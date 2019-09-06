import sys
import warnings

from PyQt5.QtWidgets import (QApplication, QMainWindow, QMenu, QVBoxLayout, 
    QSizePolicy, QMessageBox, QWidget, QPushButton)
#from PyQt5.QtGui import QIcon
from PyQt5 import uic

import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

from mpl_toolkits.mplot3d import axes3d, proj3d
from matplotlib.patches import FancyArrowPatch

from itertools import product, combinations
from scipy.linalg import expm, norm

class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
        
    
def cComponents(c):
    return np.array([np.real(c), np.imag(c)])
        
        
def M(axis, theta):
    return expm(np.cross(np.eye(3), axis/norm(axis)*theta))

def EulerRot(alpha, beta, gamma):
    z_ax = np.array([0,0,1])
    y_ax = np.array([0,1,0])
    return np.matmul(np.matmul(M(z_ax, alpha), M(y_ax, beta)), M(z_ax, gamma))

def RotatePoints(x, y, z, A):
    points = zip(x, y, z)
    vec = np.array([np.dot(A, p) for p in points])
    return vec.T

def WignerD(alpha, beta, gamma):
    '''
    Wigner D-matrix for J=1.
    '''
    c = np.cos(beta)
    s = np.sin(beta)
    c2 = np.cos(beta/2)
    s2 = np.sin(beta/2)
    little_d = np.array([[c2**2, -s/np.sqrt(2), s2**2],
                          [s/np.sqrt(2), c, -s/np.sqrt(2)],
                          [s2**2, s/np.sqrt(2), c2**2]])
    
    expon = np.exp(-1j* np.array([[-alpha - gamma, -alpha, -alpha + gamma], 
                                  [-gamma, 0, gamma], 
                                  [alpha - gamma, alpha, alpha + gamma]]))
    return expon*little_d


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        uic.loadUi('Visualizer.ui', self)
        self.left = 10
        self.top = 10
        self.title = 'Polarization Rotation Visualizer'
        self.width = 640
        self.height = 400
        
#        self.vec = -np.array([0,0,1.5])
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
#        self.setGeometry(self.left, self.top, self.width, self.height)
        
        self.alpha_LineEdit.returnPressed.connect(lambda: self.set_slider(0))
        self.beta_LineEdit.returnPressed.connect(lambda: self.set_slider(1))
        self.gamma_LineEdit.returnPressed.connect(lambda: self.set_slider(2))
        
        
        self.alphaSlider.valueChanged.connect(lambda: self.sliderChanged(0))
        self.betaSlider.valueChanged.connect(lambda: self.sliderChanged(1))
        self.gammaSlider.valueChanged.connect(lambda: self.sliderChanged(2))
        
        self.resetButton.clicked.connect(self.reset)
        self.rightRadioButton.clicked.connect(self.input_poln)
        self.linearRadioButton.clicked.connect(self.input_poln)
        self.leftRadioButton.clicked.connect(self.input_poln)
        
        self.polnButtonGroup.setId(self.rightRadioButton, -2)
        self.polnButtonGroup.setId(self.linearRadioButton, 0)
        self.polnButtonGroup.setId(self.leftRadioButton, 1)
        
        
        self.plot_layout = QVBoxLayout()
        self.alpha = np.radians(self.alphaSlider.value())
        self.beta = np.radians(self.betaSlider.value())
        self.gamma = np.radians(self.gammaSlider.value())
        
        self.alpha_LineEdit.setText(str(int(np.degrees(self.alpha))))
        self.beta_LineEdit.setText(str(int(np.degrees(self.beta))))
        self.gamma_LineEdit.setText(str(int(np.degrees(self.gamma))))
        
        self.pc = PlotCanvas(self, width=5, height=4)
        
        self.toolbar = NavigationToolbar(self.pc, self)  
        self.plot_layout.addWidget(self.pc)        
        self.plot_layout.addWidget(self.toolbar)        
        self.gridLayout.addLayout(self.plot_layout, 0, 0, 0, 1)
        
        
        #initialize the vector showing the direction of the light
#        self.vec = np.array([0,0,1.5])
#        self.init_vec = np.copy(self.vec)
#        self.pc.update_vec(self.vec)
        
        #initialize the polarization "state" of the light
#        self.state = np.array([0,1,0])
#        self.init_state = np.copy(self.state)
#        self.update_state()
        
        #initialize the polarization "representation" of the light
        self.inp_polzn = self.checkInputPoln()
        print(self.inp_polzn)
        r = 0.15
        N = 50
        if self.inp_polzn == 0:
            rot_init = np.array([0, np.pi/2, 0])
            self.vec = np.array([1.5,0,0]) #starts the beam along the x-axis
            self.state = np.array([0, 1, 0])
            z = np.linspace(-r, r, N)
            y = np.zeros_like(z)
            x = 0.75*np.ones_like(z)
        
        else:
            rot_init = np.array([0, 0, 0])
            self.vec = np.array([0, 0, -1.5])
            
            self.theta = np.linspace(0, 2*np.pi, N)
            x, y = r*np.cos(self.theta), r*np.sin(self.theta)
            z = -0.75*np.ones_like(x)
            
            if self.inp_polzn > 0:
                self.state = np.array([0, 0, 1])
            else:
                self.state = np.array([1, 0, 0])
                self.theta = np.flip(self.theta)
        
        self.pol_curve = np.stack((x, y, z), axis=1)
        #rotates the beam to z-axis from initial orientation
#        self.vec = np.dot(EulerRot(*rot_init), self.vec) 
        self.init_vec = np.copy(self.vec)
        self.init_state = np.copy(self.state)
        self.init_pol_curve = np.copy(self.pol_curve)
#        print(self.pol_curve)
#        print(self.state)
        self.rotate(*rot_init)
#        print(self.state)
        self.init_vec = np.copy(self.vec)
        self.init_state = np.copy(self.state)
        self.init_pol_curve = np.copy(self.pol_curve)
        
        
        self.show()
        
    
    def rotate(self, alpha=None, beta=None, gamma=None):
        '''Performs the rotation operations on the state and the indidence
        vector.'''
        if (alpha == None and beta == None and gamma == None):
            alpha, beta, gamma = self.alpha, self.beta, self.gamma
            
        self.R = EulerRot(alpha, beta, gamma)
        self.D = WignerD(alpha, beta, gamma)
        
        self.state = np.dot(self.D, self.init_state)
        self.vec = np.dot(self.R, self.init_vec)
        self.pol_curve = self.rotation_op(self.R, self.init_pol_curve)
        
        
        self.pc.update_vec(self.vec)
        self.pc.update_poln(self.pol_curve, self.inp_polzn)
        self.update_state()
        
    def rotation_op(self, mat, vectors):
        '''Generalizes the rotation operation to also run over a list of 
        vectors'''
        vectors = np.array(list(vectors))
        return np.array([np.dot(mat, v) for v in vectors])
    
    def input_poln(self):
        inputpol = self.checkInputPoln()
        if inputpol == -1:
            print('-1')
        elif inputpol == 0:
            print('0')
        elif inputpol == 1:
            print('+1')

    def checkInputPoln(self):
        polzn = self.polnButtonGroup.checkedId()
        if polzn < 0:
            polzn = int(polzn/2)
        return polzn
    
    def set_slider(self, index):
        '''Set the position of the slider by entering a value in the 
        corresponding text box.'''
        if index == 0:
            slider = self.alphaSlider
            lineEdit = self.alpha_LineEdit
            val = np.clip(int(lineEdit.text()), -180, 180)
            self.alpha = np.radians(val)
            val = self.alpha
        elif index == 1:
            slider = self.betaSlider
            lineEdit = self.beta_LineEdit
            val = np.clip(int(lineEdit.text()), -180, 180)
            self.beta = np.radians(val)
            val = self.beta
        elif index == 2:
            slider = self.gammaSlider
            lineEdit = self.gamma_LineEdit
            val = np.clip(int(lineEdit.text()), -180, 180)
            self.gamma = np.radians(val)
            val = self.gamma
        
        val = int(np.degrees(val))
        slider.setValue(val)
        lineEdit.setText(str(val))
        self.rotate()
        self.pc.update_plot()
    
    def update_plot(self):
        pass
        
    def update_state(self):
        self.negativeLineEdit.setText('{:.2f}'.format(self.state[0]))
        self.zeroLineEdit.setText('{:.2f}'.format(self.state[1]))
        self.positiveLineEdit.setText('{:.2f}'.format(self.state[2]))
    
    def reset(self):
        print('Resetting...')
        pass
    
    def sliderChanged(self, index):
        if index == 0:
            slider = self.alphaSlider
            lineEdit = self.alpha_LineEdit
            self.alpha = np.radians(slider.value())
        elif index == 1:
            slider = self.betaSlider
            lineEdit = self.beta_LineEdit
            self.beta = np.radians(slider.value())
        elif index == 2:
            slider = self.gammaSlider
            lineEdit = self.gamma_LineEdit
            self.gamma = np.radians(slider.value())
            
        lineEdit.setText(str(slider.value()))
        
        self.rotate()
        self.pc.update_plot()
    
    
        


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=7, dpi=100, vec=None, pol_curve=None):
        self.origin = np.array([0.,0.,0.])
        if (vec == None and pol_curve == None):
            self.vec = self.origin
        else:
            self.vec = vec
        self.pol_curve = pol_curve
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, self.fig)
#        self.axes = self.fig.add_subplot(111)
        self.axes = self.fig.gca(projection='3d')
#        self.axes.set_aspect('equal')
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.x_ax = self.arrow3D(self.origin, np.array([0.75, 0., 0.]))
        self.y_ax = self.arrow3D(self.origin, np.array([0., 0.75, 0.]))
        self.z_ax = self.arrow3D(self.origin, np.array([0., 0., 1.5]), color="g")
        self._init_plot()


    def _init_plot(self):
#        data = 2*np.random.rand(3, 25) - 1
        self.vec = np.array([0,0,-1.5])
        self.beam = self.arrow3D(self.vec, self.origin, lw=5, color='r')
        
        # draw bounding cube
        r = [-1, 1]
        for s, e in combinations(np.array(list(product(r, r, r))), 2):
            if np.sum(np.abs(s-e)) == r[1]-r[0]:
                self.axes.plot3D(*zip(s, e), color="C0", lw=0) 
#        points = self.axes.scatter(*(data))
        self.axes.add_artist(self.x_ax)
        self.axes.add_artist(self.y_ax)
        self.axes.add_artist(self.z_ax)
        #adds origin point
        self.axes.scatter([0], [0], [0], color="k", s=8) 
        # draw sphere representing atoms
        self.axes.scatter([0], [0], [0], color="C3", s=200, alpha=0.5)
        
        self.axes.add_artist(self.beam)
#        points.remove()
        
        self.axes.set_xticklabels([])
        self.axes.set_yticklabels([])
        self.axes.set_zticklabels([])
        self.draw()
        
    def update_plot(self):
        self.draw()
    
    def update_vec(self, vec):
        self.vec = vec
        self.beam.remove()
        self.beam = self.arrow3D(self.vec, self.origin, lw=5, color='r')
        self.axes.add_artist(self.beam)
    
    def update_poln(self, curve, inpt):
        try:
            self.pol_curve.remove()
        except AttributeError:
            pass
        self.pol_curve, = self.axes.plot3D(*curve.T, color='b')
    
    def arrow3D(self, point1, point2, color='k', lw=3):
        return Arrow3D(*zip(point1, point2), mutation_scale=20,
                            lw=lw, arrowstyle="-|>", color=color)

if __name__ == '__main__':
    def run_app():
        app = QApplication(sys.argv)
        ex = App()
        ex.show()
        app.exec_()
    run_app()