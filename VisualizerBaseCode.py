# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 16:09:29 2019

@author: Benjamin Smith
"""

import sys
from PyQt5 import QtGui, uic
from PyQt5.QtWidgets import QVBoxLayout, QApplication, QMainWindow

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import proj3d

from matplotlib.backends.backend_qt5agg import (
	FigureCanvasQTAgg as FigureCanvas,
	NavigationToolbar2QT as NavigationToolbar)
from itertools import product, combinations

from scipy.linalg import expm, norm





class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        uic.loadUi('Visualizer.ui', self)
        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.plot_layout = QVBoxLayout()        
        self.plot_layout.addWidget(self.canvas)        
        self.plot_layout.addWidget(self.toolbar)        
        self.gridLayout.addLayout(self.plot_layout, 0, 0, 1, 1)

        self.ax = self.figure.add_subplot(111)
        self.ax.plot(np.random.rand(15))
        self.show()
#        
if __name__ == '__main__':
    app = 0    
    app = QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec_())