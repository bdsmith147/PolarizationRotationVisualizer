# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 09:11:03 2019

@author: Benjamin Smith
"""

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from scipy.linalg import expm, norm
plt.close()

np.set_printoptions(suppress=True)

class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
        
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

#Cartesian coordinate unit vectors
x_ax = Arrow3D([0, 0.5], [0, 0], [0, 0], mutation_scale=20,
            lw=3, arrowstyle="-|>", color="k")
y_ax = Arrow3D([0, 0], [0, 0.5], [0, 0], mutation_scale=20,
            lw=3, arrowstyle="-|>", color="k")
z_ax = Arrow3D([0, 0], [0, 0], [0, 1.0], mutation_scale=20,
            lw=3, arrowstyle="-|>", color="g")
#Quantization axis (along z-axis)
#q_ax = Arrow3D([0.5, 0.5], [0.5, 0.5], [0.2, 1], mutation_scale=20,
#            lw=10, arrowstyle="-|>", color="C2", alpha=1)

#Umat = np.sqrt(2)*np.array([[-1, -1j, 0], [0, 0, 1/np.sqrt(2)], [1, -1j, 0]])
#Udagg = np.conj(Umat).T

polzn = 0
a, b, c = 0, 45, 0 # in degrees
alpha, beta, gamma = np.radians([a, b, c]) # Euler angles, in radians   
A = EulerRot(alpha, beta, gamma)
D = WignerD(alpha, beta, gamma)


r = 0.15
N = 100
if np.abs(polzn) == 0:
    rot_init = (0, np.pi/2, 0)
    vec = np.array([1.5,0,0]) #starts the beam along the x-axis
    vec = np.dot(EulerRot(*rot_init), vec) #rotates the beam to z-axis
    
    state = np.array([0, 1, 0])
    state = np.dot(WignerD(*rot_init), state)
    
    #Represent the linear polarization as a line transverse to the beam
    x = np.linspace(-r, r, N)
    y = np.zeros_like(x)
    z = -0.75*np.ones_like(x)

else:
    vec = -np.array([0,0,1.5])
    
    if polzn > 0:
        state = np.array([0, 0, 1])
    else:
        state = np.array([1, 0, 0])
    
    #Define the line representing the circular polarization
    theta = np.linspace(0, 2*np.pi, N)
    x, y = r*np.cos(theta), r*np.sin(theta)
    z = -0.75*np.ones_like(x)
    

vec = np.dot(A, vec)
state = np.round(np.dot(D, state), decimals=15)
print("Polarization State:")
print('\t' + str(state[0])+"\t |-1> \t+ ")
print('\t' + str(state[1])+"\t |0> \t+ ")
print('\t' + str(state[2])+"\t |+1>")
x, y, z = RotatePoints(x, y, z, A)

beam = Arrow3D([vec[0], 0], [vec[1], 0], [vec[2], 0], mutation_scale=20,
            lw=5, arrowstyle="-|>", color="r")



#%%

fig = plt.figure()
ax = fig.gca(projection='3d')
# draw bounding cube
r = [-1, 1]
for s, e in combinations(np.array(list(product(r, r, r))), 2):
    if np.sum(np.abs(s-e)) == r[1]-r[0]:
        ax.plot3D(*zip(s, e), color="C0", lw=0) 
# draw vectors
ax.add_artist(x_ax)
ax.add_artist(y_ax)
ax.add_artist(z_ax)
#adds origin point
ax.scatter([0], [0], [0], color="k", s=8) 
# draw sphere representing atoms
ax.scatter([0], [0], [0], color="C3", s=200, alpha=0.5) 
#draws beam vector
ax.add_artist(beam)


if np.abs(polzn) == 1:
    if polzn > 0:
        theta = np.flip(theta)

    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    #account for the last to first segment
    lastseg = np.array([points[-1], points[0]]).reshape(1, 2, 3)
    segments = np.append(segments, lastseg, axis=0)
    
    cmap=plt.get_cmap('seismic')
    colors=[cmap(i) for i in theta/(2*np.pi)]

    for j, seg in enumerate(segments):
        line, = ax.plot(seg[:,0], seg[:,1], seg[:,2], lw=3, color=colors[j])
        line.set_solid_capstyle('round')
else:
    ax.plot(x, y, z, lw=3, color='b')

plt.show()

#%%
plt.close()
angle = np.linspace(0, 2*np.pi, 50)
newstate = np.zeros((len(angle), 3), dtype='complex128')
for i, ang in enumerate(angle):
    dmat = WignerD(ang, np.radians(0), np.radians(0))
    newstate[i] = np.round(np.dot(dmat, state), decimals=15)

f, g, h = newstate.T
fig, ax = plt.subplots()
ax.scatter(np.real(f), np.imag(f))
plt.scatter(np.real(g), np.imag(g))
plt.scatter(np.real(h), np.imag(h))
ax.set_aspect(1)
plt.grid()
plt.show()

#cartstate = np.array([np.real(np.dot(Udagg, s)) for s in newstate])
#cartstate = cartstate / np.tile(np.linalg.norm(cartstate, axis=1), (3,1)).T
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)



#%%
R = 1/np.sqrt(2) * np.array([1, -1j])
L = 1/np.sqrt(2) * np.array([1, 1j])
psi = np.array([elem[0]*R + elem[2]*L for elem in newstate]) # the order of L and R might be wrong
Sz = np.linalg.norm(psi, axis=1)
Sx = np.array([2*np.real(elem[0]*np.conj(elem[0])) for elem in psi])
Sy = np.array([2*np.imag(elem[0]*np.conj(elem[0])) for elem in psi])


fig = plt.figure()
ax = fig.gca(projection='3d')
r = [-1, 1]
for s, e in combinations(np.array(list(product(r, r, r))), 2):
    if np.sum(np.abs(s-e)) == r[1]-r[0]:
        ax.plot3D(*zip(s, e), 'w', lw=0)

ax.plot_wireframe(x, y, z)
ax.plot(Sx, Sy, Sz, color='C1')
plt.show()

