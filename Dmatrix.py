# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 11:19:40 2020

@author: benjamin

Another way to look at this:
    Look at the reverse problem - I want ___
    polarization components with ___ complex
    phases: what are the polarizations and angles
    that produce this?
    
What kinds of polarization states might people
care about? Equal parts pi/sig+/sig-? (they would 
have to have different phases, but maybe that's ok.)

How do you conceptualize the polarization states?
They don't fit on a Poincare sphere, because this
doesn't take care of the pi component. What kind of
object does do this properly?
"""

import numpy as np
from sympy.physics.quantum.spin import Rotation as R
import sympy as sp
sp.init_printing()
from matplotlib import pyplot as plt
alpha, beta, gamma = sp.symbols('alpha beta gamma')
J = 1
m = 1
mp = 1
Rot = sp.zeros(3, 3)
for i, m in enumerate(range(-1, 2)):
    for j, mp in enumerate(range(-1, 2)):
        val = R.D(J, m, mp, alpha, beta, gamma).doit()
        Rot[i, j] = val
#        print(val)
        
## For sigma light rotated along the polar angle
N = 101
angles = np.linspace(0, 2*np.pi, N)
states = np.zeros((N, 3)) #might have to be complex
state_init = np.array([[1, 0, 0]]).T
for i, ang in enumerate(angles):
    rotMat = Rot.subs(alpha, 0).subs(gamma, 0).subs(beta, ang)
    newval = rotMat*state_init
    states[i,:] = newval.T

degrees = angles * 180/np.pi    
plt.figure()
plt.plot(degrees, states[:,0], label='-1')
plt.plot(degrees, states[:,1], label='0')
plt.plot(degrees, states[:,2], label='+1')
plt.ylim(-1.2, 1.2)
plt.xlabel('Polar Angle $\\theta$')
plt.ylabel('Component Amplitude')
plt.xticks(np.linspace(0, 360, 9))
plt.legend(loc=4)
plt.grid()
plt.title('Rotation of $\\sigma^+$-polarized light')
plt.show()


## For pi light rotated along the polar angle
states = np.zeros((N, 3)) #might have to be complex
state_init = np.array([[0, 1, 0]]).T
for i, ang in enumerate(angles):
    rotMat = Rot.subs(alpha, 0).subs(gamma, 0).subs(beta, ang)
    newval = rotMat*state_init
    states[i,:] = newval.T
    
plt.figure()
plt.plot(degrees, states[:,0], label='-1')
plt.plot(degrees, states[:,1], label='0')
plt.plot(degrees, states[:,2], label='+1')
plt.ylim(-1.2, 1.2)
plt.xlabel('Polar Angle $\\theta$')
plt.ylabel('Component Amplitude')
plt.xticks(np.linspace(0, 360, 9))
plt.legend(loc=4)
plt.grid()
plt.title('Rotation of $\\pi$-polarized light')
plt.show()


## For sigma light rotated initally 90 degrees along the polar angle
# and then all the way around the azimuthal angle
states = np.zeros((N, 3), dtype='complex') #might have to be complex
state_init = np.array([[1*np.exp(12j*np.pi/8 * 1), 0, 0]]).T
for i, ang in enumerate(angles):
    rotMat = Rot.subs(alpha, ang).subs(gamma, 0).subs(beta, np.pi/2)
    newval = rotMat*state_init
    states[i,:] = newval.T
    
plt.figure()
plt.plot(degrees, np.real(states[:,0]), label='-1')
plt.plot(degrees, np.real(states[:,1]), label='0')
plt.plot(degrees, np.real(states[:,2]), label='+1')
plt.plot(degrees, np.imag(states[:,0]), label='-1 (imag)')
plt.plot(degrees, np.imag(states[:,1]), label='0 (imag)')
plt.plot(degrees, np.imag(states[:,2]), label='+1 (imag)')
plt.ylim(-1.2, 1.2)
plt.xlabel('Azimuthal Angle $\\phi$')
plt.ylabel('Component Amplitude')
plt.xticks(np.linspace(0, 360, 9))
plt.legend(loc=4)
plt.grid()
plt.title('Rotation of $\\sigma$-polarized light')
plt.show()

plt.figure()
plt.axes().set_aspect('equal')
plt.plot(np.real(states[:,2]), np.imag(states[:,2]))
plt.grid()
plt.show()