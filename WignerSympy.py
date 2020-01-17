# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:39:23 2019

@author: Benjamin Smith
"""

from sympy.physics.quantum.spin import Rotation as r
import sympy as sp
import numpy as np


a, b, c = sp.symbols('a b c')

J = 1
Jmag = int(2*J+1)
alpha, beta, gamma = 0, -np.pi/2, 0
mat = sp.zeros(Jmag, Jmag)
states = np.linspace(-J, J, Jmag, dtype=int)
for i in range(Jmag):
    for j in range(Jmag):
        mat[i, j] = sp.nsimplify(r.D(J, states[i], states[j], a, b, c).doit(), 
           tolerance=1e-10,rational=True)

#rot_mat = mat * sp.Matrix([0, 1, 0])
print(mat)