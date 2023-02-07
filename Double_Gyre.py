# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:40:10 2023

@author: pmg124
"""

import timeit
import time

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt


from numpy import dot, multiply, diag, power
from numpy import pi, exp, sin, cos, cosh, tanh, real, imag
from numpy.linalg import inv, eig, pinv, norm, qr
from scipy.linalg import svd, svdvals, orth
from matplotlib import cm
from scipy.fftpack import fft
from pandas import DataFrame
import pandas as pd
from matplotlib import ticker
from numpy.linalg import matrix_rank

import matplotlib.tri as tri
import scipy.io

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


plt.close('all')
plt.clf()


A = 0.1;
omega = 2* np.pi/10
eps = 0.25;
Nx = 256;
Ny = 256;
Tsp = Nx*Ny;

x = np.linspace(0,2,Nx)
y1 = np.linspace(0,1,Ny)
z = y1;
y = y1.reshape(-1,1)

dt = 0.05
ts = np.arange(0,10,dt);
Nt = np.shape(ts)[0];

for j in range(0,Nt,1):
    t=ts[j]
    f=eps* np.sin(omega*t)*x**2 + (1 - 2*eps*np.sin(omega * t))*x
    psi=A* np.sin(np.pi*f)* np.sin(np.pi*y)
   
    dfdx=2*eps* np.sin(omega*t)*x + (1 - 2*eps*np.sin(omega * t))
    d2fdx2=2*eps* np.sin(omega*t)
       
    vx=-A*np.pi* np.sin(np.pi* f)*np.cos(np.pi * y)
    vy=A*np.pi* np.cos(np.pi*f)* np.sin(np.pi* y)* dfdx

    dvxdy=(np.pi)**2* A* np.sin(np.pi* f)*np.sin(np.pi* y)
    dvydx=A*np.pi* np.sin(np.pi* y)*(np.cos(np.pi*f)*d2fdx2-np.sin(np.pi*f)*dfdx*dfdx)
   
    vorticity=dvydx-dvxdy
   
    if j==0:
       vx_s=np.concatenate((np.reshape(vx,(Nx*Ny,1)),),axis=1)
       vy_s=np.concatenate((np.reshape(vy,(Nx*Ny,1)),),axis=1)
       vort_s=np.concatenate((np.reshape(vorticity,(Nx*Ny,1)),),axis=1)
    else:
       vx_s=np.concatenate((vx_s,(np.reshape(vx,(Nx*Ny,1)))),axis=1)
       vy_s=np.concatenate((vy_s,(np.reshape(vy,(Nx*Ny,1)))),axis=1)
       vort_s=np.concatenate((vort_s,(np.reshape(vorticity,(Nx*Ny,1)))),axis=1)
       

mm=100
   
       
for i in range(0,Nt):
    print(i)
    plt.figure(1)
    cp0=plt.contourf(x,y1,np.reshape(vort_s[:,i],(Nx,Ny)))   
    plt.colorbar(cp0)
    #plt.hold(True)
    cp01 = plt.streamplot(x, y1, np.reshape(vx_s[:,i],(Nx,Ny)),np.reshape(vy_s[:,i],(Nx,Ny)))
    plt.xlim(0,2)
    plt.ylim(0,1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    time.sleep(1)