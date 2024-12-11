#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:17:18 2024

@author: declanmann
"""

"""
Template Code to be used for EV320 Final Project

Variables to be changed as well as some of the plotting setup. 
Elevation Data will be based on a 100m x 100m plot of lidar on a hillslope
"""
import numpy as np 
import matplotlib.pyplot as plt 
import mpl_toolkits.mplot3d as Axes3D

## init conditons 
nx = 100
ny= 100 

dx = 2 # meters 
dy= 2 # meters 

x= np.arange(0, nx*dx, dx) # creates 1-D array of x positions 
y= np.arange(0, ny*dy, dy) # creates 1-D array for y positions 

X,Y = np.meshgrid(x,y, indexing= 'ij') # creates a 2-D coordinate system for plotting 

"""
Captial letters = 2D array 
lowercase letters = 1D array 
"""
D= 0.02 #m2/year 

dt= 5 #years 

#example will use topography 

Z= np.random.random((nx, ny))*100 # expands numbers from 0-100 instead of 0-1 
z= Z.flatten() # gives the 1-D flattended array, flattended by rows(x)

## stability check 
sx= dt * D / dx**2 
sy= dt * D / dy**2 

import sys 
if sx > 0.5: 
    print('unstable x')
    sys.exit()
if sy > 0.5: 
    print('unstable y') 
    sys.exit()

## A matrix creation 

A = np.zeros((nx*ny, nx*ny)) # no longer use nodes bc 2 dimension

for i in range(nx):
    for k in range(ny):
        ik = i*ny + k 
        # -------boundary condition 
        if i == 0: 
            A[ik,ik]= 1        #no change scenario 
        elif i == (nx-1):
            A[ik,ik]= 1
        elif k == 0:
            A[ik,ik]= 1
        elif k == (ny-1):
            A[ik,ik]= 1
        else: 
            #-------- matrix coefficients 
            A[ik,ik]= 1-2*sx - 2*sy
            A[ik, (i+1) * ny + k]= sx
            A[ik, (i-1) * ny + k]= sx
            A[ik, i * ny + k + 1]= sy
            A[ik, i * ny + k - 1]= sy

#print(A)

## plotting init condtions 
# method 1 - use a surface plot 

fig = plt.figure()
ax= fig.add_subplot(111, projection= '3d')
ax.plot_surface(X,Y,Z)

ax.set_xlabel('Distance in meters')
ax.set_ylabel('Distance in meters')
ax.set_zlabel('Elevation in meters')
# method 2 --- uses pcolormesh

fig2,ax2= plt.subplots(1,1)
cbar= ax2.pcolormesh(X,Y,Z)
ax2.set_title('Initial conditions')

fig2.colorbar(cbar, ax=ax2, label='elevation (m)')

## running time 

totaltime= 1000
time=0

while time<=totaltime:
    newz = np.dot(A,z)
    z[:]= newz 
    time += dt 

## final plot 

# method 1 - use a surface plot 
Z= z.reshape(X.shape)
fig = plt.figure()
ax= fig.add_subplot(111, projection= '3d')
ax.plot_surface(X,Y,Z)

ax.set_xlabel('Distance in meters')
ax.set_ylabel('Distance in meters')
ax.set_zlabel('Elevation in meters')

# method 2 --- uses pcolormesh

fig2,ax2= plt.subplots(1,1)
cbar1= ax2.pcolormesh(X,Y,Z)
ax2.set_title('Initial conditions')

fig2.colorbar(cbar1, ax=ax2, label='Elevation (m)')