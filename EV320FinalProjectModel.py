#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:23:12 2024

@author: declanmann
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

### -------- INITIAL CONDITIONS -------------
# import .asc data
ascii_grid = np.loadtxt("trailelevations.asc", dtype = 'float', skiprows=6)
ascii_headers = np.loadtxt("trailelevations.asc", max_rows = 6, dtype = 'str')
ascii_grid[ascii_grid==-9999]= np.nan
n_long = ascii_headers[0,1].astype(int)
n_lat = ascii_headers[1,1].astype(int)
dxy = ascii_headers[4,1].astype(float)
xllcorner = ascii_headers[2,1].astype(float)
yllcorner = ascii_headers[3,1].astype(float)


## ---- GRID FORMATION
x = np.arange(0, dxy*n_lat, dxy) + xllcorner # array of x values
y = np.arange(0, dxy*n_long, dxy) + yllcorner # array of z values
LAT, LONG = np.meshgrid(x, y, indexing='ij') # this sets up a plotting grid
nodes = n_long*n_lat 

##-- Creating the lengths of the model 
nx= 87 
ny= 99

dx=1 
dy=1
## ---- TIME STEP
dt = 10 # year


## ---- DIFFUSION
D = 0.004 # m2/year


## ---- INITIAL ELEVATION
elv_flat = ascii_grid.flatten()



### --------- PLOT INITIAL CONDITIONS -------------
fig, ax = plt.subplots(1,1) # use this method
elv_matrix = elv_flat.reshape(LONG.shape) # z is currently a 1D, but we will reshape it into the 2D coordinate form - recognizing that we want it to follow theshape of X.
c1 = ax.pcolormesh(LONG, LAT, elv_matrix, cmap = 'viridis')
fig.colorbar(c1)
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Distance (m)')
ax.set_title('Initial conditions')


fig = plt.figure()
ax= fig.add_subplot(111, projection= '3d')
ax.plot_surface(LAT,LONG, elv_matrix)

ax.set_xlabel('Distance in meters')
ax.set_ylabel('Distance in meters')
ax.set_zlabel('Elevation in meters')

## stability check 
sx= dt * D / dx**2 
sy= dt * D / dy**2 

print(sx)
print(sy)

import sys 
if sx > 0.5: 
    print('unstable x')
    sys.exit()
if sy > 0.5: 
    print('unstable y') 
    sys.exit()


### ---------- A MATRIX AND RUNNING THE MODEL

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

print(A)


## Time loop 

totaltime= 1000
time=0

while time<=totaltime:
    newz = np.dot(A,elv_flat)
    elv_flat[:]= newz 
    time += dt 
    
### ----- RUN A QUCK CHANGE -----
elv_matrix += np.random.random((LAT.shape))*5 # 5 meter random additions
### ---- SAVE ASCII OUTPUT (this can be opened in qgis easily!) -----
header = 'NCOLS %s \n' % n_long + 'NROWS %s \n' % n_lat + 'xllcorner %s \n' % xllcorner+ 'yllcorner %s \n' % yllcorner + 'cellsize %s \n' % dxy + 'NODATA_value - 9999'
np.savetxt('new_elev.asc', elv_matrix, header = header, comments = '')



# method 1 - use a surface plot 
Z= elv_matrix.reshape(LAT.shape)
fig = plt.figure()
ax= fig.add_subplot(111, projection= '3d')
ax.plot_surface(LAT,LONG,Z)

ax.set_xlabel('Distance (m)')
ax.set_ylabel('Distance (m)')
ax.set_zlabel('Elevation (m)')

# method 2 --- uses pcolormesh

fig2,ax2= plt.subplots(1,1)
cbar1= ax2.pcolormesh(LAT,LONG,Z)
ax2.set_title('Final Elevations')

fig2.colorbar(cbar1, ax=ax2, label='Elevation (m)')







