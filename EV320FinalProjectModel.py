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
ascii_grid[ascii_grid==-9999]= np.nan        # This excludes all points without reliable data. like an elevation of -9999
ascii_grid = ascii_grid[5:-2, 2:-2]

# n_long = ascii_headers[0,1].astype(int)
# n_lat = ascii_headers[1,1].astype(int)

n_lat, n_long = ascii_grid.shape
dxy = ascii_headers[4,1].astype(float)
xllcorner = ascii_headers[2,1].astype(float)
yllcorner = ascii_headers[3,1].astype(float)


## ---- GRID FORMATION
x = np.arange(0, dxy*n_lat, dxy) + xllcorner  # array of x values
y = np.arange(0, dxy*n_long, dxy) + yllcorner # array of y values
LAT, LONG = np.meshgrid(x, y, indexing='ij') # this sets up a plotting grid
nodes = n_long*n_lat 

##-- Creating the lengths of the model 
nx= n_lat 
ny= n_long

dx=1 
dy=1
## ---- TIME STEP
dt = 10 # years (should be 10 )


## ---- DIFFUSION
D = 0.004 # m2/year


## ---- INITIAL ELEVATION
elv_flat = ascii_grid.flatten()



### --------- PLOT INITIAL CONDITIONS -------------
# fig, ax = plt.subplots(1,1) # use this method
elv_matrix = elv_flat.reshape(LONG.shape) # z is currently a 1D, but we will reshape it into the 2D coordinate form - recognizing that we want it to follow theshape of X.
# c1 = ax.pcolormesh(LONG, LAT, elv_matrix, cmap = 'viridis')
# fig.colorbar(c1)
# ax.set_xlabel('Distance (m)')
# ax.set_ylabel('Distance (m)')
# ax.set_title('Initial conditions')

# create a gradient function to show slope of hillslope in plot
Z_x, Z_y= np.gradient(elv_matrix, dxy, dxy)

slope = np.sqrt(Z_x**2 + Z_y**2)
slope_normalized = slope / np.max(slope)      # scales the slope values make coloring easier



fig = plt.figure()
ax= fig.add_subplot(111, projection= '3d')
ax.plot_surface(LAT,LONG, elv_matrix, facecolors=plt.cm.viridis(slope_normalized), rstride=1, cstride=1)
ax.set_title('Initial Elevations on part of the South Skyline Trail in Eden, UT')
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Distance (m)')
ax.set_zlabel('Elevation (m)')

# add a colorbar to show the slope values in the plot
mappable = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=np.max(slope)))
mappable.set_array(slope)
fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5, label='Slope Magnitude')

## Stability Check 
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

A = np.zeros((nx*ny, nx*ny)) # no longer use nodes because 2 dimensions

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
        elif i == (nx-2):
              A[ik,ik]= 1
        elif i == (ny-2):
              A[ik,ik]= 1
        elif k == 1:
            A[ik,ik]= 1
        elif i == 1:
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

totaltime= 250    #years 
time=0


### CHANGE the maksing 
A_m = np.ma.masked_invalid(A)  # Mask invalid (NaN or Inf) values in matrix A
elv_flat_m = np.ma.masked_invalid(elv_flat)  # Mask invalid values in elevation data

print(np.any(np.isnan(A_m)))  # Check for NaNs in A_m
print(np.any(np.isnan(elv_flat_m)))  # Check for NaNs in elv_flat_m

while time<=totaltime:
    newz = np.ma.dot(A_m,elv_flat_m)
    elv_flat_m[:]= newz 
    time += dt 
    
    

# Create a slope gradient for the final plot 
Z= elv_flat_m.reshape(LAT.shape)   #final elevation variable

Zx, Zy= np.gradient(Z, dxy, dxy)

final_slope= np.sqrt(Zx**2 + Zy**2)

final_slope_normalized = final_slope / np.max(final_slope)      # scales the slope values make coloring easier

# method 1 - use a surface plot 

fig = plt.figure()
ax= fig.add_subplot(111, projection= '3d')
ax.plot_surface(LAT,LONG,Z, facecolors=plt.cm.viridis(final_slope_normalized), rstride=1, cstride=1)
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Distance (m)')
ax.set_zlabel('Elevation (m)')
ax.set_title('Final Elevations after 250 years of diffusion on the South Skyline Trail in Eden, UT')

# add color bar for final plot to show slope magnitude
mappable = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=np.max(final_slope)))
mappable.set_array(final_slope)
fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5, label='Slope Magnitude')



# method 2 --- uses pcolormesh

# fig2,ax2= plt.subplots(1,1)
# cbar1= ax2.pcolormesh(LONG, LAT,Z)
# ax2.set_title('Final Elevations after 1000 years on Lewis Peak and the South Skyline Trail in Huntsville, UT')

# fig2.colorbar(cbar1, ax=ax2, label='Elevation (m)')

## Difference 
elv_diff = Z- elv_matrix

# Method 1: Surface plot for elevation difference
fig_diff = plt.figure()
ax_diff = fig_diff.add_subplot(111, projection='3d')
ax_diff.plot_surface(LAT, LONG, elv_diff, cmap='RdBu', edgecolor='none')
ax_diff.set_xlabel('Distance (m)')
ax_diff.set_ylabel('Distance (m)')
ax_diff.set_zlabel('Elevation Difference (m)')
ax_diff.set_title('Difference in Elevation After Diffusion (Final - Initial)')


### ---- SAVE ASCII OUTPUT (this can be opened in qgis easily!) -----
header = 'NCOLS %s \n' % n_long + 'NROWS %s \n' % n_lat + 'xllcorner %s \n' % xllcorner+ 'yllcorner %s \n' % yllcorner + 'cellsize %s \n' % dxy + 'NODATA_value - 9999'
np.savetxt('Final_elev.asc', Z, header = header, comments = '')
np.savetxt('Init_elev.asc', elv_matrix, header=header, comments='')




