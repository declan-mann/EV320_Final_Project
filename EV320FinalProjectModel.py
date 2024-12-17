#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:23:12 2024

@author: declanmann
"""
"""
--------------------
My Earth Science Question: How do hiking trails affect hillslope diffusion over time? How long does it take for the 
hiking trails to be affected by hillslope diffusion? 

This model is a 2 dimensional diffsion problem
--------------------
INPUTS:
    
The research area used for this model was sourced from Lidar data to collect elevations. The research area
in this code is located in Eden, Utah along the South Skyline Trail. This trail cuts through part of the Uinta-Wasatch-Cache 
National Forest. The size of the research area is 99 meters by 87 meters, but the model only uses a 80 by 95 meter space of the
area. This is due to the lidar data having some NaN values so I trimmed the data grid, and the appearance of the final figures
is much cleaner. 

The elevations data file can be found in the GitHub repository. A google earth file outlining the research area can also be found there

This model runs over the course of 250 years and uses a diffusion coefficient of 0.004 m2/yr. This model also has a time step of
10 years. 
--------------------
OUTPUTS: 
    Two 3D surface plots with a slope gradient colormap overlayed with these titles-
    
        1. Initial Elevations of the South Skyline Trail in Eden, UT
        2. Elevations on the South Skyline Trail in Eden, Utah after 250 Years
        
    One surface plot depicitng the difference in elevations titled:
        
        1. Difference in Elevation After Diffusion (Final - Initial)
    
    Saves two text files in .asc format for exporting data
--------------------
Note: There are other lines of code to plot a colormesh figure depictiing the elevations. They are not used for the final outputs in
this model, but can be used for future use, if someone decides to use this code a template. 

"""

### -------- IMPORT MODULES ---------
import numpy as np
import matplotlib.pyplot as plt
import sys


### -------- INITIAL CONDITIONS -------------
    # import .asc (Elevation) data 
ascii_grid = np.loadtxt("trailelevations.asc", dtype = 'float', skiprows=6)     # I skip the top rows because there are headers before the data
ascii_headers = np.loadtxt("trailelevations.asc", max_rows = 6, dtype = 'str')

ascii_grid[ascii_grid==-9999]= np.nan        # This excludes all points without reliable data. like an elevation of -9999
ascii_grid = ascii_grid[5:-2, 2:-2]          # Here I trim the ascii grid to get rid of NaN values as most of them were focused around the edges/corners


# Define some grid parameters 
n_lat, n_long = ascii_grid.shape             # Number of rows/colunmns values 
dxy = ascii_headers[4,1].astype(float)       # The size of each cell 
xllcorner = ascii_headers[2,1].astype(float)      
yllcorner = ascii_headers[3,1].astype(float)        # define the coordinates of the corners


## ---- GRID FORMATION
x = np.arange(0, dxy*n_lat, dxy) + xllcorner  # array of x values corresponding to cells on the grid
y = np.arange(0, dxy*n_long, dxy) + yllcorner # array of y values corresponding to cells on the grid
LAT, LONG = np.meshgrid(x, y, indexing='ij') # this sets up a 2D plotting grid
nodes = n_long*n_lat 

##--- MODEL PARAMETERS
#Creating the lengths of the model 
nx= n_lat 
ny= n_long

dx=1    # spacing of the cells 
dy=1    # spacing of the cells 

# Time Step
dt = 10 # years

# Diffusion Coefficient
D = 0.004 # m2/year    


## ---- INITIAL ELEVATION
elv_flat = ascii_grid.flatten()     # this flattens my 2D elevation grid into a 1D array 


elv_matrix = elv_flat.reshape(LONG.shape) # elevation (z) is currently a 1D, 
# but we will reshape it into the 2D coordinate form - recognizing that we want it to follow theshape of X.

### --------- PLOT INITIAL CONDITIONS -------------

# this plot is not used in the final outputs, but kept for future uses 
    # fig, ax = plt.subplots(1,1) # use this method
    # c1 = ax.pcolormesh(LONG, LAT, elv_matrix, cmap = 'viridis')
    # fig.colorbar(c1)
    # ax.set_xlabel('Distance (m)')
    # ax.set_ylabel('Distance (m)')
    # ax.set_title('Initial conditions')


# create a gradient function to show slope of hillslope in plot
Z_x, Z_y= np.gradient(elv_matrix, dxy, dxy)     # this computes the gradient in both the x and y direction. 

slope = np.sqrt(Z_x**2 + Z_y**2)           # calculate slope
slope_normalized = slope / np.max(slope)      # Normalizes the slope values make visualization easier

# Create a 3D surface plot 
fig = plt.figure()
ax= fig.add_subplot(111, projection= '3d')
ax.plot_surface(LAT,LONG, elv_matrix, facecolors=plt.cm.viridis(slope_normalized), rstride=1, cstride=1)  #This overlays the slope/gradient in a color map over the plot
ax.set_title('Initial Elevations of the South Skyline Trail in Eden, UT')
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Distance (m)')
ax.set_zlabel('Elevation (m)')

# add a colorbar to show the slope values in the plot
mappable = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=np.max(slope)))
mappable.set_array(slope)
fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5, label='Slope Magnitude')

#----------------------
## Stability Check (Courant Condition)

sx= dt * D / dx**2      # stability x-direction
sy= dt * D / dy**2      # stability y direction

# Print these values if either are unstable to see what your courant value is
    # print(sx) 
    # print(sy)

# Here the code will interupt itself if the courant value is unstable
    # This is extremely helpful in saving time for computationally expensive models, and it lets you know something is wrong
import sys 
if sx > 0.5: 
    print('unstable x')
    sys.exit()
if sy > 0.5: 
    print('unstable y') 
    sys.exit()


### ---------- A MATRIX -----------------

A = np.zeros((nx*ny, nx*ny)) # no longer use nodes because 2 dimensions

for i in range(nx):
    for k in range(ny):
        ik = i*ny + k 
        # -------boundary conditions, may seem more extensive than the template code but again this is to help get rid of the NaN values at the edges 
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

# print(A)


###------- Time loop -----------

totaltime= 250    #250 year run time 
time=0

## MASK the NaN's 
A_m = np.ma.masked_invalid(A)  # Mask invalid (NaN or Inf) values in matrix A
elv_flat_m = np.ma.masked_invalid(elv_flat)  # Mask invalid values in elevation data

# print these if you are wonder if an NaN values remain in the elevation or matrix
    # print(np.any(np.isnan(A_m)))  # Check for NaNs in A_m
    # print(np.any(np.isnan(elv_flat_m)))  # Check for NaNs in elv_flat_m

# This time loop will update the elevations for every time step (10 years) and save it at elev_flat_m
while time<=totaltime:
    newz = np.ma.dot(A_m,elv_flat_m)
    elv_flat_m[:]= newz 
    time += dt 
    
    

# Create a slope gradient for the final plot 
Z= elv_flat_m.reshape(LAT.shape)   #final elevation variable

Zx, Zy= np.gradient(Z, dxy, dxy) # this computes the gradient in both the x and y direction. 

final_slope= np.sqrt(Zx**2 + Zy**2) # calculate the slope 

final_slope_normalized = final_slope / np.max(final_slope)      # scales the slope values make visualization easier

## method 1 - use a surface plot 

fig = plt.figure()
ax= fig.add_subplot(111, projection= '3d')
ax.plot_surface(LAT,LONG,Z, facecolors=plt.cm.viridis(final_slope_normalized), rstride=1, cstride=1) # overlays the final slopes gradient on top of the surface plot
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Distance (m)')
ax.set_zlabel('Elevation (m)')
ax.set_title('Elevations on the South Skyline Trail in Eden, Utah after 250 Years', fontsize= '15')


# add color bar for final plot to show slope magnitude
mappable = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=np.max(final_slope)))
mappable.set_array(final_slope)
fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5, label='Slope Magnitude')


# Colormesh not used in final outputs 
    # method 2 --- uses pcolormesh

    # fig2,ax2= plt.subplots(1,1)
    # cbar1= ax2.pcolormesh(LONG, LAT,Z)
    # ax2.set_title('Final Elevations after 1000 years on Lewis Peak and the South Skyline Trail in Huntsville, UT')
    
    # fig2.colorbar(cbar1, ax=ax2, label='Elevation (m)')

## Crearte a Difference Plot, to emphasize changes in elevation over the run time

elv_diff = Z- elv_matrix # calculate the difference between the final and initial elevations 

# use a surface plot because 2D diffusion
fig_diff = plt.figure()
ax_diff = fig_diff.add_subplot(111, projection='3d')
ax_diff.plot_surface(LAT, LONG, elv_diff, cmap='RdBu', edgecolor='none')
ax_diff.set_xlabel('Distance (m)')
ax_diff.set_ylabel('Distance (m)')
ax_diff.set_zlabel('Elevation Difference (m)')
ax_diff.set_title('Difference in Elevation After Diffusion (Final - Initial)', fontsize= '15')


### ---- SAVE ASCII OUTPUT for further data usage -----
header = 'NCOLS %s \n' % n_long + 'NROWS %s \n' % n_lat + 'xllcorner %s \n' % xllcorner+ 'yllcorner %s \n' % yllcorner + 'cellsize %s \n' % dxy + 'NODATA_value - 9999'
np.savetxt('Final_elev.asc', Z, header = header, comments = '')
np.savetxt('Init_elev.asc', elv_matrix, header=header, comments='')




