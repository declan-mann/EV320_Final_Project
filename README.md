# EV320_Final_Project

####--------------------

##My Earth Science Question:
    
    How do hiking trails affect hillslope diffusion over time? How are hiking trails affected by 
    hillslope diffusion?

    This model is a 2-dimensional diffusion problem

####--------------------

##INPUTS:
    
        The research area used for this model was sourced from Lidar data to collect elevations. The research area
    imported to this code is located in Eden, Utah, along the South Skyline Trail. The space chosen for this model
    has 2 parts of a hiking trail included. One hiking trail cuts around the high elevations on the hillslope, 
    and the other cuts pretty much straight across the lower hillslope. The trails cut through part of the 
    Uinta-Wasatch-Cache National Forest. On the plots, you can identify the trails as they appear a darker color than
    the rest of the hillslope. There is a valley/dip in the elevation at the lower end plot, but you can tell it is 
    not a trail because there is a sharp increase in elevation after the dark blue area. The research area is 99 
    meters by 87 meters, but the model only uses an 80 by 95-meter space of the area. This is due to the Lidar data
    having some NaN values, so I trimmed the data grid, and the appearance of the final figures is much cleaner. 

    The elevation data file can be found in the GitHub repository and it is titled "trailelevations.asc". A Google 
    Earth file outlining the research area can also be found in the repository titled "EV320.kml." The Google Earth 
    file allows you to explore the search area in an accessible way if you don't have a computer that can handle more 
    complex GIS systems. 

    This model runs for 250 years and uses a diffusion coefficient of 0.004 m2/yr. This model also has a time step of 
    10 years. 

####--------------------

##OUTPUTS: 
   
      Two 3D surface plots with a slope gradient colormap overlayed with these titles-
    
        1. "Initial Elevations of the South Skyline Trail in Eden, UT"
        2. "Elevations on the South Skyline Trail in Eden, Utah after 250 Years"
        
      One surface plot depicting the difference in elevations is titled:
        
        1. "Difference in Elevation After Diffusion (Final - Initial)"
    
      The code will also save two text files in .asc format for exporting data, like a 3d print, or to collect elevations 
      for multiple different run times. 
####--------------------

##Note: 

    There are other lines of code to plot a color mesh figure depicting the elevations. They are not used for the 
    final outputs in this model but can be used for future use if someone decides to use this code as a template. 
