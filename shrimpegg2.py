#!/usr/bin/env python

from datetime import datetime, timedelta
import numpy as np
from opendrift.models.opendrift3D import OpenDrift3DSimulation, Lagrangian3DArray
from opendrift.readers import reader_basemap_landmask
from opendrift.readers import reader_netCDF_CF_generic
from opendrift.models import basemodel
from opendrift.readers import reader_ROMS_native

import logging
import os,sys
from netCDF4 import Dataset, datetime, date2num,num2date, MFDataset
from numpy.random import RandomState
import matplotlib.pyplot as plt
import csv

import ShrimpLarvae.IBM.get_2Dvar_interp as interp2D
from ShrimpLarvae.IBM.IBMshrimp import PelagicShrimpDrift

#from ShrimpLarvae.IBM import reader_ROMS_native

#########################
# SETUP FOR ROMS PROJECT
#########################
romsDirectory='/Users/conecta/Desktop/LTRANSv2b/input/catb_his_*.nc' #Directory for ROMS files


startTime=datetime(2001,7,1,0,0,0)
endTime=datetime(2001,7,25,0,0,0)
verticalBehavior=True
N=3


# ROMS: 2KM, 24H resolution:
outputFilename='testHdiff.nc'

print "Result files will be stored as:\nnetCDF=> %s"%outputFilename
print "N is %s "%N



### Random value of Density from a gaussian distribution
def setupDensity(MeanDensity,SdDensity,NumberParticule):
    density=np.random.normal(MeanDensity,SdDensity,NumberParticule)
    return density

### Random value of Size eggs from a gaussian distribution
def setupSizeEgg(MeanSizeeggs,SdSizeeggs,NumberParticule):
    size=np.random.normal(MeanSizeeggs,SdSizeeggs,NumberParticule)*10**-3
    return size


### Function to run the simulation
def createAndRunSimulation(startTime,endTime,outputFilename,romsDirectory,verticalBehavior,N):
    #rho = float(sys.argv[4])
    Settlement=False # sys.argv[1]
    Diff = 2 # int(sys.argv[3])
### Which behaviour applying ? 0: Buoyancy, 1: Just Kh, 2: Just Kv, 3: Kv --> Kh | MLD, 4: buoyancy + Kh, >4:nothing
        # Buoyancy + Kv-->Kh | MLD, and Buoyancy + Kv+ Kh could be configured later
    Buoyancy=False
    VertiLarvae=False
    HorizDiff=False
    VertDiff=False
    if Diff == 0:
        VertiLarvae=True
        Buoyancy=True
    if Diff == 1:
        HorizDiff=True;
    if Diff == 2:
        VertDiff=True
    if Diff == 3:
        Buoyancy=True
        VertDiff=True
        HorizDiff=True
    if Diff == 4:
        HorizDiff=True
        VertiLarvae=True
        Buoyancy=True
    if Diff>4:
        print " Behavior is deactivated, no diffusivity, buoyancy or vertical swimming"
        verticalBehavior=False

    # Setup a new simulation
    o = PelagicShrimpDrift(loglevel=0)  # Set loglevel to 0 for debug information

    #######################
    # Preparing readers
    #######################
    #Do not include basemap when stranding is deactivated
    reader_roms = reader_ROMS_native.Reader(romsDirectory)
    reader_roms.zlevels= [0, -.5, -1, -3, -5, -10, -25, -50, -75, -100, -150, -200,
                         -250, -300, -400, -500, -550, -600, -650, -700, -750, -800, -900, -1000, -1500,
                         -2000, -2500, -3000]

    reader_roms.interpolation = 'linearNDFast' # Tested : nearest (work) and linearND (work also)
  
    o.add_reader([reader_roms])
    
   #########################
   #Adjusting configuration
   #########################
    o.set_config('processes:verticaladvection', True)
    o.set_config('drift:scheme', 'runge-kutta')
    o.set_config('general:coastline_action', 'stranding')

    if verticalBehavior:
        o.set_config('processes:Buoyancy', Buoyancy)
        o.set_config('processes:Settlement', Settlement)
        o.set_config('processes:turbulentmixing', VertDiff) # Activating vertical turbulences + buoyancy
        o.set_config('processes:horiz_turbulent',HorizDiff) # Activating horizontal turbulences
        o.set_config('processes:verticalmodule', VertiLarvae) # Activating only buoyancy + velocity of settlement
        o.set_config('turbulentmixing:TSprofiles', False)
        o.set_config('turbulentmixing:moduleturb',Diff)
        o.set_config('turbulentmixing:timestep', 60) # seconds
        o.set_config('turbulentmixing:verticalresolution', 1)
        o.set_config('turbulentmixing:diffusivitymodel','environment')
        o.set_config('turbulentmixing:diffusivityhorizontal','constant')

        print "Buoyancy is activated: %s, Settlement velocity is activated: %s"%(str(o.get_config('processes:Buoyancy')),str(o.get_config('processes:Settlement')))
        print "      "
        print "Vertical mixing is activated: %s , Horizontal mixing is activated: %s"%(str(o.get_config('processes:turbulentmixing')),str(o.get_config('processes:horiz_turbulent')))
        print "          "
        print "The configuration module is number:"
        print Diff
        print "          "

    #######################
    # IBM configuration
    #######################

    #######################
    # Seed particles
    #######################
    
    #Alternative 1: Fixed distribution on bottom:
    lon=[]
    lat=[]
    with open("/Users/conecta/Documents/Chap2_Doc/Biology Aristeus/Preliminary/ReleaseChap2TRUE.txt","rb") as f:
          reader=csv.reader(f, delimiter=" ")
          for row in reader:
              lon=np.append(lon,float(row[0]))
              lat=np.append(lat,float(row[1]))
              if len(lon)== N:
                  break

    #depth=data.V3[0:N]
    #z=depth(depth.tolist())[0:N]

    #Alternative 2: Bathymetry taken from longitude and latitude of the initial position of particles
    # No time interpolation
    # Function get_2Dvat_interp is written in basereader.py of opendrift.
#z,detail=interp2D.get_2DBathy_interp(reader_roms,'sea_floor_depth_below_sea_level',time=startTime,lon=lon,lat=lat, rotate_to_proj=o.proj)

    diameter=setupSizeEgg(0.33,0.0405,N) # Oocytes size at the last stages of maduration according to Demestre and fortuno 1992
                                         #  and kapiris, 2009 in mm will return diameter in m
    density = setupDensity(981,15,N) ### Density from observed density on deep Sea shrimp in kg/m3


    o.seed_elements(lon, lat, z='seafloor', density=density,time=startTime, diameter=diameter, stages_end_buoyancy=2)



    #########################
    # Run the model
    #########################
    o.run(end_time=endTime, time_step=86400/4, outfile=outputFilename)



createAndRunSimulation(startTime,endTime,outputFilename,romsDirectory,verticalBehavior,N)

