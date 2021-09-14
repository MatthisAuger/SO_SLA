#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib
#~ matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy import feature
from numba import jit
import argparse
import datetime
from netCDF4 import Dataset
import matplotlib.path as mpath
import sys
sys.path.insert(0, '/usr/home/mauger/support/')
import functions
import imp
imp.reload(functions)
from ease2 import ease_grid
from scipy import ndimage
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.convolution import convolve
from scipy.signal import welch
from scipy.interpolate import interp1d


plot_unfiltered = False
plot_filtered = False
plot_date = datetime.datetime(2019,3,15)

file_in = "/net/ether/data/proteo1/mauger/Data/SLA/v4/final_product/dt_antarctic_multimission_sea_level_uv_20130401_20193107.nc"
file_out = "/net/ether/data/proteo1/mauger/Data/SLA/v4/final_product/final_prod_filtered_nonan.nc"
bathy_file = "/net/ether/data/proteo1/mauger/Data/Bathy/Bathy_1_30_EASE.nc" #Bathymetry file must be interpolated on the same grid as the SLA product (ease2 grid)


variables = []

D = Dataset(file_in)
t = D["time"][:]
SLA = D["sla"][:]
lon = D["longitude"][:].data
lat = D["latitude"][:].data
D.close()

kernel = Gaussian2DKernel(stddev=2.5)
mask_bathy_val = 0 #threshold for bathymetry mask

D = Dataset("bathy_file")
mdt = D["bathy"][:]
D.close()

mask = np.ones(mdt.shape)
mask[np.isnan(mdt)] = np.nan
mask[mdt>mask_bathy_val] = np.nan

astropy_conv = convolve(SLA[ind,:,:], kernel)
SLA[SLA.mask] = np.nan

##############################
#Filter SLA
##############################

variables = [SLA]
variables_filtered = []
for VAR in variables:
	VAR[VAR.mask] = np.nan
	VAR = VAR.data
	date = [datetime.datetime(1950,1,1) + datetime.timedelta(days= int(x)) for x in t]
	VAR_filtered = np.ma.ones(VAR.shape)*np.nan
	for i in range(len(date)):
		astropy_conv = convolve(VAR[i,:,:], kernel)
		VAR_filtered[i,:,:] = astropy_conv * mask

	VAR_filtered.mask = np.isnan(VAR_filtered)
	variables_filtered.append(VAR_filtered)


##############################
#Dataset creation
##############################

dataset =Dataset(file_out,'w')

dataset.createDimension('X',len(lon))
dataset.createDimension('Y',len(lon[1]))
dataset.createDimension('time',len(t))

dataset.Conventions = "COARDS, CF-1.5" ;
dataset.title = "ERA_5 data on ease2_grid" ;
dataset.GMT_version = "5.2.1 (r15220) [64-bit] [MP]" ;

C = dataset.createVariable('lat', float, ('X','Y'),fill_value =np.nan)
B = dataset.createVariable("lon", float, ('X','Y'),fill_value =np.nan )
A = dataset.createVariable("time", float, ('time'),fill_value =np.nan )
D = dataset.createVariable('SLA', float, ('time','X','Y'),fill_value =np.nan)

B.long_name = "longitude"
C.long_name = "latitude"
D.long_name = "SLA"

B.units = "degrees_east"
C.units = "degrees_north"
D.units = "m"

B[:] = lon
C[:] = lat
A[:] = t
D[:] = np.ma.masked_array(variables_filtered[0],variables_filtered[0] == 0)

dataset.close()

##############################
#Plot snapshots
##############################

ind = t.tolist().index((plot_date - datetime.datetime(1950,1,1)).days)
if plot_unfiltered:
	f = plt.figure()
	ax = functions.ax_map(f,1,1,1,-50)
	cs  = ax.pcolormesh(lon,lat,SLA[:,:,ind],vmin = -0.20,vmax = 0.20, cmap = "bwr", transform = ccrs.PlateCarree())
	plt.colorbar(cs)
if plot_filtered:
	f = plt.figure()
	ax = functions.ax_map(f,1,1,1,-50)
	cs  = ax.pcolormesh(lon,lat,SLA_filtered[:,:,ind],vmin = -0.20,vmax = 0.20, cmap = "bwr", transform = ccrs.PlateCarree())
	plt.colorbar(cs)

plt.show()

