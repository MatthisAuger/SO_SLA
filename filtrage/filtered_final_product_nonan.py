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
d = datetime.datetime(2019,3,15)

variables = []

D = Dataset("/net/ether/data/proteo1/mauger/Data/SLA/v4/final_product/dt_antarctic_multimission_sea_level_uv_20130401_20193107.nc")
t = D["time"][:]
SLA = D["sla"][:]

lon = D["longitude"][:].data
lat = D["latitude"][:].data
D.close()

kernel = Gaussian2DKernel(stddev=2.5)

mask_bathy_val = 0

D = Dataset("/net/ether/data/proteo1/mauger/Data/Bathy/Bathy_1_30_EASE.nc")

mdt = D["bathy"][:]
D.close()

mask = np.ones(mdt.shape)
mask[np.isnan(mdt)] = np.nan
mask[mdt>mask_bathy_val] = np.nan



ind = t.tolist().index((d - datetime.datetime(1950,1,1)).days)
f = plt.figure()
ax = functions.ax_map(f,1,1,1,-50)
cs = ax.pcolormesh(lon,lat,SLA[ind,:,:], cmap = "bwr",vmin = -0.2,vmax = 0.2, transform = ccrs.PlateCarree())
plt.colorbar(cs)




astropy_conv = convolve(SLA[ind,:,:], kernel)

f = plt.figure()
ax = functions.ax_map(f,1,1,1,-50)
cs = ax.pcolormesh(lon,lat,astropy_conv*mask, cmap = "bwr",vmin = -0.2,vmax = 0.2, transform = ccrs.PlateCarree())
plt.colorbar(cs)




SLA[SLA.mask] = np.nan
V=SLA[ind,:,:].copy()
V[np.isnan(SLA[ind,:,:])]=0
VV=ndimage.gaussian_filter(V,sigma=2.5)

W=0*SLA[ind,:,:].copy()+1
W[np.isnan(SLA[ind,:,:])]=0
WW=ndimage.gaussian_filter(W,sigma=2.5)

VAR_filtered=np.ma.masked_array(VV/WW,np.isnan(SLA[ind,:,:]))

f = plt.figure()
ax = functions.ax_map(f,1,1,1,-50)
cs = ax.pcolormesh(lon,lat,VAR_filtered, cmap = "bwr",vmin = -0.2,vmax = 0.2, transform = ccrs.PlateCarree())
plt.colorbar(cs)
plt.show()





variables = [SLA]




variables_filtered = []
for VAR in variables:
	VAR[VAR.mask] = np.nan
	VAR = VAR.data
	date = [datetime.datetime(1950,1,1) + datetime.timedelta(days= int(x)) for x in t]


	VAR_filtered = np.ma.ones(VAR.shape)*np.nan
	for i in range(len(date)):
	#for i in range(1):
		
		astropy_conv = convolve(VAR[i,:,:], kernel)
		

		VAR_filtered[i,:,:] = astropy_conv * mask

	VAR_filtered.mask = np.isnan(VAR_filtered)


	variables_filtered.append(VAR_filtered)




f = plt.figure()
ax = functions.ax_map(f,1,1,1,-50)
cs = ax.pcolormesh(lon,lat,variables_filtered[0][ind,:,:], cmap = "bwr",vmin = -0.2,vmax = 0.2, transform = ccrs.PlateCarree())
plt.colorbar(cs)
plt.show()

f = plt.figure()
ax = functions.ax_map(f,1,1,1,-50)
cs = ax.pcolormesh(lon,lat,variables_filtered[0][5], cmap = "bwr",vmin = -0.2,vmax = 0.2, transform = ccrs.PlateCarree())
plt.colorbar(cs)
plt.show()


#dataset =Dataset("/net/ether/data/proteo1/mauger/Data/SLA/unbiased/monomission/S3A_2013_2019_unbiased_filtered_v2_werr.nc",'w')
dataset =Dataset("/net/ether/data/proteo1/mauger/Data/SLA/v4/final_product/final_prod_filtered_nonan.nc",'w')

dataset.createDimension('X',len(lon))
dataset.createDimension('Y',len(lon[1]))
dataset.createDimension('time',len(t))

dataset.Conventions = "COARDS, CF-1.5" ;
dataset.title = "ERA_5 data on ease2_grid" ;
dataset.GMT_version = "5.2.1 (r15220) [64-bit] [MP]" ;



C = dataset.createVariable('lat', float, ('X','Y'),fill_value =np.nan)

B = dataset.createVariable("lon", float, ('X','Y'),fill_value =np.nan )
A = dataset.createVariable("time", float, ('time'),fill_value =np.nan )
#D = dataset.createVariable('SLA', float, ('time','lon','lat'),fill_value =np.nan)
D = dataset.createVariable('SLA', float, ('time','X','Y'),fill_value =np.nan)


B.long_name = "longitude"
C.long_name = "latitude"
D.long_name = "SLA"



B.units = "degrees_east"
C.units = "degrees_north"


B[:] = lon
C[:] = lat
A[:] = t
D[:] = np.ma.masked_array(variables_filtered[0],variables_filtered[0] == 0)

dataset.close()

"""
f = plt.figure()
ax = functions.ax_map(f,1,1,1,-50)
cs  = ax.pcolormesh(lon,lat,SLA[:,:,2],vmin = -0.20,vmax = 0.20, cmap = "bwr", transform = ccrs.PlateCarree())

f = plt.figure()
ax = functions.ax_map(f,1,1,1,-50)
cs  = ax.pcolormesh(lon,lat,SLA_filtered[:,:,2],vmin = -0.20,vmax = 0.20, cmap = "bwr", transform = ccrs.PlateCarree())

plt.show()
"""

gA = SLA[46]
gB = np.ma.masked_array(variables_filtered[0],variables_filtered[0] == 0)[46]

fsx = 1/25
fsy = 1/25

x_start = 70
x_stop = 120

y_start = 70
y_stop = 120

def psd(g,fsx,fsy,x_start,x_stop,y_start,y_stop):


	g_slice = g[x_start:x_stop,y_start:y_stop]
	

	f,pw = welch(g_slice[:,0],25,scaling = 'spectrum')
	shp = (len(f),len(g_slice[1]))

	x_freqs = np.empty(shp)
	x_power = np.empty(shp)
	#pdb.set_trace()
	for i in range(0, shp[1]):
		f, pw = welch(g_slice[:,i], fsx, scaling='spectrum')
		x_freqs[:,i] = f
		x_power[:,i] = pw

	# echantillonne continuement les fréquences
	f_min, f_max = np.min(x_freqs), np.max(x_freqs)
	f_interp = np.linspace(f_min, f_max, 100)

	# interpole chaque spectre sur les fréquences voulues
	x_interp_power = np.empty((shp[1], 100))
	for i in range(0, shp[1]):
		interpolator = interp1d(x_freqs[:,i], x_power[:,i], fill_value=0., bounds_error=False)
		x_interp_power[i,:] = interpolator(f_interp)

	# maintenant on peut moyenner
	mean_power = np.mean(x_interp_power, axis=0)
	
	return f_interp, mean_power

f_a,pw_a = psd(gA,fsx,fsy,x_start,x_stop,y_start,y_stop)
f_b,pw_b = psd(gB,fsx,fsy,x_start,x_stop,y_start,y_stop)

plt.loglog(1./f_a,pw_a/f_a)
plt.loglog(1./f_a,pw_a/f_a-3)

plt.loglog(1./f_b,pw_b/f_b)

plt.grid()
plt.show()


