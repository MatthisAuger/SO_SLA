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
sys.path.insert(0, '../support/')
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


file_unfiltered = "/net/ether/data/proteo1/mauger/Data/SLA/v4/final_product/dt_antarctic_multimission_sea_level_uv_20130401_20193107.nc"
file_filtered = "/net/ether/data/proteo1/mauger/Data/SLA/v4/final_product/final_prod_filtered_nonan.nc"

variables = []

#Unfiltered dataset
D = Dataset(file_filtered)
t = D["time"][:]
SLA = D["sla"][:]
lon = D["longitude"][:].data
lat = D["latitude"][:].data
D.close()

#Unfiletered dataset
dataset =Dataset(file_unfiltered)
SLA_f = dataset["SLA"][:]
dataset.close()

plt.figure()

for d in [120,400,600,900,100]:
	gA = SLA[d]
	gB = SLA_f[d]

	fsx = 1/25
	fsy = 1/25

	x_start = 40
	x_stop = 150

	y_start = 40
	y_stop = 150

	def psd(g,fsx,fsy,x_start,x_stop,y_start,y_stop):


		g_slice = g[x_start:x_stop,y_start:y_stop]
		

		f,pw = welch(g_slice[:,0],25,scaling = 'density')
		shp = (len(f),len(g_slice[1]))

		x_freqs = np.empty(shp)
		x_power = np.empty(shp)

		for i in range(0, shp[1]):
			f, pw = welch(g_slice[:,i], fsx, scaling='density')
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

	plt.loglog(1./f_a,pw_a)
	plt.loglog(1./f_a,pw_a)

	plt.loglog(1./f_b,pw_b)

plt.grid()
plt.show()

plt.loglog(1./f_a,pw_a/f_a)
plt.loglog(1./f_a,pw_a/f_a)

plt.loglog(1./f_b,pw_b/f_b)

plt.grid()
plt.show()
