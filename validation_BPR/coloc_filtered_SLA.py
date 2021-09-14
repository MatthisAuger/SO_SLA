#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pdb
import datetime
import sys
import os
from scipy.interpolate import griddata
sys.path.insert(0, '..//support/')
from ease2 import ease_grid
import functions

""" Colocates tide gauge / BPR measurements with a filtered version of the SLA product
run coloc.py /archive/PMC/mauger/2015_2017/Analyse_Objective/2013_2018_v3/SLA_2013_2019_filtered.nc ./fig/BPR_filtered_AO --n=6

""" 






files = ["BPR_filt_15j_10.txt"]
name = ["BPR_Drake_No_Drift_filt_15"]

tgy = [-60.8249]
tgx = [-54.7221]


#~ python coloc_filtered_SLA.py filtered_SLA_file output_figure_folder --n=n

def al_loader(f):
    ''' read filtered SLA dataset '''
    with Dataset(f, 'r') as nc:
        lon = nc.variables['lon'][:]
        lat = nc.variables['lat'][:]
        val = nc.variables['SLA'][:]
        tim = nc.variables['time'][:]
    return lon, lat, tim/365.25+1950., val
    
def get_alt(alh, ix, iy, n=0):
    ''' get the time series at the location, averages n cells around'''
    if n == 0:
        # extract data
        v = alh[ix, iy, :]
        if len(v.compressed())==0:
            return None
        else:
            return v - np.nanmean(v)
    
    else:
        v = alh[ix-n:ix+n, iy-n:iy+n, :].filled(np.nan)
        if np.all(np.isnan(v)):
            print("nok")
            return None
        else:
            v = np.nanmean(v, axis=(0,1))
        
            return v - np.nanmean(v)
        return v - np.nanmean(v)
    
    
    
    
    
    
parser = argparse.ArgumentParser()
parser.add_argument('alt')
parser.add_argument('outdir')
parser.add_argument('--n', type=int, default=0)
args = parser.parse_args()

alx, aly, alt, alh = al_loader(args.alt)
n = args.n


#Get the position of the in situ data on the SLA product grid
grid = ease_grid(orientation='S',
                    res=25000,
                    latitude=-50)
gx, gy = grid.lonlat2index(tgx,tgy, positive = True)


for ix in range(len(gx)):
	figname = '%s/%s_%s.png' %(args.outdir, name[ix],n)
	resname = 'res/%s_%s.txt' %(name[ix],n)
	fig = plt.figure(figsize=(15,10))
	ax1 = fig.add_axes([0.1,0.1,0.8,0.7])
	ax2 = fig.add_axes([0.75,0.67,0.3,0.3], 
	    projection=ccrs.Orthographic(central_latitude=-60.))
	ax2.set_global()
	ftg = np.loadtxt("data/%s" %files[ix])
	SL = ftg[:,1]*0.001
	T = ftg[:,0]
	SL = SL[T>2010]
	T = T[T>2010]
	if n == 0:
	    #shows the position of the in situ data
	    ax2.scatter(alx[gx[ix],gy[ix]], aly[gx[ix],gy[ix] ], c='red', s=15, zorder=100, transform=ccrs.PlateCarree())
	else:
	    #shows the position of the in situ data and the n cells around
	    ax2.scatter(alx[gx[ix]-n:gx[ix]+n,gy[ix]-n:gy[ix]+n ], aly[gx[ix]-n:gx[ix]+n,gy[ix]-n:gy[ix]+n ], c='red', s=15, zorder=100, transform=ccrs.PlateCarree())
	alt_h = get_alt(alh, gx[ix], gy[ix], n=n)
	if alt_h is not None:
	    ax1.plot([datetime.datetime(2013,1,1) + datetime.timedelta(days = int((X-2013)*365.25)) for X in alt], alt_h-0.01, 'r-', lw=3, label='altimetry')

	date_SIC = np.array([1950+x/365.25 for x in np.loadtxt("data/SIC_BPR_DATE.txt")])
	SIC_BPR = np.loadtxt("SIC_BPR.txt")
	date_SIC = date_SIC[SIC_BPR>0.1]
	date_SIC = date_SIC[date_SIC<2014]
	ax1.axvspan(datetime.datetime(2013,1,1) + datetime.timedelta(days = int((np.nanmin(date_SIC)-2013)*365.25)), datetime.datetime(2013,1,1) + datetime.timedelta(days = int((np.nanmax(date_SIC)-2013)*365.25)), alpha=0.5, color='grey')
	ax1.spines['top'].set_visible(False)
	ax1.spines['right'].set_visible(False)
	ax1.grid()
	ax2.background_patch.set_alpha(0.5)
	ax1.tick_params(axis='both', which='major', labelsize=20)
	ax1.legend(fontsize=24)
	ax1.set_ylabel('sea level (m)', fontsize=24)
	ax1.set_xlim(datetime.datetime(2013,1,1),datetime.datetime(2014,1,1))
	fig.savefig(figname, dpi=300)
	plt.close(fig)

	#B = np.vstack((alt,alt_h)).T
	#np.savetxt(resname,B)
	#T = T[~SL.mask]
	#SL = SL[~SL.mask]
	#SL_BPR = griddata(T,np.array(SL),np.array(alt[:235]))
	#alt_h_corr = alt_h[:235]
	#alt_h_corr = alt_h_corr[SL_BPR>-0.7]
	#SL_BPR = SL_BPR[SL_BPR>-0.7]
	#print(name[ix], ' n = ', n, ' corr = ',np.corrcoef(SL_BPR,alt_h_corr)[1,0])
    

