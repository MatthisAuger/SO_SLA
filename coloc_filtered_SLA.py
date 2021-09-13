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
sys.path.insert(0, '/archive/PMC/mauger/support/')
from ease2 import ease_grid
import functions

#~ files = ["Rothera.txt","Scott.txt","Faraday.txt","Dumont_d_Urville.txt","King_Edward.txt","Syowa.txt","Roberts.txt","BPR_Drake_Drift.txt","BPR_Drake_No_Drift.txt","Bluff.txt"]
files = ["BPR_Drake_Drift.txt","BPR_filt_15j_10.txt","drake_south_deep_data_detrended_monthly.txt","drake_south_deep_data_monthly.txt"]
#~ name = ["Rothera","Scott","Faraday","Dumont_d_Urville","King_Edward","Syowa","Roberts","BPR_Drake_Drift","BPR_Drake_No_Drift.txt","Bluff"]
name = ["BPR_Drake_Drift","BPR_Drake_No_Drift_filt_15","drake_south_deep_data_detrended_monthly","drake_south_deep_data_monthly"]

#~ files = ["Syowa.txt","Cape_Roberts.txt"]
#~ tgy = [-67.571265,-77.850100,-65.246233,-66.66510,-54.283292,-69.000000, -77.03392,-60.8249,-60.8249,-46.597650]
tgy = [-60.8249,-60.8249,-60.8249,-60.8249]
#~ tgy = [-69.000000, -77.03392]
#~ tgx = [-68.12979,166.76900,-64.257417,140.001998,-36.496756,39.566667,163.19123,-54.7221,-54.7221,168.345400]
tgx = [-54.7221,-54.7221,-54.7221,-54.7221]
#~ tgx = [39.566667,163.19123]

#~ BatchFerme "./coloc.py /archive/PMC/mauger/2015_2017/Analyse_Objective/2013_2018_v3/AO_Gridded_ALL_23101_24989.nc ./fig/BPR --n=6" --memory 10G

#~ BatchFerme "./coloc.py /archive/PMC/mauger/2015_2017/Analyse_Objective/2013_2018_v3/SLA_2013_2019_filtered.nc ./fig/BPR_filtered_AO --n=6" --memory 10G

def tg_loader(f):
    ''' read dataset '''
    with Dataset(f, 'r') as nc:
        lon = nc.variables['longitude'][:]
        lat = nc.variables['latitude'][:]
        val = nc.variables['ssh'][:]
        tim = nc.variables['time'][:]
    return lon, lat, tim/365.25+1950., val

"""def al_loader(f):
    ''' read dataset '''
    with Dataset(f, 'r') as nc:
        lon = nc.variables['NbLongitudes'][:]
        lat = nc.variables['NbLatitudes'][:]
        #~ val = nc.variables['Grid_0001'][:,:,::30]
        val = nc.variables['Grid_0001'][:]
        var = nc.variables["Variance"][:]
        #~ tim = nc.variables['Time'][::30]
        tim = nc.variables['Time'][:]
    return lon, lat, tim/365.25+1950., val,var
"""
def al_loader(f):
    ''' read dataset '''
    with Dataset(f, 'r') as nc:
        lon = nc.variables['lon'][:]
        lat = nc.variables['lat'][:]
        #~ val = nc.variables['Grid_0001'][:,:,::30]
        val = nc.variables['SLA'][:]
        #var = nc.variables["Variance"][:]
        #~ tim = nc.variables['Time'][::30]
        tim = nc.variables['time'][:]
    return lon, lat, tim/365.25+1950., val
    
def get_alt(alh, ix, iy, n=0):
    if n == 0:

        # extract data
        v = alh[ix, iy, :]
        #~ v = alh[:,ix, iy]
        if len(v.compressed())==0:
            return None
        else:
            return v - np.nanmean(v)
    
    else:
        #~ v = alh[ix-n:ix+n, iy-n:iy+n, :]
        #~ print(v)
        v = alh[ix-n:ix+n, iy-n:iy+n, :].filled(np.nan)
        #~ print(v)
        #~ v = alh[:,ix-n:ix+n, iy-n:iy+n].filled(np.nan)
        
        if np.all(np.isnan(v)):
            print("nok")
            return None
        else:
            v = np.nanmean(v, axis=(0,1))
        
            return v - np.nanmean(v)

        v = np.nanmean(v, axis=(0,1))
        #~ print(np.ma.mean(v))
        #~ v = np.ma.mean(v, axis=(0,1))
        #~ print(v)
        #~ print(v - np.namean(v))
        #~ print(np.ma.mean(v))
        #~ print(np.nanmean(v))
        #~ print(v[~np.isnan(v)])
    
        return v - np.nanmean(v)
    
    
    
    
    
    
parser = argparse.ArgumentParser()
parser.add_argument('alt')
parser.add_argument('outdir')
parser.add_argument('--n', type=int, default=0)
args = parser.parse_args()

#alx, aly, alt, alh, var = al_loader(args.alt)
alx, aly, alt, alh = al_loader(args.alt)

grid = ease_grid(orientation='S',
                    res=25000,
                    latitude=-50)
#~ tgx = [110.516667,110.516667]
#~ tgy = [-66.266667,-66.266667]

gx, gy = grid.lonlat2index(tgx,tgy, positive = True)

for n in [0,1,2,3,4,6,8,10]:
#for n in [6]:
    """
    figvarname = '%s/var_%s.png' %(args.outdir,n)
    figvar = plt.figure(figsize=(15,15))
    ax_var = functions.ax_map(figvar,1,1,1,-50)
    cs = ax_var.pcolormesh(alx,aly,var, cmap = "viridis", vmin = 0, vmax = 0.02,transform = ccrs.PlateCarree())
    ax_var.set_title("SLA Variance (m2)")
    plt.colorbar(cs)
    """
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
        if name[ix] != "Rothera":
		SL = np.ma.masked_array(SL,SL<-0.6)
        ax1.plot([datetime.datetime(2013,1,1) + datetime.timedelta(days = int((X-2013)*365.25)) for X in T], SL - np.nanmean(SL), 
            'k-', 
            lw=3,
            label='Bottom Pressure Recorder')
        #ax1.set_title(name[ix]+"_%s" %n, fontsize = 25)
        ax2.add_feature(cfeature.LAND, zorder=30, edgecolor='#737373', facecolor='#737373')
        #~ ax2.scatter(tgx[ix], tgy[ix], c='red', s=75, zorder=100, transform=ccrs.PlateCarree())
        if n == 0:
            ax2.scatter(alx[gx[ix],gy[ix]], aly[gx[ix],gy[ix] ], c='red', s=15, zorder=100, transform=ccrs.PlateCarree())
        else:

            ax2.scatter(alx[gx[ix]-n:gx[ix]+n,gy[ix]-n:gy[ix]+n ], aly[gx[ix]-n:gx[ix]+n,gy[ix]-n:gy[ix]+n ], c='red', s=15, zorder=100, transform=ccrs.PlateCarree())

        #ax_var.scatter(tgx[ix],tgy[ix],s = 150,c = np.nanvar(SL), cmap = "viridis", vmin = 0, vmax = 0.02, transform = ccrs.PlateCarree(), zorder = 150,edgecolors='black')
        #~ print(grid.lons[gx[ix]-n:gx[ix]+n,gy[ix]-n:gy[ix]+n ],grid.lats[gx[ix]-n:gx[ix]+n,gy[ix]-n:gy[ix]+n ])
        alt_h = get_alt(alh, gx[ix], gy[ix], n=n)
        #~ print(tgx[ix],tgy[ix])
        #~ print(alh.shape,gx.shape)
        #~ print(alt_h.shape,gx.shape)
        #~ pdb.set_trace()

        #~ print(len(alt_h),alt_h)
        if alt_h is not None:
            ax1.plot([datetime.datetime(2013,1,1) + datetime.timedelta(days = int((X-2013)*365.25)) for X in alt], alt_h-0.01, 'r-', lw=3, label='altimetry')

        date_SIC = np.array([1950+x/365.25 for x in np.loadtxt("SIC_BPR_DATE.txt")])
        SIC_BPR = np.loadtxt("SIC_BPR.txt")
        date_SIC = date_SIC[SIC_BPR>0.1]
        date_SIC = date_SIC[date_SIC<2014]
        ax1.axvspan(datetime.datetime(2013,1,1) + datetime.timedelta(days = int((np.nanmin(date_SIC)-2013)*365.25)), datetime.datetime(2013,1,1) + datetime.timedelta(days = int((np.nanmax(date_SIC)-2013)*365.25)), alpha=0.5, color='grey')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.grid()
        ax2.background_patch.set_alpha(0.5)
        #ax1.xticks(fontsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        #ax1.yticks(fontsize=20)
        ax1.legend(fontsize=24)
        ax1.set_ylabel('sea level (m)', fontsize=24)
        #ax1.set_xlabel('time (yr)', fontsize=24)
        ax1.set_xlim(datetime.datetime(2013,1,1),datetime.datetime(2014,1,1))

        fig.savefig(figname, dpi=300)
        plt.close(fig)
        B = np.vstack((alt,alt_h)).T
        np.savetxt(resname,B)
        T = T[~SL.mask]
        SL = SL[~SL.mask]
        SL_BPR = griddata(T,np.array(SL),np.array(alt[:235]))
        alt_h_corr = alt_h[:235]
        alt_h_corr = alt_h_corr[SL_BPR>-0.7]
        SL_BPR = SL_BPR[SL_BPR>-0.7]
        print(name[ix], ' n = ', n, ' corr = ',np.corrcoef(SL_BPR,alt_h_corr)[1,0])
	#pdb.set_trace()
    #figvar.savefig(figvarname, dpi=300)
    #plt.close(figvar)
    

"""
('BPR_Drake_Drift', ' n = ', 2, ' corr = ', 0.026740365508224343)
('BPR_Drake_No_Drift_filt_15', ' n = ', 2, ' corr = ', 0.22440999555890384)
('drake_south_deep_data_detrended_monthly', ' n = ', 2, ' corr = ', 0.15549177126553812)
('drake_south_deep_data_monthly', ' n = ', 2, ' corr = ', -0.055832651614790806)
('BPR_Drake_Drift', ' n = ', 3, ' corr = ', 0.24792292089266946)
('BPR_Drake_No_Drift_filt_15', ' n = ', 3, ' corr = ', 0.4798482477092347)
('drake_south_deep_data_detrended_monthly', ' n = ', 3, ' corr = ', 0.2810268755599502)
('drake_south_deep_data_monthly', ' n = ', 3, ' corr = ', 0.12492465345561068)
('BPR_Drake_Drift', ' n = ', 4, ' corr = ', 0.4249714955369347)
('BPR_Drake_No_Drift_filt_15', ' n = ', 4, ' corr = ', 0.5960998991758288)
('drake_south_deep_data_detrended_monthly', ' n = ', 4, ' corr = ', 0.275888656921823)
('drake_south_deep_data_monthly', ' n = ', 4, ' corr = ', 0.2140169414749643)
('BPR_Drake_Drift', ' n = ', 6, ' corr = ', 0.4938407461259947)
('BPR_Drake_No_Drift_filt_15', ' n = ', 6, ' corr = ', 0.723436183614367)
('drake_south_deep_data_detrended_monthly', ' n = ', 6, ' corr = ', 0.4789931245418067)
('drake_south_deep_data_monthly', ' n = ', 6, ' corr = ', 0.3232588588873284)
('BPR_Drake_Drift', ' n = ', 8, ' corr = ', 0.514056773157357)
('BPR_Drake_No_Drift_filt_15', ' n = ', 8, ' corr = ', 0.7520080012666687)
('drake_south_deep_data_detrended_monthly', ' n = ', 8, ' corr = ', 0.5588925902809968)
('drake_south_deep_data_monthly', ' n = ', 8, ' corr = ', 0.4421201036410209)
('BPR_Drake_Drift', ' n = ', 10, ' corr = ', 0.5567002425443127)
('BPR_Drake_No_Drift_filt_15', ' n = ', 10, ' corr = ', 0.7528460236660954)
('drake_south_deep_data_detrended_monthly', ' n = ', 10, ' corr = ', 0.5483404133393527)
('drake_south_deep_data_monthly', ' n = ', 10, ' corr = ', 0.4653347517725505)"""

    
#run coloc.py /archive/PMC/mauger/2015_2017/Analyse_Objective/2013_2018_v3/SLA_2013_2019_filtered.nc ./fig/BPR_filtered_AO --n=6


('BPR_Drake_Drift', ' n = ', 0, ' corr = ', 0.34531978420371134)
('BPR_Drake_No_Drift_filt_15', ' n = ', 0, ' corr = ', 0.5460198514726127)
('drake_south_deep_data_detrended_monthly', ' n = ', 0, ' corr = ', 0.4049107686373806)
('drake_south_deep_data_monthly', ' n = ', 0, ' corr = ', 0.28229352835445054)
('BPR_Drake_Drift', ' n = ', 1, ' corr = ', 0.3858674959855218)
('BPR_Drake_No_Drift_filt_15', ' n = ', 1, ' corr = ', 0.5838455836550027)
('drake_south_deep_data_detrended_monthly', ' n = ', 1, ' corr = ', 0.38551641208301135)
('drake_south_deep_data_monthly', ' n = ', 1, ' corr = ', 0.2638362354370995)
('BPR_Drake_Drift', ' n = ', 2, ' corr = ', 0.4195844082940418)
('BPR_Drake_No_Drift_filt_15', ' n = ', 2, ' corr = ', 0.6248835400527477)
('drake_south_deep_data_detrended_monthly', ' n = ', 2, ' corr = ', 0.41334426736037694)
('drake_south_deep_data_monthly', ' n = ', 2, ' corr = ', 0.3026451806038319)
('BPR_Drake_Drift', ' n = ', 3, ' corr = ', 0.46148977935274876)
('BPR_Drake_No_Drift_filt_15', ' n = ', 3, ' corr = ', 0.676113662824349)
('drake_south_deep_data_detrended_monthly', ' n = ', 3, ' corr = ', 0.4508472847990183)
('drake_south_deep_data_monthly', ' n = ', 3, ' corr = ', 0.3567038500472008)
('BPR_Drake_Drift', ' n = ', 4, ' corr = ', 0.49735515670842356)
('BPR_Drake_No_Drift_filt_15', ' n = ', 4, ' corr = ', 0.7209447232904244)
('drake_south_deep_data_detrended_monthly', ' n = ', 4, ' corr = ', 0.48936799203742704)
('drake_south_deep_data_monthly', ' n = ', 4, ' corr = ', 0.4139456531822142)
('BPR_Drake_Drift', ' n = ', 6, ' corr = ', 0.5297152824756586)
('BPR_Drake_No_Drift_filt_15', ' n = ', 6, ' corr = ', 0.767038047580257)
('drake_south_deep_data_detrended_monthly', ' n = ', 6, ' corr = ', 0.5519713294142911)
('drake_south_deep_data_monthly', ' n = ', 6, ' corr = ', 0.505458462404776)
('BPR_Drake_Drift', ' n = ', 8, ' corr = ', 0.5319280054048187)
('BPR_Drake_No_Drift_filt_15', ' n = ', 8, ' corr = ', 0.7778962484049277)
('drake_south_deep_data_detrended_monthly', ' n = ', 8, ' corr = ', 0.5899995518957064)
('drake_south_deep_data_monthly', ' n = ', 8, ' corr = ', 0.5486785950341924)
('BPR_Drake_Drift', ' n = ', 10, ' corr = ', 0.5229233338628118)
('BPR_Drake_No_Drift_filt_15', ' n = ', 10, ' corr = ', 0.7806987625881109)
('drake_south_deep_data_detrended_monthly', ' n = ', 10, ' corr = ', 0.6169172830579974)
('drake_south_deep_data_monthly', ' n = ', 10, ' corr = ', 0.554405223886103)

    

