#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import argparse
import datetime
from cartopy import feature
from datetime import datetime, timedelta
from netCDF4 import Dataset
from matplotlib.colors import LogNorm
import matplotlib.path as mpath

import sys
import os
import logging
import cmocean

sys.path.insert(0, '/.autofs/home/mauger/support/')
from ease2 import ease_grid

#~ import functions
#~ import imp 
#~ imp.reload(functions)

def to_julian(date):
	date_julian = (date - datetime(1950,1,1)).days
	return(date_julian)
def view_ncfile(file, lon = "NbLongitudes", lat = "NbLatitudes", val = "Grid_0001", vmin = -0.4, vmax = 0.4, cmap = "bwr", log = False, transpose = True, Title = "",Center = False, norm = None, cbar = True):
	dataset = Dataset(file) 
	Lats = dataset.variables[lat][:]
	Lons = dataset.variables[lon][:]
	A = dataset.variables[val][:]
	dataset.close()
	if transpose:
		A = A.T
	plt.gcf()
	plt.gca()
	f = plt.figure(figsize = (10,10))
	ax = f.add_subplot(1,1,1,projection=ccrs.SouthPolarStereo(central_longitude=0.))
	ax.set_extent([-180, 180., -90., -50.], crs=ccrs.PlateCarree())
	ax.add_feature(feature.LAND, facecolor = "grey", zorder=100)
	ax.set_title(Title)	
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	if Center == True:
		#~ A = A - np.nanmean(A)
		print(np.ma.median(A))
		A = A - np.nanmean(A)
		
	if log:
		cs = ax.pcolormesh(Lons,Lats,A ,  transform=ccrs.PlateCarree(),norm = LogNorm(vmin = vmin, vmax = vmax),cmap=cmap)
	else:
		cs = ax.pcolormesh(Lons,Lats,A , cmap=cmap, transform=ccrs.PlateCarree(), vmin = vmin, vmax =vmax)
	
	if cbar :
		plt.colorbar(cs)
	return(f,ax)
	
def affect_xyz_to_ease2_grid(lon,lat, vals, resolution = 25000):
	Gridref = ease_grid(orientation = 'S', latitude = -40, res = resolution)
	G = np.ones(Gridref.lats.shape)*np.nan
	G[Gridref.lonlat2index(lon,lat)] = grid
	return(G,Gridref)
	
def ax_map(f, subploti = 1, subplotj = 1, subplotn = 1,latlim = -50,continent = "yes", fc = 'k'):
	ax = f.add_subplot(subploti,subplotj,subplotn,projection=ccrs.SouthPolarStereo(central_longitude=0.), fc = fc)
	ax.set_extent([-180, 180., -90., latlim], crs=ccrs.PlateCarree())
	if continent == "yes":
		ax.add_feature(feature.LAND, zorder=100, facecolor = "grey")
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	return(ax)

def ax_map_grid_sub(f, grid_sub,latlim = -50,continent = "yes", fc = 'k'):
	ax = f.add_subplot(grid_sub,projection=ccrs.SouthPolarStereo(central_longitude=0.), fc = fc)
	ax.set_extent([-180, 180., -90., latlim], crs=ccrs.PlateCarree())
	if continent == "yes":
		ax.add_feature(feature.LAND, zorder=100, facecolor = "grey")
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	return(ax)
	
def is_job_running(n):
    os.system('qstat > /tmp/qstat.log')
    with open('/tmp/qstat.log', 'r') as f:
        lines = f.readlines()
        for l in lines[2:]:
            logging.info(l)
            j = int(l.split()[0])
            if j==n:
                return True
    return False



def SIC_contour(ax, val, date, color = 'k', linewidth = 2):
	D = Dataset("/net/ether/data/proteo1/mauger/Data/SIC/SIC_ease2_v3.nc")

	lon = D["lon"][:]
	lat = D["lat"][:]
	time = D["time"][:]
	
	ind = np.arange(len(time))
	
	SIC = D['SIC'][ind[time== date][0],:,:].T

	#ax.contour(lon,lat,SIC,val,color = color, transform = ccrs.PlateCarree())
	ax.contour(np.ma.masked_array(lon,mask =lon<0),np.ma.masked_array(lat,mask = lon<0),np.ma.masked_array(SIC,mask = lon<0), val,transform = ccrs.PlateCarree(), c = color, linewidth = linewidth)
	ax.contour(np.ma.masked_array(lon,mask =lon>0),np.ma.masked_array(lat,mask = lon>0),np.ma.masked_array(SIC,mask = lon>0), val,transform = ccrs.PlateCarree(), c = color, linewidth = linewidth)
	"""A = plt.figure()
	ax_null = ax_map(A,1,1,1,-50)
	cs = plt.contour(lon,lat,SIC,val, transform = ccrs.PlateCarree())
	dat0 = cs.allsegs[0]
	plt.close(A)

	dat_len = [len(x) for x in dat0]
	cont = dat0[np.argmax(dat_len)]

	loncont = cont[:,0]
	latcont = cont[:,1]

	latcontpos = latcont[loncont>=-0.5]
	loncontpos = loncont[loncont>=-0.5]
	latcontneg = latcont[loncont<=0.5]
	loncontneg = loncont[loncont<=0.5]
	ax.plot(loncontpos, latcontpos,transform = ccrs.PlateCarree(),c = color,linewidth = linewidth)
	ax.plot(loncontneg[:np.argmax(loncontneg)+1], latcontneg[:np.argmax(loncontneg)+1],transform = ccrs.PlateCarree(),c = color,linewidth = linewidth)
	ax.plot(loncontneg[np.argmax(loncontneg)+1:], latcontneg[np.argmax(loncontneg)+1:],transform = ccrs.PlateCarree(),c = color,linewidth = linewidth)
	ax.plot([loncontneg[-1],loncontneg[1]], [latcontneg[-1],latcontneg[1]],transform = ccrs.PlateCarree(),c = color,linewidth = linewidth)
	ax.plot([np.max(loncont),np.min(loncont)+360], [latcont[np.argmax(loncont)],latcont[np.argmin(loncont)]],transform = ccrs.PlateCarree(),c = color,linewidth = linewidth)"""

def bathy_contour(ax,val,color = 'grey',linewidth = 3,linestyles = "--"):
	D = Dataset("/net/ether/data/proteo1/mauger/Data/Bathy/Bathy_1_30_EASE.nc")
	lon_A = D["lon"][:]
	lat_A = D["lat"][:]
	MDT_A = D["bathy"][:]
	D.close()
        

	A = plt.figure()
	ax_null = ax_map(A,1,1,1,-50)
	cs = plt.contour(lon_A,lat_A,MDT_A,[val], transform = ccrs.PlateCarree())
	dat0 = cs.allsegs[0]
	plt.close(A)

	dat_len = [len(x) for x in dat0]
	cont = dat0[np.argmax(dat_len)]

	loncont = cont[:,0]
	latcont = cont[:,1]

	latcontpos = latcont[loncont>=-0.5]
	loncontpos = loncont[loncont>=-0.5]
	latcontneg = latcont[loncont<=0.5]
	loncontneg = loncont[loncont<=0.5]
	#ax.plot(loncontpos, latcontpos,transform = ccrs.PlateCarree(),c = color,linestyle = linestyles,linewidth = linewidth)
	#ax.plot(loncontneg[:np.argmax(loncontneg)+1], latcontneg[:np.argmax(loncontneg)+1],transform = ccrs.PlateCarree(),c = color,linestyle = linestyles,linewidth = linewidth)
	#ax.plot(loncontneg[np.argmax(loncontneg)+1:], latcontneg[np.argmax(loncontneg)+1:],transform = ccrs.PlateCarree(),c = color,linestyle = linestyles,linewidth = linewidth)
	#ax.plot([loncontneg[-1],loncontneg[1]], [latcontneg[-1],latcontneg[1]],transform = ccrs.PlateCarree(),c = color,linestyle = linestyles,linewidth = linewidth)
	#ax.plot([np.max(loncont),np.min(loncont)+360], [latcont[np.argmax(loncont)],latcont[np.argmin(loncont)]],transform = ccrs.PlateCarree(),c = color,linestyle = linestyles,linewidth = linewidth)



	bat1 = np.ma.masked_array(MDT_A,mask = lon_A<0.5)
	bat2 = np.ma.masked_array(MDT_A,mask = lon_A>0.5)

	cs1 = ax.contour(lon_A,lat_A,bat1,[val],colors = color, transform = ccrs.PlateCarree(),linewidths = linewidth)
	cs2 = ax.contour(lon_A,lat_A,bat2,[val],colors = color, transform = ccrs.PlateCarree(),linewidths = linewidth)
	return(cs1,cs2)

def bathy_contour_shelftype(ax,val=-1200,linewidth = 3):
	D = Dataset("/net/ether/data/proteo1/mauger/Data/Bathy/Bathy_1_30_EASE.nc")
	bat_lon = D["lon"][:]
	bat_lat = D["lat"][:]
	bat = D["bathy"][:]
	D.close()
        
	
	
	bat1 = np.ma.masked_array(bat,mask = bat_lon<=0)
	bat2 = np.ma.masked_array(bat,mask = bat_lon>0)

	cs1 = ax.contour(bat_lon,bat_lat,bat1,[val],c = None, transform = ccrs.PlateCarree())
	cs2 = ax.contour(bat_lon,bat_lat,bat2,[val],c = None, transform = ccrs.PlateCarree())
	
	dat1 = cs1.allsegs[0][0]
	dat2 = cs2.allsegs[0][4]
	print(cs2.allsegs)
	#0 fresh, 1 dense, 2 warm
	shelf_type = np.ones(dat1[:,0].shape)*np.nan
	for i in range(len(dat1[:,0])):
		if -180<dat1[i,0]<-120:
			shelf_type[i] = 0
		if -120<dat1[i,0]<-60:
			shelf_type[i] = 2

		if -60<dat1[i,0]<=-30:
			shelf_type[i] = 1
		if -60<dat1[i,0]<=-50 and dat1[i,1]>-68:
			shelf_type[i] = 0
		if -30<dat1[i,0]<=47:
			shelf_type[i] = 0
		if 47<dat1[i,0]<=60:
			shelf_type[i] = 1
		if 60<dat1[i,0]<=100:
			shelf_type[i] = 0
		if 100<dat1[i,0]<=110:
			shelf_type[i] = 2
		if 110<dat1[i,0]<=130:
			shelf_type[i] = 0
		if 130<dat1[i,0]<=180:
			shelf_type[i] = 1
	shelf_type_0= np.copy(dat1[:,0])
	shelf_type_0[shelf_type!=0]=np.nan
	shelf_type_1 = np.copy(dat1[:,0])
	shelf_type_1[shelf_type!=1]=np.nan
	shelf_type_2 = np.copy(dat1[:,0])
	shelf_type_2[shelf_type!=2]=np.nan	
	print(shelf_type_0)
	print(shelf_type_1)
	print(shelf_type_2)
	ax.plot(shelf_type_0,dat1[:,1],color = 'green',transform = ccrs.PlateCarree(),linewidth = linewidth,label = 'fresh shelf')	
	ax.plot(shelf_type_1,dat1[:,1],color = 'purple',transform = ccrs.PlateCarree(),linewidth = linewidth, label = 'dense shelf')	
	ax.plot(shelf_type_2,dat1[:,1],color = 'orange',transform = ccrs.PlateCarree(),linewidth = linewidth, label = 'warm_shelf')	
	
	shelf_type = np.ones(dat2[:,0].shape)*np.nan
	for i in range(len(dat2[:,0])):
		if -180<dat2[i,0]<-120:
			shelf_type[i] = 0
		if -120<dat2[i,0]<-60:
			shelf_type[i] = 2

		if -60<dat2[i,0]<=-30:
			shelf_type[i] = 1
		if -60<dat2[i,0]<=-50 and dat2[i,1]>-68:
			shelf_type[i] = 0
		if -30<dat2[i,0]<=47:
			shelf_type[i] = 0
		if 47<dat2[i,0]<=60:
			shelf_type[i] = 1
		if 60<dat2[i,0]<=100:
			shelf_type[i] = 0
		if 100<dat2[i,0]<=110:
			shelf_type[i] = 2
		if 110<dat2[i,0]<=130:
			shelf_type[i] = 0
		if 130<dat2[i,0]<=180:
			shelf_type[i] = 1
	shelf_type_0= np.copy(dat2[:,0])
	shelf_type_0[shelf_type!=0]=np.nan
	shelf_type_1 = np.copy(dat2[:,0])
	shelf_type_1[shelf_type!=1]=np.nan
	shelf_type_2 = np.copy(dat2[:,0])
	shelf_type_2[shelf_type!=2]=np.nan	
	print(shelf_type)
	cs1 =  ax.plot(shelf_type_0,dat2[:,1],color = 'green',transform = ccrs.PlateCarree(),linewidth = linewidth)	
	cs2 = ax.plot(shelf_type_1,dat2[:,1],color = 'purple',transform = ccrs.PlateCarree(),linewidth = linewidth)	
	cs3 = ax.plot(shelf_type_2,dat2[:,1],color = 'orange',transform = ccrs.PlateCarree(),linewidth = linewidth)	
	return(cs1,cs2,cs3)

def is_on_shelf(lon,lat):
	G = ease_grid('S',-50,25000)
	x,y = G.lonlat2index(np.array([lon]),np.array([lat]))
	D = Dataset("/net/ether/data/proteo1/mauger/Data/Bathy/Bathy_1_30_EASE.nc")
	bat = D["bathy"][:]
	D.close()
	print(bat[x,y])
	return(bat[x,y] > -2000)

	

def shelftype(lon,lat):
	if -180<lon<-120:
		shelf_type = "fresh"
	if -120<lon<-60:
		shelf_type = "warm"

	if -60<lon<=-30:
		shelf_type = "dense"
	if -60<lon<=-50 and lat>-68:
		shelf_type = "fresh"
	if -30<lon<=47:
		shelf_type = "fresh"
	if 47<lon<=60:
		shelf_type = "dense"
	if 60<lon<=100:
		shelf_type = "fresh"
	if 100<lon<=110:
		shelf_type = "warm"
	if 110<lon<=130:
		shelf_type = "fresh"
	if 130<lon<=180:
		shelf_type = "warm"
	return(shelf_type)

def mdtmask():
	mask = np.loadtxt("/usr/home/mauger/support/mask_mdt.txt")

	return(mask)

def mdtcontour(ax,val = -180,linestyles = 'dashed',colors = 'black',linewidth = 3):
	D1 = Dataset("/net/ether/data/proteo1/mauger/Data/Armitage/CS2_combined_Southern_Ocean_2011-2016.nc")
	lat_A = D1["Latitude"][:]
	lon_A = D1["Longitude"][:]
	MDT_A = D1["MDT"][:]


	A = plt.figure()
	ax_null = ax_map(A,1,1,1,-50)
	cs = plt.contour(lon_A,lat_A,MDT_A,[val], transform = ccrs.PlateCarree())
	dat0 = cs.allsegs[0]
	plt.close(A)

	dat_len = [len(x) for x in dat0]
	cont = dat0[np.argmax(dat_len)]

	loncont = cont[:,0]
	latcont = cont[:,1]

	latcontpos = latcont[loncont>=-0.5]
	loncontpos = loncont[loncont>=-0.5]
	latcontneg = latcont[loncont<=0.5]
	loncontneg = loncont[loncont<=0.5]
	ax.plot(loncontpos, latcontpos,transform = ccrs.PlateCarree(),c = colors,linestyle = linestyles,linewidth = linewidth)
	ax.plot(loncontneg[:np.argmax(loncontneg)+1], latcontneg[:np.argmax(loncontneg)+1],transform = ccrs.PlateCarree(),c = colors,linestyle = linestyles,linewidth = linewidth)
	ax.plot(loncontneg[np.argmax(loncontneg)+1:], latcontneg[np.argmax(loncontneg)+1:],transform = ccrs.PlateCarree(),c = colors,linestyle = linestyles,linewidth = linewidth)
	ax.plot([loncontneg[-1],loncontneg[1]], [latcontneg[-1],latcontneg[1]],transform = ccrs.PlateCarree(),c = colors,linestyle = linestyles,linewidth = linewidth)
	ax.plot([np.max(loncont),np.min(loncont)+360], [latcont[np.argmax(loncont)],latcont[np.argmin(loncont)]],transform = ccrs.PlateCarree(),c = colors,linestyle = linestyles,linewidth = linewidth)



	#MDT_A_pos = np.ma.masked_array(MDT_A,mask = lon_A<=0)
	#MDT_A_neg = np.ma.masked_array(MDT_A,mask = lon_A>0)
	#ax.contour(lon_A,lat_A,MDT_A_pos,[val],transform = ccrs.PlateCarree(),colors = colors,linestyles = linestyles,linewidths = linewidth)
	#ax.contour(lon_A,lat_A,MDT_A_neg,[val],transform = ccrs.PlateCarree(),colors = colors,linestyles = linestyles,linewidths = linewidth)




def mask_bat_original(val,addlat = 0):
	D = Dataset("/net/ether/data/proteo1/mauger/Data/Bathy/Bathy_1_30_EASE.nc")
	bat_lon = D["lon"][:]
	bat_lat = D["lat"][:]
	bat = D["bathy"][:]
	D.close()

	return(bat<val)


def mask_bat_(val,addlat = 0):
	f = plt.figure()
	ax = ax_map(f,1,1,1,-50)
	(cs1,cs2) = bathy_contour(ax,val)
	cont1 = cs1.collections[0].get_paths()[0].vertices
	cont2 = cs2.collections[0].get_paths()[3].vertices
	cont = np.vstack((cont2,cont1))
	print(cont.shape,cont)
	longitude = np.arange(-180,180,1)
	latitude = np.arange(-90,-50,0.5)

	LAT,LON = np.meshgrid(latitude,longitude)

	X = np.zeros(LAT.shape)


	cont_lon = np.zeros(cont[:,0].shape)
	cont_lat = np.zeros(cont[:,0].shape)

	for i in range(len(cont)):
		cont_lon[i] = np.nanargmin((cont[i][0] - longitude)**2)
		cont_lat[i] = np.nanargmin((cont[i][1]+addlat - latitude)**2)
		X[int(cont_lon[i]),int(cont_lat[i])] = 1
		

	from scipy.interpolate import griddata
	G = ease_grid("S",-50,25000)
	DATA = np.cumsum(X,axis = 1)
	T = griddata((LON.flatten(),LAT.flatten()),DATA.flatten(),(G.lons,G.lats))
	T = np.ma.masked_array(T,mask = T>0)
	T = np.ma.masked_array(T,mask = np.isnan(T))
	return(T.mask)

def snapshot_SLA(ax,date, ncfile = "/net/ether/data/proteo1/mauger/Data/SLA/v4/final_product/dt_antarctic_multimission_sea_level_uv_20130401_20193107.nc" , vmin = -0.2, vmax = 0.2, cmap = cmocean.cm.balance, centered = True):
	D = Dataset(ncfile)
	sla = D["sla"][:]
	if centered:
		sla = (sla.T - np.ma.mean(sla,axis = 2).T).T
	time = D["time"][:]
	lon = D["longitude"][:].data
	lat = D["latitude"][:].data
	cs = ax.pcolormesh(lon,lat, sla[time.tolist().index((date - datetime(1950,1,1)).days)]*100, transform = ccrs.PlateCarree(), cmap = cmap, vmin = vmin*100, vmax = vmax*100)
	return(cs)


def snapshot_current(ax,date,component = 'U', ncfile = "/net/ether/data/proteo1/mauger/Data/SLA/v3/SO_SLA_U_V_2013_2019.nc" , vmin = -0.05, vmax = 0.05, cmap = cmocean.cm.balance):
	D = Dataset(ncfile)
	speed = D[component][:]
	time = D["time"][:]
	lon = D["longitude"][:].data
	lat = D["latitude"][:].data
	cs = ax.pcolormesh(lon,lat, speed[time.tolist().index((date - datetime(1950,1,1)).days)]*100, transform = ccrs.PlateCarree(), cmap = cmap, vmin = vmin*100, vmax = vmax*100)
	return(cs)

	

def snapshot_Arm(ax,date, ncfile = "/net/ether/data/proteo1/mauger/Data/Armitage/CS2_combined_Southern_Ocean_2011-2016.nc" , vmin = -0.2, vmax = 0.2, cmap = cmocean.cm.balance, centered = True):
	D = Dataset(ncfile)
	sla = D["SLA"][:]
	if centered:
		sla = sla - np.nanmean(sla,axis = 0)
	time = D["date"][:]
	lon = D["Longitude"][:].data
	lat = D["Latitude"][:].data
	cs = ax.pcolormesh(lon,lat, sla[time.tolist().index(int(date.strftime("%Y%m"))),:,:], transform = ccrs.PlateCarree(), cmap = cmap, vmin = vmin*100, vmax = vmax*100)
	return(cs)
	



