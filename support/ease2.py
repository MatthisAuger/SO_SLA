#!/usr/bin/env python
# coding: utf-8

import pyproj
import numpy as np
import argparse
from netCDF4 import Dataset

class ease_grid(object):
    
    '''
        a class for the manipulation of EASE2 nothern or southern hemispheric grids.
        
        EASE stands for equal area scalable earth, a description 
        of the projection and tiling is available at https://nsidc.org/data/ease
        
        the current class is loosely adapted from github.com/TUW-GEO/ease_grid
    '''
    
    __slots__ = ('_lat_bound_N',
                    '_lat_bound_S',
                    '_orientation',
                    '_proj',
                    '_res',
                    '_lons',
                    '_lats',
                    '_r0',
                    '_s0',
                    '_x_min',
                    '_y_min',
                    '_y_max',
                    '_x_max',
                    '_xx',
                    '_yy'
                    )
    
    def __init__(self,
                    orientation = 'N', 
                    latitude=0., 
                    res=100000.):
        ''' 
            constructor 
                * orientation: 'N' ('S') for Northern (Southern) hemispheric grid.
                * latitude is the lower (upper) latitude bound of the grid (defaults to 0.) for 'N' ('S') orientation.
                * res is the grid resolution in meters (defaults to 100000) 
        '''
        self._orientation = orientation
        if self._orientation == 'N':
            self._lat_bound_N = latitude
        elif self._orientation =='S':
            self._lat_bound_S = latitude
        else : 
            print("ERROR: Orientation must be 'N' (North) or 'S' (South)")
            return ("STOP") 
        self._res = res
        self.__setup()
        
    def __setup(self):
        '''
            actually creates the grid
        '''
        # only nothern hemisphere implemented
        # projection is a Lambert Azimuthal Equal Area
        if self._orientation =='N':
            self._proj = pyproj.Proj(proj='laea', 
                                    lat_0=90, 
                                    lon_0=0, 
                                    x_0=0,
                                    y_0=0,
                                    ellps='WGS84',
                                    datum='WGS84',
                                    units='m') 
        else:
            self._proj = pyproj.Proj(proj='laea', 
                                    lat_0=-90, 
                                    lon_0=0, 
                                    x_0=0,
                                    y_0=0,
                                    ellps='WGS84',
                                    datum='WGS84',
                                    units='m') 
        # x axis boundaries in the projected space
        self._x_min, _ = self._proj(-90.,self._lat_bound_S)
        self._x_max, _ = self._proj(90., self._lat_bound_S)
        
        # get extent and cell count
        x_extent = self._x_max - self._x_min
        x_count = np.round(x_extent/self._res)
        
        # we want an even number of cells
        if x_count%2 != 0:
            x_count -= 1
        
        # reestimate x axis boundaries and pave x axis
        self._x_min, self._x_max = -x_count/2*self._res, x_count/2*self._res
        x_arr = np.arange(self._x_min+self._res/2, self._x_max, self._res)
        
        # do a similar thing for y axis
        self._y_min, self._y_max = -x_count/2*self._res, x_count/2*self._res
        y_arr = np.arange(self._y_min+self._res/2, self._y_max, self._res)
        
        # pave the plane
        xx, yy = np.meshgrid(x_arr, y_arr)
        self._xx = xx
        self._yy = yy
        # set offsets for the center of the grid
        self._r0, self._s0 = x_count/2, x_count/2
        
        # compute lons/lats in geographical space from projected place coords 
        llons, llats = self._proj(xx.flatten(), yy.flatten(), inverse=True)
        self._lons = llons.reshape(np.shape(xx)).T
        self._lats = llats.reshape(np.shape(xx)).T
        
                                            
        return 0
    
    def write_dat(self, filename):
        '''
            export lon/lat to an ascii file
        '''
        with open(filename, 'w') as f:
            for lon, lat in zip(self.lons.flatten(), self.lats.flatten()):
                f.write("%.3f %.3f\n" %(lon, lat))
        return 0
        
    def write_nc(self, filename):
        '''
            export lon/lat to a netCDF file
        '''
        with Dataset(filename, 'w') as nc:
            x = nc.createDimension('index', self.nlats)
            lon = nc.createVariable('longitude', 'f4', dimensions=('index'))
            lon[:] = self.lons.flatten()
            lat = nc.createVariable('latitude', 'f4', dimensions=('index'))
            lat[:] = self.lats.flatten()
            
        return 0
        
    def lonlat2index(self, lon, lat, positive=False):
        '''
            returns grid cell index for a given lon/lat
        '''
        lon[lon>180.] -= 360.
        
        x, y = self._proj(lon, lat)
        x_i = np.int_(np.round((x - self._x_min - self._res/2)/self._res))
        y_i = np.int_(np.round((y - self._y_max - self._res/2)/self._res))
        
        if positive:
            x_i[x_i<0] += self.nx
            y_i[y_i<0] += self.ny
        
        return x_i, y_i
    
    @property
    def lats(self):
        '''
            returns lats as 2d masked array
        '''
        return self._lats
    
    @property
    def lons(self):
        '''
            returns lons as 2d masked array
        '''
        return self._lons
    @property
    def xx(self):
        '''
            returns lons as 2d masked array
        '''
        return self._xx
    @property
    def yy(self):
        '''
            returns lons as 2d masked array
        '''
        return self._yy
        
    @property
    def nx(self):
        '''
            returns number of grid points in the x direction
        '''
        return np.shape(self._lats)[0]
    
    @property
    def ny(self):
        '''
            returns number of grid points in the y direction
        '''
        return np.shape(self._lats)[1]
        
    @property
    def nelem(self):
        '''
            returns the total number of grid points
        '''
        return self.nx * self.ny
    
    @property
    def nlats(self):
        '''
            returns the total number of latitudes (= nelem)
        '''
        return len(self._lats.flatten())
        
    @property
    def shape(self):
        '''
            returns the grid shape (nx,ny)
        '''
        return np.shape(self._lats)
    
    @property
    def nlons(self):
        '''
            returns the total number of longitudes (= nelem)
        '''
        return len(self._lons.flatten())
    
    @property
    def resolution(self):
        '''
            returns the grid resolution in meters
        '''
        return self._res
        
    @property
    def lat_bound(self):
        '''
            returns the effective latitude of the grid boundary
        '''
        return np.nanmin(self.lats)


            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('outfile')
    parser.add_argument('--orientation', default = 'N', type = str)
    parser.add_argument('--res', default=100000, type=float)
    parser.add_argument('--lat_lim', default=35., type=float)
    parser.add_argument('--netcdf', default=False, action='store_true')
    args = parser.parse_args()    
    g = ease_grid(args.orientation, args.lat_lim, res=args.res)
    if args.netcdf:
        g.write_nc(args.outfile)
    else:
        g.write_dat(args.outfile)
        
