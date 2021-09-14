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
from ease2 import ease_grid
import sys
import os
import logging

import xarray as xr


def xarray_SLA():
	ds = xr.open_dataset("/net/ether/data/proteo1/mauger/Data/SLA/v1/AO_Gridded_ALL_23101_25446_mod.nc")
	ds.time.attrs = {'units' : 'days since 1950-01-01'}
	ds = xr.decode_cf(ds)
	return(ds)

def xarray_SLA_climato():
	ds = xr.open_dataset("/net/ether/data/proteo1/mauger/Data/SLA/v1/AO_Gridded_ALL_23101_25446_mod.nc")
	ds.time.attrs = {'units' : 'days since 1950-01-01'}
	ds = xr.decode_cf(ds)
	Climato = ds.groupby('time.month').mean()
	return(Climato)

def xarray_OSC():
	ds = xr.open_dataset("/net/ether/data/proteo1/mauger/Data/Ocean_Stress_Curl/Ocean_Stress_Curl_v6_mod.nc")
	ds.time.attrs = {'units' : 'days since 1950-01-01'}
	ds = xr.decode_cf(ds)
	return(ds)

def xarray_OSC_climato():
	ds = xr.open_dataset("/net/ether/data/proteo1/mauger/Data/Ocean_Stress_Curl/Ocean_Stress_Curl_v6_mod.nc")
	ds.time.attrs = {'units' : 'days since 1950-01-01'}
	ds = xr.decode_cf(ds)
	Climato = ds.groupby('time.month').mean()
	return(Climato)
