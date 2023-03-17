import shutil

import numpy as np
from datetime import date
from datetime import timedelta
from datetime import datetime

import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.pyplot import figure
from cartopy import crs as ccrs
from cartopy import feature as cf
#
import cartopy.io.img_tiles as cimgt

from netCDF4 import Dataset as netcdf_dataset
import xarray as xr
import pandas as pd

from shapely.geometry.polygon import Polygon

# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import IncrementalPCA

from PIL import Image

epoch = datetime.utcfromtimestamp(0)

def unix_time_seconds(dt):
    return (dt - epoch).total_seconds()

##shortest geographical distance between 2 points in km.
def distlatlon(lat1,lat2,lon1,lon2):
    lat1=math.radians(lat1)
    lat2=math.radians(lat2)
    lon1=math.radians(lon1)
    lon2 = math.radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (math.sin(dlat / 2) ** 2 ) + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))

    ## earth radius in KM
    r = 6371
    D = c * r ## Shortest distance between points in km
    return D

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
            elif str(col_type)[:4] == 'date':
                df[col] = df[col].values.astype('datetime64[h]')
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def sign(x):
    if x>0:
       return 1
    elif x==0:
       return 0
    else:
       return -1

def UV_to_WD(u,v):
    w = math.sqrt((u ** 2) + (v ** 2))
    if u == 0 and sign(v) == 1:
      dr = 180
    elif u == 0 and sign(v) == -1:
      dr = 0
    elif sign(u) == 1 and sign(v) == 1:
      dr = 270 - math.degrees(math.atan(abs(v / u)))
    elif sign(u) == 1 and sign(v) == -1:
      dr = 270 + math.degrees(math.atan(abs(v / u)))
    elif sign(u) == -1 and sign(v) == -1:
      dr = 90 - math.degrees(math.atan(abs(v / u)))
    elif sign(u) == -1 and sign(v) == 1:
      dr = 90 + math.degrees(math.atan(abs(v / u)))
    if dr < 0:
       dr = 0
    return w,dr


import requests
import time
import os

import pickle

random_seed = np.random.seed(42)

directory_to_load_models="C:/Users/plgeo/OneDrive/PC Desktop/MATLAB DRIVE/MSc_Meteorology/trained_models/"

fileclass=directory_to_load_models+'GB_classifier_'+'central grid.sav'
model_class = pickle.load(open(fileclass, 'rb'))
filestandclass= directory_to_load_models+'GB_classifier_standardizer_'+'central grid.sav'
standardizer_class = pickle.load(open(filestandclass,'rb'))

# model_SVR = pickle.load(open('/var/georgepl/lightning_model/ZI_SVR_central_area.sav', 'rb'))
# normalizer_svr = pickle.load(open('/var/georgepl/lightning_model/ZI_SVR_normalizer.sav', 'rb'))
file_SVR=directory_to_load_models+'ZI_SVR_'+'central area.sav'
model_SVR=pickle.load(open(file_SVR, 'rb'))
# file_features_SVR=directory_to_load_models+'ZI_SVR_features.pkl'
# features_SVR=pickle.load(open(file_features_SVR, 'rb'))
file_normalizer=directory_to_load_models+'ZI_SVR_normalizer.sav'
normalizer_svr=pickle.load(open(file_normalizer, 'rb'))

file_GBR=directory_to_load_models+'GB_ensemble_'+'central area.sav'
model_GBR=pickle.load(open(file_GBR, 'rb'))

clevs = [-0.1,0.1,1.5,5.5,11.5,23.0,50,100000]
cmap = mpl.cm.get_cmap('hot_r')

cmap_data = [cmap(x/len(clevs)+0.05) for x in range(len(clevs))]

cmap_data[0] = (1.0, 1.0, 1.0, 0.0)
cmap = mcolors.ListedColormap(cmap_data, 'lightning')
norm = mcolors.BoundaryNorm(clevs, cmap.N)

clevs2 = [0,0.5,1]
cmap_data2 = [(1.0, 1.0, 1.0, 0.0), (1.0, 0, 0, 1)]
cmap2 = mcolors.ListedColormap(cmap_data2, 'lightning')
norm2 = mcolors.BoundaryNorm(clevs2, cmap2.N)

now = datetime.now()
today = date.today()

# today = date.today() - timedelta(days=7)
if now.hour < 3:
    today = date.today() - timedelta(days=1)
    run = "12"
elif now.hour < 15:
    run = "00"
else:
    run = "12"

# run = "06"

today_edited = today.strftime("%Y%m%d")
today12 = today_edited + run

f = open("/var/georgepl/lightning_model/grid_mapping.csv")

lats = []
lons = []
lats_gfs = []
lons_gfs = []
latlons_gfs = []
latlons_mapping = {}

for line in f:
    line = line.strip()
    line = line.split(",")
    if float(line[0]) not in lats:
        lats.append(float(line[0]))
    if float(line[1]) not in lons:
        lons.append(float(line[1]))
    if float(line[2]) not in lats_gfs:
        lats_gfs.append(float(line[2]))
    if float(line[3]) not in lons_gfs:
        lons_gfs.append(float(line[3]))
    if (float(line[2]), float(line[3])) not in latlons_gfs:
        latlons_gfs.append((float(line[2]), float(line[3])))
    # if (float(line[2]), float(line[3])) not in latlons_gfs:
    latlons_mapping[line[0]+line[1]] = (float(line[2]), float(line[3]))

lats.sort()
lons.sort()

# print(latlons_mapping)

for i in range(2,121,1):
    result = [[-1] * 5 for j in range(4)]
    result_proba = [[-1] * 5 for j in range(4)]
    result_regressor = [[-0.1] * 5 for j in range(4)]
    # print(result)
    i_string = str(i).zfill(3)
    ahead_minus_1 = str(i-1).zfill(3)

    url = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?file=gfs.t'+run+'z.pgrb2.0p25.f'+i_string+'&lev_1000_mb=on&lev_100_mb=on&lev_10_m_above_ground=on&lev_10_mb=on&lev_150_mb=on&lev_15_mb=on&lev_200_mb=on&lev_20_mb=on&lev_250_mb=on&lev_300_mb=on&lev_30_mb=on&lev_350_mb=on&lev_400_mb=on&lev_40_mb=on&lev_450_mb=on&lev_500_mb=on&lev_50_mb=on&lev_550_mb=on&lev_600_mb=on&lev_650_mb=on&lev_700_mb=on&lev_70_mb=on&lev_750_mb=on&lev_800_mb=on&lev_850_mb=on&lev_900_mb=on&lev_925_mb=on&lev_950_mb=on&lev_975_mb=on&lev_mean_sea_level=on&lev_surface=on&var_ABSV=on&var_ACPCP=on&var_APCP=on&var_CAPE=on&var_CIN=on&var_CLWMR=on&var_CPRAT=on&var_DPT=on&var_HGT=on&var_HPBL=on&var_ICMR=on&var_PRMSL=on&var_RH=on&var_RWMR=on&var_SNMR=on&var_SPFH=on&var_TMP=on&var_UGRD=on&var_VGRD=on&var_VVEL=on&subregion=&leftlon='+str(min(lons_gfs)-0.3)+'&rightlon='+str(max(lons_gfs)+0.3)+'&toplat='+str(max(lats_gfs)+0.3)+'&bottomlat='+str(min(lats_gfs)-0.3)+'&dir=%2Fgfs.'+str(today_edited)+'%2F'+run+'%2Fatmos'
    while True:
        try:
            r = requests.get(url, allow_redirects=True, timeout=10)
            open('/var/georgepl/lightning_model/gfs.grib2', 'wb').write(r.content)
            if os.stat("/var/georgepl/lightning_model/gfs.grib2").st_size < 800:
                print("small")
                time.sleep(1)
                continue
            
            os.system("/var/plots/./wgrib2 /var/georgepl/lightning_model/gfs.grib2 -netcdf /var/georgepl/lightning_model/gfs.nc")
            
            fname = "/var/georgepl/lightning_model/gfs.nc"
            dataset = netcdf_dataset(fname)
            time_list = dataset.variables['time'][:]
            desired = datetime(today.year, today.month, today.day) + timedelta(hours=int(run)) + timedelta(hours=i)
            print(desired)
            if unix_time_seconds(desired) != time_list[0]:
                # print("time not match")
                # print(unix_time_seconds(desired), time_list[0])
                continue
            break
        except:
            print("Error")
            time.sleep(1)

    url = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?file=gfs.t'+run+'z.pgrb2.0p25.f'+ahead_minus_1+'&lev_surface=on&var_ACPCP=on&var_APCP=on&subregion=&leftlon='+str(min(lons_gfs)-0.3)+'&rightlon='+str(max(lons_gfs)+0.3)+'&toplat='+str(max(lats_gfs)+0.3)+'&bottomlat='+str(min(lats_gfs)-0.3)+'&dir=%2Fgfs.'+str(today_edited)+'%2F'+run+'%2Fatmos'
    while True:
        try:
            r = requests.get(url, allow_redirects=True, timeout=10)
            open('/var/georgepl/lightning_model/gfs_prev.grib2', 'wb').write(r.content)
            if os.stat("/var/georgepl/lightning_model/gfs_prev.grib2").st_size < 600:
                print("small")
                time.sleep(1)
                continue
            
            os.system("/var/plots/./wgrib2 /var/georgepl/lightning_model/gfs_prev.grib2 -netcdf /var/georgepl/lightning_model/gfs_prev.nc")
            
            fname = "/var/georgepl/lightning_model/gfs_prev.nc"
            dataset = netcdf_dataset(fname)
            time_list = dataset.variables['time'][:]
            desired = datetime(today.year, today.month, today.day) + timedelta(hours=int(run)) + timedelta(hours=i-1)
            print(desired)
            if unix_time_seconds(desired) != time_list[0]:
                # print("time not match")
                # print(unix_time_seconds(desired), time_list[0])
                continue
            break
        except:
            print("Error")
            time.sleep(1)

    fname = "/var/georgepl/lightning_model/gfs.nc"
    dataset = netcdf_dataset(fname)
    time_list = dataset.variables['time'][:]

    ds = xr.open_dataset('/var/georgepl/lightning_model/gfs.nc')
    df_gfs = ds.to_dataframe()
    df_gfs = df_gfs.reset_index()
    df_gfs = df_gfs.fillna(0)

    ds = xr.open_dataset('/var/georgepl/lightning_model/gfs_prev.nc')
    df_gfs_prev = ds.to_dataframe()
    df_gfs_prev = df_gfs_prev.reset_index()
    df_gfs_prev = df_gfs_prev.fillna(0)

    # print(df_gfs.head())
    # for col in df_gfs.columns:
    #     print(col)

    a = 17.625
    b = 243.04

    def fun(T,RH):
        RH = RH.replace(0,0.0001)
        return np.log(RH/100) + a*T/(b+T)

    df_gfs['td850'] = (b * fun(df_gfs['TMP_850mb']-273.15,df_gfs['RH_850mb'])) / (a - fun(df_gfs['TMP_850mb']-273.15,df_gfs['RH_850mb']))

    df_gfs['RH_500mb'] = df_gfs['RH_500mb']*(1/100)
    df_gfs.loc[df_gfs['RH_500mb']<0,'RH_500mb'] = 0

    df_gfs['RH_600mb'] = df_gfs['RH_600mb']*(1/100)
    df_gfs.loc[df_gfs['RH_600mb']<0,'RH_600mb'] = 0

    df_gfs['RH_700mb'] = df_gfs['RH_700mb']*(1/100)
    df_gfs.loc[df_gfs['RH_700mb']<0,'RH_700mb'] = 0

    df_gfs['RH_800mb'] = df_gfs['RH_800mb']*(1/100)
    df_gfs.loc[df_gfs['RH_800mb']<0,'RH_800mb'] = 0

    df_gfs['RH_850mb'] = df_gfs['RH_850mb']*(1/100)
    df_gfs.loc[df_gfs['RH_850mb']<0,'RH_850mb'] = 0

    df_gfs.loc[df_gfs['CAPE_surface']<0,'CAPE_surface'] = 0

    df_gfs.loc[df_gfs['CIN_surface']<0,'CIN_surface'] = 0

    df_gfs['Total totals index'] = (df_gfs['TMP_850mb']-273.15) - (df_gfs['TMP_500mb']-273.15) + (df_gfs['td850']) - (df_gfs['TMP_500mb']-273.15)

    df_gfs['PRMSL_meansealevel'] = df_gfs['PRMSL_meansealevel'] * (1/1000)

    # print(df_gfs)
    # df_gfs['ACPCP_surface'][0] = 12
    # df_gfs['ACPCP_surface'][1] = 17
    # df_gfs_prev['ACPCP_surface'][0] = 10
    # df_gfs_prev['ACPCP_surface'][1] = 3
    # print(df_gfs['ACPCP_surface'])

    df_gfs['ACPCP_surface'] = df_gfs['ACPCP_surface'] - df_gfs_prev['ACPCP_surface']
    df_gfs['CPRAT_surface'] = df_gfs['CPRAT_surface'] * 3600
    df_gfs['APCP_surface'] = df_gfs['APCP_surface'] - df_gfs_prev['APCP_surface']

    df_gfs['Vorticity (relative)_500'] = df_gfs['ABSV_500mb'] - 2*0.00007292115 * np.sin(df_gfs['latitude'] * np.pi / 180.)
    df_gfs['Vorticity (relative)_500'] = df_gfs['Vorticity (relative)_500']*(10**5)

    df_gfs['Wind_Speed_500'] = df_gfs.apply(lambda x: UV_to_WD(x['UGRD_500mb'], x['VGRD_500mb'])[0], axis=1)
    df_gfs['Wind_Speed_600'] = df_gfs.apply(lambda x: UV_to_WD(x['UGRD_600mb'], x['VGRD_600mb'])[0], axis=1)
    df_gfs['Wind_Speed_800'] = df_gfs.apply(lambda x: UV_to_WD(x['UGRD_800mb'], x['VGRD_800mb'])[0], axis=1)
    df_gfs['Wind_Speed_850'] = df_gfs.apply(lambda x: UV_to_WD(x['UGRD_850mb'], x['VGRD_850mb'])[0], axis=1)
    df_gfs['Wind_Speed_900'] = df_gfs.apply(lambda x: UV_to_WD(x['UGRD_900mb'], x['VGRD_900mb'])[0], axis=1)
    df_gfs['Wind_Speed_925'] = df_gfs.apply(lambda x: UV_to_WD(x['UGRD_925mb'], x['VGRD_925mb'])[0], axis=1)
    df_gfs['Wind_Speed_950'] = df_gfs.apply(lambda x: UV_to_WD(x['UGRD_950mb'], x['VGRD_950mb'])[0], axis=1)

    df_gfs['HPBL'] = df_gfs['HPBL_surface']

    df_gfs['Divergence'] = float('NaN')

    for lat, lon in latlons_gfs:
        lat1 = lat - 0.25
        lat2 = lat + 0.25
        lon1 = lon - 0.25
        lon2 = lon + 0.25

        df_gfs_left = df_gfs.loc[(df_gfs['latitude'] == lat) & (df_gfs['longitude'] == lon1)]

        df_gfs_right = df_gfs.loc[(df_gfs['latitude'] == lat) & (df_gfs['longitude'] == lon2)]

        df_gfs_up = df_gfs.loc[(df_gfs['latitude'] == lat2) & (df_gfs['longitude'] == lon)]

        df_gfs_down = df_gfs.loc[(df_gfs['latitude'] == lat1) & (df_gfs['longitude'] == lon)]

        DX = distlatlon(lat,lat,lon1-(lon2-lon1)/2,lon1+(lon2-lon1)/2)
        # print("dx", DX)
        DY = distlatlon(lat1-(lat2-lat1)/2,lat1+(lat2-lat1)/2,lon,lon)
        # print("dy", DY)

        D_300 = (float(df_gfs_right['UGRD_300mb']) - float(df_gfs_left['UGRD_300mb']))/DX + (float(df_gfs_up['VGRD_300mb']) - float(df_gfs_down['VGRD_300mb']))/DY
        D_300 *= 100

        df_gfs['Divergence'].loc[(df_gfs['latitude'] == lat) & (df_gfs['longitude'] == lon)] = D_300
        # print(df_gfs)


    df_gfs['Total column supercooled liquid water'] = float('NaN')
    df_gfs['Total column water'] = float('NaN')
    df_gfs['Total column water vapour'] = float('NaN')


    slw_icmr_clwmr = {}

    latlons_extended = []
    for lat, lon in latlons_gfs:
        if (lat, lon) not in latlons_extended:
            latlons_extended.append((lat, lon))
        if (lat+0.25, lon) not in latlons_extended:
            latlons_extended.append((lat+0.25, lon))
        if (lat-0.25, lon) not in latlons_extended:
            latlons_extended.append((lat-0.25, lon))
        if (lat, lon+0.25) not in latlons_extended:
            latlons_extended.append((lat, lon+0.25))
        if (lat, lon-0.25) not in latlons_extended:
            latlons_extended.append((lat, lon-0.25))

    for lat, lon in latlons_extended:

        df_gfs_current = df_gfs.loc[(df_gfs['latitude'] == lat) & (df_gfs['longitude'] == lon)]

        mixing_ratio_levels = [1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50]
        p = 1.22

        levels_negative_ght = 0
        for level in mixing_ratio_levels:
            if float(df_gfs_current['HGT_'+str(level)+'mb']) < 0:
                levels_negative_ght += 1

        height_differences = [float(df_gfs_current['HGT_1000mb'])]+[float(df_gfs_current['HGT_'+str(level_next)+'mb'] - df_gfs_current['HGT_'+str(level)+'mb']) for level, level_next in zip(mixing_ratio_levels[0:-1], mixing_ratio_levels[1:])]
        for j in range(levels_negative_ght):
            height_differences[j] = 0

        ICMR = [float(df_gfs_current['ICMR_'+str(level)+'mb']) for level in mixing_ratio_levels]

        ICMR_sum = np.sum(p*np.matmul(ICMR, height_differences))
        # print(ICMR_sum)

        RWMR = [float(df_gfs_current['RWMR_'+str(level)+'mb']) for level in mixing_ratio_levels]
        RWMR_sum = np.sum(p*np.matmul(RWMR, height_differences))
        # print(RWMR_sum)

        SNMR = [float(df_gfs_current['SNMR_'+str(level)+'mb']) for level in mixing_ratio_levels]
        SNMR_sum = np.sum(p*np.matmul(SNMR, height_differences))
        # print(SNMR_sum)

        CLWMR = [float(df_gfs_current['CLWMR_'+str(level)+'mb']) for level in mixing_ratio_levels]
        CLWMR_sum = np.sum(p*np.matmul(CLWMR, height_differences))
        # print(CLWMR_sum)

        WV = [float(df_gfs_current['SPFH_'+str(level)+'mb']/(1 - df_gfs_current['SPFH_'+str(level)+'mb'])) for level in mixing_ratio_levels]
        WV_sum = np.sum(p*np.matmul(WV, height_differences))

        TCW = ICMR_sum + RWMR_sum + SNMR_sum + CLWMR_sum + WV_sum
        # print(TCW)

        first_negative_level = 50
        first_negative_level_index = -1
        for level in mixing_ratio_levels:
            if float(df_gfs_current['TMP_'+str(level)+'mb'])-273.15 < 0:
                first_negative_level = level
                first_negative_level_index = mixing_ratio_levels.index(first_negative_level)
                break
        
        RWMR_negative_temp = [float(df_gfs_current['RWMR_'+str(level)+'mb']) for level in mixing_ratio_levels if first_negative_level >= level]
        CLWMR_negative_temp = [float(df_gfs_current['CLWMR_'+str(level)+'mb']) for level in mixing_ratio_levels if first_negative_level >= level]
        heights_negative_temp = [df_gfs_current['HGT_'+str(level)+'mb'] for level in mixing_ratio_levels if first_negative_level >= level]
        height_differences_negative_temp = [heights_negative_temp[0]]+[float(height_next - height) for height, height_next in zip(heights_negative_temp[0:-1], heights_negative_temp[1:])]
        
        levels_negative_ght = 0
        for level in mixing_ratio_levels:
            if first_negative_level >= level and float(df_gfs_current['HGT_'+str(level)+'mb']) < 0:
                levels_negative_ght += 1

        for j in range(levels_negative_ght):
            height_differences_negative_temp[j] = 0

        for index, level in enumerate(mixing_ratio_levels[first_negative_level_index:]):
            if float(df_gfs_current['TMP_'+str(level)+'mb'])-273.15 >= 0:
                height_differences_negative_temp[index] = 0

        SLW = [x + y for x, y in zip(RWMR_negative_temp, CLWMR_negative_temp)]
        SLW_sum = np.sum(p*np.matmul(SLW, height_differences_negative_temp))
        # print(SLW_sum)

        if isinstance(lat, float) and lat.is_integer():
            lat = int(lat)

        if isinstance(lon, float) and lon.is_integer():
            lon = int(lon)

        df_gfs['Total column supercooled liquid water'].loc[(df_gfs['latitude'] == lat) & (df_gfs['longitude'] == lon)] = SLW_sum
        df_gfs['Total column water'].loc[(df_gfs['latitude'] == lat) & (df_gfs['longitude'] == lon)] = TCW
        df_gfs['Total column water vapour'].loc[(df_gfs['latitude'] == lat) & (df_gfs['longitude'] == lon)] = WV_sum

        slw_icmr_clwmr[str(lat)+str(lon)] = [WV, ICMR, CLWMR]

    df_gfs['Vertical integral of divergence of moisture flux'] = float('NaN')

    # for key in slw_icmr_clwmr.keys():
    #     print(key)

    for lat, lon in latlons_gfs:
        lat1 = lat - 0.25
        lat2 = lat + 0.25
        lon1 = lon - 0.25
        lon2 = lon + 0.25

        if isinstance(lat, float) and lat.is_integer():
            lat = int(lat)

        if isinstance(lon, float) and lon.is_integer():
            lon = int(lon)

        if lat1.is_integer():
            lat1 = int(lat1)

        if lat2.is_integer():
            lat2 = int(lat2)

        if lon1.is_integer():
            lon1 = int(lon1)

        if lon2.is_integer():
            lon2 = int(lon2)

        df_gfs_left = df_gfs.loc[(df_gfs['latitude'] == lat) & (df_gfs['longitude'] == lon1)]

        df_gfs_right = df_gfs.loc[(df_gfs['latitude'] == lat) & (df_gfs['longitude'] == lon2)]

        df_gfs_up = df_gfs.loc[(df_gfs['latitude'] == lat2) & (df_gfs['longitude'] == lon)]

        df_gfs_down = df_gfs.loc[(df_gfs['latitude'] == lat1) & (df_gfs['longitude'] == lon)]

        df_gfs_current = df_gfs.loc[(df_gfs['latitude'] == lat) & (df_gfs['longitude'] == lon)]

        DX = distlatlon(lat,lat,lon1-(lon2-lon1)/2,lon1+(lon2-lon1)/2)
        # print("dx", DX)
        DY = distlatlon(lat1-(lat2-lat1)/2,lat1+(lat2-lat1)/2,lon,lon)
        # print("dy", DY)

        mixing_ratio_levels = [1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50]
        slw_left = slw_icmr_clwmr[str(lat)+str(lon1)][0]
        icmr_left = slw_icmr_clwmr[str(lat)+str(lon1)][1]
        clwmr_left = slw_icmr_clwmr[str(lat)+str(lon1)][2]
        slw_right = slw_icmr_clwmr[str(lat)+str(lon2)][0]
        icmr_right = slw_icmr_clwmr[str(lat)+str(lon2)][1]
        clwmr_right = slw_icmr_clwmr[str(lat)+str(lon2)][2]
        slw_up = slw_icmr_clwmr[str(lat2)+str(lon)][0]
        icmr_up = slw_icmr_clwmr[str(lat2)+str(lon)][1]
        clwmr_up = slw_icmr_clwmr[str(lat2)+str(lon)][2]
        slw_down = slw_icmr_clwmr[str(lat1)+str(lon)][0]
        icmr_down = slw_icmr_clwmr[str(lat1)+str(lon)][1]
        clwmr_down = slw_icmr_clwmr[str(lat1)+str(lon)][2]
        slw_current = slw_icmr_clwmr[str(lat)+str(lon)][0]
        icmr_current = slw_icmr_clwmr[str(lat)+str(lon)][1]
        clwmr_current = slw_icmr_clwmr[str(lat)+str(lon)][2]
        MFC = []
        for index,level in enumerate(mixing_ratio_levels):
            M_left = slw_left[index] + icmr_left[index] + clwmr_left[index]
            M_right = slw_right[index] + icmr_right[index] + clwmr_right[index]
            M_up = slw_up[index] + icmr_up[index] + clwmr_up[index]
            M_down = slw_down[index] + icmr_down[index] + clwmr_down[index]
            M_current = slw_current[index] + icmr_current[index] + clwmr_current[index]
            D_ = (float(df_gfs_right['UGRD_300mb']) - float(df_gfs_left['UGRD_300mb']))/DX + (float(df_gfs_up['VGRD_300mb']) - float(df_gfs_down['VGRD_300mb']))/DY
            MFC.append(-float(df_gfs_current['UGRD_'+str(level)+'mb'])*(M_right-M_left)/DX - float(df_gfs_current['VGRD_'+str(level)+'mb'])*(M_up-M_down)/DY - M_current*D_)
        # print(MFC)
        VIMFC = np.sum(MFC)*100
        # print(VIMFC)
        df_gfs['Vertical integral of divergence of moisture flux'].loc[(df_gfs['latitude'] == lat) & (df_gfs['longitude'] == lon)] = VIMFC
        # print(df_gfs)

    df_gfs['SPFH_700mb'] = df_gfs['SPFH_700mb'] * 1000
    df_gfs['SPFH_800mb'] = df_gfs['SPFH_800mb'] * 1000
    df_gfs['SPFH_850mb'] = df_gfs['SPFH_850mb'] * 1000
    df_gfs['SPFH_900mb'] = df_gfs['SPFH_900mb'] * 1000
    df_gfs['SPFH_925mb'] = df_gfs['SPFH_925mb'] * 1000
    df_gfs['SPFH_950mb'] = df_gfs['SPFH_950mb'] * 1000

    df_predict = df_gfs[['RH_500mb', 'RH_600mb', 'RH_700mb', 'RH_800mb', 'RH_850mb', 'VVEL_500mb', 'VVEL_600mb', 'VVEL_700mb', 'VVEL_800mb',
    'VVEL_850mb', 'VVEL_900mb', 'VVEL_925mb', 'VVEL_950mb', 'Divergence', 'CAPE_surface', 'CIN_surface',
    'Total column supercooled liquid water', 'Total column water', 'Total column water vapour', 'Total totals index',
    'PRMSL_meansealevel', 'ACPCP_surface', 'CPRAT_surface', 'APCP_surface', 'HPBL',
    'Vertical integral of divergence of moisture flux', 'Vorticity (relative)_500', 'SPFH_700mb', 'SPFH_800mb',
    'SPFH_850mb', 'SPFH_900mb', 'SPFH_925mb', 'SPFH_950mb', 'Wind_Speed_500', 'Wind_Speed_600', 'Wind_Speed_800',
    'Wind_Speed_850', 'Wind_Speed_900', 'Wind_Speed_925', 'Wind_Speed_950', 'latitude', 'longitude']].loc[df_gfs.apply(lambda x: (x['latitude'],x['longitude']) in latlons_gfs, axis=1)].copy()

    df_predict.rename(columns={'RH_500mb': 'Relative humidity_500',
    'RH_600mb': 'Relative humidity_600',
    'RH_700mb': 'Relative humidity_700',
    'RH_800mb': 'Relative humidity_800',
    'RH_850mb': 'Relative humidity_850',
    'VVEL_500mb': 'Vertical velocity_500',
    'VVEL_600mb': 'Vertical velocity_600',
    'VVEL_700mb': 'Vertical velocity_700',
    'VVEL_800mb': 'Vertical velocity_800',
    'VVEL_850mb': 'Vertical velocity_850',
    'VVEL_900mb': 'Vertical velocity_900',
    'VVEL_925mb': 'Vertical velocity_925',
    'VVEL_950mb': 'Vertical velocity_950',
    'CAPE_surface': 'Convective available potential energy',
    'CIN_surface': 'Convective inhibition',
    'PRMSL_meansealevel': 'Mean sea level pressure',
    'ACPCP_surface': 'Convective precipitation',
    'CPRAT_surface': 'Convective rain rate',
    'APCP_surface': 'Total precipitation',
    'HPBL': 'Boundary layer height',
    'SPFH_700mb': 'Specific humidity_700',
    'SPFH_800mb': 'Specific humidity_800',
    'SPFH_850mb': 'Specific humidity_850',
    'SPFH_900mb': 'Specific humidity_900',
    'SPFH_925mb': 'Specific humidity_925',
    'SPFH_950mb': 'Specific humidity_950'}, inplace=True)

    # lats = [37.781, 37.899, 38.017, 38.135]
    # lons = [23.4885, 23.6115, 23.7345, 23.8575, 23.9805]

    for lat_index,lat in enumerate(reversed(lats)):
        for lon_index,lon in enumerate(lons):
            # if lat < 38:
            #     x_test = df_predict.loc[(df_predict['latitude'] == 38) & (df_predict['longitude'] == 23.75)]
            #     x_test = x_test.drop(['latitude', 'longitude'], axis=1)
            # elif lat > 38:
            #     x_test = df_predict.loc[(df_predict['latitude'] == 38.25) & (df_predict['longitude'] == 23.75)]
            #     x_test = x_test.drop(['latitude', 'longitude'], axis=1)

            lat_gfs, lon_gfs = latlons_mapping[str(lat)+str(lon)]
            # print(lat_gfs, lon_gfs)

            x_test = df_predict.loc[(df_predict['latitude'] == lat_gfs) & (df_predict['longitude'] == lon_gfs)]
            x_test = x_test.drop(['latitude', 'longitude'], axis=1)

            x_test['latitude'] = lat
            x_test['longitude'] = lon

            classification_gb = model_class.predict(standardizer_class.transform(x_test.values))

            probability_gb = model_class.predict_proba(standardizer_class.transform(x_test.values))[0]

            if classification_gb[0] == 1:
                # print(lat, lon, lat_gfs, lon_gfs)
                # x_test_regressor = x_test[['Convective rain rate', 'Total column supercooled liquid water', 'Convective precipitation', 'Convective available potential energy', 'Vertical velocity_800', 'Total totals index', 'Vertical velocity_900', 'Relative humidity_700', 'Convective inhibition', 'Specific humidity_700', 'Relative humidity_800', 'Vertical integral of divergence of moisture flux', 'Specific humidity_900', 'Boundary layer height', 'Relative humidity_850', 'Specific humidity_925', 'Mean sea level pressure', 'Vertical velocity_925', 'Wind_Speed_500', 'Total column water', 'Wind_Speed_800', 'Vertical velocity_500', 'Vertical velocity_700', 'Relative humidity_500', 'Total column water vapour', 'Divergence', 'Vorticity (relative)_500', 'Wind_Speed_950', 'Specific humidity_950', 'Wind_Speed_925', 'Relative humidity_600', 'Wind_Speed_900', 'Specific humidity_800', 'Wind_Speed_600', 'Wind_Speed_850', 'Vertical velocity_600', 'latitude', 'longitude']]
                x_test_regressor = x_test[['Convective rain rate', 'Total column supercooled liquid water', 'Convective precipitation', 'Convective available potential energy', 'Vertical velocity_800', 'Total totals index', 'Vertical velocity_900', 'Relative humidity_700', 'Convective inhibition', 'Vertical velocity_850', 'Specific humidity_700', 'Relative humidity_800', 'Vertical integral of divergence of moisture flux', 'Specific humidity_900', 'Boundary layer height', 'Relative humidity_850', 'Specific humidity_925', 'Mean sea level pressure', 'Vertical velocity_925', 'Wind_Speed_500', 'Total column water', 'Wind_Speed_800', 'Vertical velocity_500', 'Vertical velocity_700', 'Relative humidity_500', 'Total column water vapour', 'latitude', 'longitude']]

                # print(x_test['Convective rain rate'], x_test['Total column supercooled liquid water'], x_test['latitude'], x_test['longitude'])

                # regression_svr = model_SVR.predict(x_test_regressor.values.reshape(1,-1))
                # regression_svr = math.expm1(normalizer_svr.inverse_transform(regression_svr.reshape(-1, 1))[:, 0])

                # result_regressor[lat_index][lon_index] = regression_svr

                # print(regression_svr)
                # print("edw",regression_svr)

                corr_factor_gbr = 1.801910513824101
                yprt2 = abs(np.expm1(model_GBR.predict(x_test_regressor.values)))[0]
                yprt2 *= corr_factor_gbr
                result_regressor[lat_index][lon_index] = yprt2
                # print(yprt2)
                # print("edw",yprt2)

                del x_test_regressor

            del x_test
            result[lat_index][lon_index] = classification_gb[0]
            result_proba[lat_index][lon_index] = probability_gb[1]

    pd.set_option('display.max_columns', None)

 
