import netCDF4 as nc
import statistics
import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import spearmanr,kendalltau,rankdata,weightedtau,kstest,shapiro,describe,zscore
from itertools import chain
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold as CV_kfold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score as CV_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingClassifier
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score,confusion_matrix,balanced_accuracy_score,log_loss,make_scorer,d2_tweedie_score,mean_poisson_deviance,mean_gamma_deviance,mean_squared_log_error,roc_auc_score,classification_report
from sklearn.metrics import mean_pinball_loss,explained_variance_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import PoissonRegressor
from sklearn.neighbors import KernelDensity
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline,make_pipeline
from xgboost.sklearn import XGBClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, rand
from fitter import Fitter,get_common_distributions,get_distributions
from scipy.optimize import curve_fit
from scipy.stats import genhalflogistic as ghlog
from scipy.stats import poisson,linregress,beta
from scipy.stats import chi2_contingency
from sklearn.linear_model import LinearRegression,ElasticNet,QuantileRegressor
from sklearn.utils import check_consistent_length,check_array
from rfpimp import feature_corr_matrix
import copy
import csv
import xlsxwriter
import warnings
import pickle
import numbers
import sys

random_seed=np.random.seed(42)

def fxn():
    warnings.warn(UserWarning)

def sign(x):
    if x>0:
       return 1
    elif x==0:
       return 0
    else:
       return -1

#shortest distance between points in km
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

## converts lat,lon to x,y egsa87
def latlon_to_xy(lat,lon):
    lat=math.radians(lat)
    lon=math.radians(lon)
    GE_WGS84_Alpha=6378137.000
    GE_WGS84_F_INV=298.257223563
    ##displace_geodetic_system
    e2=1-(1-1/GE_WGS84_F_INV)**2
    rad=GE_WGS84_Alpha/math.sqrt(1-e2*math.sin(lat)*math.sin(lat))
    ##spherical coordinates inverse transformation
    x=rad*math.cos(lat)*math.cos(lon)
    y=rad*math.cos(lat)*math.sin(lon)
    z=rad*(1-e2)*math.sin(lat)
    ##displace geocentric system
    x2=x+199.72
    y2=y-74.03
    z2=z-246.02
    ##ellipsoid xyz to philambda
    aradius=GE_WGS84_Alpha
    ##sphere xyz to philambda
    if abs(z2)<aradius:
       phi2=math.asin(z2/aradius)
    else:
        if z2>0:
           phi2=0.5*math.pi
        else:
           phi2=-0.5*math.pi
    if abs(x2)>0.001:
        lambda2=math.atan(y2/x2)
    else:
        if y2>0:
            lambda2=0.5*math.pi
        else:
            lambda2=-0.5*math.pi
    if x2<0:
       lambda2=math.pi-lambda2
    f=1/1/GE_WGS84_F_INV
    et2=e2/((1-f)*(1-f))
    phi2=math.atan(z2*(1+et2)/math.sqrt(x2**2+y2**2))
    acount=0
    aradius_old=10**30 ## a very large number
    while(abs(aradius-aradius_old)>0.00005  and acount<100):
        acount+=1
        aradius_old=aradius
        aradius=GE_WGS84_Alpha/math.sqrt(1-e2*math.sin(phi2)*math.sin(phi2)) ##ellipsoid_main_normal_section_radius(phi2)
        phi2=math.atan((z2+e2*aradius*math.sin(phi2))/math.sqrt(x2**2+y2**2))
    ##project philambda_to_xy
    kappa0=0.9996
    lambda0=24*math.pi
    lambda0=lambda0/180.00
    xoffset=500000
    yoffset=0.00
    dl=lambda2-lambda0
    t=math.tan(phi2)
    n2=(e2*math.cos(phi2)*math.cos(phi2))/(1-e2)
    L=dl*math.cos(phi2)
    Ni=GE_WGS84_Alpha/math.sqrt(1-e2*math.sin(phi2)*math.sin(phi2))
    ##Mi=ellipsoid_arc(phi2)
    e4=e2**2
    e6=e4*e2
    e8=e6*e2
    M0 = 1 + 0.75 * e2 + 0.703125 * e4 + 0.68359375 * e6 + 0.67291259765625 * e8
    M2 = 0.375 * e2 + 0.46875 * e4 + 0.5126953125 * e6 + 0.538330078125 * e8
    M4 = 0.05859375 * e4 + 0.1025390625 * e6 + 0.25 * e8
    M6 = 0.01139322916666667 * e6 + 0.025634765625 * e8
    M8 = 0.002408551504771226 * e8
    Mi = GE_WGS84_Alpha * (1 - e2) * (M0 * phi2 - M2 * math.sin(2 * phi2) + M4 * math.sin(4 * phi2) - M6 * math.sin(6 * phi2) + M8 * math.sin(8 * phi2))
    x = (((5 - 18 * t * t + t * t * t * t + 14 * n2 - 58 * t * t * n2) * L * L / 120.00 + (1 - t * t + n2) / 6.00) * L * L + 1) * L * kappa0 * Ni + xoffset
    y = Mi + (Ni * t / 2) * L * L + (Ni * t / 24) * (5 - t * t + 9 * n2 + 4 * n2 * n2) * L * L * L * L + (Ni * t / 720) * (61 - 58 * t * t) * L * L * L * L * L * L
    y = y * kappa0 + yoffset
    return x,y

## convert U and V components of wind to Wind speed and direction.
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

directory_to_read='C:/Users/plgeo/OneDrive/PC Desktop/MATLAB DRIVE/MSc_Meteorology/Input data/'

## process NETCDF files to extract all input variables at the predefined grid.

filetypo=directory_to_read+'ERA_data_part'

List_of_all_input_variables=[]
Dict_var={}
for i in range(16):
    filename=filetypo+str(i+1)+'.nc'
    ds=nc.Dataset(filename)
    ## globally used variables
    Longitude = ds.variables['longitude'][:]
    Latitude = ds.variables['latitude'][:]
    Greg_timestamp = ds.variables['time'][:]  ## timestamp in gregorian diary
    Variable_list=list(ds.variables.keys())
    for var in ds.variables.keys():
        var_info=ds.variables[var]## shows all information about the data , such as scale_factor, fillvalue, units, full name, data arrangement in dimensions etc..
        if 'level' in Variable_list:
           if var_info.long_name=='pressure_level':
              var_plevel=var_info[:].tolist()
        else:
           var_plevel=[]
        if var_info.long_name!='longitude' and var_info.long_name!='latitude' and var_info.long_name!='time' and var_info.long_name!='pressure_level':
           if var_plevel:
              for level in var_plevel:
                  if level!=1000 and level!=975:
                      ## exclude surface variables.
                      var_name=var_info.long_name+'_'+str(level)
                      lev_index=var_plevel.index(level)
                      var_dim=var_info.shape## dimensions of the data array
                      # print(var_dim)
                      var_data=np.array(var_info[:][:][:][:],dtype=float)
                      var_data=var_data[:,lev_index,:,:]
                      var_mod_dim=var_data.shape## dimensions of the modified data array
                      Dict_var[var_name]=var_data
           else:
              var_name=var_info.long_name
              var_data=np.array(var_info[:][:][:],dtype=float)
              var_mod_dim = var_data.shape  ## dimensions of the modified data array
              Dict_var[var_name] = var_data
           List_of_all_input_variables.append(var_info.long_name)
# Dict_var() contains all input variables at the selected grid domain of ERA
print(Dict_var.keys())

## Comment: The variables Divergence , U and V component of wind without a level number index refer to 300mb

# build-up dates and times of the period of study
# build timestamps
timestampstart = datetime.strptime("2018-01-01 00:00", "%Y-%m-%d %H:%M")
timestampend = datetime.strptime("2019-12-31 23:00", "%Y-%m-%d %H:%M")
timestamps_generated = pd.date_range(timestampstart, timestampend, freq='60T')
timestamps_generated=timestamps_generated.strftime("%Y%m%d %H:%M:%S")
timestampstart=timestampstart.strftime("%Y-%m-%d %H:%M")##converts datetime to string with the specified format
timestampend=timestampend.strftime("%Y-%m-%d %H:%M")

## build dates
datestart=datetime(2018,1,1)
dateend=datetime(2019,12,31)
## create a list of all dates in the predefined period
ddd = [datestart + timedelta(days=x) for x in range((dateend-datestart).days + 1)]## list of datetime objects with all dates
all_dates=[da.strftime("%Y%m%d") for da in ddd]## string format of the dates.

datestart=all_dates[0]
dateend=all_dates[-1]
#
# process Lightning data
filelight=directory_to_read+'Lightning data Athens.xlsx'
df_lightning_data=pd.read_excel(filelight,sheet_name='Lightning data Athens')
strike_date=df_lightning_data.iloc[:,0].values.tolist()
strike_time=df_lightning_data.iloc[:,1].values.tolist()
strike_lat=df_lightning_data.iloc[:,2].values.tolist()## latitude of the strike event
strike_lon=df_lightning_data.iloc[:,3].values.tolist()## longitude of the strike event
strike_type=df_lightning_data.iloc[:,5].values.tolist()## CC strike is type 2 ,CG strike is type 1
strike_ampl=np.array(df_lightning_data.iloc[:,6],dtype=float)## amplitude of the strike event
strike_time=[timee.strftime("%H:%M:%S.%f") for ind_timee,timee in enumerate(strike_time)]

strike_CG=strike_ampl[np.array(strike_type)==1]
strike_neg_CG=strike_CG[strike_CG<0]

## convert lightning strokes to lightning flashes based on the principle that all strokes occuring within a timeframe of less than 500 ms (95th percentile of flash duration) or 0.5 s are parts of the same flash.
## Alternatively, the principle that the geometric mean of the interstroke interval is 60 ms and they can be as long as several hundreds of ms.
## keeping the first stroke as flash and dropping the rest of the rows is what is performed here. Do you also want to use a separation distance? The 95th percentile of the separation distance between
## strokes of the same flash is 6 km according to CIGRE Fig 2.3. -->95% of strokes within the same flash are within 6 km.
# According to Fig. 9.3. Distributions of interstroke intervals in Arizona and São Paulo. Adapted from Saraiva et al. (2010). --> 95% of interstroke intervals fall within 280-300 ms, so instead of 0.06 or 0.1 use 0.3 secs. 99% is 0.5 secs.  The median flash duration is 180 ms and the 95th quantile is 1100 ms.
## Poelman et al. (2021) proposes to use 0.5s as max interstroke interval ( 99th percentile) given a certain distance threshold.
## Conditions to check for successive strokes of the same flash are : if they are both CG strikes ( type=1) , if they happened the same date and time up to 100millisec level and if their distance is lower than 6km.
## F.G.Zoghzoghy et. al (Lightning activity following the return stroke) propose that 97% of first subsequent stroke occur within 5.8km and 750 ms from the initial stroke and 97% of third subsequents occur within 5.3km and 750 ms. This is consistent with using 6km as max interstroke separation distance.
strokes_flash=[i+1 for i in range(len(strike_time)-1) if strike_type[i]==strike_type[i+1]==1 and strike_date[i]==strike_date[i+1] and float(strike_time[i].split(':')[0])==float(strike_time[i+1].split(':')[0]) and float(strike_time[i].split(':')[1])==float(strike_time[i+1].split(':')[1]) and abs(float(strike_time[i].split(':')[2])-float(strike_time[i+1].split(':')[2]))<0.5 and distlatlon(strike_lat[i],strike_lat[i+1],strike_lon[i],strike_lon[i+1])<6]
df_lightning_data.drop(strokes_flash,axis=0,inplace=True)

del strike_date
del strike_time
del strike_lat
del strike_lon
del strike_type
del strike_ampl

strike_date=np.array(df_lightning_data.iloc[:,0],dtype='str')
strike_time=np.array(df_lightning_data.iloc[:,1],dtype='str')
strike_lat=np.array(df_lightning_data.iloc[:,2],dtype=float)## latitude of the strike event
strike_lon=np.array(df_lightning_data.iloc[:,3],dtype=float)## longitude of the strike event
strike_type=np.array(df_lightning_data.iloc[:,5],dtype=int)## CC strike is type 2 ,CG strike is type 1
strike_ampl=np.array(df_lightning_data.iloc[:,6],dtype=float)## amplitude of the strike event
strikedate=strike_date.tolist()
dates_of_strikes=list(dict.fromkeys(strikedate)) ## a list with all thunderstorm days of the period

min_lat=np.min(strike_lat)
max_lat=np.max(strike_lat)
min_lon=np.min(strike_lon)
max_lon=np.max(strike_lon)

##define a grid whose resolution is hres degrees
## the starting point of each grid cell is the minimum of the space where each set of strikes will be summed.
hmer=float(0.118) # this is in degs and it is roughly equivalent (accuracy on the order of 10m) to 5 km meridian distance, let's find the equivalent so that it is exactly 2 km
hzon=float(0.123) # this is in degs and it is roughly equivalent (accuracy on the order of 10m) to 5 km zonal distance, let's find the equivalent so that it is exactly 2 km
x_h=12.0 ## horizontal resolution in km , defined mannually
y_h=12.0 ## horizontal resolution in km , defined mannually
standard_area=x_h*y_h ## ideal area of the grid with the above resolution in km.

##build limits for lightning 'grid' , add/subtract small numbers rounded to the same decimal as this rounding in order to include data at boundaries.
min_lat=round(min_lat,2)-0.5*hmer
max_lat=round(max_lat,2)+hmer
min_lon=round(min_lon,2)-0.5*hzon
max_lon=round(max_lon,2)+hzon

print (min_lat,max_lat,min_lon,max_lon)

## we might need to change the resolution in order to obtain more meaningful data. Too high resolution might give lot's of zeros.
lat_grid=np.arange(min_lat,max_lat,hmer) ## evenly spaced lat and lon at [start , stop ) .. the endpoint is open , so increase limit of max_lat, max_lon to include all events.
lon_grid=np.arange(min_lon,max_lon,hzon)

## These lines of code demonstrate the presence of a roughly equidistant 1km grid. variable grid_area kept for computational reasons.
Dist_mer=np.zeros((len(lat_grid)-1,))
Dist_zon=np.zeros((len(lat_grid)-1,))
for ll in range(len(lat_grid)-1):
    lat00=lat_grid[ll]
    lon00=lon_grid[ll]
    lat01=lat_grid[ll+1]
    lon01=lon_grid[ll+1]
    dist_mer=distlatlon(lat00,lat01,lon00,lon00)## meridian distance between consecutive points of the grid.
    dist_zon=distlatlon(lat00,lat00,lon00,lon01)## zonal distance between consecutive points of the grid.
    grid_area=dist_mer*dist_zon## area of each grid cell.
    area_factor=standard_area/grid_area ## must be as close to unity as possible
    Dist_mer[ll]=dist_mer
    Dist_zon[ll]=dist_zon

grid_WE,grid_NS=np.meshgrid(lon_grid,lat_grid)


print('Now let us organise data to a grid in the predefined domain\n')
print('Organise data both for regression and classification\n')
print('Timestep should be one hour\n')

# make a list with all events both in a raw form and matrix form.
index_date,List_of_event_date,List_of_event_hour,List_of_event_lat,List_of_event_lon,List_of_event_timestamps=[],[],[],[],[],[]
Strike_grid_matrix=np.zeros((strike_lat.size,lat_grid.size,lon_grid.size))
for k in range(len(strike_lat)):
    event_lat = strike_lat[k]
    event_lon = strike_lon[k]
    peak_cur = strike_ampl[k]
    event_date = strike_date[k]
    event_time = strike_time[k]
    event_time_help = event_time.split(':')
    event_hour_str=str(event_time_help[0])
    event_hour = int(event_time_help[0])
    event_min = int(event_time_help[1])
    event_sec = round(float(event_time_help[2]), 0)
    event_timestamp=event_date+' '+event_hour_str+':'+'00'+':'+'00'## event timestamp converted in hourly increments. timestamp at hour X corresponds to all data between hour X and X+1.
    List_of_event_timestamps.append(event_timestamp)
    List_of_event_date.append(event_date)
    List_of_event_hour.append(event_hour)
    List_of_event_lat.append(event_lat)
    List_of_event_lon.append(event_lon)
    counter = 0
    for i in range(len(lat_grid)-1):
        for j in range(len(lon_grid)-1):
            lat0=lat_grid[i]
            lat1=lat_grid[i+1]
            lon0=lon_grid[j]
            lon1=lon_grid[j+1]
            if (event_lat>lat0 or event_lat==lat0) and event_lat<lat1:
               counter=0.5
            else:
               counter=0
            if (event_lon>lon0 or event_lon==lon0) and event_lon<lon1:
               counter+=0.5
            else:
               counter=0
            if counter==1:
               Strike_grid_matrix[k,i,j]=counter
            else:
               Strike_grid_matrix[k,i,j]=0

# print(List_of_event_timestamps)

# # # In the following, interhour data at hour X is between hour X-1 and hour X
# # # In the following, interhour data at hour X is between hour X-1 and hour X
Strike_grid_categ=np.zeros((len(timestamps_generated),lat_grid.size,lon_grid.size),dtype=int)
Strike_grid_regr=np.zeros((len(timestamps_generated),lat_grid.size,lon_grid.size),dtype=float)
Index_event_dates=[];
event_no=0 ## counts all events
events_not_listed=[]
for dt in range(len(timestamps_generated)):
    tmstamp0 = timestamps_generated[dt]
    tmstamp_help0=tmstamp0.split(' ')
    date0=tmstamp_help0[0]
    time0=tmstamp_help0[1]
    time0_help=time0.split(':')
    hour0=int(time0_help[0])
    min0=int(time0_help[1])
    if tmstamp0 in List_of_event_timestamps:
       index_event = [x for x in range(len(List_of_event_timestamps)) if tmstamp0==List_of_event_timestamps[x]] #lists all events of the specific hour irrespective of their point
       for i in range(len(lat_grid)):
           for j in range(len(lon_grid)):
               lat0 = lat_grid[i]
               lon0 = lon_grid[j]
               # lat1 = lat_grid[i + 1]
               # lon1 = lon_grid[j + 1]
               counter_hour=[] ## measures number of lightning strikes per selected grid point per hour.
               for iii in range(len(index_event)):
                   index=index_event[iii]
                   event_hour = List_of_event_hour[index]
                   event_lat = List_of_event_lat[index]
                   event_lon = List_of_event_lon[index]
                   # if ((event_lat>lat0 or event_lat==lat0) and event_lat<lat1) and ((event_lon>lon0 or event_lon==lon0) and event_lon<lon1):
                   if (event_lat > lat0 - hmer / 2 and event_lat < lat0 + hmer / 2) and (event_lon > lon0 - hzon / 2 and event_lon < lon0 + hzon / 2):
                       count=1
                       event_no+=1
                       counter_hour.append(iii)
                   else:
                       count=0
                       events_not_listed.append(tmstamp0)
               if len(counter_hour)>0:
                   Strike_grid_categ[dt,i,j]=1 ## categorical lightning data at grid
               else:
                   Strike_grid_categ[dt,i,j]=0
               Strike_grid_regr[dt,i,j]=len(counter_hour)/area_factor ##lightning density per area, if counter_hour is an empty list, it returns zero. scale to unit km^2 area using standard_area
    else:
        for i in range(len(lat_grid)):
            for j in range(len(lon_grid)):
                count=0
                Strike_grid_categ[dt, i, j] = 0  ## categorical lightning data at grid
                Strike_grid_regr[dt, i ,j] = 0 ##lightning density per square km
# #
# #
# #
print(np.max(Strike_grid_categ))
print(np.max(Strike_grid_regr))
print(event_no)
#
if event_no != len(List_of_event_date):
   raise error('Not all observations have been gridded')
else:
    print('All observations have been gridded')
#
# ### pre-processing of input variables
#
## reestablish ERA domain

Longitude_mod=[xx for xx in Longitude if xx>22.5 and xx<24.5]
Latitude_mod=[yy for yy in Latitude if yy>37.25 and yy<38.75]
Latitude_mod.sort()## sort in ascending order
Latitude.sort()
index_long_mod=[indlo for indlo in range(len(Longitude)) if Longitude[indlo]>22.5 and Longitude[indlo]<24.5]
index_lat_mod=[indla for indla in range(len(Latitude)) if Latitude[indla]>37.25 and Latitude[indla]<38.75]

# re-write input variables in the new domain. the data is arranged as (time,latitude,longitude). Item is the variable name and vardata is the variable values. The dictionary of variables is updated with the new domain dimensions.
# also some variables are transformed to more reasonable units and the filling values are replaced with zeros.
go=9.81 ## surface gravitational acceleration in m/s^2
Wind_U={} ## store all pressure level U components for transformation purposes
Wind_V={}
Dict_var_copy=copy.copy(Dict_var)
for item in Dict_var_copy:
    vardata=Dict_var[item]
    varname=item
    vardata=vardata[:,index_lat_mod[0]:index_lat_mod[-1]+1,index_long_mod[0]:index_long_mod[-1]+1]
    var_name_name=varname.split('_')[0]
    if var_name_name=='U component of wind':
       if len(varname.split('_'))<2:
          windname=var_name_name+'_'+'300'## rename some variables to indicate the pressure level of them.
       else:
          windname=varname
       Wind_U[windname]=vardata
    if var_name_name=='V component of wind':
        if len(varname.split('_')) < 2:
           vname=var_name_name+'_'+'300'
        else:
           vname=varname
        Wind_V[vname]=vardata
    ## convert geopotential to geopotential height in dam ( 10m) , change units in other variables and change values where necessary
    if var_name_name=='Geopotential':
       varname=var_name_name+' '+'height'+'_'+varname.split('_')[1]
       vardata=vardata*(0.1/go)
    elif  var_name_name=='Convective rain rate':
        vardata=vardata*3600 ## converts from mm/s to mm/hr
    elif var_name_name=='Convective precipitation' or var_name_name=='Total precipitation':
        vardata=vardata*1000 ## switch from meters to mm
        var_help=vardata
        var_help[var_help<0.0]=0.0 ## replace filling or negative values with zero.
        vardata=var_help
    elif var_name_name=='Vertical integral of divergence of moisture flux':
        vardata=vardata*1000 ## switch from kg m**-2 s**-1 to g m**-2 s**-1
    elif var_name_name=='Specific humidity':
        vardata=vardata*1000 ## switch from kg/kg to g/g
    elif var_name_name=='Vorticity (relative)' or var_name_name=='Divergence':
        vardata=vardata*(10**5) ## switch from s**-1 to 10**-5 s**-1
    elif var_name_name=='Mean sea level pressure':
        vardata=vardata*(1/1000) ## switch from Pa to kPa
    elif var_name_name=='Temperature':
        vardata=vardata-273 ## convert from Kelvin to Celcius.
    elif var_name_name=='Relative humidity':
        vardata=vardata*(1/100) ## convert percentage to ratio.
        var_help=vardata
        var_help[var_help<0.0]=0.0 ## replace negative values with zero (no physical meaning in negative RH)
        vardata=var_help
    elif var_name_name=='Convective inhibition' or var_name_name=='Convective available potential energy':
        var_help=vardata
        var_help[var_help<0.0]=0.0 ## replace filling or negative values with zero.
        vardata=var_help
    Dict_var.update({varname:vardata})
#
# ##The U,V components of wind , the  Geopotential , the SST and the Vertical integrals of heat flux at all pressure levels  are removed from the dictionary of variables and later , U and V  will be replaced by Wind and Direction.
Dict_var={name:value for (name,value) in Dict_var.items() if name.split('_')[0] not in ['U component of wind','V component of wind' ,'Geopotential','Vertical integral of eastward heat flux','Vertical integral of northward heat flux','Sea surface temperature','Large-scale precipitation']}

# converting U,V components of wind to Wind speed and direction at all pressure levels and adding them to the list of variables
Wind_U_names=list(Wind_U.keys())
Wind_V_names=list(Wind_V.keys())
Wind_speeds={}## Α dictionary containing wind speed arrays at all pressure levels
Wind_dirs={}## Α dictionary containing wind speed arrays at all pressure levels
for i in range(len(Wind_U_names)):
     uitem=Wind_U_names[i]
     vitem=Wind_V_names[i]
     U_comp=Wind_U[uitem]
     V_comp=Wind_V[vitem]
     wname='Wind_Speed'+'_'+uitem.split('_')[1]
     dname='Wind_direction'+'_'+uitem.split('_')[1]
     Wind_speed=np.zeros((U_comp.shape))
     Direction=np.zeros((U_comp.shape))
     for t in range(U_comp.shape[0]):
        for lo in range(U_comp.shape[1]):
            for la in range(U_comp.shape[2]):
                u_c=U_comp[t,lo,la]
                v_c=V_comp[t,lo,la]
                ws,ds=UV_to_WD(u_c,v_c)
                Wind_speed[t,lo,la]=ws
                Direction[t,lo,la]=math.radians(ds) ## convert degrees to radians.
     Wind_speeds[wname]=Wind_speed
     Wind_dirs[dname]=Direction
     Dict_var[wname]=Wind_speed
     Dict_var[dname]=Direction

meteovar_names=list(Dict_var.keys())## create a list with all updated ERA variables after the above processing

# #
# ## non-parametric rank correlation per grid point of lightning data. choose only one grid point from ERA to make the correlation process simpler.
#
# ## find nearest lightning grid points to ERA central point. Be careful : The ERA data is obtained at the centre of the grid cell and not on the boundaries.
index_lat_central = [yyy for yyy in range(len(lat_grid)) if round(lat_grid[yyy],2) >= round(Latitude[index_lat_mod[2]],2)-hmer and round(lat_grid[yyy],2) < round(Latitude[index_lat_mod[2]],1)+hmer] ## it finds all the grid points of lightning data that fall within the first decimal of the ERA grid
index_lon_central = [xxx for xxx in range(len(lon_grid)) if round(lon_grid[xxx],2) >= round(Longitude[index_long_mod[4]],2)-hzon and round(lon_grid[xxx],2) <= round(Longitude[index_long_mod[4]],2)+hzon] ## remove '=' from last condition if you make higher than 5 km resolution
###print the boundaries of the ERA central grid point in the first decimal part.
# print(round(Latitude[index_lat_mod[2]],1)-0.05)
# print(round(Latitude[index_lat_mod[2]],1)+0.05)
# print(round(Longitude[index_long_mod[4]],2)-0.05)
# print(round(Longitude[index_long_mod[4]],2)+0.05)
Lat_grid_central=lat_grid[index_lat_central[0]:index_lat_central[-1]+1]## obtain the latitudes of the lightning data around central ERA grid point.
Lon_grid_central=lon_grid[index_lon_central[0]:index_lon_central[-1]+1] ## obtain the longitudes of the lightning data around central ERA grid point.

##make both kendall-tau and spearman correlation within a mesh 1*1km inside the central grid point of ERA and return a list of the correlated variables alongside the number of times each of them is correlated with the target
Strike_central_regr = Strike_grid_regr[:,index_lat_central[0]:index_lat_central[-1]+1,index_lon_central[0]:index_lon_central[-1]+1]
Strike_central_regr=Strike_central_regr.reshape(Strike_central_regr.shape[0],Strike_central_regr.shape[1]*Strike_central_regr.shape[2])
Strike_central_categ=Strike_grid_categ[:,index_lat_central[0]:index_lat_central[-1]+1,index_lon_central[0]:index_lon_central[-1]+1]
Strike_central_categ=Strike_central_categ.reshape(Strike_central_categ.shape[0],Strike_central_categ.shape[1]*Strike_central_categ.shape[2])

List_of_correlated_vars_spear={}
List_of_correlated_vars_kendall={}
## the dataset must be restricted to particular days where there were lightning strikes at the specified sub-grid. Also choose the first year only.
## builts a n element list around an element which is an array index.
def list_extension(dfo):
    Df=[]
    n=120
    for i in range(n):
        df=dfo-int(n/2)+i
        Df.append(df)
    return Df

## removes dublicate values that are aside,  from a list in ascending order.
def remove_listdublicates(DF):
    list_rem=[v for i,v in enumerate(DF) if i==0 or v!=DF[i-1]]
    return list_rem

## load trained GB classifier and get most important features.
directory_to_load_models="C:/Users/plgeo/OneDrive/PC Desktop/MATLAB DRIVE/MSc_Meteorology/trained_models/"
# fileclass=directory_to_load_models+'GB_classifier_'+'half_area_west.sav'
fileclass=directory_to_load_models+'GB_classifier_'+'central grid.sav'
model_class = pickle.load(open(fileclass, 'rb'))
feature_importance0=model_class.get_booster().get_score(importance_type='gain')
features_classifier = list(feature_importance0.keys()) ## input features of the classifier are returned with the raw they should.
## sort feature importances by gain , in descending order.
feature_gains_sorted={fname: fvalue for fname, fvalue in sorted(feature_importance0.items(), key=lambda item: item[1] , reverse=True)}
important_features=list(feature_gains_sorted.keys())

# ## IO_var is a matrix that contains the input variables at the ERA domain.
IO_var=np.zeros((Strike_grid_regr.shape[0],len(features_classifier),len(Latitude_mod),len(Longitude_mod)))
features_classifier_copy=features_classifier.copy()
del features_classifier_copy['latitude']
del features_classifier_copy['longitude']
for v in range(len(features_classifier_copy)):
      var = features_classifier_copy[v]
      input_var=Dict_var[var]
      IO_var[0:,v,0:,0:]=input_var

def list_day_extension(dfo):
    Df=[]
    n=120
    for i in range(n):
        df=dfo-int(n/2)+i
        Df.append(df)
    return Df

## Isolate indices of ERA domain lat lon and lat lon to correlate meteorological features with all available lightning data. The selected ERA domain must marginally include all lightning grid points.
lat_era_light={ind:lat for ind,lat in enumerate(Latitude_mod) for indl,lat_light in enumerate(lat_grid) if (lat<lat_light and lat>lat_light-hmer) or (lat>lat_light and lat<lat_light+hmer)}
lon_era_light = {ind:lon for ind,lon in enumerate(Longitude_mod) for indloo,lon_light in enumerate(lon_grid) if (lon<lon_light and lon>lon_light-hzon) or (lon>lon_light and lon<lon_light+hzon)}
lon_era_light_compl = {2:Longitude_mod[2]} ## adds the western 'correlatable' point of ERA.
lon_era_light_compl.update(lon_era_light)
Lat_era_light = list(lat_era_light.values()) ## returns the selected latitudes of ERA
Lon_era_light = list(lon_era_light_compl.values()) ## returns the selected longitudes of ERA
del lon_era_light
lat_era_light = list(lat_era_light.keys()) ## returns the required indices of the latitudes of ERA to use.
lon_era_light = list(lon_era_light_compl.keys())## returns the required indices of the longitudes of ERA to use.

IO_var_ = IO_var[0:,0:,lat_era_light[0]:lat_era_light[-1]+1,lon_era_light[0]:lon_era_light[-1]+1] ## further reduce the spatial dimensions of the era domain's meteorological features.

## reshape strike grid data to 2 dimensions
Strike_grid_regr=Strike_grid_regr.reshape(Strike_grid_regr.shape[0],Strike_grid_regr.shape[1]*Strike_grid_regr.shape[2])
Strike_grid_categ = Strike_grid_categ.reshape(Strike_grid_categ.shape[0],Strike_grid_categ.shape[1]*Strike_grid_categ.shape[2])

directory_to_save_figures = "C:/Users/plgeo/OneDrive/PC Desktop/MATLAB DRIVE/MSc_Meteorology/Figures/"

##Gradient boosting classification via cross-validated bayesian optimisation
## Gradient boosting trees generally require more shallow structures( less tree leafs) compared to RFs to fit the training data well .
## Also they perform better by random sampling without replacement (vs bagging of RFs) , by using half the data for each random sampling -- stochasting gradient boosting.
## Crucial for performance is the learning rate or shrinkage/regularisation factor which competes with tree depth
n_trees_gb =hp.quniform('num_boosted_rounds',800,1000,1) #(500,1000) , (400,900) ...
## depth of each tree
max_dep_gb=hp.quniform('max_depth',1,6,1)
# define ratio of features to use for each tree
num_feat_gb=hp.uniform('colsample_bytree',0.2,0.5)
# define ratio of features to use for each node(split).
num_split_gb=0.5
## define subsample fraction for stochastic gradient descent
sub_sample_gb=hp.uniform('subsample',0.350,0.600)
## define shrinkage rates (learning rates)
learn_rate_gb = hp.loguniform('learning_rate',math.log(0.001),math.log(0.0041))
## minimum loss reduction required to make a further partition on a leaf node of the tree. gamma param. makes sure that we are not splitting a node such that the overall gain from parent to children is very low.
gamma_gb=hp.uniform('gamma',0.05,0.25)
## minimum sum of weights of all observations required in a child node. controls the minimum number of samples needed to justify a split on a node.
sum_child_node=hp.quniform('min_child_weight', 3, 5, 1)
## L2-regularisation
reg_lambda=hp.uniform('reg_lambda',0.03,0.05)
## maximum delta step
max_delta_step=hp.quniform('max_delta_step',1,2,1)
gbm_grid_params = {'num_boosted_rounds': n_trees_gb,'max_depth': max_dep_gb,'subsample':sub_sample_gb,'learning_rate':learn_rate_gb,'gamma':gamma_gb,'reg_lambda':reg_lambda,'min_child_weight':sum_child_node,'max_delta_step':max_delta_step,'colsample_bytree':num_feat_gb}


## returns a report of the mean cross validation scores of classification, input is the model and the evaluation set consisting of train, validate pairs of data and output is a pandas dataframe with the classification scores
## in all cases the model is fit.
def evaluate_classifier(model,eval_set):
    train_set=eval_set[0]
    validation_set=eval_set[1]
    Xtrain_set=train_set[0]
    Ytrain_set=train_set[1]
    Xval_set=validation_set[0]
    Yval_set=validation_set[1]
    no_thund_prec,no_thund_recall,no_thund_f1,no_thund_support = 0,0,0,0
    thund_prec , thund_recall ,thund_f1 ,thund_support ,acc_f1 , acc_support = 0,0,0,0,0,0
    balanced_prec , balanced_recall , balanced_f1 , balanced_support = 0,0,0,0
    weighted_prec , weighted_recall , weighted_f1 , weighted_support = 0,0,0,0
    HSS_val,PSS_val,FA_val=0,0,0
    for val_round in range(len(Xval_set)):
        xtrain = Xtrain_set[val_round]
        ytrain = Ytrain_set[val_round]
        xval = Xval_set[val_round]
        yval = Yval_set[val_round]
        model.fit(xtrain, ytrain)  ## to display mean cross validation classification accuracy scores we have to retrain the model with the optimal hyperparams, with the splitted sets.
        ypredval = model.predict(xval)
        class_val_report = classification_report(yval, ypredval, target_names=['low activity', 'high activity'],output_dict=True, zero_division=1)
        t_n, f_p, f_n, t_p = confusion_matrix(yval, ypredval).ravel()
        hss = HSS_score(t_p, f_p, f_n, t_n)  ##Heidke skill score.
        pss = Pierce_score(t_p, f_p, f_n, t_n)  ## Pierce skill score
        fal = False_alarm(f_p,t_n)
        HSS_val+=hss
        PSS_val+=pss
        FA_val+=fal
        HSS_mean=round(HSS_val/(val_round + 1), 3)
        PSS_mean = round(PSS_val / (val_round + 1), 3)
        Fal_mean = round(FA_val/(val_round+1),3)
        value_00 = class_val_report['low activity']['precision']
        no_thund_prec += value_00
        no_thund_mean_prec = round(no_thund_prec / (val_round + 1), 3)
        value_01 = class_val_report['low activity']['recall']
        no_thund_recall += value_01
        no_thund_mean_recall = round(no_thund_recall / (val_round + 1), 3)
        value_02 = class_val_report['low activity']['f1-score']
        no_thund_f1 += value_02
        no_thund_mean_f1 = round(no_thund_f1 / (val_round + 1), 3)
        value_03 = class_val_report['low activity']['support']
        no_thund_support += value_03
        no_thund_mean_support = int(round(no_thund_support / (val_round + 1), 0))
        value_10 = class_val_report['high activity']['precision']
        thund_prec += value_10
        thund_mean_prec = round(thund_prec / (val_round + 1), 3)
        value_11 = class_val_report['high activity']['recall']
        thund_recall += value_11
        thund_mean_recall = round(thund_recall / (val_round + 1), 3)
        value_12 = class_val_report['high activity']['f1-score']
        thund_f1 += value_12
        thund_mean_f1 = round(thund_f1 / (val_round + 1), 3)
        value_13 = class_val_report['high activity']['support']
        thund_support += value_13
        thund_mean_support = int(round(thund_support / (val_round + 1), 0))
        value_22 = class_val_report['accuracy']
        acc_f1 += value_22
        acc_mean_f1 = round(acc_f1 / (val_round + 1), 3)
        value_30 = class_val_report['macro avg']['precision']
        balanced_prec += value_30
        balanced_mean_prec = round(balanced_prec / (val_round + 1), 3)
        value_31 = class_val_report['macro avg']['recall']
        balanced_recall += value_31
        balanced_mean_recall = round(balanced_recall / (val_round + 1), 3)
        value_32 = class_val_report['macro avg']['f1-score']
        balanced_f1 += value_32
        balanced_mean_f1 = round(balanced_f1 / (val_round + 1), 3)
        value_33 = class_val_report['macro avg']['support']
        balanced_support += value_33
        balanced_mean_support = int(round(balanced_support / (val_round + 1), 0))
        value_40 = class_val_report['weighted avg']['precision']
        weighted_prec += value_40
        weighted_mean_prec = round(weighted_prec / (val_round + 1), 3)
        value_41 = class_val_report['weighted avg']['recall']
        weighted_recall += value_41
        weighted_mean_recall = round(weighted_recall / (val_round + 1), 3)
        value_42 = class_val_report['weighted avg']['f1-score']
        weighted_f1 += value_42
        weighted_mean_f1 = round(weighted_f1 / (val_round + 1), 3)
        value_43 = class_val_report['weighted avg']['support']
        weighted_support += value_43
        weighted_mean_support = int(round(weighted_support / (val_round + 1), 0))
        class_val_report['low activity']['precision'] = no_thund_mean_prec
        class_val_report['low activity']['recall'] = no_thund_mean_recall
        class_val_report['low activity']['f1-score'] = no_thund_mean_f1
        class_val_report['low activity']['support'] = int(no_thund_mean_support)
        class_val_report['high activity']['precision'] = thund_mean_prec
        class_val_report['high activity']['recall'] = thund_mean_recall
        class_val_report['high activity']['f1-score'] = thund_mean_f1
        class_val_report['high activity']['support'] = int(thund_mean_support)
        class_val_report['accuracy'] = acc_mean_f1
        class_val_report['balanced accuracy'] = class_val_report['macro avg']
        del class_val_report['macro avg']
        class_val_report['balanced accuracy']['precision'] = balanced_mean_prec
        class_val_report['balanced accuracy']['recall'] = balanced_mean_recall
        class_val_report['balanced accuracy']['f1-score'] = balanced_mean_f1
        class_val_report['balanced accuracy']['support'] = int(balanced_mean_support)
        class_val_report['weighted accuracy'] = class_val_report['weighted avg']
        del class_val_report['weighted avg']
        class_val_report['weighted accuracy']['precision'] = weighted_mean_prec
        class_val_report['weighted accuracy']['recall'] = weighted_mean_recall
        class_val_report['weighted accuracy']['f1-score'] = weighted_mean_f1
        class_val_report['weighted accuracy']['support'] = int(weighted_mean_support)
    HR_mean=class_val_report['high activity']['recall'] ## hit rate
    Sens_mean=class_val_report['balanced accuracy']['recall'] ## balanced sensitivity
    F1_mean=class_val_report['balanced accuracy']['f1-score'] ## average F1 score
    class_val_report=pd.DataFrame(class_val_report).transpose()
    list_val_scores=pd.DataFrame([HSS_mean,PSS_mean,HR_mean,Fal_mean,Sens_mean,F1_mean]).transpose()
    return class_val_report,list_val_scores

def cross_val_class_set(X,Y):
    cv = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
    X_train = [];
    Y_train = [];
    X_val = [];
    Y_val = []
    for train, validate in cv.split(X,Y):
        xtra = X.loc[train,:].values
        ytra = Y[train]
        x_val = X.loc[validate,:].values
        y_val = Y[validate]
        X_train.append(xtra)
        Y_train.append(ytra)
        X_val.append(x_val)
        Y_val.append(y_val)
    eval_set=[[X_train,Y_train],[X_val,Y_val]]
    return {'cv':cv,'eval_set':eval_set}

def cross_val_set(X,Y):
    cv = CV_kfold(n_splits=5, shuffle=False, random_state=None)
    X_train = [];
    Y_train = [];
    X_val = [];
    Y_val = []
    for train, validate in cv.split(X,Y):
        x_train = X.loc[train,:].values
        y_train = Y[train]
        x_val = X.loc[validate,:].values
        y_val = Y[validate]
        X_train.append(x_train)
        Y_train.append(y_train)
        X_val.append(x_val)
        Y_val.append(y_val)
    eval_set=[[X_train,Y_train],[X_val,Y_val]]
    return {'cv':cv,'eval_set':eval_set}

def test_classifier(y1,y2):
    class_report = classification_report(y1, y2, target_names=['no thunderstorm', 'thunderstorm'],output_dict=True, zero_division=1)
    class_report['balanced accuracy'] = class_report['macro avg']
    del class_report['macro avg']
    class_report['weighted accuracy'] = class_report['weighted avg']
    del class_report['weighted avg']
    Hr=class_report['thunderstorm']['recall'] ## hit rate
    Sens=class_report['balanced accuracy']['recall'] ## balanced sensitivity
    F1=class_report['balanced accuracy']['f1-score'] ## average F1 score
    class_report = pd.DataFrame(class_report).round(decimals=3).transpose()
    tn, fp, fn, tp = confusion_matrix(y1, y2).ravel()
    Fal = False_alarm(fp, tn)
    Hss=HSS_score(tp, fp, fn, tn)
    Pss=Pierce_score(tp, fp, fn, tn)
    Hss=round(Hss,3)
    Pss=round(Pss,3)
    Fal=round(Fal,3)
    Hr=round(Hr,3)
    Sens=round(Sens,3)
    F1=round(F1,3)
    list_scores=pd.DataFrame([Hss,Pss,Hr,Fal,Sens,F1]).transpose()
    return class_report,list_scores

def test_subs_classifier(y1,y2):
    class_report = classification_report(y1, y2, target_names=['low activity', 'high activity'],output_dict=True, zero_division=1)
    class_report['balanced accuracy'] = class_report['macro avg']
    del class_report['macro avg']
    class_report['weighted accuracy'] = class_report['weighted avg']
    del class_report['weighted avg']
    Hr=class_report['high activity']['recall'] ## hit rate
    Sens=class_report['balanced accuracy']['recall'] ## balanced sensitivity
    F1=class_report['balanced accuracy']['f1-score'] ## average F1 score
    class_report = pd.DataFrame(class_report).round(decimals=3).transpose()
    tn, fp, fn, tp = confusion_matrix(y1, y2).ravel()
    Fal = False_alarm(fp, tn)
    Hss=HSS_score(tp, fp, fn, tn)
    Pss=Pierce_score(tp, fp, fn, tn)
    Hss=round(Hss,3)
    Pss=round(Pss,3)
    Fal=round(Fal,3)
    Hr=round(Hr,3)
    Sens=round(Sens,3)
    F1=round(F1,3)
    list_scores=pd.DataFrame([Hss,Pss,Hr,Fal,Sens,F1]).transpose()
    return class_report,list_scores

def HSS_score(tp,fp,fn,tn):
    hss = 2 * (tp * tn - fp * fn) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))  ##Heidke skill score. tp -->a , b-->fp, c--> fn, d-->tn , determines excess skill over random chance
    return hss

def Pierce_score(tp,fp,fn,tn):
    pss=(tp*tn-fp*fn)/((tp+fn)*(fp+tn))
    return pss

## False alarm rate
def False_alarm(pf,pt):
    fa=pf/(pf+pt)
    return fa

def nonparamPD(ytt,cp):
    NPDM = KernelDensity(bandwidth=1,kernel='gaussian')  ## stands for non-parametric probability distribution model, use defaults
    dataset_NPDM = ytt.reshape((ytt.size, 1))
    NPDM.fit(dataset_NPDM)
    probabilities = np.exp(NPDM.score_samples(dataset_NPDM))
    control_param = float(cp) ## a parameter that controls the weight sampling -- given externally and possibly integrated within the optimisation.
    epsilon = 0.0000001 ## a very small value that ensures that weights are non-negative no matter the choice of bandwidth, kernel and control parameter of the sample weights.
    fsamp = np.array([max([1 - (control_param * probabilities[pro]), epsilon]) for pro in range(len(probabilities))])  ## function to control sampling weights.
    samp_weighs = fsamp/np.mean(fsamp)
    return samp_weighs,NPDM

def eval_nonparamPD(NPDM,yens):
    yens=yens.reshape((yens.size,1))
    probv=np.exp(NPDM.score_samples(yens))
    return probv

def evaluate_knnregressor(model,eval_set):
    train_set = eval_set[0]
    validation_set = eval_set[1]
    Xtrain_set = train_set[0]
    Ytrain_set = train_set[1]
    Xval_set = validation_set[0]
    Yval_set = validation_set[1]
    RMSE, RSQ,= 0, 0
    for val_round in range(len(Xval_set)):
        xtrain = Xtrain_set[val_round]
        ytrain = Ytrain_set[val_round]
        xval = Xval_set[val_round]
        yval = Yval_set[val_round]
        model.fit(xtrain,ytrain)  ## to display mean cross validation regression scores we have to retrain the model with the optimal hyperparams, with the splitted sets.
        ypredval=model.predict(xval)
        rsq = r2_score(yval, ypredval,multioutput='variance_weighted') #multioutput='variance_weighted'
        RSQ += rsq
        RSQ_ = round(RSQ / (val_round + 1), 3)
        rmse = mean_squared_error(yval, ypredval, squared=False)
        RMSE += rmse
        RMSE_ = round(RMSE / (val_round + 1), 3)
    return RSQ_,RMSE_

def evaluate_regressor(model,eval_set,tol):
    train_set = eval_set[0]
    validation_set = eval_set[1]
    Xtrain_set = train_set[0]
    Ytrain_set = train_set[1]
    Xval_set = validation_set[0]
    Yval_set = validation_set[1]
    RMSE, RSQ, Log_loss, loss_func = 0, 0, 0, 0
    Yvalid=[];Ypredvalid=[]
    for val_round in range(len(Xval_set)):
        xtrain = Xtrain_set[val_round]
        ytrain = Ytrain_set[val_round]
        xval = Xval_set[val_round]
        yval = Yval_set[val_round]
        # xtrain_nz = np.tile(xtrain,(4,1))
        # ytrain_nz = np.tile(ytrain,4)
        xtrain_nz = xtrain
        ytrain_nz = ytrain
        model.fit(xtrain_nz,ytrain_nz)  ## to display mean cross validation regression scores we have to retrain the model with the optimal hyperparams, with the splitted sets.
        yval_nz = yval
        xval_nz = xval
        ypredval_nz=model.predict(xval_nz)
        Bias=[yval_nz[ypv]-ypredval_nz[ypv] for ypv in range(len(ypredval_nz)) if np.logical_and((yval_nz[ypv]>ypredval_nz[ypv]),(yval_nz[ypv]-ypredval_nz[ypv]>tol))] ## creates a list of positive biases on the validation set.
        Abs_error=abs(ypredval_nz-yval_nz) ## abs error only on the non-zero validation instances
        Abs_error=list(Abs_error)
        ypredval_nz=normalizer_y.inverse_transform(ypredval_nz.reshape(-1,1)).ravel()
        yval_nz=normalizer_y.inverse_transform(yval_nz.reshape(-1,1)).ravel()
        ## used only when target is log-transformed.
        ypredval_nz=np.expm1(ypredval_nz)
        yval_nz=np.expm1(yval_nz)
        ##########################################
        ##### bound the data to prevent weights from exploding-applicable only to log transformed targets. #########
        ypredval_nz[ypredval_nz<=0]=0.0001
        yval_nz[yval_nz<=0]=0.0001
        ypredval_nz[ypredval_nz>np.max(np.expm1(normalizer_y.inverse_transform(ytrain_nz.reshape(-1,1)).ravel()))]=np.max(np.expm1(normalizer_y.inverse_transform(ytrain_nz.reshape(-1,1)).ravel()))
        ######################################################
        yprval3h = np.array([sum(ypredval_nz[i:i + 2]) for i in range(0, len(ypredval_nz), 2)])
        yvalh3 = np.array([sum(yval_nz[i:i + 2]) for i in range(0, len(yval_nz), 2)])
        Yvalid.append(yval_nz)
        Ypredvalid.append(ypredval_nz)
        contigency_matrix = pd.crosstab(yval_nz, ypredval_nz)
        Xchi2, p, dof, expected_freq = chi2_contingency(contigency_matrix)
        exc_vari = Xchi2 / (dof-xval_nz.shape[1]-1)## estimates the excess variance by a pearson chi-square goodness of fit test between the estimated lightning density as derived from the probability distribution and the synthesized real lightning density, degrees of freedom are corrected with the number of parameters used to derive the prediction.
        loss_func+=np.mean([abs_error-tol for abs_error in Abs_error if abs_error>tol]) ## objective function of the optimisation process is defined as the mean absolude deviation exceeding tolerance limit , i.e. epsilon parameter.
        # loss_func+=statistics.mean(Bias)
        mean_loss_func = round(loss_func / (val_round + 1), 3)
        # rsq= d2_absolute_error_score(np.log1p(yval_nz), np.log1p(ypredval_nz))
        dev = mean_squared_log_error(yval_nz, ypredval_nz, squared=False)
        Log_loss += dev
        Log_loss_mean = round(Log_loss / (val_round + 1), 3)
        rsq = r2_score(yval_nz, ypredval_nz) #multioutput='variance_weighted'
        RSQ += rsq
        RSQ_mean = round(RSQ / (val_round + 1), 3)
        rmse = mean_squared_error(yval_nz, ypredval_nz, squared=False)
        RMSE += rmse
        RMSE_mean = round(RMSE / (val_round + 1), 3)
    return RSQ_mean,RMSE_mean,Log_loss_mean,mean_loss_func,Yvalid,Ypredvalid

## find lightning grid points with the most lightning context
Strike_overall_spatial=np.sum(Strike_central_categ,axis=0)

## in the following , lgp is the symbol for lightning grid point - pca shall run for each lightning grid point contained around the ERA central grid point.
inp_features=features_classifier_copy ## names of input variables.
inp_features.append('latitude')
inp_features.append('longitude')
inp_features.append('Lightning density') ## output feature is lightning density
feature_array=[i for i in inp_features] ## the length of the feature list is input_features+1, 1 goes for the output /target feature.

input_data_help = IO_var_ ## input data at selected subdomain of ERA.
input_data_help = input_data_help.reshape(input_data_help.shape[0],input_data_help.shape[1],input_data_help.shape[2]*input_data_help.shape[3])

index_best_corr = [0,1,6,7,8,11,12,13] ## most correlated lightning grids from the central domain.
index_2nd_best_corr = [2,3,5,10]

input_data_train0 = input_data_help[:, :, 7]
light_train = Strike_grid_regr[0:round(input_data_train0.shape[0]*0.70):, index_best_corr[0:4]]
light_train = light_train.reshape(light_train.shape[0] * 4,1,order='F')
index_train=[st for st in range(light_train.size) if light_train[st]>0]
light_train=light_train[index_train]
log_light_train=np.log1p(light_train)
z_score=zscore(log_light_train)
z_thres=2 ## no of standard deviations to consider as threshold for distinguishing between low and high lightning activity
index_low=np.where(z_score<z_thres)[0].tolist()## indices of training samples where lightning density belongs to the low/moderate region
index_high=np.where(z_score>=z_thres)[0].tolist() ## indices of training samples where lightning density belongs to the high region
light_binary_train = np.zeros((light_train.shape),dtype=int)
light_binary_train[index_high]=int(1)
light_binary_train[index_low]=int(0)
yclass_train=light_binary_train
input_data_train=np.tile(input_data_train0[0:round(input_data_train0.shape[0]*0.70):,:],(4,1))
input_data_train=input_data_train[index_train,:] ##input data used for the complementary classifier.
input_data_train_SVR=input_data_train[index_low,:] ## train SVR only for low/moderate lightning activity
# input_data_train_SVR=np.tile(input_data_train_SVR,(4,1)) ## copies of the input data used for the SV regression.
yreg_train = light_train[index_low] ## target of the SV regression, i.e. the low/moderate lightning density instances.
y_train=yreg_train
# y_train = np.tile(yreg_train,(4,1)) ## copies of the target of the SV regression to make the model train with more instances.
in_data = input_data_train ## input data of the subsequent classifier
io_dataset = pd.DataFrame(np.column_stack((in_data, yclass_train)))  ## input & output array of dim (timestamps*variables) , stacked columnwise
io_dataset.columns = inp_features
x_data=io_dataset.iloc[:,0:-1].values
yclass_train = io_dataset.iloc[:,-1].values
df_train_features_dataset = pd.DataFrame(x_data, columns=feature_array[0:-1])
df_train_regression_dataset=pd.DataFrame(input_data_train_SVR,columns=feature_array[0:-1])
light_test = Strike_grid_regr[round(input_data_train0.shape[0]*0.70):, index_best_corr[0:4]]
light_test = light_test.reshape(light_test.shape[0] * 4,1,order='F')
index_test=[st for st in range(light_test.size) if light_test[st]>0]
light_test=light_test[index_test]
log_light_test=np.log1p(light_test)
index_test_low=np.where(log_light_test<=np.max(np.log1p(y_train)))[0].tolist()## indices of training samples where lightning density belongs to the low/moderate region
index_test_high=np.where(log_light_test>np.max(np.log1p(y_train)))[0].tolist()## indices of training samples where lightning density belongs to the high region
light_binary_test = np.zeros((light_test.shape),dtype=int)
light_binary_test[index_test_high]=int(1)
light_binary_test[index_test_low]=int(0)
y_test=light_test[index_test_low]
yclass_test=light_binary_test
input_data_test=np.tile(input_data_train0[round(input_data_train0.shape[0]*0.70):,:],(4,1))
input_data_test=input_data_test[index_test,:]
input_data_test_SVR=input_data_test[index_test_low,:]
io_dataset_test=pd.DataFrame(np.column_stack((input_data_test,yclass_test)))
io_dataset_test.columns=inp_features
x_data_test=io_dataset_test.iloc[:,0:-1].values
df_test_features_dataset = pd.DataFrame(x_data_test, columns=feature_array[0:-1])
df_test_regression_dataset=pd.DataFrame(input_data_test_SVR,columns=feature_array[0:-1])
del index_train

## keep first n features for regression
def get_regression_features(n,import_features,df_f_train):
    import_features1 = import_features[0:n]
    df_train_r_dataset = df_f_train[import_features1]
    ## get the correlation matrix between selected features
    feature_correlations = feature_corr_matrix(df_train_r_dataset.loc[:, :])
    ### drop highly collinear features
    index_cl = [feature_correlations[np.logical_and(feature_correlations[import_features1[cl]] > 0.9, feature_correlations[import_features1[cl]] < 1)].index.tolist() for cl in range(len(import_features1))]
    index_cl = list(dict.fromkeys(list(chain(*index_cl))))  ## prints all collinear features in one list after removing dublicate names.
    proposed_features_to_drop = ['Vertical velocity_950','Vertical velocity_850', 'Total precipitation','Specific humidity_850','Total column water vapour','Mean sea level pressure']  ## inherited from RF regression analysis
    features_to_drop = [value_drop for ind_drop, value_drop in enumerate(index_cl) if value_drop in proposed_features_to_drop]
    index_conflict={val:ind for ind,val in enumerate(import_features1) if val in features_to_drop and (val=='Specific humidity_850' or val=='Total column water vapour')}
    if len(list(index_conflict.keys()))>1:
       for var in index_conflict.keys():
           if var=='Specific humidity_850':
              index1=index_conflict[var]
           elif var=='Total column water vapour':
              index2=index_conflict[var]
       if index1<index2:
          features_to_drop.remove('Specific humidity_850')
       else:
          features_to_drop.remove('Total column water vapour')
    [import_features1.remove(fd) for fd in features_to_drop]
    inpla='latitude'
    inplo='longitude'
    if inpla not in import_features1:
       import_features1.append(inpla)
    if inplo not in import_features1:
       import_features1.append(inplo)
    return import_features1

normalizer_x = StandardScaler()
# normalizer_x=MinMaxScaler(feature_range=(-1, 1))
normalizer_y = make_pipeline(MinMaxScaler(feature_range=(0, 1)))
eps=0.1
# ##iterative process to decide desired number of features , decision based upon RMSE min. Start by keeping all features and reduce features dimension by one iteratively until you reach 4.
n=len(important_features)
RMSE_trials={};LogL_trials={};Loss_trials={}
while n>3:
    # ## keep reduced features after collinearity removed. re-initialise the input dataset for train and test.
    important_features1=get_regression_features(n,important_features,df_train_regression_dataset)
    df_train_regr_dataset = df_train_regression_dataset.copy()
    df_train_regr_dataset = df_train_regr_dataset[important_features1]
    y_train_log = np.log1p(y_train)
    ytr_train = normalizer_y.fit_transform(y_train_log.reshape(-1, 1)).ravel()
    ## prepare regression dataset for cross-validation.
    dict_cv_trial = cross_val_set(df_train_regr_dataset, ytr_train)
    cvset_trial = dict_cv_trial['cv']
    train_validation_trial = dict_cv_trial['eval_set']
    del df_train_regr_dataset
    del y_train_log
    del ytr_train
    svrtrial = SVR(epsilon=eps, C=1.0, gamma='scale', tol=0.001, kernel='rbf', shrinking=False, cache_size=1000,verbose=False, max_iter=-1)  ## import an SVR with default behaviour.
    pipeline_trial = Pipeline(steps=[('scaler', normalizer_x), ('regressor', svrtrial)])
    RSQ_trial, RMSE_trial, logL_trial, loss_trial, yv_trial, ypv_trial = evaluate_regressor(pipeline_trial,train_validation_trial,eps)
    RMSE_trials[n]=RMSE_trial
    LogL_trials[n]=logL_trial
    Loss_trials[n]=loss_trial
    n-=1

# ## decide on desired number of features based on the minimisation of RMSE error of the trial SVR
n_min=[nfe for nfe,rmse in RMSE_trials.items() if rmse==min(list(RMSE_trials.values()))]
n_keep=int(n_min[0])
important_features1=get_regression_features(n_keep,important_features,df_train_features_dataset)
df_train_regr_dataset = df_train_regression_dataset.copy()
df_train_regr_dataset = df_train_regr_dataset[important_features1]
df_test_regr_dataset = df_test_regression_dataset.copy()
df_test_regr_dataset = df_test_regr_dataset[important_features1] ##test set has already been normalized.
y_train_log = np.log1p(y_train)
ytr_train = normalizer_y.transform(y_train_log.reshape(-1, 1)).ravel() ## numpy array is 'C' ordered
# ## prepare regression dataset for cross-validation.
dict_cv = cross_val_set(df_train_regr_dataset, ytr_train)
cvset = dict_cv['cv']
train_validation_set = dict_cv['eval_set']
# # # # #
# # # # # # ##define SV regression hyperparameter ranges for bayesian optimisation.
proposed_reg_param=[abs(describe(ytr_train).mean+2.5*math.sqrt(describe(ytr_train).variance)),abs(describe(ytr_train).mean+3.5*math.sqrt(describe(ytr_train).variance))] ## Cherkasky et.al. proposes regularisation parameter as abs(ymean+3*ystd)
reg_param_range=np.logspace(math.log(proposed_reg_param[0]),math.log(proposed_reg_param[1]),20) ## regularize around the proposal of Cherkasky et.al.
## for epsilon range, run a KNN regressor several times and compute the RMSE , then calculate epsilon range based on the proposal by Cherkasky et.al. The additive noise levels come from fitting the dataset to a knn regressor.
## range of values of the neighbours, denoted k, is between 2 and 6.
epsi_range=[]
for k in range(2,7,1):
    knnmodel=KNeighborsRegressor(n_neighbors=k)
    pipeline_knnr=Pipeline(steps=[('scaler',normalizer_x),('regressor',knnmodel)])
    RSQ_knn, RMSE_knn=evaluate_knnregressor(pipeline_knnr,train_validation_set)
    sample_size=int(ytr_train.size*0.8)
    dof_knn = sample_size/((sample_size**(1/5))*k) ## degrees of freedom of the knn-regression
    std_factor=math.sqrt(sample_size/(sample_size-dof_knn))
    noise_std=std_factor*RMSE_knn
    epsi_range.append(3*noise_std*math.sqrt(math.log(sample_size)/sample_size))

epsi_range=np.array(epsi_range) ##tolerate absolute errors between start*100% and end*100%

##for the gaussian kernel width, compute the mean distance between adjacent inputs and consider it as the width.
features_scaled=StandardScaler().fit_transform(df_train_regr_dataset)
Mean_adj_dist=[];Max_adj_dist=[]
for ind_imp in range(len(important_features1)-1):
    feature_x0=features_scaled[:,ind_imp]
    feature_x1=features_scaled[:,ind_imp+1]
    mean_adj_dist=statistics.mean([abs(feature_x0[x]-feature_x1[x+1]) for x in range(feature_x0.shape[0]-1)])
    max_adj_dist=max([abs(feature_x0[x]-feature_x1[x+1]) for x in range(feature_x0.shape[0]-1)])
    Mean_adj_dist.append(mean_adj_dist)
    Max_adj_dist.append(max_adj_dist)
Max_adj_dist.sort(reverse=True) ## sorts the list in descending order.
Mean_adj_dist.sort(reverse=True) ## sorts the list in descending order.

kernel_wid=np.linspace(1/(Max_adj_dist[0]-Mean_adj_dist[0]),1/(Max_adj_dist[-1]-Mean_adj_dist[-1]),20) ## controls width of rbf and sigmoid kernels. It is the inverse of the width.
trans_func = hp.choice('kernel',['rbf', 'sigmoid'])#'linear', 'poly',
reg_param=hp.choice('C',reg_param_range)
epsi=hp.choice('epsilon',epsi_range)
kernel_width=hp.choice('gamma',kernel_wid)
## dictionary of parameter values to make randomized search for
svr_grid_params = {'kernel': trans_func,'gamma':kernel_width,'C': reg_param,'epsilon':epsi}
# # #
ytrain_nzz = light_train.ravel() ## this 'dataset' of the target includes ALL non-zero lightning instances. (non-zero lightning density).
#
target_analysis = describe(ytrain_nzz)  ## obtain statistics of the target variable , i.e. the synthetic non-zero lightning data
print('The mean of the target lightning data is ' + str(round(target_analysis.mean, 2)) + ' strikes per 144 km^2')
print('The std of the target lightning data is ' + str(round(math.sqrt(target_analysis.variance), 2)) + ' strikes per 144 km^2')
print('The length of the SVR training set is ' + str(y_train.size))

#fit distribution procedure for synthetic lightning data constituted of non -structural zeros and non-zeros. Non-structural zeros are zeros for which the classifier has predicted a false alarm and the predicted probability of the zero case is <0.5
#change the 'ytrain_nz' to y_train to obtain the probability distribution of the whole data and to y_nz to obtain the probability distribution of the non-zero lightning data only. be-careful , some distributions are defined for 0<=x<=1 only.
## all chosen distributions should be defined for non-negative continuous variables. Distributions for which the mean and variance cannot be computed, are excluded from the analysis. e.g. halfcauchy
prob_distrs_nz = Fitter(ytrain_nzz,distributions=['betaprime', 'chi', 'chi2', 'erlang', 'expon','exponpow', 'exponweib', 'gamma', 'genexpon', 'gengamma', 'genhalflogistic','halfcauchy', 'geninvgauss', 'genlogistic', 'genpareto', 'halflogistic', 'halfnorm', 'invgauss', 'invweibull', 'loggamma', 'logistic', 'loglaplace', 'lognorm', 'loguniform','powerlaw', 'powerlognorm', 'rayleigh', 'wrapcauchy'])
prob_distrs_nz.fit()
# print(prob_distrs_nz.summary()) # the def set of the fitted distribution must be strictly non-negative and must generate strictly non-negative values. if this is not the case,change distribution.
fitted_distribution_target=prob_distrs_nz.get_best(method='sumsquare_error')
fitted_distr_nztarget=fitted_distribution_target[list(fitted_distribution_target.keys())[0]] ## gives the parameters of the best fitted distribution in a dictionary.

print('Synthetic positive lightning data best fitted PD name: ' + list(fitted_distribution_target.keys())[0])

# # ## get class from globals and create an instance in order to convert the string name of the function corresponding to the best fitted PD to a function.
import importlib
def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c
pdname=class_for_name('scipy.stats',list(fitted_distribution_target.keys())[0])
#
#
# ## apply Bayesian Cross-validated optimisation for GBM classifier and for Support vector regression.

def optfunc_SVregrmodels(svr_grid_params):
    cv = cvset
    svr_param_search = {'kernel': svr_grid_params['kernel'],'gamma': svr_grid_params['gamma'],'C': svr_grid_params['C'], 'epsilon': svr_grid_params['epsilon']}
    eps_cv = svr_param_search['epsilon']
    svregr = SVR(tol=0.001, shrinking=False, cache_size=1000, verbose=False, max_iter=-1)
    pipeline_svregr=Pipeline(steps=[('scaler',normalizer_x),('regressor',svregr)])
    pipeline_svregr['regressor'].set_params(**svr_param_search)
    RSQ_eval, RMSE_eval, LogL_eval, loss_fun,yv,ypv = evaluate_regressor(pipeline_svregr, train_validation_set, eps_cv)
    loss = RMSE_eval
    # loss = loss_fun
    return {'loss': loss, 'status': STATUS_OK, 'model_sv': pipeline_svregr, 'params': svr_param_search, 'Rscore': RSQ_eval,'error': [RMSE_eval]}
# # #
# # #
trials = Trials()
best_hyperparams_svreg = fmin(fn=optfunc_SVregrmodels, space=svr_grid_params, algo=rand.suggest, max_evals=5000,trials=trials,rstate=random_seed)
best_SVregrmodel = trials.results[np.argmin([r['loss'] for r in trials.results])]['model_sv']  # get the best model after optimization
best_params_SVR = trials.results[np.argmin([r['loss'] for r in trials.results])]['params']  # #get the best parameters after optimization
RSQ_val, RMSE_val, LogL_val, MAD,Yvali,Ypredvali = evaluate_regressor(best_SVregrmodel, train_validation_set,best_params_SVR['epsilon'])
#
print("SVR log loss on validation: " + str(LogL_val))
print("SVR rmse error on validation: " + str(RMSE_val))  ## prints the mean cross validation RMSE of the best RF model.
print('SVR Coef. of determination on validation', RSQ_val)
print(best_params_SVR)
print('\n')
#
# for ivset,vset in enumerate(Yvali):
#     yvali=vset
#     ypredvali=Ypredvali[ivset]
#     fig, ax = plt.subplots()
#     line1, = ax.plot(ypredvali, label='predicted')
#     line2, = ax.plot(yvali, label='real')
#     ax.legend(handles=[line1, line2])
#     plt.show()
# # #
#
# ## prepare subsequent classification part for cross-validation and bayesian optimisation
# dict_cv_class = cross_val_class_set(df_train_features_dataset, yclass_train)
# train_validation_class_set = dict_cv_class['eval_set']  ## obtain k-fold train/validation datasets.
# cv_class = dict_cv_class['cv']  ## obtain mode of cross validation split for evaluations involving cross-validation
# #
# y_zeros = yclass_train[yclass_train == 0]
# y_nz = yclass_train[yclass_train > 0]
# spw_sc = y_zeros.size / y_nz.size ## changing from int to float increases classification accuracy on test, the opposite decreases it.
#
# def optfunc_gbclassmodels(gbm_grid_params):
#     ## define n-fold stratified cross-validation
#     cvgbc = cv_class
#     gb_param_search = {'num_boosted_rounds': gbm_grid_params['num_boosted_rounds'],
#                        'max_depth': int(gbm_grid_params['max_depth']), 'subsample': gbm_grid_params['subsample'],
#                        'learning_rate': gbm_grid_params['learning_rate'], 'gamma': gbm_grid_params['gamma'],
#                        'reg_lambda': gbm_grid_params['reg_lambda'],
#                        'min_child_weight': gbm_grid_params['min_child_weight'],
#                        'max_delta_step': gbm_grid_params['max_delta_step'],
#                        'colsample_bytree': gbm_grid_params['colsample_bytree']}
#     gbclass = XGBClassifier(**gb_param_search, objective='binary:logistic', base_score=0.6, booster='gbtree',
#                             grow_policy='lossguide', colsample_bylevel=1,
#                             colsample_bynode=num_split_gb, importance_type='gain', scale_pos_weight=spw_sc,
#                             validate_parameters=1,
#                             random_state=random_seed, verbosity=0, threads=1)
#     # gbaccuracy = CV_score(gbclass, train_dataset, y_train, scoring='recall', cv=cvgbc, n_jobs=-5, verbose=False).mean()
#     class_val_report_gb, list_val_scores_gb = evaluate_classifier(gbclass, train_validation_class_set)
#     gbaccuracy = list_val_scores_gb.iat[0,1] ## select Pierce score as the new objective function instead of positive class recall (hit rate) or stick to hit rate.
#     loss_f=1-gbaccuracy ## it is the variable value itself, if the objective is the false alarm rate. It is the 1-variable value if the obj func is hit rate , pierce score, or heide score.
#     # print('CV accuracy:',gbaccuracy)
#     return {'loss': loss_f, 'status': STATUS_OK, 'model_gb': gbclass, 'params_gb': gb_param_search}
#
# trials_class=Trials()
# best_hyperparams_gbclass = fmin(fn=optfunc_gbclassmodels, space=gbm_grid_params, algo=tpe.suggest, max_evals=50,trials=trials_class)
# #
# best_GBclassmodel = trials_class.results[np.argmin([r['loss'] for r in trials_class.results])]['model_gb']
# best_params_GBM = trials_class.results[np.argmin([r['loss'] for r in trials_class.results])]['params_gb']  # #get the best parameters after optimization
# class_val_report_gb, list_val_scores_gb = evaluate_classifier(best_GBclassmodel, train_validation_class_set)
# print("GB classification accuracy on validation:\n",class_val_report_gb)  ## prints the mean cross validation classification report of the best GB class model.
# print("GB advanced scores on validation:\n",list_val_scores_gb)
# print('\n')
# #
# ##refit the regressor with all training set.
train_regr_dataset = df_train_regr_dataset.loc[:,:].values
best_SVregrmodel.fit(train_regr_dataset, ytr_train)  ## refit the best model with all training data, including validation set.
#
# #refit the classifier with all training set.
# best_GBclassmodel.fit(df_train_features_dataset.loc[:,:].values, yclass_train)
#
# ##test subsequent classifier.
# yhigh_predict=best_GBclassmodel.predict(df_test_features_dataset.loc[:,:].values)
# highc_report, advhigh_scores = test_subs_classifier(yclass_test, yhigh_predict)
# print('GB subs. classifier accuracy on test:\n',highc_report)
# print('GB subs. classifier advanced scores on test:\n',advhigh_scores)
# #
# # # ## test regressor alone in the low/moderate lightning activity data.
test_input=df_test_regr_dataset.loc[:,:].values
ytest_svr=best_SVregrmodel.predict(test_input)
ytest_svr=np.expm1(normalizer_y.inverse_transform(ytest_svr.reshape(-1,1)).ravel())
RSQ_regr = r2_score(y_test, ytest_svr,multioutput='variance_weighted')
print('Coefficient of determination of SVR on test:',round(RSQ_regr,3))
fig, ax = plt.subplots()
line1, = ax.plot(ytest_svr, label='predicted')
line2, = ax.plot(y_test, label='real')
ax.legend(handles=[line1, line2])
plt.show()

## extrapolation on test data
# ypred_svr = np.zeros((y_test.shape[0],))
# for inst in range(test_features_dataset.shape[0]):
#     input_ctest = df_test_features_dataset.loc[inst, :].values.reshape(1, -1)
#     input_rtest = df_test_regr_dataset.loc[inst, :].values.reshape(1, -1)
#     ytest_class = model_class.predict(input_ctest) ## the prediction of the first classifier, that decides whether there will be lightning or not.
#     if ytest_class == 0:
#         yprt = 0.0
#     else:
#         one_if_high=best_GBclassmodel.predict(input_ctest) ## the prediction of the subsequent classifier, that decides whether lightning activity will be low/ moderate or high, given that there will be some activity
#         if one_if_high==0:
#            yprt_svr = best_SVregrmodel.predict(input_rtest)
#            yprt_svr = normalizer_y.inverse_transform(yprt_svr.reshape(-1, 1))[:,0] ## SVR provides predictions for only the low/moderate region.
#            if yprt_svr<0:
#               yprt=0
#            else:
#              yprt=yprt_svr
#         elif one_if_high==1:
#             yprt=80
#     ypred_svr[inst] = yprt
# # # # # # # #
# yprednh = np.array([sum(ypred_svr[i:i + 6]) for i in range(0, len(ypred_svr), 6)])
# ytestnh = np.array([sum(y_test[i:i + 6]) for i in range(0, len(y_test), 6)])
# log_error_svr = mean_squared_log_error(y_test[y_test < 80], ypred_svr[np.where(y_test < 80)[0].tolist()],squared=False)
# error_svr = mean_squared_error(y_test[y_test < 80], ypred_svr[np.where(y_test < 80)[0].tolist()], squared=False)
# print('SVR rmse error on test: ' + str(round(error_svr, 3)) + ' ' + str(round(log_error_svr, 3)))
# RSQ_test = r2_score(y_test[y_test < 80], ypred_svr[np.where(y_test < 80)[0].tolist()],multioutput='variance_weighted')
# print('Coefficient of determination on test', round(RSQ_test, 3))
# print(describe(ypred_svr))
# # # # # # # print(describe(y_test))
# # # # # # # ## time-series plot
# fig, ax = plt.subplots()
# line1, = ax.plot(ypred_svr, label='predicted')
# line2, = ax.plot(y_test, label='real')
# ax.legend(handles=[line1, line2])
# plt.show()
# # directory_to_save_models=directory_to_load_models
# # filetosave1=directory_to_save_models+'SVR_'+'half_area_west.sav'
# # pickle.dump(best_SVregrmodel, open(filetosave1, 'wb'))
# # filetosave2=directory_to_save_models+'RF_KDE_'+'half_area_west.sav'
# # pickle.dump(model_KDE, open(filetosave2, 'wb'))