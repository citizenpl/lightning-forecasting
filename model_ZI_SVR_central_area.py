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

mv=0
while mv<len(meteovar_names):
    mname=meteovar_names[mv]
    era_var=Dict_var[mname]
    era_var=era_var[0:11686,2,4] ## chooses the nearest to the centre of lightning data, grid point of ERA , choose 2/3 of the available period for colleration analysis.
    input_info = describe(era_var)  ## descriptive statistics of the input variable , includes minmax,mean,variance, skewness and kurtosis of the sample data.
    stati, pvalue = kstest(np.sort(era_var), stats.norm.cdf(np.sort(era_var), loc=input_info.mean, scale=math.sqrt(input_info.variance)))  ## Kolmogorov Smirnoff test for normality of variable's data distribution.
    if pvalue > 0.05:
        print('The variable ' + mname + ' is probably normaly distributed , norm. test returns p_value= %.3f\n' % pvalue)
    else:
        print('The variable ' + mname + ' is probably not normaly distributed, norm. test returns p_value= %.3f\n' % pvalue)
    space_rank_kendall=0 ; space_rank=0;## measures how many times the specific variable is correlated with lightning density in the grid space around the central ERA point.
    total_rank=0 ## measures how many lightning data grid points have nonzero lightning data
    coef_list_spear=[]
    coef_list_kendall=[]
    for cc in range(Strike_central_regr.shape[1]):
        strikes_per_point=Strike_central_regr[0:11686,cc] ## choose 2/3 of the available period.
        strikes=strikes_per_point.tolist()
        indexnz_help=[list_extension(st) for st in range(len(strikes)) if strikes[st]>0] ## keeps the indices of the non-zero lightning timestamps and the +-12 hours around them and creates a nested list of timestampd indices around any non-zero element.
        indexnz_help_help=list(chain(*indexnz_help)) ## un-nests  nested list of timestamp indices from above and creates a timestamp ordered un-nested list of indices from above.
        indexnz=list(dict.fromkeys(indexnz_help_help)) ## removes all dublicate values in ascending element order. if not new list sorted, then sort afterwards.
        if indexnz:
           total_rank+=1
           Strikes_nz=np.zeros((len(indexnz),),dtype=float)
           Vars_nz=np.zeros((len(indexnz),),dtype=float)
           for index in range(len(indexnz)):
               innn=indexnz[index]
               strikes_nz=strikes_per_point[innn]
               era_vars=era_var[innn]
               Strikes_nz[index]=float("{:.3f}".format(strikes_nz)) ## floating number with 3 decimals
               Vars_nz[index]=float("{:.3f}".format(era_vars))
           coef_spear,p_spear=spearmanr(Vars_nz,Strikes_nz,axis=0,nan_policy='omit',alternative='two-sided')## spearman's correlation , nans omitted, alternative hypothesis is two-sided.
           print('The Spearman correlation index of the variable '+mname+' is coef_spear=%.3f'%coef_spear)
           ## interprete the significance at 99% confidence level. The significance test's validity is influenced by the number of samples.
           alpha=0.01
           if coef_spear!='nan' and p_spear>alpha:
             print('The variable '+mname+' is Spearman uncorrelated with Lightning density'+' since H0 fails to be rejected at p=%.3f'%p_spear)
           elif coef_spear!='nan' and p_spear<=alpha:
             print('The variable ' + mname + ' is Spearman correlated with Lightning density' + ' since H0 is rejected at p=%.3f' % p_spear)
             space_rank+=1
             coef_list_spear.append(coef_spear)
             max_corr_spear=max(coef_list_spear)
             print('\n')
           ## if the input variable is correlated to the lightning density in at least 2 points of the lightning grid around the central ERA cell , then this variable is included in the list of correlated variables. CAPE and RH700 are also included if correlation in at least 1 spatial point.
           if space_rank>3 or (mname in ['Relative humidity_700','Convective available potential energy'] and space_rank!=0):
              List_of_correlated_vars_spear[mname]=[space_rank,max_corr_spear]
           ## interprete the significance at 99% confidence level. The significance test's validity is influenced by the number of samples.
           coef_kend,p_kend=kendalltau(Vars_nz,Strikes_nz,nan_policy='omit')
           print('The Kendall correlation index of the variable '+mname+' is coef_kendal=%.3f'%coef_kend)
           if coef_kend!='nan' and p_kend>alpha:
              print('The variable '+mname+' is Kendall uncorrelated with Lightning density'+' since H0 fails to be rejected at p=%.3f'%p_kend)
           elif coef_kend!='nan' and p_kend<=alpha:
              print('The variable ' + mname + ' is Kendall correlated with Lightning density' + ' since H0 is rejected at p=%.3f'%p_kend)
              space_rank_kendall += 1
              coef_list_kendall.append(coef_kend)
              max_corr_kendall=max(coef_list_kendall) ## stores the max observed correlation in space , between input ERA variable and lightning density.
           print('\n')
           ## if the input variable is correlated to the lightning density in at least 2 points of the lightning grid around the central ERA cell , then this variable is included in the list of correlated variables. CAPE and RH700 are also included if correlation in at least 1 spatial point.
           if space_rank_kendall>3 or (mname in ['Relative humidity_700','Convective available potential energy'] and space_rank_kendall!=0):
              List_of_correlated_vars_kendall[mname]=[space_rank_kendall,max_corr_kendall]
        output_info=describe(strikes_per_point) ## descriptive statistics of the output variable , includes minmax,mean,variance, skewness and kurtosis of the sample data.
        # stati_o,pvalue_o=kstest(strikes_per_point,stats.norm.cdf(Strikes_nz,loc=output_info.mean,scale=math.sqrt(output_info.variance)))
        # if pvalue_o>0.05:
        #     print('The variable lightning density is probably normaly distributed , norm. test returns p_value= %.3f\n'%pvalue_o)
        # else:
        #     print('The variable lightning density is probably not normaly distributed, norm. test returns p_value= %.3f\n'%pvalue_o)
    mv+=1
#
# # both spearmann and kendall correlation indices are evaluated above.
# ## significance tests for normality have also been held.
#
# list with the names of the correlated variables.
listc1=list(List_of_correlated_vars_spear.keys())
listc2=list(List_of_correlated_vars_kendall.keys())
# list with the values of the correlated variables
listv1=list(List_of_correlated_vars_spear.values())
listv2=list(List_of_correlated_vars_kendall.values())

listv1r=[listv1[v1][0] for v1 in range(len(listv1))] # Spearman's spatial rank
listv1c=[listv1[v1][1] for v1 in range(len(listv1))] # Spearman's correlation indices
listv2r=[listv2[v2][0] for v2 in range(len(listv2))] # Kendall's spatial rank
listv2c=[listv2[v2][1] for v2 in range(len(listv2))] # Kendall's correlation indices

if listc1==listc2:
   print('Both correlation indices indicated the same correlated variables. The number of input variables to be selected is: '+str(len(listc1)))
   print(List_of_correlated_vars_spear.items())
   list_input_var = listc1
   Processed_data = np.dstack((np.vstack(listc1), np.vstack(listv1r) , np.vstack(listv1c)))
else:
    print('The one correlation index indicated different variables than the other')
    print(List_of_correlated_vars_spear.items())
    print(List_of_correlated_vars_kendall.items())
    list_input_var=listc2
    Processed_data = np.dstack((np.vstack(listc2), np.vstack(listv2r), np.vstack(listv2c)))

del cc

# ## PCA analysis on the variables derived from primary correlation analysis, for only the central point of ERA.
# ## IO_var is a matrix that contains the input variables at the ERA domain.
IO_var=np.zeros((Strike_grid_regr.shape[0],len(list_input_var),len(Latitude_mod),len(Longitude_mod)))
for v in range(len(list_input_var)):
      var = list_input_var[v]
      input_var=Dict_var[var]
      IO_var[0:,v,0:,0:]=input_var

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

def list_day_extension(dfo):
    Df=[]
    n=120
    for i in range(n):
        df=dfo-int(n/2)+i
        Df.append(df)
    return Df

def cross_val_class_set(X,Y,cv):
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

## This function takes the target variable at each original or log-transformed form and returns a pool of sampling weights based on the frequency of each target sample.
def nonparamPD(ytt,cp,prnz):
    NPDM = KernelDensity(bandwidth=1,kernel='gaussian')  ## stands for non-parametric probability distribution model, use defaults
    dataset_NPDM = ytt.reshape((ytt.size, 1))
    NPDM.fit(dataset_NPDM)
    probabilities = np.exp(NPDM.score_samples(dataset_NPDM))
    prnz=prnz.tolist()
    control_param = float(cp) ## a parameter that controls the weight sampling -- given externally and possibly integrated within the optimisation.
    epsilon = 0.0000001 ## a very small value that ensures that weights are non-negative no matter the choice of bandwidth, kernel and control parameter of the sample weights.
    fsamp = np.array([max([1-(control_param*probabilities[pro]),epsilon])*(prnz[pro]/0.5) if ytt[pro]>0 else max([1-(control_param*probabilities[pro]),epsilon])*(0.5/(1-prnz[pro]))  for pro in range(len(probabilities))]) ## function to control sampling weights.
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
    RMSE, RSQ, Log_loss, loss_func,excess_variance = 0, 0, 0, 0, 0
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
        excess_variance += exc_vari
        mean_excess_variance = round(excess_variance / (val_round + 1), 3)
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
    return RSQ_mean,RMSE_mean,Log_loss_mean,mean_loss_func,Yvalid,Ypredvalid,mean_excess_variance


## find lightning grid points with the most lightning context
Strike_overall_spatial=np.sum(Strike_central_categ,axis=0)

## in the following , lgp is the symbol for lightning grid point - pca shall run for each lightning grid point contained around the ERA central grid point.
inp_features=list_input_var ## names of input variables.
inp_features.append('latitude')
inp_features.append('longitude')
inp_features.append('Lightning density') ## output feature is lightning density
inp_features.append('Lightning density')
feature_array=[i for i in inp_features] ## the length of the feature list is input_features+1, 1 goes for the output /target feature.

var_thres = 0.99 ## explained variance threshold - modifiable
n_drop = len(feature_array) - 2  ## initially , number of dropped features is such that only 1 PC component is used. then add components one by one until explained variance reaches variance threshold
input_data = IO_var[0:17000, 0:, 2, 4]  ## 1year input data at central point of ERA domain. try :,:,2,4 , :,:,2,5 , :,:,3,4 , :,:,3,5
lon1=(Lon_grid_central[0]-float(str(Lon_grid_central[0])[0]+'0'))*np.ones((17000,1))
lon2=(Lon_grid_central[1]-float(str(Lon_grid_central[1])[0]+'0'))*np.ones((17000,1))
lat1=(Lat_grid_central[0]-float(str(Lat_grid_central[0])[0]+'0'))*np.ones((17000,1))
lat2=(Lat_grid_central[1]-float(str(Lat_grid_central[1])[0]+'0'))*np.ones((17000,1))
input_data1=np.column_stack((input_data,lat1,lon1))
input_data2=np.column_stack((input_data,lat1,lon2))
input_data3=np.column_stack((input_data,lat2,lon1))
input_data4=np.column_stack((input_data,lat2,lon2))
input_data=np.concatenate((input_data1,input_data2,input_data3,input_data4))
light_help=Strike_central_regr[0:17000,:]
light =  np.reshape(light_help,(light_help.shape[0]*4,1),order='F') ## target output, which is going to be modified.
light_binary_help = Strike_central_categ[0:17000,:]
light_binary = np.reshape(light_binary_help,(light_binary_help.shape[0]*4,1),order='F')  ## categorical lightning data(binary)
index_train_help = [list_extension(st) for st in range(len(light)) if light[st] > 0]  ## keeps the indices of the non-zero lightning timestamps and the +-12 hours around them and creates a nested list of timestampd indices around any non-zero element.
index_train_help_help = list(chain(*index_train_help))  ## un-nests  nested list of timestamp indices from above and creates a timestamp ordered un-nested list of indices from above.
index_train = list(dict.fromkeys(index_train_help_help))  ## removes all dublicate values in ascending element order. if not new list sorted, then sort afterwards.
yreg = light[index_train]
yreg_pos=yreg[yreg>0]
ycat = light_binary[index_train]
in_data = input_data[index_train, 0:]
outlier_detection=zscore(np.log1p(yreg_pos))
z_thres=3 ## no of standard deviations to consider as threshold for distinguishing between low and high lightning activity
index_within=np.where(outlier_detection<=z_thres)[0].tolist()
yreg_within=yreg_pos[index_within].tolist()
yreg[yreg>max(yreg_within)]=max(yreg_within)
############
io_data_class = np.column_stack((in_data,yreg,ycat))
io_dataset_class = pd.DataFrame(io_data_class)
io_dataset_class.columns = inp_features
xclass_train, xclass_test, yclass_train, yclass_test = train_test_split(io_dataset_class.iloc[:, 0:-1].values,io_dataset_class.iloc[:, -1].values,test_size=1 / 4.0, random_state=random_seed,shuffle=True)
y_train=xclass_train[:,-1]
y_train_pos=y_train[y_train>0]
y_test=xclass_test[:,-1]
df_train_class_dataset = pd.DataFrame(xclass_train[np.where(y_train>0)[0].tolist(),0:-1], columns=feature_array[0:-2])
df_test_class_dataset = pd.DataFrame(xclass_test[:,0:-1], columns=feature_array[0:-2])
# Strike_categ_synthetic = pd.DataFrame(ycat, index=timestamps_generated[index_train])
del index_train_help_help
del index_train_help
del index_train

directory_to_save_figures = "C:/Users/plgeo/OneDrive/PC Desktop/MATLAB DRIVE/MSc_Meteorology/Figures/"

# load trained GB classifier and get most important features.
directory_to_load_models="C:/Users/plgeo/OneDrive/PC Desktop/MATLAB DRIVE/MSc_Meteorology/trained_models/"
fileclass=directory_to_load_models+'GB_classifier_'+'central grid.sav'
filestandclass= directory_to_load_models+'GB_classifier_standardizer_'+'central grid.sav'
model_class = pickle.load(open(fileclass, 'rb'))
standardizer_class = pickle.load(open(filestandclass,'rb'))

feature_importance0=model_class.get_booster().get_score(importance_type='gain')
feature_importance=model_class.feature_importances_
features_classifier = np.array(feature_array[0:-2])
sorted_idx = feature_importance.argsort()
## sort feature importances by gain , in descending order.
feature_gains_sorted={fname: fvalue for fname, fvalue in sorted(feature_importance0.items(), key=lambda item: item[1] , reverse=True)}
important_features0=list(feature_gains_sorted.keys())
important_features = list(features_classifier[sorted_idx])
important_features.reverse()

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
    important_features1=get_regression_features(n,important_features,df_train_class_dataset)
    df_train_regr_dataset = df_train_class_dataset.copy()
    df_train_regr_dataset = df_train_regr_dataset[important_features1]
    y_train_log = np.log1p(y_train_pos)
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
    RSQ_trial, RMSE_trial, logL_trial, loss_trial, yv_trial, ypv_trial,exc_trial = evaluate_regressor(pipeline_trial,train_validation_trial,eps)
    RMSE_trials[n]=RMSE_trial
    LogL_trials[n]=logL_trial
    Loss_trials[n]=loss_trial
    n-=1

# ## decide on desired number of features based on the minimisation of RMSE error of the trial SVR
n_min=[nfe for nfe,rmse in RMSE_trials.items() if rmse==min(list(RMSE_trials.values()))]
n_keep=int(n_min[0])
important_features1=get_regression_features(n_keep,important_features,df_train_class_dataset)
df_train_regr_dataset = df_train_class_dataset.copy()
df_train_regr_dataset = df_train_regr_dataset[important_features1]
df_test_regr_dataset = df_test_class_dataset.copy()
df_test_regr_dataset = df_test_regr_dataset[important_features1] ##test set has already been normalized.
y_train_log = np.log1p(y_train_pos)
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
    epsi_range.append(noise_std*math.sqrt(math.log(sample_size)/sample_size))

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

ytrain_nzz = yreg.ravel() ## this 'dataset' of the target includes ALL lightning instances. (lightning density).
#
target_analysis = describe(ytrain_nzz)  ## obtain statistics of the target variable , i.e. the synthetic non-zero lightning data
print('The mean of the target lightning data is ' + str(round(target_analysis.mean, 2)) + ' strikes per 144 km^2')
print('The std of the target lightning data is ' + str(round(math.sqrt(target_analysis.variance), 2)) + ' strikes per 144 km^2')
print('The length of the SVR training set is ' + str(y_train_pos.size))

#
# ## apply Bayesian Cross-validated optimisation for GBM classifier and for Support vector regression.

def optfunc_SVregrmodels(svr_grid_params):
    cv = cvset
    svr_param_search = {'kernel': svr_grid_params['kernel'],'gamma': svr_grid_params['gamma'],'C': svr_grid_params['C'], 'epsilon': svr_grid_params['epsilon']}
    eps_cv = svr_param_search['epsilon']
    svregr = SVR(tol=0.001, shrinking=False, cache_size=1000,verbose=False, max_iter=-1)
    pipeline_svregr=Pipeline(steps=[('scaler',normalizer_x),('regressor',svregr)])
    pipeline_svregr['regressor'].set_params(**svr_param_search)
    RSQ_eval, RMSE_eval, LogL_eval, loss_fun,yv,ypv,exc_eval = evaluate_regressor(pipeline_svregr, train_validation_set, eps_cv)
    loss = RMSE_eval
    # loss = loss_fun
    return {'loss': loss, 'status': STATUS_OK, 'model_sv': pipeline_svregr, 'params': svr_param_search, 'Rscore': RSQ_eval,'error': [RMSE_eval,exc_eval]}
# # #
# # #
trials = Trials()
best_hyperparams_svreg = fmin(fn=optfunc_SVregrmodels, space=svr_grid_params, algo=rand.suggest, max_evals=1000,trials=trials,rstate=random_seed)
best_SVregrmodel = trials.results[np.argmin([r['loss'] for r in trials.results])]['model_sv']  # get the best model after optimization
best_params_SVR = trials.results[np.argmin([r['loss'] for r in trials.results])]['params']  # #get the best parameters after optimization
RSQ_val, RMSE_val, LogL_val, MAD,Yvali,Ypredvali,exc_val = evaluate_regressor(best_SVregrmodel, train_validation_set,best_params_SVR['epsilon'])

print("SVR log loss on validation: " + str(LogL_val))
print("SVR rmse error on validation: " + str(RMSE_val))  ## prints the mean cross validation RMSE of the best RF model.
print('SVR Coef. of determination on validation', RSQ_val)
print('SVR Excess variance', exc_val)
print(best_params_SVR)
print('\n')

# # ##refit the regressor with all training set.
train_regr_dataset = df_train_regr_dataset.loc[:,:].values
best_SVregrmodel.fit(train_regr_dataset, ytr_train)  ## refit the best model with all training data, including validation set.

# # ## extrapolation on test data

## GB classifier
yclass_predict=model_class.predict(standardizer_class.transform(df_test_class_dataset.loc[:,:].values))
class_report, adv_scores = test_classifier(yclass_test, yclass_predict)

ypred_svr = np.zeros((y_test.shape[0],))
for inst in range(xclass_test.shape[0]):
    input_ctest = df_test_class_dataset.loc[inst, :].values.reshape(1, -1)
    input_rtest = df_test_regr_dataset.loc[inst,:].values.reshape(1,-1)
    ytest_class = model_class.predict(standardizer_class.transform(input_ctest))
    if ytest_class[0]==0:
       yprt=0.0
    else:
       yprt_svr=best_SVregrmodel.predict(input_rtest)
       yprt_svr = math.expm1(normalizer_y.inverse_transform(yprt_svr.reshape(-1, 1))[:, 0])
       if yprt_svr<1:
          yprt=0
       else:
          yprt=yprt_svr
    ypred_svr[inst] = yprt
log_error_svr= round(mean_squared_log_error(y_test[y_test>0],ypred_svr[np.where(y_test>0)[0].tolist()],squared=False),3)
error_svr = round(mean_squared_error(y_test[y_test>0], ypred_svr[np.where(y_test>0)[0].tolist()], squared=False),3)
print('SVR rmse error on test: ' + str(error_svr) + ' ' + str(log_error_svr))
RSQ_test = round(r2_score(y_test[y_test>0],ypred_svr[np.where(y_test>0)[0].tolist()], multioutput='variance_weighted'),3)
RSQ_test_all = round(r2_score(y_test,ypred_svr, multioutput='variance_weighted'),3)
correl_spear,p_test=spearmanr(y_test[y_test>0],ypred_svr[np.where(y_test>0)[0].tolist()],axis=0,nan_policy='omit',alternative='greater')
correl_spear_all,p_test_all=spearmanr(y_test,ypred_svr,axis=0,nan_policy='omit',alternative='greater')
print('Coefficient of determination on test: ' + str(RSQ_test))
print('Spearman correlation on test: ' + str(round(correl_spear,3)))
## time-series plot
fig,ax=plt.subplots(figsize=(12, 8))
line1, = ax.plot(ypred_svr, label='predicted',linewidth=2)
line2, = ax.plot(y_test, label='real',linewidth=1.5)
ax.legend(handles=[line1, line2],loc='upper left')
plt.xlabel('hourly time steps')
plt.ylabel('Lightning density(strikes/144 km^2)')
plt.title('Lightning strikes SVR prediction test central area')
plt.savefig(directory_to_save_figures + 'Lightning strikes SVR prediction test central area.png', dpi='figure',
            format=None, metadata=None,
            bbox_inches='tight', pad_inches=0.5,
            facecolor='auto', edgecolor='auto',
            backend=None, transparent=False)
plt.close()
#
directory_to_save_models=directory_to_load_models
filetosave1=directory_to_save_models+'ZI_SVR_'+'central area.sav'
pickle.dump(best_SVregrmodel, open(filetosave1, 'wb'))
filetosavevars = directory_to_save_models+'ZI_SVR_features.pkl'
pickle.dump(important_features1,open(filetosavevars,'wb'))
filetosave2=directory_to_save_models+'ZI_SVR_normalizer.sav'
pickle.dump(normalizer_y,open(filetosave2,'wb'))
print(len(important_features1))
print(train_regr_dataset.shape[1])
stat_predict=describe(ypred_svr)
stat_real=describe(y_test)
Keys = ['nobs', 'minmax', 'mean', 'variance', 'skewness','kurtosis']
stats_predict={}
stats_real={}
count = 0
for key in Keys:
    if key=='minmax':
       key1=key[0:3]
       key2=key[3:]
       stats_predict[key1] = stat_predict[count][0]
       stats_predict[key2] = stat_predict[count][1]
       stats_real[key1] = stat_real[count][0]
       stats_real[key2] = stat_real[count][1]
    else:
       stats_predict[key]=stat_predict[count]
       stats_real[key]=stat_real[count]
    count += 1
directory_to_save_results="C:/Users/plgeo/OneDrive/PC Desktop/MATLAB DRIVE/MSc_Meteorology/regression results/"
filetosave3=directory_to_save_results+'SV_regression_central_area_validation_results.xlsx'
filetowrite3=pd.ExcelWriter(filetosave3, engine='xlsxwriter')##excel file to save processed data
regr_valid=pd.DataFrame(np.array([LogL_val,RMSE_val,RSQ_val]),index=['log loss','RMSE','Rsquared'],columns=['validation'])
regr_valid.to_excel(filetowrite3,sheet_name='central area', engine="xlsxwriter", startrow=1, startcol=0, header=True)
df_best_svr = pd.DataFrame(list(best_params_SVR.values()), index=list(best_params_SVR.keys()),columns=['value'])
df_head1 = pd.DataFrame([], columns=["", "", "Bayesian optimized ZI-SVR best hyperparameters", "", ""])
df_head1.to_excel(filetowrite3, sheet_name='central area', engine="xlsxwriter", startrow=1, startcol=8, header=True)
df_best_svr.to_excel(filetowrite3, sheet_name='central area', engine="xlsxwriter", startrow=2, startcol=8, header=True)
regr_test=pd.DataFrame(np.array([log_error_svr,error_svr,RSQ_test,RSQ_test_all,correl_spear,correl_spear_all]),index=['log loss','RMSE','Rsquared','Rsquared_all','Spearman','Spearman_all'],columns=['test'])
regr_test.to_excel(filetowrite3,sheet_name='central area', engine="xlsxwriter", startrow=5, startcol=0, header=True)
stat_describe=pd.DataFrame(np.array([list(stats_predict.values()),list(stats_real.values())]),index=['predicted','real'],columns=[list(stats_predict.keys())])
stat_describe.to_excel(filetowrite3,sheet_name='central area', engine="xlsxwriter", startrow=13, startcol=0, header=True)
filetowrite3.save()
