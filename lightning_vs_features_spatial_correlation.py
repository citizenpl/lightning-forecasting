import netCDF4 as nc
import statistics
import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import spearmanr,kendalltau,rankdata,weightedtau,kstest,shapiro,describe
from itertools import chain
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold as CV_kfold
from sklearn.model_selection import cross_val_score as CV_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingClassifier
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score,confusion_matrix,balanced_accuracy_score,log_loss,make_scorer,d2_tweedie_score,mean_poisson_deviance,mean_gamma_deviance,mean_squared_log_error,roc_auc_score,classification_report
from sklearn.metrics import mean_pinball_loss,explained_variance_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import PoissonRegressor
from sklearn.neighbors import KernelDensity
from sklearn.svm import SVC, LinearSVC
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
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

## advanced scores for zero-inflated regression
def _num_samples(x):
    message = "Expected sequence or array-like, got %s" % type(x)
    if hasattr(x, "fit") and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError(
                "Singleton array %r cannot be considered a valid collection." % x
            )
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

def _check_reg_targets(y_true, y_pred, multioutput, dtype="numeric"):
    check_consistent_length(y_true, y_pred)
    y_true = check_array(y_true, ensure_2d=False, dtype=dtype)
    y_pred = check_array(y_pred, ensure_2d=False, dtype=dtype)

    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError(
            "y_true and y_pred have different number of output ({0}!={1})".format(
                y_true.shape[1], y_pred.shape[1]
            )
        )

    n_outputs = y_true.shape[1]
    allowed_multioutput_str = ("raw_values", "uniform_average", "variance_weighted")
    if isinstance(multioutput, str):
        if multioutput not in allowed_multioutput_str:
            raise ValueError(
                "Allowed 'multioutput' string values are {}. "
                "You provided multioutput={!r}".format(
                    allowed_multioutput_str, multioutput
                )
            )
    elif multioutput is not None:
        multioutput = check_array(multioutput, ensure_2d=False)
        if n_outputs == 1:
            raise ValueError("Custom weights are useful only in multi-output cases.")
        elif n_outputs != len(multioutput):
            raise ValueError(
                "There must be equally many custom weights (%d) as outputs (%d)."
                % (len(multioutput), n_outputs)
            )
    y_type = "continuous" if n_outputs == 1 else "continuous-multioutput"

    return y_type, y_true, y_pred, multioutput

def d2_pinball_score(y_true, y_pred, *, sample_weight=None, alpha=0.5, multioutput="uniform_average"):
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)

    if _num_samples(y_pred) < 2:
        msg = "D^2 score is not well-defined with less than two samples."
        warnings.warn(msg, UndefinedMetricWarning)
        return float("nan")

    numerator = mean_pinball_loss(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        alpha=alpha,
        multioutput="raw_values",
    )

    if sample_weight is None:
        y_quantile = np.tile(
            np.percentile(y_true, q=alpha * 100, axis=0), (len(y_true), 1)
        )
    else:
        sample_weight = _check_sample_weight(sample_weight, y_true)
        y_quantile = np.tile(
            _weighted_percentile(
                y_true, sample_weight=sample_weight, percentile=alpha * 100
            ),
            (len(y_true), 1),
        )

    denominator = mean_pinball_loss(
        y_true,
        y_quantile,
        sample_weight=sample_weight,
        alpha=alpha,
        multioutput="raw_values",
    )

    nonzero_numerator = numerator != 0
    nonzero_denominator = denominator != 0
    valid_score = nonzero_numerator & nonzero_denominator
    output_scores = np.ones(y_true.shape[1])

    output_scores[valid_score] = 1 - (numerator[valid_score]  / denominator[valid_score])
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            # return scores individually
            return output_scores
        elif multioutput == "uniform_average":
            # passing None as weights to np.average results in uniform mean
            avg_weights = None
        else:
            raise ValueError(
                "multioutput is expected to be 'raw_values' "
                "or 'uniform_average' but we got %r"
                " instead." % multioutput
            )
    else:
        avg_weights = multioutput

    return np.average(output_scores, weights=avg_weights)

def d2_absolute_error_score(
    y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
    return d2_pinball_score(
        y_true, y_pred, sample_weight=sample_weight, alpha=0.5, multioutput=multioutput)

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

# print (min_lat,max_lat,min_lon,max_lon)

# we might need to change the resolution in order to obtain more meaningful data. Too high resolution might give lot's of zeros.
lat_grid=np.arange(min_lat,max_lat,hmer) ## evenly spaced lat and lon at [start , stop ) .. the endpoint is open , so increase limit of max_lat, max_lon to include all events.
lon_grid=np.arange(min_lon,max_lon,hzon)

print(lon_grid)
print(lat_grid)

grid_WE,grid_NS=np.meshgrid(lon_grid,lat_grid)
print(grid_WE)
print(grid_NS)
#
#
# ## These lines of code demonstrate the presence of a roughly equidistant 1km grid. variable grid_area kept for computational reasons.
# Dist_mer=np.zeros((len(lat_grid)-1,))
# Dist_zon=np.zeros((len(lat_grid)-1,))
# for ll in range(len(lat_grid)-1):
#     lat00=lat_grid[ll]
#     lon00=lon_grid[ll]
#     lat01=lat_grid[ll+1]
#     lon01=lon_grid[ll+1]
#     dist_mer=distlatlon(lat00,lat01,lon00,lon00)## meridian distance between consecutive points of the grid.
#     dist_zon=distlatlon(lat00,lat00,lon00,lon01)## zonal distance between consecutive points of the grid.
#     grid_area=dist_mer*dist_zon## area of each grid cell.
#     area_factor=standard_area/grid_area ## must be as close to unity as possible
#     Dist_mer[ll]=dist_mer
#     Dist_zon[ll]=dist_zon
# print('Now let us organise data to a grid in the predefined domain\n')
# print('Organise data both for regression and classification\n')
# print('Timestep should be one hour\n')
#
# # make a list with all events both in a raw form and matrix form.
# index_date,List_of_event_date,List_of_event_hour,List_of_event_lat,List_of_event_lon,List_of_event_timestamps=[],[],[],[],[],[]
# Strike_grid_matrix=np.zeros((strike_lat.size,lat_grid.size,lon_grid.size))
# for k in range(len(strike_lat)):
#     event_lat = strike_lat[k]
#     event_lon = strike_lon[k]
#     peak_cur = strike_ampl[k]
#     event_date = strike_date[k]
#     event_time = strike_time[k]
#     event_time_help = event_time.split(':')
#     event_hour_str=str(event_time_help[0])
#     event_hour = int(event_time_help[0])
#     event_min = int(event_time_help[1])
#     event_sec = round(float(event_time_help[2]), 0)
#     event_timestamp=event_date+' '+event_hour_str+':'+'00'+':'+'00'## event timestamp converted in hourly increments. timestamp at hour X corresponds to all data between hour X and X+1.
#     List_of_event_timestamps.append(event_timestamp)
#     List_of_event_date.append(event_date)
#     List_of_event_hour.append(event_hour)
#     List_of_event_lat.append(event_lat)
#     List_of_event_lon.append(event_lon)
#     counter = 0
#     for i in range(len(lat_grid)-1):
#         for j in range(len(lon_grid)-1):
#             lat0=lat_grid[i]
#             lat1=lat_grid[i+1]
#             lon0=lon_grid[j]
#             lon1=lon_grid[j+1]
#             if (event_lat>lat0 or event_lat==lat0) and event_lat<lat1:
#                counter=0.5
#             else:
#                counter=0
#             if (event_lon>lon0 or event_lon==lon0) and event_lon<lon1:
#                counter+=0.5
#             else:
#                counter=0
#             if counter==1:
#                Strike_grid_matrix[k,i,j]=counter
#             else:
#                Strike_grid_matrix[k,i,j]=0
#
# # print(List_of_event_timestamps)
#
# # # # In the following, interhour data at hour X is between hour X-1 and hour X
# Strike_grid_categ=np.zeros((len(timestamps_generated),lat_grid.size,lon_grid.size),dtype=int)
# Strike_grid_regr=np.zeros((len(timestamps_generated),lat_grid.size,lon_grid.size),dtype=float)
# Index_event_dates=[];
# event_no=0 ## counts all events
# events_not_listed=[]
# for dt in range(len(timestamps_generated)):
#     tmstamp0 = timestamps_generated[dt]
#     tmstamp_help0=tmstamp0.split(' ')
#     date0=tmstamp_help0[0]
#     time0=tmstamp_help0[1]
#     time0_help=time0.split(':')
#     hour0=int(time0_help[0])
#     min0=int(time0_help[1])
#     if tmstamp0 in List_of_event_timestamps:
#        index_event = [x for x in range(len(List_of_event_timestamps)) if tmstamp0==List_of_event_timestamps[x]] #lists all events of the specific hour irrespective of their point
#        for i in range(len(lat_grid)):
#            for j in range(len(lon_grid)):
#                lat0 = lat_grid[i]
#                lon0 = lon_grid[j]
#                # lat1 = lat_grid[i + 1]
#                # lon1 = lon_grid[j + 1]
#                counter_hour=[] ## measures number of lightning strikes per selected grid point per hour.
#                for iii in range(len(index_event)):
#                    index=index_event[iii]
#                    event_hour = List_of_event_hour[index]
#                    event_lat = List_of_event_lat[index]
#                    event_lon = List_of_event_lon[index]
#                    # if ((event_lat>lat0 or event_lat==lat0) and event_lat<lat1) and ((event_lon>lon0 or event_lon==lon0) and event_lon<lon1):
#                    if (event_lat > lat0 - hmer / 2 and event_lat < lat0 + hmer / 2) and (event_lon > lon0 - hzon / 2 and event_lon < lon0 + hzon / 2):
#                        count=1
#                        event_no+=1
#                        counter_hour.append(iii)
#                    else:
#                        count=0
#                        events_not_listed.append(tmstamp0)
#                if len(counter_hour)>0:
#                    Strike_grid_categ[dt,i,j]=1 ## categorical lightning data at grid
#                else:
#                    Strike_grid_categ[dt,i,j]=0
#                Strike_grid_regr[dt,i,j]=len(counter_hour)/area_factor ##lightning density per area, if counter_hour is an empty list, it returns zero. scale to unit km^2 area using standard_area
#     else:
#         for i in range(len(lat_grid)):
#             for j in range(len(lon_grid)):
#                 count=0
#                 Strike_grid_categ[dt, i, j] = 0  ## categorical lightning data at grid
#                 Strike_grid_regr[dt, i ,j] = 0 ##lightning density per square km
# # #
# # #
# print(np.max(Strike_grid_categ))
# print(np.max(Strike_grid_regr))
# print(event_no)
# #
# if event_no != len(List_of_event_date):
#    raise error('Not all observations have been gridded')
# else:
#     print('All observations have been gridded')
# #
# # ### pre-processing of input variables
# #
# ## reestablish ERA domain
#
# Longitude_mod=[xx for xx in Longitude if xx>22.5 and xx<24.5]
# Latitude_mod=[yy for yy in Latitude if yy>37.25 and yy<38.75]
# Latitude_mod.sort()## sort in ascending order
# Latitude.sort()
# index_long_mod=[indlo for indlo in range(len(Longitude)) if Longitude[indlo]>22.5 and Longitude[indlo]<24.5]
# index_lat_mod=[indla for indla in range(len(Latitude)) if Latitude[indla]>37.25 and Latitude[indla]<38.75]
#
# # re-write input variables in the new domain. the data is arranged as (time,latitude,longitude). Item is the variable name and vardata is the variable values. The dictionary of variables is updated with the new domain dimensions.
# # also some variables are transformed to more reasonable units and the filling values are replaced with zeros.
# go=9.81 ## surface gravitational acceleration in m/s^2
# Wind_U={} ## store all pressure level U components for transformation purposes
# Wind_V={}
# Dict_var_copy=copy.copy(Dict_var)
# for item in Dict_var_copy:
#     vardata=Dict_var[item]
#     varname=item
#     vardata=vardata[:,index_lat_mod[0]:index_lat_mod[-1]+1,index_long_mod[0]:index_long_mod[-1]+1]
#     var_name_name=varname.split('_')[0]
#     if var_name_name=='U component of wind':
#        if len(varname.split('_'))<2:
#           windname=var_name_name+'_'+'300'## rename some variables to indicate the pressure level of them.
#        else:
#           windname=varname
#        Wind_U[windname]=vardata
#     if var_name_name=='V component of wind':
#         if len(varname.split('_')) < 2:
#            vname=var_name_name+'_'+'300'
#         else:
#            vname=varname
#         Wind_V[vname]=vardata
#     ## convert geopotential to geopotential height in dam ( 10m) , change units in other variables and change values where necessary
#     if var_name_name=='Geopotential':
#        varname=var_name_name+' '+'height'+'_'+varname.split('_')[1]
#        vardata=vardata*(0.1/go)
#     elif  var_name_name=='Convective rain rate':
#         vardata=vardata*3600 ## converts from mm/s to mm/hr
#     elif var_name_name=='Convective precipitation' or var_name_name=='Total precipitation':
#         vardata=vardata*1000 ## switch from meters to mm
#         var_help=vardata
#         var_help[var_help<0.0]=0.0 ## replace filling or negative values with zero.
#         vardata=var_help
#     elif var_name_name=='Vertical integral of divergence of moisture flux':
#         vardata=vardata*1000 ## switch from kg m**-2 s**-1 to g m**-2 s**-1
#     elif var_name_name=='Specific humidity':
#         vardata=vardata*1000 ## switch from kg/kg to g/g
#     elif var_name_name=='Vorticity (relative)' or var_name_name=='Divergence':
#         vardata=vardata*(10**5) ## switch from s**-1 to 10**-5 s**-1
#     elif var_name_name=='Mean sea level pressure':
#         vardata=vardata*(1/1000) ## switch from Pa to kPa
#     elif var_name_name=='Temperature':
#         vardata=vardata-273 ## convert from Kelvin to Celcius.
#     elif var_name_name=='Relative humidity':
#         vardata=vardata*(1/100) ## convert percentage to ratio.
#         var_help=vardata
#         var_help[var_help<0.0]=0.0 ## replace negative values with zero (no physical meaning in negative RH)
#         vardata=var_help
#     elif var_name_name=='Convective inhibition' or var_name_name=='Convective available potential energy':
#         var_help=vardata
#         var_help[var_help<0.0]=0.0 ## replace filling or negative values with zero.
#         vardata=var_help
#     Dict_var.update({varname:vardata})
# #
# # ##The U,V components of wind , the  Geopotential , the SST and the Vertical integrals of heat flux at all pressure levels  are removed from the dictionary of variables and later , U and V  will be replaced by Wind and Direction.
# Dict_var={name:value for (name,value) in Dict_var.items() if name.split('_')[0] not in ['U component of wind','V component of wind' ,'Geopotential','Vertical integral of eastward heat flux','Vertical integral of northward heat flux','Sea surface temperature','Large-scale precipitation']}
#
# # converting U,V components of wind to Wind speed and direction at all pressure levels and adding them to the list of variables
# Wind_U_names=list(Wind_U.keys())
# Wind_V_names=list(Wind_V.keys())
# Wind_speeds={}## Α dictionary containing wind speed arrays at all pressure levels
# Wind_dirs={}## Α dictionary containing wind speed arrays at all pressure levels
# for i in range(len(Wind_U_names)):
#      uitem=Wind_U_names[i]
#      vitem=Wind_V_names[i]
#      U_comp=Wind_U[uitem]
#      V_comp=Wind_V[vitem]
#      wname='Wind_Speed'+'_'+uitem.split('_')[1]
#      dname='Wind_direction'+'_'+uitem.split('_')[1]
#      Wind_speed=np.zeros((U_comp.shape))
#      Direction=np.zeros((U_comp.shape))
#      for t in range(U_comp.shape[0]):
#         for lo in range(U_comp.shape[1]):
#             for la in range(U_comp.shape[2]):
#                 u_c=U_comp[t,lo,la]
#                 v_c=V_comp[t,lo,la]
#                 ws,ds=UV_to_WD(u_c,v_c)
#                 Wind_speed[t,lo,la]=ws
#                 Direction[t,lo,la]=math.radians(ds) ## convert degrees to radians.
#      Wind_speeds[wname]=Wind_speed
#      Wind_dirs[dname]=Direction
#      Dict_var[wname]=Wind_speed
#      Dict_var[dname]=Direction
#
# meteovar_names=list(Dict_var.keys())## create a list with all updated ERA variables after the above processing
#
# #
# # ## find nearest lightning grid points to ERA central point. Be careful : The ERA data is obtained at the centre of the grid cell and not on the boundaries.
# index_lat_central = [yyy for yyy in range(len(lat_grid)) if round(lat_grid[yyy],2) >= round(Latitude[index_lat_mod[2]],2)-hmer and round(lat_grid[yyy],2) < round(Latitude[index_lat_mod[2]],1)+hmer] ## it finds all the grid points of lightning data that fall within the first decimal of the ERA grid
# index_lon_central = [xxx for xxx in range(len(lon_grid)) if round(lon_grid[xxx],2) >= round(Longitude[index_long_mod[4]],2)-hzon and round(lon_grid[xxx],2) <= round(Longitude[index_long_mod[4]],2)+hzon] ## remove '=' from last condition if you make higher than 5 km resolution
# Lat_grid_central=lat_grid[index_lat_central[0]:index_lat_central[-1]+1]## obtain the latitudes of the lightning data around central ERA grid point.
# Lon_grid_central=lon_grid[index_lon_central[0]:index_lon_central[-1]+1] ## obtain the longitudes of the lightning data around central ERA grid point.
#
# ##make both kendall-tau and spearman correlation within a mesh 1*1km inside the central grid point of ERA and return a list of the correlated variables alongside the number of times each of them is correlated with the target
# Strike_central_regr = Strike_grid_regr[:,index_lat_central[0]:index_lat_central[-1]+1,index_lon_central[0]:index_lon_central[-1]+1]
# Strike_central_regr=Strike_central_regr.reshape(Strike_central_regr.shape[0],Strike_central_regr.shape[1]*Strike_central_regr.shape[2])
# Strike_central_categ=Strike_grid_categ[:,index_lat_central[0]:index_lat_central[-1]+1,index_lon_central[0]:index_lon_central[-1]+1]
# Strike_central_categ=Strike_central_categ.reshape(Strike_central_categ.shape[0],Strike_central_categ.shape[1]*Strike_central_categ.shape[2])
#
# ## load optimized & trained GB classifier and get most important features.
# directory_to_load_models="C:/Users/plgeo/OneDrive/PC Desktop/MATLAB DRIVE/MSc_Meteorology/trained_models/"
# fileclass=directory_to_load_models+'GB_classifier_'+'central grid.sav'
# model_class = pickle.load(open(fileclass, 'rb'))
# feature_importance=model_class.get_booster().get_score(importance_type='gain')
# sorted_idx = np.array(list(feature_importance.values())).argsort()
# features_classifier = list(feature_importance.keys()) ## input features of the classifier are returned with the raw they should.
# ## sort feature importances by gain , in descending order.
# feature_gains_sorted={fname: fvalue for fname, fvalue in sorted(feature_importance.items(), key=lambda item: item[1] , reverse=True)}
# important_features=list(feature_gains_sorted.keys())
# print(features_classifier)
# print(important_features)

# # ## IO_var is a matrix that contains the input variables at the ERA domain.
# IO_var=np.zeros((Strike_grid_regr.shape[0],len(features_classifier),len(Latitude_mod),len(Longitude_mod)))
# features_classifier_copy=features_classifier.copy()
# features_classifier_copy.remove('latitude')
# features_classifier_copy.remove('longitude')
# for v in range(len(features_classifier_copy)):
#       var = features_classifier_copy[v]
#       input_var=Dict_var[var]
#       IO_var[0:,v,0:,0:]=input_var
# #
# def list_extension(dfo):
#     Df=[]
#     n=120
#     for i in range(n):
#         df=dfo-int(n/2)+i
#         Df.append(df)
#     return Df
#
# ## Isolate indices of ERA domain lat lon and lat lon to correlate meteorological features with all available lightning data. The selected ERA domain must marginally include all lightning grid points.
# lat_era_light={ind:lat for ind,lat in enumerate(Latitude_mod) for indl,lat_light in enumerate(lat_grid) if (lat<lat_light and lat>lat_light-hmer) or (lat>lat_light and lat<lat_light+hmer)}
# lon_era_light = {ind:lon for ind,lon in enumerate(Longitude_mod) for indloo,lon_light in enumerate(lon_grid) if (lon<lon_light and lon>lon_light-hzon) or (lon>lon_light and lon<lon_light+hzon)}
# lon_era_light_compl = {2:Longitude_mod[2]} ## adds the western 'correlatable' point of ERA.
# lon_era_light_compl.update(lon_era_light)
# Lat_era_light = list(lat_era_light.values()) ## returns the selected latitudes of ERA
# Lon_era_light = list(lon_era_light_compl.values()) ## returns the selected longitudes of ERA
# del lon_era_light
# lat_era_light = list(lat_era_light.keys()) ## returns the required indices of the latitudes of ERA to use.
# lon_era_light = list(lon_era_light_compl.keys())## returns the required indices of the longitudes of ERA to use.
#
# IO_var_ = IO_var[0:,0:,lat_era_light[0]:lat_era_light[-1]+1,lon_era_light[0]:lon_era_light[-1]+1] ## further reduce the spatial dimensions of the era domain's meteorological features.
#
#
# ## reshape strike grid data to 2 dimensions
# Strike_grid_regr=Strike_grid_regr.reshape(Strike_grid_regr.shape[0],Strike_grid_regr.shape[1]*Strike_grid_regr.shape[2])
# Strike_grid_categ = Strike_grid_categ.reshape(Strike_grid_categ.shape[0],Strike_grid_categ.shape[1]*Strike_grid_categ.shape[2])
#
# ## find lightning grid points with the most lightning context
# Strike_overall_spatial=np.sum(Strike_central_categ,axis=0)
#
# ## in the following , lgp is the symbol for lightning grid point - pca shall run for each lightning grid point contained around the ERA central grid point.
# inp_features=features_classifier_copy ## names of input variables.
# inp_features.append('Lightning density') ## output feature is lightning density
# feature_array=[i for i in inp_features] ## the length of the feature list is input_features+1, 1 goes for the output /target feature.
#
# input_data_help = IO_var_[0:17000, :, :, :]  ## input data at selected subdomain of ERA.
# input_data_help = input_data_help.reshape(input_data_help.shape[0],input_data_help.shape[1],input_data_help.shape[2]*input_data_help.shape[3])
# Strike_grid_regr=Strike_grid_regr[0:17000,:]
# Strike_grid_categ = Strike_grid_categ[0:17000,:]
# directory_to_save_figures = "C:/Users/plgeo/OneDrive/PC Desktop/MATLAB DRIVE/MSc_Meteorology/Figures/"
#
# def get_features(n,import_features):
#     import_features1=import_features[0:n]
#     if 'latitude' in import_features1:
#         import_features1.remove('latitude')
#     if 'longitude' in import_features1:
#         import_features1.remove('longitude')
#     return import_features1
#
# ## keep first n features, start from all features and go down to 4. check and record spatial correlations in all cases.
# ## keep the most correlated ERA point for each lightning grid point wrt number of features used( keep the feature importance order of the GBM central area) for usage in training regressors for various parts of the area.
# directory_to_save = "C:/Users/plgeo/OneDrive/PC Desktop/MATLAB DRIVE/MSc_Meteorology/"
# filetosave = directory_to_save + 'Spatial_correlation_ERA_lightning_density.xlsx'
# filetosave1= directory_to_save +'Spatial_correlation_per_no_of_features_ERA_Lightning_grid.xlsx'
# filetowrite=pd.ExcelWriter(filetosave, engine='xlsxwriter')  ##excel file to save processed data
# filetowrite1=pd.ExcelWriter(filetosave1, engine='xlsxwriter')  ##excel file to save processed data
# n=len(important_features)
# while n>3:
#     n_keep = n  ## features to keep for feature importances application for regression.
#     important_features1 = get_features(n_keep,important_features)  ## drop some known collinear features, before proceeding to spatial correlation analysis.
#     # spatial correlation analysis
#     input_regression = np.zeros((IO_var_.shape[0], len(important_features1), len(Latitude_mod), len(Longitude_mod)))
#     for v in range(len(important_features1)):
#         var1 = important_features1[v]
#         input_var1 = Dict_var[var1]
#         input_regression[:, v, :, :] = input_var1
#
#     input_regression = input_regression[0:17000, :, lat_era_light[0]:lat_era_light[-1] + 1,lon_era_light[0]:lon_era_light[-1] + 1]
#     input_regression = input_regression.reshape(input_regression.shape[0], input_regression.shape[1],input_regression.shape[2] * input_regression.shape[3])
#
#     Coefs_corr = np.zeros((12, 20))
#     for kk in range(input_regression.shape[2]):
#         for jj in range(Strike_grid_regr.shape[1]):
#             coef = 0
#             for fe in range(len(important_features1)):
#                 input_cor = input_regression[0:round(input_regression.shape[0] * 0.75), fe, kk]
#                 output_cor = Strike_grid_regr[0:round(input_regression.shape[0] * 0.75), jj]
#                 indexnz_help = [list_extension(st) for st in range(len(output_cor)) if output_cor[st] > 0]  ## keeps the indices of the non-zero lightning timestamps and the +-12 hours around them and creates a nested list of timestampd indices around any non-zero element.
#                 indexnz_help_help = list(chain(*indexnz_help))  ## un-nests  nested list of timestamp indices from above and creates a timestamp ordered un-nested list of indices from above.
#                 indexnz = list(dict.fromkeys(indexnz_help_help))  ## removes all dublicate values in ascending element order. if not new list sorted, then sort afterwards.
#                 coef_spear, p_spear = spearmanr(input_cor[indexnz], output_cor[indexnz], axis=0, nan_policy='omit',alternative='two-sided')
#                 coef += coef_spear
#                 coef_mean = round(coef / len(important_features1), 3)
#             Coefs_corr[kk, jj] = coef_mean
#     coefs_corr = pd.DataFrame(Coefs_corr, index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
#                               columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,17,18,19])
#     sheet_name=str(len(important_features1))+' '+'features'
#     Coefs_corr_trans=np.transpose(Coefs_corr)
#     Coefs_list=Coefs_corr_trans.tolist()
#     ERA_char={l:index_max for l in range(len(Coefs_list)) for index_max,val_max in enumerate(Coefs_list[l]) if val_max==np.nanmax(Coefs_list[l])}
#     ERA_df_char=pd.DataFrame.from_dict(ERA_char,orient='index')
#     coefs_corr.to_excel(filetowrite, sheet_name=sheet_name, engine="xlsxwriter", startrow=1, startcol=0, header=True)
#     ERA_df_char.to_excel(filetowrite1,sheet_name=sheet_name,engine="xlsxwriter", startrow=1, startcol=0, header=True)
#     n-=1
# filetowrite.save()
# filetowrite1.save()
#
