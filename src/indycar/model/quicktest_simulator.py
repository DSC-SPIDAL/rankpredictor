#!/usr/bin/env python
# coding: utf-8

# ### stint simulator
# based on: stint_simulator_shortterm_pitmodel.py
# 
# 
# long term predictor by continuously regressive forecasting at each pitstop
# 
# 
# support:
# + train/test split by ratio or event
# + incremental training evaluation(adjust ratio)
# + go beyond curtrack and zerotrack by modeling the track status
# + halfwin mode(0:no, 1:halfwin, 2:continous)
# + split by stage, support all events (todo)
# 
# + disturbance analysis by adding disturbance to oracle trackstatus and lapstatus
# 
# + rank prediction directly
# + rank prediction by laptime2rank,timediff2rank
# + laptime,lapstatus prediction
# 

# In[1]:
import ipdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import logging
from optparse import OptionParser


import mxnet as mx
from mxnet import gluon
import pickle
import json
import random, math
import inspect
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from gluonts.dataset.common import ListDataset
from gluonts.dataset.util import to_pandas
from pathlib import Path
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deep_factor import DeepFactorEstimator
from gluonts.model.deepstate import DeepStateEstimator
from gluonts.trainer import Trainer
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator, MultivariateEvaluator
from gluonts.distribution.multivariate_gaussian import MultivariateGaussianOutput
from gluonts.model.predictor import Predictor
from gluonts.model.prophet import ProphetPredictor
from gluonts.model.r_forecast import RForecastPredictor
from indycar.model.NaivePredictor import NaivePredictor
from indycar.model.ZeroPredictor import ZeroPredictor
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from indycar.model.pitmodel import PitModelSimple, PitModelMLP
from indycar.model.deeparw import DeepARWeightEstimator
# In[2]:


import os
random.seed()
os.getcwd()
#GPUID = 1


# ### global constants

# In[3]:


#
# remove NaN at the tail
# there should be no nans in the middle of the ts
COL_LAPTIME=0
COL_RANK=1
COL_TRACKSTATUS=2
COL_LAPSTATUS=3
COL_TIMEDIFF=4
COL_CAUTION_LAPS_INSTINT=5
COL_LAPS_INSTINT= 6
COL_ELAPSEDTIME= 7
COL_LAP2NEXTPIT= 8

# added new features
COL_LEADER_PITCNT = 9
COL_TOTAL_PITCNT = 10
COL_SHIFT_TRACKSTATUS = 11
COL_SHIFT_LAPSTATUS = 12
COL_SHIFT_LEADER_PITCNT = 13
COL_SHIFT_TOTAL_PITCNT = 14

COL_LASTFEATURE = 14
# dynamically extended space in simulation
COL_TRACKSTATUS_SAVE = COL_LASTFEATURE+1
COL_LAPSTATUS_SAVE = COL_LASTFEATURE+2
COL_CAUTION_LAPS_INSTINT_SAVE = COL_LASTFEATURE+3
COL_LAPS_INSTINT_SAVE= COL_LASTFEATURE+4

COL_ENDPOS = COL_LASTFEATURE+5


FEATURE_STATUS = 2
FEATURE_PITAGE = 4
FEATURE_LEADER_PITCNT = 8
FEATURE_TOTAL_PITCNT = 16
FEATURE_SHIFT_TRACKSTATUS = 32
FEATURE_SHIFT_LAPSTATUS = 64
FEATURE_SHIFT_LEADER_PITCNT = 128
FEATURE_SHIFT_TOTAL_PITCNT  = 256

_feature2str= {
    FEATURE_STATUS : ("FEATURE_STATUS",'S'),
    FEATURE_PITAGE : ("FEATURE_PITAGE",'A'),
    FEATURE_LEADER_PITCNT : ("FEATURE_LEADER_PITCNT",'L'),
    FEATURE_TOTAL_PITCNT : ("FEATURE_TOTAL_PITCNT",'T'),
    FEATURE_SHIFT_TRACKSTATUS : ("FEATURE_SHIFT_TRACKSTATUS",'Y'),
    FEATURE_SHIFT_LAPSTATUS : ("FEATURE_SHIFT_LAPSTATUS",'P'),
    FEATURE_SHIFT_LEADER_PITCNT : ("FEATURE_SHIFT_LEADER_PITCNT",'L'),
    FEATURE_SHIFT_TOTAL_PITCNT  : ("FEATURE_SHIFT_TOTAL_PITCNT",'T')
    }



# oracle mode
MODE_ORACLE = 1024  # oracle = track + lap
MODE_ORACLE_TRACKONLY = 1
MODE_ORACLE_LAPONLY = 2   
   

# oracle mode for training
MODE_NOLAP = 1   
MODE_NOTRACK = 2

# predicting mode
MODE_TESTZERO = 4
MODE_TESTCURTRACK = 8

MODE_PREDTRACK = 16
MODE_PREDPIT = 32

# disturbe analysis
MODE_DISTURB_CLEARTRACK = 64
MODE_DISTURB_ADJUSTTRACK = 128
MODE_DISTURB_ADJUSTPIT = 256


_mode_map = {MODE_ORACLE:'MODE_ORACLE',MODE_ORACLE_TRACKONLY:'MODE_ORACLE_TRACKONLY',
            MODE_ORACLE_LAPONLY:'MODE_ORACLE_LAPONLY',
             MODE_TESTZERO:'MODE_TESTZERO',MODE_TESTCURTRACK:'MODE_TESTCURTRACK',
             MODE_PREDTRACK:'MODE_PREDTRACK',MODE_PREDPIT:'MODE_PREDPIT',
            MODE_DISTURB_CLEARTRACK:'MODE_DISTURB_CLEARTRACK',MODE_DISTURB_ADJUSTTRACK:'MODE_DISTURB_ADJUSTTRACK',
            MODE_DISTURB_ADJUSTPIT:'MODE_DISTURB_ADJUSTPIT'}
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def test_flag(a, bitflag):
    return (a & bitflag) ==  bitflag

#_feature_mode = FEATURE_STATUS
def decode_feature_mode(feature_mode):
    
    retstr = []
    short_ret = []
    for feature in _feature2str.keys():
        if test_flag(feature_mode, feature):
            retstr.append(_feature2str[feature][0])
            short_ret.append(_feature2str[feature][1])
        else:
            short_ret.append('0')

    print(' '.join(retstr))
    
    return ''.join(short_ret)


def add_leader_cnt(selmat, rank_col=COL_RANK, pit_col=COL_LAPSTATUS, shift_len = 0, 
                   dest_col = COL_LEADER_PITCNT,
                   verbose = False):
    """
    add a new feature into mat(car, feature, lap)
    
    shift rank status
    
    input:
        sel_mat : laptime_data array [car, feature, lap]
    
    """
    dim1, dim2, dim3 = selmat.shape
    
    # rerank by the rank_col
    idx = np.argsort(selmat[:, rank_col,:], axis=0)
    true_rank = np.argsort(idx, axis=0).astype(np.float)

    # get leaderCnt by sorted pits
    pits = np.zeros((dim1,dim3))
    
    for lap in range(shift_len, dim3):
        col = idx[:, lap-shift_len]
        pits[:, lap] = selmat[col, pit_col, lap]
    
    leaderCnt = np.nancumsum(pits, axis=0) - pits
    
    if verbose:
        print('pits:\n')
        print(pits[:,190:])
        print('leaderCnt raw:\n')
        print(leaderCnt[:,190:])
    
    #remove nans
    nanidx = np.isnan(leaderCnt)
    leaderCnt[nanidx] = 0
    
    if verbose:
        print('leaderCnt after remove nan:\n')
        print(leaderCnt[:,190:])
    
    if dest_col == -1:
        #create a new data
        newmat = np.zeros((dim1,dim2+1,dim3))
        dest_col = dim2
        newmat[:,:dim2,:] = selmat.copy()
    else:
        #update mode
        newmat = selmat
    
    for lap in range(dim3):
        col = idx[:, lap]
        newmat[col, dest_col, lap] = leaderCnt[:, lap]
        
    # sync length to COL_RANK
    for rec in newmat:
        nans, x= nan_helper(rec[rank_col,:])
        nan_count = np.sum(nans)
        if nan_count > 0:
            #todo, some invalid nan, remove them
            #rec[dim2, np.isnan(rec[dim2,:])] = 0
            rec[dest_col, -nan_count:] = np.nan
    
    return newmat

def add_allpit_cnt(selmat, rank_col=COL_RANK, pit_col=COL_LAPSTATUS, 
                   dest_col = COL_TOTAL_PITCNT,verbose = False):
    """
    add a new feature into mat(car, feature, lap)
    
    total pits in a lap
    
    input:
        sel_mat : laptime_data array [car, feature, lap]
    
    """
    dim1, dim2, dim3 = selmat.shape

    #calc totalCnt vector for 
    totalCnt = np.nansum(selmat[:, pit_col, :], axis=0).reshape((-1))
    
    if verbose:
        print('pits:\n')
        print(pits[:,190:])
        print('totalCnt raw:\n')
        print(totalCnt[190:])
    
    #remove nans
    nanidx = np.isnan(totalCnt)
    totalCnt[nanidx] = 0
    
    if verbose:
        print('totalCnt after remove nan:\n')
        print(totalCnt[190:])
    
    if dest_col == -1:
        #create a new data
        newmat = np.zeros((dim1,dim2+1,dim3))
        dest_col = dim2
        newmat[:,:dim2,:] = selmat.copy()
    else:
        #update mode
        newmat = selmat

    for car in range(dim1):
        newmat[car, dest_col, :] = totalCnt
        
    # sync length to COL_RANK
    for rec in newmat:
        nans, x= nan_helper(rec[rank_col,:])
        nan_count = np.sum(nans)
        if nan_count > 0:
            #todo, some invalid nan, remove them
            #rec[dim2, np.isnan(rec[dim2,:])] = 0
            rec[dest_col, -nan_count:] = np.nan
    
    return newmat

def add_shift_feature(selmat, rank_col=COL_RANK, shift_col=COL_LAPSTATUS, shift_len = 2, 
                      dest_col = -1,verbose = False):
    """
    add a new feature into mat(car, feature, lap)
    
    shift features left in a lap
    
    warning: these are oracle features, be careful not to let future rank positions leaking
    
    input:
        sel_mat : laptime_data array [car, feature, lap]
    
    """
    dim1, dim2, dim3 = selmat.shape

    if dest_col == -1:
        #create a new data
        newmat = np.zeros((dim1,dim2+1,dim3))
        dest_col = dim2
        newmat[:,:dim2,:] = selmat.copy()
    else:
        #update mode
        newmat = selmat
    
    for car in range(dim1):
        # set empty status by default
        newmat[car, dest_col, :] = np.nan
        
        # get valid laps
        rec = selmat[car]
        nans, x= nan_helper(rec[rank_col,:])
        nan_count = np.sum(nans)
        recnnz = rec[shift_col, ~np.isnan(rec[rank_col,:])]
        reclen = len(recnnz)

        #shift copy
        newmat[car, dest_col, :reclen] = 0
        #newmat[car, dim2, :-shift_len] = selmat[car, shift_col, shift_len:]
        newmat[car, dest_col, :reclen-shift_len] = recnnz[shift_len:]
        
    # sync length to COL_RANK
    #for rec in newmat:
    #    nans, x= nan_helper(rec[rank_col,:])
    #    nan_count = np.sum(nans)
    #    if nan_count > 0:
    #        #todo, some invalid nan, remove them
    #        #rec[dim2, np.isnan(rec[dim2,:])] = 0
    #        rec[dim2, -nan_count:] = np.nan
    
    return newmat


def update_laptimedata(prediction_length, freq, 
                       test_event = 'Indy500-2018',
                       train_ratio=0.8,
                       context_ratio = 0.,
                       shift_len = -1, verbose = False):
    """
    update the features in laptime data
    
    3. create new features
    
    input: 
        laptime_data   ; global var
    output:
        data  ; new representation of laptime_data
    """
    global laptime_data

    #inplace update
    #_laptime_data = laptime_data.copy()
    _laptime_data = laptime_data

    #get test event
    test_idx = -1
    for idx, _data in enumerate(laptime_data):
        if events[_data[0]] == _test_event:
            test_idx = idx
            break
    
    # check shift len
    if shift_len < 0:
        shift_len = prediction_length
    if verbose:
        print('update_laptimedata shift len:', shift_len, test_idx)
    
    #_data: eventid, carids, datalist[carnumbers, features, lapnumber]->[laptime, rank, track, lap]]
    new_data = []
    if test_idx >= 0:
        _data = laptime_data[test_idx]
        # use to check the dimension of features
        input_feature_cnt = _data[2].shape[1]

        if verbose:
            if input_feature_cnt < COL_LASTFEATURE + 1:
                print('create new features mode, feature_cnt:', input_feature_cnt)
            else:
                print('update features mode, feature_cnt:', input_feature_cnt)
        
        # add new features
        # add leaderPitCnt
        #if _data[0]==0:
        #    verbose = True
        #else:
        #    verbose = False
        verbose = False
            

        dest_col = -1 if input_feature_cnt < COL_LASTFEATURE + 1 else COL_LEADER_PITCNT
        data2_intermediate = add_leader_cnt(_data[2], shift_len = shift_len, dest_col=dest_col, verbose = verbose)
        
        # add totalPit
        dest_col = -1 if input_feature_cnt < COL_LASTFEATURE + 1 else COL_TOTAL_PITCNT
        data2_intermediate = add_allpit_cnt(data2_intermediate, dest_col=dest_col)
        
        #
        # add shift features, a fixed order, see the MACROS 
        #COL_SHIFT_TRACKSTATUS = 11
        #COL_SHIFT_LAPSTATUS = 12
        #COL_SHIFT_LEADER_PITCNT = 13
        #COL_SHIFT_TOTAL_PITCNT = 14
        #
        dest_col = -1 if input_feature_cnt < COL_LASTFEATURE + 1 else COL_SHIFT_TRACKSTATUS
        data2_intermediate = add_shift_feature(data2_intermediate, dest_col=dest_col,
                                               shift_col=COL_TRACKSTATUS, shift_len = shift_len)
        
        dest_col = -1 if input_feature_cnt < COL_LASTFEATURE + 1 else COL_SHIFT_LAPSTATUS
        data2_intermediate = add_shift_feature(data2_intermediate, dest_col=dest_col,
                                               shift_col=COL_LAPSTATUS, shift_len = shift_len)
        
        dest_col = -1 if input_feature_cnt < COL_LASTFEATURE + 1 else COL_SHIFT_LEADER_PITCNT
        data2_intermediate = add_shift_feature(data2_intermediate, dest_col=dest_col,
                                               shift_col=COL_LEADER_PITCNT, shift_len = shift_len)
        
        dest_col = -1 if input_feature_cnt < COL_LASTFEATURE + 1 else COL_SHIFT_TOTAL_PITCNT
        data2_intermediate = add_shift_feature(data2_intermediate, dest_col=dest_col,
                                                   shift_col=COL_TOTAL_PITCNT, shift_len = shift_len)
            
        # final
        data2_newfeature = data2_intermediate
            
        #new_data.append([_data[0], _data[1], data2_newfeature])
        laptime_data[test_idx][2] = data2_newfeature
        
    return laptime_data


def get_real_features(feature_mode, rec, endpos):
    """
    construct the real value feature vector from feature_mode

    legacy code:
        real_features = {
            FEATURE_STATUS:[rec[COL_TRACKSTATUS,:],rec[COL_LAPSTATUS,:]],
            FEATURE_PITAGE:[rec[COL_TRACKSTATUS,:],rec[COL_LAPSTATUS,:],rec[COL_LAPS_INSTINT,:]],
            FEATURE_LEADERPITCNT:[rec[COL_TRACKSTATUS,:],rec[COL_LAPSTATUS,:],rec[COL_LEADER_PITCNT,:]],
            FEATURE_TOTALPITCNT:[rec[COL_TRACKSTATUS,:],rec[COL_LAPSTATUS,:],rec[COL_TOTAL_PITCNT,:]]
        }    
    
        real_features[feature_mode]
        
        
        COL_LEADER_PITCNT = 9
        COL_TOTAL_PITCNT = 10
        COL_SHIFT_TRACKSTATUS = 11
        COL_SHIFT_LAPSTATUS = 12
        COL_SHIFT_LEADER_PITCNT = 13
        COL_SHIFT_TOTAL_PITCNT = 14


        FEATURE_STATUS = 2
        FEATURE_PITAGE = 4
        FEATURE_LEADER_PITCNT = 8
        FEATURE_TOTAL_PITCNT = 16
        FEATURE_SHIFT_TRACKSTATUS = 32
        FEATURE_SHIFT_LAPSTATUS = 64
        FEATURE_SHIFT_LEADER_PITCNT = 128
        FEATURE_SHIFT_TOTAL_PITCNT  = 256        
    
    """
    
    features = []
    
    #check endpos
    if endpos <=0 :
        endpos = rec.shape[1]
    
    if test_flag(feature_mode, FEATURE_STATUS):
        features.append(rec[COL_TRACKSTATUS,:endpos])
        features.append(rec[COL_LAPSTATUS,:endpos])
        
    if test_flag(feature_mode, FEATURE_PITAGE):
        features.append(rec[COL_LAPS_INSTINT,:endpos])
        
    if test_flag(feature_mode, FEATURE_LEADER_PITCNT):
        features.append(rec[COL_LEADER_PITCNT,:endpos])
        
    if test_flag(feature_mode, FEATURE_TOTAL_PITCNT):
        features.append(rec[COL_TOTAL_PITCNT,:endpos])    
        
    if test_flag(feature_mode, FEATURE_SHIFT_TRACKSTATUS):
        features.append(rec[COL_SHIFT_TRACKSTATUS,:endpos])    
        
    if test_flag(feature_mode, FEATURE_SHIFT_LAPSTATUS):
        features.append(rec[COL_SHIFT_LAPSTATUS,:endpos])    

    if test_flag(feature_mode, FEATURE_SHIFT_LEADER_PITCNT):
        features.append(rec[COL_SHIFT_LEADER_PITCNT,:endpos])    

    if test_flag(feature_mode, FEATURE_SHIFT_TOTAL_PITCNT):
        features.append(rec[COL_SHIFT_TOTAL_PITCNT,:endpos])    
        
        
    return features

def get_real_features(feature_mode, rec, endpos):
    """
    construct the real value feature vector from feature_mode

    legacy code:
        real_features = {
            FEATURE_STATUS:[rec[COL_TRACKSTATUS,:],rec[COL_LAPSTATUS,:]],
            FEATURE_PITAGE:[rec[COL_TRACKSTATUS,:],rec[COL_LAPSTATUS,:],rec[COL_LAPS_INSTINT,:]],
            FEATURE_LEADERPITCNT:[rec[COL_TRACKSTATUS,:],rec[COL_LAPSTATUS,:],rec[COL_LEADER_PITCNT,:]],
            FEATURE_TOTALPITCNT:[rec[COL_TRACKSTATUS,:],rec[COL_LAPSTATUS,:],rec[COL_TOTAL_PITCNT,:]]
        }    
    
        real_features[feature_mode]
        
        
        COL_LEADER_PITCNT = 9
        COL_TOTAL_PITCNT = 10
        COL_SHIFT_TRACKSTATUS = 11
        COL_SHIFT_LAPSTATUS = 12
        COL_SHIFT_LEADER_PITCNT = 13
        COL_SHIFT_TOTAL_PITCNT = 14


        FEATURE_STATUS = 2
        FEATURE_PITAGE = 4
        FEATURE_LEADER_PITCNT = 8
        FEATURE_TOTAL_PITCNT = 16
        FEATURE_SHIFT_TRACKSTATUS = 32
        FEATURE_SHIFT_LAPSTATUS = 64
        FEATURE_SHIFT_LEADER_PITCNT = 128
        FEATURE_SHIFT_TOTAL_PITCNT  = 256        
    
    """
    
    features = []
    
    #check endpos
    if endpos <=0 :
        endpos = rec.shape[1]
    
    if test_flag(feature_mode, FEATURE_STATUS):
        features.append(rec[COL_TRACKSTATUS,:endpos])
        features.append(rec[COL_LAPSTATUS,:endpos])
        
    if test_flag(feature_mode, FEATURE_PITAGE):
        features.append(rec[COL_LAPS_INSTINT,:endpos])
        
    if test_flag(feature_mode, FEATURE_LEADER_PITCNT):
        features.append(rec[COL_LEADER_PITCNT,:endpos])
        
    if test_flag(feature_mode, FEATURE_TOTAL_PITCNT):
        features.append(rec[COL_TOTAL_PITCNT,:endpos])    
        
    if test_flag(feature_mode, FEATURE_SHIFT_TRACKSTATUS):
        features.append(rec[COL_SHIFT_TRACKSTATUS,:endpos])    
        
    if test_flag(feature_mode, FEATURE_SHIFT_LAPSTATUS):
        features.append(rec[COL_SHIFT_LAPSTATUS,:endpos])    

    if test_flag(feature_mode, FEATURE_SHIFT_LEADER_PITCNT):
        features.append(rec[COL_SHIFT_LEADER_PITCNT,:endpos])    

    if test_flag(feature_mode, FEATURE_SHIFT_TOTAL_PITCNT):
        features.append(rec[COL_SHIFT_TOTAL_PITCNT,:endpos])    
        
        
    return features
#
# interface with QuickTest
#
def set_laptimedata(newdata):
    global laptime_data

    #get test event
    test_idx = -1
    for idx, _data in enumerate(laptime_data):
        if events[_data[0]] == _test_event:
            test_idx = idx
            break


    print('Set a new global laptime_data, shape=', len(newdata), newdata[test_idx][2].shape)
    laptime_data = newdata

#
#
#
def load_data(event, year=0):
    #inputfile = '../data/final/C_'+ event +'-' + year + '-final.csv'
    if year>0:
        inputfile = '../data/final/C_'+ event +'-' + year + '.csv'
    else:
        inputfile = '../data/final/C_'+ event +'.csv'
    #outputprefix = year +'-' + event + '-'
    dataset = pd.read_csv(inputfile)
    #dataset.info(verbose=True)    
    
    final_lap = max(dataset.completed_laps)
    total_laps = final_lap + 1

    # get records for the cars that finish the race
    completed_car_numbers= dataset[dataset.completed_laps == final_lap].car_number.values
    completed_car_count = len(completed_car_numbers)

    #print('count of completed cars:', completed_car_count)
    #print('completed cars:', completed_car_numbers)

    #make a copy
    alldata = dataset.copy()
    dataset = dataset[dataset['car_number'].isin(completed_car_numbers)]
    rankdata = alldata.rename_axis('MyIdx').sort_values(by=['elapsed_time','MyIdx'], ascending=True)
    rankdata = rankdata.drop_duplicates(subset=['car_number', 'completed_laps'], keep='first')
    
    cldata = make_cl_data(dataset)
    acldata = make_cl_data(alldata)

    return alldata, rankdata, acldata

# make indy car completed_laps dataset
# car_number, completed_laps, rank, elapsed_time, rank_diff, elapsed_time_diff 
def make_cl_data(dataset):

    # pick up data with valid rank
    rankdata = dataset.rename_axis('MyIdx').sort_values(by=['elapsed_time','MyIdx'], ascending=True)
    rankdata = rankdata.drop_duplicates(subset=['car_number', 'completed_laps'], keep='first')

    # resort by car_number, lap
    uni_ds = rankdata.sort_values(by=['car_number', 'completed_laps', 'elapsed_time'], ascending=True)    
    #uni_ds = uni_ds.drop(["unique_id", "best_lap", "current_status", "track_status", "lap_status",
    #                  "laps_behind_leade","laps_behind_prec","overall_rank","pit_stop_count",
    #                  "last_pitted_lap","start_position","laps_led"], axis=1)
    
    uni_ds = uni_ds.drop(["unique_id", "best_lap", 
                      "laps_behind_leade","laps_behind_prec","overall_rank","pit_stop_count",
                      "last_pitted_lap","start_position","laps_led"], axis=1)
        
    carnumber = set(uni_ds['car_number'])
    #print('cars:', carnumber)
    #print('#cars=', len(carnumber))
   
    # faster solution , uni_ds already sorted by car_number and lap
    uni_ds['rank_diff'] = uni_ds['rank'].diff()
    mask = uni_ds.car_number != uni_ds.car_number.shift(1)
    uni_ds['rank_diff'][mask] = 0
    
    uni_ds['time_diff'] = uni_ds['elapsed_time'].diff()
    mask = uni_ds.car_number != uni_ds.car_number.shift(1)
    uni_ds['time_diff'][mask] = 0
    
    #df = uni_ds[['car_number','completed_laps','rank','elapsed_time','rank_diff','time_diff']]
    df = uni_ds[['car_number','completed_laps','rank','elapsed_time',
                 'rank_diff','time_diff',"current_status", "track_status", "lap_status"]]
    
    return df


# In[5]:


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def test_flag(a, bitflag):
    return (a & bitflag) ==  bitflag

# pit model is separate for each car
def load_model(prediction_length, model_name,trainid,epochs=1000, exproot='../models/remote'):
    with mx.Context(mx.gpu(7)):    
        pred_ret = []

        #rootdir = f'../models/{exproot}/{_dataset_id}/{_task_id}-{trainid}/'
        rootdir = f'{exproot}/{_dataset_id}/{_task_id}-{trainid}/'

        # deepAR-Oracle
        if model_name == 'curtrack':
            model=f'deepAR-Oracle-{_task_id}-curtrack-indy-f1min-t{prediction_length}-e1000-r1_curtrack_t{prediction_length}'
            modeldir = rootdir + model
            print(f'predicting model={model_name}, plen={prediction_length}')
            predictor =  Predictor.deserialize(Path(modeldir))
            print(f'loading model...done!, ctx:{predictor.ctx}')

        elif model_name == 'zerotrack':
            model=f'deepAR-Oracle-{_task_id}-nolap-zerotrack-indy-f1min-t{prediction_length}-e1000-r1_zerotrack_t{prediction_length}'
            modeldir = rootdir + model
            print(f'predicting model={model_name}, plen={prediction_length}')
            predictor =  Predictor.deserialize(Path(modeldir))
            print(f'loading model...done!, ctx:{predictor.ctx}')
        #deeparw-oracle
        elif model_name == 'weighted-oracle':
            model=f'deepARW-Oracle-{_task_id}-all-indy-f1min-t{prediction_length}-e1000-r1_oracle_t{prediction_length}'
            modeldir = rootdir + model
            print(f'predicting model={model_name}, plen={prediction_length}')
            predictor =  Predictor.deserialize(Path(modeldir))
            print(f'loading model...done!, ctx:{predictor.ctx}')
            
        # deepAR-Oracle
        elif model_name == 'oracle' or model_name == 'pitmodel':
            #
            # debug for weighted model
            #

            #model=f'deepARW-Oracle-{_task_id}-all-indy-f1min-t{prediction_length}-e1000-r1_oracle_t{prediction_length}'
            model=f'deepARW-Oracle-{_task_id}-all-indy-f1min-t{prediction_length}-e{epochs}-r1_oracle_t{prediction_length}'
            modeldir = rootdir + model
            print(f'predicting model={model_name}, plen={prediction_length}')
            predictor =  Predictor.deserialize(Path(modeldir))
            print(f'loading model...{model}...done!, ctx:{predictor.ctx}')



        # deepAR-Oracle
        elif model_name == 'oracle-laponly':
            model=f'deepAR-Oracle-{_task_id}-all-indy-f1min-t{prediction_length}-e1000-r1_oracle-laponly_t{prediction_length}'
            modeldir = rootdir + model
            print(f'predicting model={model_name}, plen={prediction_length}')
            predictor =  Predictor.deserialize(Path(modeldir))
            print(f'loading model...done!, ctx:{predictor.ctx}')
        # deepAR-Oracle
        elif model_name == 'oracle-trackonly':
            model=f'deepAR-Oracle-{_task_id}-all-indy-f1min-t{prediction_length}-e1000-r1_oracle-trackonly_t{prediction_length}'
            modeldir = rootdir + model
            print(f'predicting model={model_name}, plen={prediction_length}')
            predictor =  Predictor.deserialize(Path(modeldir))
            print(f'loading model...done!, ctx:{predictor.ctx}')
        # deepAR
        elif model_name == 'deepAR':
            model=f'deepAR-{_task_id}-all-indy-f1min-t{prediction_length}-e1000-r1_deepar_t{prediction_length}'
            modeldir = rootdir + model
            print(f'predicting model={model_name}, plen={prediction_length}')
            predictor =  Predictor.deserialize(Path(modeldir))
            print(f'loading model...done!, ctx:{predictor.ctx}')

        # naive
        elif model_name == 'naive':
            print(f'predicting model={model_name}, plen={prediction_length}')
            predictor =  NaivePredictor(freq= freq, prediction_length = prediction_length)
        # zero, zero keeps the rank unchange
        elif model_name == 'zero':
            print(f'predicting model={model_name}, plen={prediction_length}')
            predictor =  ZeroPredictor(freq= freq, prediction_length = prediction_length)

        # arima
        elif model_name == 'arima':
            print(f'predicting model={model_name}, plen={prediction_length}')
            predictor =  RForecastPredictor(method_name='arima',freq= freq, 
                                            prediction_length = prediction_length,trunc_length=60)
        else:
            print(f'error: model {model_name} not support yet!')

        return predictor         


#
# simulator 
#

def get_pitlaps(verbose = True, prediction_length=2):
    """
    collect pitlaps info from COL_LAPSTATUS

    input:
        laptime_data    ;
        _test_event     ;
        events
        _train_len      ; minimum laps for a ts(otherwise, discard)
        global_car_ids  ; carno-> carid mapping

    return:
        pitlaps ; [] array of laps which are pitstop for some car

    """
    run_ts = _run_ts
    #all_pitlaps = []  # carno -> pitstops
    all_pitlaps = {}  # carno -> pitstops
    max_lap = 0

    for _data in laptime_data:
        if events[_data[0]] != _test_event:
            continue

        #statistics on the ts length
        ts_len = [ _entry.shape[1] for _entry in _data[2]]
        max_lap = int(np.max(ts_len))

        # process for each ts
        for rowid in range(_data[2].shape[0]):
            # rec[features, lapnumber] -> [laptime, rank, track_status, lap_status,timediff]]
            rec = _data[2][rowid].copy()
            #remove nan(only tails)
            nans, x= nan_helper(rec[run_ts,:])
            nan_count = np.sum(nans)             
            rec = rec[:, ~np.isnan(rec[run_ts,:])]

            # remove short ts
            totallen = rec.shape[1]
            if ( totallen < _train_len + prediction_length):
                if verbose:
                    print(f'a short ts: carid={_data[1][rowid]}，len={totallen}')
                continue                

            carno = _data[1][rowid]
            carid = global_carids[_data[1][rowid]]


            static_cat = [carid]    

            #print(f'carno:{carno}, totallen={totallen}')
            #first, get target a copy    
            # target can be COL_XXSTATUS
            lap_status = rec[COL_LAPSTATUS, :]

            pitstops = np.where(lap_status == 1)[0]

            # filter out inlaps (when _inlap_status > 0)
            if _inlap_status !=0:
                if _inlap_status == 1:
                    #remove inlaps in previous of pit stops
                    pitstops_tmp = [pitstops[x] for x in range(1, len(pitstops), 2)]
                    pitstops = pitstops_tmp

                elif _inlap_status == 2:
                    #remove inlaps in next lap of pit stops
                    pitstops_tmp = [pitstops[x] for x in range(0, len(pitstops), 2)]
                    pitstops = pitstops_tmp


            #all_pitlaps.append(list(pitstops))
            all_pitlaps[carno] = list(pitstops)

            # append the end lap
            if _include_endpit:
                all_pitlaps[carno].append(totallen-1)


    #retrurn
    allset = []
    for l in all_pitlaps.keys():
        allset.extend(all_pitlaps[l])

    ret_pitlaps = sorted(list(set(allset)))

    return ret_pitlaps, all_pitlaps, max_lap


def get_nextpit(pitlaps, startlap):
    """
    input:
        pitlaps ; array of pitstops for all the cars
        startlap    ; 
    return
        nextpit ; nextpit for all the cars, nan for non-pit

    """
    nextpit = []
    nextpit_map = {}
    nextpit_hit = []

    #find hits
    for carno in pitlaps.keys():
        rec = pitlaps[carno]
        #search for startlap
        for lap in rec:
            if lap ==startlap:
                nextpit_hit.append(carno)

    #normal search
    for carno in pitlaps.keys():
        rec = pitlaps[carno]
        #search for startlap
        found = False
        for lap in rec:
            if lap > startlap:
                nextpit.append(lap)
                nextpit_map[carno] = lap
                found = True
                break
        if not found:
            #nextpit.append(np.nan)
            nextpit.append(-1)

            #todo, set to the end
            #nextpit.append(199)

    #get maxpit from nextpit_hit
    maxpit = -1
    for carno in nextpit_hit:
        if carno in nextpit_map:
            maxpit = max(nextpit_map[carno], maxpit)

    #return
    #return nextpit_map, max(nextpit)
    return nextpit_map, maxpit

def sim_init():
    """
    extend laptime data space to COL_ENDPOS
    save the lapstatus in laptime_data

    """
    global laptime_data


    #get test event
    test_idx = -1
    for idx, _data in enumerate(laptime_data):
        if events[_data[0]] == _test_event:
            test_idx = idx
            break

    print('sim_init: input laptime_data, shape=', len(laptime_data), laptime_data[test_idx][2].shape, test_idx)
    #update this laptime record
    if test_idx >= 0:
        _data = laptime_data[test_idx][2]

        dim1, dim2, dim3 = _data.shape

        if dim2 < COL_ENDPOS:
            #create a new data
            newmat = np.zeros((dim1, COL_ENDPOS, dim3))
            newmat[:,:dim2,:] = _data.copy()
        else:
            newmat = _data

        #save pit model related features
        newmat[:,COL_TRACKSTATUS_SAVE,:] = newmat[:,COL_TRACKSTATUS, :]
        newmat[:,COL_LAPSTATUS_SAVE,:] = newmat[:,COL_LAPSTATUS, :]
        newmat[:,COL_CAUTION_LAPS_INSTINT_SAVE,:] = newmat[:,COL_CAUTION_LAPS_INSTINT, :]
        newmat[:,COL_LAPS_INSTINT_SAVE, :] = newmat[:,COL_LAPS_INSTINT, :]

        # reset 
        if dim2 < COL_ENDPOS:
            laptime_data[test_idx][2] = newmat

    print('sim_init: after laptime_data, shape=', len(laptime_data), laptime_data[test_idx][2].shape)


def update_lapstatus(startlap):
    """
    update the whole lapstatus data
    """
    run_ts = _run_ts
    for _data in laptime_data:
        if events[_data[0]] != _test_event:
            continue

        #statistics on the ts length
        ts_len = [ _entry.shape[1] for _entry in _data[2]]
        max_lap = int(np.max(ts_len))

        # process for each ts
        for rowid in range(_data[2].shape[0]):
            # rec[features, lapnumber] -> [laptime, rank, track_status, lap_status,timediff]]
            rec = _data[2][rowid]
            carno = _data[1][rowid]
            update_onets(rec, startlap, carno)



_pitmodel = None

def update_onets(rec, startlap, carno):
    """
    update lapstatus after startlap basedon tsrec by pit prediction model

    input:
        tsrec   ; a ts with multiple features COL_XXX

    return:
        tsrec    ; updated for COL_LAPSTATUS, COL_CAUTION_LAPS_INSTINT, COL_LAPS_INSTINT

    """
    # loop from startlap
    nans, x= nan_helper(rec[_run_ts,:])
    nan_count = np.sum(nans)             
    recx = rec[:, ~np.isnan(rec[_run_ts,:])]
    
    # remove short ts
    totallen = recx.shape[1]
    if startlap >= totallen:
        return
    #totallen = tsrec.shape[1]
    #ipdb.set_trace()

    #reset status :startlap + 1
    endpos = startlap + 1
    rec[COL_TRACKSTATUS,:] = 0
    rec[COL_LAPSTATUS,:] = 0
    rec[COL_TRACKSTATUS,:endpos] = rec[COL_TRACKSTATUS_SAVE, :endpos]
    rec[COL_LAPSTATUS,:endpos] = rec[COL_LAPSTATUS_SAVE, :endpos]
    rec[COL_CAUTION_LAPS_INSTINT,:endpos] = rec[COL_CAUTION_LAPS_INSTINT_SAVE, :endpos]
    rec[COL_LAPS_INSTINT, :endpos] = rec[COL_LAPS_INSTINT_SAVE, :endpos]

    debug_report('start update_onets', rec[COL_LAPSTATUS], startlap, carno)

    #loop on predict nextpit pos
    curpos = startlap
    while True:
        caution_laps_instint = int(rec[COL_CAUTION_LAPS_INSTINT, curpos])
        laps_instint = int(rec[COL_LAPS_INSTINT, curpos])

        pred_pit_laps = _pitmodel.predict(caution_laps_instint, laps_instint)

        nextpos = curpos + pred_pit_laps - laps_instint

        #debug
        #if carno == 12:
        #    print('pitmodel: startlap={}, laps_instint={}, cuation_laps={}, \
        #            nextpos={}'.format(curpos, laps_instint, caution_laps_instint, nextpos))


        if nextpos >= totallen:
            nextpos = totallen - 1
 
            rec[COL_CAUTION_LAPS_INSTINT, curpos+1: nextpos+1] = caution_laps_instint
            for _pos in range(curpos+1, nextpos+1):
                rec[COL_LAPS_INSTINT, _pos] = rec[COL_LAPS_INSTINT, _pos - 1] + 1

            break

        else:
            #if (pred_pit_laps > laps_instint) and (nextpos < totallen):
            # a valid pit
            rec[COL_LAPSTATUS, nextpos] = 1
            if _inlap_status != 0:
                #inlap is 'P'
                if _inlap_status == 1 :
                    #rec[COL_LAPSTATUS, nextpos-1] = _inlap_status
                    rec[COL_LAPSTATUS, nextpos-1] = 1
                else:
                    #todo: no boudary check
                    #rec[COL_LAPSTATUS, nextpos+1] = _inlap_status
                    rec[COL_LAPSTATUS, nextpos+1] = 1

            rec[COL_CAUTION_LAPS_INSTINT, curpos+1: nextpos] = caution_laps_instint
            rec[COL_CAUTION_LAPS_INSTINT, nextpos] = 0
            for _pos in range(curpos+1, nextpos):
                rec[COL_LAPS_INSTINT, _pos] = rec[COL_LAPS_INSTINT, _pos - 1] + 1
            rec[COL_LAPS_INSTINT, nextpos] = 0

        #go forward
        curpos = nextpos

    debug_report('after update_onets', rec[COL_LAPSTATUS], startlap, carno)

    return

def debug_pitmodel(startlap, carno, laps_instint, caution_laps_instint, samplecnt=1000):
    """
    test the pitmodel
    ret:
        list of predictions of nextpit
    """
    ret = []
    for runid in range(samplecnt):

        pred_pit_laps = _pitmodel.predict(caution_laps_instint, laps_instint)

        nextpos = startlap + pred_pit_laps - laps_instint
        ret.append(nextpos)

    return ret


#debug tracking status
#status matrix  :  laps x ( endCol x 5 features)
#features: target, lapstatus, lap_instint, caution_instint, trackstatus
_status_mat = {}  # stepid -> status matrix
def debug_report_mat(startlap, maxnext):
    """
    output the status of the simulation

    """
    fixedWidth = 5
    endCol = 4

    run_ts = _run_ts
    for _data in laptime_data:
        if events[_data[0]] != _test_event:
            continue

        #statistics on the ts length
        ts_len = [ _entry.shape[1] for _entry in _data[2]]
        max_lap = int(np.max(ts_len))

        #header  carno | lap#...
        #fixed width

        # process for each ts
        for rowid in range(_data[2].shape[0]):
            # rec[features, lapnumber] -> [laptime, rank, track_status, lap_status,timediff]]
            rec = _data[2][rowid]

_debug_carlist = []
#_debug_carlist = [12]
def debug_report_ts(msg, rec, startlap, carno, col= COL_LAPSTATUS):
    if carno not in _debug_carlist:
        return
    print(f'--------- {msg}: {startlap} ----------')
    print(rec[col, : startlap + 1])
    print('='*10)
    print(rec[col, startlap + 1:])
def debug_report(msg, rec, startlap, carno):
    if carno not in _debug_carlist:
        return
    print(f'--------- {msg}: {startlap} ----------')
    print(rec[: startlap + 1])
    print('='*10)
    print(rec[startlap + 1:])

def debug_print(msg):
    if len(_debug_carlist) > 0:
        print(msg)

# works on predicted lap status
def sim_onestep_pred(predictor, prediction_length, freq, 
                   startlap, endlap,
                   oracle_mode = MODE_ORACLE,
                   sample_cnt = 100,
                   verbose = False
                ):
    """
    input:
        parameters  ; same as longterm_predict, make_dataset_byevent
        startlap
        endlap

    return:
        forecast    ; {}, carno -> 5 x totallen matrix 
            0,: -> lapstatus
            1,: -> true target
            2,: -> pred target
            3,  -> placeholder
            4,  -> placeholder
        forecast_samples; save the samples, the farest samples
            {}, carno -> samplecnt of the target

    """    
    run_ts= _run_ts 
    test_event = _test_event    
    feature_mode = _feature_mode
    context_ratio = _context_ratio
    train_len = _train_len

    start = pd.Timestamp("01-01-2019", freq=freq)  # can be different for each time series

    test_set = []
    forecasts_et = {}
    forecasts_samples = {}

    _laptime_data = laptime_data.copy()

    endpos = startlap + prediction_length + 1
    #while(endpos <= endlap + prediction_length + 1):
    while(endpos <= endlap + prediction_length):
        #make the testset
        #_data: eventid, carids, datalist[carnumbers, features, lapnumber]->[laptime, rank, track, lap]]
        _test = []
        for _data in _laptime_data:

            if events[_data[0]] != test_event:
                #jump out
                continue

            #statistics on the ts length
            ts_len = [ _entry.shape[1] for _entry in _data[2]]
            max_len = int(np.max(ts_len))

            #ipdb.set_trace()
            if verbose:
                print(f'after ====event:{events[_data[0]]}, prediction_len={prediction_length},train_len={train_len}, max_len={np.max(ts_len)}, min_len={np.min(ts_len)}, cars={_data[2].shape[0]}')

            # process for each ts
            for rowid in range(_data[2].shape[0]):
                # rec[features, lapnumber] -> [laptime, rank, track_status, lap_status,timediff]]
                rec = _data[2][rowid].copy()
                rec_raw = _data[2][rowid].copy()
                
                #remove nan(only tails)
                nans, x= nan_helper(rec[run_ts,:])
                nan_count = np.sum(nans)             
                rec = rec[:, ~np.isnan(rec[run_ts,:])]
                
                # remove short ts
                totallen = rec.shape[1]
                if ( totallen < train_len + prediction_length):
                    if verbose:
                        print(f'a short ts: carid={_data[1][rowid]}，len={totallen}')
                    continue                
                
                if endpos > totallen:
                    continue

                carno = _data[1][rowid]
                carid = global_carids[_data[1][rowid]]
                
                static_cat = [carid]    
                 


                #first, get target a copy    
                # target can be COL_XXSTATUS
                #target_val = rec[run_ts,:].copy().astype(np.float32)

                lap_status = rec[COL_LAPSTATUS, :].copy()
                track_status = rec[COL_TRACKSTATUS, :].copy()
                pitage_status = rec[COL_LAPS_INSTINT,:].copy()
                
                

                # <3, totallen> 
                if carno not in forecasts_et:
                    forecasts_et[carno] = np.zeros((5, totallen))
                    forecasts_et[carno][:,:] = np.nan
                    forecasts_et[carno][0,:] = rec[COL_LAPSTATUS_SAVE, :].copy()
                    forecasts_et[carno][1,:] = rec[run_ts,:].copy().astype(np.float32)
                    forecasts_et[carno][2,:] = rec[run_ts,:].copy().astype(np.float32)
                    # for p-risk
                    forecasts_samples[carno] = np.zeros((sample_cnt))

                # forecasts_et will be updated by forecasts
                target_val = forecasts_et[carno][2,:]
                
                # selection of features
                if test_flag(oracle_mode, MODE_NOTRACK) or test_flag(oracle_mode, MODE_ORACLE_LAPONLY):                
                    rec[COL_TRACKSTATUS, :] = 0
                if test_flag(oracle_mode, MODE_NOLAP) or test_flag(oracle_mode, MODE_ORACLE_TRACKONLY):                
                    rec[COL_LAPSTATUS, :] = 0

                test_rec_cnt = 0

                # RUN Prediction for single record
                
                track_rec = rec[COL_TRACKSTATUS, :endpos].copy()
                lap_rec = rec[COL_LAPSTATUS, :endpos].copy()
                pitage_rec = rec[COL_LAPS_INSTINT, :endpos].copy()
                
                caution_laps_instint = int(rec[COL_CAUTION_LAPS_INSTINT, endpos -prediction_length - 1])
                laps_instint = int(rec[COL_LAPS_INSTINT, endpos -prediction_length - 1])
                

                # test mode
                if test_flag(oracle_mode, MODE_TESTCURTRACK):
                    # since nan does not work, use cur-val instead
                    track_rec[-prediction_length:] = track_rec[-prediction_length - 1]
                    #track_rec[-prediction_length:] = random.randint(0,1)
                    #lap_rec[-prediction_length:] = lap_rec[-prediction_length - 1]
                    lap_rec[-prediction_length:] = 0
                    #for pitage, just assume there is no pit
                    start_pitage = pitage_rec[-prediction_length - 1]
                    pitage_rec[-prediction_length:] = np.array([x+start_pitage+1 for x in range(prediction_length)])
                    
                elif test_flag(oracle_mode, MODE_TESTZERO):
                    #set prediction part as nan
                    #track_rec[-prediction_length:] = np.nan
                    #lap_rec[-prediction_length:] = np.nan
                    track_rec[-prediction_length:] = 0
                    lap_rec[-prediction_length:] = 0        
                    #for pitage, just assume there is no pit
                    start_pitage = pitage_rec[-prediction_length - 1]
                    pitage_rec[-prediction_length:] = np.array([x+start_pitage+1 for x in range(prediction_length)])

                if test_flag(oracle_mode, MODE_PREDPIT):

                    #todo

                    #lap_rec[-prediction_length:] = get_pit_model(caution_laps_instint,
                    #                                            laps_instint,prediction_length)

                    #for pitage, use the predicted lap info to update pitage
                    start_pitage = pitage_rec[-prediction_length - 1]
                    for pos in range(prediction_length):
                        if lap_rec[-prediction_length + pos]==0:
                            pitage_rec[-prediction_length + pos] = start_pitage+1
                        else:
                            #new pit
                            start_pitage = 0
                            pitage_rec[-prediction_length + pos] = start_pitage

                # add to test set

                #train real features
                real_features = get_real_features(feature_mode, rec, endpos)

                _test.append({'target': target_val[:endpos].astype(np.float32), 
                        'start': start, 
                        'feat_static_cat': static_cat,
                        'feat_dynamic_real': real_features
                         }
                      )   

                test_rec_cnt += 1

                #debug
                #debug_report('simu_onestep', rec, startlap, carno, col= _run_ts)
                #debug_report(f'simu_onestep: {startlap}-{endlap}, endpos={endpos}', target_val[:endpos], startlap, carno)

        # end of for each ts

        # RUN Prediction here
        test_ds = ListDataset(_test, freq=freq)

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_ds,  # test dataset
            predictor=predictor,  # predictor
            num_samples=sample_cnt,  # number of sample paths we want for evaluation
        )

        forecasts = list(forecast_it)
        tss = list(ts_it)

        #save the forecast results
        ds_iter =  iter(test_ds)
        for idx in range(len(test_ds)):
            test_rec = next(ds_iter)
            #global carid
            carno = decode_carids[test_rec['feat_static_cat'][0]]

            if _use_mean:
                forecast_laptime_mean = np.mean(forecasts[idx].samples, axis=0).reshape((prediction_length))
            else:
                forecast_laptime_mean = np.median(forecasts[idx].samples, axis=0).reshape((prediction_length))
            
            #update the forecasts , ready to use in the next prediction(regresive forecasting)
            forecasts_et[carno][2, len(tss[idx]) - prediction_length:len(tss[idx])] = forecast_laptime_mean.copy()

            #save the samples, the farest samples
            forecasts_samples[carno][:] = forecasts[idx].samples[:,-1].reshape(-1)
 

        #go forward
        endpos += prediction_length


    return forecasts_et, forecasts_samples


# pred pit differs to true pit
def get_acc_onestint_pred(forecasts, startlap, nextpit, nextpit_pred, trim=2, currank = False):
    """
    input:
        trim     ; steady lap of the rank (before pit_inlap, pit_outlap)
        forecasts;  carno -> [5,totallen]
                0; lap_status
                3; true_rank
                4; pred_rank
        startlap ; eval for the stint start from startlap only
        nextpit  ; array of next pitstop for all cars
    output:
        carno, stintid, startrank, endrank, diff, sign

    """
    rankret = []
    for carno in forecasts.keys():
        lapnum = len(forecasts[carno][1,:])
        true_rank = forecasts[carno][3,:]
        pred_rank = forecasts[carno][4,:]


        #lap status condition
        if _inlap_status == 0:
            lapstatus_cont = (forecasts[carno][0, startlap] == 1)
        elif _inlap_status == 1:
            lapstatus_cont = ((forecasts[carno][0, startlap] == 1) and (forecasts[carno][0, startlap-1] == 1))
        elif _inlap_status == 2:
            lapstatus_cont = ((forecasts[carno][0, startlap] == 1) and (forecasts[carno][0, startlap+1] == 1))


        if carno in _debug_carlist:
            _debug_msg = 'startlap=%d, total=%d, pitstop status = %s, nextpit=%s, nextpit_pred=%s'%(startlap, lapnum, lapstatus_cont, 
                     'none' if (carno not in nextpit) else nextpit[carno],
                     'none' if (carno not in nextpit_pred) else nextpit_pred[carno],
                     )
            debug_print(_debug_msg)


        # check the lap status
        #if ((startlap < lapnum) and (forecasts[carno][0, startlap] == 1)):
        if ((startlap < lapnum) and (lapstatus_cont == True)):

            startrank = true_rank[startlap-trim]
            

            if not carno in nextpit:
                continue

            pitpos = nextpit[carno]
            if np.isnan(pitpos):
                continue

            #todo, use the true prediction that longer than maxlap
            if _force_endpit_align:
                if not carno in nextpit_pred:
                    #continue
                    pitpos_pred = pitpos
                else:
                    pitpos_pred = nextpit_pred[carno]
                    if np.isnan(pitpos_pred):
                        pitpos_pred = pitpos
            else:

                if not carno in nextpit_pred:
                    continue
                pitpos_pred = nextpit_pred[carno]
                if np.isnan(pitpos_pred):
                    #set prediction to the end
                    continue

            endrank = true_rank[pitpos-trim]
            #endrank_pred = true_rank[pitpos_pred-trim]


            diff = endrank - startrank
            sign = get_sign(diff)



            if currank:
                #force into currank model, zero doesn't work here
                pred_endrank = startrank
                pred_diff = pred_endrank - startrank
                pred_sign = get_sign(pred_diff)

            else:
                pred_endrank = pred_rank[pitpos_pred-trim]
                pred_diff = pred_endrank - startrank
                pred_sign = get_sign(pred_diff)

            rankret.append([carno, startlap, startrank, 
                            endrank, diff, sign,
                            pred_endrank, pred_diff, pred_sign,
                            pitpos, pitpos_pred
                            ])

    return rankret

# pred pit differs to true pit
def get_acc_onestep_shortterm(forecasts, startlap, endlap, trim=0, currank = False):
    """
    input:
        trim     ; steady lap of the rank (before pit_inlap, pit_outlap)
        forecasts;  carno -> [5,totallen]
                0; lap_status
                3; true_rank
                4; pred_rank
        startlap ; eval for the stint start from startlap only
        nextpit  ; array of next pitstop for all cars
    output:
        carno, stintid, startrank, endrank, diff, sign

    """
    rankret = []
    for carno in forecasts.keys():
        lapnum = len(forecasts[carno][1,:])
        true_rank = forecasts[carno][3,:]
        pred_rank = forecasts[carno][4,:]

        # check the lap status
        #if ((startlap < lapnum) and (forecasts[carno][0, startlap] == 1)):
        if startlap < lapnum:

            startrank = true_rank[startlap-trim]
            if np.isnan(endlap):
                continue

            endrank = true_rank[endlap-trim]

            diff = endrank - startrank
            sign = get_sign(diff)

            if currank:
                #force into currank model, zero doesn't work here
                pred_endrank = startrank
                pred_diff = pred_endrank - startrank
                pred_sign = get_sign(pred_diff)

            else:
                pred_endrank = pred_rank[endlap-trim]
                pred_diff = pred_endrank - startrank
                pred_sign = get_sign(pred_diff)

            rankret.append([carno, startlap, startrank, 
                            endrank, diff, sign,
                            pred_endrank, pred_diff, pred_sign
                            ])

    return rankret



# works when pred pitstop == true pitstop
def get_acc_onestint(forecasts, startlap, nextpit, trim=2, currank = False):
    """
    input:
        trim     ; steady lap of the rank (before pit_inlap, pit_outlap)
        forecasts;  carno -> [5,totallen]
                0; lap_status
                3; true_rank
                4; pred_rank
        startlap ; eval for the stint start from startlap only
        nextpit  ; array of next pitstop for all cars
    output:
        carno, stintid, startrank, endrank, diff, sign

    """
    rankret = []
    for carno in forecasts.keys():
        lapnum = len(forecasts[carno][1,:])
        true_rank = forecasts[carno][3,:]
        pred_rank = forecasts[carno][4,:]

        # check the lap status
        if ((startlap < lapnum) and (forecasts[carno][0, startlap] == 1)):

            startrank = true_rank[startlap-trim]
            
            if not carno in nextpit:
                continue

            pitpos = nextpit[carno]
            if np.isnan(pitpos):
                continue

            endrank = true_rank[pitpos-trim]


            diff = endrank - startrank
            sign = get_sign(diff)

            if currank:
                #force into currank model, zero doesn't work here
                pred_endrank = startrank
                pred_diff = pred_endrank - startrank
                pred_sign = get_sign(pred_diff)

            else:
                pred_endrank = pred_rank[pitpos-trim]
                pred_diff = pred_endrank - startrank
                pred_sign = get_sign(pred_diff)

            rankret.append([carno, startlap, startrank, 
                            endrank, diff, sign,
                            pred_endrank, pred_diff, pred_sign
                            ])

    return rankret

#
# simulation
#
def run_simulation_stint(predictor, prediction_length, freq, 
                   carno, stintid, loopcnt,
                   datamode = MODE_ORACLE):
    """
    simulation for one car at specific stint
    input:
        carno   ; 
        stintid ;

    step:
        1. init the lap status model
        2. loop on each pit lap
            1. onestep simulation
            2. eval stint performance
    """

    rankret = []

    # the ground truth
    allpits, pitmat, maxlap = get_pitlaps()
    sim_init()

    #init samples array
    full_samples = {}
    full_tss = {}


    #here, test only one stint for carno and stintid
    pitlap = pitmat[carno][stintid]

    for runid in range(loopcnt):
    #for pitlap in allpits:
        #1. update lap status
        debug_print(f'start pitlap: {pitlap}')
        if not (isinstance(_pitmodel, str) and _pitmodel == 'oracle'):
            update_lapstatus(pitlap)

        debug_print(f'update lapstatus done.')
        #2. get maxnext
        allpits_pred, pitmat_pred, maxlap = get_pitlaps()
        nextpit, maxnext = get_nextpit(pitmat, pitlap)
        nextpit_pred, maxnext_pred = get_nextpit(pitmat_pred, pitlap)

        #only for one car
        maxnext = nextpit[carno]
        maxnext_pred = nextpit_pred[carno]

        #debug
        if len(_debug_carlist) > 0:
            _testcar = _debug_carlist[0]
            if _testcar in nextpit and _testcar in nextpit_pred:
                #print('nextpit:', nextpit[12], nextpit_pred[12], 'maxnext:', maxnext, maxnext_pred)
                #debugstr = f'nextpit: {nextpit[]}, {nextpit_pred[12]}, maxnext: {maxnext}, {maxnext_pred}'
                debugstr = 'nextpit: %d, %d, maxnext: %d, %d'%(nextpit[_testcar], nextpit_pred[_testcar]
                        , maxnext, maxnext_pred)

                debug_print(debugstr)

        #run one step sim from pitlap to maxnext
        #to get the forecast_sample, set max = mexnext_pred only, 
        #rather than max(maxnext,maxnext_pred)
        #
        forecast, forecast_samples = sim_onestep_pred(predictor, prediction_length, freq,
                pitlap, maxnext_pred,
                oracle_mode = datamode,
                sample_cnt = 100
                )

        debug_print(f'simulation done: {len(forecast)}')
        # calc rank from this result
        if _exp_id=='rank' or _exp_id=='timediff2rank':
            forecasts_et = eval_stint_direct(forecast, 2)
        elif _exp_id=='laptime2rank':
            forecasts_et = eval_stint_bylaptime(forecast, 2, global_start_offset[_test_event])

        else:
            print(f'Error, {_exp_id} evaluation not support yet')
            return

        ## evaluate for this stint
        ret = get_acc_onestint_pred(forecasts_et, pitlap, nextpit, nextpit_pred, trim=_trim)

        #add endlap
        #_ = [x.append(maxnext_pred) for x in ret]
        rankret.extend(ret)

        ## add to full_samples
        #eval_full_samples(maxnext_pred,
        #        forecast_samples, forecast, 
        #        full_samples, full_tss)

        

    #add to df
    df = pd.DataFrame(rankret, columns =['carno', 'startlap', 'startrank', 
                                         'endrank', 'diff', 'sign',
                                         'pred_endrank', 'pred_diff', 'pred_sign',
                                         'endlap','pred_endlap'
                                        ])

    return df, full_samples, full_tss, maxnext_pred


def run_simulation_pred(predictor, prediction_length, freq, 
                   datamode = MODE_ORACLE, verbose = False):
    """
    step:
        1. init the lap status model
        2. loop on each pit lap
            1. onestep simulation
            2. eval stint performance
    """

    rankret = []

    # the ground truth
    allpits, pitmat, maxlap = get_pitlaps()
    sim_init()

    for pitlap in allpits:
        #1. update lap status
        debug_print(f'start pitlap: {pitlap}')
        if not (isinstance(_pitmodel, str) and _pitmodel == 'oracle'):
            update_lapstatus(pitlap)

        debug_print(f'update lapstatus done.')
        #2. get maxnext
        allpits_pred, pitmat_pred, maxlap = get_pitlaps()
        nextpit, maxnext = get_nextpit(pitmat, pitlap)
        nextpit_pred, maxnext_pred = get_nextpit(pitmat_pred, pitlap)

        #debug
        if len(_debug_carlist) > 0:
            _testcar = _debug_carlist[0]
            if _testcar in nextpit and _testcar in nextpit_pred:
                #print('nextpit:', nextpit[12], nextpit_pred[12], 'maxnext:', maxnext, maxnext_pred)
                #debugstr = f'nextpit: {nextpit[]}, {nextpit_pred[12]}, maxnext: {maxnext}, {maxnext_pred}'
                debugstr = 'nextpit: %d, %d, maxnext: %d, %d'%(nextpit[_testcar], nextpit_pred[_testcar]
                        , maxnext, maxnext_pred)

                debug_print(debugstr)

        #run one step sim from pitlap to maxnext
        forecast, forecast_samples = sim_onestep_pred(predictor, prediction_length, freq,
                pitlap, max(maxnext, maxnext_pred),
                oracle_mode = datamode,
                sample_cnt = 100,
                verbose = verbose
                )

        debug_print(f'simulation done: {len(forecast)}')
        # calc rank from this result
        if _exp_id=='rank' or _exp_id=='timediff2rank':
            forecasts_et = eval_stint_direct(forecast, 2)
        elif _exp_id=='laptime2rank':
            forecasts_et = eval_stint_bylaptime(forecast, 2, global_start_offset[_test_event])

        else:
            print(f'Error, {_exp_id} evaluation not support yet')
            break

        # evaluate for this stint
        ret = get_acc_onestint_pred(forecasts_et, pitlap, nextpit, nextpit_pred, trim=_trim)
        rankret.extend(ret)


    #add to df
    df = pd.DataFrame(rankret, columns =['carno', 'startlap', 'startrank', 
                                         'endrank', 'diff', 'sign',
                                         'pred_endrank', 'pred_diff', 'pred_sign',
                                         'endlap','pred_endlap'
                                        ])

    return df

#prediction of shorterm + pred pit model
def run_simulation_shortterm(predictor, prediction_length, freq, 
                   datamode = MODE_ORACLE, 
                   sample_cnt = 100,
                   verbose = False
                   ):
    """
    step:
        1. init the lap status model
        2. loop on each pit lap
            1. onestep simulation
            2. eval stint performance
    """
    global laptime_data

    rankret = []

    # the ground truth
    allpits, pitmat, maxlap = get_pitlaps()
    sim_init()

    #init samples array
    full_samples = {}
    full_tss = {}

    for pitlap in range(10, maxlap-prediction_length):
        #1. update lap status
        debug_print(f'start pitlap: {pitlap}')
        if not (isinstance(_pitmodel, str) and _pitmodel == 'oracle'):
            update_lapstatus(pitlap)

            #update the featuers
            laptime_data = update_laptimedata(prediction_length, freq, 
                        test_event = _test_event,
                        train_ratio=0, context_ratio = 0.,shift_len = prediction_length)

        debug_print(f'update lapstatus done.')
        #run one step sim from pitlap to maxnext
        forecast, forecast_samples = sim_onestep_pred(predictor, prediction_length, freq,
                pitlap, pitlap + prediction_length,
                oracle_mode = datamode,
                sample_cnt = sample_cnt,
                verbose = verbose
                )

        debug_print(f'simulation done: {len(forecast)}')
        # calc rank from this result
        if _exp_id=='rank' or _exp_id=='timediff2rank':
            forecasts_et = eval_stint_direct(forecast, prediction_length)
        elif _exp_id=='laptime2rank':
            forecasts_et = eval_stint_bylaptime(forecast, prediction_length, global_start_offset[_test_event])

        else:
            print(f'Error, {_exp_id} evaluation not support yet')
            break

        # evaluate for this stint
        #ret = get_acc_onestint_pred(forecasts_et, pitlap, nextpit, nextpit_pred)
        ret = get_acc_onestep_shortterm(forecasts_et, pitlap, pitlap+prediction_length)

        rankret.extend(ret)

        # add to full_samples
        evalbyrank = False if _exp_id == 'laptime2rank' else True
        eval_full_samples(pitlap + prediction_length,
                forecast_samples, forecast, 
                full_samples, full_tss, evalbyrank=evalbyrank)

    print('evalbyrank:', evalbyrank)

    #add to df
    df = pd.DataFrame(rankret, columns =['carno', 'startlap', 'startrank', 
                                         'endrank', 'diff', 'sign',
                                         'pred_endrank', 'pred_diff', 'pred_sign',
                                        ])

    return df, full_samples, full_tss



def eval_stint_bylaptime(forecasts_et, prediction_length, start_offset):
    """
    evaluate stint rank by laptime forecasting
    input:
        forecast    ; {}, carno -> 5 x totallen matrix 
            0,: -> lapstatus
            1,: -> true target
            2,: -> pred target
            3,  -> placeholder
            4,  -> placeholder
        start_offset[]; elapsed time for lap0, for one specific event
        prediction_length ; 
    return:
        forecasts  
    """

    #get car list for this lap
    carlist = list(forecasts_et.keys())
    #print('carlist:', carlist)

    caridmap={key:idx for idx, key in enumerate(carlist)}    

    #convert to elapsedtime
    #todo, Indy500 - > 200 max laps
    maxlap = 200
    
    elapsed_time = np.zeros((2, len(carlist), maxlap))
    elapsed_time[:,:] = np.nan
    
    for carno in forecasts_et.keys():

        #start_offset is global var
        if isinstance(start_offset, pd.core.frame.DataFrame):
            offset = start_offset[(start_offset['car_number']==carno)].elapsed_time.values[0] 
        
        lapnum = len(forecasts_et[carno][1,:])
        
        laptime_array = forecasts_et[carno][1,:]
        elapsed = np.cumsum(laptime_array) + offset
        elapsed_time[0, caridmap[carno],:lapnum] = elapsed
        
        laptime_array = forecasts_et[carno][2,:]
        elapsed = np.cumsum(laptime_array) + offset
        elapsed_time[1, caridmap[carno],:lapnum] = elapsed
        
        #maxlap = max(maxlap, len(forecasts_et[carno][1,:]))

    #calculate rank, support nan
    idx = np.argsort(elapsed_time[0], axis=0)
    true_rank = np.argsort(idx, axis=0)

    idx = np.argsort(elapsed_time[1], axis=0)
    pred_rank = np.argsort(idx, axis=0)
        
    # save the rank back
    for carno in forecasts_et.keys():
        lapnum = len(forecasts_et[carno][1,:])
        
        forecasts_et[carno][3,:] = true_rank[caridmap[carno],:lapnum]
        forecasts_et[carno][4,:] = pred_rank[caridmap[carno],:lapnum]
    
    return forecasts_et

#
#
def eval_full_samples(lap, forecast_samples, forecast, full_samples, full_tss, maxlap=200, evalbyrank = True):
    """
    input:
        lap  ; lap number
        forecast_samples; {} cano -> samples ore pred target
        forecast    ; {}, carno -> 5 x totallen matrix 
            1,: -> true target
            2,: -> pred target
    return:
        full_samples
        full_tss
    """

    #get car list for this lap
    carlist = list(forecast.keys())
    #print('carlist:', carlist)

    caridmap={key:idx for idx, key in enumerate(carlist)}    

    #convert to elapsedtime
    #todo, Indy500 - > 200 max laps
    samplecnt = len(forecast_samples[carlist[0]])
    
    #diff_time = np.zeros((len(carlist), 1))
    diff_time = np.zeros((len(carlist), maxlap))
    diff_time_hat = np.zeros((len(carlist), samplecnt))
    diff_time[:,:] = np.nan
    diff_time_hat[:,:] = np.nan
    
    for carno in carlist:
        #diff_time[caridmap[carno],0] = forecast[carno][1, lap]
        maxlen = len(forecast[carno][1, :])

        diff_time[caridmap[carno],:maxlen] = forecast[carno][1, :]
        
        diff_time_hat[caridmap[carno],:] = forecast_samples[carno]
        
    
    if evalbyrank == True:
        #calculate rank, support nan
        idx = np.argsort(diff_time, axis=0)
        true_rank = np.argsort(idx, axis=0)

        idx = np.argsort(diff_time_hat, axis=0)
        pred_rank = np.argsort(idx, axis=0)
    else:
        true_rank = diff_time
        pred_rank = diff_time_hat
            
    # save the rank back
    for carno in carlist:
        if carno not in full_tss:
            #init
            full_tss[carno] = np.zeros((maxlap))
            full_samples[carno] = np.zeros((samplecnt, maxlap))
            full_tss[carno][:] = np.nan
            full_samples[carno][:,:] = np.nan
            full_tss[carno][:lap] = true_rank[caridmap[carno]][:lap]

        full_tss[carno][lap] = true_rank[caridmap[carno]][lap]
        
        full_samples[carno][:, lap] = pred_rank[caridmap[carno],:]
    
    return 


def eval_stint_direct(forecasts_et, prediction_length):
    """
    evaluate rank by timediff forecasting
    input:
        forecast    ; {}, carno -> 5 x totallen matrix 
            0,: -> lapstatus
            1,: -> true target
            2,: -> pred target
            3,  -> placeholder
            4,  -> placeholder
        start_offset[]; elapsed time for lap0, for one specific event
        prediction_length ; 
    return:
        forecasts  
    """

    #get car list for this lap
    carlist = list(forecasts_et.keys())
    #print('carlist:', carlist)

    caridmap={key:idx for idx, key in enumerate(carlist)}    

    #convert to elapsedtime
    #todo, Indy500 - > 200 max laps
    maxlap = 200
    
    diff_time = np.zeros((2, len(carlist), maxlap))
    diff_time[:,:] = np.nan
    
    for carno in forecasts_et.keys():

        lapnum = len(forecasts_et[carno][1,:])
        
        timediff_array = forecasts_et[carno][1,:]
        diff_time[0, caridmap[carno],:lapnum] = timediff_array
        
        timediff_array = forecasts_et[carno][2,:]
        diff_time[1, caridmap[carno],:lapnum] = timediff_array
        
        #maxlap = max(maxlap, len(forecasts_et[carno][1,:]))

    #calculate rank, support nan
    idx = np.argsort(diff_time[0], axis=0)
    true_rank = np.argsort(idx, axis=0)

    idx = np.argsort(diff_time[1], axis=0)
    pred_rank = np.argsort(idx, axis=0)
        
    # save the rank back
    for carno in forecasts_et.keys():
        lapnum = len(forecasts_et[carno][1,:])
        
        forecasts_et[carno][3,:] = true_rank[caridmap[carno],:lapnum]
        forecasts_et[carno][4,:] = pred_rank[caridmap[carno],:lapnum]
    
    return forecasts_et



#calc rank
def eval_stint_rank(forecasts_et, prediction_length, start_offset):
    """
    evaluate rank by laptime forecasting
    
    input:
        test_ds       ; must be test set for a single event, because test_ds itself does not 
                        contain features to identify the eventid
        start_offset[]; elapsed time for lap0, for one specific event
        tss,forecasts ; forecast result
        prediction_length ; 
    return:
        rank_ret      ;  [lap, elapsed_time, true_rank, pred_rank] 
        forecasts_et  ;  {[completed_laps][carno]} ->(elapsed_time, elapsed_time_pred)
        
    """

    #get car list for this lap
    carlist = list(forecasts_et.keys())
    #print('carlist:', carlist)

    caridmap={key:idx for idx, key in enumerate(carlist)}    

    #convert to elapsedtime
    #todo, Indy500 - > 200 max laps
    maxlap = 200
    
    elapsed_time = np.zeros((2, len(carlist), maxlap))
    elapsed_time[:,:] = np.nan
    
    for carno in forecasts_et.keys():

        #start_offset is global var
        if isinstance(start_offset, pd.core.frame.DataFrame):
            offset = start_offset[(start_offset['car_number']==carno)].elapsed_time.values[0] 
        
        lapnum = len(forecasts_et[carno][1,:])
        
        laptime_array = forecasts_et[carno][1,:]
        elapsed = np.cumsum(laptime_array) + offset
        elapsed_time[0, caridmap[carno],:lapnum] = elapsed
        
        laptime_array = forecasts_et[carno][2,:]
        elapsed = np.cumsum(laptime_array) + offset
        elapsed_time[1, caridmap[carno],:lapnum] = elapsed
        
        #maxlap = max(maxlap, len(forecasts_et[carno][1,:]))

    #calculate rank, support nan
    idx = np.argsort(elapsed_time[0], axis=0)
    true_rank = np.argsort(idx, axis=0)

    idx = np.argsort(elapsed_time[1], axis=0)
    pred_rank = np.argsort(idx, axis=0)
        
    # save the rank back
    for carno in forecasts_et.keys():
        lapnum = len(forecasts_et[carno][1,:])
        
        forecasts_et[carno][3,:] = true_rank[caridmap[carno],:lapnum]
        forecasts_et[carno][4,:] = pred_rank[caridmap[carno],:lapnum]
    
    return forecasts_et


# In[13]:


def get_sign(diff):
    if diff > 0:
        sign = 1
    elif diff < 0:
        sign = -1
    else:
        sign = 0
    return sign
                
#
# configurataion
#
# model path:  <_dataset_id>/<_task_id>-<trainid>/
#_dataset_id = 'indy2013-2018-nocarid'
_dataset_id = 'indy2013-2018'
_test_event = 'Indy500-2018'
#_test_event = 'Indy500-2019'
_train_len = 40
_test_train_len = 40

_feature_mode = FEATURE_STATUS
_context_ratio = 0.

#_task_id = 'timediff'  # rank,laptime, the trained model's task
#_run_ts = COL_TIMEDIFF   #COL_LAPTIME,COL_RANK
#_exp_id='timediff2rank'  #rank, laptime, laptim2rank, timediff2rank... 
#
#_task_id = 'lapstatus'  # rank,laptime, the trained model's task
#_run_ts = COL_LAPSTATUS   #COL_LAPTIME,COL_RANK
#_exp_id='lapstatus'  #rank, laptime, laptim2rank, timediff2rank... 

_task_id = 'laptime'  # rank,laptime, the trained model's task
_run_ts = COL_LAPTIME   #COL_LAPTIME,COL_RANK
_exp_id='laptime2rank'  #rank, laptime, laptim2rank, timediff2rank... 

_inlap_status = 1
_force_endpit_align = False
_include_endpit = False

#_use_mean = False   # mean or median to get prediction from samples
_use_mean = True   # mean or median to get prediction from samples

# In[16]:
global_start_offset = {}
global_carids = {}
laptime_data = None
laptime_data_save = None
freq = "1min"
decode_carids = {}

years = ['2013','2014','2015','2016','2017','2018','2019']
events = [f'Indy500-{x}' for x in years]
events_id={key:idx for idx, key in enumerate(events)}
dbid = f'Indy500_{years[0]}_{years[-1]}_v9_p{_inlap_status}'

def init(pitmodel = ''):
    global global_carids, laptime_data, global_start_offset, decode_carids,_pitmodel
    global dbid, _inlap_status

    dbid = f'Indy500_{years[0]}_{years[-1]}_v9_p{_inlap_status}'

    stagedata = {}
    for event in events:
        #dataid = f'{event}-{year}'
        #alldata, rankdata, acldata, flagdata
        stagedata[event] = load_data(event)
    
        alldata, rankdata, acldata = stagedata[event]
        
        #offset
        global_start_offset[event] = rankdata[rankdata['completed_laps']==0][['car_number','elapsed_time']]
    
    # start from here
    import pickle
    #with open('laptime_rank_timediff_fulltest-oracle-%s.pickle'%year, 'rb') as f:
    laptimefile = f'laptime_rank_timediff_pit-oracle-{dbid}.pickle'
    with open(laptimefile, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        global_carids, laptime_data = pickle.load(f, encoding='latin1')

        laptime_data_save = laptime_data
    
    decode_carids={carid:carno for carno, carid in global_carids.items()}
    print(f'init: load dataset {laptimefile} with {len(laptime_data)} races, {len(global_carids)} cars')

    if not isinstance(pitmodel, str):
        _pitmodel = PitModelSimple(top8=(True if pitmodel==0 else False))
        print(f'init pitmodel as PitModelSimple')
    elif pitmodel=='oracle':
        _pitmodel = pitmodel
    else:
        _pitmodel = PitModelMLP(modelfile = pitmodel)
        print(f'init pitmodel as PitModelMLP(pitmodel)')


def get_evalret(df):
    correct = df[df['sign']==df['pred_sign']]
    acc = len(correct)/len(df)

    mae1 = np.sum(np.abs(df['pred_diff'].values - df['diff'].values))/len(df)

    rmse = math.sqrt(mean_squared_error(df['pred_diff'].values , df['diff'].values))
    mae = mean_absolute_error(df['pred_diff'].values , df['diff'].values)
    r2 = r2_score(df['pred_diff'].values , df['diff'].values)

    #naive result
    n_correct = df[df['startrank']==df['endrank']]
    acc_naive = len(n_correct)/len(df)
    mae_naive = np.mean(np.abs(df['diff'].values))

    rmse_naive = math.sqrt(mean_squared_error(df['startrank'].values , df['endrank'].values))
    r2_naive = r2_score(df['startrank'].values , df['endrank'].values)

    #print(f'pred: acc={acc}, mae={mae},{mae1}, rmse={rmse},r2={r2}, acc_naive={acc_naive}, mae_naive={mae_naive}, {mae_naive1}')
    #print(f'pred: acc={acc}, mae={mae}, rmse={rmse},r2={r2}, acc_naive={acc_naive}, mae_naive={mae_naive}')
    print('model: acc={%.2f}, mae={%.2f}, rmse={%.2f},r2={%.2f}, {%d}\n \
           naive: acc={%.2f}, mae={%.2f}, rmse={%.2f},r2={%.2f}'%(
               acc, mae, rmse, r2, len(df),
               acc_naive, mae_naive, rmse_naive, r2_naive
            )
        )
    
    return np.array([[acc, mae, rmse, r2],[acc_naive, mae_naive, rmse_naive, r2_naive]])

    #print(f'pred: acc={acc}, mae={mae},{mae1}, rmse={rmse},r2={r2}, acc_naive={acc_naive}, mae_naive={mae_naive}')
    
    #return acc, mae, rmse, r2

def get_evalret_shortterm(df):
    maxlap = np.max(df['startlap'].values)
    minlap = np.min(df['startlap'].values)
 
    top1 = df[df['endrank']==0]
    top1_pred = df[df['pred_endrank']==0]

    correct = top1_pred[top1_pred['pred_endrank']==top1_pred['endrank']]
    #acc = len(correct)/len(top1_pred)
    acc = len(correct)/(len(top1_pred) + 1e-10)

    rmse = math.sqrt(mean_squared_error(df['pred_endrank'].values , df['endrank'].values))
    mae = mean_absolute_error(df['pred_endrank'].values , df['endrank'].values)
    r2 = r2_score(df['pred_endrank'].values , df['endrank'].values)
    mae1 = np.sum(np.abs(df['pred_endrank'].values  - df['endrank'].values))
    mae1 = mae1/ (maxlap -minlap +1)


    #naive result
    top1_naive = df[df['startrank']==0]
    n_correct = top1_naive[top1_naive['startrank']==top1_naive['endrank']]
    acc_naive = len(n_correct)/len(top1_naive)
    mae_naive = np.mean(np.abs(df['diff'].values))

    mae_naive1 = np.sum(np.abs(df['diff'].values))
    mae_naive1 = mae_naive1 / (maxlap - minlap + 1)

    rmse_naive = math.sqrt(mean_squared_error(df['startrank'].values , df['endrank'].values))
    r2_naive = r2_score(df['startrank'].values , df['endrank'].values)

    #print(f'pred: acc={acc}, mae={mae},{mae1}, rmse={rmse},r2={r2}, acc_naive={acc_naive}, mae_naive={mae_naive}, {mae_naive1}')
    #print(f'pred: acc={acc}, mae={mae}, rmse={rmse},r2={r2}, acc_naive={acc_naive}, mae_naive={mae_naive}')
    print('model: acc={%.2f}, mae={%.2f}, rmse={%.2f},r2={%.2f}, {%d}\n \
           naive: acc={%.2f}, mae={%.2f}, rmse={%.2f},r2={%.2f}'%(
               acc, mae, rmse, r2, len(top1_pred),
               acc_naive, mae_naive, rmse_naive, r2_naive
            )
        )
    
    return np.array([[acc, mae, rmse, r2],[acc_naive, mae_naive, rmse_naive, r2_naive]])

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    # logging configure
    import logging.config
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # cmd argument parser
    usage = 'stint_predictor_fastrun.py --datasetid datasetid --testevent testevent --task taskid '
    parser = OptionParser(usage)
    parser.add_option("--task", dest="taskid", default='laptime')
    parser.add_option("--datasetid", dest="datasetid", default='indy2013-2018')
    parser.add_option("--testevent", dest="testevent", default='Indy500-2018')
    parser.add_option("--contextratio", dest="contextratio", default=0.)
    parser.add_option("--trim", dest="trim", type=int, default=0)

    opt, args = parser.parse_args()

    #set global parameters
    _dataset_id = opt.datasetid
    _test_event = opt.testevent
    _trim = opt.trim
    if opt.taskid == 'laptime':
        _task_id = 'laptime'  # rank,laptime, the trained model's task
        _run_ts = COL_LAPTIME   #COL_LAPTIME,COL_RANK
        _exp_id='laptime2rank'  #rank, laptime, laptim2rank, timediff2rank... 

    elif opt.taskid == 'timediff':
        _task_id = 'timediff'  # rank,laptime, the trained model's task
        _run_ts = COL_TIMEDIFF   #COL_LAPTIME,COL_RANK
        _exp_id='timediff2rank'  #rank, laptime, laptim2rank, timediff2rank... 

    elif opt.taskid == 'rank':
        _task_id = 'rank'  # rank,laptime, the trained model's task
        _run_ts = COL_RANK   #COL_LAPTIME,COL_RANK
        _exp_id='rank'  #rank, laptime, laptim2rank, timediff2rank... 
    else:
        logger.error('taskid:%s not support yet!', opt.taskid)
        sys.exit(-1)

    if _dataset_id=='' or _test_event=='':
        logger.error('datasetid and testevnet cannot be null')
        sys.exit(-1)

    if _dataset_id.find('pitage') > 0:
        _feature_mode = FEATURE_PITAGE

    logger.info('Start evaluation, dataset=%s, testevent=%s, taskid=%s', _dataset_id, _test_event,
            _task_id)



    init()

    mytest()


