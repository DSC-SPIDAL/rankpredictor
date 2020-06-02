#!/usr/bin/env python
# coding: utf-8

# ### stint simulator
# based on: Stint-Predictor-Fastrun
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
# share the memory
#COL_LAPSTATUS_PRED = 8   # for dynamic lapstatus predictions
#LAPSTATUS SAVED in forecast_et
COL_LAPSTATUS_SAVE = 0   #laptime no use
COL_CAUTION_LAPS_INSTINT_SAVE=7
COL_LAPS_INSTINT_SAVE= 8


FEATURE_STATUS = 2
FEATURE_PITAGE = 4

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


# In[ ]:





# In[4]:


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

#
# remove NaN at the tail
# there should be no nans in the middle of the ts
def get_modestr(a):
    modestr = ''
    for key in _mode_map:
        if test_flag(a, key):
            modestr += '%s,'%(_mode_map[key])
            
    return modestr

# endpos -> vector of prediction_length
_track_pred  = {}
_track_true  = {}
def init_track_model():
    global _track_pred,_track_true
    _track_pred = {}
    _track_true  = {}
    
def get_track_model(track_rec, endpos, prediction_length, context_len=10):
    """
    return the predicted track status
    """
    global _track_pred,_track_true
    # this is the perfect track model for Indy500 2018
    track_model = [6,4,4,5,6,6,4]
    if endpos in _track_pred:
        return _track_pred[endpos]
    else:
        #get yflag lap count from the start pred point
        yflaplen = 0
        for i in range(1, context_len):
            if track_rec[- prediction_length - i] == 1:
                yflaplen += 1
            else:
                break
                
        #laps remain, fill into the future
        trackpred = np.array([0 for x in range(prediction_length)])
        
        yflap_pred = random.choice(track_model)
        if yflaplen > 0 and yflap_pred > yflaplen:
            trackpred[:(yflap_pred - yflaplen)] = 1
        _track_pred[endpos] = trackpred
        
        _track_true[endpos]  = track_rec[- prediction_length:].copy()
        
        return trackpred

    
# endpos -> vector of prediction_length
_track_adjust  = {}
def init_adjust_track_model():
    global _track_adjust
    _track_adjust = {}
    
def adjust_track_model(track_rec, endpos, prediction_length, tailpos):
    """
    input:
        tailpos ; <0 end pos of 1
    return the predicted track status
    """
    global _track_adjust
    # this is the perfect track model for Indy500 2018
    track_model = [-1,0,1]
    if endpos in _track_adjust:
        return _track_adjust[endpos]
    else:
        yflap_adjust = random.choice(track_model)
        
        #laps remain, fill into the future
        trackadjust = track_rec[-prediction_length:].copy()
        if yflap_adjust == -1:
            trackadjust[tailpos] = 0
        elif yflap_adjust == 1:
            trackadjust[tailpos] = 0
            if (tailpos + 1) <= -1:
                trackadjust[tailpos+1] = 1
        
        _track_adjust[endpos] = trackadjust
        
        return trackadjust

# carno -> lap_status
_lap_adjust = {}    
_empirical_model = {}
def init_adjust_pitmodel():
    global _lap_adjust
    _lap_adjust = {}    
    _empirical_model = {}

def get_adjust_lapstatus(carno, lapstatus, force = True):
    """
    init the lapstatus for each car, save it for future reference
    
    input:
        carno;
        lapstatus  ; the trueth
    
    """
    if carno not in _lap_adjust:
        #adjust it
        lapadjust = lapstatus.copy()
        for pos in range(0, len(lapstatus)):
            if lapadjust[pos] == 1:

                success = False

                while(not success):
                    # adjust this pit lap position
                    pos_adjust = get_random_choice(_adjust_model)

                    new_pos = pos + pos_adjust

                    if new_pos >= 0 and new_pos < len(lapstatus):
                        #valid
                        lapadjust[pos] = 0
                        lapadjust[new_pos] = 1
                        success = True
                        
                        #add statistics
                        if pos_adjust not in _empirical_model:
                            _empirical_model[pos_adjust] = 1
                        else:
                            _empirical_model[pos_adjust] += 1

                    if force==False:
                        break

        _lap_adjust[carno] = lapadjust

    return _lap_adjust[carno]
        
        
def build_random_model(modeldict):
    """
    input:
        modeldict ; {val: probability}
    return:
        model  ;  [val, cdf]
    """
    # val, cdf
    cdf = 0
    model = np.zeros((len(modeldict), 2))
    for idx, val in enumerate(sorted(modeldict.keys())):
        model[idx, 0] = val
        model[idx, 1] = cdf + modeldict[val]
        cdf = model[idx, 1]
        
    #normalize
    model[:, 1] = model[:, 1]/cdf
    return model
    
def print_model(model, iscdf=True):
    """
    input:
        model  ;  [val, cdf]
    """
    sorted_model = model[np.argsort(model[:, 0])]
    cdf = 0
    
    sumval = 1.
    if not iscdf:
        sumval = np.sum(sorted_model[:,1])
    
    ret = []
    for row in sorted_model:
        ret.append((row[0], (row[1]-cdf)/sumval))
        if iscdf:
            cdf = row[1]
    #output
    print(['%d:%.3f'%(x[0],x[1]) for x in ret])
    
    
def get_random_choice(model):
    """
    input:
        model  ;  [val, cdf]
    return:
        val according to its probability
    """
    
    target = np.random.rand()
    idx = np.sum(model[:,1] < target)
    return int(model[idx,0])
    
#_modeldict={-2:0.1,-1:0.2,0:0.4, 1:0.2, 2:0.1 }
_modeldict={-2:0.1,-1:0.2,0:0.05, 1:0.2, 2:0.1 }
_adjust_model = build_random_model(_modeldict)

def adjust_pit_model(lap_rec, prediction_length, force=True):
    """
    input:
        tailpos ; <0 end pos of 1
    return the predicted lap status
    """
    #laps remain, fill into the future
    lapadjust = lap_rec[-prediction_length:].copy()
    for pos in range(0, prediction_length):
        if lapadjust[pos] == 1:
            
            success = False
            
            while(not success):
                # adjust this pit lap position
                pos_adjust = get_random_choice(_adjust_model)

                new_pos = pos + pos_adjust

                if new_pos >= 0 and new_pos < prediction_length:
                    #valid
                    lapadjust[pos] = 0
                    lapadjust[new_pos] = 1
                    success = True
                    
                if force==False:
                    break
                    
    return lapadjust

def adjust_pit_model_fix(lap_rec, endpos, prediction_length):
    """
    input:
        tailpos ; <0 end pos of 1
    return the predicted lap status
    """
    adjust_model = [-1,0,1]
    lap_adjust = random.choice(adjust_model)
        
    #laps remain, fill into the future
    lapadjust = lap_rec[-prediction_length:].copy()
    for pos in range(0, prediction_length):
        if lapadjust[pos] == 1:
            # adjust this pit lap position
            pos_adjust = random.choice(adjust_model)

            if pos_adjust == -1:
                if (pos - 1 >= 0):
                    lapadjust[pos] = 0
                    lapadjust[pos - 1] = 1
            elif pos_adjust == 1:
                if (pos + 1 < prediction_length):
                    lapadjust[pos] = 0
                    lapadjust[pos + 1] = 1

    return lapadjust
    
# pit model is separate for each car
def get_pit_model(cuation_laps_instint, laps_instint, prediction_length):
    """
    return the predicted pit status
    """
    # this is the perfect empirical pit model for Indy500 2018
    pit_model_all = [[33, 32, 35, 32, 35, 34, 35, 34, 37, 32, 37, 30, 33, 36, 35, 33, 36, 30, 31, 33, 36, 37, 35, 34, 34, 33, 37, 35, 39, 32, 36, 35, 34, 32, 36, 32, 31, 36, 33, 33, 35, 37, 40, 32, 32, 34, 35, 36, 33, 37, 35, 37, 34, 35, 39, 32, 31, 37, 32, 35, 36, 39, 35, 36, 34, 35, 33, 33, 34, 32, 33, 34],
                [45, 44, 46, 44, 43, 46, 45, 43, 41, 48, 46, 43, 47, 45, 49, 44, 48, 42, 44, 46, 45, 45, 43, 44, 44, 43, 46]]
    pit_model_top8 = [[33, 32, 35, 33, 36, 33, 36, 33, 37, 35, 36, 33, 37, 34],
                 [46, 45, 43, 48, 46, 45, 45, 43]]
    
    pit_model = pit_model_all
    
    if cuation_laps_instint>10:
        #use low model
        pred_pit_laps = random.choice(pit_model[0])
    else:
        pred_pit_laps = random.choice(pit_model[1])
                
    #laps remain, fill into the future
    pitpred = np.array([0 for x in range(prediction_length)])
    
    if (pred_pit_laps > laps_instint) and (pred_pit_laps <= laps_instint + prediction_length):
        pitpred[pred_pit_laps - laps_instint - 1] = 1
         
    return pitpred    
    
def make_dataset_byevent(runs, prediction_length, freq, 
                       useeid = False,
                       run_ts= COL_LAPTIME, 
                       test_event = 'Indy500-2018',
                       test_cars = [],  
                       use_global_dict = True,
                       oracle_mode = MODE_ORACLE,
                       half_moving_win = 0,
                       train_ratio=0.8,
                       log_transform = False,
                       context_ratio = 0.,
                       verbose = False
                ):
    """
    split the ts to train and test part by the ratio
    
    input:
        oracle_mode: false to simulate prediction in real by 
                set the covariates of track and lap status as nan in the testset
        half_moving_win  ; extend to 0:-1 ,1:-1/2plen, 2:-plen
    
    """    
    run_ts= _run_ts 
    test_event = _test_event
    feature_mode = _feature_mode
    
    init_track_model()
    init_adjust_track_model()
    
    start = pd.Timestamp("01-01-2019", freq=freq)  # can be different for each time series

    train_set = []
    test_set = []
    
    #select run
    if runs>=0:
        _laptime_data = [laptime_data[runs].copy()]
    else:
        _laptime_data = laptime_data.copy()
    
   
    #add statistics for adjust test
    # trackstatus, lapstatus
    mae = [0,0]
    
    #_data: eventid, carids, datalist[carnumbers, features, lapnumber]->[laptime, rank, track, lap]]
    for _data in _laptime_data:
        _train = []
        _test = []
        
        if events[_data[0]] == test_event:
            test_mode = True
        
        else:
            test_mode = False
            #jump out
            continue            
            
        #statistics on the ts length
        ts_len = [ _entry.shape[1] for _entry in _data[2]]
        max_len = int(np.max(ts_len))
        train_len = int(np.max(ts_len) * train_ratio)
        
        if context_ratio != 0.:
            # add this part to train set
            context_len = int(np.max(ts_len) * context_ratio)
        else:    
            context_len = prediction_length*2
        if context_len < 10:
            context_len = 10
            
        if verbose:
            #print(f'====event:{events[_data[0]]}, train_len={train_len}, max_len={np.max(ts_len)}, min_len={np.min(ts_len)}')
            print(f'====event:{events[_data[0]]}, prediction_len={prediction_length},train_len={train_len}, max_len={np.max(ts_len)}, min_len={np.min(ts_len)},context_len={context_len}')
                
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
            
            if use_global_dict:
                carno = _data[1][rowid]
                carid = global_carids[_data[1][rowid]]
            else:
                #simulation dataset, todo, fix the carids as decoder
                carno = rowid
                carid = rowid
                
            
            if useeid:
                static_cat = [carid, _data[0]]    
            else:
                static_cat = [carid]    
                
            #first, get target a copy    
            # target can be COL_XXSTATUS
            target_val = rec[run_ts,:].copy().astype(np.float32)
            if log_transform:
                target_val = np.log(target_val + 1.0)                
            
            # adjust for disturbance analysis
            if test_mode and test_flag(oracle_mode, MODE_DISTURB_ADJUSTPIT):
                lap_status = rec[COL_LAPSTATUS, :].copy()
                rec[COL_LAPSTATUS, :] = get_adjust_lapstatus(carno, lap_status)
                
            # selection of features
            if test_flag(oracle_mode, MODE_NOTRACK) or test_flag(oracle_mode, MODE_ORACLE_LAPONLY):                
                rec[COL_TRACKSTATUS, :] = 0
            if test_flag(oracle_mode, MODE_NOLAP) or test_flag(oracle_mode, MODE_ORACLE_TRACKONLY):                
                rec[COL_LAPSTATUS, :] = 0

            test_rec_cnt = 0
            if not test_mode:
                
                # all go to train set
                _train.append({'target': target_val, 
                                'start': start, 
                                'feat_static_cat': static_cat,
                                'feat_dynamic_real': [rec[COL_TRACKSTATUS,:],
                                       rec[COL_LAPSTATUS,:]]
                              }
                              )
            else:
                # reset train_len
                #context_len = prediction_length*2
                #if context_len < 10:
                #    context_len = 10
                
                #context_len = train_len
                
                # multiple test ts(rolling window as half of the prediction_length)

                #step = -int(prediction_length/2) if half_moving_win else -prediction_length
                if half_moving_win == 1:
                    step = -int(prediction_length/2)
                elif half_moving_win == 2:
                    step = -prediction_length
                else:
                    step = -1
                
                #bug fix, fixed the split point for all cars/ts
                for endpos in range(max_len, context_len+prediction_length, 
                                    step):

                    #check if enough for this ts
                    if endpos > totallen:
                        continue
                    
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
 
                    # predicting with status model
                    if test_flag(oracle_mode, MODE_PREDTRACK):
                        predrec = get_track_model(track_rec, endpos, prediction_length)
                        track_rec[-prediction_length:] = predrec
                        #lap_rec[-prediction_length:] = 0
                        
                    if test_flag(oracle_mode, MODE_PREDPIT):
                        #predrec = get_track_model(track_rec, endpos, prediction_length)
                        #track_rec[-prediction_length:] = predrec
                        lap_rec[-prediction_length:] = get_pit_model(caution_laps_instint,
                                                                    laps_instint,prediction_length)
                        
                        #for pitage, use the predicted lap info to update pitage
                        start_pitage = pitage_rec[-prediction_length - 1]
                        for pos in range(prediction_length):
                            if lap_rec[-prediction_length + pos]==0:
                                pitage_rec[-prediction_length + pos] = start_pitage+1
                            else:
                                #new pit
                                start_pitage = 0
                                pitage_rec[-prediction_length + pos] = start_pitage
                        
                    # disturbe analysis
                    if test_flag(oracle_mode, MODE_DISTURB_CLEARTRACK):
                        # clear the oracle track status
                        # future 1s in trackstatus
                        # pattern like 0 1 xx
                        for _pos in range(-prediction_length + 1, -1):
                            if track_rec[_pos - 1] == 0:
                                track_rec[_pos] = 0
                                
                    if test_flag(oracle_mode, MODE_DISTURB_ADJUSTTRACK):
                        # adjust the end position of track, or caution lap length
                        # find the end of caution laps
                        _tail = 0
                        for _pos in range(-1,-prediction_length + 1,-1):
                            if track_rec[_pos] == 1:
                                #find the tail
                                _tail = _pos
                                break
                        if _tail != 0:
                            #found
                            adjustrec = adjust_track_model(track_rec, endpos, prediction_length, _tail)
                            track_rec[-prediction_length:] = adjustrec
                        
                    #if test_flag(oracle_mode, MODE_DISTURB_ADJUSTPIT):
                    #    # adjust the position of pit
                    #    if np.sum(lap_rec[-prediction_length:]) > 0:
                    #        adjustrec = adjust_pit_model(lap_rec, endpos, prediction_length)
                    #        lap_rec[-prediction_length:] = adjustrec
                        
                    #okay, end of adjustments, test difference here
                    # rec_raw .vs. track_rec, lap_rec
                    track_rec_raw = rec_raw[COL_TRACKSTATUS, :endpos]
                    lap_rec_raw = rec_raw[COL_LAPSTATUS, :endpos]
                    
                    mae[0] = mae[0] + np.nansum(np.abs(track_rec[-prediction_length:] - track_rec_raw[-prediction_length:]))
                    mae[1] = mae[1] + np.nansum(np.abs(lap_rec[-prediction_length:] - lap_rec_raw[-prediction_length:]))

                    if feature_mode == FEATURE_STATUS:
                        _test.append({'target': target_val[:endpos].astype(np.float32), 
                                'start': start, 
                                'feat_static_cat': static_cat,
                                'feat_dynamic_real': [track_rec,lap_rec]
                                 }
                              )   
                    elif feature_mode == FEATURE_PITAGE:
                        _test.append({'target': target_val[:endpos].astype(np.float32), 
                                    'start': start, 
                                    'feat_static_cat': static_cat,
                                    'feat_dynamic_real': [track_rec,lap_rec,pitage_rec]
                                     }
                                  )                       
                    test_rec_cnt += 1
            
            #add one ts
            if verbose:
                print(f'carno:{carno}, totallen:{totallen}, nancount:{nan_count}, test_reccnt:{test_rec_cnt}')

        train_set.extend(_train)
        test_set.extend(_test)

    print(f'train len:{len(train_set)}, test len:{len(test_set)}, mae_track:{mae[0]},mae_lap:{mae[1]},')
    
    train_ds = ListDataset(train_set, freq=freq)
    test_ds = ListDataset(test_set, freq=freq)    
    
    return train_ds, test_ds, train_set, test_set

def save_dataset(datafile,freq, prediction_length, cardinality, train_ds, test_ds):
    with open(datafile, 'wb') as f:
        #pack [global_carids, laptime_data]
        savedata = [freq, prediction_length, cardinality, train_ds, test_ds]
        #savedata = [freq, train_set, test_set]
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(savedata, f, pickle.HIGHEST_PROTOCOL)        


# ### test for Indy500

# In[6]:


def predict(test_ds,predictor):
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,  # test dataset
        predictor=predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
    )

    forecasts = list(forecast_it)
    tss = list(ts_it)
    print(f'tss len={len(tss)}, forecasts len={len(forecasts)}')
    
    return tss, forecasts
 
def load_model(prediction_length, model_name,trainid='2018',exproot='remote'):
    with mx.Context(mx.gpu(7)):    
        pred_ret = []

        rootdir = f'../models/{exproot}/{_dataset_id}/{_task_id}-{trainid}/'
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
            
        # deepAR-Oracle
        elif model_name == 'oracle':
            model=f'deepAR-Oracle-{_task_id}-all-indy-f1min-t{prediction_length}-e1000-r1_oracle_t{prediction_length}'
            modeldir = rootdir + model
            print(f'predicting model={model_name}, plen={prediction_length}')
            predictor =  Predictor.deserialize(Path(modeldir))
            print(f'loading model...done!, ctx:{predictor.ctx}')
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


# In[7]:


#calc rank
def eval_rank_bytimediff(test_ds,tss,forecasts,prediction_length):
    """
    timediff models
    
    works for one event only
    
    """

    carlist = []

    # carno-lap# -> elapsed_time[] array
    forecasts_et = dict()

    ds_iter =  iter(test_ds)
    for idx in range(len(test_ds)):
        test_rec = next(ds_iter)
        #global carid
        carno = decode_carids[test_rec['feat_static_cat'][0]]
        #print('car no:', carno)

        if carno not in carlist:
            carlist.append(carno)

        # calc elapsed time
        prediction_len = forecasts[idx].samples.shape[1]
        if prediction_length != prediction_len:
            print('error: prediction_len does not match, {prediction_length}:{prediction_len}')
            return []
        
        #forecast_laptime_mean = np.mean(forecasts[idx].samples, axis=0).reshape((prediction_len,1))
        forecast_laptime_mean = np.median(forecasts[idx].samples, axis=0).reshape((prediction_len,1))
        
        timediff_array = tss[idx].values.copy()

        #save the prediction
        completed_laps = len(tss[idx]) - prediction_len + 1
        #print('car no:', carno, 'completed_laps:', completed_laps)
        #key = '%s-%s'%(carno, completed_laps)
        #forecasts_et[key] = elapsed_time[-prediction_len:].copy()
        if completed_laps not in forecasts_et:
            forecasts_et[completed_laps] = {}
        forecasts_et[completed_laps][carno] = [timediff_array[-prediction_len:].copy(),
                                                   forecast_laptime_mean.copy()]


    # calc rank
    rank_ret = []
    for lap in forecasts_et.keys():
        #get car list for this lap
        carlist = list(forecasts_et[lap].keys())
        #print('carlist:', carlist)

        caridmap={key:idx for idx, key in enumerate(carlist)}

        #fill in data
        time_diff = np.zeros((2, len(carlist), prediction_len))
        for carno in carlist:
            carid = caridmap[carno]
            time_diff[0, carid, :] = forecasts_et[lap][carno][0].reshape((prediction_len))
            time_diff[1, carid, :] = forecasts_et[lap][carno][1].reshape((prediction_len))

        #calculate rank    
        idx = np.argsort(time_diff[0], axis=0)
        true_rank = np.argsort(idx, axis=0)

        idx = np.argsort(time_diff[1], axis=0)
        pred_rank = np.argsort(idx, axis=0)

        rank_ret.append([lap, time_diff, true_rank, pred_rank])
        
    return rank_ret,forecasts_et
    
#calc rank
def eval_laptime(test_ds,tss,forecasts,prediction_length, start_offset):
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

    carlist = []

    # carno-lap# -> elapsed_time[] array
    forecasts_et = dict()

    ds_iter =  iter(test_ds)
    for idx in range(len(test_ds)):
        test_rec = next(ds_iter)
        #global carid
        carno = decode_carids[test_rec['feat_static_cat'][0]]
        #print('car no:', carno)

        if carno not in carlist:
            carlist.append(carno)

        #start_offset is global var
        #offset = start_offset[(start_offset['car_number']==carno)].elapsed_time.values[0] 
        #print('start_offset:', offset)

        # calc elapsed time
        prediction_len = forecasts[idx].samples.shape[1]
        if prediction_length != prediction_len:
            print('error: prediction_len does not match, {prediction_length}:{prediction_len}')
            return []
        
        forecast_laptime_mean = np.mean(forecasts[idx].samples, axis=0).reshape((prediction_len,1))
        #forecast_laptime_mean = np.median(forecasts[idx].samples, axis=0).reshape((prediction_len,1))

        laptime_array = tss[idx].values.copy()
        #elapsed_time = np.cumsum(laptime_array) + offset

        laptime_array_hat = tss[idx].values.copy()
        laptime_array_hat[-prediction_len:] = forecast_laptime_mean 
        #elapsed_time_hat = np.cumsum(laptime_array) + offset

        #save the prediction
        completed_laps = len(tss[idx]) - prediction_len + 1
        #print('car no:', carno, 'completed_laps:', completed_laps)
        #key = '%s-%s'%(carno, completed_laps)
        #forecasts_et[key] = elapsed_time[-prediction_len:].copy()
        if completed_laps not in forecasts_et:
            forecasts_et[completed_laps] = {}
        #forecasts_et[completed_laps][carno] = [elapsed_time[-prediction_len-1:].copy(),
        #                                           elapsed_time_hat[-prediction_len-1:].copy()]
        forecasts_et[completed_laps][carno] = [laptime_array[-prediction_len:].copy(),
                                                   laptime_array_hat[-prediction_len:].copy()]


    # calc rank
    rank_ret = []
    for lap in forecasts_et.keys():
        #get car list for this lap
        carlist = list(forecasts_et[lap].keys())
        #print('carlist:', carlist)

        caridmap={key:idx for idx, key in enumerate(carlist)}

        #fill in data
        #elapsed_time = np.zeros((2, len(carlist), prediction_len+1))
        lap_time = np.zeros((2, len(carlist), prediction_len))
        for carno in carlist:
            carid = caridmap[carno]
            lap_time[0, carid, :] = forecasts_et[lap][carno][0].reshape((prediction_len))
            lap_time[1, carid, :] = forecasts_et[lap][carno][1].reshape((prediction_len))

        #calculate rank    
        #idx = np.argsort(elapsed_time[0], axis=0)
        #true_rank = np.argsort(idx, axis=0)
        true_laptime = lap_time[0]

        #idx = np.argsort(elapsed_time[1], axis=0)
        #pred_rank = np.argsort(idx, axis=0)
        pred_laptime = lap_time[1]

        rank_ret.append([lap, lap_time, true_laptime, pred_laptime])
        
    return rank_ret,forecasts_et    

#calc rank
def eval_rank(test_ds,tss,forecasts,prediction_length, start_offset):
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

    carlist = []

    # carno-lap# -> elapsed_time[] array
    forecasts_et = dict()

    ds_iter =  iter(test_ds)
    for idx in range(len(test_ds)):
        test_rec = next(ds_iter)
        #global carid
        carno = decode_carids[test_rec['feat_static_cat'][0]]
        #print('car no:', carno)

        if carno not in carlist:
            carlist.append(carno)

        #start_offset is global var
        if isinstance(start_offset, pd.core.frame.DataFrame):
            offset = start_offset[(start_offset['car_number']==carno)].elapsed_time.values[0] 
        
            
        #print('start_offset:', offset)

        # calc elapsed time
        prediction_len = forecasts[idx].samples.shape[1]
        if prediction_length != prediction_len:
            print('error: prediction_len does not match, {prediction_length}:{prediction_len}')
            return []
     
        if _use_mean:
            forecast_laptime_mean = np.mean(forecasts[idx].samples, axis=0).reshape((prediction_len,1))
        else:
            forecast_laptime_mean = np.median(forecasts[idx].samples, axis=0).reshape((prediction_len,1))

        if isinstance(start_offset, pd.core.frame.DataFrame):
            #print('eval_rank:laptime2rank')
            laptime_array = tss[idx].values.copy()
            elapsed_time = np.cumsum(laptime_array) + offset

            laptime_array = tss[idx].values.copy()
            laptime_array[-prediction_len:] = forecast_laptime_mean 
            elapsed_time_hat = np.cumsum(laptime_array) + offset
        else:
            #print('eval_rank:rank-direct')
            # rank directly
            elapsed_time  = tss[idx].values.copy()

            elapsed_time_hat = tss[idx].values.copy()
            elapsed_time_hat[-prediction_len:] = forecast_laptime_mean             

        #save the prediction
        completed_laps = len(tss[idx]) - prediction_len + 1
        #print('car no:', carno, 'completed_laps:', completed_laps)
        #key = '%s-%s'%(carno, completed_laps)
        #forecasts_et[key] = elapsed_time[-prediction_len:].copy()
        if completed_laps not in forecasts_et:
            forecasts_et[completed_laps] = {}
        #forecasts_et[completed_laps][carno] = [elapsed_time[-prediction_len-1:].copy(),
        #                                           elapsed_time_hat[-prediction_len-1:].copy()]
        forecasts_et[completed_laps][carno] = [elapsed_time[-prediction_len:].copy(),
                                                   elapsed_time_hat[-prediction_len:].copy()]


    # calc rank
    rank_ret = []
    for lap in forecasts_et.keys():
        #get car list for this lap
        carlist = list(forecasts_et[lap].keys())
        #print('carlist:', carlist)

        caridmap={key:idx for idx, key in enumerate(carlist)}

        #fill in data
        #elapsed_time = np.zeros((2, len(carlist), prediction_len+1))
        elapsed_time = np.zeros((2, len(carlist), prediction_len))
        for carno in carlist:
            carid = caridmap[carno]
            elapsed_time[0, carid, :] = forecasts_et[lap][carno][0].reshape((prediction_len))
            elapsed_time[1, carid, :] = forecasts_et[lap][carno][1].reshape((prediction_len))

        #calculate rank    
        idx = np.argsort(elapsed_time[0], axis=0)
        true_rank = np.argsort(idx, axis=0)

        idx = np.argsort(elapsed_time[1], axis=0)
        pred_rank = np.argsort(idx, axis=0)

        rank_ret.append([lap, elapsed_time, true_rank, pred_rank])
        
    return rank_ret,forecasts_et    
    
def get_acc(rank_ret,prediction_length, verbose = False):   
    """
    input:
        rank_ret: [lap, elapsed_time, true_rank, pred_rank], use [2][3] columns
    return:
        ((metrics...)
         (record count...))
         
    the result can be used to calculate micro/macro metrics
    """
    # evaluate
    #top1 accuracy
    top1acc = 0
    top1acc_farmost = 0
    top5acc = 0
    top5acc_farmost = 0
    tau = 0
    rmse = 0.
    mae = 0.
    
    for rec in rank_ret:
        trueRank = rec[2]
        predRank = rec[3]

        #top1 , rank = 0, first col is not prediction
        top1acc += np.sum((trueRank==0) & (predRank==0)) 
        
        top1acc_farmost += np.sum((trueRank[:,-1]==0) & (predRank[:,-1]==0))
        
        #top5
        top5acc += np.sum((trueRank<5) & (predRank<5)) 
        
        top5acc_farmost += np.sum((trueRank[:,-1]<5) & (predRank[:,-1]<5))
        
        # tau
        tao, _ = stats.kendalltau(trueRank, predRank)
        tau += tao
        
        #rmse
        rmse += mean_squared_error(predRank,trueRank)
        
        #mae
        mae += np.sum(np.abs(predRank - trueRank))
        
    recnt = len(rank_ret)
    if recnt > 0:
        top1acc = top1acc *1.0/ (recnt*prediction_length)
        top1acc_farmost = top1acc_farmost *1.0/ recnt
        top5acc = top5acc *1.0/ (5*recnt*prediction_length)
        top5acc_farmost = top5acc_farmost *1.0/ (5*recnt)
        tau = tau/recnt
        rmse = rmse/recnt
        
        mae = mae/recnt

        #debug only
        if _run_ts == COL_LAPSTATUS:
            tau = mae
        
        if verbose:
            print(f'total:{len(rank_ret)}, prediction_length:{prediction_length}') 
            print('top1acc=', top1acc,
                  'top1acc_farmost=', top1acc_farmost,
                  'top5acc=', top5acc,
                  'top5acc_farmost=', top5acc_farmost,
                 )
            print('tau = ', tau,
                 'rmse = ', rmse,
                 'mae = ', mae)
    
    return ((top1acc,top1acc_farmost,top5acc,top5acc_farmost,tau,rmse),
            (recnt*prediction_length,recnt,5*recnt*prediction_length,5*recnt,recnt,recnt))

#
# simulation 
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

def get_nextpit_raw(pitlaps, startlap):
    """
    input:
        pitlaps ; array of pitstops for all the cars
        startlap    ; 
    return
        nextpit ; nextpit for all the cars, nan for non-pit

    """
    nextpit = []
    nextpit_map = {}

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

    #return
    return nextpit_map, max(nextpit)


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
    save the lapstatus in laptime_data
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

            #save pit model related features
            rec[COL_LAPSTATUS_SAVE,:] = rec[COL_LAPSTATUS, :]
            rec[COL_CAUTION_LAPS_INSTINT_SAVE,:] = rec[COL_CAUTION_LAPS_INSTINT, :]
            rec[COL_LAPS_INSTINT_SAVE, :] = rec[COL_LAPS_INSTINT, :]
 

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
    rec[COL_LAPSTATUS,:] = 0
    rec[COL_LAPSTATUS,:endpos] = rec[COL_LAPSTATUS_SAVE, :endpos]
    rec[COL_CAUTION_LAPS_INSTINT,:endpos] = rec[COL_CAUTION_LAPS_INSTINT_SAVE, :endpos]
    rec[COL_LAPS_INSTINT, :endpos] = rec[COL_LAPS_INSTINT_SAVE, :endpos]
    #rec[COL_LAPSTATUS,:] = rec[COL_LAPSTATUS_SAVE, :]
    #rec[COL_CAUTION_LAPS_INSTINT,:] = rec[COL_CAUTION_LAPS_INSTINT_SAVE, :]
    #rec[COL_LAPS_INSTINT, :] = rec[COL_LAPS_INSTINT_SAVE, :]

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
                    if nextpos+1 < rec.shape[1]:
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
                if feature_mode == FEATURE_STATUS:
                    _test.append({'target': target_val[:endpos].astype(np.float32), 
                            'start': start, 
                            'feat_static_cat': static_cat,
                            'feat_dynamic_real': [track_rec,lap_rec]
                             }
                          )   
                elif feature_mode == FEATURE_PITAGE:
                    _test.append({'target': target_val[:endpos].astype(np.float32), 
                                'start': start, 
                                'feat_static_cat': static_cat,
                                'feat_dynamic_real': [track_rec,lap_rec,pitage_rec]
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


# works on lapstatus ground truth
def sim_onestep_ex(predictor, prediction_length, freq, 
                   startlap, endlap,
                   oracle_mode = MODE_ORACLE,
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


    """    
    run_ts= _run_ts 
    test_event = _test_event    
    feature_mode = _feature_mode
    context_ratio = _context_ratio
    train_len = _train_len

    start = pd.Timestamp("01-01-2019", freq=freq)  # can be different for each time series

    test_set = []
    forecasts_et = {}

    _laptime_data = laptime_data.copy()

    endpos = startlap + prediction_length + 1
    #while(endpos <= endlap + 1):
    while(endpos <= endlap + prediction_length + 1):
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
                    forecasts_et[carno][0,:] = rec[COL_LAPSTATUS, :].copy()
                    forecasts_et[carno][1,:] = rec[run_ts,:].copy().astype(np.float32)
                    forecasts_et[carno][2,:] = rec[run_ts,:].copy().astype(np.float32)
                    #forecasts_et[carno][2,:endpos] = rec[run_ts,:endpos].copy().astype(np.float32)
                
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
                if feature_mode == FEATURE_STATUS:
                    _test.append({'target': target_val[:endpos].astype(np.float32), 
                            'start': start, 
                            'feat_static_cat': static_cat,
                            'feat_dynamic_real': [track_rec,lap_rec]
                             }
                          )   
                elif feature_mode == FEATURE_PITAGE:
                    _test.append({'target': target_val[:endpos].astype(np.float32), 
                                'start': start, 
                                'feat_static_cat': static_cat,
                                'feat_dynamic_real': [track_rec,lap_rec,pitage_rec]
                                 }
                              )   
                test_rec_cnt += 1

        # end of for each ts

        # RUN Prediction here
        test_ds = ListDataset(_test, freq=freq)

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_ds,  # test dataset
            predictor=predictor,  # predictor
            num_samples=100,  # number of sample paths we want for evaluation
        )

        forecasts = list(forecast_it)
        tss = list(ts_it)

        #save the forecast results
        ds_iter =  iter(test_ds)
        for idx in range(len(test_ds)):
            test_rec = next(ds_iter)
            #global carid
            carno = decode_carids[test_rec['feat_static_cat'][0]]

            forecast_laptime_mean = np.mean(forecasts[idx].samples, axis=0).reshape((prediction_length))
            
            #update the forecasts , ready to use in the next prediction(regresive forecasting)
            forecasts_et[carno][2, len(tss[idx]) - prediction_length:len(tss[idx])] = forecast_laptime_mean.copy()
 

        #go forward
        endpos += prediction_length

    #clear the unpred part
    for carno in forecasts_et.keys():
        forecasts_et[carno][2, endlap+1:] = np.nan

    return forecasts_et


def sim_onestep(predictor, prediction_length, freq, 
                   startlap, endlap,
                   oracle_mode = MODE_ORACLE,
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


    """    
    run_ts= _run_ts 
    test_event = _test_event    
    feature_mode = _feature_mode
    context_ratio = _context_ratio
    train_len = _train_len

    start = pd.Timestamp("01-01-2019", freq=freq)  # can be different for each time series

    test_set = []
    forecasts_et = {}

    _laptime_data = laptime_data.copy()

    #add statistics for adjust test
    # trackstatus, lapstatus
    mae = [0,0]

    #_data: eventid, carids, datalist[carnumbers, features, lapnumber]->[laptime, rank, track, lap]]
    for _data in _laptime_data:
        _test = []

        if events[_data[0]] != test_event:
            #jump out
            continue

        #statistics on the ts length
        ts_len = [ _entry.shape[1] for _entry in _data[2]]
        max_len = int(np.max(ts_len))

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
            
            carno = _data[1][rowid]
            carid = global_carids[_data[1][rowid]]
            
            static_cat = [carid]    
                
            #first, get target a copy    
            # target can be COL_XXSTATUS
            target_val = rec[run_ts,:].copy().astype(np.float32)
            lap_status = rec[COL_LAPSTATUS, :].copy()
            track_status = rec[COL_TRACKSTATUS, :].copy()
            pitage_status = rec[COL_LAPS_INSTINT,:].copy()
            # <3, totallen> 
            forecasts_et[carno] = np.zeros((5, totallen))
            forecasts_et[carno][:,:] = np.nan
            forecasts_et[carno][0,:] = rec[COL_LAPSTATUS, :].copy()
            forecasts_et[carno][1,:] = rec[run_ts,:].copy().astype(np.float32)
            forecasts_et[carno][2,:] = rec[run_ts,:].copy().astype(np.float32)
            
            # selection of features
            if test_flag(oracle_mode, MODE_NOTRACK) or test_flag(oracle_mode, MODE_ORACLE_LAPONLY):                
                rec[COL_TRACKSTATUS, :] = 0
            if test_flag(oracle_mode, MODE_NOLAP) or test_flag(oracle_mode, MODE_ORACLE_TRACKONLY):                
                rec[COL_LAPSTATUS, :] = 0

            test_rec_cnt = 0

            if True:
                #bug fix, fixed the split point for all cars/ts
                #for endpos in range(max_len, context_len+prediction_length,step):
                #step = prediction_length
                #for endpos in range(startlap + prediction_length, endlap, step):
                endpos = startlap + prediction_length
                while(endpos < endlap and endpos < totallen):

                    # RUN Prediction for single record
                    _test = []
                    
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


                    if feature_mode == FEATURE_STATUS:
                        _test.append({'target': target_val[:endpos].astype(np.float32), 
                                'start': start, 
                                'feat_static_cat': static_cat,
                                'feat_dynamic_real': [track_rec,lap_rec]
                                 }
                              )   
                    elif feature_mode == FEATURE_PITAGE:
                        _test.append({'target': target_val[:endpos].astype(np.float32), 
                                    'start': start, 
                                    'feat_static_cat': static_cat,
                                    'feat_dynamic_real': [track_rec,lap_rec,pitage_rec]
                                     }
                                  )   


                    # RUN Prediction here, for single record
                    test_ds = ListDataset(_test, freq=freq)

                    forecast_it, ts_it = make_evaluation_predictions(
                        dataset=test_ds,  # test dataset
                        predictor=predictor,  # predictor
                        num_samples=100,  # number of sample paths we want for evaluation
                    )

                    forecasts = list(forecast_it)
                    tss = list(ts_it)

                    #get prediction result 
                    forecast_laptime_mean = np.mean(forecasts[0].samples, axis=0).reshape((prediction_length))
                    #update target_val
                    target_val[endpos-prediction_length:endpos] = forecast_laptime_mean 
                    rec[COL_TRACKSTATUS, endpos-prediction_length:endpos] = track_rec[-prediction_length:]
                    rec[COL_LAPSTATUS, endpos-prediction_length:endpos] = lap_rec[-prediction_length:]
                    rec[COL_LAPS_INSTINT, endpos-prediction_length:endpos] = pitage_rec[-prediction_length:]

                    #save forecast
                    #save the prediction
                    completed_laps = len(tss[0]) - prediction_length + 1
                    #print('car no:', carno, 'completed_laps:', completed_laps)
                    forecasts_et[carno][2, len(tss[0]) - prediction_length:len(tss[0])] = forecast_laptime_mean.copy()


                    test_rec_cnt += 1

                    #go forward
                    endpos += prediction_length

            #one ts
            if verbose:
                print(f'carno:{carno}, totallen:{totallen}, nancount:{nan_count}, test_reccnt:{test_rec_cnt}')

    return forecasts_et

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
                   datamode = MODE_ORACLE):
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
                   sample_cnt = 100):
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

    #init samples array
    full_samples = {}
    full_tss = {}

    for pitlap in range(10, maxlap-prediction_length):
        #1. update lap status
        debug_print(f'start pitlap: {pitlap}')
        if not (isinstance(_pitmodel, str) and _pitmodel == 'oracle'):
            update_lapstatus(pitlap)

        debug_print(f'update lapstatus done.')
        #run one step sim from pitlap to maxnext
        forecast, forecast_samples = sim_onestep_pred(predictor, prediction_length, freq,
                pitlap, pitlap + prediction_length,
                oracle_mode = datamode,
                sample_cnt = sample_cnt
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
        #ret = get_acc_onestint_pred(forecasts_et, pitlap, nextpit, nextpit_pred)
        ret = get_acc_onestep_shortterm(forecasts_et, pitlap, pitlap+prediction_length)

        rankret.extend(ret)

        # add to full_samples
        eval_full_samples(pitlap + prediction_length,
                forecast_samples, forecast, 
                full_samples, full_tss)

    #add to df
    df = pd.DataFrame(rankret, columns =['carno', 'startlap', 'startrank', 
                                         'endrank', 'diff', 'sign',
                                         'pred_endrank', 'pred_diff', 'pred_sign',
                                        ])

    return df, full_samples, full_tss






# oracle sim
def run_simulation(predictor, prediction_length, freq, 
                   datamode = MODE_ORACLE):
    """
    step:
        1. init the lap status model
        2. loop on each pit lap
            1. onestep simulation
            2. eval stint performance
    """

    rankret = []

    allpits, pitmat, maxlap = get_pitlaps()

    for pitlap in allpits:

        print(f'start pitlap: {pitlap}')
        nextpit, maxnext = get_nextpit(pitmat, pitlap)

        #run one step sim from pitlap to maxnext
        forecast = sim_onestep_ex(predictor, prediction_length, freq,
                pitlap, maxnext,
                oracle_mode = datamode
                )

        print(f'simulation done: {len(forecast)}')
        # calc rank from this result
        if _exp_id=='rank' or _exp_id=='timediff2rank':
            forecasts_et = eval_stint_direct(forecast, 2)
        elif _exp_id=='laptime2rank':
            forecasts_et = eval_stint_bylaptime(forecast, 2, global_start_offset[_test_event])

        else:
            print(f'Error, {_exp_id} evaluation not support yet')
            break

        # evaluate for this stint
        ret = get_acc_onestint(forecasts_et, pitlap, nextpit)
        rankret.extend(ret)


    #add to df
    df = pd.DataFrame(rankret, columns =['carno', 'startlap', 'startrank', 
                                         'endrank', 'diff', 'sign',
                                         'pred_endrank', 'pred_diff', 'pred_sign',
                                        ])

    return df


#
# ------------
#
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
def eval_full_samples(lap, forecast_samples, forecast, full_samples, full_tss, maxlap=200):
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
        

    #calculate rank, support nan
    idx = np.argsort(diff_time, axis=0)
    true_rank = np.argsort(idx, axis=0)

    idx = np.argsort(diff_time_hat, axis=0)
    pred_rank = np.argsort(idx, axis=0)
        
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


def get_sign(diff):
    if diff > 0:
        sign = 1
    elif diff < 0:
        sign = -1
    else:
        sign = 0
    return sign
                
def get_stint_acc(forecasts, trim=2, currank = False):
    """
    input:
        trim     ; steady lap of the rank (before pit_inlap, pit_outlap)
        forecasts;  carno -> [5,totallen]
                0; lap_status
                3; true_rank
                4; pred_rank
    output:
        carno, stintid, startrank, endrank, diff, sign
        
    """
    rankret = []
    for carno in forecasts.keys():
        lapnum = len(forecasts[carno][1,:])
        true_rank = forecasts[carno][3,:]
        pred_rank = forecasts[carno][4,:]
        
        pitpos_list = np.where(forecasts[carno][0,:] == 1)[0]
        
        stintid = 0
        startrank = true_rank[0]
        
        for pitpos in pitpos_list:
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
            
            rankret.append([carno, stintid, startrank, 
                            endrank, diff, sign,
                            pred_endrank, pred_diff, pred_sign
                            ])
            
            stintid += 1
            startrank = true_rank[pitpos-trim]
            
        #end
        if pitpos_list[-1] < lapnum - 1:
            endrank = true_rank[-1]
            diff = endrank - startrank
            sign = get_sign(diff)

            if currank:
                #force into currank model, zero doesn't work here
                pred_endrank = startrank
                pred_diff = pred_endrank - startrank
                pred_sign = get_sign(pred_diff)
            else:
                pred_endrank = pred_rank[-1]
                pred_diff = pred_endrank - startrank
                pred_sign = get_sign(pred_diff)

            rankret.append([carno, stintid, startrank, 
                            endrank, diff, sign,
                            pred_endrank, pred_diff, pred_sign
                            ])

    #add to df
    df = pd.DataFrame(rankret, columns =['carno', 'stintid', 'startrank', 
                                         'endrank', 'diff', 'sign',
                                         'pred_endrank', 'pred_diff', 'pred_sign',
                                        ])

    return df

#
# configurataion
#
# model path:  <_dataset_id>/<_task_id>-<trainid>/
#_dataset_id = 'indy2013-2018-nocarid'
_dataset_id = 'indy2013-2018'
_test_event = 'Indy500-2018'
#_test_event = 'Indy500-2019'
_train_len = 40

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

_use_mean = True   # mean or median to get prediction from samples

# In[16]:
global_start_offset = {}
global_carids = {}
laptime_data = None
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


def runtest(modelname, model, datamode, naivemode, trainid= "2018"):

    forecast = run_exp(2,2, train_ratio =0.1 , trainid = trainid, 
               datamode=datamode, model=model)


    if _exp_id=='rank' or _exp_id=='timediff2rank':
        forecasts_et = eval_stint_direct(forecast, 2)
    elif _exp_id=='laptime2rank':
        forecasts_et = eval_stint_bylaptime(forecast, 2, global_start_offset[_test_event])

    else:
        print(f'Error, {_exp_id} evaluation not support yet')
        return 0,0, 0,0

    df = get_stint_acc(forecasts_et, currank = naivemode, trim= _trim)

    correct = df[df['sign']==df['pred_sign']]
    acc = len(correct)/len(df)

    mae1 = np.sum(np.abs(df['pred_diff'].values - df['diff'].values))/len(df)

    rmse = mean_squared_error(df['pred_diff'].values , df['diff'].values)
    mae = mean_absolute_error(df['pred_diff'].values , df['diff'].values)
    r2 = r2_score(df['pred_diff'].values , df['diff'].values)

    print(f'pred: acc={acc}, mae={mae},{mae1}, rmse={rmse},r2={r2}')
    
    return acc, mae, rmse, r2

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



#
# evaluation code in Evaluate-forecasts-paper
#
#straight implementation of prisk
def quantile_loss(target, quantile_forecast, q):
    return 2.0 * np.nansum(
        np.abs(
            (quantile_forecast - target)
            * ((target <= quantile_forecast) - q)
        )
    )

def abs_target_sum(target): 
    return np.nansum(np.abs(target)) 

def prisk(full_samples, full_tss, verbose = False):
    """
    calculate prisk by convert <samples, tss> into gluonts format
    
    """
    carlist = full_tss.keys()
    tss = []
    forecasts = []
    forecasts_mean = []
    freq = '1min'
    start = pd.Timestamp("01-01-2019", freq=freq) 

    for car in carlist:
        testcar = car
        fc = SampleForecast(samples = full_samples[testcar][:, 12:], freq=freq, start_date=start + 12)

        samples = np.mean(full_samples[testcar][:, 12:], axis =0, keepdims=True)
        fc_mean = SampleForecast(samples = samples, freq=freq, start_date=start + 12)

        index = pd.date_range(start='2019-01-01 00:00:00', freq = 'T', periods = len(full_tss[testcar]))
        ts = pd.DataFrame(index = index, data = full_tss[testcar])    

        tss.append(ts)
        forecasts.append(fc)
        forecasts_mean.append(fc_mean)

    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9]) 
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(tss))
    if verbose:
        print(json.dumps(agg_metrics, indent=4))  
    
    print(agg_metrics["wQuantileLoss[0.1]"], agg_metrics["wQuantileLoss[0.5]"],agg_metrics["wQuantileLoss[0.9]"])
    
    return agg_metrics

def prisk_direct_bysamples(full_samples, full_tss, quantiles=[0.1,0.5,0.9], startid = 12, verbose=False):
    """
    calculate prisk by <samples, tss> directly (equal to gluonts implementation)
    
    target: endrank
    forecast: pred_endrank
    item_id: <carno, startlap>
    """
    
    carlist = full_tss.keys()
    
    prisk = np.zeros((len(carlist), len(quantiles)))
    target_sum = np.zeros((len(carlist)))
    aggrisk = np.zeros((len(quantiles)))
    
    for carid, carno in enumerate(carlist):

        # for this car
        forecast = full_samples[carno]
        target = full_tss[carno]
        
        #calc quantiles
        # len(quantiles) x 1
        quantile_forecasts = np.quantile(forecast, quantiles, axis=0)
        
        for idx, q in enumerate(quantiles):
            q_forecast = quantile_forecasts[idx]
            prisk[carid, idx] = quantile_loss(target[startid:], q_forecast[startid:], q)
            target_sum[carid] = abs_target_sum(target[startid:])
            
        if verbose==True and carno==3:
            print('target:', target[startid:])
            print('forecast:', q_forecast[startid:])
            print('target_sum:', target_sum[carid])
            
            print('quantile_forecasts:', quantile_forecasts[:,startid:])
        
    #agg
    #aggrisk = np.mean(prisk, axis=0)
    prisk_sum = np.nansum(prisk, axis=0)
    if verbose==True:
        print('prisk:',prisk)
        print('prisk_sum:',prisk_sum)
        print('target_sum:',target_sum)
    for idx, q in enumerate(quantiles):
        aggrisk[idx] = np.divide(prisk_sum[idx], np.sum(target_sum))
    
    agg_metrics = {}
    for idx, q in enumerate(quantiles):
        agg_metrics[f'wQuantileLoss[{q}]'] = aggrisk[idx]
        
    print(agg_metrics.values())
    
    return agg_metrics, aggrisk

def df2samples(dfall, prediction_len=2, samplecnt=1):
    """
    convert a df into <samples, tss> format
    
    this version works for the output of ml modles which contains only 1 sample
    """
    carlist = set(dfall.carno.values)
    full_samples = {}
    full_tss = {}

    startlaps = {}
    for car in carlist:
        startlaps[car] = set(dfall[dfall['carno']==car].startlap.values)
        
    #empty samples
    for carid, carno in enumerate(carlist):
        full_tss[carno] = np.zeros((200))
        full_tss[carno][:] = np.nan
        full_samples[carno] = np.zeros((samplecnt,200))
        full_samples[carno][:] = np.nan
        
        for startlap in startlaps[carno]:
            dfrec = dfall[(dfall['carno']==carno) & (dfall['startlap']==startlap)]
            
            curlap = int(dfrec.startlap.values[0] + prediction_len)
            target = dfrec.endrank.values[0]
            forecast = dfrec.pred_endrank.values[0]
            
            for idx in range(samplecnt):
                full_samples[carno][idx,curlap] = forecast
                
            full_tss[carno][curlap] = target
    
    return full_samples, full_tss
            
def df2samples(dfall, prediction_len=2, samplecnt=1):
    """
    convert a df into <samples, tss> format
    
    this version works for the output of ml modles which contains only 1 sample
    """
    carlist = set(dfall.carno.values)
    full_samples = {}
    full_tss = {}

    startlaps = {}
    for car in carlist:
        startlaps[car] = set(dfall[dfall['carno']==car].startlap.values)
        
    #empty samples
    for carid, carno in enumerate(carlist):
        full_tss[carno] = np.zeros((200))
        full_tss[carno][:] = np.nan
        full_samples[carno] = np.zeros((samplecnt,200))
        full_samples[carno][:] = np.nan
        
        for startlap in startlaps[carno]:
            dfrec = dfall[(dfall['carno']==carno) & (dfall['startlap']==startlap)]
            
            curlap = int(dfrec.startlap.values[0] + prediction_len)
            target = dfrec.endrank.values[0]
            forecast = dfrec.pred_endrank.values[0]
            
            for idx in range(samplecnt):
                full_samples[carno][idx,curlap] = forecast
                
            full_tss[carno][curlap] = target
    
    return full_samples, full_tss
    
    
def runs2samples(runret, errlist):
    """
    for stint results only
    
    get samples from the runs
    
    input:
        runret  ; list of result df <carno,startlap,startrank,endrank,diff,sign,pred_endrank,pred_diff,pred_sign,endlap,pred_endlap>
        errlist ; <car, startlap> list
    return:
        samples, tss
    """
    samplecnt = len(runret)
    carlist = set(runret[0].carno.values)
    full_samples = {}
    full_tss = {}
    
    #concat all dfs
    dfall = pd.concat(runret)
    
    
    startlaps = {}
    for car in carlist:
        startlaps[car] = set(dfall[dfall['carno']==car].startlap.values)
        
    #empty samples
    for carid, carno in enumerate(carlist):
        full_tss[carno] = np.zeros((200))
        full_tss[carno][:] = np.nan
        full_samples[carno] = np.zeros((samplecnt,200))
        full_samples[carno][:] = np.nan
        
        for startlap in startlaps[carno]:
            
            thisrec = [carno,startlap]
            if thisrec in errlist:
                continue
            
            dfrec = dfall[(dfall['carno']==carno) & (dfall['startlap']==startlap)]
            
            curlap = int(dfrec.startlap.values[0])
            target = dfrec.endrank.values[0]
            forecast = dfrec.pred_endrank.to_numpy()
            
            #if carno==12:
            #    print('forecast.shape', forecast.shape)
            
            full_samples[carno][:,curlap] = forecast
                
            full_tss[carno][curlap] = target
    
    return full_samples, full_tss    


#
# following functions works for short-term results only
# 

#def get_allsamples(year=2018, model='pitmodel'):
def get_allsamples_ex(dfx):
    """
    dfx is the results of multiple runs, ret of test_model call
    dfx[runid] -> < df, samples, tss>
    """
    runs = list(dfx.keys())
    runcnt = len(runs)
    
    full_samples = {}
    full_tss = dfx[runs[0]][2]
    carlist = list(full_tss.keys())
    samplecnt, lapcnt = dfx[runs[0]][1][carlist[0]].shape
    
    print('sacmplecnt:', samplecnt, 'lapcnt:',lapcnt,'runcnt:', runcnt)
    
    #empty samples
    for carid, carno in enumerate(carlist):
        full_samples[carno] = np.zeros((runcnt, lapcnt))
    
    for runid in runs:
        #one run
        tss = dfx[runid][2]
        forecast = dfx[runid][1]
        
        for carid, carno in enumerate(carlist):
            #get mean for this run
            forecast_mean = np.nanmean(forecast[carno], axis=0)
            full_samples[carno][runid, :] = forecast_mean
            
            #if carno==3 and runid == 0:
            #    print('forecast:',forecast_mean)
            
    return full_samples, full_tss


def do_rerank(dfout, short=True):
    """
    carno','startlap','startrank','endrank','diff','sign','pred_endrank','pred_diff','pred_sign','endlap','pred_endlap
    
    output of prediction of target can be float
    
    resort the endrank globally
    
    """
    
    cols=['carno','startlap','startrank','endrank','diff','sign','pred_endrank','pred_diff','pred_sign','endlap','pred_endlap']
    colid={x:id for id,x in enumerate(cols)}
    
    #df = dfout.sort_values(by=['startlap','carno'])
    print('rerank...')
    laps = set(dfout.startlap.values)
    
    dfs = []
    for lap in laps:
        df = dfout[dfout['startlap']==lap].to_numpy()
        
        #print('in',df)
        
        idx = np.argsort(df[:,colid['pred_endrank']], axis=0)
        true_rank = np.argsort(idx, axis=0)
    
        df[:,colid['pred_endrank']] = true_rank
        
        #reset preds 
        df[:,colid['pred_diff']] = df[:,colid['pred_endrank']] - df[:,colid['endrank']]

        for rec in df:
            if rec[colid['pred_diff']] == 0:
                rec[colid['pred_sign']] = 0
            elif rec[colid['pred_diff']] > 0:
                rec[colid['pred_sign']] = 1
            else:
                rec[colid['pred_sign']] = -1        
        
        #print('out',df)
        if len(dfs) == 0:
            dfs = df
        else:
            dfs = np.vstack((dfs, df))
        #dfs.append(df)
        #np.vstack(df)
        
    #dfret = pd.concat(dfs)
    #data = np.array(dfs)
    if short:
        dfret = pd.DataFrame(dfs.astype(int), columns = cols[:-2])
    else:
        dfret = pd.DataFrame(dfs.astype(int), columns = cols)
    return dfret


#
# empty main
#
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


