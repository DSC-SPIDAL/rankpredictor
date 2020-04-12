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
import random
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

   
def run_prediction_ex(test_ds, prediction_length, model_name,trainid):
    with mx.Context(mx.gpu(7)):    
        pred_ret = []

        rootdir = f'../models/remote/{_dataset_id}/{_task_id}-{trainid}/'
        # deepAR-Oracle
        if model_name == 'curtrack':
            model=f'deepAR-Oracle-{_task_id}-curtrack-indy-f1min-t{prediction_length}-e1000-r1_curtrack_t{prediction_length}'
            modeldir = rootdir + model
            print(f'predicting model={model_name}, plen={prediction_length}')
            predictor =  Predictor.deserialize(Path(modeldir))
            print(f'loading model...done!, ctx:{predictor.ctx}')
            tss, forecasts = predict(test_ds,predictor)
            pred_ret = [tss, forecasts]

        elif model_name == 'zerotrack':
            model=f'deepAR-Oracle-{_task_id}-nolap-zerotrack-indy-f1min-t{prediction_length}-e1000-r1_zerotrack_t{prediction_length}'
            modeldir = rootdir + model
            print(f'predicting model={model_name}, plen={prediction_length}')
            predictor =  Predictor.deserialize(Path(modeldir))
            print(f'loading model...done!, ctx:{predictor.ctx}')
            tss, forecasts = predict(test_ds,predictor)
            pred_ret = [tss, forecasts]
            
        # deepAR-Oracle
        elif model_name == 'oracle':
            model=f'deepAR-Oracle-{_task_id}-all-indy-f1min-t{prediction_length}-e1000-r1_oracle_t{prediction_length}'
            modeldir = rootdir + model
            print(f'predicting model={model_name}, plen={prediction_length}')
            predictor =  Predictor.deserialize(Path(modeldir))
            print(f'loading model...done!, ctx:{predictor.ctx}')
            tss, forecasts = predict(test_ds,predictor)
            pred_ret = [tss, forecasts]
        # deepAR-Oracle
        elif model_name == 'oracle-laponly':
            model=f'deepAR-Oracle-{_task_id}-all-indy-f1min-t{prediction_length}-e1000-r1_oracle-laponly_t{prediction_length}'
            modeldir = rootdir + model
            print(f'predicting model={model_name}, plen={prediction_length}')
            predictor =  Predictor.deserialize(Path(modeldir))
            print(f'loading model...done!, ctx:{predictor.ctx}')
            tss, forecasts = predict(test_ds,predictor)
            pred_ret = [tss, forecasts]
        # deepAR-Oracle
        elif model_name == 'oracle-trackonly':
            model=f'deepAR-Oracle-{_task_id}-all-indy-f1min-t{prediction_length}-e1000-r1_oracle-trackonly_t{prediction_length}'
            modeldir = rootdir + model
            print(f'predicting model={model_name}, plen={prediction_length}')
            predictor =  Predictor.deserialize(Path(modeldir))
            print(f'loading model...done!, ctx:{predictor.ctx}')
            tss, forecasts = predict(test_ds,predictor)
            pred_ret = [tss, forecasts]
        # deepAR
        elif model_name == 'deepAR':
            model=f'deepAR-{_task_id}-all-indy-f1min-t{prediction_length}-e1000-r1_deepar_t{prediction_length}'
            modeldir = rootdir + model
            print(f'predicting model={model_name}, plen={prediction_length}')
            predictor =  Predictor.deserialize(Path(modeldir))
            print(f'loading model...done!, ctx:{predictor.ctx}')
            tss, forecasts = predict(test_ds,predictor)
            pred_ret = [tss, forecasts]

        # naive
        elif model_name == 'naive':
            print(f'predicting model={model_name}, plen={prediction_length}')
            predictor =  NaivePredictor(freq= freq, prediction_length = prediction_length)
            tss, forecasts = predict(test_ds,predictor)
            pred_ret = [tss, forecasts]
        # zero, zero keeps the rank unchange
        elif model_name == 'zero':
            print(f'predicting model={model_name}, plen={prediction_length}')
            predictor =  ZeroPredictor(freq= freq, prediction_length = prediction_length)
            tss, forecasts = predict(test_ds,predictor)
            pred_ret = [tss, forecasts]

        # arima
        elif model_name == 'arima':
            print(f'predicting model={model_name}, plen={prediction_length}')
            predictor =  RForecastPredictor(method_name='arima',freq= freq, 
                                            prediction_length = prediction_length,trunc_length=60)
            tss, forecasts = predict(test_ds,predictor)
            pred_ret = [tss, forecasts]
        else:
            print(f'error: model {model_name} not support yet!')

        return pred_ret     
    
def load_model(prediction_length, model_name,trainid):
    with mx.Context(mx.gpu(7)):    
        pred_ret = []

        rootdir = f'../models/remote/{_dataset_id}/{_task_id}-{trainid}/'
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
        
        forecast_laptime_mean = np.mean(forecasts[idx].samples, axis=0).reshape((prediction_len,1))
        #forecast_laptime_mean = np.median(forecasts[idx].samples, axis=0).reshape((prediction_len,1))

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
    


# In[ ]:





# In[8]:


def run_test(runs, plens, half, trainids, train_ratio, testfunc, datamode='', models=[]):
    """
    
    input:
        plens=[2,5,10]
        half=[False]
        #trainids = ["indy500-r0.2","indy500-r0.4","indy500"]
        trainids = ["r0.5"]
        #half=[True,False]
        #plens=[2]
        runs = 5
        train_ratio=0.5 
        exp_id='mean-splitbystage-predpit'
        
        testfunc ; run_exp_predpit, run_exp_predtrack, run_exp ...

    return:
    
        dfret  ; average result of multiple runs
                 dataframe['model' , 'prediction_length', 'halfmode','datamode','trainid',
                         'top1acc','top1acc_farmost','top5acc','top5acc_farmost','tau','rmse',
                         'top1acc_std','top1acc_farmost_std','top5acc_std','top5acc_farmost_std','tau_std','rmse_std']
                          
        alldata_ret  ; for debug
            [runid][halfmode,plen,trainid] -> (pred_ret, test_ds, rank_ret)
                pred_ret[model] ->　[tss, forecasts]
                test_ds[model] ->　test_ds
                rank_ret[model] -> ([lap, elapsed_time, true_rank, pred_rank],forecast_ret)
                    forecast_ret[completed_laps][carno] -> (elapsed_time, elapsed_time_hat)
    
    """
    if plens == [] or half == [] or trainids == []:
        print("error with empty settings")
        return
    
    #testfunc or (datamode & models)
    if isinstance(testfunc,str) and (datamode == '' or models == []):
        print("error with testfunc")
        return

    allret = []
    alldata_ret = []
    for runid in range(runs):
        exp_data = []
        exp_result = []

        for halfmode in half:
            for plen in plens:
                for trainid in trainids:
                    print('='*10)
                    if not isinstance(testfunc,str):
                        pred_ret, test_ds, rank_ret, metric_ret = testfunc(plen, halfmode, 
                                                            train_ratio=train_ratio,
                                                            trainid=trainid)
                    else:
                        pred_ret, test_ds, rank_ret, metric_ret = run_exp(plen, halfmode, 
                                                            train_ratio=train_ratio,
                                                            trainid=trainid, 
                                                            datamode=datamode,
                                                            models=models)
                        

                    #save 
                    exp_data.append((pred_ret, test_ds, rank_ret))
                    exp_result.extend(metric_ret)

        #save result
        result = pd.DataFrame(exp_result, columns = ['model' , 'prediction_length', 'halfmode',
                                           'datamode','trainid',
                                           'top1acc','top1acc_farmost','top5acc',
                                           'top5acc_farmost','tau','rmse'])

        #result['runid'] = [runid for x in range(len(result))]
        allret.append(result)
        alldata_ret.append(exp_data)

    #final
    rowcnt = len(allret[0])
    metrics = np.empty((runs, rowcnt, 6))
    for runid, ret in enumerate(allret):
        metrics[runid, :,:] = ret[['top1acc','top1acc_farmost','top5acc',
                                           'top5acc_farmost','tau','rmse']].values


    #average
    averagemat = np.mean(metrics[:,:,:], axis=0)
    stdmat = np.std(metrics[:,:,:], axis=0)
    dfhead = allret[0][['model' , 'prediction_length', 'halfmode', 'datamode','trainid']]
    
    
    dfaverage = pd.DataFrame(averagemat, columns = ['top1acc','top1acc_farmost','top5acc',
                                       'top5acc_farmost','tau','rmse'])
    dfstd = pd.DataFrame(stdmat, columns = ['top1acc_std','top1acc_farmost_std','top5acc_std',
                                       'top5acc_farmost_std','tau_std','rmse_std'])
    dfret = pd.concat([dfhead, dfaverage, dfstd], axis=1)

    #if exp_id != '':
    #    dfret.to_csv(f'laptime2rank-evaluate-indy500-{exp_id}-result.csv', float_format='%.3f')

    return dfret, alldata_ret


def checkret_status(dataret, runid = 0, idx = 0,model='oracle'):
    """
    check the test_ds track and lap status
    
        alldata_ret  ; for debug
            [runid][halfmode,plen,trainid] -> (pred_ret, test_ds, rank_ret)
                pred_ret[model] ->　[tss, forecasts]
                test_ds[model] ->　test_ds
                rank_ret[model] -> ([lap, elapsed_time, true_rank, pred_rank],forecast_ret)
                    forecast_ret[completed_laps][carno] -> (elapsed_time, elapsed_time_hat)
    
    """

    _, plen = dataret[runid][idx][0][model][1][0].samples.shape
    test_ds = dataret[runid][idx][1][model]
   
    
    ds_iter =  iter(test_ds)
    yfcnt = 0
    pitcnt = 0
    for recid in range(len(test_ds)):
        test_rec = next(ds_iter)
        
        carno = decode_carids[test_rec['feat_static_cat'][0]]
        
        track_rec,lap_rec = test_rec['feat_dynamic_real']
        yfcnt += np.sum(track_rec[-plen:])
        pitcnt += np.sum(lap_rec[-plen:])
        
    print('yfcnt:', yfcnt, 'pitcnt:',pitcnt)


def get_ref_oracle_testds(plens, halfs, train_ratio=0.8, test_cars = []):           
    
    testset = {}
    for prediction_length in plens:
        for half_moving_win in halfs:
            train_ds, test_ds,_,_ = make_dataset_byevent(events_id[_test_event], prediction_length,freq, 
                                         oracle_mode=MODE_ORACLE,
                                         run_ts = _run_ts,
                                         test_cars=test_cars,
                                         half_moving_win= half_moving_win,
                                         train_ratio=train_ratio)    
            
            # get key
            key = '%d-%d'%(prediction_length,half_moving_win)
            testset[key] = test_ds
            
    return testset




def checkret_confusionmat(dataret, ref_testset, runid= 0, testid = '', model='oracle'):
    """
    output the 4x4 confusion matrix split by track and lap status
    
    input:
        ref_oracle_testds  ; oracle test ds
    
    """
    plen_length = len(dataret[runid])
    
    dflist = []
    for idx in range(plen_length):
        _, plen = dataret[runid][idx][0][model][1][0].samples.shape
        test_ds = dataret[runid][idx][1][model]
        rank_ret = dataret[runid][idx][2][model][0]

        key = '%d-%d'%(plen,0)
        if key not in ref_testset:
            print(f'error, {key} not found in ref_testset')
            continue
        
        ref_oracle_testds = ref_testset[key]
        if len(ref_oracle_testds) != len(test_ds):
            print('error, size of testds mismatch', len(ref_oracle_testds), len(test_ds))
            continue

        # confusion matrix for <trackstatus, lapstatus> type: 00,01,10,11
        # lap(start lap of prediction)  -> type
        lapmap = {}
        ds_iter =  iter(ref_oracle_testds)
        for recid in range(len(ref_oracle_testds)):
            test_rec = next(ds_iter) 

            carno = decode_carids[test_rec['feat_static_cat'][0]]

            track_rec,lap_rec = test_rec['feat_dynamic_real']
            yfcnt = np.sum(track_rec[-plen:])
            pitcnt = np.sum(lap_rec[-plen:])    

            #laptype = ('0' if yfcnt==0 else '1') + ('0' if pitcnt==0 else '1')

            lap = len(track_rec) - plen + 1
            if lap not in lapmap:
                #lapmap[lap] = laptype
                lapmap[lap] = (yfcnt, pitcnt)
            else:
                oldtype = lapmap[lap]
                lapmap[lap] = (yfcnt + oldtype[0], pitcnt + oldtype[1])


        #split the rank_ret by laptype
        types=['00','10','01','11']
        acc_ret = []
        for laptype in types:
            check_ret = []
            for item in rank_ret:
                typecnt = lapmap[item[0]]

                thetype = ('0' if typecnt[0]==0 else '1') + ('0' if typecnt[1]==0 else '1')

                if thetype == laptype:
                    check_ret.append(item)
            # get acc
            metrics = get_acc(check_ret,plen)
            recret = [testid, plen, laptype, len(check_ret)]
            recret.extend(metrics[0])
            acc_ret.append(recret)

        #add all test
        metrics = get_acc(rank_ret,plen)
        recret = [testid, plen, 'aa', len(rank_ret)]
        recret.extend(metrics[0])
        acc_ret.append(recret)
        
        _dfacc = pd.DataFrame(acc_ret, columns = ['testid','plen',
                                'type','reccnt','top1acc','top1acc_farmost','top5acc',
                                'top5acc_farmost','tau','rmse'])
        
        dflist.append(_dfacc)
    
    dfacc = pd.concat(dflist, axis=0)
    
    return dfacc


# In[9]:


def check_testds(datamode, test_cars=[]):
    """
    report mae, etc
    """
    for prediction_length in plens:
        for half_moving_win in half:
            train_ds, test_ds,_,_ = make_dataset_byevent(events_id[_test_event], prediction_length,freq, 
                                         oracle_mode=datamode,
                                         run_ts = _run_ts,
                                         test_cars=test_cars,
                                         half_moving_win= half_moving_win,
                                         train_ratio=train_ratio)
            
def dotest(config):
    acclist = []
    dflist = []
    for model in config.keys():
        conf = config[model]
        for teststr in conf.keys():
            testfunc = teststr
            datamode = conf[teststr]


            df, dataret = run_test(runs, plens, half, trainids, 
                                   train_ratio, testfunc, datamode=datamode,models=[model])

            #concat
            acc = checkret_confusionmat(dataret, ref_testset, 
                                        testid = teststr, model=model)
            dflist.append(df)
            acclist.append(acc)

    dfret = pd.concat(dflist, axis=0)
    dfacc = pd.concat(acclist, axis=0)
    return dfret, dfacc

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
            #all_pitlaps.append(list(pitstops))
            all_pitlaps[carno] = list(pitstops)

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
            nextpit.append(np.nan)

    #return
    return nextpit_map, max(nextpit)

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


# difference test on pit strategy
_pitstrategy_testcar = 12
_pitstrategy_lowmode = True
def update_onets(rec, startlap, carno):
    """
    update lapstatus after startlap basedon tsrec by pit prediction model

    input:
        tsrec   ; a ts with multiple features COL_XXX

    return:
        tsrec    ; updated for COL_LAPSTATUS, COL_CAUTION_LAPS_INSTINT, COL_LAPS_INSTINT

    """
    # this is the perfect empirical pit model for Indy500 2018
    pit_model_all = [[33, 32, 35, 32, 35, 34, 35, 34, 37, 32, 37, 30, 33, 36, 35, 33, 36, 30, 31, 33, 36, 37, 35, 34, 34, 33, 37, 35, 39, 32, 36, 35, 34, 32, 36, 32, 31, 36, 33, 33, 35, 37, 40, 32, 32, 34, 35, 36, 33, 37, 35, 37, 34, 35, 39, 32, 31, 37, 32, 35, 36, 39, 35, 36, 34, 35, 33, 33, 34, 32, 33, 34],
                [45, 44, 46, 44, 43, 46, 45, 43, 41, 48, 46, 43, 47, 45, 49, 44, 48, 42, 44, 46, 45, 45, 43, 44, 44, 43, 46]]
    pit_model_top8 = [[33, 32, 35, 33, 36, 33, 36, 33, 37, 35, 36, 33, 37, 34],
                 [46, 45, 43, 48, 46, 45, 45, 43]]
    
    #pit_model = pit_model_all
    pit_model = pit_model_top8
 
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


        if carno == _pitstrategy_testcar:
            # check strategy for test car
            if _pitstrategy_lowmode:
                if caution_laps_instint <= 10:
                    #use low model
                    pred_pit_laps = min(pit_model[0])
                else:
                    pred_pit_laps = min(pit_model[1])
            else:
                if caution_laps_instint <= 10:
                    #use low model
                    pred_pit_laps = max(pit_model[0])
                else:
                    pred_pit_laps = max(pit_model[1])
        else:
            retry = 0
            while retry < 10:
                if caution_laps_instint <= 10:
                    #use low model
                    pred_pit_laps = random.choice(pit_model[0])
                else:
                    pred_pit_laps = random.choice(pit_model[1])
    
                if pred_pit_laps <= laps_instint:
                    retry += 1
                    if retry == 10:
                        pred_pit_laps = laps_instint + 1
                    continue
                else:
                    break

        nextpos = curpos + pred_pit_laps - laps_instint

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
            rec[COL_CAUTION_LAPS_INSTINT, curpos+1: nextpos] = caution_laps_instint
            rec[COL_CAUTION_LAPS_INSTINT, nextpos] = 0
            for _pos in range(curpos+1, nextpos):
                rec[COL_LAPS_INSTINT, _pos] = rec[COL_LAPS_INSTINT, _pos - 1] + 1
            rec[COL_LAPS_INSTINT, nextpos] = 0

        #go forward
        curpos = nextpos

    debug_report('after update_onets', rec[COL_LAPSTATUS], startlap, carno)

    return


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
                debug_report(f'simu_onestep: {startlap}-{endpos}', target_val[:endpos], startlap, carno)

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


    return forecasts_et




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

        # check the lap status
        if ((startlap < lapnum) and (forecasts[carno][0, startlap] == 1)):

            startrank = true_rank[startlap-trim]
            
            if not carno in nextpit:
                continue

            pitpos = nextpit[carno]
            if np.isnan(pitpos):
                continue

            if not carno in nextpit_pred:
                continue

            pitpos_pred = nextpit_pred[carno]
            if np.isnan(pitpos_pred):
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

# pred sim
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
        update_lapstatus(pitlap)

        debug_print(f'update lapstatus done.')
        #2. get maxnext
        allpits_pred, pitmat_pred, maxlap = get_pitlaps()
        nextpit, maxnext = get_nextpit(pitmat, pitlap)
        nextpit_pred, maxnext_pred = get_nextpit(pitmat_pred, pitlap)

        #debug
        #_debug_carlist
        if 12 in nextpit and 12 in nextpit_pred:
            #print('nextpit:', nextpit[12], nextpit_pred[12], 'maxnext:', maxnext, maxnext_pred)
            debugstr = f'nextpit: {nextpit[12]}, {nextpit_pred[12]}, maxnext: {maxnext}, {maxnext_pred}'
            debug_print(debugstr)

        #run one step sim from pitlap to maxnext
        forecast = sim_onestep_pred(predictor, prediction_length, freq,
                pitlap, maxnext_pred,
                oracle_mode = datamode
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
        ret = get_acc_onestint_pred(forecasts_et, pitlap, nextpit, nextpit_pred)
        rankret.extend(ret)


    #add to df
    df = pd.DataFrame(rankret, columns =['carno', 'startlap', 'startrank', 
                                         'endrank', 'diff', 'sign',
                                         'pred_endrank', 'pred_diff', 'pred_sign',
                                        ])

    return df






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
def longterm_predict(predictor, runs, prediction_length, freq, 
                       useeid = False,
                       run_ts= COL_LAPTIME, 
                       test_event = 'Indy500-2018',
                       test_cars = [],  
                       use_global_dict = True,
                       oracle_mode = MODE_ORACLE,
                       half_moving_win = 0,
                       train_ratio=0.8,
                       log_transform = False,
                       verbose = False
                ):
    """
    split the ts to train and test part by the ratio
    
    input:
        oracle_mode: false to simulate prediction in real by 
                set the covariates of track and lap status as nan in the testset
        half_moving_win  ; extend to 0:-1 ,1:-1/2plen, 2:-plen
    
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
    
    init_track_model()
    init_adjust_track_model()
    
    start = pd.Timestamp("01-01-2019", freq=freq)  # can be different for each time series

    train_set = []
    test_set = []
    forecasts_et = {}
    
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
            lap_status = rec[COL_LAPSTATUS, :].copy()
            track_status = rec[COL_TRACKSTATUS, :].copy()
            pitage_status = rec[COL_LAPS_INSTINT,:].copy()
            # <3, totallen> 
            forecasts_et[carno] = np.zeros((5, totallen))
            forecasts_et[carno][:,:] = np.nan
            forecasts_et[carno][0,:] = rec[COL_LAPSTATUS, :].copy()
            forecasts_et[carno][1,:] = rec[run_ts,:].copy().astype(np.float32)
            forecasts_et[carno][2,:] = rec[run_ts,:].copy().astype(np.float32)
            
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
                    step = int(prediction_length/2)
                elif half_moving_win == 2:
                    step = prediction_length
                else:
                    step = 1
                
                #bug fix, fixed the split point for all cars/ts
                #for endpos in range(max_len, context_len+prediction_length,step):
                for endpos in range(context_len+prediction_length, max_len, step):

                    #check if enough for this ts
                    if endpos > totallen:
                        break
                        
                    # RUN Prediction for single record
                    _test = []
                    
                    # check pitstop(stint) in the last prediction
                    # use ground truth of target before the last pitstop
                    if np.sum(lap_status[endpos-2*prediction_length:endpos-prediction_length]) > 0:
                        # pit found 
                        # adjust endpos 
                        pitpos = np.where(lap_status[endpos-2*prediction_length:endpos-prediction_length] == 1)
                        endpos = endpos-2*prediction_length + pitpos[0][0] + prediction_length + 1
                        #print('endpos:',endpos,pitpos)
                        
                        #check if enough for this ts
                        if endpos > totallen:
                            break                        
                            
                        #reset target, status
                        target_val = rec[run_ts,:].copy().astype(np.float32)
                        rec[COL_LAPSTATUS, :] = lap_status
                        rec[COL_TRACKSTATUS, :] = track_status   
                        rec[COL_LAPS_INSTINT, :] = pitage_status  
                    
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
            
            #one ts
            if verbose:
                print(f'carno:{carno}, totallen:{totallen}, nancount:{nan_count}, test_reccnt:{test_rec_cnt}')

        #train_set.extend(_train)
        #test_set.extend(_test)

    #print(f'train len:{len(train_set)}, test len:{len(test_set)}, mae_track:{mae[0]},mae_lap:{mae[1]},')
    
    #train_ds = ListDataset(train_set, freq=freq)
    #test_ds = ListDataset(test_set, freq=freq)    
    
    return forecasts_et


# In[12]:

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


def run_exp(prediction_length, half_moving_win, train_ratio=0.8, trainid="r0.8",
                   test_event='Indy500-2018', test_cars = [], 
                   datamode = MODE_ORACLE,model = 'oracle'):
    """
    dependency: test_event, test on one event only
    
    """
    retdf = []
    pred_ret = {}
    ds_ret = {}
    rank_result = {}
    predictor = {}

    #for model in models:
    print('exp:',inspect.stack()[0][3],'model:', model, 
          'datamode:', get_modestr(datamode),'eval:', _exp_id )
    predictor[model] = load_model(prediction_length, model,
                                       trainid=trainid)

    ### create test dataset
    forecasts = longterm_predict(predictor[model],
                                   events_id[_test_event], prediction_length,freq, 
                                     oracle_mode=datamode,
                                     run_ts = _run_ts,
                                     test_cars=test_cars,
                                     half_moving_win= half_moving_win,
                                     train_ratio=train_ratio
                                    )

    #forecasts = eval_stint_rank(forecasts_et, prediction_length, 
    #                            global_start_offset[test_event])
            
    return forecasts


# In[14]:


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


# In[16]:
global_start_offset = {}
global_carids = {}
laptime_data = None
freq = "1min"
decode_carids = {}

years = ['2013','2014','2015','2016','2017','2018','2019']
events = [f'Indy500-{x}' for x in years]
events_id={key:idx for idx, key in enumerate(events)}
dbid = f'Indy500_{years[0]}_{years[-1]}_v9'

def init():
    global global_carids, laptime_data, global_start_offset, decode_carids

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
    with open(f'laptime_rank_timediff_pit-oracle-{dbid}.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        global_carids, laptime_data = pickle.load(f, encoding='latin1') 
    
    decode_carids={carid:carno for carno, carid in global_carids.items()}
    print(f'init: load dataset with {len(laptime_data)} races, {len(global_carids)} cars')


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

    rmse = mean_squared_error(df['pred_diff'].values , df['diff'].values)
    mae = mean_absolute_error(df['pred_diff'].values , df['diff'].values)
    r2 = r2_score(df['pred_diff'].values , df['diff'].values)

    #naive result
    n_correct = df[df['startrank']==df['endrank']]
    acc_naive = len(n_correct)/len(df)
    mae_naive = np.mean(np.abs(df['diff'].values))


    print(f'pred: acc={acc}, mae={mae},{mae1}, rmse={rmse},r2={r2}, acc_naive={acc_naive}, mae_naive={mae_naive}')
    
    return acc, mae, rmse, r2


# In[20]:
def mytest():

    savefile = f'stint-evluate-{_exp_id}-d{_dataset_id}-t{_test_event}-c{_context_ratio}_trim{_trim}.csv'
    if os.path.exists(savefile):
        print(f'{savefile} already exists, bye')

        retdf = pd.read_csv(savefile)
        return

    config = {'fulloracle':['oracle',MODE_ORACLE,False],
              'laponly':['oracle',MODE_ORACLE_LAPONLY,False],
              'notracklap':['oracle',MODE_NOTRACK + MODE_NOLAP,False],
              'fullpred':['oracle',MODE_PREDTRACK + MODE_PREDPIT,False],
              'curtrack':['oracle',MODE_TESTCURTRACK,False],
              'zerotrack':['oracle',MODE_TESTZERO,False],
              'predtrack':['oracle',MODE_PREDTRACK + MODE_ORACLE_TRACKONLY,False],
              'predpit':['oracle',MODE_PREDPIT + MODE_ORACLE_LAPONLY,False],
              'deepAR':['deepAR',MODE_ORACLE,False],
              'naive':['zero',MODE_ORACLE, True],
             }
    
    cols = ['runid','acc','mae', 'rmse', 'r2']
    
    result = []
    for modelname in config.keys():
        acc, mae, rmse, r2 = runtest(modelname, config[modelname][0],
                           config[modelname][1],config[modelname][2])
        
        result.append([modelname, acc, mae, rmse, r2])
        
    retd = pd.DataFrame(result,columns=cols)
    
    retd.to_csv(f'stint-evluate-{_exp_id}-d{_dataset_id}-t{_test_event}-c{_context_ratio}.csv', float_format='%.3f')
    
    return retd
    

# In[ ]:

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
    parser.add_option("--trim", dest="trim", type=int, default=2)

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


