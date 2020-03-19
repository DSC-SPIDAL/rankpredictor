#!/usr/bin/env python
# coding: utf-8

# ## Laptime2Rank-evaluate-fulltest-disturbance
# 
# based on: Laptime2Rank-evaluate-fulltest
# 
# rank prediction by laptime forecasting models
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

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import mxnet as mx
from mxnet import gluon
import pickle
import json
import random
from scipy import stats
from sklearn.metrics import mean_squared_error
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
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# In[2]:


import os
random.seed()
os.getcwd()
#GPUID = 1


# In[3]:


def load_data(event, year):
    inputfile = '../data/final/C_'+ event +'-' + year + '-final.csv'
    outputprefix = year +'-' + event + '-'
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


# In[4]:


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
COL_LAPTIME=0
COL_RANK=1
COL_TRACKSTATUS=2
COL_LAPSTATUS=3
COL_TIMEDIFF=4
COL_CAUTION_LAPS_INSTINT=5
COL_LAPS_INSTINT= 6
MODE_ORACLE = 0
MODE_NOLAP = 1
MODE_NOTRACK = 2
MODE_TESTZERO = 4
MODE_TESTCURTRACK = 8

MODE_PREDTRACK = 16
MODE_PREDPIT = 32

# disturbe analysis
MODE_DISTURB_CLEARTRACK = 64
MODE_DISTURB_ADJUSTTRACK = 128
MODE_DISTURB_ADJUSTPIT = 256


#MODE_STR={MODE_ORACLE:'oracle', MODE_NOLAP:'nolap',MODE_NOTRACK:'notrack',MODE_TEST:'test'}

def make_dataset(runs, prediction_length, freq, 
                       useeid = False,
                       run_ts=COL_LAPTIME, 
                       train_ratio = 0.8,
                       use_global_dict = True,
                       oracle_mode = MODE_ORACLE,
                       test_cars = [],
                       half_moving_win = True 
                ):
    """
    split the ts to train and test part by the ratio
    
    oracle_mode: false to simulate prediction in real by 
        set the covariates of track and lap status as nan in the testset
            
    
    """    
    start = pd.Timestamp("01-01-2019", freq=freq)  # can be different for each time series

    train_set = []
    test_set = []
    
    #select run
    if runs>=0:
        _laptime_data = [laptime_data[runs].copy()]
    else:
        _laptime_data = laptime_data.copy()
    
   
    #_data: eventid, carids, datalist[carnumbers, features, lapnumber]->[laptime, rank, track, lap]]
    for _data in _laptime_data:
        _train = []
        _test = []
        
        #statistics on the ts length
        ts_len = [ _entry.shape[1] for _entry in _data[2]]
        max_len = int(np.max(ts_len))
        train_len = int(np.max(ts_len) * train_ratio)
        
        print(f'====event:{events[_data[0]]}, train_len={train_len}, max_len={np.max(ts_len)}, min_len={np.min(ts_len)}')
                
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
            if ( totallen < train_len + prediction_length):
                print(f'a short ts: carid={_data[1][rowid]}，len={totallen}')
                continue                
            
            if use_global_dict:
                carno = _data[1][rowid]
                carid = global_carids[_data[1][rowid]]
            else:
                #simulation dataset, todo, fix the carids as decoder
                carno = rowid
                carid = rowid

            #eval on carids
            if test_cars and (carno not in test_cars):
                continue                
            
            if useeid:
                static_cat = [carid, _data[0]]    
            else:
                static_cat = [carid]    
                
            # selection of features
            if test_flag(oracle_mode, MODE_NOTRACK):                
                rec[COL_TRACKSTATUS, :] = 0
            if test_flag(oracle_mode, MODE_NOLAP):                
                rec[COL_LAPSTATUS, :] = 0
                
            # split and add to dataset record
            _train.append({'target': rec[run_ts,:train_len].astype(np.float32), 
                            'start': start, 
                            'feat_static_cat': static_cat,
                            'feat_dynamic_real': [rec[COL_TRACKSTATUS,:train_len],
                                   rec[COL_LAPSTATUS,:train_len]]
                          }
                          )
            
            # multiple test ts(rolling window as half of the prediction_length)
            test_rec_cnt = 0
            step = -int(prediction_length/2) if half_moving_win else -prediction_length
            for endpos in range(max_len, train_len+prediction_length, step):
                
                
                #check if enough for this ts
                if endpos > totallen:
                    continue
                        
                track_rec = rec[COL_TRACKSTATUS, :endpos].copy()
                lap_rec = rec[COL_LAPSTATUS, :endpos].copy()
                
                # test mode
                if test_flag(oracle_mode, MODE_TESTCURTRACK):
                    # since nan does not work, use cur-val instead
                    track_rec[-prediction_length:] = track_rec[-prediction_length - 1]
                    #track_rec[-prediction_length:] = random.randint(0,1)
                    #lap_rec[-prediction_length:] = lap_rec[-prediction_length - 1]
                    lap_rec[-prediction_length:] = 0
                elif test_flag(oracle_mode, MODE_TESTZERO):
                    #set prediction part as nan
                    #track_rec[-prediction_length:] = np.nan
                    #lap_rec[-prediction_length:] = np.nan
                    track_rec[-prediction_length:] = 0
                    lap_rec[-prediction_length:] = 0                    
                
                _test.append({'target': rec[run_ts,:endpos].astype(np.float32), 
                            'start': start, 
                            'feat_static_cat': static_cat,
                            'feat_dynamic_real': [track_rec,lap_rec]
                            #'feat_dynamic_real': [rec[COL_TRACKSTATUS,:endpos],
                            #       rec[COL_LAPSTATUS,:endpos]] 
                             }
                          )   
                test_rec_cnt += 1
            
            #add one ts
            print(f'carno:{carno}, totallen:{totallen}, nancount:{nan_count}, test_reccnt:{test_rec_cnt}')

        train_set.extend(_train)
        test_set.extend(_test)

    print(f'prediction_length:{prediction_length},train len:{len(train_set)}, test len:{len(test_set)}')
    
    train_ds = ListDataset(train_set, freq=freq)
    test_ds = ListDataset(test_set, freq=freq)    
    
    return train_ds, test_ds, train_set, test_set



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
    
def adjust_pit_model(lap_rec, endpos, prediction_length):
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
                       run_ts=COL_LAPTIME, 
                       test_event = 'Indy500',
                       test_cars = [],  
                       use_global_dict = True,
                       oracle_mode = MODE_ORACLE,
                       half_moving_win = 0,
                       train_ratio=0.8
                ):
    """
    split the ts to train and test part by the ratio
    
    input:
        oracle_mode: false to simulate prediction in real by 
                set the covariates of track and lap status as nan in the testset
        half_moving_win  ; extend to 0:-1 ,1:-1/2plen, 2:-plen
    
    """    
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
    
   
    #_data: eventid, carids, datalist[carnumbers, features, lapnumber]->[laptime, rank, track, lap]]
    for _data in _laptime_data:
        _train = []
        _test = []
        
        if events[_data[0]] == test_event:
            test_mode = True
        
        else:
            test_mode = False
            
            
        #statistics on the ts length
        ts_len = [ _entry.shape[1] for _entry in _data[2]]
        max_len = int(np.max(ts_len))
        train_len = int(np.max(ts_len) * train_ratio)
        
        
        print(f'====event:{events[_data[0]]}, train_len={train_len}, max_len={np.max(ts_len)}, min_len={np.min(ts_len)}')
                
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
            if ( totallen < train_len + prediction_length):
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
                
            # selection of features
            if test_flag(oracle_mode, MODE_NOTRACK):                
                rec[COL_TRACKSTATUS, :] = 0
            if test_flag(oracle_mode, MODE_NOLAP):                
                rec[COL_LAPSTATUS, :] = 0

            test_rec_cnt = 0
            if not test_mode:
                
                # all go to train set
                _train.append({'target': rec[run_ts,:].astype(np.float32), 
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
                
                context_len = train_len
                
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
                    
                    caution_laps_instint = int(rec[COL_CAUTION_LAPS_INSTINT, endpos -prediction_length - 1])
                    laps_instint = int(rec[COL_LAPS_INSTINT, endpos -prediction_length - 1])
                    

                    # test mode
                    if test_flag(oracle_mode, MODE_TESTCURTRACK):
                        # since nan does not work, use cur-val instead
                        track_rec[-prediction_length:] = track_rec[-prediction_length - 1]
                        #track_rec[-prediction_length:] = random.randint(0,1)
                        #lap_rec[-prediction_length:] = lap_rec[-prediction_length - 1]
                        lap_rec[-prediction_length:] = 0
                    elif test_flag(oracle_mode, MODE_PREDTRACK):
                        predrec = get_track_model(track_rec, endpos, prediction_length)
                        track_rec[-prediction_length:] = predrec
                        lap_rec[-prediction_length:] = 0
                    elif test_flag(oracle_mode, MODE_PREDPIT):
                        predrec = get_track_model(track_rec, endpos, prediction_length)
                        track_rec[-prediction_length:] = predrec
                        lap_rec[-prediction_length:] = get_pit_model(caution_laps_instint,
                                                                    laps_instint,prediction_length)
                    elif test_flag(oracle_mode, MODE_TESTZERO):
                        #set prediction part as nan
                        #track_rec[-prediction_length:] = np.nan
                        #lap_rec[-prediction_length:] = np.nan
                        track_rec[-prediction_length:] = 0
                        lap_rec[-prediction_length:] = 0        
                        
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
                        
                    if test_flag(oracle_mode, MODE_DISTURB_ADJUSTPIT):
                        # adjust the position of pit
                        if np.sum(lap_rec[-prediction_length:]) > 0:
                            adjustrec = adjust_pit_model(lap_rec, endpos, prediction_length)
                            lap_rec[-prediction_length:] = adjustrec
                        

                    _test.append({'target': rec[run_ts,:endpos].astype(np.float32), 
                                'start': start, 
                                'feat_static_cat': static_cat,
                                'feat_dynamic_real': [track_rec,lap_rec]
                                #'feat_dynamic_real': [rec[COL_TRACKSTATUS,:endpos],
                                #       rec[COL_LAPSTATUS,:endpos]] 
                                 }
                              )   
                    test_rec_cnt += 1
            
            #add one ts
            print(f'carno:{carno}, totallen:{totallen}, nancount:{nan_count}, test_reccnt:{test_rec_cnt}')

        train_set.extend(_train)
        test_set.extend(_test)

    print(f'train len:{len(train_set)}, test len:{len(test_set)}')
    
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

# In[5]:


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

        rootdir = f'../models/remote/laptime-{trainid}/'
        # deepAR-Oracle
        if model_name == 'curtrack':
            model=f'deepAR-Oracle-laptime-curtrack-indy-f1min-t{prediction_length}-e1000-r1_curtrack_t{prediction_length}'
            modeldir = rootdir + model
            print(f'predicting model={model_name}, plen={prediction_length}')
            predictor =  Predictor.deserialize(Path(modeldir))
            print(f'loading model...done!, ctx:{predictor.ctx}')
            tss, forecasts = predict(test_ds,predictor)
            pred_ret = [tss, forecasts]

        elif model_name == 'zerotrack':
            model=f'deepAR-Oracle-laptime-nolap-zerotrack-indy-f1min-t{prediction_length}-e1000-r1_zerotrack_t{prediction_length}'
            modeldir = rootdir + model
            print(f'predicting model={model_name}, plen={prediction_length}')
            predictor =  Predictor.deserialize(Path(modeldir))
            print(f'loading model...done!, ctx:{predictor.ctx}')
            tss, forecasts = predict(test_ds,predictor)
            pred_ret = [tss, forecasts]
            
        # deepAR-Oracle
        elif model_name == 'oracle':
            model=f'deepAR-Oracle-laptime-all-indy-f1min-t{prediction_length}-e1000-r1_oracle_t{prediction_length}'
            modeldir = rootdir + model
            print(f'predicting model={model_name}, plen={prediction_length}')
            predictor =  Predictor.deserialize(Path(modeldir))
            print(f'loading model...done!, ctx:{predictor.ctx}')
            tss, forecasts = predict(test_ds,predictor)
            pred_ret = [tss, forecasts]

        # deepAR
        elif model_name == 'deepAR':
            model=f'deepAR-laptime-all-indy-f1min-t{prediction_length}-e1000-r1_deepar_t{prediction_length}'
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


# In[6]:


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
        offset = start_offset[(start_offset['car_number']==carno)].elapsed_time.values[0] 
        #print('start_offset:', offset)

        # calc elapsed time
        prediction_len = forecasts[idx].samples.shape[1]
        if prediction_length != prediction_len:
            print('error: prediction_len does not match, {prediction_length}:{prediction_len}')
            return []
        
        forecast_laptime_mean = np.mean(forecasts[idx].samples, axis=0).reshape((prediction_len,1))
        #forecast_laptime_mean = np.median(forecasts[idx].samples, axis=0).reshape((prediction_len,1))

        laptime_array = tss[idx].values.copy()
        elapsed_time = np.cumsum(laptime_array) + offset

        laptime_array = tss[idx].values.copy()
        laptime_array[-prediction_len:] = forecast_laptime_mean 
        elapsed_time_hat = np.cumsum(laptime_array) + offset

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
            elapsed_time[0, carid, :] = forecasts_et[lap][carno][0]
            elapsed_time[1, carid, :] = forecasts_et[lap][carno][1]

        #calculate rank    
        idx = np.argsort(elapsed_time[0], axis=0)
        true_rank = np.argsort(idx, axis=0)

        idx = np.argsort(elapsed_time[1], axis=0)
        pred_rank = np.argsort(idx, axis=0)

        rank_ret.append([lap, elapsed_time, true_rank, pred_rank])
        
    return rank_ret,forecasts_et
   
def get_acc(rank_ret,prediction_length):   
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
        
    recnt = len(rank_ret)
    top1acc = top1acc *1.0/ (recnt*prediction_length)
    top1acc_farmost = top1acc_farmost *1.0/ recnt
    top5acc = top5acc *1.0/ (5*recnt*prediction_length)
    top5acc_farmost = top5acc_farmost *1.0/ (5*recnt)
    tau = tau/recnt
    rmse = rmse/recnt
    
        
    print(f'total:{len(rank_ret)}, prediction_length:{prediction_length}') 
    print('top1acc=', top1acc,
          'top1acc_farmost=', top1acc_farmost,
          'top5acc=', top5acc,
          'top5acc_farmost=', top5acc_farmost,
         )
    print('tau = ', tau,
         'rmse = ', rmse)
    
    return ((top1acc,top1acc_farmost,top5acc,top5acc_farmost,tau,rmse),
            (recnt*prediction_length,recnt,5*recnt*prediction_length,5*recnt,recnt,recnt))
    


# In[7]:


def run_exp(prediction_length, half_moving_win, train_ratio=0.8, trainid="r0.8",
                   test_event='Indy500', test_cars = []):           
    """
    dependency: test_event, test on one event only
    
    """
    
    models = ['oracle','deepAR','naive']
    #,'arima']
    retdf = []
    pred_ret = {}
    ds_ret = {}
    
    ### create test dataset
    train_ds, test_ds,_,_ = make_dataset_byevent(events_id[test_event], prediction_length,freq, 
                                         oracle_mode=MODE_ORACLE,
                                         run_ts = COL_LAPTIME,
                                         test_cars=test_cars,
                                         half_moving_win= half_moving_win,
                                         train_ratio=train_ratio)
    
    for model in models:
        print('model:', model)
        tss, forecasts = run_prediction_ex(test_ds, prediction_length, model,trainid=trainid)
        pred_ret[model] = [tss, forecasts]
        ds_ret[model] = test_ds

        rank_ret,_ = eval_rank(test_ds,tss,forecasts,prediction_length,global_start_offset[test_event])
        metrics = get_acc(rank_ret,prediction_length)
        ret = [model, prediction_length, half_moving_win,trainid]
        ret.extend(metrics[0])
        retdf.append(ret)
    
    # special model with test_ds
    models = ['curtrack']        
    train_ds, test_ds,_,_ = make_dataset_byevent(events_id[test_event], prediction_length,freq, 
                                         oracle_mode=MODE_TESTCURTRACK,
                                         run_ts = COL_LAPTIME,
                                         test_cars=test_cars,
                                         half_moving_win= half_moving_win,
                                         train_ratio=train_ratio)
    
    for model in models:
        print('model:', model)
        tss, forecasts = run_prediction_ex(test_ds, prediction_length, model,trainid=trainid)
        pred_ret[model] = [tss, forecasts]
        ds_ret[model] = test_ds
        
        rank_ret,_ = eval_rank(test_ds,tss,forecasts,prediction_length,global_start_offset[test_event])
        metrics = get_acc(rank_ret,prediction_length)
        ret = [model, prediction_length, half_moving_win,trainid]
        ret.extend(metrics[0])
        retdf.append(ret)

    # zerotrack
    models = ['zerotrack']        
    train_ds, test_ds,_,_ = make_dataset_byevent(events_id[test_event], prediction_length,freq, 
                                         oracle_mode=MODE_TESTZERO,
                                         run_ts = COL_LAPTIME,
                                         test_cars=test_cars,
                                         half_moving_win= half_moving_win,
                                         train_ratio=train_ratio)
    
    for model in models:
        print('model:', model)
        tss, forecasts = run_prediction_ex(test_ds, prediction_length, model,trainid=trainid)
        pred_ret[model] = [tss, forecasts]
        ds_ret[model] = test_ds
        
        rank_ret,_ = eval_rank(test_ds,tss,forecasts,prediction_length,global_start_offset[test_event])
        metrics = get_acc(rank_ret,prediction_length)
        ret = [model, prediction_length, half_moving_win,trainid]
        ret.extend(metrics[0])
        retdf.append(ret)
    
        
    return pred_ret, ds_ret, retdf



def run_exp_predtrack(prediction_length, half_moving_win, train_ratio=0.8, trainid="r0.8",
                                        test_event='Indy500', test_cars = []):

    """
    dependency: test_event, test on one event only
    
    """
    
    models = ['oracle','curtrack','zerotrack']
    #,'arima']
    retdf = []
    pred_ret = {}
    ds_ret = {}
    
    ### create test dataset
    train_ds, test_ds,_,_ = make_dataset_byevent(events_id[test_event], prediction_length,freq, 
                                         oracle_mode=MODE_PREDTRACK,
                                         run_ts = COL_LAPTIME,
                                         test_cars=test_cars,
                                         half_moving_win= half_moving_win,
                                         train_ratio=train_ratio)
    
    for model in models:
        print('model:', model)
        tss, forecasts = run_prediction_ex(test_ds, prediction_length, model,trainid=trainid)
        pred_ret[model] = [tss, forecasts]
        ds_ret[model] = test_ds

        rank_ret,_ = eval_rank(test_ds,tss,forecasts,prediction_length,global_start_offset[test_event])
        metrics = get_acc(rank_ret,prediction_length)
        ret = [model, prediction_length, half_moving_win,trainid]
        ret.extend(metrics[0])
        retdf.append(ret)
    
    return pred_ret, ds_ret, retdf

def run_exp_predpit(prediction_length, half_moving_win, train_ratio=0.8, trainid="r0.8",
                   test_event='Indy500', test_cars = []):
    """
    dependency: test_event, test on one event only
    
    """
    
    models = ['oracle','curtrack','zerotrack']
    #,'arima']
    retdf = []
    pred_ret = {}
    ds_ret = {}
    
    ### create test dataset
    train_ds, test_ds,_,_ = make_dataset_byevent(events_id[test_event], prediction_length,freq, 
                                         oracle_mode=MODE_PREDPIT,
                                         run_ts = COL_LAPTIME,
                                         test_cars=test_cars,
                                         half_moving_win= half_moving_win,
                                         train_ratio=train_ratio)
    
    for model in models:
        print('model:', model)
        tss, forecasts = run_prediction_ex(test_ds, prediction_length, model,trainid=trainid)
        pred_ret[model] = [tss, forecasts]
        ds_ret[model] = test_ds

        rank_ret,_ = eval_rank(test_ds,tss,forecasts,prediction_length,global_start_offset[test_event])
        metrics = get_acc(rank_ret,prediction_length)
        ret = [model, prediction_length, half_moving_win,trainid]
        ret.extend(metrics[0])
        retdf.append(ret)
    
    return pred_ret, ds_ret, retdf


# In[8]:


#MODE_DISTURB_CLEARTRACK = 64
#MODE_DISTURB_ADJUSTTRACK = 128
#MODE_DISTURB_ADJUSTPIT = 256
def run_exp_cleartrack(prediction_length, half_moving_win, train_ratio=0.8, trainid="r0.8",
                   test_event='Indy500', test_cars = []):
    """
    dependency: test_event, test on one event only
    
    """
    
    models = ['oracle']
    #,'arima']
    retdf = []
    pred_ret = {}
    ds_ret = {}
    
    ### create test dataset
    train_ds, test_ds,_,_ = make_dataset_byevent(events_id[test_event], prediction_length,freq, 
                                         oracle_mode=MODE_DISTURB_CLEARTRACK,
                                         run_ts = COL_LAPTIME,
                                         test_cars=test_cars,
                                         half_moving_win= half_moving_win,
                                         train_ratio=train_ratio)
    
    for model in models:
        print('model:', model)
        tss, forecasts = run_prediction_ex(test_ds, prediction_length, model,trainid=trainid)
        pred_ret[model] = [tss, forecasts]
        ds_ret[model] = test_ds

        rank_ret,_ = eval_rank(test_ds,tss,forecasts,prediction_length,global_start_offset[test_event])
        metrics = get_acc(rank_ret,prediction_length)
        ret = [model, prediction_length, half_moving_win,trainid]
        ret.extend(metrics[0])
        retdf.append(ret)
    
    return pred_ret, ds_ret, retdf

def run_exp_adjusttrack(prediction_length, half_moving_win, train_ratio=0.8, trainid="r0.8",
                   test_event='Indy500', test_cars = []):
    """
    dependency: test_event, test on one event only
    
    """
    
    models = ['oracle']
    #,'arima']
    retdf = []
    pred_ret = {}
    ds_ret = {}
    
    ### create test dataset
    train_ds, test_ds,_,_ = make_dataset_byevent(events_id[test_event], prediction_length,freq, 
                                         oracle_mode=MODE_DISTURB_CLEARTRACK + MODE_DISTURB_ADJUSTTRACK,
                                         run_ts = COL_LAPTIME,
                                         test_cars=test_cars,
                                         half_moving_win= half_moving_win,
                                         train_ratio=train_ratio)
    
    for model in models:
        print('model:', model)
        tss, forecasts = run_prediction_ex(test_ds, prediction_length, model,trainid=trainid)
        pred_ret[model] = [tss, forecasts]
        ds_ret[model] = test_ds

        rank_ret,_ = eval_rank(test_ds,tss,forecasts,prediction_length,global_start_offset[test_event])
        metrics = get_acc(rank_ret,prediction_length)
        ret = [model, prediction_length, half_moving_win,trainid]
        ret.extend(metrics[0])
        retdf.append(ret)
    
    return pred_ret, ds_ret, retdf

def run_exp_adjustpit(prediction_length, half_moving_win, train_ratio=0.8, trainid="r0.8",
                   test_event='Indy500', test_cars = []):
    """
    dependency: test_event, test on one event only
    
    """
    
    models = ['oracle']
    #,'arima']
    retdf = []
    pred_ret = {}
    ds_ret = {}
    
    ### create test dataset
    train_ds, test_ds,_,_ = make_dataset_byevent(events_id[test_event], prediction_length,freq, 
                                         oracle_mode=MODE_DISTURB_ADJUSTPIT,
                                         run_ts = COL_LAPTIME,
                                         test_cars=test_cars,
                                         half_moving_win= half_moving_win,
                                         train_ratio=train_ratio)
    
    for model in models:
        print('model:', model)
        tss, forecasts = run_prediction_ex(test_ds, prediction_length, model,trainid=trainid)
        pred_ret[model] = [tss, forecasts]
        ds_ret[model] = test_ds

        rank_ret,_ = eval_rank(test_ds,tss,forecasts,prediction_length,global_start_offset[test_event])
        metrics = get_acc(rank_ret,prediction_length)
        ret = [model, prediction_length, half_moving_win,trainid]
        ret.extend(metrics[0])
        retdf.append(ret)
    
    return pred_ret, ds_ret, retdf

def run_exp_adjustall(prediction_length, half_moving_win, train_ratio=0.8, trainid="r0.8",
                   test_event='Indy500', test_cars = []):
    """
    dependency: test_event, test on one event only
    
    """
    
    models = ['oracle']
    #,'arima']
    retdf = []
    pred_ret = {}
    ds_ret = {}
    
    ### create test dataset
    train_ds, test_ds,_,_ = make_dataset_byevent(events_id[test_event], prediction_length,freq, 
                                         oracle_mode=MODE_DISTURB_CLEARTRACK +MODE_DISTURB_ADJUSTPIT +MODE_DISTURB_ADJUSTTRACK,
                                         run_ts = COL_LAPTIME,
                                         test_cars=test_cars,
                                         half_moving_win= half_moving_win,
                                         train_ratio=train_ratio)
    
    for model in models:
        print('model:', model)
        tss, forecasts = run_prediction_ex(test_ds, prediction_length, model,trainid=trainid)
        pred_ret[model] = [tss, forecasts]
        ds_ret[model] = test_ds

        rank_ret,_ = eval_rank(test_ds,tss,forecasts,prediction_length,global_start_offset[test_event])
        metrics = get_acc(rank_ret,prediction_length)
        ret = [model, prediction_length, half_moving_win,trainid]
        ret.extend(metrics[0])
        retdf.append(ret)
    
    return pred_ret, ds_ret, retdf


# ### init

# In[ ]:





# In[9]:


#
# parameters
#
#year = '2017'
year = '2018'
#event = 'Toronto'
#https://www.racing-reference.info/season-stats/2018/O/#
events_totalmiles=[256,500,372,268,500,310]
events_laplen = [1.022,2.5,1.5,0.894,2.5,1.25]
events = ['Phoenix','Indy500','Texas','Iowa','Pocono','Gateway']
#events = ['Gateway']

#events = ['Indy500']
#events = ['Phoenix']
events_id={key:idx for idx, key in enumerate(events)}
#works for only one event


# In[10]:


stagedata = {}
global_start_offset = {}
global_carids = {}
traindata = None

cur_carid = 0
for event in events:
    #alldata, rankdata, acldata, flagdata
    stagedata[event] = load_data(event, year)
    
    alldata, rankdata, acldata = stagedata[event]
    #carlist = set(acldata['car_number'])
    #laplist = set(acldata['completed_laps'])
    #print('%s: carno=%d, lapnum=%d'%(event, len(carlist), len(laplist)))

    #offset
    global_start_offset[event] = rankdata[rankdata['completed_laps']==0][['car_number','elapsed_time']]
    
    #build the carid map
    #for car in carlist:
    #    if car not in global_carids:
    #        global_carids[car] = cur_carid
    #        cur_carid += 1


# In[11]:


# start from here
import pickle
with open('laptime_rank_timediff_pit-oracle-%s.pickle'%year, 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    global_carids, laptime_data = pickle.load(f, encoding='latin1')


# In[12]:


freq = "1min"
#decode global_carids
decode_carids={carid:carno for carno, carid in global_carids.items()}
    
#useeid = False
#interpolate = False
#ipstr = '-ip' if interpolate else '-noip'
#ipstr = '%s-%s'%('ip' if interpolate else 'noip', 'eid' if useeid else 'noeid')
#if useeid:
#    cardinality = [len(global_carids), len(laptime_data)]
#else:
#    cardinality = [len(global_carids)]


# In[13]:


def run_test(runs, plens, half, trainids, train_ratio, testfunc):
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
                 dataframe['model' , 'prediction_length', 'halfmode','trainid',
                         'top1acc','top1acc_farmost','top5acc','top5acc_farmost','tau','rmse',
                         'top1acc_std','top1acc_farmost_std','top5acc_std','top5acc_farmost_std','tau_std','rmse_std']
                          
    
    """
    if plens == [] or half == [] or trainids == []:
        print("error with empty settings")
        return

    allret = []
    for runid in range(runs):
        exp_data = []
        exp_result = []

        for halfmode in half:
            for plen in plens:
                for trainid in trainids:
                    print('='*20)
                    pred_ret, test_ds, metric_ret = testfunc(plen, halfmode, 
                                                            train_ratio=train_ratio,
                                                            trainid=trainid)

                    #save 
                    exp_data.append((pred_ret, test_ds))
                    exp_result.extend(metric_ret)

        #save result
        result = pd.DataFrame(exp_result, columns = ['model' , 'prediction_length', 'halfmode',
                                           'trainid',
                                           'top1acc','top1acc_farmost','top5acc',
                                           'top5acc_farmost','tau','rmse'])

        #result['runid'] = [runid for x in range(len(result))]
        allret.append(result)


    #final
    rowcnt = len(allret[0])
    metrics = np.empty((runs, rowcnt, 6))
    for runid, ret in enumerate(allret):
        metrics[runid, :,:] = ret[['top1acc','top1acc_farmost','top5acc',
                                           'top5acc_farmost','tau','rmse']].values


    #average
    averagemat = np.mean(metrics[:,:,:], axis=0)
    stdmat = np.std(metrics[:,:,:], axis=0)
    result = allret[0][['model' , 'prediction_length', 'halfmode',
                                           'trainid',
                                   'top1acc','top1acc_farmost','top5acc',
                                           'top5acc_farmost','tau','rmse']].values
    result[:,4:] = averagemat

    dfret = pd.DataFrame(result, columns = ['model' , 'prediction_length', 'halfmode',
                                       'trainid',
                                       'top1acc','top1acc_farmost','top5acc',
                                       'top5acc_farmost','tau','rmse'])
    dfstd = pd.DataFrame(stdmat, columns = ['top1acc_std','top1acc_farmost_std','top5acc_std',
                                       'top5acc_farmost_std','tau_std','rmse_std'])
    dfret = pd.concat([dfret, dfstd], axis=1)

    dfret.to_csv(f'laptime2rank-evaluate-indy500-{exp_id}-result.csv', float_format='%.3f')

    return dfret


# ### loop test

# In[14]:


test_result = {}


# In[ ]:


#half=[True, False]
#plens=[2,5,10,20,30]
plens=[2,5,10]
half=[0]
trainids = ["indy500-r0.2","indy500-r0.4","indy500"]
#trainids = ["r0.5","r0.6"]
runs = 5
train_ratio=0.4

#trainids = ["indy500"]
#runs = 1
#plens=[2]

#testfunc = run_exp_cleartrack
#exp_id=f'mean-splitbyevent-adjustcleartrack-r{runs}-t{train_ratio}'
#test_result[exp_id] = run_test(runs, plens, half, trainids, train_ratio, testfunc)
#
#testfunc = run_exp_adjusttrack
#exp_id=f'mean-splitbyevent-adjusttrack-r{runs}-t{train_ratio}'
#test_result[exp_id] = run_test(runs, plens, half, trainids, train_ratio, testfunc)

testfunc = run_exp_adjustpit
exp_id=f'mean-splitbyevent-adjustpit-r{runs}-t{train_ratio}'
test_result[exp_id] = run_test(runs, plens, half, trainids, train_ratio, testfunc)

testfunc = run_exp_adjustall
exp_id=f'mean-splitbyevent-adjustall-r{runs}-t{train_ratio}'
test_result[exp_id] = run_test(runs, plens, half, trainids, train_ratio, testfunc)

#testfunc = run_exp
#exp_id=f'mean-splitbyevent-baseline-r{runs}-t{train_ratio}'
#test_result[exp_id] = run_test(runs, plens, half, trainids, train_ratio, testfunc)


# In[ ]:


#half=[True, False]
#plens=[2,5,10,20,30]
plens=[2,5,10]
half=[0]
#trainids = ["indy500-r0.2","indy500-r0.4","indy500"]
trainids = ["r0.5","r0.6","r0.7"]
runs = 5
train_ratio=0.7


# ### test

# In[ ]:





# In[ ]:




