#!/usr/bin/env python
# coding: utf-8

"""
RankNet QuickTest goes through the following steps

     makedb laptime
     makedb gluonts
     train model
     evaluate model
     draw figures

version 0.4
supported features:
    forecast_mode: shortterm, stint
    trainmodel  : deepAR  , deepARW-Oracle, deepAR-multi
    testmodel    : standard, oracle,pitmodel, joint

Usage: RankNet-QuickTest.py <configfile> [options]

    options overwrite the configurations for quick experiments needs, include:
    _forecast_mode      ;
    trainmodel          ;
    testmodel           ;
    _joint_train        ; False/True
    loopcnt             ; 100/2
    _pitmodel_bias      ; 0/2,4
    year                ; 2018/2019
    _test_event         ; Indy500-2018, Indy500-2019

"""

import logging
from optparse import OptionParser

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import random
import mxnet as mx
from mxnet import gluon
import pickle
import json
import copy
from gluonts.dataset.common import ListDataset
from gluonts.dataset.util import to_pandas
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from pathlib import Path
import configparser

from gluonts.model.deep_factor import DeepFactorEstimator
from gluonts.model.deepstate import DeepStateEstimator
from gluonts.trainer import Trainer
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator, MultivariateEvaluator
from gluonts.model.predictor import Predictor
from gluonts.model.prophet import ProphetPredictor
from gluonts.model.r_forecast import RForecastPredictor
from gluonts.dataset.util import to_pandas

from gluonts.distribution.neg_binomial import NegativeBinomialOutput
from gluonts.distribution.student_t import StudentTOutput
from gluonts.distribution.multivariate_gaussian import MultivariateGaussianOutput

from indycar.model.NaivePredictor import NaivePredictor
from indycar.model.deeparw import DeepARWeightEstimator
from indycar.model.deepar import DeepAREstimator
from indycar.model.transformerw import TransformerWeightedEstimator
from indycar.model.transformerf import TransformerFullLossEstimator
from indycar.model.transformerwf import TransformerWeightedFullLossEstimator
from indycar.model.transformerwfm import TransformerWeightedFullLossMaskedEstimator

#import indycar.model.stint_simulator_shortterm_pitmodel as stint
import indycar.model.quicktest_simulator as stint
#from indycar.model.global_variables import _hi
import indycar.model.global_variables as gvar

from gluonts.model.transformer import TransformerEstimator


logger = logging.getLogger(__name__)


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
    print('cars:', carnumber)
    print('#cars=', len(carnumber))
   
    # faster solution , uni_ds already sorted by car_number and lap
    uni_ds['rank_diff'] = uni_ds['rank'].diff()
    mask = uni_ds.car_number != uni_ds.car_number.shift(1)
    uni_ds['rank_diff'][mask] = 0
    
    uni_ds['time_diff'] = uni_ds['elapsed_time'].diff()
    mask = uni_ds.car_number != uni_ds.car_number.shift(1)
    uni_ds['time_diff'][mask] = 0
    
    #df = uni_ds[['car_number','completed_laps','rank','elapsed_time','rank_diff','time_diff']]
    #df = uni_ds[['car_number','completed_laps','rank',
    #             'rank_diff','time_diff',"current_status", "track_status", "lap_status",'elapsed_time']]
    
    df = uni_ds[['car_number','completed_laps','time_diff','rank','track_status', 'lap_status','elapsed_time']]
    
    return df

def make_lapstatus_data(dataset):
    final_lap = max(dataset.completed_laps)
    total_laps = final_lap + 1

    # get records for the cars that finish the race
    completed_car_numbers= dataset[dataset.completed_laps == final_lap].car_number.values
    completed_car_count = len(completed_car_numbers)

    print('count of completed cars:', completed_car_count)
    print('completed cars:', completed_car_numbers)
    
    #pick up one of them
    onecar = dataset[dataset['car_number']==completed_car_numbers[0]]
    onecar = onecar.drop_duplicates(subset=['car_number', 'completed_laps'], keep='first')
    return onecar[['completed_laps','track_status']]

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

    print('count of completed cars:', completed_car_count)
    print('completed cars:', completed_car_numbers)

    #make a copy
    alldata = dataset.copy()
    dataset = dataset[dataset['car_number'].isin(completed_car_numbers)]
    rankdata = alldata.rename_axis('MyIdx').sort_values(by=['elapsed_time','MyIdx'], ascending=True)
    rankdata = rankdata.drop_duplicates(subset=['car_number', 'completed_laps'], keep='first')
    
    cldata = make_cl_data(dataset)
    flagdata = make_lapstatus_data(dataset)
    acldata = make_cl_data(alldata)

    return alldata, rankdata, acldata, flagdata

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

def get_lap2nextpit(lap_status, maxlap):
    """
    input:
        lapstatus  ; array of 0/1 indicating pitstops for each lap, nan means incomplete race
        maxlap     ; the max lap number of the race
    output:
        lap2nextpit ; array of the lap gap to the next pit for each lap
    
    """
    
    #pitstops = np.where(lap_status==1)[0]
                    
    pitstops = list(np.where(lap_status==1)[0])
    #if not len(lap_status) < maxlap:
    nans, x= nan_helper(lap_status)
    nan_count = np.sum(nans)      
    if nan_count == 0:
        #complete cars
        # the last stint, to the end
        pitstops.append(maxlap)
    
    lap2nextpit = np.zeros_like(lap_status)
    lap2nextpit[:] = np.nan
    
    #guard
    if len(pitstops)==0:
        return lap2nextpit
    
    idx = 0
    for lap in range(len(lap_status)):
        if lap < pitstops[idx]:
            lap2nextpit[lap] = pitstops[idx] - lap
        else:
            idx += 1
            if idx < len(pitstops):
                lap2nextpit[lap] = pitstops[idx] - lap
            else:
                break
            
    return lap2nextpit

def get_lapdata(acldata):
    """
    input:
        acldata['car_number','completed_laps','time_diff','rank','track_status', 'lap_status','elapsed_time']
    
        timediff: [car_number, completed_laps] -> elapsed time diff to leader
    output:
        lapdata = acldata[['car_number','completed_laps',
                           'time_diff','rank','track_status', 'lap_status','time_behind']].to_numpy()
    """
    COL_COMPLETED_LAPS = 1
    COL_ELAPSED_TIME = 6
    
    maxlap = np.max(acldata['completed_laps'].values)
    #'car_number','completed_laps','time_diff','rank','track_status', 'lap_status','time_behind'
    time_behind = []
    
    for lap in range(1, maxlap+1):
        this_lap = acldata[acldata['completed_laps']==lap][
            ['car_number','completed_laps','time_diff','rank',
             'track_status', 'lap_status','elapsed_time']].values
        
        if len(this_lap) == 0:
            continue

        min_elapsed_time = np.nanmin(this_lap[:,COL_ELAPSED_TIME].astype(np.float))
        #print(f'lap:{lap}, min_elapsed_time:{min_elapsed_time}')
        
        for row in this_lap:
            car_number = int(row[0])
            time_diff = row[2]
            rank = row[3]
            track_status = row[4]
            lap_status = row[5]
            
            timebehind = float(row[COL_ELAPSED_TIME]) - min_elapsed_time
            #
            time_behind.append([car_number, lap, time_diff,rank,track_status, lap_status,
                                timebehind, float(row[COL_ELAPSED_TIME])])
    
    #return
    lapdata = np.array(time_behind)
    return lapdata



# features: laptime, rank, track_status, lap_status, timediff
LAPTIME = 0
RANK = 1
TRACK_STATUS = 2
LAP_STATUS = 3
TIME_BEHIND = 4
CAUTION_LAPS_INSTINT = 5 
LAPS_INSTINT = 6
ELAPSED_TIME = 7
LAP2NEXTPIT = 8

_featureCnt = 9
        
def get_laptime_dataset(stagedata, inlap_status = 0):
    """
    #add caution_laps_instint, laps_instint
    
    input: (alldata, rankdata, acldata, flagdata)
    output: laptime & rank data
    
    [(
    eventid,
    carids : rowid -> carno,
    datalist: #car_number x features x #totallaps (padded by Nan)
        entry: [[laptime, rank, track_status, lap_status,
                caution_laps_instint, laps_instint]]
    )]
    """
    laptime_data = []
    for event in stagedata.keys():
        
        print(f'start event: {event}')
        
        laptime_rec = []
        eventid = gvar.events_id[event]
        
        alldata, rankdata, acldata, flagdata = stagedata[event]
        carlist = set(acldata['car_number'])
        laplist = set(acldata['completed_laps'])
        totalcars = len(carlist)
        #totallaps = len(laplist)
        totallaps = max(laplist) + 1
        print('totallaps:', event, totallaps, len(laplist))
        


        #carnumber -> carid
        carids={key:idx for idx, key in enumerate(carlist)}
        decode_carids={idx:key for idx, key in enumerate(carlist)}

        #init
        lap_instint = {carids[x]:0 for x in carlist}
        caution_instint = {carids[x]:0 for x in carlist}        
        
        #array: car_number x lap
        #laptime = np.zeros((totalcars, totallaps-1))
        #rank = np.zeros((totalcars, totallaps-1))
        laptime = np.empty((totalcars, totallaps-1))
        rank = np.empty((totalcars, totallaps-1))
        laptime[:] = np.NaN
        rank[:] = np.NaN
        

        datalist = np.empty((totalcars, _featureCnt, totallaps-1))
        datalist[:] = np.NaN
        
        #lapdata = acldata[['car_number','completed_laps',
        #                   'time_diff','rank','track_status', 'lap_status','elapsed_time']].to_numpy()
        
        #'car_number','completed_laps','time_diff','rank','track_status', 'lap_status','time_behind'
        lapdata = get_lapdata(acldata)
        
        
        for row in lapdata:
            #completed_laps
            if int(row[1]) == 0:
                continue
                
            #add to data array
            car_number = carids[int(row[0])]
            completed_laps = int(row[1])-1
            time_diff = float(row[2])
            rank = int(row[3])
            track_status = 1 if row[4]=='Y' else 0
            lap_status = 1 if row[5]=='P' else 0
            time_behind = float(row[6])
            
            datalist[car_number, LAPTIME, completed_laps] = time_diff
            datalist[car_number, RANK, completed_laps] = rank
            datalist[car_number, TRACK_STATUS, completed_laps] = track_status
            datalist[car_number, LAP_STATUS, completed_laps] = lap_status
            datalist[car_number, TIME_BEHIND, completed_laps] = time_behind

            datalist[car_number, ELAPSED_TIME, completed_laps] = float(row[7])

            
            #stint status
            if track_status == 1:
                caution_instint[car_number] += 1
            lap_instint[car_number] += 1
            if lap_status == 1:
                #new stint
                lap_instint[car_number] = 0
                caution_instint[car_number] = 0
                
                # add inlap feature into lap_Status
                # set the previous lap to inlap status
                
                # what does it mean?
                
                if (inlap_status!=0):
                    if inlap_status == 1:
                        # set the previous lap of 'P'
                        if completed_laps > 0:
                            #datalist[car_number, LAP_STATUS, completed_laps-1] = INLAP_STATUS
                            datalist[car_number, LAP_STATUS, completed_laps-1] = 1
                    else:
                        # set the next lap of 'P'
                        if completed_laps +1 < totallaps:
                            #datalist[car_number, LAP_STATUS, completed_laps-1] = INLAP_STATUS
                            datalist[car_number, LAP_STATUS, completed_laps + 1] = 1
                
            
            datalist[car_number, LAPS_INSTINT, completed_laps] = lap_instint[car_number]
            datalist[car_number, CAUTION_LAPS_INSTINT, completed_laps] = caution_instint[car_number]
                

                
        #update lap2nextpit in datalist
        for caridx in range(datalist.shape[0]):
            lap_status = datalist[caridx, LAP_STATUS, :]
            #pit status
            lap2nextpit = get_lap2nextpit(lap_status, totallaps-1)
            datalist[caridx, LAP2NEXTPIT, :] = lap2nextpit        
                
        #add one record
        laptime_data.append([eventid, decode_carids, datalist])
        # push this event into stage dataframe
        print('event=%s, records=%s'%(event, datalist.shape))
        
    
    return laptime_data


# In[ ]:


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
COL_ELAPSED_TIME= 7
COL_LAP2NEXTPIT = 8
#_featureCnt = 9

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


MODE_ORACLE = 0
MODE_NOLAP = 1
MODE_NOTRACK = 2
MODE_TESTZERO = 4
MODE_TESTCURTRACK = 8
#MODE_STR={MODE_ORACLE:'oracle', MODE_NOLAP:'nolap',MODE_NOTRACK:'notrack',MODE_TEST:'test'}

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


def prepare_laptimedata(laptime_data, 
                        prediction_length, freq, 
                       test_event = 'Indy500-2018',
                       train_ratio=0.8,
                       context_ratio = 0.,
                       shift_len = -1):
    """
    prepare the laptime data for training
    
    1. remove short ts
    2. rerank the tss
    3. create new features
    
    input: 
        laptime_data   ; global var
    output:
        data  ; new representation of laptime_data
    
    """
    _laptime_data = laptime_data.copy()
    
    test_eventid = gvar.events_id[test_event]
    train_events = gvar._train_events

    run_ts = COL_RANK
    
    # check shift len
    if shift_len < 0:
        shift_len = prediction_length
    print('prepare_laptimedata shift len:', shift_len)
    
    #_data: eventid, carids, datalist[carnumbers, features, lapnumber]->[laptime, rank, track, lap]]
    new_data = []
    for _data in _laptime_data:
        #if events[_data[0]] == test_event:
        test_mode = False
        if _data[0] == test_eventid:
            test_mode = True
        #elif _data[0] in train_events:
        #    test_mode = False
        #else:
        #    #skip this event
        #    print('skip this event:', _data[0])
        #    continue
        
        #statistics on the ts length
        ts_len = [ _entry.shape[1] for _entry in _data[2]]
        train_len = int(np.max(ts_len) * train_ratio)
        if train_len == 0:
            #use global train_len
            train_len = gvar._train_len if not test_mode else gvar._test_train_len
        
        if context_ratio != 0.:
            # add this part to train set
            context_len = int(np.max(ts_len) * context_ratio)
        else:    
            context_len = prediction_length*2
        if context_len < 10:
            context_len = 10
        
        print(f'before ====event:{gvar.events[_data[0]]}, prediction_len={prediction_length},train_len={train_len}, max_len={np.max(ts_len)}, min_len={np.min(ts_len)},context_len={context_len}')

        #rerank due to short ts removed
        #if run_ts == COL_RANK and dorerank == True:
        if True:
            sel_rows = []
            
            # use to check the dimension of features
            input_feature_cnt = _data[2].shape[1]
            if input_feature_cnt < COL_LASTFEATURE + 1:
                print('create new features mode, feature_cnt:', input_feature_cnt)
            else:
                print('update features mode, feature_cnt:', input_feature_cnt)
            
            for rowid in range(_data[2].shape[0]):
                # rec[features, lapnumber] -> [laptime, rank, track_status, lap_status,timediff]]
                rec = _data[2][rowid].copy()
                #remove nan(only tails)
                nans, x= nan_helper(rec[run_ts,:])
                nan_count = np.sum(nans)             
                rec = rec[:, ~np.isnan(rec[run_ts,:])]
                
                totallen = rec.shape[1]
                if ( totallen < train_len + prediction_length):
                    print(f'rerank a short ts: carid={_data[1][rowid]}，len={totallen}')
                    continue 
                else:
                    sel_rows.append(rowid)
                    
            #get selected matrix
            sel_idx = np.array(sel_rows)
            selmat = _data[2][sel_idx]
            
            # check the format of _data
            #ipdb.set_trace()
            
            mask = np.isnan(selmat[:,COL_RANK,:])
            
            idx = np.argsort(selmat[:,COL_RANK,:], axis=0)
            true_rank = np.argsort(idx, axis=0).astype(np.float)
            true_rank[mask] = np.nan
            
            if test_mode:
                #
                # for historical code mismatch, simulation does not run rerank
                #
                _data[2][sel_idx,COL_RANK,:] = true_rank + 1
            else:
                _data[2][sel_idx,COL_RANK,:] = true_rank
            
            # update the carno dict
            new_carids = {}
            for rowid in range(len(sel_idx)):
                carid = sel_idx[rowid]
                carno = _data[1][carid]
                new_carids[rowid] = carno

                
            # add new features
            # add leaderPitCnt
            if _data[0]==0:
                verbose = True
            else:
                verbose = False
                

            dest_col = -1 if input_feature_cnt < COL_LASTFEATURE + 1 else COL_LEADER_PITCNT
            data2_intermediate = add_leader_cnt(_data[2][sel_idx], shift_len = shift_len, dest_col=dest_col, verbose = verbose)
            
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
            
            # leader_pitcnt can not be shift, target leaking, just do not use it
            dest_col = -1 if input_feature_cnt < COL_LASTFEATURE + 1 else COL_SHIFT_LEADER_PITCNT
            data2_intermediate = add_shift_feature(data2_intermediate, dest_col=dest_col,
                                                   shift_col=COL_LEADER_PITCNT, shift_len = shift_len)
            
            dest_col = -1 if input_feature_cnt < COL_LASTFEATURE + 1 else COL_SHIFT_TOTAL_PITCNT
            data2_intermediate = add_shift_feature(data2_intermediate, dest_col=dest_col,
                                                   shift_col=COL_TOTAL_PITCNT, shift_len = shift_len)
            
            # final
            data2_newfeature = data2_intermediate
            
        new_data.append([_data[0], new_carids, data2_newfeature])
        
    return new_data


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

def make_dataset_byevent(_laptime_data, 
                       prediction_length, freq, 
                       useeid = False,
                       run_ts=COL_LAPTIME, 
                       test_event = 'Indy500-2018',
                       use_global_dict = True,
                       oracle_mode = MODE_ORACLE,
                       half_moving_win = True,
                       train_ratio=0.8,
                       log_transform = False,
                       context_ratio = 0.,
                       dorerank = True,
                       joint_train = 0,
                       test_cars = []  
                ):
    """
    split the ts to train and test part by the ratio
    
    oracle_mode: false to simulate prediction in real by 
        set the covariates of track and lap status as nan in the testset
            
    
    """    
    #global setting
    feature_mode = gvar._feature_mode
    
    start = pd.Timestamp("01-01-2019", freq=freq)  # can be different for each time series

    train_set = []
    test_set = []
    
    
    totalTSCnt = 0
    totalTSLen = 0
    test_eventid = gvar.events_id[test_event]
    train_events = gvar._train_events
    
    #_data: eventid, carids, datalist[carnumbers, features, lapnumber]->[laptime, rank, track, lap]]
    for _data in _laptime_data:
        _train = []
        _test = []
        
        if _data[0] == test_eventid:
            test_mode = True
        elif _data[0] in train_events:
            test_mode = False
        else:
            #skip this event
            print('skip this event:', _data[0])
            continue
            
        #statistics on the ts length
        ts_len = [ _entry.shape[1] for _entry in _data[2]]
        train_len = int(np.max(ts_len) * train_ratio)
        if train_len == 0:
            #use global train_len
            train_len = gvar._train_len if not test_mode else gvar._test_train_len
        
        if context_ratio != 0.:
            # add this part to train set
            context_len = int(np.max(ts_len) * context_ratio)
        else:    
            context_len = prediction_length*2
        if context_len < 10:
            context_len = 10
        
        print(f'after ====event:{gvar.events[_data[0]]}, prediction_len={prediction_length},train_len={train_len}, max_len={np.max(ts_len)}, min_len={np.min(ts_len)},context_len={context_len}')

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
            
            totalTSCnt += 1
            totalTSLen += totallen
            
            if ( totallen < train_len + prediction_length):
                print(f'a short ts: carid={_data[1][rowid]}，len={totallen}')
                continue                
            
            if use_global_dict:
                carno = _data[1][rowid]
                carid = gvar.global_carids[_data[1][rowid]]
            else:
                #simulation dataset, todo, fix the carids as decoder
                carno = rowid
                carid = rowid
                
            #check carno in test_cars, testmode only
            if len(test_cars)>0 and carno not in test_cars:
                continue
                
            if useeid:
                static_cat = [carid, _data[0]]    
            else:
                static_cat = [carid]    
                
            #first, get target a copy    
            # target can be COL_XXSTATUS
            if joint_train:
                target_cols = [run_ts, COL_LAPSTATUS]
                target_val = rec[target_cols].copy().astype(np.float32)                
            else:
                target_val = rec[run_ts,:].copy().astype(np.float32)
                
            if log_transform:
                target_val = np.log(target_val + 1.0)
            
            # selection of features
            if test_flag(oracle_mode, MODE_NOTRACK):                
                rec[COL_TRACKSTATUS, :] = 0
            if test_flag(oracle_mode, MODE_NOLAP):                
                rec[COL_LAPSTATUS, :] = 0

            test_rec_cnt = 0
            if not test_mode:
                # all go to train set
                real_features = get_real_features(feature_mode, rec, -1)
                
                _train.append({'target': target_val, 
                            'start': start, 
                            'feat_static_cat': static_cat,
                            'feat_dynamic_real': real_features
                          })
                    
            else:
                # reset train_len
                if context_ratio != 0.:
                    # all go to train set
                    #add [0, context_len] to train set 
                    # all go to train set
                    if joint_train:
                        _train.append({'target': target_val[:,:context_len],  
                                'start': start, 
                                'feat_static_cat': static_cat,
                                'feat_dynamic_real': get_real_features(feature_mode, rec, context_len)
                              })
                    else:
                        _train.append({'target': target_val[:context_len],  
                                'start': start, 
                                'feat_static_cat': static_cat,
                                'feat_dynamic_real': get_real_features(feature_mode, rec, context_len)
                              })
                              
                # testset
                # multiple test ts(rolling window as half of the prediction_length)
                #step = -int(prediction_length/2) if half_moving_win else -prediction_length
                step = -1
                for endpos in range(totallen, context_len+prediction_length, 
                                    step):

                    track_rec = rec[COL_TRACKSTATUS, :endpos].copy()
                    lap_rec = rec[COL_LAPSTATUS, :endpos].copy()
                    pitage_rec = rec[COL_LAPS_INSTINT, :endpos].copy()

                    real_features = get_real_features(feature_mode, rec, endpos)
                    
                    if joint_train:                    
                        _test.append({'target': target_val[:,:endpos], 
                            'start': start, 
                            'feat_static_cat': static_cat,
                            'feat_dynamic_real': real_features
                             })
                        
                    else:
                        _test.append({'target': target_val[:endpos], 
                            'start': start, 
                            'feat_static_cat': static_cat,
                            'feat_dynamic_real': real_features
                             })
                                 
                    test_rec_cnt += 1
            
            #check feature cnt
            featureCnt = len(real_features)
            
            #add one ts
            print(f'carno:{carno}, totallen:{totallen}, nancount:{nan_count}, test_reccnt:{test_rec_cnt},featureCnt:{featureCnt}')

        train_set.extend(_train)
        test_set.extend(_test)

    print(f'train len:{len(train_set)}, test len:{len(test_set)}, totsl TsCnt:{totalTSCnt}, total ts len:{totalTSLen}')
    
    train_ds = ListDataset(train_set, freq=freq,one_dim_target= False if joint_train else True)
    test_ds = ListDataset(test_set, freq=freq,one_dim_target= False if joint_train else True)    
    
    return train_ds, test_ds, train_set, test_set


# In[ ]:


def init_estimator(model, gpuid, epochs=100, batch_size = 32, 
        target_dim = 3, distr_output = None, use_feat_static = True):
    
    if int(gpuid) < 0:
        ctx = "cpu"
    else:
        ctx = "gpu(%s)"%gpuid

    #global vars
    prediction_length = gvar.prediction_length
    context_length = gvar.context_length
    freq = gvar.freq

    if model == 'deepAR':
        if use_feat_static:
            estimator = DeepAREstimator(
                prediction_length=prediction_length,
                context_length= context_length,
                use_feat_static_cat=True,
                cardinality=cardinality,
                use_feat_dynamic_real=False,
                distr_output = distr_output,
                freq=freq,
                lags_seq=gvar._lags_seq,
                trainer=Trainer(ctx=ctx, 
                                batch_size = batch_size,
                                epochs=epochs, 
                                learning_rate=1e-3, 
                                num_batches_per_epoch=100
                               )
            )
        else:
            estimator = DeepAREstimator(
                prediction_length=prediction_length,
                context_length= context_length,
                use_feat_static_cat=False,
                #cardinality=cardinality,
                use_feat_dynamic_real=False,
                distr_output = distr_output,
                freq=freq,
                lags_seq=gvar._lags_seq,
                trainer=Trainer(ctx=ctx, 
                                batch_size = batch_size,
                                epochs=epochs, 
                                learning_rate=1e-3, 
                                num_batches_per_epoch=100
                               )
        )
        
    elif model == 'deepAR-Oracle':

        if use_feat_static:
            estimator = DeepAREstimator(
                prediction_length=prediction_length,
                context_length= context_length,
                use_feat_static_cat=use_feat_static,
                cardinality=cardinality,
                use_feat_dynamic_real=True,
                distr_output = distr_output,
                freq=freq,
                lags_seq=gvar._lags_seq,
                trainer=Trainer(ctx=ctx, 
                                batch_size = batch_size,
                                epochs=epochs, 
                                learning_rate=1e-3, 
                                num_batches_per_epoch=100
                               )
                )
        else:
            estimator = DeepAREstimator(
                prediction_length=prediction_length,
                context_length= context_length,
                use_feat_static_cat=use_feat_static,
                #cardinality=cardinality,
                use_feat_dynamic_real=True,
                distr_output = distr_output,
                freq=freq,
                lags_seq=gvar._lags_seq,
                trainer=Trainer(ctx=ctx, 
                                batch_size = batch_size,
                                epochs=epochs, 
                                learning_rate=1e-3, 
                                num_batches_per_epoch=100
                               )
                )
    elif model == 'deepARW-Oracle' or model == 'RankNet':

        if use_feat_static:
            estimator = DeepARWeightEstimator(
                prediction_length=prediction_length,
                context_length= context_length,
                use_feat_static_cat=use_feat_static,
                cardinality=cardinality,
                use_feat_dynamic_real=True,
                distr_output = distr_output,
                freq=freq,
                lags_seq=gvar._lags_seq,
                weight_coef=gvar._weight_coef,
                trainer=Trainer(ctx=ctx, 
                                batch_size = batch_size,
                                epochs=epochs, 
                                learning_rate=1e-3, 
                                #hybridize=False,
                                num_batches_per_epoch=100
                               )
                )
        else:
            estimator = DeepARWeightEstimator(
                prediction_length=prediction_length,
                context_length= context_length,
                use_feat_static_cat=use_feat_static,
                #cardinality=cardinality,
                use_feat_dynamic_real=True,
                distr_output = distr_output,
                freq=freq,
                lags_seq=gvar._lags_seq,
                weight_coef=gvar._weight_coef,
                trainer=Trainer(ctx=ctx, 
                                batch_size = batch_size,
                                epochs=epochs, 
                                learning_rate=1e-3, 
                                #hybridize=False,
                                num_batches_per_epoch=100
                               )
                )
    elif model == 'Transformer':

        if use_feat_static:
            estimator = TransformerEstimator(
                prediction_length=prediction_length,
                context_length= context_length,
                use_feat_static_cat=use_feat_static,
                cardinality=cardinality,
                use_feat_dynamic_real=False,
                distr_output = distr_output,
                freq=freq,
                lags_seq=gvar._lags_seq,
                trainer=Trainer(ctx=ctx, 
                                batch_size = batch_size,
                                epochs=epochs, 
                                learning_rate=1e-3, 
                                #hybridize=False,
                                num_batches_per_epoch=100
                               )
                )
        else:
            estimator = TransformerEstimator(
                prediction_length=prediction_length,
                context_length= context_length,
                use_feat_static_cat=use_feat_static,
                #cardinality=cardinality,
                use_feat_dynamic_real=False,
                distr_output = distr_output,
                freq=freq,
                lags_seq=gvar._lags_seq,
                trainer=Trainer(ctx=ctx, 
                                batch_size = batch_size,
                                epochs=epochs, 
                                learning_rate=1e-3, 
                                #hybridize=False,
                                num_batches_per_epoch=100
                               )
                )
    elif model == 'Transformer-Oracle':

        if use_feat_static:
            estimator = TransformerEstimator(
                prediction_length=prediction_length,
                context_length= context_length,
                use_feat_static_cat=use_feat_static,
                cardinality=cardinality,
                use_feat_dynamic_real=True,
                distr_output = distr_output,
                freq=freq,
                lags_seq=gvar._lags_seq,
                trainer=Trainer(ctx=ctx, 
                                batch_size = batch_size,
                                epochs=epochs, 
                                learning_rate=1e-3, 
                                #hybridize=False,
                                num_batches_per_epoch=100
                               )
                )
        else:
            estimator = TransformerEstimator(
                prediction_length=prediction_length,
                context_length= context_length,
                use_feat_static_cat=use_feat_static,
                #cardinality=cardinality,
                use_feat_dynamic_real=True,
                distr_output = distr_output,
                freq=freq,
                lags_seq=gvar._lags_seq,
                trainer=Trainer(ctx=ctx, 
                                batch_size = batch_size,
                                epochs=epochs, 
                                learning_rate=1e-3, 
                                #hybridize=False,
                                num_batches_per_epoch=100
                               )
                )
    elif model == 'TransformerW-Oracle':

        if use_feat_static:
            estimator = TransformerWeightedEstimator(
                prediction_length=prediction_length,
                context_length= context_length,
                use_feat_static_cat=use_feat_static,
                cardinality=cardinality,
                use_feat_dynamic_real=True,
                distr_output = distr_output,
                freq=freq,
                lags_seq=gvar._lags_seq,
                trainer=Trainer(ctx=ctx, 
                                batch_size = batch_size,
                                epochs=epochs, 
                                learning_rate=1e-3, 
                                #hybridize=False,
                                num_batches_per_epoch=100
                               )
                )
        else:
            estimator = TransformerWeightedEstimator(
                prediction_length=prediction_length,
                context_length= context_length,
                use_feat_static_cat=use_feat_static,
                #cardinality=cardinality,
                use_feat_dynamic_real=True,
                distr_output = distr_output,
                freq=freq,
                lags_seq=gvar._lags_seq,
                trainer=Trainer(ctx=ctx, 
                                batch_size = batch_size,
                                epochs=epochs, 
                                learning_rate=1e-3, 
                                #hybridize=False,
                                num_batches_per_epoch=100
                               )
                )
    elif model == 'TransformerWF-Oracle' or model == 'RankNet-Transformer':

        if use_feat_static:
            estimator = TransformerWeightedFullLossEstimator(
                prediction_length=prediction_length,
                context_length= context_length,
                use_feat_static_cat=use_feat_static,
                cardinality=cardinality,
                use_feat_dynamic_real=True,
                distr_output = distr_output,
                freq=freq,
                lags_seq=gvar._lags_seq,
                trainer=Trainer(ctx=ctx, 
                                batch_size = batch_size,
                                epochs=epochs, 
                                learning_rate=1e-3, 
                                #hybridize=False,
                                num_batches_per_epoch=100
                               )
                )
        else:
            estimator = TransformerWeightedFullLossEstimator(
                prediction_length=prediction_length,
                context_length= context_length,
                use_feat_static_cat=use_feat_static,
                #cardinality=cardinality,
                use_feat_dynamic_real=True,
                distr_output = distr_output,
                freq=freq,
                lags_seq=gvar._lags_seq,
                trainer=Trainer(ctx=ctx, 
                                batch_size = batch_size,
                                epochs=epochs, 
                                learning_rate=1e-3, 
                                #hybridize=False,
                                num_batches_per_epoch=100
                               )
                )
    elif model == 'TransformerWFM-Oracle':

        if use_feat_static:
            estimator = TransformerWeightedFullLossMaskedEstimator(
                prediction_length=prediction_length,
                context_length= context_length,
                use_feat_static_cat=use_feat_static,
                cardinality=cardinality,
                use_feat_dynamic_real=True,
                distr_output = distr_output,
                freq=freq,
                lags_seq=gvar._lags_seq,
                weight_coef=gvar._weight_coef,
                trainer=Trainer(ctx=ctx, 
                                batch_size = batch_size,
                                epochs=epochs, 
                                learning_rate=1e-3, 
                                #hybridize=False,
                                num_batches_per_epoch=100
                               )
                )
        else:
            estimator = TransformerWeightedFullLossMaskedEstimator(
                prediction_length=prediction_length,
                context_length= context_length,
                use_feat_static_cat=use_feat_static,
                #cardinality=cardinality,
                use_feat_dynamic_real=True,
                distr_output = distr_output,
                freq=freq,
                lags_seq=gvar._lags_seq,
                weight_coef=gvar._weight_coef,
                trainer=Trainer(ctx=ctx, 
                                batch_size = batch_size,
                                epochs=epochs, 
                                learning_rate=1e-3, 
                                #hybridize=False,
                                num_batches_per_epoch=100
                               )
                )
    
    elif model == 'TransformerF-Oracle':

        if use_feat_static:
            estimator = TransformerFullLossEstimator(
                prediction_length=prediction_length,
                context_length= context_length,
                use_feat_static_cat=use_feat_static,
                cardinality=cardinality,
                use_feat_dynamic_real=True,
                distr_output = distr_output,
                freq=freq,
                lags_seq=gvar._lags_seq,
                trainer=Trainer(ctx=ctx, 
                                batch_size = batch_size,
                                epochs=epochs, 
                                learning_rate=1e-3, 
                                #hybridize=False,
                                num_batches_per_epoch=100
                               )
                )
        else:
            estimator = TransformerFullLossEstimator(
                prediction_length=prediction_length,
                context_length= context_length,
                use_feat_static_cat=use_feat_static,
                #cardinality=cardinality,
                use_feat_dynamic_real=True,
                distr_output = distr_output,
                freq=freq,
                lags_seq=gvar._lags_seq,
                trainer=Trainer(ctx=ctx, 
                                batch_size = batch_size,
                                epochs=epochs, 
                                learning_rate=1e-3, 
                                #hybridize=False,
                                num_batches_per_epoch=100
                               )
                )
            
            
    elif model == 'deepAR-multi':
        estimator = DeepAREstimator(
            prediction_length=prediction_length,
            context_length= context_length,
            use_feat_static_cat=use_feat_static,
            #cardinality=cardinality,
            use_feat_dynamic_real=False,
            freq=freq,
            lags_seq=gvar._lags_seq,
            trainer=Trainer(ctx=ctx, 
                            batch_size = batch_size,
                            epochs=epochs, 
                            learning_rate=1e-3, 
                            num_batches_per_epoch=100
                           ),
            distr_output=MultivariateGaussianOutput(dim=target_dim),
        )
    elif model == 'deepARW-multi' or model == 'RankNet-Joint':
        estimator = DeepARWeightEstimator(
            prediction_length=prediction_length,
            context_length= context_length,
            use_feat_static_cat=use_feat_static,
            #cardinality=cardinality,
            use_feat_dynamic_real=False,
            freq=freq,
            lags_seq=gvar._lags_seq,
            weight_coef=gvar._weight_coef,
            trainer=Trainer(ctx=ctx, 
                            batch_size = batch_size,
                            epochs=epochs, 
                            learning_rate=1e-3, 
                            num_batches_per_epoch=100
                           ),
            distr_output=MultivariateGaussianOutput(dim=target_dim),
        )


    elif model == 'simpleFF':
        estimator = SimpleFeedForwardEstimator(
            num_hidden_dimensions=[10],
            prediction_length=prediction_length,
            context_length= context_length,
            freq=freq,
            lags_seq=gvar._lags_seq,
            trainer=Trainer(ctx=ctx, 
                            batch_size = batch_size,
                            epochs=epochs,
                            learning_rate=1e-3,
                            hybridize=False,
                            num_batches_per_epoch=100
                           )
        )
    elif model == 'deepFactor':
        estimator = DeepFactorEstimator(
            prediction_length=prediction_length,
            context_length= context_length,
            freq=freq,
            lags_seq=gvar._lags_seq,
            trainer=Trainer(ctx=ctx, 
                            batch_size = batch_size,
                            epochs=epochs, 
                            learning_rate=1e-3, 
                            num_batches_per_epoch=100
                           )
        )
    elif model == 'deepState':
        estimator = DeepStateEstimator(
            prediction_length=prediction_length,
            use_feat_static_cat=True,
            cardinality=cardinality,
            freq=freq,
            lags_seq=gvar._lags_seq,
            trainer=Trainer(ctx=ctx, 
                            batch_size = batch_size,
                            epochs=epochs, 
                            learning_rate=1e-3, 
                            num_batches_per_epoch=100
                           )
        )
    elif model == 'ets':
        estimator = RForecastPredictor(method_name='ets',freq= freq, prediction_length = prediction_length)
    elif model == 'prophet':

        estimator = ProphetPredictor(freq= freq, prediction_length = prediction_length)
    elif model == 'arima':
        estimator = RForecastPredictor(method_name='arima',freq= freq, prediction_length = prediction_length, trunc_length = context_length)
    elif model == 'naive':
        estimator = NaivePredictor(freq= freq, prediction_length = prediction_length)
    else:
        logger.error('model %s not support yet, quit', model)
        sys.exit(-1)


    return estimator


# In[ ]:


#
# simulation engine general
#
def init_simulation(datasetid, testevent, taskid, runts, expid, predictionlen, 
               featuremode = stint.FEATURE_STATUS,
               pitmodel = 0, 
               inlapmode=0,
               train_len = 40,test_train_len=40,
               joint_train = False,
               pitmodel_bias= 0, prepared_laptimedata = None):
    """
    input:
        prepared_laptimedata   ; global
    
    """
    
    
    #
    # configurataion
    #
    # model path:  <_dataset_id>/<_task_id>-<trainid>/
    #_dataset_id = 'indy2013-2018-nocarid'
    
    stint._inlap_status = inlapmode
    
    stint.init(gvar.LAPTIME_DATASET, pitmodel, pitmodel_bias= pitmodel_bias)
    
    stint._dataset_id = datasetid
    stint._test_event = testevent
    #_test_event = 'Indy500-2019'

    stint._feature_mode = featuremode
    stint._context_ratio = 0.

    stint._task_id = taskid  # rank,laptime, the trained model's task
    stint._run_ts = runts   #COL_LAPTIME,COL_RANK
    stint._exp_id=expid  #rank, laptime, laptim2rank, timediff2rank... 

    stint._use_mean = True
    
    stint._train_len = train_len
    stint._test_train_len = test_train_len
    
    stint._joint_train = joint_train

    # todo: add into stint code
    #here add new laptime_data with new features
    #
    stint.set_laptimedata(prepared_laptimedata)
    
def simulation(datasetid, testevent, taskid, runts, expid, predictionlen, 
               datamode, loopcnt, featuremode = stint.FEATURE_STATUS,
              pitmodel = 0, model = 'oracle', inlapmode=0, train_len = 40,test_train_len=40,
              forecastmode = 'shortterm', joint_train = False, 
               pitmodel_bias= 0, prepared_laptimedata = None,
               epochs = 1000):
    """
    input:
        prepared_laptimedata   ; global
    
    """
    
    #
    # configurataion
    #
    # model path:  <_dataset_id>/<_task_id>-<trainid>/
    #_dataset_id = 'indy2013-2018-nocarid'
    
    stint._inlap_status = inlapmode
    
    stint.init(gvar.LAPTIME_DATASET, pitmodel, pitmodel_bias= pitmodel_bias)
    
   
    stint._dataset_id = datasetid
    stint._test_event = testevent
    #_test_event = 'Indy500-2019'

    stint._feature_mode = featuremode
    stint._context_ratio = 0.

    stint._task_id = taskid  # rank,laptime, the trained model's task
    stint._run_ts = runts   #COL_LAPTIME,COL_RANK
    stint._exp_id=expid  #rank, laptime, laptim2rank, timediff2rank... 

    stint._use_mean = True
    
    stint._train_len = train_len
    stint._test_train_len = test_train_len
    
    stint._joint_train = joint_train

    if forecastmode == 'stint':
        stint._trim = 0
        stint._debug_carlist=[]
        stint._force_endpit_align = False
        stint._include_endpit = True    
 
    # todo: add into stint code
    #here add new laptime_data with new features
    #
    stint.set_laptimedata(prepared_laptimedata)
    #stint.set_laptimedata(laptime_data)

    predictor = stint.load_model(predictionlen, model,trainid='indy500',epochs = epochs, exproot='./')

    ret2 = {}
    for i in range(loopcnt):
        #df, full_samples, full_tss
        if forecastmode == 'shortterm':
            ret2[i] = stint.run_simulation_shortterm(predictor, predictionlen, stint.freq, datamode=datamode, verbose=False)
        elif forecastmode == 'stint':
            ret2[i] = stint.run_simulation_pred(predictor, predictionlen, stint.freq, datamode=datamode)
        else:
            print('forecastmode not support:', forecastmode)
            break
            

    acc = []
    for i in ret2.keys():
        
        if forecastmode == 'shortterm':
            df = ret2[i][0]
            _x = stint.get_evalret_shortterm(df)
        elif forecastmode == 'stint':
            df = ret2[i]
            _x = stint.get_evalret(df)
        
        acc.append(_x)

    b = np.array(acc)
    print(np.mean(b, axis=0))
    
    #save keys
    #stint._pitmodel.save_keys('pitmodel-keys.pickle')
    
    return b, ret2

def long_predict(predictor, sampleCnt = 100):
    """
    use the farest samples only
    
    input:
        test_ds    ; global var
        predictor
    
    """
    
    def get_start(idx):
        td = forecasts[idx].start_date - start_time
        return td.days*24*60 + td.seconds//60
    
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,  # test dataset
        predictor=predictor,  # predictor
        num_samples=sampleCnt,  # number of sample paths we want for evaluation
    )

    forecasts = list(forecast_it)
    tss = list(ts_it)
    print(f'tss len={len(tss)}, forecasts len={len(forecasts)}')
    
    start_time, row = next(tss[0].iterrows())

    first_start = get_start(-1)
    last_start = get_start(0)
    print(first_start, last_start)    
    
    import copy
    target = copy.deepcopy(forecasts[-1])

    #100, 10
    nsample, npredict = target.samples.shape
    print('sampel# x predictlen: ', nsample, npredict)
    
    newsamples = np.zeros((nsample, last_start - first_start + npredict))
    newsamples[:,:] = np.nan

    for idx in range(len(forecasts)):
        #copy samples
        start_pos = get_start(idx)

        pos = start_pos - first_start
        #copy sample to block
        #newsamples[:, pos:pos + npredict] = forecasts[idx].samples
        newsamples[:, pos + npredict - 1] = forecasts[idx].samples[:,-1]
        

    target.samples = newsamples

    #plot_prob_forecasts_ex([tss[0]],[target],output)
    
    return target, tss[0]



    


def get_alldf(dfx, year=2018, forecast_mode = 'shortterm'):
    #dfx = ret[f'{model}-RANK-{year}-inlap-nopitage']
    #dfx = ret[f'{model}-TIMEDIFF-{year}-noinlap-nopitage']
    
    samples = dfx.keys()
    retdfs = []
    for id in samples:
        if forecast_mode == 'shortterm':
            df = dfx[id][0]
        else:
            df = dfx[id]
        retdfs.append(df)
        
    if len(retdfs) > 1:
        dfout = pd.concat(retdfs)
    else:
        dfout = retdfs[0]
        
    return dfout
    
def get_alldf_mode(dfx, year=2018,mode=0, forecast_mode = 'shortterm'):
    """
    mode: 
        0; mode
        1; mean
        2; median
    """
    dfall = get_alldf(dfx, year=year, forecast_mode = forecast_mode)
    
    cars = set(dfall.carno.values)
    startlaps = {}
    for car in cars:
        startlaps[car] = set(dfall[dfall['carno']==car].startlap.values)
        
    retdf = []
    for car in cars:
        for startlap in startlaps[car]:
            dfrec = dfall[(dfall['carno']==car) & (dfall['startlap']==startlap)]
            
            #get mode
            if mode == 0:
                pred_endrank = stats.mode(dfrec.pred_endrank.values).mode[0]
                #pred_endlap =  stats.mode(dfrec.pred_endlap.values).mode[0]
            elif mode == 1:
                #use mean
                pred_endrank = np.mean(dfrec.pred_endrank.values)
                #pred_endlap =  np.mean(dfrec.pred_endlap.values)
            elif mode == 2:
                #use mean
                pred_endrank = np.median(dfrec.pred_endrank.values)
                #pred_endlap =  np.median(dfrec.pred_endlap.values)
            
            firstrec = dfrec.to_numpy()[0,:]
            firstrec[6] = pred_endrank
            firstrec[7] = pred_endrank - firstrec[2]
            if firstrec[7] == 0:
                firstrec[8] = 0
            elif firstrec[7] > 0:
                firstrec[8] = 1
            else:
                firstrec[8] = -1
                
            #endlap, pred_endlap
            
        
            retdf.append(firstrec)
        
    #dfout = pd.concat(retdf)
    if forecast_mode == 'shortterm':
        dfout = pd.DataFrame(retdf, columns =['carno', 'startlap', 'startrank',    
                                         'endrank', 'diff', 'sign',
                                         'pred_endrank', 'pred_diff', 'pred_sign',
                                         #'endlap','pred_endlap'
                                        ])
    else:
        dfout = pd.DataFrame(retdf, columns =['carno', 'startlap', 'startrank',    
                                         'endrank', 'diff', 'sign',
                                         'pred_endrank', 'pred_diff', 'pred_sign',
                                         'endlap','pred_endlap'
                                        ])
        
    print('df size:', len(dfout))
    return dfout

def get_allsamples(dfx, year=2018):
    
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


def prisk_direct_bysamples2(full_samples, full_tss, quantiles=[0.1,0.5,0.9], startid = 12, verbose=False):
    """
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


# In[ ]:


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

def clear_samples(full_samples, full_tss, clearidx):
    """
    clear the laps in clearidx
    """
    import copy
    ret_samples = copy.deepcopy(full_samples)
    ret_tss = copy.deepcopy(full_tss)
    
    
    carlist = full_tss.keys()
    
    for carid, carno in enumerate(carlist):
        forecast = ret_samples[carno]
        target = ret_tss[carno]
        
        forecast[:, clearidx] = np.nan
        target[clearidx] = np.nan
        
        ret_samples[carno] = forecast
        ret_tss[carno] = target
        
    return ret_samples, ret_tss

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


# In[ ]:


def long_predict_bymloutput_multirun(output, dfin, test_ds, predictor, sampleCnt=100):
    """
    input:
        test_ds
        predictor
    
    """
    def get_start(idx):
        td = forecasts[idx].start_date - start_time
        return td.days*24*60 + td.seconds//60
    
    forecast_it, ts_it = make_evaluation_predictions(
        dataset = test_ds,  # test dataset
        predictor = predictor,  # predictor
        num_samples=sampleCnt,  # number of sample paths we want for evaluation
    )

    forecasts = list(forecast_it)
    tss = list(ts_it)
    print(f'tss len={len(tss)}, forecasts len={len(forecasts)}')
    
    start_time, row = next(tss[0].iterrows())

    first_start = get_start(-1)
    last_start = get_start(0)
    print('first start:', first_start, 'last start:', last_start)    
    
    import copy
    target = copy.deepcopy(forecasts[-1])

    #100, 10
    nsample, npredict = target.samples.shape
    print('sampel# x predictlen: ', nsample, npredict)
    
    newsamples = np.zeros((nsample, last_start - first_start + npredict))
    newsamples[:,:] = np.nan
    
    for idx in range(len(forecasts)):
        #copy samples
        start_pos = get_start(idx)

        pos = start_pos - first_start + npredict - 1
        #copy sample to block
        #newsamples[:, pos:pos + npredict] = forecasts[idx].samples
        #newsamples[:, pos + npredict - 1] = forecasts[idx].samples[:,-1]
        
        # get prediction from ml output
        # pos = laps
        # 1 ... 10 | 11 <- start pos in forecasts
        # 0 ...  9 | 10 <- 9 is the startlap
        #
        startlap = start_pos  - 2
        #print('start pos:', start_pos, 'pos:',pos, 'startlap:', startlap)
        
        _rec = dfin[dfin['startlap']== startlap]
        if len(_rec) > 0:
            # rank start from 1 for visualization
            pred_val = _rec.pred_endrank.values
            
            #pred_val = _rec.pred_endrank.values
            #make sure shape match, 100 samples
            
            #newsamples[:, pos + npredict - 1] = pred_val + 1
            newsamples[:, pos] = pred_val + 1
            #print('startlap:', startlap, 'predrank:', pred_val)

    target.samples = newsamples
    
    print('multirun target samples:', target.samples.shape)

    #plot_prob_forecasts_ex([tss[0]],[target],output)
    
    return target,tss[0]

def long_predict_bymloutput(output, dfin, test_ds, predictor):
    """
    input:
        test_ds
        predictor
    
    """
    def get_start(idx):
        td = forecasts[idx].start_date - start_time
        return td.days*24*60 + td.seconds//60
    
    forecast_it, ts_it = make_evaluation_predictions(
        dataset = test_ds,  # test dataset
        predictor = predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
    )

    forecasts = list(forecast_it)
    tss = list(ts_it)
    print(f'tss len={len(tss)}, forecasts len={len(forecasts)}')
    
    start_time, row = next(tss[0].iterrows())

    first_start = get_start(-1)
    last_start = get_start(0)
    print('first start:', first_start, 'last start:', last_start)    
    
    import copy
    target = copy.deepcopy(forecasts[-1])

    #100, 10
    nsample, npredict = target.samples.shape
    print('sampel# x predictlen: ', nsample, npredict)
    
    newsamples = np.zeros((nsample, last_start - first_start + npredict))
    newsamples[:,:] = np.nan
    
    for idx in range(len(forecasts)):
        #copy samples
        start_pos = get_start(idx)

        pos = start_pos - first_start + npredict - 1
        #copy sample to block
        #newsamples[:, pos:pos + npredict] = forecasts[idx].samples
        #newsamples[:, pos + npredict - 1] = forecasts[idx].samples[:,-1]
        
        # get prediction from ml output
        # pos = laps
        # 1 ... 10 | 11 <- start pos in forecasts
        # 0 ...  9 | 10 <- 9 is the startlap
        #
        startlap = start_pos  - 2
        #print('start pos:', start_pos, 'pos:',pos, 'startlap:', startlap)
        
        _rec = dfin[dfin['startlap']== startlap]
        if len(_rec) > 0:
            # rank start from 1 for visualization
            pred_val = _rec.pred_endrank.values[0]
            
            #pred_val = _rec.pred_endrank.values
            #make sure shape match, 100 samples
            
            #newsamples[:, pos + npredict - 1] = pred_val + 1
            newsamples[:, pos] = pred_val + 1
            #print('startlap:', startlap, 'predrank:', pred_val)

    target.samples = newsamples
    
    print('target samples:', target.samples.shape)

    #plot_prob_forecasts_ex([tss[0]],[target],output)
    
    return target,tss[0]

def long_predict_bysamples(output, samples, tss, test_ds, predictor):
    """
    use the farest samples only
    
    input:
        samples
        tss
    
    """
    
    def get_start(idx):
        td = forecasts[idx].start_date - start_time
        return td.days*24*60 + td.seconds//60
    
    forecast_it, ts_it = make_evaluation_predictions(
        dataset = test_ds,  # test dataset
        predictor= predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
    )

    forecasts = list(forecast_it)
    tss = list(ts_it)
    print(f'tss len={len(tss)}, forecasts len={len(forecasts)}')
    
    start_time, row = next(tss[0].iterrows())

    first_start = get_start(-1)
    last_start = get_start(0)
    print(first_start, last_start)    
    
    import copy
    target = copy.deepcopy(forecasts[-1])

    #100, 10
    nsample, npredict = target.samples.shape
    print('sampel# x predictlen: ', nsample, npredict)
    
    #sample array size: last_start - first_start + npredict
    arraysize = last_start - first_start + npredict
    
    #error here
    #target.samples = samples[:,-len(forecasts)-1:] + 1
    #target.samples = samples[:, 10 + npredict:] + 1
    target.samples = samples[:, first_start:first_start + arraysize] + 1

    print('long_predict_bysamples==>target samples shape:', target.samples.shape)
    #plot_prob_forecasts_ex([tss[0]],[target],output)
    
    return target, tss[0]

#
# different idx format to bymloutput
#
def long_predict_bydf(output, dfin, test_ds, predictor):
    """
    input:
        test_ds
        predictor
    
    """
    def get_start(idx):
        td = forecasts[idx].start_date - start_time
        return td.days*24*60 + td.seconds//60
    
    forecast_it, ts_it = make_evaluation_predictions(
        dataset = test_ds,  # test dataset
        predictor = predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
    )

    forecasts = list(forecast_it)
    tss = list(ts_it)
    print(f'tss len={len(tss)}, forecasts len={len(forecasts)}')
    
    start_time, row = next(tss[0].iterrows())

    first_start = get_start(-1)
    last_start = get_start(0)
    print('first start:', first_start, 'last start:', last_start)    
    
    import copy
    target = copy.deepcopy(forecasts[-1])

    #100, 10
    nsample, npredict = target.samples.shape
    print('sampel# x predictlen: ', nsample, npredict)
    
    newsamples = np.zeros((nsample, last_start - first_start + npredict))
    newsamples[:,:] = np.nan
    
    for idx in range(len(forecasts)):
        #copy samples
        start_pos = get_start(idx)

        pos = start_pos - first_start + npredict - 1
        #copy sample to block
        #newsamples[:, pos:pos + npredict] = forecasts[idx].samples
        #newsamples[:, pos + npredict - 1] = forecasts[idx].samples[:,-1]
        
        # get prediction from ml output
        # pos = laps
        # 1 ... 10 | 11 <- start pos in forecasts
        # 0 ...  9 | 10 <- 9 is the startlap
        #
        startlap = start_pos  - 1
        #print('start pos:', start_pos, 'pos:',pos, 'startlap:', startlap)
        
        _rec = dfin[dfin['startlap']== startlap]
        if len(_rec) > 0:
            # rank start from 1 for visualization
            pred_val = _rec.pred_endrank.values[0]
            
            #pred_val = _rec.pred_endrank.values
            #make sure shape match, 100 samples
            
            #newsamples[:, pos + npredict - 1] = pred_val + 1
            newsamples[:, pos] = pred_val + 1
            #print('startlap:', startlap, 'predrank:', pred_val)

    target.samples = newsamples
    
    print('target samples:', target.samples.shape)

    #plot_prob_forecasts_ex([tss[0]],[target],output)
    
    return target,tss[0]

def get_ranknet_multirun(retdata, testcar, test_ds, predictor, sampleCnt=100):
    dfs = []
    #for id in range(samplecnt):
    for id in retdata.keys():
        #ret['pitmodel-RANK-2018-inlap-nopitage']
        df = retdata[id][0]
        df = df[df['carno']==testcar]
        dfs.append(df)

    dfin_ranknet = pd.concat(dfs)

    print('dfin_ranknet size:', len(dfin_ranknet))
    
    #modify to fit to ml model format
    dfin_ranknet['startlap'] = dfin_ranknet['startlap'] - 1
    dfin_ranknet['startrank'] = dfin_ranknet['startrank'] - 1
    dfin_ranknet['endrank'] = dfin_ranknet['endrank'] - 1
                
    target_ranknet, tss_ranknet = long_predict_bymloutput_multirun('ranknet-rank', dfin_ranknet, 
            test_ds, predictor, sampleCnt=sampleCnt)                
                
    return target_ranknet, tss_ranknet


# In[ ]:


def ploth(ts_entry, forecast_entry, pits,caution, pitstop,outputfile,
                   colors = ['r','g','m'],
                   plabels= ['observed','svr','arima','ranknet'],
                   ylabel = 'RANK'):

    #plot_length = int(forecast_entry[0].samples.shape[1] *1.2) 
    #plot_length = forecast_entry[0].samples.shape[1] + 10 
    
    #prediction_intervals = (50.0, 90.0)
    prediction_intervals = [90.0]
    
    #legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals]

    figcnt = len(forecast_entry)
    
    #fig, axs = plt.subplots(figcnt,1, figsize=(8,6))
    fig, axs = plt.subplots(1, figcnt, figsize=(12,3*figcnt))

    #colors = ['r','g','m']
    #plabels = ['observed','svr','arima','ranknet']
    
    for idx in range(figcnt):
        ax = plt.subplot(figcnt, 1, idx+1)
        #ax = plt.subplot(1, figcnt, idx+1)
        #ts_entry.iloc[-plot_length:,0].plot(ax=axs, linewidth=1)  # plot the time series
        #ts_entry.iloc[-plot_length:,0].plot(ax=axs[idx], linewidth=1)  # plot the time series
        #plot_length = int(forecast_entry[idx].samples.shape[1] *1.2) 
        ts_entry[idx].iloc[:,0].plot(linewidth=1, color='b',
                                            marker='*', alpha=0.7, zorder=-1, label=plabels[0]) 


        # currank
        sv = ts_entry[idx].iloc[:,0].to_numpy()
        start = pd.Timestamp("01-01-2019", freq='1min') + 2
        date_index = pd.date_range(start, periods = len(sv)-2, freq='1min')
        df2 = pd.DataFrame(sv[:-2], index=date_index)        
        df2.iloc[:,0].plot(linewidth=0.5, color='k',
                                            marker='+', alpha=0.7, zorder=-1, label='CurRank') 
        
        
    #for idx in range(len(forecast_entry)):
    #    forecast_entry[idx].copy_dim(0).plot(prediction_intervals=prediction_intervals, color='g')
    
        forecast_entry[idx].copy_dim(0).plot(prediction_intervals=prediction_intervals, 
                                             color=colors[idx],label=plabels[idx+1], zorder=10)
        #forecast_entry[1].copy_dim(0).plot(prediction_intervals=prediction_intervals, color='b')
        #forecast_entry[2].copy_dim(0).plot(prediction_intervals=prediction_intervals, color='r')
        
        #add mean line, compare with median
        #if forecast_entry[idx].samples.shape[0] > 1:
        if idx>3:
            mean_forecast = copy.deepcopy(forecast_entry[idx])
            mean_forecast.samples = np.mean(mean_forecast.samples, axis=0).reshape((1,-1))
            mean_forecast.copy_dim(0).plot(prediction_intervals=prediction_intervals, 
                                                 color='g',label='use-mean', zorder=10)
        
        
        if idx == figcnt-1:
            ax.set_xlabel('Lap')
        #if idx==0:
        ax.set_ylabel(ylabel)
        if idx==0:
            plt.title(outputfile)        
    
        locs, labels = plt.xticks() 
        #plt.xticks(locs, range(len(locs)))
        start_loc = locs[0]        
        #offset = range(0, 200, 5)
        offset = range(0, gvar.maxlap, 5)
        #new_locs = range(start_loc , start_loc+200, 10)
        new_locs = [start_loc + x for x in offset]
        #new_labels = [str(x-start_loc + 1) for x in new_locs]
        new_labels = [str(x+1) for x in offset]
        plt.xticks(new_locs, new_labels)

        if figcnt==1 or idx < figcnt -1:
            print('xlim:', plt.xlim())
            xl, xr = plt.xlim()
            xlim_h = len(ts_entry[idx])
            
            #xlim_h = 100
            ax.set_xlim((xl+0,xl+xlim_h))
        elif idx == figcnt - 1:
            xlim_h = len(ts_entry[idx])
            
            #xlim_h = 100
            ax.set_xlim((xl+0,xl+xlim_h))
        
        if ylabel=='RANK':
            ax.set_ylim((-5,+40))
        else:
            ax.set_ylim((25,175))
            
        #ax.set_xlim((80,110))
        ax.set_zorder(-1)
        plt.grid(which="both", zorder=-1)
        ax.set_axisbelow(True)
        
        l=plt.legend(prop={'size': 10},loc='upper left')
        l.set_zorder(0.6)
        
        #add racestatus
        if ylabel=='RANK':
            ax.plot(xl+pits[:,0]-1,pits[:,1],'^',color='r', label='PitStop', linewidth=2,alpha=0.7, zorder=-1)
            add_status(ax,xl, caution, pitstop)
        else:
            ax.plot(xl+pits[:,0]-1,pits[:,2],'^',color='r', label='PitStop', linewidth=2,alpha=0.7, zorder=-1)
            add_status(ax,xl, caution, pitstop,y=27, height=3)
        
        
    
    plt.show()
    fig.tight_layout()
    fig.savefig(outputfile + '.pdf')    
    
def plotcar(carno):
    """
    input:
        alldata, rankdata; global data
    """
    #target_svr, target_rf,target_arima, target_oracle, target_ranknet_1run = savedata[carno]
    #target_oracle(by longpredict), tss_oracle_multirun,tss_ranknet_multirun
    tsss, targets = alldata[carno]
    
    pits, cautions, caution, pitstop,ranks,laptimes = get_racestatus(carno, gvar.rankdata)
    print(np.where(pitstop==1))
    
    ploth(tsss[:5], targets[:5], pits, caution, pitstop,
               'ranknet-rf-rank-forecast-%d'%carno,
                   colors = ['y','c','g','m','r'],
                   plabels= ['observed','SVR','RF','Arima','RrankNet-Oracle','RrankNet-MLP'])
    
def plotcar_laptime(carno):
    """
    input:
        alldata, rankdata; global data
    """
    #target_svr, target_rf,target_arima, target_oracle, target_ranknet_1run = savedata[carno]
    #target_oracle(by longpredict), tss_oracle_multirun,tss_ranknet_multirun
    tsss, targets = alldata[carno]
    
    pits, cautions, caution, pitstop,ranks,laptimes = get_racestatus(carno, gvar.rankdata)
    print(np.where(pitstop==1))
    
    ploth(tsss, targets, pits, caution, pitstop,
               'ranknet-oracle-laptime-forecast-%d'%carno,
                   colors = ['m','r'],
                   plabels= ['observed','RrankNet-Oracle','RrankNet-MLP'],
                ylabel='LapTime')
    
    
def plotrank(outputfile, mode='RANK' ):
    """
    input:
        alldata, rankdata; global data
    """
    
    figcnt = len(alldata)
    fig, axs = plt.subplots(1, figcnt, figsize=(12,3*figcnt))

    carlist = list(alldata.keys())
    
    for idx, carno in enumerate(carlist):
        #target_svr, target_rf,target_arima, target_oracle, target_ranknet_1run = savedata[carno]
        #target_oracle(by longpredict), tss_oracle_multirun,tss_ranknet_multirun
        tsss, targets = alldata[carno]

        pits, cautions, caution, pitstop,ranks,laptimes = get_racestatus(carno, gvar.rankdata)
        print(np.where(pitstop==1))
    
        ax = plt.subplot(figcnt, 1, idx+1)
        
        if mode == 'RANK':
            ax.plot(ranks, linewidth=1, color='b',marker='*', alpha=0.7, zorder=-1, label='Rank') 
            ax.set_ylim((-5,+35))
            ax.plot(pits[:,0]-1,pits[:,1],'^',color='r', label='PitStop', linewidth=2,alpha=0.7, zorder=-1)
            #add racestatus
            add_status(ax,0, caution, pitstop)

        else:
            ax.plot(laptimes, linewidth=1, color='b',marker='*', alpha=0.7, zorder=-1, label='LapTime') 
            ax.set_ylim((30,140))
            ax.plot(pits[:,0]-1,pits[:,2],'^',color='r', label='PitStop', linewidth=2,alpha=0.7, zorder=-1)
            #add racestatus
            add_status(ax,0, caution, pitstop,y=32, height=5)
        
        #ax.set_xlim((0,200))
        ax.set_xlim((0,gvar.maxlap))
        
        ax.set_ylabel('car-%d'%carno)
        
        
    plt.show()
    fig.tight_layout()
    fig.savefig(outputfile + '.pdf')    
    
def plotcarx(carno):
    """
    input:
        alldata, rankdata; global data
    """
    #target_svr, target_rf,target_arima, target_oracle, target_ranknet_1run = savedata[carno]
    #target_oracle(by longpredict), tss_oracle_multirun,tss_ranknet_multirun
    tsss, targets = alldata[carno]
    
    oracle_tss, oracle_targets = oracledata[carno]
    
    tsss[2] = oracle_tss[1]
    targets[2] = oracle_targets[1]
    
    pits, cautions, caution, pitstop,ranks,laptimes = get_racestatus(carno, gvar.rankdata)
    print(np.where(pitstop==1))
    
    ploth(tsss[:5], targets[:5], pits, caution, pitstop,
               'ranknet-rf-rank-forecast-%d'%carno,
                   colors = ['y','c','g','m','r'],
                   plabels= ['observed','SVR','RF','Weighted-Oracle','RrankNet-Oracle','RrankNet-MLP'])
    
    
def plotoracle(alldata, carno, destdir):
    """
    input:
        alldata, rankdata; global data
    """
    
    outputfile = destdir + 'ranknet-oracle-forecast-%d'%carno
    
    #target_svr, target_rf,target_arima, target_oracle, target_ranknet_1run = savedata[carno]
    #target_oracle(by longpredict), tss_oracle_multirun,tss_ranknet_multirun
    tsss, targets = alldata[carno]
    
    pits, cautions, caution, pitstop,ranks,laptimes = get_racestatus(carno, gvar.rankdata)
    print(np.where(pitstop==1))
    
    ploth(tsss, targets, pits, caution, pitstop,
               outputfile,
               colors = ['y','c','g','m','r'],
               plabels= ['observed','1run-samples','1run-df','multimean','norerank-multimean','mrun-samples'])    
    

def plotallcars(alldata, outputfile, drawid = 0, 
               colors = ['g','c','m','r','y'],
               plabels= ['observed','1run-samples','1run-df','multimean','norerank-multimean','mrun-samples'],
               ylabel='RANK'):
    """
    plot a single fig for all cars
    
    input:
        prediction_length,freq   ; global var
        alldata, rankdata; global data
        drawid : long prediction result index in alldata[carno] to draw
    """
    figcnt = len(alldata)
    fig, axs = plt.subplots(1, figcnt, figsize=(12,3*figcnt))
    prediction_intervals = [90.0]
    #legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals]

    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 12,
        }    
    
    carlist = list(alldata.keys())
    
    for idx, carno in enumerate(carlist):
        #target_svr, target_rf,target_arima, target_oracle, target_ranknet_1run = savedata[carno]
        #target_oracle(by longpredict), tss_oracle_multirun,tss_ranknet_multirun
        ts_entry, forecast_entry = alldata[carno]

        pits, cautions, caution, pitstop,ranks,laptimes = get_racestatus(carno, gvar.rankdata)
        print(np.where(pitstop==1))
    
        ax = plt.subplot(figcnt, 1, idx+1)
    
        # observed
        ts_entry[drawid].iloc[:,0].plot(linewidth=1, color='b',
                                            marker='*', alpha=0.7, zorder=-1, label=plabels[0]) 


        # currank
        sv = ts_entry[drawid].iloc[:,0].to_numpy()
        start = pd.Timestamp("01-01-2019", freq=gvar.freq) + gvar.prediction_length
        date_index = pd.date_range(start, periods = len(sv) - gvar.prediction_length, freq = gvar.freq)
        df2 = pd.DataFrame(sv[:- gvar.prediction_length], index=date_index)        
        df2.iloc[:,0].plot(linewidth=0.5, color='k',
                            marker='+', alpha=0.7, zorder=-1, label='CurRank') 
        
        #forecast
        forecast_entry[drawid].copy_dim(0).plot(prediction_intervals=prediction_intervals, 
                                             color=colors[drawid],label=plabels[drawid+1], zorder=10)
        
        if idx == figcnt-1:
            ax.set_xlabel('Lap')
        ax.set_ylabel(ylabel)
        
    
        locs, labels = plt.xticks() 
        #plt.xticks(locs, range(len(locs)))
        start_loc = locs[0]        
        #offset = range(0, 200, 5)
        offset = range(0, gvar.maxlap, 5)
        #new_locs = range(start_loc , start_loc+200, 10)
        new_locs = [start_loc + x for x in offset]
        #new_labels = [str(x-start_loc + 1) for x in new_locs]
        new_labels = [str(x+1) for x in offset]
        plt.xticks(new_locs, new_labels)

        if figcnt==1 or idx < figcnt -1:
            print('xlim:', plt.xlim())
            xl, xr = plt.xlim()
            xlim_h = len(ts_entry[drawid])
            
            #xlim_h = 100
            ax.set_xlim((xl+0,xl+xlim_h))
        elif idx == figcnt - 1:
            xlim_h = len(ts_entry[drawid])
            
            #xlim_h = 100
            ax.set_xlim((xl+0,xl+xlim_h))
            
        #plt.title(outputfile)        
        plt.text(xl + xlim_h - 15, 35, f'car-{carno}',fontdict=font)
            
        
        if ylabel=='RANK':
            ax.set_ylim((-5,+40))
        else:
            ax.set_ylim((25,175))
            
        #ax.set_xlim((80,110))
        ax.set_zorder(-1)
        plt.grid(which="both", zorder=-1)
        ax.set_axisbelow(True)
        
        l=plt.legend(prop={'size': 10},loc='upper left')
        l.set_zorder(0.6)
        
        #add racestatus
        if ylabel=='RANK':
            ax.plot(xl+pits[:,0]-1,pits[:,1],'^',color='r', label='PitStop', linewidth=2,alpha=0.7, zorder=-1)
            add_status(ax,xl, caution, pitstop)
        else:
            ax.plot(xl+pits[:,0]-1,pits[:,2],'^',color='r', label='PitStop', linewidth=2,alpha=0.7, zorder=-1)
            add_status(ax,xl, caution, pitstop,y=27, height=3)
        
        
    
    plt.show()
    fig.tight_layout()
    fig.savefig(outputfile + '.pdf')      
    

def get_racestatus_all(rankdata):
    df12 = rankdata
    
    data = df12[['completed_laps','rank','last_laptime','time_behind_leader']].values
    pitstop = df12[['lap_status']].values
    caution = df12[['track_status']].values
    pitstop = np.array([1 if x=='P' else 0 for x in pitstop])
    caution = np.array([1 if x=='Y' else 0 for x in caution])
    
    pitidx = np.where(pitstop == 1)
    pits = data[pitidx]
    pitlaps = sorted(set(list(pits[:,0].astype(int))))
    
    cautionidx = np.where(caution == 1)
    cautions = data[cautionidx]
    cautionlaps = sorted(set(list(cautions[:,0].astype(int))))

    return pitlaps, cautionlaps


def get_racestatus(carno, rankdata):
    df12 = rankdata[rankdata['car_number']==carno]
    #
    # completed_laps start from 0
    # in array mode completed_laps=1 should indexed by 0
    #
    data = df12[['completed_laps','rank','last_laptime','time_behind_leader']].values
    pitstop = df12[['lap_status']].values
    caution = df12[['track_status']].values
    pitstop = np.array([1 if x=='P' else 0 for x in pitstop])
    caution = np.array([1 if x=='Y' else 0 for x in caution])
    pitidx = np.where(pitstop == 1)
    pits = data[pitidx]
    yidx = np.where(caution == 1)
    cautions = data[yidx]
    
    ranks = df12[['rank']].values
    laptimes = df12[['last_laptime']].values

    #return pits, cautions, caution, pitstop
    return pits, cautions, caution[1:], pitstop[1:], ranks[1:],laptimes[1:]


#red = '#ff8080'
red = 'red'
#yellow = '#8080ff'
yellow = 'yellow'
#green = '#80ff80'
green = 'green'

def add_status(axs,xl, caution, pitstop, y=-4, height=2):
    """
    input:
        caution, pitstop : race status
    """
    maxlap = min(len(caution), len(pitstop))
    for lap in range(maxlap):
        fc = green
        if caution[lap] == 1:
            fc = yellow
        if pitstop[lap] == 1:
            fc = red
        ec = fc
        rectangle = plt.Rectangle((lap+xl-0.5,y), 1, height, fc=fc,ec=ec)
        #plt.gca().add_patch(rectangle)
        axs.add_patch(rectangle)    


# In[ ]:


#
# stint evaluation
#
def eval_bydf(testdf, bydf, forcematch=True, force2int=False):
    #collect only records in bydf <carno and startlap>
    cars = set(bydf.carno.values)
    startlaps = {}
    for car in cars:
        startlaps[car] = set(bydf[bydf['carno']==car].startlap.values)
        
    retdf = []
    for car in cars:
        for startlap in startlaps[car]:    
            dfrec = testdf[(testdf['carno']==car) & (testdf['startlap']==startlap)]
            
            #check match
            if forcematch:
                a = testdf[(testdf['carno']==car) & (testdf['startlap']==startlap)].to_numpy().astype(int)
                b = bydf[(bydf['carno']==car) & (bydf['startlap']==startlap)].to_numpy().astype(int)

                if len(a)!=0 and len(b)!=0:
                    #compare 
                    #startrank, endrank
                    if not ((a[0][2] == b[0][2]) and (a[0][3] == b[0][3])):
                        #print('mismatch:', a, b)            
                        continue
            
            retdf.append(dfrec)
        
    dfout = pd.concat(retdf)
    
    if force2int:
        dfdata = dfout.to_numpy().astype(int)
        dfout = pd.DataFrame(dfdata, columns =['carno', 'startlap', 'startrank',    
                                         'endrank', 'diff', 'sign',
                                         'pred_endrank', 'pred_diff', 'pred_sign',
                                         'endlap','pred_endlap'
                                        ])
    
    dfout = dfout.sort_values(by=['carno','startlap'])
    
    print('df size:', len(dfout))
    #return acc
    accret = stint.get_evalret(dfout)[0]
    
    return dfout  , accret


def eval_sync(testdf, errlist, force2int=False):
    """
    eval df result by sync with the errlist detected
    remove the records in errlist
    
    """
    #collect only records in bydf <carno and startlap>
    cars = set(testdf.carno.values)
    startlaps = {}
    for car in cars:
        startlaps[car] = set(testdf[testdf['carno']==car].startlap.values)
        
    retdf = []
    for car in cars:
        for startlap in startlaps[car]:    
            dfrec = testdf[(testdf['carno']==car) & (testdf['startlap']==startlap)]
            
            #check match
            this_rec = [car, startlap]
            if this_rec in errlist:
                continue
            
            retdf.append(dfrec)
        
    dfout = pd.concat(retdf)
    
    if force2int:
        dfdata = dfout.to_numpy().astype(int)
        dfout = pd.DataFrame(dfdata, columns =['carno', 'startlap', 'startrank',    
                                         'endrank', 'diff', 'sign',
                                         'pred_endrank', 'pred_diff', 'pred_sign',
                                         'endlap','pred_endlap'
                                        ])
    
    dfout = dfout.sort_values(by=['carno','startlap'])
    
    print('df size:', len(dfout))
    #return acc
    accret = stint.get_evalret(dfout)[0]
    
    return dfout  , accret

def cmp_df(testdf, bydf):
    """
    df can be different, minor difference for the rank when RankNet removes short ts
    """
    #collect only records in bydf <carno and startlap>
    cars = set(bydf.carno.values)
    startlaps = {}
    for car in cars:
        startlaps[car] = set(bydf[bydf['carno']==car].startlap.values)
        
    err_list = []
    retdf = []
    errcnt = 0
    for car in cars:
        for startlap in startlaps[car]:    
            a = testdf[(testdf['carno']==car) & (testdf['startlap']==startlap)].to_numpy().astype(int)
            b = bydf[(bydf['carno']==car) & (bydf['startlap']==startlap)].to_numpy().astype(int)
            
            if len(a)!=0 and len(b)!=0:
                #compare 
                #startrank, endrank
                if not ((a[0][2] == b[0][2]) and (a[0][3] == b[0][3])):
                    print('mismatch:', a, b)
                    errcnt += 1
                    err_list.append([car, startlap])
            else:
                errcnt += 1
                print('mismatch empty:', a, b)
                err_list.append([car, startlap])
                
                
    print('errcnt:', errcnt)
    return errcnt, err_list

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
        full_tss[carno] = np.zeros((gvar.maxlap))
        full_tss[carno][:] = np.nan
        full_samples[carno] = np.zeros((samplecnt,gvar.maxlap))
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

def df2samples_ex(dfall, samplecnt=100,errlist=[]):
    """
    for stint results only
    
    get samples from the runs
    
    input:
        runret  ; list of result df <carno,startlap,startrank,endrank,diff,sign,pred_endrank,pred_diff,pred_sign,endlap,pred_endlap>
        errlist ; <car, startlap> list
    return:
        samples, tss
    """
    #samplecnt = len(runret)
    full_samples = {}
    full_tss = {}
    
    carlist = set(dfall.carno.values)
    
    startlaps = {}
    for car in carlist:
        startlaps[car] = set(dfall[dfall['carno']==car].startlap.values)
        
    #empty samples
    for carid, carno in enumerate(carlist):
        full_tss[carno] = np.zeros((gvar.maxlap))
        full_tss[carno][:] = np.nan
        full_samples[carno] = np.zeros((samplecnt,gvar.maxlap))
        full_samples[carno][:] = np.nan
        
        for startlap in startlaps[carno]:
            
            thisrec = [carno,startlap]
            if thisrec in errlist:
                continue
            
            dfrec = dfall[(dfall['carno']==carno) & (dfall['startlap']==startlap)]
 
            # save to the endlap
            #curlap = int(dfrec.startlap.values[0])
            curlap = int(dfrec.endlap.values[0])

            target = dfrec.endrank.values[0]
            forecast = dfrec.pred_endrank.to_numpy()
            
            #if carno==12:
            #    print('forecast.shape', forecast.shape)
            
            full_samples[carno][:,curlap] = forecast
                
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
        full_tss[carno] = np.zeros((gvar.maxlap))
        full_tss[carno][:] = np.nan
        full_samples[carno] = np.zeros((samplecnt,gvar.maxlap))
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


# In[ ]:


def get_config():
    config = [
        _savedata,
        _skip_overwrite,
        _inlap_status,
        _feature_mode,
        _featureCnt,
        freq ,
        _train_len,
        prediction_length,
        context_ratio,
        context_length,
        contextlen,
        dataset,
        epochs,
        gpuid,
        _use_weighted_model,
        trainmodel,
        _use_cate_feature,
        use_feat_static,
        distroutput,
        batch_size,
        loopcnt,
        _test_event,
        testmodel,
        pitmodel,
        year
    ]
    
    return config


def test_global():
    gvar._hi += 200

def get_event_info(event):
    #eid = event.split('-')[0]
    return gvar._race_info[event]

