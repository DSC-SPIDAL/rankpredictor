#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import mxnet as mx
from mxnet import gluon
import pickle
import json
import random
import inspect
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
from indycar.model.ZeroPredictor import ZeroPredictor
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#from indycar.model.stint_predictor_fastrun import *
import indycar.model.stint_simulator as stint


# In[4]:


def simulation(datasetid, testevent, taskid, runts, expid, predictionlen, datamode, loopcnt, featuremode = stint.FEATURE_STATUS):
    #
    # configurataion
    #
    # model path:  <_dataset_id>/<_task_id>-<trainid>/
    #_dataset_id = 'indy2013-2018-nocarid'
    stint.init()
    stint._dataset_id = datasetid
    stint._test_event = testevent
    #_test_event = 'Indy500-2019'

    stint._feature_mode = featuremode
    stint._context_ratio = 0.

    stint._task_id = taskid  # rank,laptime, the trained model's task
    stint._run_ts = runts   #COL_LAPTIME,COL_RANK
    stint._exp_id=expid  #rank, laptime, laptim2rank, timediff2rank... 

    stint._train_len = 40
    predictor = stint.load_model(predictionlen, 'oracle',trainid='2018')

    ret2 = {}
    for i in range(loopcnt):
        ret2[i] = stint.run_simulation_pred(predictor, predictionlen, stint.freq, datamode=datamode)

    acc = []
    for i in ret2.keys():
        df = ret2[i]
        _x = stint.get_evalret(df)
        acc.append(_x)

    b = np.array(acc)
    print(np.mean(b, axis=0))
    
    return b, ret2


# ### test low mode

# In[5]:

testcar = 12
runid = 0
loopcnt = 50
if len(sys.argv) > 1:
    testcar = int(sys.argv[1])

if len(sys.argv) > 2:
    runid = sys.argv[2]
if len(sys.argv) > 3:
    loopcnt = int(sys.argv[3])


print('testcar:', testcar, 'runid:', runid, 'loopcnt:', loopcnt)


stint._pitstrategy_lowmode = True
stint._pitstrategy_testcar = testcar
acc, ret = simulation('indy2013-2018', 'Indy500-2018', 
                    'timediff',stint.COL_TIMEDIFF,'timediff2rank',
                   2, stint.MODE_ORACLE_LAPONLY,loopcnt)

df = pd.concat(ret)
dftestcar_low = df[df['carno']==testcar]
df.to_csv(f'test-strategy-df-lowmode-c{testcar}-r{runid}.csv')

stint._pitstrategy_lowmode = False
stint._pitstrategy_testcar = testcar
acc, ret_high = simulation('indy2013-2018', 'Indy500-2018', 
                    'timediff',stint.COL_TIMEDIFF,'timediff2rank',
                   2, stint.MODE_ORACLE_LAPONLY,loopcnt)
df = pd.concat(ret_high)
dftestcar_high = df[df['carno']==testcar]
df.to_csv(f'test-strategy-df-highmode-c{testcar}-r{runid}.csv')

stint.get_evalret(dftestcar_low)
stint.get_evalret(dftestcar_high)

print('sample cnt:', len(dftestcar_low))

#check the difference between two distribution of pred_rank
# mode x category   pred_sign
f_obs = np.zeros((2, 3))

predsign = dftestcar_low.pred_sign
for idx, sign in enumerate([-1,0,1]):
    f_obs[0, idx] = np.sum(predsign == sign)
predsign = dftestcar_high.pred_sign
for idx, sign in enumerate([-1,0,1]):
    f_obs[1, idx] = np.sum(predsign == sign)

from scipy import stats
chi, pval, freedom = stats.chi2_contingency(f_obs)[0:3]

print('chi2 test:', chi, pval, freedom)


