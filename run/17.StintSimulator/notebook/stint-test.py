#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ipdb
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


#
# configurataion
#
# model path:  <_dataset_id>/<_task_id>-<trainid>/
#_dataset_id = 'indy2013-2018-nocarid'
stint.init()
stint._dataset_id = 'indy2013-2018-nocarid-context40'
stint._test_event = 'Indy500-2019'
#_test_event = 'Indy500-2019'

stint._feature_mode = stint.FEATURE_STATUS
stint._context_ratio = 0.

stint._task_id = 'timediff'  # rank,laptime, the trained model's task
stint._run_ts = stint.COL_TIMEDIFF   #COL_LAPTIME,COL_RANK
stint._exp_id='timediff2rank'  #rank, laptime, laptim2rank, timediff2rank... 

stint._train_len = 40
predictor = stint.load_model(2, 'oracle',trainid='2018')


# In[155]:


df = stint.run_simulation_pred(predictor, 2, stint.freq, datamode=stint.MODE_ORACLE)


