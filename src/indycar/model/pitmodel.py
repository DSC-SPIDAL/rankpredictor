#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import random
import mxnet as mx
from mxnet import gluon
import pickle
import json
from gluonts.dataset.common import ListDataset
from gluonts.dataset.util import to_pandas

import inspect
from scipy import stats
from pathlib import Path 
from sklearn.metrics import mean_squared_error
from gluonts.dataset.util import to_pandas
from pathlib import Path
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator, MultivariateEvaluator
from gluonts.model.predictor import Predictor
from gluonts.distribution.neg_binomial import NegativeBinomialOutput
from gluonts.distribution.student_t import StudentTOutput
from gluonts.model.forecast import SampleForecast

from indycar.model.mlp import MLPEstimator

class PitModelBase():

    def __init__(self, modelfile=''):
        self.model = {}
        self.name = ''
        self.keys = {}
        
        if modelfile:
            self.load_model(modelfile)

    def load_model(self, modelfile):
        with open(modelfile, 'rb') as f:
            self.name, self.model = pickle.load(f, encoding='latin1')
            print(f'init model:{self.name}')

    def save_keys(self, keyfile):
        with open(keyfile, 'wb') as f:
            savedata = self.keys
            pickle.dump(savedata, f, pickle.HIGHEST_PROTOCOL)        
        print(f'save keys to {keyfile}.')

    def load_keys(self, keyfile):
        with open(keyfile, 'rb') as f:
            self.keys = pickle.load(f, encoding='latin1')
            print(f'load {len(data)} keys from {keyfile}')
            #self.keys = {}
            #for d in data:
            #    self.keys[d] = 1

    def save_model(self, modelname, test_ds, forecasts, scaler):
        pass

    def predict(self, *args):
        pass

    def forecast_ds(self, test_ds, forecasts):
        """
        test_ds as testset, the unsclaed input
        forecasts ; the template
        """
        
        plen = len(test_ds)
        sample_cnt = forecasts[0].samples.shape[0]
        assert(plen == len(forecasts))
        

        #build a new forecasts object
        nf = []
        for fc in forecasts:
            nfc = SampleForecast(samples = np.zeros_like(fc.samples), 
                                 freq=fc.freq, start_date=fc.start_date)
            nf.append(nfc)
        
        for idx, rec in enumerate(test_ds):
            feat = rec[1:]
                    
            onecast = np.zeros((sample_cnt))
            for i in range(sample_cnt):
                onecast[i] = self.predict(feat[0], feat[[1]])
        
            nf[idx].samples = onecast

        return nf


### pitmodel

class PitModelSimple(PitModelBase):
    # this is the perfect empirical pit model for Indy500 2018
    pit_model_all = [[33, 32, 35, 32, 35, 34, 35, 34, 37, 32, 37, 30, 33, 36, 35, 33, 36, 30, 31, 33, 36, 37, 35, 34, 34, 33, 37, 35, 39, 32, 36, 35, 34, 32, 36, 32, 31, 36, 33, 33, 35, 37, 40, 32, 32, 34, 35, 36, 33, 37, 35, 37, 34, 35, 39, 32, 31, 37, 32, 35, 36, 39, 35, 36, 34, 35, 33, 33, 34, 32, 33, 34],
                [45, 44, 46, 44, 43, 46, 45, 43, 41, 48, 46, 43, 47, 45, 49, 44, 48, 42, 44, 46, 45, 45, 43, 44, 44, 43, 46]]
    pit_model_top8 = [[33, 32, 35, 33, 36, 33, 36, 33, 37, 35, 36, 33, 37, 34],
                 [46, 45, 43, 48, 46, 45, 45, 43]]
 
    def __init__(self, modelfile='', top8 = False, retry = 10):
        super().__init__(modelfile)

        self.retry = retry
        if top8:
            self.model = self.pit_model_top8
        else:
            self.model = self.pit_model_all

    def predict(self, *args):

        retry = 0
        caution_laps_instint = args[0]
        laps_instint = args[1]

        key = '-'.join([str(int(x)) for x in args])
        self.keys[key] = 1

        while retry < self.retry:
            if caution_laps_instint <= 10:
                #use low model
                pred_pit_laps = random.choice(self.model[0])
            else:
                pred_pit_laps = random.choice(self.model[1])
    
            if pred_pit_laps <= laps_instint:
                retry += 1
                if retry == 10:
                    pred_pit_laps = laps_instint + 1
                continue
            else:
                break

        return pred_pit_laps


class PitModelMLP(PitModelBase):
    """
     <caution_lap, pitage> -> [distribution]    
     distribution := sorted cdf [val:probability, val2:p2, ...]
         [0,:] -> val
         [1,:] -> cdf p

     no scaler, raw feat and target

    """
    def __init__(self, modelfile=''):
        super().__init__(modelfile)

    def save_model(self, modelname, test_ds, forecasts, scaler):

        model = {}

        #get the sclaer for the first column(lap2nextpit)
        sc, scf = '', ''
        if isinstance(scaler, StandardScaler):
            sc = StandardScaler()
            sc.scale_ = scaler.scale_[0]
            sc.mean_ = scaler.mean_[0]
            sc.var_ = scaler.var_[0]

            scf = StandardScaler()
            scf.scale_ = scaler.scale_[1:]
            scf.mean_ = scaler.mean_[1:]
            scf.var_ = scaler.var_[1:]

        
        for idx, rec in enumerate(test_ds):
            feat = rec[1:]
                
            key = '-'.join([str(int(x)) for x in feat])
            
            if not key in model:
            
                samples = forecasts[idx].samples.reshape(-1)
                
                if not isinstance(sc, str):
                    samples = sc.inverse_transform(samples)
                
                #force to prediction to be valid lap2nextpit
                samples = samples.astype(int)
                samples = samples[samples > 0]

                #
                valset = set(list(samples))
                plen = len(valset)
                distr = np.zeros((2, plen))
                distr[0, :] = sorted(valset)
                smap = {val:id for id, val in enumerate(distr[0, :])}
                for s in samples:
                    distr[1,smap[s]] += 1
                tsum = np.sum(distr[1,:])
                distr[1, :] /= tsum
                distr[1, :] = np.cumsum(distr[1, :])

                model[key] = distr
                
        #save model
        self.model = model
        self.name = modelname
        with open(modelname, 'wb') as f:
            savedata = [self.name, self.model]
            pickle.dump(savedata, f, pickle.HIGHEST_PROTOCOL)        
        print(f'save model {modelname} with {len(self.model)} keys.')
                
    def predict(self, *args):
        key = '-'.join([str(int(x)) for x in args])


        #if key in self.model:
        try:
            distr = self.model[key]
            
            #[0, 1.)
            p = np.random.random()  
            i = np.sum(distr[1,:] < p)
            
            # return totallen
            return int(distr[0,i]) + args[1]
        except:
            #exception
            #todo, backto special model
            print(f'ERROR: key {key} not found in model')
            return 1 + args[1]

