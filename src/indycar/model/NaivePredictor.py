#!/usr/bin/env python
# coding: utf-8
import mxnet as mx
from mxnet import gluon
import numpy as np
import json

from typing import Callable, Dict, Iterator, NamedTuple, Optional, List

from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.model.predictor import Predictor
from gluonts.model.forecast import SampleForecast


class NaivePredictor(Predictor):

    @validated()
    def __init__(self, 
                 freq: str,
                 prediction_length: int) -> None:
        self.prediction_length=prediction_length
        self.freq = freq
    
    def predict(
            self, dataset: Dataset, num_samples: int = 100, **kwargs
            ) -> Iterator[SampleForecast]:


        for entry in dataset:
            train_length = len(entry["target"])
            prediction_length = self.prediction_length
            start = entry["start"]
            target = entry["target"]
            feat_dynamic_real = entry.get("feat_dynamic_real", [])

            #forecast_samples = self._run_prophet(data, params)
            #target_dim = target.shape[0] if len(target.shape) > 1 else 1
            if len(target.shape) > 1:
                #multivariate
                target_dim = target.shape[0] 
                target_len = target.shape[1]
            else:
                target_dim = 1
                target_len = target.shape[0]

            if target_dim ==1 :
                forecast_samples = np.zeros((num_samples, prediction_length))
                #navie prediction with the last status of target
                #forecast_samples[:] = target[-prediction_length]
                forecast_samples[:] = target[-1]
            else:
                forecast_samples = np.zeros((num_samples, prediction_length, target_dim))

                #forecast_samples[:,:] = target[-prediction_length]
                forecast_samples[:,:] = target[-1]

            yield SampleForecast(
                samples=forecast_samples,
                start_date=start + target_len,
                freq=self.freq,
                )

