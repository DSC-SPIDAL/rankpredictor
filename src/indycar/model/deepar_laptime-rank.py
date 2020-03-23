#!/usr/bin/env python
# coding: utf-8

# # DeepAR on laptime&rank dataset
# 
# laptime&rank dataset
# <eventid, carids, laptime (totalcars x totallaps), rank (totalcars x totallaps)>; filled with NaN

# In[1]:


# Third-party imports
import mxnet as mx
from mxnet import gluon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

from pathlib import Path


print("deepar.py <ts_type> <epochs> <gpuid>")

import sys
if len(sys.argv)!=4:
    exit(-1)

ts_type = int(sys.argv[1])
epochs = int(sys.argv[2])
gpudevice = int(sys.argv[3])

runid='deepar_indy_e%d_ts%d'%(epochs, ts_type)

# In[2]:


import pickle
with open('laptime_rank-2018.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    global_carids, laptime_data = pickle.load(f, encoding='latin1')


# In[3]:


events = ['Phoenix','Indy500','Texas','Iowa','Pocono','Gateway']
events_id={key:idx for idx, key in enumerate(events)}


# In[4]:


print(f"events: {events}")


# To download one of the built-in datasets, simply call get_dataset with one of the above names. GluonTS can re-use the saved dataset so that it does not need to be downloaded again: simply set `regenerate=False`.

# In[5]:


laptime_data[2][2].astype(np.float32)


# In[6]:


# global configuration
prediction_length = 50
freq = "1H"
cardinality = [len(global_carids)]
TS_LAPTIME=2
TS_RANK=3
run_ts = ts_type


# In[7]:


from gluonts.dataset.common import ListDataset
start = pd.Timestamp("01-01-2019", freq=freq)  # can be different for each time series

train_set = []
test_set = []

#_data: eventid, carids, laptime array
for _data in laptime_data:
    #_train = [{'target': x.astype(np.float32), 'start': start} 
    #        for x in _data[2][:, :-prediction_length]]
    #_test = [{'target': x.astype(np.float32), 'start': start} 
    #        for x in _data[2]]
    
    #map rowid -> carno -> global_carid
    #carids = list(_data[1].values())
    #global_carid = global_carids[_data[1][rowid]]
    
    _train = [{'target': _data[run_ts][rowid, :-prediction_length].astype(np.float32), 'start': start, 
               'feat_static_cat': global_carids[_data[1][rowid]]}             
            for rowid in range(_data[run_ts].shape[0]) ]
    _test = [{'target': _data[run_ts][rowid, :].astype(np.float32), 'start': start, 
              'feat_static_cat': global_carids[_data[1][rowid]]} 
            for rowid in range(_data[run_ts].shape[0]) ]
    
    train_set.extend(_train)
    test_set.extend(_test)


# In[8]:


# train dataset: cut the last window of length "prediction_length", add "target" and "start" fields
train_ds = ListDataset(train_set, freq=freq)
# test dataset: use the whole dataset, add "target" and "start" fields
test_ds = ListDataset(test_set, freq=freq)


# In general, the datasets provided by GluonTS are objects that consists of three main members:
# 
# - `dataset.train` is an iterable collection of data entries used for training. Each entry corresponds to one time series
# - `dataset.test` is an iterable collection of data entries used for inference. The test dataset is an extended version of the train dataset that contains a window in the end of each time series that was not seen during training. This window has length equal to the recommended prediction length.
# - `dataset.metadata` contains metadata of the dataset such as the frequency of the time series, a recommended prediction horizon, associated features, etc.

# In[9]:


from gluonts.dataset.util import to_pandas

# ## Training an existing model (`Estimator`)
# 
# GluonTS comes with a number of pre-built models. All the user needs to do is configure some hyperparameters. The existing models focus on (but are not limited to) probabilistic forecasting. Probabilistic forecasts are predictions in the form of a probability distribution, rather than simply a single point estimate.
# 
# We will begin with GulonTS's pre-built feedforward neural network estimator, a simple but powerful forecasting model. We will use this model to demonstrate the process of training a model, producing forecasts, and evaluating the results.
# 
# GluonTS's built-in feedforward neural network (`SimpleFeedForwardEstimator`) accepts an input window of length `context_length` and predicts the distribution of the values of the subsequent `prediction_length` values. In GluonTS parlance, the feedforward neural network model is an example of `Estimator`. In GluonTS, `Estimator` objects represent a forecasting model as well as details such as its coefficients, weights, etc.
# 
# In general, each estimator (pre-built or custom) is configured by a number of hyperparameters that can be either common (but not binding) among all estimators (e.g., the `prediction_length`) or specific for the particular estimator (e.g., number of layers for a neural network or the stride in a CNN).
# 
# Finally, each estimator is configured by a `Trainer`, which defines how the model will be trained i.e., the number of epochs, the learning rate, etc.

# In[12]:


from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer


# In[13]:


estimator = DeepAREstimator(
    prediction_length=prediction_length,
    context_length=2*prediction_length,
    use_feat_static_cat=True,
    cardinality=cardinality,
    freq=freq,
    trainer=Trainer(ctx="gpu(%d)"%gpudevice, 
                    epochs="%d"%epochs, 
                    learning_rate=1e-3, 
                    num_batches_per_epoch=64
                   )
)


# After specifying our estimator with all the necessary hyperparameters we can train it using our training dataset `dataset.train` by invoking the `train` method of the estimator. The training algorithm returns a fitted model (or a `Predictor` in GluonTS parlance) that can be used to construct forecasts.

# In[14]:


predictor = estimator.train(train_ds)

outputfile=runid
if not os.path.exists(outputfile):
    os.mkdir(outputfile)

predictor.serialize(Path(outputfile))


# With a predictor in hand, we can now predict the last window of the `dataset.test` and evaluate our model's performance.
# 
# GluonTS comes with the `make_evaluation_predictions` function that automates the process of prediction and model evaluation. Roughly, this function performs the following steps:
# 
# - Removes the final window of length `prediction_length` of the `dataset.test` that we want to predict
# - The estimator uses the remaining data to predict (in the form of sample paths) the "future" window that was just removed
# - The module outputs the forecast sample paths and the `dataset.test` (as python generator objects)

# In[15]:


from gluonts.evaluation.backtest import make_evaluation_predictions


# In[16]:


forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,  # test dataset
    predictor=predictor,  # predictor
    num_samples=100,  # number of sample paths we want for evaluation
)


# First, we can convert these generators to lists to ease the subsequent computations.

# In[17]:


forecasts = list(forecast_it)
tss = list(ts_it)



# Indy500 Car 12 WillPower
ts_entry = tss[52]
# first entry of the forecast list
forecast_entry = forecasts[52]



# In[28]:


print(f"Number of sample paths: {forecast_entry.num_samples}")
print(f"Dimension of samples: {forecast_entry.samples.shape}")
print(f"Start date of the forecast window: {forecast_entry.start_date}")
print(f"Frequency of the time series: {forecast_entry.freq}")


# We can also do calculations to summarize the sample paths, such computing the mean or a quantile for each of the 48 time steps in the forecast window.

# In[29]:


print(f"Mean of the future window:\n {forecast_entry.mean}")
print(f"0.5-quantile (median) of the future window:\n {forecast_entry.quantile(0.5)}")


# `Forecast` objects have a `plot` method that can summarize the forecast paths as the mean, prediction intervals, etc. The prediction intervals are shaded in different colors as a "fan chart".

# In[30]:


def plot_prob_forecasts(ts_entry, forecast_entry):
    plot_length = 150 
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.savefig(runid + '.pdf')


# In[31]:


plot_prob_forecasts(ts_entry, forecast_entry)


# We can also evaluate the quality of our forecasts numerically. In GluonTS, the `Evaluator` class can compute aggregate performance metrics, as well as metrics per time series (which can be useful for analyzing performance across heterogeneous time series).

# In[32]:


from gluonts.evaluation import Evaluator


# In[33]:


evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))


# Aggregate metrics aggregate both across time-steps and across time series.

# In[34]:


print(json.dumps(agg_metrics, indent=4))


# Individual metrics are aggregated only across time-steps.

# In[35]:


item_metrics.head()


