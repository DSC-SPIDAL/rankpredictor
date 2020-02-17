#!/usr/bin/env python
# coding: utf-8

# # Prophet on telemetry ts dataset
# 
# refer to telemetry_dataset_gluonts

# In[1]:


# Third-party imports
import mxnet as mx
from mxnet import gluon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os,sys
from optparse import OptionParser
import pickle
from pathlib import Path


# In[2]:


### test on one run
from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
from gluonts.model.prophet import ProphetPredictor

def evaluate_model(test_ds,predictor):
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,  # test dataset
        predictor=predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)   

    # evaluation
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))
    print(json.dumps(agg_metrics, indent=4))     
    
    #plot a example
    #ts_entry = tss[7]
    #forecast_entry = forecasts[7]
    #plot_prob_forecasts(ts_entry, forecast_entry) 
    
    return tss, forecasts
 
    
def plot_prob_forecasts(ts_entry, forecast_entry, outputfile):
    plot_length = 800 
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    #ts_entry[-plot_length:].dropna().plot(ax=ax)  # plot the time series
    plt.plot(ts_entry[-plot_length:].index, ts_entry[-plot_length:].values)
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.savefig(outputfile + '.pdf')
    
# prophet
def evaluate_prophet(test_ds,prediction_length,freq):
    predictor = ProphetPredictor(freq= freq, prediction_length = prediction_length)
    return evaluate_model(test_ds, predictor)
  
    
def run_prophet(prediction_length,freq):
    train_ds, test_ds = make_dataset(runs, prediction_length,freq)
    evaluate_prophet(test_ds,prediction_length,freq)

def run_prophet_nonan(prediction_length,freq):
    train_ds, test_ds = make_dataset_nonan(prediction_length,freq)
    evaluate_prophet(test_ds,prediction_length,freq)
    
   


# ## Datasets
# 

# In[13]:


import pickle
### load indy
with open('telemetry-gluonts-all-2018.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    freq, prediction_length, cardinality,train_ds, test_ds = pickle.load(f, encoding='latin1')
    #freq, train_set, test_set = pickle.load(f, encoding='latin1')
    #train_ds = ListDataset(train_set, freq=freq)    
    #test_ds = ListDataset(test_set, freq=freq)  


# In[4]:


events = ['Phoenix','Indy500','Texas','Iowa','Pocono','Gateway']
events_id={key:idx for idx, key in enumerate(events)}


# In[5]:


print(f"events: {events}")


# In[14]:


from gluonts.dataset.util import to_pandas
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# global configuration
TS_VSPEED=1
TS_DISTANCE=2
run_ts = TS_VSPEED


# In[16]:


tss, forecast =evaluate_prophet(test_ds,prediction_length,freq)
ts_entry = tss[7]
forecast_entry = forecast[7]
plot_prob_forecasts(ts_entry, forecast_entry, 'prophet-tele-00') 


# In[12]:


# test all 
#run_prophet_nonan(-1, 50, '1D')


# ### R-predictor

# In[17]:


from gluonts.model.r_forecast import RForecastPredictor
est = RForecastPredictor(method_name='ets',freq= freq, prediction_length = prediction_length)
arima = RForecastPredictor(method_name='arima',freq= freq, prediction_length = prediction_length)
#
##train_ds, test_ds = make_dataset_nonan(1, prediction_length,freq)
##train_ds, test_ds = make_dataset(prediction_length,freq)
#
tss, forecast = evaluate_model(test_ds, est)
#
#
## In[18]:
#
#
ts_entry = tss[7]
forecast_entry = forecast[7]
plot_prob_forecasts(ts_entry, forecast_entry, 'ets-tele-00')
#
#
## In[19]:
#
#
tss, forecast = evaluate_model(test_ds, arima)
ts_entry = tss[7]
forecast_entry = forecast[7]
plot_prob_forecasts(ts_entry, forecast_entry,'arima-tele-00')
#
#
## ### DeepAR
#
## In[21]:
#
#
#with open('telemetry-2018.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
#    global_carids, telemetry_data_indy = pickle.load(f, encoding='latin1')


exit(0)


from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
#cardinality = [len(global_carids)]
estimator = DeepAREstimator(
    prediction_length=prediction_length,
    context_length = 3*prediction_length,
    use_feat_static_cat=True,
    cardinality=cardinality,
    freq=freq,
    trainer=Trainer(ctx="gpu(2)", 
                    epochs=500, 
                    learning_rate=1e-3, 
                    num_batches_per_epoch=64
                   )
)


# In[ ]:


#train_ds, test_ds, train_set, test_set = make_dataset_interpolate(prediction_length,freq)
#train_ds, test_ds, train_set, test_set = make_dataset_interpolate(prediction_length,'1S')


# In[23]:


predictor = estimator.train(train_ds)

modeldir = 'deepar-tele'
if not os.path.exists(modeldir):
    os.mkdir(modeldir)

predictor.serialize(Path(modeldir))


# In[24]:


from gluonts.evaluation.backtest import make_evaluation_predictions
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,  # test dataset
    predictor=predictor,  # predictor
    num_samples=100,  # number of sample paths we want for evaluation
)


# In[25]:


forecasts = list(forecast_it)


# In[26]:


tss = list(ts_it)


# In[ ]:


ts_entry = tss[7]
forecast_entry = forecasts[7]
plot_prob_forecasts(ts_entry, forecast_entry, 'deepar-tele-00')


evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))
print(json.dumps(agg_metrics, indent=4))

