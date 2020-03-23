#!/usr/bin/env python
# coding: utf-8

# # Prophet on laptime&rank dataset
# 
# https://gluon-ts.mxnet.io/api/gluonts/gluonts.model.prophet.html
# 
# laptime&rank dataset
# <eventid, carids, laptime (totalcars x totallaps), rank (totalcars x totallaps)>; filled with NaN

# In[1]:


# Third-party imports
get_ipython().run_line_magic('matplotlib', 'inline')
import mxnet as mx
from mxnet import gluon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


# In[2]:


### test on one run
from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
from gluonts.model.prophet import ProphetPredictor
from gluonts.model.r_forecast import RForecastPredictor
from gluonts.dataset.util import to_pandas
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

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

#
# remove NaN at the tail
# there should be no nans in the middle of the ts
def make_dataset(runs, prediction_length, freq, 
                       run_ts=2, train_ratio = 0.8,
                       use_global_dict = True):
    """
    split the ts to train and test part by the ratio
    """    
    start = pd.Timestamp("01-01-2019", freq=freq)  # can be different for each time series

    train_set = []
    test_set = []
    
    #select run
    if runs>=0:
        _laptime_data = [laptime_data[runs].copy()]
    else:
        _laptime_data = laptime_data.copy()
    
   
    #_data: eventid, carids, laptime array
    for _data in _laptime_data:
        _train = []
        _test = []
        
        #statistics on the ts length
        ts_len = [ x.shape[0] for x in _data[run_ts]]
        train_len = int(np.max(ts_len) * train_ratio)

        print(f'====event:{events[_data[0]]}, train_len={train_len}, max_len={np.max(ts_len)}, min_len={np.min(ts_len)}')

        for rowid in range(_data[run_ts].shape[0]):
            rec = _data[run_ts][rowid, :].copy()
            #remove nan
            nans, x= nan_helper(rec)
            nan_count = np.sum(nans)             
            rec = rec[~np.isnan(rec)]
            
            # remove short ts
            totallen = rec.shape[0]
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
                
            # split and add to dataset record
            _train.append({'target': rec[:train_len].astype(np.float32), 
                            'start': start, 
                            'feat_static_cat': carid}
                          )
            
            # multiple test ts(rolling window as half of the prediction_length)
            test_rec_cnt = 0
            for endpos in range(totallen, train_len+prediction_length, -int(prediction_length/2)):
                _test.append({'target': rec[:endpos].astype(np.float32), 
                            'start': start, 
                            'feat_static_cat': carid}
                          )   
                test_rec_cnt += 1
            
            #add one ts
            print(f'carno:{carno}, totallen:{totallen}, nancount:{nan_count}, test_reccnt:{test_rec_cnt}')

        train_set.extend(_train)
        test_set.extend(_test)

    # train dataset: cut the last window of length "prediction_length", add "target" and "start" fields
    train_ds = ListDataset(train_set, freq=freq)
    # test dataset: use the whole dataset, add "target" and "start" fields
    test_ds = ListDataset(test_set, freq=freq)    
    return train_ds, test_ds, train_set, test_set

def evaluate_model(test_ds,predictor, output=''):
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
    ts_entry = tss[7]
    forecast_entry = forecasts[7]
    plot_prob_forecasts(ts_entry, forecast_entry, output)     
    
def plot_prob_forecasts(ts_entry, forecast_entry, output):
    plot_length = 50 
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    if output:
        plt.savefig(output + '.pdf')
    plt.show() 
    
# prophet
def run_prophet(dataset, prediction_length,freq, output=''):
    predictor = ProphetPredictor(freq= freq, prediction_length = prediction_length)
    evaluate_model(dataset, predictor, output)
    
# ets
def run_ets(dataset, prediction_length,freq, output=''):
    predictor = RForecastPredictor(method_name='ets',freq= freq, prediction_length = prediction_length)
    evaluate_model(dataset, predictor, output)

# arima
def run_ets(dataset, prediction_length,freq, output=''):
    predictor = RForecastPredictor(method_name='arima',freq= freq, prediction_length = prediction_length)
    evaluate_model(dataset, predictor, output)


# ## Indy Dataset 
# 

# In[3]:


import pickle
### load indy
with open('laptime_rank-2018.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    global_carids, laptime_data_indy = pickle.load(f, encoding='latin1')


# In[4]:


events = ['Phoenix','Indy500','Texas','Iowa','Pocono','Gateway']
events_id={key:idx for idx, key in enumerate(events)}


# In[5]:


print(f"events: {events}")


# In[6]:


laptime_data = laptime_data_indy
laptime_data[2][2].astype(np.float32)


# In[7]:


# global configuration
prediction_length = 5
freq = "1min"
cardinality = [len(global_carids)]
TS_LAPTIME=2
TS_RANK=3
run_ts = TS_LAPTIME


# In[8]:


#run on indy500 dataset
train_ds, test_ds,_,_ = make_dataset(1, prediction_length,freq)


# In[9]:


get_ipython().run_line_magic('debug', '')


# In[ ]:


output = f'Prophet-indy-indy500'
run_prophet(test_ds, prediction_length, freq, output)
output = f'ETS-indy-indy500'
run_ets(test_ds, prediction_length, freq, output)
output = f'ARIMA-indy-indy500'
run_arima(test_ds, prediction_length, freq, output)


# In[ ]:





# In[ ]:


# test all 
train_ds, test_ds,_,_ = make_dataset(-1, prediction_length,freq)
output = f'Prophet-indy-all'
run_prophet(test_ds, prediction_length, freq, output)
output = f'ETS-indy-all'
run_ets(test_ds, prediction_length, freq, output)
output = f'ARIMA-indy-all'
run_arima(test_ds, prediction_length, freq, output)


# In[ ]:


entry = next(iter(train_ds))
train_series = to_pandas(entry)
entry = next(iter(test_ds))
test_series = to_pandas(entry)
test_series.plot()
plt.axvline(train_series.index[-1], color='r') # end of train dataset
plt.grid(which="both")
plt.legend(["test series", "end of train series"], loc="upper left")
plt.show()


# Individual metrics are aggregated only across time-steps.

# In[ ]:


item_metrics.head()


# In[ ]:


item_metrics.plot(x='MSIS', y='MASE', kind='scatter')
plt.grid(which="both")
plt.show()


# In[ ]:





# ### test on sim-indy dataset

# In[ ]:


import pickle
with open('sim-indy500-laptime-2018.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    laptime_data_simindy = pickle.load(f, encoding='latin1')
    
print(f"number of runs: {len(laptime_data)}")


# In[ ]:


laptime_data = laptime_data_simindy
#run on indy500 dataset
train_ds, test_ds,_,_ = make_dataset(1, prediction_length,freq, use_global_dict=False)
output = f'Prophet-simindy-indy500'
run_prophet(test_ds, prediction_length, freq, output)


# In[ ]:


get_ipython().run_line_magic('debug', '')


# In[ ]:


output = f'ETS-simindy-indy500'
run_ets(test_ds, prediction_length, freq, output)
output = f'ARIMA-simindy-indy500'
run_arima(test_ds, prediction_length, freq, output)


# In[ ]:


# test all 
#train_ds, test_ds,_,_ = make_dataset(-1, prediction_length,freq)
#output = f'Prophet-simindy-all'
#run_prophet(test_ds, prediction_length, freq, output)
#output = f'ETS-simindy-all'
#run_ets(test_ds, prediction_length, freq, output)
#output = f'ARIMA-simindy-all'
#run_arima(test_ds, prediction_length, freq, output)


# In[ ]:




