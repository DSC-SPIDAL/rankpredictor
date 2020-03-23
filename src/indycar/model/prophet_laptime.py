import mxnet as mx
from mxnet import gluon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle

with open('sim-indy500-laptime-2018.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    laptime_data = pickle.load(f, encoding='latin1')
        
    print(f"number of runs: {len(laptime_data)}")
    from gluonts.dataset.common import ListDataset
    prediction_length = 50
    freq = "5m"
    start = pd.Timestamp("01-01-2019", freq=freq)  # can be different for each time series

    train_set = []
    test_set = []
    cardinality = []
    #_data: eventid, carids, laptime array
    for _data in laptime_data:

        _train = [{'target': _data[2][rowid, :-prediction_length].astype(np.float32), 'start': start, 
            'feat_static_cat': rowid}             
            for rowid in range(_data[2].shape[0]) ]
        _test = [{'target': _data[2][rowid, :].astype(np.float32), 'start': start, 'feat_static_cat': rowid} 
            for rowid in range(_data[2].shape[0]) ]

    train_set.extend(_train)
    test_set.extend(_test)
        
    train_ds = ListDataset(train_set, freq=freq)
    test_ds = ListDataset(test_set, freq=freq)  


from gluonts.model.prophet import ProphetPredictor

predictor = ProphetPredictor(freq= freq, prediction_length = prediction_length)
predictions = list(predictor.predict(test_ds))
