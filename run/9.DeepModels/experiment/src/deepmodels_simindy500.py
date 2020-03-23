#!/usr/bin/env python
# coding: utf-8
"""
Deep Models on the Indy500 simulation dataset

simulation dataset:

laptime&rank dataset <eventid, carids, laptime (totalcars x totallaps), rank (totalcars x totallaps)>; filled with NaN

deep models:

deepAR, deepstate, deepFactor



"""
# # DeepAR on simulation indy500 laptime dataset
# 
# laptime dataset
# <eventid, carids, laptime (totalcars x totallaps)>

# Third-party imports
import mxnet as mx
from mxnet import gluon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import logging
import os,sys
from optparse import OptionParser


import pickle
from pathlib import Path
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deep_factor import DeepFactorEstimator
from gluonts.model.deepstate import DeepStateEstimator
from gluonts.trainer import Trainer
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator

logger = logging.getLogger(__name__)
 

#global variables
prediction_length = 50
context_length = 100
test_length = 50
freq = "1H"
 
def load_dataset(inputfile):

    with open(inputfile, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        laptime_data = pickle.load(f, encoding='latin1')
    
    print(f"number of runs: {len(laptime_data)}")
    
    start = pd.Timestamp("01-01-2019", freq=freq)  # can be different for each time series
    
    train_set = []
    test_set = []
    cardinality = []
    #_data: eventid, carids, laptime array
    for _data in laptime_data:
        #_train = [{'target': x.astype(np.float32), 'start': start} 
        #        for x in _data[2][:, :-prediction_length]]
        #_test = [{'target': x.astype(np.float32), 'start': start} 
        #        for x in _data[2]]
        carids = list(_data[1].values())
        _train = [{'target': _data[2][rowid, :-test_length].astype(np.float32), 'start': start, 
                   'feat_static_cat': rowid}
                for rowid in range(_data[2].shape[0]) ]
        _test = [{'target': _data[2][rowid, :].astype(np.float32), 'start': start, 'feat_static_cat': rowid} 
                for rowid in range(_data[2].shape[0]) ]
        
        train_set.extend(_train)
        test_set.extend(_test)
        cardinality.append(len(carids))
    # train dataset: cut the last window of length "test_length", add "target" and "start" fields
    train_ds = ListDataset(train_set, freq=freq)
    # test dataset: use the whole dataset, add "target" and "start" fields
    test_ds = ListDataset(test_set, freq=freq)

    return train_ds, test_ds





def plot_prob_forecasts(ts_entry, forecast_entry, outputfile):
    plot_length = 150 
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    figcnt = len(ts_entry)

    #fig, axs = plt.subplots(figcnt, 1, figsize=(10, 7))

    #for idx in range(figcnt):

    #    ts_entry[idx][-plot_length:].plot(ax=axs[idx])  # plot the time series
    #    forecast_entry[idx].plot(prediction_intervals=prediction_intervals, color='g')
    #    axs[idx].grid(which="both")
    #    axs[idx].legend(legend, loc="upper left")
    
    for idx in range(figcnt):
        fig, axs = plt.subplots(1, 1, figsize=(10, 7))

        ts_entry[idx][-plot_length:].plot(ax=axs)  # plot the time series
        forecast_entry[idx].plot(prediction_intervals=prediction_intervals, color='g')
        plt.grid(which="both")
        plt.legend(legend, loc="upper left")
        plt.savefig(outputfile + '-%d.pdf'%idx)


def evaluate_model(estimator, train_ds, test_ds, outputfile):
    predictor = estimator.train(train_ds)
    
    #if not os.path.exists(outputfile):
    #    os.mkdir(outputfile)

    #predictor.serialize(Path(outputfile))

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,  # test dataset
        predictor=predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
    )
    
    forecasts = list(forecast_it)
    tss = list(ts_it)
    logger.info(f'tss len={len(tss)}, forecasts len={len(forecasts)}')
    
    # car12@rank1, car1@rank16, car7@rank33, the index is 7,0,4 accordingly
    # Indy500 Car 12 WillPower
    ts_entry = [tss[7],tss[0],tss[4]]
    forecast_entry = [forecasts[7],forecasts[0],forecasts[4]]

    plot_prob_forecasts(ts_entry, forecast_entry, outputfile)
    
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))
    
    
    logger.info(json.dumps(agg_metrics, indent=4))
    

def init_estimator(model, gpuid, epochs=100):

    if model == 'deepAR':
        estimator = DeepAREstimator(
            prediction_length=prediction_length,
            context_length= context_length,
            use_feat_static_cat=True,
            cardinality=[33],
            freq=freq,
            trainer=Trainer(ctx="gpu(%s)"%gpuid, 
                            epochs=epochs, 
                            learning_rate=1e-3, 
                            num_batches_per_epoch=100
                           )
        )
    elif model == 'simpleFF':
        estimator = SimpleFeedForwardEstimator(
            num_hidden_dimensions=[10],
            prediction_length=prediction_length,
            context_length= context_length,
            freq=freq,
            trainer=Trainer(ctx="gpu(%s)"%gpuid, 
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
            trainer=Trainer(ctx="gpu(%s)"%gpuid, 
                            epochs=epochs, 
                            learning_rate=1e-3, 
                            num_batches_per_epoch=100
                           )
        )
    elif model == 'deepState':
        estimator = DeepStateEstimator(
            prediction_length=prediction_length,
            use_feat_static_cat=True,
            cardinality=[33],
            freq=freq,
            trainer=Trainer(ctx="gpu(%s)"%gpuid, 
                            epochs=epochs, 
                            learning_rate=1e-3, 
                            num_batches_per_epoch=100
                           )
        )
        
    else:
        logger.error('model %s not support yet, quit', model)
        sys.exit(-1)


    return estimator

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    # logging configure
    import logging.config
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # cmd argument parser
    usage = 'deepar_simindy500.py --epochs epochs --input inputpicklefile --output outputfile'
    parser = OptionParser(usage)
    parser.add_option("--input", dest="inputfile", default='sim-indy500-laptime-2018.pickle')
    parser.add_option("--output", dest="outputfile")
    parser.add_option("--epochs", dest="epochs", default=100)
    parser.add_option("--model", dest="model", default="deepAR")
    parser.add_option("--gpuid", dest="gpuid", default=0)
    parser.add_option("--contextlen", dest="contextlen", default=100)
    parser.add_option("--predictionlen", dest="predictionlen", default=50)
    parser.add_option("--testlen", dest="testlen", default=50)

    opt, args = parser.parse_args()

    #set the global length
    prediction_length = int(opt.predictionlen)
    context_length = int(opt.contextlen)
    test_length = int(opt.testlen)
    runid = f'-i{opt.outputfile}-e{opt.epochs}-m{opt.model}-p{opt.predictionlen}-c{opt.contextlen}-t{opt.testlen}'
    logger.info("runid=%s", runid)
            

    train_ds, test_ds = load_dataset(opt.inputfile)
    estimator = init_estimator(opt.model, opt.gpuid, opt.epochs)

    evaluate_model(estimator, train_ds, test_ds, opt.outputfile)





