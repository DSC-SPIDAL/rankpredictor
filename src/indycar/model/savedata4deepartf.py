#!/usr/bin/env python
# coding: utf-8
"""
Gluonts Models on the Indy dataset

dataset:
freq, prediction_length, cardinality,train_ds, test_ds

models:
1. classical models
naive,
arima, ets, prophet

2. deep models
deepAR, deepstate, deepFactor
deepAR-Oracle


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

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from indycar.model.NaivePredictor import NaivePredictor
from indycar.model.deeparsavedata import DeepARSaveDataEstimator


logger = logging.getLogger(__name__)
 

#global variables
prediction_length = 50
context_length = 100
freq = "1H"
 

events = ['Phoenix','Indy500','Texas','Iowa','Pocono','Gateway']
events_id={key:idx for idx, key in enumerate(events)}

cardinality = [0]
TS_LAPTIME=2
TS_RANK=3

def load_dataset(inputfile):
    global freq, prediction_length, cardinality

    with open(inputfile, 'rb') as f:
        # have to specify it.
        freq, prediction_length, cardinality,train_ds, test_ds = pickle.load(f, encoding='latin1')
    
    logger.info(f"number of cars: {cardinality}")
    
    return train_ds, test_ds


def plot_prob_forecasts(ts_entry, forecast_entry, outputfile):

    plot_length = context_length 
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
        #ts_entry[idx][-plot_length:].plot(ax=axs)  # plot the time series
        #forecast_entry[idx].plot(prediction_intervals=prediction_intervals, color='g')
        ts_entry[idx].iloc[-plot_length:,0].plot(ax=axs)  # plot the time series
        forecast_entry[idx].copy_dim(0).plot(prediction_intervals=prediction_intervals, color='g')
        
        plt.grid(which="both")
        plt.legend(legend, loc="upper left")
        plt.savefig(outputfile + '-%d.pdf'%idx)

def evaluate_model_old(estimator, train_ds, test_ds, outputfile, samplecnt = 100):
    predictor = estimator.train(train_ds)
    
    if not os.path.exists(outputfile):
        os.mkdir(outputfile)

    predictor.serialize(Path(outputfile))

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,  # test dataset
        predictor=predictor,  # predictor
        num_samples=samplecnt,  # number of sample paths we want for evaluation
    )
    
    forecasts = list(forecast_it)
    tss = list(ts_it)
    logger.info(f'tss len={len(tss)}, forecasts len={len(forecasts)}')
    
    # car12@rank1, car1@rank16, car7@rank33, the index is 7,0,4 accordingly
    # Indy500 Car 12 WillPower
    #offset = 52-7
    offset = 0
    ts_entry = [tss[7+offset],tss[0+offset],tss[4+offset]]
    forecast_entry = [forecasts[7+offset],forecasts[0+offset],forecasts[4+offset]]

    plot_prob_forecasts(ts_entry, forecast_entry, outputfile)
    
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))
    
    
    logger.info(json.dumps(agg_metrics, indent=4))
    
def evaluate_model_uni(predictor, evaluator, test_ds, outputfile):

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
    #offset = 52-7
    offset = 0
    ts_entry = [tss[7+offset],tss[0+offset],tss[4+offset]]
    forecast_entry = [forecasts[7+offset],forecasts[0+offset],forecasts[4+offset]]

    #plot_prob_forecasts(ts_entry, forecast_entry, outputfile)
    
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))
    
    
    logger.info(json.dumps(agg_metrics, indent=4))
 

def evaluate_model(predictor, evaluator, test_ds, outputfile, samplecnt = 100):
    
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,  # test dataset
        predictor=predictor,  # predictor
        num_samples=samplecnt,  # number of sample paths we want for evaluation
    )
    
    forecasts = list(forecast_it)
    tss = list(ts_it)
    logger.info(f'tss len={len(tss)}, forecasts len={len(forecasts)}')
    
    #convert to univariate format
    # tss: <ts_len, #feature>
    # forecasts.sample: < 100, prediction_length, #feature>
   
    #tss_n = []
    #for ts in tss:
    #    tse = ts.to_numpy()
    #    tss_n.append(tse[:,0].reshape((tse.shape[0])))
    #cast_n = []
    #for fc in forecasts:
    #    nfc = fc
    #    fcs = fc.samples.shape
    #    nsamples = fc.samples[:,:,0].reshape((fcs[0], fcs[1]))
    #    nfc.samples = nsamples
    #    cast_n.append(nfc)
    #tss = tss_n
    #forecasts = cast_n


    # car12@rank1, car1@rank16, car7@rank33, the index is 7,0,4 accordingly
    # Indy500 Car 12 WillPower
    #offset = 52-7
    offset = 0
    ts_entry = [tss[7+offset],tss[0+offset],tss[4+offset]]
    forecast_entry = [forecasts[7+offset],forecasts[0+offset],forecasts[4+offset]]

    #debug
    #print(f'ts_entry shape:{ts_entry[0].shape}, forecast:{forecast_entry[0].samples.shape}')

    plot_prob_forecasts(ts_entry, forecast_entry, outputfile)
    
    #evaluator = MultivariateEvaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))
    
    
    logger.info(json.dumps(agg_metrics, indent=4))
    

def init_estimator(model, gpuid, epochs=100, batch_size = 32, 
        target_dim = 3, distr_output = None, use_feat_static = True):
    
    if int(gpuid) < 0:
        ctx = "cpu"
    else:
        ctx = "gpu(%s)"%gpuid

    if model == 'deepAR-Oracle':

        if use_feat_static:
            estimator = DeepARSaveDataEstimator(
                prediction_length=prediction_length,
                context_length= context_length,
                use_feat_static_cat=use_feat_static,
                cardinality=cardinality,
                use_feat_dynamic_real=True,
                distr_output = distr_output,
                freq=freq,
                trainer=Trainer(ctx=ctx, 
                                batch_size = batch_size,
                                epochs=epochs, 
                                learning_rate=1e-3, 
                                hybridize=False,
                                num_batches_per_epoch=100
                               )
                )
        else:
            estimator = DeepARSaveDataEstimator(
                prediction_length=prediction_length,
                context_length= context_length,
                use_feat_static_cat=use_feat_static,
                #cardinality=cardinality,
                use_feat_dynamic_real=True,
                distr_output = distr_output,
                freq=freq,
                trainer=Trainer(ctx=ctx, 
                                batch_size = batch_size,
                                epochs=epochs, 
                                learning_rate=1e-3, 
                                hybridize=False,
                                num_batches_per_epoch=100
                               )
                )
    elif model == 'deepARW-Oracle':

        if use_feat_static:
            estimator = DeepARWEstimator(
                prediction_length=prediction_length,
                context_length= context_length,
                use_feat_static_cat=use_feat_static,
                cardinality=cardinality,
                use_feat_dynamic_real=True,
                distr_output = distr_output,
                freq=freq,
                trainer=Trainer(ctx=ctx, 
                                batch_size = batch_size,
                                epochs=epochs, 
                                learning_rate=1e-3, 
                                num_batches_per_epoch=100
                               )
                )
        else:
            estimator = DeepARWEstimator(
                prediction_length=prediction_length,
                context_length= context_length,
                use_feat_static_cat=use_feat_static,
                #cardinality=cardinality,
                use_feat_dynamic_real=True,
                distr_output = distr_output,
                freq=freq,
                trainer=Trainer(ctx=ctx, 
                                batch_size = batch_size,
                                epochs=epochs, 
                                learning_rate=1e-3, 
                                num_batches_per_epoch=100
                               )
                )
            
    elif model == 'deepAR-nocarid':
        estimator = DeepAREstimator(
            prediction_length=prediction_length,
            context_length= context_length,
            use_feat_static_cat=use_feat_static,
            cardinality=cardinality,
            use_feat_dynamic_real=True,
            distr_output = distr_output,
            freq=freq,
            trainer=Trainer(ctx=ctx, 
                            batch_size = batch_size,
                            epochs=epochs, 
                            learning_rate=1e-3, 
                            num_batches_per_epoch=100
                           )
        )
    elif model == 'deepAR-multi':
        estimator = DeepAREstimator(
            prediction_length=prediction_length,
            context_length= context_length,
            use_feat_static_cat=use_feat_static,
            cardinality=cardinality,
            freq=freq,
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
        estimator = RForecastPredictor(method_name='arima',freq= freq, prediction_length = prediction_length, trunc_length = 200)
    elif model == 'naive':
        estimator = NaivePredictor(freq= freq, prediction_length = prediction_length)
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
    parser.add_option("--batch_size", dest="batch_size", default=32)
    #parser.add_option("--predictionlen", dest="predictionlen", default=50)
    #parser.add_option("--testlen", dest="testlen", default=50)
    parser.add_option("--nosave", dest="nosave", action="store_true", default=False)
    parser.add_option("--evalmode", dest="evalmode", action="store_true", default=False)
    parser.add_option("--distr_output", dest="distr_output", default='student')
    parser.add_option("--nocarid", dest="nocarid", action="store_true", default=False)

    #obsolete
    parser.add_option("--mode", dest="mode", default='train')
    parser.add_option("--savedata", dest="savedata", default='savedata')


    opt, args = parser.parse_args()

    #set the global length
    #prediction_length = int(opt.predictionlen)
    context_length = int(opt.contextlen)
    #test_length = int(opt.testlen)
    #ts_type = int(opt.ts_type)
    #train_ds, test_ds = load_dataset(opt.inputfile, ts_type)
    train_ds, test_ds = load_dataset(opt.inputfile)

    #get target dim
    entry = next(iter(train_ds))
    target_dim = entry['target'].shape
    target_dim = target_dim[0] if len(target_dim) > 1 else 1
    logger.info('target_dim:%s', target_dim)

    runid = f'-i{opt.outputfile}-e{opt.epochs}-m{opt.model}-p{prediction_length}-c{opt.contextlen}-f{freq}-dim{target_dim}-dstr{opt.distr_output}'
    logger.info("runid=%s", runid)
            

    # train
    classical_models = ['ets', 'arima', 'prophet', 'naive']

    distr_outputs ={'student':StudentTOutput(),
                    'negbin':NegativeBinomialOutput()
                    }
    if opt.distr_output in distr_outputs:
        distr_output = distr_outputs[opt.distr_output]
    else:
        logger.error('output distr no found:%s', opt.distr_output)
        exit(-1)

    use_feat_static = True
    if opt.nocarid:
        use_feat_static = False

    estimator = init_estimator(opt.model, opt.gpuid, 
            opt.epochs, opt.batch_size,target_dim, distr_output = distr_output,use_feat_static = use_feat_static)
    
    if opt.evalmode == False:
        if opt.model in classical_models:
            predictor = estimator
        else:
            predictor = estimator.train(train_ds)

            data = estimator.network.savedata
        #if not opt.nosave:
        #    if not os.path.exists(opt.outputfile):
        #        os.mkdir(opt.outputfile)
        #    
        #    logger.info('Start to save the model to %s', opt.outputfile)
        #    predictor.serialize(Path(opt.outputfile))
        #    logger.info('End of saving the model.')

    else:
        if not os.path.exists(opt.outputfile):
            logger.error(f'error:{outputfile} not exists')
            exit(-1)

        logger.info('Start to load the model from %s', opt.outputfile)
        predictor =  Predictor.deserialize(Path(opt.outputfile))
        logger.info('End of loading the model.')


    # evaluate
    if opt.evalmode == True:
        #if opt.multi!=0:
        if target_dim > 1:
            logger.info('Start MultivariateEvaluator')
            evaluator = MultivariateEvaluator(quantiles=[0.1, 0.5, 0.9])
        else:
            logger.info('Start Evaluator')
            evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])

        #forece to single item
        print('batch size:', predictor.batch_size)
        predictor.batch_size = 1
        print('batch size reset:', predictor.batch_size)
        evaluate_model(predictor, evaluator, test_ds, opt.outputfile, samplecnt=1)
        #evaluate_model_uni(predictor, evaluator, test_ds, opt.outputfile)

        #
        #predictor.prediction_net.rnn.summary()
        data = predictor.prediction_net.savedata

    #target = estimator.network.savetarget
    #other = estimator.network.saveother
    print('len(savedata):', data.keys())
    savefile = opt.savedata
    with open(savefile, 'wb') as f:
        savedata = [data['input'], data['target'],0]
        #pickle.dump([data,target,other], f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(savedata, f, pickle.HIGHEST_PROTOCOL)
        #pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    with open("alldata-" + savefile, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    logger.info('Save data size=%d to %s'%(len(data), savefile))




