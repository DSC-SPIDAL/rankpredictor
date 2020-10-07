#!/usr/bin/env python
# coding: utf-8

# ### test deepar tensorflow

from numpy.random import normal
import tqdm
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

#from indycar.model.deepartf.dataset.time_series import MockTs
from indycar.model.deepartfve.dataset.time_series import RecordTs
from indycar.model.deepartfve.model_eager.lstm_ve import DeepAR
#from indycar.model.deepartfve.model.lstm import DeepAR

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def get_sample_prediction_gaussian(sample, fn):
    sample = np.array(sample).reshape(1, _seq_len, 1)
    output = fn([sample])
    samples = []
    for mu,sigma in zip(output[0].reshape(_seq_len), output[1].reshape(_seq_len)):
        samples.append(normal(loc=mu, scale=np.sqrt(sigma), size=1)[0])
    return np.array(samples)

def get_sample_prediction(sample, model, verbose = False):
    sample = np.array(sample).reshape(1, _seq_len, -1)
    output = model.predict([sample])
    
    output2 = np.zeros((_seq_len, len(output)))
    for idx, x in enumerate(output):
        output2[:,idx] = x.reshape(_seq_len)
        
    #output2 = np.array(output).reshape(_seq_len, -1)
    if verbose:
        print('output.shape=',[len(x) for x in output])
        print('output:', output)
        print('output2.shape=', output2.shape)
        print('output2:', output2)
    
    samples = []
    #for theta in zip(output[0].reshape(_seq_len), output[1].reshape(_seq_len)):
    for theta in output2:
        samples.append(model.get_sample(theta))
    return np.array(samples)


def predict(model):
    #batch = ts.next_batch(_batch_size, _seq_len)
    batch = ts.next_batch(-1, _seq_len)

    #get_sample_prediction(batch[0], model, verbose=True)
    #get_sample_prediction(batch[0], model, verbose=True)
    
    ress = []
    for i in tqdm.tqdm(range(300)):
        #ress.append(get_sample_prediction(batch[0], model.predict_theta_from_input))
        ress.append(get_sample_prediction(batch[0], model))

    res_df = pd.DataFrame(ress).T
    tot_res = res_df

    plt.plot(batch[1].reshape(_seq_len), linewidth=6)
    tot_res['mu'] = tot_res.apply(lambda x: np.mean(x), axis=1)
    tot_res['upper'] = tot_res.apply(lambda x: np.mean(x) + np.std(x), axis=1)
    tot_res['lower'] = tot_res.apply(lambda x: np.mean(x) - np.std(x), axis=1)
    tot_res['two_upper'] = tot_res.apply(lambda x: np.mean(x) + 2*np.std(x), axis=1)
    tot_res['two_lower'] = tot_res.apply(lambda x: np.mean(x) - 2*np.std(x), axis=1)

    plt.plot(tot_res.mu, 'bo')
    plt.plot(tot_res.mu, linewidth=2)
    plt.fill_between(x = tot_res.index, y1=tot_res.lower, y2=tot_res.upper, alpha=0.5)
    plt.fill_between(x = tot_res.index, y1=tot_res.two_lower, y2=tot_res.two_upper, alpha=0.5)
    plt.title('Prediction uncertainty')
    
    return batch, tot_res

import tensorflow as tf
#tf.config.experimental_run_functions_eagerly(False)
#tf.config.experimental_run_functions_eagerly(True)
#tf.compat.v1.disable_eager_execution()


ts = RecordTs('savedata_drank_e1.pickle')
_context_len = 40
_prediction_len = 2
_batch_size = 32
_seq_len = _context_len + _prediction_len


# train
model3 = DeepAR(ts, epochs=10, distribution='StudentT', with_custom_nn_structure=DeepAR.encoder_decoder)
model3.fit(context_len=_context_len, prediction_len=_prediction_len, input_dim = 33)


# prediction
#batch, df = predict(model3)

