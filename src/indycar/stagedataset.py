import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import os

def plot_examples(X,y,ypreds=None,nm_ypreds=None):
    fig = plt.figure(figsize=(16,10))
    fig.subplots_adjust(hspace = 0.32,wspace = 0.15)
    count = 1
    n_ts = X.shape[0]
    num_per_row = 2 if n_ts > 2 else 1
    for irow in range(n_ts):
        ax = fig.add_subplot(int(n_ts+num_per_row -1)/num_per_row,num_per_row,count)
        #ax.set_ylim(-0.5,0.5)
        ax.plot(X[irow,:,0],"--",label="x1")
        ax.plot(y[irow,:,:],marker='.',label="y",linewidth=3,alpha = 0.5)
        ax.set_title("{:}th time series sample".format(irow))
        if ypreds is not None:
            for ypred,nm in zip(ypreds,nm_ypreds):
                ax.plot(ypred[irow,:,:],marker='.',label=nm)   
        count += 1
    plt.legend()
    plt.show()
    #if ypreds is not None:
    #    for y_pred, nm_ypred in zip(ypreds,nm_ypreds):
    #        loss = np.mean( (y_pred[:,D:,:].flatten() - y[:,D:,:].flatten())**2)
    #        print("The final validation loss of {} is {:7.6f}".format(
    #            nm_ypred,loss))
            
def plot_vectors(y,ypred,y_idx,nm_ypreds='pred'):
    fig = plt.figure(figsize=(16,10))
    fig.subplots_adjust(hspace = 0.32,wspace = 0.15)
    count = 1
    n_ts = y_idx.shape[0]
    num_per_row = 2 if n_ts > 2 else 1
    start, end = 0, 0
    for irow in range(n_ts):
        end += y_idx[irow]
        
        ax = fig.add_subplot(int(n_ts+num_per_row -1)/num_per_row,num_per_row,count)
        #ax.set_ylim(-0.5,0.5)
        #ax.plot(X[irow,:,0],"--",label="x1")
        ax.plot(y[start:end],marker='.',label="y",linewidth=3,alpha = 0.5)
        ax.set_title("{:}th time series sample".format(irow))
        #if ypreds is not None:
        #    for ypred,nm in zip(ypreds,nm_ypreds):
        #        ax.plot(ypred[start:end],marker='.',label=nm)   
        ax.plot(ypred[start:end],marker='.',label='pred')   
        
        count += 1
        
        start = end
        
    plt.legend()
    plt.show()
    #if ypreds is not None:
    #    for y_pred, nm_ypred in zip(ypreds,nm_ypreds):
    loss = np.mean( np.abs(ypred.flatten() - y.flatten()))
    print("The final validation loss MAE is {:7.6f}".format(loss))
            
            
# load multiple datasets
def load_datalist(datalist):
    #add 
    db = []
    lens = []
    for id, f in enumerate(datalist):
        data = pd.read_csv(f)
        data['dbid'] = id + 1
        db.append(data)
        
        carNumber = len(set(data.car_number))
        lens.append(carNumber)
        print('load %s, len=%d'%(f, data.shape[0]))
        
    
    alldata = None
    for d in db:
        #update car_number with the dbid
        d['car_number'] += d['dbid'] * 1000
        if alldata is None:
            alldata = d
        else:
            alldata = alldata.append(d)
    
    #scaler
    scaler = MinMaxScaler()
    alldata[['rank_diff_raw', 'time_diff_raw']] = alldata[['rank_diff', 'time_diff']]
    alldata[['rank_diff', 'time_diff']] = scaler.fit_transform(alldata[['rank_diff', 'time_diff']])
    
    return scaler, alldata, lens

def generate_data(dataset, D= 1, target='rank', shuffle = False):
    # dataset with multiple events, car_number is encoded with event id
    # T is the max len
    
    carNumber = len(set(dataset.car_number))
    T = 0
    for car, group in dataset.groupby('car_number'):
        T = max(T, group.shape[0])
    print('carNumber = %d, max T =%d'%(carNumber, T))
    
    #variable len of time series
    x_train , y_train = [], []
    for car, group in dataset.groupby('car_number'):
        x = list(group.time_diff)
    
        if target == 'rank':
            y = list(group.rank_diff)
        elif target =='time':
            y = list(group.time_diff)
        else:
            print('error in target setting as', target)
            return None

        #get train/label
        retlen = len(x) - D
        if retlen <=0 :
            print('error with record, too short, car = %d, len=%d'%(car,len(x)))
            continue
        
        #output
        x_train.append(x[:retlen])
        y_train.append(y[D:])
        
    if len(x_train) != carNumber:
        print('error in carNumber')
        return x_train, y_train, x
    
    #convert to np array
    X = np.zeros((carNumber, T-D, 1))
    Y = np.zeros((carNumber, T-D, 1))
    W = np.zeros((carNumber, T-D))
    for car in range(carNumber):
        reclen = len(x_train[car])
        X[car, :reclen, 0] = np.array(x_train[car])
        Y[car, :reclen, 0] = np.array(y_train[car])        
        W[car, :reclen] = 1
        
    if shuffle:
        idx = np.random.permutation(carNumber)
        X = X[idx]
        Y = Y[idx]
        W = W[idx]

    return X, Y, W

def read_list(listfile):
    datalist = []
    with open(listfile, 'r') as inf:
        for l in inf:
            datalist.append(l.strip())
    return datalist

#from sklearn.utils import check_arrays
def mean_absolute_percentage_error(y_true, y_pred): 
    #y_true, y_pred = check_arrays(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)
    idx = y_true != 0

    #return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return np.mean(np.abs((y_true[idx] - y_pred[idx]) / y_true[idx])) * 100

def generate_feature_vectors(X,Y,W, vect_len = 10):
    '''
    X shape: <samples, series_len, 1>
    Y shape: <samples, series_len, 1>
    W shape: <samples, series_len>        
    vect_len: output feature vector length
    return:
    vect_x, vect_y
    ts_idx
    
    '''
    ts_cnt = X.shape[0]
    #ts_cnt = 1
    vect_x = []
    vect_y = []
    ts_idx = []
    for rid in range(ts_cnt):
        ts_x = X[rid,:,0]
        ts_y = Y[rid,:,0]
        ts_len = int(np.sum(W[rid]))
        if ts_len < vect_len:
            #ts is not long enough, skip it
            print('len[%d]=%d is too short, skip'%(rid, ts_len))
            continue
        #extract multiple feature vectors from this ts
        ts_idx.append(ts_len - vect_len + 1)
        for i in range(ts_len - vect_len + 1):
            #not padding
            vect_x.append(ts_x[i:i+vect_len])
            vect_y.append(ts_y[i+vect_len-1])
            
    return np.array(vect_x), np.array(vect_y), np.array(ts_idx)  


def predict(model_name,model, x_test, y_test_in, scaler=None, target='time'):
    #prediction
    if model_name == 'lstm':
        n,m = x_test.shape
        y_pred = model.predict(x_test.reshape((n,m,1)))
    else:
        y_pred = model.predict(x_test)
        
    #flatten y
    y_test = y_test_in.copy().flatten()
    y_pred = y_pred.flatten()
    
    #mae
    mae = metrics.mean_absolute_error(y_test, y_pred)
    
    #inverse scale to get original values
    tmae = 0.
    if scaler is not None:
        n = y_test.shape[0]
        Y_true = np.zeros((n,2))
        if target == 'time':
            Y_true[:,1] = y_test
        else:
            Y_true[:,0] = y_test        
        Y_true = scaler.inverse_transform(Y_true)
        Y_pred = np.zeros((n,2))
        if target == 'time':
            Y_pred[:,1] = y_pred.reshape((n))
        else:
            Y_pred[:,0] = y_pred.reshape((n))        
        Y_pred = scaler.inverse_transform(Y_pred)
        
        
        tmae = metrics.mean_absolute_error(Y_true, Y_pred)
        tmape = mean_absolute_percentage_error(Y_true, Y_pred)
    
    # print
    print('%s model mae=%f, raw mae=%f, raw mape=%f'%(model_name, mae, tmae, tmape))
    
    return y_pred, Y_pred, mae, tmae, tmape

#===================================================================================

# make indy car completed_laps dataset
# car_number, completed_laps, rank, elapsed_time, rank_diff, elapsed_time_diff 
def make_cl_data(dataset):

    # pick up data with valid rank
    rankdata = dataset.rename_axis('MyIdx').sort_values(by=['elapsed_time','MyIdx'], ascending=True)
    rankdata = rankdata.drop_duplicates(subset=['car_number', 'completed_laps'], keep='first')

    # resort by car_number, lap
    uni_ds = rankdata.sort_values(by=['car_number', 'completed_laps', 'elapsed_time'], ascending=True)    
    uni_ds = uni_ds.drop(["unique_id", "best_lap", "current_status", "track_status", "lap_status",
                      "laps_behind_leade","laps_behind_prec","overall_rank","pit_stop_count",
                      "last_pitted_lap","start_position","laps_led"], axis=1)
    
    carnumber = set(uni_ds['car_number'])
    print('cars:', carnumber)
    print('#cars=', len(carnumber))
   
    # faster solution , uni_ds already sorted by car_number and lap
    uni_ds['rank_diff'] = uni_ds['rank'].diff()
    mask = uni_ds.car_number != uni_ds.car_number.shift(1)
    uni_ds['rank_diff'][mask] = 0
    
    uni_ds['time_diff'] = uni_ds['elapsed_time'].diff()
    mask = uni_ds.car_number != uni_ds.car_number.shift(1)
    uni_ds['time_diff'][mask] = 0
    
    df = uni_ds[['car_number','completed_laps','rank','elapsed_time','rank_diff','time_diff']]
    
    return df


def make_lapstatus_data(dataset):
    final_lap = max(dataset.completed_laps)
    total_laps = final_lap + 1

    # get records for the cars that finish the race
    completed_car_numbers= dataset[dataset.completed_laps == final_lap].car_number.values
    completed_car_count = len(completed_car_numbers)

    print('count of completed cars:', completed_car_count)
    print('completed cars:', completed_car_numbers)
    
    #pick up one of them
    onecar = dataset[dataset['car_number']==completed_car_numbers[0]]
    onecar = onecar.drop_duplicates(subset=['car_number', 'completed_laps'], keep='first')
    return onecar[['completed_laps','track_status']]

def load_data(event, year):
    inputfile = '../data/final/C_'+ event +'-' + year + '-final.csv'
    outputprefix = year +'-' + event + '-'
    dataset = pd.read_csv(inputfile)
    #dataset.info(verbose=True)    
    
    final_lap = max(dataset.completed_laps)
    total_laps = final_lap + 1

    # get records for the cars that finish the race
    completed_car_numbers= dataset[dataset.completed_laps == final_lap].car_number.values
    completed_car_count = len(completed_car_numbers)

    print('count of completed cars:', completed_car_count)
    print('completed cars:', completed_car_numbers)

    #make a copy
    alldata = dataset.copy()
    dataset = dataset[dataset['car_number'].isin(completed_car_numbers)]
    rankdata = alldata.rename_axis('MyIdx').sort_values(by=['elapsed_time','MyIdx'], ascending=True)
    rankdata = rankdata.drop_duplicates(subset=['car_number', 'completed_laps'], keep='first')
    
    cldata = make_cl_data(dataset)
    flagdata = make_lapstatus_data(dataset)
    acldata = make_cl_data(alldata)

    return alldata, rankdata, acldata, flagdata

