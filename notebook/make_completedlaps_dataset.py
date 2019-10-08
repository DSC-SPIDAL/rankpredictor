# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# to use only one GPU.
# use this on r-001
# otherwise comment
import os, sys


# ## Load Data

# In[2]:

import os
os.getcwd()


def load_dataset(inputfile):
    print('load dataset:', inputfile)
    dataset = pd.read_csv(inputfile)
    dataset.sort_values(by=['car_number', 'completed_laps', 'elapsed_time'], ascending=True).head(40)
    #dataset.info(verbose=True)
    return dataset

 
def get_finished_cars(dataset):
    final_lap = max(dataset.completed_laps)
    total_laps = final_lap + 1
    
    # get records for the cars that finish the race
    completed_car_numbers= dataset[dataset.completed_laps == final_lap].car_number.values
    
    completed_car_count = len(completed_car_numbers)
    
    print('final lap:', final_lap)
    print('count of completed cars:', completed_car_count)
    print('completed cars:', completed_car_numbers)
    #make a copy
    ret = dataset[dataset['car_number'].isin(completed_car_numbers)]
    return ret

# make indy car completed_laps dataset
# car_number, completed_laps, rank, elapsed_time, rank_diff, elapsed_time_diff 
def make_cl_data(dataset):
    uni_ds = dataset.drop(["unique_id", "best_lap", "current_status", "track_status", "lap_status",
                      "laps_behind_leade","laps_behind_prec","overall_rank","pit_stop_count",
                      "last_pitted_lap","start_position","laps_led"], axis=1)
    uni_ds=uni_ds.sort_values(by=['car_number', 'completed_laps', 'elapsed_time'], ascending=True)
    uni_ds=uni_ds.drop_duplicates(subset=['car_number', 'completed_laps'], keep='first')
    
    carnumber = set(uni_ds['car_number'])
    print('cars:', carnumber)
    
    # faster solution , uni_ds already sorted by car_number and lap
    uni_ds['rank_diff'] = uni_ds['rank'].diff()
    mask = uni_ds.car_number != uni_ds.car_number.shift(1)
    uni_ds['rank_diff'][mask] = 0
    
    uni_ds['time_diff'] = uni_ds['elapsed_time'].diff()
    mask = uni_ds.car_number != uni_ds.car_number.shift(1)
    uni_ds['time_diff'][mask] = 0
    
    df = uni_ds[['car_number','completed_laps','rank','elapsed_time','rank_diff','time_diff']]
    
    return df




#outputprefix = sys.argv[1]
#inputfile = 'C_' + outputprefix + '.log'
inputfile = sys.argv[1]
#inputfile = 'C_' + outputprefix + '.log'
outputprefix = inputfile[2:inputfile.rfind('-final')] + '-'

dataset = load_dataset(inputfile)
alldata = dataset.copy()

dataset = get_finished_cars(dataset)

cldata = make_cl_data(dataset)
cldata.to_csv(outputprefix + 'completed_laps_diff.csv')


cldata = make_cl_data(alldata)
cldata.to_csv(outputprefix + 'all_completed_laps_diff.csv')


