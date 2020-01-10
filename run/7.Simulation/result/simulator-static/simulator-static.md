# Imports


```python
%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random

```

## Load Data


```python
import os
os.getcwd()

```




    '/scratch/hpda/indycar/predictor/notebook/6.SectionRank'




```python
#
# parameters
#
#year = '2017'
year = '2018'
#event = 'Toronto'
event = 'Indy500'

inputfile = '../data/final/C_'+ event +'-' + year + '-final.csv'
outputprefix = year +'-' + event + '-'
dataset = pd.read_csv(inputfile)
dataset.info(verbose=True)
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 18500 entries, 0 to 18499
    Data columns (total 21 columns):
    rank                    18500 non-null int64
    car_number              18500 non-null int64
    unique_id               18500 non-null object
    completed_laps          18500 non-null int64
    elapsed_time            18500 non-null float64
    last_laptime            18500 non-null float64
    lap_status              18500 non-null object
    best_laptime            18500 non-null float64
    best_lap                18500 non-null object
    time_behind_leader      18500 non-null float64
    laps_behind_leade       18500 non-null object
    time_behind_prec        18500 non-null float64
    laps_behind_prec        18500 non-null object
    overall_rank            18500 non-null object
    overall_best_laptime    18500 non-null float64
    current_status          18500 non-null object
    track_status            18500 non-null object
    pit_stop_count          18500 non-null object
    last_pitted_lap         18500 non-null object
    start_position          18500 non-null object
    laps_led                18500 non-null object
    dtypes: float64(6), int64(3), object(12)
    memory usage: 3.0+ MB


### The Simulator

simple model without DNF

1. laptime, modeled by average lap time on green laps
2. pitstop, uniform distributed in pit window(10 laps)
3. pitime, modeled by inlap, outlap time



```python
#green laps
alldata = dataset.copy()
carnos = np.sort(list(set(alldata.car_number.values)))
rankdata = alldata.rename_axis('MyIdx').sort_values(by=['elapsed_time','MyIdx'], ascending=True)

# since the flag changes in the middle of a lap, != 'Y' does not work here
#greendata = rankdata[rankdata['track_status']!='Y']
yellow_laps = rankdata[rankdata['track_status']=='Y'].completed_laps.values
green_laps = set(rankdata.completed_laps.values) - set(yellow_laps)
greendata = rankdata[rankdata['completed_laps'].isin(green_laps)]

# car_number, startpos, norm_lap, in_lap, out_lap
statdata = np.zeros((len(carnos), 8))
for idx, car in enumerate(carnos):
        thiscar = greendata[greendata['car_number']==car]
        
        pit_laps = thiscar[thiscar['lap_status']=='P'].completed_laps.values
        in_lap = thiscar[thiscar['completed_laps'].isin(pit_laps)].last_laptime.values
        out_laps = [x+1 for x in pit_laps]
        out_lap = thiscar[thiscar['completed_laps'].isin(out_laps)].last_laptime.values
        
        normal_laps = set(thiscar.completed_laps.values) - set(pit_laps) -set(out_laps)
        _laps = [x if x-1 in normal_laps else -1 for x in normal_laps]
        _laps=np.array(_laps)
        normal_laps = _laps[_laps>0]
        norm_lap = thiscar[thiscar['completed_laps'].isin(normal_laps)].last_laptime.values
        
        #save statistics
        statdata[idx, 0] = car
        startPos = thiscar[thiscar['completed_laps']==1].start_position.values[0]
        statdata[idx, 1] = int(startPos, 16)
        statdata[idx, 2] = np.mean(norm_lap)
        statdata[idx, 3] = np.std(norm_lap)
        statdata[idx, 4] = np.mean(in_lap)
        statdata[idx, 5] = np.std(in_lap)
        statdata[idx, 6] = np.mean(out_lap)
        statdata[idx, 7] = np.std(out_lap)
        
df = pd.DataFrame({'car_number':statdata[:,0].astype(int),'start_position':statdata[:,1].astype(int),'norm_lap_mean':statdata[:,2],'norm_lap_std':statdata[:,3],
                   'in_lap_mean':statdata[:,4],'in_lap_std':statdata[:,5],'out_lap_mean':statdata[:,6],'out_lap_std':statdata[:,7]})        
df.to_csv(outputprefix + 'simulator.csv')
simdf = df.copy()
```


```python
df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>car_number</th>
      <th>in_lap_mean</th>
      <th>in_lap_std</th>
      <th>norm_lap_mean</th>
      <th>norm_lap_std</th>
      <th>out_lap_mean</th>
      <th>out_lap_std</th>
      <th>start_position</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>54.339015</td>
      <td>8.424845e-01</td>
      <td>41.743706</td>
      <td>0.766367</td>
      <td>66.104833</td>
      <td>0.730250</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>54.281610</td>
      <td>7.251726e-01</td>
      <td>41.611673</td>
      <td>0.676782</td>
      <td>66.911113</td>
      <td>0.401542</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>54.295602</td>
      <td>3.190570e-01</td>
      <td>41.946153</td>
      <td>0.770234</td>
      <td>68.036082</td>
      <td>0.729633</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>64.027260</td>
      <td>8.913270e-01</td>
      <td>42.018802</td>
      <td>0.778819</td>
      <td>55.488440</td>
      <td>0.238874</td>
      <td>18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>76.500685</td>
      <td>5.074079e+00</td>
      <td>43.346298</td>
      <td>1.081907</td>
      <td>58.039793</td>
      <td>0.983255</td>
      <td>28</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9</td>
      <td>54.458102</td>
      <td>1.106836e+00</td>
      <td>42.022496</td>
      <td>0.701250</td>
      <td>66.476525</td>
      <td>0.545905</td>
      <td>9</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10</td>
      <td>67.298100</td>
      <td>1.421085e-14</td>
      <td>42.409261</td>
      <td>1.136934</td>
      <td>55.727200</td>
      <td>0.000000</td>
      <td>29</td>
    </tr>
    <tr>
      <th>7</th>
      <td>12</td>
      <td>53.857717</td>
      <td>1.671692e+00</td>
      <td>41.485985</td>
      <td>0.776325</td>
      <td>65.957933</td>
      <td>0.901091</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>13</td>
      <td>55.341700</td>
      <td>7.105427e-15</td>
      <td>41.927896</td>
      <td>0.640302</td>
      <td>67.616600</td>
      <td>0.000000</td>
      <td>7</td>
    </tr>
    <tr>
      <th>9</th>
      <td>14</td>
      <td>55.449998</td>
      <td>1.114515e+00</td>
      <td>41.540961</td>
      <td>0.629124</td>
      <td>66.876700</td>
      <td>0.556106</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
# pit window
# the first pit is more reasonable, the pit window should at least to be 7 laps
#
maxpitcnt = max([int(x,16) for x in rankdata.pit_stop_count.values])

for pit in range(1,maxpitcnt):
    pit_laps= np.sort(list(rankdata[(rankdata['pit_stop_count']==('%x'%(pit))) 
                                & (rankdata['lap_status']=='P')].completed_laps.values))
    print('%d:%d, %d'%(pit, min(pit_laps), max(pit_laps)))
    
    
```

    1:29, 36
    2:35, 70
    3:52, 108
    4:60, 141
    5:96, 175
    6:107, 197
    7:120, 196
    8:138, 195
    9:180, 191


#### Simulator


```python
def run_simulator(rankdata, simdf, savemodel=''):
    """
    input: simdf, rankdata
    
    simulator output the same data format as rankdata(C_xxx)
        #rank, car_number, completed_laps,elapsed_time,last_laptime
        #lap_status,track_status, pit_stop_count, last_pitted_lap
    """
    
    #init
    random.seed()

    maxlaps = max(set(rankdata.completed_laps.values))
    cols=['rank', 'car_number', 'completed_laps','elapsed_time','last_laptime',
          'lap_status','track_status', 'pit_stop_count', 'last_pitted_lap']
    colid={key:idx for idx, key in enumerate(cols)}

    # fixed pit strategy
    # max laps = 38
    # pit window = 8
    # uniform distribution in [last_pit+38-8, last_pit+38]
    pit_maxlaps = 38
    pit_window = 8
    carnos = np.sort(list(set(simdf.car_number.values)))
    #carnos = simdf.car_number.values
    carid = {key:idx for idx, key in enumerate(carnos)}

    data = np.zeros((len(carnos)*maxlaps, len(cols)))
    #print('maxlaps=%d, data shape=%s'%(maxlaps, data.shape))    
    
    # fixed pit strategy
    # max laps = 38
    # pit window = 8
    # uniform distribution in [last_pit+38-8, last_pit+38]
    for car in carnos:
        curlap = 0
        pit_cnt = 0
        while curlap < maxlaps:
            #set the next pit lap
            #uniform in [curlap + ]
            right = curlap + pit_maxlaps
            if right > maxlaps:
                # no need to pitstop
                break
            left = curlap + pit_maxlaps - pit_window
            pit_lap = int(random.uniform(left, right))
            #set it
            data[carid[car] * maxlaps + pit_lap, colid['lap_status']] = 1
            data[carid[car] * maxlaps + pit_lap, colid['pit_stop_count']] = pit_cnt
            data[carid[car] * maxlaps + pit_lap, colid['last_pitted_lap']] = pit_lap

            pit_cnt += 1
            curlap = pit_lap
            
    # simulate the lap time
    # startPenalty = startPosition * 0.11(s)

    for car in carnos:
        last_ispit = 0
        param = simdf[simdf['car_number']==car]
        elapsed_time = param.start_position * 0.11
        for lap in range(maxlaps):
            cur_ispit = data[carid[car] * maxlaps + lap, colid['lap_status']]
            if last_ispit:
                #use out_lap
                laptime = random.gauss(param['out_lap_mean'],param['out_lap_std'])
            elif cur_ispit:
                #use in_lap
                laptime = random.gauss(param['in_lap_mean'],param['in_lap_std'])
            else:
                #use norm_lap
                laptime = random.gauss(param['norm_lap_mean'],param['norm_lap_std'])

            data[carid[car] * maxlaps + lap, colid['last_laptime']] = laptime
            elapsed_time += laptime
            data[carid[car] * maxlaps + lap, colid['elapsed_time']] = elapsed_time

            data[carid[car] * maxlaps + lap, colid['car_number']] = car
            #start from lap 1
            data[carid[car] * maxlaps + lap, colid['completed_laps']] = lap + 1

            #update and goto next lap
            last_ispit = cur_ispit

    # update the rank
    # carnumber = len(carnos)
    for lap in range(maxlaps):
        elapsed_time = [data[carid[car] * maxlaps + lap, colid['elapsed_time']] for car in carnos]
        indice = np.argsort(elapsed_time)
        rank = np.arange(len(carnos))
        out = np.arange(len(carnos))
        out[indice] = rank + 1
        for car in carnos:
            data[carid[car] * maxlaps + lap, colid['rank']] = int(out[carid[car]])

    #save data
    #rank, car_number, completed_laps,elapsed_time,last_laptime
    #lap_status,track_status, pit_stop_count, last_pitted_lap
    df = pd.DataFrame({'rank': data[:, 0].astype(int), 'car_number': data[:, 1].astype(int),
                       'completed_laps': data[:, 2].astype(int),
                       'elapsed_time': data[:, 3], 'last_laptime': data[:, 4], 
                       'lap_status': [ 'P' if x==1 else 'T' for x in data[:, 5]],
                       'track_status': [ 'G' for x in data[:, 6]],
                       'pit_stop_count': data[:, 7], 'last_pitted_lap': data[:, 8]})
    if savemodel:
        df.to_csv(savemodel)
    
    return df
```


```python
df = run_simulator(rankdata, simdf, outputprefix + 'simulator_completedlaps.csv')
```


```python
df[df['completed_laps']==101]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>car_number</th>
      <th>completed_laps</th>
      <th>elapsed_time</th>
      <th>lap_status</th>
      <th>last_laptime</th>
      <th>last_pitted_lap</th>
      <th>pit_stop_count</th>
      <th>rank</th>
      <th>track_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100</th>
      <td>1</td>
      <td>101</td>
      <td>4308.946522</td>
      <td>P</td>
      <td>54.062757</td>
      <td>100.0</td>
      <td>2.0</td>
      <td>6</td>
      <td>G</td>
    </tr>
    <tr>
      <th>300</th>
      <td>3</td>
      <td>101</td>
      <td>4321.426490</td>
      <td>T</td>
      <td>66.543354</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10</td>
      <td>G</td>
    </tr>
    <tr>
      <th>500</th>
      <td>4</td>
      <td>101</td>
      <td>4356.291667</td>
      <td>T</td>
      <td>41.569234</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>23</td>
      <td>G</td>
    </tr>
    <tr>
      <th>700</th>
      <td>6</td>
      <td>101</td>
      <td>4356.963060</td>
      <td>T</td>
      <td>41.830193</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>24</td>
      <td>G</td>
    </tr>
    <tr>
      <th>900</th>
      <td>7</td>
      <td>101</td>
      <td>4548.046122</td>
      <td>T</td>
      <td>43.550836</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>33</td>
      <td>G</td>
    </tr>
    <tr>
      <th>1100</th>
      <td>9</td>
      <td>101</td>
      <td>4319.942692</td>
      <td>T</td>
      <td>42.449789</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8</td>
      <td>G</td>
    </tr>
    <tr>
      <th>1300</th>
      <td>10</td>
      <td>101</td>
      <td>4367.371384</td>
      <td>T</td>
      <td>43.764535</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>26</td>
      <td>G</td>
    </tr>
    <tr>
      <th>1500</th>
      <td>12</td>
      <td>101</td>
      <td>4305.026491</td>
      <td>T</td>
      <td>39.509822</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>G</td>
    </tr>
    <tr>
      <th>1700</th>
      <td>13</td>
      <td>101</td>
      <td>4353.528541</td>
      <td>T</td>
      <td>67.616600</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>21</td>
      <td>G</td>
    </tr>
    <tr>
      <th>1900</th>
      <td>14</td>
      <td>101</td>
      <td>4262.959606</td>
      <td>T</td>
      <td>41.098201</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>G</td>
    </tr>
    <tr>
      <th>2100</th>
      <td>15</td>
      <td>101</td>
      <td>4355.304506</td>
      <td>T</td>
      <td>55.987469</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>22</td>
      <td>G</td>
    </tr>
    <tr>
      <th>2300</th>
      <td>17</td>
      <td>101</td>
      <td>4430.211647</td>
      <td>T</td>
      <td>42.789169</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>31</td>
      <td>G</td>
    </tr>
    <tr>
      <th>2500</th>
      <td>18</td>
      <td>101</td>
      <td>4316.497802</td>
      <td>T</td>
      <td>41.153194</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7</td>
      <td>G</td>
    </tr>
    <tr>
      <th>2700</th>
      <td>19</td>
      <td>101</td>
      <td>4346.207239</td>
      <td>T</td>
      <td>41.895730</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>19</td>
      <td>G</td>
    </tr>
    <tr>
      <th>2900</th>
      <td>20</td>
      <td>101</td>
      <td>4323.551889</td>
      <td>T</td>
      <td>42.557845</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>11</td>
      <td>G</td>
    </tr>
    <tr>
      <th>3100</th>
      <td>21</td>
      <td>101</td>
      <td>4291.338001</td>
      <td>T</td>
      <td>41.877024</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>G</td>
    </tr>
    <tr>
      <th>3300</th>
      <td>22</td>
      <td>101</td>
      <td>4266.444854</td>
      <td>T</td>
      <td>41.079094</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>G</td>
    </tr>
    <tr>
      <th>3500</th>
      <td>23</td>
      <td>101</td>
      <td>4319.971125</td>
      <td>T</td>
      <td>40.491372</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9</td>
      <td>G</td>
    </tr>
    <tr>
      <th>3700</th>
      <td>24</td>
      <td>101</td>
      <td>4351.173946</td>
      <td>T</td>
      <td>43.554592</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20</td>
      <td>G</td>
    </tr>
    <tr>
      <th>3900</th>
      <td>25</td>
      <td>101</td>
      <td>4331.575289</td>
      <td>T</td>
      <td>41.715543</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>14</td>
      <td>G</td>
    </tr>
    <tr>
      <th>4100</th>
      <td>26</td>
      <td>101</td>
      <td>4399.964065</td>
      <td>T</td>
      <td>41.335177</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>29</td>
      <td>G</td>
    </tr>
    <tr>
      <th>4300</th>
      <td>27</td>
      <td>101</td>
      <td>4333.368894</td>
      <td>T</td>
      <td>42.599018</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>17</td>
      <td>G</td>
    </tr>
    <tr>
      <th>4500</th>
      <td>28</td>
      <td>101</td>
      <td>4294.753030</td>
      <td>T</td>
      <td>41.974022</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>G</td>
    </tr>
    <tr>
      <th>4700</th>
      <td>29</td>
      <td>101</td>
      <td>4325.919702</td>
      <td>T</td>
      <td>43.715521</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12</td>
      <td>G</td>
    </tr>
    <tr>
      <th>4900</th>
      <td>30</td>
      <td>101</td>
      <td>4361.112899</td>
      <td>T</td>
      <td>41.705822</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>25</td>
      <td>G</td>
    </tr>
    <tr>
      <th>5100</th>
      <td>32</td>
      <td>101</td>
      <td>4431.399492</td>
      <td>T</td>
      <td>43.232561</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>32</td>
      <td>G</td>
    </tr>
    <tr>
      <th>5300</th>
      <td>33</td>
      <td>101</td>
      <td>4419.812339</td>
      <td>T</td>
      <td>40.457185</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>30</td>
      <td>G</td>
    </tr>
    <tr>
      <th>5500</th>
      <td>59</td>
      <td>101</td>
      <td>4370.955543</td>
      <td>T</td>
      <td>43.160472</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>27</td>
      <td>G</td>
    </tr>
    <tr>
      <th>5700</th>
      <td>60</td>
      <td>101</td>
      <td>4331.858680</td>
      <td>T</td>
      <td>41.796018</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>15</td>
      <td>G</td>
    </tr>
    <tr>
      <th>5900</th>
      <td>64</td>
      <td>101</td>
      <td>4327.624918</td>
      <td>T</td>
      <td>41.998043</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>13</td>
      <td>G</td>
    </tr>
    <tr>
      <th>6100</th>
      <td>66</td>
      <td>101</td>
      <td>4388.883638</td>
      <td>T</td>
      <td>41.344231</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>28</td>
      <td>G</td>
    </tr>
    <tr>
      <th>6300</th>
      <td>88</td>
      <td>101</td>
      <td>4332.109365</td>
      <td>T</td>
      <td>42.496802</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>16</td>
      <td>G</td>
    </tr>
    <tr>
      <th>6500</th>
      <td>98</td>
      <td>101</td>
      <td>4344.051600</td>
      <td>T</td>
      <td>42.856482</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>18</td>
      <td>G</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df['completed_laps']==200]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>car_number</th>
      <th>completed_laps</th>
      <th>elapsed_time</th>
      <th>lap_status</th>
      <th>last_laptime</th>
      <th>last_pitted_lap</th>
      <th>pit_stop_count</th>
      <th>rank</th>
      <th>track_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>199</th>
      <td>1</td>
      <td>200</td>
      <td>8541.454103</td>
      <td>T</td>
      <td>42.564135</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6</td>
      <td>G</td>
    </tr>
    <tr>
      <th>399</th>
      <td>3</td>
      <td>200</td>
      <td>8514.072033</td>
      <td>T</td>
      <td>42.350526</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>G</td>
    </tr>
    <tr>
      <th>599</th>
      <td>4</td>
      <td>200</td>
      <td>8590.997951</td>
      <td>T</td>
      <td>42.645031</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>13</td>
      <td>G</td>
    </tr>
    <tr>
      <th>799</th>
      <td>6</td>
      <td>200</td>
      <td>8624.569536</td>
      <td>T</td>
      <td>42.510448</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>24</td>
      <td>G</td>
    </tr>
    <tr>
      <th>999</th>
      <td>7</td>
      <td>200</td>
      <td>8975.151789</td>
      <td>T</td>
      <td>42.325585</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>33</td>
      <td>G</td>
    </tr>
    <tr>
      <th>1199</th>
      <td>9</td>
      <td>200</td>
      <td>8594.504606</td>
      <td>T</td>
      <td>42.687133</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>15</td>
      <td>G</td>
    </tr>
    <tr>
      <th>1399</th>
      <td>10</td>
      <td>200</td>
      <td>8681.266491</td>
      <td>T</td>
      <td>41.461483</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>26</td>
      <td>G</td>
    </tr>
    <tr>
      <th>1599</th>
      <td>12</td>
      <td>200</td>
      <td>8504.072296</td>
      <td>T</td>
      <td>41.431504</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>G</td>
    </tr>
    <tr>
      <th>1799</th>
      <td>13</td>
      <td>200</td>
      <td>8577.419192</td>
      <td>T</td>
      <td>43.321706</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10</td>
      <td>G</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>14</td>
      <td>200</td>
      <td>8496.436174</td>
      <td>T</td>
      <td>41.912857</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>G</td>
    </tr>
    <tr>
      <th>2199</th>
      <td>15</td>
      <td>200</td>
      <td>8577.371116</td>
      <td>T</td>
      <td>43.043372</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9</td>
      <td>G</td>
    </tr>
    <tr>
      <th>2399</th>
      <td>17</td>
      <td>200</td>
      <td>8762.357394</td>
      <td>T</td>
      <td>42.648065</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>31</td>
      <td>G</td>
    </tr>
    <tr>
      <th>2599</th>
      <td>18</td>
      <td>200</td>
      <td>8586.619258</td>
      <td>T</td>
      <td>42.247758</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>11</td>
      <td>G</td>
    </tr>
    <tr>
      <th>2799</th>
      <td>19</td>
      <td>200</td>
      <td>8660.187653</td>
      <td>T</td>
      <td>43.994959</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>25</td>
      <td>G</td>
    </tr>
    <tr>
      <th>2999</th>
      <td>20</td>
      <td>200</td>
      <td>8577.238694</td>
      <td>T</td>
      <td>41.142803</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8</td>
      <td>G</td>
    </tr>
    <tr>
      <th>3199</th>
      <td>21</td>
      <td>200</td>
      <td>8556.308713</td>
      <td>T</td>
      <td>42.653847</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7</td>
      <td>G</td>
    </tr>
    <tr>
      <th>3399</th>
      <td>22</td>
      <td>200</td>
      <td>8512.093653</td>
      <td>T</td>
      <td>41.592563</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>G</td>
    </tr>
    <tr>
      <th>3599</th>
      <td>23</td>
      <td>200</td>
      <td>8592.806270</td>
      <td>T</td>
      <td>42.440640</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>14</td>
      <td>G</td>
    </tr>
    <tr>
      <th>3799</th>
      <td>24</td>
      <td>200</td>
      <td>8611.839859</td>
      <td>T</td>
      <td>43.456550</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>21</td>
      <td>G</td>
    </tr>
    <tr>
      <th>3999</th>
      <td>25</td>
      <td>200</td>
      <td>8612.726758</td>
      <td>T</td>
      <td>43.251119</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>22</td>
      <td>G</td>
    </tr>
    <tr>
      <th>4199</th>
      <td>26</td>
      <td>200</td>
      <td>8711.761132</td>
      <td>T</td>
      <td>43.479587</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>29</td>
      <td>G</td>
    </tr>
    <tr>
      <th>4399</th>
      <td>27</td>
      <td>200</td>
      <td>8589.852396</td>
      <td>T</td>
      <td>42.170604</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12</td>
      <td>G</td>
    </tr>
    <tr>
      <th>4599</th>
      <td>28</td>
      <td>200</td>
      <td>8538.054207</td>
      <td>T</td>
      <td>41.867358</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>G</td>
    </tr>
    <tr>
      <th>4799</th>
      <td>29</td>
      <td>200</td>
      <td>8598.513273</td>
      <td>T</td>
      <td>42.691738</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>18</td>
      <td>G</td>
    </tr>
    <tr>
      <th>4999</th>
      <td>30</td>
      <td>200</td>
      <td>8596.053254</td>
      <td>T</td>
      <td>42.729305</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>17</td>
      <td>G</td>
    </tr>
    <tr>
      <th>5199</th>
      <td>32</td>
      <td>200</td>
      <td>8746.356481</td>
      <td>T</td>
      <td>42.857868</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>30</td>
      <td>G</td>
    </tr>
    <tr>
      <th>5399</th>
      <td>33</td>
      <td>200</td>
      <td>8779.878658</td>
      <td>T</td>
      <td>43.756987</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>32</td>
      <td>G</td>
    </tr>
    <tr>
      <th>5599</th>
      <td>59</td>
      <td>200</td>
      <td>8682.515545</td>
      <td>T</td>
      <td>44.102395</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>27</td>
      <td>G</td>
    </tr>
    <tr>
      <th>5799</th>
      <td>60</td>
      <td>200</td>
      <td>8609.335102</td>
      <td>T</td>
      <td>41.643110</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20</td>
      <td>G</td>
    </tr>
    <tr>
      <th>5999</th>
      <td>64</td>
      <td>200</td>
      <td>8599.742854</td>
      <td>T</td>
      <td>41.939070</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>19</td>
      <td>G</td>
    </tr>
    <tr>
      <th>6199</th>
      <td>66</td>
      <td>200</td>
      <td>8704.539425</td>
      <td>T</td>
      <td>42.181555</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>28</td>
      <td>G</td>
    </tr>
    <tr>
      <th>6399</th>
      <td>88</td>
      <td>200</td>
      <td>8623.015960</td>
      <td>T</td>
      <td>42.170190</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>23</td>
      <td>G</td>
    </tr>
    <tr>
      <th>6599</th>
      <td>98</td>
      <td>200</td>
      <td>8595.168102</td>
      <td>T</td>
      <td>41.747804</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>16</td>
      <td>G</td>
    </tr>
  </tbody>
</table>
</div>



### run simulation


```python
# total runs, model runs = 80%
runs = 120
modelruns = 100

maxlaps = max(set(rankdata.completed_laps.values))
carnos = np.sort(list(set(simdf.car_number.values)))
carid = {key:idx for idx, key in enumerate(carnos)}

#laps = [100,200]
laps = range(1,201)
#contigency matrix
#<lap, carno, rank> -> count
cmat = np.zeros((len(laps), len(carnos), len(carnos)))
#simulation result 
simretdf = []
#<run, lap, carno> -> rank
simdata = np.zeros((runs, len(laps), len(carnos)))

# run simulator
for run in range(runs):
    df = run_simulator(rankdata, simdf)
    simretdf.append(df)
    
    for idx, lap in enumerate(laps):
        data = df[df['completed_laps']==lap][['car_number','rank']].to_numpy()
        for pt in data:
            simdata[run, idx, carid[pt[0]]] = pt[1]
    
    
# build statistics model
for run in range(modelruns):    
    df = simretdf[run]
    
    #save rank@lap100 and rank@lap200
    for idx, lap in enumerate(laps):
        data = df[df['completed_laps']==lap][['car_number','rank']].to_numpy()
        for pt in data:
            cmat[idx, carid[pt[0]], pt[1]-1] += 1
    
print('simulation finished!')

#get the test data
simtestdata = simdata[modelruns-runs:,:,:]
print('sim cmat:', cmat.shape)
print('sim testdata:', simtestdata.shape)
```

    simulation finished!
    ('sim cmat:', (200, 33, 33))
    ('sim testdata:', (20, 200, 33))



```python
#save the simulation result
retfile = outputprefix + 'simulator-run%d.dat'%runs
cmatsave = cmat.reshape((-1, len(carnos)*len(carnos)))
np.savetxt(retfile, cmatsave)

retfile = outputprefix + 'simulator-data%d.dat'%runs
simdatasave = simdata.reshape((-1, len(laps)*len(carnos)))
np.savetxt(retfile, simdatasave)

```

### check the result


```python
#check the result
laps = [1,15,32,50,150,200]
#check rank 1@lap100, @lap200
for lap in laps:
    print('rank1@lap',lap, ':', cmat[lap-1,:,0])
    
#print('rank1@lap200:',cmat[1,:,0])
```

    ('rank1@lap', 1, ':', array([ 8.,  9.,  0.,  0.,  0.,  2.,  0., 19.,  1.,  2.,  0.,  0.,  8.,
            1., 18.,  2., 28.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
            1.,  0.,  0.,  0.,  0.,  0.,  0.]))
    ('rank1@lap', 15, ':', array([ 3.,  8.,  0.,  0.,  0.,  0.,  0., 34.,  0.,  9.,  0.,  0.,  0.,
            0., 29.,  0., 16.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.]))
    ('rank1@lap', 32, ':', array([ 0.,  9.,  1.,  0.,  0.,  0.,  0., 38.,  0., 16.,  0.,  0.,  0.,
            0., 24.,  0., 11.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.]))
    ('rank1@lap', 50, ':', array([ 0.,  4.,  0.,  0.,  0.,  0.,  0., 61.,  0.,  5.,  0.,  0.,  0.,
            0., 18.,  0., 10.,  0.,  0.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.]))
    ('rank1@lap', 150, ':', array([ 0.,  0.,  0.,  0.,  0.,  0.,  0., 88.,  0.,  5.,  0.,  0.,  0.,
            0.,  6.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.]))
    ('rank1@lap', 200, ':', array([ 0.,  0.,  0.,  0.,  0.,  0.,  0., 77.,  0., 10.,  0.,  0.,  0.,
            0., 13.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.]))


#### check the result of will power


```python
cmat[:,7,0]
```




    array([19., 21., 27., 26., 29., 32., 27., 29., 27., 27., 28., 30., 29.,
           29., 34., 30., 33., 31., 33., 38., 42., 38., 40., 37., 41., 39.,
           39., 38., 42., 38., 37., 38., 33., 31., 30., 22., 14., 26., 55.,
           60., 56., 60., 60., 60., 59., 60., 58., 62., 62., 61., 61., 65.,
           66., 61., 64., 66., 62., 62., 62., 62., 61., 62., 61., 63., 53.,
           47., 45., 42., 38., 29., 24., 24., 34., 49., 65., 68., 69., 71.,
           72., 70., 71., 71., 75., 70., 69., 73., 65., 69., 72., 71., 69.,
           74., 72., 68., 68., 65., 63., 58., 53., 54., 52., 46., 38., 37.,
           40., 46., 52., 57., 68., 71., 74., 77., 80., 81., 80., 80., 81.,
           79., 81., 79., 81., 82., 83., 81., 80., 80., 82., 78., 76., 74.,
           70., 66., 64., 57., 51., 49., 43., 40., 46., 57., 64., 74., 79.,
           79., 84., 85., 89., 88., 87., 88., 86., 85., 85., 86., 87., 88.,
           85., 87., 89., 87., 87., 84., 81., 78., 76., 70., 59., 62., 58.,
           52., 53., 50., 51., 57., 64., 72., 81., 83., 86., 89., 88., 88.,
           88., 88., 87., 87., 88., 88., 87., 88., 88., 87., 87., 85., 80.,
           79., 78., 79., 77., 77.])




```python
leader = rankdata[rankdata['rank']==1]
willpower_in_lead = leader[leader['car_number']==12].completed_laps.values
willpower_pred_lead = cmat[:,7,0]
willpower_in_lead
```




    array([ 92,  93,  94, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
           118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 141, 142,
           143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
           156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
           169, 170, 196, 197, 198, 199, 200])




```python
willpower_in_lead.astype(int)-1
```




    array([ 91,  92,  93, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
           117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 140, 141,
           142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154,
           155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,
           168, 169, 195, 196, 197, 198, 199])




```python
willpower_pred_lead[willpower_in_lead.astype(int)-1]
```




    array([74., 72., 68., 57., 68., 71., 74., 77., 80., 81., 80., 80., 81.,
           79., 81., 79., 81., 82., 83., 81., 80., 80., 82., 78., 64., 74.,
           79., 79., 84., 85., 89., 88., 87., 88., 86., 85., 85., 86., 87.,
           88., 85., 87., 89., 87., 87., 84., 81., 78., 76., 70., 59., 62.,
           58., 52., 79., 78., 79., 77., 77.])




```python
laps = range(1,201)
plt.plot(laps, willpower_pred_lead,'.')
plt.plot(willpower_in_lead, willpower_pred_lead[willpower_in_lead.astype(int)-1],'+')
plt.show()
```


![png](output_24_0.png)


#### performance of the simulator prediction


```python
#model preidction on first 80 simulation results 
laps = range(1,201)
simulator_pred = []
for lap in laps:
    simulator_pred.append(np.argmax(cmat[lap-1,:,0]))
    
simulator_pred = np.array(simulator_pred)
simulator_pred
```




    array([16, 16,  7,  7,  7,  7, 14, 14, 14, 14, 14, 14,  7, 14,  7, 14,  7,
           14,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
            7,  7,  1,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
            7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
            7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
            7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
            7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
            7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
            7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
            7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
            7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
            7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7])




```python
#
#  over the groundtruth
#
# test on the latest 20
testruns = runs - modelruns
testret = np.zeros((testruns, len(simulator_pred)))
for run in range(modelruns, runs):
    df = simretdf[run]
    leader = df[df['rank']==1]
    leader = leader.rename_axis('MyIdx').sort_values(by=['completed_laps','MyIdx'], ascending=True)
    ground_truth = []
    for car in leader.car_number.values:
        ground_truth.append(carid[car] )
    ground_truth = np.array(ground_truth)
    
    idx = ((simulator_pred - ground_truth) == 0)
    testret[run-modelruns,:] = idx
#show result    
print('mean rank1 accuracy:', np.mean(testret), np.mean(testret, axis=1))

#check rank1@lap10, fixed prediction
laps = [1,15,32,50,150,200]
for lap in laps:
    print('mean rank1@lap',lap, 'accuracy:', np.mean(testret[:,lap-1]))
#print('mean rank1@lap10 accuracy:', np.mean(testret[:,9]))
#print('mean rank1@lap10 accuracy:', np.mean(testret[:,9]))

```

    ('mean rank1 accuracy:', 0.56575, array([0.26 , 0.725, 0.865, 0.15 , 0.625, 0.9  , 0.075, 0.425, 0.58 ,
           0.435, 0.46 , 0.805, 0.93 , 0.055, 0.645, 0.67 , 0.71 , 0.885,
           0.49 , 0.625]))
    ('mean rank1@lap', 1, 'accuracy:', 0.15)
    ('mean rank1@lap', 15, 'accuracy:', 0.35)
    ('mean rank1@lap', 32, 'accuracy:', 0.45)
    ('mean rank1@lap', 50, 'accuracy:', 0.45)
    ('mean rank1@lap', 150, 'accuracy:', 0.85)
    ('mean rank1@lap', 200, 'accuracy:', 0.8)



```python
    run = modelruns + 6
    df = simretdf[run]
    leader = df[df['rank']==1]
    
    leader = leader.rename_axis('MyIdx').sort_values(by=['completed_laps','MyIdx'], ascending=True)
    
    ground_truth = []
    for car in leader.car_number.values:
        ground_truth.append(carid[car] )
    ground_truth = np.array(ground_truth)
    ground_truth
```




    array([12, 15, 15,  7,  7,  7,  1, 12, 16, 23, 23, 23,  1,  1,  7,  7,  7,
            7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 14,  7, 14, 14, 14, 14, 14,
           14,  9,  9,  9, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
           14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
           14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
           14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
           14,  9,  9,  9, 22, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
           14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
            9,  9,  9,  9,  9,  9,  9, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
           14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,  9,  9,  9,
            9,  9,  9,  9,  9,  9,  9,  9,  9, 14, 14, 14, 14, 14, 14, 14, 14,
           14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14])




```python
leader
```


```python
#
#  over the realrace truth
#
ground_truth = []
leader = rankdata[rankdata['rank']==1]
leader = leader.rename_axis('MyIdx').sort_values(by=['completed_laps','MyIdx'], ascending=True)
for car in leader.car_number.values:
    ground_truth.append(carid[car] )
len(ground_truth)
ground_truth = np.array(ground_truth)
ground_truth
```




    array([14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
           14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,  0, 15, 15,
           15, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
           13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14,  9,  9, 14, 14, 14,
           14, 14, 14, 14, 14,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,
            9,  9,  9,  9,  9, 14, 14,  7,  7,  7, 29, 12, 10, 10, 10, 10, 10,
           10, 10, 10, 10, 13, 13,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
            7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 22, 12, 12, 12,  0,  0, 10,
           10, 10, 23, 23, 23,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
            7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
            7, 14, 14, 21, 16, 23, 29, 29,  3,  3, 29, 29, 29, 29, 29, 29, 29,
           29, 29, 29, 29, 29, 29, 19, 19, 19,  7,  7,  7,  7,  7])




```python
idx = ((simulator_pred - ground_truth[1:]) == 0)
print(np.mean(idx), np.mean(idx[100:]),np.mean(idx[:50]),np.mean(idx[:100]),np.mean(idx[:150]))
```

    (0.34, 0.56, 0.18, 0.12, 0.2866666666666667)



```python
# very low accuracy
idx = (simulator_pred - ground_truth[1:]) == 0
simulator_pred[idx]

```




    array([14, 14, 14, 14, 14, 14, 14, 14, 14,  7,  7,  7,  7,  7,  7,  7,  7,
            7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
            7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
            7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7])



### conclusion

even such a simple simulator demonstrates the uncertainty of the long term rank distribution.


```python

```
