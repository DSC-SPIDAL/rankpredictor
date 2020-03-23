# Rank1 baseline model

input: laptime&rank dataset
<eventid, carids, laptime (totalcars x totallaps), rank (totalcars x totallaps)>; filled with NaN

evaluate the rank1 prediction task on baseline

1. predict the car number of rank1 2 laps later
2. CurRank model




```python
# Third-party imports
%matplotlib inline
import mxnet as mx
from mxnet import gluon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
```

## Datasets



```python
import pickle
with open('laptime_rank-2018.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    global_carids, laptime_data = pickle.load(f, encoding='latin1')
```


```python
events = ['Phoenix','Indy500','Texas','Iowa','Pocono','Gateway']
events_id={key:idx for idx, key in enumerate(events)}
```


```python
print(f"events: {events}")
```

    events: ['Phoenix', 'Indy500', 'Texas', 'Iowa', 'Pocono', 'Gateway']



```python
print(laptime_data[1][3].shape)
rank1 = np.nanargmin(laptime_data[1][3],axis=0)
print(len(rank1), rank1[-1])
print(np.array(rank1))
np.array([laptime_data[1][1][x] for x in rank1])
```

    (33, 200)
    200 7
    [14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14
     14 14 14 14 14 14  0 15 15 15 14 14 14 14 14 14 14 14 14 14 14 14 14 14
     14 14 13 13 13 13 13 14 14 14 14 14 14 14  9  9 14 14 14 14 14 14 14 14
      9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9 14 14  7  7  7 29 12
     10 10 10 10 10 10 10 10 10 13 13  7  7  7  7  7  7  7  7  7  7  7  7  7
      7  7  7  7  7  7  7  7 22 12 12 12  0  0 10 10 10 23 23 23  7  7  7  7
      7  7  7  7  7  7  7  7  7  7  7  7  7  7  7  7  7  7  7  7  7  7  7  7
      7  7 14 14 21 16 23 29 29  3  3 29 29 29 29 29 29 29 29 29 29 29 29 29
     19 19 19  7  7  7  7  7]





    array([20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
           20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,  1, 21, 21, 21,
           20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 19,
           19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 14, 14, 20, 20, 20, 20,
           20, 20, 20, 20, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
           14, 14, 14, 14, 20, 20, 12, 12, 12, 64, 18, 15, 15, 15, 15, 15, 15,
           15, 15, 15, 19, 19, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
           12, 12, 12, 12, 12, 12, 12, 12, 12, 28, 18, 18, 18,  1,  1, 15, 15,
           15, 29, 29, 29, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
           12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
           20, 20, 27, 22, 29, 64, 64,  6,  6, 64, 64, 64, 64, 64, 64, 64, 64,
           64, 64, 64, 64, 64, 25, 25, 25, 12, 12, 12, 12, 12])




```python
TS_RANK=3
rank1_ts = []
#_data: eventid, carids, laptime array
for _data in laptime_data:
    #rank data
    rank1_index = np.nanargmin(_data[TS_RANK],axis=0)
    rank1_ts.append([_data[1][x] for x in rank1_index])

```


```python
#task: predict 2 laps later
#predict_len = 2
def eval_curmodel(eventid, predict_len = 2):
    indy_ts = rank1_ts[eventid]

    #start lap 100 
    y = np.array(indy_ts[100 + predict_len:])
    y_pred = np.array(indy_ts[100:-predict_len])

    #evaluate
    accuracy = np.sum(y_pred == y)*1.0 / len(y)
    print('predict_len=', predict_len, 'accuracy=', accuracy)

#test
for eventid in range(len(events)):
    print('evaluate for', events[eventid])
    for tlen in range(2,10):
        eval_curmodel(eventid, tlen)      


```

    evaluate for Phoenix
    predict_len= 2 accuracy= 0.8581081081081081
    predict_len= 3 accuracy= 0.7959183673469388
    predict_len= 4 accuracy= 0.7397260273972602
    predict_len= 5 accuracy= 0.6896551724137931
    predict_len= 6 accuracy= 0.6527777777777778
    predict_len= 7 accuracy= 0.6153846153846154
    predict_len= 8 accuracy= 0.5774647887323944
    predict_len= 9 accuracy= 0.5390070921985816
    evaluate for Indy500
    predict_len= 2 accuracy= 0.6938775510204082
    predict_len= 3 accuracy= 0.6185567010309279
    predict_len= 4 accuracy= 0.5833333333333334
    predict_len= 5 accuracy= 0.5368421052631579
    predict_len= 6 accuracy= 0.5106382978723404
    predict_len= 7 accuracy= 0.4838709677419355
    predict_len= 8 accuracy= 0.45652173913043476
    predict_len= 9 accuracy= 0.42857142857142855
    evaluate for Texas
    predict_len= 2 accuracy= 0.958904109589041
    predict_len= 3 accuracy= 0.9517241379310345
    predict_len= 4 accuracy= 0.9444444444444444
    predict_len= 5 accuracy= 0.9370629370629371
    predict_len= 6 accuracy= 0.9295774647887324
    predict_len= 7 accuracy= 0.9219858156028369
    predict_len= 8 accuracy= 0.9142857142857143
    predict_len= 9 accuracy= 0.9064748201438849
    evaluate for Iowa
    predict_len= 2 accuracy= 0.9696969696969697
    predict_len= 3 accuracy= 0.9543147208121827
    predict_len= 4 accuracy= 0.9489795918367347
    predict_len= 5 accuracy= 0.9435897435897436
    predict_len= 6 accuracy= 0.9381443298969072
    predict_len= 7 accuracy= 0.9326424870466321
    predict_len= 8 accuracy= 0.9270833333333334
    predict_len= 9 accuracy= 0.9214659685863874
    evaluate for Pocono
    predict_len= 2 accuracy= 0.9183673469387755
    predict_len= 3 accuracy= 0.8762886597938144
    predict_len= 4 accuracy= 0.8541666666666666
    predict_len= 5 accuracy= 0.8315789473684211
    predict_len= 6 accuracy= 0.8297872340425532
    predict_len= 7 accuracy= 0.8279569892473119
    predict_len= 8 accuracy= 0.8260869565217391
    predict_len= 9 accuracy= 0.8241758241758241
    evaluate for Gateway
    predict_len= 2 accuracy= 0.9041095890410958
    predict_len= 3 accuracy= 0.8689655172413793
    predict_len= 4 accuracy= 0.8333333333333334
    predict_len= 5 accuracy= 0.8041958041958042
    predict_len= 6 accuracy= 0.7746478873239436
    predict_len= 7 accuracy= 0.75177304964539
    predict_len= 8 accuracy= 0.7357142857142858
    predict_len= 9 accuracy= 0.7194244604316546



```python

```
