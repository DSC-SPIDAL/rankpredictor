
### stage_model

prediction models on stage dataset

data format:
    target , eventid ,    car_number,    stageid,     features...


```python
%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# to use only one GPU.
# use this on r-001
# otherwise comment
import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"
```


```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import xgboost as xgb

```


```python
# bulid regression model
classifiers = ['currank','avgrank','dice','lr','lrl1','lsvc','lsvcl2','rf','lrbias','xgb']
def get_classifier(classifier = 'lr'):
    
    class_weight = None
    
    if classifier == "lsvc":
        clf = LinearSVC(penalty='l1',dual=False, tol=1e-3, class_weight=class_weight )
    elif classifier == "lsvcl2":
        clf = LinearSVC(penalty='l2', tol=1e-4, class_weight=class_weight)
    elif classifier == 'rf':
        #clf = RandomForestClassifier(n_estimators=100, n_jobs=4,criterion='entropy', min_samples_split=1,class_weight = class_weight)
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1,criterion='entropy', class_weight = class_weight)
    elif classifier == 'lr':
        clf = LogisticRegression(class_weight = class_weight, n_jobs=-1, fit_intercept = False, verbose = 0)
    elif classifier == 'lrbias':
        clf = LogisticRegression(class_weight = class_weight, n_jobs=-1, fit_intercept = True, verbose = 1)
    elif classifier == 'lrl1':
        clf = LogisticRegression(class_weight = class_weight, penalty='l1',n_jobs=-1)
    elif classifier == 'xgb':
        clf = xgb.XGBClassifier(booster = 'gbtree', nthread = -1, subsample = 1, n_estimators = 600, colsample_bytree = 1, max_depth = 3, min_child_weight = 1)
    elif classifier == 'dice':
        clf = RandomDice('1234')
    elif classifier == 'currank':
        clf = CurRank()
    elif classifier == 'avgrank':
        clf = AverageRank()        
    else:
        clf = None
        
    return clf


class CurRank():
    """
    predict with current rank
    """
    def __init__(self):
        pass
    def fit(self, x, y):
        pass
    def predict(self, test_x):
        pred_y = [0 for x in range(test_x.shape[0])]
        return np.array(pred_y)
    
class AverageRank():
    """
    print('[*] predict with average rankchg (change_in_rank_all):idx = 15')
    change_in_rank_all = test[:,15]
    pred_y_avg = np.array([1 if x > 0 else (-1 if x < 0 else 0) for x in change_in_rank_all])
    """
    def __init__(self):
        pass
    def fit(self, x, y):
        pass
    def predict(self, test_x):
        pred_y = []
        for x in test_x:
            #13, change_in_rank_all
            pred_y.append(x[13])
        pred_y_avg = np.array([1 if x > 0 else (-1 if x < 0 else 0) for x in pred_y])
        return np.array(pred_y_avg)   

class RandomDice():
    """
    a random dice model
    """
    def __init__(self, seed='1234'):
        self.dist = []
        self.val = []
        random.seed(seed)
    
    def fit(self, x, y):
        total = y.shape[0]
        yval = set(y)
        
        ratio = 0.
        for val in yval:
            self.val.append(val)
            ratio += np.sum(y==val)*1.0 / total
            self.dist.append(ratio)
            
    def predict(self, test_x):
        pred_y = []
        for x in test_x:
            dice = random.random()
            #search in self.dist
            find_idx = -1
            for idx, ratio in enumerate(self.dist):
                if dice <= ratio:
                    find_idx = idx
                    break
            
            #or the last one match
            pred_y.append(self.val[find_idx])
            
        return np.array(pred_y)

def evaluate(test_y, pred_y):
    precision = metrics.precision_score(test_y, pred_y, average=None) 
    recall = metrics.recall_score(test_y, pred_y, average=None)
    f1 = metrics.f1_score(test_y, pred_y, average=None)
    accuracy = metrics.accuracy_score(test_y, pred_y)
    print('precision=%s, recall=%s, f1=%s, accuracy=%.2f'%(precision,recall, f1, accuracy))
    return accuracy
    
#
#features
#    cols=[Myidx, 'target','eventid','car_number','stageid',
#             'firststage','pit_in_caution','start_position',
#             'start_rank','start_rank_ratio','top_pack','bottom_pack',
#             'average_rank','average_rank_all',
#             'change_in_rank','change_in_rank_all','rate_of_change','rate_of_change_all']    
def split_by_eventid(stagedata, eventid):
    """
    split by eventid
    """
    #if not eventid in stagedata:
    #    print('error, %d not found in stagedata'%eventid)
    #    return
    
    train = stagedata[stagedata['eventid'] != eventid].to_numpy()
    test  = stagedata[stagedata['eventid'] == eventid].to_numpy()

    #2:car_number
    train_x = train[:,2:]
    train_y = np.array([1 if x > 0 else (-1 if x < 0 else 0) for x in train[:,1]])
    test_x = test[:,2:]
    test_y = np.array([1 if x > 0 else (-1 if x < 0 else 0) for x in test[:,1]])
    
    return train, test, train_x, train_y, test_x, test_y


def split_by_stageid(stagedata, stageid):
    """
    split by stageid
    """
    #if not eventid in stagedata:
    #    print('error, %d not found in stagedata'%eventid)
    #    return
    
    train = stagedata[stagedata['stageid'] <= stageid].to_numpy()
    test  = stagedata[stagedata['stageid'] > stageid].to_numpy()

    train_x = train[:,2:]
    train_y = np.array([1 if x > 0 else (-1 if x < 0 else 0) for x in train[:,1]])
    test_x = test[:,2:]
    test_y = np.array([1 if x > 0 else (-1 if x < 0 else 0) for x in test[:,1]])
    
    return train, test, train_x, train_y, test_x, test_y


### baseline
def baseline_model():
    #1. predict with current rank, rankchg = 0
    print('[*] predict with current rank, rankchg = 0')
    pred_y_simple = np.zeros_like(test_y)
    score1 = evaluate(test_y, pred_y_simple)

    #2. predict with average rankchg (change_in_rank_all):idx = 15
    print('[*] predict with average rankchg (change_in_rank_all):idx = 15')
    change_in_rank_all = test[:,15]
    pred_y_avg = np.array([1 if x > 0 else (-1 if x < 0 else 0) for x in change_in_rank_all])
    score2 = evaluate(test_y, pred_y_avg)
    return score1, score2

def classifier_model(name='lr'):
    ### test learning models
    print('[*] predict with %s model'%name)
    clf = get_classifier(name)
    clf.fit(train_x, train_y)

    pred_y = clf.predict(test_x)
    score = evaluate(test_y, pred_y)
    return score
```


```python
#load data
stagedata = pd.read_csv('stage-2018.csv')
stagedata.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 805 entries, 0 to 804
    Data columns (total 18 columns):
    Unnamed: 0            805 non-null int64
    target                805 non-null int64
    eventid               805 non-null int64
    car_number            805 non-null int64
    stageid               805 non-null int64
    firststage            805 non-null int64
    pit_in_caution        805 non-null int64
    start_position        805 non-null int64
    start_rank            805 non-null int64
    start_rank_ratio      805 non-null float64
    top_pack              805 non-null int64
    bottom_pack           805 non-null int64
    average_rank          805 non-null float64
    average_rank_all      805 non-null float64
    change_in_rank        805 non-null int64
    change_in_rank_all    805 non-null float64
    rate_of_change        805 non-null int64
    rate_of_change_all    805 non-null float64
    dtypes: float64(5), int64(13)
    memory usage: 113.3 KB



```python
stagedata
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
      <th>Unnamed: 0</th>
      <th>target</th>
      <th>eventid</th>
      <th>car_number</th>
      <th>stageid</th>
      <th>firststage</th>
      <th>pit_in_caution</th>
      <th>start_position</th>
      <th>start_rank</th>
      <th>start_rank_ratio</th>
      <th>top_pack</th>
      <th>bottom_pack</th>
      <th>average_rank</th>
      <th>average_rank_all</th>
      <th>change_in_rank</th>
      <th>change_in_rank_all</th>
      <th>rate_of_change</th>
      <th>rate_of_change_all</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>7</td>
      <td>0.304348</td>
      <td>0</td>
      <td>0</td>
      <td>7.000000</td>
      <td>7.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-5</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>7</td>
      <td>0.304348</td>
      <td>0</td>
      <td>0</td>
      <td>7.025641</td>
      <td>7.025641</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>2</td>
      <td>0.086957</td>
      <td>1</td>
      <td>0</td>
      <td>2.328947</td>
      <td>3.921739</td>
      <td>-5</td>
      <td>-2.500000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>-4</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>5</td>
      <td>0.217391</td>
      <td>1</td>
      <td>0</td>
      <td>5.736842</td>
      <td>4.523256</td>
      <td>3</td>
      <td>-0.666667</td>
      <td>8</td>
      <td>1.500000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>0.043478</td>
      <td>1</td>
      <td>0</td>
      <td>3.071429</td>
      <td>4.166667</td>
      <td>-4</td>
      <td>-1.500000</td>
      <td>-7</td>
      <td>-1.333333</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>-1</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>16</td>
      <td>0.695652</td>
      <td>0</td>
      <td>0</td>
      <td>16.000000</td>
      <td>16.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>-3</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>16</td>
      <td>15</td>
      <td>0.652174</td>
      <td>0</td>
      <td>0</td>
      <td>15.025641</td>
      <td>15.025641</td>
      <td>-1</td>
      <td>-1.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>3</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>16</td>
      <td>12</td>
      <td>0.521739</td>
      <td>0</td>
      <td>0</td>
      <td>14.500000</td>
      <td>14.976744</td>
      <td>-3</td>
      <td>-2.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>7</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>16</td>
      <td>15</td>
      <td>0.652174</td>
      <td>0</td>
      <td>0</td>
      <td>16.887324</td>
      <td>16.166667</td>
      <td>3</td>
      <td>-0.333333</td>
      <td>6</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>-1</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>16</td>
      <td>22</td>
      <td>0.956522</td>
      <td>0</td>
      <td>1</td>
      <td>10.600000</td>
      <td>15.932773</td>
      <td>7</td>
      <td>1.500000</td>
      <td>4</td>
      <td>2.666667</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>-1</td>
      <td>0</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>16</td>
      <td>21</td>
      <td>0.913043</td>
      <td>0</td>
      <td>1</td>
      <td>21.660377</td>
      <td>17.697674</td>
      <td>-1</td>
      <td>1.000000</td>
      <td>-8</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>-1</td>
      <td>0</td>
      <td>4</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>16</td>
      <td>20</td>
      <td>0.869565</td>
      <td>0</td>
      <td>1</td>
      <td>20.041667</td>
      <td>18.209091</td>
      <td>-1</td>
      <td>0.666667</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>3</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0.217391</td>
      <td>1</td>
      <td>0</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>-3</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>8</td>
      <td>0.347826</td>
      <td>0</td>
      <td>0</td>
      <td>7.897436</td>
      <td>7.897436</td>
      <td>3</td>
      <td>3.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>-3</td>
      <td>0</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0.217391</td>
      <td>1</td>
      <td>0</td>
      <td>5.112676</td>
      <td>6.100000</td>
      <td>-3</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>2</td>
      <td>0.086957</td>
      <td>1</td>
      <td>0</td>
      <td>3.047619</td>
      <td>4.988439</td>
      <td>-3</td>
      <td>-1.000000</td>
      <td>0</td>
      <td>-3.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>-1</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>0.260870</td>
      <td>0</td>
      <td>0</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>-2</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>5</td>
      <td>0.217391</td>
      <td>1</td>
      <td>0</td>
      <td>5.025641</td>
      <td>5.025641</td>
      <td>-1</td>
      <td>-1.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>-2</td>
      <td>0</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>3</td>
      <td>0.130435</td>
      <td>1</td>
      <td>0</td>
      <td>3.138889</td>
      <td>3.801802</td>
      <td>-2</td>
      <td>-1.500000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>0.043478</td>
      <td>1</td>
      <td>0</td>
      <td>2.825397</td>
      <td>3.448276</td>
      <td>-2</td>
      <td>-1.666667</td>
      <td>0</td>
      <td>-0.500000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>1</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>17</td>
      <td>17</td>
      <td>0.739130</td>
      <td>0</td>
      <td>0</td>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>-8</td>
      <td>0</td>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>17</td>
      <td>18</td>
      <td>0.782609</td>
      <td>0</td>
      <td>1</td>
      <td>17.974359</td>
      <td>17.974359</td>
      <td>1</td>
      <td>1.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22</td>
      <td>-3</td>
      <td>0</td>
      <td>9</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>17</td>
      <td>10</td>
      <td>0.434783</td>
      <td>0</td>
      <td>0</td>
      <td>11.301370</td>
      <td>13.625000</td>
      <td>-8</td>
      <td>-3.500000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>23</td>
      <td>-2</td>
      <td>0</td>
      <td>9</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>17</td>
      <td>7</td>
      <td>0.304348</td>
      <td>0</td>
      <td>0</td>
      <td>8.622951</td>
      <td>11.861272</td>
      <td>-3</td>
      <td>-3.333333</td>
      <td>5</td>
      <td>-2.000000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>24</td>
      <td>-1</td>
      <td>0</td>
      <td>9</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>17</td>
      <td>5</td>
      <td>0.217391</td>
      <td>1</td>
      <td>0</td>
      <td>6.563636</td>
      <td>10.583333</td>
      <td>-2</td>
      <td>-3.000000</td>
      <td>1</td>
      <td>-1.000000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>25</td>
      <td>-1</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>11</td>
      <td>0.478261</td>
      <td>0</td>
      <td>0</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>26</td>
      <td>-5</td>
      <td>0</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>11</td>
      <td>10</td>
      <td>0.434783</td>
      <td>0</td>
      <td>0</td>
      <td>10.051282</td>
      <td>10.051282</td>
      <td>-1</td>
      <td>-1.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>27</td>
      <td>-1</td>
      <td>0</td>
      <td>10</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>11</td>
      <td>5</td>
      <td>0.217391</td>
      <td>1</td>
      <td>0</td>
      <td>6.162162</td>
      <td>7.504425</td>
      <td>-5</td>
      <td>-3.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>28</td>
      <td>15</td>
      <td>0</td>
      <td>10</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>11</td>
      <td>4</td>
      <td>0.173913</td>
      <td>1</td>
      <td>0</td>
      <td>4.758621</td>
      <td>6.573099</td>
      <td>-1</td>
      <td>-2.333333</td>
      <td>4</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>29</th>
      <td>29</td>
      <td>1</td>
      <td>0</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0.130435</td>
      <td>1</td>
      <td>0</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>775</th>
      <td>74</td>
      <td>-1</td>
      <td>5</td>
      <td>27</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>0.142857</td>
      <td>1</td>
      <td>0</td>
      <td>3.701754</td>
      <td>3.701754</td>
      <td>1</td>
      <td>1.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>776</th>
      <td>75</td>
      <td>1</td>
      <td>5</td>
      <td>27</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0.095238</td>
      <td>1</td>
      <td>0</td>
      <td>3.064516</td>
      <td>3.369748</td>
      <td>-1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>777</th>
      <td>76</td>
      <td>-1</td>
      <td>5</td>
      <td>27</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>0.142857</td>
      <td>1</td>
      <td>0</td>
      <td>2.981481</td>
      <td>3.248555</td>
      <td>1</td>
      <td>0.333333</td>
      <td>2</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>778</th>
      <td>77</td>
      <td>0</td>
      <td>5</td>
      <td>28</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0.238095</td>
      <td>1</td>
      <td>0</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>779</th>
      <td>78</td>
      <td>1</td>
      <td>5</td>
      <td>28</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0.238095</td>
      <td>1</td>
      <td>0</td>
      <td>5.254545</td>
      <td>5.254545</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>780</th>
      <td>79</td>
      <td>-3</td>
      <td>5</td>
      <td>28</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>6</td>
      <td>0.285714</td>
      <td>0</td>
      <td>0</td>
      <td>6.298246</td>
      <td>5.785714</td>
      <td>1</td>
      <td>0.500000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>781</th>
      <td>80</td>
      <td>0</td>
      <td>5</td>
      <td>28</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>0.142857</td>
      <td>1</td>
      <td>0</td>
      <td>5.017857</td>
      <td>5.529762</td>
      <td>-3</td>
      <td>-0.666667</td>
      <td>-4</td>
      <td>-1.500000</td>
    </tr>
    <tr>
      <th>782</th>
      <td>81</td>
      <td>16</td>
      <td>5</td>
      <td>28</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>0.142857</td>
      <td>1</td>
      <td>0</td>
      <td>3.000000</td>
      <td>5.514793</td>
      <td>0</td>
      <td>-0.500000</td>
      <td>3</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>783</th>
      <td>82</td>
      <td>-8</td>
      <td>5</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>13</td>
      <td>0.619048</td>
      <td>0</td>
      <td>0</td>
      <td>13.000000</td>
      <td>13.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>784</th>
      <td>83</td>
      <td>-3</td>
      <td>5</td>
      <td>30</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>13</td>
      <td>5</td>
      <td>0.238095</td>
      <td>1</td>
      <td>0</td>
      <td>12.250000</td>
      <td>12.250000</td>
      <td>-8</td>
      <td>-8.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>785</th>
      <td>84</td>
      <td>7</td>
      <td>5</td>
      <td>30</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>13</td>
      <td>2</td>
      <td>0.095238</td>
      <td>1</td>
      <td>0</td>
      <td>13.080645</td>
      <td>12.672131</td>
      <td>-3</td>
      <td>-5.500000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>786</th>
      <td>85</td>
      <td>0</td>
      <td>5</td>
      <td>30</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>13</td>
      <td>9</td>
      <td>0.428571</td>
      <td>0</td>
      <td>0</td>
      <td>11.163636</td>
      <td>12.203390</td>
      <td>7</td>
      <td>-1.333333</td>
      <td>10</td>
      <td>7.500000</td>
    </tr>
    <tr>
      <th>787</th>
      <td>86</td>
      <td>0</td>
      <td>5</td>
      <td>59</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>20</td>
      <td>0.952381</td>
      <td>0</td>
      <td>1</td>
      <td>20.000000</td>
      <td>20.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>788</th>
      <td>87</td>
      <td>-1</td>
      <td>5</td>
      <td>59</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>20</td>
      <td>20</td>
      <td>0.952381</td>
      <td>0</td>
      <td>1</td>
      <td>20.000000</td>
      <td>20.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>789</th>
      <td>88</td>
      <td>-2</td>
      <td>5</td>
      <td>59</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>20</td>
      <td>19</td>
      <td>0.904762</td>
      <td>0</td>
      <td>1</td>
      <td>19.890909</td>
      <td>19.890909</td>
      <td>-1</td>
      <td>-1.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>790</th>
      <td>89</td>
      <td>-2</td>
      <td>5</td>
      <td>59</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>20</td>
      <td>17</td>
      <td>0.809524</td>
      <td>0</td>
      <td>1</td>
      <td>19.685185</td>
      <td>19.788991</td>
      <td>-2</td>
      <td>-1.500000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>791</th>
      <td>90</td>
      <td>2</td>
      <td>5</td>
      <td>59</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>20</td>
      <td>15</td>
      <td>0.714286</td>
      <td>0</td>
      <td>0</td>
      <td>18.090909</td>
      <td>19.219512</td>
      <td>-2</td>
      <td>-1.666667</td>
      <td>0</td>
      <td>-0.500000</td>
    </tr>
    <tr>
      <th>792</th>
      <td>91</td>
      <td>0</td>
      <td>5</td>
      <td>59</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>20</td>
      <td>17</td>
      <td>0.809524</td>
      <td>0</td>
      <td>1</td>
      <td>16.954545</td>
      <td>18.569565</td>
      <td>2</td>
      <td>-0.750000</td>
      <td>4</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>793</th>
      <td>92</td>
      <td>-6</td>
      <td>5</td>
      <td>88</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>19</td>
      <td>0.904762</td>
      <td>0</td>
      <td>1</td>
      <td>19.000000</td>
      <td>19.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>794</th>
      <td>93</td>
      <td>-2</td>
      <td>5</td>
      <td>88</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>19</td>
      <td>13</td>
      <td>0.619048</td>
      <td>0</td>
      <td>0</td>
      <td>13.547170</td>
      <td>13.547170</td>
      <td>-6</td>
      <td>-6.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>795</th>
      <td>94</td>
      <td>0</td>
      <td>5</td>
      <td>88</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>19</td>
      <td>11</td>
      <td>0.523810</td>
      <td>0</td>
      <td>0</td>
      <td>11.807692</td>
      <td>12.685714</td>
      <td>-2</td>
      <td>-4.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>796</th>
      <td>95</td>
      <td>9</td>
      <td>5</td>
      <td>88</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>19</td>
      <td>11</td>
      <td>0.523810</td>
      <td>0</td>
      <td>0</td>
      <td>11.000000</td>
      <td>12.654206</td>
      <td>0</td>
      <td>-2.666667</td>
      <td>2</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>797</th>
      <td>96</td>
      <td>-1</td>
      <td>5</td>
      <td>88</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>19</td>
      <td>20</td>
      <td>0.952381</td>
      <td>0</td>
      <td>1</td>
      <td>19.603774</td>
      <td>14.956250</td>
      <td>9</td>
      <td>0.250000</td>
      <td>9</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>798</th>
      <td>97</td>
      <td>-1</td>
      <td>5</td>
      <td>88</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>19</td>
      <td>19</td>
      <td>0.904762</td>
      <td>0</td>
      <td>1</td>
      <td>19.166667</td>
      <td>15.250000</td>
      <td>-1</td>
      <td>0.000000</td>
      <td>-10</td>
      <td>1.250000</td>
    </tr>
    <tr>
      <th>799</th>
      <td>98</td>
      <td>0</td>
      <td>5</td>
      <td>88</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>19</td>
      <td>18</td>
      <td>0.857143</td>
      <td>0</td>
      <td>1</td>
      <td>18.019231</td>
      <td>15.892857</td>
      <td>-1</td>
      <td>-0.166667</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>800</th>
      <td>99</td>
      <td>-2</td>
      <td>5</td>
      <td>98</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>10</td>
      <td>0.476190</td>
      <td>0</td>
      <td>0</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>801</th>
      <td>100</td>
      <td>0</td>
      <td>5</td>
      <td>98</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>8</td>
      <td>0.380952</td>
      <td>0</td>
      <td>0</td>
      <td>7.773585</td>
      <td>7.773585</td>
      <td>-2</td>
      <td>-2.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>802</th>
      <td>101</td>
      <td>0</td>
      <td>5</td>
      <td>98</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>8</td>
      <td>0.380952</td>
      <td>0</td>
      <td>0</td>
      <td>8.660714</td>
      <td>8.229358</td>
      <td>0</td>
      <td>-1.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>803</th>
      <td>102</td>
      <td>5</td>
      <td>5</td>
      <td>98</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>8</td>
      <td>0.380952</td>
      <td>0</td>
      <td>0</td>
      <td>9.472727</td>
      <td>8.646341</td>
      <td>0</td>
      <td>-0.666667</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>804</th>
      <td>103</td>
      <td>1</td>
      <td>5</td>
      <td>98</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>13</td>
      <td>0.619048</td>
      <td>0</td>
      <td>0</td>
      <td>14.021739</td>
      <td>9.823810</td>
      <td>5</td>
      <td>0.750000</td>
      <td>5</td>
      <td>2.333333</td>
    </tr>
  </tbody>
</table>
<p>805 rows × 18 columns</p>
</div>




```python
cols = ['runid','trainsize','testsize','testdistribution']
cols.extend(classifiers)
print('cols:%s'%cols)
retdf = pd.DataFrame([],columns=cols)

eventsname = ['Phoenix','Indy500','Texas','Iowa','Pocono','Gateway']
events = set(stagedata['eventid'])
for eventid in events:
    print('Testset = %s'%eventsname[eventid])
    
    train, test, train_x, train_y, test_x, test_y = split_by_eventid(stagedata, eventid)
    test_distribution = '+:%d,0:%d,-:%d'%(np.sum(test_y>0),np.sum(test_y==0),np.sum(test_y<0))
    #print('Testset by stageid= %s, trainsize=%d, testsize=%d, dist=%s'%
    #      (stageid, train_x.shape[0], test_x.shape[0], test_distribution))
    
    #record
    rec = [eventsname[eventid],train_x.shape[0],test_x.shape[0],test_distribution]
    
    acc = [0 for x in range(len(classifiers))]
    for idx, clf in enumerate(classifiers):
        acc[idx] = classifier_model(clf)

    rec.extend(acc)
    print('rec:%s'%rec)
    
    #new df
    df = pd.DataFrame([rec],columns=cols)
    retdf = pd.concat([retdf, df])        
    
retdf.to_csv('crossvalid_stagedata_splitbyevent.csv')
df_event = retdf
```

    cols:['runid', 'trainsize', 'testsize', 'testdistribution', 'currank', 'avgrank', 'dice', 'lr', 'lrl1', 'lsvc', 'lsvcl2', 'rf', 'lrbias', 'xgb']
    Testset = Phoenix
    [*] predict with currank model
    precision=[0.         0.14035088 0.        ], recall=[0. 1. 0.], f1=[0.         0.24615385 0.        ], accuracy=0.14
    [*] predict with avgrank model
    precision=[0.48275862 0.16129032 0.16      ], recall=[0.46666667 0.3125     0.10526316], f1=[0.47457627 0.21276596 0.12698413], accuracy=0.32
    [*] predict with dice model
    precision=[0.52173913 0.2        0.31578947], recall=[0.4        0.375      0.31578947], f1=[0.45283019 0.26086957 0.31578947], accuracy=0.37
    [*] predict with lr model
    precision=[0.625      0.27272727 0.4516129 ], recall=[0.75       0.1875     0.36842105], f1=[0.68181818 0.22222222 0.4057971 ], accuracy=0.54
    [*] predict with lrl1 model
    precision=[0.62666667 0.27272727 0.46428571], recall=[0.78333333 0.1875     0.34210526], f1=[0.6962963  0.22222222 0.39393939], accuracy=0.55
    [*] predict with lsvc model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.62121212 0.25       0.41666667], recall=[0.68333333 0.1875     0.39473684], f1=[0.65079365 0.21428571 0.40540541], accuracy=0.52
    [*] predict with lsvcl2 model
    precision=[0.625      0.375      0.66666667], recall=[0.91666667 0.1875     0.31578947], f1=[0.74324324 0.25       0.42857143], accuracy=0.61
    [*] predict with rf model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.63076923 0.26315789 0.6       ], recall=[0.68333333 0.3125     0.47368421], f1=[0.656      0.28571429 0.52941176], accuracy=0.56
    [*] predict with lrbias model
    [LibLinear]precision=[0.63768116 0.25       0.45454545], recall=[0.73333333 0.1875     0.39473684], f1=[0.68217054 0.21428571 0.42253521], accuracy=0.54
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.58333333 0.21875    0.54545455], recall=[0.58333333 0.4375     0.31578947], f1=[0.58333333 0.29166667 0.4       ], accuracy=0.47
    rec:['Phoenix', 691, 114, '+:38,0:16,-:60', 0.14035087719298245, 0.32456140350877194, 0.3684210526315789, 0.543859649122807, 0.5526315789473685, 0.5175438596491229, 0.6140350877192983, 0.5614035087719298, 0.543859649122807, 0.47368421052631576]
    Testset = Indy500
    [*] predict with currank model
    precision=[0.         0.20888889 0.        ], recall=[0. 1. 0.], f1=[0.         0.34558824 0.        ], accuracy=0.21
    [*] predict with avgrank model
    precision=[0.4017094  0.24528302 0.23636364], recall=[0.48958333 0.27659574 0.15853659], f1=[0.44131455 0.26       0.18978102], accuracy=0.32
    [*] predict with dice model
    precision=[0.47619048 0.26315789 0.38095238], recall=[0.52083333 0.31914894 0.29268293], f1=[0.49751244 0.28846154 0.33103448], accuracy=0.40
    [*] predict with lr model
    precision=[0.56834532 0.33333333 0.58441558], recall=[0.82291667 0.06382979 0.54878049], f1=[0.67234043 0.10714286 0.56603774], accuracy=0.56
    [*] predict with lrl1 model
    precision=[0.5620438  0.27272727 0.58441558], recall=[0.80208333 0.06382979 0.54878049], f1=[0.66094421 0.10344828 0.56603774], accuracy=0.56
    [*] predict with lsvc model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.55555556 0.3        0.57746479], recall=[0.83333333 0.06382979 0.5       ], f1=[0.66666667 0.10526316 0.53594771], accuracy=0.55
    [*] predict with lsvcl2 model
    precision=[0.59813084 0.32142857 0.56451613], recall=[0.66666667 0.38297872 0.42682927], f1=[0.63054187 0.34951456 0.48611111], accuracy=0.52
    [*] predict with rf model
    precision=[0.54054054 0.7        0.59701493], recall=[0.83333333 0.14893617 0.48780488], f1=[0.6557377  0.24561404 0.53691275], accuracy=0.56
    [*] predict with lrbias model
    [LibLinear]precision=[0.5620438  0.27272727 0.58441558], recall=[0.80208333 0.06382979 0.54878049], f1=[0.66094421 0.10344828 0.56603774], accuracy=0.56
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.57024793 0.31707317 0.55555556], recall=[0.71875    0.27659574 0.42682927], f1=[0.6359447  0.29545455 0.48275862], accuracy=0.52
    rec:['Indy500', 580, 225, '+:82,0:47,-:96', 0.2088888888888889, 0.3244444444444444, 0.39555555555555555, 0.5644444444444444, 0.5555555555555556, 0.5511111111111111, 0.52, 0.5644444444444444, 0.5555555555555556, 0.52]
    Testset = Texas
    [*] predict with currank model
    precision=[0.         0.26771654 0.        ], recall=[0. 1. 0.], f1=[0.         0.42236025 0.        ], accuracy=0.27
    [*] predict with avgrank model
    precision=[0.37037037 0.3255814  0.2       ], recall=[0.37037037 0.41176471 0.15384615], f1=[0.37037037 0.36363636 0.17391304], accuracy=0.31
    [*] predict with dice model
    precision=[0.30357143 0.23333333 0.31707317], recall=[0.31481481 0.20588235 0.33333333], f1=[0.30909091 0.21875    0.325     ], accuracy=0.29
    [*] predict with lr model
    precision=[0.6440678  0.57894737 0.55102041], recall=[0.7037037  0.32352941 0.69230769], f1=[0.67256637 0.41509434 0.61363636], accuracy=0.60
    [*] predict with lrl1 model
    precision=[0.63076923 0.625      0.56521739], recall=[0.75925926 0.29411765 0.66666667], f1=[0.68907563 0.4        0.61176471], accuracy=0.61
    [*] predict with lsvc model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.63636364 0.61111111 0.60465116], recall=[0.77777778 0.32352941 0.66666667], f1=[0.7        0.42307692 0.63414634], accuracy=0.62
    [*] predict with lsvcl2 model
    precision=[0.66666667 0.55555556 0.34042553], recall=[0.18518519 0.29411765 0.82051282], f1=[0.28985507 0.38461538 0.48120301], accuracy=0.41
    [*] predict with rf model
    precision=[0.69565217 0.31818182 0.37288136], recall=[0.59259259 0.20588235 0.56410256], f1=[0.64       0.25       0.44897959], accuracy=0.48
    [*] predict with lrbias model
    [LibLinear]precision=[0.63492063 0.57894737 0.57777778], recall=[0.74074074 0.32352941 0.66666667], f1=[0.68376068 0.41509434 0.61904762], accuracy=0.61
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.55263158 0.35294118 0.38888889], recall=[0.38888889 0.17647059 0.71794872], f1=[0.45652174 0.23529412 0.5045045 ], accuracy=0.43
    rec:['Texas', 678, 127, '+:39,0:34,-:54', 0.2677165354330709, 0.31496062992125984, 0.29133858267716534, 0.5984251968503937, 0.6062992125984252, 0.6220472440944882, 0.4094488188976378, 0.48031496062992124, 0.6062992125984252, 0.4330708661417323]
    Testset = Iowa
    [*] predict with currank model
    precision=[0.         0.25688073 0.        ], recall=[0. 1. 0.], f1=[0.         0.40875912 0.        ], accuracy=0.26
    [*] predict with avgrank model
    precision=[0.30232558 0.20689655 0.35135135], recall=[0.33333333 0.21428571 0.30952381], f1=[0.31707317 0.21052632 0.32911392], accuracy=0.29
    [*] predict with dice model
    precision=[0.25531915 0.18518519 0.31428571], recall=[0.30769231 0.17857143 0.26190476], f1=[0.27906977 0.18181818 0.28571429], accuracy=0.26
    [*] predict with lr model
    precision=[0.39393939 0.15625    0.72727273], recall=[0.66666667 0.17857143 0.19047619], f1=[0.4952381  0.16666667 0.30188679], accuracy=0.36
    [*] predict with lrl1 model
    precision=[0.39130435 0.1        0.8       ], recall=[0.69230769 0.10714286 0.19047619], f1=[0.5        0.10344828 0.30769231], accuracy=0.35
    [*] predict with lsvc model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.36842105 0.08695652 0.8       ], recall=[0.71794872 0.07142857 0.19047619], f1=[0.48695652 0.07843137 0.30769231], accuracy=0.35
    [*] predict with lsvcl2 model
    precision=[0.41176471 0.09090909 0.40740741], recall=[0.17948718 0.03571429 0.78571429], f1=[0.25       0.05128205 0.53658537], accuracy=0.38
    [*] predict with rf model
    precision=[0.35616438 0.10526316 0.52941176], recall=[0.66666667 0.07142857 0.21428571], f1=[0.46428571 0.08510638 0.30508475], accuracy=0.34
    [*] predict with lrbias model
    [LibLinear]precision=[0.3880597  0.12903226 0.72727273], recall=[0.66666667 0.14285714 0.19047619], f1=[0.49056604 0.13559322 0.30188679], accuracy=0.35
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.33333333 0.13513514 0.5       ], recall=[0.46153846 0.17857143 0.21428571], f1=[0.38709677 0.15384615 0.3       ], accuracy=0.29
    rec:['Iowa', 696, 109, '+:42,0:28,-:39', 0.25688073394495414, 0.29357798165137616, 0.25688073394495414, 0.3577981651376147, 0.3486238532110092, 0.3486238532110092, 0.3761467889908257, 0.3394495412844037, 0.3486238532110092, 0.29357798165137616]
    Testset = Pocono
    [*] predict with currank model
    precision=[0.         0.48412698 0.        ], recall=[0. 1. 0.], f1=[0.         0.65240642 0.        ], accuracy=0.48
    [*] predict with avgrank model
    precision=[0.265625   0.44117647 0.10714286], recall=[0.47222222 0.24590164 0.10344828], f1=[0.34       0.31578947 0.10526316], accuracy=0.28
    [*] predict with dice model
    precision=[0.23214286 0.34615385 0.18181818], recall=[0.36111111 0.14754098 0.27586207], f1=[0.2826087  0.20689655 0.21917808], accuracy=0.24
    [*] predict with lr model
    precision=[0.40384615 0.66666667 0.26470588], recall=[0.58333333 0.06557377 0.62068966], f1=[0.47727273 0.11940299 0.37113402], accuracy=0.34
    [*] predict with lrl1 model
    precision=[0.4        0.6        0.25352113], recall=[0.55555556 0.04918033 0.62068966], f1=[0.46511628 0.09090909 0.36      ], accuracy=0.33
    [*] predict with lsvc model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.41176471 0.5        0.26027397], recall=[0.58333333 0.01639344 0.65517241], f1=[0.48275862 0.03174603 0.37254902], accuracy=0.33
    [*] predict with lsvcl2 model
    precision=[0.36363636 0.         0.19444444], recall=[0.88888889 0.         0.24137931], f1=[0.51612903 0.         0.21538462], accuracy=0.31
    [*] predict with rf model
    precision=[0.41304348 0.66666667 0.24324324], recall=[0.52777778 0.06557377 0.62068966], f1=[0.46341463 0.11940299 0.34951456], accuracy=0.33
    [*] predict with lrbias model
    [LibLinear]precision=[0.41176471 0.6        0.25714286], recall=[0.58333333 0.04918033 0.62068966], f1=[0.48275862 0.09090909 0.36363636], accuracy=0.33
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.41304348 0.63157895 0.27868852], recall=[0.52777778 0.19672131 0.5862069 ], f1=[0.46341463 0.3        0.37777778], accuracy=0.38
    rec:['Pocono', 679, 126, '+:29,0:61,-:36', 0.48412698412698413, 0.2777777777777778, 0.23809523809523808, 0.3412698412698413, 0.3253968253968254, 0.3253968253968254, 0.30952380952380953, 0.3253968253968254, 0.3333333333333333, 0.38095238095238093]
    Testset = Gateway
    [*] predict with currank model
    precision=[0.         0.26923077 0.        ], recall=[0. 1. 0.], f1=[0.         0.42424242 0.        ], accuracy=0.27
    [*] predict with avgrank model
    precision=[0.36956522 0.29411765 0.29166667], recall=[0.4047619  0.35714286 0.20588235], f1=[0.38636364 0.32258065 0.24137931], accuracy=0.33
    [*] predict with dice model
    precision=[0.34782609 0.13043478 0.2       ], recall=[0.38095238 0.10714286 0.20588235], f1=[0.36363636 0.11764706 0.20289855], accuracy=0.25
    [*] predict with lr model
    precision=[0.8125     0.34722222 0.5625    ], recall=[0.30952381 0.89285714 0.26470588], f1=[0.44827586 0.5        0.36      ], accuracy=0.45
    [*] predict with lrl1 model
    precision=[0.90909091 0.33333333 0.5       ], recall=[0.23809524 0.96428571 0.17647059], f1=[0.37735849 0.49541284 0.26086957], accuracy=0.41
    [*] predict with lsvc model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.90909091 0.33333333 0.5       ], recall=[0.23809524 0.96428571 0.17647059], f1=[0.37735849 0.49541284 0.26086957], accuracy=0.41
    [*] predict with lsvcl2 model
    precision=[0.5        0.26666667 0.30120482], recall=[0.07142857 0.14285714 0.73529412], f1=[0.125      0.18604651 0.42735043], accuracy=0.31
    [*] predict with rf model
    precision=[0.54054054 0.30555556 0.35483871], recall=[0.47619048 0.39285714 0.32352941], f1=[0.50632911 0.34375    0.33846154], accuracy=0.40
    [*] predict with lrbias model
    [LibLinear]precision=[0.83333333 0.325      0.5       ], recall=[0.23809524 0.92857143 0.17647059], f1=[0.37037037 0.48148148 0.26086957], accuracy=0.40
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.55263158 0.29268293 0.36      ], recall=[0.5        0.42857143 0.26470588], f1=[0.525      0.34782609 0.30508475], accuracy=0.40
    rec:['Gateway', 701, 104, '+:34,0:28,-:42', 0.2692307692307692, 0.3269230769230769, 0.25, 0.4519230769230769, 0.41346153846153844, 0.41346153846153844, 0.3076923076923077, 0.40384615384615385, 0.40384615384615385, 0.40384615384615385]



```python
df_event
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
      <th>runid</th>
      <th>trainsize</th>
      <th>testsize</th>
      <th>testdistribution</th>
      <th>currank</th>
      <th>avgrank</th>
      <th>dice</th>
      <th>lr</th>
      <th>lrl1</th>
      <th>lsvc</th>
      <th>lsvcl2</th>
      <th>rf</th>
      <th>lrbias</th>
      <th>xgb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Phoenix</td>
      <td>691</td>
      <td>114</td>
      <td>+:38,0:16,-:60</td>
      <td>0.140351</td>
      <td>0.324561</td>
      <td>0.368421</td>
      <td>0.543860</td>
      <td>0.552632</td>
      <td>0.517544</td>
      <td>0.614035</td>
      <td>0.561404</td>
      <td>0.543860</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Indy500</td>
      <td>580</td>
      <td>225</td>
      <td>+:82,0:47,-:96</td>
      <td>0.208889</td>
      <td>0.324444</td>
      <td>0.395556</td>
      <td>0.564444</td>
      <td>0.555556</td>
      <td>0.551111</td>
      <td>0.520000</td>
      <td>0.564444</td>
      <td>0.555556</td>
      <td>0.520000</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Texas</td>
      <td>678</td>
      <td>127</td>
      <td>+:39,0:34,-:54</td>
      <td>0.267717</td>
      <td>0.314961</td>
      <td>0.291339</td>
      <td>0.598425</td>
      <td>0.606299</td>
      <td>0.622047</td>
      <td>0.409449</td>
      <td>0.480315</td>
      <td>0.606299</td>
      <td>0.433071</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Iowa</td>
      <td>696</td>
      <td>109</td>
      <td>+:42,0:28,-:39</td>
      <td>0.256881</td>
      <td>0.293578</td>
      <td>0.256881</td>
      <td>0.357798</td>
      <td>0.348624</td>
      <td>0.348624</td>
      <td>0.376147</td>
      <td>0.339450</td>
      <td>0.348624</td>
      <td>0.293578</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Pocono</td>
      <td>679</td>
      <td>126</td>
      <td>+:29,0:61,-:36</td>
      <td>0.484127</td>
      <td>0.277778</td>
      <td>0.238095</td>
      <td>0.341270</td>
      <td>0.325397</td>
      <td>0.325397</td>
      <td>0.309524</td>
      <td>0.325397</td>
      <td>0.333333</td>
      <td>0.380952</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Gateway</td>
      <td>701</td>
      <td>104</td>
      <td>+:34,0:28,-:42</td>
      <td>0.269231</td>
      <td>0.326923</td>
      <td>0.250000</td>
      <td>0.451923</td>
      <td>0.413462</td>
      <td>0.413462</td>
      <td>0.307692</td>
      <td>0.403846</td>
      <td>0.403846</td>
      <td>0.403846</td>
    </tr>
  </tbody>
</table>
</div>




```python
retdf = pd.DataFrame([],columns=cols)

for stageid in range(8):
    train, test, train_x, train_y, test_x, test_y =split_by_stageid(stagedata, stageid)
    test_distribution = '+:%d,0:%d,-:%d'%(np.sum(test_y>0),np.sum(test_y==0),np.sum(test_y<0))
    #print('Testset by stageid= %s, trainsize=%d, testsize=%d, dist=%s'%
    #      (stageid, train_x.shape[0], test_x.shape[0], test_distribution))
    
    #record
    rec = ['stage%d'%stageid,train_x.shape[0],test_x.shape[0],test_distribution]
    
    acc = [0 for x in range(len(classifiers))]
    for idx, clf in enumerate(classifiers):
        acc[idx] = classifier_model(clf)

    rec.extend(acc)
    print('rec:%s'%rec)
    
    #new df
    df = pd.DataFrame([rec],columns=cols)
    retdf = pd.concat([retdf, df])  
    
retdf.to_csv('crossvalid_stagedata_splitbystage.csv')
df_stage = retdf
```

    [*] predict with currank model
    precision=[0.         0.25613497 0.        ], recall=[0. 1. 0.], f1=[0.         0.40781441 0.        ], accuracy=0.26
    [*] predict with avgrank model
    precision=[0.37172775 0.22535211 0.23115578], recall=[0.53787879 0.09580838 0.2081448 ], f1=[0.43962848 0.13445378 0.21904762], accuracy=0.31
    [*] predict with dice model
    precision=[0.41197183 0.25806452 0.3956044 ], recall=[0.44318182 0.28742515 0.32579186], f1=[0.4270073  0.27195467 0.3573201 ], accuracy=0.36
    [*] predict with lr model
    precision=[0.49633252 0.36842105 0.56603774], recall=[0.76893939 0.41916168 0.13574661], f1=[0.60326895 0.39215686 0.2189781 ], accuracy=0.46
    [*] predict with lrl1 model
    precision=[0.49608355 0.36097561 0.46875   ], recall=[0.71969697 0.44311377 0.13574661], f1=[0.58732612 0.39784946 0.21052632], accuracy=0.45
    [*] predict with lsvc model
    precision=[0.47074468 0.3438914  0.27272727], recall=[0.67045455 0.45508982 0.0678733 ], f1=[0.553125   0.39175258 0.10869565], accuracy=0.41
    [*] predict with lsvcl2 model
    precision=[0.46966292 0.35121951 1.        ], recall=[0.79166667 0.43113772 0.00904977], f1=[0.58956276 0.38709677 0.01793722], accuracy=0.43
    [*] predict with rf model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.44444444 0.32850242 0.33108108], recall=[0.5        0.40718563 0.22171946], f1=[0.47058824 0.36363636 0.26558266], accuracy=0.38
    [*] predict with lrbias model
    [LibLinear]precision=[0.49234694 0.35576923 0.48076923], recall=[0.73106061 0.44311377 0.11312217], f1=[0.58841463 0.39466667 0.18315018], accuracy=0.45
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.4497992  0.29338843 0.34782609], recall=[0.42424242 0.4251497  0.25339367], f1=[0.43664717 0.34718826 0.29319372], accuracy=0.37
    rec:['stage0', 153, 652, '+:221,0:167,-:264', 0.2561349693251534, 0.3128834355828221, 0.36349693251533743, 0.4647239263803681, 0.450920245398773, 0.4110429447852761, 0.4340490797546012, 0.38190184049079756, 0.44785276073619634, 0.3665644171779141]
    [*] predict with currank model
    precision=[0.         0.26305609 0.        ], recall=[0. 1. 0.], f1=[0.         0.41653905 0.        ], accuracy=0.26
    [*] predict with avgrank model
    precision=[0.34674923 0.24242424 0.24223602], recall=[0.57435897 0.05882353 0.20967742], f1=[0.43243243 0.09467456 0.22478386], accuracy=0.31
    [*] predict with dice model
    precision=[0.39312977 0.22881356 0.37956204], recall=[0.52820513 0.19852941 0.27956989], f1=[0.45076586 0.21259843 0.32198142], accuracy=0.35
    [*] predict with lr model
    precision=[0.42746114 0.44642857 0.58666667], recall=[0.84615385 0.18382353 0.23655914], f1=[0.56798623 0.26041667 0.33716475], accuracy=0.45
    [*] predict with lrl1 model
    precision=[0.44722222 0.45614035 0.62      ], recall=[0.82564103 0.19117647 0.33333333], f1=[0.58018018 0.26943005 0.43356643], accuracy=0.48
    [*] predict with lsvc model
    precision=[0.46449704 0.41489362 0.58823529], recall=[0.80512821 0.28676471 0.2688172 ], f1=[0.5891182  0.33913043 0.36900369], accuracy=0.48
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.51648352 0.29927007 0.52631579], recall=[0.24102564 0.60294118 0.43010753], f1=[0.32867133 0.4        0.47337278], accuracy=0.40
    [*] predict with rf model
    precision=[0.43801653 0.33333333 0.5106383 ], recall=[0.81538462 0.14705882 0.25806452], f1=[0.56989247 0.20408163 0.34285714], accuracy=0.44
    [*] predict with lrbias model
    [LibLinear]precision=[0.4296875  0.44827586 0.58666667], recall=[0.84615385 0.19117647 0.23655914], f1=[0.56994819 0.26804124 0.33716475], accuracy=0.45
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.4787234  0.34482759 0.53781513], recall=[0.69230769 0.29411765 0.34408602], f1=[0.56603774 0.31746032 0.41967213], accuracy=0.46
    rec:['stage1', 288, 517, '+:186,0:136,-:195', 0.26305609284332687, 0.30754352030947774, 0.3520309477756286, 0.4526112185686654, 0.4816247582205029, 0.4758220502901354, 0.40425531914893614, 0.43907156673114117, 0.45454545454545453, 0.4622823984526112]
    [*] predict with currank model
    precision=[0.         0.29166667 0.        ], recall=[0. 1. 0.], f1=[0.        0.4516129 0.       ], accuracy=0.29
    [*] predict with avgrank model
    precision=[0.28630705 0.28571429 0.21311475], recall=[0.52272727 0.05357143 0.18571429], f1=[0.36997319 0.09022556 0.19847328], accuracy=0.26
    [*] predict with dice model
    precision=[0.3627451  0.17567568 0.33018868], recall=[0.56060606 0.11607143 0.25      ], f1=[0.44047619 0.13978495 0.28455285], accuracy=0.32
    [*] predict with lr model
    precision=[0.49142857 0.         0.48543689], recall=[0.65151515 0.         0.71428571], f1=[0.56026059 0.         0.57803468], accuracy=0.48
    [*] predict with lrl1 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.44298246 0.16666667 0.53333333], recall=[0.76515152 0.00892857 0.57142857], f1=[0.56111111 0.01694915 0.55172414], accuracy=0.47
    [*] predict with lsvc model
    precision=[0.49238579 0.         0.51086957], recall=[0.73484848 0.         0.67142857], f1=[0.58966565 0.         0.58024691], accuracy=0.50
    [*] predict with lsvcl2 model
    precision=[0.51136364 0.28057554 0.54140127], recall=[0.34090909 0.34821429 0.60714286], f1=[0.40909091 0.31075697 0.57239057], accuracy=0.44
    [*] predict with rf model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.432      0.40625    0.56862745], recall=[0.81818182 0.11607143 0.41428571], f1=[0.56544503 0.18055556 0.47933884], accuracy=0.47
    [*] predict with lrbias model
    [LibLinear]precision=[0.47953216 0.         0.48076923], recall=[0.62121212 0.         0.71428571], f1=[0.54125413 0.         0.57471264], accuracy=0.47
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.425      0.36923077 0.46218487], recall=[0.64393939 0.21428571 0.39285714], f1=[0.51204819 0.27118644 0.42471042], accuracy=0.43
    rec:['stage2', 421, 384, '+:140,0:112,-:132', 0.2916666666666667, 0.2630208333333333, 0.3177083333333333, 0.484375, 0.4739583333333333, 0.4973958333333333, 0.4401041666666667, 0.4661458333333333, 0.4739583333333333, 0.4270833333333333]
    [*] predict with currank model
    precision=[0.         0.34496124 0.        ], recall=[0. 1. 0.], f1=[0.        0.5129683 0.       ], accuracy=0.34
    [*] predict with avgrank model
    precision=[0.22929936 0.29411765 0.21428571], recall=[0.46153846 0.05617978 0.1978022 ], f1=[0.30638298 0.09433962 0.20571429], accuracy=0.23
    [*] predict with dice model
    precision=[0.28985507 0.4375     0.34722222], recall=[0.51282051 0.23595506 0.27472527], f1=[0.37037037 0.30656934 0.30674847], accuracy=0.33
    [*] predict with lr model
    precision=[0.46875    0.18181818 0.52941176], recall=[0.76923077 0.02247191 0.69230769], f1=[0.58252427 0.04       0.6       ], accuracy=0.48
    [*] predict with lrl1 model
    precision=[0.4295302  0.09090909 0.59183673], recall=[0.82051282 0.01123596 0.63736264], f1=[0.56387665 0.02       0.61375661], accuracy=0.48
    [*] predict with lsvc model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.4609375  0.125      0.52459016], recall=[0.75641026 0.01123596 0.7032967 ], f1=[0.57281553 0.02061856 0.60093897], accuracy=0.48
    [*] predict with lsvcl2 model
    precision=[0.33802817 0.4        0.6       ], recall=[0.92307692 0.06741573 0.1978022 ], f1=[0.49484536 0.11538462 0.29752066], accuracy=0.37
    [*] predict with rf model
    precision=[0.40625    0.5        0.53846154], recall=[0.83333333 0.11235955 0.46153846], f1=[0.54621849 0.18348624 0.49704142], accuracy=0.45
    [*] predict with lrbias model
    [LibLinear]precision=[0.464      0.16666667 0.51239669], recall=[0.74358974 0.02247191 0.68131868], f1=[0.57142857 0.03960396 0.58490566], accuracy=0.47
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.3880597  0.51724138 0.48421053], recall=[0.66666667 0.16853933 0.50549451], f1=[0.49056604 0.25423729 0.49462366], accuracy=0.44
    rec:['stage3', 547, 258, '+:91,0:89,-:78', 0.3449612403100775, 0.22868217054263565, 0.3333333333333333, 0.4844961240310077, 0.47674418604651164, 0.4806201550387597, 0.37209302325581395, 0.45348837209302323, 0.4728682170542636, 0.437984496124031]
    [*] predict with currank model
    precision=[0.         0.35810811 0.        ], recall=[0. 1. 0.], f1=[0.         0.52736318 0.        ], accuracy=0.36
    [*] predict with avgrank model
    precision=[0.21348315 0.22222222 0.2       ], recall=[0.40425532 0.03773585 0.20833333], f1=[0.27941176 0.06451613 0.20408163], accuracy=0.21
    [*] predict with dice model
    precision=[0.33802817 0.4375     0.28888889], recall=[0.5106383  0.26415094 0.27083333], f1=[0.40677966 0.32941176 0.27956989], accuracy=0.34
    [*] predict with lr model
    precision=[0.61904762 0.51515152 0.53424658], recall=[0.55319149 0.32075472 0.8125    ], f1=[0.58426966 0.39534884 0.6446281 ], accuracy=0.55
    [*] predict with lrl1 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.62222222 0.53333333 0.53424658], recall=[0.59574468 0.30188679 0.8125    ], f1=[0.60869565 0.38554217 0.6446281 ], accuracy=0.56
    [*] predict with lsvc model
    precision=[0.63636364 0.53333333 0.52702703], recall=[0.59574468 0.30188679 0.8125    ], f1=[0.61538462 0.38554217 0.63934426], accuracy=0.56
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.46296296 0.47435897 0.75      ], recall=[0.53191489 0.69811321 0.25      ], f1=[0.4950495 0.5648855 0.375    ], accuracy=0.50
    [*] predict with rf model
    precision=[0.42465753 0.53333333 0.62222222], recall=[0.65957447 0.30188679 0.58333333], f1=[0.51666667 0.38554217 0.60215054], accuracy=0.51
    [*] predict with lrbias model
    [LibLinear]precision=[0.65116279 0.5625     0.53424658], recall=[0.59574468 0.33962264 0.8125    ], f1=[0.62222222 0.42352941 0.6446281 ], accuracy=0.57
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.43636364 0.39534884 0.54      ], recall=[0.5106383  0.32075472 0.5625    ], f1=[0.47058824 0.35416667 0.55102041], accuracy=0.46
    rec:['stage4', 657, 148, '+:48,0:53,-:47', 0.3581081081081081, 0.20945945945945946, 0.34459459459459457, 0.5540540540540541, 0.5608108108108109, 0.5608108108108109, 0.5, 0.5067567567567568, 0.5743243243243243, 0.4594594594594595]
    [*] predict with currank model
    precision=[0.     0.3625 0.    ], recall=[0. 1. 0.], f1=[0.         0.53211009 0.        ], accuracy=0.36
    [*] predict with avgrank model
    precision=[0.19565217 0.28571429 0.18518519], recall=[0.36       0.06896552 0.19230769], f1=[0.25352113 0.11111111 0.18867925], accuracy=0.20
    [*] predict with dice model
    precision=[0.32352941 0.26666667 0.35483871], recall=[0.44       0.13793103 0.42307692], f1=[0.37288136 0.18181818 0.38596491], accuracy=0.33
    [*] predict with lr model
    precision=[0.625      0.5        0.72222222], recall=[0.4        0.79310345 0.5       ], f1=[0.48780488 0.61333333 0.59090909], accuracy=0.57
    [*] predict with lrl1 model
    precision=[0.625      0.52083333 0.75      ], recall=[0.4        0.86206897 0.46153846], f1=[0.48780488 0.64935065 0.57142857], accuracy=0.59
    [*] predict with lsvc model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.61111111 0.52272727 0.66666667], recall=[0.44       0.79310345 0.46153846], f1=[0.51162791 0.63013699 0.54545455], accuracy=0.57
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.25      0.3943662 1.       ], recall=[0.08       0.96551724 0.03846154], f1=[0.12121212 0.56       0.07407407], accuracy=0.39
    [*] predict with rf model
    precision=[0.55263158 0.58823529 0.68      ], recall=[0.84       0.34482759 0.65384615], f1=[0.66666667 0.43478261 0.66666667], accuracy=0.60
    [*] predict with lrbias model
    [LibLinear]precision=[0.625      0.5106383  0.70588235], recall=[0.4        0.82758621 0.46153846], f1=[0.48780488 0.63157895 0.55813953], accuracy=0.57
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.44444444 0.51851852 0.5       ], recall=[0.48       0.48275862 0.5       ], f1=[0.46153846 0.5        0.5       ], accuracy=0.49
    rec:['stage5', 725, 80, '+:26,0:29,-:25', 0.3625, 0.2, 0.325, 0.575, 0.5875, 0.575, 0.3875, 0.6, 0.575, 0.4875]
    [*] predict with currank model
    precision=[0.         0.34210526 0.        ], recall=[0. 1. 0.], f1=[0.         0.50980392 0.        ], accuracy=0.34
    [*] predict with avgrank model
    precision=[0.27272727 0.33333333 0.1       ], recall=[0.42857143 0.15384615 0.09090909], f1=[0.33333333 0.21052632 0.0952381 ], accuracy=0.24
    [*] predict with dice model
    precision=[0.2        0.3        0.30769231], recall=[0.21428571 0.23076923 0.36363636], f1=[0.20689655 0.26086957 0.33333333], accuracy=0.26
    [*] predict with lr model
    precision=[0.625      0.42307692 1.        ], recall=[0.35714286 0.84615385 0.36363636], f1=[0.45454545 0.56410256 0.53333333], accuracy=0.53
    [*] predict with lrl1 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.625      0.42307692 1.        ], recall=[0.35714286 0.84615385 0.36363636], f1=[0.45454545 0.56410256 0.53333333], accuracy=0.53
    [*] predict with lsvc model
    precision=[0.625      0.42307692 1.        ], recall=[0.35714286 0.84615385 0.36363636], f1=[0.45454545 0.56410256 0.53333333], accuracy=0.53
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.5        0.45454545 0.6       ], recall=[0.21428571 0.76923077 0.54545455], f1=[0.3        0.57142857 0.57142857], accuracy=0.50
    [*] predict with rf model
    precision=[0.55       0.42857143 0.63636364], recall=[0.78571429 0.23076923 0.63636364], f1=[0.64705882 0.3        0.63636364], accuracy=0.55
    [*] predict with lrbias model
    [LibLinear]precision=[0.71428571 0.42307692 0.8       ], recall=[0.35714286 0.84615385 0.36363636], f1=[0.47619048 0.56410256 0.5       ], accuracy=0.53
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.61111111 0.55555556 0.63636364], recall=[0.78571429 0.38461538 0.63636364], f1=[0.6875     0.45454545 0.63636364], accuracy=0.61
    rec:['stage6', 767, 38, '+:11,0:13,-:14', 0.34210526315789475, 0.23684210526315788, 0.2631578947368421, 0.5263157894736842, 0.5263157894736842, 0.5263157894736842, 0.5, 0.5526315789473685, 0.5263157894736842, 0.6052631578947368]
    [*] predict with currank model
    precision=[0.   0.25 0.  ], recall=[0. 1. 0.], f1=[0.  0.4 0. ], accuracy=0.25
    [*] predict with avgrank model
    precision=[0.3 0.  0. ], recall=[0.375 0.    0.   ], f1=[0.33333333 0.         0.        ], accuracy=0.19
    [*] predict with dice model
    precision=[0.42857143 0.2        0.25      ], recall=[0.375 0.25  0.25 ], f1=[0.4        0.22222222 0.25      ], accuracy=0.31
    [*] predict with lr model
    precision=[0.66666667 0.25       1.        ], recall=[0.25 0.75 0.25], f1=[0.36363636 0.375      0.4       ], accuracy=0.38
    [*] predict with lrl1 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[1.   0.25 0.5 ], recall=[0.25 0.75 0.25], f1=[0.4        0.375      0.33333333], accuracy=0.38
    [*] predict with lsvc model
    precision=[0.66666667 0.25       1.        ], recall=[0.25 0.75 0.25], f1=[0.36363636 0.375      0.4       ], accuracy=0.38
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)


    precision=[0.5 0.  0. ], recall=[1. 0. 0.], f1=[0.66666667 0.         0.        ], accuracy=0.50
    [*] predict with rf model
    precision=[0.5  0.25 0.5 ], recall=[0.375 0.25  0.75 ], f1=[0.42857143 0.25       0.6       ], accuracy=0.44
    [*] predict with lrbias model
    [LibLinear]precision=[1.   0.25 0.5 ], recall=[0.25 0.75 0.25], f1=[0.4        0.375      0.33333333], accuracy=0.38
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.5        0.2        0.42857143], recall=[0.25 0.25 0.75], f1=[0.33333333 0.22222222 0.54545455], accuracy=0.38
    rec:['stage7', 789, 16, '+:4,0:4,-:8', 0.25, 0.1875, 0.3125, 0.375, 0.375, 0.375, 0.5, 0.4375, 0.375, 0.375]



```python
df_stage
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
      <th>runid</th>
      <th>trainsize</th>
      <th>testsize</th>
      <th>testdistribution</th>
      <th>currank</th>
      <th>avgrank</th>
      <th>dice</th>
      <th>lr</th>
      <th>lrl1</th>
      <th>lsvc</th>
      <th>lsvcl2</th>
      <th>rf</th>
      <th>lrbias</th>
      <th>xgb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>stage0</td>
      <td>153</td>
      <td>652</td>
      <td>+:221,0:167,-:264</td>
      <td>0.256135</td>
      <td>0.312883</td>
      <td>0.363497</td>
      <td>0.464724</td>
      <td>0.450920</td>
      <td>0.411043</td>
      <td>0.434049</td>
      <td>0.381902</td>
      <td>0.447853</td>
      <td>0.366564</td>
    </tr>
    <tr>
      <th>0</th>
      <td>stage1</td>
      <td>288</td>
      <td>517</td>
      <td>+:186,0:136,-:195</td>
      <td>0.263056</td>
      <td>0.307544</td>
      <td>0.352031</td>
      <td>0.452611</td>
      <td>0.481625</td>
      <td>0.475822</td>
      <td>0.404255</td>
      <td>0.439072</td>
      <td>0.454545</td>
      <td>0.462282</td>
    </tr>
    <tr>
      <th>0</th>
      <td>stage2</td>
      <td>421</td>
      <td>384</td>
      <td>+:140,0:112,-:132</td>
      <td>0.291667</td>
      <td>0.263021</td>
      <td>0.317708</td>
      <td>0.484375</td>
      <td>0.473958</td>
      <td>0.497396</td>
      <td>0.440104</td>
      <td>0.466146</td>
      <td>0.473958</td>
      <td>0.427083</td>
    </tr>
    <tr>
      <th>0</th>
      <td>stage3</td>
      <td>547</td>
      <td>258</td>
      <td>+:91,0:89,-:78</td>
      <td>0.344961</td>
      <td>0.228682</td>
      <td>0.333333</td>
      <td>0.484496</td>
      <td>0.476744</td>
      <td>0.480620</td>
      <td>0.372093</td>
      <td>0.453488</td>
      <td>0.472868</td>
      <td>0.437984</td>
    </tr>
    <tr>
      <th>0</th>
      <td>stage4</td>
      <td>657</td>
      <td>148</td>
      <td>+:48,0:53,-:47</td>
      <td>0.358108</td>
      <td>0.209459</td>
      <td>0.344595</td>
      <td>0.554054</td>
      <td>0.560811</td>
      <td>0.560811</td>
      <td>0.500000</td>
      <td>0.506757</td>
      <td>0.574324</td>
      <td>0.459459</td>
    </tr>
    <tr>
      <th>0</th>
      <td>stage5</td>
      <td>725</td>
      <td>80</td>
      <td>+:26,0:29,-:25</td>
      <td>0.362500</td>
      <td>0.200000</td>
      <td>0.325000</td>
      <td>0.575000</td>
      <td>0.587500</td>
      <td>0.575000</td>
      <td>0.387500</td>
      <td>0.600000</td>
      <td>0.575000</td>
      <td>0.487500</td>
    </tr>
    <tr>
      <th>0</th>
      <td>stage6</td>
      <td>767</td>
      <td>38</td>
      <td>+:11,0:13,-:14</td>
      <td>0.342105</td>
      <td>0.236842</td>
      <td>0.263158</td>
      <td>0.526316</td>
      <td>0.526316</td>
      <td>0.526316</td>
      <td>0.500000</td>
      <td>0.552632</td>
      <td>0.526316</td>
      <td>0.605263</td>
    </tr>
    <tr>
      <th>0</th>
      <td>stage7</td>
      <td>789</td>
      <td>16</td>
      <td>+:4,0:4,-:8</td>
      <td>0.250000</td>
      <td>0.187500</td>
      <td>0.312500</td>
      <td>0.375000</td>
      <td>0.375000</td>
      <td>0.375000</td>
      <td>0.500000</td>
      <td>0.437500</td>
      <td>0.375000</td>
      <td>0.375000</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
