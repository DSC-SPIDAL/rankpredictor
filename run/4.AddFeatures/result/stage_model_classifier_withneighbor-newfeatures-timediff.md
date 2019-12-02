
### stage_model_classifier

prediction models of sign classifiers on stage dataset

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
        clf = xgb.XGBClassifier(booster = 'gbtree', nthread = -1, subsample = 1, 
                                n_estimators = 600, colsample_bytree = 1, max_depth = 6, min_child_weight = 1)
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
suffix='-withneighbor-newfeatures-timediff'
stagedata = pd.read_csv('stage-2018%s.csv'%suffix)
stagedata.fillna(0, inplace=True)
stagedata.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 805 entries, 0 to 804
    Data columns (total 35 columns):
    Unnamed: 0                   805 non-null int64
    target                       805 non-null int64
    eventid                      805 non-null int64
    car_number                   805 non-null int64
    stageid                      805 non-null int64
    firststage                   805 non-null int64
    pit_in_caution               805 non-null int64
    start_position               805 non-null int64
    start_rank                   805 non-null int64
    start_rank_ratio             805 non-null float64
    top_pack                     805 non-null int64
    bottom_pack                  805 non-null int64
    average_rank                 805 non-null float64
    average_rank_all             805 non-null float64
    change_in_rank               805 non-null int64
    change_in_rank_all           805 non-null float64
    rate_of_change               805 non-null int64
    rate_of_change_all           805 non-null float64
    laptime_green_mean_prev      805 non-null float64
    laptime_green_std_prev       805 non-null float64
    laptime_green_mean_all       805 non-null float64
    laptime_green_std_all        805 non-null float64
    laptime_mean_prev            805 non-null float64
    laptime_std_prev             805 non-null float64
    laptime_mean_all             805 non-null float64
    laptime_std_all              805 non-null float64
    laps_prev                    805 non-null int64
    laps_after_last_pitstop      805 non-null int64
    pittime_prev                 805 non-null float64
    prev_nb0_change_in_rank      805 non-null int64
    prev_nb1_change_in_rank      805 non-null int64
    prev_nb2_change_in_rank      805 non-null int64
    follow_nb0_change_in_rank    805 non-null int64
    follow_nb1_change_in_rank    805 non-null int64
    follow_nb2_change_in_rank    805 non-null int64
    dtypes: float64(14), int64(21)
    memory usage: 220.2 KB



```python
stagedata.head(5)
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
      <th>...</th>
      <th>laptime_std_all</th>
      <th>laps_prev</th>
      <th>laps_after_last_pitstop</th>
      <th>pittime_prev</th>
      <th>prev_nb0_change_in_rank</th>
      <th>prev_nb1_change_in_rank</th>
      <th>prev_nb2_change_in_rank</th>
      <th>follow_nb0_change_in_rank</th>
      <th>follow_nb1_change_in_rank</th>
      <th>follow_nb2_change_in_rank</th>
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
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0.00000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>...</td>
      <td>3.538589</td>
      <td>39</td>
      <td>39</td>
      <td>11.54325</td>
      <td>-2</td>
      <td>-1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>-1</td>
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
      <td>...</td>
      <td>7.902623</td>
      <td>76</td>
      <td>72</td>
      <td>59.63585</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-5</td>
      <td>-5</td>
      <td>1</td>
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
      <td>...</td>
      <td>6.817462</td>
      <td>57</td>
      <td>53</td>
      <td>40.43850</td>
      <td>-1</td>
      <td>-2</td>
      <td>-3</td>
      <td>-6</td>
      <td>-8</td>
      <td>-5</td>
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
      <td>...</td>
      <td>6.182861</td>
      <td>56</td>
      <td>52</td>
      <td>39.51240</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>-2</td>
      <td>-3</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
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
    
retdf.to_csv('crossvalid_stagedata_splitbyevent%s.csv'%suffix)
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
    precision=[0.65079365 0.15384615 0.47368421], recall=[0.68333333 0.125      0.47368421], f1=[0.66666667 0.13793103 0.47368421], accuracy=0.54
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


    precision=[0.64615385 0.14285714 0.48571429], recall=[0.7        0.125      0.44736842], f1=[0.672      0.13333333 0.46575342], accuracy=0.54
    [*] predict with lsvc model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.64516129 0.14285714 0.44736842], recall=[0.66666667 0.125      0.44736842], f1=[0.6557377  0.13333333 0.44736842], accuracy=0.52
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.55813953 0.2173913  0.6       ], recall=[0.8        0.3125     0.07894737], f1=[0.65753425 0.25641026 0.13953488], accuracy=0.49
    [*] predict with rf model
    precision=[0.64814815 0.27777778 0.52380952], recall=[0.58333333 0.3125     0.57894737], f1=[0.61403509 0.29411765 0.55      ], accuracy=0.54
    [*] predict with lrbias model
    [LibLinear]precision=[0.65079365 0.14285714 0.45945946], recall=[0.68333333 0.125      0.44736842], f1=[0.66666667 0.13333333 0.45333333], accuracy=0.53
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.58974359 0.14285714 0.475     ], recall=[0.38333333 0.3125     0.5       ], f1=[0.46464646 0.19607843 0.48717949], accuracy=0.41
    rec:['Phoenix', 691, 114, '+:38,0:16,-:60', 0.14035087719298245, 0.32456140350877194, 0.3684210526315789, 0.5350877192982456, 0.5350877192982456, 0.5175438596491229, 0.49122807017543857, 0.543859649122807, 0.5263157894736842, 0.41228070175438597]
    Testset = Indy500
    [*] predict with currank model
    precision=[0.         0.20888889 0.        ], recall=[0. 1. 0.], f1=[0.         0.34558824 0.        ], accuracy=0.21
    [*] predict with avgrank model
    precision=[0.4017094  0.24528302 0.23636364], recall=[0.48958333 0.27659574 0.15853659], f1=[0.44131455 0.26       0.18978102], accuracy=0.32
    [*] predict with dice model
    precision=[0.47619048 0.26315789 0.38095238], recall=[0.52083333 0.31914894 0.29268293], f1=[0.49751244 0.28846154 0.33103448], accuracy=0.40
    [*] predict with lr model
    precision=[0.54605263 0.16666667 0.62686567], recall=[0.86458333 0.0212766  0.51219512], f1=[0.66935484 0.03773585 0.56375839], accuracy=0.56
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


    precision=[0.53691275 0.125      0.61764706], recall=[0.83333333 0.0212766  0.51219512], f1=[0.65306122 0.03636364 0.56      ], accuracy=0.55
    [*] predict with lsvc model
    precision=[0.5308642  0.25       0.61016949], recall=[0.89583333 0.0212766  0.43902439], f1=[0.66666667 0.03921569 0.5106383 ], accuracy=0.55
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.51234568 0.23404255 0.5       ], recall=[0.86458333 0.23404255 0.09756098], f1=[0.64341085 0.23404255 0.16326531], accuracy=0.45
    [*] predict with rf model
    precision=[0.51428571 0.26666667 0.4       ], recall=[0.75       0.34042553 0.12195122], f1=[0.61016949 0.29906542 0.18691589], accuracy=0.44
    [*] predict with lrbias model
    [LibLinear]precision=[0.54605263 0.16666667 0.62686567], recall=[0.86458333 0.0212766  0.51219512], f1=[0.66935484 0.03773585 0.56375839], accuracy=0.56
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.56034483 0.34615385 0.54385965], recall=[0.67708333 0.38297872 0.37804878], f1=[0.61320755 0.36363636 0.44604317], accuracy=0.51
    rec:['Indy500', 580, 225, '+:82,0:47,-:96', 0.2088888888888889, 0.3244444444444444, 0.39555555555555555, 0.56, 0.5466666666666666, 0.5466666666666666, 0.4533333333333333, 0.43555555555555553, 0.56, 0.5066666666666667]
    Testset = Texas
    [*] predict with currank model
    precision=[0.         0.26771654 0.        ], recall=[0. 1. 0.], f1=[0.         0.42236025 0.        ], accuracy=0.27
    [*] predict with avgrank model
    precision=[0.37037037 0.3255814  0.2       ], recall=[0.37037037 0.41176471 0.15384615], f1=[0.37037037 0.36363636 0.17391304], accuracy=0.31
    [*] predict with dice model
    precision=[0.30357143 0.23333333 0.31707317], recall=[0.31481481 0.20588235 0.33333333], f1=[0.30909091 0.21875    0.325     ], accuracy=0.29
    [*] predict with lr model
    precision=[0.60465116 0.4        0.51851852], recall=[0.48148148 0.35294118 0.71794872], f1=[0.53608247 0.375      0.60215054], accuracy=0.52
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


    precision=[0.64       0.46666667 0.53191489], recall=[0.59259259 0.41176471 0.64102564], f1=[0.61538462 0.4375     0.58139535], accuracy=0.56
    [*] predict with lsvc model
    precision=[0.61363636 0.37142857 0.54166667], recall=[0.5        0.38235294 0.66666667], f1=[0.55102041 0.37681159 0.59770115], accuracy=0.52
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.52054795 0.36170213 0.57142857], recall=[0.7037037 0.5       0.1025641], f1=[0.5984252  0.41975309 0.17391304], accuracy=0.46
    [*] predict with rf model
    precision=[0.62       0.31578947 0.4137931 ], recall=[0.57407407 0.17647059 0.61538462], f1=[0.59615385 0.22641509 0.49484536], accuracy=0.48
    [*] predict with lrbias model
    [LibLinear]precision=[0.60869565 0.41176471 0.53191489], recall=[0.51851852 0.41176471 0.64102564], f1=[0.56       0.41176471 0.58139535], accuracy=0.53
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.57142857 0.5        0.41176471], recall=[0.51851852 0.14705882 0.71794872], f1=[0.54368932 0.22727273 0.52336449], accuracy=0.48
    rec:['Texas', 678, 127, '+:39,0:34,-:54', 0.2677165354330709, 0.31496062992125984, 0.29133858267716534, 0.5196850393700787, 0.5590551181102362, 0.5196850393700787, 0.4645669291338583, 0.48031496062992124, 0.5275590551181102, 0.48031496062992124]
    Testset = Iowa
    [*] predict with currank model
    precision=[0.         0.25688073 0.        ], recall=[0. 1. 0.], f1=[0.         0.40875912 0.        ], accuracy=0.26
    [*] predict with avgrank model
    precision=[0.30232558 0.20689655 0.35135135], recall=[0.33333333 0.21428571 0.30952381], f1=[0.31707317 0.21052632 0.32911392], accuracy=0.29
    [*] predict with dice model
    precision=[0.25531915 0.18518519 0.31428571], recall=[0.30769231 0.17857143 0.26190476], f1=[0.27906977 0.18181818 0.28571429], accuracy=0.26
    [*] predict with lr model
    precision=[0.4109589  0.16666667 0.66666667], recall=[0.76923077 0.10714286 0.28571429], f1=[0.53571429 0.13043478 0.4       ], accuracy=0.41
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


    precision=[0.40277778 0.16666667 0.63157895], recall=[0.74358974 0.10714286 0.28571429], f1=[0.52252252 0.13043478 0.39344262], accuracy=0.40
    [*] predict with lsvc model
    precision=[0.4109589  0.06666667 0.57142857], recall=[0.76923077 0.03571429 0.28571429], f1=[0.53571429 0.04651163 0.38095238], accuracy=0.39
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.39189189 0.10526316 0.6875    ], recall=[0.74358974 0.07142857 0.26190476], f1=[0.51327434 0.08510638 0.37931034], accuracy=0.39
    [*] predict with rf model
    precision=[0.37037037 0.         0.63157895], recall=[0.76923077 0.         0.28571429], f1=[0.5        0.         0.39344262], accuracy=0.39
    [*] predict with lrbias model
    [LibLinear]precision=[0.40277778 0.16666667 0.63157895], recall=[0.74358974 0.10714286 0.28571429], f1=[0.52252252 0.13043478 0.39344262], accuracy=0.40
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.35384615 0.25       0.5       ], recall=[0.58974359 0.17857143 0.28571429], f1=[0.44230769 0.20833333 0.36363636], accuracy=0.37
    rec:['Iowa', 696, 109, '+:42,0:28,-:39', 0.25688073394495414, 0.29357798165137616, 0.25688073394495414, 0.41284403669724773, 0.4036697247706422, 0.3944954128440367, 0.3853211009174312, 0.3853211009174312, 0.4036697247706422, 0.3669724770642202]
    Testset = Pocono
    [*] predict with currank model
    precision=[0.         0.48412698 0.        ], recall=[0. 1. 0.], f1=[0.         0.65240642 0.        ], accuracy=0.48
    [*] predict with avgrank model
    precision=[0.265625   0.44117647 0.10714286], recall=[0.47222222 0.24590164 0.10344828], f1=[0.34       0.31578947 0.10526316], accuracy=0.28
    [*] predict with dice model
    precision=[0.23214286 0.34615385 0.18181818], recall=[0.36111111 0.14754098 0.27586207], f1=[0.2826087  0.20689655 0.21917808], accuracy=0.24
    [*] predict with lr model
    precision=[0.31818182 0.64705882 0.30769231], recall=[0.58333333 0.36065574 0.27586207], f1=[0.41176471 0.46315789 0.29090909], accuracy=0.40
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


    precision=[0.33802817 0.60714286 0.25925926], recall=[0.66666667 0.27868852 0.24137931], f1=[0.44859813 0.38202247 0.25      ], accuracy=0.38
    [*] predict with lsvc model
    precision=[0.36486486 0.54545455 0.23333333], recall=[0.75       0.19672131 0.24137931], f1=[0.49090909 0.28915663 0.23728814], accuracy=0.37
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.23863636 0.33333333 0.17142857], recall=[0.58333333 0.01639344 0.20689655], f1=[0.33870968 0.03125    0.1875    ], accuracy=0.22
    [*] predict with rf model
    precision=[0.40540541 0.60869565 0.24242424], recall=[0.41666667 0.2295082  0.55172414], f1=[0.4109589  0.33333333 0.33684211], accuracy=0.36
    [*] predict with lrbias model
    [LibLinear]precision=[0.31818182 0.66666667 0.33333333], recall=[0.58333333 0.36065574 0.31034483], f1=[0.41176471 0.46808511 0.32142857], accuracy=0.41
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.41666667 0.51351351 0.28301887], recall=[0.41666667 0.31147541 0.51724138], f1=[0.41666667 0.3877551  0.36585366], accuracy=0.39
    rec:['Pocono', 679, 126, '+:29,0:61,-:36', 0.48412698412698413, 0.2777777777777778, 0.23809523809523808, 0.40476190476190477, 0.38095238095238093, 0.36507936507936506, 0.2222222222222222, 0.35714285714285715, 0.4126984126984127, 0.3888888888888889]
    Testset = Gateway
    [*] predict with currank model
    precision=[0.         0.26923077 0.        ], recall=[0. 1. 0.], f1=[0.         0.42424242 0.        ], accuracy=0.27
    [*] predict with avgrank model
    precision=[0.36956522 0.29411765 0.29166667], recall=[0.4047619  0.35714286 0.20588235], f1=[0.38636364 0.32258065 0.24137931], accuracy=0.33
    [*] predict with dice model
    precision=[0.34782609 0.13043478 0.2       ], recall=[0.38095238 0.10714286 0.20588235], f1=[0.36363636 0.11764706 0.20289855], accuracy=0.25
    [*] predict with lr model
    precision=[0.83333333 0.42622951 0.51612903], recall=[0.23809524 0.92857143 0.47058824], f1=[0.37037037 0.58426966 0.49230769], accuracy=0.50
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


    precision=[1.         0.3880597  0.48275862], recall=[0.19047619 0.92857143 0.41176471], f1=[0.32       0.54736842 0.44444444], accuracy=0.46
    [*] predict with lsvc model
    precision=[1.         0.37142857 0.5       ], recall=[0.14285714 0.92857143 0.41176471], f1=[0.25       0.53061224 0.4516129 ], accuracy=0.44
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.66666667 0.5        0.34939759], recall=[0.14285714 0.21428571 0.85294118], f1=[0.23529412 0.3        0.4957265 ], accuracy=0.39
    [*] predict with rf model
    precision=[0.55102041 0.5        0.45714286], recall=[0.64285714 0.35714286 0.47058824], f1=[0.59340659 0.41666667 0.46376812], accuracy=0.51
    [*] predict with lrbias model
    [LibLinear]precision=[1.         0.39393939 0.5       ], recall=[0.19047619 0.92857143 0.44117647], f1=[0.32       0.55319149 0.46875   ], accuracy=0.47
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.58333333 0.38095238 0.46153846], recall=[0.5        0.57142857 0.35294118], f1=[0.53846154 0.45714286 0.4       ], accuracy=0.47
    rec:['Gateway', 701, 104, '+:34,0:28,-:42', 0.2692307692307692, 0.3269230769230769, 0.25, 0.5, 0.46153846153846156, 0.4423076923076923, 0.3942307692307692, 0.5096153846153846, 0.47115384615384615, 0.47115384615384615]



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
      <td>0.535088</td>
      <td>0.535088</td>
      <td>0.517544</td>
      <td>0.491228</td>
      <td>0.543860</td>
      <td>0.526316</td>
      <td>0.412281</td>
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
      <td>0.560000</td>
      <td>0.546667</td>
      <td>0.546667</td>
      <td>0.453333</td>
      <td>0.435556</td>
      <td>0.560000</td>
      <td>0.506667</td>
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
      <td>0.519685</td>
      <td>0.559055</td>
      <td>0.519685</td>
      <td>0.464567</td>
      <td>0.480315</td>
      <td>0.527559</td>
      <td>0.480315</td>
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
      <td>0.412844</td>
      <td>0.403670</td>
      <td>0.394495</td>
      <td>0.385321</td>
      <td>0.385321</td>
      <td>0.403670</td>
      <td>0.366972</td>
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
      <td>0.404762</td>
      <td>0.380952</td>
      <td>0.365079</td>
      <td>0.222222</td>
      <td>0.357143</td>
      <td>0.412698</td>
      <td>0.388889</td>
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
      <td>0.500000</td>
      <td>0.461538</td>
      <td>0.442308</td>
      <td>0.394231</td>
      <td>0.509615</td>
      <td>0.471154</td>
      <td>0.471154</td>
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
    
retdf.to_csv('crossvalid_stagedata_splitbystage%s.csv'%suffix)
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
    precision=[0.51075269 0.36619718 0.50746269], recall=[0.71969697 0.46706587 0.15384615], f1=[0.59748428 0.41052632 0.23611111], accuracy=0.46
    [*] predict with lsvc model
    precision=[0.48199446 0.35744681 0.25      ], recall=[0.65909091 0.50299401 0.06334842], f1=[0.5568     0.41791045 0.10108303], accuracy=0.42
    [*] predict with lsvcl2 model
    precision=[0.5        0.35329341 0.40186916], recall=[0.31060606 0.35329341 0.58371041], f1=[0.38317757 0.35329341 0.47601476], accuracy=0.41
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


    precision=[0.44863014 0.30909091 0.32142857], recall=[0.49621212 0.40718563 0.20361991], f1=[0.47122302 0.35142119 0.24930748], accuracy=0.37
    [*] predict with lrbias model
    [LibLinear]precision=[0.49234694 0.35576923 0.48076923], recall=[0.73106061 0.44311377 0.11312217], f1=[0.58841463 0.39466667 0.18315018], accuracy=0.45
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.4375     0.30677291 0.29655172], recall=[0.42424242 0.46107784 0.19457014], f1=[0.43076923 0.36842105 0.23497268], accuracy=0.36
    rec:['stage0', 153, 652, '+:221,0:167,-:264', 0.2561349693251534, 0.3128834355828221, 0.36349693251533743, 0.4647239263803681, 0.46319018404907975, 0.4171779141104294, 0.41411042944785276, 0.37423312883435583, 0.44785276073619634, 0.3558282208588957]
    [*] predict with currank model
    precision=[0.         0.26305609 0.        ], recall=[0. 1. 0.], f1=[0.         0.41653905 0.        ], accuracy=0.26
    [*] predict with avgrank model
    precision=[0.34674923 0.24242424 0.24223602], recall=[0.57435897 0.05882353 0.20967742], f1=[0.43243243 0.09467456 0.22478386], accuracy=0.31
    [*] predict with dice model
    precision=[0.39312977 0.22881356 0.37956204], recall=[0.52820513 0.19852941 0.27956989], f1=[0.45076586 0.21259843 0.32198142], accuracy=0.35
    [*] predict with lr model
    precision=[0.46511628 0.28525641 0.53781513], recall=[0.20512821 0.65441176 0.34408602], f1=[0.28469751 0.39732143 0.41967213], accuracy=0.37
    [*] predict with lrl1 model
    precision=[0.48780488 0.30612245 0.53383459], recall=[0.1025641  0.77205882 0.38172043], f1=[0.16949153 0.43841336 0.44514107], accuracy=0.38
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


    precision=[0.5        0.29131653 0.53846154], recall=[0.14358974 0.76470588 0.30107527], f1=[0.22310757 0.42190669 0.3862069 ], accuracy=0.36
    [*] predict with lsvcl2 model
    precision=[0.5        0.27678571 0.50793651], recall=[0.01538462 0.91176471 0.17204301], f1=[0.02985075 0.42465753 0.25702811], accuracy=0.31
    [*] predict with rf model
    precision=[0.43007916 0.32258065 0.44859813], recall=[0.83589744 0.07352941 0.25806452], f1=[0.56794425 0.11976048 0.32764505], accuracy=0.43
    [*] predict with lrbias model
    [LibLinear]precision=[0.4691358  0.28481013 0.53333333], recall=[0.19487179 0.66176471 0.34408602], f1=[0.27536232 0.39823009 0.41830065], accuracy=0.37
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.4380531  0.23076923 0.43661972], recall=[0.50769231 0.13235294 0.5       ], f1=[0.47030879 0.1682243  0.46616541], accuracy=0.41
    rec:['stage1', 288, 517, '+:186,0:136,-:195', 0.26305609284332687, 0.30754352030947774, 0.3520309477756286, 0.3733075435203095, 0.379110251450677, 0.36363636363636365, 0.30754352030947774, 0.4274661508704062, 0.3713733075435203, 0.40618955512572535]
    [*] predict with currank model
    precision=[0.         0.29166667 0.        ], recall=[0. 1. 0.], f1=[0.        0.4516129 0.       ], accuracy=0.29
    [*] predict with avgrank model
    precision=[0.28630705 0.28571429 0.21311475], recall=[0.52272727 0.05357143 0.18571429], f1=[0.36997319 0.09022556 0.19847328], accuracy=0.26
    [*] predict with dice model
    precision=[0.3627451  0.17567568 0.33018868], recall=[0.56060606 0.11607143 0.25      ], f1=[0.44047619 0.13978495 0.28455285], accuracy=0.32
    [*] predict with lr model
    precision=[0.34364261 0.28125    0.49180328], recall=[0.75757576 0.08035714 0.21428571], f1=[0.47281324 0.125      0.29850746], accuracy=0.36
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


    precision=[0.35211268 0.27272727 0.5       ], recall=[0.75757576 0.10714286 0.2       ], f1=[0.48076923 0.15384615 0.28571429], accuracy=0.36
    [*] predict with lsvc model
    precision=[0.35202492 0.31578947 0.56818182], recall=[0.85606061 0.05357143 0.17857143], f1=[0.49889625 0.09160305 0.27173913], accuracy=0.38
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.29059829 0.23404255 0.40909091], recall=[0.25757576 0.09821429 0.64285714], f1=[0.27309237 0.13836478 0.5       ], accuracy=0.35
    [*] predict with rf model
    precision=[0.41403509 0.69565217 0.56578947], recall=[0.89393939 0.14285714 0.30714286], f1=[0.56594724 0.23703704 0.39814815], accuracy=0.46
    [*] predict with lrbias model
    [LibLinear]precision=[0.33797909 0.28571429 0.48387097], recall=[0.73484848 0.08928571 0.21428571], f1=[0.46300716 0.13605442 0.2970297 ], accuracy=0.36
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.42173913 0.425      0.53508772], recall=[0.73484848 0.15178571 0.43571429], f1=[0.5359116  0.22368421 0.48031496], accuracy=0.46
    rec:['stage2', 421, 384, '+:140,0:112,-:132', 0.2916666666666667, 0.2630208333333333, 0.3177083333333333, 0.3619791666666667, 0.3645833333333333, 0.375, 0.3515625, 0.4609375, 0.3567708333333333, 0.4557291666666667]
    [*] predict with currank model
    precision=[0.         0.34496124 0.        ], recall=[0. 1. 0.], f1=[0.        0.5129683 0.       ], accuracy=0.34
    [*] predict with avgrank model
    precision=[0.22929936 0.29411765 0.21428571], recall=[0.46153846 0.05617978 0.1978022 ], f1=[0.30638298 0.09433962 0.20571429], accuracy=0.23
    [*] predict with dice model
    precision=[0.28985507 0.4375     0.34722222], recall=[0.51282051 0.23595506 0.27472527], f1=[0.37037037 0.30656934 0.30674847], accuracy=0.33
    [*] predict with lr model
    precision=[0.44166667 0.21052632 0.51260504], recall=[0.67948718 0.04494382 0.67032967], f1=[0.53535354 0.07407407 0.58095238], accuracy=0.46
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


    precision=[0.42519685 0.2        0.52252252], recall=[0.69230769 0.04494382 0.63736264], f1=[0.52682927 0.0733945  0.57425743], accuracy=0.45
    [*] predict with lsvc model
    precision=[0.43089431 0.13333333 0.51666667], recall=[0.67948718 0.02247191 0.68131868], f1=[0.52736318 0.03846154 0.58767773], accuracy=0.45
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.65217391 0.40243902 0.44444444], recall=[0.19230769 0.37078652 0.74725275], f1=[0.2970297  0.38596491 0.55737705], accuracy=0.45
    [*] predict with rf model
    precision=[0.36315789 0.71428571 0.46296296], recall=[0.88461538 0.11235955 0.27472527], f1=[0.51492537 0.19417476 0.34482759], accuracy=0.40
    [*] predict with lrbias model
    [LibLinear]precision=[0.43333333 0.2        0.50847458], recall=[0.66666667 0.04494382 0.65934066], f1=[0.52525253 0.0733945  0.57416268], accuracy=0.45
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.38853503 0.52173913 0.47435897], recall=[0.78205128 0.13483146 0.40659341], f1=[0.51914894 0.21428571 0.43786982], accuracy=0.43
    rec:['stage3', 547, 258, '+:91,0:89,-:78', 0.3449612403100775, 0.22868217054263565, 0.3333333333333333, 0.4573643410852713, 0.4496124031007752, 0.45348837209302323, 0.4496124031007752, 0.40310077519379844, 0.4496124031007752, 0.4263565891472868]
    [*] predict with currank model
    precision=[0.         0.35810811 0.        ], recall=[0. 1. 0.], f1=[0.         0.52736318 0.        ], accuracy=0.36
    [*] predict with avgrank model
    precision=[0.21348315 0.22222222 0.2       ], recall=[0.40425532 0.03773585 0.20833333], f1=[0.27941176 0.06451613 0.20408163], accuracy=0.21
    [*] predict with dice model
    precision=[0.33802817 0.4375     0.28888889], recall=[0.5106383  0.26415094 0.27083333], f1=[0.40677966 0.32941176 0.27956989], accuracy=0.34
    [*] predict with lr model
    precision=[0.56521739 0.42424242 0.49152542], recall=[0.27659574 0.52830189 0.60416667], f1=[0.37142857 0.47058824 0.54205607], accuracy=0.47
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


    precision=[0.54166667 0.42622951 0.50793651], recall=[0.27659574 0.49056604 0.66666667], f1=[0.36619718 0.45614035 0.57657658], accuracy=0.48
    [*] predict with lsvc model
    precision=[0.52       0.44067797 0.5       ], recall=[0.27659574 0.49056604 0.66666667], f1=[0.36111111 0.46428571 0.57142857], accuracy=0.48
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.5        0.18181818 0.40594059], recall=[0.38297872 0.03773585 0.85416667], f1=[0.43373494 0.0625     0.55033557], accuracy=0.41
    [*] predict with rf model
    precision=[0.43181818 0.61764706 0.65384615], recall=[0.80851064 0.39622642 0.35416667], f1=[0.56296296 0.48275862 0.45945946], accuracy=0.51
    [*] predict with lrbias model
    [LibLinear]precision=[0.59090909 0.4057971  0.49122807], recall=[0.27659574 0.52830189 0.58333333], f1=[0.37681159 0.45901639 0.53333333], accuracy=0.47
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.45614035 0.53333333 0.5       ], recall=[0.55319149 0.45283019 0.47916667], f1=[0.5        0.48979592 0.4893617 ], accuracy=0.49
    rec:['stage4', 657, 148, '+:48,0:53,-:47', 0.3581081081081081, 0.20945945945945946, 0.34459459459459457, 0.47297297297297297, 0.4797297297297297, 0.4797297297297297, 0.41216216216216217, 0.5135135135135135, 0.46621621621621623, 0.49324324324324326]
    [*] predict with currank model
    precision=[0.     0.3625 0.    ], recall=[0. 1. 0.], f1=[0.         0.53211009 0.        ], accuracy=0.36
    [*] predict with avgrank model
    precision=[0.19565217 0.28571429 0.18518519], recall=[0.36       0.06896552 0.19230769], f1=[0.25352113 0.11111111 0.18867925], accuracy=0.20
    [*] predict with dice model
    precision=[0.32352941 0.26666667 0.35483871], recall=[0.44       0.13793103 0.42307692], f1=[0.37288136 0.18181818 0.38596491], accuracy=0.33
    [*] predict with lr model
    precision=[0.5        0.47619048 0.57692308], recall=[0.24       0.68965517 0.57692308], f1=[0.32432432 0.56338028 0.57692308], accuracy=0.51
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


    precision=[0.5        0.48780488 0.55555556], recall=[0.24       0.68965517 0.57692308], f1=[0.32432432 0.57142857 0.56603774], accuracy=0.51
    [*] predict with lsvc model
    precision=[0.5        0.48717949 0.55555556], recall=[0.28       0.65517241 0.57692308], f1=[0.35897436 0.55882353 0.56603774], accuracy=0.51
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.52380952 0.42592593 0.8       ], recall=[0.44       0.79310345 0.15384615], f1=[0.47826087 0.55421687 0.25806452], accuracy=0.47
    [*] predict with rf model
    precision=[0.43243243 0.48148148 0.6875    ], recall=[0.64       0.44827586 0.42307692], f1=[0.51612903 0.46428571 0.52380952], accuracy=0.50
    [*] predict with lrbias model
    [LibLinear]precision=[0.5        0.47619048 0.57692308], recall=[0.24       0.68965517 0.57692308], f1=[0.32432432 0.56338028 0.57692308], accuracy=0.51
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.57142857 0.62962963 0.68      ], recall=[0.64       0.5862069  0.65384615], f1=[0.60377358 0.60714286 0.66666667], accuracy=0.62
    rec:['stage5', 725, 80, '+:26,0:29,-:25', 0.3625, 0.2, 0.325, 0.5125, 0.5125, 0.5125, 0.475, 0.5, 0.5125, 0.625]
    [*] predict with currank model
    precision=[0.         0.34210526 0.        ], recall=[0. 1. 0.], f1=[0.         0.50980392 0.        ], accuracy=0.34
    [*] predict with avgrank model
    precision=[0.27272727 0.33333333 0.1       ], recall=[0.42857143 0.15384615 0.09090909], f1=[0.33333333 0.21052632 0.0952381 ], accuracy=0.24
    [*] predict with dice model
    precision=[0.2        0.3        0.30769231], recall=[0.21428571 0.23076923 0.36363636], f1=[0.20689655 0.26086957 0.33333333], accuracy=0.26
    [*] predict with lr model
    precision=[0.57142857 0.3        0.36363636], recall=[0.28571429 0.46153846 0.36363636], f1=[0.38095238 0.36363636 0.36363636], accuracy=0.37
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


    precision=[0.57142857 0.31578947 0.41666667], recall=[0.28571429 0.46153846 0.45454545], f1=[0.38095238 0.375      0.43478261], accuracy=0.39
    [*] predict with lsvc model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.66666667 0.35       0.41666667], recall=[0.28571429 0.53846154 0.45454545], f1=[0.4        0.42424242 0.43478261], accuracy=0.42
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[1.         0.33333333 0.28571429], recall=[0.07142857 0.23076923 0.72727273], f1=[0.13333333 0.27272727 0.41025641], accuracy=0.32
    [*] predict with rf model
    precision=[0.52173913 0.5        0.66666667], recall=[0.85714286 0.23076923 0.54545455], f1=[0.64864865 0.31578947 0.6       ], accuracy=0.55
    [*] predict with lrbias model
    [LibLinear]precision=[0.6        0.33333333 0.33333333], recall=[0.21428571 0.53846154 0.36363636], f1=[0.31578947 0.41176471 0.34782609], accuracy=0.37
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.5625 0.5    0.5   ], recall=[0.64285714 0.23076923 0.72727273], f1=[0.6        0.31578947 0.59259259], accuracy=0.53
    rec:['stage6', 767, 38, '+:11,0:13,-:14', 0.34210526315789475, 0.23684210526315788, 0.2631578947368421, 0.3684210526315789, 0.39473684210526316, 0.42105263157894735, 0.3157894736842105, 0.5526315789473685, 0.3684210526315789, 0.5263157894736842]
    [*] predict with currank model
    precision=[0.   0.25 0.  ], recall=[0. 1. 0.], f1=[0.  0.4 0. ], accuracy=0.25
    [*] predict with avgrank model
    precision=[0.3 0.  0. ], recall=[0.375 0.    0.   ], f1=[0.33333333 0.         0.        ], accuracy=0.19
    [*] predict with dice model
    precision=[0.42857143 0.2        0.25      ], recall=[0.375 0.25  0.25 ], f1=[0.4        0.22222222 0.25      ], accuracy=0.31
    [*] predict with lr model
    precision=[0.         0.09090909 0.        ], recall=[0.   0.25 0.  ], f1=[0.         0.13333333 0.        ], accuracy=0.06
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
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)


    precision=[0.         0.11111111 0.28571429], recall=[0.   0.25 0.5 ], f1=[0.         0.15384615 0.36363636], accuracy=0.19
    [*] predict with lsvc model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)


    precision=[0.         0.11111111 0.28571429], recall=[0.   0.25 0.5 ], f1=[0.         0.15384615 0.36363636], accuracy=0.19
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)


    precision=[0.   0.25 0.  ], recall=[0. 1. 0.], f1=[0.  0.4 0. ], accuracy=0.25
    [*] predict with rf model
    precision=[0.44444444 0.2        1.        ], recall=[0.5  0.25 0.5 ], f1=[0.47058824 0.22222222 0.66666667], accuracy=0.44
    [*] predict with lrbias model
    [LibLinear]precision=[0.         0.1        0.16666667], recall=[0.   0.25 0.25], f1=[0.         0.14285714 0.2       ], accuracy=0.12
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)


    precision=[0.4 0.2 0.5], recall=[0.25 0.25 0.75], f1=[0.30769231 0.22222222 0.6       ], accuracy=0.38
    rec:['stage7', 789, 16, '+:4,0:4,-:8', 0.25, 0.1875, 0.3125, 0.0625, 0.1875, 0.1875, 0.25, 0.4375, 0.125, 0.375]



```python
#xgb max_tree_depth=3
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
      <td>0.463190</td>
      <td>0.417178</td>
      <td>0.414110</td>
      <td>0.374233</td>
      <td>0.447853</td>
      <td>0.355828</td>
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
      <td>0.373308</td>
      <td>0.379110</td>
      <td>0.363636</td>
      <td>0.307544</td>
      <td>0.427466</td>
      <td>0.371373</td>
      <td>0.406190</td>
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
      <td>0.361979</td>
      <td>0.364583</td>
      <td>0.375000</td>
      <td>0.351562</td>
      <td>0.460938</td>
      <td>0.356771</td>
      <td>0.455729</td>
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
      <td>0.457364</td>
      <td>0.449612</td>
      <td>0.453488</td>
      <td>0.449612</td>
      <td>0.403101</td>
      <td>0.449612</td>
      <td>0.426357</td>
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
      <td>0.472973</td>
      <td>0.479730</td>
      <td>0.479730</td>
      <td>0.412162</td>
      <td>0.513514</td>
      <td>0.466216</td>
      <td>0.493243</td>
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
      <td>0.512500</td>
      <td>0.512500</td>
      <td>0.512500</td>
      <td>0.475000</td>
      <td>0.500000</td>
      <td>0.512500</td>
      <td>0.625000</td>
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
      <td>0.368421</td>
      <td>0.394737</td>
      <td>0.421053</td>
      <td>0.315789</td>
      <td>0.552632</td>
      <td>0.368421</td>
      <td>0.526316</td>
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
      <td>0.062500</td>
      <td>0.187500</td>
      <td>0.187500</td>
      <td>0.250000</td>
      <td>0.437500</td>
      <td>0.125000</td>
      <td>0.375000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#xgb max_tree_depth=6
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
      <td>0.463190</td>
      <td>0.417178</td>
      <td>0.414110</td>
      <td>0.374233</td>
      <td>0.447853</td>
      <td>0.355828</td>
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
      <td>0.373308</td>
      <td>0.379110</td>
      <td>0.363636</td>
      <td>0.307544</td>
      <td>0.427466</td>
      <td>0.371373</td>
      <td>0.406190</td>
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
      <td>0.361979</td>
      <td>0.364583</td>
      <td>0.375000</td>
      <td>0.351562</td>
      <td>0.460938</td>
      <td>0.356771</td>
      <td>0.455729</td>
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
      <td>0.457364</td>
      <td>0.449612</td>
      <td>0.453488</td>
      <td>0.449612</td>
      <td>0.403101</td>
      <td>0.449612</td>
      <td>0.426357</td>
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
      <td>0.472973</td>
      <td>0.479730</td>
      <td>0.479730</td>
      <td>0.412162</td>
      <td>0.513514</td>
      <td>0.466216</td>
      <td>0.493243</td>
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
      <td>0.512500</td>
      <td>0.512500</td>
      <td>0.512500</td>
      <td>0.475000</td>
      <td>0.500000</td>
      <td>0.512500</td>
      <td>0.625000</td>
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
      <td>0.368421</td>
      <td>0.394737</td>
      <td>0.421053</td>
      <td>0.315789</td>
      <td>0.552632</td>
      <td>0.368421</td>
      <td>0.526316</td>
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
      <td>0.062500</td>
      <td>0.187500</td>
      <td>0.187500</td>
      <td>0.250000</td>
      <td>0.437500</td>
      <td>0.125000</td>
      <td>0.375000</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
