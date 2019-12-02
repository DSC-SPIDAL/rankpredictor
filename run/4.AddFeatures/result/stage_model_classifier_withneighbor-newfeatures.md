
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
suffix='-withneighbor-newfeatures'
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
      <td>2.948049</td>
      <td>39</td>
      <td>39</td>
      <td>31.76610</td>
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
      <td>7.710188</td>
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
      <td>6.682523</td>
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
      <td>6.078419</td>
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
    precision=[0.63934426 0.2        0.44186047], recall=[0.65  0.125 0.5  ], f1=[0.6446281  0.15384615 0.4691358 ], accuracy=0.53
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


    precision=[0.63333333 0.         0.4375    ], recall=[0.63333333 0.         0.55263158], f1=[0.63333333 0.         0.48837209], accuracy=0.52
    [*] predict with lsvc model
    precision=[0.62711864 0.15384615 0.42857143], recall=[0.61666667 0.125      0.47368421], f1=[0.62184874 0.13793103 0.45      ], accuracy=0.50
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.62962963 0.15625    0.35714286], recall=[0.56666667 0.3125     0.26315789], f1=[0.59649123 0.20833333 0.3030303 ], accuracy=0.43
    [*] predict with rf model
    precision=[0.64150943 0.27777778 0.48837209], recall=[0.56666667 0.3125     0.55263158], f1=[0.60176991 0.29411765 0.51851852], accuracy=0.53
    [*] predict with lrbias model
    [LibLinear]precision=[0.63934426 0.         0.43478261], recall=[0.65       0.         0.52631579], f1=[0.6446281  0.         0.47619048], accuracy=0.52
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.575      0.11111111 0.5       ], recall=[0.38333333 0.25       0.5       ], f1=[0.46       0.15384615 0.5       ], accuracy=0.40
    rec:['Phoenix', 691, 114, '+:38,0:16,-:60', 0.14035087719298245, 0.32456140350877194, 0.3684210526315789, 0.5263157894736842, 0.5175438596491229, 0.5, 0.4298245614035088, 0.5263157894736842, 0.5175438596491229, 0.40350877192982454]
    Testset = Indy500
    [*] predict with currank model
    precision=[0.         0.20888889 0.        ], recall=[0. 1. 0.], f1=[0.         0.34558824 0.        ], accuracy=0.21
    [*] predict with avgrank model
    precision=[0.4017094  0.24528302 0.23636364], recall=[0.48958333 0.27659574 0.15853659], f1=[0.44131455 0.26       0.18978102], accuracy=0.32
    [*] predict with dice model
    precision=[0.47619048 0.26315789 0.38095238], recall=[0.52083333 0.31914894 0.29268293], f1=[0.49751244 0.28846154 0.33103448], accuracy=0.40
    [*] predict with lr model
    precision=[0.57342657 0.38888889 0.640625  ], recall=[0.85416667 0.14893617 0.5       ], f1=[0.68619247 0.21538462 0.56164384], accuracy=0.58
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


    precision=[0.56643357 0.36842105 0.63492063], recall=[0.84375    0.14893617 0.48780488], f1=[0.67782427 0.21212121 0.55172414], accuracy=0.57
    [*] predict with lsvc model
    precision=[0.56737589 0.30769231 0.61971831], recall=[0.83333333 0.08510638 0.53658537], f1=[0.67510549 0.13333333 0.5751634 ], accuracy=0.57
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.609375   0.2278481  0.33333333], recall=[0.40625    0.76595745 0.01219512], f1=[0.4875     0.35121951 0.02352941], accuracy=0.34
    [*] predict with rf model
    precision=[0.54966887 0.28846154 0.45454545], recall=[0.86458333 0.31914894 0.12195122], f1=[0.67206478 0.3030303  0.19230769], accuracy=0.48
    [*] predict with lrbias model
    [LibLinear]precision=[0.56028369 0.35       0.625     ], recall=[0.82291667 0.14893617 0.48780488], f1=[0.66666667 0.20895522 0.54794521], accuracy=0.56
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.5826087  0.34042553 0.53968254], recall=[0.69791667 0.34042553 0.41463415], f1=[0.63507109 0.34042553 0.46896552], accuracy=0.52
    rec:['Indy500', 580, 225, '+:82,0:47,-:96', 0.2088888888888889, 0.3244444444444444, 0.39555555555555555, 0.5777777777777777, 0.5688888888888889, 0.5688888888888889, 0.3377777777777778, 0.48, 0.56, 0.52]
    Testset = Texas
    [*] predict with currank model
    precision=[0.         0.26771654 0.        ], recall=[0. 1. 0.], f1=[0.         0.42236025 0.        ], accuracy=0.27
    [*] predict with avgrank model
    precision=[0.37037037 0.3255814  0.2       ], recall=[0.37037037 0.41176471 0.15384615], f1=[0.37037037 0.36363636 0.17391304], accuracy=0.31
    [*] predict with dice model
    precision=[0.30357143 0.23333333 0.31707317], recall=[0.31481481 0.20588235 0.33333333], f1=[0.30909091 0.21875    0.325     ], accuracy=0.29
    [*] predict with lr model
    precision=[0.61538462 0.5        0.54545455], recall=[0.59259259 0.29411765 0.76923077], f1=[0.60377358 0.37037037 0.63829787], accuracy=0.57
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


    precision=[0.63157895 0.5        0.58      ], recall=[0.66666667 0.29411765 0.74358974], f1=[0.64864865 0.37037037 0.65168539], accuracy=0.59
    [*] predict with lsvc model
    precision=[0.61818182 0.47058824 0.54545455], recall=[0.62962963 0.23529412 0.76923077], f1=[0.62385321 0.31372549 0.63829787], accuracy=0.57
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.43835616 0.3        0.28571429], recall=[0.59259259 0.35294118 0.1025641 ], f1=[0.50393701 0.32432432 0.1509434 ], accuracy=0.38
    [*] predict with rf model
    precision=[0.60869565 0.35       0.37704918], recall=[0.51851852 0.20588235 0.58974359], f1=[0.56       0.25925926 0.46      ], accuracy=0.46
    [*] predict with lrbias model
    [LibLinear]precision=[0.61538462 0.52380952 0.55555556], recall=[0.59259259 0.32352941 0.76923077], f1=[0.60377358 0.4        0.64516129], accuracy=0.57
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.64       0.375      0.39130435], recall=[0.59259259 0.08823529 0.69230769], f1=[0.61538462 0.14285714 0.5       ], accuracy=0.49
    rec:['Texas', 678, 127, '+:39,0:34,-:54', 0.2677165354330709, 0.31496062992125984, 0.29133858267716534, 0.5669291338582677, 0.5905511811023622, 0.5669291338582677, 0.3779527559055118, 0.4566929133858268, 0.5748031496062992, 0.4881889763779528]
    Testset = Iowa
    [*] predict with currank model
    precision=[0.         0.25688073 0.        ], recall=[0. 1. 0.], f1=[0.         0.40875912 0.        ], accuracy=0.26
    [*] predict with avgrank model
    precision=[0.30232558 0.20689655 0.35135135], recall=[0.33333333 0.21428571 0.30952381], f1=[0.31707317 0.21052632 0.32911392], accuracy=0.29
    [*] predict with dice model
    precision=[0.25531915 0.18518519 0.31428571], recall=[0.30769231 0.17857143 0.26190476], f1=[0.27906977 0.18181818 0.28571429], accuracy=0.26
    [*] predict with lr model
    precision=[0.3974359  0.07692308 0.66666667], recall=[0.79487179 0.03571429 0.28571429], f1=[0.52991453 0.04878049 0.4       ], accuracy=0.40
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


    precision=[0.39473684 0.07142857 0.63157895], recall=[0.76923077 0.03571429 0.28571429], f1=[0.52173913 0.04761905 0.39344262], accuracy=0.39
    [*] predict with lsvc model
    precision=[0.3974359  0.07692308 0.66666667], recall=[0.79487179 0.03571429 0.28571429], f1=[0.52991453 0.04878049 0.4       ], accuracy=0.40
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.3373494  0.         0.46666667], recall=[0.71794872 0.         0.16666667], f1=[0.45901639 0.         0.24561404], accuracy=0.32
    [*] predict with rf model
    precision=[0.35526316 0.09090909 0.54545455], recall=[0.69230769 0.03571429 0.28571429], f1=[0.46956522 0.05128205 0.375     ], accuracy=0.37
    [*] predict with lrbias model
    [LibLinear]precision=[0.4025974  0.07142857 0.66666667], recall=[0.79487179 0.03571429 0.28571429], f1=[0.53448276 0.04761905 0.4       ], accuracy=0.40
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.33333333 0.17647059 0.52173913], recall=[0.58974359 0.10714286 0.28571429], f1=[0.42592593 0.13333333 0.36923077], accuracy=0.35
    rec:['Iowa', 696, 109, '+:42,0:28,-:39', 0.25688073394495414, 0.29357798165137616, 0.25688073394495414, 0.4036697247706422, 0.3944954128440367, 0.4036697247706422, 0.3211009174311927, 0.3669724770642202, 0.4036697247706422, 0.3486238532110092]
    Testset = Pocono
    [*] predict with currank model
    precision=[0.         0.48412698 0.        ], recall=[0. 1. 0.], f1=[0.         0.65240642 0.        ], accuracy=0.48
    [*] predict with avgrank model
    precision=[0.265625   0.44117647 0.10714286], recall=[0.47222222 0.24590164 0.10344828], f1=[0.34       0.31578947 0.10526316], accuracy=0.28
    [*] predict with dice model
    precision=[0.23214286 0.34615385 0.18181818], recall=[0.36111111 0.14754098 0.27586207], f1=[0.2826087  0.20689655 0.21917808], accuracy=0.24
    [*] predict with lr model
    precision=[0.24489796 0.5        0.19230769], recall=[0.66666667 0.01639344 0.17241379], f1=[0.35820896 0.03174603 0.18181818], accuracy=0.24
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


    precision=[0.26851852 0.5        0.25      ], recall=[0.80555556 0.01639344 0.13793103], f1=[0.40277778 0.03174603 0.17777778], accuracy=0.27
    [*] predict with lsvc model
    precision=[0.28070175 0.         0.25      ], recall=[0.88888889 0.         0.10344828], f1=[0.42666667 0.         0.14634146], accuracy=0.28
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.23595506 0.         0.13888889], recall=[0.58333333 0.         0.17241379], f1=[0.336      0.         0.15384615], accuracy=0.21
    [*] predict with rf model
    precision=[0.41666667 0.59090909 0.23214286], recall=[0.55555556 0.21311475 0.44827586], f1=[0.47619048 0.31325301 0.30588235], accuracy=0.37
    [*] predict with lrbias model
    [LibLinear]precision=[0.24489796 0.         0.18518519], recall=[0.66666667 0.         0.17241379], f1=[0.35820896 0.         0.17857143], accuracy=0.23
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.4        0.6        0.30434783], recall=[0.55555556 0.29508197 0.48275862], f1=[0.46511628 0.3956044  0.37333333], accuracy=0.41
    rec:['Pocono', 679, 126, '+:29,0:61,-:36', 0.48412698412698413, 0.2777777777777778, 0.23809523809523808, 0.23809523809523808, 0.2698412698412698, 0.2777777777777778, 0.20634920634920634, 0.36507936507936506, 0.23015873015873015, 0.4126984126984127]
    Testset = Gateway
    [*] predict with currank model
    precision=[0.         0.26923077 0.        ], recall=[0. 1. 0.], f1=[0.         0.42424242 0.        ], accuracy=0.27
    [*] predict with avgrank model
    precision=[0.36956522 0.29411765 0.29166667], recall=[0.4047619  0.35714286 0.20588235], f1=[0.38636364 0.32258065 0.24137931], accuracy=0.33
    [*] predict with dice model
    precision=[0.34782609 0.13043478 0.2       ], recall=[0.38095238 0.10714286 0.20588235], f1=[0.36363636 0.11764706 0.20289855], accuracy=0.25
    [*] predict with lr model
    precision=[0.65       0.38888889 0.56666667], recall=[0.30952381 0.75       0.5       ], f1=[0.41935484 0.51219512 0.53125   ], accuracy=0.49
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


    precision=[0.8        0.40322581 0.59259259], recall=[0.28571429 0.89285714 0.47058824], f1=[0.42105263 0.55555556 0.52459016], accuracy=0.51
    [*] predict with lsvc model
    precision=[0.78571429 0.40983607 0.55172414], recall=[0.26190476 0.89285714 0.47058824], f1=[0.39285714 0.56179775 0.50793651], accuracy=0.50
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.54545455 0.31578947 0.54545455], recall=[0.71428571 0.42857143 0.17647059], f1=[0.6185567  0.36363636 0.26666667], accuracy=0.46
    [*] predict with rf model
    precision=[0.47727273 0.5        0.45      ], recall=[0.5        0.35714286 0.52941176], f1=[0.48837209 0.41666667 0.48648649], accuracy=0.47
    [*] predict with lrbias model
    [LibLinear]precision=[0.75      0.3968254 0.6      ], recall=[0.28571429 0.89285714 0.44117647], f1=[0.4137931  0.54945055 0.50847458], accuracy=0.50
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.51351351 0.31707317 0.42307692], recall=[0.45238095 0.46428571 0.32352941], f1=[0.48101266 0.37681159 0.36666667], accuracy=0.41
    rec:['Gateway', 701, 104, '+:34,0:28,-:42', 0.2692307692307692, 0.3269230769230769, 0.25, 0.49038461538461536, 0.5096153846153846, 0.5, 0.46153846153846156, 0.47115384615384615, 0.5, 0.41346153846153844]



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
      <td>0.526316</td>
      <td>0.517544</td>
      <td>0.500000</td>
      <td>0.429825</td>
      <td>0.526316</td>
      <td>0.517544</td>
      <td>0.403509</td>
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
      <td>0.577778</td>
      <td>0.568889</td>
      <td>0.568889</td>
      <td>0.337778</td>
      <td>0.480000</td>
      <td>0.560000</td>
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
      <td>0.566929</td>
      <td>0.590551</td>
      <td>0.566929</td>
      <td>0.377953</td>
      <td>0.456693</td>
      <td>0.574803</td>
      <td>0.488189</td>
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
      <td>0.403670</td>
      <td>0.394495</td>
      <td>0.403670</td>
      <td>0.321101</td>
      <td>0.366972</td>
      <td>0.403670</td>
      <td>0.348624</td>
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
      <td>0.238095</td>
      <td>0.269841</td>
      <td>0.277778</td>
      <td>0.206349</td>
      <td>0.365079</td>
      <td>0.230159</td>
      <td>0.412698</td>
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
      <td>0.490385</td>
      <td>0.509615</td>
      <td>0.500000</td>
      <td>0.461538</td>
      <td>0.471154</td>
      <td>0.500000</td>
      <td>0.413462</td>
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
    precision=[0.49739583 0.36585366 0.44444444], recall=[0.72348485 0.4491018  0.12669683], f1=[0.58950617 0.40322581 0.1971831 ], accuracy=0.45
    [*] predict with lsvc model
    precision=[0.48087432 0.34893617 0.23529412], recall=[0.66666667 0.49101796 0.05429864], f1=[0.55873016 0.4079602  0.08823529], accuracy=0.41
    [*] predict with lsvcl2 model
    precision=[0.58333333 0.39583333 0.35384615], recall=[0.07954545 0.22754491 0.83257919], f1=[0.14       0.28897338 0.49662618], accuracy=0.37
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


    precision=[0.47916667 0.29439252 0.32666667], recall=[0.52272727 0.37724551 0.22171946], f1=[0.5        0.33070866 0.26415094], accuracy=0.38
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
    rec:['stage0', 153, 652, '+:221,0:167,-:264', 0.2561349693251534, 0.3128834355828221, 0.36349693251533743, 0.4647239263803681, 0.450920245398773, 0.41411042944785276, 0.3726993865030675, 0.3834355828220859, 0.44785276073619634, 0.3558282208588957]
    [*] predict with currank model
    precision=[0.         0.26305609 0.        ], recall=[0. 1. 0.], f1=[0.         0.41653905 0.        ], accuracy=0.26
    [*] predict with avgrank model
    precision=[0.34674923 0.24242424 0.24223602], recall=[0.57435897 0.05882353 0.20967742], f1=[0.43243243 0.09467456 0.22478386], accuracy=0.31
    [*] predict with dice model
    precision=[0.39312977 0.22881356 0.37956204], recall=[0.52820513 0.19852941 0.27956989], f1=[0.45076586 0.21259843 0.32198142], accuracy=0.35
    [*] predict with lr model
    precision=[0.36       0.14285714 0.72727273], recall=[0.83076923 0.05882353 0.04301075], f1=[0.50232558 0.08333333 0.08121827], accuracy=0.34
    [*] predict with lrl1 model
    precision=[0.37435897 0.16455696 0.625     ], recall=[0.74871795 0.09558824 0.16129032], f1=[0.4991453  0.12093023 0.25641026], accuracy=0.37
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


    precision=[0.36754177 0.14285714 0.71428571], recall=[0.78974359 0.08823529 0.05376344], f1=[0.50162866 0.10909091 0.1       ], accuracy=0.34
    [*] predict with lsvcl2 model
    precision=[0.37071651 0.19444444 0.625     ], recall=[0.61025641 0.25735294 0.05376344], f1=[0.46124031 0.22151899 0.0990099 ], accuracy=0.32
    [*] predict with rf model
    precision=[0.41919192 0.25       0.50588235], recall=[0.85128205 0.06617647 0.2311828 ], f1=[0.56175973 0.10465116 0.31734317], accuracy=0.42
    [*] predict with lrbias model
    [LibLinear]precision=[0.36080178 0.14285714 0.75      ], recall=[0.83076923 0.05882353 0.0483871 ], f1=[0.50310559 0.08333333 0.09090909], accuracy=0.35
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.43597561 0.27868852 0.46875   ], recall=[0.73333333 0.125      0.32258065], f1=[0.54684512 0.17258883 0.38216561], accuracy=0.43
    rec:['stage1', 288, 517, '+:186,0:136,-:195', 0.26305609284332687, 0.30754352030947774, 0.3520309477756286, 0.344294003868472, 0.3655705996131528, 0.3404255319148936, 0.31721470019342357, 0.42166344294003866, 0.34622823984526113, 0.425531914893617]
    [*] predict with currank model
    precision=[0.         0.29166667 0.        ], recall=[0. 1. 0.], f1=[0.        0.4516129 0.       ], accuracy=0.29
    [*] predict with avgrank model
    precision=[0.28630705 0.28571429 0.21311475], recall=[0.52272727 0.05357143 0.18571429], f1=[0.36997319 0.09022556 0.19847328], accuracy=0.26
    [*] predict with dice model
    precision=[0.3627451  0.17567568 0.33018868], recall=[0.56060606 0.11607143 0.25      ], f1=[0.44047619 0.13978495 0.28455285], accuracy=0.32
    [*] predict with lr model
    precision=[0.364      0.375      0.49019608], recall=[0.68939394 0.10714286 0.35714286], f1=[0.47643979 0.16666667 0.41322314], accuracy=0.40
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
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.36042403 0.30769231 0.5       ], recall=[0.77272727 0.03571429 0.31428571], f1=[0.49156627 0.064      0.38596491], accuracy=0.39
    [*] predict with lsvc model
    precision=[0.38132296 0.33333333 0.51694915], recall=[0.74242424 0.02678571 0.43571429], f1=[0.50385604 0.04958678 0.47286822], accuracy=0.42
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.34883721 0.         0.53846154], recall=[0.90909091 0.         0.15      ], f1=[0.50420168 0.         0.23463687], accuracy=0.37
    [*] predict with rf model
    precision=[0.39642857 0.45454545 0.51219512], recall=[0.84090909 0.08928571 0.3       ], f1=[0.53883495 0.14925373 0.37837838], accuracy=0.42
    [*] predict with lrbias model
    [LibLinear]precision=[0.3625498  0.4        0.48543689], recall=[0.68939394 0.10714286 0.35714286], f1=[0.47519582 0.16901408 0.41152263], accuracy=0.40
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.42857143 0.51351351 0.52586207], recall=[0.75       0.16964286 0.43571429], f1=[0.54545455 0.25503356 0.4765625 ], accuracy=0.47
    rec:['stage2', 421, 384, '+:140,0:112,-:132', 0.2916666666666667, 0.2630208333333333, 0.3177083333333333, 0.3984375, 0.390625, 0.421875, 0.3671875, 0.4244791666666667, 0.3984375, 0.4661458333333333]
    [*] predict with currank model
    precision=[0.         0.34496124 0.        ], recall=[0. 1. 0.], f1=[0.        0.5129683 0.       ], accuracy=0.34
    [*] predict with avgrank model
    precision=[0.22929936 0.29411765 0.21428571], recall=[0.46153846 0.05617978 0.1978022 ], f1=[0.30638298 0.09433962 0.20571429], accuracy=0.23
    [*] predict with dice model
    precision=[0.28985507 0.4375     0.34722222], recall=[0.51282051 0.23595506 0.27472527], f1=[0.37037037 0.30656934 0.30674847], accuracy=0.33
    [*] predict with lr model
    precision=[0.43859649 0.3        0.44776119], recall=[0.64102564 0.03370787 0.65934066], f1=[0.52083333 0.06060606 0.53333333], accuracy=0.44
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


    precision=[0.42241379 0.2        0.43939394], recall=[0.62820513 0.02247191 0.63736264], f1=[0.50515464 0.04040404 0.52017937], accuracy=0.42
    [*] predict with lsvc model
    precision=[0.4375     0.22222222 0.45255474], recall=[0.62820513 0.02247191 0.68131868], f1=[0.51578947 0.04081633 0.54385965], accuracy=0.44
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.54545455 0.24193548 0.38918919], recall=[0.07692308 0.16853933 0.79120879], f1=[0.13483146 0.1986755  0.52173913], accuracy=0.36
    [*] predict with rf model
    precision=[0.37566138 0.6875     0.56603774], recall=[0.91025641 0.12359551 0.32967033], f1=[0.53183521 0.20952381 0.41666667], accuracy=0.43
    [*] predict with lrbias model
    [LibLinear]precision=[0.42982456 0.27272727 0.44360902], recall=[0.62820513 0.03370787 0.64835165], f1=[0.51041667 0.06       0.52678571], accuracy=0.43
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.39285714 0.41176471 0.57534247], recall=[0.84615385 0.07865169 0.46153846], f1=[0.53658537 0.13207547 0.51219512], accuracy=0.45
    rec:['stage3', 547, 258, '+:91,0:89,-:78', 0.3449612403100775, 0.22868217054263565, 0.3333333333333333, 0.437984496124031, 0.42248062015503873, 0.437984496124031, 0.36046511627906974, 0.43410852713178294, 0.43023255813953487, 0.44573643410852715]
    [*] predict with currank model
    precision=[0.         0.35810811 0.        ], recall=[0. 1. 0.], f1=[0.         0.52736318 0.        ], accuracy=0.36
    [*] predict with avgrank model
    precision=[0.21348315 0.22222222 0.2       ], recall=[0.40425532 0.03773585 0.20833333], f1=[0.27941176 0.06451613 0.20408163], accuracy=0.21
    [*] predict with dice model
    precision=[0.33802817 0.4375     0.28888889], recall=[0.5106383  0.26415094 0.27083333], f1=[0.40677966 0.32941176 0.27956989], accuracy=0.34
    [*] predict with lr model
    precision=[0.54545455 0.47368421 0.46753247], recall=[0.38297872 0.33962264 0.75      ], f1=[0.45      0.3956044 0.576    ], accuracy=0.49
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


    precision=[0.57575758 0.5        0.46753247], recall=[0.40425532 0.35849057 0.75      ], f1=[0.475      0.41758242 0.576     ], accuracy=0.50
    [*] predict with lsvc model
    precision=[0.55882353 0.55882353 0.475     ], recall=[0.40425532 0.35849057 0.79166667], f1=[0.4691358  0.43678161 0.59375   ], accuracy=0.51
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.36144578 0.57142857 0.43333333], recall=[0.63829787 0.37735849 0.27083333], f1=[0.46153846 0.45454545 0.33333333], accuracy=0.43
    [*] predict with rf model
    precision=[0.38554217 0.56097561 0.625     ], recall=[0.68085106 0.43396226 0.3125    ], f1=[0.49230769 0.4893617  0.41666667], accuracy=0.47
    [*] predict with lrbias model
    [LibLinear]precision=[0.55882353 0.48648649 0.46753247], recall=[0.40425532 0.33962264 0.75      ], f1=[0.4691358 0.4       0.576    ], accuracy=0.49
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.44615385 0.47619048 0.53658537], recall=[0.61702128 0.37735849 0.45833333], f1=[0.51785714 0.42105263 0.49438202], accuracy=0.48
    rec:['stage4', 657, 148, '+:48,0:53,-:47', 0.3581081081081081, 0.20945945945945946, 0.34459459459459457, 0.4864864864864865, 0.5, 0.5135135135135135, 0.42567567567567566, 0.47297297297297297, 0.49324324324324326, 0.4797297297297297]
    [*] predict with currank model
    precision=[0.     0.3625 0.    ], recall=[0. 1. 0.], f1=[0.         0.53211009 0.        ], accuracy=0.36
    [*] predict with avgrank model
    precision=[0.19565217 0.28571429 0.18518519], recall=[0.36       0.06896552 0.19230769], f1=[0.25352113 0.11111111 0.18867925], accuracy=0.20
    [*] predict with dice model
    precision=[0.32352941 0.26666667 0.35483871], recall=[0.44       0.13793103 0.42307692], f1=[0.37288136 0.18181818 0.38596491], accuracy=0.33
    [*] predict with lr model
    precision=[0.58823529 0.5        0.51851852], recall=[0.4        0.62068966 0.53846154], f1=[0.47619048 0.55384615 0.52830189], accuracy=0.53
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


    precision=[0.66666667 0.5        0.48275862], recall=[0.4        0.62068966 0.53846154], f1=[0.5        0.55384615 0.50909091], accuracy=0.53
    [*] predict with lsvc model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.625 0.5   0.5  ], recall=[0.4        0.62068966 0.53846154], f1=[0.48780488 0.55384615 0.51851852], accuracy=0.53
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.45       0.5483871  0.77777778], recall=[0.72       0.5862069  0.26923077], f1=[0.55384615 0.56666667 0.4       ], accuracy=0.53
    [*] predict with rf model
    precision=[0.4        0.58333333 0.625     ], recall=[0.64       0.48275862 0.38461538], f1=[0.49230769 0.52830189 0.47619048], accuracy=0.50
    [*] predict with lrbias model
    [LibLinear]precision=[0.61538462 0.47368421 0.48275862], recall=[0.32       0.62068966 0.53846154], f1=[0.42105263 0.53731343 0.50909091], accuracy=0.50
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.6        0.56666667 0.6       ], recall=[0.72       0.5862069  0.46153846], f1=[0.65454545 0.57627119 0.52173913], accuracy=0.59
    rec:['stage5', 725, 80, '+:26,0:29,-:25', 0.3625, 0.2, 0.325, 0.525, 0.525, 0.525, 0.525, 0.5, 0.5, 0.5875]
    [*] predict with currank model
    precision=[0.         0.34210526 0.        ], recall=[0. 1. 0.], f1=[0.         0.50980392 0.        ], accuracy=0.34
    [*] predict with avgrank model
    precision=[0.27272727 0.33333333 0.1       ], recall=[0.42857143 0.15384615 0.09090909], f1=[0.33333333 0.21052632 0.0952381 ], accuracy=0.24
    [*] predict with dice model
    precision=[0.2        0.3        0.30769231], recall=[0.21428571 0.23076923 0.36363636], f1=[0.20689655 0.26086957 0.33333333], accuracy=0.26
    [*] predict with lr model
    precision=[0.42857143 0.35       0.36363636], recall=[0.21428571 0.53846154 0.36363636], f1=[0.28571429 0.42424242 0.36363636], accuracy=0.37
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


    precision=[0.42857143 0.38888889 0.38461538], recall=[0.21428571 0.53846154 0.45454545], f1=[0.28571429 0.4516129  0.41666667], accuracy=0.39
    [*] predict with lsvc model
    precision=[0.5        0.47368421 0.38461538], recall=[0.21428571 0.69230769 0.45454545], f1=[0.3        0.5625     0.41666667], accuracy=0.45
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.33333333 0.33333333 0.5       ], recall=[0.14285714 0.76923077 0.09090909], f1=[0.2        0.46511628 0.15384615], accuracy=0.34
    [*] predict with rf model
    precision=[0.43478261 0.44444444 0.66666667], recall=[0.71428571 0.30769231 0.36363636], f1=[0.54054054 0.36363636 0.47058824], accuracy=0.47
    [*] predict with lrbias model
    [LibLinear]precision=[0.75       0.41666667 0.3       ], recall=[0.21428571 0.76923077 0.27272727], f1=[0.33333333 0.54054054 0.28571429], accuracy=0.42
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.58823529 0.5        0.4       ], recall=[0.71428571 0.23076923 0.54545455], f1=[0.64516129 0.31578947 0.46153846], accuracy=0.50
    rec:['stage6', 767, 38, '+:11,0:13,-:14', 0.34210526315789475, 0.23684210526315788, 0.2631578947368421, 0.3684210526315789, 0.39473684210526316, 0.4473684210526316, 0.34210526315789475, 0.47368421052631576, 0.42105263157894735, 0.5]
    [*] predict with currank model
    precision=[0.   0.25 0.  ], recall=[0. 1. 0.], f1=[0.  0.4 0. ], accuracy=0.25
    [*] predict with avgrank model
    precision=[0.3 0.  0. ], recall=[0.375 0.    0.   ], f1=[0.33333333 0.         0.        ], accuracy=0.19
    [*] predict with dice model
    precision=[0.42857143 0.2        0.25      ], recall=[0.375 0.25  0.25 ], f1=[0.4        0.22222222 0.25      ], accuracy=0.31
    [*] predict with lr model
    precision=[0.         0.16666667 0.2       ], recall=[0.   0.25 0.5 ], f1=[0.         0.2        0.28571429], accuracy=0.19
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


    precision=[1.         0.25       0.27272727], recall=[0.125 0.25  0.75 ], f1=[0.22222222 0.25       0.4       ], accuracy=0.31
    [*] predict with lsvc model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[1.         0.25       0.27272727], recall=[0.125 0.25  0.75 ], f1=[0.22222222 0.25       0.4       ], accuracy=0.31
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)


    precision=[0.   0.   0.25], recall=[0. 0. 1.], f1=[0.  0.  0.4], accuracy=0.25
    [*] predict with rf model
    precision=[0.55555556 0.2        1.        ], recall=[0.625 0.25  0.5  ], f1=[0.58823529 0.22222222 0.66666667], accuracy=0.50
    [*] predict with lrbias model
    [LibLinear]precision=[0.         0.16666667 0.2       ], recall=[0.   0.25 0.5 ], f1=[0.         0.2        0.28571429], accuracy=0.19
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


    precision=[0.5        0.28571429 0.66666667], recall=[0.375 0.5   0.5  ], f1=[0.42857143 0.36363636 0.57142857], accuracy=0.44
    rec:['stage7', 789, 16, '+:4,0:4,-:8', 0.25, 0.1875, 0.3125, 0.1875, 0.3125, 0.3125, 0.25, 0.5, 0.1875, 0.4375]



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
      <td>0.450920</td>
      <td>0.414110</td>
      <td>0.372699</td>
      <td>0.383436</td>
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
      <td>0.344294</td>
      <td>0.365571</td>
      <td>0.340426</td>
      <td>0.317215</td>
      <td>0.421663</td>
      <td>0.346228</td>
      <td>0.425532</td>
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
      <td>0.398438</td>
      <td>0.390625</td>
      <td>0.421875</td>
      <td>0.367188</td>
      <td>0.424479</td>
      <td>0.398438</td>
      <td>0.466146</td>
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
      <td>0.437984</td>
      <td>0.422481</td>
      <td>0.437984</td>
      <td>0.360465</td>
      <td>0.434109</td>
      <td>0.430233</td>
      <td>0.445736</td>
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
      <td>0.486486</td>
      <td>0.500000</td>
      <td>0.513514</td>
      <td>0.425676</td>
      <td>0.472973</td>
      <td>0.493243</td>
      <td>0.479730</td>
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
      <td>0.525000</td>
      <td>0.525000</td>
      <td>0.525000</td>
      <td>0.525000</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>0.587500</td>
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
      <td>0.447368</td>
      <td>0.342105</td>
      <td>0.473684</td>
      <td>0.421053</td>
      <td>0.500000</td>
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
      <td>0.187500</td>
      <td>0.312500</td>
      <td>0.312500</td>
      <td>0.250000</td>
      <td>0.500000</td>
      <td>0.187500</td>
      <td>0.437500</td>
    </tr>
  </tbody>
</table>
</div>




```python
#xgb max_tree_depth=6
df_stage
```


```python

```
