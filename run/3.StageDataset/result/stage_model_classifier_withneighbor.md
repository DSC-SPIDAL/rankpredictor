
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
suffix='-withneighbor'
stagedata = pd.read_csv('stage-2018%s.csv'%suffix)
stagedata.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 805 entries, 0 to 804
    Data columns (total 24 columns):
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
    prev_nb0_change_in_rank      805 non-null int64
    prev_nb1_change_in_rank      805 non-null int64
    prev_nb2_change_in_rank      805 non-null int64
    follow_nb0_change_in_rank    805 non-null int64
    follow_nb1_change_in_rank    805 non-null int64
    follow_nb2_change_in_rank    805 non-null int64
    dtypes: float64(5), int64(19)
    memory usage: 151.0 KB



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
      <th>change_in_rank</th>
      <th>change_in_rank_all</th>
      <th>rate_of_change</th>
      <th>rate_of_change_all</th>
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
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
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
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
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
      <td>-5</td>
      <td>-2.500000</td>
      <td>0</td>
      <td>0.000000</td>
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
      <td>3</td>
      <td>-0.666667</td>
      <td>8</td>
      <td>1.500000</td>
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
      <td>-4</td>
      <td>-1.500000</td>
      <td>-7</td>
      <td>-1.333333</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>-2</td>
      <td>-3</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
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
    precision=[0.61971831 0.22222222 0.47058824], recall=[0.73333333 0.125      0.42105263], f1=[0.67175573 0.16       0.44444444], accuracy=0.54
    [*] predict with lrl1 model
    precision=[0.625      0.22222222 0.48484848], recall=[0.75       0.125      0.42105263], f1=[0.68181818 0.16       0.45070423], accuracy=0.55
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


    precision=[0.609375   0.15384615 0.43243243], recall=[0.65       0.125      0.42105263], f1=[0.62903226 0.13793103 0.42666667], accuracy=0.50
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.6440678  0.21428571 0.69230769], recall=[0.63333333 0.5625     0.23684211], f1=[0.63865546 0.31034483 0.35294118], accuracy=0.49
    [*] predict with rf model
    precision=[0.59375    0.26666667 0.48571429], recall=[0.63333333 0.25       0.44736842], f1=[0.61290323 0.25806452 0.46575342], accuracy=0.52
    [*] predict with lrbias model
    [LibLinear]precision=[0.62857143 0.2        0.47058824], recall=[0.73333333 0.125      0.42105263], f1=[0.67692308 0.15384615 0.44444444], accuracy=0.54
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.54901961 0.13333333 0.54545455], recall=[0.46666667 0.25       0.47368421], f1=[0.5045045  0.17391304 0.50704225], accuracy=0.44
    rec:['Phoenix', 691, 114, '+:38,0:16,-:60', 0.14035087719298245, 0.32456140350877194, 0.3684210526315789, 0.543859649122807, 0.5526315789473685, 0.5, 0.49122807017543857, 0.5175438596491229, 0.543859649122807, 0.43859649122807015]
    Testset = Indy500
    [*] predict with currank model
    precision=[0.         0.20888889 0.        ], recall=[0. 1. 0.], f1=[0.         0.34558824 0.        ], accuracy=0.21
    [*] predict with avgrank model
    precision=[0.4017094  0.24528302 0.23636364], recall=[0.48958333 0.27659574 0.15853659], f1=[0.44131455 0.26       0.18978102], accuracy=0.32
    [*] predict with dice model
    precision=[0.47619048 0.26315789 0.38095238], recall=[0.52083333 0.31914894 0.29268293], f1=[0.49751244 0.28846154 0.33103448], accuracy=0.40
    [*] predict with lr model
    precision=[0.57246377 0.30769231 0.58108108], recall=[0.82291667 0.08510638 0.52439024], f1=[0.67521368 0.13333333 0.55128205], accuracy=0.56
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


    precision=[0.56934307 0.33333333 0.57534247], recall=[0.8125     0.10638298 0.51219512], f1=[0.6695279  0.16129032 0.54193548], accuracy=0.56
    [*] predict with lsvc model
    precision=[0.56462585 0.30769231 0.58461538], recall=[0.86458333 0.08510638 0.46341463], f1=[0.68312757 0.13333333 0.5170068 ], accuracy=0.56
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.54545455 0.35714286 0.57352941], recall=[0.8125     0.10638298 0.47560976], f1=[0.65271967 0.16393443 0.52      ], accuracy=0.54
    [*] predict with rf model
    precision=[0.50649351 0.63636364 0.53333333], recall=[0.8125     0.14893617 0.3902439 ], f1=[0.624      0.24137931 0.45070423], accuracy=0.52
    [*] predict with lrbias model
    [LibLinear]precision=[0.57352941 0.33333333 0.58108108], recall=[0.8125     0.10638298 0.52439024], f1=[0.67241379 0.16129032 0.55128205], accuracy=0.56
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.55223881 0.34615385 0.55384615], recall=[0.77083333 0.19148936 0.43902439], f1=[0.64347826 0.24657534 0.48979592], accuracy=0.53
    rec:['Indy500', 580, 225, '+:82,0:47,-:96', 0.2088888888888889, 0.3244444444444444, 0.39555555555555555, 0.56, 0.5555555555555556, 0.5555555555555556, 0.5422222222222223, 0.52, 0.56, 0.5288888888888889]
    Testset = Texas
    [*] predict with currank model
    precision=[0.         0.26771654 0.        ], recall=[0. 1. 0.], f1=[0.         0.42236025 0.        ], accuracy=0.27
    [*] predict with avgrank model
    precision=[0.37037037 0.3255814  0.2       ], recall=[0.37037037 0.41176471 0.15384615], f1=[0.37037037 0.36363636 0.17391304], accuracy=0.31
    [*] predict with dice model
    precision=[0.30357143 0.23333333 0.31707317], recall=[0.31481481 0.20588235 0.33333333], f1=[0.30909091 0.21875    0.325     ], accuracy=0.29
    [*] predict with lr model
    precision=[0.60714286 0.5        0.53061224], recall=[0.62962963 0.32352941 0.66666667], f1=[0.61818182 0.39285714 0.59090909], accuracy=0.56
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


    precision=[0.60655738 0.5        0.56818182], recall=[0.68518519 0.32352941 0.64102564], f1=[0.64347826 0.39285714 0.60240964], accuracy=0.57
    [*] predict with lsvc model
    precision=[0.609375   0.54166667 0.61538462], recall=[0.72222222 0.38235294 0.61538462], f1=[0.66101695 0.44827586 0.61538462], accuracy=0.60
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.5915493  0.40909091 0.58823529], recall=[0.77777778 0.26470588 0.51282051], f1=[0.672      0.32142857 0.54794521], accuracy=0.56
    [*] predict with rf model
    precision=[0.66       0.36363636 0.41818182], recall=[0.61111111 0.23529412 0.58974359], f1=[0.63461538 0.28571429 0.4893617 ], accuracy=0.50
    [*] predict with lrbias model
    [LibLinear]precision=[0.60714286 0.54166667 0.55319149], recall=[0.62962963 0.38235294 0.66666667], f1=[0.61818182 0.44827586 0.60465116], accuracy=0.57
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.5952381  0.27777778 0.40298507], recall=[0.46296296 0.14705882 0.69230769], f1=[0.52083333 0.19230769 0.50943396], accuracy=0.45
    rec:['Texas', 678, 127, '+:39,0:34,-:54', 0.2677165354330709, 0.31496062992125984, 0.29133858267716534, 0.5590551181102362, 0.5748031496062992, 0.5984251968503937, 0.5590551181102362, 0.5039370078740157, 0.5748031496062992, 0.44881889763779526]
    Testset = Iowa
    [*] predict with currank model
    precision=[0.         0.25688073 0.        ], recall=[0. 1. 0.], f1=[0.         0.40875912 0.        ], accuracy=0.26
    [*] predict with avgrank model
    precision=[0.30232558 0.20689655 0.35135135], recall=[0.33333333 0.21428571 0.30952381], f1=[0.31707317 0.21052632 0.32911392], accuracy=0.29
    [*] predict with dice model
    precision=[0.25531915 0.18518519 0.31428571], recall=[0.30769231 0.17857143 0.26190476], f1=[0.27906977 0.18181818 0.28571429], accuracy=0.26
    [*] predict with lr model
    precision=[0.38095238 0.17647059 0.66666667], recall=[0.61538462 0.21428571 0.19047619], f1=[0.47058824 0.19354839 0.2962963 ], accuracy=0.35
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


    precision=[0.38461538 0.16129032 0.69230769], recall=[0.64102564 0.17857143 0.21428571], f1=[0.48076923 0.16949153 0.32727273], accuracy=0.36
    [*] predict with lsvc model
    precision=[0.38028169 0.14814815 0.72727273], recall=[0.69230769 0.14285714 0.19047619], f1=[0.49090909 0.14545455 0.30188679], accuracy=0.36
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.40425532 0.25       0.48148148], recall=[0.48717949 0.07142857 0.61904762], f1=[0.44186047 0.11111111 0.54166667], accuracy=0.43
    [*] predict with rf model
    precision=[0.37681159 0.24       0.73333333], recall=[0.66666667 0.21428571 0.26190476], f1=[0.48148148 0.22641509 0.38596491], accuracy=0.39
    [*] predict with lrbias model
    [LibLinear]precision=[0.38461538 0.1875     0.66666667], recall=[0.64102564 0.21428571 0.19047619], f1=[0.48076923 0.2        0.2962963 ], accuracy=0.36
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.31746032 0.13793103 0.52941176], recall=[0.51282051 0.14285714 0.21428571], f1=[0.39215686 0.14035088 0.30508475], accuracy=0.30
    rec:['Iowa', 696, 109, '+:42,0:28,-:39', 0.25688073394495414, 0.29357798165137616, 0.25688073394495414, 0.3486238532110092, 0.3577981651376147, 0.3577981651376147, 0.43119266055045874, 0.3944954128440367, 0.3577981651376147, 0.30275229357798167]
    Testset = Pocono
    [*] predict with currank model
    precision=[0.         0.48412698 0.        ], recall=[0. 1. 0.], f1=[0.         0.65240642 0.        ], accuracy=0.48
    [*] predict with avgrank model
    precision=[0.265625   0.44117647 0.10714286], recall=[0.47222222 0.24590164 0.10344828], f1=[0.34       0.31578947 0.10526316], accuracy=0.28
    [*] predict with dice model
    precision=[0.23214286 0.34615385 0.18181818], recall=[0.36111111 0.14754098 0.27586207], f1=[0.2826087  0.20689655 0.21917808], accuracy=0.24
    [*] predict with lr model
    precision=[0.41176471 0.58333333 0.28571429], recall=[0.58333333 0.1147541  0.62068966], f1=[0.48275862 0.19178082 0.39130435], accuracy=0.37
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


    precision=[0.41176471 0.75       0.28358209], recall=[0.58333333 0.09836066 0.65517241], f1=[0.48275862 0.17391304 0.39583333], accuracy=0.37
    [*] predict with lsvc model
    precision=[0.41176471 0.66666667 0.27536232], recall=[0.58333333 0.06557377 0.65517241], f1=[0.48275862 0.11940299 0.3877551 ], accuracy=0.35
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.         0.48305085 0.33333333], recall=[0.         0.93442623 0.06896552], f1=[0.         0.63687151 0.11428571], accuracy=0.47
    [*] predict with rf model
    precision=[0.37037037 0.66666667 0.27272727], recall=[0.55555556 0.06557377 0.62068966], f1=[0.44444444 0.11940299 0.37894737], accuracy=0.33
    [*] predict with lrbias model
    [LibLinear]precision=[0.41176471 0.63636364 0.265625  ], recall=[0.58333333 0.1147541  0.5862069 ], f1=[0.48275862 0.19444444 0.3655914 ], accuracy=0.36
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.46666667 0.62857143 0.32608696], recall=[0.58333333 0.36065574 0.51724138], f1=[0.51851852 0.45833333 0.4       ], accuracy=0.46
    rec:['Pocono', 679, 126, '+:29,0:61,-:36', 0.48412698412698413, 0.2777777777777778, 0.23809523809523808, 0.36507936507936506, 0.36507936507936506, 0.3492063492063492, 0.46825396825396826, 0.3333333333333333, 0.35714285714285715, 0.4603174603174603]
    Testset = Gateway
    [*] predict with currank model
    precision=[0.         0.26923077 0.        ], recall=[0. 1. 0.], f1=[0.         0.42424242 0.        ], accuracy=0.27
    [*] predict with avgrank model
    precision=[0.36956522 0.29411765 0.29166667], recall=[0.4047619  0.35714286 0.20588235], f1=[0.38636364 0.32258065 0.24137931], accuracy=0.33
    [*] predict with dice model
    precision=[0.34782609 0.13043478 0.2       ], recall=[0.38095238 0.10714286 0.20588235], f1=[0.36363636 0.11764706 0.20289855], accuracy=0.25
    [*] predict with lr model
    precision=[0.8125     0.38235294 0.55      ], recall=[0.30952381 0.92857143 0.32352941], f1=[0.44827586 0.54166667 0.40740741], accuracy=0.48
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


    precision=[0.85714286 0.35135135 0.4375    ], recall=[0.28571429 0.92857143 0.20588235], f1=[0.42857143 0.50980392 0.28      ], accuracy=0.43
    [*] predict with lsvc model
    precision=[0.8        0.35135135 0.46666667], recall=[0.28571429 0.92857143 0.20588235], f1=[0.42105263 0.50980392 0.28571429], accuracy=0.43
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.59090909 0.30263158 0.66666667], recall=[0.30952381 0.82142857 0.11764706], f1=[0.40625    0.44230769 0.2       ], accuracy=0.38
    [*] predict with rf model
    precision=[0.58333333 0.3902439  0.48148148], recall=[0.5        0.57142857 0.38235294], f1=[0.53846154 0.46376812 0.42622951], accuracy=0.48
    [*] predict with lrbias model
    [LibLinear]precision=[0.8        0.36619718 0.5       ], recall=[0.28571429 0.92857143 0.26470588], f1=[0.42105263 0.52525253 0.34615385], accuracy=0.45
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.43333333 0.4047619  0.375     ], recall=[0.30952381 0.60714286 0.35294118], f1=[0.36111111 0.48571429 0.36363636], accuracy=0.40
    rec:['Gateway', 701, 104, '+:34,0:28,-:42', 0.2692307692307692, 0.3269230769230769, 0.25, 0.4807692307692308, 0.4326923076923077, 0.4326923076923077, 0.38461538461538464, 0.4807692307692308, 0.4519230769230769, 0.40384615384615385]



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
      <td>0.500000</td>
      <td>0.491228</td>
      <td>0.517544</td>
      <td>0.543860</td>
      <td>0.438596</td>
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
      <td>0.555556</td>
      <td>0.555556</td>
      <td>0.542222</td>
      <td>0.520000</td>
      <td>0.560000</td>
      <td>0.528889</td>
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
      <td>0.559055</td>
      <td>0.574803</td>
      <td>0.598425</td>
      <td>0.559055</td>
      <td>0.503937</td>
      <td>0.574803</td>
      <td>0.448819</td>
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
      <td>0.348624</td>
      <td>0.357798</td>
      <td>0.357798</td>
      <td>0.431193</td>
      <td>0.394495</td>
      <td>0.357798</td>
      <td>0.302752</td>
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
      <td>0.365079</td>
      <td>0.365079</td>
      <td>0.349206</td>
      <td>0.468254</td>
      <td>0.333333</td>
      <td>0.357143</td>
      <td>0.460317</td>
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
      <td>0.480769</td>
      <td>0.432692</td>
      <td>0.432692</td>
      <td>0.384615</td>
      <td>0.480769</td>
      <td>0.451923</td>
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
    precision=[0.49867374 0.37209302 0.4       ], recall=[0.71212121 0.47904192 0.10859729], f1=[0.58658346 0.41884817 0.17081851], accuracy=0.45
    [*] predict with lsvc model
    precision=[0.51734104 0.35341365 0.36842105], recall=[0.6780303  0.52694611 0.09502262], f1=[0.58688525 0.42307692 0.15107914], accuracy=0.44
    [*] predict with lsvcl2 model
    precision=[0.54098361 0.24703892 0.        ], recall=[0.125     0.8742515 0.       ], f1=[0.20307692 0.38522427 0.        ], accuracy=0.27
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
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)


    precision=[0.46644295 0.33333333 0.35947712], recall=[0.52651515 0.4011976  0.24886878], f1=[0.49466192 0.36413043 0.29411765], accuracy=0.40
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
    rec:['stage0', 153, 652, '+:221,0:167,-:264', 0.2561349693251534, 0.3128834355828221, 0.36349693251533743, 0.4647239263803681, 0.44785276073619634, 0.44171779141104295, 0.2745398773006135, 0.4003067484662577, 0.44785276073619634, 0.3558282208588957]
    [*] predict with currank model
    precision=[0.         0.26305609 0.        ], recall=[0. 1. 0.], f1=[0.         0.41653905 0.        ], accuracy=0.26
    [*] predict with avgrank model
    precision=[0.34674923 0.24242424 0.24223602], recall=[0.57435897 0.05882353 0.20967742], f1=[0.43243243 0.09467456 0.22478386], accuracy=0.31
    [*] predict with dice model
    precision=[0.39312977 0.22881356 0.37956204], recall=[0.52820513 0.19852941 0.27956989], f1=[0.45076586 0.21259843 0.32198142], accuracy=0.35
    [*] predict with lr model
    precision=[0.42384106 0.35772358 0.57608696], recall=[0.65641026 0.32352941 0.28494624], f1=[0.51509054 0.33976834 0.38129496], accuracy=0.44
    [*] predict with lrl1 model
    precision=[0.44202899 0.3671875  0.55752212], recall=[0.62564103 0.34558824 0.33870968], f1=[0.51804671 0.35606061 0.42140468], accuracy=0.45
    [*] predict with lsvc model
    precision=[0.45289855 0.3515625  0.5840708 ], recall=[0.64102564 0.33088235 0.35483871], f1=[0.53078556 0.34090909 0.44147157], accuracy=0.46
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


    precision=[0.42222222 0.38461538 0.58095238], recall=[0.77948718 0.14705882 0.32795699], f1=[0.54774775 0.21276596 0.41924399], accuracy=0.45
    [*] predict with rf model
    precision=[0.44623656 0.4        0.5       ], recall=[0.85128205 0.10294118 0.29569892], f1=[0.58553792 0.16374269 0.37162162], accuracy=0.45
    [*] predict with lrbias model
    [LibLinear]precision=[0.42244224 0.3553719  0.56989247], recall=[0.65641026 0.31617647 0.28494624], f1=[0.51405622 0.33463035 0.37992832], accuracy=0.43
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.48375451 0.33663366 0.51079137], recall=[0.68717949 0.25       0.38172043], f1=[0.56779661 0.28691983 0.43692308], accuracy=0.46
    rec:['stage1', 288, 517, '+:186,0:136,-:195', 0.26305609284332687, 0.30754352030947774, 0.3520309477756286, 0.43520309477756286, 0.44874274661508706, 0.4564796905222437, 0.4506769825918762, 0.45454545454545453, 0.4332688588007737, 0.4622823984526112]
    [*] predict with currank model
    precision=[0.         0.29166667 0.        ], recall=[0. 1. 0.], f1=[0.        0.4516129 0.       ], accuracy=0.29
    [*] predict with avgrank model
    precision=[0.28630705 0.28571429 0.21311475], recall=[0.52272727 0.05357143 0.18571429], f1=[0.36997319 0.09022556 0.19847328], accuracy=0.26
    [*] predict with dice model
    precision=[0.3627451  0.17567568 0.33018868], recall=[0.56060606 0.11607143 0.25      ], f1=[0.44047619 0.13978495 0.28455285], accuracy=0.32
    [*] predict with lr model
    precision=[0.41810345 0.4        0.48591549], recall=[0.73484848 0.03571429 0.49285714], f1=[0.53296703 0.06557377 0.4893617 ], accuracy=0.44
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


    precision=[0.4015444  0.4        0.49090909], recall=[0.78787879 0.05357143 0.38571429], f1=[0.53196931 0.09448819 0.432     ], accuracy=0.43
    [*] predict with lsvc model
    precision=[0.43243243 0.4        0.49342105], recall=[0.72727273 0.03571429 0.53571429], f1=[0.54237288 0.06557377 0.51369863], accuracy=0.46
    [*] predict with lsvcl2 model
    precision=[0.43262411 0.         0.42436975], recall=[0.46212121 0.         0.72142857], f1=[0.44688645 0.         0.53439153], accuracy=0.42
    [*] predict with rf model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.41245136 0.43478261 0.53846154], recall=[0.8030303  0.08928571 0.4       ], f1=[0.54498715 0.14814815 0.45901639], accuracy=0.45
    [*] predict with lrbias model
    [LibLinear]precision=[0.4185022  0.38461538 0.48611111], recall=[0.71969697 0.04464286 0.5       ], f1=[0.52924791 0.08       0.49295775], accuracy=0.44
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.41422594 0.40425532 0.5       ], recall=[0.75       0.16964286 0.35      ], f1=[0.53369272 0.23899371 0.41176471], accuracy=0.43
    rec:['stage2', 421, 384, '+:140,0:112,-:132', 0.2916666666666667, 0.2630208333333333, 0.3177083333333333, 0.4427083333333333, 0.4270833333333333, 0.4557291666666667, 0.421875, 0.4479166666666667, 0.4427083333333333, 0.4348958333333333]
    [*] predict with currank model
    precision=[0.         0.34496124 0.        ], recall=[0. 1. 0.], f1=[0.        0.5129683 0.       ], accuracy=0.34
    [*] predict with avgrank model
    precision=[0.22929936 0.29411765 0.21428571], recall=[0.46153846 0.05617978 0.1978022 ], f1=[0.30638298 0.09433962 0.20571429], accuracy=0.23
    [*] predict with dice model
    precision=[0.28985507 0.4375     0.34722222], recall=[0.51282051 0.23595506 0.27472527], f1=[0.37037037 0.30656934 0.30674847], accuracy=0.33
    [*] predict with lr model
    precision=[0.42176871 0.16666667 0.54545455], recall=[0.79487179 0.02247191 0.59340659], f1=[0.55111111 0.03960396 0.56842105], accuracy=0.46
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


    precision=[0.39156627 0.18181818 0.58024691], recall=[0.83333333 0.02247191 0.51648352], f1=[0.53278689 0.04       0.54651163], accuracy=0.44
    [*] predict with lsvc model
    precision=[0.41721854 0.2        0.51546392], recall=[0.80769231 0.02247191 0.54945055], f1=[0.55021834 0.04040404 0.53191489], accuracy=0.45
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.35911602 0.2        0.54166667], recall=[0.83333333 0.01123596 0.42857143], f1=[0.5019305  0.0212766  0.47852761], accuracy=0.41
    [*] predict with rf model
    precision=[0.39344262 0.63636364 0.59375   ], recall=[0.92307692 0.07865169 0.41758242], f1=[0.55172414 0.14       0.49032258], accuracy=0.45
    [*] predict with lrbias model
    [LibLinear]precision=[0.42281879 0.16666667 0.54639175], recall=[0.80769231 0.02247191 0.58241758], f1=[0.55506608 0.03960396 0.56382979], accuracy=0.46
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.38787879 0.5        0.55223881], recall=[0.82051282 0.14606742 0.40659341], f1=[0.52674897 0.22608696 0.46835443], accuracy=0.44
    rec:['stage3', 547, 258, '+:91,0:89,-:78', 0.3449612403100775, 0.22868217054263565, 0.3333333333333333, 0.4573643410852713, 0.4418604651162791, 0.44573643410852715, 0.4069767441860465, 0.45348837209302323, 0.4573643410852713, 0.4418604651162791]
    [*] predict with currank model
    precision=[0.         0.35810811 0.        ], recall=[0. 1. 0.], f1=[0.         0.52736318 0.        ], accuracy=0.36
    [*] predict with avgrank model
    precision=[0.21348315 0.22222222 0.2       ], recall=[0.40425532 0.03773585 0.20833333], f1=[0.27941176 0.06451613 0.20408163], accuracy=0.21
    [*] predict with dice model
    precision=[0.33802817 0.4375     0.28888889], recall=[0.5106383  0.26415094 0.27083333], f1=[0.40677966 0.32941176 0.27956989], accuracy=0.34
    [*] predict with lr model
    precision=[0.57142857 0.51351351 0.46052632], recall=[0.42553191 0.35849057 0.72916667], f1=[0.48780488 0.42222222 0.56451613], accuracy=0.50
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


    precision=[0.57142857 0.51428571 0.46153846], recall=[0.42553191 0.33962264 0.75      ], f1=[0.48780488 0.40909091 0.57142857], accuracy=0.50
    [*] predict with lsvc model
    precision=[0.55555556 0.43333333 0.43902439], recall=[0.42553191 0.24528302 0.75      ], f1=[0.48192771 0.31325301 0.55384615], accuracy=0.47
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.37681159 0.52380952 0.44827586], recall=[0.55319149 0.20754717 0.54166667], f1=[0.44827586 0.2972973  0.49056604], accuracy=0.43
    [*] predict with rf model
    precision=[0.38461538 0.55555556 0.55882353], recall=[0.63829787 0.37735849 0.39583333], f1=[0.48       0.4494382  0.46341463], accuracy=0.47
    [*] predict with lrbias model
    [LibLinear]precision=[0.58333333 0.52777778 0.46052632], recall=[0.44680851 0.35849057 0.72916667], f1=[0.5060241  0.42696629 0.56451613], accuracy=0.51
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.43396226 0.44897959 0.45652174], recall=[0.4893617  0.41509434 0.4375    ], f1=[0.46       0.43137255 0.44680851], accuracy=0.45
    rec:['stage4', 657, 148, '+:48,0:53,-:47', 0.3581081081081081, 0.20945945945945946, 0.34459459459459457, 0.5, 0.5, 0.46621621621621623, 0.42567567567567566, 0.46621621621621623, 0.5067567567567568, 0.44594594594594594]
    [*] predict with currank model
    precision=[0.     0.3625 0.    ], recall=[0. 1. 0.], f1=[0.         0.53211009 0.        ], accuracy=0.36
    [*] predict with avgrank model
    precision=[0.19565217 0.28571429 0.18518519], recall=[0.36       0.06896552 0.19230769], f1=[0.25352113 0.11111111 0.18867925], accuracy=0.20
    [*] predict with dice model
    precision=[0.32352941 0.26666667 0.35483871], recall=[0.44       0.13793103 0.42307692], f1=[0.37288136 0.18181818 0.38596491], accuracy=0.33
    [*] predict with lr model
    precision=[0.69230769 0.52083333 0.63157895], recall=[0.36       0.86206897 0.46153846], f1=[0.47368421 0.64935065 0.53333333], accuracy=0.57
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


    precision=[0.64285714 0.54347826 0.65      ], recall=[0.36       0.86206897 0.5       ], f1=[0.46153846 0.66666667 0.56521739], accuracy=0.59
    [*] predict with lsvc model
    precision=[0.66666667 0.5        0.56521739], recall=[0.4        0.72413793 0.5       ], f1=[0.5        0.5915493  0.53061224], accuracy=0.55
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.66666667 0.5106383  0.54166667], recall=[0.24       0.82758621 0.5       ], f1=[0.35294118 0.63157895 0.52      ], accuracy=0.54
    [*] predict with rf model
    precision=[0.51162791 0.68421053 0.61111111], recall=[0.88       0.44827586 0.42307692], f1=[0.64705882 0.54166667 0.5       ], accuracy=0.57
    [*] predict with lrbias model
    [LibLinear]precision=[0.69230769 0.53191489 0.65      ], recall=[0.36       0.86206897 0.5       ], f1=[0.47368421 0.65789474 0.56521739], accuracy=0.59
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.62068966 0.55555556 0.66666667], recall=[0.72       0.51724138 0.61538462], f1=[0.66666667 0.53571429 0.64      ], accuracy=0.61
    rec:['stage5', 725, 80, '+:26,0:29,-:25', 0.3625, 0.2, 0.325, 0.575, 0.5875, 0.55, 0.5375, 0.575, 0.5875, 0.6125]
    [*] predict with currank model
    precision=[0.         0.34210526 0.        ], recall=[0. 1. 0.], f1=[0.         0.50980392 0.        ], accuracy=0.34
    [*] predict with avgrank model
    precision=[0.27272727 0.33333333 0.1       ], recall=[0.42857143 0.15384615 0.09090909], f1=[0.33333333 0.21052632 0.0952381 ], accuracy=0.24
    [*] predict with dice model
    precision=[0.2        0.3        0.30769231], recall=[0.21428571 0.23076923 0.36363636], f1=[0.20689655 0.26086957 0.33333333], accuracy=0.26
    [*] predict with lr model
    precision=[0.66666667 0.40740741 0.4       ], recall=[0.28571429 0.84615385 0.18181818], f1=[0.4  0.55 0.25], accuracy=0.45
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


    precision=[0.66666667 0.39285714 0.5       ], recall=[0.28571429 0.84615385 0.18181818], f1=[0.4        0.53658537 0.26666667], accuracy=0.45
    [*] predict with lsvc model
    precision=[0.8       0.4137931 0.5      ], recall=[0.28571429 0.92307692 0.18181818], f1=[0.42105263 0.57142857 0.26666667], accuracy=0.47
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)


    precision=[0.         0.35135135 1.        ], recall=[0.         1.         0.09090909], f1=[0.         0.52       0.16666667], accuracy=0.37
    [*] predict with rf model
    precision=[0.48       0.5        0.57142857], recall=[0.85714286 0.23076923 0.36363636], f1=[0.61538462 0.31578947 0.44444444], accuracy=0.50
    [*] predict with lrbias model
    [LibLinear]precision=[0.8        0.42857143 0.4       ], recall=[0.28571429 0.92307692 0.18181818], f1=[0.42105263 0.58536585 0.25      ], accuracy=0.47
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.57894737 0.5        0.46153846], recall=[0.78571429 0.23076923 0.54545455], f1=[0.66666667 0.31578947 0.5       ], accuracy=0.53
    rec:['stage6', 767, 38, '+:11,0:13,-:14', 0.34210526315789475, 0.23684210526315788, 0.2631578947368421, 0.4473684210526316, 0.4473684210526316, 0.47368421052631576, 0.3684210526315789, 0.5, 0.47368421052631576, 0.5263157894736842]
    [*] predict with currank model
    precision=[0.   0.25 0.  ], recall=[0. 1. 0.], f1=[0.  0.4 0. ], accuracy=0.25
    [*] predict with avgrank model
    precision=[0.3 0.  0. ], recall=[0.375 0.    0.   ], f1=[0.33333333 0.         0.        ], accuracy=0.19
    [*] predict with dice model
    precision=[0.42857143 0.2        0.25      ], recall=[0.375 0.25  0.25 ], f1=[0.4        0.22222222 0.25      ], accuracy=0.31
    [*] predict with lr model
    precision=[1.  0.3 0. ], recall=[0.25 0.75 0.  ], f1=[0.4        0.42857143 0.        ], accuracy=0.31
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


    precision=[1.  0.3 0. ], recall=[0.375 0.75  0.   ], f1=[0.54545455 0.42857143 0.        ], accuracy=0.38
    [*] predict with lsvc model
    precision=[1.   0.25 0.  ], recall=[0.25 0.75 0.  ], f1=[0.4   0.375 0.   ], accuracy=0.31
    [*] predict with lsvcl2 model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)


    precision=[0.   0.25 0.  ], recall=[0. 1. 0.], f1=[0.  0.4 0. ], accuracy=0.25
    [*] predict with rf model
    precision=[0.7 0.5 1. ], recall=[0.875 0.5   0.5  ], f1=[0.77777778 0.5        0.66666667], accuracy=0.69
    [*] predict with lrbias model
    [LibLinear]precision=[1.         0.27272727 0.        ], recall=[0.25 0.75 0.  ], f1=[0.4 0.4 0. ], accuracy=0.31
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1300: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 48.
      " = {}.".format(effective_n_jobs(self.n_jobs)))


    precision=[0.71428571 0.5        0.6       ], recall=[0.625 0.5   0.75 ], f1=[0.66666667 0.5        0.66666667], accuracy=0.62
    rec:['stage7', 789, 16, '+:4,0:4,-:8', 0.25, 0.1875, 0.3125, 0.3125, 0.375, 0.3125, 0.25, 0.6875, 0.3125, 0.625]



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
      <td>0.447853</td>
      <td>0.441718</td>
      <td>0.274540</td>
      <td>0.400307</td>
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
      <td>0.435203</td>
      <td>0.448743</td>
      <td>0.456480</td>
      <td>0.450677</td>
      <td>0.454545</td>
      <td>0.433269</td>
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
      <td>0.442708</td>
      <td>0.427083</td>
      <td>0.455729</td>
      <td>0.421875</td>
      <td>0.447917</td>
      <td>0.442708</td>
      <td>0.434896</td>
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
      <td>0.441860</td>
      <td>0.445736</td>
      <td>0.406977</td>
      <td>0.453488</td>
      <td>0.457364</td>
      <td>0.441860</td>
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
      <td>0.500000</td>
      <td>0.500000</td>
      <td>0.466216</td>
      <td>0.425676</td>
      <td>0.466216</td>
      <td>0.506757</td>
      <td>0.445946</td>
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
      <td>0.550000</td>
      <td>0.537500</td>
      <td>0.575000</td>
      <td>0.587500</td>
      <td>0.612500</td>
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
      <td>0.447368</td>
      <td>0.447368</td>
      <td>0.473684</td>
      <td>0.368421</td>
      <td>0.500000</td>
      <td>0.473684</td>
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
      <td>0.312500</td>
      <td>0.375000</td>
      <td>0.312500</td>
      <td>0.250000</td>
      <td>0.687500</td>
      <td>0.312500</td>
      <td>0.625000</td>
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
      <td>0.447853</td>
      <td>0.441718</td>
      <td>0.274540</td>
      <td>0.400307</td>
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
      <td>0.435203</td>
      <td>0.448743</td>
      <td>0.456480</td>
      <td>0.450677</td>
      <td>0.454545</td>
      <td>0.433269</td>
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
      <td>0.442708</td>
      <td>0.427083</td>
      <td>0.455729</td>
      <td>0.421875</td>
      <td>0.447917</td>
      <td>0.442708</td>
      <td>0.434896</td>
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
      <td>0.441860</td>
      <td>0.445736</td>
      <td>0.406977</td>
      <td>0.453488</td>
      <td>0.457364</td>
      <td>0.441860</td>
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
      <td>0.500000</td>
      <td>0.500000</td>
      <td>0.466216</td>
      <td>0.425676</td>
      <td>0.466216</td>
      <td>0.506757</td>
      <td>0.445946</td>
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
      <td>0.550000</td>
      <td>0.537500</td>
      <td>0.575000</td>
      <td>0.587500</td>
      <td>0.612500</td>
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
      <td>0.447368</td>
      <td>0.447368</td>
      <td>0.473684</td>
      <td>0.368421</td>
      <td>0.500000</td>
      <td>0.473684</td>
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
      <td>0.312500</td>
      <td>0.375000</td>
      <td>0.312500</td>
      <td>0.250000</td>
      <td>0.687500</td>
      <td>0.312500</td>
      <td>0.625000</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
