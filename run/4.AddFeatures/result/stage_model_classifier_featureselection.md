
### stage_model_classifier_featureselection

prediction models of sign classifiers on stage dataset

data format:
    target , eventid ,    car_number,    stageid,     features...


```python
%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.feature_selection import RFE

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
#classifiers = ['currank','avgrank','dice','lr','lrl1','lsvc','lsvcl2','rf','lrbias','xgb']
classifiers = ['lr','lsvc','rf','xgb']
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

def classifier_model(name='lr', feature_num = 10):
    ### test learning models
    print('[*] predict with %s model'%name)
    clf = get_classifier(name)
    
    selector = RFE(clf, feature_num, step=1)
    
    features = selector.fit(train_x, train_y)

    pred_y = selector.predict(test_x)
    score = evaluate(test_y, pred_y)
    return score, features.support_
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

selFeatureNumber = 10
featureNames=['eventid','car_number','stageid',
             'firststage','pit_in_caution','start_position',
             'start_rank','start_rank_ratio','top_pack','bottom_pack',
             'average_rank','average_rank_all',
             'change_in_rank','change_in_rank_all','rate_of_change','rate_of_change_all',
             'laptime_green_mean_prev','laptime_green_std_prev','laptime_green_mean_all','laptime_green_std_all', 
             'laptime_mean_prev','laptime_std_prev','laptime_mean_all','laptime_std_all', 
             'laps_prev','laps_after_last_pitstop','pittime_prev',     
             'prev_nb0_change_in_rank','prev_nb1_change_in_rank','prev_nb2_change_in_rank',
             'follow_nb0_change_in_rank','follow_nb1_change_in_rank','follow_nb2_change_in_rank']
    
eventsname = ['Phoenix','Indy500','Texas','Iowa','Pocono','Gateway']
events = set(stagedata['eventid'])

weights = np.zeros((len(featureNames)))

for eventid in events:
    print('Testset = %s'%eventsname[eventid])
    
    train, test, train_x, train_y, test_x, test_y = split_by_eventid(stagedata, eventid)
    test_distribution = '+:%d,0:%d,-:%d'%(np.sum(test_y>0),np.sum(test_y==0),np.sum(test_y<0))
    #print('Testset by stageid= %s, trainsize=%d, testsize=%d, dist=%s'%
    #      (stageid, train_x.shape[0], test_x.shape[0], test_distribution))
    
    #record
    rec = [eventsname[eventid],train_x.shape[0],test_x.shape[0],test_distribution]
    
    acc = [0 for x in range(len(classifiers))]
    #features = np.array((len(classifiers), selFeatureNumber))
    features =[0 for x in range(len(classifiers))]
    
    for idx, clf in enumerate(classifiers):
        acc[idx], features[idx] = classifier_model(clf, feature_num= selFeatureNumber)
        fnames = [featureNames[id] if val else '' for id,val in enumerate(features[idx])]
        print('clf:%d :%s',idx, ','.join(fnames))
        
        #add to feature weights
        for id,val in enumerate(features[idx]):
            if val:
                weights[id] += 1

    rec.extend(acc)
    print('rec:%s'%rec)
    
    
    #new df
    df = pd.DataFrame([rec],columns=cols)
    retdf = pd.concat([retdf, df])        

#weights analysis
idx = np.argsort(-weights)
fnames = [featureNames[id] for id in idx]
print('feature weights:',','.join(fnames))
    
    
retdf.to_csv('crossvalid_stagedata_splitbyevent%s.csv'%suffix)
df_event = retdf
```

    cols:['runid', 'trainsize', 'testsize', 'testdistribution', 'lr', 'lsvc', 'rf', 'xgb']
    Testset = Phoenix
    [*] predict with lr model


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


    precision=[0.6031746  0.         0.44680851], recall=[0.63333333 0.         0.55263158], f1=[0.61788618 0.         0.49411765], accuracy=0.52
    clf:%d :%s 0 eventid,,stageid,firststage,pit_in_caution,,start_rank,start_rank_ratio,top_pack,bottom_pack,average_rank,,,,,,,,,laptime_green_std_all,,,,,,,,,,,,,
    [*] predict with lsvc model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.6        0.3        0.46153846], recall=[0.65       0.1875     0.47368421], f1=[0.624      0.23076923 0.46753247], accuracy=0.53
    clf:%d :%s 1 ,,stageid,firststage,pit_in_caution,,start_rank,start_rank_ratio,top_pack,bottom_pack,average_rank,average_rank_all,,,,,,,,laptime_green_std_all,,,,,,,,,,,,,
    [*] predict with rf model
    precision=[0.6440678 0.25      0.6      ], recall=[0.63333333 0.3125     0.55263158], f1=[0.63865546 0.27777778 0.57534247], accuracy=0.56
    clf:%d :%s 2 ,car_number,,,,,start_rank,start_rank_ratio,,,average_rank,average_rank_all,,,,,,,laptime_green_mean_all,,,laptime_std_prev,,laptime_std_all,,laps_after_last_pitstop,pittime_prev,,,,,,
    [*] predict with xgb model
    precision=[0.56896552 0.20833333 0.40625   ], recall=[0.55       0.3125     0.34210526], f1=[0.55932203 0.25       0.37142857], accuracy=0.45
    clf:%d :%s 3 ,,stageid,,pit_in_caution,,start_rank,,,,,,,,rate_of_change,rate_of_change_all,,,,,,,,,,,,,prev_nb1_change_in_rank,prev_nb2_change_in_rank,follow_nb0_change_in_rank,follow_nb1_change_in_rank,follow_nb2_change_in_rank
    rec:['Phoenix', 691, 114, '+:38,0:16,-:60', 0.5175438596491229, 0.5263157894736842, 0.5614035087719298, 0.4473684210526316]
    Testset = Indy500
    [*] predict with lr model


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


    precision=[0.5620438  0.22222222 0.59493671], recall=[0.80208333 0.04255319 0.57317073], f1=[0.66094421 0.07142857 0.58385093], accuracy=0.56
    clf:%d :%s 0 eventid,,stageid,firststage,pit_in_caution,,start_rank,start_rank_ratio,top_pack,bottom_pack,average_rank,,,,,rate_of_change_all,,,,,,,,,,,,,,,,,
    [*] predict with lsvc model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.55405405 0.53846154 0.625     ], recall=[0.85416667 0.14893617 0.48780488], f1=[0.67213115 0.23333333 0.54794521], accuracy=0.57
    clf:%d :%s 1 eventid,,stageid,firststage,pit_in_caution,,start_rank,start_rank_ratio,top_pack,bottom_pack,average_rank,,,,,,,,,laptime_green_std_all,,,,,,,,,,,,,
    [*] predict with rf model
    precision=[0.54054054 0.35294118 0.55813953], recall=[0.83333333 0.25531915 0.29268293], f1=[0.6557377 0.2962963 0.384    ], accuracy=0.52
    clf:%d :%s 2 ,car_number,,,,,start_rank,start_rank_ratio,,,average_rank,average_rank_all,,,,,,,laptime_green_mean_all,,laptime_mean_prev,,laptime_mean_all,laptime_std_all,,,pittime_prev,,,,,,
    [*] predict with xgb model
    precision=[0.51754386 0.31168831 0.5       ], recall=[0.61458333 0.5106383  0.20731707], f1=[0.56190476 0.38709677 0.29310345], accuracy=0.44
    clf:%d :%s 3 ,,,,pit_in_caution,,start_rank,,,,average_rank,,,,,rate_of_change_all,,,laptime_green_mean_all,,,,,,,,,prev_nb0_change_in_rank,,prev_nb2_change_in_rank,follow_nb0_change_in_rank,follow_nb1_change_in_rank,follow_nb2_change_in_rank
    rec:['Indy500', 580, 225, '+:82,0:47,-:96', 0.56, 0.5733333333333334, 0.5155555555555555, 0.4444444444444444]
    Testset = Texas
    [*] predict with lr model


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


    precision=[0.63380282 0.73333333 0.63414634], recall=[0.83333333 0.32352941 0.66666667], f1=[0.72       0.44897959 0.65      ], accuracy=0.65
    clf:%d :%s 0 eventid,,stageid,firststage,pit_in_caution,,start_rank,start_rank_ratio,top_pack,bottom_pack,average_rank,,,change_in_rank_all,,,,,,,,,,,,,,,,,,,
    [*] predict with lsvc model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.61764706 0.9        0.57142857], recall=[0.77777778 0.26470588 0.71794872], f1=[0.68852459 0.40909091 0.63636364], accuracy=0.62
    clf:%d :%s 1 eventid,,stageid,firststage,pit_in_caution,,start_rank,start_rank_ratio,top_pack,bottom_pack,average_rank,,,,,,,,,laptime_green_std_all,,,,,,,,,,,,,
    [*] predict with rf model
    precision=[0.58064516 0.38095238 0.43181818], recall=[0.66666667 0.23529412 0.48717949], f1=[0.62068966 0.29090909 0.45783133], accuracy=0.50
    clf:%d :%s 2 ,car_number,,,,,,start_rank_ratio,,,average_rank,average_rank_all,,,,,,,laptime_green_mean_all,,laptime_mean_prev,laptime_std_prev,,laptime_std_all,,laps_after_last_pitstop,pittime_prev,,,,,,
    [*] predict with xgb model
    precision=[0.41176471 0.19230769 0.33333333], recall=[0.12962963 0.14705882 0.71794872], f1=[0.1971831  0.16666667 0.45528455], accuracy=0.31
    clf:%d :%s 3 ,,stageid,,pit_in_caution,,,,,bottom_pack,,,,,rate_of_change,rate_of_change_all,,,laptime_green_mean_all,,laptime_mean_prev,,,laptime_std_all,,,,prev_nb0_change_in_rank,,,,,follow_nb2_change_in_rank
    rec:['Texas', 678, 127, '+:39,0:34,-:54', 0.6456692913385826, 0.6220472440944882, 0.49606299212598426, 0.31496062992125984]
    Testset = Iowa
    [*] predict with lr model


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


    precision=[0.4084507  0.16       0.61538462], recall=[0.74358974 0.14285714 0.19047619], f1=[0.52727273 0.1509434  0.29090909], accuracy=0.38
    clf:%d :%s 0 eventid,,stageid,firststage,pit_in_caution,,start_rank,start_rank_ratio,top_pack,bottom_pack,,average_rank_all,,change_in_rank_all,,,,,,,,,,,,,,,,,,,
    [*] predict with lsvc model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.39240506 0.15789474 0.63636364], recall=[0.79487179 0.10714286 0.16666667], f1=[0.52542373 0.12765957 0.26415094], accuracy=0.38
    clf:%d :%s 1 eventid,,stageid,firststage,pit_in_caution,,start_rank,,top_pack,bottom_pack,average_rank,average_rank_all,,change_in_rank_all,,,,,,,,,,,,,,,,,,,
    [*] predict with rf model
    precision=[0.35135135 0.09090909 0.54166667], recall=[0.66666667 0.03571429 0.30952381], f1=[0.46017699 0.05128205 0.39393939], accuracy=0.37
    clf:%d :%s 2 ,car_number,,,,,start_rank,start_rank_ratio,,,average_rank,average_rank_all,,,,,,,laptime_green_mean_all,,,laptime_std_prev,,laptime_std_all,laps_prev,,pittime_prev,,,,,,
    [*] predict with xgb model
    precision=[0.41025641 0.25       0.30434783], recall=[0.82051282 0.07142857 0.16666667], f1=[0.54700855 0.11111111 0.21538462], accuracy=0.38
    clf:%d :%s 3 ,,stageid,,pit_in_caution,,,,,,,,change_in_rank,,rate_of_change,rate_of_change_all,,,,,,,,,,,,prev_nb0_change_in_rank,prev_nb1_change_in_rank,prev_nb2_change_in_rank,,follow_nb1_change_in_rank,follow_nb2_change_in_rank
    rec:['Iowa', 696, 109, '+:42,0:28,-:39', 0.3761467889908257, 0.3761467889908257, 0.3669724770642202, 0.3761467889908257]
    Testset = Pocono
    [*] predict with lr model


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


    precision=[0.35       0.55555556 0.25      ], recall=[0.77777778 0.16393443 0.24137931], f1=[0.48275862 0.25316456 0.24561404], accuracy=0.36
    clf:%d :%s 0 ,,stageid,firststage,pit_in_caution,,start_rank,start_rank_ratio,top_pack,bottom_pack,average_rank,,,,,,,,,laptime_green_std_all,,,,laptime_std_all,,,,,,,,,
    [*] predict with lsvc model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.36486486 0.47619048 0.19354839], recall=[0.75       0.16393443 0.20689655], f1=[0.49090909 0.24390244 0.2       ], accuracy=0.34
    clf:%d :%s 1 eventid,,stageid,firststage,pit_in_caution,,start_rank,,top_pack,bottom_pack,,average_rank_all,,,,,,,,laptime_green_std_all,,,,laptime_std_all,,,,,,,,,
    [*] predict with rf model
    precision=[0.36538462 0.63333333 0.29545455], recall=[0.52777778 0.31147541 0.44827586], f1=[0.43181818 0.41758242 0.35616438], accuracy=0.40
    clf:%d :%s 2 ,car_number,,,,,start_rank,start_rank_ratio,,,average_rank,average_rank_all,,,,,,,,,,laptime_std_prev,laptime_mean_all,laptime_std_all,,laps_after_last_pitstop,pittime_prev,,,,,,
    [*] predict with xgb model
    precision=[0.30555556 0.53571429 0.20588235], recall=[0.30555556 0.49180328 0.24137931], f1=[0.30555556 0.51282051 0.22222222], accuracy=0.38
    clf:%d :%s 3 ,,stageid,,pit_in_caution,,,,,,,,,,rate_of_change,rate_of_change_all,,,laptime_green_mean_all,,laptime_mean_prev,,laptime_mean_all,laptime_std_all,,,,prev_nb0_change_in_rank,,,,,follow_nb2_change_in_rank
    rec:['Pocono', 679, 126, '+:29,0:61,-:36', 0.35714285714285715, 0.3412698412698413, 0.40476190476190477, 0.38095238095238093]
    Testset = Gateway
    [*] predict with lr model


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


    precision=[0.66666667 0.30136986 0.46153846], recall=[0.28571429 0.78571429 0.17647059], f1=[0.4        0.43564356 0.25531915], accuracy=0.38
    clf:%d :%s 0 eventid,,stageid,firststage,pit_in_caution,,start_rank,start_rank_ratio,top_pack,bottom_pack,average_rank,,,change_in_rank_all,,,,,,,,,,,,,,,,,,,
    [*] predict with lsvc model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.61538462 0.33333333 0.52631579], recall=[0.19047619 0.85714286 0.29411765], f1=[0.29090909 0.48       0.37735849], accuracy=0.40
    clf:%d :%s 1 eventid,,stageid,firststage,pit_in_caution,,start_rank,,top_pack,bottom_pack,average_rank,,,,,,,,,laptime_green_std_all,,,,,,,,,,,,,follow_nb2_change_in_rank
    [*] predict with rf model
    precision=[0.48387097 0.46153846 0.44827586], recall=[0.71428571 0.21428571 0.38235294], f1=[0.57692308 0.29268293 0.41269841], accuracy=0.47
    clf:%d :%s 2 ,car_number,,,,,start_rank,start_rank_ratio,,,average_rank,average_rank_all,,,,,,,laptime_green_mean_all,,,laptime_std_prev,,laptime_std_all,,laps_after_last_pitstop,pittime_prev,,,,,,
    [*] predict with xgb model
    precision=[0.47457627 0.39130435 0.45454545], recall=[0.66666667 0.32142857 0.29411765], f1=[0.55445545 0.35294118 0.35714286], accuracy=0.45
    clf:%d :%s 3 ,,stageid,,pit_in_caution,,start_rank,,,,average_rank,,,,,,laptime_green_mean_prev,,,,laptime_mean_prev,,,,,laps_after_last_pitstop,,prev_nb0_change_in_rank,,prev_nb2_change_in_rank,,,follow_nb2_change_in_rank
    rec:['Gateway', 701, 104, '+:34,0:28,-:42', 0.38461538461538464, 0.40384615384615385, 0.47115384615384615, 0.4519230769230769]
    feature weights: start_rank,pit_in_caution,average_rank,stageid,start_rank_ratio,bottom_pack,firststage,top_pack,eventid,laptime_std_all,average_rank_all,laptime_green_mean_all,laptime_green_std_all,follow_nb2_change_in_rank,car_number,rate_of_change_all,pittime_prev,prev_nb0_change_in_rank,laps_after_last_pitstop,laptime_mean_prev,laptime_std_prev,prev_nb2_change_in_rank,change_in_rank_all,rate_of_change,follow_nb1_change_in_rank,laptime_mean_all,prev_nb1_change_in_rank,follow_nb0_change_in_rank,laps_prev,change_in_rank,laptime_green_mean_prev,start_position,laptime_green_std_prev



```python
#weights analysis
idx = np.argsort(-weights)
fnames = [featureNames[id] for id in idx]
print('feature weights:',','.join(fnames))
```

    feature weights: start_rank,pit_in_caution,average_rank,stageid,start_rank_ratio,bottom_pack,firststage,top_pack,eventid,laptime_std_all,average_rank_all,laptime_green_mean_all,laptime_green_std_all,follow_nb2_change_in_rank,car_number,rate_of_change_all,pittime_prev,prev_nb0_change_in_rank,laps_after_last_pitstop,laptime_mean_prev,laptime_std_prev,prev_nb2_change_in_rank,change_in_rank_all,rate_of_change,follow_nb1_change_in_rank,laptime_mean_all,prev_nb1_change_in_rank,follow_nb0_change_in_rank,laps_prev,change_in_rank,laptime_green_mean_prev,start_position,laptime_green_std_prev



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
      <th>lr</th>
      <th>lsvc</th>
      <th>rf</th>
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
      <td>0.517544</td>
      <td>0.526316</td>
      <td>0.508772</td>
      <td>0.447368</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Indy500</td>
      <td>580</td>
      <td>225</td>
      <td>+:82,0:47,-:96</td>
      <td>0.560000</td>
      <td>0.573333</td>
      <td>0.466667</td>
      <td>0.444444</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Texas</td>
      <td>678</td>
      <td>127</td>
      <td>+:39,0:34,-:54</td>
      <td>0.645669</td>
      <td>0.622047</td>
      <td>0.464567</td>
      <td>0.314961</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Iowa</td>
      <td>696</td>
      <td>109</td>
      <td>+:42,0:28,-:39</td>
      <td>0.376147</td>
      <td>0.376147</td>
      <td>0.376147</td>
      <td>0.376147</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Pocono</td>
      <td>679</td>
      <td>126</td>
      <td>+:29,0:61,-:36</td>
      <td>0.357143</td>
      <td>0.341270</td>
      <td>0.373016</td>
      <td>0.380952</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Gateway</td>
      <td>701</td>
      <td>104</td>
      <td>+:34,0:28,-:42</td>
      <td>0.384615</td>
      <td>0.403846</td>
      <td>0.432692</td>
      <td>0.451923</td>
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
        #acc[idx] = classifier_model(clf)
        acc[idx], features[idx] = classifier_model(clf, feature_num= selFeatureNumber)
        fnames = [featureNames[id] if val else '' for id,val in enumerate(features[idx])]
        print('clf:%d :%s',idx, ','.join(fnames))

    rec.extend(acc)
    print('rec:%s'%rec)
    
    #new df
    df = pd.DataFrame([rec],columns=cols)
    retdf = pd.concat([retdf, df])  
    
retdf.to_csv('crossvalid_stagedata_splitbystage%s.csv'%suffix)
df_stage = retdf
```

    [*] predict with lr model
    precision=[0.49633252 0.36842105 0.56603774], recall=[0.76893939 0.41916168 0.13574661], f1=[0.60326895 0.39215686 0.2189781 ], accuracy=0.46
    clf:%d :%s 0 eventid,car_number,,,,start_position,start_rank,start_rank_ratio,top_pack,bottom_pack,average_rank,average_rank_all,,,,,,,,,,,,,laps_prev,,,,,,,,
    [*] predict with lsvc model
    precision=[0.49438202 0.35833333 0.26785714], recall=[0.66666667 0.51497006 0.0678733 ], f1=[0.56774194 0.42260442 0.10830325], accuracy=0.42
    clf:%d :%s 1 eventid,car_number,,,,start_position,start_rank,start_rank_ratio,top_pack,bottom_pack,average_rank,average_rank_all,,,,,,,,,,,,,laps_prev,,,,,,,,
    [*] predict with rf model


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


    precision=[0.45051195 0.30985915 0.32876712], recall=[0.5        0.39520958 0.21719457], f1=[0.47396768 0.34736842 0.26158038], accuracy=0.38
    clf:%d :%s 2 eventid,car_number,,,,start_position,start_rank,start_rank_ratio,top_pack,bottom_pack,average_rank,average_rank_all,,,,,,,,,,,,,laps_prev,,,,,,,,
    [*] predict with xgb model
    precision=[0.4375     0.30677291 0.29655172], recall=[0.42424242 0.46107784 0.19457014], f1=[0.43076923 0.36842105 0.23497268], accuracy=0.36
    clf:%d :%s 3 eventid,car_number,,,,start_position,,start_rank_ratio,,bottom_pack,,,,,,,,,,,,,,,,laps_after_last_pitstop,pittime_prev,prev_nb0_change_in_rank,prev_nb1_change_in_rank,prev_nb2_change_in_rank,,,
    rec:['stage0', 153, 652, '+:221,0:167,-:264', 0.4647239263803681, 0.42484662576687116, 0.3773006134969325, 0.3558282208588957]
    [*] predict with lr model


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
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.43928571 0.31007752 0.60185185], recall=[0.63076923 0.29411765 0.34946237], f1=[0.51789474 0.30188679 0.44217687], accuracy=0.44
    clf:%d :%s 0 eventid,,,firststage,pit_in_caution,,start_rank,start_rank_ratio,top_pack,bottom_pack,average_rank,,,,,,,,,,,,,,,,,,,prev_nb2_change_in_rank,,follow_nb1_change_in_rank,
    [*] predict with lsvc model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.42857143 0.32307692 0.59813084], recall=[0.61538462 0.30882353 0.34408602], f1=[0.50526316 0.31578947 0.43686007], accuracy=0.44
    clf:%d :%s 1 eventid,,,,pit_in_caution,,start_rank,start_rank_ratio,top_pack,bottom_pack,average_rank,,,,,,,,,,,,,,,,,,,prev_nb2_change_in_rank,follow_nb0_change_in_rank,follow_nb1_change_in_rank,
    [*] predict with rf model
    precision=[0.42028986 0.2739726  0.41414141], recall=[0.74358974 0.14705882 0.22043011], f1=[0.53703704 0.19138756 0.2877193 ], accuracy=0.40
    clf:%d :%s 2 eventid,car_number,,,,start_position,start_rank,start_rank_ratio,,,average_rank,average_rank_all,,,,,,,,,,,,laptime_std_all,laps_prev,,pittime_prev,,,,,,
    [*] predict with xgb model
    precision=[0.36708861 0.22641509 0.37162162], recall=[0.59487179 0.08823529 0.29569892], f1=[0.45401174 0.12698413 0.32934132], accuracy=0.35
    clf:%d :%s 3 ,,,,,,,,,,,,,,,,laptime_green_mean_prev,laptime_green_std_prev,,,,laptime_std_prev,,,laps_prev,,,prev_nb0_change_in_rank,prev_nb1_change_in_rank,prev_nb2_change_in_rank,follow_nb0_change_in_rank,follow_nb1_change_in_rank,follow_nb2_change_in_rank
    rec:['stage1', 288, 517, '+:186,0:136,-:195', 0.4410058027079304, 0.437137330754352, 0.3984526112185687, 0.3539651837524178]
    [*] predict with lr model


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


    precision=[0.48387097 0.         0.45414847], recall=[0.56818182 0.         0.74285714], f1=[0.52264808 0.         0.56368564], accuracy=0.47
    clf:%d :%s 0 eventid,,stageid,firststage,pit_in_caution,,,start_rank_ratio,top_pack,bottom_pack,average_rank,,,,,,laptime_green_mean_prev,,laptime_green_mean_all,,,,,,,,,,,,,,
    [*] predict with lsvc model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)


    precision=[0.47282609 0.         0.49      ], recall=[0.65909091 0.         0.7       ], f1=[0.55063291 0.         0.57647059], accuracy=0.48
    clf:%d :%s 1 eventid,,stageid,firststage,pit_in_caution,,start_rank,,top_pack,bottom_pack,average_rank,,,,,,,laptime_green_std_prev,,laptime_green_std_all,,,,,,,,,,,,,
    [*] predict with rf model
    precision=[0.43866171 0.5625     0.56716418], recall=[0.89393939 0.24107143 0.27142857], f1=[0.58852868 0.3375     0.36714976], accuracy=0.48
    clf:%d :%s 2 eventid,car_number,,,,start_position,start_rank,start_rank_ratio,,,average_rank,average_rank_all,,,,,,,,,,,laptime_mean_all,laptime_std_all,laps_prev,,,,,,,,
    [*] predict with xgb model
    precision=[0.36397059 0.28571429 0.50649351], recall=[0.75       0.08928571 0.27857143], f1=[0.49009901 0.13605442 0.359447  ], accuracy=0.39
    clf:%d :%s 3 ,,,,,,,,,,,,change_in_rank,change_in_rank_all,,,,,,,laptime_mean_prev,,laptime_mean_all,,,,,prev_nb0_change_in_rank,prev_nb1_change_in_rank,prev_nb2_change_in_rank,follow_nb0_change_in_rank,follow_nb1_change_in_rank,follow_nb2_change_in_rank
    rec:['stage2', 421, 384, '+:140,0:112,-:132', 0.4661458333333333, 0.4817708333333333, 0.4765625, 0.3854166666666667]
    [*] predict with lr model


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


    precision=[0.44554455 0.33333333 0.4527027 ], recall=[0.57692308 0.03370787 0.73626374], f1=[0.5027933  0.06122449 0.56066946], accuracy=0.45
    clf:%d :%s 0 ,,stageid,firststage,pit_in_caution,,,start_rank_ratio,top_pack,bottom_pack,,,,,,,laptime_green_mean_prev,laptime_green_std_prev,laptime_green_mean_all,laptime_green_std_all,,,,,,,,,,,,,
    [*] predict with lsvc model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)


    precision=[0.45689655 0.         0.48591549], recall=[0.67948718 0.         0.75824176], f1=[0.54639175 0.         0.59227468], accuracy=0.47
    clf:%d :%s 1 eventid,,stageid,firststage,pit_in_caution,,start_rank,,top_pack,bottom_pack,average_rank,,,,,,,laptime_green_std_prev,,laptime_green_std_all,,,,,,,,,,,,,
    [*] predict with rf model
    precision=[0.38121547 0.6        0.51612903], recall=[0.88461538 0.1011236  0.35164835], f1=[0.53281853 0.17307692 0.41830065], accuracy=0.43
    clf:%d :%s 2 ,car_number,,,,start_position,start_rank,start_rank_ratio,,,average_rank,average_rank_all,,,,,,,,,,laptime_std_prev,laptime_mean_all,laptime_std_all,,,pittime_prev,,,,,,
    [*] predict with xgb model
    precision=[0.34502924 0.47058824 0.52857143], recall=[0.75641026 0.08988764 0.40659341], f1=[0.47389558 0.1509434  0.45962733], accuracy=0.40
    clf:%d :%s 3 ,,,,,,start_rank,,,,average_rank,,,change_in_rank_all,rate_of_change,rate_of_change_all,,,,,,,,,,,,prev_nb0_change_in_rank,prev_nb1_change_in_rank,prev_nb2_change_in_rank,follow_nb0_change_in_rank,,follow_nb2_change_in_rank
    rec:['stage3', 547, 258, '+:91,0:89,-:78', 0.44573643410852715, 0.4728682170542636, 0.4263565891472868, 0.40310077519379844]
    [*] predict with lr model


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


    precision=[0.625      0.59375    0.52941176], recall=[0.63829787 0.35849057 0.75      ], f1=[0.63157895 0.44705882 0.62068966], accuracy=0.57
    clf:%d :%s 0 eventid,,stageid,firststage,pit_in_caution,,start_rank,start_rank_ratio,top_pack,bottom_pack,,average_rank_all,,,,,,,,,,,,,,laps_after_last_pitstop,,,,,,,
    [*] predict with lsvc model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.58333333 0.61538462 0.45977011], recall=[0.59574468 0.1509434  0.83333333], f1=[0.58947368 0.24242424 0.59259259], accuracy=0.51
    clf:%d :%s 1 eventid,,stageid,firststage,pit_in_caution,,start_rank,,top_pack,bottom_pack,average_rank,average_rank_all,,,,,,,,laptime_green_std_all,,,,,,,,,,,,,
    [*] predict with rf model
    precision=[0.37078652 0.53125    0.48148148], recall=[0.70212766 0.32075472 0.27083333], f1=[0.48529412 0.4        0.34666667], accuracy=0.43
    clf:%d :%s 2 ,car_number,,,,,start_rank,start_rank_ratio,,,average_rank,average_rank_all,,,,,,,laptime_green_mean_all,,laptime_mean_prev,laptime_std_prev,,laptime_std_all,,laps_after_last_pitstop,,,,,,,
    [*] predict with xgb model
    precision=[0.3943662  0.34883721 0.29411765], recall=[0.59574468 0.28301887 0.20833333], f1=[0.47457627 0.3125     0.24390244], accuracy=0.36
    clf:%d :%s 3 ,,,,pit_in_caution,,,,,,,,,change_in_rank_all,,rate_of_change_all,,laptime_green_std_prev,,,,,,,,,,prev_nb0_change_in_rank,prev_nb1_change_in_rank,prev_nb2_change_in_rank,follow_nb0_change_in_rank,follow_nb1_change_in_rank,follow_nb2_change_in_rank
    rec:['stage4', 657, 148, '+:48,0:53,-:47', 0.5743243243243243, 0.5135135135135135, 0.42567567567567566, 0.3581081081081081]
    [*] predict with lr model


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


    precision=[0.65217391 0.525      0.88235294], recall=[0.6        0.72413793 0.57692308], f1=[0.625      0.60869565 0.69767442], accuracy=0.64
    clf:%d :%s 0 eventid,,stageid,firststage,pit_in_caution,,start_rank,start_rank_ratio,top_pack,bottom_pack,average_rank,,,change_in_rank_all,,,,,,,,,,,,,,,,,,,
    [*] predict with lsvc model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.64       0.72222222 0.56756757], recall=[0.64       0.44827586 0.80769231], f1=[0.64       0.55319149 0.66666667], accuracy=0.62
    clf:%d :%s 1 eventid,,stageid,firststage,pit_in_caution,,start_rank,,top_pack,bottom_pack,average_rank,,,change_in_rank_all,,,,,,laptime_green_std_all,,,,,,,,,,,,,
    [*] predict with rf model
    precision=[0.45714286 0.53571429 0.64705882], recall=[0.64       0.51724138 0.42307692], f1=[0.53333333 0.52631579 0.51162791], accuracy=0.53
    clf:%d :%s 2 ,car_number,,,,,start_rank,start_rank_ratio,,,average_rank,average_rank_all,,,,,,,laptime_green_mean_all,,laptime_mean_prev,laptime_std_prev,,laptime_std_all,,,pittime_prev,,,,,,
    [*] predict with xgb model
    precision=[0.51851852 0.41025641 0.64285714], recall=[0.56       0.55172414 0.34615385], f1=[0.53846154 0.47058824 0.45      ], accuracy=0.49
    clf:%d :%s 3 ,,stageid,,pit_in_caution,,,,,,,,,,rate_of_change,rate_of_change_all,,,laptime_green_mean_all,,,laptime_std_prev,,laptime_std_all,,,,,,,follow_nb0_change_in_rank,follow_nb1_change_in_rank,follow_nb2_change_in_rank
    rec:['stage5', 725, 80, '+:26,0:29,-:25', 0.6375, 0.625, 0.525, 0.4875]
    [*] predict with lr model


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


    precision=[0.63636364 0.41176471 0.6       ], recall=[0.5        0.53846154 0.54545455], f1=[0.56       0.46666667 0.57142857], accuracy=0.53
    clf:%d :%s 0 eventid,,stageid,firststage,pit_in_caution,,start_rank,start_rank_ratio,top_pack,bottom_pack,average_rank,average_rank_all,,,,,,,,,,,,,,,,,,,,,
    [*] predict with lsvc model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[0.61538462 0.42857143 0.54545455], recall=[0.57142857 0.46153846 0.54545455], f1=[0.59259259 0.44444444 0.54545455], accuracy=0.53
    clf:%d :%s 1 eventid,,stageid,firststage,pit_in_caution,,start_rank,,top_pack,bottom_pack,average_rank,average_rank_all,,,,,,,,laptime_green_std_all,,,,,,,,,,,,,
    [*] predict with rf model
    precision=[0.54545455 0.44444444 0.38888889], recall=[0.42857143 0.30769231 0.63636364], f1=[0.48       0.36363636 0.48275862], accuracy=0.45
    clf:%d :%s 2 ,car_number,,,,,start_rank,start_rank_ratio,,,average_rank,average_rank_all,,,,,,,laptime_green_mean_all,,,laptime_std_prev,,laptime_std_all,laps_prev,,pittime_prev,,,,,,
    [*] predict with xgb model
    precision=[0.42857143 0.57142857 0.3       ], recall=[0.64285714 0.30769231 0.27272727], f1=[0.51428571 0.4        0.28571429], accuracy=0.42
    clf:%d :%s 3 ,,stageid,,pit_in_caution,,start_rank,,,,average_rank,average_rank_all,,,,rate_of_change_all,,,,,,,,,,laps_after_last_pitstop,,,prev_nb1_change_in_rank,prev_nb2_change_in_rank,,,follow_nb2_change_in_rank
    rec:['stage6', 767, 38, '+:11,0:13,-:14', 0.5263157894736842, 0.5263157894736842, 0.4473684210526316, 0.42105263157894735]
    [*] predict with lr model


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


    precision=[0.83333333 0.2        0.6       ], recall=[0.625 0.25  0.75 ], f1=[0.71428571 0.22222222 0.66666667], accuracy=0.56
    clf:%d :%s 0 eventid,,stageid,firststage,pit_in_caution,,start_rank,start_rank_ratio,top_pack,bottom_pack,average_rank,,,,,,,,,laptime_green_std_all,,,,,,,,,,,,,
    [*] predict with lsvc model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    precision=[1.    0.    0.375], recall=[0.625 0.    0.75 ], f1=[0.76923077 0.         0.5       ], accuracy=0.50
    clf:%d :%s 1 eventid,,stageid,firststage,pit_in_caution,,start_rank,,top_pack,bottom_pack,average_rank,average_rank_all,,,,,,,,laptime_green_std_all,,,,,,,,,,,,,
    [*] predict with rf model
    precision=[0.  0.  0.2], recall=[0.   0.   0.25], f1=[0.         0.         0.22222222], accuracy=0.06
    clf:%d :%s 2 ,car_number,,,,,start_rank,start_rank_ratio,,,average_rank,average_rank_all,,,,,,,laptime_green_mean_all,,laptime_mean_prev,laptime_std_prev,,laptime_std_all,,,pittime_prev,,,,,,
    [*] predict with xgb model
    precision=[0.75       0.28571429 0.4       ], recall=[0.375 0.5   0.5  ], f1=[0.5        0.36363636 0.44444444], accuracy=0.44
    clf:%d :%s 3 ,,stageid,,pit_in_caution,,start_rank,,,,average_rank,,,,,rate_of_change_all,,,laptime_green_mean_all,,,,,laptime_std_all,,,,,prev_nb1_change_in_rank,prev_nb2_change_in_rank,,,follow_nb2_change_in_rank
    rec:['stage7', 789, 16, '+:4,0:4,-:8', 0.5625, 0.5, 0.0625, 0.4375]



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
      <th>lr</th>
      <th>lsvc</th>
      <th>rf</th>
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
      <td>0.464724</td>
      <td>0.424847</td>
      <td>0.377301</td>
      <td>0.355828</td>
    </tr>
    <tr>
      <th>0</th>
      <td>stage1</td>
      <td>288</td>
      <td>517</td>
      <td>+:186,0:136,-:195</td>
      <td>0.441006</td>
      <td>0.437137</td>
      <td>0.398453</td>
      <td>0.353965</td>
    </tr>
    <tr>
      <th>0</th>
      <td>stage2</td>
      <td>421</td>
      <td>384</td>
      <td>+:140,0:112,-:132</td>
      <td>0.466146</td>
      <td>0.481771</td>
      <td>0.476562</td>
      <td>0.385417</td>
    </tr>
    <tr>
      <th>0</th>
      <td>stage3</td>
      <td>547</td>
      <td>258</td>
      <td>+:91,0:89,-:78</td>
      <td>0.445736</td>
      <td>0.472868</td>
      <td>0.426357</td>
      <td>0.403101</td>
    </tr>
    <tr>
      <th>0</th>
      <td>stage4</td>
      <td>657</td>
      <td>148</td>
      <td>+:48,0:53,-:47</td>
      <td>0.574324</td>
      <td>0.513514</td>
      <td>0.425676</td>
      <td>0.358108</td>
    </tr>
    <tr>
      <th>0</th>
      <td>stage5</td>
      <td>725</td>
      <td>80</td>
      <td>+:26,0:29,-:25</td>
      <td>0.637500</td>
      <td>0.625000</td>
      <td>0.525000</td>
      <td>0.487500</td>
    </tr>
    <tr>
      <th>0</th>
      <td>stage6</td>
      <td>767</td>
      <td>38</td>
      <td>+:11,0:13,-:14</td>
      <td>0.526316</td>
      <td>0.526316</td>
      <td>0.447368</td>
      <td>0.421053</td>
    </tr>
    <tr>
      <th>0</th>
      <td>stage7</td>
      <td>789</td>
      <td>16</td>
      <td>+:4,0:4,-:8</td>
      <td>0.562500</td>
      <td>0.500000</td>
      <td>0.062500</td>
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
