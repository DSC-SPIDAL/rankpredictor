
### stage_model_regressor

prediction models of chg_of_rank_in_stage on stage dataset

data format:
    target , eventid ,    car_number,    stageid,     features...


```python
%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# to use only one GPU.
# use this on r-001
# otherwise comment
import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"
```


```python
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.linear_model.ridge import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn.svm.classes import SVR
from sklearn.utils import shuffle
from sklearn import metrics
import xgboost as xgb
```


```python
# bulid regression model
regressors = ['currank','avgrank','dice','lasso','ridge','rf','svr','xgb']
def get_regressor(regressor = 'lr'):
    if regressor == "lasso":
        clf = LassoCV(cv=5, random_state=0)
    elif regressor == "ridge":
        clf = RidgeCV(alphas=np.logspace(-6, 6, 13))
    elif regressor == "rf":
        clf = RandomForestRegressor(n_estimators=100)
    elif regressor == 'svr':
        clf = SVR(kernel='rbf')
    elif regressor == 'xgb':
        clf = xgb.XGBRegressor(objective="reg:linear", random_state=42, max_depth=3)
    elif regressor == 'dice':
        clf = RandomDice('1234')
    elif regressor == 'currank':
        clf = CurRank()
    elif regressor == 'avgrank':
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
        #pred_y_avg = np.array([1 if x > 0 else (-1 if x < 0 else 0) for x in pred_y])
        pred_y_avg = pred_y
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
    mae = metrics.mean_absolute_error(test_y, pred_y) 
    rmse = math.sqrt(metrics.mean_squared_error(test_y, pred_y))
    r2 = metrics.r2_score(test_y, pred_y)
    print('rmse=%.2f, mae=%.2f, r2=%.2f'%(rmse, mae, r2))
    return rmse, mae, r2
    
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
    #train_y = np.array([1 if x > 0 else (-1 if x < 0 else 0) for x in train[:,1]])
    train_y = train[:,1]
    test_x = test[:,2:]
    #test_y = np.array([1 if x > 0 else (-1 if x < 0 else 0) for x in test[:,1]])
    test_y = test[:,1]
    
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
    #train_y = np.array([1 if x > 0 else (-1 if x < 0 else 0) for x in train[:,1]])
    train_y = train[:,1]
    test_x = test[:,2:]
    #test_y = np.array([1 if x > 0 else (-1 if x < 0 else 0) for x in test[:,1]])
    test_y = test[:,1]
    
    return train, test, train_x, train_y, test_x, test_y


def regressor_model(name='svr'):
    ### test learning models
    print('[*] predict with %s model'%name)
    clf = get_regressor(name)
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


### model on data split by event


```python
cols = ['runid','trainsize','testsize','testdistribution']
cols.extend(regressors)
print('cols:%s'%cols)
retdf0 = pd.DataFrame([],columns=cols)
retdf1 = pd.DataFrame([],columns=cols)

eventsname = ['Phoenix','Indy500','Texas','Iowa','Pocono','Gateway']
events = set(stagedata['eventid'])
for eventid in events:
    print('Testset = %s'%eventsname[eventid])
    
    train, test, train_x, train_y, test_x, test_y = split_by_eventid(stagedata, eventid)
    test_distribution = '+:%d,0:%d,-:%d'%(np.sum(test_y>0),np.sum(test_y==0),np.sum(test_y<0))
    #print('Testset by stageid= %s, trainsize=%d, testsize=%d, dist=%s'%
    #      (stageid, train_x.shape[0], test_x.shape[0], test_distribution))
    
    #record
    rec0 = [eventsname[eventid],train_x.shape[0],test_x.shape[0],test_distribution]
    rec1 = [eventsname[eventid],train_x.shape[0],test_x.shape[0],test_distribution]
    
    acc0 = [0 for x in range(len(regressors))]
    acc1 = [0 for x in range(len(regressors))]
    for idx, clf in enumerate(regressors):
        acc0[idx] = regressor_model(clf)[0]
        acc1[idx] = regressor_model(clf)[2]

    rec0.extend(acc0)
    rec1.extend(acc1)
    #print('rec:%s'%rec)
    
    #new df
    df = pd.DataFrame([rec0],columns=cols)
    retdf0 = pd.concat([retdf0, df])        
    
    df = pd.DataFrame([rec1],columns=cols)
    retdf1 = pd.concat([retdf1, df])        

    
retdf0.to_csv('regressors_stagedata_splitbyevent%s_rmse.csv'%suffix)
retdf1.to_csv('regressors_stagedata_splitbyevent%s_r2.csv'%suffix)

df_event_rmse = retdf0
df_event_r2 = retdf1
```

    cols:['runid', 'trainsize', 'testsize', 'testdistribution', 'currank', 'avgrank', 'dice', 'lasso', 'ridge', 'rf', 'svr', 'xgb']
    Testset = Phoenix
    [*] predict with currank model
    rmse=4.73, mae=3.20, r2=-0.00
    [*] predict with currank model
    rmse=4.73, mae=3.20, r2=-0.00
    [*] predict with avgrank model
    rmse=5.61, mae=3.83, r2=-0.40
    [*] predict with avgrank model
    rmse=5.61, mae=3.83, r2=-0.40
    [*] predict with dice model
    rmse=7.01, mae=4.88, r2=-1.19
    [*] predict with dice model
    rmse=7.01, mae=4.88, r2=-1.19
    [*] predict with lasso model
    rmse=4.44, mae=2.95, r2=0.12
    [*] predict with lasso model
    rmse=4.44, mae=2.95, r2=0.12
    [*] predict with ridge model
    rmse=4.42, mae=2.90, r2=0.13
    [*] predict with ridge model
    rmse=4.42, mae=2.90, r2=0.13
    [*] predict with rf model
    rmse=4.53, mae=3.18, r2=0.08
    [*] predict with rf model
    rmse=4.61, mae=3.28, r2=0.05
    [*] predict with svr model
    rmse=4.70, mae=3.13, r2=0.01
    [*] predict with svr model
    rmse=4.70, mae=3.13, r2=0.01
    [*] predict with xgb model
    [16:49:47] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=4.26, mae=2.99, r2=0.19
    [*] predict with xgb model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    [16:49:48] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=4.26, mae=2.99, r2=0.19
    Testset = Indy500
    [*] predict with currank model
    rmse=6.17, mae=3.90, r2=-0.00
    [*] predict with currank model
    rmse=6.17, mae=3.90, r2=-0.00
    [*] predict with avgrank model
    rmse=6.94, mae=4.75, r2=-0.27
    [*] predict with avgrank model
    rmse=6.94, mae=4.75, r2=-0.27
    [*] predict with dice model
    rmse=7.55, mae=5.28, r2=-0.50
    [*] predict with dice model
    rmse=7.55, mae=5.28, r2=-0.50
    [*] predict with lasso model
    rmse=5.69, mae=3.99, r2=0.15
    [*] predict with lasso model
    rmse=5.69, mae=3.99, r2=0.15
    [*] predict with ridge model
    rmse=5.51, mae=3.98, r2=0.20
    [*] predict with ridge model
    rmse=5.51, mae=3.98, r2=0.20
    [*] predict with rf model
    rmse=5.76, mae=4.28, r2=0.12
    [*] predict with rf model
    rmse=5.62, mae=4.14, r2=0.17
    [*] predict with svr model
    rmse=6.14, mae=3.91, r2=0.00
    [*] predict with svr model
    rmse=6.14, mae=3.91, r2=0.00
    [*] predict with xgb model
    [16:49:49] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=5.82, mae=4.07, r2=0.11
    [*] predict with xgb model
    [16:49:49] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    rmse=5.82, mae=4.07, r2=0.11
    Testset = Texas
    [*] predict with currank model
    rmse=4.26, mae=2.60, r2=-0.01
    [*] predict with currank model
    rmse=4.26, mae=2.60, r2=-0.01
    [*] predict with avgrank model
    rmse=5.48, mae=3.73, r2=-0.67
    [*] predict with avgrank model
    rmse=5.48, mae=3.73, r2=-0.67
    [*] predict with dice model
    rmse=6.73, mae=4.68, r2=-1.52
    [*] predict with dice model
    rmse=6.73, mae=4.68, r2=-1.52
    [*] predict with lasso model
    rmse=3.78, mae=2.53, r2=0.21
    [*] predict with lasso model
    rmse=3.78, mae=2.53, r2=0.21
    [*] predict with ridge model
    rmse=3.76, mae=2.51, r2=0.22
    [*] predict with ridge model
    rmse=3.76, mae=2.51, r2=0.22
    [*] predict with rf model
    rmse=3.96, mae=2.70, r2=0.13
    [*] predict with rf model
    rmse=4.00, mae=2.74, r2=0.11
    [*] predict with svr model
    rmse=4.13, mae=2.59, r2=0.05
    [*] predict with svr model
    rmse=4.13, mae=2.59, r2=0.05
    [*] predict with xgb model
    [16:49:50] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=4.02, mae=2.69, r2=0.10
    [*] predict with xgb model
    [16:49:50] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    rmse=4.02, mae=2.69, r2=0.10
    Testset = Iowa
    [*] predict with currank model
    rmse=3.98, mae=2.57, r2=0.00
    [*] predict with currank model
    rmse=3.98, mae=2.57, r2=0.00
    [*] predict with avgrank model
    rmse=5.80, mae=3.85, r2=-1.12
    [*] predict with avgrank model
    rmse=5.80, mae=3.85, r2=-1.12
    [*] predict with dice model
    rmse=6.45, mae=4.61, r2=-1.63
    [*] predict with dice model
    rmse=6.45, mae=4.61, r2=-1.63
    [*] predict with lasso model
    rmse=3.69, mae=2.69, r2=0.14
    [*] predict with lasso model
    rmse=3.69, mae=2.69, r2=0.14
    [*] predict with ridge model
    rmse=3.86, mae=2.81, r2=0.06
    [*] predict with ridge model
    rmse=3.86, mae=2.81, r2=0.06
    [*] predict with rf model
    rmse=4.02, mae=2.91, r2=-0.02
    [*] predict with rf model
    rmse=3.98, mae=2.91, r2=0.00
    [*] predict with svr model
    rmse=3.91, mae=2.60, r2=0.04
    [*] predict with svr model
    rmse=3.91, mae=2.60, r2=0.04
    [*] predict with xgb model
    [16:49:52] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    rmse=4.06, mae=3.01, r2=-0.04
    [*] predict with xgb model
    [16:49:52] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=4.06, mae=3.01, r2=-0.04
    Testset = Pocono
    [*] predict with currank model
    rmse=2.48, mae=1.29, r2=-0.04
    [*] predict with currank model
    rmse=2.48, mae=1.29, r2=-0.04
    [*] predict with avgrank model
    rmse=2.93, mae=1.92, r2=-0.45
    [*] predict with avgrank model
    rmse=2.93, mae=1.92, r2=-0.45
    [*] predict with dice model
    rmse=5.61, mae=3.88, r2=-4.34
    [*] predict with dice model
    rmse=5.61, mae=3.88, r2=-4.34
    [*] predict with lasso model
    rmse=2.49, mae=1.97, r2=-0.05
    [*] predict with lasso model
    rmse=2.49, mae=1.97, r2=-0.05
    [*] predict with ridge model
    rmse=2.57, mae=2.04, r2=-0.12
    [*] predict with ridge model
    rmse=2.57, mae=2.04, r2=-0.12
    [*] predict with rf model
    rmse=3.07, mae=2.22, r2=-0.60
    [*] predict with rf model
    rmse=3.01, mae=2.18, r2=-0.53
    [*] predict with svr model
    rmse=2.38, mae=1.40, r2=0.04
    [*] predict with svr model
    rmse=2.38, mae=1.40, r2=0.04
    [*] predict with xgb model
    [16:49:53] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=2.91, mae=2.24, r2=-0.44


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    [*] predict with xgb model
    [16:49:54] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=2.91, mae=2.24, r2=-0.44
    Testset = Gateway
    [*] predict with currank model
    rmse=3.41, mae=2.16, r2=-0.00
    [*] predict with currank model
    rmse=3.41, mae=2.16, r2=-0.00
    [*] predict with avgrank model
    rmse=4.30, mae=2.81, r2=-0.59
    [*] predict with avgrank model
    rmse=4.30, mae=2.81, r2=-0.59
    [*] predict with dice model
    rmse=5.89, mae=4.16, r2=-1.98
    [*] predict with dice model
    rmse=5.89, mae=4.16, r2=-1.98
    [*] predict with lasso model
    rmse=3.11, mae=2.12, r2=0.17
    [*] predict with lasso model
    rmse=3.11, mae=2.12, r2=0.17
    [*] predict with ridge model
    rmse=3.11, mae=2.08, r2=0.17
    [*] predict with ridge model
    rmse=3.11, mae=2.08, r2=0.17
    [*] predict with rf model
    rmse=3.54, mae=2.46, r2=-0.07
    [*] predict with rf model
    rmse=3.67, mae=2.58, r2=-0.16
    [*] predict with svr model
    rmse=3.34, mae=2.21, r2=0.04
    [*] predict with svr model
    rmse=3.34, mae=2.21, r2=0.04
    [*] predict with xgb model
    [16:49:55] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    rmse=3.32, mae=2.32, r2=0.05
    [*] predict with xgb model
    [16:49:55] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=3.32, mae=2.32, r2=0.05


### model on data split by stage


```python
retdf0 = pd.DataFrame([],columns=cols)
retdf1 = pd.DataFrame([],columns=cols)

for stageid in range(8):
    train, test, train_x, train_y, test_x, test_y =split_by_stageid(stagedata, stageid)
    test_distribution = '+:%d,0:%d,-:%d'%(np.sum(test_y>0),np.sum(test_y==0),np.sum(test_y<0))
    #print('Testset by stageid= %s, trainsize=%d, testsize=%d, dist=%s'%
    #      (stageid, train_x.shape[0], test_x.shape[0], test_distribution))
    
    #record
    rec0 = ['stage%d'%stageid,train_x.shape[0],test_x.shape[0],test_distribution]
    rec1 = ['stage%d'%stageid,train_x.shape[0],test_x.shape[0],test_distribution]
    
    acc0 = [0 for x in range(len(regressors))]
    acc1 = [0 for x in range(len(regressors))]
    for idx, clf in enumerate(regressors):
        acc0[idx] = regressor_model(clf)[0]
        acc1[idx] = regressor_model(clf)[2]

    rec0.extend(acc0)
    rec1.extend(acc1)
    #print('rec:%s'%rec)
    
    #new df
    df = pd.DataFrame([rec0],columns=cols)
    retdf0 = pd.concat([retdf0, df])  
    
    df = pd.DataFrame([rec1],columns=cols)
    retdf1 = pd.concat([retdf1, df])  

retdf0.to_csv('regressor_stagedata_splitbystage%s_rmse.csv'%suffix)
retdf1.to_csv('regressor_stagedata_splitbystage%s_r2.csv'%suffix)

df_stage_rmse = retdf0
df_stage_r2 = retdf1
```

    [*] predict with currank model
    rmse=4.75, mae=2.85, r2=-0.00
    [*] predict with currank model
    rmse=4.75, mae=2.85, r2=-0.00
    [*] predict with avgrank model
    rmse=5.87, mae=3.91, r2=-0.53
    [*] predict with avgrank model
    rmse=5.87, mae=3.91, r2=-0.53
    [*] predict with dice model
    rmse=6.16, mae=4.36, r2=-0.68
    [*] predict with dice model
    rmse=6.16, mae=4.36, r2=-0.68
    [*] predict with lasso model
    rmse=4.85, mae=3.19, r2=-0.05
    [*] predict with lasso model
    rmse=4.85, mae=3.19, r2=-0.05
    [*] predict with ridge model
    rmse=4.96, mae=3.25, r2=-0.09
    [*] predict with ridge model
    rmse=4.96, mae=3.25, r2=-0.09
    [*] predict with rf model
    rmse=4.98, mae=3.33, r2=-0.10
    [*] predict with rf model
    rmse=5.05, mae=3.38, r2=-0.13
    [*] predict with svr model
    rmse=4.76, mae=2.93, r2=-0.01
    [*] predict with svr model
    rmse=4.76, mae=2.93, r2=-0.01
    [*] predict with xgb model
    [16:49:56] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=5.13, mae=3.46, r2=-0.17
    [*] predict with xgb model
    [16:49:56] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=5.13, mae=3.46, r2=-0.17
    [*] predict with currank model
    rmse=4.68, mae=2.72, r2=-0.00
    [*] predict with currank model
    rmse=4.68, mae=2.72, r2=-0.00
    [*] predict with avgrank model
    rmse=5.53, mae=3.64, r2=-0.39
    [*] predict with avgrank model
    rmse=5.53, mae=3.64, r2=-0.39
    [*] predict with dice model
    rmse=6.86, mae=4.75, r2=-1.15
    [*] predict with dice model
    rmse=6.86, mae=4.75, r2=-1.15
    [*] predict with lasso model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    rmse=4.45, mae=3.12, r2=0.10
    [*] predict with lasso model
    rmse=4.45, mae=3.12, r2=0.10
    [*] predict with ridge model
    rmse=4.69, mae=3.36, r2=-0.00
    [*] predict with ridge model
    rmse=4.69, mae=3.36, r2=-0.00
    [*] predict with rf model
    rmse=4.55, mae=3.40, r2=0.06
    [*] predict with rf model
    rmse=4.62, mae=3.41, r2=0.02
    [*] predict with svr model
    rmse=4.78, mae=2.90, r2=-0.04
    [*] predict with svr model
    rmse=4.78, mae=2.90, r2=-0.04
    [*] predict with xgb model
    [16:49:56] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=4.92, mae=3.52, r2=-0.11
    [*] predict with xgb model
    [16:49:56] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=4.92, mae=3.52, r2=-0.11
    [*] predict with currank model
    rmse=4.90, mae=2.74, r2=-0.01
    [*] predict with currank model
    rmse=4.90, mae=2.74, r2=-0.01
    [*] predict with avgrank model
    rmse=5.68, mae=3.66, r2=-0.35
    [*] predict with avgrank model
    rmse=5.68, mae=3.66, r2=-0.35
    [*] predict with dice model
    rmse=7.05, mae=5.03, r2=-1.08
    [*] predict with dice model
    rmse=7.05, mae=5.03, r2=-1.08
    [*] predict with lasso model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    rmse=4.74, mae=2.93, r2=0.06
    [*] predict with lasso model
    rmse=4.74, mae=2.93, r2=0.06
    [*] predict with ridge model
    rmse=4.62, mae=3.03, r2=0.11
    [*] predict with ridge model
    rmse=4.62, mae=3.03, r2=0.11
    [*] predict with rf model
    rmse=4.64, mae=3.12, r2=0.10
    [*] predict with rf model
    rmse=4.66, mae=3.14, r2=0.09
    [*] predict with svr model
    rmse=5.00, mae=2.94, r2=-0.04
    [*] predict with svr model
    rmse=5.00, mae=2.94, r2=-0.04
    [*] predict with xgb model
    [16:49:57] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=4.80, mae=3.18, r2=0.03
    [*] predict with xgb model
    [16:49:58] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=4.80, mae=3.18, r2=0.03
    [*] predict with currank model
    rmse=4.73, mae=2.52, r2=-0.02
    [*] predict with currank model
    rmse=4.73, mae=2.52, r2=-0.02
    [*] predict with avgrank model
    rmse=5.54, mae=3.42, r2=-0.40
    [*] predict with avgrank model
    rmse=5.54, mae=3.42, r2=-0.40
    [*] predict with dice model
    rmse=7.01, mae=4.77, r2=-1.24
    [*] predict with dice model
    rmse=7.01, mae=4.77, r2=-1.24
    [*] predict with lasso model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    rmse=4.76, mae=3.06, r2=-0.03
    [*] predict with lasso model
    rmse=4.76, mae=3.06, r2=-0.03
    [*] predict with ridge model
    rmse=4.87, mae=3.19, r2=-0.08
    [*] predict with ridge model
    rmse=4.87, mae=3.19, r2=-0.08
    [*] predict with rf model
    rmse=4.62, mae=3.10, r2=0.03
    [*] predict with rf model
    rmse=4.61, mae=3.13, r2=0.03
    [*] predict with svr model
    rmse=4.82, mae=2.70, r2=-0.06
    [*] predict with svr model
    rmse=4.82, mae=2.70, r2=-0.06
    [*] predict with xgb model
    [16:49:59] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=5.12, mae=3.39, r2=-0.19
    [*] predict with xgb model
    [16:49:59] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    rmse=5.12, mae=3.39, r2=-0.19
    [*] predict with currank model
    rmse=4.91, mae=2.63, r2=-0.01
    [*] predict with currank model
    rmse=4.91, mae=2.63, r2=-0.01
    [*] predict with avgrank model
    rmse=5.65, mae=3.43, r2=-0.33
    [*] predict with avgrank model
    rmse=5.65, mae=3.43, r2=-0.33
    [*] predict with dice model
    rmse=6.45, mae=4.41, r2=-0.74
    [*] predict with dice model
    rmse=6.45, mae=4.41, r2=-0.74
    [*] predict with lasso model
    rmse=4.51, mae=2.85, r2=0.15
    [*] predict with lasso model
    rmse=4.51, mae=2.85, r2=0.15
    [*] predict with ridge model
    rmse=4.56, mae=2.91, r2=0.13
    [*] predict with ridge model
    rmse=4.56, mae=2.91, r2=0.13
    [*] predict with rf model
    rmse=4.43, mae=2.91, r2=0.18
    [*] predict with rf model
    rmse=4.54, mae=2.98, r2=0.14
    [*] predict with svr model
    rmse=4.95, mae=2.73, r2=-0.02
    [*] predict with svr model
    rmse=4.95, mae=2.73, r2=-0.02
    [*] predict with xgb model
    [16:50:00] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=4.69, mae=2.91, r2=0.08
    [*] predict with xgb model
    [16:50:00] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    rmse=4.69, mae=2.91, r2=0.08
    [*] predict with currank model
    rmse=5.05, mae=2.70, r2=-0.00
    [*] predict with currank model
    rmse=5.05, mae=2.70, r2=-0.00
    [*] predict with avgrank model
    rmse=5.85, mae=3.41, r2=-0.34
    [*] predict with avgrank model
    rmse=5.85, mae=3.41, r2=-0.34
    [*] predict with dice model
    rmse=6.65, mae=4.58, r2=-0.74
    [*] predict with dice model
    rmse=6.65, mae=4.58, r2=-0.74
    [*] predict with lasso model
    rmse=4.39, mae=2.74, r2=0.24
    [*] predict with lasso model
    rmse=4.39, mae=2.74, r2=0.24
    [*] predict with ridge model
    rmse=4.37, mae=2.71, r2=0.25
    [*] predict with ridge model
    rmse=4.37, mae=2.71, r2=0.25
    [*] predict with rf model
    rmse=4.49, mae=2.83, r2=0.21
    [*] predict with rf model
    rmse=4.30, mae=2.76, r2=0.27
    [*] predict with svr model
    rmse=5.08, mae=2.79, r2=-0.01
    [*] predict with svr model
    rmse=5.08, mae=2.79, r2=-0.01
    [*] predict with xgb model
    [16:50:02] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    rmse=4.52, mae=2.78, r2=0.20
    [*] predict with xgb model
    [16:50:02] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=4.52, mae=2.78, r2=0.20
    [*] predict with currank model
    rmse=3.85, mae=2.26, r2=-0.00
    [*] predict with currank model
    rmse=3.85, mae=2.26, r2=-0.00
    [*] predict with avgrank model
    rmse=4.36, mae=2.79, r2=-0.28
    [*] predict with avgrank model
    rmse=4.36, mae=2.79, r2=-0.28
    [*] predict with dice model
    rmse=7.17, mae=4.53, r2=-2.47
    [*] predict with dice model
    rmse=7.17, mae=4.53, r2=-2.47
    [*] predict with lasso model
    rmse=2.74, mae=1.95, r2=0.49
    [*] predict with lasso model
    rmse=2.74, mae=1.95, r2=0.49
    [*] predict with ridge model
    rmse=2.68, mae=1.84, r2=0.52
    [*] predict with ridge model
    rmse=2.68, mae=1.84, r2=0.52
    [*] predict with rf model
    rmse=3.65, mae=2.59, r2=0.10
    [*] predict with rf model
    rmse=3.38, mae=2.40, r2=0.23
    [*] predict with svr model
    rmse=3.86, mae=2.35, r2=-0.00
    [*] predict with svr model
    rmse=3.86, mae=2.35, r2=-0.00
    [*] predict with xgb model
    [16:50:04] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    rmse=2.92, mae=2.14, r2=0.43
    [*] predict with xgb model
    [16:50:04] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=2.92, mae=2.14, r2=0.43
    [*] predict with currank model
    rmse=2.68, mae=1.81, r2=-0.01
    [*] predict with currank model
    rmse=2.68, mae=1.81, r2=-0.01
    [*] predict with avgrank model
    rmse=3.05, mae=2.28, r2=-0.31
    [*] predict with avgrank model
    rmse=3.05, mae=2.28, r2=-0.31
    [*] predict with dice model
    rmse=5.99, mae=4.19, r2=-4.07
    [*] predict with dice model
    rmse=5.99, mae=4.19, r2=-4.07
    [*] predict with lasso model
    rmse=1.87, mae=1.36, r2=0.51
    [*] predict with lasso model
    rmse=1.87, mae=1.36, r2=0.51
    [*] predict with ridge model
    rmse=1.77, mae=1.24, r2=0.56
    [*] predict with ridge model
    rmse=1.77, mae=1.24, r2=0.56
    [*] predict with rf model
    rmse=3.60, mae=2.38, r2=-0.82
    [*] predict with rf model
    rmse=2.86, mae=1.92, r2=-0.15
    [*] predict with svr model
    rmse=2.67, mae=1.82, r2=-0.00
    [*] predict with svr model
    rmse=2.67, mae=1.82, r2=-0.00
    [*] predict with xgb model
    [16:50:05] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    rmse=2.28, mae=1.73, r2=0.27
    [*] predict with xgb model
    [16:50:05] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=2.28, mae=1.73, r2=0.27



```python
df_event_rmse
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
      <th>lasso</th>
      <th>ridge</th>
      <th>rf</th>
      <th>svr</th>
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
      <td>4.734161</td>
      <td>5.606406</td>
      <td>7.008766</td>
      <td>4.440868</td>
      <td>4.420807</td>
      <td>4.532616</td>
      <td>4.700286</td>
      <td>4.259167</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Indy500</td>
      <td>580</td>
      <td>225</td>
      <td>+:82,0:47,-:96</td>
      <td>6.165135</td>
      <td>6.936325</td>
      <td>7.545124</td>
      <td>5.692075</td>
      <td>5.510547</td>
      <td>5.764011</td>
      <td>6.144507</td>
      <td>5.818417</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Texas</td>
      <td>678</td>
      <td>127</td>
      <td>+:39,0:34,-:54</td>
      <td>4.257462</td>
      <td>5.478100</td>
      <td>6.726373</td>
      <td>3.777977</td>
      <td>3.755482</td>
      <td>3.963985</td>
      <td>4.134338</td>
      <td>4.018192</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Iowa</td>
      <td>696</td>
      <td>109</td>
      <td>+:42,0:28,-:39</td>
      <td>3.981609</td>
      <td>5.800698</td>
      <td>6.453077</td>
      <td>3.691717</td>
      <td>3.864756</td>
      <td>4.015204</td>
      <td>3.909159</td>
      <td>4.056615</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Pocono</td>
      <td>679</td>
      <td>126</td>
      <td>+:29,0:61,-:36</td>
      <td>2.475275</td>
      <td>2.928532</td>
      <td>5.611072</td>
      <td>2.492176</td>
      <td>2.565253</td>
      <td>3.073110</td>
      <td>2.382271</td>
      <td>2.910115</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Gateway</td>
      <td>701</td>
      <td>104</td>
      <td>+:34,0:28,-:42</td>
      <td>3.412364</td>
      <td>4.301330</td>
      <td>5.885935</td>
      <td>3.110094</td>
      <td>3.105241</td>
      <td>3.535631</td>
      <td>3.341364</td>
      <td>3.323695</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_event_r2
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
      <th>lasso</th>
      <th>ridge</th>
      <th>rf</th>
      <th>svr</th>
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
      <td>-0.000416</td>
      <td>-0.403019</td>
      <td>-1.192692</td>
      <td>0.119701</td>
      <td>0.127637</td>
      <td>0.051859</td>
      <td>0.013850</td>
      <td>0.190263</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Indy500</td>
      <td>580</td>
      <td>225</td>
      <td>+:82,0:47,-:96</td>
      <td>-0.003011</td>
      <td>-0.269636</td>
      <td>-0.502288</td>
      <td>0.145009</td>
      <td>0.198673</td>
      <td>0.165833</td>
      <td>0.003690</td>
      <td>0.106633</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Texas</td>
      <td>678</td>
      <td>127</td>
      <td>+:39,0:34,-:54</td>
      <td>-0.007943</td>
      <td>-0.668763</td>
      <td>-1.515918</td>
      <td>0.206306</td>
      <td>0.215730</td>
      <td>0.111354</td>
      <td>0.049512</td>
      <td>0.102166</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Iowa</td>
      <td>696</td>
      <td>109</td>
      <td>+:42,0:28,-:39</td>
      <td>0.000000</td>
      <td>-1.122479</td>
      <td>-1.626736</td>
      <td>0.140314</td>
      <td>0.057835</td>
      <td>0.001241</td>
      <td>0.036061</td>
      <td>-0.038031</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Pocono</td>
      <td>679</td>
      <td>126</td>
      <td>+:29,0:61,-:36</td>
      <td>-0.038432</td>
      <td>-0.453555</td>
      <td>-4.336088</td>
      <td>-0.052662</td>
      <td>-0.115300</td>
      <td>-0.532270</td>
      <td>0.038136</td>
      <td>-0.435330</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Gateway</td>
      <td>701</td>
      <td>104</td>
      <td>+:34,0:28,-:42</td>
      <td>-0.001344</td>
      <td>-0.591028</td>
      <td>-1.979225</td>
      <td>0.168199</td>
      <td>0.170793</td>
      <td>-0.160898</td>
      <td>0.039892</td>
      <td>0.050019</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_stage_rmse
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
      <th>lasso</th>
      <th>ridge</th>
      <th>rf</th>
      <th>svr</th>
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
      <td>4.747780</td>
      <td>5.870429</td>
      <td>6.158315</td>
      <td>4.852986</td>
      <td>4.960882</td>
      <td>4.975656</td>
      <td>4.760141</td>
      <td>5.130541</td>
    </tr>
    <tr>
      <th>0</th>
      <td>stage1</td>
      <td>288</td>
      <td>517</td>
      <td>+:186,0:136,-:195</td>
      <td>4.682987</td>
      <td>5.526302</td>
      <td>6.856642</td>
      <td>4.452702</td>
      <td>4.690667</td>
      <td>4.549374</td>
      <td>4.778549</td>
      <td>4.923725</td>
    </tr>
    <tr>
      <th>0</th>
      <td>stage2</td>
      <td>421</td>
      <td>384</td>
      <td>+:140,0:112,-:132</td>
      <td>4.903761</td>
      <td>5.681358</td>
      <td>7.051706</td>
      <td>4.742792</td>
      <td>4.619030</td>
      <td>4.636804</td>
      <td>4.997688</td>
      <td>4.804186</td>
    </tr>
    <tr>
      <th>0</th>
      <td>stage3</td>
      <td>547</td>
      <td>258</td>
      <td>+:91,0:89,-:78</td>
      <td>4.734420</td>
      <td>5.541856</td>
      <td>7.011065</td>
      <td>4.760917</td>
      <td>4.867947</td>
      <td>4.615638</td>
      <td>4.821914</td>
      <td>5.118272</td>
    </tr>
    <tr>
      <th>0</th>
      <td>stage4</td>
      <td>657</td>
      <td>148</td>
      <td>+:48,0:53,-:47</td>
      <td>4.914815</td>
      <td>5.646990</td>
      <td>6.450435</td>
      <td>4.512124</td>
      <td>4.559429</td>
      <td>4.432090</td>
      <td>4.953117</td>
      <td>4.690186</td>
    </tr>
    <tr>
      <th>0</th>
      <td>stage5</td>
      <td>725</td>
      <td>80</td>
      <td>+:26,0:29,-:25</td>
      <td>5.054701</td>
      <td>5.848562</td>
      <td>6.650188</td>
      <td>4.393868</td>
      <td>4.365724</td>
      <td>4.487634</td>
      <td>5.079438</td>
      <td>4.523294</td>
    </tr>
    <tr>
      <th>0</th>
      <td>stage6</td>
      <td>767</td>
      <td>38</td>
      <td>+:11,0:13,-:14</td>
      <td>3.852545</td>
      <td>4.358174</td>
      <td>7.174516</td>
      <td>2.740342</td>
      <td>2.676179</td>
      <td>3.645943</td>
      <td>3.860519</td>
      <td>2.920000</td>
    </tr>
    <tr>
      <th>0</th>
      <td>stage7</td>
      <td>789</td>
      <td>16</td>
      <td>+:4,0:4,-:8</td>
      <td>2.680951</td>
      <td>3.045832</td>
      <td>5.994789</td>
      <td>1.867283</td>
      <td>1.767208</td>
      <td>3.596909</td>
      <td>2.668724</td>
      <td>2.280309</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_stage_r2
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
      <th>lasso</th>
      <th>ridge</th>
      <th>rf</th>
      <th>svr</th>
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
      <td>-0.000316</td>
      <td>-0.529311</td>
      <td>-0.682983</td>
      <td>-0.045139</td>
      <td>-0.092129</td>
      <td>-0.130434</td>
      <td>-0.005531</td>
      <td>-0.168106</td>
    </tr>
    <tr>
      <th>0</th>
      <td>stage1</td>
      <td>288</td>
      <td>517</td>
      <td>+:186,0:136,-:195</td>
      <td>-0.000986</td>
      <td>-0.393963</td>
      <td>-1.145879</td>
      <td>0.095040</td>
      <td>-0.004272</td>
      <td>0.024175</td>
      <td>-0.042256</td>
      <td>-0.106547</td>
    </tr>
    <tr>
      <th>0</th>
      <td>stage2</td>
      <td>421</td>
      <td>384</td>
      <td>+:140,0:112,-:132</td>
      <td>-0.005719</td>
      <td>-0.349964</td>
      <td>-1.079728</td>
      <td>0.059224</td>
      <td>0.107682</td>
      <td>0.090224</td>
      <td>-0.044615</td>
      <td>0.034710</td>
    </tr>
    <tr>
      <th>0</th>
      <td>stage3</td>
      <td>547</td>
      <td>258</td>
      <td>+:91,0:89,-:78</td>
      <td>-0.019516</td>
      <td>-0.396919</td>
      <td>-1.235778</td>
      <td>-0.030960</td>
      <td>-0.077835</td>
      <td>0.034083</td>
      <td>-0.057547</td>
      <td>-0.191537</td>
    </tr>
    <tr>
      <th>0</th>
      <td>stage4</td>
      <td>657</td>
      <td>148</td>
      <td>+:48,0:53,-:47</td>
      <td>-0.008557</td>
      <td>-0.331435</td>
      <td>-0.737257</td>
      <td>0.149943</td>
      <td>0.132025</td>
      <td>0.138870</td>
      <td>-0.024338</td>
      <td>0.081527</td>
    </tr>
    <tr>
      <th>0</th>
      <td>stage5</td>
      <td>725</td>
      <td>80</td>
      <td>+:26,0:29,-:25</td>
      <td>-0.004151</td>
      <td>-0.344332</td>
      <td>-0.738105</td>
      <td>0.241244</td>
      <td>0.250933</td>
      <td>0.271718</td>
      <td>-0.014004</td>
      <td>0.195886</td>
    </tr>
    <tr>
      <th>0</th>
      <td>stage6</td>
      <td>767</td>
      <td>38</td>
      <td>+:11,0:13,-:14</td>
      <td>-0.000747</td>
      <td>-0.280672</td>
      <td>-2.470676</td>
      <td>0.493665</td>
      <td>0.517098</td>
      <td>0.228321</td>
      <td>-0.004894</td>
      <td>0.425097</td>
    </tr>
    <tr>
      <th>0</th>
      <td>stage7</td>
      <td>789</td>
      <td>16</td>
      <td>+:4,0:4,-:8</td>
      <td>-0.013774</td>
      <td>-0.308505</td>
      <td>-4.068871</td>
      <td>0.508206</td>
      <td>0.559508</td>
      <td>-0.150119</td>
      <td>-0.004548</td>
      <td>0.266583</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
