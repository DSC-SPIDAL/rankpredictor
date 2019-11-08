
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

    
retdf0.to_csv('regressors_stagedata_splitbyevent_rmse.csv')
retdf1.to_csv('regressors_stagedata_splitbyevent_r2.csv')

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
    rmse=4.45, mae=2.89, r2=0.12
    [*] predict with lasso model
    rmse=4.45, mae=2.89, r2=0.12
    [*] predict with ridge model
    rmse=4.42, mae=2.86, r2=0.13
    [*] predict with ridge model
    rmse=4.42, mae=2.86, r2=0.13
    [*] predict with rf model
    rmse=4.54, mae=3.20, r2=0.08
    [*] predict with rf model
    rmse=4.61, mae=3.20, r2=0.05
    [*] predict with svr model
    rmse=4.67, mae=3.12, r2=0.03
    [*] predict with svr model
    rmse=4.67, mae=3.12, r2=0.03
    [*] predict with xgb model
    [11:31:54] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=4.29, mae=2.99, r2=0.18
    [*] predict with xgb model
    [11:31:54] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    rmse=4.29, mae=2.99, r2=0.18
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
    rmse=5.65, mae=3.97, r2=0.16
    [*] predict with lasso model
    rmse=5.65, mae=3.97, r2=0.16
    [*] predict with ridge model
    rmse=5.49, mae=3.99, r2=0.20
    [*] predict with ridge model
    rmse=5.49, mae=3.99, r2=0.20
    [*] predict with rf model
    rmse=5.74, mae=4.27, r2=0.13
    [*] predict with rf model
    rmse=5.74, mae=4.32, r2=0.13
    [*] predict with svr model
    rmse=6.14, mae=3.90, r2=0.01
    [*] predict with svr model
    rmse=6.14, mae=3.90, r2=0.01
    [*] predict with xgb model
    [11:31:55] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=5.75, mae=4.00, r2=0.13
    [*] predict with xgb model
    [11:31:55] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=5.75, mae=4.00, r2=0.13


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


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
    rmse=3.79, mae=2.51, r2=0.20
    [*] predict with lasso model
    rmse=3.79, mae=2.51, r2=0.20
    [*] predict with ridge model
    rmse=3.77, mae=2.47, r2=0.21
    [*] predict with ridge model
    rmse=3.77, mae=2.47, r2=0.21
    [*] predict with rf model
    rmse=4.01, mae=2.84, r2=0.10
    [*] predict with rf model
    rmse=4.04, mae=2.88, r2=0.09
    [*] predict with svr model
    rmse=4.12, mae=2.57, r2=0.06
    [*] predict with svr model
    rmse=4.12, mae=2.57, r2=0.06
    [*] predict with xgb model
    [11:31:56] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=4.08, mae=2.77, r2=0.08
    [*] predict with xgb model
    [11:31:56] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    rmse=4.08, mae=2.77, r2=0.08
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
    rmse=3.74, mae=2.75, r2=0.12
    [*] predict with lasso model
    rmse=3.74, mae=2.75, r2=0.12
    [*] predict with ridge model
    rmse=3.83, mae=2.78, r2=0.07
    [*] predict with ridge model
    rmse=3.83, mae=2.78, r2=0.07
    [*] predict with rf model
    rmse=4.07, mae=2.89, r2=-0.04
    [*] predict with rf model
    rmse=4.17, mae=3.00, r2=-0.10
    [*] predict with svr model
    rmse=3.92, mae=2.61, r2=0.03
    [*] predict with svr model
    rmse=3.92, mae=2.61, r2=0.03
    [*] predict with xgb model
    [11:31:57] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=4.12, mae=3.06, r2=-0.07
    [*] predict with xgb model
    [11:31:57] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    rmse=4.12, mae=3.06, r2=-0.07
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
    rmse=2.50, mae=1.99, r2=-0.06
    [*] predict with lasso model
    rmse=2.50, mae=1.99, r2=-0.06
    [*] predict with ridge model
    rmse=2.57, mae=2.06, r2=-0.12
    [*] predict with ridge model
    rmse=2.57, mae=2.06, r2=-0.12
    [*] predict with rf model
    rmse=3.25, mae=2.43, r2=-0.79
    [*] predict with rf model
    rmse=3.22, mae=2.46, r2=-0.76
    [*] predict with svr model
    rmse=2.34, mae=1.41, r2=0.07
    [*] predict with svr model
    rmse=2.34, mae=1.41, r2=0.07
    [*] predict with xgb model
    [11:31:59] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=3.02, mae=2.37, r2=-0.55
    [*] predict with xgb model
    [11:31:59] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    rmse=3.02, mae=2.37, r2=-0.55
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
    rmse=3.09, mae=2.11, r2=0.18
    [*] predict with lasso model
    rmse=3.09, mae=2.11, r2=0.18
    [*] predict with ridge model
    rmse=3.08, mae=2.07, r2=0.19
    [*] predict with ridge model
    rmse=3.08, mae=2.07, r2=0.19
    [*] predict with rf model
    rmse=3.49, mae=2.47, r2=-0.05
    [*] predict with rf model
    rmse=3.45, mae=2.42, r2=-0.03
    [*] predict with svr model
    rmse=3.35, mae=2.21, r2=0.03
    [*] predict with svr model
    rmse=3.35, mae=2.21, r2=0.03
    [*] predict with xgb model
    [11:32:00] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=3.31, mae=2.32, r2=0.06
    [*] predict with xgb model
    [11:32:00] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    rmse=3.31, mae=2.32, r2=0.06


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

retdf0.to_csv('regressor_stagedata_splitbystage_rmse.csv')
retdf1.to_csv('regressor_stagedata_splitbystage_r2.csv')

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
    rmse=5.03, mae=3.36, r2=-0.12
    [*] predict with rf model
    rmse=5.00, mae=3.35, r2=-0.11
    [*] predict with svr model
    rmse=4.76, mae=2.92, r2=-0.00
    [*] predict with svr model
    rmse=4.76, mae=2.92, r2=-0.00
    [*] predict with xgb model
    [11:34:49] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=5.13, mae=3.46, r2=-0.17
    [*] predict with xgb model
    [11:34:49] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
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


    rmse=4.27, mae=2.93, r2=0.17
    [*] predict with lasso model
    rmse=4.27, mae=2.93, r2=0.17
    [*] predict with ridge model
    rmse=4.51, mae=3.19, r2=0.07
    [*] predict with ridge model
    rmse=4.51, mae=3.19, r2=0.07
    [*] predict with rf model
    rmse=4.60, mae=3.42, r2=0.04
    [*] predict with rf model
    rmse=4.53, mae=3.34, r2=0.06
    [*] predict with svr model
    rmse=4.74, mae=2.86, r2=-0.03
    [*] predict with svr model


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    rmse=4.74, mae=2.86, r2=-0.03
    [*] predict with xgb model
    [11:34:49] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=4.93, mae=3.49, r2=-0.11
    [*] predict with xgb model
    [11:34:49] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=4.93, mae=3.49, r2=-0.11
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
    rmse=4.54, mae=2.95, r2=0.14
    [*] predict with lasso model
    rmse=4.54, mae=2.95, r2=0.14
    [*] predict with ridge model
    rmse=4.62, mae=3.01, r2=0.11
    [*] predict with ridge model
    rmse=4.62, mae=3.01, r2=0.11
    [*] predict with rf model
    rmse=4.60, mae=3.06, r2=0.12
    [*] predict with rf model
    rmse=4.62, mae=3.07, r2=0.11
    [*] predict with svr model
    rmse=4.98, mae=2.91, r2=-0.04
    [*] predict with svr model
    rmse=4.98, mae=2.91, r2=-0.04
    [*] predict with xgb model
    [11:34:50] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=4.76, mae=3.11, r2=0.05
    [*] predict with xgb model
    [11:34:50] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=4.76, mae=3.11, r2=0.05
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


    rmse=4.70, mae=3.02, r2=-0.00
    [*] predict with lasso model
    rmse=4.70, mae=3.02, r2=-0.00
    [*] predict with ridge model
    rmse=4.76, mae=3.09, r2=-0.03
    [*] predict with ridge model
    rmse=4.76, mae=3.09, r2=-0.03
    [*] predict with rf model
    rmse=4.67, mae=3.09, r2=0.01
    [*] predict with rf model
    rmse=4.67, mae=3.12, r2=0.01
    [*] predict with svr model
    rmse=4.81, mae=2.69, r2=-0.05
    [*] predict with svr model
    rmse=4.81, mae=2.69, r2=-0.05
    [*] predict with xgb model
    [11:34:51] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=5.18, mae=3.35, r2=-0.22
    [*] predict with xgb model
    [11:34:51] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=5.18, mae=3.35, r2=-0.22
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


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    rmse=4.49, mae=2.85, r2=0.16
    [*] predict with lasso model
    rmse=4.49, mae=2.85, r2=0.16
    [*] predict with ridge model
    rmse=4.52, mae=2.89, r2=0.15
    [*] predict with ridge model
    rmse=4.52, mae=2.89, r2=0.15
    [*] predict with rf model
    rmse=4.69, mae=3.01, r2=0.08
    [*] predict with rf model
    rmse=4.71, mae=3.10, r2=0.07
    [*] predict with svr model
    rmse=4.95, mae=2.73, r2=-0.02
    [*] predict with svr model
    rmse=4.95, mae=2.73, r2=-0.02
    [*] predict with xgb model
    [11:34:52] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=4.66, mae=3.05, r2=0.09
    [*] predict with xgb model
    [11:34:52] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    rmse=4.66, mae=3.05, r2=0.09
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
    rmse=4.37, mae=2.77, r2=0.25
    [*] predict with lasso model
    rmse=4.37, mae=2.77, r2=0.25
    [*] predict with ridge model
    rmse=4.33, mae=2.70, r2=0.26
    [*] predict with ridge model
    rmse=4.33, mae=2.70, r2=0.26
    [*] predict with rf model
    rmse=4.99, mae=3.33, r2=0.02
    [*] predict with rf model
    rmse=4.82, mae=3.16, r2=0.09
    [*] predict with svr model
    rmse=5.08, mae=2.78, r2=-0.01
    [*] predict with svr model
    rmse=5.08, mae=2.78, r2=-0.01
    [*] predict with xgb model
    [11:34:54] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=4.49, mae=2.86, r2=0.21
    [*] predict with xgb model
    [11:34:54] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    rmse=4.49, mae=2.86, r2=0.21
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
    rmse=2.73, mae=1.94, r2=0.50
    [*] predict with lasso model
    rmse=2.73, mae=1.94, r2=0.50
    [*] predict with ridge model
    rmse=2.62, mae=1.82, r2=0.54
    [*] predict with ridge model
    rmse=2.62, mae=1.82, r2=0.54
    [*] predict with rf model
    rmse=4.21, mae=2.94, r2=-0.19
    [*] predict with rf model
    rmse=4.21, mae=3.03, r2=-0.20
    [*] predict with svr model
    rmse=3.85, mae=2.35, r2=0.00
    [*] predict with svr model
    rmse=3.85, mae=2.35, r2=0.00
    [*] predict with xgb model
    [11:34:55] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=3.18, mae=2.42, r2=0.32
    [*] predict with xgb model
    [11:34:55] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    rmse=3.18, mae=2.42, r2=0.32
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
    rmse=1.73, mae=1.27, r2=0.58
    [*] predict with lasso model
    rmse=1.73, mae=1.27, r2=0.58
    [*] predict with ridge model
    rmse=1.67, mae=1.16, r2=0.61
    [*] predict with ridge model
    rmse=1.67, mae=1.16, r2=0.61
    [*] predict with rf model
    rmse=3.41, mae=2.25, r2=-0.64
    [*] predict with rf model
    rmse=3.02, mae=1.83, r2=-0.28
    [*] predict with svr model
    rmse=2.69, mae=1.86, r2=-0.02
    [*] predict with svr model
    rmse=2.69, mae=1.86, r2=-0.02
    [*] predict with xgb model
    [11:34:56] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    rmse=2.71, mae=2.21, r2=-0.03
    [*] predict with xgb model
    [11:34:56] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost_1572314959925/work/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.


    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /scratch/hpda/anaconda3/envs/predictor/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    rmse=2.71, mae=2.21, r2=-0.03



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
      <td>4.447984</td>
      <td>4.415062</td>
      <td>4.544965</td>
      <td>4.672830</td>
      <td>4.293633</td>
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
      <td>5.654568</td>
      <td>5.492028</td>
      <td>5.739337</td>
      <td>6.135708</td>
      <td>5.750406</td>
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
      <td>3.786871</td>
      <td>3.771194</td>
      <td>4.013697</td>
      <td>4.117686</td>
      <td>4.076657</td>
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
      <td>3.738325</td>
      <td>3.830665</td>
      <td>4.066765</td>
      <td>3.915588</td>
      <td>4.118872</td>
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
      <td>2.498792</td>
      <td>2.572223</td>
      <td>3.249308</td>
      <td>2.340347</td>
      <td>3.019592</td>
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
      <td>3.086153</td>
      <td>3.075081</td>
      <td>3.486320</td>
      <td>3.352309</td>
      <td>3.314166</td>
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
      <td>0.116878</td>
      <td>0.129902</td>
      <td>0.053207</td>
      <td>0.025337</td>
      <td>0.177105</td>
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
      <td>0.156239</td>
      <td>0.204050</td>
      <td>0.130859</td>
      <td>0.006541</td>
      <td>0.127395</td>
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
      <td>0.202564</td>
      <td>0.209153</td>
      <td>0.090855</td>
      <td>0.057154</td>
      <td>0.075849</td>
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
      <td>0.118470</td>
      <td>0.074384</td>
      <td>-0.096279</td>
      <td>0.032888</td>
      <td>-0.070137</td>
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
      <td>-0.058258</td>
      <td>-0.121369</td>
      <td>-0.758839</td>
      <td>0.071693</td>
      <td>-0.545354</td>
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
      <td>0.180956</td>
      <td>0.186822</td>
      <td>-0.025420</td>
      <td>0.033592</td>
      <td>0.055459</td>
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
      <td>5.027814</td>
      <td>4.755545</td>
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
      <td>4.270439</td>
      <td>4.510827</td>
      <td>4.595698</td>
      <td>4.744955</td>
      <td>4.928826</td>
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
      <td>4.543633</td>
      <td>4.617479</td>
      <td>4.597718</td>
      <td>4.980098</td>
      <td>4.755490</td>
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
      <td>4.699386</td>
      <td>4.758406</td>
      <td>4.672375</td>
      <td>4.810762</td>
      <td>5.175445</td>
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
      <td>4.491933</td>
      <td>4.519653</td>
      <td>4.692315</td>
      <td>4.951282</td>
      <td>4.664548</td>
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
      <td>4.374176</td>
      <td>4.331226</td>
      <td>4.990362</td>
      <td>5.076617</td>
      <td>4.487230</td>
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
      <td>2.726765</td>
      <td>2.624649</td>
      <td>4.206698</td>
      <td>3.850762</td>
      <td>3.181817</td>
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
      <td>1.728240</td>
      <td>1.670028</td>
      <td>3.411449</td>
      <td>2.686300</td>
      <td>2.706842</td>
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
      <td>-0.107700</td>
      <td>-0.003591</td>
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
      <td>0.167609</td>
      <td>0.071259</td>
      <td>0.064858</td>
      <td>-0.027653</td>
      <td>-0.108841</td>
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
      <td>0.136575</td>
      <td>0.108281</td>
      <td>0.109080</td>
      <td>-0.037275</td>
      <td>0.054180</td>
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
      <td>-0.004484</td>
      <td>-0.029873</td>
      <td>0.007399</td>
      <td>-0.052661</td>
      <td>-0.218305</td>
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
      <td>0.157534</td>
      <td>0.147104</td>
      <td>0.071921</td>
      <td>-0.023579</td>
      <td>0.091541</td>
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
      <td>0.248030</td>
      <td>0.262724</td>
      <td>0.085499</td>
      <td>-0.012878</td>
      <td>0.208657</td>
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
      <td>0.498669</td>
      <td>0.535515</td>
      <td>-0.196821</td>
      <td>0.000179</td>
      <td>0.317380</td>
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
      <td>0.578720</td>
      <td>0.606621</td>
      <td>-0.283273</td>
      <td>-0.017823</td>
      <td>-0.033450</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
