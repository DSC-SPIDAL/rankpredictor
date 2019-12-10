#!/usr/bin/env python
# coding: utf-8

# ### stage model prediction interface
# 
# A stage, or a stint, is the section of laps between two consecutive pitstops for a car.
# The models predict the change of the ranks for the next stage when a car enters into the pit lane(or from the beginning).
# 
# There are two prediction models: 
# 
# 1. sign model to predict the sign of rank change (-1 rank improve, 0 no change, 1 rank goes worse)
# 2. value model to predict the value of rank change (integer number)
# 
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.linear_model.ridge import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn.svm.classes import SVR
from sklearn import metrics
import xgboost as xgb
import os


# In[2]:


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
    
def classifier_model(name='lr'):
    ### test learning models
    print('[*] predict with %s model'%name)
    clf = get_classifier(name)
    clf.fit(train_x, train_y)

    pred_y = clf.predict(test_x)
    score = evaluate(test_y, pred_y)
    return score


# In[3]:


#load data
suffix='-withneighbor-newfeatures-timediff'
stagedata = pd.read_csv('stage-2018%s.csv'%suffix)
stagedata.fillna(0, inplace=True)
stagedata.info()


# In[4]:


stagedata.head(5)


# ### load the pre-trained models

# In[5]:


import pickle 
eventsname = ['Phoenix','Indy500','Texas','Iowa','Pocono','Gateway']
events = set(stagedata['eventid'])
#for eventid in events:
eventid = 1   # Indy500
signmodel = 'signmodel-' + eventsname[eventid] + '-lsvc' + '.pkl'
valuemodel = 'valuemodel-' + eventsname[eventid] + '-lasso' + '.pkl'


# In[6]:


EMPTY = 100
def predict(carno, stageid):
    #
    # stageid is the id of pitstop, start from 0
    #
    #find input x <eventid, car_num, stageid>
    input_x = []
    for x in test_x:
        if ((x[1] == carno) and (x[2] == stageid)):
            input_x = x.reshape((1,-1))
            pred_y = clf.predict(input_x)
            return int(pred_y[0])
    else:
        return EMPTY


# ### test the sign of rank change prediction

# In[7]:


#load model and predict
with open(signmodel, 'rb') as fin:
    clf, test_x, test_y = pickle.load(fin)
    
yhat = clf.predict(test_x)

#check carno 12
carno=12
idx = (test_x[:,1]==carno)
_yhat = yhat[idx]

ret_y = []
for stageid in range(10):
    Y = predict(carno, stageid)
    if Y == EMPTY:
        break
    ret_y.append(Y)
    
print('trueth:', test_y[idx])
print('prediction:', _yhat)
print('prediction:', ret_y)


# ### test the value of rank change prediction

# In[8]:


#load model and predict
with open(valuemodel, 'rb') as fin:
    clf, test_x, test_y = pickle.load(fin)
    
yhat = clf.predict(test_x).astype(int)

#check carno 12
carno=12
idx = (test_x[:,1]==carno)
_yhat = yhat[idx]

ret_y = []
for stageid in range(10):
    Y = predict(carno, stageid)
    if Y == EMPTY:
        break
    ret_y.append(Y)
    
#predict(12, 3)
print('trueth:', test_y[idx])
print('prediction:', _yhat)
print('prediction:', ret_y)


# In[ ]:




