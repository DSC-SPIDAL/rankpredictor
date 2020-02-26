#!/usr/bin/env python
# coding: utf-8

# ### collect metrics from gluonts results
# 
# input: run, metric, value
# 
# log-deepAR-vspeed-nolap-indy-f1min-t5-e1000-r1,    mean_wQuantileLoss, 0.03997013105238816,
# 
# output: data frame of run# x # metric

# In[2]:


import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import sys

# In[13]:

if len(sys.argv) != 2:
    print('usage: collect_metric.py <inputfile>')
    sys.exit(-1)

inputfile=sys.argv[1]

dataset = pd.read_csv(inputfile)

#dataset.info()


# In[14]:


#dataset.head(10)


# In[18]:


runids =dataset.runid.unique()
print(len(runids))
metrics = dataset.metric.unique()
print(metrics)


# In[28]:


#bulid map
idmap={key:idx for idx, key in enumerate(runids)}
metricmap={key:idx for idx, key in enumerate(metrics)}

#build metric matrix
mat = np.zeros((len(runids), len(metrics)))
for index, row in dataset.iterrows():
    mat[idmap[row['runid']], metricmap[row['metric']]] = row['value']
    

#build the data frame
met = pd.DataFrame(data=mat, columns=metrics)


# In[30]:


met.insert(0,'runid', runids)


# In[32]:


met


# In[33]:


met.to_csv(inputfile + '-result.csv', index=False)


# In[ ]:




