
test sota models
=====================

change settings in global_variable.py


```
#=========sota-oracle=============
#use_driverid = True
#use_dynamic_real = False
#use_time_feat = True
#use_simpledb = True
#DeepFactor (sota3)
race=Indy500; suffix=${race}-sota; sota=deepFactorX; python RankNet-QuickTest-Slim.py config/weighted-noinlap-S0LTYP0T-nocate-c60-drank-oracle.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=2 --test_event=${race}-2019 --trainmodel=${sota} --testmodel=${sota} --suffix=${suffix} --gpuid=0 --trainrace=${race} --context_len=60 --prediction_len=2 --use_cat_feat=1 --weight_coef=1 2>&1 | tee logs/${suffix}_${sota}.log;

# train with gpu, predict with cpu only(gpuid=-2)
race=Indy500; suffix=${race}-sota; sota=deepState; python RankNet-QuickTest-Slim.py config/weighted-noinlap-S0LTYP0T-nocate-c60-drank-oracle.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=2 --test_event=${race}-2019 --trainmodel=${sota} --testmodel=${sota} --suffix=${suffix} --gpuid=0 --trainrace=${race} --context_len=60 --prediction_len=2 --use_cat_feat=1 --weight_coef=1 2>&1 | tee logs/${suffix}_${sota}.log;

race=Indy500; suffix=${race}-sota; sota=nbeats; python RankNet-QuickTest-Slim.py config/weighted-noinlap-S0LTYP0T-nocate-c60-drank-oracle.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=2 --test_event=${race}-2019 --trainmodel=${sota} --testmodel=${sota} --suffix=${suffix} --gpuid=0 --trainrace=${race} --context_len=60 --prediction_len=2 --use_cat_feat=1 --weight_coef=1 2>&1 | tee logs/${suffix}_${sota}.log;

race=Indy500; suffix=${race}-sota-oracle; sota=deepAR; python RankNet-QuickTest-Slim.py config/weighted-noinlap-S0000000-nocate-c60-drank-oracle.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=2 --test_event=${race}-2019 --trainmodel=${sota} --testmodel=${sota} --suffix=${suffix} --gpuid=0 --trainrace=${race} --context_len=60 --prediction_len=2 --use_cat_feat=1 --weight_coef=1 2>&1 | tee logs/${suffix}_${sota}.log;

#=========sota-oracle=============
#use_driverid = True
#use_dynamic_real = True
#use_time_feat = False
#use_simpledb = False
#DeepFactor-Oracle  (sota5)
race=Indy500; suffix=${race}-sota-oracle; sota=deepFactorX; python RankNet-QuickTest-Slim.py config/weighted-noinlap-S0LTYP0T-nocate-c60-drank-oracle.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=2 --test_event=${race}-2019 --trainmodel=${sota} --testmodel=${sota} --suffix=${suffix} --gpuid=0 --trainrace=${race} --context_len=60 --prediction_len=2 --use_cat_feat=1 --weight_coef=1 2>&1 | tee logs/${suffix}_${sota}.log;

# train with gpu, predict with cpu only
race=Indy500; suffix=${race}-sota-oracle; sota=deepState; python RankNet-QuickTest-Slim.py config/weighted-noinlap-S0LTYP0T-nocate-c60-drank-oracle.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=2 --test_event=${race}-2019 --trainmodel=${sota} --testmodel=${sota} --suffix=${suffix} --gpuid=0 --trainrace=${race} --context_len=60 --prediction_len=2 --use_cat_feat=1 --weight_coef=1 2>&1 | tee logs/${suffix}_${sota}.log;

race=Indy500; suffix=${race}-sota-oracle; sota=deepState; python RankNet-QuickTest-Slim.py config/weighted-noinlap-S0000000-nocate-c60-drank-oracle.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=2 --test_event=${race}-2019 --trainmodel=${sota} --testmodel=${sota} --suffix=${suffix} --gpuid=0 --trainrace=${race} --context_len=60 --prediction_len=2 --use_cat_feat=1 --weight_coef=1 2>&1 | tee logs/${suffix}_${sota}.log;

race=Indy500; suffix=${race}-sota-oracle; sota=deepAR-Oracle; python RankNet-QuickTest-Slim.py config/weighted-noinlap-S0000000-nocate-c60-drank-oracle.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=2 --test_event=${race}-2019 --trainmodel=${sota} --testmodel=${sota} --suffix=${suffix} --gpuid=0 --trainrace=${race} --context_len=60 --prediction_len=2 --use_cat_feat=1 --weight_coef=1 2>&1 | tee logs/${suffix}_${sota}.log;

```
