#!/usr/bin/env python
# coding: utf-8

# ## QuickTest Slim
# 
# based on : RankNet-QuickTest-Joint
# 
#     makedb laptime
#     makedb gluonts
#     train model
#     evaluate model
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import random
import mxnet as mx
from mxnet import gluon
import pickle
import json
import copy
from gluonts.dataset.common import ListDataset
from gluonts.dataset.util import to_pandas
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from pathlib import Path
import configparser

from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deep_factor import DeepFactorEstimator
from gluonts.model.deepstate import DeepStateEstimator
from gluonts.trainer import Trainer
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator, MultivariateEvaluator
from gluonts.model.predictor import Predictor
from gluonts.model.prophet import ProphetPredictor
from gluonts.model.r_forecast import RForecastPredictor
from gluonts.dataset.util import to_pandas

from gluonts.distribution.neg_binomial import NegativeBinomialOutput
from gluonts.distribution.student_t import StudentTOutput
from gluonts.distribution.multivariate_gaussian import MultivariateGaussianOutput

from indycar.model.NaivePredictor import NaivePredictor
from indycar.model.deeparw import DeepARWeightEstimator

#import indycar.model.stint_simulator_shortterm_pitmodel as stint
import indycar.model.quicktest_simulator as stint

# import all functions 
#from indycar.model.global_variables import _hi
import indycar.model.global_variables as gvar
from indycar.model.quicktest_modules import *


# ## run

# In[2]:
### run
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

# logging configure
import logging.config
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

# cmd argument parser
usage = 'RankNet-QuickTest.py <configfile> [options]'
parser = OptionParser(usage)
parser.add_option("--forecast_mode", dest="forecast_mode", default="")
parser.add_option("--trainmodel", default='', dest="trainmodel")
parser.add_option("--testmodel", default='', dest="testmodel")
parser.add_option("--joint_train", action="store_true", default=False, dest="joint_train")
parser.add_option("--loopcnt", default=-1,type='int',  dest="loopcnt")
parser.add_option("--gpuid", default=-1,type='int',  dest="gpuid")
parser.add_option("--pitmodel_bias", default=-1, type='int', dest="pitmodel_bias")
parser.add_option("--year", default='', dest="year")
parser.add_option("--test_event", default='', dest="test_event")
parser.add_option("--suffix", default='', dest="suffix")
parser.add_option("--dataroot", default='test/', dest="dataroot")

opt, args = parser.parse_args()
print(len(args), opt.joint_train)

#check validation
if len(args) != 1:
    logger.error(globals()['__doc__'] % locals())
    sys.exit(-1)

configfile = args[0]

base=os.path.basename(configfile)
configname = os.path.splitext(base)[0]

WorkRootDir = 'QuickTestOutput'
#configname = 'weighted-noinlap-nopitage-nocate-c60-drank'
#configfile = f'{configname}.ini'

if not os.path.exists(configfile):
    print('config file not exists error:', configfile)
    sys.exit(-1)

if configfile != '':
    config = configparser.RawConfigParser()
    #config.read(WorkRootDir + '/' + configfile)
    config.read(configfile)

    #set them back
    section = "RankNet-QuickTest"
    
    _savedata = config.getboolean(section, "_savedata")
    _skip_overwrite = config.getboolean(section, "_skip_overwrite")
    _inlap_status = config.getint(section, "_inlap_status") #0
    _feature_mode = config.getint(section, "_feature_mode") #FEATURE_STATUS
    _featureCnt = config.getint(section, "_featureCnt") #9
    freq = config.get(section, "freq") #"1min"
    _train_len = config.getint(section, "_train_len") #40
    prediction_length = config.getint(section, "prediction_length") #2
    context_ratio = config.getfloat(section, "context_ratio") #0.
    context_length =  config.getint(section, "context_length") #40
    
    dataset= config.get(section, "dataset") #'rank'
    epochs = config.getint(section, "epochs") #1000
    gpuid = config.getint(section, "gpuid") #5
    _use_weighted_model = config.getboolean(section, "_use_weighted_model")
    trainmodel = config.get(section, "trainmodel") #'deepARW-Oracle' if _use_weighted_model else 'deepAR-Oracle'
    
    _use_cate_feature = config.getboolean(section, "_use_cate_feature")
    
    distroutput = config.get(section, "distroutput") #'student'
    batch_size = config.getint(section, "batch_size") #32
    loopcnt = config.getint(section, "loopcnt") #2
    _test_event = config.get(section, "_test_event") #'Indy500-2018'
    testmodel = config.get(section, "testmodel") #'oracle'
    pitmodel = config.get(section, "pitmodel") #'oracle'
    year = config.get(section, "year") #'2018'
    
    contextlen = context_length
    use_feat_static = _use_cate_feature 

    #config1 = get_config()
    
else:
    print('Warning, please use config file')
    sys.exit(0)


# In[3]:
# new added parameters
_draw_figs = False
_test_train_len = 40
_joint_train = False
_pitmodel_bias = 0
#shortterm, stint
#_forecast_mode = 'stint'
_forecast_mode = 'shortterm'

#load arguments overwites
if opt.forecast_mode != '':
    _forecast_mode = opt.forecast_mode
if opt.trainmodel != '':
    trainmodel = opt.trainmodel
if opt.testmodel != '':
    testmodel = opt.testmodel
if opt.joint_train != False:
    _joint_train = True
if opt.gpuid >= 0:
    gpuid = opt.gpuid
if opt.loopcnt > 0:
    loopcnt = opt.loopcnt
if opt.pitmodel_bias >= 0:
    _pitmodel_bias = opt.pitmodel_bias
if opt.year != '':
    year = opt.year
if opt.test_event != '':
    _test_event = opt.test_event
if opt.suffix:
    _debugstr = f'-{opt.suffix}'
else:
    _debugstr = ''
dataroot = opt.dataroot

#discard year
year = _test_event

if testmodel == 'pitmodel':
    testmodel = 'pitmodel%s'%(_pitmodel_bias if _pitmodel_bias!=0 else '')

#featurestr = {FEATURE_STATUS:'nopitage',FEATURE_PITAGE:'pitage',FEATURE_LEADERPITCNT:'leaderpitcnt'}
#cur_featurestr = featurestr[_feature_mode]
print('current configfile:', configfile)
cur_featurestr = decode_feature_mode(_feature_mode)
print('feature_mode:', _feature_mode, cur_featurestr)
print('testmodel:', testmodel)
print('pitmodel:', pitmodel)
#print('year:', year)
print('test_event:', _test_event)


# In[4]:


#
# string map
#
inlapstr = {0:'noinlap',1:'inlap',2:'outlap'}
weightstr = {True:'weighted',False:'noweighted'}
catestr = {True:'cate',False:'nocate'}

#
# input data parameters
#
#events = ['Phoenix','Indy500','Texas','Iowa','Pocono','Gateway']
#events_totalmiles=[256,500,372,268,500,310]
#events_laplen = [1.022,2.5,1.5,0.894,2.5,1.25]
events_info = {
    'Phoenix':(256, 1.022, 250),'Indy500':(500,2.5,200),'Texas':(372,1.5,248),
    'Iowa':(268,0.894,300),'Pocono':(500,2.5,200),'Gateway':(310,1.25,248)
}

years = ['2013','2014','2015','2016','2017','2018','2019']
events = [f'Indy500-{x}' for x in years]

events.extend(['Phoenix-2018','Texas-2018','Texas-2019','Pocono-2018','Pocono-2019','Iowa-2018','Iowa-2019',
              'Gateway-2018','Gateway-2019'])

events_id={key:idx for idx, key in enumerate(events)}

#dbid = f'Indy500_{years[0]}_{years[-1]}_v{_featureCnt}_p{_inlap_status}'
dbid = f'IndyCar_d{len(events)}_v{_featureCnt}_p{_inlap_status}'
_dataset_id = '%s-%s'%(inlapstr[_inlap_status], cur_featurestr)

_train_events = [events_id[x] for x in [f'Indy500-{x}' for x in ['2013','2014','2015','2016','2017']]]
#
# internal parameters
#
distr_outputs ={'student':StudentTOutput(),
                'negbin':NegativeBinomialOutput()
                }
distr_output = distr_outputs[distroutput]

#
#
#
experimentid = f'{weightstr[_use_weighted_model]}-{inlapstr[_inlap_status]}-{cur_featurestr}-{catestr[_use_cate_feature]}-c{context_length}{_debugstr}'

#
#
#
outputRoot = f"{WorkRootDir}/{experimentid}/"
version = f'IndyCar-d{len(events)}-endlap'

# standard output file names
LAPTIME_DATASET = f'{outputRoot}/laptime_rank_timediff_pit-oracle-{dbid}.pickle' 
STAGE_DATASET = f'{outputRoot}/stagedata-{dbid}.pickle' 
# year related
SIMULATION_OUTFILE = f'{outputRoot}/{_test_event}/{_forecast_mode}-dfout-{trainmodel}-indy500-{dataset}-{inlapstr[_inlap_status]}-{cur_featurestr}-{testmodel}-l{loopcnt}-alldata.pickle'
EVALUATION_RESULT_DF = f'{outputRoot}/{_test_event}/{_forecast_mode}-evaluation_result_d{dataset}_m{testmodel}.csv'
LONG_FORECASTING_DFS = f'{outputRoot}/{_test_event}/{_forecast_mode}-long_forecasting_dfs_d{dataset}_m{testmodel}.pickle'
FORECAST_FIGS_DIR = f'{outputRoot}/{_test_event}/{_forecast_mode}-forecast-figs-d{dataset}_m{testmodel}/'


# In[5]:


# set global vars
gvar._savedata =                            _savedata
gvar._skip_overwrite =                      _skip_overwrite
gvar._inlap_status =                        _inlap_status
gvar._feature_mode =                        _feature_mode
gvar._featureCnt =                          _featureCnt
gvar.freq  =                                freq 
gvar._train_len =                           _train_len
gvar.prediction_length =                    prediction_length
gvar.context_ratio =                        context_ratio
gvar.context_length =                       context_length
gvar.contextlen =                           contextlen
gvar.dataset =                              dataset
gvar.epochs =                               epochs
gvar.gpuid =                                gpuid
gvar._use_weighted_model =                  _use_weighted_model
gvar.trainmodel =                           trainmodel
gvar._use_cate_feature =                    _use_cate_feature
gvar.use_feat_static =                      use_feat_static
gvar.distroutput =                          distroutput
gvar.batch_size =                           batch_size
gvar.loopcnt =                              loopcnt
gvar._test_event =                          _test_event
gvar.testmodel =                            testmodel
gvar.pitmodel =                             pitmodel
gvar.year =                                year
gvar._forecast_mode = _forecast_mode
gvar._test_train_len = _test_train_len
gvar._joint_train = _joint_train
gvar._pitmodel_bias = _pitmodel_bias
gvar.events = events
gvar.events_id  = events_id
gvar.events_info = events_info
gvar._train_events = _train_events

gvar.maxlap = get_event_info(_test_event)[2]
gvar.dbid = dbid
gvar.LAPTIME_DATASET = LAPTIME_DATASET


# ### 1. make laptime dataset

# In[6]:


stagedata = {}
global_carids = {}
os.makedirs(outputRoot, exist_ok=True)
os.makedirs(f'{outputRoot}/{_test_event}', exist_ok=True)

#check the dest files first
if _skip_overwrite and os.path.exists(LAPTIME_DATASET) and os.path.exists(STAGE_DATASET):
        #
        # load data
        #
        print('Load laptime and stage dataset:',LAPTIME_DATASET, STAGE_DATASET)
        with open(LAPTIME_DATASET, 'rb') as f:
            global_carids, laptime_data = pickle.load(f, encoding='latin1') 
        with open(STAGE_DATASET, 'rb') as f:
            stagedata = pickle.load(f, encoding='latin1') 
    
else:    
    cur_carid = 0
    for event in events:
        #dataid = f'{event}-{year}'
        #alldata, rankdata, acldata, flagdata
        stagedata[event] = load_data(event)

        alldata, rankdata, acldata, flagdata = stagedata[event]
        carlist = set(acldata['car_number'])
        laplist = set(acldata['completed_laps'])
        print('%s: carno=%d, lapnum=%d'%(event, len(carlist), len(laplist)))

        #build the carid map
        for car in carlist:
            if car not in global_carids:
                global_carids[car] = cur_carid
                cur_carid += 1

    laptime_data = get_laptime_dataset(stagedata, inlap_status = _inlap_status)

    if _savedata:
        import pickle
        #stintdf.to_csv('laptime-%s.csv'%year)
        #savefile = outputRoot + f'laptime_rank_timediff_pit-oracle-{dbid}.pickle' 
        savefile = LAPTIME_DATASET
        print(savefile)
        with open(savefile, 'wb') as f:
            #pack [global_carids, laptime_data]
            savedata = [global_carids, laptime_data]
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(savedata, f, pickle.HIGHEST_PROTOCOL)

        #savefile = outputRoot + f'stagedata-{dbid}.pickle' 
        savefile = STAGE_DATASET
        print(savefile)
        with open(savefile, 'wb') as f:
            #pack [global_carids, laptime_data]
            savedata = stagedata
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(savedata, f, pickle.HIGHEST_PROTOCOL)    
        
#update global var
gvar.global_carids = global_carids


# ### 2. make gluonts db

# In[7]:


outdir = outputRoot + _dataset_id
os.makedirs(outdir, exist_ok=True)

if dataset == 'laptime':
    subdir = 'laptime-indy500'
    os.makedirs(f'{outdir}/{subdir}', exist_ok=True)
    _run_ts = COL_LAPTIME
elif dataset == 'timediff':
    subdir = 'timediff-indy500'
    os.makedirs(f'{outdir}/{subdir}', exist_ok=True)
    _run_ts = COL_TIMEDIFF
elif dataset == 'rank':
    subdir = 'rank-indy500'
    os.makedirs(f'{outdir}/{subdir}', exist_ok=True)
    _run_ts = COL_RANK
else:
    print('error, dataset not support: ', dataset)
    
_task_dir = f'{outdir}/{subdir}/'

#
#dbname, train_ds, test_ds = makedbs()   
#
useeid = False
interpolate = False
#ipstr = '-ip' if interpolate else '-noip'
ipstr = '%s-%s'%('ip' if interpolate else 'noip', 'eid' if useeid else 'noeid')
jointstr = '-joint' if _joint_train else ''

dbname = _task_dir + f'gluontsdb-{dataset}-oracle-{ipstr}-all-all-f{freq}-t{prediction_length}-r{_test_event}-indy-{year}{jointstr}.pickle'
laptimedb = _task_dir + f'gluontsdb-{dataset}-oracle-{ipstr}-all-all-f{freq}-t{prediction_length}-r{_test_event}-indy-{year}-newlaptimedata.pickle'

#check the dest files first
if _skip_overwrite and os.path.exists(dbname) and os.path.exists(laptimedb):
        print('Load Gluonts Dataset:',dbname)
        with open(dbname, 'rb') as f:
            freq, prediction_length, cardinality, train_ds, test_ds = pickle.load(f, encoding='latin1') 
        print('.......loaded data, freq=', freq, 'prediction_length=', prediction_length)
        print('Load New Laptime Dataset:',laptimedb)
        with open(laptimedb, 'rb') as f:
            prepared_laptimedata = pickle.load(f, encoding='latin1') 
        
else:
    if useeid:
        cardinality = [len(global_carids), len(laptime_data)]
    else:
        cardinality = [len(global_carids)]

    prepared_laptimedata = prepare_laptimedata(laptime_data,
                           prediction_length, freq, test_event = _test_event,
                           train_ratio=0, context_ratio = 0.,shift_len = prediction_length)

    train_ds, test_ds,_,_ = make_dataset_byevent(prepared_laptimedata, 
                                        prediction_length,freq,
                                         useeid=useeid, run_ts=_run_ts,
                                        test_event=_test_event, log_transform =False,
                                        context_ratio=0, train_ratio = 0, joint_train = _joint_train)    


    if _savedata:
        print('Save Gluonts Dataset:',dbname)
        with open(dbname, 'wb') as f:
            savedata = [freq, prediction_length, cardinality, train_ds, test_ds]
            pickle.dump(savedata, f, pickle.HIGHEST_PROTOCOL)

        print('Save preprocessed laptime Dataset:',laptimedb)
        with open(laptimedb, 'wb') as f:
            pickle.dump(prepared_laptimedata, f, pickle.HIGHEST_PROTOCOL)
        


# ### 3. train the model

# In[8]:


id='oracle'
run=1
runid=f'{trainmodel}-{dataset}-all-indy-f1min-t{prediction_length}-e{epochs}-r{run}_{id}_t{prediction_length}'
modelfile = _task_dir + runid

if _skip_overwrite and os.path.exists(modelfile):
    print('Model checkpoint found at:',modelfile)

else:
    #get target dim
    entry = next(iter(train_ds))
    target_dim = entry['target'].shape
    target_dim = target_dim[0] if len(target_dim) > 1 else 1
    print('target_dim:%s', target_dim)

    estimator = init_estimator(trainmodel, gpuid, 
            epochs, batch_size,target_dim, distr_output = distr_output,use_feat_static = use_feat_static)

    predictor = estimator.train(train_ds)

    if _savedata:
        os.makedirs(modelfile, exist_ok=True)

        print('Start to save the model to %s', modelfile)
        predictor.serialize(Path(modelfile))
        print('End of saving the model.')


# 
# ### 4. evaluate the model

# In[9]:


lapmode = _inlap_status
fmode = _feature_mode
runts = dataset
mid = f'{testmodel}-%s-%s-%s-%s'%(runts, year, inlapstr[lapmode], cur_featurestr)
datasetid = outputRoot + _dataset_id

if _skip_overwrite and os.path.exists(SIMULATION_OUTFILE):
    print('Load Simulation Results:',SIMULATION_OUTFILE)
    with open(SIMULATION_OUTFILE, 'rb') as f:
        dfs,acc,ret,pret = pickle.load(f, encoding='latin1') 
    print('.......loaded data, ret keys=', ret.keys())
    
    
    # init the stint module
    #
    # in test mode, set all train_len = 40 to unify the evaluation results
    #
    init_simulation(datasetid, _test_event, 'rank',stint.COL_RANK,'rank',prediction_length, 
                    pitmodel=pitmodel, inlapmode=lapmode,featuremode =fmode,
                    train_len = _test_train_len, pitmodel_bias= _pitmodel_bias, prepared_laptimedata = prepared_laptimedata)    

else:
    #run simulation
    acc, ret, pret = {}, {}, {}

    #lapmode = _inlap_status
    #fmode = _feature_mode
    #runts = dataset
    #mid = f'{testmodel}-%s-%s-%s-%s'%(runts, year, inlapstr[lapmode], featurestr[fmode])

    if runts == 'rank':
        acc[mid], ret[mid] = simulation(datasetid, _test_event, 
                    'rank',stint.COL_RANK,'rank',
                   prediction_length, stint.MODE_ORACLE,loopcnt, 
                      pitmodel=pitmodel, model=testmodel, inlapmode=lapmode,featuremode =fmode,
                    train_len = _test_train_len, forecastmode = _forecast_mode, joint_train = _joint_train,
                    pitmodel_bias= _pitmodel_bias, prepared_laptimedata = prepared_laptimedata,
                    epochs = epochs)
    else:
        acc[mid], ret[mid] = simulation(datasetid, _test_event, 
                    'timediff',stint.COL_TIMEDIFF,'timediff2rank',
                   prediction_length, stint.MODE_ORACLE,loopcnt, 
                      pitmodel=pitmodel, model=testmodel, inlapmode=lapmode,featuremode =fmode,
                    train_len = _test_train_len, forecastmode = _forecast_mode, joint_train = _joint_train,
                    pitmodel_bias= _pitmodel_bias, prepared_laptimedata = prepared_laptimedata,
                    epochs = epochs)

    if _forecast_mode == 'shortterm':
        allsamples, alltss = get_allsamples(ret[mid], year=year)
        _, pret[mid]= prisk_direct_bysamples(allsamples, alltss)
        print(pret[mid])
    

    dfs={}

    mode=1
    df = get_alldf_mode(ret[mid], year=year,mode=mode, forecast_mode = _forecast_mode)
    name = '%s_%s'%(testmodel, 'mean' if mode==1 else ('mode' if mode==0 else 'median'))
    if year not in dfs:
        dfs[year] = {}
    dfs[year][name] = df

    _trim = 0
    _include_final = True
    _include_stintlen = True
    include_str = '1' if _include_final else '0'
    stint_str = '1' if _include_stintlen else ''            
    #simulation_outfile=outputRoot + f'shortterm-dfout-oracle-indy500-{dataset}-{inlapstr[_inlap_status]}-{featurestr[_feature_mode]}-2018-oracle-l{loopcnt}-alldata-weighted.pickle'

    with open(SIMULATION_OUTFILE, 'wb') as f:
        savedata = [dfs,acc,ret,pret]
        pickle.dump(savedata, f, pickle.HIGHEST_PROTOCOL)
        
#alias
ranknetdf = dfs   
ranknet_ret = ret


# In[10]:


# ### 5. final evaluation

# In[11]:


if _skip_overwrite and os.path.exists(EVALUATION_RESULT_DF):
    print('Load Evaluation Results:',EVALUATION_RESULT_DF)
    oracle_eval_result = pd.read_csv(EVALUATION_RESULT_DF)

else:    
    # get pit laps, pit-covered-laps
    # pitdata[event] = [pitlaps, pitcoveredlaps]
    with open('pitcoveredlaps-alldata-g1.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        pitdata = pickle.load(f, encoding='latin1') 


    ##-------------------------------------------------------------------------------
    if _forecast_mode == 'shortterm':

        #
        # Model,SignAcc,MAE,50-Risk,90-Risk
        # 
        cols = ['Year','Model','ExpID','laptype','Top1Acc','SignAcc', 'MAE','50-Risk','90-Risk']
        plen = prediction_length
        usemeanstr='mean'

        #load data
        # dfs,acc,ret,pret

        retdata = []

        #oracle
        dfx = ret[mid]
        allsamples, alltss = get_allsamples(dfx, year=year)
        #_, pret[mid]= prisk_direct_bysamples(ret[mid][0][1], ret[mid][0][2])
        _, prisk_vals = prisk_direct_bysamples(allsamples, alltss)

        dfout = do_rerank(ranknetdf[year][f'{testmodel}_mean'])
        accret = stint.get_evalret_shortterm(dfout)[0]
        #fsamples, ftss = runs2samples_ex(ranknet_ret[f'oracle-RANK-{year}-inlap-nopitage'],[])
        #_, prisk_vals = prisk_direct_bysamples(fsamples, ftss)
        retdata.append([year,f'{testmodel}',configname,'all', accret[0], accret[4], accret[1], prisk_vals[1], prisk_vals[2]])

        for laptype in ['normal','pit']:
            # select the set
            pitcoveredlaps = pitdata[_test_event][1]
            normallaps = set([x for x in range(1,201)]) - pitcoveredlaps

            if laptype == 'normal':
                sellaps = normallaps
                clearlaps = pitcoveredlaps
            else:
                sellaps = pitcoveredlaps
                clearlaps = normallaps


            # pitcoveredlaps start idx = 1
            startlaps = [x-plen-1 for x in sellaps]
            #sellapidx = np.array([x-1 for x in sellaps])
            clearidx = np.array([x-1 for x in clearlaps])
            print('sellaps:', len(sellaps), 'clearlaps:',len(clearlaps))

            #oracle
            #outfile=f'shortterm-dfout-ranknet-indy500-rank-inlap-nopitage-20182019-oracle-l10-alldata-weighted.pickle'
            #_all = load_dfout_all(outfile)[0]
            #ranknetdf, acc, ret, pret = _all[0],_all[1],_all[2],_all[3]

            dfout = do_rerank(ranknetdf[year][f'{testmodel}_mean'])

            allsamples, alltss = get_allsamples(dfx, year=year)


            allsamples, alltss = clear_samples(allsamples, alltss,clearidx)

            _, prisk_vals = prisk_direct_bysamples(allsamples, alltss)

            dfout = dfout[dfout['startlap'].isin(startlaps)]
            accret = stint.get_evalret_shortterm(dfout)[0]

            print(year, laptype,f'RankNet-{testmodel}',accret[0], accret[4], accret[1], prisk_vals[1], prisk_vals[2])
            retdata.append([year, f'{testmodel}',configname,laptype, accret[0], accret[4], accret[1], prisk_vals[1], prisk_vals[2]])
            
    ##-------------------------------------------------------------------------------
    elif _forecast_mode == 'stint':
        if testmodel == 'oracle':
            #datafile=f'stint-dfout-mlmodels-indy500-tr2013_2017-te2018_2019-end1-oracle-t0-tuned.pickle'
            datafile=f'{dataroot}/stint-dfout-mlmodels-{version}-end1-oracle-t0-tuned.pickle'
        else:
            #datafile=f'stint-dfout-mlmodels-indy500-tr2013_2017-te2018_2019-end1-normal-t0-tuned.pickle'
            datafile=f'{dataroot}/stint-dfout-mlmodels-{version}-end1-normal-t0-tuned.pickle'
        #preddf = load_dfout(outfile)
        with open(datafile, 'rb') as f:
            preddf = pickle.load(f, encoding='latin1')[0] 
        #preddf_oracle = load_dfout(outfile)
        ranknet_ret = ret 

        #discard old year
        #year <- _test_event

        errlist = {}
        errcnt, errlist[year] = cmp_df(ranknetdf[year][f'{testmodel}_mean'], preddf[_test_event]['lasso'])
        
        retdata = []
        #
        # Model,SignAcc,MAE,50-Risk,90-Risk
        # 
        cols = ['Year','Model','ExpID','laptype','SignAcc','MAE','50-Risk','90-Risk']
        models = {'currank':'CurRank','rf':'RandomForest','svr_lin':'SVM','xgb':'XGBoost'}

        for clf in ['currank','rf','svr_lin','xgb']:
            print('year:',year,'clf:',clf)
            dfout, accret = eval_sync(preddf[_test_event][clf],errlist[year])
            fsamples, ftss = df2samples_ex(dfout)
            _, prisk_vals = prisk_direct_bysamples(fsamples, ftss)

            retdata.append([_test_event,models[clf],configname,'all', accret[0], accret[1], prisk_vals[1], prisk_vals[2]])
            

        dfout, accret = eval_sync(ranknetdf[year][f'{testmodel}_mean'], errlist[year],force2int=True)
        #fsamples, ftss = df2samples(dfout)
        fsamples, ftss = runs2samples(ranknet_ret[mid],errlist[f'{year}'])
        _, prisk_vals = prisk_direct_bysamples(fsamples, ftss)
        retdata.append([_test_event,f'{testmodel}',configname,'all',accret[0], accret[1], prisk_vals[1], prisk_vals[2]])


        # split evaluation
        if True:
            for laptype in ['normal','pit']:
                # select the set
                pitcoveredlaps = pitdata[_test_event][1]
                normallaps = set([x for x in range(1,201)]) - pitcoveredlaps

                if laptype == 'normal':
                    sellaps = normallaps
                    clearlaps = pitcoveredlaps
                else:
                    sellaps = pitcoveredlaps
                    clearlaps = normallaps


                # pitcoveredlaps start idx = 1
                #startlaps = [x-plen-1 for x in sellaps]
                startlaps = [x-1 for x in sellaps]
                #sellapidx = np.array([x-1 for x in sellaps])
                clearidx = np.array([x-1 for x in clearlaps])
                print('sellaps:', len(sellaps), 'clearlaps:',len(clearlaps))

                # evaluation start
                for clf in ['currank','rf','svr_lin','xgb']:
                    dfout, accret = eval_sync(preddf[_test_event][clf],errlist[year])

                    dfout = dfout[dfout['endlap'].isin(startlaps)]
                    accret = stint.get_evalret(dfout)[0]

                    fsamples, ftss = df2samples_ex(dfout)

                    #fsamples, ftss = clear_samples(fsamples, ftss, clearidx)

                    _, prisk_vals = prisk_direct_bysamples(fsamples, ftss)


                    #dfout = dfout[dfout['endlap'].isin(startlaps)]
                    #accret = stint.get_evalret(dfout)[0]

                    retdata.append([_test_event,models[clf],configname,laptype, accret[0], accret[1], prisk_vals[1], prisk_vals[2]])
                    

                dfout, accret = eval_sync(ranknetdf[year][f'{testmodel}_mean'], errlist[year],force2int=True)
                dfout = dfout[dfout['endlap'].isin(startlaps)]
                accret = stint.get_evalret(dfout)[0]


                #fsamples, ftss = df2samples(dfout)
                fsamples, ftss = runs2samples(ranknet_ret[mid],errlist[f'{year}'])

                #fsamples, ftss = clear_samples(fsamples, ftss, clearidx)

                _, prisk_vals = prisk_direct_bysamples(fsamples, ftss)

                #dfout = dfout[dfout['endlap'].isin(startlaps)]
                #accret = stint.get_evalret(dfout)[0]

                retdata.append([_test_event,f'{testmodel}',configname,laptype,accret[0], accret[1], prisk_vals[1], prisk_vals[2]])


    # end of evaluation
    oracle_eval_result = pd.DataFrame(data=retdata, columns=cols)
    if _savedata:
        oracle_eval_result.to_csv(EVALUATION_RESULT_DF)    


# ### 6. Draw forecasting results

# In[12]:


if _forecast_mode == 'shortterm' and _joint_train == False:
    if _skip_overwrite and os.path.exists(LONG_FORECASTING_DFS):
        fname = LONG_FORECASTING_DFS
        print('Load Long Forecasting Data:',fname)
        with open(fname, 'rb') as f:
            alldata = pickle.load(f, encoding='latin1') 
        print('.......loaded data, alldata keys=', alldata.keys())

    else:    

        oracle_ret = ret    
        mid = f'{testmodel}-%s-%s-%s-%s'%(runts, year, inlapstr[lapmode], cur_featurestr)
        print('eval mid:', mid, f'{testmodel}_ret keys:', ret.keys())

        ## init predictor
        _predictor =  NaivePredictor(freq= freq, prediction_length = prediction_length)

        oracle_dfout = do_rerank(dfs[year][f'{testmodel}_mean'])
        carlist = set(list(oracle_dfout.carno.values))
        carlist = [int(x) for x in carlist]
        print('carlist:', carlist,'len:',len(carlist))

        #carlist = [13, 7, 3, 12]
        #carlist = [13]    

        retdata = {}
        for carno in carlist:
            print("*"*40)
            print('Run models for carno=', carno)
            # create the test_ds first
            test_cars = [carno]

            #train_ds, test_ds, trainset, testset = stint.make_dataset_byevent(events_id[_test_event], 
            #                                 prediction_length,freq, 
            #                                 oracle_mode=stint.MODE_ORACLE,
            #                                 run_ts = _run_ts,
            #                                 test_event = _test_event,
            #                                 test_cars=test_cars,
            #                                 half_moving_win = 0,
            #                                 train_ratio = 0.01)

            train_ds, test_ds, trainset, testset = make_dataset_byevent(prepared_laptimedata, prediction_length,freq,
                                             useeid=useeid, run_ts=_run_ts,
                                            test_event=_test_event, log_transform =False,
                                            context_ratio=0, train_ratio = 0,
                                            joint_train = _joint_train,
                                            test_cars = test_cars)    


            if (len(testset) <= 10 + prediction_length):
                print('ts too short, skip ', len(testset))
                continue

            #by first run samples
            samples = oracle_ret[mid][0][1][test_cars[0]]
            tss  = oracle_ret[mid][0][2][test_cars[0]]
            target_oracle1, tss_oracle1 = long_predict_bysamples('1run-samples', samples, tss, test_ds, _predictor)

            #by first run output df(_use_mean = true, already reranked)
            df = oracle_ret[mid][0][0]
            dfin_oracle = df[df['carno']==test_cars[0]]
            target_oracle2, tss_oracle2 = long_predict_bydf(f'{testmodel}-1run-dfout', dfin_oracle, test_ds, _predictor)        


            #by multi-run mean at oracle_dfout
            df = oracle_dfout
            dfin_oracle = df[df['carno']==test_cars[0]]
            target_oracle3, tss_oracle3 = long_predict_bydf(f'{testmodel}-multimean', dfin_oracle, test_ds, _predictor)        


            #no rerank
            df = ranknetdf[year][f'{testmodel}_mean']
            dfin_oracle = df[df['carno']==test_cars[0]]
            target_oracle4, tss_oracle4 = long_predict_bydf(f'{testmodel}-norerank-multimean', dfin_oracle, test_ds, _predictor)        


            #by multiple runs
            target_oracle_multirun, tss_oracle_multirun = get_ranknet_multirun(
                                    oracle_ret[mid], 
                                    test_cars[0], test_ds, _predictor,sampleCnt=loopcnt)

            retdata[carno] = [[tss_oracle1,tss_oracle2,tss_oracle3,tss_oracle4,tss_oracle_multirun],
                               [target_oracle1,target_oracle2,target_oracle3,target_oracle4,target_oracle_multirun]]

        alldata = retdata    

        if _savedata:
            with open(LONG_FORECASTING_DFS, 'wb') as f:
                pickle.dump(alldata, f, pickle.HIGHEST_PROTOCOL)  
            
           


# In[13]:


if _draw_figs:
    if _forecast_mode == 'shortterm' and _joint_train == False:
        destdir = FORECAST_FIGS_DIR

        if _skip_overwrite and os.path.exists(destdir):
            print('Long Forecasting Figures at:',destdir)

        else:
            with open(STAGE_DATASET, 'rb') as f:
                stagedata = pickle.load(f, encoding='latin1') 
                _alldata, rankdata, _acldata, _flagdata = stagedata[_test_event]

            #set gobal variable
            gvar.rankdata = rankdata
            #destdir = outputRoot + 'oracle-forecast-figs/'
            os.makedirs(destdir, exist_ok=True)

            for carno in alldata:
                plotoracle(alldata, carno, destdir)

            #draw summary result
            outputfile = destdir + f'{configname}'
            plotallcars(alldata, outputfile, drawid = 0)


# final output
pd.set_option("display.max_rows", None, "display.max_columns", None)
print(oracle_eval_result)





