# global settings
#
#_savedata = False
_savedata = True
_skip_overwrite = True

#inlap status = 
# 0 , no inlap
# 1 , set previous lap
# 2 , set the next lap
_inlap_status = 0

#
# featuremode in [FEATURE_STATUS, FEATURE_PITAGE]:
#
_feature_mode = 0
_featureCnt = 9

#
# training parameters
#
freq = "1min"
_train_len = 60
prediction_length = 2

context_ratio = 0.
context_length =  60
contextlen = context_length

dataset='rank'
epochs = 1000
#epochs = 10
gpuid = 5

#'deepAR-Oracle','deepARW-Oracle'
_use_weighted_model = True
trainmodel = 'deepARW-Oracle' if _use_weighted_model else 'deepAR-Oracle'

_use_cate_feature = False
use_feat_static = _use_cate_feature 

distroutput = 'student'
batch_size = 32

#
# test parameters
#
loopcnt = 2
_test_event = 'Indy500-2018'
testmodel = 'oracle'
pitmodel = 'oracle'
year = '2018'

_test_train_len = 40
_joint_train = False
_pitmodel_bias = 0
_forecast_mode = 'shortterm'

events = []
events_id ={}
global_carids = {}
