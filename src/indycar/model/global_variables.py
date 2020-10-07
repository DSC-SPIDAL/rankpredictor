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
test_context_ratio = 0.
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
learning_rate = 1e-3
patience = 10
use_validation = False

# using only one lags
_lags_seq = [1]

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
events_info = {}
_race_info = {}
global_carids = {}
_train_events = []
trainrace = 'Indy500'

maxlap = 200
dbid = ''
LAPTIME_DATASET = ''
rankdata = None

#
# new experimental vars
#
# static_cat type: 0: carid; 1:carid, eid; 2:tsid
# fail, carid is hard coded in feat_static_cat
static_cat_type = 0
