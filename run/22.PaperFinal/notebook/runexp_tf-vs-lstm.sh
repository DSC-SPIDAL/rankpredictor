echo "run stint simulation"

config=$1
trainmodel=$2
testmodel=$3
gpuid=$4


export CUDA_VISIBLE_DEVICES=$gpuid

#./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0000000-nocate-c60-drank-oracle.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=2 --year=Indy500-2018 --test_event=Indy500-2018 --trainmodel=deepAR-Oracle --testmodel=deepAR-Oracle --suffix=tf-vs-lstm --gpuid=0
#
#./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0000000-nocate-c60-drank-pitmodel.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=20 --year=Indy500-2018 --test_event=Indy500-2018 --trainmodel=deepAR-Oracle --testmodel=deepAR-MLP --suffix=tf-vs-lstm --gpuid=0
#
#./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0000000-nocate-c60-drank-oracle.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=2 --year=Indy500-2018 --test_event=Indy500-2018 --trainmodel=Transformer-Oracle --testmodel=Transformer-Oracle --suffix=tf-vs-lstm --gpuid=0
#
#./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0000000-nocate-c60-drank-pitmodel.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=20 --year=Indy500-2018 --test_event=Indy500-2018 --trainmodel=Transformer-Oracle --testmodel=Transformer-MLP --suffix=tf-vs-lstm --gpuid=0

#./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-${config}-nocate-c60-drank-oracle.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=2 --year=Indy500-2018 --test_event=Indy500-2018 --trainmodel=${trainmodel} --testmodel=${testmodel} --suffix=tf-vs-lstm --gpuid=0

testmodel=`echo $testmodel | sed 's/Oracle/MLP/g'`
./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-${config}-nocate-c60-drank-pitmodel.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=100 --year=Indy500-2018 --test_event=Indy500-2018 --trainmodel=${trainmodel} --testmodel=${testmodel} --suffix=tf-vs-lstm --gpuid=0
