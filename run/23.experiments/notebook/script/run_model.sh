
if [ $# -ne 9 ]; then
    echo "usage: runexp_paper.sh <featuremode> <forecastmode> <pitmodel> <trainmodel> <testmodel> <testevent> <loopcnt> <gpuid> <suffix>"
    exit -1
fi

featuremode=$1
forecastmode=$2
pitmodel=$3
trainmodel=$4
testmodel=$5
testevent=$6
loopcnt=$7
gpuid=$8
suffix=$9

export CUDA_VISIBLE_DEVICES=$gpuid


if echo "$trainmodel" | grep -q "multi"; then
    #joint train
    echo "python RankNet-QuickTest-Slim.py config/weighted-noinlap-${featuremode}-nocate-c60-drank-${pitmodel}.ini --forecast_mode=${forecastmode} --pitmodel_bias=0 --loopcnt=${loopcnt} --test_event=${testevent} --trainmodel=${trainmodel} --testmodel=${testmodel} --suffix=${suffix} --gpuid=0 --joint_train | tee $logfile"
    python RankNet-QuickTest-Slim.py config/weighted-noinlap-${featuremode}-nocate-c60-drank-${pitmodel}.ini --forecast_mode=${forecastmode} --pitmodel_bias=0 --loopcnt=${loopcnt} --test_event=${testevent} --trainmodel=${trainmodel} --testmodel=${testmodel} --suffix=${suffix} --gpuid=0 --joint_train | tee $logfile


else
    echo "python RankNet-QuickTest-Slim.py config/weighted-noinlap-${featuremode}-nocate-c60-drank-${pitmodel}.ini --forecast_mode=${forecastmode} --pitmodel_bias=0 --loopcnt=${loopcnt} --test_event=${testevent} --trainmodel=${trainmodel} --testmodel=${testmodel} --suffix=${suffix} --gpuid=0 | tee $logfile"
    python RankNet-QuickTest-Slim.py config/weighted-noinlap-${featuremode}-nocate-c60-drank-${pitmodel}.ini --forecast_mode=${forecastmode} --pitmodel_bias=0 --loopcnt=${loopcnt} --test_event=${testevent} --trainmodel=${trainmodel} --testmodel=${testmodel} --suffix=${suffix} --gpuid=0 | tee $logfile

fi


