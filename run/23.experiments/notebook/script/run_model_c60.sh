
if [ $# -ne 10 ]; then
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

shift
trainrace=$9


# get log name
#VAR=""
#for ELEMENT in $@; do
#  VAR+="${ELEMENT}"
#done
#VAR=`echo $VAR |sed 's/\//_/g'`
#echo $VAR
logfile="logs/${trainrace}_${trainmodel}_${featuremode}_${trainmodel}_${testmodel}_${forecastmode}_${pitmodel}_${testevent}_${loopcnt}_${suffix}.log"


export CUDA_VISIBLE_DEVICES=$gpuid


if echo "$trainmodel" | grep -q "multi"; then
    #joint train
    echo "python RankNet-QuickTest-Slim.py config/weighted-noinlap-${featuremode}-nocate-c60-drank-${pitmodel}.ini --forecast_mode=${forecastmode} --pitmodel_bias=0 --loopcnt=${loopcnt} --test_event=${testevent} --trainmodel=${trainmodel} --testmodel=${testmodel} --suffix=${suffix} --gpuid=0 --joint_train --trainrace=${trainrace} | tee $logfile"
    python RankNet-QuickTest-Slim.py config/weighted-noinlap-${featuremode}-nocate-c60-drank-${pitmodel}.ini --forecast_mode=${forecastmode} --pitmodel_bias=0 --loopcnt=${loopcnt} --test_event=${testevent} --trainmodel=${trainmodel} --testmodel=${testmodel} --suffix=${suffix} --gpuid=0 --joint_train  --trainrace=${trainrace} | tee $logfile


else
    echo "python RankNet-QuickTest-Slim.py config/weighted-noinlap-${featuremode}-nocate-c60-drank-${pitmodel}.ini --forecast_mode=${forecastmode} --pitmodel_bias=0 --loopcnt=${loopcnt} --test_event=${testevent} --trainmodel=${trainmodel} --testmodel=${testmodel} --suffix=${suffix} --gpuid=0  --trainrace=${trainrace} | tee $logfile"
    python RankNet-QuickTest-Slim.py config/weighted-noinlap-${featuremode}-nocate-c60-drank-${pitmodel}.ini --forecast_mode=${forecastmode} --pitmodel_bias=0 --loopcnt=${loopcnt} --test_event=${testevent} --trainmodel=${trainmodel} --testmodel=${testmodel} --suffix=${suffix} --gpuid=0  --trainrace=${trainrace} | tee $logfile

fi


