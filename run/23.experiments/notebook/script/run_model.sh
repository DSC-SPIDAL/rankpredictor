
if [ $# -lt 10 ]; then
    echo "usage: runexp_paper.sh <featuremode> <forecastmode> <pitmodel> <trainmodel> <testmodel> <testevent> <loopcnt> <gpuid> <suffix> [trainrace] [predictionlen] [contextlen]"
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
shift
predictionlen=$9
shift
contextlen=$9

if [ -z $predictionlen ] ; then
    predictionlen=-1
fi

if [ -z $contextlen ] ; then
    contextlen=-1
fi


configfile="weighted-noinlap-${featuremode}-nocate-c60-drank-${pitmodel}.ini"
# get log name
#VAR=""
#for ELEMENT in $@; do
#  VAR+="${ELEMENT}"
#done
#VAR=`echo $VAR |sed 's/\//_/g'`
#echo $VAR
logfile="logs/${trainrace}_N${trainmodel}_E${featuremode}_A${trainmodel}_M${testmodel}_F${forecastmode}_P${pitmodel}_T${testevent}_L${loopcnt}_C${contextlen}_P${predictionlen}_S${suffix}.log"


export CUDA_VISIBLE_DEVICES=$gpuid


if echo "$trainmodel" | grep -q "multi"; then
    #joint train
    echo "python RankNet-QuickTest-Slim.py config/${configfile}  --forecast_mode=${forecastmode} --pitmodel_bias=0 --loopcnt=${loopcnt} --test_event=${testevent} --trainmodel=${trainmodel} --testmodel=${testmodel} --suffix=${suffix} --gpuid=0 --joint_train --trainrace=${trainrace} --context_len=${contextlen} --prediction_len=${predictionlen}" | tee $logfile
    python RankNet-QuickTest-Slim.py config/${configfile}  --forecast_mode=${forecastmode} --pitmodel_bias=0 --loopcnt=${loopcnt} --test_event=${testevent} --trainmodel=${trainmodel} --testmodel=${testmodel} --suffix=${suffix} --gpuid=0 --joint_train  --trainrace=${trainrace}  --context_len=${contextlen} --prediction_len=${predictionlen} 2>&1 | tee -a $logfile


else
    echo "python RankNet-QuickTest-Slim.py config/${configfile} --forecast_mode=${forecastmode} --pitmodel_bias=0 --loopcnt=${loopcnt} --test_event=${testevent} --trainmodel=${trainmodel} --testmodel=${testmodel} --suffix=${suffix} --gpuid=0  --trainrace=${trainrace} --context_len=${contextlen} --prediction_len=${predictionlen}" | tee $logfile
    python RankNet-QuickTest-Slim.py config/${configfile}  --forecast_mode=${forecastmode} --pitmodel_bias=0 --loopcnt=${loopcnt} --test_event=${testevent} --trainmodel=${trainmodel} --testmodel=${testmodel} --suffix=${suffix} --gpuid=0  --trainrace=${trainrace}  --context_len=${contextlen} --prediction_len=${predictionlen} 2>&1 | tee -a $logfile

fi


