
if [ $# -le 10 ]; then
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
shift
contextlen=$9

if [ -z $contextlen ] ; then
    contextlen=60
fi


# get log name
#VAR=""
#for ELEMENT in $@; do
#  VAR+="${ELEMENT}"
#done
#VAR=`echo $VAR |sed 's/\//_/g'`
#echo $VAR
logfile="logs/${trainrace}_N${trainmodel}_E${featuremode}_A${trainmodel}_M${testmodel}_F${forecastmode}_P${pitmodel}_T${testevent}_L${loopcnt}_C${contextlen}_S${suffix}.log"


export CUDA_VISIBLE_DEVICES=$gpuid


if echo "$trainmodel" | grep -q "multi"; then
    #joint train
    echo "python RankNet-QuickTest-Slim.py config/weighted-noinlap-${featuremode}-nocate-c${contextlen}-drank-${pitmodel}.ini --forecast_mode=${forecastmode} --pitmodel_bias=0 --loopcnt=${loopcnt} --test_event=${testevent} --trainmodel=${trainmodel} --testmodel=${testmodel} --suffix=${suffix} --gpuid=0 --joint_train --trainrace=${trainrace} " | tee $logfile
    python RankNet-QuickTest-Slim.py config/weighted-noinlap-${featuremode}-nocate-c${contextlen}-drank-${pitmodel}.ini --forecast_mode=${forecastmode} --pitmodel_bias=0 --loopcnt=${loopcnt} --test_event=${testevent} --trainmodel=${trainmodel} --testmodel=${testmodel} --suffix=${suffix} --gpuid=0 --joint_train  --trainrace=${trainrace} | tee -a $logfile


else
    echo "python RankNet-QuickTest-Slim.py config/weighted-noinlap-${featuremode}-nocate-c${contextlen}-drank-${pitmodel}.ini --forecast_mode=${forecastmode} --pitmodel_bias=0 --loopcnt=${loopcnt} --test_event=${testevent} --trainmodel=${trainmodel} --testmodel=${testmodel} --suffix=${suffix} --gpuid=0  --trainrace=${trainrace}" | tee $logfile
    python RankNet-QuickTest-Slim.py config/weighted-noinlap-${featuremode}-nocate-c${contextlen}-drank-${pitmodel}.ini --forecast_mode=${forecastmode} --pitmodel_bias=0 --loopcnt=${loopcnt} --test_event=${testevent} --trainmodel=${trainmodel} --testmodel=${testmodel} --suffix=${suffix} --gpuid=0  --trainrace=${trainrace} | tee -a $logfile

fi


