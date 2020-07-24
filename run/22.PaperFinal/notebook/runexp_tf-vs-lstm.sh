echo "run stint simulation"

config=$1
trainmodel=$2
testmodel=$3
gpuid=$4
year=$5

# get log name
VAR=""
for ELEMENT in $@; do
  VAR+="${ELEMENT}"
done
VAR=`echo $VAR |sed 's/\//_/g'`
echo $VAR
logfile="logs/${VAR}.log"

export CUDA_VISIBLE_DEVICES=$gpuid

#./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-${config}-nocate-c60-drank-oracle.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=2 --year=Indy500-${year} --test_event=Indy500-${year} --trainmodel=${trainmodel} --testmodel=${testmodel} --suffix=tf-vs-lstm --gpuid=0
python RankNet-QuickTest-Slim.py QuickTestOutput/weighted-noinlap-${config}-nocate-c60-drank-oracle.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=2 --year=Indy500-${year} --test_event=Indy500-${year} --trainmodel=${trainmodel} --testmodel=${testmodel} --suffix=tf-vs-lstm --gpuid=0 | tee $logfile


testmodel=`echo $testmodel | sed 's/Oracle/MLP/g'`
#./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-${config}-nocate-c60-drank-pitmodel.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=100 --year=Indy500-${year} --test_event=Indy500-${year} --trainmodel=${trainmodel} --testmodel=${testmodel} --suffix=tf-vs-lstm --gpuid=0
python RankNet-QuickTest-Slim.py QuickTestOutput/weighted-noinlap-${config}-nocate-c60-drank-pitmodel.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=100 --year=Indy500-${year} --test_event=Indy500-${year} --trainmodel=${trainmodel} --testmodel=${testmodel} --suffix=tf-vs-lstm --gpuid=0 | tee $logfile
