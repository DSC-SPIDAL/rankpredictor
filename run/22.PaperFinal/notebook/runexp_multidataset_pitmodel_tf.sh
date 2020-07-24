echo "run stint simulation"
export CUDA_VISIBLE_DEVICES=1

testevents=(Indy500-2018 Indy500-2019 Phoenix-2018 Texas-2018 Texas-2019 Pocono-2018 Pocono-2019 Iowa-2018 Iowa-2019 Gateway-2018 Gateway-2019)
testevents=(Phoenix-2018 Texas-2018 Pocono-2018 Iowa-2018 Gateway-2018)

if [ $# -lt 0 ] ; then
    testevents=($1)
fi

echo 'Run for test events:'$testevents

run_quicktest()
{
logfile="logs/multidataset_pitmodel_tf-${trainmodel}-${test_event}-l${loopcnt}.log"


python RankNet-QuickTest-Slim.py "$@" | tee $logfile
}


trainmodel="TransformerWF-Oracle"
testmodel="TransformerWF-MLP"

for test_event in ${testevents[*]}; do

    echo $test_event
    run_quicktest QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-pitmodel.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=100 --year=${test_event} --test_event=${test_event} --suffix=multidata-tf --trainmodel=${trainmodel} --testmodel=${testmodel} --gpuid=0 

done



