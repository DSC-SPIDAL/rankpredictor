echo "run stint simulation"
export CUDA_VISIBLE_DEVICES=1

testevents=(Indy500-2018 Indy500-2019 Phoenix-2018 Texas-2018 Texas-2019 Pocono-2018 Pocono-2019 Iowa-2018 Iowa-2019 Gateway-2018 Gateway-2019)
#testevents=(Indy500-2018 Indy500-2019 Phoenix-2018 Texas-2018 Texas-2019 )
testevents=(Pocono-2018 Pocono-2019 Iowa-2018)
#testevents=(Iowa-2019 Gateway-2018 Gateway-2019)

if [ $# -lt 0 ] ; then
    testevents=($1)
fi

echo 'Run for test events:'$testevents

for test_event in ${testevents[*]}; do

    echo $test_event
    
#    ./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-oracle.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=2 --year=${test_event} --test_event=${test_event} --suffix=debug
#    ./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-oracle.ini --forecast_mode=stint --pitmodel_bias=0 --loopcnt=2 --year=${test_event} --test_event=${test_event} --suffix=debug
    
    ./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-pitmodel.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=100 --year=${test_event} --test_event=${test_event} --suffix=multidata
#    ./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-pitmodel.ini --forecast_mode=stint --pitmodel_bias=0 --loopcnt=20 --year=${test_event} --test_event=${test_event} --suffix=debug
    

done



