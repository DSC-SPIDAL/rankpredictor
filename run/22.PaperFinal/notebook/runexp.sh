echo "run stint simulation"
export CUDA_VISIBLE_DEVICES=7
./run_quicktest.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-pitmodel.ini --forecast_mode=stint --pitmodel_bias=0 --loopcnt=100 --year=2018 --test_event=Indy500-2018
./run_quicktest.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-pitmodel.ini --forecast_mode=stint --pitmodel_bias=0 --loopcnt=100 --year=2019 --test_event=Indy500-2019

./run_quicktest.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-pitmodel.ini --forecast_mode=stint --pitmodel_bias=2 --loopcnt=100 --year=2018 --test_event=Indy500-2018
./run_quicktest.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-pitmodel.ini --forecast_mode=stint --pitmodel_bias=2 --loopcnt=100 --year=2019 --test_event=Indy500-2019
