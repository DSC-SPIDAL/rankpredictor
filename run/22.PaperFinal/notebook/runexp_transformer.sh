echo "run stint simulation"
export CUDA_VISIBLE_DEVICES=5

####  oracle and standard 

#shortterm
./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-oracle.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=2 --year=2018 --test_event=Indy500-2018 --trainmodel=Transformer --testmodel=Transformer --debug
./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-oracle.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=2 --year=2019 --test_event=Indy500-2019 --trainmodel=Transformer --testmodel=Transformer --debug


./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-oracle.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=2 --year=2018 --test_event=Indy500-2018 --trainmodel=Transformer-Oracle --testmodel=Transformer-Oracle --debug

./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-oracle.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=2 --year=2019 --test_event=Indy500-2019 --trainmodel=Transformer-Oracle --testmodel=Transformer-Oracle --debug



#stint
./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-oracle.ini --forecast_mode=stint --pitmodel_bias=0 --loopcnt=2 --year=2018 --test_event=Indy500-2018 --trainmodel=Transformer --testmodel=Transformer --debug

./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-oracle.ini --forecast_mode=stint --pitmodel_bias=0 --loopcnt=2 --year=2019 --test_event=Indy500-2019 --trainmodel=Transformer --testmodel=Transformer --debug


./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-oracle.ini --forecast_mode=stint --pitmodel_bias=0 --loopcnt=2 --year=2018 --test_event=Indy500-2018 --trainmodel=Transformer-Oracle --testmodel=Transformer-Oracle --debug

./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-oracle.ini --forecast_mode=stint --pitmodel_bias=0 --loopcnt=2 --year=2019 --test_event=Indy500-2019 --trainmodel=Transformer-Oracle --testmodel=Transformer-Oracle --debug



#### pitmodel
#shortterm
./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-pitmodel.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=100 --year=2018 --test_event=Indy500-2018 --trainmodel=Transformer-Oracle --testmodel=Transformer-MLP --debug

./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-pitmodel.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=100 --year=2019 --test_event=Indy500-2019 --trainmodel=Transformer-Oracle --testmodel=Transformer-MLP --debug



#stint
./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-pitmodel.ini --forecast_mode=stint --pitmodel_bias=0 --loopcnt=100 --year=2018 --test_event=Indy500-2018 --trainmodel=Transformer-Oracle --testmodel=Transformer-MLP --debug

./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-pitmodel.ini --forecast_mode=stint --pitmodel_bias=0 --loopcnt=100 --year=2019 --test_event=Indy500-2019 --trainmodel=Transformer-Oracle --testmodel=Transformer-MLP --debug




