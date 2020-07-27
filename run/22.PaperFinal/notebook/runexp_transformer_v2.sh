echo "run stint simulation"
export CUDA_VISIBLE_DEVICES=6

####  oracle and standard 

#shortterm
./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-oracle.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=2 --year=2018 --test_event=Indy500-2018 --trainmodel=Transformer --testmodel=Transformer --suffix=transformer2
./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-oracle.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=2 --year=2019 --test_event=Indy500-2019 --trainmodel=Transformer --testmodel=Transformer --suffix=transformer2


./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-oracle.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=2 --year=2018 --test_event=Indy500-2018 --trainmodel=Transformer-Oracle --testmodel=Transformer-Oracle --suffix=transformer2

./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-oracle.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=2 --year=2019 --test_event=Indy500-2019 --trainmodel=Transformer-Oracle --testmodel=Transformer-Oracle --suffix=transformer2



#stint
./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-oracle.ini --forecast_mode=stint --pitmodel_bias=0 --loopcnt=2 --year=2018 --test_event=Indy500-2018 --trainmodel=Transformer --testmodel=Transformer --suffix=transformer2

./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-oracle.ini --forecast_mode=stint --pitmodel_bias=0 --loopcnt=2 --year=2019 --test_event=Indy500-2019 --trainmodel=Transformer --testmodel=Transformer --suffix=transformer2


./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-oracle.ini --forecast_mode=stint --pitmodel_bias=0 --loopcnt=2 --year=2018 --test_event=Indy500-2018 --trainmodel=Transformer-Oracle --testmodel=Transformer-Oracle --suffix=transformer2

./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-oracle.ini --forecast_mode=stint --pitmodel_bias=0 --loopcnt=2 --year=2019 --test_event=Indy500-2019 --trainmodel=Transformer-Oracle --testmodel=Transformer-Oracle --suffix=transformer2



#### pitmodel
#shortterm
./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-pitmodel.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=100 --year=2018 --test_event=Indy500-2018 --trainmodel=Transformer-Oracle --testmodel=Transformer-MLP --suffix=transformer2

./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-pitmodel.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=100 --year=2019 --test_event=Indy500-2019 --trainmodel=Transformer-Oracle --testmodel=Transformer-MLP --suffix=transformer2



#stint
./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-pitmodel.ini --forecast_mode=stint --pitmodel_bias=0 --loopcnt=100 --year=2018 --test_event=Indy500-2018 --trainmodel=Transformer-Oracle --testmodel=Transformer-MLP --suffix=transformer2

./run_quicktest_slim.sh QuickTestOutput/weighted-noinlap-S0LTYP0T-nocate-c60-drank-pitmodel.ini --forecast_mode=stint --pitmodel_bias=0 --loopcnt=100 --year=2019 --test_event=Indy500-2019 --trainmodel=Transformer-Oracle --testmodel=Transformer-MLP --suffix=transformer2



