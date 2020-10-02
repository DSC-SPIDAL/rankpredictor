races=(Indy500 Pocono Iowa Texas); 
all=(0 0.1 0.2 0.3 0.4 0.5); 
for race in ${races[*]}; do 
    for ratio in ${all[*]}; do 
        test_ratio=0.5; 
        python RankNet-QuickTest-Slim.py config/weighted-noinlap-S0LTYP0T-nocate-c60-drank-oracle.ini --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=1 --test_event=${race}-2019 --trainmodel=deepARW-Oracle --testmodel=deepARW-Oracle --suffix=${race}-ctxratio${ratio} --gpuid=0 --trainrace=${race} --context_len=60 --prediction_len=2 --weight_coef=9 --context_ratio=${test_ratio} 2>&1 | tee  logs/${race}_ctxratio${ratio}.log; 
    done; 
done
