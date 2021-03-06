
export CUDA_VISIBLE_DEVICES=$1
trainrace=Indy500,Pocono,Iowa,Texas
models=(deepARW-Oracle TransformerWFM-Oracle);
#models=(deepARW-Oracle );
#models=(TransformerWFM-Oracle);
#testsets=(Indy500-2018 Indy500-2019)
#testsets=(Pocono-2019 Texas-2019 Iowa-2019)
testsets=(Texas-2019 Iowa-2019)
#id=alldatalr2b128p20
id=alldatalr2b128
for model in ${models[*]}; do
     for testdata in ${testsets[*]}; do     
#python RankNet-QuickTest-Slim.py config/weighted-noinlap-S0LTYP0T-nocate-c60-drank-oracle.ini --patience 10 --lr 1e-2 --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=1 --test_event=${testdata} --trainmodel=${model} --testmodel=${model} --suffix=lstmvstf${id} --gpuid=0 --trainrace=${trainrace} --context_len=60 --prediction_len=2 --weight_coef=9 2>&1 | tee  -a logs/lstmvstf_${trainrace}_${testdata}_${model}_${id}.log;
#python RankNet-QuickTest-Slim.py config/weighted-noinlap-S0LTYP0T-nocate-c60-drank-oracle.ini --patience 20 --lr 1e-2 --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=1 --test_event=${testdata} --trainmodel=${model} --testmodel=${model} --suffix=lstmvstf${id} --gpuid=0 --trainrace=${trainrace} --context_len=60 --prediction_len=2 --weight_coef=9 --batch_size=128 2>&1 | tee  -a logs/lstmvstf_${trainrace}_${testdata}_${model}_${id}.log;
     python RankNet-QuickTest-Slim.py config/weighted-noinlap-S0LTYP0T-nocate-c60-drank-oracle.ini  --forecast_mode=shortterm --pitmodel_bias=0 --loopcnt=1 --test_event=${testdata} --trainmodel=${model} --testmodel=${model} --suffix=lstmvstf${id} --gpuid=0 --trainrace=${trainrace} --context_len=60 --prediction_len=2 --weight_coef=9 --batch_size=128 2>&1 | tee  -a logs/lstmvstf_${trainrace}_${testdata}_${model}_${id}.log;

done

done
