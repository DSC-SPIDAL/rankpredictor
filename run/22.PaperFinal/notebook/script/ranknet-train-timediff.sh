#
# test <arima, deepAR, and deepAR-Oracle+testcurtrack>
#
id='oracle'

train_ratio=Indy500

predict_lens=(2 4 6 8)
runs=(1)
root=/scratch_hdd/hpda/indycar/test



gpuid=4
epochs=1000
if [ ! -z $1 ]; then
    epochs=$1
fi
echo "epochs=${epochs}"

dataset='timediff'

logfile=$id-execution-time.csv
log_start()
{
STARTTIME=$(date +%s)
}

log_elapsed()
{
ENDTIME=$(date +%s)
echo "elapsedtime: $model $run $predict_len $(($ENDTIME - $STARTTIME))" 
echo "$model $run $predict_len $(($ENDTIME - $STARTTIME))" >>$logfile

STARTTIME=$(date +%s)
}

datatopdir="ranknet-train"
dataids=('inlap-pitage' 'inlap-pitage' 'outlap-pitage' 'inlap-nopitage' 'inlap-nopitage' 'outlap-nopitage')

for tid in ${dataids[*]}; do
    #prepare the directory
    workdir="indy2013-2018-${tid}/${dataset}-indy500"
    mkdir -p $workdir

    pushd $workdir

    #train
    for predict_len in ${predict_lens[*]}; do
    
        contextlen=$((predict_len * 2))
        
        if [ $contextlen -lt 10 ]; then contextlen=10; fi
        datacfg="--contextlen 40 --batch_size 32 --nocarid"
        
        for run in ${runs[*]}; do
        
            log_start
            
            model='deepAR-Oracle'
            runid=${model}-${dataset}-all-indy-f1min-t${predict_len}-e$epochs-r${run}_${id}_t${predict_len}
            db=${dataset}-oracle-noip-noeid-all-all-f1min-t${predict_len}-r${train_ratio}-2018-gluonts-indy-2018.pickle
            echo "python -m indycar.model.gluonts_models $datacfg --gpuid $gpuid --model $model --epochs $epochs --input $root/data/${datatopdir}/${tid}/${dataset}-indy500/$db --output $runid 2>&1 |tee log-$runid.log"
            #python -m indycar.model.gluonts_models $datacfg --gpuid $gpuid --model $model --epochs $epochs --input $root/data/indy2013-2018-inlap-nopitage/rank-indy500/$db --output $runid 2>&1 |tee log-$runid.log
            python -m indycar.model.gluonts_models $datacfg --gpuid $gpuid --model $model --epochs $epochs --input $root/data/${datatopdir}/${tid}/${dataset}-indy500/$db --output $runid 2>&1 |tee log-$runid.log
            
            model=$model-all
            log_elapsed
        
        done
    
    done

    #goback home
    popd

done

