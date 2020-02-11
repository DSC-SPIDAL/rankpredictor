

model='deepFactor'
epochs=2000
#epochs=1
id=DF02
root=/scratch/hpda/indycar/test

datacfg='--predictionlen 50 --contextlen 100 --testlen 50'

gpuid=3
runid=$id-200laps
echo "python $root/src/deepmodels_simindy500.py $datacfg --gpuid $gpuid --model $model --epochs $epochs --input $root/data/sim-indy500-laptime-2018-200laps.pickle --output $runid 2>&1 |tee log-$runid.log"

gpuid=4
runid=$id-500laps
echo "python $root/src/deepmodels_simindy500.py $datacfg --gpuid $gpuid --model $model --epochs $epochs --input $root/data/sim-indy500-laptime-2018-500laps.pickle --output $runid 2>&1 |tee log-$runid.log"

gpuid=5
runid=$id-200laps-360runs
echo "python $root/src/deepmodels_simindy500.py $datacfg --gpuid $gpuid --model $model --epochs $epochs --input $root/data/sim-indy500-laptime-2018-200laps-360runs.pickle --output $runid 2>&1 |tee log-$runid.log"



