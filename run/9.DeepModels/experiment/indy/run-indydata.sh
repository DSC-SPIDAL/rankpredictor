

model='deepAR'
epochs=2000
#epochs=1
id=indy-AR00
root=/scratch/hpda/indycar/test

datacfg='--predictionlen 20 --contextlen 100 --testlen 50'

gpuid=0
runid=$id-laptime
echo "python $root/src/deepmodels_indy.py --ts 2 $datacfg --gpuid $gpuid --model $model --epochs $epochs --input $root/data/laptime_rank-2018.pickle --output $runid 2>&1 |tee log-$runid.log"

gpuid=1
runid=$id-rank
echo "python $root/src/deepmodels_indy.py --ts 3 $datacfg --gpuid $gpuid --model $model --epochs $epochs --input $root/data/laptime_rank-2018.pickle --output $runid 2>&1 |tee log-$runid.log"

id=indy-AR01
datacfg='--predictionlen 50 --contextlen 100 --testlen 50'

gpuid=2
runid=$id-laptime
echo "python $root/src/deepmodels_indy.py --ts 2 $datacfg --gpuid $gpuid --model $model --epochs $epochs --input $root/data/laptime_rank-2018.pickle --output $runid 2>&1 |tee log-$runid.log"

gpuid=4
runid=$id-rank
echo "python $root/src/deepmodels_indy.py --ts 3 $datacfg --gpuid $gpuid --model $model --epochs $epochs --input $root/data/laptime_rank-2018.pickle --output $runid 2>&1 |tee log-$runid.log"





model='deepFactor'
#epochs=2000
gpuid=2
id=indy-DF00
runid=$id-laptime
echo "python $root/src/deepmodels_indy.py --ts 2 $datacfg --gpuid $gpuid --model $model --epochs $epochs --input $root/data/laptime_rank-2018.pickle --output $runid 2>&1 |tee log-$runid.log"

runid=$id-rank
echo "python $root/src/deepmodels_indy.py --ts 3 $datacfg --gpuid $gpuid --model $model --epochs $epochs --input $root/data/laptime_rank-2018.pickle --output $runid 2>&1 |tee log-$runid.log"


model='deepState'
#epochs=2000
gpuid=4
id=indy-DS00
runid=$id-laptime
echo "python $root/src/deepmodels_indy.py --ts 2 $datacfg --gpuid $gpuid --model $model --epochs $epochs --input $root/data/laptime_rank-2018.pickle --output $runid 2>&1 |tee log-$runid.log"

gpuid=6
runid=$id-rank
echo "python $root/src/deepmodels_indy.py --ts 3 $datacfg --gpuid $gpuid --model $model --epochs $epochs --input $root/data/laptime_rank-2018.pickle --output $runid 2>&1 |tee log-$runid.log"









