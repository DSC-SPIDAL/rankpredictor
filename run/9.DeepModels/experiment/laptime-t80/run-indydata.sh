

epochs=1000
#epochs=1
root=/scratch/hpda/indycar/test

datacfg='--contextlen 50'

model='deepAR'
id=AR
gpuid=0
runid=$id-laptime-indy500
db=laptime-gluonts-indy500-2018.pickle
echo "python $root/src/deepmodels_indy_gluontsdb.py --ts 2 $datacfg --gpuid $gpuid --model $model --epochs $epochs --input $root/data/$db --output $runid 2>&1 |tee log-$runid.log"

gpuid=1
id=AR1K
runid=$id-laptime-all
db=laptime-gluonts-all-2018.pickle
echo "python $root/src/deepmodels_indy_gluontsdb.py --ts 2 $datacfg --gpuid $gpuid --model $model --epochs $epochs --input $root/data/$db --output $runid 2>&1 |tee log-$runid.log"









