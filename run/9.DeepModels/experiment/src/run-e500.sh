

epochs=$1
runid=$2


runid=$2-200laps
python deepar_simindy500.py --epochs $epochs --output $runid 2>&1 |tee log-$runid.log

runid=$2-500laps
python deepar_simindy500.py --epochs $epochs --input sim-indy500-laptime-2018-500laps.pickle --output $runid 2>&1 |tee log-$runid.log

