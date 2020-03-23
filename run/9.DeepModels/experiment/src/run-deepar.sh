

epochs=$1
runid=$2

python deepar_simindy500.py --epochs $epochs --output $runid 2>&1 |tee log-$runid.log

