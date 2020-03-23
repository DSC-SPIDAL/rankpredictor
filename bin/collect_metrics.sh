#!/bin/bash

#
# collect metrics from log file of gluonts runs
#
#    "MSE": 635.8188582674634,
#    "abs_error": 4237654.489504814,
#    "abs_target_sum": 44722770.209438324,
#    "abs_target_mean": 157.3998490995593,
#    "seasonal_error": 4.3722497967176235,
#    "MASE": 4.108789588031062,
#    "sMAPE": 0.12484209456341018,
#    "MSIS": 72.18060697651568,
#    "QuantileLoss[0.1]": 2124778.6685033725,
#    "Coverage[0.1]": 0.17600771537963783,
#    "QuantileLoss[0.5]": 4237654.482759959,
#    "Coverage[0.5]": 0.4564933983141022,
#    "QuantileLoss[0.9]": 2583325.463880241,
#    "Coverage[0.9]": 0.7175409802995689,
#    "RMSE": 25.21544880162682,
#    "NRMSE": 0.1601999553740196,
#    "ND": 0.09475384618751762,
#    "wQuantileLoss[0.1]": 0.04750999677687581,
#    "wQuantileLoss[0.5]": 0.09475384603670284,
#    "wQuantileLoss[0.9]": 0.057763091413667714,
#    "mean_wQuantileLoss": 0.06667564474241545,
#    "MAE_Coverage": 0.1006577789219889
#

if [ $# -ne '1' ] ; then
    #echo 'collect_metrics.sh <inputdir> <output>'
    echo 'collect_metrics.sh <output>'
    exit -1
fi

#inputdir=$1
output=$1

metrics=("MASE" "sMAPE" "RMSE" "NRMSE" "wQuantileLoss\[0\.5\]" "wQuantileLoss\[0\.9\]" "mean_wQuantileLoss")

tmpfile=.metrics

#clean 
#cat /dev/null >$tmpfile
echo "runid,metric,value," > $tmpfile

#get metrics
for metric in ${metrics[*]}; do
    #grep "$metric" $inputdir/*.log >> $tmpfile
    grep "$metric" *.log >> $tmpfile
done

#format
sed -i 's/:/,/g' $tmpfile
sed -i 's/"//g' $tmpfile
sed -i 's/ //g' $tmpfile
sed -i 's/\.log//' $tmpfile

#transform

#output
mv $tmpfile $output
collect_metric.py $output
