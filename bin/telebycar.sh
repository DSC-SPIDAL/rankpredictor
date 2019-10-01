#!/bin/bash

if [ $# -ne '3' ] ; then
    echo 'getbycar.sh <carno> <telemetryfile> <output>'
    exit -1
fi

carno=$1
data=$2
output=$3
echo "Get telemetry data by car number $carno, output to ${output}-rpm-valid.csv"
grep -P "P\t${carno}\t" $data >${output}-raw.csv

gawk '{printf("2017-05-28 %s,%s\n", $3, $6)}' ${output}-raw.csv > ${output}-rpm.csv

#add header to the output
echo "timestamp,value" > ${output}-rpm-valid.csv
#filter out records with invalid timestamp
grep -P " [0-9]*:[0-9]*:[0-9]*\." ${output}-rpm.csv >> ${output}-rpm-valid.csv

