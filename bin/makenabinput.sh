#!/bin/bash

if [ $# -ne '3' ] ; then
    echo 'getbycar.sh <telemetryfile> <col> <output>'
    exit -1
fi

data=$1
col=$2
output=$3

#add header to the output
echo "timestamp,value" > ${output}

echo "Get telemetry data ${data}, output to ${output}"

#gstr="'{printf("\""2018-05-27 %s,%s\n"\"", \$1, \$${col})}'"
gstr="{printf("\""2018-05-27 %s,%s\n"\"", \$1, \$${col})}"
echo "gstr=$gstr"
gawk "${gstr}" ${data} >> ${output}


