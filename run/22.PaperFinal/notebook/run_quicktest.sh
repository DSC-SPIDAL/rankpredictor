#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: run_quicktest.sh <configfiles...>"
    exit
fi

while [ $# -gt 0 ]; do

    configfile=$1
    
    #get configname
    filename=$(basename -- "$configfile")
    extension="${filename##*.}"
    filename="${filename%.*}"
    
    mkdir -p logs
    python RankNet-QuickTest.py $configfile | tee logs/${filename}.log
    
    shift

done
