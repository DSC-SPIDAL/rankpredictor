#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: run_quicktest.sh <configfiles> [options]"
    exit
fi


configfile=$1

#get configname
filename=$(basename -- "$configfile")
extension="${filename##*.}"
filename="${filename%.*}"

mkdir -p logs
python RankNet-QuickTest-Slim.py "$@" | tee logs/${filename}.log

