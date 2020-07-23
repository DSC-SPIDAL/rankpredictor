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

VAR=""
for ELEMENT in $@; do
  VAR+="${ELEMENT}"
done
VAR=`echo $VAR |sed 's/\//_/g'`
echo $VAR

mkdir -p logs
python RankNet-QuickTest-Slim.py "$@" | tee logs/${VAR}.log

