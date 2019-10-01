input=$1
threshold=$2
python -m indycar.nabcsv --input results/numenta/indycar/numenta_${input}.csv --output anomaly_${threshold}_${input}.csv --threshold ${threshold}
python -m indycar.plot --plot anomaly --input anomaly_${threshold}_${input}.csv --output ${input}_${threshold}.pdf --flaginfo ../flaginfo.csv
