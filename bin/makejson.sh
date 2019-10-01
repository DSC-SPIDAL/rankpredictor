
input=$1
fname=`basename $input`

mkdir -p indycar/indycar
cp $input indycar/indycar
#make an empty labels file
python -c 'import sys; print("{\n\t\"indycar/%s\": []\n}"%(sys.argv[1]))' ${fname} > indycar/indycar_labels.json
