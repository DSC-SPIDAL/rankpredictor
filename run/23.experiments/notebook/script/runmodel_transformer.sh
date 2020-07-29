if [ $# -ne 4 ]; then
	echo "usage runmodel_xxx.sh <gpuid> <suffix>"
	exit 0
fi

gpuid=$1
suffix=$2
testevent=$3
trainrace=$4

script/run_model.sh S0LTYP0T shortterm oracle TransformerWFM-Oracle TransformerWFM-Oracle $testevent 2 $gpuid $suffix $trainrace
script/run_model.sh S0LTYP0T shortterm pitmodel TransformerWFM-Oracle TransformerWFM-MLP $testevent 100 $gpuid $suffix $trainrace
script/run_model.sh S0LTYP0T stint oracle TransformerWFM-Oracle TransformerWFM-Oracle $testevent 2 $gpuid $suffix $trainrace
script/run_model.sh S0LTYP0T stint pitmodel TransformerWFM-Oracle TransformerWFM-MLP $testevent 100 $gpuid $suffix $trainrace
