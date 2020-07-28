if [ $# -ne 3 ]; then
	echo "usage runmodel_xxx.sh <gpuid> <suffix>"
	exit 0
fi

gpuid=$1
suffix=$2
testevent=$3

script/run_model.sh S0LTYP0T shortterm oracle deepARW-multi deepARW-multi $testevent 2 $gpuid $suffix
script/run_model.sh S0LTYP0T stint oracle deepARW-multi deepARW-multi $testevent 2 $gpuid $suffix
