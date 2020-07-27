if [ $# -ne 2 ]; then
	echo "usage runmodel_xxx.sh <gpuid> <suffix>"
	exit 0
fi

gpuid=$1
suffix=$2

./run_model.sh S0LTYP0T shortterm oracle deepARW-multi deepARW-multi Indy500-2019 2 $gpuid $suffix
./run_model.sh S0LTYP0T stint oracle deepARW-multi deepARW-multi Indy500-2019 2 $gpuid $suffix
