if [ $# -lt 4 ]; then
	echo "usage runmodel_xxx.sh <gpuid> <suffix>"
	exit 0
fi

gpuid=$1
suffix=$2
testevent=$3
shift
shift
shift
allargs="$@"
trainrace=$1

echo $allargs
echo "script/run_model.sh S0LTYP0T shortterm oracle deepARW-Oracle deepARW-Oracle $testevent 2 $gpuid $suffix $trainrace $allargs"

