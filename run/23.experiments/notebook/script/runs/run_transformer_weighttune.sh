#runmodel_xxx.sh <gpuid> <suffix> <testevent> [trainrace] [predictionlen] [contextlen]"
#script/run_model.sh S0LTYP0T shortterm oracle deepARW-Oracle deepARW-Oracle $testevent 2 $gpuid $suffix $allargs

if [ $# -lt 2 ]; then
    echo "usage: runexp_paper.sh <gpuid> <weight>"
    exit -1
fi

gpuid=$1
weight=$2
script/run_model.sh S0LTYP0T shortterm oracle TransformerWFM-Oracle TransformerWFM-Oracle Indy500-2018 2 $gpuid wtune_${weight} Indy500 2 60 ${weight}
