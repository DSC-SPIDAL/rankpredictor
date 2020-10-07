if [ $# -ne 2 ] ; then
    echo "usage: run_adv.sh <batchsize> <usecore>"
    exit 0
fi

batchsize=$1
usecore=$2

cmdline="/scratch_hdd/hpda/anaconda3/envs/ranknet/bin/python opt_testve_eager.py --usecore=${usecore} --batchsize=${batchsize}"
resDir=./advi_deepar_b${batchsize}_c${usecore}

echo "Result dir: $resDir"
if [ -d $resDir ]; then
    echo "remove existed result folder"
    rm -rf $resDir
fi

#advixe-cl -collect roofline -stacks -project-dir ${resDir} $cmdline
advixe-cl -collect survey -project-dir ${resDir} $cmdline

#advixe-cl --collect tripcounts --project-dir ${resDir} --flop --no-trip-counts -- $cmdline
advixe-cl --collect tripcounts --project-dir ${resDir} --flop -- $cmdline
