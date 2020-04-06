datasetid=$1

testevents=('Indy500-2018' 'Indy500-2019')
tasks=('laptime' 'rank' 'timediff')

for testevent in ${testevents[*]}; do

for task in ${tasks[*]}; do
    echo "python -m indycar.model.stint-predictor-fastrun --datasetid $datasetid --testevent $testevent --task $task"
    python -m indycar.model.stint-predictor-fastrun --datasetid $datasetid --testevent $testevent --task $task
done
done




#python -m indycar.model.stint-predictor-fastrun --testevent Indy500-2019 --datasetid indy2013-2018-pitage
#python -m indycar.model.stint-predictor-fastrun --testevent Indy500-2019 --datasetid indy2013-2018-pitageonly
#python -m indycar.model.stint-predictor-fastrun --testevent Indy500-2019 --datasetid indy2013-2018-pitage --task rank
#python -m indycar.model.stint-predictor-fastrun --testevent Indy500-2019 --datasetid indy2013-2018-pitageonly --task rank
##python -m indycar.model.stint-predictor-fastrun --testevent Indy500-2019 --datasetid indy2013-2018 --task rank

