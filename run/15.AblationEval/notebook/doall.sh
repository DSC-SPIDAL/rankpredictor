tasks=('indy2013-2018' 'indy2013-2018-pitage' 'indy2013-2018-track+pitage')

for task in ${tasks[*]}; do
#./runeval.sh $task
#./runeval.sh $task-nocarid
    ./runeval.sh $task-nocarid-context40

#./runstint.sh $task
#./runstint.sh $task-nocarid
    ./runstint.sh $task-nocarid-context40

done
