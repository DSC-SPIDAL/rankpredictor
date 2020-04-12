
cars=(12  20 9 27 28 22 29 1)
runs=(0)

for car in ${cars[*]}; do 

for run in ${runs[*]}; do

echo "============ $car ========================"
echo "python stint-test-strategy.py $car $run $1"
python stint-test-strategy.py $car $run $1
echo "===================================="

done

done
