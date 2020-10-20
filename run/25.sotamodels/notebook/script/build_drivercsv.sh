
for logfile in `ls final/*.log`; do grep "^\$E" ${logfile} | sed -e 's/\xa6/\t/g' | gawk -F'\t' '{print($5","$7","$11)}'  |sort |uniq >`basename $logfile`.driver; done

for logfile in `ls *.driver`; do
fname=`basename $logfile`
race=`echo $fname | gawk -F'-' '{print $1}'`
year=`echo $fname | gawk -F'-' '{print $2}'`
sed "s/^/$race,$year,&/" $logfile >>drivers.csv
done
