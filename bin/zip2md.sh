
name=`basename $1`;
dname=`realpath $1`;
dirn=`dirname $dname`
fname=${name%.zip}; 
mkdir -p $fname; 

cd $fname; 
echo "unzip to $fname"
echo unzip "$dirn/$name" 
unzip "$dirn/$name" 

cd ..
