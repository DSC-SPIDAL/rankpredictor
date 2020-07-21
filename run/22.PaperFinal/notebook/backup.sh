rm backup.tgz

find . -name "*.ini" >.filelist
find . -name "*evalua*.csv" >>.filelist
find . -name "*weight*.pdf" >>.filelist
tar czvf backup.tgz -T .filelist
