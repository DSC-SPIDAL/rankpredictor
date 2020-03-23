

makelog()
{
    event=$1
    python -m indycar.rplog_stream --extract --input final/$event-2018-final.log --output $event-2018
}

makelog Indy500
makelog Gateway
makelog Pocono
makelog Iowa
makelog Texas
makelog Phoenix

