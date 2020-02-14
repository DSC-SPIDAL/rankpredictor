#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
File Parser for the eRP log file

Extract telemetry data with the help of the timing information.
+ race start timepoint (by $F command)
+ flag status in each lap ($F)
+ pitstop status in each lap ($C)

input format:
    <<INDYCAR Timing Data Document v2018.1>>
    $P�911�16:05:54.134�0.00�0.000�0�0�0�0�0.00�0�0�0�0�0�0�0�0�0.00�0.00�0.00�0.00��0�0�0�0��3��0�0�0�7�0�39.7904429�-86.2387456
    $C?U?0?R.I?1?20?0?0?0?DF5C0?T?0?0?0?0?0?0?1?0?Active?G?0?0?1?0
    $F?N?57?R.I?G?0?0?0?0?0?0?0??0?0.0000

output format:
    meta info:
        lapno, flag, pitstop,

    carno#, timestamp, lapdistance, vspeed, rpm, gear, brake, throttle, steering,lapno, flag, pitstop, anomalyscore

Usage:
    rplogex.py --extract --input inputfile --output outputfile
    example:
    python -m indycar.rplog --extract --input eRPGenerator_TGMLP_20170528_Indianapolis500_Race.log --output indy2017
    create indy2017-carno#.csv files

"""

import string
import sys,time
import os, os.path
import logging
from optparse import OptionParser
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


#constants
CMDTYPE = 0
#telemetry $P
CARNO = 1
TIMESTAMP = 2
LAPDIST = 3
MPH = 4
RPM = 5
GEAR = 6
BRAKE = 7
THROTTLE = 8
STEERING = 9
# flag $F
F_PREAMBLE = 3
F_TRACKSTATUS = 4
F_LAPNO = 5
F_GREENTIME = 6
F_YELLOWTIME = 8
# complete $C

# schema
COLUMNS=['carno', 'timestamp', 'lapdistance', 'vspeed', 'rpm', 'gear', 'brake', 'throttle', 'steering','lapno', 'flag', 'pitstop', 'anomalyscore']


#new version for multiple cmds
RECSTARTCOL = 4
COMPLETED_LAPS=['rank','car_number','unique_id','completed_laps','elapsed_time','last_laptime','lap_status', \
    'best_laptime', 'best_lap', 'time_behind_leader', 'laps_behind_leade', 'time_behind_prec',  \
    'laps_behind_prec', 'overall_rank', 'overall_best_laptime', 'current_status', 'track_status',   \
    'pit_stop_count', 'last_pitted_lap', 'start_position', 'laps_led']

#$O Overall result
OVERALL_RESULT=['resultid','deleted','marker','rank','overall_rank','start_position','best_lap_time','best_lap','last_lap_time','laps','total_time','last_warm_up_qual_time','lap1_qual_time','lap2_qual_time','lap3_qual_time','lap4_qual_time','total_qual_time','status','diff','gap','on_track','pit_stops','last_pit_lap','since_pit_laps','flag_status','no','first_name','last_name','class','equipment','license','team','total_entrant_points','total_driver_points','comment','overtake_remain','overtake_active','tire_type','best_speed','last_speed','average_speed','entrantid','teamid','driverid','offtrack']


#$L crossing information
CROSSING_INFO=['car_number','unique_identifier','time_line','source','elapsed_time','track_status','crossing_status']

#$S
COMPLETED_SECTIONS=['car_number','unique_identifier','section_identifier','elapsed_time','last_section_time','last_lap']

CMDMAP={'C':COMPLETED_LAPS,'O':OVERALL_RESULT,'L':CROSSING_INFO,'S':COMPLETED_SECTIONS}


def _buliddict(l):
    """
        build dict with key->id
    """
    return {k:id for id, k in enumerate(l)}

def _hex2int(hexstr):
    return int(hexstr, 16)

def _timestr(timestamp, scale=10000):
    s, ms = divmod(timestamp, scale)
    hours, remainder = divmod(s, 3600)
    minutes, seconds = divmod(remainder, 60)
    timestr = '{:02}:{:02}:{:02}.{:04}'.format(int(hours), int(minutes), int(seconds), int(ms))
    return timestr

def _gettime(timestr, scale=1000.):
    tmms = 0
    tmall = timestr.split('.')
    tms = [int(x) for x in tmall[0].split(':')]
    #bug fix, race start at noon
    #$P?S1?31:39.521?, Gateway-2018
    #if len(tms) == 3 and len(tmall) == 2:
    #    tmms = (tms[0] * 3600 + tms[1] * 60 + tms[2])*scale + float(tmall[1])
    if len(tmall) == 2:
        if len(tms) == 3:
            tmms = (tms[0] * 3600 + tms[1] * 60 + tms[2])*scale + float(tmall[1])
        elif len(tms) == 2:
            tmms = (tms[0] * 60 + tms[1])*scale + float(tmall[1])

    return tmms
 
class rplog:
    """
    File parser following the eRP definition
    """
    def __init__(self, logfile, outfile='', deli=chr(0xa6)):
        self.DELI = deli
        self.outf = None
        try:
            self.infname = logfile
            self.outfname = outfile
            #self.inf = open(logfile,'r',encoding="latin-1")
            self.inf = open(logfile,'r')
        
            #self.df = pd.DataFrame([], columns = COLUMNS);
            self.outfname = outfile
            self.outf = {}
            self.lastrec = {}

            logger.info('open log file successfully : %s', logfile)
        except:
            logger.error('open log file failed : %s', logfile)
            self.inf = None
            self.infname = ''
            self.outfname = ''

        #init cmd buffer
        self.dict_c = _buliddict(COMPLETED_LAPS)
        self.outf_c = open('C_' + outfile, 'w')
        self.outf_c.write(','.join(COMPLETED_LAPS) + '\n')




    def rewind(self):
        self.inf.seek(0)
        self.errcnt = 0


    def writeRecord(self, items):
        """
        Check the timestamp and write when it's valid
        """
        carno = items[CARNO]
        if not carno in self.outf:
            self.outf[carno] = open('%s-%s.csv'%(self.outfname, carno), 'w')
            self.lastrec[carno] = 0

        tmms = _gettime(items[TIMESTAMP])
        if tmms > self.lastrec[carno]:
            #bugfix, the format of datetime
            writeStr = '\t'.join(items[2:10])
            if len(items[2].split(':')) !=3:
                writeStr = '00:' + writeStr

            #self.outf[carno].write('\t'.join(items[2:10]))
            self.outf[carno].write(writeStr)
            self.outf[carno].write('\n')

            self.lastrec[carno] = tmms
        else:
            #bad records
            self.errcnt += 1


    def extract(self, save_telemetry = False):
        """
        Find the start point, $F  R.I G
        """
        self.rewind()

        #time in ms
        curTime = 0
        startTime = 0

        raceStartFlag = False
        flagStatus = 'G'
        flagInfo = [('G',0,0),('Y',0,0)] #<flagstatus, length, accumtime>


        for line in self.inf:
            #logger.info('type of line: %s', type(line))
            items = line.strip().split(self.DELI)
            cmd = items[0]
            #logger.info('items of line: %s', items)
            if cmd == '$F':
                #debug, print F cmd
                logger.info('\t'.join(items))

                if items[F_PREAMBLE] == 'R.I':
                    if raceStartFlag == False and items[F_TRACKSTATUS] == 'G':
                        #start of the game
                        raceStartFlag = True
                        flagStatus = 'G'
                        startTime = curTime
                        logger.info('Start point found at %s', _timestr(startTime))

                        continue
                    if raceStartFlag:
                        #check the flag changes
                        newflag = items[F_TRACKSTATUS] 
                        if newflag == 'K':
                            newflag = 'G'
                        if newflag != flagStatus:
                            #flag change, time is 10 thousands sec
                            if newflag == 'Y':
                                ntime = int('0x%s'%items[F_GREENTIME], 16)
                            else:
                                ntime = int('0x%s'%items[F_YELLOWTIME], 16)

                            otime = flagInfo[-2][2]
                            flagInfo.append((flagStatus, ntime - otime, ntime))

                            flagStatus = newflag


            elif cmd == '$C':
                if raceStartFlag:
                    # convert time
                    for key in self.dict_c:
                        if key.find('time') != -1:
                            id = self.dict_c[key] + RECSTARTCOL
                            items[id] = str(_hex2int(items[id])*1.0/10000)

                    #rank
                    items[RECSTARTCOL] = str(_hex2int(items[RECSTARTCOL]))
                    #completed_laps
                    id = RECSTARTCOL + self.dict_c['completed_laps']
                    items[id] = str(_hex2int(items[id]))

                    #save completed laps info
                    self.outf_c.write(','.join(items[RECSTARTCOL:]) + '\n')

            elif cmd == '$P':
                if items[TIMESTAMP].find(':') > 0:
                    tmall = items[TIMESTAMP].split('.')
                    tms = [int(x) for x in tmall[0].split(':')]
                    
                    #bug fix, race start at noon
                    #$P?S1?31:39.521?, Gateway-2018
                    if len(tmall) == 2:
                        if len(tms) == 3:
                            tmms = (tms[0] * 3600 + tms[1] * 60 + tms[2])*1000. + float(tmall[1])
                        elif len(tms) == 2:
                            tmms = (tms[0] * 60 + tms[1])*1000. + float(tmall[1])
 
                        if tmms > curTime:
                            curTime = tmms

                if raceStartFlag and save_telemetry:
                    #save records from the start point
                    self.writeRecord(items)

        #write out the flagInfo
        outf = open('%s-flag.csv'%(self.outfname), 'w')
        timestamp = startTime * 10
        for idx, rec in enumerate(flagInfo[2:]):
            #write <timestamp, len, flag>
            timestr = _timestr(timestamp)
            outf.write('%s\t%s\t%s\t%x\n'%(timestr, rec[1], rec[0], rec[2]))
            timestamp += rec[1]

        #final
        logger.info('Finish with errcnt=%s', self.errcnt)

    def extract_cmd(self, cmdType, keep = False):
        """
        extract the records of cmdType
        """
        logger.info('Start extract with cmdType:%s', cmdType)
        cnt = 0
        output = []
        for line in self.inf:
            #logger.info('type of line: %s', type(line))
            items = line.strip().split(self.DELI)
            cmd = items[0]
            #logger.info('items of line: %s', items)
            if cmd == cmdType:
                if self.outf is not None:
                    self.outf.write("\t".join(items))
                    self.outf.write("\n")
                cnt += 1
                
                #if keep:
                #    #keep in memory
                #    output.append(items)
                output.append(items)

        logger.info('End extraction with %d records', cnt)
        logger.info('Return record example %s', output[0])
        return output

    def get_telemetry(self, data, carno, col = LAPDIST):
        """
        load telemetry data for specific column
        Input:
            data    ; loaded data returned by extract()
            carno   ; select a car
            col     ; select a metric

        Return:
            np array of [[timestamp metric]]
        """
        logger.info('Start load telemetry for car:%s', 'all' if carno == -1 else carno)
        cnt = 0
        output = []   
        for items in data:
            if carno == items[CARNO] and items[TIMESTAMP].find(':') > 0:
                tmall = items[TIMESTAMP].split('.')
                tms = [int(x) for x in tmall[0].split(':')]
                tmms = (tms[0] * 3600 + tms[1] * 60 + tms[2])*1000. + float(tmall[1])
                
                metric = float(items[col])
                output.append((tmms, metric))
                cnt += 1

        logger.info('End load telemetry with %d records', cnt)
        return np.array(output)


if __name__=="__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    # logging configure
    import logging.config
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.DEBUG)
    logger.info("running %s" % ' '.join(sys.argv))

    # cmd argument parser
    usage = 'rplog.py --extract --telemetry --input inputfile --output outputfile'
    parser = OptionParser(usage)
    parser.add_option("--input", dest="inputfile")
    parser.add_option("--output", dest="outputfile")
    parser.add_option("--telemetry", action="store_true")
    parser.add_option("--extract", action="store_true")

    opt, args = parser.parse_args()

    if opt.inputfile is None:
        logger.error(globals()['__doc__'] % locals())
        sys.exit(1)

    rplog = rplog(opt.inputfile, opt.outputfile)

    rplog.extract(opt.telemetry)
