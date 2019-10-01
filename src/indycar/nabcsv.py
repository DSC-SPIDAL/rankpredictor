#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
CSV File for NAB

Usage:
    nabcsv.py [--cmd cmdtype] --threshold threshold --input inputfile --output outputfile 
    cmdtype ; score by default, to convert nab result file
            ; flag, to convert flaginfo.csv 
            ; prune, to remove datapoints not in green flag status

"""
import sys,os,re
import datetime
import random
import numpy as np
import logging
from optparse import OptionParser


logger = logging.getLogger(__name__)

def prune(csvfile, flagfile, outfile):
    """
    csvfile ; timestamp
    flagfile; np array

    """
    flaginfo = np.loadtxt(flagfile)
    inf = open(csvfile,'r')
    outf = open(outfile, 'w')
    
    lineid = 0
    prunecnt = 0
    curend = flaginfo[0,0] + flaginfo[0,1]
    curflag = flaginfo[0,2]
    curidx = 0
    rowcnt, colcnt = flaginfo.shape

    for line in inf:
        if lineid == 0:
            #head
            outf.write('%s'%(line))
            lineid += 1
            continue

        items = line.strip().split(',')
        tmall = (items[0].split())[1].split('.')
        tms = [int(x) for x in tmall[0].split(':')]
        tmms = (tms[0] * 3600 + tms[1] * 60 + tms[2])*1000. + int(tmall[1])

        if tmms >= curend:
            #get to the end, check flag
            curidx += 1
            if curidx < rowcnt:
                curend = flaginfo[curidx,0] + flaginfo[curidx,1]
                curflag = flaginfo[curidx,2]
            else:
                curflag = 0
        
        if (curflag == 0):
            outf.write('%s'%(line))
        else:
            prunecnt += 1
        lineid += 1

    logger.info('%d lines processed, %d pruned.', lineid, prunecnt)

    inf.close()
    outf.close()



def convert_flag(csvfile, outfile):
    """
    input: (flag result csv file)
        timestamp(10 thousands), len, flag, ....
    ret:
        ms, len, flag
    """

    inf = open(csvfile,'r')
    outf = open(outfile, 'w')
    
    lineid = 0
    for line in inf:
        items = line.strip().split('\t')
        tmall = items[0].split('.')
        tms = [int(x) for x in tmall[0].split(':')]
        tmms = (tms[0] * 3600 + tms[1] * 60 + tms[2])*1000. + int(tmall[1])/10

        length = int(items[1])/10

        outf.write('%s %s %s\n'%(tmms, length, 0 if items[2]=='G' else 1))
        lineid += 1

    logger.info('%d lines converted.', lineid)

    inf.close()
    outf.close()

def _timestr(timestamp, scale=10000):
    s, ms = divmod(timestamp, scale)
    hours, remainder = divmod(s, 3600)
    minutes, seconds = divmod(remainder, 60)
    timestr = '{:02}:{:02}:{:02}.{:03}'.format(int(hours), int(minutes), int(seconds), int(ms))
    return timestr



def convert_confuse(csvfile, outfile):
    """
    add random timestamp

    input: (flag result csv file)
        timestamp(10 thousands), len, flag, ....
    """

    inf = open(csvfile,'r')
    outf = open(outfile, 'w')
    
    lineid = 0
    timestamp = 0
    for line in inf:
        if lineid == 0:
            #head
            outf.write('%s'%(line))
            lineid += 1
            continue

        items = line.strip().split(',')
        #tmall = (items[0].split())[1].split('.')
        #tms = [int(x) for x in tmall[0].split(':')]
        #tmms = (tms[0] * 3600 + tms[1] * 60 + tms[2])*1000. + int(tmall[1])

        #get random timestamp
        timestamp += random.randint(1,100)

        outf.write('2000-01-01 %s,%s\n'%(_timestr(timestamp, scale=1000), items[1]))
        lineid += 1

    logger.info('%d lines converted.', lineid)

    inf.close()
    outf.close()


def convert_score(csvfile, outfile, threshold):
    """
    input: (NAB result csv file)
        timestamp, value, anomly_score, ....
    ret:
        ms, value, 1/0
    """

    inf = open(csvfile,'r')
    outf = open(outfile, 'w')
    
    lineid = 0
    anomalycnt = 0
    for line in inf:
        #skip header
        if lineid == 0 :
            lineid += 1
            header = line.strip().split(',')
            #outf.write('%s,%s,%s\n'%(header[0],header[1],'anomaly'))
            continue

        items = line.strip().split(',')
        tmall = (items[0].split())[1].split('.')
        tms = [int(x) for x in tmall[0].split(':')]
        tmms = (tms[0] * 3600 + tms[1] * 60 + tms[2])*1000. + float(tmall[1])
 
        score = 1 if float(items[2]) > threshold else 0

        outf.write('%s %s %s\n'%(tmms, items[1], score))
        lineid += 1
        if score == 1: 
            anomalycnt += 1

    logger.info('%d lines converted, %d anomaly records found.', lineid, anomalycnt)

    inf.close()
    outf.close()

if __name__=="__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    # logging configure
    import logging.config
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.DEBUG)
    logger.info("running %s" % ' '.join(sys.argv))

    # cmd argument parser
    usage = 'nabcsv.py --threshold threshold --input inputfile --output outputfile'
    parser = OptionParser(usage)
    parser.add_option("--input", dest="inputfile")
    parser.add_option("--output", dest="outputfile")
    parser.add_option("--threshold", type=float, dest="threshold")
    parser.add_option("--flagfile", default="")
    parser.add_option("--cmd", dest="cmd", default='score')


    opt, args = parser.parse_args()

    if opt.inputfile is None:
        logger.error(globals()['__doc__'] % locals())
        sys.exit(1)

    if opt.cmd == 'score':
        convert_score(opt.inputfile, opt.outputfile, opt.threshold)
    elif opt.cmd == 'flag':
        convert_flag(opt.inputfile, opt.outputfile)
    elif opt.cmd == 'prune':
        prune(opt.inputfile, opt.flagfile,opt.outputfile)
    elif opt.cmd == 'confuse':
        convert_confuse(opt.inputfile, opt.outputfile)
 

