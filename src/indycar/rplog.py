#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
File Parser for the eRP log file

input format:
    $P�911�16:05:54.134�0.00�0.000�0�0�0�0�0.00�0�0�0�0�0�0�0�0�0.00�0.00�0.00�0.00��0�0�0�0��3��0�0�0�7�0�39.7904429�-86.2387456

Usage:
    rplog.py --extract --cmdType cmdType --input inputfile --output outputfile
    example:
    python -m indycar.rplog --extract --cmdType \$P --input eRPGenerator_TGMLP_20170528_Indianapolis500_Race.log --output P.log

"""

import string
import sys,time
import os, os.path
import logging
from optparse import OptionParser
import numpy as np

logger = logging.getLogger(__name__)


#constants
#telemetry $P
CARNO = 1
TIMESTAMP = 2
LAPDIST = 3
MPH = 4
RPM = 5


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
            self.inf = open(logfile,'r')
            if outfile:
                self.outf = open(outfile, 'w')
        except:
            logger.error('open log file failed : %s', logfile)
            self.inf = None
            self.outf = None
            self.infname = ''
            self.outfname = ''

        logger.info('open log file successfully : %s', logfile)

    def rewind(self):
        self.inf.seek(0)

    def extract(self, cmdType, keep = False):
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
    usage = 'rplog.py --extract --cmdType cmdType --input inputfile --output outputfile'
    parser = OptionParser(usage)
    parser.add_option("--input", dest="inputfile")
    parser.add_option("--output", dest="outputfile")
    parser.add_option("--cmdType", dest="cmdType")
    parser.add_option("--extract", action="store_true")

    opt, args = parser.parse_args()

    if opt.inputfile is None:
        logger.error(globals()['__doc__'] % locals())
        sys.exit(1)

    rplog = rplog(opt.inputfile, opt.outputfile)

    rplog.extract(opt.cmdType)
