#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Dataset for training a model

CompletedLaps
    car_number, completed_laps, rank, elapsed_time, rank_diff, elapsed_time_diff

"""

import string
import sys,time
import os, os.path
import logging
from optparse import OptionParser
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class IndyData:
    """
    dataset class
    """
    def __init__(self, logfile, outfile='', deli=chr(0xa6)):
        logger.info('open log file successfully : %s', logfile)

    def make_completed_laps_data(self, intputfile, outputfile):
        """
        input:
            cmd c log extracted from rplog
        output:
            data frame save to outputfile
        """
        pass


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
