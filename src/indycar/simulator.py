#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
IndyCar Simulator

laptime = ForcastedTime + FuelAdj + Random
pitstop :
    pit window,  10 laps
    in lap penalty
    pit time
    out lap penalty




"""
import sys,os,re
import datetime
import random
import numpy as np
import logging
from optparse import OptionParser


logger = logging.getLogger(__name__)

class IndySimulator():
    """
    """
    def __init__(self):
        pass

    def load_config(self, configfile):
        pass

    def run(self):





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
 

