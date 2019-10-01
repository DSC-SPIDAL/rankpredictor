#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Plot engine for the eRP analysis

plottype:
    scatter ;   such as <timestamp, metric> scatter plot
    scatter_w_anomaly   ; scatter with anomly markers

Usage:
    plot.py --plot plotname --input inputfile --output outputfile 

"""
import sys,os,re
import datetime
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import matplotlib.ticker as ticker
from optparse import OptionParser
from rplog import *


logger = logging.getLogger(__name__)

class PlotConf():
    def __init__(self, configfile):
        """ 
        Configure file
        title : .....
        """
        basicname = ['title', 'xlable','ylabel']

        config = {}
        conf = open(configfile, 'r')
        for line in conf:
            tokens = line.strip().split(':')
            config[tokens[0]] = tokens[1]

        for name in basicname:
            if name not in config:
                config[name] = name

        self.config = config

    def __getitem__(self, name):
        if name in self.config:
            return self.config[name]
        else:
            return None


class PlotEngine():

    def __init__(self, use_shortest_x = True):
        self.ploters = {
            "scatter":self.plot_scatter,
            "anomaly":self.plot_anomaly,
            "flaginfo":self.plot_flaginfo
        }

        # init default subplot
        self.init_subplot(1,1)
        self.set_subplot(1,1)

        self.colors_orig=['r','b','g', 'm','c','y','k','r','b','m','g','c','y','k']*5
        self.colors_orig_shade=['#ff8080','#8080ff','#80ff80', 'm','c','y','k','r','b','m','g','c','y','k']
        #self.colors=[(name, hex) for name, hex in matplotlib.colors.cnames.iteritems()]
        #self.colors=[hex for name, hex in matplotlib.colors.cnames.iteritems()]
        self.colors_stackbar=['r','y','b','g', 'm','c','y','k','r','y','b','g','m','c','y','k']*5
        self.linestyle=['-','--','-.',':']
        self.marker=['.','o','^','s','*','3']
        #default setting
        self.colors = self.colors_orig
        self.lines = ['.-'] * 40
        self.lines2 = ['--'] * 40

        self.use_shortest_x = use_shortest_x

        self.dataset={
            'pubmed2m':149031108
        }

    def init_data(self, data, perfname):
        """
        perfdata is PerfData object
        perfname is {"name":label} object
        """
        self.perfdata = data
        self.perfname = perfname

    def init_subplot(self,*args):
        """
        set up sub plot info

        """
        #f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
        self.fig, self.ax = plt.subplots(*args)
        logger.info('init_subplots as %s, axarr shape=%s', args, self.ax)
        #self.fig.set_size_inches(9.25*1.5, 5.25*1.5)

        x = args[0]
        y = args[1]
        if x == 1 and y == 1:
            self.axtype = 1
        elif x == 1 or y == 1:
            self.axtype = 2
        else:
            self.axtype = 3

    def set_subplot(self, x, y):
        if self.axtype == 1:
            self.curax = self.ax
        elif self.axtype == 2:
            x = max(x, y)
            self.curax = self.ax[x-1]
        else:
            self.curax = self.ax[x-1,y-1]
   
        logger.info('set_subplot curax = %s', self.curax)

    def savefig(self, figname):
        # plt.savefig(figname, dpi=300)
        self.fig.tight_layout()
        self.fig.savefig(figname)

    def autolabel(self, rects, realnum = 0):
        """
        input:
            bar
        """
        for rect in rects:
            height = rect.get_height()
            if realnum > 0:
                self.curax.text(rect.get_x()+rect.get_width()/2., 1.01*height, '%.2f'%(float(height)/realnum),                    ha='center', va='bottom')
            else:
                self.curax.text(rect.get_x()+rect.get_width()/2., 1.01*height, '%.0f'%float(height),                    ha='center', va='bottom')

    def autolabel_stack(self, rects):
        """
        input:
            (bar1, bar2)
        """
        old_height = []
        for rect in rects[0]:
            height = rect.get_height()
            self.curax.text(rect.get_x()+rect.get_width()/2., 1.01*height, '%d'%int(height),
                        ha='center', va='bottom')
            old_height.append(height)

        # stack the next one
        idx = 0
        for rect in rects[1]:
            height2 = rect.get_height()
            self.curax.text(rect.get_x()+rect.get_width()/2., 1.01*(old_height[idx]+height2), '%d'%int(height2),
                        ha='center', va='bottom')
            idx += 1

    ##################################################
    def plot(self, plotname, fig, conf):
        if plotname in self.ploters:
            logger.info('='*20)
            logger.info('StartPlot: %s',plotname)
            logger.info('='*20)
            return self.ploters[plotname](fig, conf)
        else:
            logger.error('plotname %s not support yet', plotname)


    ######################
    def plot_scatter(self, figname, conf):
        """

        """

        #setup the line style and mark, colors, etc
        colors = self.colors
        lines = self.lines

        if 'lines' in conf:
            lines = conf['lines']
        if 'colors' in conf:
            colors = conf['colors']

        origin_x = self.perfdata[:,0]
        #self.curax.scatter(self.perfdata[:,0], self.perfdata[:,1], c = colors)
        self.curax.plot(self.perfdata[:,0], self.perfdata[:,1],'.')


        xtick_scale = 1000
        if 'xtick_scale' in conf:
            xtick_scale = conf['xtick_scale']
 
        xticks = [('%.2f'%(x/xtick_scale)) for x in origin_x]
        #self.curax.set_xticklabels( xticks)
        
        if 'xlabel' in conf:
            self.curax.set_xlabel(conf['xlabel'])
        else:
            self.curax.set_xlabel('timestamp(x%.0e)'%xtick_scale)
        #self.curax.set_ylabel('RPM')
        if 'ylabel' in conf:
            self.curax.set_ylabel(conf['ylabel'])
        else:
            self.curax.set_ylabel('vspeed')

        if 'title' in conf:
            self.curax.set_title(conf['title'])
        else:
            self.curax.set_title('RPM .vs. Time')
 
        if not 'nolegend' in conf:
            if 'loc' in conf:
                self.curax.legend(loc = conf['loc'])
            else:
                self.curax.legend(loc = 2)
        if figname:
            self.savefig(figname)

    def plot_flaginfo(self, figname, conf):
        """
        timestamp, length, flag
        """
        rowcnt, colcnt = self.perfdata.shape

        x = [0,0]
        y = [-20,-20]
        for idx in range(rowcnt):
            x[0] = self.perfdata[idx,0]
            x[1] = x[0] + self.perfdata[idx,1]
            color = 'g' if self.perfdata[idx,2] == 0 else 'y'

            self.curax.plot(x, y,'.-', markersize = 10, c = color, linewidth = 10)

    def plot_anomaly(self, figname, conf):
        """
        teimstamp, value, anomaly(0/1)
        """

        #setup the line style and mark, colors, etc
        colors = self.colors
        lines = self.lines

        if 'lines' in conf:
            lines = conf['lines']
        if 'colors' in conf:
            colors = conf['colors']

        origin_x = self.perfdata[:,0]
        #self.curax.scatter(self.perfdata[:,0], self.perfdata[:,1], c = colors)
        #self.curax.plot(self.perfdata[:,0], self.perfdata[:,1],'-', markersize = 1)
        self.curax.plot(self.perfdata[:,0], self.perfdata[:,1],'.', markersize = 1)
        anomaly = self.perfdata[self.perfdata[:,2] == 1]
        self.curax.scatter(anomaly[:,0], anomaly[:,1], s = 40, marker = '^', c = 'r')


        xtick_scale = 1000
        if 'xtick_scale' in conf:
            xtick_scale = conf['xtick_scale']
 
        #xticks = [('%.2f'%(x/xtick_scale)) for x in origin_x]
        #xticks = ['%s:%s.%s'%((int(x/1000)%3600)/60, int(x/1000)%60,int(x)%1000) for x in origin_x]
        #xticks = ['%s:%s'%((int(x/1000)%3600)/60, int(x/1000)%60) for x in origin_x]
        #self.curax.set_xtick(xticks[origin_x[::1000]])
        #self.curax.set_xticklabels(xticks[::1000])
        #get the time span
        tickspan = int((origin_x[-1] - origin_x[0])/20)
        tickval = range(int(origin_x[0]), int(origin_x[-1]), tickspan)
        xticks = ['%s:%s:%s'%(int(x/1000)/3600,(int(x/1000)%3600)/60, int(x/1000)%60) for x in tickval]

        self.curax.set_xticks(tickval)
        self.curax.set_xticklabels(xticks)
        
        if 'xlabel' in conf:
            self.curax.set_xlabel(conf['xlabel'])
        else:
            self.curax.set_xlabel('timestamp')
        #self.curax.set_ylabel('RPM')
        if 'ylabel' in conf:
            self.curax.set_ylabel(conf['ylabel'])
        else:
            self.curax.set_ylabel('vspeed')

        
        if 'title' in conf:
            self.curax.set_title(conf['title'])
        else:
            self.curax.set_title('RPM .vs. Time')
 
        if not 'nolegend' in conf:
            if 'loc' in conf:
                self.curax.legend(loc = conf['loc'])
            else:
                self.curax.legend(loc = 2)
        if figname:
            self.savefig(figname)


if __name__=="__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    # logging configure
    import logging.config
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.DEBUG)
    logger.info("running %s" % ' '.join(sys.argv))

    # cmd argument parser
    usage = 'plot.py --plot plotname --input inputfile --output outputfile'
    parser = OptionParser(usage)
    parser.add_option("--input", dest="inputfile")
    parser.add_option("--output", dest="outputfile")
    parser.add_option("--plot", dest="plotname")
    parser.add_option("--flaginfo", default='')

    opt, args = parser.parse_args()

    if opt.inputfile is None:
        logger.error(globals()['__doc__'] % locals())
        sys.exit(1)

    if opt.inputfile.find('log') > 0:
        rplog = rplog(opt.inputfile)
        tdata = rplog.extract('$P', keep = True)
        pdata = rplog.get_telemetry(tdata, '9', RPM)
        logger.info('shape of pdata: %s', pdata.shape)
        np.savetxt('pdata.csv', pdata)
    #elif opt.inputfile.find('npy') > 0:
    #    pdata = np.loadtxt(opt.inputfile)
    #elif opt.inputfile.find('csv') > 0:
    #    #load original .csv input to NAB
    #    pdata = loadnabcsv(opt.inputfile)
    else:
        pdata = np.loadtxt(opt.inputfile)


    #load flaginfo
    flaginfo = None
    if opt.flaginfo:
        flaginfo = np.loadtxt(opt.flaginfo)

    plt.rcParams.update({'figure.figsize': (32,3)})

    ploter = PlotEngine()
    ploter.init_data(pdata, '')
    #ploter.plot(opt.plotname, opt.outputfile, {'title':opt.inputfile})
    ploter.plot(opt.plotname, '', {'title':opt.inputfile})

    ploter.init_data(flaginfo, '')
    ploter.plot('flaginfo', '', {})
    ploter.savefig(opt.outputfile)





