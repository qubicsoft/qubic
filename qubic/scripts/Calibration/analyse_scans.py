# %matplotlib notebook
from matplotlib import rc
#rc('figure',figsize=(16,8))
#rc('font',size=12)
#rc('text',usetex=False)

from qubicpack import qubicpack as qp
import fibtools as ft
import plotters as p
import lin_lib as ll
import demodulation_lib as dl
import satorchipy as stpy
from pysimulators import FitsArray

import numpy as np
from matplotlib.pyplot import *
import matplotlib.mlab as mlab
import scipy.ndimage.filters as f
import glob
import string
import scipy.signal as scsig
from scipy import interpolate
import datetime as dt
import sys


# n = 'ScanMap_Speed_VE4_El'
# days = ['2019-03-30', '2019-03-31']
# freq_mod = 1. ## Hz
# doit = 1

n = sys.argv[1]
freq_mod = float(sys.argv[2])
doit = int(sys.argv[3])
replace = int(sys.argv[4])
method = sys.argv[5]
days = sys.argv[6:]
print n
print freq_mod
print doit
print days



print(n)
print days
dirs = []
elevations=[]
for d in days:
    dd = glob.glob('/qubic/Data/Calib-TD/'+d+'/*'+n+'*')
    for i in xrange(len(dd)): 
        #print dd[i]
        truc = str.split(dd[i],'_')
        the_el = truc[-1]
        try:
            elfloat = np.float(the_el)
            elevations.append(np.float(the_el))
            dirs.append(dd[i])
        except:
            print 'File '+dd[i]+' has a format that des not comply with expectations => rejected'
            continue
    print '  * ',d,' : {} files'.format(len(dd))
print '  => Total = {} files'.format(len(dirs))
print '  => Elevation ranges from {} to {}'.format(np.min(elevations),np.max(elevations))


labels = []
dir_time = []
for d in dirs:
    bla = str.split(d,'__')
    blo = str.split(bla[0],'/')
    labels.append(bla[1])
    dir_time.append(blo[-1])

for i in xrange(len(labels)): 
    print i, labels[i], dir_time[i], 'Elevation: ', elevations[i]


if doit==1:
	#### Parameters
	ppp = 1./freq_mod
	lowcut = 0.3
	highcut = 10.

	nbins = 250

	reload(dl)
	reload(ft)
	savedir = './'

	ids=0

	for ii in xrange(len(dirs)):
		thedir = dirs[ii]
		print ''
		print '##############################################################'
		print 'Directory {} / {} :'.format(ii, len(dirs)), thedir
		print '##############################################################'
		f0 = glob.glob(savedir+'alltes_unbinned_{}_{}.fits'.format(n,elevations[ii]))
		f1 = glob.glob(savedir+'angles_unbinned_{}_{}.fits'.format(n,elevations[ii]))
		filesalreadydone = ((f0!=[]) & (f1!=[]))
		filesnotdone = bool(not filesalreadydone)
		dothejob = bool(replace or filesnotdone)
		if dothejob:
			if filesalreadydone:
				print 'files already exist on disk but I was asked to replace them so doing the job'
			else: 
				print 'Files do not exist so doing the job'
			allsb = []
			all_az_el_azang = []
			for iasic in [0,1]:
			    print '======== ASIC {} ====================='.format(iasic)
			    AsicNum = iasic+1
			    a = qp()
			    a.read_qubicstudio_dataset(thedir, asic=AsicNum)
			    data=a.azel_etc(TES=None)
			    data['t_src'] += 7200
			    unbinned, binned = dl.general_demodulate(ppp, data, 
			                                            lowcut, highcut,
			                                            nbins=nbins, median=True, method=method, 
			                                            doplot=False, rebin=False, verbose=False)
			    all_az_el_azang.append(np.array([unbinned['az'], unbinned['el'], unbinned['az_ang']]))
			    allsb.append(unbinned['sb'])
			sh0 = allsb[0].shape
			sh1 = allsb[1].shape
			mini = np.min([sh0[1], sh1[1]])
			sb = np.append(allsb[0][:,:mini], allsb[1][:,:mini], axis=0)
			az_el_azang = np.array(all_az_el_azang[0][:,:mini])
			print az_el_azang.shape
			print sb.shape
			FitsArray(sb).save(savedir+'alltes_unbinned_{}_{}.fits'.format(n,elevations[ii]))
			FitsArray(az_el_azang).save(savedir+'angles_unbinned_{}_{}.fits'.format(n,elevations[ii]))
		else:
			if filesalreadydone:
				print 'files already exist on disk and I was asked not to replace them so doing nothing'
			else: 
				print 'This should not happen... There is a bug'












