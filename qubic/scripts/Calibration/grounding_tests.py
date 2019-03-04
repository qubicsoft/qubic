from Calibration import fibtools as ft
from Calibration.plotters import *

from matplotlib.pyplot import *
from pysimulators import FitsArray
import matplotlib.mlab as mlab
import scipy.ndimage.filters as f
import glob
from qubicpack import qubicpack as qp
import string

basedir = '/Users/hamilton/Qubic/Grounding/'
dirs = glob.glob(basedir+'/*')

motor = []
grounding = []
for i in xrange(len(dirs)):
    bla = string.split(dirs[i],'_')
    grounding.append(bla[-2])
    motor.append(bla[-1])


a1 = qp()
a1.read_qubicstudio_dataset(dirs[0], asic=1)
a2 = qp()
a2.read_qubicstudio_dataset(dirs[0], asic=2)

nsamples = len(a1.timeline(TES=66))
FREQ_SAMPLING = 1./a1.sample_period()
spectrum, freq = mlab.psd(2
a1.timeline(TES=66), Fs=FREQ_SAMPLING, NFFT=nsamples, window=mlab.window_hanning)


ok1 = np.array(FitsArray('/Users/hamilton/Qubic/ExternalSource/ScanAz2019-01-30_OK_Asic1.fits'), dtype=bool)
ok2 = np.array(FitsArray('/Users/hamilton/Qubic/ExternalSource/ScanAz2019-01-30_OK_Asic2.fits'), dtype=bool)
ok = np.append(ok1,ok2)


clf()
plot(freq, spectrum)
xscale('log')
yscale('log')

allspecs = []
allfreqs = []
medspec = []
for i in xrange(len(dirs)):
    a1 = qp()
    a1.read_qubicstudio_dataset(dirs[i], asic=1)
    a2 = qp()
    a2.read_qubicstudio_dataset(dirs[i], asic=2)
    nsamples = len(a1.timeline(TES=66))
    FREQ_SAMPLING = 1./a1.sample_period()

    specs = np.zeros((256, nsamples/2+1))
    for j in xrange(128):
        spectrum, freq = mlab.psd(a1.timeline(TES=j+1), Fs=FREQ_SAMPLING, NFFT=nsamples, window=mlab.window_hanning)
        specs[j,:] = spectrum
        spectrum, freq = mlab.psd(a2.timeline(TES=j+1), Fs=FREQ_SAMPLING, NFFT=nsamples, window=mlab.window_hanning)
        specs[j+128,:] = spectrum
    allspecs.append(specs)
    allfreqs.append(freq)
    medspec.append(np.median(specs[ok,:],axis=0))




for theTES in xrange(256):
    median=False
    filt = 10
    #cpls = [[0,1], [3,2], [0,3], [1,2]]
    cpls = [[0,1], [4,5], [0,4], [1,5]]
    clf()
    for j in xrange(len(cpls)):
        subplot(len(cpls),1,j+1)
        xscale('log')
        yscale('log')
        xlim(0.9, FREQ_SAMPLING/2)
        ylim(1e5, 1e8)
        for i in cpls[j]:
            if median:
                toplot = medspec[i]
            else:
                toplot = allspecs[i][theTES,:]
            plot(allfreqs[i], f.gaussian_filter1d(toplot,filt), label=grounding[i]+'-'+motor[i])
        legend(loc='lower left', fontsize=8)
        xlabel('Frequency [Hz]')
        if j==0: 
            if median:
                title('Median of {} good TES'.format(ok.sum()))
                fname = 'median.png'
            else:
                title('TES={} - Status={}'.format(theTES, ok[theTES]))
                fname = 'tes{}.png'.format(theTES)
    #savefig(fname)
    show()
    raw_input('Press a key')










