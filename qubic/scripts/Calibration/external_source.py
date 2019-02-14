# from Calibration import fibtools as ft
# import Calibration.plotters as p

import fibtools as ft
import plotters as p

import numpy as np
from matplotlib.pyplot import *
from pysimulators import FitsArray

import matplotlib.mlab as mlab
import scipy.ndimage.filters as f

################################################ INPUT FILES ######################################

basedir = '/home/louisemousset/QUBIC/Qubic_work/Calibration/2019-01-24'
# basedir = '/home/james/fibdata/'
# basedir = '/Users/hamilton/Qubic/ExternalSource/'


##### Cal Lamps List of files - Select one a bit further
###################### Src at 150 GHz ####################
name = 'ExtSrc'
fnum = 150
asic1 = basedir + '/2019-01-24_17.25.20__CalibSource 150GHz/Sums/science-asic1-2019.01.24.172520.fits'
fff = 0.5
dc = 0.5
initpars = [dc, 0.1, 0.5, 1.]
infos150 = {'name': name, 'fnum': fnum, 'asic': asic1, 'fff': fff, 'dc': dc, 'initpars': initpars}

###################### Src at 130 GHz ####################
name = 'ExtSrc'
fnum = 130
asic1 = basedir + '/2019-01-24_17.45.49__CalibSource 130GHz/Sums/science-asic1-2019.01.24.174549.fits'
fff = 0.5
dc = 0.5
initpars = [dc, 0.1, 1.2, 1.]
infos130 = {'name': name, 'fnum': fnum, 'asic': asic1, 'fff': fff, 'dc': dc, 'initpars': initpars}

###################### Src at 160 GHz ####################
name = 'ExtSrc'
fnum = 160
asic1 = basedir + '/2019-01-24_17.34.57__CalibSource 160GHz/Sums/science-asic1-2019.01.24.173457.fits'
fff = 0.5
dc = 0.5
initpars = [dc, 0.1, 0.9, 1.]
infos160 = {'name': name, 'fnum': fnum, 'asic': asic1, 'fff': fff, 'dc': dc, 'initpars': initpars}

###################### Src at 165 GHz ####################
name = 'ExtSrc'
fnum = 165
asic1 = basedir + '/2019-01-24_17.55.31__CalibSource 165GHz/Sums/science-asic1-2019.01.24.175531.fits'
fff = 0.5
dc = 0.5
initpars = [dc, 0.1, 1.4, 1.]
infos165 = {'name': name, 'fnum': fnum, 'asic': asic1, 'fff': fff, 'dc': dc, 'initpars': initpars}

############################################################
the_info = infos150
asic1 = the_info['asic']
fff = the_info['fff']
dc = the_info['dc']

# Select ASIC
asic = 1
if asic == 1:
    theasic = asic1
else:
    theasic = asic2

FREQ_SAMPLING = (2e6 / 128 / 100)
time, dd, a = ft.qs2array(theasic, FREQ_SAMPLING)
ndet, nsamples = np.shape(dd)

best_det = 9

###### TOD Example #####
theTES = best_det
clf()
plot(time, dd[theTES, :])
xlabel('Time [s]')
ylabel('Current [nA]')
title(theTES)
show()
# raw_input('press any key')


###### TOD Power Spectrum #####
theTES = best_det
frange = [0.3, 15]  # range of plot frequencies desired
filt = 5
spectrum, freq = mlab.psd(dd[theTES, :], Fs=FREQ_SAMPLING, NFFT=nsamples, window=mlab.window_hanning)
filtered_spec = f.gaussian_filter1d(spectrum, filt)

p.FreqResp(freq, frange, filtered_spec, theTES, fff)

#### Filtering out the signal from the PT
freqs_pt = [1.72383, 3.24323, 3.44727, 5.69583, 6.7533, 9.64412, 12.9874]  # frequencies of pics from PT
bw_0 = 0.005  # Bandwidth of the filter

notch = ft.notch_array(freqs_pt, bw_0)

p.FiltFreqResp(theTES, frange, fff, filt, dd, notch, FREQ_SAMPLING, nsamples, freq, spectrum, filtered_spec)
############################################################################




############################################################################
# Fold the data at the modulation period of the fibers
# Signal is also bandpass filtered before folding
nper = 1  # number of periods
fff = 0.5 / nper
lowcut = 0.01
highcut = 20.
nbins = 50 * nper

# Not filtered
folded, tt, folded_nonorm = ft.fold_data(time, dd, 1. / fff, lowcut, highcut, nbins)

# Filtered
folded_notch, tt, folded_notch_nonorm = ft.fold_data(time, dd, 1. / fff, lowcut, highcut, nbins, notch=notch)

# plot folded TES data along with a guess for source signal
pars = [dc, 0.05, 0., 1.2]
p.FoldedFiltTES(tt, pars, theTES, folded, folded_notch)

# now plot it with a fitting of the source signal parameters
guess = [dc, 0.06, 0., 1.2]
bla = ft.do_minuit(tt, folded[theTES, :], np.ones(len(tt)), guess, functname=ft.simsig,
                   rangepars=[[0., 1.], [0., 1], [0., 1], [0., 1]], fixpars=[0, 0, 0, 0], force_chi2_ndf=True,
                   verbose=False, nohesse=True)
p.FoldedTESFreeFit(tt, bla, theTES, folded)

###################### Now loop on the files ##################################
infos = [infos130, infos150, infos160, infos165]

pars = []
for i in xrange(len(infos)):
    the_info = infos[i]
    fnum = the_info['fnum']
    asic1 = the_info['asic']
    fff = the_info['fff']
    dc = the_info['dc']
    initpars = the_info['initpars']
    tt, folded1, okfinal1, allparams1, allerr1, allchi21, ndf1 = ft.run_asic(fnum, 0, fff, dc, asic1, 1,
                                                                             reselect_ok=False, lowcut=0.05,
                                                                             highcut=15., nbins=50,
                                                                             rangepars=[[0., 1.], [0., 0.5],
                                                                                        [0., 1. / fff], [0., 5000.]],
                                                                             initpars=initpars,
                                                                             okfile='TES_OK_ExtSrc150GHz_asic1.fits')
    pars.append(allparams1)

img_amp = np.zeros((128, len(infos))) + np.nan
img_tau = np.zeros((128, len(infos))) + np.nan
for i in xrange(len(infos)):
    img_amp[okfinal1, i] = pars[i][okfinal1, 3]
    img_tau[okfinal1, i] = pars[i][okfinal1, 1]

# Time response tau
clf()
for i in xrange(len(infos)):
    subplot(2, 2, i + 1)
    imshow(ft.image_asics(data1=img_tau[:, i]), vmin=0, vmax=0.2, interpolation='nearest')
    title('tau {} {} GHz'.format(infos[i]['name'], infos[i]['fnum']))
    colorbar()
tight_layout()

# Amplitude
clf()
for i in xrange(len(infos)):
    subplot(2, 2, i + 1)
    imshow(ft.image_asics(data1=img_amp[:, i]), vmin=0, vmax=500, interpolation='nearest')
    title('Amp {} {} GHz'.format(infos[i]['name'], infos[i]['fnum']))
    colorbar()
tight_layout()

freqs = np.array([infos[i]['fnum'] for i in xrange(len(infos))])
ampsok = img_amp[okfinal1, :]
av_shape = np.zeros(len(infos))
err_shape = np.zeros(len(infos))
for i in xrange(len(infos)):
    av_shape[i], err_shape[i] = ft.meancut(ampsok[:, i] / ampsok[:, 1], 4)

clf()
errorbar(freqs, av_shape, yerr=err_shape / np.sqrt(okfinal1.sum()), fmt='ko-')
ylim(0, 1.2)
xlabel('Freq. [GHz]')
ylabel('Relative Signal amplitude (w.r.t. 150 GHz)')
# savefig('filter.png')
