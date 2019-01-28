from Fibres import fibtools as ft
import matplotlib.pyplot as plt
from pysimulators import FitsArray
import matplotlib.mlab as mlab
import scipy.ndimage.filters as f


################################################ INPUT FILES ######################################

##### Cal Lamps List of files - Select one a bit further
###################### Src at 150 GHz ####################
name = 'ExtSrc'
fnum = 150
asic1 = '/Users/hamilton/Qubic/ExternalSource/2019-01-24_17.25.20__CalibSource 150GHz/Sums/science-asic1-2019.01.24.172520.fits'
fff = 0.5
dc = 0.5
initpars = [dc, 0.1, 0.5, 1.]
infos150 = {'name':name, 'fnum':fnum, 'asic':asic1, 'fff':fff, 'dc':dc, 'initpars':initpars}

###################### Src at 130 GHz ####################
name = 'ExtSrc'
fnum = 130
asic1 = '/Users/hamilton/Qubic/ExternalSource/2019-01-24_17.45.49__CalibSource 130GHz/Sums/science-asic1-2019.01.24.174549.fits'
fff = 0.5
dc = 0.5
initpars = [dc, 0.1, 1.2, 1.]
infos130 = {'name':name, 'fnum':fnum, 'asic':asic1, 'fff':fff, 'dc':dc, 'initpars':initpars}

###################### Src at 160 GHz ####################
name = 'ExtSrc'
fnum = 160
asic1 = '/Users/hamilton/Qubic/ExternalSource/2019-01-24_17.34.57__CalibSource 160GHz/Sums/science-asic1-2019.01.24.173457.fits'
fff = 0.5
dc = 0.5
initpars = [dc, 0.1, 0.9, 1.]
infos160 = {'name':name, 'fnum':fnum, 'asic':asic1, 'fff':fff, 'dc':dc, 'initpars':initpars}

###################### Src at 165 GHz ####################
name = 'ExtSrc'
fnum = 165
asic1 = '/Users/hamilton/Qubic/ExternalSource/2019-01-24_17.55.31__CalibSource 165GHz/Sums/science-asic1-2019.01.24.175531.fits'
fff = 0.5
dc = 0.5
initpars = [dc, 0.1, 1.4, 1.]
infos165 = {'name':name, 'fnum':fnum, 'asic':asic1, 'fff':fff, 'dc':dc, 'initpars':initpars}


############################################################
the_info = infos150
name = the_info['name']
fnum = the_info['fnum']
asic1 = the_info['asic']
fff = the_info['fff']
dc = the_info['dc']







asic = 1
### Select ASIC
if asic==1:
	theasic = asic1
else:
	theasic = asic2

FREQ_SAMPLING = (2e6/128/100)    
time, dd, a = ft.qs2array(theasic, FREQ_SAMPLING)
ndet, nsamples = np.shape(dd)

best_det = 9

###### TOD Example #####
theTES = best_det
for i in xrange(256):
    clf()   
    theTES=i
    plot(time, dd[theTES,:])
    xlabel('Time [s]')
    ylabel('Current [nA]')
    title(i)
    show()
    raw_input('press any key')




###### TOD Power Spectrum #####
theTES=best_det
frange = [0.3, 15]
spectrum, freq = mlab.psd(dd[theTES,:], Fs=FREQ_SAMPLING, NFFT=nsamples, window=mlab.window_hanning)

filt = 5
clf()
xlim(frange[0], frange[1])
rng = (freq > frange[0]) & (freq < frange[1])
filtered_spec = f.gaussian_filter1d(spectrum, filt)
loglog(freq[rng], filtered_spec[rng], label='Data')
title('Tes #{}'.format(theTES+1))
ylim(np.min(filtered_spec[rng])*0.8, np.max(filtered_spec[rng])*1.2)
xlabel('Freq [Hz]')
ylabel('Power Spectrum [$nA^2.Hz^{-1}$]')
#### Show where the signal is expected
for ii in xrange(30): plot(np.array([fff,fff])*(ii+1),[1e-20,1e-10],'r--', alpha=0.3)
#### PT frequencies
fpt = 1.724
for ii in xrange(30): plot(np.array([fpt,fpt])*(ii+1),[1e-20,1e-10],'k--', alpha=0.3)



#### Filtering out the signal from the PT
freqs_pt = [1.72383, 3.24323, 3.44727, 5.69583, 6.7533, 9.64412, 12.9874]
bw_0 = 0.005
notch = []
for i in xrange(len(freqs_pt)):
	notch.append([freqs_pt[i], bw_0*(1+i)])

sigfilt = dd[theTES,:]
for i in xrange(len(notch)):
	sigfilt = ft.notch_filter(sigfilt, notch[i][0], notch[i][1], FREQ_SAMPLING)

spectrum_f, freq_f = mlab.psd(sigfilt, Fs=FREQ_SAMPLING, NFFT=nsamples, window=mlab.window_hanning)

clf()
xlim(frange[0], frange[1])
rng = (freq > frange[0]) & (freq < frange[1])
loglog(freq[rng], filtered_spec[rng], label='Data')
loglog(freq[rng], f.gaussian_filter1d(spectrum_f,filt)[rng], label='Filt')
title('Tes #{}'.format(theTES+1))
ylim(np.min(filtered_spec[rng])*0.8, np.max(filtered_spec[rng])*1.2)
xlabel('Freq [Hz]')
ylabel('Power Spectrum [$nA^2.Hz^{-1}$]')
#### Show where the signal is expected
for ii in xrange(20): plot(np.array([fff,fff])*(ii+1),[1e-20,1e-10],'r--', alpha=0.3)
#### PT frequencies
for ii in xrange(len(freqs_pt)): plot(np.array([freqs_pt[ii],freqs_pt[ii]]),[1e-20,1e-10],'k--', alpha=0.3)

############################################################################




############################################################################
### Fold the data at the modulation period of the fibers
### Signal is also badpass filtered before folding
nper = 1
fff = 0.5/nper
lowcut = 0.01
highcut = 20.
nbins=50*nper

folded_notch, tt, folded_notch_nonorm = ft.fold_data(time, dd, 1./fff, lowcut, highcut, nbins,
	notch = notch)

### Plot it along with a guess for fiber signal
theTES = best_det
for i in xrange(128):
    clf()
    theTES = i
    plot(tt, folded_notch_nonorm[theTES,:]*1e9, label='Data TES #{} (with Notch filter)'.format(theTES))
    legend()
    ylabel('Current [nA]')
    xlabel('time [s]')
    show()
    raw_input('press a key')



tt, folded1, okfinal1, allparams1, allerr1, allchi21, ndf1 = ft.run_asic(fnum, 0, fff, dc, 
	asic1, 1, reselect_ok=False, lowcut=0.05, highcut=15., nbins=50, 
	nointeractive=False, doplot=True, notch=notch, lastpassallfree=False, okfile='TES_OK_ExtSrc150GHz_asic1.fits', name=name)






#### Now loop on the files
infos = [infos130, infos150, infos160, infos165]

the_info = infos150
name = the_info['name']
fnum = the_info['fnum']
asic1 = the_info['asic']
fff = the_info['fff']
dc = the_info['dc']
initpars = the_info['initpars']
tt, folded1, okfinal1, allparams1, allerr1, allchi21, ndf1 = ft.run_asic(fnum, 0, fff, dc, 
	asic1, 1, reselect_ok=False, lowcut=0.05, highcut=15., nbins=50, 
	nointeractive=False, doplot=True, notch=notch, lastpassallfree=False, okfile='TES_OK_ExtSrc150GHz_asic1.fits', name=name,
    initpars=initpars)






