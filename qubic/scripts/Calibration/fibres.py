import numpy as np
import fibtools as ft
import matplotlib.pyplot as plt
from pysimulators import FitsArray
import matplotlib.mlab as mlab
import scipy.ndimage.filters as f
from plotters import FreqResp


################################################ INPUT FILES ######################################

##### Fiber_2 Fiber@1V; Freq=1Hz; DutyCycle=30%; Voffset_TES=3V
fib = 2
Vtes = 3.
fff = 1.
dc = 0.3
asic1 = '/home/james/fibdata/2018-12-20/2018-12-20_17.27.22__Fiber_2/Sums/science-asic1-2018.12.20.172722.fits'
asic2 = '/home/james/fibdata/2018-12-20/2018-12-20_17.27.22__Fiber_2/Sums/science-asic2-2018.12.20.172722.fits'


##### Fiber 3: Fiber@1V; Freq=1.5Hz; DutyCycle=50%; Voffset_TES=3V
# fib = 3
# Vtes = 3.
# fff = 1.5
# dc = 0.5
# asic1 = '/Users/hamilton/Qubic/Fibres/2018-12-20/2018-12-20_17.52.08__Fiber_3/Sums/science-asic1-2018.12.20.175208.fits'
# asic2 = '/Users/hamilton/Qubic/Fibres/2018-12-20/2018-12-20_17.52.08__Fiber_3/Sums/science-asic2-2018.12.20.175208.fits'

# ##### Fiber 4: Fiber@1V; Freq=1Hz; DutyCycle=50%; Voffset_TES=2.6V
# fib = 4
# Vtes = 2.6
# fff = 1.
# dc = 0.5
# asic1 = '/home/louisemousset/QUBIC/Qubic_work/Calibration/2018-12-20/2018-12-20_17.27.22__Fiber_2/Sums/science-asic1-2018.12.20.172722.fits'
# asic2 = '/home/louisemousset/QUBIC/Qubic_work/Calibration/2018-12-20/2018-12-20_17.27.22__Fiber_2/Sums/science-asic2-2018.12.20.172722.fits'
############################################################################



############################# Reading files Example ########################
asic = 1
reselect_ok = True

### Select ASIC
if asic==1:
	theasic = asic1
else:
	theasic = asic2

FREQ_SAMPLING = (2e6/128/100)    
time, dd, a = ft.qs2array(theasic, FREQ_SAMPLING)
ndet, nsamples = np.shape(dd)
#### Selection of detectors for which the signal is obvious
good_dets=[37, 45, 49, 70, 77, 90, 106, 109, 110]
best_det = 37

###### TOD Example #####
theTES = best_det
clf()
plot(time, dd[theTES,:])
xlabel('Time [s]')
ylabel('Current [nA]')

###### TOD Power Spectrum #####

#theTES=best_det
frange = [0.3, 15]
filt = 5
FreqResp(best_det, frange, fff, filt)
#spectrum, freq = mlab.psd(dd[theTES,:], Fs=FREQ_SAMPLING, NFFT=nsamples, window=mlab.window_hanning)
#filt = 5
#clf()
#xlim(frange[0], frange[1])
#rng = (freq > frange[0]) & (freq < frange[1])
#filtered_spec = f.gaussian_filter1d(spectrum, filt)
#loglog(freq[rng], filtered_spec[rng], label='Data')
#title('Tes #{}'.format(theTES+1))
#ylim(np.min(filtered_spec[rng])*0.8, np.max(filtered_spec[rng])*1.2)
#xlabel('Freq [Hz]')
#ylabel('Power Spectrum [$nA^2.Hz^{-1}$]')
##### Show where the signal is expected
#for ii in xrange(10): plot(np.array([fff,fff])*(ii+1),[1e-20,1e-10],'r--', alpha=0.3)
##### PT frequencies
#fpt = 1.724
#for ii in xrange(10): plot(np.array([fpt,fpt])*(ii+1),[1e-20,1e-10],'k--', alpha=0.3)


###NEW PLOT WITH FILTERED OVERPLOT
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
for ii in xrange(10): plot(np.array([fff,fff])*(ii+1),[1e-20,1e-10],'r--', alpha=0.3)
#### PT frequencies
fpt = 1.724
for ii in xrange(10): plot(np.array([fpt,fpt])*(ii+1),[1e-20,1e-10],'k--', alpha=0.3)

############################################################################



############################################################################
### Fold the data at the modulation period of the fibers
### Signal is also badpass filtered before folding
lowcut = 0.5
highcut = 15.
nbins=50
folded, tt, folded_nonorm = ft.fold_data(time, dd, 1./fff, lowcut, highcut, nbins)

folded_notch, tt, folded_notch_nonorm = ft.fold_data(time, dd, 1./fff, lowcut, highcut, nbins,
	notch = notch)

### Plot it along with a guess for fiber signal
theTES = best_det
plt.clf()
plt.plot(tt, folded[theTES,:], label='Data TES #{}'.format(theTES))
plt.plot(tt, folded_notch[theTES,:], label='Data TES #{} (with Notch filter)'.format(theTES))
plt.plot(tt, ft.simsig(tt, [dc, 0.05, 0.0, 1.2]), label='Expected')
plt.legend()
plt.ylabel('Current [nA]')
plt.xlabel('time [s]')


#### Now fit the fiber signal 
#### NB errors come from renormalizing chi2/ndf to 1
guess = [dc, 0.06, 0., 1.2]
bla = ft.do_minuit(tt, folded[theTES,:], np.ones(len(tt)),
	guess, functname=ft.simsig,
	rangepars=[[0.,1.], [0., 1], [0.,1], [0., 1]], fixpars=[0,0,0,0], 
	force_chi2_ndf=True, verbose=False, nohesse=True)
params =  bla[1]
err = bla[2]

plt.clf()
plt.plot(tt, folded[theTES,:], label='Data TES #{}'.format(theTES))
plt.plot(tt, ft.simsig(tt, bla[1]), label='Fitted: \n cycle={0:8.3f}+/-{1:8.3f} \n tau = {2:8.3f}+/-{3:8.3f}s \n t0 = {4:8.3f}+/-{5:8.3f}s \n amp = {6:8.3f}+/-{7:8.3f}'.format(params[0], err[0], params[1], err[1], params[2], err[2], params[3], err[3]))
plt.legend()
plt.ylabel('Current [nA]')
plt.xlabel('time [s]')



##################################################
### Now do a kind a multipass analysis where
### TES exhibiting nice signal are manually picked
### I tried an automated algorithm but it was not
### very satisfying...
### Running the following with the keyword
### reselect_ok = True 
### will go through all TES and ask for a [y] if it 
### is a valid signal.
### This is to be done twice: first pass is used 
### to determine the av. start time from the best TES
### Second time fits with this start time
### as well as the duty cycle of the fibers forced.
### Therefore at the end we have a fit of two
### variables for each TES: time constant and
### signal amplitude
### Once the reselect_ok=True case has been done,
### a file is created with the list of valid TES
### and the code can be ran in a much faster way
### with reslect_ok=False
##################################################
#### Asic 1
tt, folded1, okfinal1, allparams1, allerr1, allchi21, ndf1 = ft.run_asic(fib, Vtes, fff, dc, 
	asic1, 1, reselect_ok=False, lowcut=0.5, highcut=15., nbins=50, 
	nointeractive=False, doplot=True, notch=notch)
plt.savefig('fib{}_ASIC1_summary.png'.format(fib))


#### Asic 2
tt, folded2, okfinal2, allparams2, allerr2, allchi22, ndf2 = ft.run_asic(fib, Vtes, fff, dc, 
	asic2, 2, reselect_ok=False, lowcut=0.5, highcut=15., nbins=50, 
	nointeractive=False, doplot=True, notch=notch)
plt.savefig('fib{}_ASIC2_summary.png'.format(fib))


#### Additional cuts:
tau_max = 0.3
okfinal1 = okfinal1 * (allparams1[:,1] < tau_max)
okfinal2 = okfinal2 * (allparams2[:,1] < tau_max)

#### Combine data
folded = np.append(folded1, folded2, axis=0)
okfinal = np.append(okfinal1, okfinal2, axis=0)
allparams = np.append(allparams1, allparams2, axis=0)
allerr = np.append(allerr1, allerr2, axis=0)
allchi2 = np.append(allchi21, allchi22, axis=0)
ndf = ndf1

plt.figure()
plt.clf()
med = False
rng = [0,0.4]
plt.subplot(2,2,1)
plt.hist(allparams[okfinal, 1],range=rng,bins=30,label='All ({}) '.format(okfinal.sum())+ft.statstr(allparams[okfinal,1]*1000, median=med)+' ms', alpha=0.5)
plt.hist(allparams1[okfinal1, 1],range=rng,bins=30,label='Asic1 ({})'.format(okfinal1.sum())+ft.statstr(allparams1[okfinal1,1]*1000, median=med)+' ms', alpha=0.5)
plt.hist(allparams1[okfinal2, 1],range=rng,bins=30,label='Asic2 ({})'.format(okfinal2.sum())+ft.statstr(allparams2[okfinal2,1]*1000, median=med)+' ms', alpha=0.5)
plt.xlabel('Tau [sec]')
plt.legend(fontsize=7, frameon=False)
plt.title('Fib {} - Tau [s]'.format(fib))

plt.subplot(2,2,2)
plt.hist(allparams[okfinal, 3],range=[0,1],bins=15,label='All ({}) '.format(okfinal.sum())+ft.statstr(allparams[okfinal,3], median=med)+' nA', alpha=0.5)
plt.hist(allparams1[okfinal1, 3],range=[0,1],bins=15,label='Asic1 ({}) '.format(okfinal1.sum())+ft.statstr(allparams1[okfinal1,3], median=med)+' nA', alpha=0.5)
plt.hist(allparams1[okfinal2, 3],range=[0,1],bins=15,label='Asic2 ({}) '.format(okfinal2.sum())+ft.statstr(allparams2[okfinal2,3], median=med)+' nA', alpha=0.5)
plt.xlabel('Amp [nA]')
plt.legend(fontsize=7, frameon=False)
plt.title('Fib {} - Amp [nA]'.format(fib))

plt.subplot(2,2,3)
imtau = ft.image_asics(data1=allparams1[:,1], data2=allparams2[:,1])	
plt.imshow(imtau,vmin=0,vmax=0.5)
plt.title('Tau [s] - Fiber {}'.format(fib,asic))
plt.colorbar()

subplot(2,2,4)
imamp = ft.image_asics(data1=allparams1[:,3], data2=allparams2[:,3])	
imshow(imamp,vmin=0,vmax=1)
title('Amp [nA] - Fiber {}'.format(fib,asic))
colorbar()
plt.tight_layout()
savefig('fib{}_summary.png'.format(fib))


##### Compare TES with Thermometers
thermos = np.zeros(128, dtype=bool)
thnum = [3, 35, 67, 99]
thermos[thnum] = True

figure()
clf()
subplot(2,1,1)
plot(tt, np.mean(folded1[okfinal1 * ~thermos,:], axis=0), 'b', lw=2, label='Valid TES average')
plot(tt, np.mean(folded1[thermos,:],axis=0), 'r', lw=2, label='Thermometers')
title('Fib = {} - ASIC 1'.format(fib))
legend(loc='upper left', fontsize=8)
subplot(2,1,2)
plot(tt, np.mean(folded2[okfinal2 * ~thermos,:], axis=0), 'b', lw=2, label='Valid TES average')
plot(tt, np.mean(folded2[thermos,:],axis=0), 'r', lw=2, label='Thermometers')
title('Fib = {} - ASIC 2'.format(fib))
savefig('fib{}_thermoVsTES.png'.format(fib))

