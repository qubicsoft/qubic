import numpy as np
from Calibration import fibtools as ft
import matplotlib.pyplot as plt
from pysimulators import FitsArray
import matplotlib.mlab as mlab
import scipy.ndimage.filters as f
from Calibration.plotters import *



################################################ INPUT FILES ######################################

#basedir = '/home/louisemousset/QUBIC/Qubic_work/Calibration/'
#basedir = '/home/james/fibdata/'
basedir = '/Users/hamilton/CMB/Qubic/Fibres/'

##### Fiber_2 Fiber@1V; Freq=1Hz; DutyCycle=30%; Voffset_TES=3V
fib = 2
Vtes = 3.
fff = 1.
dc = 0.3
asic1 = basedir + '/2018-12-20/2018-12-20_17.27.22__Fiber_2/Sums/science-asic1-2018.12.20.172722.fits'
asic2 = basedir + '/2018-12-20/2018-12-20_17.27.22__Fiber_2/Sums/science-asic2-2018.12.20.172722.fits'


##### Fiber 3: Fiber@1V; Freq=1.5Hz; DutyCycle=50%; Voffset_TES=3V
# fib = 3
# Vtes = 3.
# fff = 1.5
# dc = 0.5
# asic1 = basedir +  '/2018-12-20/2018-12-20_17.52.08__Fiber_3/Sums/science-asic1-2018.12.20.175208.fits'
# asic2 = basedir + '/2018-12-20/2018-12-20_17.52.08__Fiber_3/Sums/science-asic2-2018.12.20.175208.fits'

# ##### Fiber 4: Fiber@1V; Freq=1Hz; DutyCycle=50%; Voffset_TES=2.6V
# fib = 4
# Vtes = 2.6
# fff = 1.
# dc = 0.5
# asic1 = basedir +  '/2018-12-20/2018-12-20_17.27.22__Fiber_2/Sums/science-asic1-2018.12.20.172722.fits'
# asic2 = basedir + '/2018-12-20/2018-12-20_17.27.22__Fiber_2/Sums/science-asic2-2018.12.20.172722.fits'
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
plt.clf()
plt.plot(time, dd[theTES,:])
plt.xlabel('Time [s]')
plt.ylabel('Current [nA]')


###### TOD Power Spectrum #####

#theTES=best_det
frange = [0.3, 15]
filt = 5
FreqResp(best_det, frange, fff, filt, dd, FREQ_SAMPLING, nsamples)

###NEW PLOT WITH FILTERED OVERPLOT
#### Filtering out the signal from the PT
freqs_pt = [1.72383, 3.24323, 3.44727, 5.69583, 6.7533, 9.64412, 12.9874]
bw_0 = 0.005
spectrum, freq = mlab.psd(dd[theTES,:], Fs=FREQ_SAMPLING, NFFT=nsamples, window=mlab.window_hanning)
filtered_spec = f.gaussian_filter1d(spectrum, filt)
notch = FiltFreqResp(theTES, frange, fff, filt, freqs_pt, bw_0, dd, 
			 FREQ_SAMPLING, nsamples, freq, spectrum, filtered_spec)

############################################################################



############################################################################
### Fold the data at the modulation period of the fibers
### Signal is also badpass filtered before folding

#set up band pass filter
theTES=45
lowcut = 0.5
highcut = 15
nbins=50
folded, tt, folded_nonorm = ft.fold_data(time, dd, 1./fff, lowcut, highcut, nbins)

folded_notch, tt, folded_notch_nonorm = ft.fold_data(time, dd, 1./fff, lowcut, highcut, nbins,
	notch = notch)

pars = [dc, 0.05, 0., 1.2]
#plot folded TES data
FoldedFiltTES(tt, pars, theTES, folded, folded_notch)

### Plot it along with a guess for fiber signal
#### Now fit the fiber signal 
#### NB errors come from renormalizing chi2/ndf to 1
guess = [dc, 0.06, 0., 1.2]
bla = ft.do_minuit(tt, folded[theTES,:], np.ones(len(tt)),
	guess, functname=ft.simsig,
	rangepars=[[0.,1.], [0., 1], [0.,1], [0., 1]], fixpars=[0,0,0,0], 
	force_chi2_ndf=True, verbose=False, nohesse=True)
#plot the free fit
FoldedTESFreeFit(tt, bla, theTES, folded)


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
#run ASIC analysis code
tt, folded1, okfinal1, allparams1, allerr1, allchi21, ndf1 = ft.run_asic(fib, Vtes, fff, dc, 
	asic1, 1, reselect_ok=True, lowcut=0.5, highcut=15., nbins=50, 
	nointeractive=False, doplot=True, notch=notch)
#run ASIC plotter


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

Allplots(fib, allparams, allparams1, allparams2, okfinal, okfinal1, okfinal2, asic, med=False, rng=[0,0.4])

##### Compare TES with Thermometers
thermos = np.zeros(128, dtype=bool)
thnum = [3, 35, 67, 99]
thermos[thnum] = True

TESvsThermo(fib, tt, folded1, folded2, okfinal1, okfinal2, thermos)
