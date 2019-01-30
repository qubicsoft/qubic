import numpy as np
import scipy.ndimage.filters as f
from qubicpack import qubicpack as qp
from scipy.signal import butter, lfilter, iirnotch
import scipy.signal as scsig
import scipy.stats
from qubic.utils import progress_bar
from scipy.ndimage.filters import correlate1d
import time
import iminuit
import math
#from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from pysimulators import FitsArray


def statstr(x,divide=False, median=False):
	if median:
		m = np.median(x[np.isfinite(x)])
	else:
		m = np.mean(x[np.isfinite(x)])
	s = np.std(x[np.isfinite(x)])
	if divide==True:
		s /= len(x[np.isfinite(x)])
	return '{0:6.2f} +/- {1:6.2f}'.format(m,s)



def image_asics(data1=None, data2=None, all1=False):
	if all1:
		nn = len(data1)/2
		data2 = data1[nn:]
		data1 = data1[0:nn]
	if data1 is not None:
		a1 = qp()
		a1.assign_asic(1)
		a1.pix_grid[1,16] = 1005
		a1.pix_grid[2,16] = 1006
		a1.pix_grid[3,16] = 1007
		a1.pix_grid[4,16] = 1008
	if data2 is not None:
		a2 = qp()
		a2.assign_asic(2)
		a2.pix_grid[0,15] = 1004
		a2.pix_grid[0,14] = 1003
		a2.pix_grid[0,13] = 1002
		a2.pix_grid[0,12] = 1001

	nrows = 17
	ncols = 17
	img = np.zeros((nrows,ncols))+np.nan
	for row in range(nrows):
		for col in range(ncols):
			if data1 is not None:
				physpix=a1.pix_grid[row,col]
				if physpix in a1.TES2PIX[0]:
					TES=a1.pix2tes(physpix)
					img[row,col] = data1[TES-1]
			if data2 is not None:
				physpix=a2.pix_grid[row,col]
				if physpix in a2.TES2PIX[1]:
					TES=a2.pix2tes(physpix)
					img[row,col] = data2[TES-1]
	return img


###############################################################################
################################### Fitting ###################################
###############################################################################
### Generic polynomial function ##########
def thepolynomial(x,pars):
    f=np.poly1d(pars)
    return(f(x))
  
### Class defining the minimizer and the data
class MyChi2:
    def __init__(self,xin,yin,covarin,functname):
        self.x=xin
        self.y=yin
        self.covar=covarin
        self.invcov=np.linalg.inv(covarin)
        self.functname=functname
            
    def __call__(self,*pars):
        val=self.functname(self.x,pars)
        chi2=np.dot(np.dot(self.y-val,self.invcov),self.y-val)
        return(chi2)
        
### Call Minuit
def do_minuit(x,y,covarin,guess,functname=thepolynomial, fixpars = None, chi2=None, rangepars=None, nohesse=False, force_chi2_ndf=False, verbose=True):
    # check if covariance or error bars were given
    covar=covarin
    if np.size(np.shape(covarin)) == 1:
        err=covarin
        covar=np.zeros((np.size(err),np.size(err)))
        covar[np.arange(np.size(err)),np.arange(np.size(err))]=err**2
                                    
    # instantiate minimizer
    if chi2 is None:
        chi2=MyChi2(x,y,covar,functname)
        #nohesse=False
    else:
        nohesse=True
    # variables
    ndim=np.size(guess)
    parnames=[]
    for i in range(ndim): parnames.append('c'+np.str(i))
    # initial guess
    theguess=dict(zip(parnames,guess))
    # fixed parameters
    dfix = {}
    if fixpars is not None:
        for i in xrange(len(parnames)): dfix['fix_'+parnames[i]]=fixpars[i]
    else:
        for i in xrange(len(parnames)): dfix['fix_'+parnames[i]]=False
    # range for parameters
    drng = {}
    if rangepars is not None:
        for i in xrange(len(parnames)): drng['limit_'+parnames[i]]=rangepars[i]
    else:
        for i in xrange(len(parnames)): drng['limit_'+parnames[i]]=False
    #stop
    # Run Minuit
    if verbose: print('Fitting with Minuit')
    theargs = dict(theguess.items() + dfix.items())
    if rangepars is not None: theargs.update(dict(theguess.items() + drng.items()))
    m = iminuit.Minuit(chi2,forced_parameters=parnames,errordef=1.,**theargs)
    m.migrad()
    if nohesse==False:
        m.hesse()
    # build np.array output
    parfit=[]
    for i in parnames: parfit.append(m.values[i])
    errfit=[]
    for i in parnames: errfit.append(m.errors[i])
    ndimfit = int(np.sqrt(len(m.errors)))
    covariance=np.zeros((ndimfit,ndimfit))
    if fixpars is not None:
        parnamesfit = []
        for i in xrange(len(parnames)):
            if fixpars[i] == False: parnamesfit.append(parnames[i])
            if fixpars[i] == True: errfit[i]=0
    else:
        parnamesfit = parnames
    if m.covariance:
        for i in xrange(ndimfit):
            for j in xrange(ndimfit):
                covariance[i,j]=m.covariance[(parnamesfit[i],parnamesfit[j])]

    chisq = chi2(*parfit)
    ndf = np.size(x)-ndim
    if force_chi2_ndf:
        correct = chisq/ndf
        if verbose: print('correcting errorbars to have chi2/ndf=1 - correction = {}'.format(chisq))
    else:
        correct = 1.
    if verbose:
        print(np.array(parfit))
        print(np.array(errfit)*np.sqrt(correct))
        print('Chi2=',chisq)
        print('ndf=',ndf)
    return(m,np.array(parfit), np.array(errfit)*np.sqrt(correct), np.array(covariance)*correct,chi2(*parfit), ndf)

###############################################################################
###############################################################################


def profile(xin,yin,range=None,nbins=10,fmt=None,plot=True, dispersion=True, log=False):
  ok = np.isfinite(xin) * np.isfinite(yin)
  x = xin[ok]
  y = yin[ok]
  if range == None:
    mini = np.min(x)
    maxi = np.max(x)
  else:
    mini = range[0]
    maxi = range[1]
  if log==False:
    xx = np.linspace(mini,maxi,nbins+1)
  else:
    xx = np.logspace(np.log10(mini), np.log10(maxi), nbins+1)
  xmin = xx[0:nbins]
  xmax = xx[1:]
  yval = np.zeros(nbins)
  xc = np.zeros(nbins)
  dy = np.zeros(nbins)
  dx = np.zeros(nbins)
  nn = np.zeros(nbins)
  for i in np.arange(nbins):
    ok = (x > xmin[i]) & (x < xmax[i])
    nn[i] =  np.sum(ok)
    yval[i] = np.mean(y[ok])
    xc[i] = np.mean(x[ok])
    if dispersion: 
      fact = 1
    else:
      fact = np.sqrt(len(y[ok]))
    dy[i] = np.std(y[ok])/fact
    dx[i] = np.std(x[ok])/fact
  if plot:
      if fmt is None: fmt='ro' 
      plt.errorbar(xc, yval, xerr=dx, yerr=dy, fmt=fmt)
  ok = nn != 0
  return xc[ok], yval[ok], dx[ok], dy[ok]


def exponential_filter1d(input, sigma, axis=-1, output=None,
                      mode="reflect", cval=0.0, truncate=10.0):
    """One-dimensional Exponential filter.
    Parameters
    ----------
    %(input)s
    sigma : scalar
        Tau of exponential kernel
    %(axis)s
    %(output)s
    %(mode)s
    %(cval)s
    truncate : float
        Truncate the filter at this many standard deviations.
        Default is 4.0.
    Returns
    -------
    gaussian_filter1d : ndarray
    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    # calculate the kernel:
    for ii in range(1, lw + 1):
        tmp = math.exp(-float(ii) / sd)
        weights[lw + ii] = tmp*0
        weights[lw - ii] = tmp
        sum += tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return correlate1d(input, weights, axis, output, mode, cval, 0)


def qs2array(file, FREQ_SAMPLING):
	a = qp()
	a.read_fits(file)
	npix = a.NPIXELS
	nsamples = len(a.timeline(TES=1))
	dd = np.zeros((npix, nsamples))
	for i in xrange(npix):
		dd[i,:] = a.timeline(TES=i+1)
		##### Normalisation en courant
		Rfb=100e3
		NbSamplesPerSum = 64.
		gain=1./2.**7*20./2.**16/(NbSamplesPerSum*Rfb)
		dd[i,:] = gain * dd[i,:]
	time = np.arange(nsamples)/FREQ_SAMPLING
	return time, dd, a


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scsig.lfilter(b, a, data)
    return y

def notch_filter(data, f0, bw, fs):
	Q = f0/bw
	b, a = scsig.iirnotch(f0/fs*2, Q)
	y = scsig.lfilter(b, a, data)
	return y

def meancut(data, nsig):
	dd, mini, maxi = scipy.stats.sigmaclip(data, low=nsig, high=nsig)
	return np.mean(dd)



def simsig(x,pars):
	dx = x[1]-x[0]
	cycle = np.nan_to_num(pars[0])
	ctime = np.nan_to_num(pars[1])
	t0 = np.nan_to_num(pars[2])
	amp = np.nan_to_num(pars[3])
	sim_init = np.zeros(len(x))
	ok = x < (cycle*(np.max(x)))
	sim_init[ok] = 1.
	sim_init_shift = np.interp((x-t0) % max(x), x, sim_init)
	#thesim = -1*f.gaussian_filter1d(sim_init_shift, ctime, mode='wrap')
	thesim = -1*exponential_filter1d(sim_init_shift, ctime/dx, mode='wrap')
	thesim = (thesim-np.mean(thesim))/np.std(thesim) * amp
	return thesim

def simsig_nonorm(x,pars):
	dx = x[1]-x[0]
	cycle = np.nan_to_num(pars[0])
	ctime = np.nan_to_num(pars[1])
	t0 = np.nan_to_num(pars[2])
	amp = np.nan_to_num(pars[3])
	sim_init = np.zeros(len(x))
	ok = x < (cycle*(np.max(x)))
	sim_init[ok] = amp
	sim_init_shift = np.interp((x-t0) % max(x), x, sim_init)
	thesim = -1*exponential_filter1d(sim_init_shift, ctime/dx, mode='wrap')
	thesim = (thesim-np.mean(thesim))
	return thesim

def fold_data(time, dd, period, lowcut, highcut, nbins, notch=None):
	tfold = time % period
	FREQ_SAMPLING = 1./(time[1]-time[0])
	sh = np.shape(dd)
	ndet = sh[0]
	folded = np.zeros((ndet,nbins))
	folded_nonorm = np.zeros((ndet,nbins))
	bar = progress_bar(ndet, 'Detectors ')
	for THEPIX in xrange(ndet):
		bar.update()
		data = dd[THEPIX,:]
		filt = scsig.butter(3, [lowcut/FREQ_SAMPLING, highcut/FREQ_SAMPLING], btype='bandpass', output='sos')
		newdata = scsig.sosfilt(filt, data)
		if notch is not None:
			for i in xrange(len(notch)):
				ftocut = notch[i][0]
				bw = notch[i][1]
				newdata = notch_filter(newdata, ftocut, bw, FREQ_SAMPLING)
		t, yy, dx, dy = profile(tfold,newdata,range=[0, period],nbins=nbins,dispersion=False, plot=False)
		folded[THEPIX,:] = (yy-np.mean(yy))/np.std(yy)
		folded_nonorm[THEPIX,:] = (yy-np.mean(yy))
	return folded, t, folded_nonorm


def fit_average(t, folded, fff, dc, fib, Vtes, 
						initpars=[0.3, 0.06, 0.1, 0.6], 
						fixpars = [0,0,0,0],
						doplot=True, functname=simsig, clear=True):
	sh = np.shape(folded)
	npix = sh[0]
	nbins = sh[1]
	####### Average folded data
	av = np.median(np.nan_to_num(folded),axis=0)

	####### Fit
	bla = do_minuit(t, av, np.ones(len(t)), initpars, functname=functname, 
		rangepars=[[0.,1.], [0., 1], [0.,1], [0., 1]], fixpars=fixpars, 
		force_chi2_ndf=True, verbose=False, nohesse=True)
	params_av = bla[1]
	err_av = bla[2]

	if doplot:
		plt.ion()
		if clear: plt.clf()
		plt.xlim(0,1./fff)
		for i in xrange(npix):
			plt.plot(t, folded[i,:], alpha=0.1, color='k')
		plt.plot(t,av,color='b',lw=4,alpha=0.3, label='Median')
		plt.plot(t, functname(t, bla[1]), 'r--',lw=4, 
			label='Fitted average of {8:} pixels \n cycle={0:8.3f}+/-{1:8.3f} \n tau = {2:8.3f}+/-{3:8.3f}s \n t0 = {4:8.3f}+/-{5:8.3f}s \n amp = {6:8.3f}+/-{7:8.3f}'.format(params_av[0], 
				err_av[0], params_av[1], err_av[1], params_av[2], err_av[2], params_av[3], err_av[3],npix))
		plt.legend(fontsize=7,frameon=False, loc='lower right')
		plt.xlabel('Time(sec)')
		plt.ylabel('Stacked')
		plt.title('Fiber {}: Freq_Fiber={}Hz - Cycle={}% - Vtes={}V'.format(fib,fff,dc*100,Vtes))
		plt.show()
		time.sleep(0.1)
	return(av, params_av, err_av)


def fit_all(t, folded, av, fff, dc, fib, Vtes, 
						initpars=None, 
						fixpars = [0,0,0,0],
						doplot=True, stop_each=False, functname=simsig):

	sh = np.shape(folded)
	npix = sh[0]
	nbins = sh[1]
	print('       Got {} pixels to fit'.format(npix))

	##### Now fit each TES fixing cycle to dc and t0 to the one fitted on the median
	allparams = np.zeros((npix,4))
	allerr = np.zeros((npix,4))
	allchi2 = np.zeros(npix)
	bar = progress_bar(npix, 'Detectors ')
	ok = np.zeros(npix,dtype=bool)
	for i in xrange(npix):
		bar.update()
		thedd = folded[i,:]
		#### First a fit with no error correction in order to have a chi2 distribution
		theres = do_minuit(t, thedd, np.ones(len(t)), initpars, functname=functname,
			fixpars = fixpars, 
			rangepars=[[0.,1.], [0., 10], [0.,1], [0., 1]], 
			force_chi2_ndf=True, verbose=False, nohesse=True)
		chi2 = theres[4]
		ndf = theres[5]
		params = theres[1]
		err = theres[2]
		allparams[i,:] = params
		allerr[i,:] = err
		allchi2[i] = theres[4]
		
		#initialise plot figure
		
		plt.figure()
		if stop_each:
			plt.clf() # was clf
			plt.ion()
			plt.plot(t, thedd, color='k')
			plt.plot(t,av,color='b',lw=4, alpha=0.2, label='Median')
			plt.plot(t,	functname(t, theres[1]), 'r--', lw=4, 
			label='Fitted: \n cycle={0:8.3f}+/-{1:8.3f} \n tau = {2:8.3f}+/-{3:8.3f}s \n t0 = {4:8.3f}+/-{5:8.3f}s \n amp = {6:8.3f}+/-{7:8.3f}'.format(params[0], 
				err[0], params[1], err[1], params[2], err[2], params[3], err[3]))
			plt.legend()
			plt.pause(3)
			plt.draw()
			plt.show()
			msg = 'TES #{}'.format(i)
			if i in [3, 35, 67, 99]:
				msg = 'Channel #{} - BEWARE THIS IS A THERMOMETER !'.format(i)
			plt.title(msg)
			#Changing so 'i' select prompts plot inversion
			bla=raw_input("Press [y] if fit OK, [i] to invert, other key otherwise...")
			if (bla=='y'):
				ok[i]=True
				
			#invert to check if TES okay, 
			#thedd refers to the indexed TES in loop
			elif (bla == 'i'):
				plt.plot(t, thedd*-1.0, color='olive')
				plt.pause(3)
				plt.draw()
				plt.show()
				ibla = raw_input("Press [y] if INVERTED fit OK, otherwise anykey")
				#and invert thedd in the original datset
				if (ibla == 'y'):
					ok[i]=True
					folded[i,:] = thedd * -1.0
					
			print(ok[i])
			
	return allparams, allerr, allchi2, ndf,ok


def run_asic(fib, Vtes, fff, dc, theasicfile, asic, reselect_ok=False, lowcut=0.5, highcut=15., nbins=50, nointeractive=False, doplot=True, notch=None, lastpassallfree=False):
	### Read data
	FREQ_SAMPLING = (2e6/128/100)    
	time, dd, a = qs2array(theasicfile, FREQ_SAMPLING)
	ndet, nsamples = np.shape(dd)

	### Fold the data at the modulation period of the fibers
	### Signal is also badpass filtered before folding
	folded, tt, folded_nonorm = fold_data(time, dd, 1./fff, lowcut, highcut, nbins, notch=notch)

	if nointeractive==True:
		releselct_ok=False
		answer = 'n'
	else:
		if reselect_ok==True:
			print('\n\n')
			answer=raw_input('This will overwrite the file for OK TES. Are you sure you want to proceed [y/n]')
		else:
			answer='n'

	if answer=='y':
		print('Now going to reselct the OK TES and overwrite the corresponding file')
		#### Pass 1 - allows to obtain good values for t0 basically
		#### Now perform the fit on the median folded data
		print('')
		print('FIRST PASS')
		print('First Pass is only to have a good guess of the t0, your selection should be very conservative - only high S/N')
		av, params, err = fit_average(tt, folded, fff, dc, 
				fib, Vtes, 
				initpars = [dc, 0.06, 0., 0.6],
				fixpars = [0,0,0,0],
				doplot=True)

		#### And the fit on all data with this as a first guess forcing some parameters - it returns the list of OK detectorsy
		allparams, allerr, allchi2, ndf, ok = fit_all(tt, folded, av, fff, dc, fib, Vtes, 
				initpars = [dc, params[1], params[2], params[3]],
				fixpars = [1,0,1,0],stop_each=True)

		#### Pass 2
		#### Refit with only the above selected ones in order to have good t0
		#### Refit the median of the OK detectors
		print('')
		print('SECOND PASS')
		print('Second pass is the final one, please select the pixels that seem OK')
		av, params, err = fit_average(tt, folded[ok,:], fff, dc, 
				fib, Vtes, 
				initpars = [dc, 0.1, 0., 1.],
				fixpars = [0,0,0,0],
				doplot=True)

		#### And the fit on all data with this as a first guess forcing some parameters - it returns the list of OK detectors
		allparams, allerr, allchi2, ndf, ok = fit_all(tt, folded, av, fff, dc, fib, Vtes, 
				initpars = [dc, params[1], params[2], params[3]],
				fixpars = [1,0,1,0],stop_each=True)


		#### Final Pass
		#### The refit them all with only tau and amp as free parameters 
		#### also do not normalize amplitudes of folded
		allparams, allerr, allchi2, ndf, ok_useless = fit_all(tt, folded_nonorm*1e9, 
				av, fff, dc, fib, Vtes, 
				initpars = [dc, params[1], params[2], params[3]],
				fixpars = [1,0,1,0], functname=simsig_nonorm)

		okfinal = ok * (allparams[:,1] < 1.)
		### Make sure no thermometer is included
		okfinal[[3, 35, 67, 99]] = False
		# Save the list of OK bolometers
		FitsArray(okfinal.astype(int)).save('TES-OK-fib{}-asic{}.fits'.format(fib,asic))
	else:
		okfinal=np.array(FitsArray('TES-OK-fib{}-asic{}.fits'.format(fib,asic))).astype(bool)

	if doplot==False:
		### Now redo the fits one last time
		av, params, err = fit_average(tt, folded[okfinal,:], fff, dc, 
				fib, Vtes, 
				initpars = [dc, 0.1, 0., 1.],
				fixpars = [0,0,0,0],
				doplot=False,clear=False)

		allparams, allerr, allchi2, ndf, ok_useless = fit_all(tt, folded_nonorm*1e9, 
				av, fff, dc, fib, Vtes, 
				initpars = [dc, params[1], params[2], params[3]],
				fixpars = [1,0,1,0], functname=simsig_nonorm)
	else:
		plt.figure(figsize=(6,8))
		plt.subplot(3,1,1)
		### Now redo the fits one last time
		av, params, err = fit_average(tt, folded[okfinal,:], fff, dc, 
				fib, Vtes, 
				initpars = [dc, 0.1, 0., 1.],
				fixpars = [0,0,0,0],
				doplot=True,clear=False)

		if lastpassallfree:
			fixed = [0, 0, 0, 0]
		else: 
			fixed = [1, 0, 1, 0]
		allparams, allerr, allchi2, ndf, ok_useless = fit_all(tt, folded_nonorm*1e9, 
				av, fff, dc, fib, Vtes, 
				initpars = [dc, params[1], params[2], params[3]],
				fixpars = fixed, functname=simsig_nonorm)

		plt.subplot(3,2,3)
		plt.hist(allparams[okfinal,1],range=[0,1],bins=30,label=statstr(allparams[okfinal,1]))
		plt.xlabel('Tau [sec]')
		plt.legend()
		plt.title('Asic {} - Fib {}'.format(asic, fib))
		plt.subplot(3,2,4)
		plt.hist(allparams[okfinal,3],range=[0,1],bins=30, label=statstr(allparams[okfinal,3]))
		plt.legend()
		plt.xlabel('Amp [nA]')

		pars = allparams
		tau = pars[:,1]
		tau[~okfinal]=np.nan
		amp = pars[:,3]
		amp[~okfinal]=np.nan

		if asic==1:
			tau1 = tau
			tau2 = None
			amp1 = amp
			amp2 = None
		else:
			tau1 = None
			tau2 = tau
			amp1 = None
			amp2 = amp

		plt.subplot(3,2,5)
		imtau = image_asics(data1=tau1, data2=tau2)	
		plt.imshow(imtau,vmin=0,vmax=0.5, interpolation='nearest')
		plt.title('Tau - Fiber {} - asic {}'.format(fib,asic))
		plt.colorbar()
		plt.subplot(3,2,6)
		imamp = image_asics(data1=amp1, data2=amp2)	
		plt.imshow(imamp,vmin=0,vmax=1, interpolation='nearest')
		plt.colorbar()
		plt.title('Amp - Fiber {} - asic {}'.format(fib, asic))
		plt.tight_layout()
		

	return tt, folded, okfinal, allparams, allerr, allchi2, ndf


def calibrate(fib, pow_maynooth, allparams, allerr, allok, cutparam=None, cuterr=None, bootstrap=None):
	img_maynooth = image_asics(pow_maynooth, all1=True)

	plt.clf()
	plt.subplot(2,2,1)
	plt.plot(allparams[allok,3], allerr[allok,3],'k.')
	if cuterr is not None:
		thecut_err = cuterr
	else: 
		thecut_err=1e10
	if cutparam is not None:
		thecut_amp = cutparam
	else:
		thecut_amp = 1e10
	
	newok = allok * (allerr[:,3] < thecut_err) * (allparams[:,3] < thecut_amp)
	plt.plot([np.min(allparams[allok,3]), np.max(allparams[allok,3])], [thecut_err, thecut_err], 'g--')
	plt.plot([thecut_amp, thecut_amp], [np.min(allerr[allok,3]), np.max(allerr[allok,3])], 'g--')
	plt.plot(allparams[newok,3], allerr[newok,3],'r.')
	allparams[~newok,:]=np.nan
	plt.ylabel('$\sigma_{amp}$ [nA]')
	plt.xlabel('Amp Fib{} [nA]'.format(fib))


	plt.subplot(2,2,3)
	plt.errorbar(pow_maynooth[newok], allparams[newok,3], yerr=allerr[newok,3],fmt='r.')
	xx = pow_maynooth[newok]
	yy = allparams[newok,3]
	yyerr = allerr[newok,3]
	res = do_minuit(xx, yy, yyerr, np.array([1.,0]), fixpars=[0,0])
	paramfit = res[1]
	if bootstrap is None:
		errfit = res[2]
		typerr = 'Minuit'
	else:
		bsres = []
		bar = progress_bar(bootstrap, 'Bootstrap')
		for i in xrange(bootstrap):
			bar.update()
			order = np.argsort(np.random.rand(len(xx)))
			xxbs = xx.copy()
			yybs = yy[order]
			yybserr = yyerr[order]
			theres = do_minuit(xxbs, yybs, yybserr, np.array([1.,0]), fixpars=[0,0], verbose=False)
			bsres.append(theres[1])
		bsres = np.array(bsres)
		errfit = np.std(bsres, axis=0)
		typerr = 'Bootstrap'

	xxx = np.linspace(0, np.max(pow_maynooth), 100)
	plt.plot(xxx, thepolynomial(xxx, res[1]), 'g', lw=3, label='a={0:8.3f} +/- {1:8.3f} \n b={2:8.3f} +/- {3:8.3f}'.format(paramfit[0], errfit[0],paramfit[1], errfit[1]))
	if bootstrap is not None:
		bsdata = np.zeros((bootstrap,len(xxx)))
		for i in xrange(bootstrap):
			bsdata[i,:] = thepolynomial(xxx, bsres[i,:])
		mm = np.mean(bsdata, axis=0)
		ss = np.std(bsdata, axis=0)
		plt.fill_between(xxx, mm-ss, y2=mm+ss, color='b', alpha=0.3)
		plt.fill_between(xxx, mm-2*ss, y2=mm+2*ss, color='b', alpha=0.2)
		plt.fill_between(xxx, mm-3*ss, y2=mm+3*ss, color='b', alpha=0.1)
		plt.plot(xxx,mm,'b', label='Mean bootstrap')
		# indices = np.argsort(np.random.rand(bootstrap))[0:1000]
		# for i in xrange(len(indices)):
		# 	plot(xxx, thepolynomial(xxx, bsres[indices[i],:]), 'k', alpha=0.01)
	plt.ylim(0,np.max(allparams[newok,3])*1.1)
	plt.xlim(np.min(pow_maynooth[newok])*0.99,np.max(pow_maynooth[newok])*1.01)
	plt.ylabel('Amp Fib{} [nA]'.format(fib))
	plt.xlabel('Maynooth [mW]')
	plt.legend(fontsize=8, framealpha=0.5)


	plt.subplot(2,2,2)
	plt.imshow(img_maynooth,vmin=np.min(pow_maynooth), vmax=np.max(pow_maynooth), interpolation='nearest')
	plt.colorbar()
	plt.title('Maynooth [mW]')

	plt.subplot(2,2,4)
	plt.img = image_asics(allparams[:,3]/res[1][0], all1=True)
	plt.imshow(img, interpolation='nearest')
	plt.colorbar()
	plt.title('Amp Fib{}  converted to mW'.format(fib))
	plt.tight_layout()

	return res[1], res[2], newok





