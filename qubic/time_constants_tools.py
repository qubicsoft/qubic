import numpy as np
from numpy import *
from matplotlib.pyplot import *
from matplotlib import pyplot as plt
import iminuit
from iminuit.cost import LeastSquares
import scipy.ndimage.filters as scfilt
import scipy.ndimage.filters as f
from scipy.interpolate import interp1d
import gc
from scipy.integrate import simpson as simps
import math
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from scipy.signal import savgol_filter

import qubic as qubic
from qubic import *
from qubic.qubicdict import qubicDict
from qubic.instrument import QubicInstrument



#qubic related
import qubic
from qubic.qubicdict import qubicDict
from qubic.instrument import QubicInstrument
from qubicpack.qubicfp import qubicfp
from qubic import fibtools as ft
from qubicpack.utilities import Qubic_DataDir
from qubic import selfcal_lib as scal
from qubic import time_domain_tools as tdt

def normalize(x):
	"""
	(x-np.nanmean(x))/np.nanstd(x)
	"""
	return (x-np.nanmean(x))/np.nanstd(x)

def plot_folded_data_on_FP(datain, time = None, datain_error = None, tes_signal_ok = np.ones(256,dtype=bool), analytical_function = None, eval_domain = None, params_function = None, save=True, figname = 'figname', format='png', doplot = False ,**kwargs):

    basedir = Qubic_DataDir()
    dictfilename = basedir + '/dicts/global_source_oneDet_multiband.dict'
#    d = qubic.qubicdict.qubicDict()
    d = qubicDict()
    d.read_from_file(dictfilename)
#    q = qubic.QubicInstrument(d)
    q = QubicInstrument(d)

    
    x=np.linspace(-0.0504, -0.0024, 17)
    y=np.linspace(-0.0024, -0.0504, 17)

    X, Y = np.meshgrid(x, y)
    
    tes_to_plot = np.ones(256,dtype=bool)
    tes_ok = tes_signal_ok
    discarded_tes = ~tes_signal_ok
    
    for i in [3,35,67,99,3+128,35+128,67+128,99+128]: ##extract thermometers
        tes_to_plot[i] = False
        discarded_tes[i] = False
    
    TES_asic1 = np.arange(1,129)
    TES_asic2 = np.arange(1,129)
    TES_asic1_notherm = TES_asic1[tes_to_plot[:128]]
    TES_asic2_notherm = TES_asic2[tes_to_plot[128:]]
    TES_asic1_ok = TES_asic1[tes_ok[:128]]
    TES_asic2_ok = TES_asic2[tes_ok[128:]]
    discarded_tes_asic1 = TES_asic1[discarded_tes[:128]]
    discarded_tes_asic2 = TES_asic2[discarded_tes[128:]]
    
    plt.ioff()
    
    fig, axs = subplots(nrows=17, ncols=17, figsize=(50, 50))

#    if time is None:
#    	time = np.arange(1,len(datain)+1)
#    else:
#    	if eval_domain is None:
#    	    eval_domain = time

    for j in [1,2]:
        if j==1:
            for tes in TES_asic1_notherm:

                xtes, ytes, FP_index, index_q= scal.TES_Instru2coord(TES=tes, ASIC=j, q=q, frame='ONAFP', verbose=False)
                ind=np.where((np.round(xtes, 4) == np.round(X, 4)) & (np.round(ytes, 4) == np.round(Y, 4)))

                if datain_error is not None:
                    axs[ind[0][0], ind[1][0]].errorbar(time, datain[tes-1], yerr=datain_error[tes-1], fmt='bo', label='{}'.format(tes), alpha=0.5, **kwargs)
                else:
                    axs[ind[0][0], ind[1][0]].scatter(time, datain[tes-1], fmt='bo', label='{}'.format(tes), alpha=0.5, **kwargs)
                leg = axs[ind[0][0], ind[1][0]].legend(handlelength=0, handletextpad=0, fancybox=True,fontsize=22,loc='upper center')
                for item in leg.legendHandles:
                    item.set_visible(False)
                axs[ind[0][0], ind[1][0]].get_xaxis().set_visible(False)
                axs[ind[0][0], ind[1][0]].get_yaxis().set_visible(False)

                if analytical_function is not None:
                    axs[ind[0][0], ind[1][0]].plot(eval_domain, analytical_function(eval_domain,params_function[tes-1]),color = 'black')
                
                if tes in discarded_tes_asic1:
                    axs[ind[0][0], ind[1][0]].set_facecolor('xkcd:salmon')

        elif j==2:
            for tes in TES_asic2_notherm:
                
                xtes, ytes, FP_index, index_q= scal.TES_Instru2coord(TES=tes, ASIC=j, q=q, frame='ONAFP', verbose=False)
                ind=np.where((np.round(xtes, 4) == np.round(X, 4)) & (np.round(ytes, 4) == np.round(Y, 4)))

                if datain_error is not None:
                    axs[ind[0][0], ind[1][0]].errorbar(time, datain[tes-1+128], yerr=datain_error[tes-1+128], fmt='ro', label='{}'.format(tes), alpha=0.5, **kwargs)
                else:
                    axs[ind[0][0], ind[1][0]].scatter(time, datain[tes-1+128], fmt='ro', label='{}'.format(tes), alpha=0.5, **kwargs)
                leg = axs[ind[0][0], ind[1][0]].legend(handlelength=0, handletextpad=0, fancybox=True,fontsize=22,loc='upper center')
                for item in leg.legendHandles:
                    item.set_visible(False)
                axs[ind[0][0], ind[1][0]].get_xaxis().set_visible(False)
                axs[ind[0][0], ind[1][0]].get_yaxis().set_visible(False)
                
                if analytical_function is not None:
                    axs[ind[0][0], ind[1][0]].plot(eval_domain, analytical_function(eval_domain,params_function[tes-1+128]),color = 'black')

                if tes in discarded_tes_asic2:
                    axs[ind[0][0], ind[1][0]].set_facecolor('xkcd:salmon')
#    if doplot:
#    	show()
#    	            
#    if save:
#    	show()
    savefig(figname+'.'+format, bbox_inches="tight",format=format)
    close(fig)
    plt.ion()


class asymsig_spl_class:
    def __init__(self, x, y, err, nbspl):
        #print('Number of spline: {}'.format(nbspl))
        self.spl = tdt.MySplineFitting(x, y, err, nbspl)
        
    def __call__(self, x, pars):
        pars_asymsig = pars[:5]
        asym_sig = simsig_asym_nooffset(x, pars_asymsig)
        pars_spl = pars[5:]
        splvals = self.spl.with_alpha(x, pars_spl)
        return asym_sig + splvals

def simsig_asym(x, pars): #needs 6 parameters
        dx = x[1] - x[0]
        cycle = np.nan_to_num(pars[0])
        ctime_rise = np.nan_to_num(pars[1])
        ctime_fall = np.nan_to_num(pars[2])
        t0 = np.nan_to_num(pars[3])
        amp = np.nan_to_num(pars[4])
        offset = np.nan_to_num(pars[5])
        sim_init = np.zeros(len(x))
        ok = x < (cycle * (np.max(x)))
        sim_init[ok] = -1 + np.exp(-x[ok] / ctime_rise)
        if ok.sum() > 0:
                endval = sim_init[ok][-1]
        else:
                endval = -1.
        sim_init[~ok] = -np.exp(-(x[~ok] - x[~ok][0]) / ctime_fall) + 1 + endval
        thesim = np.interp((x - t0) % max(x), x, sim_init)
        thesim = thesim * amp + offset
        return np.nan_to_num(thesim)

def simsig_asym_nooffset(x, pars): #needs 5 parameters
        dx = x[1] - x[0]
        cycle = np.nan_to_num(pars[0])
        ctime_rise = np.nan_to_num(pars[1])
        ctime_fall = np.nan_to_num(pars[2])
        t0 = np.nan_to_num(pars[3])
        amp = np.nan_to_num(pars[4])
        sim_init = np.zeros(len(x))
        ok = x < (cycle * (np.max(x)))
        sim_init[ok] = -1 + np.exp(-x[ok] / ctime_rise)
        if ok.sum() > 0:
                endval = sim_init[ok][-1]
        else:
                endval = -1.
        sim_init[~ok] = -np.exp(-(x[~ok] - x[~ok][0]) / ctime_fall) + 1 + endval
        thesim = np.interp((x - t0) % max(x), x, sim_init)
        thesim = thesim * amp
        return np.nan_to_num(thesim)

def asymsig_poly(x, pars):
    pars_asymsig = pars[:5]
    asym_sig = simsig_asym_nooffset(x, pars_asymsig)
    pars_poly = pars[5:][::-1]
    polvals = np.poly1d(pars_poly)(x)
    return asym_sig + polvals



def fit_one(t, tofit, errors, initguess, nparams_ext, fctfit = asymsig_poly, fixpars = None, limits=None, scan=None):
    ok = np.isfinite(tofit) & (errors != 0)
    myls = LeastSquares(t[ok], tofit[ok], errors[ok], fctfit)
    if nparams_ext == 0:
        guess = initguess
    else:
        guess = np.append(initguess, np.zeros(nparams_ext-1)+initguess[-1])
    try:
        m = iminuit.Minuit(myls, guess)
        ## Limits
        if limits is not None:
            mylimits = []
            for k in range(len(guess)):
                mylimits.append((None, None))
            for k in range(len(limits)):
                mylimits[limits[k][0]] = (limits[k][1], limits[k][2])
            m.limits = mylimits
        ## Fixed parameters
        if fixpars is not None:
            for k in range(len(guess)):
                m.fixed["x{}".format(k)]=False
            for k in range(len(fixpars)):
                m.fixed["x{}".format(fixpars[k])]=True
        if scan is not None:
            print('scanning',scan)
            m.scan(ncall=scan)
        m.migrad()  # finds minimum of least_squares function
        m.hesse()   # accurately computes uncertainties
        ch2 = m.fval
        ndf = len(t[ok]) - m.nfit
        return m, ch2, ndf
    except:
        print('Minuit Failed')
        return 0., 0., 0.

def run_DBSCAN(results, doplot=False, parnames = None, eps_cpar = 1.3, min_samples_cpar = 10):
    clustering = DBSCAN(eps=eps_cpar, min_samples=min_samples_cpar).fit(np.nan_to_num(results))
    labels = clustering.labels_
    nfound = len(np.unique(np.sort(labels)))
    unique_labels = unique(labels)  
    colors = [plt.cm.jet(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    
    nnn = np.shape(results)[1]
    if doplot:
        if nnn>1:
            figure(figsize=(11,8))
            for i in range(nnn):
                for j in range(i+1, nnn):
                    subplot(nnn-1, nnn-1, i*(nnn-1)+j)
                    if parnames is None:
                        xlabel('Param {}'.format(j))
                        ylabel('Param {}'.format(i))
                    else:
                        xlabel(parnames[j])
                        ylabel(parnames[i])
                    plot(results[:,j], results[:,i], 'k.')
                    for k in range(len(unique_labels)):
                        thisone = labels == unique_labels[k]
                        plot(results[thisone,j],results[thisone,i], '.',
                                label='Type {} : n={}'.format(unique_labels[k],thisone.sum()))
                    if (i+j-1)==0: legend()
        elif nnn==1:
            figure()
            if parnames is None:
                xlabel('TES Number')
                ylabel('Parameter')
            else:
                xlabel('TES Number')
                ylabel(parnames[0])
            plot(np.arange(1,257),results, 'k.')
            for k in range(len(unique_labels)):
                thisone = labels == unique_labels[k]
                plot(np.arange(1,257)[thisone],results[thisone], '.',
                        label='Type {} : n={}'.format(unique_labels[k],thisone.sum()))
        legend()
        tight_layout()
    return (labels)

def run_OPTICS(results, doplot=False, parnames = None, min_samples_optics = 10):
    clustering = OPTICS(min_samples=min_samples_optics).fit(np.nan_to_num(results))
    labels = clustering.labels_
    nfound = len(np.unique(np.sort(labels)))
    unique_labels = unique(labels)  
    colors = [plt.cm.jet(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    
    nnn = np.shape(results)[1]
    if doplot:
        if nnn>1:
            figure(figsize=(11,8))
            for i in range(nnn):
                for j in range(i+1, nnn):
                    subplot(nnn-1, nnn-1, i*(nnn-1)+j)
                    if parnames is None:
                        xlabel('Param {}'.format(j))
                        ylabel('Param {}'.format(i))
                    else:
                        xlabel(parnames[j])
                        ylabel(parnames[i])
                    plot(results[:,j], results[:,i], 'k.')
                    for k in range(len(unique_labels)):
                        thisone = labels == unique_labels[k]
                        plot(results[thisone,j],results[thisone,i], '.',
                                label='Type {} : n={}'.format(unique_labels[k],thisone.sum()))
                    if (i+j-1)==0: legend()
        elif nnn==1:
            figure()
            if parnames is None:
                xlabel('TES Number')
                ylabel('Parameter')
            else:
                xlabel('TES Number')
                ylabel(parnames[0])
            plot(np.arange(1,257),results, 'k.')
            for k in range(len(unique_labels)):
                thisone = labels == unique_labels[k]
                plot(np.arange(1,257)[thisone],results[thisone], '.',
                        label='Type {} : n={}'.format(unique_labels[k],thisone.sum()))
        legend()
        tight_layout()
    return (labels)


def compute_tc_squaremod(thedatadir, nbins = 100, lowcut = None, highcut = None, notch = None, fmod = None, dutycycle = None, typefit = 'just_exp', nparams_ext_spl=4, nparams_ext_poly=1, doplot = False, doplot_onebyone = True, verbose = False,save_path =None):

	"""
	Compute the time constants from a square modulation
	
	thedatadir : 		directory where to read the dataset
	nbins (= 100) : 	nbins to fold the data
	lowcut (= None) : 	lowcut frequency. See fibtools.fold_data() for more info.
	highcut (= None) : 	highcut frequency. See fibtools.fold_data() for more info.
	notch (= None) : 	several frequencies to notch filter. See fibtools.fold_data() for more info.
	fmod (= None) : 	modulation frequency, this is only required if the calsource information is not available in the housekeeping data.
	dutycycle (= None) : 	dutycycle in %, this is only required if the calsource information is not available in the housekeeping data.
	typefit (= 'just_exp') : 	on top of the exponential behaviour we can chose to also fit a slow varying function. 'just_exp' means an exponential model, 'spl' means adding a slow varying function with splines on top of the exponential behaviour (nparams_ext_spl must be >=4 and defines the number of polynomial parameters) and 'poly' means adding a slow varying function with polynomials on top of the exponential behaviour (nparams_ext_poly must be >=1 and nparams_ext_poly-1 defines the degree of the polynomial).
	nparams_ext_spl (=4) :	must be >=4 and defines the number of spline parameters.
	nparams_ext_poly (=1) :	must be >=1 and nparams_ext_poly-1 defines the degree of the polynomial.
	doplot (= False) :	to show several plots.
	doplot_onebyone (= True) :	to show one plot per TES (fit and folded data, three plots per figure). If doplot= False, then doplot_onebyone will be also False.
	verbose (= True) :	to show some intermediate output messages.
	save_path (= None) :	if not None, one dictionary (with all the information, see Output) and one focal plane plot (with the fits and folded data) per dataset.
	
	Output:
	d :	dictionary with all the relevant information. Elaborate...
	

	"""
	
	dataset_info = str.split(thedatadir,'/')[-1]
	
	d_results = {}
	d_results['dataset_info'] = dataset_info
	
	a = qubicfp()
	
	if not verbose:
		a.assign_verbosity(0)
	
	a.read_qubicstudio_dataset(thedatadir)

	minVbias_asic1 = a.asic(1).min_bias
	minVbias_asic2 = a.asic(2).min_bias
	maxVbias_asic1 = a.asic(1).max_bias
	maxVbias_asic2 = a.asic(2).max_bias

	tt, alltod = a.tod()
	calsource_dict = a.calsource_info()
	
	shape = None
	
	try:
		RF = calsource_dict['calsource']['frequency'] # in GHz
		fmod = calsource_dict['modulator']['frequency'] # in Hz
		dutycycle = calsource_dict['modulator']['duty_cycle'] # in %
		shape = calsource_dict['modulator']['shape']
#		amplifier_invert = calsource_dict['amplifier']['invert']
		
	except:
		calsource_analysis = False
		print('No calsource information.')
		if fmod is None or dutycycle is None:
			raise Exception('Insert modulation frequency and dutycyle in the arguments.')	

	if shape is not None:
		if shape == 'square':
			calsource_analysis = True
		else:
			raise Exception('ERROR: The shape of the modulation is not square. This analysis is intended to be performed for square modulation. Returning an empty dictionary.')
#			d_results['Exception'] = 'The shape of the modulation is not square.'
#			
#			return d_results
			
	else:
		calsource_analysis = False
		print('No calsource analysis performed since no calsource data is available')
	
	if calsource_analysis:
	
		caltime, calsourcedata = a.calsource()
		caldata = []
		caldata.append(calsourcedata)
		caldata = np.asarray(caldata)
		caldata = caldata[0,:]
	#	if amplifier_invert == 'ON':
	#		caldata = caldata[0,:]
	#	else:
	#		caldata = -caldata[0,:]
	
	del(a)
	gc.collect()
	
	period = 1./ fmod
	
	
	if calsource_analysis:
		try:
			print('Folding the calsource data.')
	 		########## Folding
			folded_cal, t_fold_cal, folded_nonorm_cal, dfolded_cal, dfolded_nonorm_cal, newdata_cal, fn_cal, nn_cal= ft.fold_data(caltime,
	 												np.reshape(caldata, (1,len(caldata))),
	 						                                                period, nbins, lowcut = lowcut, highcut = highcut,
	 						                                                notch = notch, median = True, rebin = False,
	 						                                                verbose = verbose, return_error = True,
	 						                                                return_noise_harmonics = 30)
			folded_cal = folded_cal[0,:]
			dfolded_cal = dfolded_cal[0,:]
			folded_nonorm_cal = folded_nonorm_cal[0,:]
			dfolded_nonorm_cal = dfolded_nonorm_cal[0,:]

			t_cal = t_fold_cal.copy()
			
			print('Folding the calsource data finished')
	 	
		except:
	 		print('Error when folding the calsource data.')
	 		if caldata is None:
	 			print('No calsource data.')

		if doplot:
			figure()
			plot(caltime, caldata)
			xlabel('t [s]')
			ylabel('Calsource data [a.u.]')
			
			figure()
			############ Power spectrum
			subplot(2,1,1)
			spectrum_f, freq_f = ft.power_spectrum(caltime, caldata, rebin=True)
			plot(freq_f, scfilt.gaussian_filter1d(spectrum_f,1),label='Calsource Data')
			yscale('log')
			xscale('log')
			xlabel('Frequency [Hz]')
			ylabel('Power Spectrum')
			title('Calsource data')
		
			for i in range(20):
			    axvline(1./period*i,color='k',linestyle='--',alpha=0.3)
		
			if lowcut is not None:
			    axvline(lowcut,color='k')
			if highcut is not None:
			    axvline(highcut,color='k')
			legend()
		
			subplot(2,1,2)
			errorbar(t_cal, folded_nonorm_cal, yerr=dfolded_nonorm_cal, fmt='ro',
				label='Filtered Data {} < f < {} Hz'.format(lowcut,highcut))
			xlim(0,period)
			xlabel('Time [sec]')
			ylabel('Folded Signal Calsource [ADU]')
			grid()
			legend()
		
			########## New Power spectrum
			spectrum_f2, freq_f2 = ft.power_spectrum(caltime, newdata_cal[0,:], rebin=True)
			subplot(2,1,1)
			plot(freq_f2, scfilt.gaussian_filter1d(spectrum_f2,1),label='Filtered data')
		
			plot(fn_cal, nn_cal[0,:]**2,'ro-', label='Noise level between peaks')
			grid()
			legend()

			tight_layout()

	#	try:
		print('Fitting the calibration source data.')

		###Fit type
		if typefit == 'spl':
		### Instanciate timecst+spline object
			nparams_ext = nparams_ext_spl
			fctfit = asymsig_spl_class(t, tofit, errors, nparams_ext)

		elif typefit == 'poly':
		### Instanciate timecst+polynomials object
			nparams_ext = nparams_ext_poly
			fctfit = asymsig_poly

		elif typefit == 'just_exp':
		### Instanciate just timecst object
			nparams_ext = 0
			fctfit = simsig_asym

		else:
			print('Give a valid typefit: \'just_exp\', \'spl\' or \'poly\' ')

		### Vectors to fit

		tofit = np.reshape(folded_nonorm_cal, nbins)
		errors = np.reshape(dfolded_nonorm_cal, nbins)
		t = t_cal

		### Initial guess

		pnames = ['cycle', 'risetime', 'falltime', 't0']

		risetime = 0.05
		falltime = 0.05

		smoothed_tofit = savgol_filter(tofit, int(nbins/5), 3)

		tstart = t[argmin(np.gradient(smoothed_tofit))]
		amplitude = (np.max(tofit) - np.min(tofit))
		offset = np.max(tofit)
		guess_1 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]

		tstart = t[argmin(np.gradient(smoothed_tofit))]
		amplitude = -(np.max(tofit) - np.min(tofit))
		offset = np.min(tofit)
		guess_2 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]

		tstart = t[argmax(np.gradient(smoothed_tofit))]
		amplitude = (np.max(tofit) - np.min(tofit))
		offset = np.max(tofit)
		guess_3 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]

		tstart = t[argmax(np.gradient(smoothed_tofit))]
		amplitude = -(np.max(tofit) - np.min(tofit))
		offset = np.min(tofit)
		guess_4 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]

		tstart = t[argmin(np.gradient(tofit))]
		amplitude = (np.max(tofit) - np.min(tofit))
		offset = np.max(tofit)
		guess_5 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]

		tstart = t[argmin(np.gradient(tofit))]
		amplitude = -(np.max(tofit) - np.min(tofit))
		offset = np.min(tofit)
		guess_6 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]

		tstart = t[argmax(np.gradient(tofit))]
		amplitude = (np.max(tofit) - np.min(tofit))
		offset = np.max(tofit)
		guess_7 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]

		tstart = t[argmax(np.gradient(tofit))]
		amplitude = -(np.max(tofit) - np.min(tofit))
		offset = np.min(tofit)
		guess_8 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]


		if nparams_ext == 0:
		    allguess_1 = guess_1
		    allguess_2 = guess_2
		    allguess_3 = guess_3
		    allguess_4 = guess_4
		    allguess_5 = guess_5
		    allguess_6 = guess_6
		    allguess_7 = guess_7
		    allguess_8 = guess_8
		else:
		    allguess_1 = np.append(guess_1, np.zeros(nparams_ext-1) + guess_1[-1])
		    allguess_2 = np.append(guess_2, np.zeros(nparams_ext-1) + guess_2[-1])
		    allguess_3 = np.append(guess_3, np.zeros(nparams_ext-1) + guess_3[-1])
		    allguess_4 = np.append(guess_4, np.zeros(nparams_ext-1) + guess_4[-1])
		    allguess_5 = np.append(guess_5, np.zeros(nparams_ext-1) + guess_5[-1])
		    allguess_6 = np.append(guess_6, np.zeros(nparams_ext-1) + guess_6[-1])
		    allguess_7 = np.append(guess_7, np.zeros(nparams_ext-1) + guess_7[-1])
		    allguess_8 = np.append(guess_8, np.zeros(nparams_ext-1) + guess_8[-1])

		guesses = [guess_1, guess_2, guess_3, guess_4, guess_5, guess_6, guess_7, guess_8]
		allguesses = [allguess_1, allguess_2, allguess_3, allguess_4, allguess_5, allguess_6, allguess_7, allguess_8]

		difference_guess_1 = np.abs(simps((tofit-fctfit(t, allguess_1))**2,t))
		difference_guess_2 = np.abs(simps((tofit-fctfit(t, allguess_2))**2,t))
		difference_guess_3 = np.abs(simps((tofit-fctfit(t, allguess_3))**2,t))
		difference_guess_4 = np.abs(simps((tofit-fctfit(t, allguess_4))**2,t))
		difference_guess_5 = np.abs(simps((tofit-fctfit(t, allguess_5))**2,t))
		difference_guess_6 = np.abs(simps((tofit-fctfit(t, allguess_6))**2,t))
		difference_guess_7 = np.abs(simps((tofit-fctfit(t, allguess_7))**2,t))
		difference_guess_8 = np.abs(simps((tofit-fctfit(t, allguess_8))**2,t))

		difference_guesses_tofit = [difference_guess_1, difference_guess_2, difference_guess_3, difference_guess_4, difference_guess_5, difference_guess_6, difference_guess_7, difference_guess_8]

		guess = guesses[np.argmin(difference_guesses_tofit)]
		allguess = allguesses[np.argmin(difference_guesses_tofit)]

		### Limits
		# limits = [[0, 0., 1], [1, 0., 1.], [2, 0., 1.], [3, 0., period]]

		limits = [[0,np.maximum(dutycycle/100-0.2,0.),dutycycle/100+0.2], [1,0., risetime*10],
				   [2,0., falltime*10], [3,allguess[3]-period/2,allguess[3]+period/2],
				   [4,-1.2*np.abs(allguess[4]),1.2*np.abs(allguess[4])],[5,np.min(tofit)-0.1*np.abs(allguess[4]),np.max(tofit)+0.1*np.abs(allguess[4])]]

		### Fixed parameters
		fixpars = []

		m, ch2, ndf = fit_one(t, tofit, errors, guess, nparams_ext, fctfit = fctfit, limits=limits, fixpars=fixpars)
		
		# m_1, ch2_1, ndf_1 = fit_one(t, tofit, errors, guess, nparams_ext, fctfit = fctfit, limits=limits, fixpars=fixpars)
		# difference_1 = np.abs(simps((tofit-fctfit(t, m_1.values))**2,t))

		# guess[3] = t[argmin(-np.gradient(tofit))]
		# m_2, ch2_2, ndf_2 = fit_one(t, tofit, errors, guess, nparams_ext, fctfit = fctfit, limits=limits, fixpars=fixpars)
		# difference_2 = np.abs(simps((tofit-fctfit(t, m_2.values))**2,t))

		# if difference_2 > difference_1:
		#     m = m_1
		#     ch2 = ch2_1
		#     ndf = ndf_1
		#     guess[3] = t[argmin(np.gradient(tofit))]
		# else:
		#     m = m_2
		#     ch2 = ch2_2
		#     ndf = ndf_2

		ch2vals_cal = ch2
		ndfvals_cal = ndf
		dcfit_cal = m.values[0]
		dcerr_cal = m.errors[0]
		risefit_cal = m.values[1]
		riseerr_cal = m.errors[1]
		fallfit_cal = m.values[2]
		fallerr_cal = m.errors[2]
		t0fit_cal = m.values[3]
		t0err_cal = m.errors[3]
		ampfit_cal = m.values[4]
		amperr_cal = m.errors [4]
		validfit_cal = m.valid
		
		d_cal = {'dutycycle':dcfit_cal, 'dutycyle_error':dcerr_cal, 'risetime':risefit_cal, 'risetime_error' : riseerr_cal, 'falltime' : fallfit_cal, 'falltime_error' : fallerr_cal, 't0' : t0fit_cal, 't0_error' : t0err_cal, 'amplitude' : ampfit_cal, 'amplitude_error' :amperr_cal, 'ch2' : ch2vals_cal, 'ndf' : ndfvals_cal, 'valid_minuitfit' : validfit_cal,'calsource_info':calsource_dict}
		
		print('Fitting the calibration source data finished.')	

		if doplot:
			
			### Plotting different guesses
#			guess_fct_1 = fctfit(t, allguess_1)
#			myguesspars_1 = allguess_1.copy()
#			myguesspars_1[4] = 0
#			myslowguess_1 = fctfit(t, myguesspars_1)
#
#			guess_fct_2 = fctfit(t, allguess_2)
#			myguesspars_2 = allguess_2.copy()
#			myguesspars_2[4] = 0
#			myslowguess_2 = fctfit(t, myguesspars_2)
#
#			guess_fct_3 = fctfit(t, allguess_3)
#			myguesspars_3 = allguess_3.copy()
#			myguesspars_3[4] = 0
#			myslowguess_3 = fctfit(t, myguesspars_3)
#
#			guess_fct_4 = fctfit(t, allguess_4)
#			myguesspars_4 = allguess_4.copy()
#			myguesspars_4[4] = 0
#			myslowguess_4 = fctfit(t, myguesspars_4)
#
#			guess_fct_5 = fctfit(t, allguess_5)
#			myguesspars_5 = allguess_5.copy()
#			myguesspars_5[4] = 0
#			myslowguess_5 = fctfit(t, myguesspars_5)
#
#			guess_fct_6 = fctfit(t, allguess_6)
#			myguesspars_6 = allguess_6.copy()
#			myguesspars_6[4] = 0
#			myslowguess_6 = fctfit(t, myguesspars_6)
#
#			guess_fct_7 = fctfit(t, allguess_7)
#			myguesspars_7 = allguess_7.copy()
#			myguesspars_7[4] = 0
#			myslowguess_7 = fctfit(t, myguesspars_7)
#
#			guess_fct_8 = fctfit(t, allguess_8)
#			myguesspars_8 = allguess_8.copy()
#			myguesspars_8[4] = 0
#			myslowguess_8 = fctfit(t, myguesspars_8)
#
			guess_fct = fctfit(t, allguess)
			myguesspars = allguess.copy()
			myguesspars[4] = 0
			myslowguess = fctfit(t, myguesspars)

			figure()

#			plot(t, guess_fct_1, label='guess 1',color='C0')
#			axvline(allguess_1[3],linestyle='--',color='C0',label='t0 1')
#			# # plot(t, myslowguess_1, label='slow guess 1',color='C0')
#
#			plot(t, guess_fct_2, label='guess 2',color='C1')
#			axvline(allguess_2[3],linestyle='--',color='C1',label='t0 2')
#			# # plot(t, myslowguess_2, label='slow guess 2',color='C1')
#
#			plot(t, guess_fct_3, label='guess 3',color='C2')
#			axvline(allguess_3[3],linestyle='--',color='C2',label='t0 3')
#			# # plot(t, myslowguess_3, label='slow guess 3',color='C2')
#
#			plot(t, guess_fct_4, label='guess 4',color='C3')
#			axvline(allguess_4[3],linestyle='--',color='C3',label='t0 4')
#			# # plot(t, myslowguess_4, label='slow guess 4',color='C3')
#
#			plot(t, guess_fct_5, label='guess 5',color='C4')
#			axvline(allguess_5[3],linestyle='--',color='C4',label='t0 5')
#			# # plot(t, myslowguess_5, label='slow guess 5',color='C4')
#
#			plot(t, guess_fct_6, label='guess 6',color='C5')
#			axvline(allguess_6[3],linestyle='--',color='C5',label='t0 6')
#			# # plot(t, myslowguess_6, label='slow guess 6',color='C5')
#
#			plot(t, guess_fct_7, label='guess 7',color='C6')
#			axvline(allguess_7[3],linestyle='--',color='C6',label='t0 7')
#			# # plot(t, myslowguess_7, label='slow guess 7',color='C6')
#
#			plot(t, guess_fct_8, label='guess 8',color='C7')
#			axvline(allguess_8[3],linestyle='--',color='C7',label='t0 8')
#			# # plot(t, myslowguess_8, label='slow guess 8',color='C7')
#
			plot(t, guess_fct, label='best guess')
			axvline(allguess[3],linestyle='--',label='t0')
			plot(t, myslowguess, label='slow guess')

#			plot(t,smoothed_tofit,label='Smoothed folded')
#			plot(t,np.gradient(smoothed_tofit),label='Diff smoothed folded')
			errorbar(t, tofit, yerr=errors, fmt='ro', label='Data', alpha=0.5)

			grid()
			legend()
			tight_layout

			figure(figsize=(11,4))
			### Plot the fit
			subplot(1,3,1)
			# plot(t, guess_fct, label='guess')
			# plot(t, myslowguess, label='slow guess')
			errorbar(t, tofit, yerr=errors, fmt='ro', label='Data', alpha=0.5)
			plot(t, fctfit(t, m.values), label="Time-cst + "+typefit)
			fit_info = [
			f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {ch2:.1f} / {ndf}",
			]
			for i in range(4):
				vi = m.values[i]
				ei = m.errors[i]
				fit_info.append(f"{pnames[i]} = ${vi:.3f} \\pm {ei:.3f}$")
			grid()
			legend(title="\n".join(fit_info));


			### Plot the data and fit corrected for slow-variations fitted with splines or polynomial
			# slow variations are obtained with the same params but amplitude 0
			myslowpars = np.array(m.values)
			myslowpars[4] = 0.
			myslow = fctfit(t, myslowpars)

			subplot(1,3,2)
			errorbar(t, tofit, yerr=errors, fmt='ro', label='Data', alpha=0.5)
			plot(t, fctfit(t, m.values), label='Fitted')
			plot(t, myslow, label='Slow component')
			grid()
			legend()

			subplot(1,3,3)
			errorbar(t, tofit-myslow, yerr=errors, fmt='ro', label='Data Corrected', alpha=0.5)
			plot(t, fctfit(t, m.values)-myslow, label='Time-CSt Fit')
			grid()
			legend(title="\n".join(fit_info))

	#	except:
	#		print('Error when fitting the calibration source data to compute the their time constants (+powermeter ones), probably no calsource data available')
	else:
		d_cal = {'No external calibration source analysis'}
	
	##Now for all the TODs

	##first define de Vbias
	
	minVbias= min(minVbias_asic1,minVbias_asic2,maxVbias_asic1,maxVbias_asic2)
	maxVbias= max(minVbias_asic1,minVbias_asic2,maxVbias_asic1,maxVbias_asic2)	
	
	if (maxVbias-minVbias)/np.mean([maxVbias,minVbias]) < 0.001:
		Vbias = np.mean([maxVbias,minVbias])
	else:
		Vbias = 'Vbias not well defined'
		print('Different min and max Vbias, or different ASIC\'s Vbias')

	upper_satval = 4.15*1e6
	lower_satval = -4.15*1e6

	frac_sat_pertes = np.zeros(256)

	for i in range(256):
	    mask1 = alltod[i] > upper_satval
	    mask2 = alltod[i] < lower_satval
	    frac_sat_pertes[i] = (np.sum(mask1)+np.sum(mask2))/len(alltod[i])

	nonsaturated_tes = frac_sat_pertes == 0
	fraction_no_saturated_tes = np.sum(nonsaturated_tes) / 256
	fraction_saturated_tes = np.sum(~nonsaturated_tes) / 256

	d_ok = {} # dictionary to store the boolean ok's array for different criteria
		
	d_ok['Saturation'] = nonsaturated_tes

	if doplot:
		figure()
		for i in range(256):
		    plot(tt,alltod[i])
		plot(tt,np.ones(len(tt))*upper_satval,color='black')
		plot(tt,np.ones(len(tt))*lower_satval,color='black')
		title('{} % detectors reaches saturation'.format(100*(fraction_saturated_tes)))
		xlabel('Time [s]')
		ylabel('ADU')
		tight_layout
		
		figure()
		for i in range(256):
			if nonsaturated_tes[i]:
				plot(tt,alltod[i])
		plot(tt,np.ones(len(tt))*upper_satval,color='black')
		plot(tt,np.ones(len(tt))*lower_satval,color='black')
		title('Timelines for nonsaturated TESs')
		xlabel('Time [s]')
		ylabel('ADU')
		tight_layout
		
		spectra = []
		smooth_param = 2
		smooth_spectra = []

		for i in np.arange(256):
		    spectrum_f, freq_f = ft.power_spectrum(tt, alltod[i], rebin=True)
		    spectra.append(spectrum_f)
		    smooth_spectrum_f = f.gaussian_filter1d(spectra[i],smooth_param)
		    smooth_spectra.append(smooth_spectrum_f)
		    
		spectra = np.asarray(spectra)
		smooth_spectra = np.asarray(smooth_spectra)
		
		figure()

		# notch = np.array([[0.852, 0.003, 1],
		#                   [1.724, 0.003, 3],
		#                   [2.35, 0.03, 1],
		#                   [6.939, 0.003, 1]])

		# for i in range(notch.shape[0]):
		#     nharms = notch[i,2].astype(int)
		#     for j in range(nharms):
		#         if j==0:
		#             axvline(notch[i,0]*(j+1),linestyle='-.',color='blue')   
		#         else:
		#             axvline(notch[i,0]*(j+1),linestyle='-.',color='blue') 

		for i in np.arange(10):
			if i==0:
				axvline(i*fmod,linestyle='--',color='gray',label='fmod')
			else:
				axvline(i*fmod,linestyle='--',color='gray')

		for i in np.arange(256):
			if nonsaturated_tes[i]:
				plot(freq_f, smooth_spectra[i],'k-',alpha=0.1)

# 		yscale('log')
# 		xscale('log')
# 		legend()
# 		xlabel('Frequency [Hz]')
# 		ylabel('Smoothed spectra')
# 		tight_layout			

	print('Folding TOD\'s timelines')	
	
	folded, t_fold, folded_nonorm, dfolded, dfolded_nonorm, newdata, fn, nn= ft.fold_data(tt, alltod, period, nbins, lowcut=lowcut,
						highcut=highcut, notch=notch, median=True, rebin=False, verbose=verbose, return_error=True,
						return_noise_harmonics=30)

	t = t_fold.copy()

	print('Folding TOD\'s timelines finished')
	
	if doplot:
		spectra = []
		smooth_param = 2
		smooth_spectra = []

		for i in np.arange(256):
		    spectrum_f, freq_f = ft.power_spectrum(tt, newdata[i], rebin=True)
		    spectra.append(spectrum_f)
		    smooth_spectrum_f = f.gaussian_filter1d(spectra[i],smooth_param)
		    smooth_spectra.append(smooth_spectrum_f)
		    
		spectra = np.asarray(spectra)
		smooth_spectra = np.asarray(smooth_spectra)
		
		# notch = np.array([[0.852, 0.003, 1],
		#                   [1.724, 0.003, 3],
		#                   [2.35, 0.03, 1],
		#                   [6.939, 0.003, 1]])

		# for i in range(notch.shape[0]):
		#     nharms = notch[i,2].astype(int)
		#     for j in range(nharms):
		#         if j==0:
		#             axvline(notch[i,0]*(j+1),linestyle='-.',color='blue')   
		#         else:
		#             axvline(notch[i,0]*(j+1),linestyle='-.',color='blue') 

# 		for i in np.arange(10):
# 			if i==0:
# 				axvline(i*fmod,linestyle='--',color='gray',label='fmod')
# 			else:
# 				axvline(i*fmod,linestyle='--',color='gray')

		for i in np.arange(256):
			if nonsaturated_tes[i]:
				plot(freq_f, smooth_spectra[i],'b-',alpha=0.1)

		yscale('log')
		xscale('log')
		legend()
		xlabel('Frequency [Hz]')
		ylabel('Smoothed spectra')
		tight_layout

#	try:
	print('Fitting no normalized folded data.')

	###Fit type
	if typefit == 'spl':
	### Instanciate timecst+spline object
		nparams_ext = nparams_ext_spl
		fctfit = asymsig_spl_class(t, tofit, errors, nparams_ext)

	elif typefit == 'poly':
	### Instanciate timecst+polynomials object
		nparams_ext = nparams_ext_poly
		fctfit = asymsig_poly

	elif typefit == 'just_exp':
	### Instanciate just timecst object
		nparams_ext = 0
		fctfit = simsig_asym

	else:
		print('Give a valid typefit: \'just_exp\', \'spl\' or \'poly\' ')

	mean_fold = np.nanmean(folded[nonsaturated_tes],axis=0)
	median_fold = np.nanmedian(folded[nonsaturated_tes],axis=0)
	smoothed_median_fold = savgol_filter(median_fold, int(nbins/5), 3)
	sigma_fold = np.nanstd(folded[nonsaturated_tes],axis=0)
	error_mean = sigma_fold / np.sqrt(len(nonsaturated_tes)) # revisit this
	errors_median = 1.253 * sigma_fold / np.sqrt(len(nonsaturated_tes)) #revisit this

	### Fit the folded median

	tofit = median_fold
	errors = errors_median

	### Initial guess

	pnames = ['cycle', 'risetime', 'falltime', 't0']

	risetime = 0.1
	falltime = 0.1

	smoothed_tofit = savgol_filter(tofit, int(nbins/5), 3)

	tstart = t[argmin(np.gradient(smoothed_tofit))]
	amplitude = (np.max(tofit) - np.min(tofit))
	offset = np.max(tofit)
	guess_1 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]

	tstart = t[argmin(np.gradient(smoothed_tofit))]
	amplitude = -(np.max(tofit) - np.min(tofit))
	offset = np.min(tofit)
	guess_2 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]

	tstart = t[argmax(np.gradient(smoothed_tofit))]
	amplitude = (np.max(tofit) - np.min(tofit))
	offset = np.max(tofit)
	guess_3 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]

	tstart = t[argmax(np.gradient(smoothed_tofit))]
	amplitude = -(np.max(tofit) - np.min(tofit))
	offset = np.min(tofit)
	guess_4 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]

	tstart = t[argmin(np.gradient(tofit))]
	amplitude = (np.max(tofit) - np.min(tofit))
	offset = np.max(tofit)
	guess_5 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]

	tstart = t[argmin(np.gradient(tofit))]
	amplitude = -(np.max(tofit) - np.min(tofit))
	offset = np.min(tofit)
	guess_6 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]

	tstart = t[argmax(np.gradient(tofit))]
	amplitude = (np.max(tofit) - np.min(tofit))
	offset = np.max(tofit)
	guess_7 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]

	tstart = t[argmax(np.gradient(tofit))]
	amplitude = -(np.max(tofit) - np.min(tofit))
	offset = np.min(tofit)
	guess_8 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]


	if nparams_ext == 0:
	    allguess_1 = guess_1
	    allguess_2 = guess_2
	    allguess_3 = guess_3
	    allguess_4 = guess_4
	    allguess_5 = guess_5
	    allguess_6 = guess_6
	    allguess_7 = guess_7
	    allguess_8 = guess_8
	else:
	    allguess_1 = np.append(guess_1, np.zeros(nparams_ext-1) + guess_1[-1])
	    allguess_2 = np.append(guess_2, np.zeros(nparams_ext-1) + guess_2[-1])
	    allguess_3 = np.append(guess_3, np.zeros(nparams_ext-1) + guess_3[-1])
	    allguess_4 = np.append(guess_4, np.zeros(nparams_ext-1) + guess_4[-1])
	    allguess_5 = np.append(guess_5, np.zeros(nparams_ext-1) + guess_5[-1])
	    allguess_6 = np.append(guess_6, np.zeros(nparams_ext-1) + guess_6[-1])
	    allguess_7 = np.append(guess_7, np.zeros(nparams_ext-1) + guess_7[-1])
	    allguess_8 = np.append(guess_8, np.zeros(nparams_ext-1) + guess_8[-1])

	guesses = [guess_1, guess_2, guess_3, guess_4, guess_5, guess_6, guess_7, guess_8]
	allguesses = [allguess_1, allguess_2, allguess_3, allguess_4, allguess_5, allguess_6, allguess_7, allguess_8]

	difference_guess_1 = np.abs(simps((tofit-fctfit(t, allguess_1))**2,t))
	difference_guess_2 = np.abs(simps((tofit-fctfit(t, allguess_2))**2,t))
	difference_guess_3 = np.abs(simps((tofit-fctfit(t, allguess_3))**2,t))
	difference_guess_4 = np.abs(simps((tofit-fctfit(t, allguess_4))**2,t))
	difference_guess_5 = np.abs(simps((tofit-fctfit(t, allguess_5))**2,t))
	difference_guess_6 = np.abs(simps((tofit-fctfit(t, allguess_6))**2,t))
	difference_guess_7 = np.abs(simps((tofit-fctfit(t, allguess_7))**2,t))
	difference_guess_8 = np.abs(simps((tofit-fctfit(t, allguess_8))**2,t))

	difference_guesses_tofit = [difference_guess_1, difference_guess_2, difference_guess_3, difference_guess_4, difference_guess_5, difference_guess_6, difference_guess_7, difference_guess_8]

	guess = guesses[np.argmin(difference_guesses_tofit)]
	allguess = allguesses[np.argmin(difference_guesses_tofit)]

	### Limits
	# limits = [[0, 0., 1], [1, 0., 1.], [2, 0., 1.], [3, 0., period]]

	limits = [[0,np.maximum(dutycycle/100-0.2,0.),dutycycle/100+0.2], [1,0., risetime*10],
			   [2,0., falltime*10], [3,allguess[3]-period/2,allguess[3]+period/2],
			   [4,-1.2*np.abs(allguess[4]),1.2*np.abs(allguess[4])],[5,np.min(tofit)-0.1*np.abs(allguess[4]),np.max(tofit)+0.1*np.abs(allguess[4])]]

	### Fixed parameters
	fixpars = []

	### Run minuit

	m, ch2, ndf = fit_one(t, tofit, errors, guess, nparams_ext, fctfit = fctfit, limits=limits, fixpars=fixpars)
	
	if m != 0:
		ch2vals_folded_median = ch2
		ndfvals_folded_median = ndf
		dcfit_folded_median = m.values[0]
		dcerr_folded_median = m.errors[0]
		risefit_folded_median = m.values[1]
		riseerr_folded_median = m.errors[1]
		fallfit_folded_median = m.values[2]
		fallerr_folded_median = m.errors[2]
		t0fit_folded_median = m.values[3]
		t0err_folded_median = m.errors[3]
		ampfit_folded_median = m.values[4]
		amperr_folded_median = m.errors[4]
		validfit_folded_median = m.valid
		allpars_folded_median = np.array(m.values)
		allerrs_folded_median = np.array(m.errors)
	else:
		print('Folded median nonsaturated TES\'s')
	
	d_folded_median = {'dutycycle':dcfit_folded_median, 'dutycyle_error':dcerr_folded_median, 'risetime':risefit_folded_median, 'risetime_error' : riseerr_folded_median, 'falltime' : fallfit_folded_median, 'falltime_error' : fallerr_folded_median, 't0' : t0fit_folded_median, 't0_error' : t0err_folded_median, 'amplitude' : ampfit_folded_median, 'amplitude_error' :amperr_folded_median, 'ch2' : ch2vals_folded_median, 'ndf' : ndfvals_folded_median, 'valid_minuitfit' : validfit_folded_median}

	if doplot:			
		figure()
		for order,i in enumerate(np.arange(256)):#[nonsaturated_tes]
			if order == 0:
				plot(t,folded[i,:], 'k-',alpha=0.1,label='Folded data for all detectors')
			plot(t, folded[i,:], 'k-',alpha=0.1)
		plot(t,median_fold,'bo',label='Median over nonsaturated folded')
		plot(t,fctfit(t,allpars_folded_median),color='blue',label='Fitted median')
		if calsource_analysis:
			plot(t_cal,folded_cal,color='red',label='Folded calsource data')
		ylim(-2,2)
		legend()
		xlabel('Time [s]')
		ylabel('Stacked folded data')
		tight_layout


	ch2vals = np.zeros(256)
	ndfvals = np.zeros(256)
	dcfit = np.zeros(256)
	dcerr = np.zeros(256)
	risefit = np.zeros(256)
	riseerr = np.zeros(256)
	fallfit = np.zeros(256)
	fallerr = np.zeros(256)
	t0fit   = np.zeros(256)
	t0err   = np.zeros(256)
	ampfit  = np.zeros(256)
	amperr  = np.zeros(256)
	validfit = np.zeros(256, dtype=bool)

	if nparams_ext == 0:
		allpars = np.zeros(( 256, 6 ))
		allerrs = np.zeros(( 256, 6 ))	

	else:
		allpars = np.zeros(( 256, 5 + nparams_ext ))
		allerrs = np.zeros(( 256, 5 + nparams_ext ))

	d_alltod_nonorm = {}
	
	rc('figure',figsize=(11,4))
	nh = 3	
	
	for i in range(256):

		### Vectors to fit

		tofit = folded_nonorm[i]
		errors = dfolded_nonorm[i]

		### Initial guess

		pnames = ['cycle', 'risetime', 'falltime', 't0']

		risetime = 0.1
		falltime = 0.1

		smoothed_tofit = savgol_filter(tofit, int(nbins/5), 3)

		tstart = t[argmin(np.gradient(smoothed_tofit))]
		# tstart = t[argmin(np.gradient(tofit))]
		amplitude = (np.max(tofit) - np.min(tofit))
		offset = np.max(tofit)
		guess_1 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]

		tstart = t[argmin(np.gradient(smoothed_tofit))]
		# tstart = t[argmin(np.gradient(tofit))]
		amplitude = -(np.max(tofit) - np.min(tofit))
		offset = np.min(tofit)
		guess_2 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]

		tstart = t[argmax(np.gradient(smoothed_tofit))]
		# tstart = t[argmin(np.gradient(tofit))]
		amplitude = (np.max(tofit) - np.min(tofit))
		offset = np.max(tofit)
		guess_3 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]

		tstart = t[argmax(np.gradient(smoothed_tofit))]
		# tstart = t[argmin(np.gradient(tofit))]
		amplitude = -(np.max(tofit) - np.min(tofit))
		offset = np.min(tofit)
		guess_4 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]

		tstart = t[argmin(np.gradient(smoothed_median_fold))]
		# tstart = t[argmin(np.gradient(tofit))]
		amplitude = (np.max(tofit) - np.min(tofit))
		offset = np.max(tofit)
		guess_5 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]

		tstart = t[argmin(np.gradient(smoothed_median_fold))]
		# tstart = t[argmin(np.gradient(tofit))]
		amplitude = -(np.max(tofit) - np.min(tofit))
		offset = np.min(tofit)
		guess_6 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]

		tstart = t[argmax(np.gradient(smoothed_median_fold))]
		# tstart = t[argmin(np.gradient(tofit))]
		amplitude = (np.max(tofit) - np.min(tofit))
		offset = np.max(tofit)
		guess_7 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]

		tstart = t[argmax(np.gradient(smoothed_median_fold))]
		# tstart = t[argmin(np.gradient(tofit))]
		amplitude = -(np.max(tofit) - np.min(tofit))
		offset = np.min(tofit)
		guess_8 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]


		if nparams_ext == 0:
		    allguess_1 = guess_1
		    allguess_2 = guess_2
		    allguess_3 = guess_3
		    allguess_4 = guess_4
		    allguess_5 = guess_5
		    allguess_6 = guess_6
		    allguess_7 = guess_7
		    allguess_8 = guess_8
		else:
		    allguess_1 = np.append(guess_1, np.zeros(nparams_ext-1) + guess_1[-1])
		    allguess_2 = np.append(guess_2, np.zeros(nparams_ext-1) + guess_2[-1])
		    allguess_3 = np.append(guess_3, np.zeros(nparams_ext-1) + guess_3[-1])
		    allguess_4 = np.append(guess_4, np.zeros(nparams_ext-1) + guess_4[-1])
		    allguess_5 = np.append(guess_5, np.zeros(nparams_ext-1) + guess_5[-1])
		    allguess_6 = np.append(guess_6, np.zeros(nparams_ext-1) + guess_6[-1])
		    allguess_7 = np.append(guess_7, np.zeros(nparams_ext-1) + guess_7[-1])
		    allguess_8 = np.append(guess_8, np.zeros(nparams_ext-1) + guess_8[-1])

		guesses = [guess_1, guess_2, guess_3, guess_4, guess_5, guess_6, guess_7, guess_8]
		allguesses = [allguess_1, allguess_2, allguess_3, allguess_4, allguess_5, allguess_6, allguess_7, allguess_8]

		difference_guess_1 = np.abs(simps((tofit-fctfit(t, allguess_1))**2,t))
		difference_guess_2 = np.abs(simps((tofit-fctfit(t, allguess_2))**2,t))
		difference_guess_3 = np.abs(simps((tofit-fctfit(t, allguess_3))**2,t))
		difference_guess_4 = np.abs(simps((tofit-fctfit(t, allguess_4))**2,t))
		difference_guess_5 = np.abs(simps((tofit-fctfit(t, allguess_5))**2,t))
		difference_guess_6 = np.abs(simps((tofit-fctfit(t, allguess_6))**2,t))
		difference_guess_7 = np.abs(simps((tofit-fctfit(t, allguess_7))**2,t))
		difference_guess_8 = np.abs(simps((tofit-fctfit(t, allguess_8))**2,t))

		difference_guesses_tofit = [difference_guess_1, difference_guess_2, difference_guess_3, difference_guess_4, difference_guess_5, difference_guess_6, difference_guess_7, difference_guess_8]

		guess = guesses[np.argmin(difference_guesses_tofit)]
		allguess = allguesses[np.argmin(difference_guesses_tofit)]

		### Limits
		# limits = [[0, 0., 1], [1, 0., 1.], [2, 0., 1.], [3, 0., period]]

		limits = [[0,np.maximum(dutycycle/100-0.2,0.),dutycycle/100+0.2], [1,0., risetime*10],
				   [2,0., falltime*10], [3,allguess[3]-period/2,allguess[3]+period/2],
				   [4,-1.2*np.abs(allguess[4]),1.2*np.abs(allguess[4])],[5,np.min(tofit)-0.1*np.abs(allguess[4]),np.max(tofit)+0.1*np.abs(allguess[4])]]

		### Fixed parameters
		fixpars = []

		### Run minuit

		m, ch2, ndf = fit_one(t, tofit, errors, guess, nparams_ext, fctfit = fctfit, limits=limits, fixpars=fixpars)

# 		m_1, ch2_1, ndf_1 = fit_one(t, tofit, errors, guess, nparams_ext, fctfit = fctfit, limits=limits, fixpars=fixpars)
#
# 		difference_1 = np.abs(simps((tofit-fctfit(t, m_1.values))**2,t))
#
# 		guess[3] = t[argmin(-np.gradient(tofit))]
#
# 		m_2, ch2_2, ndf_2 = fit_one(t, tofit, errors, guess, nparams_ext, fctfit = fctfit, limits=limits, fixpars=fixpars)
#
# 		difference_2 = np.abs(simps((tofit-fctfit(t, m_2.values))**2,t))
#
# 		if difference_2 > difference_1:
# 			m = m_1
# 			ch2 = ch2_1
# 			ndf = ndf_1
# 			guess[3] = t[argmin(np.gradient(tofit))]
# 		else:
# 			m = m_2
# 			ch2 = ch2_2
# 			ndf = ndf_2		

		if m != 0:
			ch2vals[i] = ch2
			ndfvals[i] = ndf
			dcfit[i] = m.values[0]
			dcerr[i] = m.errors[0]
			risefit[i] = m.values[1]
			riseerr[i] = m.errors[1]
			fallfit[i] = m.values[2]
			fallerr[i] = m.errors[2]
			t0fit[i] = m.values[3]
			t0err[i] = m.errors[3]
			ampfit[i] = m.values[4]
			amperr[i] = m.errors[4]
			validfit[i] = m.valid
			allpars[i,:] = np.array(m.values)
			allerrs[i,:] = np.array(m.errors)
		else:
			print('TES# {}'.format(i+1))		
		
		if doplot and doplot_onebyone:
		
			if ((i)%nh) == 0:
				show()
				fig, axs = plt.subplots(2, nh, sharex=True)
				fig.subplots_adjust(hspace=0)

			if m != 0:
				fit_info = [
				    f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {ch2:.1f} / {ndf}",
				]
				fit_info.append(f"{pnames[1]} = ${m.values[1]:.2f} \\pm {m.errors[1]:.2f}$")
				fit_info.append(f"{pnames[2]} = ${m.values[2]:.2f} \\pm {m.errors[2]:.2f}$")
				fit_info.append(f"{pnames[3]} = ${m.values[3]:.2f} \\pm {m.errors[3]:.2f}$")

			#### We plot the data and fit with slow components on the top
			axs[0][i%nh].set_title('TES #{}'.format(i+1))
			if m !=0:
				myslowpars = np.array(m.values)
				myslowpars[4] = 0.
				myslow = fctfit(t, myslowpars)
				axs[0][i%nh].errorbar(t, tofit, yerr=errors, fmt='ko', label='Data', alpha=0.5)
				axs[0][i%nh].plot(t, fctfit(t, allguess), 'r', lw=3, linestyle = '--', label='First guess')
				axs[0][i%nh].plot(t, myslow, 'm', lw=3, label='Slow part',zorder=4)
				axs[0][i%nh].plot(t, fctfit(t, m.values), 'r', lw=3, label='Tcst + '+typefit+' fit',zorder=5)
				axs[0][i%nh].legend(fontsize=8, framealpha=0, loc='lower right')

				#### and we plot the data correct for slow component with only the time-cst fit on the bottom
				axs[1][i%nh].errorbar(t, tofit-myslow, yerr=errors, fmt='ko', label='Data Corrected', alpha=0.5)
				axs[1][i%nh].plot(t, fctfit(t, m.values)-myslow, 'b', lw=3, label='Tcst fit',zorder=5)
				axs[1][i%nh].legend(fontsize=8, framealpha=0, title="\n".join(fit_info), title_fontsize=8, loc='lower right')

	d_alltod_nonorm = {'dutycycle':dcfit, 'dutycyle_error':dcerr, 'risetime':risefit, 'risetime_error' : riseerr, 'falltime' : fallfit, 'falltime_error' : fallerr, 't0' : t0fit, 't0_error' : t0err, 'amplitude' : ampfit, 'amplitude_error' :amperr, 'ch2' : ch2vals, 'ndf' : ndfvals, 'valid_minuitfit' : validfit,'saturated_timeline_fraction':frac_sat_pertes}
	
	print('Fitting No normalized folded data finished')				

	print('Fitting normalized folded data.')

	### Fit the folded data
	
	ch2vals_folded = np.zeros(256)
	ndfvals_folded = np.zeros(256)
	dcfit_folded = np.zeros(256)
	dcerr_folded = np.zeros(256)
	risefit_folded = np.zeros(256)
	riseerr_folded = np.zeros(256)
	fallfit_folded = np.zeros(256)
	fallerr_folded = np.zeros(256)
	t0fit_folded   = np.zeros(256)
	t0err_folded   = np.zeros(256)
	ampfit_folded  = np.zeros(256)
	amperr_folded  = np.zeros(256)
	validfit_folded = np.zeros(256, dtype=bool)

	if nparams_ext == 0:
		allpars_folded = np.zeros(( 256, 6 ))
		allerrs_folded = np.zeros(( 256, 6 ))	

	else:
		allpars_folded = np.zeros(( 256, 5 + nparams_ext ))
		allerrs_folded = np.zeros(( 256, 5 + nparams_ext ))

	residuals = np.empty(256)
	residuals[:] = np.nan
	residuals_fit = np.empty(256)
	residuals_fit[:] = np.nan

	
	for i in range(256):

		### Vectors to fit

		tofit = folded[i]
		errors = dfolded[i]

		### Initial guess

		pnames = ['cycle', 'risetime', 'falltime', 't0']

		risetime = 0.1
		falltime = 0.1

		smoothed_tofit = savgol_filter(tofit, int(nbins/5), 3)


		tstart = t[argmin(np.gradient(smoothed_tofit))]
		# tstart = t[argmin(np.gradient(tofit))]
		amplitude = (np.max(tofit) - np.min(tofit))
		offset = np.max(tofit)
		guess_1 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]

		tstart = t[argmin(np.gradient(smoothed_tofit))]
		# tstart = t[argmin(np.gradient(tofit))]
		amplitude = -(np.max(tofit) - np.min(tofit))
		offset = np.min(tofit)
		guess_2 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]

		tstart = t[argmax(np.gradient(smoothed_tofit))]
		# tstart = t[argmin(np.gradient(tofit))]
		amplitude = (np.max(tofit) - np.min(tofit))
		offset = np.max(tofit)
		guess_3 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]

		tstart = t[argmax(np.gradient(smoothed_tofit))]
		# tstart = t[argmin(np.gradient(tofit))]
		amplitude = -(np.max(tofit) - np.min(tofit))
		offset = np.min(tofit)
		guess_4 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]

		tstart = t[argmin(np.gradient(smoothed_median_fold))]
		# tstart = t[argmin(np.gradient(tofit))]
		amplitude = (np.max(tofit) - np.min(tofit))
		offset = np.max(tofit)
		guess_5 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]

		tstart = t[argmin(np.gradient(smoothed_median_fold))]
		# tstart = t[argmin(np.gradient(tofit))]
		amplitude = -(np.max(tofit) - np.min(tofit))
		offset = np.min(tofit)
		guess_6 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]

		tstart = t[argmax(np.gradient(smoothed_median_fold))]
		# tstart = t[argmin(np.gradient(tofit))]
		amplitude = (np.max(tofit) - np.min(tofit))
		offset = np.max(tofit)
		guess_7 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]

		tstart = t[argmax(np.gradient(smoothed_median_fold))]
		# tstart = t[argmin(np.gradient(tofit))]
		amplitude = -(np.max(tofit) - np.min(tofit))
		offset = np.min(tofit)
		guess_8 = [dutycycle/100, risetime, falltime, tstart, amplitude, offset]


		if nparams_ext == 0:
		    allguess_1 = guess_1
		    allguess_2 = guess_2
		    allguess_3 = guess_3
		    allguess_4 = guess_4
		    allguess_5 = guess_5
		    allguess_6 = guess_6
		    allguess_7 = guess_7
		    allguess_8 = guess_8
		else:
		    allguess_1 = np.append(guess_1, np.zeros(nparams_ext-1) + guess_1[-1])
		    allguess_2 = np.append(guess_2, np.zeros(nparams_ext-1) + guess_2[-1])
		    allguess_3 = np.append(guess_3, np.zeros(nparams_ext-1) + guess_3[-1])
		    allguess_4 = np.append(guess_4, np.zeros(nparams_ext-1) + guess_4[-1])
		    allguess_5 = np.append(guess_5, np.zeros(nparams_ext-1) + guess_5[-1])
		    allguess_6 = np.append(guess_6, np.zeros(nparams_ext-1) + guess_6[-1])
		    allguess_7 = np.append(guess_7, np.zeros(nparams_ext-1) + guess_7[-1])
		    allguess_8 = np.append(guess_8, np.zeros(nparams_ext-1) + guess_8[-1])

		guesses = [guess_1, guess_2, guess_3, guess_4, guess_5, guess_6, guess_7, guess_8]
		allguesses = [allguess_1, allguess_2, allguess_3, allguess_4, allguess_5, allguess_6, allguess_7, allguess_8]

		difference_guess_1 = np.abs(simps((tofit-fctfit(t, allguess_1))**2,t))
		difference_guess_2 = np.abs(simps((tofit-fctfit(t, allguess_2))**2,t))
		difference_guess_3 = np.abs(simps((tofit-fctfit(t, allguess_3))**2,t))
		difference_guess_4 = np.abs(simps((tofit-fctfit(t, allguess_4))**2,t))
		difference_guess_5 = np.abs(simps((tofit-fctfit(t, allguess_5))**2,t))
		difference_guess_6 = np.abs(simps((tofit-fctfit(t, allguess_6))**2,t))
		difference_guess_7 = np.abs(simps((tofit-fctfit(t, allguess_7))**2,t))
		difference_guess_8 = np.abs(simps((tofit-fctfit(t, allguess_8))**2,t))

		difference_guesses_tofit = [difference_guess_1, difference_guess_2, difference_guess_3, difference_guess_4, difference_guess_5, difference_guess_6, difference_guess_7, difference_guess_8]

		guess = guesses[np.argmin(difference_guesses_tofit)]
		allguess = allguesses[np.argmin(difference_guesses_tofit)]

		### Limits
		# limits = [[0, 0., 1], [1, 0., 1.], [2, 0., 1.], [3, 0., period]]

		limits = [[0,np.maximum(dutycycle/100-0.2,0.),dutycycle/100+0.2], [1,0., risetime*10],
				   [2,0., falltime*10], [3,allguess[3]-period/2,allguess[3]+period/2],
				   [4,-1.2*np.abs(allguess[4]),1.2*np.abs(allguess[4])],[5,np.min(tofit)-0.1*np.abs(allguess[4]),np.max(tofit)+0.1*np.abs(allguess[4])]]

		### Fixed parameters
		fixpars = []

		### Run minuit

		m, ch2, ndf = fit_one(t, tofit, errors, guess, nparams_ext, fctfit = fctfit, limits=limits, fixpars=fixpars)
		
		if m != 0:
			ch2vals_folded[i] = ch2
			ndfvals_folded[i] = ndf
			dcfit_folded[i] = m.values[0]
			dcerr_folded[i] = m.errors[0]
			risefit_folded[i] = m.values[1]
			riseerr_folded[i] = m.errors[1]
			fallfit_folded[i] = m.values[2]
			fallerr_folded[i] = m.errors[2]
			t0fit_folded[i] = m.values[3]
			t0err_folded[i] = m.errors[3]
			ampfit_folded[i] = m.values[4]
			amperr_folded[i] = m.errors[4]
			validfit_folded[i] = m.valid
			allpars_folded[i,:] = np.array(m.values)
			allerrs_folded[i,:] = np.array(m.errors)

			residuals_fit[i] = simps((fctfit(t,m.values)-fctfit(t,allpars_folded_median))**2,t)	
			
		else:
			print('TES# {}'.format(i+1))

		residuals[i] = simps((tofit-median_fold)**2,t)

	d_alltod_norm = {'dutycycle':dcfit_folded, 'dutycyle_error':dcerr_folded, 'risetime':risefit_folded, 'risetime_error' : riseerr_folded, 'falltime' : fallfit_folded, 'falltime_error' : fallerr_folded, 't0' : t0fit_folded, 't0_error' : t0err_folded, 'amplitude' : ampfit_folded, 'amplitude_error' :amperr_folded, 'ch2' : ch2vals_folded, 'ndf' : ndfvals_folded, 'valid_minuitfit' : validfit_folded,'residuals_folded_median':residuals, 'residuals_fit_folded_median':residuals_fit}

	print('Fitting normalized folded data finished')				
	
	### different types of discarding process
	## by clustering the residuals [ integrate((folded-foldedmedian[nonsaturated])**2) ]
	results = np.array([residuals]).T
	labels = run_DBSCAN(results, doplot=doplot, parnames = ['Residuals'],eps_cpar=0.6,min_samples_cpar = 20)
	total_labels = np.max(labels)+1 #without considering the noisy data in label=-1 that could appear
	
	if total_labels == 0:
		ok_dbs_residuals = np.zeros(256,dtype=bool)
		print('All data considered as noisy when clustering')
	else:
		n_samples_in_cluster = np.zeros(total_labels)
		for i in np.arange(total_labels):
			n_samples_in_cluster[i] = np.sum(labels==i)
		
		ok_dbs_residuals = (labels==np.argmax(n_samples_in_cluster))
	
	d_ok['Residuals'] = ok_dbs_residuals

# 	## by clustering the residuals [ integrate((fitted_folded-fitted_foldedmedian[nonsaturated])**2) ]
# 	results = np.array([residuals_fit]).T
# 	labels = run_DBSCAN(results, doplot=doplot,eps_cpar=0.6,min_samples_cpar = 20)
# 	total_labels = np.max(labels)+1 #without considering the noisy data in label=-1 that could appear
#
# 	if total_labels == 0:
# 		ok_dbs_residuals_fit = np.zeros(256,dtype=bool)
# 		print('All data considered as noisy when clustering')
# 	else:
# 		n_samples_in_cluster = np.zeros(total_labels)
# 		for i in np.arange(total_labels):
# 			n_samples_in_cluster[i] = np.sum(labels==i)
# 		
# 		ok_dbs_residuals_fit = (labels==np.argmax(n_samples_in_cluster))
# 	
# 	d_ok['Residuals_fit'] = ok_dbs_residuals_fit

# 	## by clustering the residuals [ integrate((folded-foldedmedian[nonsaturated])**2) ] with OPTICS
# 	results = np.array([residuals]).T
# 	labels = run_OPTICS(results, doplot=doplot,min_samples_optics = 20)
# 	total_labels = np.max(labels)+1 #without considering the noisy data in label=-1 that could appear
# 	if total_labels == 0:
# 		ok_optics_residuals = np.zeros(256,dtype=bool)
# 		print('All data considered as noisy when clustering')
# 	else:
# 		n_samples_in_cluster = np.zeros(total_labels)
# 		for i in np.arange(total_labels):
# 			n_samples_in_cluster[i] = np.sum(labels==i)
# 		
# 		ok_optics_residuals = (labels==np.argmax(n_samples_in_cluster))
# 	
# 	d_ok['Residuals-with-OPTICS'] = ok_optics_residuals

	## by clustering the parameters, errors and ch2
	results = np.array([residuals, risefit, fallfit, ch2vals]).T #, t0fit, t0err, ampfit, amperr
	labels = run_DBSCAN(results, doplot=doplot, parnames = ['Residuals','Risetime','Falltime','Ch2'], eps_cpar=2,min_samples_cpar = 20)
	total_labels = np.max(labels)+1 #without considering the noisy data in label=-1 that could appear

	if total_labels == 0:
		ok_dbs_parerrch2 = np.zeros(256,dtype=bool)
		print('All data considered as noisy when clustering')
	else:
		n_samples_in_cluster = np.zeros(total_labels)
		for i in np.arange(total_labels):
			n_samples_in_cluster[i] = np.sum(labels==i)
		
		ok_dbs_parerrch2 = (labels==np.argmax(n_samples_in_cluster))
	
	d_ok['Params-Errors-Ch2'] = ok_dbs_parerrch2

# 	## by clustering the parameters, errors and ch2 with OPTICS
# 	results = np.array([risefit, riseerr, fallfit, fallerr, t0fit, t0err, ampfit, amperr, ch2vals]).T
# 	labels = run_OPTICS(results, doplot=doplot,min_samples_optics = 20)
# 	total_labels = np.max(labels)+1 #without considering the noisy data in label=-1 that could appear
#
# 	if total_labels == 1:
# 		ok_optics_parerrch2 = np.zeros(256,dtype=bool)
# 		print('All data considered as noisy when clustering')
# 	else:
# 		n_samples_in_cluster = np.zeros(total_labels)
# 		for i in np.arange(total_labels):
# 			n_samples_in_cluster[i] = np.sum(labels==i)
# 		
# 		ok_optics_parerrch2 = (labels==np.argmax(n_samples_in_cluster))
# 	
# 	d_ok['Params-Errors-Ch2-with-OPTICS'] = ok_optics_parerrch2


	d_results['Vbias'] = Vbias
	d_results['calsource'] = d_cal
	d_results['folded_median'] = d_folded_median
	d_results['ok'] = d_ok
	d_results['alltod_folded_nonorm'] = d_alltod_nonorm
	d_results['alltod_folded_norm'] = d_alltod_norm



	if save_path is not None:
		wdir = os.getcwd()
		os.chdir(save_path)
		dictname = 'd__{}.npy'.format(d_results['dataset_info'])
		np.save(dictname,d_results)

		plot_folded_data_on_FP(folded_nonorm, time = t, datain_error = dfolded_nonorm, tes_signal_ok = nonsaturated_tes * ok_dbs_residuals, analytical_function = fctfit, eval_domain = t, params_function = allpars, save=True, figname = 'Folded-data-and-fits-on-FP-residualsdb-'+dataset_info)

	# 	plot_folded_data_on_FP(folded, time = t, datain_error = dfolded, tes_signal_ok = nonsaturated_tes * ok_dbs_residuals_fit, analytical_function = fctfit, eval_domain = t, params_function = allpars_folded, save=True, figname = 'Folded-data-and-fits-on-FP-residualsfitdb-'+dataset_info)

	# 	plot_folded_data_on_FP(folded_nonorm, time = t, datain_error = dfolded_nonorm, tes_signal_ok = nonsaturated_tes * ok_optics_residuals, analytical_function = fctfit, eval_domain = t, params_function = allpars, save=True, figname = 'Folded-data-and-fits-on-FP-residualsop'+dataset_info)			

	# 	plot_folded_data_on_FP(folded_nonorm, time = t, datain_error = dfolded_nonorm, tes_signal_ok = nonsaturated_tes * ok_dbs_parerrch2, analytical_function = fctfit, eval_domain = t, params_function = allpars, save=True, figname = 'Folded-data-and-fits-on-FP-parerrch2db'+dataset_info)			

	# 	plot_folded_data_on_FP(folded_nonorm, time = t, datain_error = dfolded_nonorm, tes_signal_ok = nonsaturated_tes * ok_optics_parerrch2, analytical_function = fctfit, eval_domain = t, params_function = allpars, save=True, figname = 'Folded-data-and-fits-on-FP-parerrch2op'+dataset_info)			

		os.chdir(wdir)


# 	except:
# 		print('Error when fitting folded TOD data in the first pass.')


	
	return d_results


# # to improve:
# when computing residuals, inverted signal are not corrected and then give a false bad result
# distinguish between the discarded TES by saturation and residuals in the focal plane plot

# definir la mediana
# fitear la mediana
# usar esos parámetros fiteados como los guess para fitear la normalized folded data

# ahora usar esos parámetros fiteados como los guess (salvo amp y offset) para fitear la nonorm folded data
