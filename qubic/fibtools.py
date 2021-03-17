# -*- coding: utf-8 -*-
from __future__ import division, print_function
import iminuit
import math
from matplotlib.pyplot import *
from pysimulators import FitsArray
import matplotlib.mlab as mlab
import sys

import scipy.signal as scsig
import scipy.stats
from scipy.ndimage.filters import correlate1d, gaussian_filter1d
import glob
from astropy.io import fits
from iminuit.util import describe, make_func_code
from functools import partial, update_wrapper
from qubic.utils import progress_bar

from qubicpack import qubicpack as qp
from qubicpack.pix2tes import assign_pix_grid, assign_pix2tes, tes2pix, pix2tes, TES2PIX

pix_grid = assign_pix_grid()
TES2PIX = assign_pix2tes()


def printnow(truc):
	print(truc)
	sys.stdout.flush()


def isfloat(s):
	try:
		a = float(s)
		return True
	except ValueError:
		pass
		return False


def statstr(x, divide=False, median=False, cut=None):
	if median:
		m = np.median(x[np.isfinite(x)])
		s = np.std(x[np.isfinite(x)])
	elif cut is not None:
		m, s = meancut(x[np.isfinite(x)], cut, disp=False)
	else:
		m = np.mean(x[np.isfinite(x)])
		s = np.std(x[np.isfinite(x)])
	if divide:
		nn = len(x[np.isfinite(x)])
		s /= nn
	return '{0:6.5f} +/- {1:6.5f}'.format(m, s)

def qgrid():
	for i in range(17):
		axvline(x=i-0.5, alpha=0.3, color='k')
		axhline(y=i-0.5, alpha=0.3, color='k')


def image_asics(data1=None, data2=None, all1=None):
	"""
	Return an image of detectors on the focal plane.
	Each asic has 124 TES and 4 thermometers.

	Parameters
	----------
	data1 :
	data2 :
	all1 : array
		signal from 2 asics (256 detectors)

	Returns
	-------

	"""

	if all1 is not None:
		nn = int(len(all1) / 2)
		data2 = all1[nn:]
		data1 = all1[0:nn]

	if data1 is not None:
		pix_grid[1, 16] = 1005
		pix_grid[2, 16] = 1006
		pix_grid[3, 16] = 1007
		pix_grid[4, 16] = 1008
	if data2 is not None:
		pix_grid[0, 15] = 1004
		pix_grid[0, 14] = 1003
		pix_grid[0, 13] = 1002
		pix_grid[0, 12] = 1001
	nrows = 17
	ncols = 17
	img = np.zeros((nrows, ncols)) + np.nan
	for row in range(nrows):
		for col in range(ncols):
			physpix = pix_grid[row, col]
			TES, asic = pix2tes(physpix)
			if data1 is not None and asic == 1:
				img[row, col] = data1[TES - 1]
			if data2 is not None and asic == 2:
				img[row, col] = data2[TES - 1]
	return img


"""
################################## Fitting ###################################
##############################################################################
"""


def thepolynomial(x, pars):
	"""
	Generic polynomial function

	"""

	f = np.poly1d(pars)
	return f(x)


class MyChi2:
	"""
	Class defining the minimizer and the data
	"""

	def __init__(self, xin, yin, covarin, functname, extra_args=None):
		self.x = xin
		self.y = yin
		self.covar = covarin
		self.invcov = np.linalg.inv(covarin)
		self.functname = functname
		self.extra_args = extra_args
	def __call__(self, *pars, extra_args=None):
		val = self.functname(self.x, pars, extra_args=self.extra_args)
		chi2 = np.dot(np.dot(self.y - val, self.invcov), self.y - val)
		return chi2

class Chi2Minimizer(object):
	"""
	Parent classes with the model to minimize
	"""

	def __init__(self, functname, xin, yin, covarin, extra_args = None):
		self.x = xin
		self.y = yin
		self.covar = covarin
		self.invcov = np.linalg.inv(covarin)
		self.functname = functname
		self.extra_args = extra_args
 
	def __call__(self, *pars, extra_args = None):
		val = self.functname(self.x, pars, extra_args = self.extra_args)
		chi2 = np.dot(np.dot(self.y - val, self.invcov), self.y - val)
		return chi2

class Chi2Implement(Chi2Minimizer):
	def __init__(self, functname, x, y, covarin, extra_args=None):
		super().__init__(functname, x, y, covarin, extra_args = extra_args)
		self.func_code = make_func_code(describe(functname)[1:-2])

class MyChi2_nocov:
	"""
	Class defining the minimizer and the data
	"""

	def __init__(self, xin, yin, invcovarin, functname):
		self.x = xin
		self.y = yin
		self.functname = functname
	
	def __call__(self, *pars):
		val = self.functname(self.x, pars)
		chi2 = np.sum((self.y - val) ** 2)
		# chi2 = np.dot(self.y - val, self.y - val)
		return chi2
		


# ## Call Minuit
def do_minuit(x, y, covarin, guess, functname=thepolynomial, fixpars=None, chi2=None, rangepars=None, nohesse=False,
			  force_chi2_ndf=False, verbose=True, minos=True, extra_args=None, print_level=0, force_diag=False,
			  nsplit=1, ncallmax=10000, precision=None):

	# check if covariance or error bars were given
	covar = covarin.copy()
	if np.size(np.shape(covarin)) == 1:
		if force_diag:
			covar = covarin.copy()
		else:
			err = covarin
			covar = np.zeros((np.size(err), np.size(err)))
			covar[np.arange(np.size(err)), np.arange(np.size(err))] = err ** 2
	# instantiate minimizer
	if chi2 is None:
		chi2 = MyChi2(x, y, covar, functname, extra_args=extra_args)
	else:
		chi2 = Chi2Implement(functname, x, y, covar, extra_args=extra_args)

		# nohesse=False
	#elif chi2.__name__ is 'MyChi2_nocov':
	#    chi2 = chi2(x, y, covar, functname)

	# variables
	ndim = np.size(guess)
	parnames = []
	for i in range(ndim):
		parnames.append('c' + np.str(i))
	# initial guess
	theguess = dict(zip(parnames, guess))
	# fixed parameters
	dfix = {}
	if fixpars is not None:
		for i in range(len(parnames)):
			dfix['fix_' + parnames[i]] = fixpars[i]
	else:
		for i in range(len(parnames)):
			dfix['fix_' + parnames[i]] = False
	# range for parameters
	drng = {}
	dstep = {}
	if rangepars is not None:
		step_norm = 100
		for i in range(len(parnames)):
			drng['limit_' + parnames[i]] = rangepars[i]
			dstep['error_' + parnames[i]] = (rangepars[i][1] - rangepars[i][0]) / step_norm
	else:
		for i in range(len(parnames)):
			drng['limit_' + parnames[i]] = False
			dstep['error_' + parnames[i]] = False

	# Run Minuit
	if verbose: print('Fitting with Minuit')
	ver = sys.version_info
	if ver.major < 3:
		theargs = dict(theguess.items() + dfix.items() + dstep.items())
		if rangepars is not None: theargs.update(dict(theguess.items() + drng.items()))
	else:
		for k in dfix.keys():
			theguess[k] = dfix[k]
		theargs = theguess
		if rangepars is not None:
			for k in drng.keys():
				theguess[k] = drng[k]
		theargs.update(theguess)

	if isinstance(chi2, MyChi2):
		m = iminuit.Minuit(chi2, forced_parameters=parnames, errordef=0.1, print_level=print_level, **theargs)
		m.migrad(ncall=ncallmax * nsplit, nsplit=nsplit, precision=precision)

	elif isinstance(chi2, Chi2Implement):
		#if verbose:
		#	print("Minimizer object: ", chi2.__dict__)
		#	print("Guess: ", *guess)
		#	print("ncallmax, nsplit, precision: ", ncallmax, nsplit, precision)	
		# if iminuit.version==2.2
		#m = iminuit.Minuit(chi2, *guess)
		#m.migrad(ncall = ncallmax * nsplit)
		#if iminuit.version==1.3
		m  = iminuit.Minuit(chi2, forced_parameters=parnames, errordef=0.1, print_level=print_level, **theargs)
		m.migrad(ncall=ncallmax * nsplit, nsplit=nsplit, precision=precision)
	# print('Migrad Done')
	if minos:
		try:
			m.minos()
			if verbose: print('Minos Done')
		except:
			if verbose: print('Minos Failed !')
	if nohesse is False:
		try:
			m.hesse()
			if verbose: print('Hesse Done')
		except:
			if verbose: print('Hesse failed !')
	# build np.array output
	parfit = []
	for i in parnames: parfit.append(m.values[i])

	errfit = []
	for i in parnames: errfit.append(m.errors[i])
	if fixpars is not None:
		parnamesfit = []
		for i in range(len(parnames)):
			if fixpars[i] is False:
				parnamesfit.append(parnames[i])
			if fixpars[i]:
				errfit[i] = 0
	else:
		parnamesfit = parnames
	ndimfit = len(parnamesfit)  # int(np.sqrt(len(m.errors)))
	covariance = np.zeros((ndimfit, ndimfit))
	if m.covariance:
		for i in range(ndimfit):
			for j in range(ndimfit):
				covariance[i, j] = m.covariance[(parnamesfit[i], parnamesfit[j])]

	if isinstance(chi2, MyChi2):
		chisq = chi2(*parfit)
	elif isinstance(chi2, Chi2Implement):
		ChiEvaluate = Chi2Minimizer(functname, x, y, covar) 
		chisq = ChiEvaluate(*parfit) 
		
	ndf = np.size(x) - ndim
	if force_chi2_ndf:
		correct = chisq / ndf
		if verbose:
			print('correcting errorbars to have chi2/ndf=1 - correction = {}'.format(chisq))
	else:
		correct = 1.
	if verbose:
		print(np.array(parfit))
		print(np.array(errfit) * np.sqrt(correct))
		print('Chi2=', chisq)
		print('ndf=', ndf)
	return m, np.array(parfit), np.array(errfit) * np.sqrt(correct), np.array(covariance) * correct, chi2(*parfit), ndf, chi2 


# ##############################################################################
# ##############################################################################


def profile(xin, yin, rng=None, nbins=10, fmt=None, plot=True, dispersion=True, log=False,
			median=False, cutbad=True, rebin_as_well=None, clip=None, mode=False):
	"""
	"""
	ok = np.isfinite(xin) * np.isfinite(yin)
	x = xin[ok]
	y = yin[ok]
	if rng is None:
		mini = np.min(x)
		maxi = np.max(x)
	else:
		mini = rng[0]
		maxi = rng[1]
	if log is False:
		xx = np.linspace(mini, maxi, nbins + 1)
	else:
		xx = np.logspace(np.log10(mini), np.log10(maxi), nbins + 1)
	xmin = xx[0:nbins]
	xmax = xx[1:]
	yval = np.zeros(nbins)
	xc = np.zeros(nbins)
	dy = np.zeros(nbins)
	dx = np.zeros(nbins)
	nn = np.zeros(nbins)
	if rebin_as_well is not None:
		nother = len(rebin_as_well)
		others = np.zeros((nbins, nother))
	else:
		others = None
	for i in np.arange(nbins):
		ok = (x > xmin[i]) & (x < xmax[i])
		newy = y[ok]
		if clip is not None:
			for k in np.arange(3):
				newy, mini, maxi = scipy.stats.sigmaclip(newy, low=clip, high=clip)
		nn[i] = len(newy)
		if median:
			yval[i] = np.median(y[ok])
		elif mode:
			mm, ss = meancut(y[ok], 3)
			hh = np.histogram(y[ok], bins=int(np.min([len(y[ok]) / 30, 100])), range=[mm - 5 * ss, mm + 5 * ss])
			idmax = np.argmax(hh[0])
			yval[i] = 0.5 * (hh[1][idmax + 1] + hh[1][idmax])
		else:
			yval[i] = np.mean(y[ok])
		xc[i] = (xmax[i] + xmin[i]) / 2
		if rebin_as_well is not None:
			for o in range(nother):
				others[i, o] = np.mean(rebin_as_well[o][ok])
		if dispersion:
			fact = 1
		else:
			fact = np.sqrt(len(y[ok]))
		dy[i] = np.std(y[ok]) / fact
		dx[i] = np.std(x[ok]) / fact
	if plot:
		if fmt is None:
			fmt = 'ro'
		errorbar(xc, yval, xerr=dx, yerr=dy, fmt=fmt)
	ok = nn != 0
	if cutbad:
		if others is None:
			return xc[ok], yval[ok], dx[ok], dy[ok], others
		else:
			return xc[ok], yval[ok], dx[ok], dy[ok], others[ok, :]
	else:
		yval[~ok] = 0
		dy[~ok] = 0
		return xc, yval, dx, dy, others


def exponential_filter1d(input, sigma, axis=-1, output=None, mode="reflect", cval=0.0, truncate=10.0, power=1):
	"""
	One-dimensional Exponential filter.

	Parameters
	----------
	input
	sigma : scalar
		Tau of exponential kernel
	axis : int, optional
		The axis of input along which to calculate. 
		Default is -1.

	output : array or dtype, optional
		The array in which to place the output, or the dtype of the returned array. 
		By default an array of the same dtype as input will be created.
		
	mode : {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}, optional
		The mode parameter determines how the input array is extended beyond its boundaries. 
		Default is ‘reflect’.
	cval : scalar, optional
		Value to fill past edges of input if mode is ‘constant’. Default is 0.0.
	truncate : float
		Truncate the filter at this many standard deviations.
		Default is 4.0.

	"""
	sd = float(sigma)
	# make the radius of the filter equal to truncate standard deviations
	lw = int(truncate * sd + 0.5)
	weights = [0.0] * (2 * lw + 1)
	weights[lw] = 1.0
	sum = 1.0
	# calculate the kernel:
	for ii in range(1, lw + 1):
		tmp = math.exp(-(float(ii) / sd) ** power)
		weights[lw + ii] = tmp * 0
		weights[lw - ii] = tmp
		sum += tmp
	for ii in range(2 * lw + 1):
		weights[ii] /= sum
	return correlate1d(input, weights, axis, output, mode, cval, 0)


def qs2array(file, FREQ_SAMPLING, timerange=None):
	"""
	Loads qubic instance to create 'dd' which is TOD for each TES
	Also normalises raw data
	Also returns 'time' which is linear time array
	Can also specify a timerange

	Parameters
	----------
	file : fits file
		File containing data.
	FREQ_SAMPLING
	timerange : array
		Time range, low and high boundaries

	Returns
	-------

	"""
	a = qp()
	a.read_fits(file)
	npix = a.NPIXELS
	nsamples = len(a.timeline(TES=1))
	dd = np.zeros((npix, nsamples))
	##### Normalisation en courant
	# Rfb=100e3 #changing to read from pystudio dictionary
	Rfb = a.Rfeedback
	NbSamplesPerSum = 64.  # this could also be a.NPIXELS_sampled
	gain = 1. / 2. ** 7 * 20. / 2. ** 16 / (NbSamplesPerSum * Rfb)

	for i in range(npix):
		dd[i, :] = a.timeline(TES=i + 1)
		dd[i, :] = gain * dd[i, :]

	time = np.arange(nsamples) / FREQ_SAMPLING

	if timerange is not None:
		print('Selecting time range: {} to {} sec'.format(timerange[0], timerange[1]))
		oktime = (time >= timerange[0]) & (time <= timerange[1])
		time = time[oktime]
		dd = dd[:, oktime]

	return time, dd, a


def read_hkintern(basedir, thefieldname=None):
	"""

	Parameters
	----------
	basedir : str
		directory of the file
	thefieldname : str

	Returns
	-------
	newdate : array
		New time array of te measurement
	hk : array
		Angle position given by the encoder (number of encoder steps).
	"""
	hkinternfile = glob.glob(basedir + '/Hks/hk-intern*')
	hk = fits.open(hkinternfile[0])
	nfields = hk[1].header['TFIELDS']
	fields = {}
	for idx in range(nfields):
		fieldno = idx + 1
		ttype = 'TTYPE%i' % fieldno
		fieldname = hk[1].header[ttype]
		fields[fieldname] = fieldno

	if thefieldname is None:
		print('List of available fields:')
		print('-------------------------')
		for idx in range(nfields):
			print(fields.keys()[idx])
		return None
	else:
		gpsdate = hk[1].data.field(fields['GPSDate'] - 1)  # in ms
		pps = hk[1].data.field(fields['Platform-PPS'] - 1)  # 0 and 1
		pps[0] = 1
		gpsdate[0] -= 1000
		ppson = pps == 1
		indices = np.arange(len(gpsdate))
		newdate = np.interp(indices, indices[ppson], gpsdate[ppson] + 1000) * 1e-3

		# read the azimuth position
		fieldno = fields[thefieldname]
		hk = hk[1].data.field(fieldno - 1)
		return newdate, hk


def butter_bandpass(lowcut, highcut, fs, order=5):
	"""

	Parameters
	----------
	lowcut : scalar
	highcut : scalar
	fs :
	order : int
		order of the filter

	Returns
	-------
	b, a : (ndarray, ndarray)
		Numerator (b) and denominator (a) polynomials of the IIR filter.

	"""
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = scsig.butter(order, [low, high], btype='band', output='ba')
	return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	"""

	Parameters
	----------
	data : array like
	lowcut
	highcut
	fs
	order

	Returns
	-------
	y : array
		The output of the digital filter.

	"""
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = scsig.lfilter(b, a, data)
	return y


def notch_array(freqs, bw):
	"""
	Returns array to be used with notch_filter

	Parameters
	----------
	freqs : list
		Frequencies to filter
	bw : scalar
		The filter bandwidth

	Returns
	-------

	"""
	notch = []

	for i in range(len(freqs)):
		notch.append([freqs[i], bw * (1 + i)])

	return notch


def notch_filter(data, f0, bw, fs):
	"""

	Parameters
	----------
	data
	f0
	bw
	fs

	Returns
	-------
	y : array
		Output of the notch filter
	"""
	Q = f0 / bw
	b, a = scsig.iirnotch(f0 / fs * 2, Q)
	y = scsig.lfilter(b, a, data)
	return y


def meancut(data, nsig, med=False, disp=True):
	"""
	Parameters
	----------
	data: array like
	nsig: float
		Lower and upper bound factor of sigma clipping.
	med: bool
		If True, perform the median and not the mean.
	disp: bool
		If True, return the dispersion (STD),
		if False, return the error on the mean (STD/sqrt(N))
	Returns
	-------
	The mean/median and the dispersion/error.

	"""
	dd = data.copy()
	for i in range(10):
		dd, mini, maxi = scipy.stats.sigmaclip(dd, low=nsig, high=nsig)
	if disp:
		sc = 1
	else:
		sc = np.sqrt(len(dd))
	if med:
		return np.median(dd), np.std(dd) / sc
	else:
		return np.mean(dd), np.std(dd) / sc


def weighted_mean(x, dx, dispersion=True):
	"""
	Calculated the weighted mean of data, errors
	If dispersion is True (default) the error on the mean comes from the RMS of the data, otherwise
	the error on the weighted mean is analytically calculated from input errors
	"""
	w = 1. / dx ** 2
	sumw = np.sum(w)
	mm = np.sum(w * x) / sumw
	if dispersion:
		ss = np.std(x)
	else:
		ss = 1. / np.sqrt(sumw)
	return mm, ss


def simsig(x, pars, extra_args=None):
	"""

	Parameters
	----------
	x : list
	pars : list
		List with 4 parameters: cycle, ctime, initial time, amplitude

	Returns
	-------
	A simulated signal.

	"""
	dx = x[1] - x[0]
	cycle = np.nan_to_num(pars[0])
	ctime = np.nan_to_num(pars[1])
	t0 = np.nan_to_num(pars[2])
	amp = np.nan_to_num(pars[3])
	sim_init = np.zeros(len(x))
	ok = x < (cycle * (np.max(x)))
	sim_init[ok] = 1.

	# Add a phase
	sim_init_shift = np.interp((x - t0) % max(x), x, sim_init)

	# Convolved by a filter
	# thesim = -1 * gaussian_filter1d(sim_init_shift, ctime, mode='wrap')
	thesim = -1 * exponential_filter1d(sim_init_shift, ctime / dx, mode='wrap')

	# Normalization
	thesim = (thesim - np.mean(thesim)) / np.std(thesim) * amp

	return np.nan_to_num(thesim)


def simsig_nonorm(x, pars):
	"""
	Same as simsig but without normalisation.
	"""
	dx = x[1] - x[0]
	cycle = np.nan_to_num(pars[0])
	ctime = np.nan_to_num(pars[1])
	t0 = np.nan_to_num(pars[2])
	amp = np.nan_to_num(pars[3])
	sim_init = np.zeros(len(x))
	ok = x < (cycle * (np.max(x)))
	sim_init[ok] = amp

	# Add a phase
	sim_init_shift = np.interp((x - t0) % max(x), x, sim_init)

	# Convolved by a filter
	thesim = -1 * exponential_filter1d(sim_init_shift, ctime / dx, mode='wrap')

	# Center the signal
	thesim = (thesim - np.mean(thesim))
	return thesim


def simsig_asym(x, pars, extra_args=None):
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


def simsig_fringes(time, stable_time, params):
	"""
	Simulate a TOD signal obtained during the fringe measurement.
	This function was done to make a fit.
	Parameters
	----------
	time : array
		Time sampling.
	stable_time: float
		Stable time [s] on each step.
	params : list
		ctime, starting time, the 6 amplitudes.

	Returns
	-------
	The simulated signal.

	"""
	dt = time[1] - time[0]
	tf = time[-1]
	npoints = len(time)

	ctime = params[0]
	t0 = params[1]
	amp = params[2:]

	sim_init = np.zeros(npoints)

	for i in range(6):
		a = int(npoints / tf * stable_time * i)
		b = int((stable_time * i + stable_time) * npoints / tf)
		sim_init[a: b] = amp[i]

	# Add a phase
	sim_init_shift = np.interp((time - t0) % max(time), time, sim_init)

	# Convolved by an exponential filter
	thesim = exponential_filter1d(sim_init_shift, ctime / dt, mode='wrap')

	return np.array(thesim).astype(np.float64)


def fold_data(time, dd, period, lowcut, highcut, nbins,
			  notch=None, rebin=None,
			  median=False, mode=False, clip=None,
			  return_error=False,
			  return_noise_harmonics=None,
			  silent=False, verbose=None):
	"""

	Parameters
	----------
	time : array
	dd : array
		Data signal.
	period : float
		Data will be folded on this period.
	lowcut : float
		Low cut for the band filter.
	highcut : float
		High cut for the band filter.
	nbins
	notch
	return_error
	"""
	tfold = time % period
	FREQ_SAMPLING = 1. / (time[1] - time[0])
	sh = np.shape(dd)
	ndet = sh[0]

	if return_noise_harmonics is not None:
		# We estimate the noise in between the harmonics of the signal between harm=1 and
		# nharm=return_noise_harmonics
		# First we find the corresponding frequencies, below we measure the noise
		nharm = return_noise_harmonics
		margin = 0.2
		fmin = np.zeros(nharm)
		fmax = np.zeros(nharm)
		fnoise = np.zeros(nharm)
		noise = np.zeros((ndet, nharm))
		for i in range(nharm):
			fmin[i] = 1. / period * (i + 1) * (1 + margin / (i + 1))
			fmax[i] = 1. / period * (i + 2) * (1 - margin / (i + 1))
			fnoise[i] = 0.5 * (fmin[i] + fmax[i])

	folded = np.zeros((ndet, nbins))
	folded_nonorm = np.zeros((ndet, nbins))
	dfolded = np.zeros((ndet, nbins))
	dfolded_nonorm = np.zeros((ndet, nbins))
	if not silent:
		bar = progress_bar(ndet, 'Detectors ')
	for THEPIX in range(ndet):
		if not silent:
			bar.update()
		data = dd[THEPIX, :]
		newdata = filter_data(time, data, lowcut, highcut, notch=notch, rebin=rebin, verbose=verbose)
		t, yy, dx, dy, others = profile(tfold, newdata,
										nbins=nbins, dispersion=False, plot=False,
										cutbad=False, median=median, mode=mode, clip=clip)
		folded[THEPIX, :] = (yy - np.mean(yy)) / np.std(yy)
		folded_nonorm[THEPIX, :] = (yy - np.mean(yy))
		dfolded[THEPIX, :] = dy / np.std(yy)
		dfolded_nonorm[THEPIX, :] = dy
		if return_noise_harmonics is not None:
			spectrum, freq = power_spectrum(time, newdata, rebin=True)
			for i in range(nharm):
				ok = (freq >= fmin[i]) & (freq < fmax[i])
				noise[THEPIX, i] = np.sqrt(np.mean(spectrum[ok]))

	if return_error:
		if return_noise_harmonics is not None:
			return folded, t, folded_nonorm, dfolded, dfolded_nonorm, newdata, fnoise, noise
		else:
			return folded, t, folded_nonorm, dfolded, dfolded_nonorm, newdata
	else:
		if return_noise_harmonics is not None:
			return folded, t, folded_nonorm, newdata, fnoise, noise
		else:
			return folded, t, folded_nonorm, newdata


def power_spectrum(time_in, data_in, rebin=True):
	if rebin:
		### Resample the data on a regular grid
		time = np.linspace(time_in[0], time_in[-1], len(time_in))
		data = np.interp(time, time_in, data_in)
	else:
		time = time_in
		data = data_in

	spectrum_f, freq_f = mlab.psd(data, Fs=1. / (time[1] - time[0]), NFFT=len(data), window=mlab.window_hanning)
	return spectrum_f, freq_f


def filter_data(time_in, data_in, lowcut, highcut, rebin=True, verbose=False, notch=None, order=5):
	sh = np.shape(data_in)
	if rebin:
		if verbose: printnow('Rebinning before Filtering')
		### Resample the data on a regular grid
		time = np.linspace(time_in[0], time_in[-1], len(time_in))
		if len(sh) == 1:
			data = np.interp(time, time_in, data_in)
		else:
			data = vec_interp(time, time_in, data_in)
	else:
		if verbose: printnow('No rebinning before Filtering')
		time = time_in
		data = data_in

	FREQ_SAMPLING = 1. / ((np.max(time) - np.min(time)) / len(time))
	filt = scsig.butter(order, [2 * lowcut / FREQ_SAMPLING, 2 * highcut / FREQ_SAMPLING], btype='bandpass',
						output='sos')
	if len(sh) == 1:
		dataf = scsig.sosfilt(filt, data)
	else:
		dataf = scsig.sosfilt(filt, data, axis=1)

	if notch is not None:
		for i in range(len(notch)):
			ftocut = notch[i][0]
			bw = notch[i][1]
			nharmonics = notch[i][2].astype(int)
			if verbose: print('Notching {} Hz with width {} and {} harmonics'.format(ftocut, bw, nharmonics))
			for j in range(nharmonics):
				dataf = notch_filter(dataf, ftocut * (j + 1), bw, FREQ_SAMPLING)

	return dataf


def vec_interp(x, xin, yin):
	sh = np.shape(yin)
	nvec = sh[0]
	yout = np.zeros_like(yin)
	for i in range(nvec):
		yout[i, :] = np.interp(x, xin, yin[i, :])
	return yout


def fit_average(t, folded, fff, dc, fib, Vtes, initpars=None, fixpars=[0, 0, 0, 0], doplot=True, functname=simsig,
				clear=True, name='fib'):
	"""

	Parameters
	----------
	t
	folded
	fff : float
		Modulation frequency of the external source.
	dc : float
		Duty cycle of the modulation.
	fib
	Vtes
	initpars
	fixpars
	doplot : bool
		If true make the plot.
	functname
	clear : bool
		If true, clear the window before plotting.

	name

	Returns
	-------

	"""
	sh = np.shape(folded)
	npix = sh[0]
	nbins = sh[1]
	####### Average folded data
	av = np.median(np.nan_to_num(folded), axis=0)

	if initpars is None:
		# derivatives = np.gradient(av)
		# src_on = np.min()
		# try to detect the start time

		nnn = 100
		t0 = np.linspace(0, 1. / fff, nnn)
		diff2 = np.zeros(nnn)
		for i in range(nnn):
			diff2[i] = np.sum((av - functname(t, [dc, 0.1, t0[i], 1.])) ** 2)
		ttry = t0[np.argmin(diff2)]

		bla = do_minuit(t, av, np.ones(len(t)), [dc, 0.1, ttry, 1.], functname=functname,
						rangepars=[[0., 1.], [0., 0.2], [0., 1. / fff], [0., 20.]], fixpars=[1, 1, 0, 1],
						force_chi2_ndf=True, verbose=True, nohesse=True)
		initpars = [dc, 0.1, bla[1][2], 1.]
	# ion()
	# clf()
	# xlim(0,1./fff)
	# plot(t,av,color='b',lw=4,alpha=0.3, label='Median')
	# plot(t, functname(t, bla[1]), 'r--',lw=4)
	# show()

	####### Fit
	bla = do_minuit(t, av, np.ones(len(t)), initpars, functname=functname,
					rangepars=[[0., 1.], [0., 0.5], [0., 1. / fff], [0., 2.]], fixpars=fixpars, force_chi2_ndf=True,
					verbose=False, nohesse=True)
	params_av = bla[1]
	err_av = bla[2]

	if doplot:
		ion()
		if clear:
			clf()
		xlim(0, 1. / fff)
		for i in range(npix):
			plot(t, folded[i, :], alpha=0.1, color='k')
		plot(t, av, color='b', lw=4, alpha=0.3, label='Median')
		plot(t, functname(t, bla[1]), 'r--', lw=4,
			 label='Fitted average of {8:} pixels \n cycle={0:8.3f}+/-{1:8.3f} \n tau = {2:8.3f}+/-{3:8.3f}s \n t0 = {4:8.3f}+/-{5:8.3f}s \n amp = {6:8.3f}+/-{7:8.3f}'.format(
				 params_av[0], err_av[0], params_av[1], err_av[1], params_av[2], err_av[2], params_av[3], err_av[3],
				 npix))
		legend(fontsize=7, frameon=False, loc='lower left')
		xlabel('Time(sec)')
		ylabel('Stacked')
		title('{} {}: Freq_Mod={}Hz - Cycle={}% - Vtes={}V'.format(name, fib, fff, dc * 100, Vtes))
		show()
		time.sleep(0.1)
	return av, params_av, err_av


def fit_all(t, folded, av, initpars=None, fixpars=[0, 0, 0, 0],
			stop_each=False, functname=simsig, rangepars=None):
	"""

	Parameters
	----------
	t
	folded
	av
	initpars
	fixpars
	stop_each
	functname
	rangepars

	Returns
	-------

	"""
	npix, nbins = np.shape(folded)
	print('       Got {} pixels to fit'.format(npix))
	##### Now fit each TES fixing cycle to dc and t0 to the one fitted on the median
	allparams = np.zeros((npix, 4))
	allerr = np.zeros((npix, 4))
	allchi2 = np.zeros(npix)
	bar = progress_bar(npix, 'Detectors ')
	ok = np.zeros(npix, dtype=bool)
	for i in range(npix):
		bar.update()
		thedd = folded[i, :]
		#### First a fit with no error correction in order to have a chi2 distribution
		theres = do_minuit(t, thedd, np.ones(len(t)), initpars, functname=functname, fixpars=fixpars,
						   rangepars=rangepars, force_chi2_ndf=True, verbose=False, nohesse=True)
		ndf = theres[5]
		params = theres[1]
		err = theres[2]
		allparams[i, :] = params
		allerr[i, :] = err
		allchi2[i] = theres[4]
		if stop_each:
			clf()
			plot(t, thedd, color='k')
			plot(t, av, color='b', lw=4, alpha=0.2, label='Median')
			plot(t, functname(t, theres[1]), 'r--', lw=4,
				 label='Fitted: \n cycle={0:8.3f}+/-{1:8.3f} \n tau = {2:8.3f}+/-{3:8.3f}s \n t0 = {4:8.3f}+/-{5:8.3f}s \n amp = {6:8.3f}+/-{7:8.3f}'.format(
					 params[0], err[0], params[1], err[1], params[2], err[2], params[3], err[3]))
			legend()
			show()
			msg = 'TES #{}'.format(i)
			if i in [3, 35, 67, 99]:
				msg = 'Channel #{} - BEWARE THIS IS A THERMOMETER !'.format(i)
			title(msg)
			# Changing so 'i' select prompts plot inversion
			bla = raw_input("Press [y] if fit OK, [i] to invert, other key otherwise...")
			if bla == 'y':
				ok[i] = True
			# invert to check if TES okay,
			# thedd refers to the indexed TES in loop
			if bla == 'i':
				# clf()
				plot(t, thedd * (-1.0), color='olive')
				show()
				ibla = raw_input("Press [y] if INVERTED fit OK, otherwise anykey")
				# and invert thedd in the original datset
				if ibla == 'y':
					ok[i] = True
					folded[i, :] = thedd * -1.0

			print(ok[i])
	return allparams, allerr, allchi2, ndf, ok


def run_asic(idnum, Vtes, fff, dc, theasicfile, asic, reselect_ok=False, lowcut=0.5, highcut=15., nbins=50,
			 nointeractive=False, doplot=True, notch=None, lastpassallfree=False, name='fib', okfile=None,
			 initpars=None, timerange=None, removesat=False, stop_each=False, rangepars=None):
	"""

	Parameters
	----------
	idnum
	Vtes
	fff : float
		Modulation frequency of the external source
	dc : float
		Duty cycle of the modulation
	theasicfile
	asic
	reselect_ok : bool
		If true, you will select the good TES one by one, if False, you will use the file created before.
	lowcut
	highcut
	nbins
	nointeractive
	doplot
	notch
	lastpassallfree
	name
	okfile
	initpars
	timerange
	removesat
	stop_each
	rangepars

	Returns
	-------

	"""
	fib = idnum
	### Read data
	# GOING TO TEST PASSING IN THESE VARS
	FREQ_SAMPLING = (2e6 / 128 / 100)
	time, dd, a = qs2array(theasicfile, FREQ_SAMPLING, timerange=timerange)
	ndet, nsamples = np.shape(dd)

	### Fold the data at the modulation period of the fibers
	### Signal is also badpass filtered before folding
	folded, tt, folded_nonorm = fold_data(time, dd, 1. / fff, lowcut, highcut, nbins, notch=notch)

	if nointeractive:
		reselect_ok = False
		answer = 'n'
	else:
		if reselect_ok:
			print('\n\n')
			answer = raw_input('This will overwrite the file for OK TES. Are you sure you want to proceed [y/n]')
		else:
			answer = 'n'

	if answer == 'y':
		print('Now going to reselect the OK TES and overwrite the corresponding file')
		#### Pass 1 - allows to obtain good values for t0 basically
		#### Now perform the fit on the median folded data
		print('')
		print('FIRST PASS')
		print('First Pass is only to have a good guess of the t0, '
			  'your selection should be very conservative - only high S/N')
		# if initpars == Noy
		# ne:
		# 	initpars = [dc, 0.06, 0., 0.6]
		av, params, err = fit_average(tt, folded, fff, dc, fib, Vtes, initpars=initpars, fixpars=[0, 0, 0, 0],
									  doplot=True, name=name)

		#### And the fit on all data with this as a first guess forcing some parameters
		#### it returns the list of OK detectorsy
		allparams, allerr, allchi2, ndf, ok = fit_all(tt, folded, av, initpars=[dc, params[1], params[2], params[3]],
													  fixpars=[1, 0, 1, 0], rangepars=rangepars, stop_each=True)

		#### Pass 2
		#### Refit with only the above selected ones in order to have good t0
		#### Refit the median of the OK detectors
		print('')
		print('SECOND PASS')
		print('Second pass is the final one, please select the pixels that seem OK')
		av, params, err = fit_average(tt, folded[ok, :], fff, dc, fib, Vtes, initpars=initpars, fixpars=[0, 0, 0, 0],
									  doplot=True, name=name)

		#### And the fit on all data with this as a first guess forcing some parameters
		#### it returns the list of OK detectors
		allparams, allerr, allchi2, ndf, ok = fit_all(tt, folded, av, initpars=[dc, params[1], params[2], params[3]],
													  rangepars=rangepars, fixpars=[1, 0, 1, 0], stop_each=True)

		#### Final Pass
		#### The refit them all with only tau and amp as free parameters
		#### also do not normalize amplitudes of folded
		allparams, allerr, allchi2, ndf, ok_useless = fit_all(tt, folded_nonorm * 1e9, av,
															  initpars=[dc, params[1], params[2], params[3]],
															  rangepars=rangepars, fixpars=[1, 0, 1, 0],
															  functname=simsig_nonorm)

		okfinal = ok * (allparams[:, 1] < 1.)
		### Make sure no thermometer is included
		okfinal[[3, 35, 67, 99]] = False
		# Save the list of OK bolometers
		if okfile is None:
			FitsArray(okfinal.astype(int)).save('TES-OK-{}{}-asic{}.fits'.format(name, fib, asic))
		else:
			FitsArray(okfinal.astype(int)).save(okfile)
	else:
		# if initpars is None:
		# 	initpars = [dc, 0.06, 0., 0.6]
		if okfile is None:
			okfinal = np.array(FitsArray('TES-OK-{}{}-asic{}.fits'.format(name, fib, asic))).astype(bool)
		else:
			okfinal = np.array(FitsArray(okfile)).astype(bool)
		if removesat:
			#### remove pixels looking saturated
			saturated = (np.min(folded_nonorm, axis=1) < removesat)
			okfinal = (okfinal * ~saturated).astype(bool)

	if doplot is False:
		### Now redo the fits one last time
		av, params, err = fit_average(tt, folded[okfinal, :], fff, dc, fib, Vtes, initpars=initpars,
									  fixpars=[0, 0, 0, 0], doplot=False, clear=False, name=name)

		allparams, allerr, allchi2, ndf, ok_useless = fit_all(tt, folded_nonorm * 1e9, av,
															  initpars=[dc, params[1], params[2], params[3]],
															  fixpars=[1, 0, 1, 0], functname=simsig_nonorm,
															  rangepars=rangepars)
	else:
		figure(figsize=(6, 8))
		subplot(3, 1, 1)
		### Now redo the fits one last time
		av, params, err = fit_average(tt, folded[okfinal, :], fff, dc, fib, Vtes, initpars=initpars,
									  fixpars=[0, 0, 0, 0], doplot=True, clear=False, name=name)
		print(params)
		print(err)

		if lastpassallfree:
			fixed = [0, 0, 0, 0]
		else:
			fixed = [1, 0, 1, 0]
		allparams, allerr, allchi2, ndf, ok_useless = fit_all(tt, folded_nonorm * 1e9, av,
															  initpars=[dc, params[1], params[2], params[3]],
															  fixpars=fixed, functname=simsig_nonorm,
															  stop_each=stop_each, rangepars=rangepars)

		subplot(3, 2, 3)
		mmt, sst = meancut(allparams[okfinal, 1], 3)
		hist(allparams[okfinal, 1], range=[0, mmt + 4 * sst], bins=10, label=statstr(allparams[okfinal, 1], cut=3))
		xlabel('Tau [sec]')
		legend()
		title('Asic {} - {} {}'.format(name, asic, fib))
		subplot(3, 2, 4)
		mma, ssa = meancut(allparams[okfinal, 3], 3)
		hist(allparams[okfinal, 3], range=[0, mma + 4 * ssa], bins=10, label=statstr(allparams[okfinal, 3], cut=3))
		legend()
		xlabel('Amp [nA]')

		pars = allparams
		tau = pars[:, 1]
		tau[~okfinal] = np.nan
		amp = pars[:, 3]
		amp[~okfinal] = np.nan

		if asic == 1:
			tau1 = tau
			tau2 = None
			amp1 = amp
			amp2 = None
		else:
			tau1 = None
			tau2 = tau
			amp1 = None
			amp2 = amp

		subplot(3, 2, 5)
		imtau = image_asics(data1=tau1, data2=tau2)
		imshow(imtau, vmin=0, vmax=mmt + 4 * sst, cmap='viridis', interpolation='nearest')
		title('Tau - {} {} - asic {}'.format(name, fib, asic))
		colorbar()
		subplot(3, 2, 6)
		imamp = image_asics(data1=amp1, data2=amp2)
		imshow(imamp, vmin=0, vmax=mma + 6 * ssa, cmap='viridis', interpolation='nearest')
		colorbar()
		title('Amp - {} {} - asic {}'.format(name, fib, asic))
		tight_layout()

	return tt, folded, okfinal, allparams, allerr, allchi2, ndf


def calibrate(fib, pow_maynooth, allparams, allerr, allok, cutparam=None, cuterr=None, bootstrap=None):
	"""

	Parameters
	----------
	fib
	pow_maynooth
	allparams
	allerr
	allok
	cutparam
	cuterr
	bootstrap

	Returns
	-------

	"""
	img_maynooth = image_asics(all1=pow_maynooth)

	clf()
	subplot(2, 2, 1)
	plot(allparams[allok, 3], allerr[allok, 3], 'k.')
	if cuterr is not None:
		thecut_err = cuterr
	else:
		thecut_err = 1e10
	if cutparam is not None:
		thecut_amp = cutparam
	else:
		thecut_amp = 1e10

	newok = allok * (allerr[:, 3] < thecut_err) * (allparams[:, 3] < thecut_amp)
	plot([np.min(allparams[allok, 3]), np.max(allparams[allok, 3])], [thecut_err, thecut_err], 'g--')
	plot([thecut_amp, thecut_amp], [np.min(allerr[allok, 3]), np.max(allerr[allok, 3])], 'g--')
	plot(allparams[newok, 3], allerr[newok, 3], 'r.')
	allparams[~newok, :] = np.nan
	ylabel('$\sigma_{amp}$ [nA]')
	xlabel('Amp Fib{} [nA]'.format(fib))

	subplot(2, 2, 3)
	errorbar(pow_maynooth[newok], allparams[newok, 3], yerr=allerr[newok, 3], fmt='r.')
	xx = pow_maynooth[newok]
	yy = allparams[newok, 3]
	yyerr = allerr[newok, 3]
	res = do_minuit(xx, yy, yyerr, np.array([1., 0]), fixpars=[0, 0])
	paramfit = res[1]
	if bootstrap is None:
		errfit = res[2]
		typerr = 'Minuit'
	else:
		bsres = []
		bar = progress_bar(bootstrap, 'Bootstrap')
		for i in range(bootstrap):
			bar.update()
			order = np.argsort(np.random.rand(len(xx)))
			xxbs = xx.copy()
			yybs = yy[order]
			yybserr = yyerr[order]
			theres = do_minuit(xxbs, yybs, yybserr, np.array([1., 0]), fixpars=[0, 0], verbose=False)
			bsres.append(theres[1])
		bsres = np.array(bsres)
		errfit = np.std(bsres, axis=0)
		typerr = 'Bootstrap'

	xxx = np.linspace(0, np.max(pow_maynooth), 100)
	plot(xxx, thepolynomial(xxx, res[1]), 'g', lw=3,
		 label='a={0:8.3f} +/- {1:8.3f} \n b={2:8.3f} +/- {3:8.3f}'.format(paramfit[0], errfit[0], paramfit[1],
																		   errfit[1]))
	if bootstrap is not None:
		bsdata = np.zeros((bootstrap, len(xxx)))
		for i in range(bootstrap):
			bsdata[i, :] = thepolynomial(xxx, bsres[i, :])
		mm = np.mean(bsdata, axis=0)
		ss = np.std(bsdata, axis=0)
		fill_between(xxx, mm - ss, y2=mm + ss, color='b', alpha=0.3)
		fill_between(xxx, mm - 2 * ss, y2=mm + 2 * ss, color='b', alpha=0.2)
		fill_between(xxx, mm - 3 * ss, y2=mm + 3 * ss, color='b', alpha=0.1)
		plot(xxx, mm, 'b', label='Mean bootstrap')

	# indices = np.argsort(np.random.rand(bootstrap))[0:1000]
	# for i in range(len(indices)):
	# 	plot(xxx, thepolynomial(xxx, bsres[indices[i],:]), 'k', alpha=0.01)
	ylim(0, np.max(allparams[newok, 3]) * 1.1)
	xlim(np.min(pow_maynooth[newok]) * 0.99, np.max(pow_maynooth[newok]) * 1.01)
	ylabel('Amp Fib{} [nA]'.format(fib))
	xlabel('Maynooth [mW]')
	legend(fontsize=8, framealpha=0.5)

	subplot(2, 2, 2)
	imshow(img_maynooth, vmin=np.min(pow_maynooth), vmax=np.max(pow_maynooth), interpolation='nearest')
	colorbar()
	title('Maynooth [mW]')

	subplot(2, 2, 4)
	img = image_asics(all1=allparams[:, 3] / res[1][0])
	imshow(img, interpolation='nearest')
	colorbar()
	title('Amp Fib{}  converted to mW'.format(fib))
	tight_layout()

	return res[1], res[2], newok
