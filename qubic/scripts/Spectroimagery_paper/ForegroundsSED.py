# -*- coding: utf-8 -*-
#
# Author: Martín M. Gamboa Lerena.
# Date: Feb 15th 2021
#

import os
import sys
import glob
from importlib import reload
import gc
import time
# Specific science modules
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pickle 
import astropy.io as fits
from lmfit import Model

# Specific qubic modules
import qubic
from qubicpack.utilities import Qubic_DataDir
from qubic import QubicSkySim as qss
from qubic.polyacquisition import compute_freq
from qubic import ReadMC as rmc
from qubic import create_acquisition_operator_REC
from pysimulators import FitsArray

from scipy.optimize import curve_fit
import scipy.constants
from qubic import mcmc
import qubic.AnalysisMC as amc

plt.rc('text',usetex=False)
plt.rc('font', size=16)
from mpl_toolkits.mplot3d import Axes3D
import qubic.AnalysisMC as amc

# This module assumes QubicDictionary is prepared from script.

def make_covTD(d):
	"""
	Usually coverage map is provided in a separate file. But if not the case, this method can compute a coverage map
	Parameters:
	d: Qubic dictionary
	Return:
	cov: coverage map in a.QubicMultibandAcquisition shape (nfreq, npix).
	"""

	pointing = qubic.get_pointing(d)
	q= qubic.QubicMultibandInstrument(d)
	s= qubic.QubicScene(d)
	nf_sub_rec = d['nf_recon']
	_, nus_edge, nus, _, _, _ = qubic.compute_freq(d['filter_nu'] / 1e9, 
												   nf_sub_rec, d['filter_relative_bandwidth'])
	arec = qubic.QubicMultibandAcquisition(q, pointing, s, d, nus_edge)
	cov = arec.get_coverage()
	return cov

def coverage(dictionaries, regions, bands, makeCov = False, filename = None):
	"""
	Read or make the coverage maps for the bands and regions 

	Assumes one config: FI or TD (info will be read it from dictionatries[0])
	Parameters: 
		dictionaries:
			array of Qubic dictionaries
		regions: 
			sky regions where coverage will be computed or read
		bands: 
			150 or 220GHz
		filename:
			Array of filenames of coverages maps. #filenames = #regions + #bands. Default: read this 4 files 
			["doc/FastSimulator/Data/DataFastSimulator_FI150Q_coverage.fits'",
			 "doc/FastSimulator/Data/DataFastSimulator_FI220Q_coverage.fits'",
			 "doc/FastSimulator/Data/DataFastSimulator_FI150G_coverage.fits'",
			 "doc/FastSimulator/Data/DataFastSimulator_FI220G_coverage.fits'" ]
	Return: 
		coveragesmaps:  
			maps with format priority: region (in regions) and then band (in bands ascendent order). 
			It take the first letter in regions to use in name when reading maps from files.
			Shape: (len(regions) +len(bands), npix)
	"""


	#regions = ['Qubic_field', 'GalCen_field']
	#bands = ['150', '220']

	global_dir = Qubic_DataDir(datafile='instrument.py', datadir=os.environ['QUBIC_DATADIR'])
	if filename == None:
		import itertools
		config = dictionaries[0]['config']
		filename = [global_dir + '/doc/FastSimulator/Data/DataFastSimulator_{}{}{}_coverage.fits'.format(config, jband, ireg[0]) \
		for ireg, jband in itertools.product(regions, bands)]
	else:
		filename = filename

	coveragesmaps = np.zeros((len(regions) * len(bands), 12 * dictionaries[0]['nside'] ** 2, ))
	if makeCov:
		#Save coverage maps with format priority: region (in regions) and then band (in bands ascendent order) 
		cov = np.shape((len(regions)+len(bands), 
						dictionaries[0]['nf_recon'], 
						12 * dictionaries[0]['nside'] ** 2))
		for jr, region in enumerate(regions):
			for jb, band in enumerate(bands):
				index = len(bands) * jr + jb
				cov[index] = make_cov[dictionaries[index]]
				coveragesmaps[index] = np.sum(cov[index], axis = 0)  # Average the bands
				coveragesmaps[index] /= np.max(coveragesmaps[index]) # Normalize by the max
	else:
		for jr, region in enumerate(regions):
			for jb, band in enumerate(bands):
				index = len(bands) * jr + jb
				#print(index, len(coveragesmaps), len(regions), len(bands))
				coveragesmaps[index] = FitsArray(filename[index])

	return coveragesmaps

def _plot_covs(regions, bands, coveragesmaps, centers, config = "FI", plotdif = False):
	
	lacarte = np.zeros((len(coveragesmaps), 200, 200))
	X,Y=np.meshgrid(range(200), range(200))

	fig = plt.figure(figsize=(18,6))
	for ic, icov in enumerate(coveragesmaps):
		lacarte[ic] = hp.gnomview(icov, rot = centers[ic], reso = 13, 
							 return_projected_map = True, no_plot = True)
		
		ax = fig.add_subplot(1, len(coveragesmaps), ic + 1, projection='3d')
		ax.set_title('cov{} {}-patch {}'.format(bands[ic//2], regions[ic//2][0], config), fontsize=16)
		ax.plot_surface(X,Y, lacarte[ic], cmap=plt.cm.viridis, linewidth=0.2)

	plt.show()
	
	if plotdif:
		fig = plt.figure(figsize=(20,5))
		ax = fig.add_subplot(1, len(coveragesmaps), 1, projection='3d')
		ax.set_title('cov 150 Q - GC', fontsize=16)
		ax.plot_surface(X,Y, lacarte[0]-lacarte[2], cmap=plt.cm.viridis, linewidth=0.2)

		ax = fig.add_subplot(1, len(coveragesmaps), 2, projection='3d')
		ax.set_title('cov 220 Q - GC', fontsize=16)
		ax.plot_surface(X,Y, lacarte[1]-lacarte[3], cmap=plt.cm.viridis, linewidth=0.2)

		ax = fig.add_subplot(1, len(coveragesmaps), 3, projection='3d')
		ax.set_title('cov 150 Q - 220 Q', fontsize=16)
		ax.plot_surface(X,Y, lacarte[0] - lacarte[1], cmap=plt.cm.viridis, linewidth=0.2)

		ax = fig.add_subplot(1, len(coveragesmaps), 4, projection='3d')
		ax.set_title('cov 150 GC - 220 GC', fontsize=16)
		ax.plot_surface(X,Y, lacarte[2] - lacarte[3], cmap=plt.cm.viridis, linewidth=0.2)

		plt.show()

	return

def foreground_signal(dictionaries, sky_configuration, sky = 'T',
	seed = None, verbose = False):
	
	"""
	Averaging manually the maps into a band if nf_sub != nfrecon, otherwise, foreground are computed 
		in nf_recon sub-bands


	Assumes . dictionaries[:]['nf_sub'] == dictionaries[:]['nf_recon']
			. all regions and frequencies uses same sky configuration. 

	Parameters: 
		dictionaries: 
			array of dictionaries. #dicts = #regions + #bands_needed
		sky_configuration:
			dictionary with PySM format to build foregrounds
		sky: 
			'T' or 'P' for temperature or polarization study
		seed: 
			fix seed for simulations
	"""

	# Generate foregrounds
	##### QubicSkySim instanciation
	seed = seed

	QubicSkyObject = []
	foreground_maps = np.zeros((len(dictionaries), dictionaries[0]['nf_recon'], 12 * dictionaries[0]['nside'] ** 2, 3))
	for ic, idict in enumerate(dictionaries):

		QubicSkyObject.append(qss.Qubic_sky(sky_configuration, idict))

		if idict['nf_sub'] != idict['nf_recon']:
			_, nus_edge_in, nus_in, _, _, _ = qubic.compute_freq(idict['filter_nu'] / 1e9, 
														   idict['nf_sub'],
														   idict['filter_relative_bandwidth'])
			_, nus_edge_out, nus_out, _, _, _ = qubic.compute_freq(idict['filter_nu'] / 1e9,  
																   idict['nf_recon'],
																   idict['filter_relative_bandwidth'])
	
			if verbose: print('Computing maps averaging')
			# Generate convolved sky of dust without noise 
			dust_map_in = QubicSkyObject[ic].get_fullsky_convolved_maps(FWHMdeg = None, verbose = False)
			if verbose: print('=== Done {} map ===='.format(idict['filter_nu'] / 1e9, regions[ic//len(regions)][0]))
		
			for i in range(idict['nf_recon']):
				inband = (nus_in > nus_edge_out[i]) & (nus_in < nus_edge_out[i + 1])
				foreground_maps[ic, ...] = np.mean(dust_map_in[inband, ...], axis=0)    

		elif idict['nf_sub'] == idict['nf_recon']:
			# Now averaging maps into reconstruction sub-bands maps
			foreground_maps[ic] = QubicSkyObject[ic].get_fullsky_convolved_maps(FWHMdeg = None, verbose = False)

	return foreground_maps


def noise_qss(dictionaries, sky_configuration, coverages, realizations, verbose = False):
	
	"""
	Assume all dictionaries have the same 'effective_duration' and same 'nf_recon'


	"""

	##### Getting FastSimulator output maps
	noise = np.zeros((len(dictionaries), realizations, dictionaries[0]['nf_recon'], 12 * dictionaries[0]['nside']**2,3))

	QubicSkyObject = []
	for ic, idict in enumerate(dictionaries):

		QubicSkyObject.append(qss.Qubic_sky(sky_configuration, idict))
		if verbose: print(type(QubicSkyObject[ic]))

		for i in range(realizations):

			if verbose: print(type(coverages[ic]))
			
			noise[ic, i, ...], _ = \
				QubicSkyObject[ic].get_partial_sky_maps_withnoise(spatial_noise=False, coverage = coverages[ic], 
															noise_only = True, Nyears = idict['effective_duration'],
															verbose = verbose)        
			if verbose: print('=== Done interation #{} ===='.format(i+1))
	
	return noise

def foreground_with_noise(dictionaries, sky_configuration, regions, bands, realizations, sky = 'T',
	coverages = None, seed = None, verbose = False, ud_grade = False, nside_out = None):

	"""
	
	"""

	if ud_grade:
		if nside_out == None: 
			raise ValueError("You ask for ud_grade maps but nside_out is None. Please specify nside_out")

	fground_maps = foreground_signal(dictionaries, sky_configuration, verbose = verbose)
	
	if coverages == None:
		coverages = coverage(dictionaries, regions, bands)
	
	noise = noise_qss(dictionaries, sky_configuration, coverages, realizations, verbose = verbose) 

	outmaps = []
	stdmaps = [] # Useless

	for ic, idict in enumerate(dictionaries):
		noisy_frgrounds = np.zeros(np.shape(noise)[1:])
		for j in range(realizations):
			if verbose: print( np.shape(noise), np.shape(fground_maps), np.shape(noisy_frgrounds))
			noisy_frgrounds[j, ...] = noise[ic, j, ...] + fground_maps[ic]
		outmaps.append(np.mean(noisy_frgrounds, axis = 0))#, np.std(noisymaps150Q, axis = 0)
		stdmaps.append(np.std(noisy_frgrounds, axis = 0))

	return np.array(outmaps), coverages, np.array(stdmaps)

def _mask_maps(maps, coverages, nf_recon):

	cov = []
	for j, ic in enumerate(coverages):
		icov = np.zeros_like(ic, dtype = bool)
		#print(cov.shape)
		covmsk = np.where(ic > 0.01*np.max(ic))
		icov[covmsk] = 1

		for jsub in range(nf_recon):
			maps[j, jsub, ~icov, 0] = hp.UNSEEN

		cov.append(icov)

	return maps, cov

# ##################### preparing MCMC runs
# from lmfit import Model

def LinModel(x, a, b):
	return a + x**b

def QuadModel(x, a, b,c):
	return a*x**2 + b*x + c

def Bnu(nuGHz, temp):
	h = scipy.constants.h
	c = scipy.constants.c
	k = scipy.constants.k
	nu = nuGHz * 1e9
	return 2 * h * nu ** 3 / c ** 2 / (np.exp(h * nu / k / temp ) - 1 )

def ThermDust_Planck353(x, pars, extra_args = None):
	"""

	"""
	c0 = pars[0]
	c1 = pars[1]
	T = 19.6
	nu0 = 353
	bnu = Bnu(x, T)
	#bnu0 = Bnu(nu0, T)
	#print(bnu / bnu0)
	#return a * bnu / bnu0 * (x / nu0) ** (b / 2)
	return c0 * 1e18 * bnu * (x / nu0) ** (c1 / 2)

def ThermDust_Planck353_pointer(x, *pars, extra_args = None):
	T = 19.6
	nu0 = 353
	bnu = Bnu(x, T)
	return pars[0] * 1e18 * bnu * (x / nu0) ** (pars[1] / 2)

def ThermDust_Planck545(x, pars, extra_args = None):
	"""
	Three parameter model for thermal dust [ arXiv:1502.01588]:
		s_d = A_d * (nu / nu0)**(b_d+1) * [(exp(gamma*nu0)-1) / (exp(gamma*nu)-1) ]
		gamma = h / k_b / T_d

		[s_d] = brightness temperature (uK)
		nu0 = 545GHz

	Parameters: 
		A --> A_d: amplitude
		b --> b_d: spectral index
		T --> T_d: temperature
		x --> nu [GHz] array
	"""
	A = pars[0]
	b = pars[1]
	
	T = 23
	h = scipy.constants.h
	k = scipy.constants.k
	nu0 = 545 #GHz

	gamma = h / k / T

	return A * (x / nu0) ** (b + 1) * (np.exp(gamma * nu0) - 1)/(np.exp(gamma * x) - 1)

def ThermDust_Planck545_pointer(x, *pars, extra_args = None):
	T = 23
	h = scipy.constants.h
	k = scipy.constants.k
	nu0 = 545 #GHz

	gamma = h / k / T

	return pars[0] * (x / nu0) ** (pars[1] + 1) * (np.exp(gamma * nu0) - 1)/(np.exp(gamma * x) - 1)

def Synchrotron_storja(x, pars, extra_args = None):
	"""
	Two parameter model for Synchrotron effect [ arXiv:1502.01588] and [arxiv: 1108.4822] (Strong, Orlando & Jaffe):

		s_s = A_s * (nu0 / nu)**2 * f_s(nu/alpha) / f_s(nu0/alpha)
		f_s = external template
		
		alpha > 0, spatially constant
		A_s > 0
		[s_s] = brightness temperature (uK)
		nu0 = 408MHz
	
	Ver también: 	https://ui.adsabs.harvard.edu/link_gateway/2013MNRAS.436.2127O/PUB_PDF (2013)
					https://arxiv.org/pdf/1207.3675.pdf (Delabrouille 2012)
					https://arxiv.org/pdf/1108.4822.pdf (Strong, Orlando & Jaffe 2011)
					https://arxiv.org/pdf/1106.4821.pdf (Aksoy $ Lewicki 2011)
	Parameters: 
		x --> nu [GHz] array

		A 	--> A_s: amplitude
		alpha-> alpha: spectral parameter 



	"""

	c0 = pars[0]
	c1 = pars[1]

	h = scipy.constants.h
	c = scipy.constants.c
	k = scipy.constants.k
	return c0 * 1e10 * x ** (- c1)

def Synchrotron_storja_pointer(x, *pars, extra_args = None):
	h = scipy.constants.h
	c = scipy.constants.c
	k = scipy.constants.k
	return pars[0] * 1e10 * x ** (- pars[1])

def Synchrotron_Planck(x, pars, extra_args = None ):
	"""
	x: frequency array [in GHz]
	A: Amplitude [in uK?]
	alpha: shift normalization in frequency 
	b: spectral index
	"""
	
	c0 = pars[0]
	c1 = pars[1]
	c2 = pars[2] 
	h = scipy.constants.h
	c = scipy.constants.c
	k = scipy.constants.k

	return c0 * 1e5 * (x / c1) ** (- c2)

def Synchrotron_Planck_pointer(x, *pars, extra_args = None ):
	h = scipy.constants.h
	c = scipy.constants.c
	k = scipy.constants.k

	return pars[0] * 1e5 * (x / pars[1]) ** (- pars[2])

def DustSynch_model(x, pars, extra_args = None):
	c0 = pars[0]
	c1 = pars[1]
	T = 19.6
	nu0 = 353
	bnu = Bnu(x, T)
	
	c2 = pars[2]
	c3 = pars[3]

	h = scipy.constants.h
	c = scipy.constants.c
	k = scipy.constants.k

	return c0 * 1e18 * bnu * (x / nu0) ** (c1 / 2) +  c2 * 1e10 * x ** (- c3)

def DustSynch_model_pointer(x, *pars, extra_args = None):
	c0 = pars[0]
	c1 = pars[1]
	T = 19.6
	nu0 = 353
	bnu = Bnu(x, T)
	
	c2 = pars[2]
	c3 = pars[3]
	#c4 = pars[4] 
	h = scipy.constants.h
	c = scipy.constants.c
	k = scipy.constants.k

	return pars[0] * 1e18 * bnu * (x / nu0) ** (pars[1] / 2) + pars[2] * 1e10 * x ** (- pars[3])

def PixSED_Xstk(nus, maps, FuncModel, pix, pix_red, istk, covMat, nus_edge,
		   maxfev = 10000, initP0 = None, verbose = False, chi2 = None,
		  nsamples = 5000):
	
	#print(np.shape(covMat[:, :, istk, pix_red]))
	#popt, pcov = curve_fit(ThermDust_Planck353_l, nus, maps[:, pix, istk], 
	#						sigma = covMat[:, :, istk, pix_red], absolute_sigma=True,
	#						maxfev = maxfev, p0 = initP0)
	popt = initP0

	if verbose:
		print("Calling LogLikelihood", popt, pcov)
		print("xvals, yvals", nus, maps[:, pix, istk])
	myfit = mcmc.LogLikelihood(xvals = nus, yvals = maps[:, pix, istk], chi2 = chi2,
							   errors = covMat[:, :, istk, pix_red], 
							   model = FuncModel, p0 = popt)
	#print("myfit info: " )
	fit_prep = myfit.run(nsamples)
	#print("Doing chain")
	flat_samples = fit_prep.get_chain(discard = nsamples//2, thin=32, flat=True)
	#print("Samples ", flat_samples, np.shape(flat_samples))
	nspls = flat_samples.shape[0]
	#Generating realizations for parameters of the model (fake X(nu))
	
	x = np.linspace(nus_edge[0], nus_edge[-1], nsamples//2)
	vals = np.zeros((len(x), nspls))
	for i in range(len(x)):
		for j in range(nspls):
			vals[i, j] = FuncModel(x[i], flat_samples[j, :])
	
	mvals = np.mean(vals, axis=1)
	svals = np.std(vals, axis=1)
	
	return mvals, svals, x, flat_samples


def foregrounds_run_mcmc(dictionaries, fgr_map, Cp_prime, FuncModel,
					nus_out, nus_edge, pixs, pixs_red = None, chi2 = None,
					samples = 5000, verbose = True, initP0 = None):
	t0 = time.time()

	MeanVals = np.zeros((len(dictionaries), samples//2, 3))
	StdVals = np.zeros((len(dictionaries), samples//2, 3))
	xarr = np.zeros((len(dictionaries), samples//2, 3))
	# ndim = 
	#2496 if samples == 5k or 4992 if samples == 10k 7488 is sample == 15k 
	ndim = 2496
	_flat_samples = np.zeros((len(dictionaries), ndim, len(initP0)))

	for istk in range(3):
		if verbose: print("======== Doing {} Stokes parameter =============".format(dictionaries[0]['kind'][istk]))
		for j in range(len(dictionaries)):
			
			if verbose: 
				print(np.shape(dictionaries), np.shape(fgr_map[j]), np.shape(Cp_prime[j]), np.shape(nus_out[j]), 
							np.shape(nus_edge[j])) 
				print("initP0 ", initP0)
			MeanVals[j, :, istk], StdVals[j, :, istk], xarr[j, :, istk], _flat_samples[j] = \
														PixSED_Xstk(nus_out[j], fgr_map[j], FuncModel, 
																	pixs[j], pixs_red[j], istk, Cp_prime[j], nus_edge[j], 
																	chi2 = chi2, initP0 = initP0, nsamples = samples)
	print('Done in {:.2f} min'.format((time.time()-t0)/60 ))

	return MeanVals, StdVals, xarr[:,:,0], _flat_samples

def udgrade_maps(fground_maps, noise, new_nside, nf_recon, nreals):
	
	"""
	Upgrade or Degrade foreground maps. 

	It returns foreground maps UD-graded, std (useless) and noise maps UD-graded for each noise realization
	"""
	
	npix_ud = 12 * new_nside **2 

	fgr_map_ud = np.zeros((len(fground_maps), nf_recon, npix_ud, 3))

	noise_ud_i = np.zeros((len(fground_maps), nreals, nf_recon, npix_ud, 3))
	
	maps_ud_i = np.zeros_like(noise_ud_i)

	for bandreg in range(len(fground_maps)):
		for irec in range(nf_recon):
			fgr_map_ud[bandreg, irec] = hp.ud_grade(fground_maps[bandreg, irec].T, new_nside).T
			for ireal in range(nreals):
				noise_ud_i[bandreg, ireal, irec] = hp.ud_grade(noise[bandreg, ireal, irec].T, new_nside).T
				maps_ud_i[bandreg, ireal, ...] = noise_ud_i[bandreg, ireal, ...] + fgr_map_ud[bandreg]
	#

	maps_ud, std_ud = np.mean(maps_ud_i, axis = 1), np.std(maps_ud_i, axis = 1)

	return maps_ud, std_ud, fgr_map_ud, noise_ud_i

def make_fit_SED(xSED, xarr, Imvals, Isvals, FuncModel, fgr_map_ud, pixs_ud, nf_recon, 
				initP0 = None, maxfev = 1000):


	# NEW (18 Feb 2021)
	ErrBar2 = lambda Q, U, Qerr, Uerr: np.sqrt( Q ** 2 * Qerr ** 2 + U ** 2 * Uerr ** 2) / \
					np.sqrt( Q ** 2 + U ** 2)
	
	#if not hasattr(initP0, '__len__'):
	#	raise ValueError("{}: You have to provide the init, otherwise curve_fit cannot \
	#				determine the number of parameters".format(FuncModel._name__))

	# last dimenssion ==2 because polarization is P = sqrt(Q**2 + U**2)
	ySED = np.zeros((len(fgr_map_ud), nf_recon, 2))
	popt = np.zeros((len(fgr_map_ud), len(initP0), 2))
	pcov = np.zeros((len(fgr_map_ud), len(initP0), len(initP0), 2))
	#With polarization

	# Modeling fit to map values
	for icomp in range(2):
		for j in range(len(fgr_map_ud)):
			if icomp == 0:
				ySED[j, :, icomp] = fgr_map_ud[j][:,pixs_ud[j],0]
				#print("FuncModel,xSED, ySED", FuncModel, xSED[j], ySED[j,:,icomp])
				#print("curve_fit", curve_fit(f = FuncModel, xdata = xSED[j], 
				#							ydata = ySED[j,:,icomp], p0 = initP0)[0],)
				auxpopt, auxcov = curve_fit(f = FuncModel, xdata = xSED[j], 
											ydata = ySED[j, :, icomp], p0 = initP0, maxfev = maxfev)
				print("auxpopt, auxpcov", auxpopt, auxcov)
				popt[j, :, icomp], pcov[j, :, :, icomp] = auxpopt, auxcov
				#print("==== Parameters for optimization (SED fitting)")
				#print(popt[j, :, icomp], pcov[j, :, :, icomp])
			else:
				ySED[j, :, icomp] = np.sqrt(fgr_map_ud[j][:,pixs_ud[j], 1] ** 2 + \
											fgr_map_ud[j][:,pixs_ud[j], 2] ** 2)
				auxpopt, auxcov = curve_fit(f = FuncModel, xdata = xSED[j], 
											ydata = ySED[j, :, icomp], p0 = initP0, maxfev = maxfev)
				popt[j, :, icomp], pcov[j, :, :, icomp] = auxpopt, auxcov

	ySED_fit = np.zeros((len(fgr_map_ud), len(xarr[0]), 2 ))
	Pmean = np.zeros((len(fgr_map_ud), len(xarr[0])) )
	Perr = np.zeros((len(fgr_map_ud), len(xarr[0])) )
	for icomp in range(2):
		for j in range(len(fgr_map_ud)):
			if icomp == 0:
				ySED_fit[j,:,icomp] = FuncModel(xarr[j, :], *popt[j, :, icomp])
			else:
				# Prepare maps to plot and errorbars after MCMC
				Pmean[j] = np.sqrt(Imvals[j, :, 1] ** 2 + Imvals[j, :, 2] ** 2)
				#Error 
				Perr[j, :] = ErrBar2(Imvals[j, :, 1], Imvals[j, :, 2], 
								   Isvals[j, :, 1], Isvals[j, :, 2])
				
				ySED_fit[j,:,icomp] = FuncModel(xarr[j, :], *popt[j, :, icomp])

	return ySED_fit, Pmean, Perr

def _plot_exampleSED(dictionary, center, nus_out, maskmaps, mapsarray = False, 
					DeltaTheta = 0, DeltaPhi = 0, savefig = False):

	"""
	Plot an example of Figure 10 (map + SED ) in paper 1

	===============
	Parameters: 
		dictionary:
			QUBIC dictionary
		center: 
			center of the FOV for the map (maskmap)
		nus_out:
			frequencies where was reconstructed the map
		maskmap:
			a sky map. 'mask' prefix it's because is recommended to use masked maps (unseen values) in the unobserved region
	===============
	Return:
		Figure with 2 subplots. At left a sky region (healpix projection), right subplot the SED.
	"""

	capsize=3
	plt.rc('font', size=16)

	pixG = [hp.ang2pix(dictionary['nside'], np.pi / 2 - np.deg2rad(center[1] + DeltaTheta ), 
					   np.deg2rad(center[0] + DeltaPhi) ), ]

	fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(14,5),)
	ax = ax.ravel()
	IPIXG = pixG[0] 
	color = ['r','g','k']
	label = ['dust', 'synchrotron', 'dust+synchrotron']
	if mapsarray:
		for j, imap in enumerate(maskmaps):
			ax[1].plot(nus_out, imap[:,IPIXG,0], 'o', color=color[j], label = label[j])
		ax[1].legend()
		ax[0].cla()	
		plt.axes(ax[0])
		hp.gnomview(maskmaps[-1][-1,:,0], reso = 15,hold = True, title = ' ',unit = r'$\mu$K', notext =True,
					min = 0 ,
					max = 0.23 * np.max(maskmaps[-1][-1,:,0]), rot = center)
	else:
		ax[1].plot(nus_out, maskmaps[:,IPIXG,0], 'o-', color='r')
		ax[0].cla()
		plt.axes(ax[0])
		hp.gnomview(maskmaps[-1,:,0], reso = 15,hold = True, title = ' ',unit = r'$\mu$K', notext =True,
					min = 0 ,
					max = 0.23 * np.max(maskmaps[-1,:,0]), rot = center)
	hp.projscatter(hp.pix2ang(dictionary['nside'], IPIXG), marker = '*', color = 'r',s = 180)
	ax[1].set_ylabel(r'$I_\nu$ [$\mu$K]')
	ax[1].set_xlabel(r'$\nu$[GHz]')
	dpar = 10
	dmer = 20
	ax[1].grid()
	#Watch out, the names are wrong (change it)
	mer_coordsG = [ center[0] - dmer,   center[0], center[0] + dmer]
	long_coordsG = [center[1] - 2*dpar, center[1] - dpar, center[1], 
					center[1] + dpar,   center[1] + 2 * dpar]
	#paralels
	for ilong in long_coordsG:
		plt.text(np.deg2rad(mer_coordsG[0] - 13), 1.1*np.deg2rad(ilong), 
				 r'{}$\degree$'.format(ilong))
	#meridians
	for imer in mer_coordsG:
		if imer < 0:
			jmer = imer + 360
			ip, dp = divmod(jmer/15,1)
		else:
			ip, dp = divmod(imer/15,1)
		if imer == 0:
			plt.text(-np.deg2rad(imer + 3), np.deg2rad(long_coordsG[-1] + 6), 
				 r'{}$\degree$'.format(int(ip) ))
		else:
			plt.text(-np.deg2rad(imer + 3), np.deg2rad(long_coordsG[-1] + 6), 
				 r'{}$\degree$'.format(imer))

	hp.projtext(mer_coordsG[1] + 2, long_coordsG[0] - 6, '$l$',  color = 'k', lonlat=True)
	hp.projtext(mer_coordsG[2] + 12.5, long_coordsG[2] - 1, '$b$', rotation = 90, color = 'k', lonlat=True)
	hp.graticule(dpar = dpar, dmer = dmer, alpha = 0.6, verbose = False)
	plt.tight_layout()
	if savefig:
		plt.savefig('SED-components.svg', format = 'svg',  bbox_inches='tight')
		plt.savefig('SED-components.pdf', format = 'pdf',  bbox_inches='tight')
		plt.savefig('SED-components',  bbox_inches='tight')
	plt.show()
