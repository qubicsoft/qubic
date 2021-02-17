#
# Author: Martín M. Gamboa Lerena.
# Date: Feb 15th 2021
#

import os
import sys
import glob
from importlib import reload
import gc
# Specific science modules
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pickle 
import astropy.io as fits

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
	seed = None, verbose = False):

	fground_maps = foreground_signal(dictionaries, sky_configuration, verbose = verbose)
	
	coverages = coverage(dictionaries, regions, bands)

	noise = noise_qss(dictionaries, sky_configuration, coverages, realizations, verbose = verbose) 

	outmaps = []

	for ic, idict in enumerate(dictionaries):
		noisy_frgrounds = np.zeros(np.shape(noise)[1:])
		for j in range(realizations):
			if verbose: print( np.shape(noise), np.shape(fground_maps), np.shape(noisy_frgrounds))
			noisy_frgrounds[j, ...] = noise[ic, j, ...] + fground_maps[ic]
		outmaps.append(np.mean(noisy_frgrounds, axis = 0))#, np.std(noisymaps150Q, axis = 0)

	return np.array(outmaps), coverages

def _mask_maps(maps, coverages, nf_recon):

	for j, icov in enumerate(coverages):
		cov = np.zeros_like(icov, dtype = bool)
		print(cov.shape)
		covmsk = np.where(icov > 0.01*np.max(icov))
		cov[covmsk] = 1

		for jsub in range(nf_recon):
			maps[j, jsub, ~cov, 0] = hp.UNSEEN

	return maps

