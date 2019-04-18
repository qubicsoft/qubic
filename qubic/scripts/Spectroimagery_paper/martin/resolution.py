import numpy as np
import healpy as hp
import qubic
import os
import sys
from qubic.io import *
import matplotlib.pyplot as mp
from scipy.optimize import curve_fit
import datetime as dt

def NameRun(d):
	"""
	Create same file output name
	"""

	now = dt.datetime.now()
	name = now.strftime("%Y%m%d")#str(now.year)+str(now.month).zfill(2)+str(now.day).zfill(2)+'/'
	
	try:
		os.path.isdir(name)
	except:
		os.mkdir(name)
	
	#name = bla+'_modular_{}_{}'.format(str(int(d['filter_nu']/1e9)), d['nhwp_angles'])
	return name

def NameCalib(method):
	""" 
	Generate output name of calibration file given a mathod used

	Input: 
		method: 'fit', 'sigma'

	Return:
		METHODname: with format day(hardcoded)+method+'calibration.txt'
	"""
	
	fitname = '20190326_fitcalibration.txt'
	signame = '20190326_sigmacalibration.txt'

	if method == 'fit':
		return fitname
	elif method == 'sigma':
		return signame

def Parameters(d, which = 'all'):
	"""
	Define parameters used several times in the resolution

	Input: 
		d: QUBIC dictionary type
		which: comfortability to not pass all the parameters
	Return: 
		Parameters chosen for the applicant

	"""
	nsideLow = d['nside']
	nsideHigh = d['nside']*2
	reso = 1.5
	size = 200
	# 1) convert sigma to fwhm
	sigma2fwhm = np.sqrt(8*np.log(2))

	if which == 'all':
		return nsideLow, nsideHigh, reso, size, sigma2fwhm
	if which == 'sigma2fwhm':
		return sigma2fwhm
	if which == 'nside':
		return nsideLow, nsideHigh
	if which == 'HealPar':
		return reso, size

def ParametersMC():
	"""
	Parameters for the Monte-Carlo simulation to build calibration files

	Return: 
		n_subpop: Number random realization of noise
		fwhm_ini: initial fwhm 
		fwhm_end: end fwhm
		sample: number of points within the range(fwhm_ini, fwhm_end)
		step_fwhm: step size 
		amplitude: parameter whit sense only for an old version
	"""
	n_subpop = 30
	fwhm_ini = 0.21
	fwhm_end = 0.70
	sample = 5
	step_fwhm = (fwhm_end - fwhm_ini) / sample
	amplitude =  np.array([1.,])

	return n_subpop, fwhm_ini, fwhm_end, sample, step_fwhm, amplitude

def f(val, fwhm, sigma2fwhm):
	"""
	Gaussian function without normalization 

	Input: 
		val: (float) Where to be evaluated
		fwhm: (float) equiv to sigma 
		sigma2fwhm: Convert sigma to fwhm or viceversa

	Return: 
		Function evaluated in val
	"""

	return np.nan_to_num(np.exp(-0.5*val**2/(np.radians(fwhm)/sigma2fwhm)**2))

def normalization(x,mapa):
	"""
    Normalization for 2-D f(x,y)

	Input: 
		x: (array) coordinate system where each points is evaluated
		mapa: f(x) 

	Return: 
		Value for normalization (twice, because 2-D, marginalized)
	"""
	ef = np.trapz((np.trapz(mapa,x,axis=0)),x)
	return 1/ef

def gaussian2d((x,y), x0, y0, varx, vary):
	"""
	Function to fit

	Input:
		(x,y): grid where the 2D function is defined
		x0, y0: Mean of the gaussian (parameter)
		varx, vary: variances of the gaussian (parameter)

	Return:
		Gaussian  
	"""
	gauss = 1/(2*np.pi*varx*vary)*np.exp(-((x-x0)**2/(2*varx**2)+(y-y0)**2/(2*vary**2)))
	return gauss.ravel()

def FitMethod(maparray, d, size = 200, reso = 1.5):
	"""
	Method who fit a gaussian function given some parameters. 
	The parameters for the calibration and for the used must be the same.

	Input:
		maparray: N-array of maps
		d: QUBIC dictionary
		size, reso: To use in healpy (xsize, reso)
	Return:
		N-array of fwhm fitted for each map (N-map)

	"""
	
	# Define cartesian coordinates to extract the map (return_projected_map = True)
	sigma2fwhm = np.sqrt(8*np.log(2))
	x_map = np.linspace(-size/2,size/2,size)*reso/60.
	y_map = x_map
	x_map, y_map = np.meshgrid(x_map, y_map)
	xdata_map = x_map.ravel(),y_map.ravel()

	input_fwhm_fit = np.empty((len(maparray)))#d['nf_sub']))
	popt = []
	for i,m in enumerate(maparray):
		norm_fit = normalization(x_map[0],m)
		ydata_map = (norm_fit * m).ravel()
		popt_map, pcov_map = curve_fit(gaussian2d, xdata_map, ydata_map, method='trf')
		input_fwhm_fit[i] = abs((popt_map[2]+popt_map[3])/2*sigma2fwhm)

	return input_fwhm_fit

def SigmaMethod(maparray, d, size = 200, reso = 1.5):
	"""
	Method who compute the fwhm taken a N-map array as a gaussian distribution function. 
	The parameters for the calibration and for the used must be the same.

	Input:
		maparray: N-array of maps
		d: QUBIC dictionary
		size, reso: To use in healpy (xsize, reso)
	Return:
		N-array of fwhm computed for each map (N-map)

	"""
	
	# Define cartesian coordinates to extract the map (return_projected_map = True)
	sigma2fwhm = Parameters(d, which = 'sigma2fwhm')
	x_map = np.linspace(-size/2,size/2,size)*reso/60.
	y_map = x_map
	x_map, y_map = np.meshgrid(x_map, y_map)
	xdata_map = x_map.ravel(),y_map.ravel()
	x = x_map[0]
	x2 = x*x
	input_fwhm_sigma=np.empty((len(maparray)))#d['nf_sub']))
	
	for i,mi in enumerate(maparray):
		maski = mi > np.max(mi)*0.01
		mi[~maski] = 0
		norm_sig = normalization(x,mi)
		m = norm_sig * mi
		gx = np.trapz(m, x, axis = 0)
		sigma_xl2 = 0.
		sigma_xl2 = np.trapz(x2*gx,x) - (np.trapz(x*gx,x))**2
		input_fwhm_sigma[i] = np.sqrt(sigma_xl2)*sigma2fwhm

	return input_fwhm_sigma

def GenerateMaps(d, nus_in, p=None ):

	"""
	Compute input maps to use in: calibration (for both methods Fit and Sigma) & QUBIC pipeline. 

	Input:
		d: QUBIC dictionary
		nus_in: frequencies where compute the point source map

	Return:
		input_maps: partition of the sky where the point source is.
		m0: point source (already integrated over pixels) [RING ordered]

	"""

	if p:
		if p.fix_az:
			center_gal = (d['fix_azimuth']['az'],d['fix_azimuth']['el'])
		elif not p.fix_az:
			center_gal = qubic.equ2gal(d['RA_center'], d['DEC_center'])
	nsideLow, nsideHigh, reso, size, sigma2fwhm = Parameters(d) 
	#center_gal = qubic.equ2gal(d['RA_center'], d['DEC_center'])
	pixel = hp.pixelfunc.ang2pix(nsideHigh, np.deg2rad(90-center_gal[1]), np.deg2rad(center_gal[0]), nest = True)
	vec_pix = hp.pix2vec(nsideHigh, pixel, nest = True)
	vec_pixeles = hp.pix2vec(nsideHigh, np.arange(12*nsideHigh**2), nest = True )
	ang_pixeles = np.arccos(np.dot(vec_pix,vec_pixeles))
	
	#mask
	mask = np.rad2deg(ang_pixeles) < d['dtheta']
	
	# Generate Gaussian maps - model of point source with FWHM (or Sigma) given by nus_in
	c0 = np.zeros((d['nf_sub'],12*nsideHigh**2,3))
	noise = np.zeros((d['nf_sub'], 12*nsideHigh**2,1))
	
	T = d['temperature']
	amplitude = 1e22
	
	for i, n_i in enumerate(nus_in):
		print 'Map {}'.format(i)
		fwhm_in = 61.347409/n_i # nus to fwhm
		for j,each in enumerate(ang_pixeles):
			if mask[j] == True:
				c0[i,j,0] = amplitude*f(each, fwhm_in, sigma2fwhm)
	c0[:,:,1] = c0[:,:,0]
	c0[:,:,2] = c0[:,:,0]
	m0 = np.empty((d['nf_sub'],12*nsideLow**2,3))
	m0[:,:,0] = hp.ud_grade(c0[:,:,0], nsideLow, order_in = 'NESTED', order_out = 'RING')
	
	input_maps = np.empty((d['nf_sub'],size,size))
	for i, mapa in enumerate(m0):
		input_maps[i] = hp.gnomview(mapa[:,0], rot = center_gal,  
	                            reso = 1.5, xsize = size,
	                            return_projected_map=True)
	mp.close('all')

	return input_maps, m0

