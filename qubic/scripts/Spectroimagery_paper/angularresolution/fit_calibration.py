import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
import qubic
import os
import sys
import pylab
from qubic.io import *
from scipy.optimize import curve_fit
import random as rd
from resolution import *

d = qubic.qubicdict.qubicDict()
d.read_from_file(sys.argv[1])
angmask = 8.

"""

Sensitive parameters for calibration (same in QUBIC pipeline)
	nsideHigh: nside to build maps and then integer over pixels
	nsideLow
	reso: hp.gnomview() parameter
	size: hp.gnomview() parameter

"""

nsideLow, nsideHigh, reso, size, sigma2fwhm = Parameters(d)#, reso = 3.5)# size= 200

_, nus_edge_in, _, _, _, _ = qubic.compute_freq(d['filter_nu']/1e9, d['nf_sub'],
    d['filter_relative_bandwidth']) 

if d['config'] == 'FI':
	CteConf = 61.34
elif d['config'] == 'TD':
	CteConf = 153.36
	
n_subpop, fwhm_ini, fwhm_end, sample, step_fwhm, amplitude = ParametersMC(fwhm_ini = CteConf/nus_edge_in[0], 
																			fwhm_end = CteConf/nus_edge_in[-1],
																			sample = 50)
rename =str(reso).replace('.','-')
#outputname = NameCalib(method = 'fit')+str(rename)+'-{}-{}.txt'.format(d['nside'], int(d['filter_nu']/1e9)) # Could be 'fit' or 'sigma'
outputname = sys.argv[2]+'_fitcalibration1-5-256-{}.txt'.format(int(d['filter_nu']/1e9))
# Compute the parameter space domain
fwhm = np.arange(fwhm_ini, fwhm_end, step_fwhm)

# compute freq's from angular resolution using P = 20 and deltaX = 1.4cm
if d['config'] == 'FI':
	nus = 61.347409/fwhm
elif d['config'] == 'TD':
	nus = 153.36/fwhm

					### ====  									==== 
					### ====  									==== 
					### ====  Compute the fwhm_j from fwhm_theo ==== 
					### ====  									==== 
					### ====  									====

### ====  			Preparing maps			==== 
### ====  									====

# Center of the Gaussian -> fixed
center_gal = qubic.equ2gal(d['RA_center'], d['DEC_center'])

""" The calibration can be done using a single pixel painted and then smoothing with a given fwhm_i (hp.smoothing(fwhm = fwhm_i))
 - onePx = True-  or can be done computing the gaussian using f function - onePx = False -. 
"""

onePx = True
if onePx == True:
	pixel = hp.pixelfunc.ang2pix(nsideHigh, np.deg2rad(90-center_gal[1]), np.deg2rad(center_gal[0]))
elif onePx == False:
	pixel = hp.pixelfunc.ang2pix(nsideHigh, np.deg2rad(90-center_gal[1]), np.deg2rad(center_gal[0]), nest = True)

# used only if onePx = False
vec_pix = hp.pix2vec(nsideHigh, pixel, nest = True)
vec_pixeles = hp.pix2vec(nsideHigh, np.arange(12*nsideHigh**2), nest = True )
ang_pixeles = np.arccos(np.dot(vec_pix,vec_pixeles))
mask = np.rad2deg(ang_pixeles) < angmask
# eslaF = xPeno fi ylno desu

# Cartesian coordinates to map the field extracted
x_map = np.linspace(-size/2,size/2,size)*reso/60
y_map = x_map

x_map, y_map = np.meshgrid(x_map, y_map)

xdata_map = x_map.ravel(),y_map.ravel()

x = x_map[0]
x2 = x*x

"""
Variables: 
	
	deltaFwhm_fit: difference between average angular resolution measured and theoretical one
	varianza_fit: variance
	ellip:  (NOT IMPLEMENTED)ellipticity of the fitted gaussian. Following BICEP2/keck definition (arXiv:0906.4069)
			e = (sigma_a - sigma_b) / (sigma_a + sigma_b)
"""

deltaFwhm_fit = np.zeros( (len(fwhm), 1) )
#position_center_fit = np.zeros((n_subpop,1))
varianza_fit = np.zeros( (len(fwhm), 1) )
ellip = np.zeros( (len(fwhm), ) )

for f_i, fwhm_i in enumerate(fwhm):
	
	print('=== Computing map {} ==='.format(f_i))
		
	f0_ud = np.zeros((n_subpop, 12*nsideHigh**2,))

	## 1) if gaussian compute:
	if onePx == False:
		for i,each in enumerate(ang_pixeles):
			if mask[i] == True:
				f0_ud[0,i] = 1e-6*f(each, fwhm = fwhm_i, sigma2fwhm = sigma2fwhm) #1e-6 --> pues \mu K
	elif onePx == True:
		## 2) if 1pixel painted
		f0_ud[0,pixel] = 1.
		f0_ud[0,:] = hp.smoothing(f0_ud[0,:], fwhm = np.deg2rad(fwhm_i))
		## end 2)
	
	#same input-map for each sub-sample 
	f0_ud[:,:] = f0_ud[0,:]

	#Noise. 0.01 of the mean of x0, luego armar array de 30 mapas noise.
	noise = np.empty((n_subpop,12*nsideHigh**2))
			
	# Noise amplitude?
	amp = 2.*np.mean(f0_ud[0,:])
	for i in range(n_subpop):
		noise[i,:] = amp*np.random.random(np.shape(f0_ud[0]))
		f0_ud[i,:] += noise[i,:]
		
	m0_ud = np.zeros((n_subpop, hp.nside2npix(nsideLow)), dtype=np.float)
		
	if onePx == False:
		for i in range(n_subpop):
			m0_ud[i,:] = hp.ud_grade(f0_ud[i,:], nsideLow, order_in = 'NESTED', order_out = 'RING')
	elif onePx == True:
		for i in range(n_subpop):
			m0_ud[i,:] = hp.ud_grade(f0_ud[i,:], nsideLow, order_in = 'RING', order_out = 'RING')

	maps_subpop = np.zeros((n_subpop,size,size))
			
	for i, mapa in enumerate(m0_ud):
	    maps_subpop[i,:,:] = hp.gnomview(mapa[:], rot = center_gal,  
		                            reso = reso, xsize = size,
		                            return_projected_map=True)
	mp.close('all')

	aux = size/2

	# Fit Method

	ave_fwhm = np.zeros((n_subpop,))
	#position_center_fit[f_i,0] = offset
	ellip_fit = np.zeros((n_subpop,1))
	ave_fwhm = FitMethod(maps_subpop, d)
	
	ellip[f_i] = np.mean(ellip_fit, axis=0)
	deltaFwhm_fit[f_i,0] = np.mean(ave_fwhm)-fwhm_i # fwhm_m - fwhm_r
	varianza_fit[f_i,0] = np.var(np.asarray(ave_fwhm), ddof = 1) #check it.. something is rare

st_dev = np.sqrt(varianza_fit)

np.savetxt(outputname, np.transpose([nus, deltaFwhm_fit[:,0], st_dev[:,0]]), fmt = ['%10.5f','%10.5f','%11.9f'],
 newline = '\n', comments = '# ', header = ' \t nus \t DeltaFWHM \t Standard Dev' ) 