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

"""Sensitive parameters for calibration (same in QUBIC pipeline)
	nsideHigh: nside to build maps and then integer over pixels
	nsideLow
	reso: hp.gnomview() parameter
	size: hp.gnomview() parameter
"""

nsideLow, nsideHigh, reso, size, sigma2fwhm = Parameters(d) #reso = 1.5 size= 200
n_subpop, fwhm_ini, fwhm_end, sample, step_fwhm, amplitude = ParametersMC() #n_subpop = 30 fwhm_ini = 0.21 fwhm_end = 0.70 sample = 50, amplitude =  np.array([1.,])
outputname = NameCalib(method = 'sigma') # Could be 'fit' or 'sigma'

# Compute the parameter space domain
fwhm = np.arange(fwhm_ini, fwhm_end, step_fwhm)

# compute freq's from angular resolution using P = 20 and deltaX = 1.4cm
nus = 61.347409/fwhm

					### ====  									==== 
					### ====  									==== 
					### ====  Compute the fwhm_j from fwhm_theo ==== 
					### ====  									==== 
					### ====  									====

### ====  			Preparing maps			==== 
### ====  									====

#Center of the Gaussian -> fixed

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
mask = np.rad2deg(ang_pixeles) < 5
# eslaF = xPeno fi ylno desu

# Cartesian coordinates to map the field extracted
x_map = np.linspace(-size/2,size/2,size)*reso/60
y_map = x_map

x_map, y_map = np.meshgrid(x_map, y_map)

x = x_map[0]
x2 = x*x

deltaFwhm_sigma = np.zeros( (len(fwhm), 1) )
varianza_sigma = np.zeros( (len(fwhm), 1) )

#Loop over the fwhm:

for f_i, fwhm_i in enumerate(fwhm):
	
	print '=== Computing map {} ==='.format(f_i)
	
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
	amp = 1.6*np.mean(f0_ud[0,:])

	for i in range(n_subpop):
		#noise[i,:] = amp*np.random.normal(loc = 1e-6, scale = 1e-6, size=np.shape(f0_ud[0]))
		noise[i,:] = amp*np.random.random(np.shape(f0_ud[0]))
		f0_ud[i,:] += noise[i,:]

	m0_ud = np.zeros((n_subpop, hp.nside2npix(nsideLow)), dtype=np.float)
			
	if onePx == False:
		for i in range(n_subpop):
			m0_ud[i,:] = hp.ud_grade(f0_ud[i,:], nsideLow, order_in = 'NESTED', order_out = 'RING')
	elif onePx == True:
		for i in range(n_subpop):
			m0_ud[i,:] = hp.ud_grade(f0_ud[i,:], nsideLow, order_in = 'RING', order_out = 'RING')

	maps_subpop = np.empty((n_subpop,size,size))
			
	for i, mapa in enumerate(m0_ud):
	    maps_subpop[i,:,:] = hp.gnomview(mapa[:], rot = center_gal,  
	                            reso = 1.5, xsize = size,
	                            return_projected_map=True)
	mp.close('all')
			
	aux = size/2

	# Sigma Method

	ave_fwhm = np.zeros((n_subpop,))
	ave_fwhm = SigmaMethod(maps_subpop, d)
			
	deltaFwhm_sigma[f_i,0] = np.mean(ave_fwhm-fwhm_i)
	varianza_sigma[f_i,0] = np.var(np.asarray(ave_fwhm), ddof = 1) #check it.. something is rare

st_dev = np.sqrt(varianza_sigma)
np.savetxt(outputname, np.transpose([nus, deltaFwhm_sigma[:,0], st_dev[:,0]]), fmt = ['%10.5f','%10.5f','%11.9f'],
 newline = '\n', comments = '# ', header = ' \t nus \t DeltaFWHM \t Standard Dev' ) 