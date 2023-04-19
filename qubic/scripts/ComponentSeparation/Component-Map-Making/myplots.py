import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import sys
import pickle
sys.path.append('/home/regnier/work/regnier/mypackages')
import plotter as p
import os
import qubic
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator


N = 84
nstk = 3
nside = 64
nside_fit = 0
convolution = False
noisy = True
band = 150220
comp_name = ['CMB', 'DUST', 'SYNCHROTRON']
nc = len(comp_name)

path = os.getcwd()
path_i = f'GAL_CMB_DUST_SYNC_band{band}_convolution{convolution}_noise{noisy}_nsidefit{nside_fit}/'
fullpath = path + '/' + path_i

analysis = p.Analysis(fullpath, N, nside, nc, nside_fit, thr=0., convolution=convolution, path_to_save=f'PLOTS_'+path_i)


#analysis.plot_spectra(-1, log=True, figsize=(8, 8), comp_name=['CMB', 'DUST'])
#analysis.make_gif_spectra(fps=24, comp_name=['CMB', 'DUST'], log=True, figsize=(8, 6))
#analysis.plot_spectra_few_iterations([0, 40, 100, 200, 500], comp_name=['CMB', 'DUST'], log=True, figsize=(8, 6))    #[0, 40, 100, 200, 500]


### RMS as function of iteration ###
analysis.plot_rms_maps(i=-1, comp_name=comp_name)

### Beta as function of iterations ###
#analysis.plot_beta(i=-1, figsize=(6, 6), truth=1.54)

### Reconstructed gain ###
#analysis.plot_FP_gain(i=-1, iFP=0, vmin=0.7, vmax=1.3)
#analysis.plot_FP_gain(i=-1, iFP=1, vmin=0.7, vmax=1.3)

### Histograms of residuals recontructed gain ###
#analysis.plot_hist_residuals_gain(i=-1, iFP=0, c='blue', bins=20, figsize=(6, 6))
#analysis.plot_hist_residuals_gain(i=-1, iFP=1, c='red', bins=20, figsize=(6, 6))

### Gnomview of each components ###
#center = qubic.equ2gal(0, -57)
center = qubic.equ2gal(100, -157)
reso = 20

#analysis.plot_maps_without_input(i=-1, center=center, reso=reso, istk=0, comp_name=comp_name, figsize=(7, 9), min=-200, max=200, rmin=-10, rmax=10)
#analysis.plot_maps_without_input(i=-1, center=center, reso=reso, istk=1, comp_name=comp_name, figsize=(7, 9), min=-8, max=8, rmin=-1, rmax=1)
#analysis.plot_maps_without_input(i=-1, center=center, reso=reso, istk=2, comp_name=comp_name, figsize=(7, 9), min=-8, max=8, rmin=-1, rmax=1)
rr = 4
analysis.plot_maps(i=-1, center=center, reso=reso, istk=0, comp_name=comp_name, figsize=(10, 10), min=-200, max=200, rmin=-rr, rmax=rr)
analysis.plot_maps(i=-1, center=center, reso=reso, istk=1, comp_name=comp_name, figsize=(10, 10), min=-6, max=6, rmin=-rr, rmax=rr)
analysis.plot_maps(i=-1, center=center, reso=reso, istk=2, comp_name=comp_name, figsize=(10, 10), min=-6, max=6, rmin=-rr, rmax=rr)


### GIF of convergence ###
fps=10
analysis.gif_maps(fps=fps, center=center, reso=reso, istk=0, comp_name=comp_name, figsize=(9, 9), min=-250, max=250, rmin=-20, rmax=20)
analysis.gif_maps(fps=fps, center=center, reso=reso, istk=1, comp_name=comp_name, figsize=(9, 9), min=-4, max=4, rmin=-4, rmax=4)
#analysis.gif_maps(fps=fps, center=center, reso=reso, istk=2, comp_name=comp_name, figsize=(9, 9), min=-4, max=4, rmin=-4, rmax=4)

