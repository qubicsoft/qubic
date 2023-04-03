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


N = 300
nstk = 3
nside = 256
nside_fit = 0
convolution = False
noisy = True
band = 150220
comp_name = ['CMB', 'DUST', 'CO']
nc = len(comp_name)

path = os.getcwd()
path_i = f'GAL_NOGAIN_P143217353_CMB_DUST_CO_band{band}_convolution{convolution}_noise{noisy}_nsidefit{nside_fit}/'
fullpath = path + '/' + path_i

analysis = p.Analysis(fullpath, N, nside, nc, nside_fit, thr=0.1, convolution=convolution, path_to_save=f'PLOTS_'+path_i)


analysis.plot_spectra(-1, log=True, figsize=(8, 8), comp_name=['CMB', 'DUST'])
#analysis.make_gif_spectra(fps=24, comp_name=['CMB', 'DUST'], log=True, figsize=(8, 6))
#analysis.plot_spectra_few_iterations([0, 40, 100, 200, 500], comp_name=['CMB', 'DUST'], log=True, figsize=(8, 6))    #[0, 40, 100, 200, 500]


### RMS as function of iteration ###
analysis.plot_rms_maps(i=-1, comp_name=comp_name)

### Reconstructed gain ###
analysis.plot_FP_gain(i=-1, iFP=0, vmin=0.9, vmax=1.1)
analysis.plot_FP_gain(i=-1, iFP=1, vmin=0.9, vmax=1.1)

### Histograms of residuals recontructed gain ###
analysis.plot_hist_residuals_gain(i=-1, iFP=0, c='blue', bins=40, figsize=(6, 6))
analysis.plot_hist_residuals_gain(i=-1, iFP=1, c='red', bins=40, figsize=(6, 6))

### Gnomview of each components ###
#center = qubic.equ2gal(0, -57)
center = qubic.equ2gal(100, -157)
reso = 15

analysis.plot_maps_without_input(i=-1, center=center, reso=reso, istk=0, comp_name=comp_name, figsize=(6, 8), min=-50, max=100, rmin=-100, rmax=100)
analysis.plot_maps_without_input(i=-1, center=center, reso=reso, istk=1, comp_name=comp_name, figsize=(6, 8), min=-8, max=8, rmin=-8, rmax=8)
analysis.plot_maps_without_input(i=-1, center=center, reso=reso, istk=2, comp_name=comp_name, figsize=(6, 8), min=-8, max=8, rmin=-8, rmax=8)

analysis.plot_maps(i=-1, center=center, reso=reso, istk=0, comp_name=comp_name, figsize=(6, 6), min=-50, max=100, rmin=-100, rmax=100)
analysis.plot_maps(i=-1, center=center, reso=reso, istk=1, comp_name=comp_name, figsize=(6, 6), min=-6, max=6, rmin=-8, rmax=8)
analysis.plot_maps(i=-1, center=center, reso=reso, istk=2, comp_name=comp_name, figsize=(6, 6), min=-6, max=6, rmin=-8, rmax=8)


### GIF of convergence ###
fps=24
analysis.gif_maps(fps=fps, center=center, reso=15, istk=0, comp_name=comp_name, figsize=(9, 9), min=-250, max=250, rmin=-30, rmax=30)
analysis.gif_maps(fps=fps, center=center, reso=15, istk=1, comp_name=comp_name, figsize=(9, 9), min=-6, max=6, rmin=-6, rmax=6)
analysis.gif_maps(fps=fps, center=center, reso=15, istk=2, comp_name=comp_name, figsize=(9, 9), min=-6, max=6, rmin=-6, rmax=6)

