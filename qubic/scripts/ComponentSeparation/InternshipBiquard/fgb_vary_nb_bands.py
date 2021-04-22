import random
import healpy as hp
import glob
from scipy.optimize import curve_fit
import pickle
from importlib import reload
import time
import scipy
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import pylab
from pylab import arange, show, cm
from astropy import units as uq
import gc

# Specific qubic modules
from qubicpack.utilities import Qubic_DataDir
from pysimulators import FitsArray
import pysm3
import pysm3.units as u
import pysm3.utils as utils
import qubic
from qubic import QubicSkySim as qss
from qubic import fibtools as ft
from qubic import camb_interface as qc
from qubic import SpectroImLib as si
from qubic import NamasterLib as nam
from qubic import mcmc
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

# FGBuster functions module
from fgbuster import get_instrument, get_sky, get_observation, ilc, basic_comp_sep, harmonic_ilc, weighted_comp_sep, \
    multi_res_comp_sep, harmonic_ilc_alm  # Predefined instrumental and sky-creation configurations
from fgbuster.visualization import corner_norm, plot_component
from fgbuster.mixingmatrix import MixingMatrix
from fgbuster.observation_helpers import _rj2cmb, _jysr2rj, get_noise_realization

# Imports needed for component separation
from fgbuster import (separation_recipes, xForecast, CMB, Dust, Synchrotron, FreeFree, PowerLaw,  # sky-fitting model
                      basic_comp_sep)  # separation routine

import ComponentSeparation

# Widgets
import ipywidgets as widgets

reload(qss)
reload(ft)

print("Importation done")

### Some initializations, to be replaced with specific path, or to modify in bash
# os.environ['QUBIC_DATADIR'] = '/home/mathias/Bureau/qubic/qubic'
# os.environ['QUBIC_DICT'] = '/home/mathias/Bureau/qubic/qubic/dicts'
global_dir = Qubic_DataDir(datafile='instrument.py', datadir=os.environ['QUBIC_DATADIR'])
print('global_dir define')
### Qubic dictionaries for 150GHz and 220Ghz
config_150, config_220 = 'FI-150', 'FI-220'
dictfilename150 = global_dir + '/doc/FastSimulator/FastSimDemo_{}.dict'.format(config_150)
dictfilename220 = global_dir + '/doc/FastSimulator/FastSimDemo_{}.dict'.format(config_220)
d150, d220 = qubic.qubicdict.qubicDict(), qubic.qubicdict.qubicDict()
d150.read_from_file(dictfilename150)
d220.read_from_file(dictfilename220)
qub_dic = {'150': d150, '220': d220}
center = qubic.equ2gal(d220['RA_center'], d220['DEC_center'])

### Read some stuff
# Read dictionary chosen
d150['focal_length'] = 0.3
d150['nside'] = 256

# Define the number of reconstruction bands:
nbands = 3
d150['nf_recon'] = nbands
d150['nf_sub'] = nbands

npix = 12 * d150['nside'] ** 2
Nf = int(d150['nf_sub'])
band = d150['filter_nu'] / 1e9
filter_relative_bandwidth = d150['filter_relative_bandwidth']
a, nus_edge, nus_in, d, e, Nbbands_in = qubic.compute_freq(band, Nf, filter_relative_bandwidth)

print("Dictionnaries loaded")

cov = pickle.load(open('/pbs/home/m/mregnier/coverage_dtheta_40_pointing_6000.pkl', 'rb'))
cov150 = cov['coverage']

okpix_inside = (cov150 > (0.5 * np.max(cov150)))
cov_qubic = np.loadtxt('/pbs/home/m/mregnier/mask_fgbuster')
okpix_qubic = cov_qubic != 0

print("coverage loaded")

if j == 3:
    freqs = [136.984337, 148.954079, 161.969742]
    fwhmdegs = [0.42999269, 0.39543908, 0.36366215]
elif j == 4:
    freqs = [135.50540829, 144.29216391, 153.6486907, 163.61193508]
    fwhmdegs = [0.43468571, 0.40821527, 0.38335676, 0.36001202]
elif j == 5:
    freqs = [134.63280448, 141.57278853, 148.8705114, 156.54441361, 164.61388627]
    fwhmdegs = [0.43750306, 0.41605639, 0.39566106, 0.37626551, 0.35782075]
else:
    raise TypeError('Incorrect frequency number')

fwhm_final = np.max(fwhmdegs) + 1e-8  # All maps will have this resolution

# QubicSkySim options
verbose = False
integration_into_band = True
nunu_correlation = True
FWHMdeg = fwhmdegs
spatial_noise = True
nyears = 4e2

okpix = (cov150 > (0.1 * np.max(cov150)))

nsim = sys.argv[1]
nband = [3, 4, 5]

beta_d_3bands_150 = []
beta_d_4bands_150 = []
beta_d_5bands_150 = []
Stkp = 'IQU'

## For 150 GHz
for i, j in enumerate(nband):

    ### Read some stuff
    d = d150
    # Read dictionary chosen
    d['focal_length'] = 0.3
    d['nside'] = 256

    # Define the number of reconstruction bands:
    nbands = j
    d['nf_recon'] = nbands
    d['nf_sub'] = nbands

    npix = 12 * d['nside'] ** 2
    Nf = int(d['nf_sub'])
    band = d['filter_nu'] / 1e9
    filter_relative_bandwidth = d['filter_relative_bandwidth']
    a, nus_edge, nus_in, dp, e, Nbbands_in = qubic.compute_freq(band, Nf, filter_relative_bandwidth)

    cov = pickle.load(open('/pbs/home/m/mregnier/coverage_dtheta_40_pointing_6000.pkl', 'rb'))
    cov150 = cov['coverage']

    okpix_inside = (cov150 > (0.5 * np.max(cov150)))
    cov_qubic = np.loadtxt('/pbs/home/m/mregnier/mask_fgbuster')
    okpix_qubic = cov_qubic != 0

    if j == 3:
        freqs = [136.984337, 148.954079, 161.969742]
        fwhmdegs = [0.42999269, 0.39543908, 0.36366215]
    elif j == 4:
        freqs = [135.50540829, 144.29216391, 153.6486907, 163.61193508]
        fwhmdegs = [0.43468571, 0.40821527, 0.38335676, 0.36001202]
    elif j == 5:
        freqs = [134.63280448, 141.57278853, 148.8705114, 156.54441361, 164.61388627]
        fwhmdegs = [0.43750306, 0.41605639, 0.39566106, 0.37626551, 0.35782075]
    else:
        raise TypeError('Incorrect frequency number')

    okpix = (cov150 > (0.1 * np.max(cov150)))

    for m in range(nsim):
        seed = m
        sky_config = {'dust': 'd0', 'cmb': seed}
        Qubic_sky_150 = qss.Qubic_sky(sky_config, d)

        CMBdust150, CMBdust_noiseless, CMBdust_noise, P = Qubic_sky_150.get_partial_sky_maps_withnoise(coverage=cov150,
                                                                                                       Nyears=nyears,
                                                                                                       verbose=verbose,
                                                                                                       FWHMdeg=FWHMdeg,
                                                                                                       seed=seed,
                                                                                                       spatial_noise=spatial_noise,
                                                                                                       nunu_correlation=nunu_correlation,
                                                                                                       integrate_into_band=integration_into_band
                                                                                                       )
        CMBdust150 = np.transpose(CMBdust150, (0, 2, 1))
        CMBdust_noiseless = np.transpose(CMBdust_noiseless, (0, 2, 1))
        CMBdust_noise = np.transpose(CMBdust_noise, (0, 2, 1))

        for f in range(j):

            R = ComponentSeparation.CompSep(d).fg_buster(CMBdust150,
                                                         [CMB(), Dust(freqs[f], temp=20.)], freq=freqs,
                                                         fwhmdeg=fwhmdegs,
                                                         target=fwhm_final, okpix=okpix_inside, Stokesparameter=Stkp)

            print(f, R.x[0])
            if j == 3:
                # array_est_fg_3bands[m, 0, f, :, okpix_inside] = R.s[0, :, :].T
                # array_est_fg_3bands[m, 1, f, :, okpix_inside] = R.s[1, :, :].T
                beta_d_3bands_150.append(R.x[0])
            elif j == 4:
                # array_est_fg_4bands[m, 0, f, :, okpix_inside] = R.s[0, :, :].T
                # array_est_fg_4bands[m, 1, f, :, okpix_inside] = R.s[1, :, :].T
                beta_d_4bands_150.append(R.x[0])
            elif j == 5:
                # array_est_fg_5bands[m, 0, f, :, okpix_inside] = R.s[0, :, :].T
                # array_est_fg_5bands[m, 1, f, :, okpix_inside] = R.s[1, :, :].T
                beta_d_5bands_150.append(R.x[0])
            else:
                pass

        del CMBdust150
        del CMBdust_noiseless
        del CMBdust_noise
        del P

beta_d_3bands_220 = []
beta_d_4bands_220 = []
beta_d_5bands_220 = []
Stkp = 'IQU'

## For 220 GHz
for i, j in enumerate(nband):

    ### Read some stuff
    d = d220
    # Read dictionary chosen
    d['focal_length'] = 0.3
    d['nside'] = 256

    # Define the number of reconstruction bands:
    nbands = j
    d['nf_recon'] = nbands
    d['nf_sub'] = nbands

    npix = 12 * d['nside'] ** 2
    Nf = int(d['nf_sub'])
    band = d['filter_nu'] / 1e9
    filter_relative_bandwidth = d['filter_relative_bandwidth']
    a, nus_edge, nus_in, dp, e, Nbbands_in = qubic.compute_freq(band, Nf, filter_relative_bandwidth)

    cov = pickle.load(open('/pbs/home/m/mregnier/coverage_dtheta_40_pointing_6000.pkl', 'rb'))
    cov150 = cov['coverage']

    okpix_inside = (cov150 > (0.5 * np.max(cov150)))
    cov_qubic = np.loadtxt('/pbs/home/m/mregnier/mask_fgbuster')
    okpix_qubic = cov_qubic != 0

    if nbands == 3:
        freqs220 = [200.9103609, 218.46598318, 237.55562228]
        fwhmdegs = [0.42999269, 0.39543908, 0.36366215]
    elif nbands == 4:
        freqs220 = [198.74126549, 211.62850707, 225.35141303, 239.96417145]
        fwhmdegs = [0.43468571, 0.40821527, 0.38335676, 0.36001202]
    elif nbands == 5:
        freqs220 = [197.46144657, 207.64008985, 218.34341672, 229.59847329, 241.43369986]
        fwhmdegs = [0.43750306, 0.41605639, 0.39566106, 0.37626551, 0.35782075]
    else:
        raise TypeError('Incorrect frequency number')

    okpix = (cov150 > (0.1 * np.max(cov150)))

    for m in range(nsim):
        seed = m
        sky_config = {'dust': 'd0', 'cmb': seed}
        Qubic_sky_220 = qss.Qubic_sky(sky_config, d)

        CMBdust220, CMBdust_noiseless, CMBdust_noise, P = Qubic_sky_220.get_partial_sky_maps_withnoise(coverage=cov150,
                                                                                                       Nyears=nyears,
                                                                                                       verbose=verbose,
                                                                                                       FWHMdeg=fwhmdegs,
                                                                                                       seed=seed,
                                                                                                       spatial_noise=spatial_noise,
                                                                                                       nunu_correlation=nunu_correlation,
                                                                                                       integrate_into_band=integration_into_band
                                                                                                       )
        CMBdust220 = np.transpose(CMBdust220, (0, 2, 1))
        CMBdust_noiseless = np.transpose(CMBdust_noiseless, (0, 2, 1))
        CMBdust_noise = np.transpose(CMBdust_noise, (0, 2, 1))

        for f in range(j):

            R = ComponentSeparation.CompSep(d).fg_buster(CMBdust220,
                                                         [CMB(), Dust(freqs220[f], temp=20.)], freq=freqs220,
                                                         fwhmdeg=fwhmdegs,
                                                         target=fwhm_final, okpix=okpix_inside, Stokesparameter=Stkp)

            print(f, R.x[0])
            if j == 3:
                # array_est_fg_3bands[m, 0, f, :, okpix_inside] = R.s[0, :, :].T
                # array_est_fg_3bands[m, 1, f, :, okpix_inside] = R.s[1, :, :].T
                beta_d_3bands_220.append(R.x[0])
            elif j == 4:
                # array_est_fg_4bands[m, 0, f, :, okpix_inside] = R.s[0, :, :].T
                # array_est_fg_4bands[m, 1, f, :, okpix_inside] = R.s[1, :, :].T
                beta_d_4bands_220.append(R.x[0])
            elif j == 5:
                # array_est_fg_5bands[m, 0, f, :, okpix_inside] = R.s[0, :, :].T
                # array_est_fg_5bands[m, 1, f, :, okpix_inside] = R.s[1, :, :].T
                beta_d_5bands_220.append(R.x[0])
            else:
                pass

        del CMBdust220
        del CMBdust_noiseless
        del CMBdust_noise
        del P

beta_d_3bands_comb = []
beta_d_4bands_comb = []
beta_d_5bands_comb = []
Stkp = 'IQU'

for i, j in enumerate(nband):

    ### Read some stuff

    # Read dictionary chosen
    d150['focal_length'] = 0.3
    d150['nside'] = 256

    npix = 12 * d150['nside'] ** 2
    Nf = int(d150['nf_sub'])
    band = d150['filter_nu'] / 1e9
    filter_relative_bandwidth = d150['filter_relative_bandwidth']

    # Read dictionary chosen
    d220['focal_length'] = 0.3
    d220['nside'] = 256

    # Define the number of reconstruction bands:
    nbands = j
    d150['nf_recon'] = nbands
    d150['nf_sub'] = nbands
    # Define the number of reconstruction bands:
    nbands = j
    d220['nf_recon'] = nbands
    d220['nf_sub'] = nbands

    npix = 12 * d220['nside'] ** 2
    Nf = int(d220['nf_sub'])
    band = d220['filter_nu'] / 1e9
    filter_relative_bandwidth = d220['filter_relative_bandwidth']

    a, nus_edge, nus_in, dp, e, Nbbands_in = qubic.compute_freq(band, Nf, filter_relative_bandwidth)

    cov = pickle.load(open('/pbs/home/m/mregnier/coverage_dtheta_40_pointing_6000.pkl', 'rb'))
    cov150 = cov['coverage']

    okpix_inside = (cov150 > (0.5 * np.max(cov150)))
    cov_qubic = np.loadtxt('/pbs/home/m/mregnier/mask_fgbuster')
    okpix_qubic = cov_qubic != 0

    if j == 3:
        freqs150 = [136.984337, 148.954079, 161.969742]
        fwhmdegs = [0.42999269, 0.39543908, 0.36366215]
    elif j == 4:
        freqs150 = [135.50540829, 144.29216391, 153.6486907, 163.61193508]
        fwhmdegs = [0.43468571, 0.40821527, 0.38335676, 0.36001202]
    elif j == 5:
        freqs150 = [134.63280448, 141.57278853, 148.8705114, 156.54441361, 164.61388627]
        fwhmdegs = [0.43750306, 0.41605639, 0.39566106, 0.37626551, 0.35782075]
    else:
        raise TypeError('Incorrect frequency number')

    if nbands == 3:
        freqs220 = [200.9103609, 218.46598318, 237.55562228]
        fwhmdegs = [0.42999269, 0.39543908, 0.36366215]
    elif nbands == 4:
        freqs220 = [198.74126549, 211.62850707, 225.35141303, 239.96417145]
        fwhmdegs = [0.43468571, 0.40821527, 0.38335676, 0.36001202]
    elif nbands == 5:
        freqs220 = [197.46144657, 207.64008985, 218.34341672, 229.59847329, 241.43369986]
        fwhmdegs = [0.43750306, 0.41605639, 0.39566106, 0.37626551, 0.35782075]
    else:
        raise TypeError('Incorrect frequency number')

    okpix = (cov150 > (0.1 * np.max(cov150)))

    for m in range(nsim):
        seed = m
        sky_config = {'dust': 'd0', 'cmb': seed}
        Qubic_sky_150 = qss.Qubic_sky(sky_config, d150)
        Qubic_sky_220 = qss.Qubic_sky(sky_config, d220)

        CMBdust150, CMBdust_noiseless, CMBdust_noise, P = Qubic_sky_150.get_partial_sky_maps_withnoise(coverage=cov150,
                                                                                                       Nyears=nyears,
                                                                                                       verbose=verbose,
                                                                                                       FWHMdeg=FWHMdeg,
                                                                                                       seed=seed,
                                                                                                       spatial_noise=spatial_noise,
                                                                                                       nunu_correlation=nunu_correlation,
                                                                                                       integrate_into_band=integration_into_band
                                                                                                       )
        CMBdust220, CMBdust_noiseless, CMBdust_noise, P = Qubic_sky_220.get_partial_sky_maps_withnoise(coverage=cov150,
                                                                                                       Nyears=nyears,
                                                                                                       verbose=verbose,
                                                                                                       FWHMdeg=FWHMdeg,
                                                                                                       seed=seed,
                                                                                                       spatial_noise=spatial_noise,
                                                                                                       nunu_correlation=nunu_correlation,
                                                                                                       integrate_into_band=integration_into_band
                                                                                                       )
        CMBdust150 = np.transpose(CMBdust150, (0, 2, 1))
        CMBdust_noiseless = np.transpose(CMBdust_noiseless, (0, 2, 1))
        CMBdust_noise = np.transpose(CMBdust_noise, (0, 2, 1))

        CMBdust220 = np.transpose(CMBdust220, (0, 2, 1))
        CMBdust_noiseless = np.transpose(CMBdust_noiseless, (0, 2, 1))
        CMBdust_noise = np.transpose(CMBdust_noise, (0, 2, 1))

        CMBdust = np.zeros(((2 * j, 3, npix)))
        CMBdust[:j] = CMBdust150
        CMBdust[j:] = CMBdust220

        freqs_combined = freqs150 + freqs220

        fwhmdegs_combined = fwhmdegs + fwhmdegs

        for f in range(2 * j):
            nbands = j * 2
            d150['nf_recon'] = nbands
            d150['nf_sub'] = nbands

            nbands = j * 2
            d220['nf_recon'] = nbands
            d220['nf_sub'] = nbands

            if f < j:
                dic = d150
            else:
                dic = d220

            R = ComponentSeparation.CompSep(dic).fg_buster(CMBdust,
                                                           [CMB(), Dust(freqs_combined[f], temp=20.)],
                                                           freq=freqs_combined, fwhmdeg=fwhmdegs_combined,
                                                           target=fwhm_final, okpix=okpix_inside, Stokesparameter=Stkp)

            print('Reconstruction : ', f, R.x[0])
            if j == 3:
                # array_est_fg_3bands[m, 0, f, :, okpix_inside] = R.s[0, :, :].T
                # array_est_fg_3bands[m, 1, f, :, okpix_inside] = R.s[1, :, :].T
                beta_d_3bands_comb.append(R.x[0])
            elif j == 4:
                # array_est_fg_4bands[m, 0, f, :, okpix_inside] = R.s[0, :, :].T
                # array_est_fg_4bands[m, 1, f, :, okpix_inside] = R.s[1, :, :].T
                beta_d_4bands_comb.append(R.x[0])
            elif j == 5:
                # array_est_fg_5bands[m, 0, f, :, okpix_inside] = R.s[0, :, :].T
                # array_est_fg_5bands[m, 1, f, :, okpix_inside] = R.s[1, :, :].T
                beta_d_5bands_comb.append(R.x[0])
            else:
                pass

        del CMBdust150
        del CMBdust220
        del CMBdust
        del CMBdust_noiseless
        del CMBdust_noise
        del P

beta_d_3bands_220 = beta_d_3bands_220[::3]
beta_d_4bands_220 = beta_d_4bands_220[::4]
beta_d_5bands_220 = beta_d_5bands_220[::5]

beta_d_3bands_150 = beta_d_3bands_150[::3]
beta_d_4bands_150 = beta_d_4bands_150[::4]
beta_d_5bands_150 = beta_d_5bands_150[::5]

beta_d_3bands_comb = beta_d_3bands_comb[::3]
beta_d_4bands_comb = beta_d_4bands_comb[::4]
beta_d_5bands_comb = beta_d_5bands_comb[::5]

# In[ ]:


mean_beta_3bands_150 = np.mean(beta_d_3bands_150)
mean_beta_4bands_150 = np.mean(beta_d_4bands_150)
mean_beta_5bands_150 = np.mean(beta_d_5bands_150)

std_beta_3bands_150 = np.std(beta_d_3bands_150)
std_beta_4bands_150 = np.std(beta_d_4bands_150)
std_beta_5bands_150 = np.std(beta_d_5bands_150)

mean_beta_3bands_220 = np.mean(beta_d_3bands_220)
mean_beta_4bands_220 = np.mean(beta_d_4bands_220)
mean_beta_5bands_220 = np.mean(beta_d_5bands_220)

std_beta_3bands_220 = np.std(beta_d_3bands_220)
std_beta_4bands_220 = np.std(beta_d_4bands_220)
std_beta_5bands_220 = np.std(beta_d_5bands_220)

mean_beta_3bands_comb = np.mean(beta_d_3bands_comb)
mean_beta_4bands_comb = np.mean(beta_d_4bands_comb)
mean_beta_5bands_comb = np.mean(beta_d_5bands_comb)

std_beta_3bands_comb = np.std(beta_d_3bands_comb)
std_beta_4bands_comb = np.std(beta_d_4bands_comb)
std_beta_5bands_comb = np.std(beta_d_5bands_comb)

print("Saving...")
data_dict = {'beta_d_150_3bands': beta_d_3bands_150[::3], 'beta_d_150_4bands': beta_d_4bands_150[::4],
             'beta_d_150_5bands': beta_d_5bands_150[::5], 'beta_d_220_3bands': beta_d_3bands_220[::3],
             'beta_d_220_4bands': beta_d_4bands_220[::4], 'beta_d_220_5bands': beta_d_5bands_220[::5],
             'beta_d_combined_3bands': beta_d_3bands_comb[::3], 'beta_d_combined_4bands': beta_d_3bands_comb[::4],
             'beta_d_combined_5bands': beta_d_5bands_comb[::5]}
files = open(sys.argv[2] + 'MC_{}_beta_d_all_bands_{}realisations.pkl'.format(int(sys.argv[2]), nsim), 'wb')
pickle.dump(data_dict, files)
files.close()
print("Saved")
