from __future__ import division
from pyoperators import pcg
from pysimulators import profile

#### QUBIC packages ####
import qubic
import pymaster as nmt
from qubicpack.utilities import Qubic_DataDir
from qubic import QubicSkySim as qss
from pysimulators import FitsArray
from qubic import fibtools as ft
from qubic import camb_interface as qc
from qubic import SpectroImLib as si
from qubic import mcmc
from qubic import AnalysisMC as amc

import os
import sys
import os.path as op
sys.path.append('/pbs/home/t/tlaclave/mypackages')
import instrument as instr
import frequency_acquisition as Acq

# Display packages
import healpy as hp
import matplotlib.pyplot as plt
import gc

# General packages
import numpy as np
import importlib
import camb
from astropy.io import fits
from pyoperators import *
import copy
import random
import configparser

#### Function to read the config file and extract informations ####
def load_config(config_file):
    # Créer un objet ConfigParser
    config = configparser.ConfigParser()

    # Lire le fichier de configuration
    config.read(config_file)

    # Itérer sur chaque section et option
    external = []
    global_variables = {}
    for section in config.sections():
        for option in config.options(section):
            
            # Récupérer la valeur de chaque option de configuration
            value = config.get(section, option)

            # Convertir la valeur en int, float ou bool si c'est possible
            if value.isdigit():
                value = int(value)
            elif '.' in value and all(part.isdigit() for part in value.split('.')):
                value = float(value)
            elif value.lower() in ['true', 'false']:
                value = value.lower() == 'true'

            # Définir chaque option de configuration en tant que variable globale
            globals()[option] = value

            global_variables[str(option)] = value
            
    return external, global_variables

#### PyOperators packages ####
from pyoperators import (
    BlockColumnOperator, BlockDiagonalOperator, BlockRowOperator,
    CompositionOperator, DiagonalOperator, I, IdentityOperator,
    MPIDistributionIdentityOperator, MPI, proxy_group, ReshapeOperator,
    rule_manager, pcg, Operator)

from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator


#### Load the global variables in the config file ####
external = load_config('config_mapmaking.ini')
print(sent_job)
if sent_job == True:
    id_index = int(sys.argv[1])
    print(id_index)
    qubic_config = f'{str(sys.argv[2])}'
    print(qubic_config)

dict_parameters = external[1]
print(dict_parameters)

#### Acquisition file ####
importlib.reload(Acq)

#### fill the dict ####
dictfilename = 'dicts/pipeline_demo.dict'

#### Read dictionary chosen ####
d_TOD = qubic.qubicdict.qubicDict()
d_TOD.read_from_file(dictfilename)

d_TOD['nf_recon'] = 1
d_TOD['nf_sub'] = 1
d_TOD['nside'] = nside
npix = 12*nside**2
d_TOD['RA_center'] = int(ra)
d_TOD['DEC_center'] = int(dec)
center = qubic.equ2gal(d_TOD['RA_center'], d_TOD['DEC_center'])
d_TOD['effective_duration'] = effective_duration
d_TOD['npointings'] = npointings
d_TOD['filter_nu'] = band * 1e9
d_TOD['photon_noise'] = photon_noise
d_TOD['noiseless'] = noiseless
d_TOD['config'] = 'FI'
d_TOD['MultiBand'] = True
d_TOD['planck'] = True
d_TOD['dtheta'] = 15

#### Usefull variable to name the plots + chose where the results are saved ####
sub_band_number = nf_recon
path = str(dir_name)
if beta == 'None':
    beta = None

#### Create the two dicts for 150 & 220 GHz ####
d150 = d_TOD
d220 = copy.deepcopy(d_TOD)
d220['filter_nu'] = 220 * 1e9

#### Qubic acquisitions ####
nf_tod = nf_recon*fact_sub
qubic_acquisition_150 = Acq.QubicIntegrated(d150, Nsub=nf_tod, Nrec=nf_recon)
qubic_acquisition_220 = Acq.QubicIntegrated(d220, Nsub=nf_tod, Nrec=nf_recon)

#### Create sky config ####
if sky == 'cmb':
    sky_config = {'cmb':1}
    sky_name = 'CMB'
    bandpass = False
elif sky == 'dust':
    sky_config = {'cmb':1, 'dust':'d0'}
    sky_name = 'CMB_Dust'
    bandpass = True

#### Create the wide band or txo bands acquisition ####
if qubic_config == 'wide':
    qubic_acquisition = Acq.QubicWideBand(qubic_acquisition_150, qubic_acquisition_220)
elif qubic_config == 'two':
    qubic_acquisition = Acq.QubicTwoBands(qubic_acquisition_150, qubic_acquisition_220)

#### Create Nsub sky_maps ####
sky = Acq.Sky(sky_config, qubic_acquisition)
sky_nu = sky.scale_component(beta)

#### Create QUBIC TOD ####
TOD_QUBIC = qubic_acquisition.get_TOD(sky_config, beta = beta, noise = noise_qubic, bandpass_correction = bandpass)
#TOD_QUBIC_noiseless = qubic_acquisition.get_TOD(sky_config, beta = beta, noise = False, bandpass_correction = bandpass)

#### Create the QUBIC scene ####
scene = qubic_acquisition.scene

#### Create coverage ####
coverage = qubic_acquisition.get_coverage()
seenpix = coverage/coverage.max() > seenpix_lim

#### Planck acquisitions ####
planck_acquisition_143 = Acq.PlanckAcquisition(143, scene)
planck_acquisition_217 = Acq.PlanckAcquisition(217, scene)

#### Create Planck TOD for each frequency ####
planck_143 = np.mean(sky_nu[:nf_tod], axis=0).ravel()
planck_217 = np.mean(sky_nu[nf_tod:2*nf_tod], axis=0).ravel()
# Create Planck noise for each frequency
planck_noise_143 = planck_acquisition_143.get_noise().ravel()
planck_noise_217 = planck_acquisition_217.get_noise().ravel()
# Build the TODs
TOD_PLANCK_143 = planck_143 + planck_noise_143 * noise_planck_level
#TOD_PLANCK_143_noiseless = planck_143 + planck_noise_143 * 0
TOD_PLANCK_217 = planck_217 + planck_noise_217 * noise_planck_level
#TOD_PLANCK_217_noiseless = planck_217 + planck_noise_217 * 0

### Create TODs noisy ####
if qubic_config == 'wide':
    TOD = TOD_QUBIC.ravel()
    TOD = np.r_[TOD, TOD_PLANCK_143]
    TOD = np.r_[TOD, TOD_PLANCK_217]
    
elif qubic_config == 'two':
    TOD_QUBIC_150 = TOD_QUBIC[:992]
    TOD_QUBIC_220 = TOD_QUBIC[992:2*992]
    TOD = TOD_QUBIC_150.ravel()
    TOD = np.r_[TOD, TOD_PLANCK_143]
    TOD = np.r_[TOD, TOD_QUBIC_220.ravel()]
    TOD = np.r_[TOD, TOD_PLANCK_217]

else:
    print('########### error qubic config ###########')

TOD = np.array(TOD)

### Create TODs noiseless ####
# if qubic_config == 'wide':
#     TOD_noiseless = TOD_QUBIC_noiseless.ravel()
#     TOD_noiseless = np.r_[TOD_noiseless, TOD_PLANCK_143_noiseless]
#     TOD_noiseless = np.r_[TOD_noiseless, TOD_PLANCK_217_noiseless]

# elif qubic_config == 'two':
#     TOD_QUBIC_150_noiseless = TOD_QUBIC_noiseless[:992]
#     TOD_QUBIC_220_noiseless = TOD_QUBIC_noiseless[992:2*992]
#     TOD_noiseless = TOD_QUBIC_150_noiseless.ravel()
#     TOD_noiseless = np.r_[TOD_noiseless, TOD_PLANCK_143_noiseless]
#     TOD_noiseless = np.r_[TOD_noiseless, TOD_QUBIC_220_noiseless.ravel()]
#     TOD_noiseless = np.r_[TOD_noiseless, TOD_PLANCK_217_noiseless]
    
# TOD_noiseless = np.array(TOD_noiseless)

#### QUBIC Planck Joint acquisition ####
if qubic_config == "wide":
    qubic_config = "Wide_Band"
    qubicplanck_acquisition = Acq.QubicPlanckMultiBandAcquisition(qubic_acquisition, [planck_acquisition_143, planck_acquisition_217])
elif qubic_config == "two":
    qubic_config = "Two_Bands"
    qubicplanck_acquisition = Acq.QubicPlanckMultiBandAcquisition(qubic_acquisition, [planck_acquisition_143, planck_acquisition_217])

#### Create operators ####
H = qubicplanck_acquisition.get_operator()
invN = qubicplanck_acquisition.get_invntt_operator()

#### Build A & b objects ####
A = H.T * invN * H
b_noise = H.T * invN * TOD
#b_noiseless = H.T * invN * TOD_noiseless

#### Find the solution with PCG ####
tol = 10**(-tol)
if sky_ini == 'None':
    sky_ini = None
else :
    sky_ini = sky_nu
solution_pcg_noise = pcg(A, b_noise, x0=sky_ini, disp=True, tol=tol, maxiter=max_iter)
solution_noise = solution_pcg_noise['x']

#solution_pcg_noiseless = pcg(A, b_noiseless, x0=sky_ini, disp=True, tol=tol, maxiter=max_iter)
#solution_noiseless = solution_pcg_noiseless['x']

#### Create the correct frequency for the sky 
sky_sub = []
devider = nf_tod // nf_recon
devide_indice = 0
for index in range(nf_recon*2):
    sky_sub.append(np.mean(sky_nu[devide_indice: devide_indice + devider]))
    devide_indice += devider

#### Create a list which will contains residuals ####
residual = (solution_noise - sky_nu[:nf_recon])

#### Frequency map making ####
# m_sub[:, ~seenpix, :] = hp.UNSEEN
# solution['x'][~seenpix, :] = hp.UNSEEN
stk = ['I', 'Q', 'U']
sub_band = list(str(sub_band))
if plot_save == "True":
    for sub_band_index in range(len(sub_band)):
        if sub_band[sub_band_index] == '1':
            plt.figure(figsize=(12, 12))
            k=0
            reso=25
            for i in range(3):
                if i == 0:
                    m = 300
                    minr = 20
                else:
                    m = 8
                    minr = 8
            
                hp.gnomview(sky_nu[sub_band_index, :, i], rot=center, reso=reso, cmap='jet', sub=(3, 3, k+1), min=-m, max=m, title=f'{stk[i]} - Input')
                hp.gnomview(solution_noise[sub_band_index, :, i], rot=center, reso=reso, cmap='jet', sub=(3, 3, k+2), min=-m, max=m, title=f'{stk[i]} - Output')
                hp.gnomview(residual[sub_band_index, :, i], rot=center, reso=reso, cmap='jet', sub=(3, 3, k+3), min=-minr, max=minr, title=f'{stk[i]} - Residuals')
                k+=3

            plt.suptitle(f'Frequency Map - {qubic_config} - ' + f'{sky_name} - ' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number}_sub_bands' + f' - {round(qubic_acquisition.nueff[sub_band_index],1)} GHz', fontsize = 10, va = 'center')
            plt.savefig(path + 'map/' + f'map_{qubic_config}_' + f'{sky_name}_' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number}_sub_bands' + f'_{round(qubic_acquisition.nueff[sub_band_index], 1)}GHz.png')

#### RMS ####
rms_profil_list = []
for sub_band_index in range(len(sub_band)):
    if sub_band[sub_band_index] == '1':
        plt.figure()
        rms_profil = qss.get_angular_profile(np.array([residual[sub_band_index][ :, 0], residual[sub_band_index][ :, 1], residual[sub_band_index][ :, 2]]).T, nbins=30, separate=True, center=center, thmax=30)
        rms_profil_list.append(rms_profil)
        if plot_save == "True":
            xx, yyI, yyQ, yyU = rms_profil
            plt.plot(xx, yyI, 'r.', label='I')
            plt.plot(xx, yyQ, 'b*', label='Q')
            plt.plot(xx, yyU, 'gs', label='U')
            plt.xlabel("Degrees")
            plt.ylabel('RMS')
            plt.legend()
            plt.title(f'RMS - {qubic_config} - ' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number}_sub_bands_' + f'{sky_name}' + f'_{round(qubic_acquisition.nueff[sub_band_index], 1)}GHz')
            plt.savefig(path + 'rms/' + f'RMS_noise_{qubic_config}_' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number}_sub_bands_' + f'{sky_name}' + f'_{round(qubic_acquisition.nueff[sub_band_index], 1)}GHz.png')

# Planck's pixels set at 0
# need to set "seenpix_lim" at a higher value
for pix in range(len(seenpix)):
    if seenpix[pix] == False:
        for stk_ind in range(3):
            for sb in range(2*nf_recon):
                residual[sb,pix,stk_ind] = 0
#                residual_noiseless[sb, pix, stk_ind] = 0

#### Covariance Matrix ####
cov_res = np.zeros((nf_recon*2, nf_recon*2, 3))
print(np.shape(cov_res), np.shape(residual))
fig, ax = plt.subplots(1, 3, figsize = (15, 8))
for idx in range(3):
    ax[idx].set_xticks(np.arange(nf_recon*2))
    ax[idx].set_yticks(np.arange(nf_recon*2))
    cov_res[:,:,idx] = np.cov(residual[:,:,idx])
    ax[idx].imshow(cov_res[:,:,idx])
    ax[idx].set_xlabel('sub_bands', fontsize = 20)
    ax[idx].set_ylabel('sub_bands', fontsize = 20)
    ax[idx].set_title(f'{stk[idx]} - Residual', fontsize = 30)
    for (i, j), z in np.ndenumerate(cov_res[:,:,idx]):
        ax[idx].text(j, i, '{:0.5f}'.format(z), ha='center', va='center')
plt.tight_layout()
plt.suptitle(f'Covariance Matrix - Residuals - {qubic_config} - ' + f'{sky_name} - ' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number} sub bands', fontsize = 25, va = 'center')

if plot_save == "True":
    plt.savefig(path + 'cov_matrix/' + f'cov_{qubic_config}_' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number}_sub_bands_' + f'{sky_name}.png')

#### Covariance Matrix of the difference between solution with noise and solution noiseless ####
# sol_diff = solution_noise - solution_noiseless
# cov_res = np.zeros((nf_recon*2, nf_recon*2, 3))

# fig, ax = plt.subplots(1, 3, figsize = (18, 18))
# for idx in range(3):
#     ax[idx].set_xticks(np.arange(nf_recon*2))
#     ax[idx].set_yticks(np.arange(nf_recon*2))
#     cov_res[:,:,idx] = np.cov(sol_diff[:,:,idx])
#     ax[idx].imshow(cov_res[:,:,idx])
#     ax[idx].set_xlabel('sub_bands', fontsize = 20)
#     ax[idx].set_ylabel('sub_bands', fontsize = 20)
#     ax[idx].set_title(f'{stk[idx]} - Residual', fontsize = 30)
#     for (i, j), z in np.ndenumerate(cov_res[:,:,idx]):
#         ax[idx].text(j, i, '{:0.5f}'.format(z), ha='center', va='center')
# plt.suptitle(f'Covariance Matrix - Diff CMB+Dust - noiseless - {qubic_config} - ' + f'{sky_name} - ' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number} sub bands', fontsize = 25, va = 'center')
# plt.tight_layout()
# plt.savefig(path + 'cov_matrix/' + f'Diff_sol_{qubic_config}_' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number}_sub_bands_' + f'{sky_name}.png')

#### Correlation Matrix ####
corr_res = np.zeros((nf_recon*2, nf_recon*2, 3))

fig, ax = plt.subplots(1, 3, figsize = (15, 8))
for idx in range(3):
    ax[idx].set_xticks(np.arange(nf_recon*2))
    ax[idx].set_yticks(np.arange(nf_recon*2))
    corr_res[:,:,idx] = np.corrcoef(residual[:,:,idx])
    ax[idx].imshow(corr_res[:,:,idx])
    ax[idx].set_xlabel('sub_bands', fontsize = 20)
    ax[idx].set_ylabel('sub_bands', fontsize = 20)
    ax[idx].set_title(f'{stk[idx]} - Residual', fontsize = 30)
    for (i, j), z in np.ndenumerate(corr_res[:,:,idx]):
        ax[idx].text(j, i, '{:0.5f}'.format(z), ha='center', va='center')
plt.tight_layout()
plt.suptitle(f'Correlation Matrix - Residuals - {qubic_config} - ' + f'{sky_name} - ' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number} sub bands', fontsize = 25, va = 'center')

if plot_save == "True":
    plt.savefig(path + 'corr_matrix/' + f'corr_{qubic_config}_' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number}_sub_bands_' + f'{sky_name}.png')

if plot_save != 'True':
    print("No plot asked !")

#### Export the data ####
import pickle
mydict = {'sky':sky_sub,
         'solution_noise':solution_noise,
         'residual':residual,
         'Nf_recon':nf_recon,
         'fact_sub':fact_sub,
         'coverage':coverage,
         'parameters':dict_parameters,
         'frequencies':qubic_acquisition.nueff,
         'RMS':rms_profil_list,
         'covariance':cov_res,
         'correlation':corr_res}

output = open(path + 'data/' + f'{qubic_config}_' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_' + f'{sub_band_number}_sub_bands_' + f'{sky_name}_' + f'{id_index}.pkl', 'wb')
pickle.dump(mydict, output)
output.close()
print(qubic_acquisition.nueff)