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

#### Create the two dicts for 150 & 220 GHz ####
d150 = d_TOD
d220 = copy.deepcopy(d_TOD)
d220['filter_nu'] = 220 * 1e9

#### Create the true frequencies for sub maps ####
# if nf_tod != 1 :
#     qubic_freq_150 = qubic.compute_freq(150, Nfreq=nf_recon)[1]
#     qubic_freq_220 = qubic.compute_freq(220, Nfreq=nf_recon)[1]
#     nus_150, nus_220, nus = [], [], []
#     for index in range(nf_recon):
#             nus_150.append(0.5*(qubic_freq_150[index + 1] + qubic_freq_150[index]))
#             nus_220.append(0.5*(qubic_freq_220[index + 1] + qubic_freq_220[index]))
#     nus = nus_150 + nus_220
# else : 
#     nus = [150, 220]f

#### Qubic acquisitions ####
nf_tod = nf_recon*fact_sub
qubic_acquisition_150_tod = Acq.QubicIntegrated(d150, Nsub=nf_tod, Nrec=nf_tod, integration = '')
qubic_acquisition_220_tod = Acq.QubicIntegrated(d220, Nsub=nf_tod, Nrec=nf_tod, integration = '')
qubic_acquisition_150_rec = Acq.QubicIntegrated(d150, Nsub=nf_tod, Nrec=nf_recon, integration = '')
qubic_acquisition_220_rec = Acq.QubicIntegrated(d220, Nsub=nf_tod, Nrec=nf_recon, integration = '')

#### Create sky config ####
if sky == 'cmb':
    sky_config = {'cmb':1}
    sky_name = 'CMB'
elif sky == 'dust':
    sky_config = {'cmb':1, 'dust':'d0'}
    sky_name = 'CMB_Dust'

#### Create the wide band or txo bands acquisition ####
if qubic_config == 'wide':
    qubic_acquisition_tod = Acq.QubicWideBand(qubic_acquisition_150_tod, qubic_acquisition_220_tod)
    qubic_acquisition_rec = Acq.QubicWideBand(qubic_acquisition_150_rec, qubic_acquisition_220_rec)
elif qubic_config == 'two':
    qubic_acquisition_tod = Acq.QubicTwoBands(qubic_acquisition_150_tod, qubic_acquisition_220_tod)
    qubic_acquisition_rec = Acq.QubicTwoBands(qubic_acquisition_150_rec, qubic_acquisition_220_rec)

#### Create Nsub map ####
m_sub = qubic_acquisition_150_tod.get_PySM_maps(sky_config, nus = qubic_acquisition_tod.nueff)

#### Create the QUBIC scene ####
scene = qubic_acquisition_tod.scene

#### Create coverage ####
coverage = qubic_acquisition_tod.get_coverage()
seenpix = coverage/coverage.max() > seenpix_lim

#### Planck acquisitions ####
planck_acquisition_150 = Acq.PlanckAcquisition(143, scene)
planck_acquisition_220 = Acq.PlanckAcquisition(217, scene)

#### QUBIC Planck Joint acquisition ####
if qubic_config == "wide":
    qubic_config = "Wide_Band"
    qubicplanck_acquisition_tod = Acq.QubicPlanckMultiBandAcquisition(qubic_acquisition_tod, [planck_acquisition_150, planck_acquisition_220])
    qubicplanck_acquisition_rec = Acq.QubicPlanckMultiBandAcquisition(qubic_acquisition_rec, [planck_acquisition_150, planck_acquisition_220])
elif qubic_config == "two":
    qubic_config = "Two_Bands"
    qubicplanck_acquisition_tod = Acq.QubicPlanckMultiBandAcquisition(qubic_acquisition_tod, [planck_acquisition_150, planck_acquisition_220])
    qubicplanck_acquisition_rec = Acq.QubicPlanckMultiBandAcquisition(qubic_acquisition_rec, [planck_acquisition_150, planck_acquisition_220])

#### QUBIC Planck operators TOD ####
H_qp_tod = qubicplanck_acquisition_tod.get_operator()
qubicplanck_noise = qubicplanck_acquisition_tod.get_noise()

#### QUBIC Planck operators ####
H_qp = qubicplanck_acquisition_rec.get_operator()
invN_qp = qubicplanck_acquisition_rec.get_invntt_operator()

#### Build the TOD ####
TOD_noise = H_qp_tod(m_sub) + qubicplanck_noise * noise_qubicplanck_level
TOD_noiseless = H_qp_tod(m_sub) + qubicplanck_noise * 1e-15

#### Build A & b objects ####
A = H_qp.T * invN_qp * H_qp
b_noise = H_qp.T * invN_qp * TOD_noise
b_noiseless = H_qp.T * invN_qp * TOD_noiseless

#### Find the solution with PCG ####
tol = 10**(-tol)
if m_sub_ini == 'None':
    m_sub_ini = None
else :
    m_sub_ini = m_sub
solution_pcg_noise = pcg(A, b_noise, x0=m_sub_ini, disp=True, tol=tol, maxiter=max_iter)
solution_noise = solution_pcg_noise['x']

solution_pcg_noiseless = pcg(A, b_noiseless, x0=m_sub_ini, disp=True, tol=tol, maxiter=max_iter)
solution_noiseless = solution_pcg_noiseless['x']

#### Create a list which will contains residuals ####
residual_noise = solution_noise - m_sub
residual_noiseless = solution_noiseless - m_sub

#### Export the data ####
import pickle
mydict = {'sky':m_sub,
         'solution_noise':solution_noise,
         'solution_noiseless':solution_noiseless,
         'Nf_TOD':nf_tod,
         'Nf_recon':nf_recon,
         'fact_sub':fact_sub,
         'parameters':dict_parameters,
         'frequencies':qubic_acquisition.nueff}

output = open(path + 'data/' + f'{qubic_config}_' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_' + f'{sub_band_number}_sub_bands_' + f'{sky_name}.pkl', 'wb')
pickle.dump(mydict, output)
output.close()


#### Frequency map making ####
# m_sub[:, ~seenpix, :] = hp.UNSEEN
# solution['x'][~seenpix, :] = hp.UNSEEN
stk = ['I', 'Q', 'U']
sub_band = list(str(sub_band))
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
        
            hp.gnomview(m_sub[sub_band_index, :, i], rot=center, reso=reso, cmap='jet', sub=(3, 3, k+1), min=-m, max=m, title=f'{stk[i]} - Input')
            hp.gnomview(solution_noise[sub_band_index, :, i], rot=center, reso=reso, cmap='jet', sub=(3, 3, k+2), min=-m, max=m, title=f'{stk[i]} - Output')
            hp.gnomview(residual_noise[sub_band_index, :, i], rot=center, reso=reso, cmap='jet', sub=(3, 3, k+3), min=-minr, max=minr, title=f'{stk[i]} - Residuals')
            k+=3

        plt.suptitle(f'Frequency Map - {qubic_config} - ' + f'{sky_name} - ' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number}_sub_bands' + f' - {round(qubic_acquisition.nueff[sub_band_index])} GHz', fontsize = 10, va = 'center')
        plt.savefig(path + 'map/' + f'map_{qubic_config}_' + f'{sky_name}_' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number}_sub_bands'+ f'_{round(qubic_acquisition.nueff[sub_band_index], 1)}GHz.png')

#### RMS ####
for sub_band_index in range(len(sub_band)):
    if sub_band[sub_band_index] == '1':
        plt.figure()
        xx, yyI, yyQ, yyU = qss.get_angular_profile(np.array([residual_noise[sub_band_index, :, 0], residual_noise[sub_band_index, :, 1], residual_noise[sub_band_index, :, 2]]).T, nbins=30, separate=True, center=center, thmax=30)
        plt.plot(xx, yyI, 'r.', label='I')
        plt.plot(xx, yyQ, 'b*', label='Q')
        plt.plot(xx, yyU, 'gs', label='U')
        plt.xlabel("Degrees")
        plt.ylabel('RMS')
        plt.legend()
        plt.title(f'RMS - {qubic_config} - ' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number}_sub_bands_' + f'{sky_name}')
        plt.savefig(path + 'rms/' + f'RMS_noise_{qubic_config}_' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number}_sub_bands_' + f'{sky_name}.png')

for pix in range(len(seenpix)):
    if seenpix[pix] == False:
        for stk_ind in range(3):
            for sb in range(fact_sub*nf_recon):
                residual_noise[sb, pix, stk_ind] = 0
                residual_noiseless[sb, pix, stk_ind] = 0

#### Covariance Matrix ####
if nf_tod != 1:
    cov_res = np.zeros((3,fact_sub*nf_recon, fact_sub*nf_recon))
else :
    cov_res = np.zeros((3, 2, 2))

fig, ax = plt.subplots(1, 3, figsize = (12, 12))
for idx in range(3):
    ax[idx].set_xticks(np.arange(fact_sub*nf_recon))
    ax[idx].set_yticks(np.arange(fact_sub*nf_recon))
    cov_res[idx, :, :] = np.cov(residual_noise[:, :, idx])
    ax[idx].imshow(cov_res[idx,:,:])
    ax[idx].set_xlabel('sub_bands', fontsize = 20)
    ax[idx].set_ylabel('sub_bands', fontsize = 20)
    ax[idx].set_title(f'{stk[idx]} - Residual', fontsize = 30)
    for (i, j), z in np.ndenumerate(cov_res[idx, :, :]):
        ax[idx].text(j, i, '{:0.5f}'.format(z), ha='center', va='center')
plt.tight_layout()
plt.suptitle(f'Covariance Matrix - Residuals - {qubic_config} - ' + f'{sky_name} - ' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number} sub bands', fontsize = 25, va = 'center')
plt.tight_layout()
plt.savefig(path + 'cov_matrix/' + f'cov_{qubic_config}_' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number}_sub_bands_' + f'{sky_name}.png')

#### Covariance Matrix of the difference between solution with noise and solution noiseless ####
sol_diff = solution_noise - solution_noiseless
if nf_tod != 1:
    cov_sol_noisy_noiseless = np.zeros((3,fact_sub*nf_recon, fact_sub*nf_recon))
else :
    cov_sol_noisy_noiseless = np.zeros((3, 2, 2))

fig, ax = plt.subplots(1, 3, figsize = (18, 18))
for idx in range(3):
    ax[idx].set_xticks(np.arange(fact_sub*nf_recon))
    ax[idx].set_yticks(np.arange(fact_sub*nf_recon))
    cov_sol_noisy_noiseless[idx,:,:] = np.cov(sol_diff[:,:,idx])
    ax[idx].imshow(cov_sol_noisy_noiseless[idx, :, :])
    ax[idx].set_xlabel('sub_bands', fontsize = 20)
    ax[idx].set_ylabel('sub_bands', fontsize = 20)
    ax[idx].set_title(f'{stk[idx]} - Solution noise-noiseless', fontsize = 30)
    for (i, j), z in np.ndenumerate(cov_sol_noisy_noiseless[idx, :, :]):
        ax[idx].text(j, i, '{:0.5f}'.format(z), ha='center', va='center')
plt.suptitle(f'Covariance Matrix - Diff CMB+Dust - noiseless - {qubic_config} - ' + f'{sky_name} - ' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number} sub bands', fontsize = 25, va = 'center')
plt.tight_layout()
plt.savefig(path + 'cov_matrix/' + f'Diff_sol_{qubic_config}_' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number}_sub_bands_' + f'{sky_name}.png')

#### Correlation Matrix ####
if nf_tod != 1:
    corr_res = np.zeros((3,fact_sub*nf_recon, fact_sub*nf_recon))
else :
    corr_res = np.zeros((3, 2, 2))

fig, ax = plt.subplots(1, 3, figsize = (15, 15))
for idx in range(3):
    ax[idx].set_xticks(np.arange(fact_sub*nf_recon))
    ax[idx].set_yticks(np.arange(fact_sub*nf_recon))
    corr_res[i,:,:] = np.corrcoef(residual_noise[:, :, i])
    ax[idx].imshow(corr_res[idx,:,:])
    ax[idx].set_xlabel('sub_bands', fontsize = 20)
    ax[idx].set_ylabel('sub_bands', fontsize = 20)
    ax[idx].set_title(f'{stk[idx]} - Residual', fontsize = 30)
    for (i, j), z in np.ndenumerate(corr_res[idx][:][:]):
        ax[idx].text(j, i, '{:0.5f}'.format(z), ha='center', va='center')
plt.tight_layout()
plt.suptitle(f'Correlation Matrix - Residuals - {qubic_config} - ' + f'{sky_name} - ' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number} sub bands', fontsize = 25, va = 'center')
plt.savefig(path + 'corr_matrix/' + f'corr_{qubic_config}_' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number}_sub_bands_' + f'{sky_name}.png')
