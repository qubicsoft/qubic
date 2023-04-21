#### QUBIC packages ####
import qubic
from qubic import QubicSkySim as qss
from qubicpack.utilities import Qubic_DataDir
import os
import sys
sys.path.append('/pbs/home/t/tlaclave/mypackages')

import frequency_acquisition as Acq

#### Display packages ####
import healpy as hp
import matplotlib.pyplot as plt
import gc

#### General packages ####
import numpy as np
import importlib
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
external = load_config('config.ini')
dict_parameters = external[1]

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
#     nus = [150, 220]

#### Qubic acquisitions ####
qubic_acquisition_150 = Acq.QubicIntegrated(d150, Nsub=nf_recon*fact_sub, Nrec=nf_recon)
qubic_acquisition_220 = Acq.QubicIntegrated(d220, Nsub=nf_recon*fact_sub, Nrec=nf_recon)

#### Create sky config ####
if sky == 'cmb':
    sky_config = {'cmb':1}
    sky_name = 'CMB'
elif sky == 'dust':
    sky_config = {'cmb':1, 'dust':'d0'}
    sky_name = 'CMB_Dust'

#### Create the wide band or txo bands acquisition ####
if qubic_config == 'wide':
    qubic_acquisition = Acq.QubicWideBand(qubic_acquisition_150, qubic_acquisition_220)
elif qubic_config == 'two':
    qubic_acquisition = Acq.QubicTwoBands(qubic_acquisition_150, qubic_acquisition_220)

#### Create Nsub map ####
m_sub = qubic_acquisition_150.get_PySM_maps(sky_config, nus = qubic_acquisition.nueff)

#### Create the QUBIC scene ####
scene = qubic_acquisition.scene

#### Create coverage ####
coverage = qubic_acquisition.get_coverage()
seenpix = coverage/coverage.max() > 0.0001

#### Planck acquisitions ####
planck_acquisition_150 = Acq.PlanckAcquisition(143, scene)
planck_acquisition_220 = Acq.PlanckAcquisition(217, scene)

#### QUBIC Planck Joint acquisition ####
if qubic_config == "wide":
    qubic_config = "Wide_Band"
    qubicplanck_acquisition = Acq.QubicPlanckMultiBandAcquisition(qubic_acquisition, [planck_acquisition_150, planck_acquisition_220])
elif qubic_config == "two":
    qubic_config = "Two_Bands"
    qubicplanck_acquisition = Acq.QubicPlanckMultiBandAcquisition(qubic_acquisition, [planck_acquisition_150, planck_acquisition_220])

#### QUBIC Planck operators ####
H_qp = qubicplanck_acquisition.get_operator()
invN_qp = qubicplanck_acquisition.get_invntt_operator()
qubicplanck_noise = qubicplanck_acquisition.get_noise()

#### Build the TOD ####
TOD_noise = H_qp(m_sub) + qubicplanck_noise * noise_qubicplanck_level
TOD_noiseless = H_qp(m_sub) + qubicplanck_noise * 1e-15

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
residual_noise, residual_noiseless = [], []
for stk_index in range(3):
    residuals_noise = solution_noise[:, :, stk_index]-m_sub[:, :, stk_index]
    residual_noise.append(residuals_noise)
    residuals_noiseless = solution_noiseless[:, :, stk_index]-m_sub[:, :, stk_index]
    residual_noiseless.append(residuals_noiseless)

#### Export the data ####
import pickle
mydict = {'sky':m_sub,
         'solution_noise':solution_noise,
         'solution_noiseless':solution_noiseless,
         'Nf_TOD':nf_tod,
         'parameters':dict_parameters}

output = open(path + 'data/' + f'data_{qubic_config}.pkl', 'wb')
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
        
            hp.gnomview(m_sub[0, :, i], rot=center, reso=reso, cmap='jet', sub=(3, 3, k+1), min=-m, max=m, title=f'{stk[i]} - Input')
            hp.gnomview(solution_noise[sub_band_index, :, i], rot=center, reso=reso, cmap='jet', sub=(3, 3, k+2), min=-m, max=m, title=f'{stk[i]} - Output')
            hp.gnomview(residual_noise[i][sub_band_index][:], rot=center, reso=reso, cmap='jet', sub=(3, 3, k+3), min=-minr, max=minr, title=f'{stk[i]} - Residuals')
            k+=3

        plt.suptitle(f'Frequency Map - {qubic_config} - ' + f'{sky_name} - ' + f'{sub_band_number}_sub_bands' + f' - {round(qubic_acquisition.nueff[sub_band_index])} GHz', fontsize = 10, va = 'center')
        plt.savefig(path + 'map/' + f'map_{qubic_config}_' + f'{sky_name}_' + f'{sub_band_number}_sub_bands'+ f'_{round(qubic_acquisition.nueff[sub_band_index], 1)}GHz.png')

#### RMS ####
plt.figure()
xx, yyI, yyQ, yyU = qss.get_angular_profile(np.array([residual_noise[0][0], residual_noise[1][0], residual_noise[2][0]]).T, nbins=30, separate=True, center=center, thmax=30)
plt.plot(xx, yyI, 'r.', label='I')
plt.plot(xx, yyQ, 'b*', label='Q')
plt.plot(xx, yyU, 'gs', label='U')
plt.xlabel("Degrees")
plt.ylabel('RMS')
plt.legend()
plt.title(f'RMS - {qubic_config} - ' + f'{sub_band_number}_sub_bands_' + f'{sky_name}')
plt.savefig(path + 'rms/' + f'RMS_noise_{qubic_config}_' + f'{sub_band_number}_sub_bands_' + f'{sky_name}.png')

#### Covariance Matrix ####
plt.figure(figsize=(15, 15))
cov_res = np.zeros((3,nf_tod,nf_tod))
for i in range(3):
    plt.subplot(1,3,i+1)
    cov_res[i,:,:] = np.cov(residual_noise[i][:][:])
    im = plt.imshow(cov_res[i,:,:])
    plt.xlabel('sub_band', fontsize = 20)
    plt.ylabel('sub_band', fontsize = 20)
    plt.title(stk[i], fontsize = 30)
    plt.colorbar(im,fraction=0.046, pad=0.04)
plt.suptitle(f'Covariance Matrix - Residuals - {qubic_config} - ' + f'{sky_name} - ' + f'{sub_band_number} sub bands', fontsize = 25, va = 'center')
plt.tight_layout()
plt.savefig(path + 'cov_matrix/' + f'cov_{qubic_config}_' + f'{sub_band_number}_sub_bands_' + f'{sky_name}.png')

#### Covariance Matrix of the difference between solution with noise and solution noiseless ####
plt.figure(figsize=(15, 15))
cov_res = np.zeros((3,nf_tod,nf_tod))
sol_diff = solution_noise - solution_noiseless
for i in range(3):
    plt.subplot(1,3,i+1)
    cov_res[i,:,:] = np.cov(sol_diff[:,:,i])
    im = plt.imshow(cov_res[i,:,:])
    plt.xlabel('sub_band', fontsize = 20)
    plt.ylabel('sub_band', fontsize = 20)
    plt.title(stk[i], fontsize = 30)
    plt.colorbar(im,fraction=0.046, pad=0.04)
plt.suptitle(f'Covariance Matrix - Diff CMB+Dust - noiseless - {qubic_config} - ' + f'{sky_name} - ' + f'{sub_band_number} sub bands', fontsize = 25, va = 'center')
plt.tight_layout()
plt.savefig(path + 'cov_matrix/' + f'Diff_sol_{qubic_config}_' + f'{sub_band_number}_sub_bands_' + f'{sky_name}.png')

#### Correlation Matrix ####
plt.figure(figsize=(15, 15))
cov_res = np.zeros((3,nf_tod,nf_tod))
for i in range(3):
    plt.subplot(1,3,i+1)
    cov_res[i,:,:] = np.corrcoef(residual_noise[i][:][:])
    im = plt.imshow(cov_res[i,:,:])
    plt.xlabel('sub_band', fontsize = 20)
    plt.ylabel('sub_band', fontsize = 20)
    plt.title(stk[i], fontsize = 30)
    plt.colorbar(im,fraction=0.046, pad=0.04)
plt.tight_layout()
plt.suptitle(f'Correlation Matrix - Residuals - {qubic_config} - ' + f'{sky_name} - ' + f'{sub_band_number} sub bands', fontsize = 25, va = 'center')
plt.savefig(path + 'corr_matrix/' + f'corr_{qubic_config}_' + f'{sub_band_number}_sub_bands_' + f'{sky_name}.png')

