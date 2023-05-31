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
sys.path.append('/sps/qubic/Users/TomLaclavere/mypackages')
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
external = load_config('compute_cov_config.ini')
dict_parameters = external[1]
print(dict_parameters)

#### import solutions computed in mapmaking.py ####
import pickle
residual_two, residual_wide = [], []
rms_wide, rms_two = [], []
cov_wide, cov_two = [], []
corr_wide, corr_two = [], []

for id_index in range(0, nmb_id):
    pickle_dict = pickle.load(open(dir_name + 'Wide_Band_' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_' + f'{sub_band_number}_sub_bands_' + f'{pointing}_pointing' + f'{sky_name}_' + f'{id_index}.pkl', 'rb'))
    residual_wide.append(pickle_dict['residual'])  
    rms_wide.append(pickle_dict['RMS'])
    cov_wide.append(pickle_dict['covariance'])
    corr_wide.append(pickle_dict['correlation'])
coverage = pickle_dict['coverage']
seenpix = coverage/np.max(coverage) < seenpix_lim
nf_recon = pickle_dict['Nf_recon']

for id_index in range(0, nmb_id):
    pickle_dict = pickle.load(open(dir_name + 'Two_Bands_' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_' + f'{sub_band_number}_sub_bands_' + f'{pointing}_pointing' + f'{sky_name}_' + f'{id_index}.pkl', 'rb'))
    residual_two.append(pickle_dict['residual'])  
    rms_two.append(pickle_dict['RMS'])
    cov_two.append(pickle_dict['covariance'])
    corr_two.append(pickle_dict['correlation'])

#### Compute the mean of the RMS noise ####
sb = [150, 220]
for idx_sb in range(nf_recon*2):
    xx_wide, yyI_wide, yyQ_wide, yyU_wide = [], [], [], []
    xx_two, yyI_two, yyQ_two, yyU_two = [], [], [], []
    for i in range(nmb_id):
        xx_wide.append(rms_wide[i][idx_sb][0])
        yyI_wide.append(rms_wide[i][idx_sb][1])
        yyQ_wide.append(rms_wide[i][idx_sb][2])
        yyU_wide.append(rms_wide[i][idx_sb][3])
        xx_two.append(rms_two[i][idx_sb][0])
        yyI_two.append(rms_two[i][idx_sb][1])
        yyQ_two.append(rms_two[i][idx_sb][2])
        yyU_two.append(rms_two[i][idx_sb][3])
    xx_wide, yyI_wide, yyQ_wide, yyU_wide = np.mean(xx_wide, axis = 0), np.mean(yyI_wide, axis = 0), np.mean(yyQ_wide, axis = 0), np.mean(yyU_wide, axis = 0)
    xx_two, yyI_two, yyQ_two, yyU_two = np.mean(xx_two, axis = 0), np.mean(yyI_two, axis = 0), np.mean(yyQ_two, axis = 0), np.mean(yyU_two, axis = 0)

    plt.figure()
    plt.plot(xx_wide, yyI_wide, 'r.', label='I')
    plt.plot(xx_wide, yyQ_wide, 'b*', label='Q')
    plt.plot(xx_wide, yyU_wide, 'gs', label='U')
    plt.xlabel("Degrees")
    plt.ylabel('RMS')
    plt.legend()
    plt.title('RMS - Wide_Band - ' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number}_sub_bands_' + f'{pointing}_pointing_' + f'{sky_name}' + f'_{sb[idx_sb]}GHz')
    plt.savefig(path + 'RMS_Wide_Band_' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number}_sub_bands_' + f'{pointing}_pointing_' + f'{sky_name}' + f'_{sb[idx_sb]}GHz.png')

    plt.figure()
    plt.plot(xx_two, yyI_two, 'r.', label='I')
    plt.plot(xx_two, yyQ_two, 'b*', label='Q')
    plt.plot(xx_two, yyU_two, 'gs', label='U')
    plt.xlabel("Degrees")
    plt.ylabel('RMS')
    plt.legend()
    plt.title('RMS - Two Bands - ' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number}_sub_bands_' + f'{pointing}_pointing_' + f'{sky_name}' + f'_{sb[idx_sb]}GHz')
    plt.savefig(path + 'RMS_Two_Bands_' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number}_sub_bands_' + f'{pointing}_pointing_' + f'{sky_name}' + f'_{sb[idx_sb]}GHz.png')

#### Compute the mean of the cov & corr matrcies ####
cov_mat_wide, cov_mat_two = np.zeros((nf_recon*2, nf_recon*2, 3)), np.zeros((nf_recon*2, nf_recon*2, 3))
corr_mat_wide, corr_mat_two = np.zeros((nf_recon*2, nf_recon*2, 3)), np.zeros((nf_recon*2, nf_recon*2, 3))

for i in range(nf_recon*2):
    for j in range(nf_recon*2):
        for k in range(3):
            list_wide = []
            list_two = []
            for index_id in range(nmb_id):
                list_wide.append(cov_wide[index_id][i][j][k])
                list_two.append(cov_two[index_id][i][j][k])
            mean_wide = np.mean(list_wide)
            mean_two = np.mean(list_two)
            cov_mat_wide[i][j][k] = mean_wide
            cov_mat_two[i][j][k] = mean_two

for i in range(nf_recon*2):
    for j in range(nf_recon*2):
        for k in range(3):
            list_wide = []
            list_two = []
            for index_id in range(nmb_id):
                list_wide.append(corr_wide[index_id][i][j][k])
                list_two.append(corr_two[index_id][i][j][k])
            mean_wide = np.mean(list_wide)
            mean_two = np.mean(list_two)
            corr_mat_wide[i][j][k] = mean_wide
            corr_mat_two[i][j][k] = mean_two

fig, ax = plt.subplots(1, 3, figsize = (18, 8))
stk = ['I', 'Q', 'U']

for idx in range(3):
    ax[idx].set_xticks(np.arange(nf_recon*2))
    ax[idx].set_yticks(np.arange(nf_recon*2))
    ax[idx].imshow(corr_mat_wide[:,:,idx], cmap = "bwr", vmax = 1, vmin = -1)
    ax[idx].set_xlabel('sub_bands', fontsize = 20)
    ax[idx].set_ylabel('sub_bands', fontsize = 20)
    ax[idx].set_title(f'{stk[idx]} - Residual', fontsize = 30)
    for (i, j), z in np.ndenumerate(corr_mat_wide[:,:,idx]):
        ax[idx].text(j, i, '{:0.5f}'.format(z), ha='center', va='center')
plt.tight_layout()
plt.suptitle('Correlation Matrix - Residuals - Wide Band - ' + f'{sky_name} - ' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_' + f'{sub_band_number}_sub_bands_' + f'{pointing}_pointing', fontsize = 25, va = 'center')
plt.savefig(path + 'corr_Wide_Band_' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_' + f'{sub_band_number}_sub_bands_' + f'{pointing}_pointing_' + f'{sky_name}.png')

fig, ax = plt.subplots(1, 3, figsize = (18, 8))

for idx in range(3):
    ax[idx].set_xticks(np.arange(nf_recon*2))
    ax[idx].set_yticks(np.arange(nf_recon*2))
    ax[idx].imshow(cov_mat_wide[:,:,idx], cmap = "bwr", vmax = 3, vmin = -3)
    ax[idx].set_xlabel('sub_bands', fontsize = 20)
    ax[idx].set_ylabel('sub_bands', fontsize = 20)
    ax[idx].set_title(f'{stk[idx]} - Residual', fontsize = 30)
    for (i, j), z in np.ndenumerate(cov_mat_wide[:,:,idx]):
        ax[idx].text(j, i, '{:0.5f}'.format(z), ha='center', va='center')
plt.tight_layout()
plt.suptitle('Covariance Matrix - Residuals - Wide Band - ' + f'{sky_name} - ' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number}_sub_bands_' + f'{pointing}_pointing', fontsize = 25, va = 'center')
plt.savefig(path + 'cov_Wide_Band_' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number}_sub_bands_' + f'{pointing}_pointing_' + f'{sky_name}.png')

fig, ax = plt.subplots(1, 3, figsize = (18, 8))
for idx in range(3):
    ax[idx].set_xticks(np.arange(nf_recon*2))
    ax[idx].set_yticks(np.arange(nf_recon*2))
    ax[idx].imshow(corr_mat_two[:,:,idx], cmap = "bwr", vmax = 1, vmin = -1)
    ax[idx].set_xlabel('sub_bands', fontsize = 20)
    ax[idx].set_ylabel('sub_bands', fontsize = 20)
    ax[idx].set_title(f'{stk[idx]} - Residual', fontsize = 30)
    for (i, j), z in np.ndenumerate(corr_mat_two[:,:,idx]):
        ax[idx].text(j, i, '{:0.5f}'.format(z), ha='center', va='center')
plt.tight_layout()
plt.suptitle('Correlation Matrix - Residuals - Two Bands - ' + f'{sky_name} - ' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number}_sub_bands_' + f'{pointing}_pointing_' , fontsize = 25, va = 'center')
plt.savefig(path + 'corr_Two_Bands_' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number}_sub_bands_' + f'{pointing}_pointing_' + f'{sky_name}.png')


fig, ax = plt.subplots(1, 3, figsize = (18, 8))
for idx in range(3):
    ax[idx].set_xticks(np.arange(nf_recon*2))
    ax[idx].set_yticks(np.arange(nf_recon*2))
    ax[idx].imshow(cov_mat_two[:,:,idx], cmap = "bwr", vmax = 3, vmin = -3)
    ax[idx].set_xlabel('sub_bands', fontsize = 20)
    ax[idx].set_ylabel('sub_bands', fontsize = 20)
    ax[idx].set_title(f'{stk[idx]} - Residual', fontsize = 30)
    for (i, j), z in np.ndenumerate(cov_mat_two[:,:,idx]):
        ax[idx].text(j, i, '{:0.5f}'.format(z), ha='center', va='center')
plt.tight_layout()
plt.suptitle('Covariance Matrix - Residuals - Two Bands - ' + f'{sky_name} - ' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number}_sub_bands_' + f'{pointing}_pointing_', fontsize = 25, va = 'center')
plt.savefig(path + 'cov_Two_Bands_' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_'  + f'{sub_band_number}_sub_bands_' + f'{pointing}_pointing_' + f'{sky_name}.png')
