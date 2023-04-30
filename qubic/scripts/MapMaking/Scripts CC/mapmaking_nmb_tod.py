#### QUBIC packages ####
import qubic
from qubic import QubicSkySim as qss
from qubicpack.utilities import Qubic_DataDir
center = qubic.equ2gal(0, -57)

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

#### Load the global variables in the config file ####
external = load_config('config_mapmaking_nmb_tod.ini')
dict_parameters = external[1]

#### Create name variables from config file ####
if sky_name == 'cmb':
    sky_name = 'CMB'
elif sky_name == 'dust':
    sky_name = 'CMB_Dust'

if qubic_config == "wide":
    qubic_config = "Wide_Band"
elif qubic_config == "two":
    qubic_config = "Two_Bands"

# import solutions computed in mapmaking.py
import pickle
sky, solution_noise, solution_noiseless, residual, sol_diff, freq = [], [], [], [], [], []
for nmb_tod in range(1, nmb_nf_tod + 1):
    pickle_dict = pickle.load(open(dir_name + f'{qubic_config}_' + f'{effective_duration}_years_' + f'{nmb_tod}_TOD_' + f'{sub_band_number}_sub_bands_' + f'{sky_name}.pkl', 'rb'))
    sky.append(pickle_dict['sky'])
    solution_noise.append(pickle_dict['solution_noise'])
    solution_noiseless.append(pickle_dict['solution_noiseless'])   
    residual.append(pickle_dict['residual'])
    sol_diff.append(pickle_dict['solution_noise'] - pickle_dict['solution_noiseless'])
    freq.append(pickle_dict['frequencies'])

dir = str(dir)

#### RMS ####
yyI, yyQ, yyU = [[],[]], [[],[]], [[], []]
for tod_index in range(nmb_nf_tod):
    for sb_index in range(sub_band_number*2):
        rms = qss.get_angular_profile(np.array([residual[tod_index][sb_index, :, 0], residual[tod_index][sb_index, :, 1], residual[tod_index][sb_index, :, 2]]).T, nbins=30, separate=True, center=center, thmax=30)
        yyI[sb_index].append(np.mean(rms[1][0:8]))
        yyQ[sb_index].append(np.mean(rms[2][0:8]))
        yyU[sb_index].append(np.mean(rms[3][0:8]))

for sb_index in range(sub_band_number*2):
    plt.figure()
    nf_tod = np.arange(nmb_nf_tod)
    plt.xlabel('Nsub')
    plt.ylabel('Noise level')
    plt.plot(nf_tod, yyI[sb_index])
    plt.title('RMS_noise_I_' + f'{effective_duration}_years_' + f'{round(freq[0][sb_index],1)}_GHz_' + f'{sub_band_number}_sub_bands_' + f'{sky_name}')
    plt.savefig(dir + 'RMS_noise_I_' + f'{effective_duration}_years_' + f'{round(freq[0][sb_index],1)}_GHz_' + f'{sub_band_number}_sub_bands_' + f'{sky_name}.png')

for sb_index in range(sub_band_number*2):
    plt.figure()
    nf_tod = np.arange(nmb_nf_tod)
    plt.plot(nf_tod, yyQ[sb_index][:])
    plt.xlabel('Nsub')
    plt.ylabel('Noise level')
    plt.title('RMS_noise_Q_' + f'{effective_duration}_years_' + f'{round(freq[0][sb_index],1)}_GHz_' + f'{sub_band_number}_sub_bands_' + f'{sky_name}')
    plt.savefig(dir + 'RMS_noise_Q_' + f'{effective_duration}_years_' + f'{round(freq[0][sb_index],1)}_GHz_' + f'{sub_band_number}_sub_bands_' + f'{sky_name}.png')

for sb_index in range(sub_band_number*2):
    plt.figure()
    nf_tod = np.arange(nmb_nf_tod)
    plt.xlabel('Nsub')
    plt.ylabel('Noise level')
    plt.plot(nf_tod, yyU[sb_index][:])
    plt.title('RMS_noise_U_' + f'{effective_duration}_years_' + f'{round(freq[0][sb_index],1)}_GHz_' + f'{sub_band_number}_sub_bands_' + f'{sky_name}')
    plt.savefig(dir + 'RMS_noise_U_' + f'{effective_duration}_years_' + f'{round(freq[0][sb_index],1)}_GHz_' + f'{sub_band_number}_sub_bands_' + f'{sky_name}.png')
