#### QUBIC packages ####
import qubic
from qubic import QubicSkySim as qss
from qubicpack.utilities import Qubic_DataDir
center = qubic.equ2gal(0, -57)

import os
import sys
sys.path.append('/sps/qubic/Users/TomLaclavere/mypackages')
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
external = load_config('mapmaking_versus__config.ini')
dict_parameters = external[1]

#### Create name variables from config file ####
if sky_name == 'cmb':
    sky_name = 'CMB'
elif sky_name == 'dust':
    sky_name = 'CMB_Dust'

if qubic_config_1 == "wide":
    qubic_config_1 = "Wide_Band"
elif qubic_config_1 == "two":
    qubic_config_1 = "Two_Bands"

if qubic_config_2 == "wide":
    qubic_config_2 = "Wide_Band"
elif qubic_config_2 == "two":
    qubic_config_2 = "Two_Bands"

# import solutions computed in mapmaking.py
import pickle

pickle_1 = pickle.load(open(dir_name + f'{qubic_config_1}_' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_' + f'{sub_band_number}_sub_bands_' + f'{sky_name}.pkl', 'rb'))
#pickle_1 = pickle.load(open(dir_name + f'{qubic_config_1}_' + f'{effective_duration}_years_' + f'{sub_band_number}_sub_bands_' + f'{sky_name}.pkl', 'rb'))

sky_1 = pickle_1['sky']
solution_noise_1 = pickle_1['solution_noise']
solution_noiseless_1 = pickle_1['solution_noiseless']       
nf_recon_1 = pickle_1['Nf_recon']
fact_sub_1 = pickle_1['fact_sub']
nf_tod_1 = nf_recon_1 * fact_sub_1
dict_parameters_1 = pickle_1['parameters']
residual_1 = solution_noise_1 - sky_1[0:1]
sol_diff_1 = solution_noise_1 - solution_noiseless_1

pickle_2 = pickle.load(open(dir_name + f'{qubic_config_2}_' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_' + f'{sub_band_number}_sub_bands_' + f'{sky_name}.pkl', 'rb'))
#pickle_2 = pickle.load(open(dir_name + f'{qubic_config_2}_' + f'{effective_duration}_years_' + f'{sub_band_number}_sub_bands_' + f'{sky_name}.pkl', 'rb'))
sky_2 = pickle_2['sky']
solution_noise_2 = pickle_2['solution_noise']
solution_noiseless_2 = pickle_2['solution_noiseless']       
nf_recon_2 = pickle_2['Nf_recon']
fact_sub_2 = pickle_2['fact_sub']
nf_tod_2 = nf_recon_2 * fact_sub_2
dict_parameters_2 = pickle_2['parameters']
residual_2 = solution_noise_2 - sky_2[0:1]
sol_diff_2 = solution_noise_2 - solution_noiseless_2

#freq = pickle_1['frequencies']
freq = [150, 220]

if nf_tod_1 != nf_tod_2 or nf_recon_1 != nf_recon_2 or fact_sub_1 != fact_sub_2:
    print('############ INCOMPATIBLE SOLUTIONS ############')
    x = 0
    stop = 10/x

dir = str(dir)
#### RMS ####

for sb_index in range(2*nf_recon_1):
    plt.figure(figsize=(10, 10))
    xx_1, yyI_1, yyQ_1, yyU_1 = qss.get_angular_profile(np.array([residual_1[sb_index, :, 0], residual_1[sb_index, :, 1], residual_1[sb_index, :, 2]]).T, nbins=30, separate=True, center=center, thmax=30)
    xx_2, yyI_2, yyQ_2, yyU_2 = qss.get_angular_profile(np.array([residual_2[sb_index, :, 0], residual_2[sb_index, :, 1], residual_2[sb_index, :, 2]]).T, nbins=30, separate=True, center=center, thmax=30)
    plt.plot(xx_1, yyI_1, 'r--', label=f'I - {qubic_config_1}')
    plt.plot(xx_1, yyQ_1, 'b--', label=f'Q - {qubic_config_1}')
    plt.plot(xx_1, yyU_1, 'g--', label=f'U - {qubic_config_1}')
    plt.plot(xx_2, yyI_2, 'r', label=f'I - {qubic_config_2}')
    plt.plot(xx_2, yyQ_2, 'b', label=f'Q - {qubic_config_2}')
    plt.plot(xx_2, yyU_2, 'g', label=f'U - {qubic_config_2}')
    plt.xlabel("Degrees")
    plt.ylabel('RMS')
    plt.legend()
    plt.title(f'RMS - Residual - {qubic_config_1} vs {qubic_config_2}_' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_' + f'{freq[sb_index]}_GHz_' + f'{sub_band_number}sub_band_' + f'{sky_name}')
    plt.savefig(dir + 'rms/' + f'RMS_noise - Residual - {qubic_config_1}_vs_{qubic_config_2}_' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_' + f'{freq[sb_index]}_GHz_' + f'{sub_band_number}_sub_bands_' + f'{sky_name}.png')

for sb_index in range(2*nf_recon_1):
    plt.figure(figsize=(10, 10))
    xx_1, yyI_1, yyQ_1, yyU_1 = qss.get_angular_profile(np.array([sol_diff_1[sb_index, :, 0], sol_diff_1[sb_index, :, 1], sol_diff_1[sb_index, :, 2]]).T, nbins=30, separate=True, center=center, thmax=30)
    xx_2, yyI_2, yyQ_2, yyU_2 = qss.get_angular_profile(np.array([sol_diff_2[sb_index, :, 0], sol_diff_2[sb_index, :, 1], sol_diff_2[sb_index, :, 2]]).T, nbins=30, separate=True, center=center, thmax=30)
    plt.plot(xx_1, yyI_1, 'r--', label=f'I - {qubic_config_1}')
    plt.plot(xx_1, yyQ_1, 'b--', label=f'Q - {qubic_config_1}')
    plt.plot(xx_1, yyU_1, 'g--', label=f'U - {qubic_config_1}')
    plt.plot(xx_2, yyI_2, 'r', label=f'I - {qubic_config_2}')
    plt.plot(xx_2, yyQ_2, 'b', label=f'Q - {qubic_config_2}')
    plt.plot(xx_2, yyU_2, 'g', label=f'U - {qubic_config_2}')
    plt.xlabel("Degrees")
    plt.ylabel('RMS')
    plt.legend()
    plt.title(f'RMS - Sol_Diff - {qubic_config_1} vs {qubic_config_2}_' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_' + f'{freq[sb_index]}_GHz_' + f'{sub_band_number}sub_band_' + f'{sky_name}')
    plt.savefig(dir + 'rms/' + f'RMS_noise - Sol_Diff - {qubic_config_1}_vs_{qubic_config_2}_' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_' + f'{freq[sb_index]}_GHz_' + f'{sub_band_number}_sub_bands_' + f'{sky_name}.png')
