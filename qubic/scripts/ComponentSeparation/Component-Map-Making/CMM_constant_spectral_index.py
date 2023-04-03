# QUBIC packages
import qubic
from qubicpack.utilities import Qubic_DataDir
from qubic.data import PATH
from qubic.io import read_map
from qubic import QubicSkySim as qss
import sys
sys.path.append('/home/regnier/work/regnier/mypackages')

import Acquisition as Acq
import pickle

# Display packages
import healpy as hp
import matplotlib.pyplot as plt

# FG-Buster packages
import component_model as c
import mixing_matrix as mm

# General packages
import numpy as np
import pysm3
import warnings
from qubic import QubicSkySim as qss
import pysm3.units as u
from importlib import reload
from pysm3 import utils
import os
import iterate
from scipy.optimize import minimize
import ComponentsMapMakingTools as CMM
import FitMultiProcessing as fmp
from functools import partial
import multiprocess as mp
import time
import os.path as op
import configparser

# PyOperators packages
from pyoperators import *
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

warnings.filterwarnings("ignore")
path = '/Users/mregnier/Desktop/PhD Regnier/MapMaking/bash'


C_1degree = HealpixConvolutionGaussianOperator(fwhm=np.deg2rad(1))
C_2degree = HealpixConvolutionGaussianOperator(fwhm=np.deg2rad(2))


### Reading and loading configuration file
def load_config(config_file):
    # Créer un objet ConfigParser
    config = configparser.ConfigParser()

    # Lire le fichier de configuration
    config.read(config_file)

    # Itérer sur chaque section et option
    external = []
    allnus = [30, 44, 70, 143, 217, 353]
    k = 0
    for section in config.sections():
        for option in config.options(section):
            
            # Récupérer la valeur de chaque option de configuration
            value = config.get(section, option)
                
            if section == 'EXTERNAL DATA':
                if value.lower() == 'true':
                    external.append(allnus[k])
                
                k+=1

            # Convertir la valeur en liste si elle est de la forme "1, 2, 3"
            if ',' in value:
                value = [x.strip() for x in value.split(',')]

            # Convertir la valeur en int, float ou bool si c'est possible
            elif value.isdigit():
                value = int(value)
            elif '.' in value and all(part.isdigit() for part in value.split('.')):
                value = float(value)
            elif value.lower() in ['true', 'false']:
                value = value.lower() == 'true'

            # Définir chaque option de configuration en tant que variable globale
            globals()[option] = value
            
    return external

#########################################################################################################
############################################## Arguments ################################################
#########################################################################################################

external = load_config('config.ini')
save_each_ite = str(prefix) + '_band{}_convolution{}_noise{}_nsidefit{}'.format(band, convolution, noisy, nside_fit)

#########################################################################################################
############################################## Dictionnary ##############################################
#########################################################################################################

comp = []
if cmb[0].lower() == 'true':
    comp.append(c.CMB())
if dust[0].lower() == 'true':
    comp.append(c.Dust(nu0=nu0_d, temp=temp))
if synchrotron[0].lower() == 'true':
    comp.append(c.Synchrotron(nu0=nu0_s))
if coline[0].lower() == 'true':
    comp.append(c.COLine(nu=float(coline[2]), active=False))

d150, center = CMM.get_dictionary(nsub, nside, pointing, 150)
d220, _ = CMM.get_dictionary(nsub, nside, pointing, 220)

#########################################################################################################
############################################## Acquisitions #############################################
#########################################################################################################

# QUBIC Acquisition
qubic150 = Acq.QubicIntegratedComponentsMapMaking(d150, Nsub=nsub, comp=comp)
qubic220 = Acq.QubicIntegratedComponentsMapMaking(d220, Nsub=nsub, comp=comp)
qubic2bands = Acq.QubicTwoBandsComponentsMapMaking(qubic150, qubic220, comp=comp)

coverage = qubic150.get_coverage()
pixok = coverage/coverage.max() > 0

if band == 150:
    myqubic = qubic150
elif band == 220:
    myqubic = qubic220
elif band == 150220:
    myqubic = qubic2bands
else:
    raise TypeError('Not right band')

# Add external data
allexp = Acq.QubicOtherIntegratedComponentsMapMaking(myqubic, external, comp=comp, nintegr=nintegr)

#########################################################################################################
############################################## Components ###############################################
#########################################################################################################

dcomp = {}
if cmb[0].lower() == 'true':
    dcomp['cmb'] = int(cmb[1])
if dust[0].lower() == 'true':
    dcomp['dust'] = str(dust[1])
if synchrotron[0].lower() == 'true':
    dcomp['synchrotron'] = str(synchrotron[1])
if coline[0].lower() == 'true':
    dcomp['coline'] = str(coline[1])

print('COMPONENTS : {}'.format(dcomp))
print('COMP : {}'.format(comp))
components = qubic150.get_PySM_maps(dcomp)

C_fake = HealpixConvolutionGaussianOperator(fwhm=np.deg2rad(fake_convolution))

for i in range(components.shape[0]):
    components[i] = C_fake(components[i])

# invN
invN = allexp.get_invntt_operator()
M = Acq.get_preconditioner(np.ones(12*allexp.nside**2))

# Input beta
beta=np.array([1.54])

#########################################################################################################
############################################## Systematics ##############################################
#########################################################################################################

# Input gain

gdet150 = np.random.normal(float(varg150[0]), float(varg150[1]), (992))
gdet150 /= gdet150[0]
gdet220 = np.random.normal(float(varg220[0]), float(varg220[1]), (992))
gdet220 /= gdet220[0]

if band == 150:
    g = gdet150
elif band == 220:
    g = gdet220
elif band == 150220:
    g = np.array([gdet150, gdet220])
else:
    raise TypeError('Not right band')

#########################################################################################################
############################################## Reconstruction ###########################################
#########################################################################################################

if convolution:
    myfwhm = np.sqrt(myqubic.allfwhm**2 - np.min(myqubic.allfwhm)**2)
else:
    myfwhm = None

isco = coline[0].lower() == 'true'
if isco == False:
    nu_co = None
else:
    nu_co = float(coline[2])

# Get reconstruction operator
Hrecon = allexp.get_operator(beta, convolution, list_fwhm=myfwhm, gain=None, nu_co=nu_co)#np.array([np.ones(992)*1.0000000001, np.ones(992)*1.0000000001]))

# Get simulated data
tod = allexp.get_observations(beta, g, components, convolution=convolution, noisy=noisy, nu_co=nu_co)

### We can make the hypothesis that Planck's astrophysical foregrounds are a good starting point. We assume no prior on the CMB.
comp_for_pcg = components.copy()
for i in range(len(comp)):
    if i == 1:
        comp_for_pcg[i] = C_1degree(components[i])
    else:
        comp_for_pcg[i] = components[i].copy() * 0



#########################################################################################################
############################################## Main Loop ################################################
#########################################################################################################


iteration = 0
kmax=3000
k=0
beta_i = beta.copy()
g_i = g.copy()
components_i = comp_for_pcg.copy()

lmin = 40
lmax = 2*d150['nside']
dl = 35
s = CMM.Spectra(lmin, lmax, dl, r=float(r), Alens=float(alens), icl=2, CMB_CL_FILE=op.join('/home/regnier/work/regnier/mypackages/Cls_Planck2018_%s.fits'))

spectra_cmb = s.get_observed_spectra(comp_for_pcg[0].T)
spectra_dust = s.get_observed_spectra(comp_for_pcg[1].T)



if save_each_ite is not None:
    current_path = os.getcwd() + '/'
    if not os.path.exists(current_path + save_each_ite):
        os.makedirs(current_path + save_each_ite)
                
    dict_i = {'maps':components, 'initial':comp_for_pcg, 'beta':beta, 'gain':g, 'allfwhm':myqubic.allfwhm, 'coverage':coverage, 'convergence':1,
              'spectra_cmb':spectra_cmb, 'spectra_dust':spectra_dust}

    fullpath = current_path + save_each_ite + '/'
    output = open(fullpath+'Iter0_maps_beta_gain_rms_maps.pkl', 'wb')
    pickle.dump(dict_i, output)
    output.close()

while k < kmax :

    #####################################
    ######## Pixels minimization ########
    #####################################
    #if k == 0:
    #    beta_i = np.random.randn(beta.shape[0], beta.shape[1]) * 0.2 + 1.54
    H_i = allexp.update_A(Hrecon, beta_i)
    H_i = allexp.update_systematic(H_i, newG=g_i, co=isco)

    A = H_i.T * invN * H_i
    b = H_i.T * invN * tod

    solution = pcg(A, b, M=M, tol=float(tol), x0=components_i, maxiter=int(maxite), disp=True)
    components_i = solution['x'].copy()
    spectra_cmb = s.get_observed_spectra(solution['x'][0].T)
    spectra_dust = s.get_observed_spectra(solution['x'][1].T)



    ###################################
    ######## Gain minimization ########
    ###################################

    H_ii = allexp.update_systematic(Hrecon, newG=np.random.randn(myqubic.number_FP, myqubic.Ndets)*0+1.000000000000001, co=isco)
    g_i = CMM.get_gain_detector(H_ii, components_i, tod, myqubic.Nsamples, myqubic.Ndets, myqubic.number_FP)
    
    if myqubic.number_FP == 2:
        g_i[0] /= g_i[0, 0]
        g_i[1] /= g_i[1, 0]
    else:
        g_i /= g_i[0]

    print(g_i[:5])
    print(g[:5])

    ###################################
    ######## Beta minimization ########
    ###################################

    #chi2 = partial(myChi2, H=H_i, solution=components_i)

    def myChi2(x, solution):
        H_i = allexp.update_A(Hrecon, x)
        fakedata = H_i(solution)
        return np.sum((fakedata - tod)**2)
        
    chi2 = partial(myChi2, solution=components_i)

    beta_i = minimize(chi2, x0=beta, method=str(method), tol=1e-4).x

    print(beta_i)
    
    if save_each_ite is not None:
        dict_i = {'maps':components_i, 'beta':beta_i, 'gain':g_i, 'allfwhm':myqubic.allfwhm, 'coverage':coverage, 'convergence':solution['error'],
                  'spectra_cmb':spectra_cmb, 'spectra_dust':spectra_dust}

        fullpath = current_path + save_each_ite + '/'
        output = open(fullpath+'Iter{}_maps_beta_gain_rms_maps.pkl'.format(k+1), 'wb')
        pickle.dump(dict_i, output)
        output.close()


    k+=1

