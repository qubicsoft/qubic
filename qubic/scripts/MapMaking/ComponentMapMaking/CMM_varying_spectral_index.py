# QUBIC packages
import os
import qubic
from qubicpack.utilities import Qubic_DataDir
from qubic.data import PATH
from qubic.io import read_map
from qubic import QubicSkySim as qss
import sys
folder_to_data = os.path.abspath(os.path.join(os.path.abspath(os.getcwd()), os.pardir))
sys.path.append(folder_to_data)
folder_to_data +=  '/'

import component_acquisition as Acq

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

### Reading and loading configuration file
def load_config(config_file):
    # Créer un objet ConfigParser
    config = configparser.ConfigParser()

    # Lire le fichier de configuration
    config.read(config_file)

    # Itérer sur chaque section et option
    external = []
    allnus = [30, 44, 70, 100, 143, 217, 353]
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

if nside_fit == 0:
    raise TypeError('You must have to put nside_fit != 0 (varying spectral index)')

#########################################################################################################
############################################## Dictionnary ##############################################
#########################################################################################################

comp = []
comp_name = []
if cmb[0].lower() == 'true':
    comp.append(c.CMB())
    comp_name.append('CMB')
if dust[0].lower() == 'true':
    comp.append(c.Dust(nu0=nu0_d, temp=temp))
    comp_name.append('DUST')
if synchrotron[0].lower() == 'true':
    comp.append(c.Synchrotron(nu0=nu0_s, beta_pl=-3))                     # We remove a template of synchrotron emission -> fixing the spectral index
    comp_name.append('SYNCHROTRON')
if coline[0].lower() == 'true':
    comp.append(c.COLine(nu=float(coline[2])/1e9, active=False))
    comp_name.append('CO')

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

pixok_nside_fit = hp.ud_grade(pixok, nside_fit)
index_fit_beta = np.where(pixok_nside_fit == True)[0]
number_of_estimated_beta = len(index_fit_beta)
number_of_loop_processes = number_of_estimated_beta // nprocess
rest_processes = number_of_estimated_beta % nprocess

print(f'Number of estimated beta : {len(index_fit_beta)}')
print(f'Number of processes      : {nprocess}')
print(f'Number of loop           : {number_of_loop_processes}')
print(f'Number of rest           : {rest_processes}')


isco = coline[0].lower() == 'true'
print('isco : ', isco)

if isco == False:
    nu_co = None
else:
    nu_co = float(coline[2])

if band == 150:
    myqubic = qubic150
    array_of_operators150 = qubic150._get_array_of_operators()

elif band == 220:
    myqubic = qubic220
    array_of_operators220 = qubic220._get_array_of_operators(nu_co=nu_co)
elif band == 150220:
    myqubic = qubic2bands
    array_of_operators150 = qubic150._get_array_of_operators()
    array_of_operators220 = qubic220._get_array_of_operators(nu_co=nu_co)
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
print('NAME COMP : {}'.format(comp_name))
components = qubic150.get_PySM_maps(dcomp)

# invN
invN = allexp.get_invntt_operator()
M = Acq.get_preconditioner(np.ones(12*allexp.nside**2))

# Input beta
beta_d = hp.read_map(folder_to_data + beta_file)
print(beta_d)
stop
beta = np.ones((12*nside_fit**2, 1))
beta[:, 0] *= 1.54

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

print(f'FWHM for Nsub : {myfwhm}')

# Get reconstruction operator
Hrecon = allexp.get_operator(beta, convolution, list_fwhm=myfwhm, gain=None, nu_co=nu_co)#np.array([np.ones(992)*1.0000000001, np.ones(992)*1.0000000001]))

# Get simulated data
tod = allexp.get_observations(beta, g, components, convolution=convolution, noisy=noisy, nu_co=nu_co)

if band == 150:
    tod_150 = tod[:(myqubic.Ndets*myqubic.Nsamples)]
elif band == 220:
    tod_220 = tod[:(myqubic.Ndets*myqubic.Nsamples)]
elif band == 150220:
    tod_150 = tod[:(myqubic.Ndets*myqubic.Nsamples)]
    tod_220 = tod[(myqubic.Ndets*myqubic.Nsamples):(myqubic.Ndets*myqubic.Nsamples*2)]

tod_external = tod[((myqubic.Ndets*myqubic.Nsamples)*2):]

if convolution:
    Ctrue = HealpixConvolutionGaussianOperator(fwhm=myqubic.allfwhm[-1])
else:
    Ctrue = HealpixConvolutionGaussianOperator(fwhm=0)
C_2degree = HealpixConvolutionGaussianOperator(fwhm=np.deg2rad(2))

### We can make the hypothesis that Planck's astrophysical foregrounds are a good starting point. We assume no prior on the CMB.
comp_for_pcg = components.copy()
for i in range(len(comp)):

    if comp_name[i] == 'CMB':
        comp_for_pcg[i] = components[i].copy() * 0
    elif comp_name[i] == 'DUST':
        comp_for_pcg[i] = components[i].copy() * 0
    elif comp_name[i] == 'SYNCHROTRON':
        comp_for_pcg[i] = Ctrue(components[i])
    elif comp_name[i] == 'CO':
        comp_for_pcg[i] = components[i] * 0
    else:
        raise TypeError(f'{comp_name[i]} not recognize')

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
def chi2_150(x, patch_id, solution, g150):

    newbeta = beta.copy()
    newbeta[patch_id] = x

    tod_s_i = tod_150.copy() * 0
    R = ReshapeOperator(((12*nside**2,1,3)), ((12*nside**2,3)))
    G150 = DiagonalOperator(g150, broadcast='rightward')
    k=0
    for ii, i in enumerate(array_of_operators150):
    
        A = CMM.get_mixing_operator(newbeta, nus=np.array([qubic150.allnus[k]]), comp=comp, nside=nside)

        Hi = G150 * i * R * A
            
        tod_s_i += Hi(solution[ii]).ravel()
        k+=1

    tod_150_norm = (tod_150)/np.std(tod_150)
    tod_s_i_norm = (tod_s_i)/np.std(tod_s_i)

    return np.sum((tod_150_norm - tod_s_i_norm)**2)
def chi2_220(x, patch_id, solution, g220):

    newbeta = beta.copy()
    newbeta[patch_id] = x

    G220 = DiagonalOperator(g220, broadcast='rightward')
    tod_s_i = tod_220.copy() * 0
    R = ReshapeOperator(((12*nside**2,1,3)), ((12*nside**2,3)))

    k=0
    for ii, i in enumerate(array_of_operators220):
        
        A = CMM.get_mixing_operator(newbeta, nus=np.array([qubic220.allnus[k]]), comp=comp, nside=nside)
        
        Hi = G220 * i * R * A
            
        tod_s_i += Hi(solution[ii]).ravel()
        k+=1

    tod_220_norm = (tod_220)/np.std(tod_220)
    tod_s_i_norm = (tod_s_i)/np.std(tod_s_i)
    return np.sum((tod_220_norm - tod_s_i_norm)**2)
def chi2_external(x, patch_id, solution):

    newbeta = beta.copy()
    newbeta[patch_id] = x
    tod_s_i = tod_external.copy() * 0

    Hexternal = Acq.OtherData(external, nside, comp).get_operator(nintegr=nintegr, beta=newbeta, convolution=False, myfwhm=None, nu_co=nu_co)

    tod_s_i = Hexternal(solution[-1])

    tod_external_norm = (tod_external)/np.std(tod_external)
    tod_s_i_norm = (tod_s_i)/np.std(tod_s_i)

    return np.sum((tod_external_norm - tod_s_i_norm)**2)
def chi2_tot(x, patch_id, solution, g150, g220):
    print(x)
    xi2_150 = chi2_150(x, patch_id, solution, g150)
    xi2_220 = chi2_220(x, patch_id, solution, g220)
    xi2_external = chi2_external(x, patch_id, solution)

    #print(f'chi2 150 : {xi2_150:.3e}, chi2 220 : {xi2_220:.3e}, chi2 external {xi2_external:.3e}')

    return xi2_150 + xi2_220 + xi2_external

if save_each_ite is not None:
    current_path = os.getcwd() + '/'
    if not os.path.exists(current_path + save_each_ite):
        os.makedirs(current_path + save_each_ite)
                
    dict_i = {'maps':components, 'initial':comp_for_pcg, 'beta':beta, 'gain':g, 'allfwhm':myqubic.allfwhm, 'coverage':coverage, 'convergence':1,
              'spectra_cmb':spectra_cmb, 'spectra_dust':spectra_dust, 'execution_time':0}

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
    H_i = allexp.update_systematic(H_i, newG=g_i)

    A = H_i.T * invN * H_i
    b = H_i.T * invN * tod

    solution = pcg(A, b, M=M, tol=tol, x0=components_i, maxiter=maxite, disp=True)
    components_i = solution['x'].copy()

    ###################################
    ######## Gain minimization ########
    ###################################

    H_ii = allexp.update_systematic(Hrecon, newG=np.array([np.ones(myqubic.Ndets)*1.0000000001]*myqubic.number_FP))
    g_i = CMM.get_gain_detector(H_ii, components_i, tod, myqubic.Nsamples, myqubic.Ndets, myqubic.number_FP)
    
    if myqubic.number_FP == 2:
        g_i[0] /= g_i[0, 0]
        g_i[1] /= g_i[1, 0]
    else:
        g_i /= g_i[0]

    print(g_i[:5])

    ###################################
    ######## Beta minimization ########
    ###################################

    components_for_beta = np.zeros((2*nsub, 3, 12*nside**2, len(comp)))

    for i in range(2*nsub):
        print(f'Convolution by gaussian kernel with FWHM = {myqubic.allfwhm[i]}')
        Ci = HealpixConvolutionGaussianOperator(fwhm = myfwhm[i])
        for jcomp in range(len(comp)):
            components_for_beta[i, :, :, jcomp] = Ci(components_i[jcomp]).T

    chi2 = partial(chi2_tot, solution=components_for_beta, g150=g_i[0], g220=g_i[1])

    start = time.time()
    for i in range(number_of_loop_processes):
            
        pix_min = i*nprocess
        pix_max = (i+1)*nprocess

        print('Doing estimation on index {}'.format(list(index_fit_beta[pix_min:pix_max])))
        
        beta_i[index_fit_beta[pix_min:pix_max]] = fmp.FitMultiProcess(chi2, nprocess, method=method, tol=1e-5, x0=np.array([1.5]), options={}).perform(list(index_fit_beta[pix_min:pix_max]))

    if rest_processes != 0:
        print('Doing estimation of {} betas on {} processes'.format(rest_processes, rest_processes))
        beta_i[index_fit_beta[-rest_processes:]] = fmp.FitMultiProcess(chi2, rest_processes, method=method, tol=1e-5, x0=np.array([1.5])).perform(list(index_fit_beta[-(rest_processes):]))
    print(beta_i)

    end = time.time()

    print('Execution time : {} s'.format(end-start))
    
    #stop
    
    #if save_each_ite is not None:
    #    dict_i = {'maps':components_i, 'beta':beta_i, 'gain':g_i, 'allfwhm':myqubic.allfwhm, 'coverage':coverage, 'convergence':solution['error']}

    #    fullpath = current_path + save_each_ite + '/'
    #    output = open(fullpath+'Iter{}_maps_beta_gain_rms_maps.pkl'.format(k+1), 'wb')
    #    pickle.dump(dict_i, output)
    #    output.close()

    k+=1

