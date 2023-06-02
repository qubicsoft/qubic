# QUBIC packages
import qubic
from qubicpack.utilities import Qubic_DataDir
from qubic.data import PATH
from qubic.io import read_map
from qubic import QubicSkySim as qss
import sys
sys.path.insert(1,'/pbs/home/n/nmirongr/qubic/qubic/scripts/MapMaking-mr')

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
import os
#import iterate
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

C_1degree = HealpixConvolutionGaussianOperator(fwhm=np.deg2rad(1))
C_2degree = HealpixConvolutionGaussianOperator(fwhm=np.deg2rad(2))

warnings.filterwarnings("ignore")
path = '/sps/qubic/Users/nahuelmg/mapmaking_results'


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
print(external)
if nside_fit != 0:
    raise TypeError('You must have to put nside_fit = 0 (constant spectral index)')

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
#coverage = C_1degree(coverage)

pixok = coverage/coverage.max() > thr
#index_pixok = np.where(pixok == True)[0]
#mask = np.ones(12*nside**2)
#mask[index_pixok] /= 1e10

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
#invN.operands[0].operands[0].operands[1] *= 1000
#invN.operands[0].operands[1].operands[1] *= 1000
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

print(f'FWHM for Nsub : {myfwhm}')

# Get reconstruction operator
Hrecon = allexp.get_operator(beta, convolution, list_fwhm=myfwhm, gain=None, nu_co=nu_co)#np.array([np.ones(992)*1.0000000001, np.ones(992)*1.0000000001]))

# Get simulated data
tod = allexp.get_observations(beta, g, components, convolution=convolution, noisy=noisy, nu_co=nu_co, pixok=pixok)

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


### We can make the hypothesis that Planck's astrophysical foregrounds are a good starting point. We assume no prior on the CMB.
comp_for_pcg = components.copy()
rr = np.random.randn(12*nside**2, 3)
for i in range(len(comp)):

    if comp_name[i] == 'CMB':
        comp_for_pcg[i] = Ctrue(components[i])*0
    elif comp_name[i] == 'DUST':
        comp_for_pcg[i] = C_2degree(components[i])#C_2degree(components[i].copy())
    elif comp_name[i] == 'SYNCHROTRON':
        comp_for_pcg[i] = C_2degree(components[i])#C_2degree(components[i].copy())# * 0#Ctrue(components[i])
    elif comp_name[i] == 'CO':
        comp_for_pcg[i] = C_2degree(components[i])#C_2degree(components[i].copy())# * 0
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
s = CMM.Spectra(lmin, lmax, dl, r=float(r), Alens=float(alens), icl=2, CMB_CL_FILE=op.join('/pbs/home/n/nmirongr/qubic/qubic/qubic/scripts/MapMaking-mr/Cls_Planck2018_%s.fits'))

spectra_cmb = s.get_observed_spectra(comp_for_pcg[0].T)
spectra_dust = s.get_observed_spectra(comp_for_pcg[1].T)

def chi2_150(x, solution, g150):

    tod_s_i = tod_150.copy() * 0
    R = ReshapeOperator(((1,12*nside**2,3)), ((12*nside**2,3)))
    G150 = DiagonalOperator(g150, broadcast='rightward')
    k=0
    for ii, i in enumerate(array_of_operators150):
    
        A = CMM.get_mixing_operator(x, nus=np.array([qubic150.allnus[k]]), comp=comp, nside=nside, active=False)

        Hi = G150 * i * R * A
            
        tod_s_i += Hi(solution[ii]).ravel()
        k+=1

    
    tod_150_norm = tod_150#/tod_150.max()#/np.std(tod_150)
    tod_s_i_norm = tod_s_i#/tod_s_i.max()#/np.std(tod_s_i)

    return np.sum((tod_150_norm - tod_s_i_norm)**2)
def chi2_220(x, solution, g220):

    G220 = DiagonalOperator(g220, broadcast='rightward')
    tod_s_ii = tod_220.copy() * 0
    R = ReshapeOperator(((1,12*nside**2,3)), ((12*nside**2,3)))

    k=0
    for ii, i in enumerate(array_of_operators220):
        if k == nsub:
            A = CMM.get_mixing_operator(x, nus=np.array([230.538]), comp=comp, nside=nside, active=True)
            Hi = G220 * i * R * A 
            tod_s_ii += Hi(solution[-1]).ravel()
            k+=1
        else:
            mynus = np.array([qubic220.allnus[k]])
            A = CMM.get_mixing_operator(x, nus=mynus, comp=comp, nside=nside, active=False)
            Hi = G220 * i * R * A 
            tod_s_ii += Hi(solution[ii+nsub]).ravel()
            k+=1
        

    tod_220_norm = tod_220#/tod_220.max()
    tod_s_ii_norm = tod_s_ii#/tod_s_ii.max()
    return np.sum((tod_220_norm - tod_s_ii_norm)**2)
def chi2_external(x, solution):

    tod_s_i = tod_external.copy() * 0

    Hexternal = Acq.OtherData(external, nside, comp).get_operator(nintegr=nintegr, beta=x, convolution=False, myfwhm=None, nu_co=nu_co)

    tod_s_i = Hexternal(solution[-1])

    
    tod_external_norm = tod_external#CMM.normalize_tod(tod_external, external, 12*nside**2)
    tod_s_i_norm = tod_s_i#CMM.normalize_tod(tod_s_i, external, 12*nside**2)

    return np.sum((tod_external_norm - tod_s_i_norm)**2)
def chi2_tot(x, solution, g150, g220):
    
    xi2_150 = chi2_150(x, solution, g150)
    xi2_220 = chi2_220(x, solution, g220)
    xi2_external = chi2_external(x, solution)
    #print(x, xi2_150, xi2_220, xi2_external)
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

    H_i = allexp.update_A(Hrecon, beta_i, co=isco)
    H_i = allexp.update_systematic(H_i, newG=g_i, co=isco)

    A = H_i.T * invN * H_i
    b = H_i.T * invN * tod

    ### PCG
    solution = pcg(A, b, M=M, tol=float(tol), x0=components_i, maxiter=int(maxite), disp=True)

    ### Synchrotron is assumed to be a template removed to the TOD
    #if synchrotron[0].lower() == 'true':
    #    index = comp_name.index('SYNCHROTRON')
    #    C = HealpixConvolutionGaussianOperator(fwhm=np.min(myqubic.allfwhm))
    #    solution['x'][index] = C(components[index]).copy()

    ### Compute spectra
    components_i = solution['x'].copy()

    spectra_cmb = s.get_observed_spectra(solution['x'][0].T)
    spectra_dust = s.get_observed_spectra(solution['x'][1].T)

    components_for_beta = np.zeros((2*nsub, len(comp), 12*nside**2, 3))

    ### We make the convolution before beta estimation to speed up the code, we avoid to make all the convolution at each iteration
    for i in range(2*nsub):
        for jcomp in range(len(comp)):
            if comp_name[jcomp] == 'CO':
                components_for_beta[i, jcomp] = components_i[jcomp]
            elif comp_name[jcomp] == 'SYNCHROTRON':
                components_for_beta[i, jcomp] = components_i[jcomp]
            else:
                components_for_beta[i, jcomp] = components_i[jcomp]

    ###################################
    ######## Gain minimization ########
    ###################################

    H_ii = allexp.update_systematic(Hrecon, newG=np.random.randn(myqubic.number_FP, myqubic.Ndets)*0+1.000000000000001, co=isco)
    g_i = CMM.get_gain_detector(H_ii, components_for_beta[-1], tod, myqubic.Nsamples, myqubic.Ndets, myqubic.number_FP)
    
    if myqubic.number_FP == 2:
        g_i[0] /= g_i[0, 0]
        g_i[1] /= g_i[1, 0]
    else:
        g_i /= g_i[0]

    print(np.mean(g_i[:5]-g[:5], axis=1))

    ###################################
    ######## Beta minimization ########
    ###################################

    ### We define new chi^2 function for beta knowing the components at iteration i
    chi2 = partial(chi2_tot, solution=components_for_beta, g150=g_i[0], g220=g_i[1])

    ### Doing minimization
    beta_i = minimize(chi2, x0=np.array([1.5]), method=str(method), tol=1e-4).x
    
    print(beta_i)
    
    ### Saving components, beta, gain, convergence, etc.. for each iteration
    if save_each_ite is not None:
        dict_i = {'maps':components_i, 'beta':beta_i, 'gain':g_i, 'allfwhm':myqubic.allfwhm, 'coverage':coverage, 'convergence':solution['error'],
                  'spectra_cmb':spectra_cmb, 'spectra_dust':spectra_dust}

        fullpath = current_path + save_each_ite + '/'
        output = open(fullpath+'Iter{}_maps_beta_gain_rms_maps.pkl'.format(k+1), 'wb')
        pickle.dump(dict_i, output)
        output.close()


    k+=1

