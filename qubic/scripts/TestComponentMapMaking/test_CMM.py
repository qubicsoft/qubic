# QUBIC packages
import qubic
import sys
import os
sys.path.append('/home/regnier/work/regnier/MapMaking')

import component_acquisition as Acq
import pickle
import gc

# Display packages
import healpy as hp
import matplotlib.pyplot as plt

# FG-Buster packages
import component_model as c

# General packages
import numpy as np
import warnings
from qubic import QubicSkySim as qss

from scipy.optimize import minimize
#import ComponentsMapMakingTools as CMM
from functools import partial
import time
import configparser

from pyoperators import MPI

# PyOperators packages
from pyoperators import *
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

warnings.filterwarnings("ignore")
path = '/home/regnier/work/regnier/MapMaking/ComponentMapMaking/forecast_wideband'

seed = 1#int(sys.argv[1])
iteration = 1#int(sys.argv[2])

### Reading and loading configuration file
def get_dict(args={}):
    
    ### Get the default dictionary
    dictfilename = 'dicts/pipeline_demo.dict'
    d = qubic.qubicdict.qubicDict()
    d.read_from_file(dictfilename)
    d['npointings'] = 9
    for i in args.keys():
        
        d[str(i)] = args[i]
    
    return d
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
def get_ultrawideband_config():
    
    nu_up = 247.5
    nu_down = 131.25
    nu_ave = np.mean(np.array([nu_up, nu_down]))
    delta = nu_up - nu_ave
    
    return nu_ave, 2*delta/nu_ave

nu_ave, delta_nu_over_nu = get_ultrawideband_config()
#########################################################################################################
############################################## Arguments ################################################
#########################################################################################################

external = load_config('config.ini')

if nside_fit != 0:
    raise TypeError('You must have to put nside_fit = 0 (constant spectral index)')

#########################################################################################################
############################################## Dictionnary ##############################################
#########################################################################################################

comp = []
comp_name = []
if cmb :
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

d = get_dict({'npointings':pointing, 'nf_recon':nsub, 'nf_sub':nsub, 'nside':nside, 'MultiBand':True, 
              'filter_nu':nu_ave*1e9, 'noiseless':False, 'comm':comm, 'nprocs_sampling':1, 'nprocs_instrument':size,
              'photon_noise':True, 'nhwp_angles':3, 'effective_duration':3, 'filter_relative_bandwidth':delta_nu_over_nu, 
              'type_instrument':'wide'})

#########################################################################################################
############################################## Acquisitions #############################################
#########################################################################################################

# QUBIC Acquisition
myqubic = Acq.QubicFullBand(d, Nsub=nsub, comp=comp, kind=type)


coverage = myqubic.get_coverage()
pixok = coverage/coverage.max() > thr



# Add external data
allexp = Acq.QubicOtherIntegratedComponentsMapMaking(myqubic, external, comp=comp, nintegr=nintegr)

# Input beta
beta=np.array([1.54])

H = allexp.get_operator(beta, convolution)


#########################################################################################################
############################################## Components ###############################################
#########################################################################################################

dcomp = {}
if cmb:
    dcomp['cmb'] = seed
if dust[0].lower() == 'true':
    dcomp['dust'] = str(dust[1])
if synchrotron[0].lower() == 'true':
    dcomp['synchrotron'] = str(synchrotron[1])
if coline[0].lower() == 'true':
    dcomp['coline'] = str(coline[1])

components = myqubic.get_PySM_maps(dcomp)


# invN
invN = allexp.get_invntt_operator()
M = Acq.get_preconditioner(np.ones(12*allexp.nside**2))


#########################################################################################################
############################################## Systematics ##############################################
#########################################################################################################


#########################################################################################################
############################################## Reconstruction ###########################################
#########################################################################################################

if convolution:
    myfwhm = np.sqrt(myqubic.allfwhm**2 - np.min(myqubic.allfwhm)**2)
else:
    myfwhm = None

# Get reconstruction operator
Hrecon = allexp.get_operator(beta, convolution, list_fwhm=myfwhm)

# Get simulated data
tod = H(components)

seed_pl = 42

if noisy:
    n = allexp.get_noise(seed_pl, ndet, pho150, pho220).ravel()
    tod += n.copy()

if convolution:
    tod = allexp.reconvolve_to_worst_resolution(tod)

if convolution:
    Ctrue = HealpixConvolutionGaussianOperator(fwhm=myqubic.allfwhm[-1], lmax=2*nside-1)
else:
    Ctrue = HealpixConvolutionGaussianOperator(fwhm=0.0, lmax=2*nside-1)

target = np.sqrt(myqubic.allfwhm[0]**2 - myqubic.allfwhm[-1]**2)
C_target = HealpixConvolutionGaussianOperator(fwhm=target)
### We can make the hypothesis that Planck's astrophysical foregrounds are a good starting point. We assume no prior on the CMB.
comp_for_pcg = components.copy()
rr = np.random.randn(12*nside**2, 3)
for i in range(len(comp)):

    if comp_name[i] == 'CMB':
        np.random.seed(42)
        comp_for_pcg[i] = Ctrue(components[i]) * (np.random.randn(12*nside**2, 3)*8)
    elif comp_name[i] == 'DUST':
        comp_for_pcg[i] = Ctrue(components[i])
    elif comp_name[i] == 'SYNCHROTRON':
        comp_for_pcg[i] = Ctrue(components[i])
    elif comp_name[i] == 'CO':
        comp_for_pcg[i] = Ctrue(components[i])
    else:
        raise TypeError(f'{comp_name[i]} not recognize')

#########################################################################################################
############################################## Main Loop ################################################
#########################################################################################################


kmax=1
k=0
beta_i = beta.copy()
components_i = comp_for_pcg.copy()


while k < kmax :

    #####################################
    ######## Pixels minimization ########
    #####################################

    H_i = allexp.update_A(Hrecon, beta_i)

    A = H_i.T * invN * H_i
    b = H_i.T * invN * tod
    
    comm.Barrier()

    ### PCG
    solution = pcg(A, b, M=M, tol=float(tol), x0=components_i, maxiter=int(maxite), disp=True)

    k+=1
    stop


