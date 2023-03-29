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


# PyOperators packages
from pyoperators import (
    BlockColumnOperator, BlockDiagonalOperator, BlockRowOperator,
    CompositionOperator, DiagonalOperator, I, IdentityOperator,
    MPIDistributionIdentityOperator, MPI, proxy_group, ReshapeOperator,
    rule_manager, pcg, Operator)

warnings.filterwarnings("ignore")
path = '/Users/mregnier/Desktop/PhD Regnier/MapMaking/bash'
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

C_1degree = HealpixConvolutionGaussianOperator(fwhm=np.deg2rad(1))

#########################################################################################################
############################################## Arguments ################################################
#########################################################################################################

convolution = False
noisy = False
tol = 1e-10
submaxiter = 20
nside_fit = 0

reload(Acq)
Nsub = int(sys.argv[1])
nside = int(sys.argv[2])
pointing = int(sys.argv[3])
band = int(sys.argv[4])
Nprocess = int(os.environ.get('NUM_CPUS'))

save_each_ite = 'P217353_CMB_DUST_CO_band{}_convolution{}_noise{}_nsidefit{}'.format(band, convolution, noisy, nside_fit)

#########################################################################################################
############################################## Dictionnary ##############################################
#########################################################################################################
nu_co = 230.538e9  # GHz
co = True

comp = [c.CMB(), c.Dust(nu0=150, temp=20), c.COLine(nu=nu_co, active=False)]

d150, center = CMM.get_dictionary(Nsub, nside, pointing, 150)
d220, _ = CMM.get_dictionary(Nsub, nside, pointing, 220)

#########################################################################################################
############################################## Acquisitions #############################################
#########################################################################################################

# QUBIC Acquisition
qubic150 = Acq.QubicIntegratedComponentsMapMaking(d150, Nsub=Nsub, comp=comp)
qubic220 = Acq.QubicIntegratedComponentsMapMaking(d220, Nsub=Nsub, comp=comp)
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
allexp = Acq.QubicOtherIntegratedComponentsMapMaking(myqubic, [217, 353], comp=comp, nintegr=5)

#########################################################################################################
############################################## Components ###############################################
#########################################################################################################

components = qubic150.get_PySM_maps({'cmb':42, 'dust':'d0', 'coline':'co2'})

# invN
invN = allexp.get_invntt_operator()
M = Acq.get_preconditioner(np.ones(12*allexp.nside**2))

# Input beta
beta=np.array([1.54])

#########################################################################################################
############################################## Systematics ##############################################
#########################################################################################################

# Input gain
sigG = 0.4
gdet150 = np.random.normal(1, sigG, (992))
gdet150 /= gdet150[0]
gdet220 = np.random.normal(1, sigG, (992))
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
print(myfwhm)
Hrecon = allexp.get_operator(beta, convolution, list_fwhm=myfwhm, gain=None, nu_co=nu_co)#np.array([np.ones(992)*1.0000000001, np.ones(992)*1.0000000001]))

# Get simulated data
tod = allexp.get_observations(beta, g, components, convolution=convolution, noisy=noisy, nu_co=nu_co)

comp_for_pcg = np.array([C_1degree(components[0]+components[1]+components[2]),
                         C_1degree(components[0]+components[1]+components[2]),
                         C_1degree(components[0]+components[1]+components[2])])

#########################################################################################################
############################################## Main Loop ################################################
#########################################################################################################

if save_each_ite is not None:
    current_path = os.getcwd() + '/'
    if not os.path.exists(current_path + save_each_ite):
        os.makedirs(current_path + save_each_ite)
                
    dict_i = {'maps':components, 'initial':comp_for_pcg, 'beta':beta, 'gain':g, 'allfwhm':myqubic.allfwhm, 'coverage':coverage, 'convergence':1}

    fullpath = current_path + save_each_ite + '/'
    output = open(fullpath+'Iter0_maps_beta_gain_rms_maps.pkl', 'wb')
    pickle.dump(dict_i, output)
    output.close()


iteration = 0
kmax=3000
k=0
beta_i = beta.copy()
g_i = g.copy()
components_i = comp_for_pcg.copy()

while k < kmax :

    #####################################
    ######## Pixels minimization ########
    #####################################
    #if k == 0:
    #    beta_i = np.random.randn(beta.shape[0], beta.shape[1]) * 0.2 + 1.54
    H_i = allexp.update_A(Hrecon, beta_i)
    H_i = allexp.update_systematic(H_i, newG=g_i, co=co)

    A = H_i.T * invN * H_i
    b = H_i.T * invN * tod

    solution = pcg(A, b, M=M, tol=tol, x0=components_i, maxiter=submaxiter, disp=True)
    components_i = solution['x'].copy()

    ###################################
    ######## Gain minimization ########
    ###################################

    H_ii = allexp.update_systematic(Hrecon, newG=np.random.randn(myqubic.number_FP, myqubic.Ndets)*0+1.000000001, co=co)
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

    beta_i = minimize(chi2, x0=beta, method='L-BFGS-B', tol=1e-6).x

    print(beta_i)
    
    if save_each_ite is not None:
        dict_i = {'maps':components_i, 'beta':beta_i, 'gain':g_i, 'allfwhm':myqubic.allfwhm, 'coverage':coverage, 'convergence':solution['error']}

        fullpath = current_path + save_each_ite + '/'
        output = open(fullpath+'Iter{}_maps_beta_gain_rms_maps.pkl'.format(k+1), 'wb')
        pickle.dump(dict_i, output)
        output.close()


    k+=1

