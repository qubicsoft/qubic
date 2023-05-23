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
import AnalysisMC as analysis

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
import scipy

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

#### Class use to compute r ####
class find_r(object):

    def __init__(self, model, nside):

        self.nside=nside
        self.lmin = 40
        self.lmax = 2*self.nside-1
        self.dl = 35
        self.cc = 0
        self.model = model

        
    def ana_likelihood(self, rv, leff, fakedata, errors, model, prior,mylikelihood=mcmc.LogLikelihood, covariance_model_funct=None, otherp=None):
        
        ll = mylikelihood(xvals=leff, yvals=fakedata, errors=errors,model = model, flatprior=prior,
                                    covariance_model_funct=covariance_model_funct)
        like = np.zeros_like(rv)
        for i in range(len(rv)):
            like[i] = np.exp(ll([rv[i]]))
            maxL = rv[like == np.max(like)]
            cumint = scipy.integrate.cumtrapz(like, x=rv)
            cumint = cumint / np.max(cumint)
            onesigma = np.interp(0.68, cumint, rv[1:])
        if otherp:
            other = np.interp(otherp, cumint, rv[1:])
            return like, cumint, onesigma, other, maxL
        else:
            return like, cumint, onesigma, maxL
        
        
    def explore_like(self, leff, cl, errors, rv, Alens=0.1, otherp=None, cov=None, sample_variance=True):

        ### Create Namaster Object
        if cov is None:
            Namaster = nam.Namaster(None, lmin=self.lmin, lmax=self.lmax, delta_ell=self.dl)
            Namaster.fsky = 0.03
        else:
            okpix = cov > (np.max(cov) * float(cc))
            maskpix = np.zeros(12*self.nside**2)
            maskpix[okpix] = 1
            Namaster = nam.Namaster(maskpix, lmin=self.lmin, lmax=self.lmax, delta_ell=self.dl)
            Namaster.fsky = 0.03

        lbinned, b = Namaster.get_binning(self.nside)

        def myBBth(ell, r):
            return self.model(r, ell)

        ### Fake data
        fakedata = cl.copy()

        if sample_variance:
            covariance_model_funct = Namaster.knox_covariance
        else:
            covariance_model_funct = None

        if otherp is None:
            like, cumint, allrlim, maxL = self.ana_likelihood(rv, leff, fakedata, errors, myBBth, [[-1,1]],covariance_model_funct=covariance_model_funct)
        else:
            like, cumint, allrlim, other, maxL = self.ana_likelihood(rv, leff, fakedata, errors, myBBth, [[-1,1]],covariance_model_funct=covariance_model_funct, otherp=otherp)

        if otherp is None:
            return like, cumint, allrlim, maxL
        else:
            return like, cumint, allrlim, other, maxL

#### Usefull functions to compute r ####
# function to compute the Dl from the Cl
def cl_to_dl(ell, cl):
    dl=np.zeros(ell.shape[0])
    for i in range(ell.shape[0]):
        dl[i]=(ell[i]*(ell[i]+1)*cl[i])/(2*np.pi)
    return dl

# Import file with PLanck's data
CMB_CL_FILE = op.join('/sps/qubic/Users/TomLaclavere/mypackages/Cls_Planck2018_%s.fits')

# Function to compute the power spectrum from PLanck datas
def get_pw_from_planck(r, Alens):
    power_spectrum = hp.read_cl(CMB_CL_FILE%'lensed_scalar')[:,:4000]
    if Alens != 1.:
        power_spectrum[2] *= Alens
    if r:
        power_spectrum += r * hp.read_cl(CMB_CL_FILE%'unlensed_scalar_and_tensor_r1')[:,:4000]
    return power_spectrum

def myBBth(r, ell):
    dlBB = cl_to_dl(ell, get_pw_from_planck(Alens=1, r=r)[2, ell.astype(int)-1])
    return dlBB

#### Load the global variables in the config file ####
external = load_config('FMM_compute_r_config.ini')
dict_parameters = external[1]
print(dict_parameters)
if dir_name == "/sps/qubic/Users/TomLaclavere/results/FMM/band150/":
    band_real = 150
elif dir_name == "/sps/qubic/Users/TomLaclavere/results/FMM/band220/":
    band_real = 220

#### import solutions computed in mapmaking.py ####
import pickle
solution = []

pickle_dict = pickle.load(open(dir_name + f'MM_maxiter{maxiter}_convolution{fc}_npointing{npointings}_nrec{nrec}_nsub{nsub}_ndet{ndet}_npho150{npho150}_npho220{npho220}_seed{seed}_iteration{1}.pkl', 'rb'))
coverage = pickle_dict['coverage']
seenpix = coverage/np.max(coverage) < seenpix_lim

for real_index in range(1, nreal + 1):
    pickle_dict = pickle.load(open(dir_name + f'MM_maxiter{maxiter}_convolution{fc}_npointing{npointings}_nrec{nrec}_nsub{nsub}_ndet{ndet}_npho150{npho150}_npho220{npho220}_seed{seed}_iteration{real_index}.pkl', 'rb'))
    sol = pickle_dict['output']
    for pix in range(len(seenpix)):
        if seenpix[pix] == True:
            for stk_ind in range(3):
                    sol[pix,stk_ind] = 0
    solution.append(sol)  

#### Create Namaster class ####
from qubic import NamasterLib as nam

lmin, lmax, delta_ell = lmin, 2*nside-1, delta_ell
namaster = nam.Namaster(weight_mask = list(~np.array(seenpix)), lmin = lmin, lmax = lmax, delta_ell = delta_ell)

#### Compute usefull varaibles ####
ell = namaster.get_binning(nside)[0]
rv = np.linspace(rmin, rmax, nmb_r)

print(np.shape(solution))

#### Compute the power spectrum for each sub-bands for each realisations ####
spectra_BB = []

for real_index in range(0, int(nreal/2)):
    solution_1 = solution[real_index]
    solution_2 = solution[real_index + int(nreal/2)]
    dl = namaster.get_spectra(map = solution_1.T, map2 = solution_2.T)[1][:, 2]
    spectra_BB.append(dl)
print(np.shape(dl))
print(np.shape(spectra_BB))

#### Compute the mean and the error ont the dl ####
mean = np.mean(spectra_BB, axis = 0)
error = np.std(spectra_BB, axis = 0)
print(mean)
print(error)

plt.figure()
plt.plot(mean)
plt.title('Mean Spectra')
plt.savefig(path + f'compute_r/Mean_Spectra_band_{band_real}_maxiter{maxiter}_convolution{fc}_npointing{npointings}_nrec{nrec}_nsub{nsub}_ndet{ndet}_npho150{npho150}_npho220{npho220}_seed{seed}')

#### Fitting to find r ####
likelihood, cumint, sigma_r, r = find_r(myBBth, nside).explore_like(ell, mean, error*np.sqrt(2), rv)

print(np.shape(mean))
print(error)
print(sigma_r)

#### plot likelihood ####
plt.figure()
plt.plot(rv, likelihood, label = f'r = {r[0]:.4f}, sigma_r = {sigma_r:.4f}')
plt.ylim((0,1))
plt.xlabel('r')
plt.ylabel('Likelihood')
plt.legend()
plt.title(f'band_{band_real}_convolution{fc}_ndet{ndet}_npho150{npho150}_npho220{npho220}')
plt.savefig(path + f'/compute_r/likelihood_band_{band_real}_maxiter{maxiter}_convolution{fc}_npointing{npointings}_nrec{nrec}_nsub{nsub}_ndet{ndet}_npho150{npho150}_npho220{npho220}_seed{seed}')
