from __future__ import division
from pyoperators import pcg
from pysimulators import profile

########################
#### QUBIC packages ####
########################
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

###################################################################
#### Function to read the config file and extract informations ####
###################################################################
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

##############################
#### PyOperators packages ####
##############################
from pyoperators import (
    BlockColumnOperator, BlockDiagonalOperator, BlockRowOperator,
    CompositionOperator, DiagonalOperator, I, IdentityOperator,
    MPIDistributionIdentityOperator, MPI, proxy_group, ReshapeOperator,
    rule_manager, pcg, Operator)

from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

#################################
#### Class used to compute r ####
#################################
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

########################################
#### Usefull functions to compute r ####
########################################
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

######################################################
#### Load the global variables in the config file ####
######################################################
external = load_config('compute_r_config.ini')
dict_parameters = external[1]
print(dict_parameters)

###################################################
#### import solutions computed in mapmaking.py ####
###################################################
import pickle
solution_two, solution_wide = [], []
cov_two, cov_wide = [], []

# Compute the list of the pixels seen by QUBIC
# WARNING : False means seen by QUBIC
pickle_dict = pickle.load(open(dir_name + 'Wide_Band_' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_' + f'{sub_band_number}_sub_bands_' + f'{pointing}_pointing' + f'{sky_name}_' + f'{0}.pkl', 'rb'))
coverage = pickle_dict['coverage']
seenpix = coverage/np.max(coverage) < seenpix_lim

# Extract solution in Wide Band & Two Bands from pickle files
# keeping only QUBIC pixels to speed up 'analysisMC.py' commands used after 
list_pix_qubix = []
for ipix in range(len(seenpix)):
    if seenpix[ipix] == False:
        list_pix_qubix.append(ipix)

for id_index in range(0, nmb_id):
    pickle_dict = pickle.load(open(dir_name + 'Two_Bands_' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_' + f'{sub_band_number}_sub_bands_' + f'{pointing}_pointing' + f'{sky_name}_' + f'{id_index}.pkl', 'rb'))
    sol = pickle_dict['solution_noise']
    sol_qubic = np.zeros((2*sub_band_number, len(list_pix_qubix), 3))
    cpt = 0
    for pix in list_pix_qubix:
        for stk_ind in range(3):
            for sb in range(2*sub_band_number):
                sol_qubic[sb, cpt, stk_ind] = sol[sb, pix, stk_ind]
        cpt += 1
    solution_two.append(sol_qubic)  

print('solution_two', np.shape(solution_two))

for id_index in range(0, nmb_id):
    pickle_dict = pickle.load(open(dir_name + 'Wide_Band_' + f'{effective_duration}_years_' + f'{nf_tod}_TOD_' + f'{sub_band_number}_sub_bands_' + f'{pointing}_pointing' + f'{sky_name}_' + f'{id_index}.pkl', 'rb'))
    sol = pickle_dict['solution_noise']
    sol_qubic = np.zeros((2*sub_band_number, len(list_pix_qubix), 3))
    cpt = 0
    for pix in list_pix_qubix:
        for stk_ind in range(3):
            for sb in range(2*sub_band_number):
                sol_qubic[sb, cpt, stk_ind] = sol[sb, pix, stk_ind]
        cpt += 1
    solution_wide.append(sol_qubic)  

# to verify what i am doing
sol_qubic_test = np.zeros((sub_band_number, len(seenpix), 3))
cpt = 0
sub_band_index = 1
for ipix in list_pix_qubix:
    for istokes in range(3):
        sol_qubic_test[0, ipix, istokes] = sol_qubic[0, cpt, istokes]
    cpt += 1

plt.figure(figsize=(10, 10))
stk = ['I', 'Q', 'U']
center = qubic.equ2gal(0, -57)
reso=25
for i in range(3):
    if i == 0:
        m = 300
        minr = 20
    else:
        m = 8
        minr = 8
    hp.gnomview(sol_qubic_test[0,:,i], rot=center, reso=reso, cmap='jet', min=-m, max=m, title=f'{stk[i]} - Input')
plt.savefig(path + 'map_test')

###############################
#### Create Namaster class ####
###############################
from qubic import NamasterLib as nam

lmin, lmax, delta_ell = lmin, 2*nside-1, delta_ell
namaster = nam.Namaster(weight_mask = list(~np.array(seenpix)), lmin = lmin, lmax = lmax, delta_ell = delta_ell)

###################################
#### Compute usefull varaibles ####
###################################
ell = namaster.get_binning(nside)[0]
rv = np.linspace(rmin, rmax, nmb_r)

############################################################################################################
#### Compute the average solution between sub-bands, using the covariance matrix, for each realisations ####
############################################################################################################
cp_two = analysis.get_Cp(solution_two)
solution_avg_two = analysis.make_weighted_av(solution_two, cp_two, verbose = True)

cp_wide = analysis.get_Cp(solution_wide)
solution_avg_wide = analysis.make_weighted_av(solution_wide, cp_wide, verbose = True)

print('solution two', np.shape(solution_two))
print('cp', np.shape(cp_two))
print('solu avg', np.shape(solution_avg_two))
print('solu avg [0]', np.shape(solution_avg_two[0]))

solution_two = np.zeros((nmb_id, len(seenpix), 3))
solution_wide = np.zeros((nmb_id, len(seenpix), 3))
cpt = 0
for qubic_pix in list_pix_qubix:
    for id_index in range(nmb_id):    
        for istokes in range(3):
            solution_two[id_index, qubic_pix, istokes] = solution_avg_two[0][id_index, cpt, istokes]
            solution_wide[id_index, qubic_pix, istokes] = solution_avg_wide[0][id_index, cpt, istokes]
    cpt += 1

print('solution_two', np.shape(solution_two))

#########################################################
#### Compute the power spectrum for each realisation ####
#########################################################
spectra_BB_wide, spectra_BB_two = [], []
for id_index in range(int(nmb_id/2)):
    # solution_1_t = np.mean(solution_two[id_index], axis = 0)
    # solution_2_t = np.mean(solution_two[id_index + int(nmb_id/2)], axis = 0)
    # dl_t = namaster.get_spectra(map = solution_1_t.T, map2 = solution_2_t.T)[1][:, 2]
    # spectra_BB_two.append(dl_t)

    dl_w = namaster.get_spectra(map = solution_wide[id_index].T, map2 = solution_wide[id_index + int(nmb_id/2)].T)[1][:, 2]
    spectra_BB_wide.append(dl_w)

    dl_t = namaster.get_spectra(map = solution_two[id_index].T, map2 = solution_two[id_index + int(nmb_id/2)].T)[1][:, 2]
    spectra_BB_two.append(dl_t)

print(np.shape(dl_t))
print(np.shape(spectra_BB_two))

###################################################
#### Compute the mean and the error ont the dl ####
###################################################
mean_wide = np.mean(spectra_BB_wide, axis = 0)
error_wide = np.std(spectra_BB_wide, axis = 0)
mean_two = np.mean(spectra_BB_two, axis = 0)
error_two = np.std(spectra_BB_two, axis = 0)
print(mean_two)
print(error_two)

###########################
#### Fitting to find r ####
###########################
likelihood_wide, cumint_wide, sigma_r_wide, r_wide = find_r(myBBth, nside).explore_like(ell, mean_wide, error_wide*np.sqrt(2), rv)
likelihood_two, cumint_two, sigma_r_two, r_two = find_r(myBBth, nside).explore_like(ell, mean_two, error_two*np.sqrt(2), rv)

print(np.shape(mean_two))
print(error_two)
print(sigma_r_two)

#########################
#### plot likelihood ####
#########################
plt.figure()
plt.plot(rv, likelihood_wide, label = f'r = {r_wide[0]:.3f}, sigma_r = {sigma_r_wide:.3f}')
plt.ylim((0,np.max(likelihood_wide)))
plt.xlabel('r')
plt.ylabel('Likelihood')
plt.legend()
plt.title('Likelihood - Wide Band')
plt.savefig(path + 'likelihood_wide_band' + f'.png')

plt.figure()
maximum = np.max(likelihood_two)
likelihood = likelihood_two / maximum
plt.plot(rv, likelihood, label = f'r = {r_two[0]:.5f}, sigma_r = {sigma_r_two:.5f}')
plt.ylim((0,1))
plt.xlabel('r')
plt.ylabel('Likelihood')
plt.legend()
plt.title('Likelihood - Two Bands')
plt.savefig(path + 'likelihood_two_bands.png')

plt.figure()
plt.plot(rv, likelihood_wide, label = f'Wide - r = {r_wide[0]:.3f}, sigma_r = {sigma_r_wide:.3f}')
plt.plot(rv, likelihood_two, label = f'Two - r = {r_two[0]:.3f}, sigma_r = {sigma_r_two:.3f}')
plt.ylim((0,1))
plt.xlabel('r')
plt.ylabel('Likelihood')
plt.legend()
plt.title('Likelihood - Comparaison')
plt.savefig(path + 'likelihood.png')
