#########################################################################################################################
#########################################################################################################################
#########                                                                                                       #########
#########      This script allow to perform the frequency map-making for QUBIC experiments. We show here how to #########
#########      take into account the angular resolution and the bandpass mismatch.                              #########
#########                                                                                                       #########
#########################################################################################################################
#########################################################################################################################

### General importations
import numpy as np
import scipy
import healpy as hp
import pysm3
import pysm3.units as u
from pysm3 import utils
import sys
import os
from scipy.optimize import minimize
import os.path as op
import configparser
import pickle
sys.path.append('/home/regnier/work/regnier/mypackages')

# FG-Buster packages
import component_model as c
import mixing_matrix as mm

### QUBIC packages
import qubic
from qubic import NamasterLib as nam
import frequency_acquisition as Acq

### PyOperators
from pyoperators import *

### PySimulators
from pysimulators import *
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

###############################################################
##################### Useful definitions ######################
###############################################################

def get_pySM_maps(sky, nu):
    return np.array(sky.get_emission(nu * u.GHz, None).T * utils.bandpass_unit_conversion(nu*u.GHz, None, u.uK_CMB))

### Reading and loading configuration file
def load_config(config_file):
    # Créer un objet ConfigParser
    config = configparser.ConfigParser()

    # Lire le fichier de configuration
    config.read(config_file)

    k = 0
    for section in config.sections():
        for option in config.options(section):
            
            # Récupérer la valeur de chaque option de configuration
            value = config.get(section, option)

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

load_config('config.ini')

###############################################################
########################## Dictionary #########################
###############################################################

dictfilename = 'dicts/pipeline_demo.dict'
d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)

d['nf_recon'] = nf_recon
d['nf_sub'] = nf_recon*fact_sub
d['nside'] = nside
npix=12*d['nside']**2
d['RA_center'] = 0
d['DEC_center'] = -57
center = qubic.equ2gal(d['RA_center'], d['DEC_center'])
d['effective_duration'] = 3
d['npointings'] = npointings
d['filter_nu'] = band * 1e9
d['photon_noise'] = not noiseless
d['noiseless'] = noiseless
d['config'] = 'FI'
d['filter_relative_bandwidth'] = 0.25
d['MultiBand'] = True
d['dtheta'] = 15
d['synthbeam_dtype'] = float

if seed is False:
    seed = np.random.randint(10000000)
sky_config = {'cmb':seed}

qubic_acquisition = Acq.QubicIntegrated(d, Nsub=nf_tod, Nrec=nf_tod)
qubic_acquisition_recon = Acq.QubicIntegrated(d, Nsub=fact_sub*nf_recon, Nrec=nf_recon)
planck_acquisition = Acq.PlanckAcquisition(band_planck, qubic_acquisition.scene)
qubicplanck_acquisition = Acq.QubicPlanckMultiBandAcquisition(qubic_acquisition, planck_acquisition)

### Coverage
cov = qubic_acquisition.get_coverage()
C_1degree  = HealpixConvolutionGaussianOperator(fwhm = np.deg2rad(1))
covnorm = cov/cov.max()
seenpix = covnorm > thr
#seenpix = C_1degree(seenpix)


###############################################################
######################### Sky model ###########################
###############################################################
print('\n***** Sky Model ******\n')

### We define foregrounds model
sky = pysm3.Sky(nside, preset_strings=['d0'])
sky_model = pysm3.Sky(nside, preset_strings=['d0'])

nu_ref = 150
### CMB realization -> for real data, we should take Planck CMB map
cmb = qubic_acquisition.get_PySM_maps({'cmb':seed}, np.array([nu_ref]))

### We compute components, we should take Planck components for real data

plancksky = np.zeros((nf_tod, 12*nside**2, 3))
skymodel = np.zeros((nf_tod, 12*nside**2, 3))
C = HealpixConvolutionGaussianOperator(fwhm=np.deg2rad(fwhm_correction))
mysky_ref = get_pySM_maps(sky, nu_ref)*0                                    ### Used for Planck data 
mysky_model = C(get_pySM_maps(sky_model, nu_ref)*0)                         ### Convolved sky model by gaussian kernel

### We define how foregrounds emit w.r.t frequency
comp = c.Dust(nu0=nu_ref, temp=20)
beta = np.array([1.54])
beta_model = np.array([beta_d_model])
allnus = qubic_acquisition.allnus

### We compute the expected SED
sed = mm.MixingMatrix(comp).evaluator(allnus)(beta)
sed_model = mm.MixingMatrix(comp).evaluator(allnus)(beta_model)

### We scale the components to reproduce frequency observations
for i in range(3):
    plancksky[:, :, i] = sed @ np.array([mysky_ref[:, i]]) + cmb[0, :, i]
    skymodel[:, :, i] = sed_model @ np.array([mysky_model[:, i]])

### Correct Planck data by the sky model
k=0
for i in range(nf_recon):
    delta = skymodel[i*fact_sub:(i+1)*fact_sub] - np.mean(skymodel[i*fact_sub:(i+1)*fact_sub], axis=0)
    for j in range(fact_sub):
        plancksky[k] -= delta[j]
        k+=1

### We making the average between two frequencies
mean_sky = np.zeros((nf_recon, 12*nside**2, 3))
for i in range(nf_recon):
    mean_sky[i] = np.mean(plancksky[i*fact_sub:(i+1)*fact_sub], axis=0)



###############################################################
######################### Making TODs #########################
###############################################################

print('\n***** Making TODs ******\n')
### We compute QUBIC TODs with the correction of the bandpass
noise = not noiseless
TOD_QUBIC = qubic_acquisition.generate_tod(config=sky_config, map_ref=mysky_model, beta=beta_model, A_ev=mm.MixingMatrix(comp).evaluator(allnus), 
                            noise=False, bandpass_correction=bandpass_correction, convolution=convolution)

### We compute Planck TODs using the previous sky

TOD_PLANCK = np.zeros((nf_recon, 12*nside**2, 3))
mrec = np.zeros((nf_recon, 12*nside**2, 3))
n_pl = planck_acquisition.get_noise() * level_noise_planck * 0
for irec in range(nf_recon):
    
    if convolution:
        target = np.min(qubic_acquisition.allfwhm[irec*fact_sub:(irec+1)*fact_sub])
    else:
        target = 0

    C = HealpixConvolutionGaussianOperator(fwhm = target)
    mrec[irec] = C(mean_sky[irec].copy() + n_pl.copy())
    TOD_PLANCK[irec] = C(mean_sky[irec].copy() + n_pl.copy())

R = ReshapeOperator(mrec.shape, (mrec.shape[0]*mrec.shape[1]*mrec.shape[2]))
TOD_PLANCK = R(TOD_PLANCK)

###############################################################
######################## Acquisitions #########################
###############################################################
print('\n***** Acquisitions ******\n')
### Create Planck and joint acquisition
planck_acquisition = Acq.PlanckAcquisition(band_planck, qubic_acquisition_recon.scene)
qubicplanck_acquisition = Acq.QubicPlanckMultiBandAcquisition(qubic_acquisition_recon, planck_acquisition)

### Create the final TOD
TOD = np.r_[TOD_QUBIC, TOD_PLANCK]

###############################################################
########################## Operators ##########################
###############################################################
print('\n***** Reconstruction Operators ******\n')

### We define here the expected angular resolution for reconstruction
myfwhm = np.array([])
for i in range(nf_recon):
    myfwhm = np.append(myfwhm, np.sqrt(qubic_acquisition.allfwhm[i*fact_sub:(i+1)*fact_sub]**2 - np.min(qubic_acquisition.allfwhm[i*fact_sub:(i+1)*fact_sub]**2)))
if convolution is False:
    myfwhm *= 0


print(f'You are reconstructing with : {myfwhm} rad')
### Reconstruction operator
H = qubicplanck_acquisition.get_operator(convolution=convolution, myfwhm=myfwhm)
invN = qubicplanck_acquisition.get_invntt_operator()

R = ReshapeOperator((1, 12*nside**2, 3), (12*nside**2, 3))

### Unpack Operator to fix pixels not seen by QUBIC
U = (
    ReshapeOperator((nf_recon * sum(seenpix) * 3), (nf_recon, sum(seenpix), 3)) *
    PackOperator(np.broadcast_to(seenpix[None, :, None], (nf_recon, seenpix.size, 3)).copy())
).T

### Compute A and b
with rule_manager(none=True):
    if nf_recon == 1:
        A = U.T * R.T * H.T * invN * H * R * U
        x_planck = mean_sky * (1 - seenpix[None, :, None])
        b = U.T ( R.T * H.T * invN * (TOD - H(R(x_planck))))
    else:
        A = U.T * H.T * invN * H * U
        x_planck = mean_sky * (1 - seenpix[None, :, None])
        b = U.T (  H.T * invN * (TOD - H(x_planck)))

### Preconditionning
M = Acq.get_preconditioner(np.ones(12*nside**2))

### PCG
print('\n***** PCG ******\n')
solution_qubic_planck = pcg(A, b, x0=None, M=M, tol=1e-25, disp=True, maxiter=maxiter)

output = mean_sky.copy()
for i in range(nf_recon):
    output[i, seenpix] = solution_qubic_planck['x'][i]

dict_i = {'output':output, 'input':mean_sky, 'allfwhm':qubic_acquisition.allfwhm, 'coverage':cov, 'seenpix':seenpix, 'covcut':thr, 'center':center, 
          'fact_sub':fact_sub, 'Nf_recon':nf_recon, 'Nf_TOD':nf_tod}

def get_spectrum(map, maskpix, lmin, lmax, dl, map2=None):
    aposize=10
    Namaster = nam.Namaster(maskpix, lmin=lmin, lmax=lmax, delta_ell=dl, aposize=aposize)
    Namaster.ell_binned, _ = Namaster.get_binning(nside)

    leff, cls, _ = Namaster.get_spectra(map, map2=map2,
                                 purify_e=False,
                                 purify_b=True,
                                 w=None,
                                 verbose=False,
                                 beam_correction=None,
                                 pixwin_correction=False)

    return leff, cls[:, 2]

map_to_nam = output.copy() - mean_sky.copy()

### To be sure that pixels not seen by QUBIC = 0
map_to_nam[:, ~seenpix, :] = 0


if spectrum:
    leff, Dls = get_spectrum(map=map_to_nam[0].T, maskpix=seenpix, lmin=40, lmax=2*nside-1, dl=35)
    print(leff)
    print(Dls)
    dict_i['leff'] = leff
    dict_i['Dl_BB'] = Dls


### If the folder is not here, you will create it
save_each_ite = str(prefix)
current_path = os.getcwd() + '/'
if not os.path.exists(current_path + save_each_ite):
    os.makedirs(current_path + save_each_ite)

fullpath = current_path + save_each_ite + '/'
output = open(fullpath+f'MM_band{band}_bandpasscorrection{bandpass_correction}_Nrec{nf_recon}_Nsub{nf_recon*fact_sub}_Ntod{nf_tod}_correction_conv{fwhm_correction}deg_noise{noise}.pkl', 'wb')
pickle.dump(dict_i, output)
output.close()
