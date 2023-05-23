#########################################################################################################################
#########################################################################################################################
#########                                                                                                       #########
#########      This script allow to perform the frequency map-making for QUBIC experiments. We show here how to #########
#########      take into account the angular resolution and the bandpass mismatch in the case of the Wide Band  #########
#########      instrument.                                                                                      #########
#########                                                                                                       #########
#########################################################################################################################
#########################################################################################################################

### General importations
import numpy as np
import healpy as hp
import pysm3.units as u
from pysm3 import utils
import sys
import os
path = os.path.dirname(os.getcwd()) + '/data/'
sys.path.append(os.path.dirname(os.getcwd()))
import os.path as op
import configparser
import pickle
import matplotlib.pyplot as plt
import time

### PyOperators
from pyoperators import *

### Modfied PCG to display execution time
from cg import pcg


### Variables
seed = int(sys.argv[1])
iteration = int(sys.argv[2])
band = int(sys.argv[3])
ndet = int(sys.argv[4])
npho150 = int(sys.argv[5])
npho220 = int(sys.argv[6])

ndet, npho150, npho220 = np.array([ndet, npho150, npho220], dtype=bool)

### QUBIC packages
import qubic
import frequency_acquisition as Acq
from noise_timeline import QubicNoise, QubicWideBandNoise
from planck_timeline import ExternalData2Timeline

### PySimulators
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

### MPI common arguments
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

t0 = time.time()

if rank == 0:
    print(f'You resquested for {size} processus.')

###############################################################
##################### Useful definitions ######################
###############################################################

def get_pySM_maps(sky, nu):
    return np.array(sky.get_emission(nu * u.GHz, None).T * utils.bandpass_unit_conversion(nu*u.GHz, None, u.uK_CMB))


def get_dict(args={}):
    
    '''
    Function for modify the qubic dictionary.
    '''
    ### Get the default dictionary
    dictfilename = 'dicts/pipeline_demo.dict'
    d = qubic.qubicdict.qubicDict()
    d.read_from_file(dictfilename)
    d['npointings'] = 9
    for i in args.keys():
        
        d[str(i)] = args[i]
    
    return d
def load_config(config_file):

    '''
    Read the configuration file.
    '''

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

### Read configuration file
load_config('config.ini')
relative_bandwidth = 0.25

### Configuration panel
if rank == 0:
    print('Configuration of the simulation :\n')
    print(f'   Nside            : {nside}')
    print(f'   Band             : {band}')
    print(f'   Npointing        : {npointings}')
    print(f'   Nrec             : {nrec}')
    print(f'   Nsub             : {nsub}')
    print(f'   Seed CMB         : {seed}')
    print(f'   Iteration CMB    : {iteration}')
    print('\nNoise configuration  :\n')
    print(f'   Detector noise   : {ndet}')
    print(f'   Photon noise 150 : {npho150}')
    print(f'   Photon noise 220 : {npho220}')

###############################################################
########################## Dictionary #########################
###############################################################

if band == 150 or band == 220:
    type = ''
    filter_nu = band
else:
    type = 'wide'
    filter_nu = 220


### Dictionary for reconstruction
d = get_dict({'npointings':npointings, 'nf_recon':nrec, 'nf_sub':nsub, 'nside':nside, 'MultiBand':True, 'period':1,
              'filter_nu':filter_nu*1e9, 'noiseless':False, 'comm':comm, 'nprocs_sampling':1, 'nprocs_instrument':size,
              'photon_noise':True, 'nhwp_angles':3, 'effective_duration':3, 'filter_relative_bandwidth':relative_bandwidth, 
              'type_instrument':type, 'TemperatureAtmosphere150':None, 'TemperatureAtmosphere220':None,
              'EmissivityAtmosphere150':None, 'EmissivityAtmosphere220':None})

### Dictionary for noise generation
dmono = get_dict({'npointings':npointings, 'nf_recon':1, 'nf_sub':1, 'nside':nside, 'MultiBand':True, 'period':1,
              'filter_nu':filter_nu*1e9, 'noiseless':False, 'comm':comm, 'nprocs_sampling':1, 'nprocs_instrument':size,
              'photon_noise':True, 'nhwp_angles':3, 'effective_duration':3, 'filter_relative_bandwidth':relative_bandwidth, 
              'type_instrument':type, 'TemperatureAtmosphere150':None, 'TemperatureAtmosphere220':None,
              'EmissivityAtmosphere150':None, 'EmissivityAtmosphere220':None})



center = qubic.equ2gal(0, -57)

### Define acquisitions
if band == 150 or band == 220:
    a = Acq.QubicIntegrated(d, Nsub=nsub, Nrec=nrec)
    atod = Acq.QubicIntegrated(d, Nsub=nsub, Nrec=nsub)
    planck_acquisition = Acq.PlanckAcquisition(band_planck, a.scene)
    joint = Acq.QubicPlanckMultiBandAcquisition(a, planck_acquisition)
else:
    a = Acq.QubicFullBand(d, Nsub=nsub, Nrec=nrec)
    atod = Acq.QubicFullBand(d, Nsub=nsub, Nrec=nsub)
    planck_acquisition143 = Acq.PlanckAcquisition(143, a.scene)
    planck_acquisition217 = Acq.PlanckAcquisition(217, a.scene)
    joint = Acq.QubicPlanckMultiBandAcquisition(a, [planck_acquisition143, planck_acquisition217])


print(a.allnus)
myfwhm = np.sqrt(a.allfwhm**2 - a.allfwhm[-1]**2)
H = joint.get_operator(convolution=convolution, myfwhm=myfwhm)
Htod = atod.get_operator(convolution=convolution)

### Compute inverse-noise covariance matrix in time domain
if band == 150 or band == 220:
    invN = joint.get_invntt_operator(True, True)
    cov = a.get_coverage()
else:
    invN = joint.get_invntt_operator(True, [True, True])
    cov = a.get_coverage()[0]

### Coverage
covnorm = cov/cov.max()
seenpix = covnorm > covcut

###############################################################
######################### Sky model ###########################
###############################################################

### We define components
skyconfig = {'cmb':seed}
if dust:
    skyconfig['dust'] = 'd0'

### Sub-maps
m_nu = ExternalData2Timeline(skyconfig, a.allnus, nrec, nside=nside, corrected_bandpass=bandpass_correction).m_nu

### Frequency maps
mean_sky = ExternalData2Timeline(skyconfig, a.allnus, nrec, nside=nside, corrected_bandpass=bandpass_correction).maps

###############################################################
######################### Making TODs #########################
###############################################################

### Factor sub
f = int(nsub/nrec)

if rank == 0:
    print('\n***** Making TODs ******\n')


### Define noise configuration
R = ReshapeOperator((12*nside**2, 3), 3*12*nside**2)
npho = False
if npho150 == True or npho220 == True:
    npho = True


### Fix the seed of Planck data for noise
comm.Barrier()
if rank == 0:
    np.random.seed(int(str(int(time.time()*1e6))[-8:]))
    seed_noise_pl = np.random.randint(1000000000)
else:
    seed_noise_pl = None

### Share seed between all precessus
seed_noise_pl = comm.bcast(seed_noise_pl, root=0)

if rank == 0:
    for i in range(size):
        print(f'seed for planck noise is {seed_noise_pl} for rank = {rank}')

### Create QUBIC noise
if band == 150 or band == 220:
    nq = QubicNoise(band, npointings, comm=comm, size=d['nprocs_instrument']).total_noise(int(ndet), int(npho)).ravel()
else:
    nq = QubicWideBandNoise(d, npointings).total_noise(int(ndet), int(npho150), int(npho220)).ravel()

### QUBIC TODs
TOD_QUBIC = Htod(m_nu).ravel() + nq


### Create Planck TODs
if band == 150 or band == 220:
    TOD_PLANCK = np.zeros((nrec, 12*nside**2, 3))
    npl = planck_acquisition.get_noise(seed_noise_pl) * level_planck_noise
    for irec in range(nrec):
        if convolution:
            C = HealpixConvolutionGaussianOperator(fwhm=np.min(a.allfwhm[irec*f:(irec+1)*f]))
        else:
            C = HealpixConvolutionGaussianOperator(fwhm=0)
        TOD_PLANCK[irec] = C(mean_sky[irec] + npl)
    TOD_PLANCK = TOD_PLANCK.ravel()
else:
    npl143 = planck_acquisition143.get_noise(iteration) * level_planck_noise
    npl217 = planck_acquisition217.get_noise(iteration) * level_planck_noise
    
    if nrec != 1:
        TOD_PLANCK = np.zeros((nrec, 12*nside**2, 3))
        for irec in range(int(nrec/2)):
            if convolution:
                C = HealpixConvolutionGaussianOperator(fwhm=np.min(a.allfwhm[irec*f:(irec+1)*f]))
            else:
                C = HealpixConvolutionGaussianOperator(fwhm=0)
        
            TOD_PLANCK[irec] = C(mean_sky[irec] + npl143)

        for irec in range(int(nrec/2), nrec):
            if convolution:
                C = HealpixConvolutionGaussianOperator(fwhm=np.min(a.allfwhm[irec*f:(irec+1)*f]))
            else:
                C = HealpixConvolutionGaussianOperator(fwhm=0)
        
            TOD_PLANCK[irec] = C(mean_sky[irec] + npl217)
        
    else:
        TOD_PLANCK = np.zeros((2*nrec, 12*nside**2, 3))
        if convolution:
            C = HealpixConvolutionGaussianOperator(fwhm=a.allfwhm[-1])
        else:
            C = HealpixConvolutionGaussianOperator(fwhm=0)

        TOD_PLANCK[0] = C(mean_sky[0] + npl143)
        TOD_PLANCK[1] = C(mean_sky[0] + npl217)
    
    TOD_PLANCK = TOD_PLANCK.ravel()

### Full TOD   
TOD = np.r_[TOD_QUBIC.ravel(), TOD_PLANCK.ravel()]

### Wait for all processus
comm.Barrier()
###############################################################
########################## Operators ##########################
###############################################################

R = ReshapeOperator((1, 12*nside**2, 3), (12*nside**2, 3))

### Unpack Operator to fix some pixel during the PCG
U = (
    ReshapeOperator((nrec * sum(seenpix) * 3), (nrec, sum(seenpix), 3)) *
    PackOperator(np.broadcast_to(seenpix[None, :, None], (nrec, seenpix.size, 3)).copy())
).T

### Compute A and b
with rule_manager(none=True):
    if nrec == 1:
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
if rank == 0:
    print('\n***** PCG ******\n')
solution_qubic_planck = pcg(A, b, x0=None, M=M, tol=1e-25, disp=True, maxiter=maxiter)

mysolution = mean_sky.copy()
mysolution[:, ~seenpix] = hp.UNSEEN
mean_sky[:, ~seenpix] = hp.UNSEEN
if nrec == 1:
    mysolution[:, seenpix] = solution_qubic_planck['x'].copy()
else:
    mysolution[:, seenpix] = solution_qubic_planck['x'].copy()


if doplot:
    if rank == 0:
        if convolution:
            Creconv = HealpixConvolutionGaussianOperator(fwhm=np.max(a.allfwhm))
            C = HealpixConvolutionGaussianOperator(fwhm=np.min(a.allfwhm))
        else:
            Creconv = HealpixConvolutionGaussianOperator(fwhm=0)
            C = HealpixConvolutionGaussianOperator(fwhm=0)
        
        if nrec == 1:
            plt.figure(figsize=(15, 5))

            hp.gnomview(Creconv(mysolution[0, :, 1]), min=-8, max=8, cmap='jet', sub=(1, 3, 1), rot=center, reso=15)
            hp.gnomview(Creconv(C(mean_sky[0, :, 1])), min=-8, max=8, cmap='jet', sub=(1, 3, 2), rot=center, reso=15)
            res = Creconv(mysolution[0, :, 1])-Creconv(C(mean_sky[0, :, 1]))
            res[~seenpix] = hp.UNSEEN
            hp.gnomview(res, min=-4, max=4, cmap='jet', sub=(1, 3, 3), rot=center, reso=15)
            plt.savefig(f'band{band}_ndet{ndet}_npho150{npho150}_npho220{npho220}_{seed}_{iteration}.png')
            plt.close()
        else:
            plt.figure(figsize=(15, 5))
            k = 0
            for irec in range(nrec):
                k+=1
                hp.gnomview(mysolution[irec, :, 1], min=-8, max=8, cmap='jet', sub=(nrec, 3, k), rot=center, reso=15)
                k+=1
                hp.gnomview(mean_sky[irec, :, 1], min=-8, max=8, cmap='jet', sub=(nrec, 3, k), rot=center, reso=15)
                k+=1
                res = mysolution[irec, :, 1]-mean_sky[irec, :, 1]
                res[~seenpix] = hp.UNSEEN
                hp.gnomview(res, min=-4, max=4, cmap='jet', sub=(nrec, 3, k), rot=center, reso=15)
            plt.savefig(f'band{band}_ndet{ndet}_npho150{npho150}_npho220{npho220}_{seed}_{iteration}.png')
            plt.close()

end = time.time()
execution_time = end - t0

if rank == 0:
    print(f'Simulation done in {execution_time} s')

dict_i = {'output':mysolution, 'input':mean_sky, 'allfwhm':a.allfwhm, 'coverage':cov, 'center':center, 'nsub':nsub, 'nrec':nrec, 
          'covcut':covcut, 'execution_time':execution_time, 'size':size}



### If the folder is not here, you will create it
if rank == 0:
    save_each_ite = f'band{band}'
    current_path = os.getcwd() + '/'
    if not os.path.exists(current_path + save_each_ite):
        os.makedirs(current_path + save_each_ite)

    fullpath = current_path + save_each_ite + '/'
    output = open(fullpath+f'MM_maxiter{maxiter}_convolution{convolution}_npointing{npointings}_nrec{nrec}_nsub{nsub}_ndet{ndet}_npho150{npho150}_npho220{npho220}_seed{seed}_iteration{iteration}.pkl', 'wb')
    pickle.dump(dict_i, output)
    output.close()