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
import os.path as op
import configparser
import pickle
import matplotlib.pyplot as plt
import time
sys.path.append('/home/regnier/work/regnier/MapMaking')

seed = int(sys.argv[1])
iteration = int(sys.argv[2])
band = int(sys.argv[3])
ndet = int(sys.argv[4])
npho150 = int(sys.argv[5])
npho220 = int(sys.argv[6])
npointings = int(sys.argv[7])

ndet, npho150, npho220 = np.array([ndet, npho150, npho220], dtype=bool)

### QUBIC packages
import qubic
import frequency_acquisition as Acq

### PyOperators
from pyoperators import *

### PySimulators
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

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
relative_bandwidth = 0.25

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

d = get_dict({'npointings':npointings, 'nf_recon':nrec, 'nf_sub':nsub, 'nside':nside, 'MultiBand':True, 
              'filter_nu':filter_nu*1e9, 'noiseless':False, 'comm':comm, 'nprocs_sampling':1, 'nprocs_instrument':size,
              'photon_noise':True, 'nhwp_angles':3, 'effective_duration':3, 'filter_relative_bandwidth':relative_bandwidth, 
              'type_instrument':type})

center = qubic.equ2gal(0, -57)
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

invN = joint.get_invntt_operator(True, [True, True])

### Coverage
cov = a.get_coverage()
covnorm = cov/cov.max()


###############################################################
######################### Sky model ###########################
###############################################################

### We define foregrounds model
skyconfig = {'cmb':seed}
if dust:
    skyconfig['dust'] = 'd0'

f = int(nsub/nrec)
sky = Acq.Sky(skyconfig, a)
sky_fg = Acq.Sky({'dust':'d0'}, a)
m_nu_fg = sky_fg.scale_component(beta=1.54)

if dust:
    m_nu = sky.scale_component(beta=1.54)
else:
    m_nu = np.array([sky.cmb*np.ones(sky.cmb.shape)]*nsub)

if fake_convolution != 0:
    print(f'Convolving input sky by constant fwhm = {fake_convolution}')
    C = HealpixConvolutionGaussianOperator(fwhm=fake_convolution)
    for inu in range(m_nu.shape[0]):
        m_nu[inu] = C(m_nu[inu])
    
k=0
if bandpass_correction:
    for i in range(nrec):
        delta = m_nu_fg[i*f:(i+1)*f] - np.mean(m_nu_fg[i*f:(i+1)*f], axis=0)
        for j in range(f):
            m_nu[k] -= delta[j]
            k+=1


mean_sky = np.zeros((nrec, 12*nside**2, 3))
for i in range(nrec):
    if rank == 0:
        print(f'Doing average of m_nu between {np.min(a.allnus[i*f:(i+1)*f])} GHz and {np.max(a.allnus[i*f:(i+1)*f])} GHz')
    mean_sky[i] = np.mean(m_nu[i*f:(i+1)*f], axis=0)


###############################################################
######################### Making TODs #########################
###############################################################

print('\n***** Making TODs ******\n')

R = ReshapeOperator((12*nside**2, 3), 3*12*nside**2)
npho = False
if npho150 == True or npho220 == True:
    npho = True

comm.Barrier()
if rank == 0:
    np.random.seed(int(str(int(time.time()*1e6))[-8:]))
    seed_noise_pl = np.random.randint(1000000000)
else:
    seed_noise_pl = None

seed_noise_pl = comm.bcast(seed_noise_pl, root=0)


for i in range(size):
    print(f'seed for planck noise is {seed_noise_pl} for rank = {rank}')

if band == 150 or band == 220:
    nq = a.get_noise(bool(ndet), bool(npho), seed=None).ravel()
else:
    nd, np150, np220 = a.get_noise(bool(ndet), bool(npho150), bool(npho220))
    nq = nd.ravel()+np150.ravel()+np220.ravel()
print(nq)
TOD_QUBIC = Htod(m_nu).ravel() + nq

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
            
TOD = np.r_[TOD_QUBIC.ravel(), TOD_PLANCK.ravel()]

comm.Barrier()
###############################################################
########################## Operators ##########################
###############################################################

A = H.T * invN * H
b = H.T * invN * TOD

### Preconditionning
M = Acq.get_preconditioner(np.ones(12*nside**2))

### PCG
print('\n***** PCG ******\n')
solution_qubic_planck = pcg(A, b, x0=None, M=M, tol=1e-25, disp=True, maxiter=maxiter)

if doplot:
    if rank == 0:
        C = HealpixConvolutionGaussianOperator(fwhm=np.min(a.allfwhm))
        if convolution:
            C_reconv = HealpixConvolutionGaussianOperator(fwhm = np.sqrt(a.allfwhm[0]**2 - a.allfwhm[-1]**2))
        else:
            C_reconv = HealpixConvolutionGaussianOperator(fwhm = 0)

        if nrec == 1:
            plt.figure(figsize=(15, 5))

            hp.gnomview(C_reconv(solution_qubic_planck['x'][:, 1]), min=-8, max=8, cmap='jet', sub=(1, 3, 1), rot=center, reso=15)
            hp.gnomview(C_reconv(C(mean_sky[0, :, 1])), min=-8, max=8, cmap='jet', sub=(1, 3, 2), rot=center, reso=15)
            hp.gnomview(C_reconv(solution_qubic_planck['x'][:, 1])-C_reconv(C(mean_sky[0, :, 1])), min=-8, max=8, cmap='jet', sub=(1, 3, 3), rot=center, reso=15)
            plt.savefig(f'test_{seed}_{iteration}.png')
            plt.close()
        else:
            plt.figure(figsize=(15, 5))
            k = 0
            for irec in range(nrec):
                k+=1
                hp.gnomview(solution_qubic_planck['x'][irec, :, 1], min=-8, max=8, cmap='jet', sub=(nrec, 3, k), rot=center, reso=15)
                k+=1
                hp.gnomview(mean_sky[irec, :, 1], min=-8, max=8, cmap='jet', sub=(nrec, 3, k), rot=center, reso=15)
                k+=1
                hp.gnomview(solution_qubic_planck['x'][irec, :, 1]-mean_sky[irec, :, 1], min=-8, max=8, cmap='jet', sub=(nrec, 3, k), rot=center, reso=15)
            plt.savefig('test.png')
            plt.close()

end = time.time()
execution_time = end - t0

if rank == 0:
    print(f'Simulation done in {execution_time} s')

dict_i = {'output':solution_qubic_planck['x'], 'input':mean_sky, 'allfwhm':a.allfwhm, 'coverage':cov, 'center':center, 'nsub':nsub, 'nrec':nrec, 'execution_time':execution_time, 'size':size}



### If the folder is not here, you will create it
if rank == 0:
    save_each_ite = f'band{band}'
    current_path = os.getcwd() + '/'
    if not os.path.exists(current_path + save_each_ite):
        os.makedirs(current_path + save_each_ite)

    fullpath = current_path + save_each_ite + '/'
    if fake_convolution != 0:
        fc = 'fake'
    else:
        if convolution:
            fc = True
        else:
            fc = False
    output = open(fullpath+f'MM_maxiter{maxiter}_convolution{fc}_npointing{npointings}_nrec{nrec}_nsub{nsub}_ndet{ndet}_npho150{npho150}_npho220{npho220}_seed{seed}_iteration{iteration}.pkl', 'wb')
    pickle.dump(dict_i, output)
    output.close()
