from pylab import *
import os
import sys
import shutil

# Specific science modules
import healpy as hp
import numpy as np
import time
import matplotlib.pyplot as plt

# Specific qubic modules
from qubic import NamasterLib as nam
from pysimulators import FitsArray
#import ReadMC as rmc
import qubic
from qubic import SpectroImLib as si
import pickle as pk
from astropy.io import fits

from mpi4py import MPI
import logging

#rc('figure', figsize=(13, 10))
#rc('font', size=15)

def save_simu_fits(maps_recon, cov, nus, nus_edge, maps_convolved,
                   save_dir, simu_name):

    if save_dir[-1] != '/':
        save_dir = save_dir+'/'

    hdu_primary = fits.PrimaryHDU()
    hdu_recon = fits.ImageHDU(data=maps_recon, name='maps_recon')
    hdu_cov = fits.ImageHDU(data=cov, name='coverage')
    hdu_nus = fits.ImageHDU(data=nus, name='central_freq', )
    hdu_nus_edge = fits.ImageHDU(data=nus_edge, name='edge_freq')
    hdu_convolved = fits.ImageHDU(data=maps_convolved, name='maps_convolved')

    the_file = fits.HDUList([hdu_primary, hdu_recon, hdu_cov, hdu_nus,
                             hdu_nus_edge, hdu_convolved])
    the_file.writeto(save_dir + simu_name, 'warn')


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

jobid = sys.argv[5]
if rank == 0:
    print('**************************')
    print('Master rank {} is speaking:'.format(rank))
    print('There are {} ranks'.format(size))
    print('**************************')
    #Log 
    logging.basicConfig(filename=jobid+'.log',level=logging.DEBUG)
    logging.Formatter('%(asctime)s:%(name)s:%(message)s')

today = datetime.datetime.now().strftime('%Y%m%d')
tf = time.time()

global_dir = '/global/homes/m/mmgamboa/qubicsoft/qubic'
if sys.argv[4].lower() == 'no':
    dictfilename = global_dir + '/dicts/spectroimaging.dict'
else:
    dictfilename = sys.argv[4]
# Repository for output files
out_dir = sys.argv[1]
if out_dir[-1] != '/':
    out_dir = out_dir + '/'
try:
    os.makedirs(out_dir)
except:
    pass

# Name of the simulation
name = today + '_' + sys.argv[2]

# Number of noise realisations
amplitude = float(sys.argv[3])

# Get the dictionary
d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)
d['dtheta'] = 10
d['nf_sub'] = eval(sys.argv[6])
d['nf_recon'] = eval(sys.argv[7])
d['npointings'] = eval(sys.argv[8])
d['tol'] = eval(sys.argv[9])
NREALS = eval(sys.argv[10])
NU0 = eval(sys.argv[11])

logging.debug('nf_sub {} and type {}'.format(d['nf_sub'], type(d['nf_sub'])))
logging.debug('nf_recon {} and type {}'.format(d['nf_recon'], type(d['nf_recon'])))
logging.debug('npointings {} and type {}'.format(d['npointings'], type(d['npointings'])))
logging.debug('tol {} and type {}'.format(d['tol'], type(d['tol'])))
comm.Barrier()

if rank == 0:
    shutil.copyfile(dictfilename, out_dir + name + '.dict')

def f(val, fwhm, sigma2fwhm):
    return np.nan_to_num(np.exp(-0.5*val**2/(np.radians(fwhm)/sigma2fwhm)**2))

# Center of the patch observed in galactic coordinates
center = qubic.equ2gal(d['RA_center'], d['DEC_center'])
    
#Compute freqs:
_, nus_edge_in, nus_in, _, _, _ = qubic.compute_freq(d['filter_nu'] / 1e9, 
                                    d['nf_sub'],
                                    d['filter_relative_bandwidth'])

# Choose freq where the point source will be placed (in frequency space)
innu0 = min(range(len(nus_in)), key=lambda i: abs(nus_in[i]-NU0))
innu0 += int(sys.argv[12])

nside = d['nside']
 
x0 = np.zeros((d['nf_sub'], 12*nside**2, 3))
psA = hp.pixelfunc.ang2pix(nside, np.deg2rad(90-center[1]), np.deg2rad(center[0]))
comm.Barrier()

#Create input map
if rank == 0:
    vecA = hp.pix2vec(nside, psA)
    valsA = hp.pix2vec(nside, np.arange(12*nside**2))
    ang_valsA = np.arccos(np.dot(vecA,valsA))

    sigma2fwhm=np.sqrt(8*np.log(2))
    cte = 64.57622#61.347409
    nu=np.array([nus_in[innu0],])
    fwhm_in = cte/nu # nus to fwhm

    # Not use a single pixel painted because rise rings when smoothing the map
        
    logging.info(': Amplitud {}'.format(amplitude) ) 
    logging.info(': nu0 {:.2f} {} {} {}'.format(NU0,nus_in[innu0],innu0, sys.argv[12]) ) 
    logging.info('DEBUGGING: f()ang_vals {}'.format(f(ang_valsA, fwhm_in[0], sigma2fwhm)) ) 
    logging.info('DEBUGGING shape {}'.format(np.shape(ang_valsA)  ) )    
    logging.info('DEBUGGING ', )
        
    x0[innu0,:,0] += amplitude*f(ang_valsA, fwhm_in[0], sigma2fwhm)

#Comparto en todos los nodo

x0 = comm.bcast(x0)
psA = comm.bcast(psA)
#Monte-carlo for map-making
mapsrec = {}

for ireal in range(NREALS):
    # Pointing strategy
    p = qubic.get_pointing(d)
  
    comm.Barrier()

    if rank == 0: print('=== Pointing DONE! ===')

    # ==== TOD making ====
    logging.debug('pointing {}'.format(p))
    logging.debug('x0 shape {}'.format(x0.shape))

    s = qubic.QubicScene(d)
    
    q = qubic.QubicMultibandInstrument(d)

    # number of sub frequencies to build the TOD
    _, nus_edge_in, _, _, _, _ = qubic.compute_freq(d['filter_nu'] / 1e9, d['nf_sub'],  # Multiband instrument model
                                                        d['filter_relative_bandwidth'])
    
    atod = qubic.QubicMultibandAcquisition(q, p, s, d, nus_edge_in)

    TOD = atod.get_observation(x0, noiseless=d['noiseless'], convolution=False)
    #TOD, maps_convolved = si.create_TOD(d, p, x0)
    
    comm.Barrier()
    
    if rank == 0: print('--------- TOD with shape: {} - Done ---------'.format(np.shape(TOD)))
    
    t0 = time.time()
    for i, nf_sub_rec in enumerate(d['nf_recon']):
        
        _, nus_edge, nus, _, _, _ = qubic.compute_freq(d['filter_nu'] / 1e9, nf_sub_rec, d['filter_relative_bandwidth'])
        arec = qubic.QubicMultibandAcquisition(q, p, s, d, nus_edge)
        cov = arec.get_coverage()
        maps_recon, nit, error = arec.tod2map(TOD, d, cov=cov)
        _, maps_convolved = arec.get_observation(x0, noiseless=d['noiseless'], convolution=True)
 
        #maps_recon, cov, nus, nus_edge, maps_convolved = si.reconstruct_maps(TOD, d, p,
        #                                                                nf_sub_rec, x0=x0)
        
        if nf_sub_rec == 1:
            maps_recon = np.reshape(maps_recon, np.shape(maps_convolved))
        comm.Barrier()

        # Look at the coverage of the sky
        if rank==0:
            cov = np.sum(cov, axis=0)
            maxcov = np.max(cov)
            unseen = cov < maxcov * 0.1
            maps_convolved=np.array(maps_convolved)
            print('debug shapes', np.shape(maps_convolved), type(maps_convolved), type(maps_convolved[0]) )
            print(np.shape(unseen))
            maps_convolved[:,unseen, :] = hp.UNSEEN
            maps_recon[:, unseen, :] = hp.UNSEEN

            #comm.Barrier()

            #if rank == 0:
            logging.info('DEBUGGING: ************* Map-Making on {} sub-map(s) - DONE *************'
                      .format(nf_sub_rec, ))

            name_map = '_nfsub{0}_nfrecon{1}_{2}.fits'.format(d['nf_sub'], d['nf_recon'][i], ireal)

            save_simu_fits(maps_recon, cov, nus, nus_edge, maps_convolved, out_dir, name + name_map)

        comm.Barrier()


    if rank == 0: print('Uhf, done it in {:3.2f} minutes'.format( (time.time() - t0) /60 ) )

if rank == 0:
    print('============== All Done in {} minutes ================'.format((time.time() - tf) / 60))


