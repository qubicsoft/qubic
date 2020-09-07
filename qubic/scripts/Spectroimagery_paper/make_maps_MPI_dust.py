from __future__ import division, print_function
import os
import sys
import time
import datetime
import shutil

import healpy as hp
import numpy as np
import pylab as plt

from pysimulators import FitsArray
from qubic import QubicSkySim as qss
import qubic
import pysm
import pysm.units as u

from qubicpack.utilities import Qubic_DataDir

import qubic.ReadMC as rmc
import qubic.SpectroImLib as si

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print('**************************')
    print('Master rank {} is speaking:'.format(rank))
    print('There are {} ranks'.format(size))
    print('**************************')

today = datetime.datetime.now().strftime('%Y%m%d')

# Repository for dictionary and input maps
if 'QUBIC_DATADIR' in os.environ:
    pass
else:
    raise NameError('You should define an environment variable QUBIC_DATADIR')

global_dir = Qubic_DataDir(datafile='instrument.py', datadir=os.environ['QUBIC_DATADIR'])
# global_dir = '/global/homes/m/mmgamboa/qubicsoft/qubic'
dictfilename = global_dir + '/dicts/' + sys.argv[4]

# Repository for output files
out_dir = sys.argv[1]
if out_dir[-1] != '/':
    out_dir = out_dir + '/'

os.makedirs(out_dir, exist_ok=True)

# Name of the simulation
name = today + '_' + sys.argv[2]

# Number of noise realisations
nreals = int(sys.argv[3])

# Get the dictionary
d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)

# Check nf_sub/nf_sub_rec is an integer
nf_sub = d['nf_sub']
for nf_sub_rec in d['nf_recon']:
    if nf_sub % nf_sub_rec != 0:
        raise ValueError('nf_sub/nf_sub_rec must be an integer.')

# Check that we do one simulation with only one reconstructed subband
# if d['nf_recon'][0] != 1:
#     raise ValueError('You should do one simulation without spectroimaging as a reference.')

# Center
center = qubic.equ2gal(d['RA_center'], d['DEC_center'])
# Save the dictionary
if rank == 0:
    shutil.copyfile(dictfilename, out_dir + name + '.dict')

# ===== Sky Creation or Reading =====
# Done only on rank0 and shared after between all ranks
if rank == 0:
    t0 = time.time()
    # Make a sky using PYSM
    seed = 42
    sky_config = {'dust': 'd1'}
    Qubic_sky = qss.Qubic_sky(sky_config, d)
    x0 = Qubic_sky.get_simple_sky_map()
    print('Input map with shape:', x0.shape)
    # hp.mollview(x0[2,:,0],rot=center)
    # plt.savefig('test-dust')
    if x0.shape[1] % (12 * d['nside'] ** 2) == 0:
        print('Good size')
    else:
        y0 = np.ones((d['nf_sub'], 12 * d['nside'] ** 2, 3))
        for i in range(d['nf_sub']):
            for j in range(3):
                y0[i, :, j] = hp.ud_grade(x0[i, :, j], d['nside'])
else:
    t0 = time.time()
    x0 = None

x0 = comm.bcast(x0)

# ==== Pointing strategy ====
# Pointing in not picklable so cannot be broadcasted
# => done on all ranks simultaneously
p = qubic.get_pointing(d)

comm.Barrier()
if rank == 0:
    print('============= All pointings done ! =============')

# =============== Noiseless ===================== #

d['noiseless'] = True
t1 = time.time()
TOD_noiseless, maps_convolved_noiseless = si.create_TOD(d, p, x0)

# Wait for all the TOD to be done (is it necessary ?)
comm.Barrier()
if rank == 0:
    print('-------------- All Noiseless TOD OK in {} minutes --------------'.format((time.time() - t1) / 60))

# ======== Reconstruction noiseless
for i, nf_sub_rec in enumerate(d['nf_recon']):
    if rank == 0:
        print('************* Map-Making on {} sub-map(s) (noiseless). STARTING *************'
              .format(nf_sub_rec))

    maps_recon_noiseless, cov_noiseless, nus, nus_edge, maps_convolved_noiseless = si.reconstruct_maps(
        TOD_noiseless, d, p,
        nf_sub_rec, x0=x0)
    if nf_sub_rec == 1:
        maps_recon_noiseless = np.reshape(maps_recon_noiseless, np.shape(maps_convolved_noiseless))
    # Look at the coverage of the sky
    cov_noiseless = np.sum(cov_noiseless, axis=0)
    maxcov_noiseless = np.max(cov_noiseless)
    unseen = cov_noiseless < maxcov_noiseless * 0.1
    maps_convolved_noiseless[:, unseen, :] = hp.UNSEEN
    maps_recon_noiseless[:, unseen, :] = hp.UNSEEN

    comm.Barrier()

    if rank == 0:
        print('************* Map-Making on {} sub-map(s) (noiseless). DONE *************'
              .format(nf_sub_rec))

        name_map = '_nfsub{0}_nfrecon{1}_noiseless{2}_nptg{3}.fits'.format(d['nf_sub'],
                                                                           d['nf_recon'][i],
                                                                           d['noiseless'],
                                                                           d['npointings'])
        rmc.save_simu_fits(maps_recon_noiseless, cov_noiseless, nus, nus_edge, maps_convolved_noiseless,
                           out_dir, name + name_map)

# ==== TOD making ====
# TOD making is intrinsically parallelized (use of pyoperators)
d['noiseless'] = False
for j in range(nreals):

    t2 = time.time()

    TOD, maps_convolved = si.create_TOD(d, p, x0)

    # Wait for all the TOD to be done (is it necessary ?)
    comm.Barrier()
    if rank == 0:
        print('-------------- All Noise TOD realisations {} - DONE in {} minutes --------------'
              .format(j, (time.time() - t2) / 60))

    # ==== Reconstruction ====
    for i, nf_sub_rec in enumerate(d['nf_recon']):
        if rank == 0:
            print('************* Map-Making on {} sub-map(s) - Realisation {} - STARTING *************'
                  .format(nf_sub_rec, j))
        maps_recon, cov, nus, nus_edge, maps_convolved = si.reconstruct_maps(TOD, d, p, nf_sub_rec, x0=x0)
        if nf_sub_rec == 1:
            maps_recon = np.reshape(maps_recon, np.shape(maps_convolved))
        # Look at the coverage of the sky
        cov = np.sum(cov, axis=0)
        maxcov = np.max(cov)
        unseen = cov < maxcov * 0.1
        maps_convolved[:, unseen, :] = hp.UNSEEN
        maps_recon[:, unseen, :] = hp.UNSEEN

        comm.Barrier()
        if rank == 0:
            print('************* Map-Making on {} sub-map(s) - Realisation {} - DONE *************'
                  .format(nf_sub_rec, j))

            name_map = '_nfsub{0}_nfrecon{1}_noiseless{2}_nptg{3}_{4}.fits'.format(d['nf_sub'],
                                                                                   d['nf_recon'][i],
                                                                                   d['noiseless'],
                                                                                   d['npointings'],
                                                                                   str(j).zfill(2) )
            rmc.save_simu_fits(maps_recon, cov, nus, nus_edge, maps_convolved, out_dir, name + name_map)

        comm.Barrier()

if rank == 0:
    print('============== All Done in {} minutes ================'.format((time.time() - t0) / 60))
