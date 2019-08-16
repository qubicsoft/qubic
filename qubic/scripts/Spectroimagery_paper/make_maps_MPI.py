from __future__ import division, print_function
import os
import sys
import time
import datetime
import shutil

import healpy as hp
import numpy as np

from pysimulators import FitsArray
import qubic

import ReadMC as rmc
import SpectroImLib as si

# MPI stuff
from mpi4py import MPI
# from pyoperators import MPI

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

if rank == 0:
    print('**************************')
    print('Master rank {} is speaking:'.format(rank))
    print('There are {} ranks'.format(size))
    print('**************************')

today = datetime.datetime.now().strftime('%Y%m%d')

# Repository for dictionary and input maps
global_dir = Qubic_DataDir(datafile='spectroimaging.dict')
dictfilename = global_dir + '/spectroimaging.dict'
dictmaps = global_dir + '/maps/'

# Repository for output maps
out_dir = sys.argv[1]
try:
    os.makedirs(out_dir)
except:
    pass

name = today + '_' + sys.argv[2]

# Number of noise realisations
nreals = int(sys.argv[3])

d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)

''' Parameters to be change for simulations:

[0] Sky creation:	d['nf_sub'] = 12 to 24 ? 4/5 diferent values?

[1] Pointing: 		d['random_pointing'] = True, 
                    d['npointings'] = [1000,1500,2000]

[2] TOD creation: 	d['noiseless'] = [True, False]
                    if False: change d['detector_nep'] only? 

[3] Reconstruction:	d['nf_sub_rec'] = [1,2,3,4,5,6,7,8] ? 
                    tol = [5e-4, 1e-4, 5e-5, 1e-5, 5e-6] :o

'''
# Check nf_sub/nf_sub_rec is an integer
nf_sub = d['nf_sub']
for nf_sub_rec in d['nf_recon']:
    if nf_sub % nf_sub_rec != 0:
        raise ValueError('nf_sub/nf_sub_rec must be an integer.')

# Check that we do one simulation with only one reconstructed subband
if d['nf_recon'][0] != 1:
    raise ValueError('You should do one simulation without spectroimaging as a reference.')

# Save the dictionary
if rank == 0:
    shutil.copyfile(dictfilename, out_dir + name + '.dict')

# ===== Sky Creation or Reading =====
# Done only on rank0 and shared after between all ranks
if rank == 0:
    t0 = time.time()
    x0 = FitsArray(dictmaps + 'nf_sub={}/nf_sub={}.fits'.format(d['nf_sub'], d['nf_sub']))
    print('Input Map with shape:', np.shape(x0))

    if x0.shape[1] % (12 * d['nside'] ** 2) == 0:
        print('Good size')
    else:
        y0 = np.ones((d['nf_sub'], 12 * d['nside'] ** 2, 3))
        for i in range(d['nf_sub']):
            for j in range(3):
                y0[i, :, j] = hp.ud_grade(x0[i, :, j], d['nside'])

    # Put I = 0
    # x0[:, :, 0] = 0.

    # Multiply Q, U maps
    # x0[:, :, 1] *= 100
    # x0[:, :, 2] *= 100

else:
    t0 = time.time()
    x0 = None

x0 = MPI.COMM_WORLD.bcast(x0)

# ==== Pointing strategy ====
# Pointing in not picklable so cannot be broadcasted
# => done on all ranks simultaneously
p = qubic.get_pointing(d)
print(rank, p.azimuth[:6], p.elevation[:6], p.pitch[:6])

MPI.COMM_WORLD.Barrier()

# =============== Noiseless ===================== #

d['noiseless'] = True
t1 = time.time()
print('-------------- Noiseless TOD - rank {} Starting --------------'.format(rank))
TOD_noiseless, maps_convolved_noiseless = si.create_TOD(d, p, x0)
print('-------------- Noiseless TOD with shape {} - Done in {} minutes on rank {} --------------'
      .format(np.shape(TOD_noiseless), (time.time() - t1) / 60, rank))

# Wait for all the TOD to be done (is it necessary ?)
MPI.COMM_WORLD.Barrier()
if rank == 0:
    print('-------------- All Noiseless TOD OK in {} minutes --------------'.format((time.time() - t1) / 60))

# Reconstruction noiseless
for i, nf_sub_rec in enumerate(d['nf_recon']):
    if rank == 0:
        print('************* Map-Making on {} sub-map(s) (noiseless) - Rank {} Starting *************'
              .format(nf_sub_rec, rank))

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
    if rank == 0:
        print('************* Map-Making on {} sub-map(s) (noiseless). Rank {} Done *************'
              .format(nf_sub_rec, rank))

    MPI.COMM_WORLD.Barrier()

    if rank == 0:
        name_map = '_nfsub{0}_nfrecon{1}_noiseless{2}_nptg{3}_tol{4}.fits'.format(d['nf_sub'],
                                                                              d['nf_recon'][i],
                                                                              d['noiseless'],
                                                                              d['npointings'],
                                                                              d['tol'])
        rmc.save_simu_fits(maps_recon_noiseless, cov_noiseless, nus, nus_edge, maps_convolved_noiseless,
                       out_dir, name + name_map)

# ==== TOD making ====
# TOD making is intrinsically parallelized (use of pyoperators)
d['Noiseless'] = False
for j in range(nreals):

    t2 = time.time()

    print('-------------- Noise TOD realisation {} - rank {} Starting --------------'.format(j, rank))
    TOD, maps_convolved = si.create_TOD(d, p, x0)
    print('-------------- Noise TOD with shape {} realisation {} - Done in {} minutes on rank {} --------------'
          .format(np.shape(TOD), j, (time.time() - t2) / 60, rank))

    # Wait for all the TOD to be done (is it necessary ?)
    MPI.COMM_WORLD.Barrier()
    if rank == 0:
        print('-------------- All Noise TOD realisation {} - Done in {} minutes --------------'
              .format(j, (time.time() - t2) / 60))

    # ==== Reconstruction ====
    for i, nf_sub_rec in enumerate(d['nf_recon']):
        if rank == 0:
            print('************* Map-Making on {} sub-map(s) - Realisation {} - Rank {} Starting *************'
                  .format(nf_sub_rec, j, rank))
        maps_recon, cov, nus, nus_edge, maps_convolved = si.reconstruct_maps(TOD, d, p, nf_sub_rec, x0=x0)
        if nf_sub_rec == 1:
            maps_recon = np.reshape(maps_recon, np.shape(maps_convolved))
        # Look at the coverage of the sky
        cov = np.sum(cov, axis=0)
        maxcov = np.max(cov)
        unseen = cov < maxcov * 0.1
        maps_convolved[:, unseen, :] = hp.UNSEEN
        maps_recon[:, unseen, :] = hp.UNSEEN
        if rank == 0:
            print('************* Map-Making on {} sub-map(s) - Realisation {} - Rank {} Done *************'
                  .format(nf_sub_rec, j, rank))

        MPI.COMM_WORLD.Barrier()

        if rank == 0:
            name_map = '_nfsub{0}_nfrecon{1}_noiseless{2}_nptg{3}_tol{4}_{5}.fits'.format(d['nf_sub'],
                                                                                          d['nf_recon'][i],
                                                                                          d['noiseless'],
                                                                                          d['npointings'],
                                                                                          d['tol'],
                                                                                          str(j).zfill(2))
            rmc.save_simu_fits(maps_recon, cov, nus, nus_edge, maps_convolved, out_dir, name + name_map)

        MPI.COMM_WORLD.Barrier()



print('============== All Done in {} minutes ================'.format((time.time() - t0) / 60))
