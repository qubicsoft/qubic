from __future__ import division
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
# from mpi4py import MPI
from pyoperators import MPI

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

if rank == 0:
    print('**************************')
    print('Master rank {} is speaking:'.format(rank))
    print('There are {} ranks'.format(size))
    print('**************************')


dictfilename = './spectroimaging.dict'
dictmaps = 'maps/'
out_dir = './TEST/'
try:
    os.makedirs(out_dir)
except:
    pass

today = datetime.datetime.now().strftime('%Y%m%d')

name = today + '_' + sys.argv[1]

# Number of noise realisations
nreals = int(sys.argv[2])

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

# Save the dictionary
shutil.copyfile(dictfilename, out_dir + name + '.dict')

# ===== Sky Creation or Reading =====
# Done only on rank0 and shared after between all ranks
if rank == 0:
    t0 = time.time()
    x0 = FitsArray(dictmaps + 'nf_sub={}/nf_sub={}.fits'.format(d['nf_sub'], d['nf_sub']))
    print('Input Map with shape:', np.shape(x0))
else:
    t0 = time.time()
    x0 = None

x0 = MPI.COMM_WORLD.bcast(x0)

# ==== Pointing strategy ====
# Pointing in not picklable so cannot be broadcasted
# => done on all ranks simultaneously
p = qubic.get_pointing(d)

# ==== TOD making ====
# TOD making is intrinsically parallelized (use of pyoperators)
for j in range(nreals):

    t1 = time.time()

    print('-------------- Noise TOD realisation {} - rank {} Starting --------------'.format(j, rank))
    TOD, maps_convolved = si.create_TOD(d, p, x0)
    print('-------------- Noise TOD with shape {} realisation {} - Done in {} minutes on rank {} --------------'
          .format(np.shape(TOD), j, (time.time() - t1) / 60, rank))

    # Wait for all the TOD to be done (is it necessary ?)
    MPI.COMM_WORLD.Barrier()
    if rank == 0:
        print('-------------- All Noise TOD realisation {} - Done in {} minutes --------------'
              .format(j, (time.time() - t1) / 60))

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

# =============== Noiseless ===================== #

d['noiseless'] = True

t2 = time.time()
print('-------------- Noiseless TOD - rank {} Starting --------------'.format(rank))
TOD_noiseless, maps_convolved_noiseless = si.create_TOD(d, p, x0)
print('-------------- Noiseless TOD with shape {} - Done in {} minutes on rank {} --------------'
      .format(np.shape(TOD_noiseless), (time.time() - t2) / 60, rank))

# Wait for all the TOD to be done (is it necessary ?)
MPI.COMM_WORLD.Barrier()
if rank == 0:
    print('-------------- All Noiseless TOD OK in {} minutes --------------'.format((time.time() - t2) / 60))

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

    name_map = '_nfsub{0}_nfrecon{1}_noiseless{2}_nptg{3}_tol{4}.fits'.format(d['nf_sub'],
                                                                              d['nf_recon'][i],
                                                                              d['noiseless'],
                                                                              d['npointings'],
                                                                              d['tol'])
    rmc.save_simu_fits(maps_recon_noiseless, cov_noiseless, nus, nus_edge, maps_convolved_noiseless,
                       out_dir, name + name_map)

print('============== All Done in {} minutes ================'.format((time.time() - t0) / 60))

MPI.Finalize()
