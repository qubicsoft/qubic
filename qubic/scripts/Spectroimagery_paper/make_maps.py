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

from qubicpack.utilities import Qubic_DataDir

import ReadMC
from qubic import SpectroImLib as si

today = datetime.datetime.now().strftime('%Y%m%d')

# Repository for dictionary and input maps
if 'QUBIC_DATADIR' in os.environ:
    pass
else:
    raise NameError('You should define an environment variable QUBIC_DATADIR')

global_dir = Qubic_DataDir(datafile='instrument.py', datadir=os.environ['QUBIC_DATADIR'])
print('global_dir', global_dir)
if sys.argv[4].lower() == 'no':
    dictfilename = global_dir + '/dicts/spectroimaging.dict'
else:
    dictfilename = global_dir + '/dicts/' + sys.argv[4]

dictmaps = global_dir + '/scripts/Spectroimagery_paper/maps/'

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
nreals = int(sys.argv[3])

# Option multFactor x QU
multFactor = int(sys.argv[5])

d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)

# Check nf_sub/nf_sub_rec is an integer
nf_sub = d['nf_sub']
for nf_sub_rec in d['nf_recon']:
    if nf_sub % nf_sub_rec != 0:
        raise ValueError('nf_sub/nf_sub_rec must be an integer.')

# Check that we do one simulation with only one reconstructed subband
if d['nf_recon'][0] != 1:
    raise ValueError('You should do one simulation without spectroimaging as a reference.')

# Save the dictionary
shutil.copyfile(dictfilename, out_dir + name + '.dict')

# ==========================================

t0 = time.time()

# ===== Sky Creation or Reading =====

x0 = FitsArray(dictmaps + 'nf_sub={}/nside{}_nfsub{}.fits'.format(nf_sub, d['nside'], nf_sub))
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
x0[:, :, 1] *= multFactor
x0[:, :, 2] *= multFactor


# ==== Pointing strategy ====

p = qubic.get_pointing(d)
print('=== Pointing DONE! ===')

# =============== Noiseless ===================== #

# ==== TOD making ====
d['noiseless'] = True
TOD_noiseless, maps_convolved_noiseless = si.create_TOD(d, p, x0)
print('--------- Noiseless TOD with shape: {} - Done ---------'.format(np.shape(TOD_noiseless)))

# Reconstruction noiseless
for i, nf_sub_rec in enumerate(d['nf_recon']):
    print('************* Map-Making on {} sub-map(s) (noiseless) *************'.format(nf_sub_rec))

    maps_recon_noiseless, cov_noiseless, nus, nus_edge, maps_convolved_noiseless = si.reconstruct_maps(
        TOD_noiseless, d, p,
        nf_sub_rec, x0=x0)
    if nf_sub_rec == 1:
        print(maps_recon_noiseless.shape, maps_convolved_noiseless.shape)
        maps_recon_noiseless = np.reshape(maps_recon_noiseless, np.shape(maps_convolved_noiseless))
    # Look at the coverage of the sky
    cov_noiseless = np.sum(cov_noiseless, axis=0)
    maxcov_noiseless = np.max(cov_noiseless)
    unseen = cov_noiseless < maxcov_noiseless * 0.1
    maps_convolved_noiseless[:, unseen, :] = hp.UNSEEN
    maps_recon_noiseless[:, unseen, :] = hp.UNSEEN
    print('************* Map-Making on {} sub-map(s) (noiseless). Done *************'.format(nf_sub_rec))

    name_map = '_nfsub{0}_nfrecon{1}_noiseless{2}_nptg{3}_tol{4}_nep{5}_nside{6}.fits'.format(d['nf_sub'],
                                                                                     d['nf_recon'][i],
                                                                                     d['noiseless'],
                                                                                     d['npointings'],
                                                                                     d['tol'],
                                                                                     d['detector_nep'],
                                                                                     d['nside'])
    ReadMC.save_simu_fits(maps_recon_noiseless, cov_noiseless, nus, nus_edge, maps_convolved_noiseless,
                          out_dir, name + name_map)

# =============== With noise ===================== #
d['noiseless'] = False
for j in range(nreals):

    TOD, maps_convolved = si.create_TOD(d, p, x0)
    print('-------- Noise TOD with shape: {} - Realisation {} - Done --------'.format(np.shape(TOD), j))

    # ==== Reconstruction with spectroimaging ====
    for i, nf_sub_rec in enumerate(d['nf_recon']):
        print('************* Map-Making on {} sub-map(s) - Realisation {}*************'.format(nf_sub_rec, j))
        maps_recon, cov, nus, nus_edge, maps_convolved = si.reconstruct_maps(TOD, d, p, nf_sub_rec, x0=x0)
        if nf_sub_rec == 1:
            maps_recon = np.reshape(maps_recon, np.shape(maps_convolved))
        # Look at the coverage of the sky
        cov = np.sum(cov, axis=0)
        maxcov = np.max(cov)
        unseen = cov < maxcov * 0.1
        maps_convolved[:, unseen, :] = hp.UNSEEN
        maps_recon[:, unseen, :] = hp.UNSEEN
        print('************* Map-Making on {} sub-map(s) - Realisation {}. Done *************'.format(nf_sub_rec, j))

        name_map = '_nfsub{0}_nfrecon{1}_noiseless{2}_nptg{3}_tol{4}_nep{5}_nside{6}_{7}.fits'.format(d['nf_sub'],
                                                                                             d['nf_recon'][i],
                                                                                             d['noiseless'],
                                                                                             d['npointings'],
                                                                                             d['tol'],
                                                                                             d['detector_nep'],
                                                                                             d['nside'],
                                                                                             str(j).zfill(2))
        ReadMC.save_simu_fits(maps_recon, cov, nus, nus_edge, maps_convolved, out_dir, name + name_map)

t1 = time.time()
print('**************** All Done in {} minutes ******************'.format((t1 - t0) / 60))
