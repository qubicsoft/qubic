from __future__ import division
import sys
import time

import healpy as hp
import numpy as np

from astropy.io import fits

from pysm.nominal import models

import qubic
from pysimulators import FitsArray

import SpectroImLib as si

dictfilename = '/home/louisemousset/QUBIC/MyGitQUBIC/qubic/qubic/scripts' \
               '/Spectroimagery_paper/spectroimaging.dict'

out_dir = '/home/louisemousset/QUBIC/Qubic_work/SpectroImagerie/SimuLouise/repeat_ptg/'
name = 'repeat_pointing_01'

# Numbers of subbands for spectroimaging
nrecon = [2]


d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)

# Print dictionary and others parameters
# Save a file with al parameters
tem = sys.stdout
sys.stdout = f = open(out_dir + name + '.txt', 'wt')

print('Simulation General Name: ' + name)
print('Dictionnary File: ' + dictfilename)
for k in d.keys():
    print(k, d[k])

print('Minimum Number of Sub Frequencies: {}'.format(noutmin))
print('Maximum Number of Sub Frequencies: {}'.format(noutmax))

sys.stdout = tem
f.close()

# ==========================================

t0 = time.time()

# ===== Sky Creation =====
sky_config = {
    'dust': models('d1', d['nside']),
    'cmb': models('c1', d['nside'])}

Qubic_sky = si.Qubic_sky(sky_config, d)
x0 = Qubic_sky.get_simple_sky_map()

hp.mollview(x0[0, :, 0])


# ==== Pointing strategy ====
p = qubic.get_pointing(d)


# ==== TOD making ====
TOD, maps_convolved = si.create_TOD(d, p, x0)

# ==== Reconstruction ====
for nf_sub_rec in nrecon: # np.arange(noutmin, noutmax + 1):
    print(nf_sub_rec)
    print('-------------------------- Map-Making on {} sub-map(s)'.format(nf_sub_rec))
    maps_recon, cov, nus, nus_edge, maps_convolved = si.reconstruct_maps(TOD, d, p, nf_sub_rec, x0=x0)
    # if nf_sub_rec == 1:
    #     maps_recon = np.reshape(maps_recon, np.shape(maps_convolved))
    # Look at the coverage of the sky
    cov = np.sum(cov, axis=0)
    maxcov = np.max(cov)
    unseen = cov < maxcov * 0.1
    maps_convolved[:, unseen, :] = hp.UNSEEN
    maps_recon[:, unseen, :] = hp.UNSEEN
    print('************************** Map-Making on {} sub-map(s)Done'.format(nf_sub_rec))

    # hdu_convolved = fits.PrimaryHDU()
    # hdu_recon.data =

    # FitsArray(nus_edge).save(name + '_nf{0}_ptg{1}'.format(nf_sub_rec, ptg) + '_nus_edges.fits')
    # FitsArray(nus).save(name + '_nf{0}_ptg{1}'.format(nf_sub_rec, ptg)+ '_nus.fits')
    FitsArray(maps_convolved).save(out_dir + name + '_nf{}_maps_convolved.fits'.format(nf_sub_rec))
    FitsArray(maps_recon).save(out_dir + name + '_nf{}_maps_recon.fits'.format(nf_sub_rec))

    t1 = time.time()
    print('************************** All Done in {} minutes'.format((t1 - t0) / 60))

