import os
import sys
import numpy as np
from itertools import combinations_with_replacement

import qubic
from qubicpack.utilities import Qubic_DataDir
from qubic import QubicSkySim as qss
from qubic import NamasterLib as nam

# To run the script: $ python Fastsim_spectroimMC.py rep_save nbands config

# Repository for dictionary and input maps
if 'QUBIC_DATADIR' in os.environ:
    pass
else:
    raise NameError('You should define an environment variable QUBIC_DATADIR')

global_dir = Qubic_DataDir(datafile='instrument.py', datadir=os.environ['QUBIC_DATADIR'])
print('global directory:', global_dir)

# Repository where maps and spectra will be saved
rep_save = sys.argv[1]

# Get a dictionary
dictionary = global_dir + '/dicts/spectroimaging_article.dict'
d = qubic.qubicdict.qubicDict()
d.read_from_file(dictionary)

# Number of bands
nbands = int(sys.argv[2])
d['nf_recon'] = nbands
d['nf_sub'] = nbands

all_nfrecon = [1, 2, 3, 4, 5, 8]
if nbands not in all_nfrecon:
    raise ValueError('Wrong number of subbands !')
print('\n nbands:', nbands)

# Config and frequency
config = sys.argv[3]
# Beam correction for Namaster
if config == 'FI150':
    beam_corr = True
elif config == 'FI220':
    beam_corr = 0.279
else:
    raise ValueError('The config should be FI150 or FI220')

d['filter_nu'] = int(config[-3:]) * 1e9

# Input sky
# seed = 42
# sky_config = {'dust': 'd1', 'cmb':seed, 'synchrotron':'s1'}
sky_config = {'dust': 'd1'}
Qubic_sky = qss.Qubic_sky(sky_config, d)
inputmaps = Qubic_sky.get_fullsky_convolved_maps(FWHMdeg=None, verbose=True)

rnd_name = qss.random_string(10)

# ================== Make maps =============================
# Getting noise realisations with FastSimulator
nreals = 6
npix = 12 * d['nside'] ** 2
noisemaps = np.zeros((nreals, nbands, npix, 3))

# qubic_coverage = np.load('/pbs/home/l/lmousset/libs/qubic/qubic/scripts/Spectroimagery_paper/maps/'
#                          'coverage_nfsub15_nptgs10000_qubicpatch.npy')

for r in range(nreals):
    noisemaps[r, ...], coverage = Qubic_sky.get_partial_sky_maps_withnoise(coverage=None,
                                                                           noise_only=True,
                                                                           spatial_noise=True)

# Make maps QUBIC = noise + CMB
qubicmaps = np.zeros_like(noisemaps)
for r in range(nreals):
    qubicmaps[r, ...] = noisemaps[r, ...] + inputmaps

unseen = coverage < np.max(coverage) * 0.1
seenmap = np.invert(unseen)
qubicmaps[:, :, unseen, :] = 0.
noisemaps[:, :, unseen, :] = 0.
inputmaps[:, unseen, :] = 0.

# Reduce it to a patch
noisepatch = noisemaps[:, :, seenmap, :]

# Save the noisy patch
np.save(rep_save + f'/noisepatch_nbands{nbands}_' + config + '_v1_galaxycenter_' + rnd_name + '.npy',
        noisepatch)

# ================== Load maps already done =============================
# imap = int(sys.argv[3])
# files = glob.glob(rep_save + f'/patch_clth_nfrecon{nfrecon}*.npy')
# print(files[imap])
# patch_clth = np.load(files[imap])
# nreals = patch_clth.shape[0]
# print(f'nreals: {nreals}')
#
# maps_clth = np.zeros((nreals, nfrecon, 12*d['nside']**2, 3))
# maps_clth[:, :, seenmap, :] = patch_clth

# ================== Power spectrum =============================
print('\n =============== Starting power spectrum ================')
# Create a Namaster object
lmin = 40
lmax = 2 * d['nside'] - 1
delta_ell = 30

mask = np.zeros(12 * d['nside'] ** 2)
mask[seenmap] = 1
Namaster = nam.Namaster(mask, lmin=lmin, lmax=lmax, delta_ell=delta_ell)
mask_apo = Namaster.get_apodized_mask()

ell_binned, b = Namaster.get_binning(d['nside'])
nbins = len(ell_binned)
print('lmin:', lmin)
print('lmax:', lmax)
print('delta_ell:', delta_ell)
print('nbins:', nbins)
print('ell binned:', ell_binned)

# Possible combinations between bands
combi = list(combinations_with_replacement(np.arange(nbands), 2))
ncombi = len(combi)
print('combi:', combi)
print('ncombi:', ncombi)


# Cross spectrum between bands but same real
print('\n =============== Cross spectrum same real starting ================')

cross_samereal_qubicmaps = np.zeros((nreals, ncombi, nbins, 4))
cross_samereal_noisemaps = np.zeros((nreals, ncombi, nbins, 4))

w = None
for real in range(nreals):
    print(f'\n Real {real}')
    for i, (band1, band2) in enumerate(combi):
        print(f'Bands {band1} {band2}')
        map1 = qubicmaps[real, band1, :, :]
        map2 = qubicmaps[real, band2, :, :]
        leff, cross_samereal_qubicmaps[real, i, :, :], w = Namaster.get_spectra(map1.T,
                                                                                mask_apo,
                                                                                map2.T,
                                                                                w=w,
                                                                                purify_e=True,
                                                                                purify_b=False,
                                                                                beam_correction=beam_corr,
                                                                                pixwin_correction=True)

        map1noise = noisemaps[real, band1, :, :]
        map2noise = noisemaps[real, band2, :, :]
        leff, cross_samereal_noisemaps[real, i, :, :], w = Namaster.get_spectra(map1noise.T,
                                                                                mask_apo,
                                                                                map2noise.T,
                                                                                w=w,
                                                                                purify_e=True,
                                                                                purify_b=False,
                                                                                beam_correction=beam_corr,
                                                                                pixwin_correction=True)
np.save(
    rep_save + f'/cross_interband_samereal_nfrecon{nbands}_qubicmaps_' + config + '_v1_galaxycenter_' + rnd_name + '.npy',
    cross_samereal_qubicmaps)
np.save(
    rep_save + f'/cross_interband_samereal_nfrecon{nbands}_noisemaps_' + config + '_v1_galaxycenter_' + rnd_name + '.npy',
    cross_samereal_noisemaps)

# Cross spectrum between bands with different real
print('\n =============== Cross spectrum mixing reals starting ================')

ncross = nreals // 2
print('ncross:', ncross)
cross_mixreals_qubicmaps = np.zeros((ncross, ncombi, nbins, 4))
cross_mixreals_noisemaps = np.zeros((ncross, ncombi, nbins, 4))

w = None
cross = 0
for c1 in range(0, nreals - 1, 2):  # do not mix pairs to avoid correlation
    c2 = c1 + 1
    print(f'\n Reals {c1} {c2}')
    for i, (band1, band2) in enumerate(combi):
        print(f'Bands {band1} {band2}')
        map1 = qubicmaps[c1, band1, :, :]
        map2 = qubicmaps[c2, band2, :, :]
        leff, cross_mixreals_qubicmaps[cross, i, :, :], w = Namaster.get_spectra(map1.T,
                                                                                 mask_apo,
                                                                                 map2.T,
                                                                                 w=w,
                                                                                 purify_e=True,
                                                                                 purify_b=False,
                                                                                 beam_correction=beam_corr,
                                                                                 pixwin_correction=True)

        map1noise = noisemaps[c1, band1, :, :]
        map2noise = noisemaps[c2, band2, :, :]
        leff, cross_mixreals_noisemaps[cross, i, :, :], w = Namaster.get_spectra(map1noise.T,
                                                                                 mask_apo,
                                                                                 map2noise.T,
                                                                                 w=w,
                                                                                 purify_e=True,
                                                                                 purify_b=False,
                                                                                 beam_correction=beam_corr,
                                                                                 pixwin_correction=True)
    cross += 1

np.save(
    rep_save + f'/cross_interband_mixreal_nfrecon{nbands}_qubicmaps_' + config + '_v1_galaxycenter_' + rnd_name + '.npy',
    cross_mixreals_qubicmaps)

np.save(
    rep_save + f'/cross_interband_mixreal_nfrecon{nbands}_noisemaps_' + config + '_v1_galaxycenter_' + rnd_name + '.npy',
    cross_mixreals_noisemaps)
