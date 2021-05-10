import sys
import glob
import numpy as np
from itertools import combinations_with_replacement

import qubic
from qubic.polyacquisition import compute_freq
from qubic import QubicSkySim as qss
from qubic import NamasterLib as nam
from qubic import ReadMC as rmc
from qubic import AnalysisMC as amc

###### To run the script: $ python Fastsim_spectroimMC.py rep_save nbands config

# Repository where maps and spectra will be saved
rep_save = sys.argv[1]

# Get a dictionary
d = qubic.qubicdict.qubicDict()
d.read_from_file('spectroimaging_article.dict')

# Number of bands
nbands = int(sys.argv[2])
d['nf_recon'] = nbands
d['nf_sub'] = nbands
print('\n nbands:', nbands)

# Config and frequency
config = sys.argv[3]
if config not in ['FI150', 'FI220']:
    raise ValueError('The config should be FI150 or FI220.')

d['filter_nu'] = int(config[-3:]) * 1e9

# Central frequencies and FWHM of each band
_, _, nus, _, _, _ = compute_freq(int(config[-3:]), nbands)
print('nus:', nus)
fwhms = [d['synthbeam_peak150_fwhm'] * 150 / nu for nu in nus]
print('fwhms', fwhms)

# Input sky
seed = 42
# sky_config = {'dust': 'd1', 'cmb':seed, 'synchrotron':'s1'}
sky_config = {'dust': 'd1'}
print('Sky config:', sky_config)
Qubic_sky = qss.Qubic_sky(sky_config, d)
inputmaps = Qubic_sky.get_fullsky_convolved_maps(FWHMdeg=None, verbose=True)

rnd_name = qss.random_string(10)

# ================== Make maps =============================
# Getting noise realisations with FastSimulator
nreals = 4
npix = 12 * d['nside'] ** 2
noisemaps = np.zeros((nreals, nbands, npix, 3))
Nyears = 3.
print('Nyears:', Nyears)

# qubic_coverage = np.load('/pbs/home/l/lmousset/libs/qubic/qubic/scripts/Spectroimagery_paper/maps/'
#                          'coverage_nfsub15_nptgs10000_qubicpatch.npy')

for r in range(nreals):
    noisemaps[r, ...], coverage = Qubic_sky.get_partial_sky_maps_withnoise(coverage=None,
                                                                           noise_only=True,
                                                                           spatial_noise=True,
                                                                           nunu_correlation=True,
                                                                           Nyears=Nyears)

# Make maps QUBIC = noise + signal
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
np.save(rep_save + f'/noisepatch_nbands{nbands}_' + config + '_v4_galaxycenter_' + rnd_name + '.npy',
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

# ------------ Maps from NERSC --------------------
# rep_mapNERSC = f'/sps/hep/qubic/Users/lmousset/SpectroImaging/mapsfromNERSC/nfrecon{nbands}/'
# fits_noise = np.sort(glob.glob(rep_mapNERSC + f'*nfrecon{nbands}_noiselessFalse*.fits',
#                               recursive=True))
# fits_noiseless = np.sort(glob.glob(rep_mapNERSC + f'*nfrecon{nbands}_noiselessTrue*.fits',
#                               recursive=True))
# nreals = len(fits_noise)
# print('nreals = ', nreals)
#
# # Get seen map (observed pixels)
# seenmap = rmc.get_seenmap(fits_noiseless[0])
# print('seenmap shape:', seenmap.shape)
#
# # Number of pixels and nside
# npix = len(seenmap)
#
# # Get reconstructed maps
# qubicmaps = np.zeros((nreals, nbands, npix, 3))
# for i, real in enumerate(fits_noise):
#     qubicmaps[i], _, _ = rmc.get_maps(real)
#
# qubicmaps[qubicmaps == hp.UNSEEN] = 0.
#
# # Compute residuals in a given way
# residuals = amc.get_residuals(fits_noise, fits_noiseless[0], 'noiseless')
# print(residuals.shape)
#
# # There is only the patch so you need to put them in a full map to plot with healpy
# noisemaps = np.zeros_like(qubicmaps)
# noisemaps[:, :, seenmap, :] = residuals


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

for real in range(nreals):
    print(f'\n Real {real}')
    for i, (band1, band2) in enumerate(combi):
        print(f'Bands {band1} {band2}')
        beam_corr = np.sqrt(fwhms[band1] * fwhms[band2])
        print('Beam correction:', beam_corr)

        map1 = qubicmaps[real, band1, :, :]
        map2 = qubicmaps[real, band2, :, :]
        leff, cross_samereal_qubicmaps[real, i, :, :], w = Namaster.get_spectra(map1.T,
                                                                                mask_apo,
                                                                                map2.T,
                                                                                w=None,
                                                                                purify_e=True,
                                                                                purify_b=False,
                                                                                beam_correction=beam_corr,
                                                                                pixwin_correction=True)

        map1noise = noisemaps[real, band1, :, :]
        map2noise = noisemaps[real, band2, :, :]
        leff, cross_samereal_noisemaps[real, i, :, :], w = Namaster.get_spectra(map1noise.T,
                                                                                mask_apo,
                                                                                map2noise.T,
                                                                                w=None,
                                                                                purify_e=True,
                                                                                purify_b=False,
                                                                                beam_correction=beam_corr,
                                                                                pixwin_correction=True)
np.save(
    rep_save + f'/IBCSsame_nfrecon{nbands}_qubicmaps_' + config + '_v4_galaxycenter_' + rnd_name + '.npy',
    cross_samereal_qubicmaps)
np.save(
    rep_save + f'/IBCSsame_nfrecon{nbands}_noisemaps_' + config + '_v4_galaxycenter_' + rnd_name + '.npy',
    cross_samereal_noisemaps)

# np.save(
#     rep_save + f'/IBCSsame_recon_{nbands}bands_150fullpipeline.npy',
#     cross_samereal_qubicmaps)
# np.save(rep_save + f'/IBCSsame_residu_{nbands}bands_150fullpipeline.npy',
#         cross_samereal_noisemaps)

# Cross spectrum between bands with different real
print('\n =============== Cross spectrum mixing reals starting ================')

ncross = nreals // 2
print('ncross:', ncross)
cross_mixreals_qubicmaps = np.zeros((ncross, ncombi, nbins, 4))
cross_mixreals_noisemaps = np.zeros((ncross, ncombi, nbins, 4))

cross = 0
for c1 in range(0, nreals - 1, 2):  # do not mix pairs to avoid correlation
    c2 = c1 + 1
    print(f'\n Reals {c1} {c2}')
    for i, (band1, band2) in enumerate(combi):
        print(f'Bands {band1} {band2}')
        beam_corr = np.sqrt(fwhms[band1] * fwhms[band2])
        print('Beam correction:', beam_corr)

        map1 = qubicmaps[c1, band1, :, :]
        map2 = qubicmaps[c2, band2, :, :]
        leff, cross_mixreals_qubicmaps[cross, i, :, :], w = Namaster.get_spectra(map1.T,
                                                                                 mask_apo,
                                                                                 map2.T,
                                                                                 w=None,
                                                                                 purify_e=True,
                                                                                 purify_b=False,
                                                                                 beam_correction=beam_corr,
                                                                                 pixwin_correction=True)

        map1noise = noisemaps[c1, band1, :, :]
        map2noise = noisemaps[c2, band2, :, :]
        leff, cross_mixreals_noisemaps[cross, i, :, :], w = Namaster.get_spectra(map1noise.T,
                                                                                 mask_apo,
                                                                                 map2noise.T,
                                                                                 w=None,
                                                                                 purify_e=True,
                                                                                 purify_b=False,
                                                                                 beam_correction=beam_corr,
                                                                                 pixwin_correction=True)
    cross += 1

np.save(
    rep_save + f'/IBCSmix_nfrecon{nbands}_qubicmaps_' + config + '_v4_galaxycenter_' + rnd_name + '.npy',
    cross_mixreals_qubicmaps)

np.save(
    rep_save + f'/IBCSmix_nfrecon{nbands}_noisemaps_' + config + '_v4_galaxycenter_' + rnd_name + '.npy',
    cross_mixreals_noisemaps)

# np.save(rep_save + f'/IBCSmix_recon_{nbands}bands_150fullpipeline.npy',
#         cross_mixreals_qubicmaps)
# np.save(rep_save + f'/IBCSmix_residu_{nbands}bands_150fullpipeline.npy',
#         cross_mixreals_noisemaps)
