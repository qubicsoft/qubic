import glob
import numpy as np
import pickle
import os
import sys

import qubic
from qubicpack.utilities import Qubic_DataDir
from qubic import QubicSkySim as qss
from qubic import NamasterLib as nam


def get_maps_from_louise(directory, nfsub, nside):
    mappatchfile = glob.glob(directory + 'residualspatch_*_nfrecon{}.pkl'.format(nfsub))[0]
    covfile = glob.glob(directory + 'coverage_*_nfrecon{}.pkl'.format(nfsub))[0]
    seenfile = glob.glob(directory + 'seenmap_*_nfrecon{}.pkl'.format(nfsub))[0]

    residualspatch = pickle.load(open(mappatchfile, "rb"))
    seenpix = pickle.load(open(seenfile, "rb"))
    covpix = pickle.load(open(covfile, "rb"))

    residuals_map = np.zeros((nfsub, 12 * nside ** 2, 3))
    residuals_map[:, seenpix, :] = residualspatch

    return residuals_map, covpix, seenpix


# Repository for dictionary and input maps
if 'QUBIC_DATADIR' in os.environ:
    pass
else:
    raise NameError('You should define an environment variable QUBIC_DATADIR')

global_dir = Qubic_DataDir(datafile='instrument.py', datadir=os.environ['QUBIC_DATADIR'])
print('global directory:', global_dir)

# Repository where maps and spectra will be saved
rep_save = sys.argv[1]

# Repository with maps obtained with the full pipeline
dirmaps = global_dir + '/scripts/Spectroimagery_paper/maps/'

# Get a dictionary
dictionary = global_dir + '/dicts/spectroimaging_article.dict'
d = qubic.qubicdict.qubicDict()
d.read_from_file(dictionary)

# Create a sky
seed = None
sky_config = {'cmb': seed}
Qubic_sky = qss.Qubic_sky(sky_config, d)

# Number of subbands
all_nfrecon = [1, 2, 3, 4, 5, 8]
nfrecon = int(sys.argv[2])
if nfrecon not in all_nfrecon:
    raise ValueError('Wrong number of subbands !')
print('\n nfrecon:', nfrecon)

# Get maps
maps, coverage, seenmap = get_maps_from_louise(dirmaps, nfrecon, d['nside'])
# np.save(rep_save + '/seenmap_nfrecon{}'.format(nfrecon)+ '.npy', seenmap)
# np.save(rep_save + '/coverage_nfrecon{}'.format(nfrecon)+ '.npy', coverage)

# Compute covariance matrices
cI, cQ, cU, _, _ = qss.get_cov_nunu(maps, coverage, QUsep=True)

# ================== Make maps =============================
# Get fitcov
myfitcovs = []
for isub in range(nfrecon):
    xx, yyfs, fitcov = qss.get_noise_invcov_profile(maps[isub, :, :],
                                                    coverage,
                                                    QUsep=True,
                                                    label='Input Map {}'.format(nfrecon),
                                                    fit=True,
                                                    norm=False,
                                                    allstokes=True,
                                                    doplot=False)
    myfitcovs.append(fitcov)

# Make many maps realisations with clnoise=None
nreals = int(sys.argv[3])
signoise = 75
nside = d['nside']
print('nside:', nside)
npix = 12 * nside ** 2
maps_clnoiseNone = np.zeros((nreals, nfrecon, npix, 3))

rnd_name = qss.random_string(10)
for i in range(nreals):
    maps_clnoiseNone[i] = Qubic_sky.create_noise_maps(signoise,
                                                      coverage,
                                                      nsub=nfrecon,
                                                      effective_variance_invcov=myfitcovs,
                                                      clnoise=None,
                                                      sub_bands_cov=[cI, cQ, cU],
                                                      verbose=False)
np.save(rep_save + '/patch_clnoiseNone_nfrecon{}_nreals{}_'.format(nfrecon, nreals) + rnd_name + '.npy',
        maps_clnoiseNone[:, :, seenmap, :])
# with open(rep_save + '/maps_clnoiseNone_nfrecon{}_nreals{}_'.format(nfrecon, nreals) + rnd_name + '.pkl', 'wb') as f:
#     pickle.dump(maps_clnoiseNone, f)

# Make many maps realisations with clnoise=clth
# Spatial correlation for noise
clth = pickle.load(open(global_dir + '/doc/FastSimulator/Data/cl_corr_noise_nersc200k.pk', 'rb'))
alpha = 4.5
clth = (clth - 1) * alpha + 1

maps_clth = np.zeros((nreals, nfrecon, npix, 3))
for i in range(nreals):
    maps_clth[i] = Qubic_sky.create_noise_maps(signoise,
                                               coverage,
                                               nsub=nfrecon,
                                               effective_variance_invcov=myfitcovs,
                                               clnoise=clth,
                                               sub_bands_cov=[cI, cQ, cU],
                                               verbose=False)
np.save(rep_save + '/patch_clth_nfrecon{}_nreals{}_'.format(nfrecon, nreals) + rnd_name + '.npy',
        maps_clth[:, :, seenmap, :])
# with open(rep_save + '/maps_clth_nfrecon{}_nreals{}_'.format(nfrecon, nreals) + rnd_name + '.pkl', 'wb') as f:
#     pickle.dump(maps_clth, f)

# # ================== Power spectrum =============================
# print('\n =============== Starting power spectrum ================')
# # Create a Namaster object
# lmin = 40
# lmax = 2 * d['nside'] - 1
# delta_ell = 30
#
# mask = np.zeros(12 * d['nside'] ** 2)
# mask[seenmap] = 1
# Namaster = nam.Namaster(mask, lmin=lmin, lmax=lmax, delta_ell=delta_ell)
# mask_apo = Namaster.get_apodized_mask()
#
# ell_binned, b = Namaster.get_binning(d['nside'])
# nbins = len(ell_binned)
# print('lmin:', lmin)
# print('lmax:', lmax)
# print('delta_ell:', delta_ell)
# print('nbins:', nbins)
# print('ell binned:', ell_binned)
#
# # Auto spectrum
# print('\n =============== Auto spectrum starting================')
# w = None
# cells_auto = np.zeros((nreals, nfrecon, nbins, 4))
# for r in range(nreals):
#     for isub in range(nfrecon):
#         maps = maps_clth[r, isub, :, :]
#         leff, cells_auto[r, isub, :, :], w = Namaster.get_spectra(maps.T,
#                                                                   mask_apo,
#                                                                   w=w,
#                                                                   purify_e=True,
#                                                                   purify_b=False,
#                                                                   beam_correction=None,
#                                                                   pixwin_correction=True)
# with open(rep_save + '/auto_spectrum_nfrecon{}_nreals{}_'.format(nfrecon, nreals) + rnd_name + '.pkl', 'wb') as f:
#     pickle.dump(cells_auto, f)
#
# # Cross spectrum
# print('\n =============== Cross spectrum starting ================')
# ncross = nreals // 2
# print('ncross:', ncross)
# cells_cross = np.zeros((ncross, nfrecon, nbins, 4))
#
# # Get spectra
# w = None
# for isub in range(nfrecon):
#     print('isub:', isub)
#     cross = 0
#     for c1 in range(0, nreals - 1, 2):  # do not mix pairs to avoid correlation
#         c2 = c1 + 1
#         c = (c1, c2)
#         print(c)
#         map1 = maps_clth[c[0], isub, :, :]
#         map2 = maps_clth[c[1], isub, :, :]
#         leff, cells_cross[cross, isub, :, :], w = Namaster.get_spectra(map1.T,
#                                                                        mask_apo,
#                                                                        map2.T,
#                                                                        w=w,
#                                                                        purify_e=True,
#                                                                        purify_b=False,
#                                                                        beam_correction=None,
#                                                                        pixwin_correction=True)
#         cross += 1
# with open(rep_save + '/cross_spectrum_nfrecon{}_nreals{}_'.format(nfrecon, nreals) + rnd_name + '.pkl', 'wb') as f:
#     pickle.dump(cells_cross, f)
