import glob
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

import ReadMC as rmc
import AnalysisMC as amc

import qubic
from qubic import equ2gal

stokes = ['I', 'Q', 'U']

# Coordinates of the zone observed in the sky
center = equ2gal(0., -57.)

# ================= Get the simulation files ================
# repository where the .fits was saved
date = '20190619'
rep_simu = './TEST/{}/'.format(date)

# Simulation name
name = 'firstdemo'

# Dictionary saved during the simulation
d = qubic.qubicdict.qubicDict()
d.read_from_file(rep_simu + date + '_' + name + '.dict')

# Get fits files names in a list
fits_files = []
for fits in glob.glob(rep_simu + date + '_' + name + '*.fits'):
    fits_files.append(fits)
    print(fits)

# Number of subbands used during the simulation
nf_recon = d['nf_recon']

# ================= Get maps ================
# Get seen map (observed pixels)
seen_map = rmc.get_seenmap(fits_files[0])

# Number of pixels and nside
npix = len(seen_map)
ns = d['nside']


# Get one full maps
maps_recon, maps_convo, maps_diff = rmc.get_maps(fits_files[1])
print('Getting maps with shape : {}'.format(maps_recon.shape))

# Look at the maps
isub = 0
plt.figure('Maps in subband {}'.format(isub))
for i in xrange(3):
    hp.gnomview(maps_convo[isub, :, i], rot=center, reso=9, sub=(3, 3, i + 1),
                title='conv ' + stokes[i])
    hp.gnomview(maps_recon[isub, :, i], rot=center, reso=9, sub=(3, 3, 3 + i + 1),
                title='recon ' + stokes[i])
    hp.gnomview(maps_diff[isub, :, i], rot=center, reso=9, sub=(3, 3, 6 + i + 1),
                title='residus ' + stokes[i])

# Get one patch
maps_recon_cut, maps_convo_cut, maps_diff_cut = rmc.get_patch(fits_files[1], seen_map)
print('Getting patches with shape : {}'.format(maps_recon_cut.shape))

# Get all patches (all noise realisations)
all_fits, all_patch_recon, all_patch_conv, all_patch_diff = rmc.get_patch_many_files(
    rep_simu, date + '_' + name + '*nfrecon2_noiselessFalse*.fits')
print('Getting all patch realizations with shape : {}'.format(all_patch_recon.shape))

# ================== Look at residuals ===============
residuals = all_patch_recon - np.mean(all_patch_recon, axis=0)

# Histogram of the residuals (first real, first subband)
plt.clf()
for i in xrange(3):
    plt.subplot(1, 3, i + 1)
    plt.hist(np.ravel(residuals[0, 0, :, i]), range=[-5, 5], bins=100)
    plt.title(stokes[i])

# ================= Make zones ============

# ================= Correlations matrices=======================
# Correlation between pixels
cov_pix, corr_pix = amc.get_covcorr_between_pix(residuals, verbose=True)

# Correlations between subbands and I, Q, U

# ================= Noise Evolution as a function of the subband number=======================
# This part should be rewritten (old)
# To do that, you need many realisations

allmeanmat = amc.get_rms_covar(nsubvals, seenmap_recon, allmaps_recon)[1]
rmsmap_cov = amc.get_rms_covarmean(nsubvals, seenmap_recon, allmaps_recon, allmeanmat)[1]
mean_rms_cov = np.sqrt(np.mean(rmsmap_cov ** 2, axis=2))

plt.plot(nsubvals, np.sqrt(nsubvals), 'k', label='Optimal $\sqrt{N}$', lw=2)
for i in xrange(3):
    plt.plot(nsubvals, mean_rms_cov[:, i] / mean_rms_cov[0, i] * np.sqrt(nsubvals), label=stokes[i], lw=2, ls='--')
plt.xlabel('Number of sub-frequencies')
plt.ylabel('Relative maps RMS')
plt.legend()


