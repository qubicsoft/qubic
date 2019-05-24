import glob
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

import Tools as tl
import ReadMC as rmc

import qubic
from qubic import equ2gal

thespec = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
stokes = ['I', 'Q', 'U']

# Coordinates of the zone observed in the sky
center = equ2gal(0., -57.)

# ================= Get the simulation files ================
# repository where the .fits was saved
rep_simu = '/home/louisemousset/Desktop/'

# Simulation name
name = '20190517_truc'

# Dictionary saved during the simulation
d = qubic.qubicdict.qubicDict()
d.read_from_file(rep_simu + name + '.dict')

# Get fits files names in a list
fits_files = []
for fits in glob.glob(rep_simu + name + '*.fits'):
    fits_files.append(fits)
    print fits

# Number of subbands used during the simulation
nf_recon = d['nf_recon']

# ================= Get the maps ================
# Get seen map (observed pixels)
seen_map = rmc.get_seenmap_new(fits_files[0])

# Number of pixels and nside
npix = len(seen_map)
ns = d['nside']

# Get full maps
maps_recon, maps_convo, residuals = rmc.get_maps(fits_files[1])
print('Getting maps with shape : {}'.format(maps_recon.shape))

# Look at the maps
isub = 0
plt.figure('Maps in subband {}'.format(isub))
for i in xrange(3):
    hp.gnomview(maps_convo[isub, :, i], rot=center, reso=9, sub=(3, 3, i + 1),
                title='conv ' + stokes[i])
    hp.gnomview(maps_recon[isub, :, i], rot=center, reso=9, sub=(3, 3, 3 + i + 1),
                title='recon ' + stokes[i])
    hp.gnomview(residuals[isub, :, i], rot=center, reso=9, sub=(3, 3, 6 + i + 1),
                title='residus ' + stokes[i])

# Get only patches to save memory
maps_recon_cut, maps_convo_cut, residuals_cut = rmc.get_patch(fits_files[1], seen_map)
print('Getting patches with shape : {}'.format(maps_recon_cut.shape))


# ================== Look at residuals ===============

# Histogram of the residuals
plt.clf()
for i in xrange(3):
    plt.subplot(1, 3, i + 1)
    plt.hist(np.ravel(residuals_cut[0, :, i]), range=[-20, 20], bins=100)
    plt.title(stokes[i])

# ================= Correlations matrices=======================
# For each Stoke parameter separately, between subbands

residuals_meanpix = np.mean(residuals_cut, axis=1)
cov = np.cov(residuals_meanpix, rowvar=False)
plt.imshow(cov, rowvar=False)

# Between subbands and between Stokes parameters

# ================= Noise Evolution as a function of the subband number=======================
# To do that, you need many realisations

allmeanmat = rmc.get_rms_covar(nsubvals, seenmap_recon, allmaps_recon)[1]
rmsmap_cov = rmc.get_rms_covarmean(nsubvals, seenmap_recon, allmaps_recon, allmeanmat)[1]
mean_rms_cov = np.sqrt(np.mean(rmsmap_cov ** 2, axis=2))

plt.plot(nsubvals, np.sqrt(nsubvals), 'k', label='Optimal $\sqrt{N}$', lw=2)
for i in xrange(3):
    plt.plot(nsubvals, mean_rms_cov[:, i] / mean_rms_cov[0, i] * np.sqrt(nsubvals), label=stokes[i], lw=2, ls='--')
plt.xlabel('Number of sub-frequencies')
plt.ylabel('Relative maps RMS')
plt.legend()

# ======================= Apply Xpoll to get spectra ============================

xpol, ell_binned, pwb = rmc.get_xpol(seenmap_conv, ns)

nbins = len(ell_binned)
print('nbins = {}'.format(nbins))

mcls, mcls_in = [], []
scls, scls_in = [], []

# Input : what we should find
mapsconv = np.zeros((12 * ns ** 2, 3))

# Output, what we find
maps_recon = np.zeros((12 * ns ** 2, 3))

for isub in xrange(len(nsubvals)):
    sh = allmaps_conv[isub].shape
    nreals = sh[0]
    nsub = sh[1]
    cells = np.zeros((6, nbins, nsub, nreals))
    cells_in = np.zeros((6, nbins, nsub, nreals))
    print(cells.shape)
    for real in xrange(nreals):
        for n in xrange(nsub):
            mapsconv[seenmap_conv, :] = allmaps_conv[isub][real, n, :, :]
            maps_recon[seenmap_conv, :] = allmaps_recon[isub][real, n, :, :]
            cells_in[:, :, n, real] = xpol.get_spectra(mapsconv)[1]
            cells[:, :, n, real] = xpol.get_spectra(maps_recon)[1]

    mcls.append(np.mean(cells, axis=3))
    mcls_in.append(np.mean(cells_in, axis=3))
    scls.append(np.std(cells, axis=3))
    scls_in.append(np.std(cells_in, axis=3))

# Plot the spectra
plt.figure('TT_EE_BB spectra')
for isub in xrange(len(nsubvals)):
    for s in [1, 2]:
        plt.subplot(4, 2, isub * 2 + s)
        plt.ylabel(thespec[s] + '_ptg=' + str((isub + 1) * 1000))
        plt.xlabel('l')
        sh = mcls[isub].shape
        nsub = sh[2]
        for k in xrange(nsub):
            p = plt.plot(ell_binned, ell_binned * (ell_binned + 1) * mcls_in[isub][s, :, k], '--')
            plt.errorbar(ell_binned, ell_binned * (ell_binned + 1) * mcls[isub][s, :, k],
                         yerr=ell_binned * (ell_binned + 1) * scls[isub][s, :, k],
                         fmt='o', color=p[0].get_color(),
                         label='subband' + str(k + 1))

        if isub == 0 and s == 1:
            plt.legend(numpoints=1, prop={'size': 7})
plt.show()
