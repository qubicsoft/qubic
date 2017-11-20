from __future__ import division, print_function

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
from numpy import pi
from qubic import (
    apodize_mask, equ2gal, plot_spectra, read_spectra, semilogy_spectra, Xpol)
from qubic.utils import progress_bar


# get input power spectra
## parameters from Planck 2013 results XV. CMB Power spectra ...,
## table 8, Planck+WP. Archiv 1303.5075
spectra = read_spectra(0)
mp.figure()
semilogy_spectra(spectra, lmax=3*512)

# compute apodization mask
nside = 256
lmin = 20
lmax = 2*nside-1
delta_ell = 20
racenter = 0.0
deccenter = -57.0
maxang = 20.

center = equ2gal(racenter, deccenter)
veccenter = hp.ang2vec(pi/2-np.radians(center[1]), np.radians(center[0]))
vecpix = hp.pix2vec(nside, np.arange(12*nside**2))
cosang = np.dot(veccenter, vecpix)
maskok = np.degrees(np.arccos(cosang)) < maxang
maskmap = apodize_mask(maskok, 5)
hp.gnomview(maskmap, rot=[racenter, deccenter], coord=['G', 'C'], reso=15)


# Xpol estimation through MC
np.random.seed(0)
xpol = Xpol(maskmap, lmin, lmax, delta_ell)
ell_binned = xpol.ell_binned
spectra_binned = xpol.bin_spectra(spectra)
nbins = len(ell_binned)

nbmc = 100
allclsout = np.zeros((nbmc, 6, nbins))
allcls = np.zeros((nbmc, 6, lmax+1))
bar = progress_bar(nbmc)
for i in np.arange(nbmc):
    maps = hp.synfast(spectra, nside, fwhm=0, pixwin=True, new=True,
                      verbose=False)
    allcls[i], allclsout[i] = xpol.get_spectra(maps)
    bar.update()


# get MC results
mclsout = np.mean(allclsout, axis=0)
sclsout = np.std(allclsout, axis=0) / np.sqrt(nbmc)
mcls = np.mean(allcls, axis=0)
scls = np.std(allcls, axis=0) / np.sqrt(nbmc)

ell = np.arange(lmax+1)
pw = hp.pixwin(nside, pol=True)
pw = [pw[0][:lmax+1], pw[1][:lmax+1]]
pwb = xpol.bin_spectra(pw)


# plot input, anafast and xpol spectra
mp.figure()
fact = len(maskmap) / maskok.sum()
kw_xpol = {'fmt': 'bo', 'markersize': 3}

mp.subplot(3, 2, 1)
mp.title('TT')
plot_spectra(ell_binned, mclsout[0, :] / pwb[0]**2, yerr=sclsout[0, :],
             label='Xpol', lmax=lmax, **kw_xpol)
plot_spectra(fact * mcls[0, :] / pw[0]**2, color='g', label='Anafast rescaled')
plot_spectra(spectra[0], color='r', label='Input')
mp.xlabel('')
mp.ylim(0, 6000)
mp.legend(loc='upper right', frameon=False, fontsize=10)

mp.subplot(3, 2, 2)
mp.title('TE')
plot_spectra(ell_binned, mclsout[3, :] / pwb[0] / pwb[1], yerr=sclsout[3, :],
             lmax=lmax, **kw_xpol)
plot_spectra(fact * mcls[3, :] / pw[0] / pw[1], color='g')
plot_spectra(spectra[3], color='r')
mp.xlabel('')
mp.ylim(-150, 150)

mp.subplot(3, 2, 3)
mp.title('EE')
plot_spectra(ell_binned, mclsout[1, :] / pwb[1]**2, yerr=sclsout[1, :],
             lmax=lmax, **kw_xpol)
plot_spectra(fact * mcls[1, :] / pw[1]**2, color='g')
plot_spectra(spectra[1], color='r')
mp.xlabel('')
mp.ylim(0, 25)

mp.subplot(3, 2, 4)
mp.title('BB')
plot_spectra(ell_binned, mclsout[2, :] / pwb[1]**2, yerr=sclsout[2, :],
             lmax=lmax, **kw_xpol)
plot_spectra(fact * mcls[2, :] / pw[1]**2, color='g')
plot_spectra(spectra[2], color='r')
mp.xlabel('')
mp.ylim(-0.005, 0.01)

mp.subplot(3, 2, 5)
mp.title('TB')
plot_spectra(ell_binned, mclsout[4, :] / pwb[1]**2, yerr=sclsout[4, :],
             lmax=lmax, **kw_xpol)
plot_spectra(fact * mcls[4, :] / pw[1]**2, color='g')
mp.axhline(0, color='r')
mp.ylim(-0.005, 0.005)

mp.subplot(3, 2, 6)
mp.title('EB')
plot_spectra(ell_binned, mclsout[5, :] / pwb[0] / pwb[1], yerr=sclsout[5, :],
             lmax=lmax, **kw_xpol)
plot_spectra(fact * mcls[5, :] / pw[0] / pw[1], color='g')
mp.axhline(0, color='r')
mp.ylim(-0.1, 0.1)


# residuals
mp.figure()
mp.subplot(3, 2, 1)
mp.title('TT')
plot_spectra(ell_binned, mclsout[0, :] / pwb[0]**2 - spectra_binned[0],
             yerr=sclsout[0, :], label='Xpol', **kw_xpol)
mp.axhline(0, color='r')

mp.subplot(3, 2, 2)
mp.title('TE')
plot_spectra(ell_binned, mclsout[3, :] / pwb[0] / pwb[1] - spectra_binned[3],
             yerr=sclsout[3, :], **kw_xpol)
mp.axhline(0, color='r')
mp.xlabel('')

mp.subplot(3, 2, 3)
mp.title('EE')
plot_spectra(ell_binned, mclsout[1, :] / pwb[1]**2 - spectra_binned[1],
             yerr=sclsout[1, :], **kw_xpol)
mp.axhline(0, color='r')
mp.xlabel('')

mp.subplot(3, 2, 4)
mp.title('BB')
plot_spectra(ell_binned, mclsout[2, :] / pwb[1]**2 - spectra_binned[2],
             yerr=sclsout[2, :], **kw_xpol)
mp.axhline(0, color='r')
mp.xlabel('')


mp.subplot(3, 2, 5)
mp.title('TB')
plot_spectra(ell_binned, mclsout[4, :] / pwb[1]**2, yerr=sclsout[4, :],
             **kw_xpol)
mp.axhline(0, color='r')

mp.subplot(3, 2, 6)
mp.title('EB')
plot_spectra(ell_binned, mclsout[5, :] / pwb[0] / pwb[1], yerr=sclsout[5, :],
             **kw_xpol)
mp.axhline(0, color='r')
