import numpy as np
import healpy as hp
import matplotlib.pyplot as mp
import pymaster as nmt
from numpy import pi
import os
from qubic import (
    apodize_mask, equ2gal, plot_spectra, read_spectra, semilogy_spectra, Xpol)

# number of simulations
nsim = 10
# nside
nside = 128
# input spectra
spectra = read_spectra(0.01)


# function that computes the cls from the maps,
# f_a and f_b are the workspaces of the corresponding fiels (see below)
def compute_master(f_a, f_b, wsp):
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    return cl_decoupled


# compute apodization mask
lmin = 20
lmax = 3 * nside - 1
delta_ell = 20
racenter = 0.0
deccenter = -57.0
maxang = 20.

center = equ2gal(racenter, deccenter)
veccenter = hp.ang2vec(pi / 2 - np.radians(center[1]), np.radians(center[0]))
vecpix = hp.pix2vec(nside, np.arange(12 * nside ** 2))
cosang = np.dot(veccenter, vecpix)
maskok = np.degrees(np.arccos(cosang)) < maxang
msk_apo = nmt.mask_apodization(maskok, 1, apotype='C1')

# Select a binning scheme
b = nmt.NmtBin(nside, nlb=20, is_Dell=True)
leff = b.get_effective_ells()
# gaussian beam
beam = hp.gauss_beam(np.radians(0.39), lmax)

# initial maps and workspaces
mp_t, mp_q, mp_u = hp.synfast(spectra,
                              nside=nside,
                              fwhm=np.radians(0.39),
                              pixwin=True,
                              new=True,
                              verbose=False)

f0 = nmt.NmtField(msk_apo, [mp_t], beam=beam)
f2 = nmt.NmtField(msk_apo, [mp_q, mp_u], beam=beam)

# We initialize two workspaces for the fields:
w = nmt.NmtWorkspace()
w.compute_coupling_matrix(f0, f0, b)
w2 = nmt.NmtWorkspace()
w2.compute_coupling_matrix(f0, f2, b)

# We now iterate over several simulations,
# computing the power spectrum for each of them
data_00 = []
data_02 = []
for i in np.arange(nsim):
    print(i, nsim)
    mp_t, mp_q, mp_u = hp.synfast(spectra,
                                  nside=nside,
                                  fwhm=np.radians(0.39),
                                  pixwin=True,
                                  new=True,
                                  verbose=False)
    f0_sim = nmt.NmtField(msk_apo, [mp_t], beam=beam)
    f2_sim = nmt.NmtField(msk_apo, [mp_q, mp_u], beam=beam)
    data_00.append(compute_master(f0_sim, f0_sim, w))
    data_02.append(compute_master(f0_sim, f2_sim, w2))
data_00 = np.array(data_00)
data_02 = np.array(data_02)
cltt_mean = np.mean(data_00, axis=0)
cltt_std = np.std(data_00, axis=0)
clte_mean = np.mean(data_02, axis=0)
clte_std = np.std(data_02, axis=0)

hp.write_cl('cls_tt_512_beam_v2.fits', cltt_mean)
hp.write_cl('scls_tt_512_beam_v2.fits', cltt_std)
hp.write_cl('cls_te_512_beam_v2.fits', clte_mean)
hp.write_cl('scls_te_512_beam_v2.fits', clte_std)
