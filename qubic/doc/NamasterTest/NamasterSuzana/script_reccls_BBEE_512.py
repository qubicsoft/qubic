import numpy as np
import healpy as hp
import matplotlib.pyplot as mp
import pymaster as nmt
import os
from qubic import (
    apodize_mask, equ2gal, plot_spectra, read_spectra, semilogy_spectra, Xpol)

# number of simulations
nsim = 100
# nside
nside = 512
# input spectra
spectra = read_spectra(0.01)
# Now we apodize the mask. The pure-B formalism requires the mask to be differentiable
# along the edges. The 'C1' and 'C2' apodization types supported by mask_apodization
# achieve this.
mask = hp.read_map("mask_qubic_20_512.fits")
msk_apo = nmt.mask_apodization(mask, 10.0, apotype='C1')
# Select a binning scheme
b = nmt.NmtBin(nside, nlb=20, is_Dell=True)
leff = b.get_effective_ells()
# gaussian beam
beam = hp.gauss_beam(np.radians(0.39), 3 * nside - 1)


def get_fields():
    mp_t, mp_q, mp_u = hp.synfast(spectra, nside=nside, fwhm=np.radians(0.39), pixwin=True, new=True, verbose=False)
    # This creates a spin-2 field with both pure E and B.
    f2 = nmt.NmtField(msk_apo, [mp_q, mp_u], beam=beam, purify_e=False, purify_b=True)
    # Note that generally it's not a good idea to purify both, since you'll lose sensitivity on E
    return f2


# We initialize two workspaces for the non-pure and pure fields:
f20 = get_fields()
w = nmt.NmtWorkspace();
w.compute_coupling_matrix(f20, f20, b)


# This wraps up the two steps needed to compute the power spectrum
# once the workspace has been initialized
def compute_master(f_a, f_b, wsp):
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    return cl_decoupled


# We now iterate over several simulations, computing the power spectrum for each of them
data = []
for i in np.arange(nsim):
    print(i, nsim)
    f2 = get_fields()
    data.append(compute_master(f2, f2, w))
data = np.array(data)
clnp_mean = np.mean(data, axis=0)
clnp_std = np.std(data, axis=0)

hp.write_cl('cls_BBEE_512.fits', clnp_mean)
hp.write_cl('scls_BBEE_512.fits', clnp_std)
