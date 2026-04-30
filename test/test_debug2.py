import numpy as np
import healpy as hp
from qubic.lib.Fitting.Qnamaster import Namaster

nside = 64
npix = hp.nside2npix(nside)

# Masque simple (tout le ciel)
mask = np.ones(npix)

# Map IQU bidon (petit signal aléatoire)
rng = np.random.default_rng(0)
I = rng.normal(scale=1e-6, size=npix)
Q = rng.normal(scale=1e-6, size=npix)
U = rng.normal(scale=1e-6, size=npix)
m = np.array([I, Q, U])

nm = Namaster(mask, lmin=10, lmax=3*nside-1, delta_ell=20, aposize=10.0, apotype="C1")

ell_b, spectra, w = nm.get_spectra(m, verbose=False)

print("ell_b shape:", ell_b.shape)
print("spectra shape:", spectra.shape, "(columns TT, EE, BB, TE)")
print("first row:", spectra[0])
print("OK spectra computed")
