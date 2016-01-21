"""
Noiseless simulation and map-making using the gaussian approximation
for the synthetic beam.

"""
from __future__ import division
from qubic import (
    create_random_pointings, gal2equ, QubicAcquisition, QubicScene,
    tod2map_all)
import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
import qubic

# read the input map
x0 = qubic.io.read_map(qubic.data.PATH + 'syn256_pol.fits', field='I_STOKES')
nside = 256

# let's take the galactic north pole as the center of the observation field
center_gal = 0, 90
center = gal2equ(center_gal[0], center_gal[1])

# sampling model
np.random.seed(0)
sampling = create_random_pointings(center, 1000, 10)
scene = QubicScene(nside, kind='I')

# acquisition model
acq = QubicAcquisition(150, sampling, scene)
y, x0_convolved = acq.get_observation(x0, convolution=True, noiseless=True)

# map-making
x, coverage = tod2map_all(acq, y, disp=True, tol=1e-4, coverage_threshold=0)
mask = coverage > 0


# some display
def display(x, title):
    x = x.copy()
    x[~mask] = np.nan
    hp.gnomview(x, rot=center_gal, reso=5, xsize=600, min=-200, max=200,
                title=title)

display(x0, 'Original map')
display(x0_convolved, 'Convolved original map')
display(x, 'Reconstructed map (gaussian approximation)')
display(x - x0_convolved, 'Residual map (gaussian approximation)')
mp.show()
