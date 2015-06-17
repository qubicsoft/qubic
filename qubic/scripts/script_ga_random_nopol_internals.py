"""
Noiseless simulation and map-making using the gaussian approximation
for the synthetic beam.

"""
from __future__ import division
from pyoperators import DiagonalOperator, pcg
from qubic import (
    create_random_pointings, gal2equ, QubicAcquisition, QubicInstrument, QubicScene)
import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
import qubic

# read the input map
x0 = qubic.io.read_map(qubic.data.PATH + 'syn256_pol.fits',
                       field='I_STOKES')

# let's take the galactic north pole as the center of the observation field
center_gal = 0, 90
center = gal2equ(center_gal[0], center_gal[1])

# sampling model
np.random.seed(0)
sampling = create_random_pointings(center, 1000, 10)

# scene model
scene = QubicScene(hp.npix2nside(x0.size), kind='I')

# instrument model
instrument = QubicInstrument(filter_nu=150e9)

# acquisition model
acq = QubicAcquisition(instrument, sampling, scene)
x0_convolved = acq.get_convolution_peak_operator()(x0)
H = acq.get_operator()
coverage = H.T(np.ones(H.shapeout))
mask = coverage > 0

# restrict the scene to the observed pixels
acq_restricted = acq[..., mask]
H_restricted = acq_restricted.get_operator()
x0_restricted = x0[mask]
y = H_restricted(x0_restricted)
invntt = acq_restricted.get_invntt_operator()

# solve for x
A = H_restricted.T * invntt * H_restricted
b = H_restricted.T(invntt(y))
solution = pcg(
    A, b, M=DiagonalOperator(1/coverage[mask]), disp=True, tol=1e-4)
x = np.zeros_like(x0)
x[mask] = solution['x']


# some display
def display(x, title, lim=200):
    x = x.copy()
    x[~mask] = np.nan
    hp.gnomview(x, rot=center_gal, reso=5, xsize=600, min=-lim, max=lim,
                title=title)

display(x0, 'Original map')
display(x0_convolved, 'Convolved original map')
display(x, 'Reconstructed map (gaussian approximation)')
display(x - x0_convolved, 'Residual map (gaussian approximation)')
mp.show()
