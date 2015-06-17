from __future__ import division
from qubic import (
    create_sweeping_pointings, equ2gal, QubicAcquisition, QubicInstrument,
    tod2map_all)
import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
import qubic

# acquisition parameters
nside = 256
racenter = 0.0      # deg
deccenter = -57.0   # deg
angspeed = 1        # deg/sec
delta_az = 15.      # deg
angspeed_psi = 0.1  # deg/sec
maxpsi = 45.        # deg
nsweeps_el = 300
duration = 24       # hours
ts = 0.1            # seconds

# get the sampling model
np.random.seed(0)
sampling = create_sweeping_pointings(
    [racenter, deccenter], duration, ts, angspeed, delta_az, nsweeps_el,
    angspeed_psi, maxpsi)

# get the instrument model with only one detector
idetector = 0
instrument = QubicInstrument()[idetector]

# get the acquisition model from the instrument and sampling models
acq = QubicAcquisition(instrument, sampling, nside=nside)

# get noiseless timeline
x0 = qubic.io.read_map(qubic.data.PATH + 'syn256_pol.fits')
y, x0_convolved = acq.get_observation(x0, convolution=True, noiseless=True)

# inversion through Preconditioned Conjugate Gradient
x, coverage = tod2map_all(acq, y, disp=True, tol=1e-3,
                          coverage_threshold=0)
mask = coverage > 0


# some display
def display(input, msg):
    for i, (kind, lim) in enumerate(zip('IQU', [200, 10, 10])):
        map = input[..., i].copy()
        map[~mask] = np.nan
        hp.gnomview(map, rot=center, reso=5, xsize=400, min=-lim, max=lim,
                    title=msg + ' ' + kind, sub=(1, 3, i+1))

center = equ2gal(racenter, deccenter)

mp.figure(1)
display(x0_convolved, 'Original map')
mp.figure(2)
display(x, 'Reconstructed map')
mp.figure(3)
display(x-x0_convolved, 'Difference map')
mp.show()
