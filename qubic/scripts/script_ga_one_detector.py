from __future__ import division
from pyoperators import DiagonalOperator, PackOperator, pcg
from pyoperators import rule_manager
from qubic import (
    QubicAcquisition, QubicInstrument, create_sweeping_pointings, equ2gal)
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
sampling.angle_hwp = np.random.random_integers(0, 7, len(sampling)) * 11.25

# get the instrument model with only one detector
idetector = 0
instrument = QubicInstrument()[idetector]

# get the acquisition model from the instrument and sampling models
acq = QubicAcquisition(instrument, sampling, nside=nside)

# get it as operators
convolution = acq.get_convolution_peak_operator()
projection = acq.get_projection_operator()
hwp = acq.get_hwp_operator()
polarizer = acq.get_polarizer_operator()

# restrict the projection to the observed sky pixels
coverage = projection.pT1()
mask = coverage > 0
projection.restrict(mask)
pack = PackOperator(mask, broadcast='rightward')

with rule_manager(inplace=True):
    H = polarizer * (hwp * projection)

# get noiseless timeline
x0 = qubic.io.read_map(qubic.data.PATH + 'syn256_pol.fits')
x0_convolved = convolution(x0)
y = H(pack(x0_convolved))

# inversion through Preconditioned Conjugate Gradient
preconditioner = DiagonalOperator(1/coverage[mask], broadcast='rightward')
solution = pcg(H.T * H, H.T(y), M=preconditioner, disp=True, tol=1e-3)
output_map = pack.T(solution['x'])


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
display(output_map, 'Reconstructed map')
mp.figure(3)
display(output_map-x0_convolved, 'Difference map')
mp.show()
