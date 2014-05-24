from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
import os
import qubic
from pyoperators import pcg, DiagonalOperator, UnpackOperator
from pysimulators import SphericalEquatorial2GalacticOperator
from pysimulators.noises import _gaussian_psd_1f
from qubic import (
    QubicAcquisition, QubicInstrument, create_sweeping_pointings, equ2gal)

# read the input map
DATAPATH = os.path.join(os.path.dirname(qubic.__file__), 'data',
                        'syn256_pol.fits')
input_map = qubic.io.read_map(DATAPATH, field='I_STOKES')


# instrument model
sigma = 10
fknee = 1  # Hz
fslope = 1
ncorr = 10
instrument = QubicInstrument('monochromatic,nopol', synthbeam_fraction=0.99,
                             detector_sigma=sigma, detector_fknee=fknee,
                             detector_fslope=fslope, detector_ncorr=ncorr)

# observation model
np.random.seed(0)
racenter = 0.0
deccenter = -57.0
angspeed = 1  # deg/sec
delta_az = 15.
angspeed_psi = 0.1
maxpsi = 45.
nsweeps_el = 300
duration = 24   # hours
ts = 20         # seconds
sampling = create_sweeping_pointings(
    [racenter, deccenter], duration, ts, angspeed, delta_az, nsweeps_el,
    angspeed_psi, maxpsi)

# acquisition model
acq = QubicAcquisition(instrument, sampling)
C = acq.get_convolution_peak_operator()
P = acq.get_projection_peak_operator()
H = P * C

# produce the Time-Ordered data
tod = H(input_map)

# noise
psd = _gaussian_psd_1f(len(acq.sampling), sigma=sigma, fknee=fknee,
                       fslope=fslope, sampling_frequency=1/ts)
invntt = acq.get_invntt_operator()
noise = acq.get_noise()

# map-making
coverage = P.pT1()
mask = coverage > 10
P.restrict(mask)
unpack = UnpackOperator(mask)

# map without covariance matrix
solution1 = pcg(P.T * P, P.T(tod + noise),
                M=DiagonalOperator(1/coverage[mask]), disp=True)
output_map1 = unpack(solution1['x'])

# map with covariance matrix
solution2 = pcg(P.T * invntt * P, (P.T * invntt)(tod + noise),
                M=DiagonalOperator(1/coverage[mask]), disp=True)
output_map2 = unpack(solution2['x'])

center = equ2gal(racenter, deccenter)


def display(x, title):
    x = x.copy()
    x[~mask] = np.nan
    hp.gnomview(x, rot=center, reso=5, xsize=600, min=-200, max=200,
                title=title)

display(C(input_map), 'Original convolved map')
#hp.projplot(np.radians(pointings[..., 0]), np.radians(pointings[..., 1]))
display(output_map1, 'Reconstructed map no invntt')
display(output_map2, 'Reconstructed map with invntt')
display(coverage, 'Coverage')

mp.show()
