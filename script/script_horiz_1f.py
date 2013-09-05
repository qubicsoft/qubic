from __future__ import division

import healpy as hp
import numpy as np
import os

from pyoperators import pcg, DiagonalOperator, UnpackOperator
from pysimulators import ProjectionInMemoryOperator
from pysimulators.noises import _gaussian_psd_1f
from qubic import QubicAcquisition, QubicInstrument, create_sweeping_pointings

kmax = 2
qubic = QubicInstrument('monochromatic,nopol')
input_map = hp.read_map('test/data/syn256.fits')

# pointing
racenter = 0.0
deccenter = -57.0
angspeed = 1  # deg/sec
delta_az = 15.
angspeed_psi = 0.1
maxpsi = 45.
nsweeps_el = 300
duration = 24   # hours
ts = 5         # seconds
pointings = create_sweeping_pointings(
    [racenter, deccenter], duration, ts, angspeed, delta_az, nsweeps_el,
    angspeed_psi, maxpsi)

# configure observation
obs = QubicAcquisition(qubic, pointings)
C = obs.get_convolution_peak_operator()
P = obs.get_projection_peak_operator(kmax=kmax)
H = P * C

# produce the Time-Ordered data
tod = H(input_map)

# noise
white = 10
alpha = 1
fknee = 0.1
psd = _gaussian_psd_1f(len(obs.pointing), sigma=white, fknee=fknee,
                       fslope=alpha, sampling_frequency=1/ts)
invntt = obs.get_invntt_operator(sigma=white, fknee=fknee, fslope=alpha,
                                 sampling_frequency=1/ts, ncorr=10)
noise = obs.get_noise(sigma=white, fknee=fknee, fslope=alpha,
                      sampling_frequency=1/ts)

# map-making
coverage = P.T(np.ones_like(tod))
mask = coverage < 10
P.matrix.pack(mask)
P_packed = ProjectionInMemoryOperator(P.matrix)
unpack = UnpackOperator(mask)

# map without covariance matrix
solution1 = pcg(P_packed.T * P_packed,
                P_packed.T(tod + noise),
                M=DiagonalOperator(1/coverage[~mask]), disp=True)
output_map1 = unpack(solution1['x'])

# map with covariance matrix
solution2 = pcg(P_packed.T * invntt * P_packed,
                P_packed.T(invntt(tod + noise)),
                M=DiagonalOperator(1/coverage[~mask]), disp=True)
output_map2 = unpack(solution2['x'])

# some display
orig = C(input_map)
orig[mask] = np.nan
hp.gnomview(orig, rot=[0, -57], reso=5, xsize=600, min=-200, max=200,
            title='Original convolved map')
hp.projplot(np.radians(pointings[..., 0]), np.radians(pointings[..., 1]))

output_map1[mask] = np.nan
hp.gnomview(output_map1, rot=[0, -57], reso=5, xsize=600, min=-200, max=200,
            title='Reconstructed map no invntt')

output_map2[mask] = np.nan
hp.gnomview(output_map2, rot=[0, -57], reso=5, xsize=600, min=-200, max=200,
            title='Reconstructed map with invntt')

hp.gnomview(coverage, rot=[0, -57], reso=5, xsize=600, min=-200, max=200,
            title='Coverage')
