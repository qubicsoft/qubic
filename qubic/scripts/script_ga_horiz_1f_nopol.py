from __future__ import division
from pyoperators import pcg, DiagonalOperator, UnpackOperator
from pysimulators.noises import _gaussian_psd_1f
from qubic import (
    QubicAcquisition, QubicScene, create_sweeping_pointings, equ2gal)
import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
import qubic

# read the input map
x0 = qubic.io.read_map(qubic.data.PATH + 'syn256_pol.fits', field='I_STOKES')
nside = 256

# instrument model
fknee = 1
fslope = 1
ncorr = 10

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
scene = QubicScene(nside, kind='I')

# acquisition model
acq = QubicAcquisition(150, sampling, scene, synthbeam_fraction=0.99,
                       detector_fknee=fknee, detector_fslope=fslope,
                       detector_ncorr=ncorr)
H_ga = acq.get_operator()
C = acq.get_convolution_peak_operator()
H = H_ga * C

# produce the Time-Ordered data
y = H(x0)

# noise
sigma = acq.instrument.detector.nep / np.sqrt(2 * sampling.period)
psd = _gaussian_psd_1f(len(acq.sampling), sigma=sigma, fknee=fknee,
                       fslope=fslope, sampling_frequency=1/ts)
invntt = acq.get_invntt_operator()
noise = acq.get_noise()
noise[...] = 0

# map-making
coverage = acq.get_coverage()
mask = coverage / coverage.max() > 0.01

acq_red = acq[..., mask]
H_ga_red = acq_red.get_operator()
# map without covariance matrix
solution1 = pcg(H_ga_red.T * H_ga_red, H_ga_red.T(y + noise),
                M=DiagonalOperator(1/coverage[mask]), disp=True)
x1 = acq_red.scene.unpack(solution1['x'])

# map with covariance matrix
solution2 = pcg(H_ga_red.T * invntt * H_ga_red,
                (H_ga_red.T * invntt)(y + noise),
                M=DiagonalOperator(1/coverage[mask]), disp=True)
x2 = acq_red.scene.unpack(solution2['x'])


def display(x, title, min=None, max=None):
    x = x.copy()
    x[~mask] = np.nan
    hp.gnomview(x, rot=center, reso=5, xsize=600, min=min, max=max,
                title=title)

center = equ2gal(racenter, deccenter)
display(C(x0), 'Original convolved map', min=-200, max=200)
#hp.projplot(np.radians(pointings[..., 0]), np.radians(pointings[..., 1]))
display(x1, 'Reconstructed map no invntt', min=-200, max=200)
display(x2, 'Reconstructed map with invntt', min=-200, max=200)
display(coverage, 'Coverage')

mp.show()
