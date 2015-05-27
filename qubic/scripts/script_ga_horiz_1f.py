from __future__ import division
from qubic import (
    QubicAcquisition, create_sweeping_pointings, equ2gal, map2tod, tod2map_all,
    tod2map_each)
import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
import qubic

x0 = qubic.io.read_map(qubic.data.PATH + 'syn256_pol.fits')

# parameters
nside = 256
racenter = 0.0      # deg
deccenter = -57.0   # deg
angspeed = 1        # deg/sec
delta_az = 15.      # deg
angspeed_psi = 0.1  # deg/sec
maxpsi = 45.        # deg
nsweeps_el = 300
duration = 24       # hours
ts = 20             # seconds

# get the sampling model
np.random.seed(0)
sampling = create_sweeping_pointings(
    [racenter, deccenter], duration, ts, angspeed, delta_az, nsweeps_el,
    angspeed_psi, maxpsi)

# get the acquisition model
acquisition = QubicAcquisition(150, sampling,
                               nside=nside,
                               synthbeam_fraction=0.99,
                               detector_tau=0.01,
                               detector_nep=1.e-17,
                               detector_fknee=1.,
                               detector_fslope=1)

# simulate the timeline
tod, x0_convolved = map2tod(acquisition, x0, convolution=True)
tod_noisy = tod + acquisition.get_noise()

# reconstruct using two methods
map_all, cov_all = tod2map_all(acquisition, tod, tol=1e-2)
map_each, cov_each = tod2map_each(acquisition, tod, tol=1e-2)


# some display
def display(map, cov, msg, sub):
    for i, (kind, lim) in enumerate(zip('IQU', [200, 10, 10])):
        map_ = map[..., i].copy()
        mask = cov == 0
        map_[mask] = np.nan
        hp.gnomview(map_, rot=center, reso=5, xsize=400, min=-lim, max=lim,
                    title=msg + ' ' + kind, sub=(3, 3, 3 * (sub-1) + i+1))

center = equ2gal(racenter, deccenter)

mp.figure(1)
display(x0_convolved, cov_all, 'Original map', 1)
display(map_all, cov_all, 'Reconstructed map', 2)
display(map_all - x0_convolved, cov_all, 'Difference map', 3)
mp.show()

mp.figure(2)
display(x0_convolved, cov_each, 'Original map', 1)
display(map_each, cov_each, 'Reconstructed map', 2)
display(map_each - x0_convolved, cov_each, 'Difference map', 3)
mp.show()
