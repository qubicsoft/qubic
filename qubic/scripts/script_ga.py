from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
import os
import qubic
from qubic import (
    QubicInstrument, create_sweeping_pointings, equ2gal, map2tod, tod2map_all,
    tod2map_each)

DATAPATH = os.path.join(os.path.dirname(qubic.__file__), 'data',
                        'syn256_pol.fits')
x0 = qubic.io.read_map(DATAPATH)

np.random.seed(0)

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
ts = 20             # seconds

# get the observation model
pointing = create_sweeping_pointings(
    [racenter, deccenter], duration, ts, angspeed, delta_az, nsweeps_el,
    angspeed_psi, maxpsi)
pointing.angle_hwp = np.random.random_integers(0, 7, pointing.size) * 11.25

# get the instrument model
instrument = QubicInstrument('monochromatic',
                             nside=nside,
                             detector_tau=0.01,
                             synthbeam_fraction=0.99)

# simulate the timeline
tod, x0_convolved = map2tod(instrument, pointing, x0)

# reconstruct using two methods
map_all, cov_all = tod2map_all(instrument, pointing, tod)
map_each, cov_each = tod2map_each(instrument, pointing, tod)


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
