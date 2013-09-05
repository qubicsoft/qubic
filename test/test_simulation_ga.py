from __future__ import division

import healpy as hp
import numpy as np
import os
from os.path import dirname, join
from pysimulators import FitsArray
from qubic import QubicCalibration, QubicConfiguration, QubicInstrument

path = join(dirname(__file__), 'data')


def test_ga():
    map_orig = hp.read_map(os.path.join(path, 'syn256.fits'))
    p = FitsArray(os.path.join(path, 'ptg_np100_10deg.fits'))
    t = FitsArray(os.path.join(path, 'tod_ndet10_np100_ga.fits'))
    c = QubicCalibration(join(path, 'calfiles'))
    q = QubicInstrument('monochromatic,nopol', c, nu=150e9, dnu_nu=0,
                        nside=256)
    obs = QubicConfiguration(q, p)
    C = obs.get_convolution_peak_operator(fwhm=np.radians(0.64883707))
    P = obs.get_projection_peak_operator(kmax=2)
    H = P * C
    t2 = H(map_orig)
    np.testing.assert_almost_equal(t, t2)
