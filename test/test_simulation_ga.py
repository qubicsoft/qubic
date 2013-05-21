from __future__ import division

import healpy as hp
import numpy as np
import os
from pysimulators import FitsArray
from qubic import QubicInstrument, QubicConfiguration

path = os.path.join(os.path.dirname(__file__), 'data')

def test_ga():
    map_orig = hp.read_map(os.path.join(path, 'syn256.fits'))
    p = FitsArray(os.path.join(path, 'ptg_np100_10deg.fits'))
    t = FitsArray(os.path.join(path, 'tod_ndet10_np100_ga.fits'))
    q = QubicInstrument(fwhm_deg=14, focal_length=0.3, nu=150e9, dnu_nu=0,
                        ndetector=16, detector_size=3e-3, nhorn=400,
                        kappa=1.344, horn_thickness=0.001, nside=256,
                        version='2.0')
    obs = QubicConfiguration(p, q)
    C = obs.get_convolution_peak_operator(fwhm=np.radians(0.64883707))
    P = obs.get_projection_peak_operator(kmax=2)
    H = P * C
    t2 = H(map_orig)
    np.testing.assert_almost_equal(t, t2)
