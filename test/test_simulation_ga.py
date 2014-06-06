from __future__ import division

import numpy as np
import os
import qubic
from pysimulators import FitsArray
from qubic import (
    QubicCalibration, QubicAcquisition, QubicInstrument, QubicSampling)

MAPPATH = os.path.join(os.path.dirname(qubic.__file__), 'data',
                       'syn256_pol.fits')
DATAPATH = os.path.join(os.path.dirname(__file__), 'data')
map_orig = qubic.io.read_map(MAPPATH, field='I_STOKES')


def test_ga():
    p = FitsArray(os.path.join(DATAPATH, 'ptg_np100_10deg.fits'))
    o = QubicSampling(azimuth=p[..., 0], elevation=p[..., 1],
                      pitch=p[..., 2])
    t = FitsArray(os.path.join(DATAPATH, 'tod_ndet16_np100_ga.fits'))
    c = QubicCalibration(os.path.join(DATAPATH, 'calfiles'))
    i = QubicInstrument('monochromatic,nopol', c, ngrids=1, nu=150e9, dnu_nu=0,
                        nside=256, synthbeam_fraction=0.99)
    acq = QubicAcquisition(i, o)
    C = acq.get_convolution_peak_operator(fwhm=np.radians(0.64883707))
    P = acq.get_projection_operator()
    H = P * C
    t2 = H(map_orig)
    np.testing.assert_almost_equal(t, t2)
