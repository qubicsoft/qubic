from __future__ import division
from pysimulators import FitsArray, BeamGaussian
from qubic import (
    QubicCalibration, QubicAcquisition, QubicInstrument, QubicSampling,
    QubicScene)
import numpy as np
import os
import qubic

DATAPATH = os.path.join(os.path.dirname(__file__), 'data')
map_orig = qubic.io.read_map(qubic.data.PATH + 'syn256_pol.fits',
                             field='I_STOKES')


def test_ga():
    p = FitsArray(os.path.join(DATAPATH, 'ptg_np100_10deg.fits'))
    o = QubicSampling(azimuth=p[..., 0], elevation=p[..., 1],
                      pitch=p[..., 2])
    s = QubicScene(256, kind='I')
    t = FitsArray(os.path.join(DATAPATH, 'tod_ndet16_np100_ga.fits'))
    c = QubicCalibration(os.path.join(DATAPATH, 'calfiles'))
    i = QubicInstrument(c, synthbeam_fraction=0.99)
    i.synthbeam.peak = BeamGaussian(np.radians(0.64883707))
    acq = QubicAcquisition(i, o, s)
    C = acq.get_convolution_peak_operator()
    P = acq.get_projection_operator()
    H = P * C
    t2 = H(map_orig)
    np.testing.assert_almost_equal(t, t2)
