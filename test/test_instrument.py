from __future__ import division

from pyoperators.utils import settingerr
from pyoperators.utils.testing import assert_equal, assert_same, assert_is_type
from pysimulators import FitsArray
from qubic import create_random_pointings, QubicAcquisition, QubicInstrument
from qubic.beams import GaussianBeam, UniformHalfSpaceBeam

q = QubicInstrument(detector_ngrids=1)
s = create_random_pointings([0, 90], 5, 10.)


def test_detector_indexing():
    expected = FitsArray('test/data/detector_indexing.fits')
    assert_same(q.detector.index, expected)


def test_beams():
    assert_is_type(q.primary_beam, GaussianBeam)
    assert_is_type(q.secondary_beam, GaussianBeam)
    a = QubicAcquisition(150, s, primary_beam=UniformHalfSpaceBeam(),
                         secondary_beam=UniformHalfSpaceBeam())
    assert_is_type(a.instrument.primary_beam, UniformHalfSpaceBeam)
    assert_is_type(a.instrument.secondary_beam, UniformHalfSpaceBeam)


def test_primary_beam():
    def primary_beam(theta, phi):
        import numpy as np
        with settingerr(invalid='ignore'):
            return theta <= np.radians(9)
    a = QubicAcquisition(150, s, primary_beam=primary_beam)
    p = a.get_projection_operator()
    assert_equal(p.matrix.ncolmax, 5)
