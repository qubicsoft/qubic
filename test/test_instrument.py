from __future__ import division
import numpy as np
from pyoperators.utils import settingerr
from pyoperators.utils.testing import assert_equal, assert_same, assert_is_type
from pysimulators import BeamGaussian, BeamUniformHalfSpace, FitsArray
from qubic import (
    create_random_pointings, QubicAcquisition, QubicInstrument, QubicSampling,
    QubicScene)

q = QubicInstrument(detector_ngrids=1)
s = create_random_pointings([0, 90], 5, 10.)


def test_detector_indexing():
    expected = FitsArray('test/data/detector_indexing.fits')
    assert_same(q.detector.index, expected)


def test_beams():
    assert_is_type(q.primary_beam, BeamGaussian)
    assert_is_type(q.secondary_beam, BeamGaussian)
    a = QubicAcquisition(150, s, primary_beam=BeamUniformHalfSpace(),
                         secondary_beam=BeamUniformHalfSpace())
    assert_is_type(a.instrument.primary_beam, BeamUniformHalfSpace)
    assert_is_type(a.instrument.secondary_beam, BeamUniformHalfSpace)


def test_primary_beam():
    def primary_beam(theta, phi):
        import numpy as np
        with settingerr(invalid='ignore'):
            return theta <= np.radians(9)
    a = QubicAcquisition(150, s, primary_beam=primary_beam)
    p = a.get_projection_operator()
    assert_equal(p.matrix.ncolmax, 5)


def test_polarizer():
    sampling = QubicSampling(0, 0, 0)
    scene = QubicScene(150, kind='I')
    instruments = [QubicInstrument(polarizer=False),
                   QubicInstrument(polarizer=True),
                   QubicInstrument(polarizer=False, detector_ngrids=1),
                   QubicInstrument(polarizer=True, detector_ngrids=1),
                   QubicInstrument(polarizer=False)[:992],
                   QubicInstrument(polarizer=True)[:992]]

    n = len(instruments[0])
    expecteds = [np.r_[np.ones(n // 2), np.zeros(n // 2)],
                 np.full(n, 0.5),
                 np.ones(n // 2),
                 np.full(n // 2, 0.5),
                 np.ones(n // 2),
                 np.full(n // 2, 0.5)]

    def func(instrument, expected):
        op = instrument.get_polarizer_operator(sampling, scene)
        sky = np.ones((len(instrument), 1))
        assert_equal(op(sky).ravel(), expected)
    for instrument, expected in zip(instruments, expecteds):
        yield func, instrument, expected
