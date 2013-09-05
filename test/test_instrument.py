from __future__ import division

import numpy as np
from pyoperators.utils.testing import assert_same
from pysimulators import FitsArray
from qubic import QubicInstrument

q = QubicInstrument('monochromatic,nopol')


def test_detector_indexing():
    expected = FitsArray('test/data/detector_indexing.fits')
    assert_same(q.detector.packed.index, expected)
