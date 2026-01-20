from pathlib import Path

import numpy as np
import pytest

from pyoperators.utils.testing import assert_same
from pysimulators import FitsArray
from qubic.lib.obsolete import Xpol

TEST_DATA = Path(__file__).parent / 'data'


@pytest.mark.parametrize('lmax', [0, 1, 2, 10])
@pytest.mark.parametrize('delta', [-1, 0, 1])
def test_xpol(lmax, delta):
    class XpolDummy(Xpol):
        def __init__(self):
            self.lmax = lmax
            self.wl = np.ones(max(n+1, 1))

    n = lmax + delta
    xpol = XpolDummy()
    mll = xpol._get_Mll(binning=False)
    expected = FitsArray(TEST_DATA / f'xpol_mll_{lmax}_{n}.fits')
    assert_same(mll, expected, atol=200)
