import pytest
from pathlib import Path

import numpy as np
from pyoperators.utils.testing import assert_same
from pysimulators import FitsArray
from qubic import Xpol

DATAPATH = Path(__file__).parent / 'data'


@pytest.mark.parametrize('lmax', [0, 1, 2, 10])
@pytest.mark.parametrize('dn', [-1, 0, 1])
def test_xpol(lmax, dn):
    n = lmax + dn
    class XpolDummy(Xpol):
        def __init__(self):
            self.lmax = lmax
            self.wl = np.ones(max(n+1, 1))

    xpol = XpolDummy()
    mll = xpol._get_Mll(binning=False)
    expected = FitsArray(str(DATAPATH / f'xpol_mll_{lmax}_{n}.fits'))
    assert_same(mll, expected, atol=200)
