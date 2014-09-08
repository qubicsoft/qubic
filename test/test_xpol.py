from __future__ import division

import numpy as np
from pyoperators.utils.testing import assert_same
from pysimulators import FitsArray
from qubic.xpol import _Mll as Mll


def test_xpol():
    def func(lmax, n):
        print lmax, n
        well = np.ones(max(n+1, 1))
        mll = Mll(lmax, well)
        expected = FitsArray('test/data/xpol_mll_{}_{}.fits'.format(lmax, n))
        assert_same(mll, expected, atol=200)
    for lmax in [0, 1, 2, 10]:
        for n in [lmax-1, lmax, lmax+1]:
            yield func, lmax, n
