from __future__ import division

import numpy as np
from qubic import QubicInstrument


def test_packunpack():
    q = QubicInstrument('monochromatic,nopol')
    d = q.detector
    data = (d.center, d.corner, d.index, d.removed, d.quadrant)

    def iseq(a, b):
        if a.dtype.kind == 'V':
            return (a.x == b.x) & (a.y == b.y)
        return a == b

    def nonfiniteorzero(x):
        if x.dtype.kind == 'V':
            return ~np.isfinite(x.x) | ~np.isfinite(x.y)
        if x.dtype == float:
            return ~np.isfinite(x)
        return x == 0

    def func(d):
        d_ = q.unpack(q.pack(d))
        assert np.all(iseq(d, d_) | nonfiniteorzero(d_))

    for d in data:
        yield func, d
