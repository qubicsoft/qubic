from __future__ import division
from astropy.time import Time
from pyoperators.utils.testing import assert_same
from pysimulators import (
    SphericalEquatorial2GalacticOperator,
    SphericalEquatorial2HorizontalOperator)
from qubic.samplings import (
    equ2gal, gal2equ, equ2hor, hor2equ, gal2hor, hor2gal, DOMECLAT, DOMECLON,
    QubicSampling)
import numpy as np


def test_sphconv():

    time = Time(QubicSampling.DEFAULT_DATE_OBS, scale='utc')

    def _pack(x):
        return np.array([x[0], x[1]]).T

    sphs = equ2gal, gal2equ, equ2hor, hor2equ, gal2hor, hor2gal
    extraargs = (), (), (0,), (0,), (0,), (0,)
    e2g = SphericalEquatorial2GalacticOperator(degrees=True)
    e2h = SphericalEquatorial2HorizontalOperator(
        'NE', time, DOMECLAT, DOMECLON, degrees=True)
    refs = e2g, e2g.I, e2h, e2h.I, e2h(e2g.I), e2g(e2h.I)

    incoords = np.array([[10, 20], [30, 40], [50, 60]])

    def func(sph, extraarg, ref):
        args = (incoords[..., 0], incoords[..., 1]) + extraarg
        outcoords = _pack(sph(*args))
        assert_same(outcoords, ref(incoords), rtol=100)

    for sph, extraarg, ref in zip(sphs, extraargs, refs):
        yield func, sph, extraarg, ref
