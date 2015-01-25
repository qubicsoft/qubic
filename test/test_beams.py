from __future__ import division

import numpy as np
from pyoperators.utils.testing import assert_same
from qubic.beams import GaussianBeam, UniformHalfSpaceBeam

angles = np.radians([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])


def test_gaussian():
    primary = GaussianBeam(14)
    secondary = GaussianBeam(14, backward=True)
    assert_same(primary(np.pi - angles, 0), secondary(angles, 0))


def test_uniform():
    beam = UniformHalfSpaceBeam()
    assert_same(beam(10, 0), 1)
