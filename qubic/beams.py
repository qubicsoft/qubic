from __future__ import division

import numpy as np
from numpy import pi

__all__ = []


class Beam(object):
    def __call__(self, theta_rad, phi_rad):
        raise NotImplementedError()


class GaussianBeam(Beam):
    """
    Axisymmetric gaussian beam.

    """
    def __init__(self, fwhm_deg, backward=False):
        self.sigma_deg = fwhm_deg / np.sqrt(8 * np.log(2))
        self.sigma_rad = np.radians(self.sigma_deg)
        self.fwhm_deg = fwhm_deg
        self.sr = 2 * pi * self.sigma_rad**2
        self.backward = bool(backward)

    def __call__(self, theta_rad, phi_rad=None):
        if self.backward:
            theta_rad = pi - theta_rad
        return np.exp(-theta_rad**2 / (2 * self.sigma_rad**2))


class UniformHalfSpaceBeam(Beam):
    """
    Uniform beam in half-space.

    """
    def __init__(self):
        self.sr = 2 * pi

    def __call__(self, theta_rad, phi_rad=None):
        return 1
