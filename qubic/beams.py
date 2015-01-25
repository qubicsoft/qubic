from __future__ import division
import healpy as hp
import numexpr as ne
import numpy as np
from pyoperators.utils import reshape_broadcast
from numpy import pi

__all__ = []


class Beam(object):
    def __init__(self, solid_angle):
        """
        Parameter
        ---------
        solid_angle : float
            The beam solid angle [sr].

        """
        self.solid_angle = float(solid_angle)

    def __call__(self, theta_rad, phi_rad):
        raise NotImplementedError()

    def healpix(self, nside):
        """
        Return the beam as a Healpix map.

        Parameter
        ---------
        nside : int
             The Healpix map's nside.

        """
        npix = hp.nside2npix(nside)
        theta, phi = hp.pix2ang(nside, np.arange(npix))
        return self(theta, phi)


class GaussianBeam(Beam):
    """
    Axisymmetric gaussian beam.

    """
    def __init__(self, fwhm_deg, backward=False):
        self.sigma_deg = fwhm_deg / np.sqrt(8 * np.log(2))
        self.sigma_rad = np.radians(self.sigma_deg)
        self.fwhm_deg = fwhm_deg
        self.backward = bool(backward)
        Beam.__init__(self, 2 * pi * self.sigma_rad**2)

    def __call__(self, theta_rad, phi_rad):
        if self.backward:
            theta_rad = pi - theta_rad
        coef = -0.5 / self.sigma_rad**2
        out = ne.evaluate('exp(coef * theta_rad**2)')
        return reshape_broadcast(out, np.broadcast(theta_rad, phi_rad).shape)


class UniformHalfSpaceBeam(Beam):
    """
    Uniform beam in half-space.

    """
    def __init__(self):
        Beam.__init__(self, 2 * pi)

    def __call__(self, theta_rad, phi_rad):
        out = 1.
        return reshape_broadcast(out, np.broadcast(theta_rad, phi_rad).shape)
