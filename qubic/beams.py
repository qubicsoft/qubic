import numpy as np
from pyoperators.utils import reshape_broadcast
import numexpr as ne
import healpy as hp
from scipy import interpolate

__all__ = ['BeamGaussian',
           'BeamFitted',
           'MultiFreqBeam']


def with_alpha(x, alpha, xspl):
    nbspl = xspl.size
    theF = np.zeros((np.size(x), nbspl))
    for i in np.arange(nbspl):
        theF[:, i] = get_spline_tofit(xspl, i, x)
    return np.dot(theF, alpha)


def get_spline_tofit(xspline, index, xx):
    yspline = np.zeros(np.size(xspline))
    yspline[index] = 1.
    tck = interpolate.splrep(xspline, yspline)
    yy = interpolate.splev(xx, tck, der=0)
    return yy


def gauss_plus(x, a, s, z):
    """
    Computes the beam profile from a 6 gaussians model as a funcion of x
    sum_i(a_i * exp(-(x-z_i)**2/2/s_i**2 + z_i -> -z_i)
    a, s, z : six components vectors
    s and z are in radians, a has no dimension.
    x : one dimensional vector of any size, in radian

    """
    x = x[..., None]
    out = (np.exp(-(x - z) ** 2 / 2 / s ** 2) + np.exp(-(x + z) ** 2 / 2 / s ** 2)) * a
    return out.sum(axis=-1)


class Beam(object):
    def __init__(self, solid_angle, nu=None):
        """
        Parameter
        ---------
        solid_angle : float
            The beam solid angle [sr].

        """
        self.solid_angle = float(solid_angle)
        if nu:
            self.nu = nu

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


class BeamGaussian(Beam):
    """
    Axisymmetric gaussian beam.

    """

    def __init__(self, fwhm, nu=150, backward=False):
        """
        Warning: The nu dependence of the fwhm is not a good approximation in the 220 GHz band.

        Parameters
        ----------
        fwhm : float
            The Full-Width-Half-Maximum of the beam, in radians.
        backward : boolean, optional
            If True, the maximum of the beam is at theta=pi.
        """
        self.nu = nu
        if 250 > self.nu > 190:
            print('Warning: The nu dependency of the gausian beam FWHM '
                  'is not a good approximation in the 220 GHz band.')
        if 170 > self.nu > 130:
            self.fwhm = fwhm * 150 / self.nu
        else:  # nu = 220
            self.fwhm = 0.1009 * np.sqrt(8 * np.log(2)) * 220 / self.nu
        self.sigma = self.fwhm / np.sqrt(8 * np.log(2))
        self.backward = bool(backward)
        Beam.__init__(self, 2 * np.pi * self.sigma ** 2)

    def __call__(self, theta, phi):
        if self.backward:
            theta = np.pi - theta
        coef = -0.5 / self.sigma ** 2
        out = ne.evaluate('exp(coef * theta**2)')
        return reshape_broadcast(out, np.broadcast(theta, phi).shape)


class BeamFitted(Beam):
    """
    Axisymmetric fitted beam.
        Warning: The nu dependence of the beam and the solid angle are not well approximated in the 220 GHz band

    """

    def __init__(self, par, omega, nu=150, backward=False):
        """
        Warning! beam and solid angle frequency dependence implementation
        in the 220 GHz band does not correctly describe the true behavior

        Parameters
        ----------
        par: the parameters of the fit
        omega : beam total solid angle
        backward : boolean, optional
            If true, the maximum of the beam is at theta=pi.
        """
        self.par = par
        self.nu = nu
        if 250 >= self.nu >= 190:
            print('Warning! beam and solid angle frequency dependence implementation '
                  'in the 220 GHz band for the fitted beam does not correctly describe '
                  'the true behavior')
        if 170 >= self.nu >= 130:
            omega *= (150 / nu) ** 2
        else:
            omega *= (220 / nu) ** 2
        self.backward = bool(backward)
        Beam.__init__(self, omega)

    def __call__(self, theta, phi):
        par = self.par
        if self.backward:
            theta = np.pi - theta
        theta_eff = theta
        if 170 >= self.nu >= 130:
            theta_eff = theta * self.nu / 150
        else:
            theta_eff = theta * self.nu / 220
        out = gauss_plus(theta_eff, par[0], par[1], par[2])
        return reshape_broadcast(out, np.broadcast(theta, phi).shape)


class MultiFreqBeam(Beam):
    """
    spline fitted multifrequency beam

    """

    def __init__(self, parth, parfr, parbeam, alpha, xspl, nu=150,
                 backward=False):
        """
        Parameters
        ----------
        parth, parfr, parbeam: angles, frequencies and beam values to be
            extrapolated
        alpha, xspl: spline parameters to evaluate the solid angle
            at frequency nu

        """
        self.nu = nu
        self.parth = parth  # input thetas
        self.parfr = parfr  # 27 input frequencies from 130 to 242 Ghz
        self.parbeam = parbeam  # input Beam values at parth and parfr
        self.alpha = alpha
        self.xspl = xspl
        self.backward = bool(backward)
        self.sp = interpolate.RectBivariateSpline(parth, parfr, parbeam)
        omega = with_alpha(nu, self.alpha, self.xspl)
        Beam.__init__(self, omega, nu=nu)

    def __call__(self, theta, phi):
        if self.backward:
            theta = np.pi - theta
        out = self.sp(theta, self.nu, grid=False)

        return reshape_broadcast(out, np.broadcast(theta, phi).shape)


class BeamUniformHalfSpace(Beam):
    """
    Uniform beam in half-space.

    """

    def __init__(self):
        Beam.__init__(self, 2 * np.pi)

    def __call__(self, theta, phi):
        out = 1.
        return reshape_broadcast(out, np.broadcast(theta, phi).shape)
