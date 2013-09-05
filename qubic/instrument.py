# coding: utf-8
from __future__ import division

import healpy as hp
try:
    import matplotlib.pyplot as mp
except:
    pass
import numpy as np
from pyoperators import MPI
from pysimulators import (Instrument, Layout, PointingMatrix,
                          ProjectionInMemoryOperator, Map, Tod)
from scipy.constants import c, pi
from .calibration import QubicCalibration
from .utils import _rotateuv, _compress_mask, _uncompress_mask

__all__ = ['QubicInstrument']


class QubicInstrument(Instrument):
    """
    The QubicInstrument class. It represents the instrument setup.

    """
    def __init__(self, name, calibration=None, removed=None, nside=256,
                 commin=MPI.COMM_WORLD, commout=MPI.COMM_WORLD, **keywords):
        """
        Parameters
        ----------
        name : str
            The module name. So far, only 'monochromatic,nopol' is available.
        calibration : QubicCalibration
            The calibration tree.
        removed : str or 2D-array of bool
            Array specifying which bolometers are removed.
        nside : int
            The Healpix nside of the sky.
        nu : float, optional
            The operating monochromatic frequency or the filter central
            frequency, in Hz.
        dnu_nu : float, optional
            The filter width.

        """
        if calibration is None:
            calibration = QubicCalibration()
        if name != 'monochromatic,nopol':
            raise ValueError("Only 'monochromatic,nopol' is implemented.")
        self.calibration = calibration
        layout = self._get_detector_layout(removed)
        Instrument.__init__(self, name, layout, commin=commin, commout=commout)
        self._init_sky(nside)
        self._init_primary_beam()
        self._init_optics(**keywords)
        self._init_horns()

    def _get_detector_layout(self, removed):
        shape, vertex, removed_, index, quadrant = \
            self.calibration.get('detarray')
        if removed is not None:
            if isinstance(removed, str):
                removed = _uncompress_mask(removed).reshape(shape)
            removed_ |= removed
        return Layout(shape, vertex=vertex, removed=removed_, index=index,
                      quadrant=quadrant)

    def _init_sky(self, nside):
        class Sky(object):
            pass
        self.sky = Sky()
        self.sky.npixel = 12 * nside**2
        self.sky.nside = nside

    def _init_primary_beam(self):
        class PrimaryBeam(object):
            def __init__(self, fwhm_deg):
                self.sigma = np.radians(fwhm_deg) / np.sqrt(8 * np.log(2))
                self.fwhm_deg = fwhm_deg
                self.fwhm_sr = 2 * pi * self.sigma**2
            def __call__(self, theta):
                return np.exp(-theta**2 / (2 * self.sigma**2))
        self.primary_beam = PrimaryBeam(self.calibration.get('primbeam'))

    def _init_optics(self, nu=150e9, dnu_nu=0, **keywords):
        class Optics(object):
            pass
        optics = Optics()
        optics.focal_length = self.calibration.get('optics')['focal length']
        optics.nu = nu
        optics.dnu_nu = dnu_nu
        self.optics = optics

    def _init_horns(self):
        class Horn(np.recarray):
            pass
        shape, center = self.calibration.get('hornarray')
        n = shape[0] * shape[1]
        dtype = [('center', float, 2)]
        horn = Horn(n, dtype=dtype)
        horn.center = center.reshape((-1, 2))
        horn.spacing = abs(center[0, 0, 0] - center[0, 1, 0])
        self.horn = horn

    def __str__(self):
        state = [('name', self.name),
                 ('nu', self.optics.nu),
                 ('dnu_nu', self.optics.dnu_nu),
                 ('nside', self.sky.nside),
                 ('removed', _compress_mask(self.detector.removed))]
        return 'Instrument:\n' + \
               '\n'.join(['    ' + a + ': ' + repr(v) for a, v in state]) + \
               '\n\nCalibration:\n' + '\n'. \
               join('    ' + l for l in str(self.calibration).splitlines())

    __repr__ = __str__

    def plot(self, autoscale=True, **keywords):
        """
        Plot detectors on the image plane.

        """
        a = mp.gca()
        vertex = self.detector.packed.vertex.reshape((-1, 4, 2))
        for v in vertex:
            a.add_patch(mp.Polygon(v, closed=True, fill=False, **keywords))
        if autoscale:
            mp.autoscale()
        mp.show()

    def get_projection_peak_operator(self, pointing, kmax=2):
        """
        Return the peak sampling operator.

        Parameters
        ----------
        pointing : ndarray
            The pointing for which the sampling is calculated.
        kmax : int, optional
            The diffraction order above which the peaks are ignored.
            For a value of 0, only the central peak is sampled.

        """
        matrix = _peak_pointing_matrix(self, kmax, pointing)
        return ProjectionInMemoryOperator(matrix, classin=Map, classout=Tod)


def _peak_angles(q, kmax):
    """
    Return the spherical coordinates (theta,phi) of the beam peaks, in radians.

    """
    ndetector = len(q.detector.packed)
    center = q.detector.packed.center
    lmbda = c / q.optics.nu
    dx = q.horn.spacing
    detvec = np.vstack([-center[..., 0],
                        -center[..., 1],
                        np.zeros(ndetector) + q.optics.focal_length]).T
    detvec.T[...] /= np.sqrt(np.sum(detvec**2, axis=1))

    kx, ky = np.mgrid[-kmax:kmax+1, -kmax:kmax+1]
    nx = detvec[:, 0, np.newaxis] - lmbda * kx.ravel() / dx
    ny = detvec[:, 1, np.newaxis] - lmbda * ky.ravel() / dx
    theta = np.arcsin(np.sqrt(nx**2 + ny**2))
    phi = np.arctan2(ny, nx)

    return theta, phi


def _peak_pointing_matrix(q, kmax, pointings):
    pointings = np.atleast_2d(pointings)
    npointing = len(pointings)
    ndetector = len(q.detector.packed)
    npeak = (2 * kmax + 1)**2
    npixel = q.sky.npixel

    pointings = np.radians(pointings)
    theta0, phi0 = _peak_angles(q, kmax)
    weight0 = q.primary_beam(theta0).astype(np.float32)
    weight0 /= np.sum(weight0, axis=-1)[..., None]

    peakvec = hp.ang2vec(theta0.ravel(), phi0.ravel())
    shape = theta0.shape

    matrix = PointingMatrix.empty((ndetector, npointing, npeak), npixel)

    for i, p in enumerate(pointings):
        theta, phi, psi = p
        newpeakvec = _rotateuv(peakvec, theta, phi, psi, inverse=True)
        newtheta, newphi = [a.reshape(shape) for a in hp.vec2ang(newpeakvec)]
        matrix[:, i, :].index = hp.ang2pix(q.sky.nside, newtheta, newphi)
        matrix[:, i, :].value = weight0

    return matrix
