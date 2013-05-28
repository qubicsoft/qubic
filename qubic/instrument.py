# coding: utf-8
from __future__ import division

import healpy as hp
try:
    import matplotlib.pyplot as mp
except:
    pass
import numpy as np
from pysimulators import PointingMatrix, ProjectionInMemoryOperator
from scipy.constants import c, pi
from .calibration import QubicCalibration
from .utils import _rotateuv, _compress_mask, _uncompress_mask

__all__ = ['QubicInstrument']

class QubicInstrument(object):

    def __init__(self, name, calibration=None, removed=None, nside=256,
                 **keywords):
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
        self.name = name
        self.calibration = calibration
        self._init_sky(nside)
        self._init_primary_beam()
        self._init_optics(**keywords)
        self._init_detectors(removed)
        self._init_horns()

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
        self.primary_beam = PrimaryBeam(self.calibration.get('fwhm'))

    def _init_optics(self, nu=150e9, dnu_nu=0, **keywords):
        class Optics(object):
            pass
        optics = Optics()
        optics.focal_length = self.calibration.get('focal length')
        optics.nu = nu
        optics.dnu_nu = dnu_nu
        self.optics = optics

    def _init_detectors(self, removed):
        class Detector(np.recarray):
            pass
        dtype = [('center', [('x', float), ('y', float)]),
                 ('corner', [('x', float), ('y', float)], 4),
                 ('index', int),
                 ('quadrant', np.int8),
                 ('masked', bool),
                 ('removed', bool)]
        shape, center, corner, removed_, index, quadrant = self.calibration.get(
            'detarray')
        if removed is not None:
            if isinstance(removed, str):
                removed = _uncompress_mask(removed).reshape(shape)
            removed_ |= removed
        removed = removed_
        detector = Detector(shape, dtype=dtype)
        detector.center.x = center[...,0]
        detector.center.y = center[...,1]
        detector.corner.x = corner[...,0]
        detector.corner.y = corner[...,1]
        detector.masked = False
        detector.removed = removed
        detector.index = index
        detector.quadrant = quadrant
        self.detector = detector

    def _init_horns(self):
        n, kappa, thickness = self.calibration.get('horn')
        class Horn(np.recarray):
            pass
        dtype = [('center', [('x', float), ('y', float)])]
        horn = Horn(n, dtype=dtype)

        nx = int(np.sqrt(n))
        if nx**2 != n:
            raise ValueError('Non-square arrays are not handled.')
        lmbda = c / self.optics.nu
        surface = kappa**2 * lmbda**2 / self.primary_beam.fwhm_sr
        radius = np.sqrt(surface / pi) + thickness
        sizex = 2 * radius * nx
        a = -sizex * 0.5 + radius + sizex * np.arange(nx) / nx
        x, y = np.meshgrid(a, a)
        horn.center.x = x.ravel()
        horn.center.y = y.ravel()
        horn.kappa = kappa
        horn.spacing = a[1] - a[0]
        horn.thickness = thickness
        self.horn = horn

    def __str__(self):
        state = [('name', self.name),
                 ('nu', self.optics.nu),
                 ('dnu_nu', self.optics.dnu_nu),
                 ('nside', self.sky.nside),
                 ('removed', _compress_mask(self.detector.removed))
                ]
        return 'Instrument:\n' + \
               '\n'.join(['    ' + a + ': ' + repr(v) for a,v in state]) + \
               '\n\nCalibration:\n' + \
               '\n'.join('    ' + l for l in str(self.calibration).splitlines())

    __repr__ = __str__

    def get_ndetectors(self):
        """ Return the number of valid detectors. """
        return int(np.sum(~self.detector.removed))

    def pack(self, x):
        """
        Convert representation from 2D to 1D, under the control of the detector
        mask 'removed'.

        """
        d = self.detector
        n = self.get_ndetectors()
        if d.shape != x.shape[:d.ndim]:
            raise ValueError("Invalid input dimensions '{}'.".format(x.shape))
        new_shape = (n,) + x.shape[d.ndim:]
        new_x = np.empty(new_shape, dtype=x.dtype).view(type(x))
        valid_index = d.index[~d.removed]
        num = np.arange(d.size, dtype=int).reshape(d.shape)[~d.removed]
        isort = np.argsort(valid_index)
        x_ = x.reshape((d.size,) + x.shape[d.ndim:])
        for i in range(n):
            new_x[i] = x_[num[isort[i]]]
        return new_x

    def unpack(self, x):
        """
        Convert representation from 1D to 2D, under the control of the detector
        mask 'removed'.

        """
        d = self.detector
        n = self.get_ndetectors()
        if self.get_ndetectors() != x.shape[0]:
            raise ValueError("Invalid input dimensions '{}'.".format(x.shape))
        new_shape = d.shape + x.shape[1:]
        new_x = np.empty(new_shape, dtype=x.dtype).view(type(x))
        #XXX improve me
        if x.dtype == float or x.dtype.kind == 'V':
            new_x[...] = np.nan
        else:
            new_x[...] = 0
        valid_index = d.index[~d.removed]
        num = np.arange(d.size, dtype=int).reshape(d.shape)[~d.removed]
        isort = np.argsort(valid_index)
        new_x_ = new_x.reshape((d.size,) + x.shape[1:])
        for i in range(n):
            new_x_[num[isort[i]]] = x[i]
        return new_x
        
    def plot(self, autoscale=True, **keywords):
        """
        Plot the detector surfaces.
        """
        a = mp.gca()
        corner = self.pack(self.detector.corner).view(float).reshape((-1,4,2))
        for c in corner:
            a.add_patch(mp.Polygon(c, closed=True, fill=False, **keywords))
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
        return ProjectionInMemoryOperator(matrix)


def _peak_angles(q, kmax):
    """
    Return the spherical coordinates (theta,phi) of the beam peaks, in radians.

    """
    ndetector = q.get_ndetectors()
    center = q.pack(q.detector.center)
    lmbda = c / q.optics.nu
    dx = q.horn.spacing
    detvec = np.vstack([-center.x,
                        -center.y,
                        np.zeros(ndetector) + q.optics.focal_length]).T
    detvec.T[...] /= np.sqrt(np.sum(detvec**2, axis=1))
    
    kx, ky = np.mgrid[-kmax:kmax+1,-kmax:kmax+1]
    nx = detvec[:,0,np.newaxis] - lmbda * kx.ravel() / dx
    ny = detvec[:,1,np.newaxis] - lmbda * ky.ravel() / dx  
    theta = np.arcsin(np.sqrt(nx**2 + ny**2))
    phi = np.arctan2(ny,nx)

    return theta, phi


def _peak_pointing_matrix(q, kmax, pointings):
    pointings = np.atleast_2d(pointings)
    npointing = len(pointings)
    ndetector = q.get_ndetectors()
    npeak = (2 * kmax + 1)**2
    npixel = q.sky.npixel

    pointings = np.radians(pointings)
    theta0, phi0 = _peak_angles(q, kmax)
    weight0 = q.primary_beam(theta0).astype(np.float32)
    weight0 /= np.sum(weight0, axis=-1)[...,None]
    
    peakvec = hp.ang2vec(theta0.ravel(), phi0.ravel())
    shape = theta0.shape

    matrix = PointingMatrix.empty((ndetector, npointing, npeak), npixel, info={})

    for i, p in enumerate(pointings):
        theta, phi, psi = p
        newpeakvec = _rotateuv(peakvec, theta, phi, psi, inverse=True)
        newtheta, newphi = [a.reshape(shape) for a in hp.vec2ang(newpeakvec)]
        matrix[:,i,:]['index'] = hp.ang2pix(q.sky.nside, newtheta, newphi)
        matrix[:,i,:]['value'] = weight0

    return matrix


