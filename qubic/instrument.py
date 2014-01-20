# coding: utf-8
from __future__ import division

try:
    import matplotlib.pyplot as mp
except:
    pass
import numpy as np
from pyoperators import Spherical2CartesianOperator, MPI
from pyoperators.utils import strenum
from pysimulators import Instrument, Layout, ProjectionOperator, _flib as flib
from pysimulators.interfaces.healpy import Cartesian2HealpixOperator
from pysimulators.sparse import FSRMatrix, FSRRotation3dMatrix
from scipy.constants import c, pi
from .calibration import QubicCalibration
from .utils import _compress_mask, _uncompress_mask

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
        names = 'monochromatic,nopol', 'monochromatic'
        if name not in names:
            raise ValueError(
                "The only modes implemented are {0}.".format(
                strenum(names, 'and')))
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
        self.sky.polarized = 'nopol' not in self.name.split(',')

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
        self.horn = self.calibration.get('hornarray')

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

    def get_projection_peak_operator(self, rotation, kmax=2):
        """
        Return the peak sampling operator.

        Parameters
        ----------
        rotation : ndarray (ntimes, 3, 3)
            The Reference-to-Instrument rotation matrix.
        kmax : int, optional
            The diffraction order above which the peaks are ignored.
            For a value of 0, only the central peak is sampled.

        """
        ndetectors = len(self.detector.packed)
        ntimes = rotation.data.shape[0]
        nside = self.sky.nside
        ncolmax = (2 * kmax + 1)**2

        theta, phi = _peak_angles(self, kmax)
        thetaphi = _pack_vector(theta, phi)  # (ndetectors, ncolmax, 2)
        direction = Spherical2CartesianOperator('zenith,azimuth')(thetaphi)
        e_nf = _replicate(direction, ntimes)  # (ndets, ntimes, ncolmax, 3)
        if nside > 8192:
            raise 'With nside={0}, pixels cannot be stored using int32.'

        if not self.sky.polarized:
            s = FSRMatrix((ndetectors*ntimes, 12*nside**2),
                          ncolmax=ncolmax, dtype=np.float32,
                          dtype_index=np.int32)
        else:
            s = FSRRotation3dMatrix((ndetectors*ntimes*3, 12*nside**2*3),
                                    ncolmax=ncolmax, dtype=np.float32,
                                    dtype_index=np.int32)

        index = s.data.index.reshape((ndetectors, ntimes, ncolmax))
        for i in xrange(ndetectors):
            # e_ni, e_nf[i] shape: (ntimes, ncolmax, 3)
            e_ni = rotation.T(e_nf[i].swapaxes(0, 1)).swapaxes(0, 1)
            index[i] = Cartesian2HealpixOperator(nside)(e_ni)
        vals = self.primary_beam(theta)  # (ndetectors, ncolmax)

        if not self.sky.polarized:
            vals /= np.sum(vals, axis=-1)[..., None]  # remove me
            value = s.data.value.reshape(ndetectors, ntimes, ncolmax)
            value[...] = vals[:, None, :]
            shapeout = (ndetectors, ntimes)
        else:
            flib.polarization.pointing_matrix_i4_m4(
                rotation.data.T, direction.T, s.data.ravel().view(np.int8),
                vals.T)
            shapeout = (ndetectors, ntimes, 3)

        return ProjectionOperator(s, shapeout=shapeout)


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


def _pack_vector(*args):
    shape = np.broadcast(*args).shape
    out = np.empty(shape + (len(args),))
    for i, arg in enumerate(args):
        out[..., i] = arg
    return out


def _replicate(v, n):
    shape = v.shape[:-2] + (n,) + v.shape[-2:]
    strides = v.strides[:-2] + (0,) + v.strides[-2:]
    return np.lib.stride_tricks.as_strided(v, shape, strides)
