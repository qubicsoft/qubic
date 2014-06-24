# coding: utf-8
from __future__ import division

try:
    import matplotlib.pyplot as mp
except:
    pass
import numexpr as ne
import numpy as np
from multiprocessing.dummy import Pool
from pyoperators import (
    DenseBlockDiagonalOperator, IdentityOperator, HomothetyOperator,
    ReshapeOperator, Rotation2dOperator, Rotation3dOperator,
    Spherical2CartesianOperator)
from pyoperators.utils import openmp_num_threads, strenum
from pysimulators import (
    Instrument, LayoutVertex, ConvolutionTruncatedExponentialOperator,
    ProjectionOperator)
from pysimulators.interfaces.healpy import Cartesian2HealpixOperator
from pysimulators.sparse import (
    FSRMatrix, FSRRotation2dMatrix, FSRRotation3dMatrix)
from scipy.constants import c, pi
from . import _flib as flib
from .calibration import QubicCalibration
from .scene import QubicScene
from .utils import _compress_mask
from .warnings import QubicDeprecationWarning, warn

__all__ = ['QubicInstrument']


class QubicInstrument(Instrument):
    """
    The QubicInstrument class. It represents the instrument setup.

    """
    def __init__(self, name=None, calibration=None,
                 detector_sigma=10, detector_fknee=0, detector_fslope=1,
                 detector_ncorr=10, detector_tau=0.01,
                 synthbeam_dtype=np.float32, synthbeam_fraction=0.99,
                 ngrids=2, **keywords):
        """
        Parameters
        ----------
        name : str
            Deprecated.
        calibration : QubicCalibration
            The calibration tree.
        detector_tau : array-like
            The detector time constants in seconds.
        detector_sigma : array-like
            The standard deviation of the detector white noise component.
        detector_fknee : array-like
            The detector 1/f knee frequency in Hertz.
        detector_fslope : array-like
            The detector 1/f slope index.
        detector_ncorr : int
            The detector 1/f correlation length.
        synthbeam_dtype : dtype, optional
            The data type for the synthetic beams (default: float32).
            It is the dtype used to store the values of the pointing matrix.
        synthbeam_fraction: float, optional
            The fraction of significant peaks retained for the computation
            of the synthetic beam.
        ngrids : int, optional
            Number of detector grids.

        """
        if name is not None:
            nside = keywords.pop('nside', None)
            skind = ", kind='I'" if 'nopol' in name else ''
            snside = ', nside={0}'.format(nside) if nside is not None else ''
            warn('Please update your code:\nacq = QubicAcquisition(150, sam'
                 'pling{0}{1})\n'.format(skind, snside),
                 QubicDeprecationWarning)
            name = name.replace(' ', '').lower()
            names = 'monochromatic', 'monochromatic,qu', 'monochromatic,nopol'
            if name not in names:
                raise ValueError(
                    "The only modes implemented are {0}.".format(
                        strenum(names, 'and')))
            self._init_deprecated_sky(name, nside)

        if calibration is None:
            calibration = QubicCalibration()
        self.calibration = calibration
        layout = self._get_detector_layout(
            ngrids, detector_sigma, detector_fknee, detector_fslope,
            detector_ncorr, detector_tau)
        Instrument.__init__(self, 'QUBIC', layout)
        self._init_primary_beam()
        self._init_optics(**keywords)
        self._init_horns()
        self._init_synthetic_beam(synthbeam_dtype, synthbeam_fraction)

    def _get_detector_layout(self, ngrids, sigma, fknee, fslope, ncorr,
                             tau):
        shape, vertex, removed_, index, quadrant = \
            self.calibration.get('detarray')
        if ngrids == 2:
            shape = (2,) + shape
            vertex = np.array([vertex, vertex])
            removed_ = np.array([removed_, removed_])
            index = np.array([index, index + np.max(index) + 1], index.dtype)
            quadrant = np.array([quadrant, quadrant + 4], quadrant.dtype)
        layout = LayoutVertex(
            shape, 4, vertex=vertex, selection=~removed_, ordering=index,
            quadrant=quadrant, sigma=sigma, fknee=fknee, fslope=fslope,
            tau=tau)
        layout.ncorr = ncorr
        layout.ngrids = ngrids
        return layout

    def _init_deprecated_sky(self, name, nside):
        names = name.split(',')
        kind = 'I' if 'nopol' in names \
                   else 'QU' if 'qu' in names \
                             else 'IQU'
        sky = QubicScene(150, kind=kind, nside=nside)
        self._deprecated_sky = sky

    def _init_primary_beam(self):
        class PrimaryBeam(object):
            def __init__(self, fwhm_deg):
                self.sigma = np.radians(fwhm_deg) / np.sqrt(8 * np.log(2))
                self.fwhm_deg = fwhm_deg
                self.fwhm_sr = 2 * pi * self.sigma**2
            def __call__(self, theta):
                return np.exp(-theta**2 / (2 * self.sigma**2))
        self.primary_beam = PrimaryBeam(self.calibration.get('primbeam'))

    def _init_optics(self, **keywords):
        class Optics(object):
            pass
        optics = Optics()
        optics.focal_length = self.calibration.get('optics')['focal length']
        self.optics = optics

    def _init_horns(self):
        self.horn = self.calibration.get('hornarray')

    def _init_synthetic_beam(self, synthbeam_dtype, synthbeam_fraction):
        class SyntheticBeam(object):
            pass
        sb = SyntheticBeam()
        sb.dtype = np.dtype(synthbeam_dtype)
        sb.fraction = synthbeam_fraction
        self.synthetic_beam = sb

    def __str__(self):
        state = [('synthbeam_fraction', self.synthetic_beam.fraction),
                 ('ngrids', self.detector.ngrids),
                 ('selection', _compress_mask(~self.detector.all.removed))]
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
        vertex = self.detector.vertex
        for v in vertex:
            a.add_patch(mp.Polygon(v, closed=True, fill=False, **keywords))
        if autoscale:
            mp.autoscale()
        mp.show()

    def get_noise(self, sampling, out=None):
        """
        Return a noisy timeline.

        """
        return Instrument.get_noise(
            self, sampling, sigma=self.detector.sigma,
            fknee=self.detector.fknee, fslope=self.detector.fslope, out=out)

    def get_detector_response_operator(self, sampling, tau=None):
        """
        Return the operator for the bolometer responses.

        """
        if tau is None:
            tau = self.detector.tau
        sampling_period = sampling.period
        shapein = len(self), len(sampling)
        if sampling_period == 0:
            return IdentityOperator(shapein)
        return ConvolutionTruncatedExponentialOperator(
            tau / sampling_period, shapein=shapein)

    def get_hwp_operator(self, sampling, scene):
        """
        Return the rotation matrix for the half-wave plate.

        """
        shape = (len(self), len(sampling))
        if scene.kind == 'I':
            return IdentityOperator(shapein=shape)
        if scene.kind == 'QU':
            return Rotation2dOperator(-4 * sampling.angle_hwp,
                                      degrees=True, shapein=shape + (2,))
        return Rotation3dOperator('X', -4 * sampling.angle_hwp,
                                  degrees=True, shapein=shape + (3,))

    def get_invntt_operator(self, sampling):
        """
        Return the inverse time-time noise correlation matrix as an Operator.

        """
        return Instrument.get_invntt_operator(
            self, sampling, sigma=self.detector.sigma,
            fknee=self.detector.fknee, fslope=self.detector.fslope,
            ncorr=self.detector.ncorr)

    def get_polarizer_operator(self, sampling, scene):
        """
        Return operator for the polarizer grid.

        """
        if scene.kind == 'I':
            return HomothetyOperator(1 / self.detector.ngrids)

        if self.detector.ngrids == 1:
            raise ValueError(
                'Polarized input not handled by a single detector grid.')

        nd = len(self)
        nt = len(sampling)
        grid = self.detector.quadrant // 4
        z = np.zeros(nd)
        data = np.array([z + 0.5, 0.5 - grid, z]).T[:, None, None, :]
        return ReshapeOperator((nd, nt, 1), (nd, nt)) * \
            DenseBlockDiagonalOperator(data, shapein=(nd, nt, 3))

    def get_projection_operator(self, sampling, scene, verbose=True):
        """
        Return the peak sampling operator.

        Parameters
        ----------
        sampling : QubicSampling
            The pointing information.
        scene : QubicScene
            The observed scene.
        verbose : bool, optional
            If true, display information about the memory allocation.

        """
        rotation = sampling.cartesian_galactic2instrument
        dtype = self.synthetic_beam.dtype
        fraction = self.synthetic_beam.fraction
        ndetectors = len(self)
        ntimes = len(sampling)
        nside = scene.nside

        theta, phi, vals = _peak_angles_fraction(self, scene, fraction)
        ncolmax = theta.shape[-1]
        thetaphi = _pack_vector(theta, phi)  # (ndetectors, ncolmax, 2)
        direction = Spherical2CartesianOperator('zenith,azimuth')(thetaphi)
        e_nf = direction[:, None, :, :]
        if nside > 8192:
            dtype_index = np.dtype(np.int64)
        else:
            dtype_index = np.dtype(np.int32)

        cls = {'I': FSRMatrix,
               'QU': FSRRotation2dMatrix,
               'IQU': FSRRotation3dMatrix}[scene.kind]
        ndims = len(scene.kind)
        s = cls((ndetectors * ntimes * ndims, 12 * nside**2 * ndims),
                ncolmax=ncolmax, dtype=dtype, dtype_index=dtype_index,
                verbose=verbose)

        index = s.data.index.reshape((ndetectors, ntimes, ncolmax))

        nthreads = openmp_num_threads()
        try:
            import mkl
            mkl.set_num_threads(1)
        except:
            pass

        def func_thread(i):
            # e_nf[i] shape: (1, ncolmax, 3)
            # e_ni shape: (ntimes, ncolmax, 3)
            e_ni = rotation.T(e_nf[i].swapaxes(0, 1)).swapaxes(0, 1)
            index[i] = Cartesian2HealpixOperator(nside)(e_ni)

        pool = Pool(nthreads)
        pool.map(func_thread, xrange(ndetectors))
        pool.close()
        pool.join()

        try:
            mkl.set_num_threads(nthreads)
        except:
            pass

        if scene.kind == 'I':
            value = s.data.value.reshape(ndetectors, ntimes, ncolmax)
            value[...] = vals[:, None, :]
            shapeout = (ndetectors, ntimes)
        else:
            func = 'pointing_matrix_rot{0}d_i{1}_m{2}'.format(
                ndims, dtype_index.itemsize, dtype.itemsize)
            try:
                getattr(flib.polarization, func)(
                    rotation.data.T, direction.T, s.data.ravel().view(np.int8),
                    vals.T)
            except AttributeError:
                raise TypeError(
                    'The projection matrix cannot be created with types: {0} a'
                    'nd {1}.'.format(dtype, dtype_index))
            if scene.kind == 'QU':
                shapeout = (ndetectors, ntimes, 2)
            else:
                shapeout = (ndetectors, ntimes, 3)

        return ProjectionOperator(s, shapeout=shapeout)


def _peak_angles(q, scene, kmax):
    """
    Return the spherical coordinates (theta,phi) of the beam peaks, in radians.

    Parameters
    ----------
    kmax : int, optional
        The diffraction order above which the peaks are ignored.
        For instance, a value of kmax=2 will model the synthetic beam by
        (2 * kmax + 1)**2 = 25 peaks and a value of kmax=0 will only sample
        the central peak.
    """
    ndetector = len(q)
    center = q.detector.center
    lmbda = c / scene.nu
    dx = q.horn.spacing
    detvec = np.vstack([-center[..., 0],
                        -center[..., 1],
                        np.zeros(ndetector) + q.optics.focal_length]).T
    detvec.T[...] /= np.sqrt(np.sum(detvec**2, axis=1))

    kx, ky = np.mgrid[-kmax:kmax+1, -kmax:kmax+1]
    nx = detvec[:, 0, np.newaxis] - lmbda * kx.ravel() / dx
    ny = detvec[:, 1, np.newaxis] - lmbda * ky.ravel() / dx
    local_dict = {'nx': nx, 'ny': ny}
    theta = ne.evaluate('arcsin(sqrt(nx**2 + ny**2))', local_dict=local_dict)
    phi = ne.evaluate('arctan2(ny, nx)', local_dict=local_dict)
    return theta, phi


def _peak_angles_fraction(q, scene, fraction):

    # there is no need to go beyond kmax=5
    theta, phi = _peak_angles(q, scene, kmax=5)
    index = _argsort(theta)
    theta = theta[index]
    phi = phi[index]
    val = q.primary_beam(theta)
    val[~np.isfinite(val)] = 0
    val /= np.sum(val, axis=-1)[:, None]
    cumval = np.cumsum(val, axis=-1)
    imaxs = cumval.shape[-1] - np.sum(cumval > fraction, axis=-1) + 1
    imax = max(imaxs)

    # slice initial arrays to discard the non-significant peaks
    theta = theta[:, :imax]
    phi = phi[:, :imax]
    val = val[:, :imax]

    # remove additional per-detector non-significant peaks
    # and remove potential NaN in theta, phi
    for idet, imax_ in enumerate(imaxs):
        val[idet, imax_:] = 0
        theta[idet, imax_:] = pi / 2 #XXX 0 leads to NaN
        phi[idet, imax_:] = 0
    val /= np.sum(val, axis=-1)[:, None]
    return theta, phi, val


def _argsort(a, axis=-1):
    i = list(np.ogrid[[slice(x) for x in a.shape]])
    i[axis] = a.argsort(axis)
    return i


def _pack_vector(*args):
    shape = np.broadcast(*args).shape
    out = np.empty(shape + (len(args),))
    for i, arg in enumerate(args):
        out[..., i] = arg
    return out
