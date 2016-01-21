# coding: utf-8
from __future__ import division

import healpy as hp
import numpy as np
from pyoperators import (
    BlockColumnOperator, BlockDiagonalOperator, BlockRowOperator,
    CompositionOperator, DiagonalOperator, I, IdentityOperator,
    MPIDistributionIdentityOperator, MPI, proxy_group, ReshapeOperator,
    rule_manager)
from pyoperators.utils.mpi import as_mpi
from pysimulators import Acquisition, FitsArray
from pysimulators.interfaces.healpy import (
    HealpixConvolutionGaussianOperator)
from .data import PATH
from .instrument import QubicInstrument
from .scene import QubicScene

__all__ = ['PlanckAcquisition',
           'QubicAcquisition',
           'QubicPlanckAcquisition']


class QubicAcquisition(Acquisition):
    """
    The QubicAcquisition class, which combines the instrument, sampling and
    scene models.

    """
    def __init__(self, instrument, sampling, scene=None, block=None,
                 calibration=None, detector_nep=4.7e-17, detector_fknee=0,
                 detector_fslope=1, detector_ncorr=10, detector_ngrids=1,
                 detector_tau=0.01, effective_duration=None,
                 filter_relative_bandwidth=0.25, photon_noise=True,
                 polarizer=True,  primary_beam=None, secondary_beam=None,
                 synthbeam_dtype=np.float32, synthbeam_fraction=0.99,
                 absolute=False, kind='IQU', nside=256, max_nbytes=None,
                 nprocs_instrument=None, nprocs_sampling=None, comm=None):
        """
        acq = QubicAcquisition(band, sampling,
                               [scene=|absolute=, kind=, nside=],
                               nprocs_instrument=, nprocs_sampling=, comm=)
        acq = QubicAcquisition(instrument, sampling,
                               [scene=|absolute=, kind=, nside=],
                               nprocs_instrument=, nprocs_sampling=, comm=)

        Parameters
        ----------
        band : int
            The module nominal frequency, in GHz.
        scene : QubicScene, optional
            The discretized observed scene (the sky).
        block : tuple of slices, optional
            Partition of the samplings.
        absolute : boolean, optional
            If true, the scene pixel values include the CMB background and the
            fluctuations in units of Kelvin, otherwise it only represents the
            fluctuations, in microKelvin.
        kind : 'I', 'QU' or 'IQU', optional
            The sky kind: 'I' for intensity-only, 'QU' for Q and U maps,
            and 'IQU' for intensity plus QU maps.
        nside : int, optional
            The Healpix scene's nside.
        instrument : QubicInstrument, optional
            The QubicInstrument instance.
        calibration : QubicCalibration, optional
            The calibration tree.
        detector_fknee : array-like, optional
            The detector 1/f knee frequency in Hertz.
        detector_fslope : array-like, optional
            The detector 1/f slope index.
        detector_ncorr : int, optional
            The detector 1/f correlation length.
        detector_nep : array-like, optional
            The detector NEP [W/sqrt(Hz)].
        detector_ngrids : 1 or 2, optional
            Number of detector grids.
        detector_tau : array-like, optional
            The detector time constants in seconds.
        effective_duration : float, optional
            If not None, the noise properties are rescaled so that this
            acquisition has an effective duration equal to the specified value,
            in years.
        filter_relative_bandwidth : float, optional
            The filter relative bandwidth Δν/ν.
        polarizer : boolean, optional
            If true, the polarizer grid is present in the optics setup.
        photon_noise : boolean, optional
            If true, the photon noise contribution is included.
        primary_beam : function f(theta [rad], phi [rad]), optional
            The primary beam transmission function.
        secondary_beam : function f(theta [rad], phi [rad]), optional
            The secondary beam transmission function.
        synthbeam_dtype : dtype, optional
            The data type for the synthetic beams (default: float32).
            It is the dtype used to store the values of the pointing matrix.
        synthbeam_fraction: float, optional
            The fraction of significant peaks retained for the computation
            of the synthetic beam.
        max_nbytes : int or None, optional
            Maximum number of bytes to be allocated for the acquisition's
            operator.
        nprocs_instrument : int, optional
            For a given sampling slice, number of procs dedicated to
            the instrument.
        nprocs_sampling : int, optional
            For a given detector slice, number of procs dedicated to
            the sampling.
        comm : mpi4py.MPI.Comm, optional
            The acquisition's MPI communicator. Note that it is transformed
            into a 2d cartesian communicator before being stored as the 'comm'
            attribute. The following relationship must hold:
                comm.size = nprocs_instrument * nprocs_sampling

        """
        if not isinstance(instrument, QubicInstrument):
            filter_nu = instrument * 1e9
            instrument = QubicInstrument(
                calibration=calibration, detector_fknee=detector_fknee,
                detector_fslope=detector_fslope, detector_ncorr=detector_ncorr,
                detector_nep=detector_nep, detector_ngrids=detector_ngrids,
                detector_tau=detector_tau, filter_nu=filter_nu,
                filter_relative_bandwidth=filter_relative_bandwidth,
                polarizer=polarizer, primary_beam=primary_beam,
                secondary_beam=secondary_beam, synthbeam_dtype=synthbeam_dtype,
                synthbeam_fraction=synthbeam_fraction)
        if scene is None:
            scene = QubicScene(absolute=absolute, kind=kind, nside=nside)
        else:
            attr = 'absolute', 'kind', 'nside'
            for a in attr:
                if locals()[a] != getattr(scene, a):
                    raise ValueError(
                        "The attribute '{}' is already specified in the input "
                        "scene.".format(a))
        Acquisition.__init__(
            self, instrument, sampling, scene, block=block,
            max_nbytes=max_nbytes, nprocs_instrument=nprocs_instrument,
            nprocs_sampling=nprocs_sampling, comm=comm)
        self.photon_noise = bool(photon_noise)
        self.effective_duration = effective_duration

    def get_coverage(self):
        """
        Return the acquisition scene coverage as given by H.T(1), normalized
        so that its integral over the sky is the number of detectors times
        the duration of the acquisition.

        """
        H = self.get_operator()
        out = H.T(np.ones((len(self.instrument), len(self.sampling))))
        if self.scene.kind != 'I':
            out = out[..., 0].copy()  # to avoid keeping QU in memory
        ndetectors = self.comm.allreduce(len(self.instrument))
        nsamplings = self.comm.allreduce(len(self.sampling))
        out *= ndetectors * nsamplings * self.sampling.period / np.sum(out)
        return out

    def get_hitmap(self, nside=None):
        """
        Return a healpy map whose values are the number of times a pointing
        hits the pixel.

        """
        if nside is None:
            nside = self.scene.nside
        ipixel = self.sampling.healpix(nside)
        npixel = 12 * nside**2
        hit = np.histogram(ipixel, bins=npixel, range=(0, npixel))[0]
        self.sampling.comm.Allreduce(MPI.IN_PLACE, as_mpi(hit), op=MPI.SUM)
        return hit

    def get_noise(self, out=None):
        out = self.instrument.get_noise(
            self.sampling, self.scene, photon_noise=self.photon_noise, out=out)
        if self.effective_duration is not None:
            out *= np.sqrt(len(self.sampling) * self.sampling.period /
                           (self.effective_duration * 31557600))
        return out
    get_noise.__doc__ = Acquisition.get_noise.__doc__

    def get_aperture_integration_operator(self):
        """
        Integrate flux density in the telescope aperture.
        Convert signal from W / m^2 / Hz into W / Hz.

        """
        return self.instrument.get_aperture_integration_operator()

    def get_convolution_peak_operator(self, **keywords):
        """
        Return an operator that convolves the Healpix sky by the gaussian
        kernel that, if used in conjonction with the peak sampling operator,
        best approximates the synthetic beam.

        """
        return self.instrument.get_convolution_peak_operator(**keywords)

    def get_detector_integration_operator(self):
        """
        Integrate flux density in detector solid angles.
        Convert W / sr into W.

        """
        return self.instrument.get_detector_integration_operator()

    def get_detector_response_operator(self):
        """
        Return the operator for the bolometer responses.

        """
        return BlockDiagonalOperator(
            [self.instrument.get_detector_response_operator(self.sampling[b])
             for b in self.block], axisin=1)

    def get_distribution_operator(self):
        """
        Return the MPI distribution operator.

        """
        return MPIDistributionIdentityOperator(self.comm)

    def get_filter_operator(self):
        """
        Return the filter operator.
        Convert units from W/Hz to W.

        """
        return self.instrument.get_filter_operator()

    def get_hwp_operator(self):
        """
        Return the operator for the bolometer responses.

        """
        return BlockDiagonalOperator(
            [self.instrument.get_hwp_operator(self.sampling[b], self.scene)
             for b in self.block], axisin=1)

    def get_invntt_operator(self):
        sigma_detector = self.instrument.detector.nep / \
            np.sqrt(2*self.sampling.period)
        if self.photon_noise:
            sigma_photon = self.instrument._get_noise_photon_nep(self.scene) /\
                           np.sqrt(2 * self.sampling.period)
        else:
            sigma_photon = 0
        out = DiagonalOperator(
            1 / (sigma_detector**2 + sigma_photon**2),
            broadcast='rightward',
            shapein=(len(self.instrument), len(self.sampling)))
        if self.effective_duration is not None:
            out /= (len(self.sampling) * self.sampling.period /
                    (self.effective_duration * 31557600))
        return out
    get_invntt_operator.__doc__ = Acquisition.get_invntt_operator.__doc__

    def get_unit_conversion_operator(self):
        """
        Convert sky temperature into W / m^2 / Hz.

        If the scene has been initialised with the 'absolute' keyword, the
        scene is assumed to include the CMB background and the fluctuations
        (in Kelvin) and the operator follows the non-linear Planck law.
        Otherwise, the scene only includes the fluctuations (in microKelvin)
        and the operator is linear (i.e. the output also corresponds to power
        fluctuations).

        """
        nu = self.instrument.filter.nu
        return self.scene.get_unit_conversion_operator(nu)

    def get_operator(self):
        """
        Return the operator of the acquisition. Note that the operator is only
        linear if the scene temperature is differential (absolute=False).

        """
        distribution = self.get_distribution_operator()
        temp = self.get_unit_conversion_operator()
        aperture = self.get_aperture_integration_operator()
        filter = self.get_filter_operator()
        projection = self.get_projection_operator()
        hwp = self.get_hwp_operator()
        polarizer = self.get_polarizer_operator()
        integ = self.get_detector_integration_operator()
        trans_inst = self.instrument.get_transmission_operator()
        trans_atm = self.scene.atmosphere.transmission
        response = self.get_detector_response_operator()

        with rule_manager(inplace=True):
            H = CompositionOperator([
                response, trans_inst, integ, polarizer, hwp * projection,
                filter, aperture, trans_atm, temp, distribution])
        if self.scene == 'QU':
            H = self.get_subtract_grid_operator()(H)
        return H

    def get_polarizer_operator(self):
        """
        Return operator for the polarizer grid.

        """
        return BlockDiagonalOperator(
            [self.instrument.get_polarizer_operator(
                self.sampling[b], self.scene) for b in self.block], axisin=1)

    def get_projection_operator(self, verbose=True):
        """
        Return the projection operator for the peak sampling.
        Convert units from W to W/sr.

        Parameters
        ----------
        verbose : bool, optional
            If true, display information about the memory allocation.

        """
        f = self.instrument.get_projection_operator
        if len(self.block) == 1:
            return BlockColumnOperator(
                [f(self.sampling[b], self.scene, verbose=verbose)
                 for b in self.block], axisout=1)
        #XXX HACK
        def callback(i):
            p = f(self.sampling[self.block[i]], self.scene, verbose=False)
            return p
        shapeouts = [(len(self.instrument), s.stop-s.start) +
                      self.scene.shape[1:] for s in self.block]
        proxies = proxy_group(len(self.block), callback, shapeouts=shapeouts)
        return BlockColumnOperator(proxies, axisout=1)

    def get_add_grids_operator(self):
        """ Return operator to add signal from detector pairs. """
        nd = len(self.instrument)
        if nd % 2 != 0:
            raise ValueError('Odd number of detectors.')
        partitionin = 2 * (len(self.instrument) // 2,)
        return BlockRowOperator([I, I], axisin=0, partitionin=partitionin)

    def get_subtract_grids_operator(self):
        """ Return operator to subtract signal from detector pairs. """
        nd = len(self.instrument)
        if nd % 2 != 0:
            raise ValueError('Odd number of detectors.')
        partitionin = 2 * (len(self.instrument) // 2,)
        return BlockRowOperator([I, -I], axisin=0, partitionin=partitionin)

    def get_observation(self, map, noiseless=False, convolution=False):
        """
        tod = map2tod(acquisition, map)
        tod, convolved_map = map2tod(acquisition, map, convolution=True)

        Parameters
        ----------
        map : I, QU or IQU maps
            Temperature, QU or IQU maps of shapes npix, (npix, 2), (npix, 3)
            with npix = 12 * nside**2
        noiseless : boolean, optional
            If True, no noise is added to the observation.
        convolution : boolean, optional
            Set to True to convolve the input map by a gaussian and return it.

        Returns
        -------
        tod : array
            The Time-Ordered-Data of shape (ndetectors, ntimes).
        convolved_map : array, optional
            The convolved map, if the convolution keyword is set.

        """
        if convolution:
            convolution = self.get_convolution_peak_operator()
            map = convolution(map)

        H = self.get_operator()
        tod = H(map)

        if not noiseless:
            tod += self.get_noise()

        if convolution:
            return tod, map

        return tod


class PlanckAcquisition(object):
    def __init__(self, band, scene, true_sky=None, factor=1, fwhm=0):
        """
        Parameters
        ----------
        band : int
            The band 150 or 220.
        scene : Scene
            The acquisition scene.
        true_sky : array of shape (npixel,) or (npixel, 3)
            The true CMB sky (temperature or polarized). The Planck observation
            will be this true sky plus a random independent gaussian noise
            realization.
        factor : 1 or 3 floats, optional
            The factor by which the Planck standard deviation is multiplied.
        fwhm : float, optional, !not used!
            The fwhm of the Gaussian used to smooth the map [radians].

        """
        if band not in (150, 220):
            raise ValueError("Invalid band '{}'.".format(band))
        if true_sky is None:
            raise ValueError('The Planck Q & U maps are not released yet.')
        if scene.kind == 'IQU' and true_sky.shape[-1] != 3:
            raise TypeError('The Planck sky shape is not (npix, 3).')
        true_sky = np.array(hp.ud_grade(true_sky.T, nside_out=scene.nside),
                            copy=False).T
        if scene.kind == 'IQU' and true_sky.shape[-1] != 3:
            raise TypeError('The Planck sky shape is not (npix, 3).')
        self.scene = scene
        self.fwhm = fwhm
        self._true_sky = true_sky
        if band == 150:
            filename = 'Variance_Planck143GHz_Kcmb2_ns256.fits'
        else:
            filename = 'Variance_Planck217GHz_Kcmb2_ns256.fits'
        sigma = 1e6 * factor * np.sqrt(FitsArray(PATH + filename))
        if scene.kind == 'I':
            sigma = sigma[:, 0]
        elif scene.kind == 'QU':
            sigma = sigma[:, :2]
        if self.scene.nside != 256:
            sigma = np.array(hp.ud_grade(sigma.T, self.scene.nside, power=2),
                             copy=False).T
        self.sigma = sigma

    _SIMULATED_PLANCK_SEED = 0

    def get_operator(self):
        return IdentityOperator(shapein=self.scene.shape)

    def get_invntt_operator(self):
        return DiagonalOperator(1 / self.sigma**2, broadcast='leftward',
                                shapein=self.scene.shape)

    def get_noise(self):
        state = np.random.get_state()
        np.random.seed(self._SIMULATED_PLANCK_SEED)
        out = np.random.standard_normal(self._true_sky.shape) * self.sigma
        np.random.set_state(state)
        return out

    def get_observation(self, noiseless=False):
        obs = self._true_sky
        if not noiseless:
            obs = obs + self.get_noise()
        return obs
        #XXX neglecting convolution effects...
        HealpixConvolutionGaussianOperator(fwhm=self.fwhm)(obs, obs)
        return obs


class QubicPlanckAcquisition(object):
    """
    The QubicPlanckAcquisition class, which combines the Qubic and Planck
    acquisitions.

    """
    def __init__(self, qubic, planck):
        """
        acq = QubicPlanckAcquisition(qubic_acquisition, planck_acquisition)

        Parameters
        ----------
        qubic_acquisition : QubicAcquisition
            The QUBIC acquisition.
        planck_acquisition : PlanckAcquisition
            The Planck acquisition.

        """
        if not isinstance(qubic, QubicAcquisition):
            raise TypeError('The first argument is not a QubicAcquisition.')
        if not isinstance(planck, PlanckAcquisition):
            raise TypeError('The second argument is not a PlanckAcquisition.')
        if qubic.scene is not planck.scene:
            raise ValueError('The Qubic and Planck scenes are different.')
        self.qubic = qubic
        self.planck = planck

    def get_noise(self):
        """
        Return a noise realization compatible with the fused noise
        covariance matrix.

        """
        noise_qubic = self.qubic.get_noise()
        noise_planck = self.planck.get_noise()
        return np.r_[noise_qubic.ravel(), noise_planck.ravel()]

    def get_operator(self):
        """
        Return the fused observation as an operator.

        """
        H_qubic = self.qubic.get_operator()
        R_qubic = ReshapeOperator(H_qubic.shapeout, H_qubic.shape[0])
        H_planck = self.planck.get_operator()
        R_planck = ReshapeOperator(H_planck.shapeout, H_planck.shape[0])
        return BlockColumnOperator(
            [R_qubic(H_qubic), R_planck(H_planck)], axisout=0)

    def get_invntt_operator(self):
        """
        Return the inverse covariance matrix of the fused observation
        as an operator.

        """
        invntt_qubic = self.qubic.get_invntt_operator()
        R_qubic = ReshapeOperator(invntt_qubic.shapeout, invntt_qubic.shape[0])
        invntt_planck = self.planck.get_invntt_operator()
        R_planck = ReshapeOperator(invntt_planck.shapeout,
                                   invntt_planck.shape[0])
        return BlockDiagonalOperator(
            [R_qubic(invntt_qubic(R_qubic.T)),
             R_planck(invntt_planck(R_planck.T))], axisout=0)

    def get_observation(self, noiseless=False, convolution=False):
        """
        Return the fused observation.

        Parameters
        ----------
        noiseless : boolean, optional
            If set, return a noiseless observation
        """
        obs_qubic_ = self.qubic.get_observation(
            self.planck._true_sky, noiseless=noiseless,
            convolution=convolution)
        obs_qubic = obs_qubic_[0] if convolution else obs_qubic_
        obs_planck = self.planck.get_observation(noiseless=noiseless)
        obs = np.r_[obs_qubic.ravel(), obs_planck.ravel()]
        if convolution:
            return obs, obs_qubic_[1]
        return obs
