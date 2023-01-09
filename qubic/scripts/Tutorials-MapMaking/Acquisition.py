# QUBIC stuff
import qubic
from qubicpack.utilities import Qubic_DataDir
from qubic.data import PATH
from qubic.io import read_map
from qubic.scene import QubicScene
from qubic.samplings import create_random_pointings, get_pointing

# General stuff
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pysm3
import warnings
warnings.filterwarnings("ignore")
import pysm3.units as u
from importlib import reload
from pysm3 import utils
# FG-Buster packages
import component_model as c
import mixing_matrix as mm
import pickle
from scipy.optimize import minimize

# PyOperators stuff
from pysimulators import *
from pyoperators import *
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator


__all__ = ['QubicAcquisition',
           'PlanckAcquisition',
           'QubicPlanckAcquisition',
           'QubicPolyAcquisition',
           'QubicMultiBandAcquisition',
           'QubicPlanckMultiBandAcquisition',
           'QubicAcquisitionTwoBands',
           'QubicMultiBandAcquisitionTwoBands']

def create_array(name, nus, nside):

    if name == 'noise':
        shape=(2, 12*nside**2, 3)
    else:
        shape=len(nus)
    pkl_file = open('AllDataSet_Components_MapMaking.pkl', 'rb')
    dataset = pickle.load(pkl_file)

    myarray=np.zeros(shape)

    for ii, i in enumerate(nus):
        myarray[ii] = dataset[name+str(i)]

    return myarray

def get_preconditioner(cov):
    if cov is not None:
        cov_inv = 1 / cov
        cov_inv[np.isinf(cov_inv)] = 0.
        preconditioner = DiagonalOperator(cov_inv, broadcast='rightward')
    else:
        preconditioner = None
    return preconditioner

def arcmin2rad(arcmin):
    return arcmin * 0.000290888

def rad2arcmin(rad):
    return rad / 0.000290888

def compute_fwhm_to_convolve(allres, target):
    s = np.sqrt(target**2 - allres**2)
    if s == np.nan:
        s = 0
    return s

def mychi2(beta, obj, Hqubic, data, solution, nsamples):

    H_for_beta = obj.get_operator(beta, convolution=False, H_qubic=Hqubic)
    fakedata = H_for_beta(solution)
    fakedata_norm = obj.normalize(fakedata, nsamples)
    print(beta)
    return np.sum((fakedata_norm - data)**2)

def fit_beta(tod, nsamples, obj, H_qubic, outputs):

    tod_norm = obj.normalize(tod, nsamples)
    r = minimize(mychi2, method='TNC', tol=1e-15, x0=np.array([1.]), args=(obj, H_qubic, tod_norm, outputs, nsamples))

    return r.x


class QubicAcquisition(Acquisition):
    """
    The QubicAcquisition class, which combines the instrument, sampling and
    scene models.
    """
    def __init__(self, instrument, sampling, scene, d):
        """
        acq = QubicAcquisition(instrument, sampling, scene, d)
        Parameters
        ----------
        instrument : QubicInstrument, optional
            The QubicInstrument instance.
        sampling : pointing
            Pointing obtained with get_pointing().
        scene : QubicScene, optional
            The discretized observed scene (the sky).
        d : dictionary with lot of parameters:
            block : tuple of slices, optional
                Partition of the samplings.
            effective_duration : float, optional
                If not None, the noise properties are rescaled so that this
                acquisition has an effective duration equal to the specified value,
                in years.
            photon_noise : boolean, optional
                If true, the photon noise contribution is included.
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
            psd : array-like, optional
                The one-sided or two-sided power spectrum density
                [signal unit/sqrt Hz].
            bandwidth : float, optional
                The PSD frequency increment [Hz].
            twosided : boolean, optional
                Whether or not the input psd is one-sided (only positive
                frequencies) or two-sided (positive and negative frequencies).
            sigma : float
                Standard deviation of the white noise component.
        """
        block = d['block']
        effective_duration = d['effective_duration']
        photon_noise = d['photon_noise']
        max_nbytes = d['max_nbytes']
        nprocs_instrument = d['nprocs_instrument']
        nprocs_sampling = d['nprocs_sampling']
        comm = d['comm']
        psd = d['psd']
        bandwidth = d['bandwidth']
        twosided = d['twosided']
        sigma = d['sigma']

        Acquisition.__init__(
            self, instrument, sampling, scene, block=block,
            max_nbytes=max_nbytes, nprocs_instrument=nprocs_instrument,
            nprocs_sampling=nprocs_sampling, comm=comm)
        self.photon_noise = bool(photon_noise)
        self.effective_duration = effective_duration
        self.bandwidth = bandwidth
        self.psd = psd
        self.twosided = twosided
        self.sigma = sigma
        self.forced_sigma = None

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
        npixel = 12 * nside ** 2
        hit = np.histogram(ipixel, bins=npixel, range=(0, npixel))[0]
        self.sampling.comm.Allreduce(MPI.IN_PLACE, as_mpi(hit), op=MPI.SUM)
        return hit

    def get_noise(self, out=None):
        out = self.instrument.get_noise(
            self.sampling, self.scene, photon_noise=self.photon_noise, out=out)
        if self.effective_duration is not None:
            nsamplings = self.comm.allreduce(len(self.sampling))
            out *= np.sqrt(nsamplings * self.sampling.period /
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

    def get_diag_invntt_operator(self):

        print('Use diagonal noise covariance matrix')

        sigma_detector = self.instrument.detector.nep / np.sqrt(2 * self.sampling.period)
        if self.photon_noise:
            sigma_photon = self.instrument._get_noise_photon_nep(self.scene) / np.sqrt(2 * self.sampling.period)
        else:
            sigma_photon = 0

        out = DiagonalOperator(1 / (sigma_detector ** 2 + sigma_photon ** 2), broadcast='rightward',
                               shapein=(len(self.instrument), len(self.sampling)))
        if self.effective_duration is not None:
            nsamplings = self.comm.allreduce(len(self.sampling))
            out /= (nsamplings * self.sampling.period / (self.effective_duration * 31557600))
        return out

    def get_invntt_operator(self):

        """
        Return the inverse time-time noise correlation matrix as an Operator.

        The input Power Spectrum Density can either be fully specified by using
        the 'bandwidth' and 'psd' keywords, or by providing the parameters of
        the gaussian distribution:
        psd = sigma**2 * (1 + (fknee/f)**fslope) / B
        where B is the sampling bandwidth equal to sampling_frequency / N.

        Parameters
        ----------
        sampling : Sampling
            The temporal sampling.
        psd : array-like, optional
            The one-sided or two-sided power spectrum density
            [signal unit/sqrt Hz].
        bandwidth : float, optional
            The PSD frequency increment [Hz].
        twosided : boolean, optional
            Whether or not the input psd is one-sided (only positive
            frequencies) or two-sided (positive and negative frequencies).
        sigma : float
            Standard deviation of the white noise component.
        sampling_frequency : float
            The sampling frequency [Hz].
        fftw_flag : string, optional
            The flags FFTW_ESTIMATE, FFTW_MEASURE, FFTW_PATIENT and
            FFTW_EXHAUSTIVE can be used to describe the increasing amount of
            effort spent during the planning stage to create the fastest
            possible transform. Usually, FFTW_MEASURE is a good compromise
            and is the default.
        nthreads : int, optional
            Tells how many threads to use when invoking FFTW or MKL. Default is
            the number of cores.

        """

        fftw_flag = 'FFTW_MEASURE'
        nthreads = None

        #if self.bandwidth is None or self.psd is None:
        if self.bandwidth is None and self.psd is not None or self.bandwidth is not None and self.psd is None:
            raise ValueError('The bandwidth or the PSD is not specified.')

        # Get sigma in Watt
        if self.instrument.detector.nep is not None:
            self.sigma = self.instrument.detector.nep / np.sqrt(2 * self.sampling.period)

            if self.photon_noise:
                sigma_photon = self.instrument._get_noise_photon_nep(self.scene) / np.sqrt(2 * self.sampling.period)
                self.sigma = np.sqrt(self.sigma ** 2 + sigma_photon ** 2)
            else:
                pass
                # sigma_photon = 0

        if self.bandwidth is None and self.psd is None and self.sigma is None:
            raise ValueError('The noise model is not specified.')


        print('In acquisition.py: self.forced_sigma={}'.format(self.forced_sigma))
        print('and self.sigma is:{}'.format(self.sigma))
        if self.forced_sigma is None:
            print('Using theoretical TES noises')
        else:
            print('Using self.forced_sigma as TES noises')
            self.sigma = self.forced_sigma.copy()

        shapein = (len(self.instrument), len(self.sampling))

        if self.bandwidth is None and self.instrument.detector.fknee == 0:
            print('diagonal case')

            out = DiagonalOperator(1 / self.sigma ** 2, broadcast='rightward',
                                   shapein=(len(self.instrument), len(self.sampling)))
            print(out.shape)
            print(out)

            if self.effective_duration is not None:
                nsamplings = self.comm.allreduce(len(self.sampling))
                out /= (nsamplings * self.sampling.period / (self.effective_duration * 31557600))
            return out

        sampling_frequency = 1 / self.sampling.period

        nsamples_max = len(self.sampling)
        fftsize = 2
        while fftsize < nsamples_max:
            fftsize *= 2

        new_bandwidth = sampling_frequency / fftsize
        if self.bandwidth is not None and self.psd is not None:
            if self.twosided:
                self.psd = _fold_psd(self.psd)
            f = np.arange(fftsize // 2 + 1, dtype=float) * new_bandwidth
            p = _unfold_psd(_logloginterp_psd(f, self.bandwidth, self.psd))
        else:
            p = _gaussian_psd_1f(fftsize, sampling_frequency, self.sigma, self.instrument.detector.fknee,
                                 self.instrument.detector.fslope, twosided=True)
        p[..., 0] = p[..., 1]
        invntt = _psd2invntt(p, new_bandwidth, self.instrument.detector.ncorr, fftw_flag=fftw_flag)

        print('non diagonal case')
        if self.effective_duration is not None:
            nsamplings = self.comm.allreduce(len(self.sampling))
            invntt /= (nsamplings * self.sampling.period / (self.effective_duration * 31557600))

        return SymmetricBandToeplitzOperator(shapein, invntt, fftw_flag=fftw_flag, nthreads=nthreads)

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

        # XXX HACK
        def callback(i):
            p = f(self.sampling[self.block[i]], self.scene, verbose=False)
            return p

        shapeouts = [(len(self.instrument), s.stop - s.start) +
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

    def get_observation(self, map, convolution=True, noiseless=False):
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

    def get_preconditioner(self, cov):
        if cov is not None:
            cov_inv = 1 / cov
            cov_inv[np.isinf(cov_inv)] = 0.
            preconditioner = DiagonalOperator(cov_inv, broadcast='rightward')
        else:
            preconditioner = None
        return preconditioner
class PlanckAcquisition:

    def __init__(self, band, scene, true_sky=None, factor=1, fwhm=0, mask=None, convolution_operator=None):
        if band not in (30, 44, 70, 143, 217, 353):
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
        self.band = band
        self._true_sky = true_sky
        if mask is not None:
            self.mask = mask
        else:
            self.mask = np.ones(scene.npixel, dtype=np.bool)


        if band == 30:
            filename = 'Variance_Planck30GHz_Kcmb2_ns256.fits'
            var=np.zeros((12*self.scene.nside**2, 3))
            for i in range(3):
                var[:, i] = hp.ud_grade(hp.fitsfunc.read_map(filename, field=i), self.scene.nside)
            sigma = 1e5 * factor * np.sqrt(var)

        elif band == 44:
            filename = 'Variance_Planck44GHz_Kcmb2_ns256.fits'
            var=np.zeros((12*self.scene.nside**2, 3))
            for i in range(3):
                var[:, i] = hp.ud_grade(hp.fitsfunc.read_map(filename, field=i), self.scene.nside)
            sigma = 1e5 * factor * np.sqrt(var)
        elif band == 70:
            filename = 'Variance_Planck70GHz_Kcmb2_ns256.fits'
            var=np.zeros((12*self.scene.nside**2, 3))
            for i in range(3):
                var[:, i] = hp.ud_grade(hp.fitsfunc.read_map(filename, field=i), self.scene.nside)
            sigma = 1e5 * factor * np.sqrt(var)
        elif band == 143:
            filename = 'Variance_Planck143GHz_Kcmb2_ns256.fits'
            sigma = 1e6 * factor * np.sqrt(FitsArray(PATH + filename))
        elif band == 217:
            filename = 'Variance_Planck217GHz_Kcmb2_ns256.fits'
            sigma = 1e6 * factor * np.sqrt(FitsArray(PATH + filename))
        else:
            filename = 'Variance_Planck353GHz_Kcmb2_ns256.fits'
            var=np.zeros((12*self.scene.nside**2, 3))
            for i in range(3):
                var[:, i] = hp.ud_grade(hp.fitsfunc.read_map(filename, field=i), self.scene.nside)
            sigma = 1e5 * factor * np.sqrt(var)

        #if d['nside']!=256:
        #    sig = np.zeros((12*d['nside']**2, 3))
        #    for i in range(3):
        #        sig[:, i]=hp.ud_grade(sigma[:, i], d['nside'])




        if scene.kind == 'I':
            sigma = sigma[:, 0]
        elif scene.kind == 'QU':
            sigma = sigma[:, :2]
        if self.scene.nside != 256:
            sigma = np.array(hp.ud_grade(sigma.T, self.scene.nside, power=2),
                             copy=False).T
        self.sigma = sigma
        if convolution_operator is None:
            self.C = IdentityOperator()
        else:
            self.C = convolution_operator


    def get_operator(self):
        return DiagonalOperator(self.mask.astype(np.int), broadcast='rightward',
                                shapein=self.scene.shape)

    def get_invntt_operator(self):
        return DiagonalOperator(1 / self.sigma ** 2, broadcast='leftward',
                                shapein=self.scene.shape)

    def get_noise(self):
        state = np.random.get_state()
        #np.random.seed(self._SIMULATED_PLANCK_SEED)
        np.random.seed(None)
        out = np.random.standard_normal(self._true_sky.shape) * self.sigma
        np.random.set_state(state)
        return out

    def get_observation(self, noiseless=False):
        obs = self._true_sky
        if not noiseless:
            obs = obs + self.C(self.get_noise())
        if len(self.scene.shape) == 2:
            for i in range(self.scene.shape[1]):
                obs[~(self.mask), i] = 0.
        else:
            obs[~(self.mask)] = 0.
        return obs
        # XXX neglecting convolution effects...
        HealpixConvolutionGaussianOperator(fwhm=self.fwhm)(obs, obs)
        return obs
class QubicPlanckAcquisition:

    def __init__(self, qubic, planck):

        self.qubic = qubic
        self.planck = planck
    def get_noise(self):

        """
        Return a noise realization compatible with the fused noise
        covariance matrix.
        """

        n = self.qubic.get_noise().ravel()
        n = np.r_[n, self.planck.get_noise().ravel()]

        return n
    def get_invntt_operator(self):
        """
        Return the inverse covariance matrix of the fused observation
        as an operator.
        """


        Operator = []
        invntt_qubic = self.qubic.get_invntt_operator()
        R_qubic = ReshapeOperator(invntt_qubic.shapeout, invntt_qubic.shape[0])
        Operator.append(R_qubic(invntt_qubic(R_qubic.T)))

        invntt_planck = self.planck.get_invntt_operator()
        R_planck = ReshapeOperator(invntt_planck.shapeout, invntt_planck.shape[0])
        Operator.append(R_planck(invntt_planck(R_planck.T)))

        return BlockDiagonalOperator(Operator, axisout=0)
    def get_operator(self, convolution=False):
        """
        Return the fused observation as an operator.
        """

        Operator = []

        H_qubic = self.qubic.get_operator(convolution=convolution)
        R_qubic = ReshapeOperator(H_qubic.shapeout, H_qubic.shape[0])
        Operator.append(R_qubic(H_qubic))

        H_planck = self.planck.get_operator()
        R_planck = ReshapeOperator(H_planck.shapeout, H_planck.shape[0])
        Operator.append(R_planck(H_planck))
        return BlockColumnOperator(Operator, axisout=0)
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
class QubicPolyAcquisition:
    def __init__(self, multiinstrument, sampling, scene, d):
        """
        acq = QubicPolyAcquisition(QubicMultibandInstrument, sampling, scene)
        Parameters
        ----------
        multiinstrument : QubicMultibandInstrument
            The sub-frequencies are set there
        sampling :
            QubicSampling instance
        scene :
            QubicScene instance
        For other parameters see documentation for the QubicAcquisition class
        """

        weights = d['weights']

        self.warnings(d)

        if d['MultiBand'] and d['nf_sub']>1:
            self.subacqs = [QubicAcquisition(multiinstrument[i],
                                             sampling, scene, d)
                            for i in range(len(multiinstrument))]
        else:
            raise ValueError('If you do not use a multiband instrument,'
                             'you should use the QubicAcquisition class'
                             'which is done for the monochromatic case.')
        for a in self[1:]:
            a.comm = self[0].comm
        self.scene = scene
        self.d = d
        if weights is None:
            self.weights = np.ones(len(self))  # / len(self)
        else:
            self.weights = weights


    def __getitem__(self, i):
        return self.subacqs[i]

    def __len__(self):
        return len(self.subacqs)

    def warnings(self, d):

        """
            This method prevent to you that beam is not a good
            approximation in the 220 GHz band.
            Also can be used to add new warnings when acquisition is created in
            specific configuration.
        """

        if d['filter_nu'] == 220e9:
            if d['beam_shape'] == 'gaussian':
                warnings.warn('The nu dependency of the gausian beam FWHM '
                        'is not a good approximation in the 220 GHz band.')
            elif d['beam_shape'] == 'fitted_beam':
                warnings.warn('Beam and solid angle frequency dependence implementation '
                        'in the 220 GHz band for the fitted beam does not correctly describe '
                        'the true behavior')


    def get_coverage(self):
        """
        Return an array of monochromatic coverage maps, one for each of subacquisitions
        """
        if len(self) == 1:
            return self.subacqs[0].get_coverage()
        return np.array([self.subacqs[i].get_coverage() for i in range(len(self))])

    def get_coverage_mask(self, coverages, covlim=0.2):
        """
        Return a healpix boolean map with True on the pixels where ALL the
            subcoverages are above covlim * subcoverage.max()
        """
        if coverages.shape[0] != len(self):
            raise ValueError('Use QubicMultibandAcquisition.get_coverage method to create input')
        if len(self) == 1:
            cov = coverages
            return cov > covlim * np.max(cov)
        observed = [(coverages[i] > covlim * np.max(coverages[i])) for i in range(len(self))]
        obs = reduce(np.logical_and, tuple(observed[i] for i in range(len(self))))
        return obs

    def _get_average_instrument_acq(self):
        """
        Create and return a QubicAcquisition instance of a monochromatic
            instrument with frequency correspondent to the mean of the
            frequency range.
        """
        if len(self) == 1:
            return self[0]
        q0 = self[0].instrument
        nu_min = q0.filter.nu
        nu_max = self[-1].instrument.filter.nu
        nep = q0.detector.nep
        fknee = q0.detector.fknee
        fslope = q0.detector.fslope

        d1 = self.d.copy()
        d1['filter_nu'] = (nu_max + nu_min) / 2.
        d1['filter_relative_bandwidth'] = (nu_max - nu_min) / ((nu_max + nu_min) / 2.)
        d1['detector_nep'] = nep
        d1['detector_fknee'] = fknee
        d1['detector_fslope'] = fslope

        q = qubic.QubicInstrument(d1, FRBW=self[0].instrument.FRBW)
        q.detector = self[0].instrument.detector
        s_ = self[0].sampling
        nsamplings = self[0].comm.allreduce(len(s_))

        d1['random_pointing'] = True
        d1['sweeping_pointing'] = False
        d1['repeat_pointing'] = False
        d1['RA_center'] = 0.
        d1['DEC_center'] = 0.
        d1['npointings'] = nsamplings
        d1['dtheta'] = 10.
        d1['period'] = s_.period

        s = get_pointing(d1)
        # s = create_random_pointings([0., 0.], nsamplings, 10., period=s_.period)
        a = QubicAcquisition(q, s, self[0].scene, d1)
        return a

    def get_noise(self):
        a = self._get_average_instrument_acq()
        return a.get_noise()

    def _get_array_of_operators(self):
        return [a.get_operator() * w for a, w in zip(self, self.weights)]

    def get_operator_to_make_TOD(self):
        """
        Return a BlockRowOperator of subacquisition operators
        In polychromatic mode it is only applied to produce the TOD
        To reconstruct maps one should use the get_operator function
        """
        if len(self) == 1:
            return self.get_operator()
        op = self._get_array_of_operators()
        return BlockRowOperator(op, new_axisin=0)

    def get_operator(self):
        """
        Return an sum of operators for subacquisitions
        """
        if len(self) == 1:
            return self[0].get_operator()
        op = np.array(self._get_array_of_operators())
        return np.sum(op, axis=0)

    def get_invntt_operator(self):
        """
        Return the inverse noise covariance matrix as operator
        """
        return self[0].get_invntt_operator()
class QubicMultibandAcquisition(QubicPolyAcquisition):
    def __init__(self, multiinstrument, sampling, scene, d, nus):
        '''
        Parameters:
        -----------
        nus : array
            edge frequencies for reconstructed subbands, for example:
            [140, 150, 160] means two bands: one from 140 to 150 GHz and
            one from 150 to 160 GHz
        Note, that number of subbands is not equal to len(self)
        Within each subband there are multiple frequencies
        Documentation for other parameters see in QubicPolyAcquisition
        '''
        QubicPolyAcquisition.__init__(self, multiinstrument, sampling, scene, d)

        if len(nus) > 1:
            self.bands = np.array([[nus[i], nus[i + 1]] for i in range(len(nus) - 1)])
        else:
            raise ValueError('The QubicMultibandAcquisition class is designed to '
                             'work with multiple reconstructed subbands. '
                             'If you reconstruct only one subband, you can use '
                             'the QubicPolyAcquisition class')
        self.nus = np.array([q.filter.nu / 1e9 for q in multiinstrument])

    def get_operator(self):
        op = np.array(self._get_array_of_operators())
        op_sum = []
        for band in self.bands:
            op_sum.append(op[(self.nus > band[0]) * (self.nus < band[1])].sum(axis=0))
        return BlockRowOperator(op_sum, new_axisin=0)
    def convolved_maps(self, m):
        _maps_convolved = np.zeros(m.shape)  # array of sky maps, each convolved with its own gaussian
        for i in range(len(self)):
            C = self[i].get_convolution_peak_operator()
            _maps_convolved[i] = C(m[i])
        return _maps_convolved

    def get_observation(self, m, convolution=True, noiseless=False):
        '''
        Return TOD for polychromatic synthesised beam,
        just the same way as QubicPolyAcquisition.get_observation does
        Parameters
        ----------
        m : np.array((N, npix, 3)) if self.scene.kind == 'IQU', else np.array((npix))
            where N = len(self) if convolution == True or
                  N = len(self.bands) if convolution == False
            Helpix map of CMB for all the frequencies
        convolution : boolean, optional [default and recommended = True]
            - if True, convolve the input map with gaussian kernel
            with width specific for each subfrequency and
            return TOD, convolved map,
            (for example, we use 4 monochromatic frequencies and divide them
                to 2 subbands)
            where TOD = [H1, H2, H3, H4] * [m_conv1, m_conv2, m_conv3, m_conv4].T
            and convolved map = [average([m_conv1, m_conv2]), average([m_conv3, m_conv4])]
            - if False, the input map is considered as already convolved
            and the return is just TOD, which is equal to
            [sum(H1, H2, H3, ...)] * input_map
        noiseless : boolean, optional [default=False]
            if False, add noise to the TOD due to the model
        '''

        if self.scene.kind != 'I':
            shape = (len(self), m.shape[1], m.shape[2])
        else:
            shape = m.shape

        if convolution:
            _maps_convolved = np.zeros(shape)  # array of sky maps, each convolved with its own gaussian
            for i in range(len(self)):
                C = self[i].get_convolution_peak_operator()
                _maps_convolved[i] = C(m[i])
            tod = self.get_operator_to_make_TOD() * _maps_convolved
        else:
            tod = self.get_operator() * m

        if not noiseless:
            tod += self.get_noise()

        return tod
class QubicPlanckMultiBandAcquisition:

    def __init__(self, qubic, planck, type):

        self.qubic = qubic
        self.scene = self.qubic.scene
        self.final_fwhm = self.qubic.final_fwhm
        #self.nfreqs = len(self.qubic.nus)
        self.planck = planck

        self.nus_edge = self.qubic.nus_edge
        self.nueff = self.qubic.nueff
        self.nfreqs = len(self.nueff)
        self.type = type
    def get_operator(self, convolution=False):

        H_qubic = self.qubic.get_operator(convolution=convolution)
        print('QUBIC shape', H_qubic.shapein, H_qubic.shapeout)
        print(H_qubic.operands[0].shapeout, H_qubic.operands[0].shape[0])
        R_qubic = ReshapeOperator(H_qubic.operands[0].shapeout, H_qubic.operands[0].shape[0])
        R_planck = ReshapeOperator((12*self.qubic.scene.nside**2, 3), (12*self.qubic.scene.nside**2*3))
        full_operator=[]

        if self.type == 'QubicIntegrated':
            if len(self.nueff) == 1:
                Operator = [R_qubic(H_qubic)]
            for i in range(self.nfreqs):
                if len(self.nueff) != 1:
                    Operator = [R_qubic(H_qubic.operands[i])]
                for j in range(self.nfreqs):
                    if i == j :
                        Operator.append(R_planck)
                    else:
                        Operator.append(R_planck*0)

                full_operator.append(BlockColumnOperator(Operator, axisout=0))
            return BlockRowOperator(full_operator, axisin=0)

        elif self.type == 'WideBand':

            f = int(self.qubic.qubic150.d['nf_sub']/self.qubic.qubic150.d['nf_recon'])
            if self.qubic.qubic150.d['nf_recon'] == 1:
                H_qubic = H_qubic.operands[0]

            k=0
            for i in range(self.nfreqs):

                if i < int(self.nfreqs/2):
                    subope = H_qubic.operands[0]
                    if self.qubic.qubic150.d['nf_recon'] != 1:
                        print(k)
                        Operator = [R_qubic(subope.operands[k])]
                    else:
                        Operator = [R_qubic(subope)]
                    k+=1
                else:
                    subope = H_qubic.operands[1]
                    if self.qubic.qubic150.d['nf_recon'] != 1:
                        print(k-self.nfreqs)
                        Operator = [R_qubic(subope.operands[k-self.nfreqs])]
                    else:
                        Operator = [R_qubic(subope)]
                    k+=1
                for j in range(self.nfreqs):
                    if i == j :
                        Operator.append(R_planck)
                    else:
                        Operator.append(R_planck*0)
                full_operator.append(BlockColumnOperator(Operator, axisout=0))
            R = ReshapeOperator((self.nfreqs, 12*self.qubic.scene.nside**2, 3), (self.nfreqs*12*self.qubic.scene.nside**2, 3))
            return BlockRowOperator(full_operator, axisin=0) * R
        elif self.type == 'TwoBands':
            if self.qubic.qubic150.d['nf_recon'] == 1:
                H_qubic = H_qubic.operands[0]

                k=0
                for i in range(self.nfreqs):
                    print(i)
                    ndets, nsamples = H_qubic.shapeout
                    R_qubic = ReshapeOperator((ndets, nsamples), ((ndets)*nsamples))

                    #print('ii', H_qubic.operands[i].shapein, H_qubic.operands[i].shapeout)
                    Operator = [R_qubic(H_qubic.operands[i])]

                    for j in range(self.nfreqs):
                        print(i, j)
                        if i == j :
                            Operator.append(R_planck)
                        else:pass
                            #Operator.append(R_planck*0)

                    full_operator.append(BlockColumnOperator(Operator, axisout=0))

                R = ReshapeOperator((self.nfreqs, 12*self.qubic.scene.nside**2, 3), (self.nfreqs*12*self.qubic.scene.nside**2, 3))
                return BlockDiagonalOperator(full_operator, axisin=0) * R
            else:
                k=0
                ndets, nsamples = H_qubic.shapeout
                Operator=[]

                for i in range(2):
                    subope = H_qubic.operands[i]
                    allsubope = []
                    for j in range(int(self.nfreqs/2)):
                        subsubope = [R_qubic(subope.operands[j])]
                        for l in range(int(self.nfreqs/2)):
                            if j == l:
                                subsubope.append(R_planck)
                            else:
                                subsubope.append(R_planck*0)
                        allsubope.append(BlockColumnOperator(subsubope, axisout=0))
                    Operator.append(BlockRowOperator(allsubope, axisin=0))
                R = ReshapeOperator((self.nfreqs, 12*self.qubic.scene.nside**2, 3), (self.nfreqs*12*self.qubic.scene.nside**2, 3))
                return BlockDiagonalOperator(Operator, axisin=0) * R

    def get_invntt_operator(self):

        if self.type == 'TwoBands':
            invntt_qubic = self.qubic.get_invntt_operator()
            invN1 = invntt_qubic.operands[0]
            invN2 = invntt_qubic.operands[1]
            ndets, nsamples = invN1.shapein
            R_qubic = ReshapeOperator((ndets, nsamples), (ndets*nsamples))

            invntt_planck = self.planck.get_invntt_operator()
            R_planck = ReshapeOperator(invntt_planck.shapeout, invntt_planck.shape[0])

            Operator = [R_qubic(invN1(R_qubic.T))]
            for i in range(int(self.nfreqs/2)):
                for j in range(int(self.nfreqs/2)):
                    if i == j:
                        Operator.append(R_planck(invntt_planck(R_planck.T)))
                    else:pass
            Operator.append(R_qubic(invN2(R_qubic.T)))
            for i in range(int(self.nfreqs/2)):
                for j in range(int(self.nfreqs/2)):
                    if i == j:
                        Operator.append(R_planck(invntt_planck(R_planck.T)))
                    else:pass

            return BlockDiagonalOperator(Operator, axisout=0)






        invntt_qubic = self.qubic.get_invntt_operator()
        R_qubic = ReshapeOperator(invntt_qubic.shapeout, invntt_qubic.shape[0])
        invntt_planck = self.planck.get_invntt_operator()
        R_planck = ReshapeOperator(invntt_planck.shapeout, invntt_planck.shape[0])
        Operator = [R_qubic(invntt_qubic(R_qubic.T))]

        if type == 'TwoBands' or 'WideBand':
            for i in range(self.nfreqs):
                Operator.append(R_planck(invntt_planck(R_planck.T)))

        return BlockDiagonalOperator(Operator, axisout=0)
    def get_noise(self):
        n = self.qubic.get_noise()
        sh = n.shape
        n = n.ravel()
        narray = np.array([])
        if self.type == 'TwoBands':
            n1 = n[:(int(sh[0]/2)*sh[1])]
            n2 = n[(int(sh[0]/2)*sh[1]):]
            for i in range(2):
                if i == 0:
                    narray = np.append(narray, n1)
                else:
                    narray = np.append(narray, n2)
                for j in range(int(self.nfreqs/2)):
                    if i == j:
                        #for l in range(int(self.nfreqs/2)):
                        narray = np.r_[narray, self.planck.get_noise().ravel()]
                    else:#pass
                        narray = np.r_[narray, self.planck.get_noise().ravel()]
        else:
            for i in range(self.nfreqs):
                n = np.r_[n, self.planck.get_noise().ravel()]
                narray = n.copy()
        return narray
    def get_observation(self, m=None, convolution=True, noiseless=False):
        """
        Return fusion observation as a sum of monochromatic fusion TODs
        """

        H = self.get_operator()
        if convolution and m is None:
            raise ValueError('Define the map, if you want to use convolution option')

        p = self.planck
        if m is None:
            m = p._true_sky
        tod_shape = len(self.qubic[0].instrument) * len(self.qubic[0].sampling) + \
                    len(self.qubic.scene.kind) * hp.nside2npix(self.qubic.scene.nside)
        tod = np.zeros(tod_shape)
        maps = np.empty((self.nfreqs, m.shape[1], m.shape[0]))
        for i in range(self.nfreqs):
            q = self.qubic[i]
            if convolution:
                C = q.get_convolution_peak_operator()
                maps[i] = C(m[i])
            else:
                maps[i] = m[i].copy()

            tod += H.operands[i](maps[i])


            #tod = H(maps)

        if not noiseless:
            np.random.seed(None)
            tod += self.get_noise()

        return tod, maps



        print(obs_qubic)

################################################################################################################
######## Proposition to replace QubicAcquisition, QubicPolyacquisition and QubicMultibandAcquisition ###########
################################################################################################################
class QubicIntegrated:

    def __init__(self, multiinstrument, sampling, scene, d, nus_edge):

        self.multiinstrument = multiinstrument
        self.sampling = sampling
        self.scene = scene
        self.d = d
        self.nus_edge = nus_edge
        self.nus = np.array([q.filter.nu / 1e9 for q in multiinstrument])

        edges=np.zeros((len(self.nus_edge)-1, 2))
        for i in range(len(self.nus_edge)-1):
            edges[i] = np.array([self.nus_edge[i], self.nus_edge[i+1]])

        self.bands = edges
        self.subacqs = [QubicAcquisition(self.multiinstrument[i], self.sampling, self.scene, self.d) for i in range(len(self.multiinstrument))]

        self.nueff = np.zeros(len(self.nus_edge)-1)
        for i in range(self.d['nf_recon']):
            self.nueff[i] = np.mean(self.bands[i])

        ### fwhm

        self.allfwhm = np.zeros(len(self.multiinstrument))
        for i in range(len(self.multiinstrument)):
            self.allfwhm[i] = self.subacqs[i].get_convolution_peak_operator().fwhm

        self.final_fwhm = np.zeros(self.d['nf_recon'])
        fact = self.d['nf_sub']/self.d['nf_recon']
        if fact != 1:
            for i in range(self.d['nf_recon']):
                self.final_fwhm[i] = np.mean(self.allfwhm[int(i*fact):int(fact*(i+1)-1)])
        else:
            for i in range(self.d['nf_recon']):
                self.final_fwhm[i] = self.allfwhm[i]

    def print_informations(self):

        print('*****************')
        print('Nf_recon : {}'.format(self.d['nf_recon']))
        print('Nf_sub : {}'.format(self.d['nf_sub']))
        print('Npix : {}'.format(12*self.scene.nside**2))
        print('*****************')
    def give_me_correctedFWHM(self):

        fwhm_max = np.max(self.final_fwhm)
        corrected = np.zeros(len(self.final_fwhm))

        for i in range(len(corrected)):
            corrected[i] = np.sqrt(fwhm_max**2 - self.final_fwhm[i]**2)

        return corrected
    def correct_reconstructed_maps(self, fwhm, solution):

        new_reconstructed_maps = solution.copy()
        for i in range(self.d['nf_recon']):
            C = HealpixConvolutionGaussianOperator(fwhm=fwhm[i])
            new_reconstructed_maps[i] = C(solution[i])
        return new_reconstructed_maps
    def _get_average_instrument_acq(self):
        """
        Create and return a QubicAcquisition instance of a monochromatic
            instrument with frequency correspondent to the mean of the
            frequency range.
        """
        #if len(self) == 1:
        #    return self[0]
        q0 = self.multiinstrument[0]
        nu_min = self.multiinstrument[0].filter.nu
        nu_max = self.multiinstrument[-1].filter.nu
        nep = q0.detector.nep
        fknee = q0.detector.fknee
        fslope = q0.detector.fslope

        d1 = self.d.copy()
        d1['filter_nu'] = (nu_max + nu_min) / 2.
        d1['filter_relative_bandwidth'] = (nu_max - nu_min) / ((nu_max + nu_min) / 2.)
        d1['detector_nep'] = nep
        d1['detector_fknee'] = fknee
        d1['detector_fslope'] = fslope

        q = qubic.QubicInstrument(d1, FRBW=q0.FRBW)
        q.detector = q0.detector
        #s_ = self.sampling
        #nsamplings = self.multiinstrument[0].comm.allreduce(len(s_))

        d1['random_pointing'] = True
        d1['sweeping_pointing'] = False
        d1['repeat_pointing'] = False
        d1['RA_center'] = 0.
        d1['DEC_center'] = 0.
        d1['npointings'] = self.d['npointings']
        d1['dtheta'] = 10.
        d1['period'] = self.d['period']

        # s = create_random_pointings([0., 0.], nsamplings, 10., period=s_.period)
        a = QubicAcquisition(q, self.sampling, self.scene, d1)
        return a
    def get_noise(self):
        a = self._get_average_instrument_acq()
        return a.get_noise()
    def _get_array_of_operators(self):
        return [a.get_operator() for a in self.subacqs]
    def get_operator_to_make_TOD(self):
        operator = self._get_array_of_operators()
        return BlockRowOperator(operator, new_axisin=0)
    def get_operator(self, convolution=False):

        op = np.array(self._get_array_of_operators())
        op_sum = []
        allnus = np.zeros(len(self.nus_edge)-1)
        for ii, band in enumerate(self.bands):
            print('Making sum from {:.2f} to {:.2f}'.format(band[0], band[1]))
            if convolution: fwhm = self.final_fwhm[ii]
            else: fwhm = 0
            C =  HealpixConvolutionGaussianOperator(fwhm=fwhm)
            op_i = op[(self.nus > band[0]) * (self.nus < band[1])].sum(axis=0)
            op_sum.append(op_i * C)
            allnus[ii]=np.mean(self.nus_edge[ii:ii+2])

        return BlockRowOperator(op_sum, new_axisin=0)
    def get_coverage(self):
        return self.subacqs[0].get_coverage()
    def get_invntt_operator(self):
        invN = self.subacqs[0].get_invntt_operator()
        R = ReshapeOperator(invN.shapeout, invN.shape[0])
        return invN
class QubicTwoBands:

    def __init__(self, q150, q220, pointing, scene, d150, d220, nus_edge150, nus_edge220):

        self.q150 = q150
        self.q220 = q220
        self.pointing = pointing
        self.scene = scene
        self.d150 = d150
        self.d = self.d150.copy()
        self.d220 = d220
        self.nus_edge150 = nus_edge150
        self.nus_edge220 = nus_edge220
        self.nus_edge = np.array([])
        self.nus_edge = np.append(self.nus_edge, self.nus_edge150)
        self.nus_edge = np.append(self.nus_edge, self.nus_edge220)


        self.qubic150 = QubicIntegrated(self.q150, self.pointing, self.scene, self.d150, nus_edge150)
        self.qubic220 = QubicIntegrated(self.q220, self.pointing, self.scene, self.d220, nus_edge220)

        self.final_fwhm = np.array([])
        self.final_fwhm = np.append(self.final_fwhm, self.qubic150.final_fwhm)
        self.final_fwhm = np.append(self.final_fwhm, self.qubic220.final_fwhm)

        self.nueff = np.array([])
        self.nueff = np.append(self.nueff, self.qubic150.nueff)
        self.nueff = np.append(self.nueff, self.qubic220.nueff)

    def get_operator(self, convolution):

        self.H150 = self.qubic150.get_operator(convolution=convolution)
        self.H220 = self.qubic220.get_operator(convolution=convolution)
        ndets, nsamples = self.H150.shapeout

        ope = [self.H150, self.H220]

        if self.d['nf_recon'] == 1:
            hh = BlockDiagonalOperator(ope, new_axisout=0)
            R = ReshapeOperator(hh.shapeout, (hh.shapeout[0]*hh.shapeout[1], hh.shapeout[2]))
            #H = BlockDiagonalOperator(ope, axisin=0)
            #R = ReshapeOperator((2*self.d['nf_recon'], 12*self.d['nside']**2, 3), H.shapein)
            return R * hh
        else:
            return BlockDiagonalOperator(ope, axisin=0)

    def get_coverage(self):
        return self.qubic150.get_coverage()

    def get_noise(self):

        self.n150 = self.qubic150.get_noise()
        self.n220 = self.qubic220.get_noise()

        return np.r_[self.n150, self.n220]

    def get_invntt_operator(self):

        self.invn150 = self.qubic150.get_invntt_operator()
        self.invn220 = self.qubic220.get_invntt_operator()

        return BlockDiagonalOperator([self.invn150, self.invn220], axisout=0)
class QubicWideBand:

    def __init__(self, qubic150, qubic220, scene):

        self.qubic150 = qubic150
        self.qubic220 = qubic220

        self.final_fwhm = np.array([])
        self.final_fwhm = np.append(self.final_fwhm, self.qubic150.final_fwhm)
        self.final_fwhm = np.append(self.final_fwhm, self.qubic220.final_fwhm)

        self.nueff = np.array([])

        for i in range(len(self.qubic150.nus_edge)-1):
            self.nueff = np.append(self.nueff, np.mean(self.qubic150.nus_edge[i:i+2]))

        for i in range(len(self.qubic220.nus_edge)-1):
            self.nueff = np.append(self.nueff, np.mean(self.qubic220.nus_edge[i:i+2]))



        self.nus_edge = np.array([])
        self.nus_edge = np.append(self.nus_edge, self.qubic150.nus_edge)
        self.nus_edge = np.append(self.nus_edge, self.qubic220.nus_edge)
        self.Nf = self.qubic150.d['nf_recon']
        self.scene = scene


    def get_operator(self, convolution=False):

        self.H150 = self.qubic150.get_operator(convolution=convolution)
        self.H220 = self.qubic220.get_operator(convolution=convolution)

        if self.qubic150.d['nf_recon']:
            H = BlockRowOperator([self.H150, self.H220], axisin=0)
            R = ReshapeOperator((2*self.qubic150.d['nf_recon'], 12*self.qubic150.d['nside']**2, 3), H.shapein)
            return H * R
        else:
            return BlockRowOperator([self.H150, self.H220], new_axisin=0)

    def get_noise(self):
        detector_noise = self.qubic150.get_noise() + self.qubic220.get_noise()
        return detector_noise

    def get_invntt_operator(self):
        return self.qubic150.get_invntt_operator() + self.qubic220.get_invntt_operator()

    def get_coverage(self):
        return self.qubic150.get_coverage()

class PlanckAcquisitionComponentsMapMaking:

    def __init__(self, band, scene, true_sky, comp, factor=1, fwhm=0, mask=None, convolution_operator=None):
        if band not in (30, 44, 70, 143, 217, 353):
            raise ValueError("Invalid band '{}'.".format(band))
        if true_sky is None:
            raise ValueError('The Planck Q & U maps are not released yet.')
        if scene.kind == 'IQU' and true_sky.shape[-1] != 3:
            raise TypeError('The Planck sky shape is not (npix, 3).')
        #true_sky = np.array(hp.ud_grade(true_sky.T, nside_out=scene.nside),
        #                    copy=False).T
        if scene.kind == 'IQU' and true_sky.shape[-1] != 3:
            raise TypeError('The Planck sky shape is not (npix, 3).')
        self.scene = scene
        self.nside = self.scene.nside
        self.fwhm = fwhm
        self.band = band
        self.comp = comp
        self._true_sky = true_sky
        if mask is not None:
            self.mask = mask
        else:
            self.mask = np.ones(scene.npixel, dtype=np.bool)


        if band == 30:
            filename = 'Variance_Planck30GHz_Kcmb2_ns256.fits'
            var=np.zeros((12*self.scene.nside**2, 3))
            for i in range(3):
                var[:, i] = hp.ud_grade(hp.fitsfunc.read_map(filename, field=i), self.scene.nside)
            sigma = 1e5 * factor * np.sqrt(var)

        elif band == 44:
            filename = 'Variance_Planck44GHz_Kcmb2_ns256.fits'
            var=np.zeros((12*self.scene.nside**2, 3))
            for i in range(3):
                var[:, i] = hp.ud_grade(hp.fitsfunc.read_map(filename, field=i), self.scene.nside)
            sigma = 1e5 * factor * np.sqrt(var)
        elif band == 70:
            filename = 'Variance_Planck70GHz_Kcmb2_ns256.fits'
            var=np.zeros((12*self.scene.nside**2, 3))
            for i in range(3):
                var[:, i] = hp.ud_grade(hp.fitsfunc.read_map(filename, field=i), self.scene.nside)
            sigma = 1e5 * factor * np.sqrt(var)
        elif band == 143:
            filename = 'Variance_Planck143GHz_Kcmb2_ns256.fits'
            sigma = 1e6 * factor * np.sqrt(FitsArray(PATH + filename))
        elif band == 217:
            filename = 'Variance_Planck217GHz_Kcmb2_ns256.fits'
            sigma = 1e6 * factor * np.sqrt(FitsArray(PATH + filename))
        else:
            filename = 'Variance_Planck353GHz_Kcmb2_ns256.fits'
            var=np.zeros((12*self.scene.nside**2, 3))
            for i in range(3):
                var[:, i] = hp.ud_grade(hp.fitsfunc.read_map(filename, field=i), self.scene.nside)
            sigma = 1e5 * factor * np.sqrt(var)

        #if d['nside']!=256:
        #    sig = np.zeros((12*d['nside']**2, 3))
        #    for i in range(3):
        #        sig[:, i]=hp.ud_grade(sigma[:, i], d['nside'])




        if scene.kind == 'I':
            sigma = sigma[:, 0]
        elif scene.kind == 'QU':
            sigma = sigma[:, :2]
        if self.scene.nside != 256:
            sigma = np.array(hp.ud_grade(sigma.T, self.scene.nside, power=2),
                             copy=False).T
        self.sigma = sigma
        if convolution_operator is None:
            self.C = IdentityOperator()
        else:
            self.C = convolution_operator

    def get_mixingmatrix(self, beta, nus):

        A = mm.MixingMatrix(*self.comp)
        #if nus == 143:
        #    nus = np.array([150])
        #elif nus == 217:
        #    nus = np.array([220])
        A_ev = A.evaluator(nus)
        A = A_ev(beta)

        return A
    def get_mixing_operator(self, beta):

        A = self.get_mixingmatrix(beta, np.array([self.band]))
        nf, nc = A.shape
        R = ReshapeOperator(((1, 12*self.nside**2, 3)), ((12*self.nside**2, 3)))
        Rp = ReshapeOperator(((12*self.nside**2, 3)), ((12*self.nside**2*3)))
        D = DenseOperator(A[0], broadcast='rightward', shapein=(nc, 12*self.nside**2, 3),
                            shapeout=(1, 12*self.nside**2, 3))

        return Rp * R * D
    def get_operator(self, beta):
        D = self.get_mixing_operator(beta)
        return DiagonalOperator(self.mask.astype(np.int), broadcast='rightward',
                                shapein=self.scene.shape) * D

    def get_invntt_operator(self):
        return DiagonalOperator(1 / self.sigma ** 2, broadcast='leftward',
                                shapein=self.scene.shape)

    def get_noise(self):
        state = np.random.get_state()
        #np.random.seed(self._SIMULATED_PLANCK_SEED)
        np.random.seed(None)
        out = np.random.standard_normal(self._true_sky[0].shape) * self.sigma
        np.random.set_state(state)
        return out

    def get_observation(self, noiseless=False):
        obs = self._true_sky
        if not noiseless:
            obs = obs + self.C(self.get_noise())
        if len(self.scene.shape) == 2:
            for i in range(self.scene.shape[1]):
                obs[~(self.mask), i] = 0.
        else:
            obs[~(self.mask)] = 0.
        return obs
        # XXX neglecting convolution effects...
        HealpixConvolutionGaussianOperator(fwhm=self.fwhm)(obs, obs)
        return obs
class QubicPlanckMultiBandAcquisitionComponentsMapMaking:

    def __init__(self, qubic, planck, type, comp):

        self.qubic = qubic
        self.scene = self.qubic.scene
        self.nside = self.scene.nside
        self.final_fwhm = self.qubic.final_fwhm
        #self.nfreqs = len(self.qubic.nus)
        self.planck = planck

        self.nus_edge = self.qubic.nus_edge
        self.nueff = self.qubic.nueff
        self.nfreqs = len(self.nueff)
        self.type = type
        self.comp = comp

    def get_mixingmatrix(self, beta, nus):

        A = mm.MixingMatrix(*self.comp)
        A_ev = A.evaluator(nus)
        A = A_ev(beta)

        return A
    def get_mixing_operator(self, beta, nus):

        A = self.get_mixingmatrix(beta, nus)
        nf, nc = A.shape
        R = ReshapeOperator(((1, 12*self.nside**2, 3)), ((12*self.nside**2, 3)))
        D = DenseOperator(A[0], broadcast='rightward', shapein=(nc, 12*self.nside**2, 3),
                            shapeout=(1, 12*self.nside**2, 3))

        return R * D
    def get_operator(self, beta, convolution=False):

        H_qubic = self.qubic.get_operator(beta, convolution=convolution)
        R_planck = PlanckAcquisitionComponentsMapMaking(self.planck.band, self.scene, self.planck._true_sky,
                    self.comp).get_operator(beta)
        R_qubic = ReshapeOperator(H_qubic.operands[0].shapeout, H_qubic.operands[0].shape[0])

        if self.type == 'QubicIntegrated':
            #if len(self.nueff) == 1:
            Operator = [R_qubic(H_qubic), R_planck]
            #Operator = BlockColumnOperator([R_qubic(H_qubic.operands[0]), R_planck, ZeroOperator()], axisout=0) + \
            #BlockColumnOperator([R_qubic(H_qubic.operands[1]), R_planck], axisout=0)
            return BlockColumnOperator(Operator, axisout=0)

#
            #    subope = H_qubic.operands[i]
            #    subsubope = [R_qubic(subope)]
            #    for l in range(self.nfreqs):
            #        if i == l:
            #            subsubope.append(R_planck)
            #        else:
            #           subsubope.append(ZeroOperator(shapein=R_planck.shapein, shapeout=R_planck.shapeout))
            #    Operator.append(BlockColumnOperator(subsubope, axisout=0))
            #    if len(self.nueff) != 1:
            #        Operator = [R_qubic(H_qubic.operands[i])]
            #    for j in range(self.nfreqs):
            #        if i == j :
            #            Operator.append(R_planck)
            #        else:
            #            Operator.append(ZeroOperator(shapein=(len(self.comp), 12*self.nside**2,3), shapeout=len(self.comp)*12*self.nside**2*3))
#
            #    full_operator.append(BlockColumnOperator(Operator, axisout=0))
            return BlockColumnOperator(Operator, axisout=0)

        elif self.type == 'WideBand':
            Operator=[R_qubic(H_qubic), R_planck]

            return BlockColumnOperator(Operator, axisout=0)
        elif self.type == 'TwoBands':
            Operator=[R_qubic(H_qubic.operands[0]), R_planck, R_qubic(H_qubic.operands[1]), R_planck]
            return BlockColumnOperator(Operator, axisout=0)

    def get_invntt_operator(self):


        invntt_qubic = self.qubic.get_invntt_operator()
        R_qubic = ReshapeOperator(invntt_qubic.shapeout, invntt_qubic.shape[0])
        invntt_planck = self.planck.get_invntt_operator()
        R_planck = ReshapeOperator(invntt_planck.shapeout, invntt_planck.shape[0])
        print(R_qubic.shapein, R_qubic.shapeout)

        if self.type == 'QubicIntegrated' or self.type == 'WideBand':

            Operator=[R_qubic(invntt_qubic(R_qubic.T)), R_planck(invntt_planck(R_planck.T))]

        else:
            ndets, ns = invntt_qubic.operands[0].shapein
            R_qubic = ReshapeOperator((ndets, ns), (ns*(ndets)))
            Operator=[R_qubic(invntt_qubic.operands[0](R_qubic.T)),
                      R_planck(invntt_planck(R_planck.T)),
                      R_qubic(invntt_qubic.operands[1](R_qubic.T)),
                      R_planck(invntt_planck(R_planck.T))]

        return BlockDiagonalOperator(Operator, axisout=0)




    def get_noise(self):

        if self.type == 'TwoBands':

            n = np.r_[self.qubic.qubic150.get_noise().ravel(),
                      self.planck.get_noise().ravel(),
                      self.qubic.qubic220.get_noise().ravel(),
                      self.planck.get_noise().ravel()]

        elif self.type == 'WideBand':

            n = np.r_[self.qubic.qubic150.get_noise().ravel()+self.qubic.qubic220.get_noise().ravel(),
                      self.planck.get_noise().ravel()]

        else:
            n = np.r_[self.qubic.get_noise().ravel(),
                      self.planck.get_noise().ravel()]

        return n

class QubicIntegratedComponentsMapMaking:

    def __init__(self, multiinstrument, sampling, scene, d, nus_edge, comp):

        self.multiinstrument = multiinstrument
        self.sampling = sampling
        self.scene = scene
        self.nside = self.scene.nside
        self.d = d
        self.nus_edge = nus_edge
        self.comp = comp
        self.nus = np.array([q.filter.nu / 1e9 for q in multiinstrument])

        edges=np.zeros((len(self.nus_edge)-1, 2))
        for i in range(len(self.nus_edge)-1):
            edges[i] = np.array([self.nus_edge[i], self.nus_edge[i+1]])

        self.bands = edges
        self.bands = self.bands[0]
        self.subacqs = [QubicAcquisition(self.multiinstrument[i], self.sampling, self.scene, self.d) for i in range(len(self.multiinstrument))]

        _, _, self.nueff, _, _, _ = qubic.compute_freq(int(self.d['filter_nu']/1e9), Nfreq=self.d['nf_sub'])
        #self.nueff = np.zeros(len(self.nus_edge)-1)
        #for i in range(self.d['nf_recon']):
        #    self.nueff[i] = np.mean(self.bands[i])

        ### fwhm

        self.final_fwhm = np.zeros(len(self.multiinstrument))
        for i in range(len(self.multiinstrument)):
            self.final_fwhm[i] = self.subacqs[i].get_convolution_peak_operator().fwhm


    def print_informations(self):

        print('*****************')
        print('Nf_recon : {}'.format(self.d['nf_recon']))
        print('Nf_sub : {}'.format(self.d['nf_sub']))
        print('Npix : {}'.format(12*self.scene.nside**2))
        print('*****************')

    def get_mixingmatrix(self, beta, nus):

        A = mm.MixingMatrix(*self.comp)
        A_ev = A.evaluator(nus)
        A = A_ev(beta)

        return A
    def get_mixing_operator(self, beta, nus):

        A = self.get_mixingmatrix(beta, nus)
        nf, nc = A.shape
        R = ReshapeOperator(((1, 12*self.nside**2, 3)), ((12*self.nside**2, 3)))
        D = DenseOperator(A[0], broadcast='rightward', shapein=(nc, 12*self.nside**2, 3),
                            shapeout=(1, 12*self.nside**2, 3))

        return R * D
    def give_me_correctedFWHM(self):

        fwhm_max = np.max(self.final_fwhm)
        corrected = np.zeros(len(self.final_fwhm))

        for i in range(len(corrected)):
            corrected[i] = np.sqrt(fwhm_max**2 - self.final_fwhm[i]**2)

        return corrected
    def correct_reconstructed_maps(self, fwhm, solution):

        new_reconstructed_maps = solution.copy()
        for i in range(self.d['nf_recon']):
            C = HealpixConvolutionGaussianOperator(fwhm=fwhm[i])
            new_reconstructed_maps[i] = C(solution[i])
        return new_reconstructed_maps
    def _get_average_instrument_acq(self):
        """
        Create and return a QubicAcquisition instance of a monochromatic
            instrument with frequency correspondent to the mean of the
            frequency range.
        """
        #if len(self) == 1:
        #    return self[0]
        q0 = self.multiinstrument[0]
        nu_min = self.multiinstrument[0].filter.nu
        nu_max = self.multiinstrument[-1].filter.nu
        nep = q0.detector.nep
        fknee = q0.detector.fknee
        fslope = q0.detector.fslope

        d1 = self.d.copy()
        d1['filter_nu'] = (nu_max + nu_min) / 2.
        d1['filter_relative_bandwidth'] = (nu_max - nu_min) / ((nu_max + nu_min) / 2.)
        d1['detector_nep'] = nep
        d1['detector_fknee'] = fknee
        d1['detector_fslope'] = fslope

        q = qubic.QubicInstrument(d1, FRBW=q0.FRBW)
        q.detector = q0.detector
        #s_ = self.sampling
        #nsamplings = self.multiinstrument[0].comm.allreduce(len(s_))

        d1['random_pointing'] = True
        d1['sweeping_pointing'] = False
        d1['repeat_pointing'] = False
        d1['RA_center'] = 0.
        d1['DEC_center'] = 0.
        d1['npointings'] = self.d['npointings']
        d1['dtheta'] = 10.
        d1['period'] = self.d['period']

        # s = create_random_pointings([0., 0.], nsamplings, 10., period=s_.period)
        a = QubicAcquisition(q, self.sampling, self.scene, d1)
        return a
    def get_noise(self):
        a = self._get_average_instrument_acq()
        return a.get_noise()
    def _get_array_of_operators(self):
        return [a.get_operator() for a in self.subacqs]
    def get_operator_to_make_TOD(self):
        operator = self._get_array_of_operators()
        return BlockRowOperator(operator, new_axisin=0)
    def get_operator(self, beta, convolution=False):

        op = self._get_array_of_operators()

        #print('Making sum from {:.2f} to {:.2f} with {:.0f} acquisitions'.format(self.bands[0], self.bands[1], self.d['nf_sub']))

        for inu, nu in enumerate(self.nueff):
            if convolution:
                C =  HealpixConvolutionGaussianOperator(fwhm=self.final_fwhm[inu])
                op[inu] *= C
            D = self.get_mixing_operator(beta, np.array([nu]))
            op[inu] = op[inu] * D


        return BlockColumnOperator([np.sum(op, axis=0)], axisout=0)
    def get_coverage(self):
        return self.subacqs[0].get_coverage()
    def get_invntt_operator(self):
        invN = self.subacqs[0].get_invntt_operator()
        R = ReshapeOperator(invN.shapeout, invN.shape[0])
        return invN
class QubicWideBandComponentsMapMaking:

    def __init__(self, qubic150, qubic220, scene, comp):

        self.qubic150 = qubic150
        self.qubic220 = qubic220
        self.comp = comp

        self.final_fwhm = np.array([])
        self.final_fwhm = np.append(self.final_fwhm, self.qubic150.final_fwhm)
        self.final_fwhm = np.append(self.final_fwhm, self.qubic220.final_fwhm)

        self.nueff = np.array([])

        self.nueff = np.append(self.nueff, self.qubic150.nueff)
        self.nueff = np.append(self.nueff, self.qubic220.nueff)
        #for i in range(len(self.qubic150.nus_edge)-1):
        #    self.nueff = np.append(self.nueff, np.mean(self.qubic150.nus_edge[i:i+2]))

        #for i in range(len(self.qubic220.nus_edge)-1):
        #    self.nueff = np.append(self.nueff, np.mean(self.qubic220.nus_edge[i:i+2]))



        self.nus_edge = np.array([])
        self.nus_edge = np.append(self.nus_edge, self.qubic150.nus_edge)
        self.nus_edge = np.append(self.nus_edge, self.qubic220.nus_edge)
        self.Nf = self.qubic150.d['nf_sub']
        self.scene = scene


    def get_operator(self, beta, convolution=False):


        self.H150 = self.qubic150.get_operator(beta, convolution=convolution)
        self.H220 = self.qubic220.get_operator(beta, convolution=convolution)
        R = ReshapeOperator(self.H150.shapeout, self.H150.shape[0])

        return R(self.H150)+R(self.H220)

    def get_noise(self):
        detector_noise = self.qubic150.get_noise() + self.qubic220.get_noise()
        return detector_noise

    def get_invntt_operator(self):
        return self.qubic150.get_invntt_operator() + self.qubic220.get_invntt_operator()

    def get_coverage(self):
        return self.qubic150.get_coverage()
class QubicTwoBandsComponentsMapMaking:

    def __init__(self, qubic150, qubic220, scene, comp):

        self.qubic150 = qubic150#QubicIntegratedComponentsMapMaking(self.q150, self.pointing, self.scene, self.d150, nus_edge150, self.comp)
        self.qubic220 = qubic220#QubicIntegratedComponentsMapMaking(self.q220, self.pointing, self.scene, self.d220, nus_edge220, self.comp)

        self.comp = comp
        self.scene = scene
        self.nside = self.scene.nside
        self.nus_edge150 = self.qubic150.nus_edge
        self.nus_edge220 = self.qubic220.nus_edge
        self.nus_edge = np.array([])
        self.nus_edge = np.append(self.nus_edge, self.nus_edge150)
        self.nus_edge = np.append(self.nus_edge, self.nus_edge220)



        self.final_fwhm = np.array([])
        self.final_fwhm = np.append(self.final_fwhm, self.qubic150.final_fwhm)
        self.final_fwhm = np.append(self.final_fwhm, self.qubic220.final_fwhm)

        self.nueff = np.array([])
        self.nueff = np.append(self.nueff, self.qubic150.nueff)
        self.nueff = np.append(self.nueff, self.qubic220.nueff)

    def get_mixingmatrix(self, beta, nus):

        A = mm.MixingMatrix(*self.comp)
        A_ev = A.evaluator(nus)
        A = A_ev(beta)

        return A
    def get_mixing_operator(self, beta, nus):

        A = self.get_mixingmatrix(beta, nus)
        nf, nc = A.shape
        R = ReshapeOperator(((1, 12*self.nside**2, 3)), ((12*self.nside**2, 3)))
        D = DenseOperator(A[0], broadcast='rightward', shapein=(nc, 12*self.nside**2, 3),
                            shapeout=(1, 12*self.nside**2, 3))

        return R * D

    def get_operator(self, beta, convolution=False):

        self.H150 = self.qubic150.get_operator(beta, convolution=convolution)
        self.H220 = self.qubic220.get_operator(beta, convolution=convolution)
        R = ReshapeOperator(self.H150.shapeout, self.H150.shape[0])
        ndets, nsamples = self.H150.shapeout
        #D143 = self.get_mixing_operator(beta, [143])
        #D217 = self.get_mixing_operator(beta, [217])
        #Hpl143 = D143 * ReshapeOperator(12*self.nside**2, 3), 3*12*self.nside**2)
        #Hpl217 = D217 * ReshapeOperator(12*self.nside**2, 3), 3*12*self.nside**2)
        Operator=[R(self.H150), R(self.H220)]

        return BlockColumnOperator(Operator, axisout=0)

    def get_coverage(self):
        return self.qubic150.get_coverage()

    def get_noise(self):

        self.n150 = self.qubic150.get_noise()
        self.n220 = self.qubic220.get_noise()

        return np.r_[self.n150, self.n220]

    def get_invntt_operator(self):

        self.invn150 = self.qubic150.get_invntt_operator()
        self.invn220 = self.qubic220.get_invntt_operator()

        return BlockDiagonalOperator([self.invn150, self.invn220], axisout=0)

###############################################################
################## Components Map-Making ######################
###############################################################



class OtherData:

    def __init__(self, nus, nside, comp):

        pkl_file = open('AllDataSet_Components_MapMaking.pkl', 'rb')
        dataset = pickle.load(pkl_file)
        self.dataset = dataset

        self.nus = nus
        self.nside = nside
        self.npix = 12*self.nside**2
        self.bw = []
        for ii, i in enumerate(self.nus):
            self.bw.append(self.dataset['bw{}'.format(i)])
        self.fwhm = arcmin2rad(create_array('fwhm', self.nus, self.nside))
        self.comp = comp


    def integrated_convolved_data(self, A, band, allnus, fwhm):

        '''
        This function allow to create operator which integrate bandpass for other experiments like Planck
        '''

        nf, nc = A.shape
        R = ReshapeOperator((1, 12*self.nside**2, 3), (12*self.nside**2, 3))

        operator = []
        verb = fwhm#*band/allnus
        #print('Beam integration from {:.4f} to {:.4f}'.format(rad2arcmin(np.min(verb)), rad2arcmin(np.max(verb))))
        for i in range(nf):
            D = DenseOperator(A[i], broadcast='rightward', shapein=(nc, 12*self.nside**2, 3), shapeout=(1, 12*self.nside**2, 3))
            C = HealpixConvolutionGaussianOperator(fwhm=verb)
            operator.append(C * R * D)

        return AdditionOperator(operator)/nf
    def get_mixingmatrix(self, nu, beta):
        A = mm.MixingMatrix(*self.comp)
        A_ev = A.evaluator(nu)
        A = A_ev(beta)
        return A
    def get_invntt_operator(self, fact=None):

        allsigma=np.array([])
        for inu, nu in enumerate(self.nus):
            if fact is None:
                f=1
            else:
                f=fact[inu]
            sigma = f * hp.ud_grade(self.dataset['noise{}'.format(nu)].T, self.nside).T
            allsigma = np.append(allsigma, sigma.ravel())

        allsigma = allsigma.ravel().copy()
        invN = DiagonalOperator(1 / allsigma ** 2, broadcast='leftward', shapein=(3*len(self.nus)*12*self.nside**2))
        R = ReshapeOperator(invN.shapeout, invN.shape[0])

        return R(invN(R.T))
    def get_operator(self, nintegr, beta, convolution):

        R2tod = ReshapeOperator((12*self.nside**2, 3), (3*12*self.nside**2))

        op = []
        for ii, i in enumerate(self.nus):
            allnus = np.linspace(i-self.bw[ii]/2, i+self.bw[ii]/2, nintegr)
            A = self.get_mixingmatrix(allnus, beta)
            if convolution: fwhm = self.fwhm[ii]
            else: fwhm = 0
            op.append(R2tod(self.integrated_convolved_data(A, i, allnus, fwhm=fwhm)))

        return BlockColumnOperator(op, axisout=0)
    def get_noise(self, fact=None):
        state = np.random.get_state()
        np.random.seed(None)
        out = np.zeros((len(self.nus), self.npix, 3))
        R2tod = ReshapeOperator((len(self.nus), 12*self.nside**2, 3), (len(self.nus)*3*12*self.nside**2))
        for inu, nu in enumerate(self.nus):
            if fact is None:
                f=1
            else:
                f=fact[inu]
            sigma = f * hp.ud_grade(self.dataset['noise{}'.format(nu)].T, self.nside).T
            out[inu] = np.random.standard_normal((self.npix,3)) * sigma
        np.random.set_state(state)
        return R2tod(out)
    def get_maps(self, tod):
        R2map = ReshapeOperator((len(self.nus)*3*12*self.nside**2), (len(self.nus), 12*self.nside**2, 3))
        return R2map(tod)
class QubicOtherIntegrated:

    def __init__(self, qubic, external_nus, comp, nintegr=1):

        self.qubic = qubic
        self.external_nus = external_nus
        self.comp = comp
        self.nside = self.qubic.scene.nside
        self.npix = 12*self.nside**2
        self.nintegr = nintegr
        self.ndets = 992

        self.allresolution = self.qubic.final_fwhm


        pkl_file = open('AllDataSet_Components_MapMaking.pkl', 'rb')
        dataset = pickle.load(pkl_file)
        self.dataset = dataset
        self.bw = []
        if self.external_nus is not None:
            for ii, i in enumerate(self.external_nus):
                self.bw.append(self.dataset['bw{}'.format(i)])

        self.fwhm = []
        if self.external_nus is not None:
            for ii, i in enumerate(self.external_nus):
                self.fwhm.append(arcmin2rad(self.dataset['fwhm{}'.format(i)]))
                self.allresolution = np.append(self.allresolution, arcmin2rad(self.dataset['fwhm{}'.format(i)]))


    def get_mixingmatrix(self, nus, beta):

        A = mm.MixingMatrix(*self.comp)
        A_ev = A.evaluator(nus)
        A = A_ev(beta)

        return A
    def get_mixing_operator(self, nus, beta):

        A = self.get_mixingmatrix(nus, beta)
        nf, nc = A.shape

        D = DenseOperator(A, broadcast='rightward', shapein=(nc, 12*self.nside**2, 3),
                            shapeout=(nf, 12*self.nside**2, 3))

        return D
    def get_external_invntt_operator(self):

        allsigma=np.array([])
        for inu, nu in enumerate(self.external_nus):
            sigma = hp.ud_grade(self.dataset['noise{}'.format(nu)].T, self.nside).T
            allsigma = np.append(allsigma, sigma.ravel())

        invN = DiagonalOperator(1 / allsigma ** 2, broadcast='leftward', shapein=(len(self.external_nus)*12*self.nside**2*3))

        R = ReshapeOperator(invN.shapeout, invN.shape[0])
        return R(invN(R.T))
    def get_operator(self, beta, convolution, H_qubic=None):

        if H_qubic is None:
            H_qubic = self.qubic.get_operator(convolution=convolution)
        R_qubic = ReshapeOperator(H_qubic.shapeout, H_qubic.shape[0])
        D = self.get_mixing_operator(self.qubic.nueff, beta)
        R = ReshapeOperator((1, self.npix, 3), (self.npix, 3))

        if len(self.qubic.nueff) == 1:
            Operator=[R_qubic(H_qubic) * R * D]   # R if nsub = 1
        else:
            Operator=[R_qubic(H_qubic) * D]

        R2tod = ReshapeOperator((1, self.npix, 3), (self.npix*3))

        if self.external_nus is not None:
            for ii, i in enumerate(self.external_nus):
                bw = self.dataset['bw{}'.format(i)]
                other = OtherData([i], self.nside, self.comp)
                Hother = other.get_operator(self.nintegr, beta, convolution=convolution)
                Operator.append(Hother)

        return BlockColumnOperator(Operator, axisout=0)
    def get_noise(self, fact=None):

        noise_qubic = self.qubic.get_noise().ravel()
        noise_external = OtherData(self.external_nus, self.nside, self.comp).get_noise(fact=fact)

        return np.r_[noise_qubic, noise_external]
    def get_maps(self):
        return ReshapeOperator(3*len(self.external_nus)*self.npix, (len(self.external_nus), self.npix, 3))
    def reconvolve_to_worst_resolution(self, tod):

        sh = tod.shape[0]
        sh_external = len(self.external_nus)*3*12*self.nside**2
        sh_1external = 3*12*self.nside**2
        shape_tod_qubic = sh - sh_external
        tod_qubic = tod[:shape_tod_qubic]
        tod_external = tod[shape_tod_qubic:]
        R = self.get_maps()
        maps_external = R(tod_external)
        for ii, i in enumerate(self.external_nus):
            target = compute_fwhm_to_convolve(self.fwhm[ii], np.max(self.allresolution))
            print(f'Reconvolution from {self.fwhm[ii]} to {np.max(self.allresolution)} with gaussian kernel {target}')
            C = HealpixConvolutionGaussianOperator(fwhm=target)
            maps_external[ii] = C(maps_external[ii])

        tod_external = R.T(maps_external)

        return np.r_[tod_qubic, tod_external]
    def get_tod(self, H, components, convolution, f=None):

        tod = H(components)
        n = self.get_noise(fact=f)
        tod += n.copy()

        if convolution:
            tod = self.reconvolve_to_worst_resolution(tod)


        return H, tod, n
    def normalize(self, tod):

        sh = tod.shape[0]
        sh_external = len(self.external_nus)*3*12*self.nside**2
        sh_1external = 3*12*self.nside**2
        shape_tod_qubic = sh - sh_external
        tod_qubic = tod[:shape_tod_qubic]
        tod_external = tod[shape_tod_qubic:]

        new_tod_qubic = tod_qubic.copy()
        new_tod_external = tod_external.copy()
        new_tod_qubic /= new_tod_qubic.max()
        new_tod_external /= new_tod_external.max()

        return np.r_[new_tod_qubic, new_tod_external]
    def get_invntt_operator(self, fact=None):

        invNq = self.qubic.get_invntt_operator()
        Rq = ReshapeOperator(invNq.shapeout, invNq.shape[0])

        invNe = OtherData(self.external_nus, self.nside, self.comp).get_invntt_operator(fact=fact)
        Re = ReshapeOperator(invNe.shapeout, invNe.shape[0])

        return BlockDiagonalOperator([Rq(invNq(Rq.T)), Re(invNe(Re.T))], axisout=0)



class PipelineReconstruction(QubicOtherIntegrated):

    def __init__(self, qubic, external_nus, comp, nintegr=1, convolution=False):

        self.convolution = convolution
        QubicOtherIntegrated.__init__(self, qubic, external_nus, comp, nintegr)
        self.Hqubic = self.qubic.get_operator(convolution=self.convolution)

    def generate_tod(self, components, beta, f=None):

        H = self.update_H(beta, convolution=self.convolution)
        _, tod, _ = self.get_tod(H, components, convolution=self.convolution, f=f)
        if self.convolution:
            tod = self.reconvolve_to_worst_resolution(tod)

        return H, tod

    def update_H(self, beta, convolution=False):
        return self.get_operator(beta, convolution=convolution, H_qubic=self.Hqubic)

    def get_invN(self, f=None):
        return self.get_invntt_operator(fact=f)


    def call_pcg(self, tod, invN, beta0, tolerance, x0, kmax=5, disp=True, maxiter=10, **kwargs):

        convergence=[1]
        M = get_preconditioner(np.ones(12*self.nside**2))
        print('Starting loop : ')
        k=0
        beta = np.array([beta0])

        while convergence[-1] > tolerance:

            print('k = {}'.format(k+1))
            H = self.update_H(beta[-1], convolution=False)
            A = H.T * invN * H
            b = H.T * invN * tod

            if k == 0:
                solution = pcg(A, b, M=M, x0=x0, tol=tolerance, disp=disp, maxiter=maxiter)
            elif k > kmax-1:
                break
            else:
                solution = pcg(A, b, M=M, x0=mysolution, tol=tolerance, disp=disp, maxiter=maxiter)

            mysolution = solution['x']
            beta_i = self.spectral_index_fitting(x0=beta[-1], args=(solution['x'], tod), **kwargs)
            beta = np.append(beta, beta_i, axis=0)

            k+=1

        return mysolution, beta


    def myChi2(self, beta, solution, data):
        H_for_beta = self.update_H(beta, convolution=False)
        fakedata = H_for_beta(solution)
        fakedata_norm = self.normalize(fakedata)
        print(beta)
        return np.sum((fakedata_norm - self.normalize(data))**2)


    def spectral_index_fitting(self, x0, args, **kwargs):

        r = minimize(self.myChi2, x0=x0, args=args, **kwargs).x

        return r






###
