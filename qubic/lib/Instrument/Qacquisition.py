import pickle
import warnings

import healpy as hp
import numpy as np
from fgbuster.mixingmatrix import MixingMatrix
from pyoperators import (
    MPI,
    AdditionOperator,
    BlockColumnOperator,
    BlockDiagonalOperator,
    BlockRowOperator,
    CompositionOperator,
    DenseBlockDiagonalOperator,
    DenseOperator,
    DiagonalOperator,
    I,
    IdentityOperator,
    MPIDistributionIdentityOperator,
    Operator,
    PackOperator,
    ReshapeOperator,
    SymmetricBandToeplitzOperator,
    proxy_group,
    rule_manager,
)
from pyoperators.utils.mpi import as_mpi
from pysimulators import (
    Acquisition,
    FitsArray,
)
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
from pysimulators.noises import (
    _fold_psd,
    _gaussian_psd_1f,
    _logloginterp_psd,
    _psd2invntt,
    _unfold_psd,
)

from qubic.data import PATH
from qubic.lib.Instrument.Qinstrument import (
    QubicInstrument,
    QubicMultibandInstrument,
    compute_freq,
)
from qubic.lib.Qsamplings import get_pointing
from qubic.lib.Qscene import QubicScene

warnings.filterwarnings("ignore")


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
        block = d["block"]
        effective_duration = d["effective_duration"]
        photon_noise = d["photon_noise"]
        max_nbytes = d["max_nbytes"]
        psd = d["psd"]
        bandwidth = d["bandwidth"]
        twosided = d["twosided"]
        sigma = d["sigma"]
        self.interp_projection = d["interp_projection"]
        comm = d["comm"]
        nprocs_instrument = d["nprocs_instrument"]
        nprocs_sampling = d["nprocs_sampling"]

        Acquisition.__init__(
            self,
            instrument,
            sampling,
            scene,
            block=block,
            max_nbytes=max_nbytes,
            nprocs_instrument=nprocs_instrument,
            nprocs_sampling=nprocs_sampling,
            comm=comm,
        )

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
        if self.scene.kind != "I":
            out = out[..., 0].copy()  # to avoid keeping QU in memory
        ndetectors = self.comm.allreduce(len(self.instrument))
        nsamplings = self.sampling.comm.allreduce(len(self.sampling))
        out *= ndetectors * nsamplings * self.sampling.period / np.sum(out)
        return out

    def get_hitmap(self, nside=None):  # ?
        """
        Return a healpy map whose values are the number of times a pointing
        hits the pixel.
        """
        if nside is None:
            nside = self.scene.nside
        ipixel = self.sampling.healpix(nside)
        npixel = 12 * nside**2
        hit = np.histogram(ipixel, bins=npixel, range=(0, npixel))[0]
        self.comm.Allreduce(MPI.IN_PLACE, as_mpi(hit), op=MPI.SUM)
        return hit

    def get_noise(self, det_noise, photon_noise, seed=None, out=None):
        np.random.seed(seed)
        out = self.instrument.get_noise(self.sampling, self.scene, det_noise, photon_noise, out=out)
        if self.effective_duration is not None:
            nsamplings = self.sampling.comm.allreduce(len(self.sampling))

            out *= np.sqrt(nsamplings * self.sampling.period / (self.effective_duration * 31557600))
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
            [self.instrument.get_detector_response_operator(self.sampling[b]) for b in self.block],
            axisin=1,
        )

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
            [self.instrument.get_hwp_operator(self.sampling[b], self.scene) for b in self.block],
            axisin=1,
        )

    def get_diag_invntt_operator(self):
        print("Use diagonal noise covariance matrix")

        sigma_detector = self.instrument.detector.nep / np.sqrt(2 * self.sampling.period)
        if self.photon_noise:
            sigma_photon = self.instrument._get_noise_photon_nep(self.scene) / np.sqrt(2 * self.sampling.period)
        else:
            sigma_photon = 0

        out = DiagonalOperator(
            1 / (sigma_detector**2 + sigma_photon**2),
            broadcast="rightward",
            shapein=(len(self.instrument), len(self.sampling)),
        )
        if self.effective_duration is not None:
            nsamplings = self.comm.allreduce(len(self.sampling))
            out /= nsamplings * self.sampling.period / (self.effective_duration * 31557600)
        return out

    def get_invntt_operator(self, det_noise, photon_noise):  # Not working in some cases?
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

        fftw_flag = "FFTW_MEASURE"
        nthreads = None

        # if self.bandwidth is None or self.psd is None:
        if self.bandwidth is None and self.psd is not None or self.bandwidth is not None and self.psd is None:
            raise ValueError("The bandwidth or the PSD is not specified.")

        # Get sigma in Watt
        self.sigma = 0
        if det_noise is not None:
            self.sigma = self.instrument.detector.nep / np.sqrt(2 * self.sampling.period)

        if photon_noise:
            sigma_photon = self.instrument._get_noise_photon_nep(self.scene) / np.sqrt(2 * self.sampling.period)
            self.sigma = np.sqrt(self.sigma**2 + sigma_photon**2)

        else:
            pass

        if self.bandwidth is None and self.psd is None and self.sigma is None:
            raise ValueError("The noise model is not specified.")

        if self.forced_sigma is None:
            pass
        else:
            self.sigma = self.forced_sigma.copy()

        shapein = (len(self.instrument), len(self.sampling))

        if self.bandwidth is None and self.instrument.detector.fknee == 0:
            out = DiagonalOperator(
                1 / self.sigma**2,
                broadcast="rightward",
                shapein=(len(self.instrument), len(self.sampling)),
            )

            if self.effective_duration is not None:
                nsamplings = self.sampling.comm.allreduce(len(self.sampling))
                out /= nsamplings * self.sampling.period / (self.effective_duration * 31557600)
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
            p = _gaussian_psd_1f(
                fftsize,
                sampling_frequency,
                self.sigma,
                self.instrument.detector.fknee,
                self.instrument.detector.fslope,
                twosided=True,
            )
        p[..., 0] = p[..., 1]
        invntt = _psd2invntt(p, new_bandwidth, self.instrument.detector.ncorr, fftw_flag=fftw_flag)

        print("non diagonal case")
        if self.effective_duration is not None:
            nsamplings = self.comm.allreduce(len(self.sampling))
            invntt /= nsamplings * self.sampling.period / (self.effective_duration * 31557600)

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
            H = CompositionOperator(
                [
                    response,
                    trans_inst,
                    integ,
                    polarizer,
                    hwp * projection,
                    filter,
                    aperture,
                    trans_atm,
                    temp,
                    distribution,
                ]
            )
        if self.scene == "QU":
            H = self.get_subtract_grid_operator()(H)
        return H

    def get_polarizer_operator(self):
        """
        Return operator for the polarizer grid.
        """
        return BlockDiagonalOperator(
            [self.instrument.get_polarizer_operator(self.sampling[b], self.scene) for b in self.block],
            axisin=1,
        )

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
                [f(self.sampling[b], self.scene, verbose=verbose, interp_projection=self.interp_projection) for b in self.block],
                axisout=1,
            )

        # XXX HACK
        def callback(i):
            p = f(self.sampling[self.block[i]], self.scene, verbose=False)
            return p

        shapeouts = [(len(self.instrument), s.stop - s.start) + self.scene.shape[1:] for s in self.block]
        proxies = proxy_group(len(self.block), callback, shapeouts=shapeouts)
        return BlockColumnOperator(proxies, axisout=1)

    def get_add_grids_operator(self):
        """Return operator to add signal from detector pairs."""
        nd = len(self.instrument)
        if nd % 2 != 0:
            raise ValueError("Odd number of detectors.")
        partitionin = 2 * (len(self.instrument) // 2,)
        return BlockRowOperator([I, I], axisin=0, partitionin=partitionin)  # ?

    def get_subtract_grids_operator(self):
        """Return operator to subtract signal from detector pairs."""
        nd = len(self.instrument)
        if nd % 2 != 0:
            raise ValueError("Odd number of detectors.")
        partitionin = 2 * (len(self.instrument) // 2,)
        return BlockRowOperator([I, -I], axisin=0, partitionin=partitionin)  # ?

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
            cov_inv[np.isinf(cov_inv)] = 0.0
            preconditioner = DiagonalOperator(cov_inv, broadcast="rightward")
        else:
            preconditioner = None
        return preconditioner


class PlanckAcquisition:
    def __init__(self, band, scene):
        if band not in (30, 44, 70, 143, 217, 353):
            raise ValueError("Invalid band '{}'.".format(band))
        self.scene = scene
        self.band = band
        self.nside = self.scene.nside

        if band == 30:
            filename = "Variance_Planck30GHz_Kcmb2_ns256.fits"
            var = np.zeros((12 * self.scene.nside**2, 3))
            for i in range(3):
                var[:, i] = hp.ud_grade(hp.fitsfunc.read_map(filename, field=i), self.scene.nside)
            sigma = np.sqrt(var)
        elif band == 44:
            filename = "Variance_Planck44GHz_Kcmb2_ns256.fits"
            var = np.zeros((12 * self.scene.nside**2, 3))
            for i in range(3):
                var[:, i] = hp.ud_grade(hp.fitsfunc.read_map(filename, field=i), self.scene.nside)
            sigma = np.sqrt(var)
        elif band == 70:
            filename = "Variance_Planck70GHz_Kcmb2_ns256.fits"
            var = np.zeros((12 * self.scene.nside**2, 3))
            for i in range(3):
                var[:, i] = hp.ud_grade(hp.fitsfunc.read_map(filename, field=i), self.scene.nside)
            sigma = np.sqrt(var)
        elif band == 143:
            filename = "Variance_Planck143GHz_Kcmb2_ns256.fits"
            self.var = np.array(FitsArray(PATH + filename))
            sigma = np.sqrt(self.var)
        elif band == 217:
            filename = "Variance_Planck217GHz_Kcmb2_ns256.fits"
            self.var = np.array(FitsArray(PATH + filename))
            sigma = np.sqrt(self.var)
        else:
            filename = "Variance_Planck353GHz_Kcmb2_ns256.fits"
            var = np.zeros((12 * self.scene.nside**2, 3))
            for i in range(3):
                var[:, i] = hp.ud_grade(hp.fitsfunc.read_map(filename, field=i), self.scene.nside)
            sigma = np.sqrt(var)

        if scene.kind == "I":
            sigma = sigma[:, 0]
        elif scene.kind == "QU":
            sigma = sigma[:, :2]
        if self.nside != 256:
            sigma = np.array(hp.ud_grade(sigma.T, self.nside, power=2), copy=False).T
        self.sigma = sigma * 1e6

    def get_operator(self, nintegr=1):
        Hp = DiagonalOperator(
            np.ones((12 * self.nside**2, 3)),
            broadcast="rightward",
            shapein=self.scene.shape,
            shapeout=np.ones((12 * self.nside**2, 3)).ravel().shape,
        )

        if nintegr == 1:
            return Hp

    def get_invntt_operator(self, beam_correction=0, mask=None, seenpix=None):
        if beam_correction != 0:
            factor = 4 * np.pi * (np.rad2deg(beam_correction) / 2.35 / np.degrees(hp.nside2resol(self.scene.nside))) ** 2
            # print(f'corrected by {factor}')
            varnew = hp.smoothing(self.var.T, fwhm=beam_correction / np.sqrt(2)) / factor
            self.sigma = 1e6 * np.sqrt(varnew.T)

        if mask is not None:
            for i in range(3):
                self.sigma[:, i] /= mask.copy()

        myweight = 1 / (self.sigma**2)

        return DiagonalOperator(myweight, broadcast="leftward", shapein=myweight.shape)

    def get_noise(self, seed):
        state = np.random.get_state()
        np.random.seed(seed)
        out = np.random.standard_normal(np.ones((12 * self.nside**2, 3)).shape) * self.sigma
        np.random.set_state(state)
        return out


class QubicMultiAcquisitions:
    """

    Instance to define the multi-frequency instrument.

    Input : - dictionary : contains QUBIC informations
            - Nsub : Number of sub-bands for integrating the physical bandwidth
            - Nrec : Number of reconstructed maps (in the case of FMM)
            - comps : List of astrophysical components (CMB, Dust, ...)
            - H : List of pointing matrix if not already computed
            - nu_co : Frequency of a line emission

    """

    def __init__(self, dictionary, nsub, nrec, comps=[], H=None, nu_co=None, sampling=None):
        ### Define class arguments
        self.dict = dictionary
        self.nsub = nsub
        self.nrec = nrec
        self.dict["nf_sub"] = self.nsub
        self.comps = comps
        self.fsub = int(self.nsub / self.nrec)

        ### Resolve issue when comm, nprocs_instrument, nprocs_sampling are None in the dictionary.
        # It will define them using codes from Acquisition in pysimulators if they are not defined by the user.
        # When dict["nprocs_instrument"] is None, the test to save MPI communicator at the end of __init__ is passing while it should not.
        comm = self.dict["comm"]
        nprocs_instrument = self.dict["nprocs_instrument"]
        nprocs_sampling = self.dict["nprocs_sampling"]

        if comm is None:
            comm = MPI.COMM_WORLD
        if nprocs_instrument is None and nprocs_sampling is None:
            nprocs_instrument = 1
            nprocs_sampling = comm.size
        elif nprocs_instrument is None:
            if nprocs_sampling < 1 or nprocs_sampling > comm.size:
                raise ValueError(f"Invalid value for nprocs_sampling '{nprocs_sampling}'.")
            nprocs_instrument = comm.size // nprocs_sampling
        else:
            if nprocs_instrument < 1 or nprocs_instrument > comm.size:
                raise ValueError(f"Invalid value for nprocs_instrument '{nprocs_instrument}'.")
            nprocs_sampling = comm.size // nprocs_instrument
        if nprocs_instrument * nprocs_sampling != comm.size:
            raise ValueError("Invalid MPI distribution of the acquisition.")

        self.dict["comm"] = comm
        self.dict["nprocs_instrument"] = nprocs_instrument
        self.dict["nprocs_sampling"] = nprocs_sampling

        # There was code duplication in the previous version
        self.allnus = []
        self.allnus_rec = []
        if self.dict["instrument_type"] == "MB":  # to be implemented on dictionary level
            print("Only the 150 GHz band will be used.")
            f_bands = [150]  # this is for the TD MonoBand instrument, the choice of the band could be implemented at dictionary level
        else:
            f_bands = [150, 220]
        for i, f_band in enumerate(f_bands):
            ### Compute frequencies on the edges
            _, _, nus_subbands_i, _, _, _ = compute_freq(f_band, Nfreq=int(self.nsub / len(f_bands)), relative_bandwidth=self.dict["filter_relative_bandwidth"])

            ### Compute the effective reconstructed frequencies if FMM is applied
            _, _, nus_i, _, _, _ = compute_freq(f_band, Nfreq=int(self.nrec / len(f_bands)), relative_bandwidth=self.dict["filter_relative_bandwidth"])

            ### Joint 150 and 220 GHz band if needed
            self.allnus += list(nus_subbands_i)
            self.allnus_rec += list(nus_i)

        ### Convert lists to numpy arrays
        self.allnus = np.array(self.allnus)
        self.allnus_rec = np.array(self.allnus_rec)

        ### Multi-frequency instrument
        self.multiinstrument = QubicMultibandInstrument(self.dict)
        print(self.multiinstrument)

        if sampling is None:
            self.sampling = get_pointing(self.dict)
        else:
            self.sampling = sampling
        self.scene = QubicScene(self.dict)
        self.npix = 12 * self.scene.nside**2

        ### Compute pointing matrix
        self.subacqs = [QubicAcquisition(self.multiinstrument[i], self.sampling, self.scene, self.dict) for i in range(len(self.multiinstrument))]

        ### CO line emission
        if nu_co is not None:
            dmono = self.dict.copy()
            dmono["filter_nu"] = nu_co * 1e9
            dmono["filter_relative_bandwidth"] = 0.05

            instrument_co = QubicInstrument(dmono, FRBW=0.25)
            self.multiinstrument.subinstruments += [instrument_co]
            self.subacqs += [QubicAcquisition(instrument_co, self.sampling, self.scene, dmono)]

            self.allnus = np.append(self.allnus, nu_co)

        for acq in self.subacqs:
            acq.comm = self.subacqs[0].comm

        ### Angular resolution
        self.allfwhm = np.zeros(len(self.multiinstrument))
        for i in range(len(self.multiinstrument)):
            self.allfwhm[i] = self.subacqs[i].get_convolution_peak_operator().fwhm

        ### Compute the pointing matrix if not already done
        if H is None:
            self.H = [self.subacqs[i].get_operator() for i in range(len(self.subacqs))]
        else:
            self.H = H

        ### Save MPI communicator
        if self.dict["nprocs_instrument"] != 1:
            self.mpidist = self.H[0].operands[-1]
            for i in range(1, len(self.H)):
                self.H[i].operands[-1] = self.mpidist

        ### Define the number of detector and sampling (for each processors)
        self.ndets = len(self.subacqs[0].instrument)
        self.nsamples = len(self.sampling)
        self.coverage = self._get_coverage()

    def _get_coverage(self):
        out = self.H[0].T(np.ones(self.H[0].T.shapein))
        if self.scene.kind != "I":
            out = out[..., 0].copy()
        out *= self.ndets * self.nsamples * self.sampling.period / np.sum(out)
        return out

    def _get_mixing_matrix(self, nus, beta):
        """

        Method to return mixing matrix.

        If beta has shape (ncomp), then the mixing matrix will have shape (nfreq, ncomp).
        If beta has shape (npix, ncomp), the the elements of the mxing matrix vary across the sky, it will have shape (npix, nfreq, ncomp)

        """

        ### Define Mixing Matrix with FGB classes
        mm = MixingMatrix(*self.comps)

        ### Compute them using the eval method at each frequency nus
        mixing_matrix_elements = mm.eval(nus, *beta)

        _sh = mixing_matrix_elements.shape
        if _sh[0] != 1:
            beta = hp.ud_grade(beta, self.scene.nside)
            mixing_matrix_elements = mm.eval(nus, *beta)

            mixing_matrix = np.transpose(mixing_matrix_elements, (1, 0, 2))
        else:
            mixing_matrix = mixing_matrix_elements[0]

        return np.round(mixing_matrix, 10)

    def _get_mixing_operator(self, A):
        """

        Method to define an operator like object for a given frequency nu, the input A should be for one frequency.
        The type of operator depends on the shape of input A.

        """

        if A.ndim == 1:  ### If constant beta across the sky
            r = ReshapeOperator((1, self.npix, 3), (self.npix, 3))
            D = r * DenseOperator(A, broadcast="rightward", shapein=(A.shape[0], self.npix, 3), shapeout=(1, self.npix, 3))

        else:  ### If varying beta across the sky
            r = ReshapeOperator((self.npix, 1, 3), (self.npix, 3))
            _, nc = A.shape

            def reshape_fct(vec, out):
                out[...] = vec.T

            R = Operator(direct=reshape_fct, transpose=reshape_fct, shapein=(nc, self.npix, 3), shapeout=(3, self.npix, nc), flags="linear")

            ### if pixelization of A is lower than the one of components
            if hp.npix2nside(A.shape[0]) != self.scene.nside:
                A = hp.ud_grade(A.T, self.scene.nside).T

            d = DenseBlockDiagonalOperator(A[:, np.newaxis, :], broadcast="rightward", shapein=(self.npix, nc))

            ### Multiply by 3 to create A matrix for I, Q and U
            D = r * BlockDiagonalOperator([d] * 3, new_axisin=0, new_axisout=2) * R

        return D


class QubicInstrumentType(QubicMultiAcquisitions):
    """
    Class providing methods necessary for all instrument types.

    """

    def __init__(self, dictionary, nsub, nrec, comps=[], H=None, nu_co=None, sampling=None):
        QubicMultiAcquisitions.__init__(self, dictionary, nsub=nsub, nrec=nrec, comps=comps, H=H, nu_co=nu_co, sampling=sampling)

        if self.dict["instrument_type"] == "DB":
            self.used_bands = [150, 220]
            self.nFocalPlanes = 2
        elif self.dict["instrument_type"] == "UWB":
            self.used_bands = [150, 220]
            self.nFocalPlanes = 1
        elif self.dict["instrument_type"] == "MB":
            self.used_bands = [150]  # this is the TD MonoBand instrument
            self.nFocalPlanes = 1
        else:
            raise TypeError(f"{self.dict['instrument_type']} is not implemented...")

    def sum_over_band(self, h, algo, gain=None):  # same for DB and UWB sum_over_band for FMM except for the return,
        # similar for CMM, with one band instead of two
        """

        Perform sum over sub-operators depending on the reconstruction algorithms (FMM or CMM)

        """

        ### Frequency Map-Making
        if algo == "FMM":
            op_sum = []
            f = int(self.nsub / self.nrec)
            h = np.array(h)
            for irec in range(self.nrec):
                imin = irec * f
                imax = (irec + 1) * f - 1
                op_sum += [h[(self.allnus >= self.allnus[imin]) * (self.allnus <= self.allnus[imax])].sum(axis=0)]

            block_list = []
            for iband in range(self.nFocalPlanes):
                edges_band = [iband * int(self.nrec // self.nFocalPlanes), (iband + 1) * int(self.nrec // self.nFocalPlanes)]  # splitting nrec op
                block_list.append(BlockRowOperator(op_sum[edges_band[0] : edges_band[1]], new_axisin=0))

            operator_H = BlockDiagonalOperator(block_list, new_axisout=0)

            return (
                ReshapeOperator(operator_H.shapeout, (self.nFocalPlanes * self.ndets * self.nsamples))
                * operator_H
                * ReshapeOperator(
                    (self.nrec, self.npix, h[0].shapein[-1]),
                    (operator_H.shapein),  # this reshape ensures that it works even for nrec=2
                )
            )

        ### Components Map-Making
        else:
            Operator_list = []
            for iband in range(self.nFocalPlanes):
                if gain is None:
                    gain_ = np.ones(self.ndets)
                else:
                    if len(gain.shape) < 2:
                        gain_ = gain[:]
                    else:
                        gain_ = gain[:, iband]

                G_band = DiagonalOperator(gain_, broadcast="rightward", shapein=(self.ndets, self.nsamples))

                edges_band = [iband * int(self.nsub // self.nFocalPlanes), (iband + 1) * int(self.nsub // self.nFocalPlanes)]  # splitting nsub h
                Operator_list.append(G_band * AdditionOperator(h[edges_band[0] : edges_band[1]]))

            return BlockColumnOperator(Operator_list, axisout=0)

    def get_operator(self, A=None, gain=None, fwhm=None):  # exactly the same for DB and UWB get_operator except for lmax=2 * self.dict["nside"] (which should be the same anyway?)
        """

        Method to generate the pointing matrix.

        mixing_matrix : array like containing mixing matrix elements. If the elements of the mixing matrix are constant across the sky,
                        mixing_matrix.shape = (nfreq, ncomp)

        """
        self.operator = []

        for isub in range(self.nsub):
            ### Compute mixing matrix operator if mixing matrix is provided
            if A is None:
                Acomp = IdentityOperator()
                algo = "FMM"
            else:
                Acomp = self._get_mixing_operator(A=A[isub])
                algo = "CMM"

            ### Compute gaussian kernel to account for angular resolution
            if fwhm is None:
                convolution = IdentityOperator()
            else:
                convolution = HealpixConvolutionGaussianOperator(fwhm=fwhm[isub], lmax=3 * self.scene.nside - 1)

            ### Compose operator as H = Proj * C * A
            with rule_manager(inplace=True):
                hi = CompositionOperator([self.H[isub], convolution, Acomp])

            self.operator.append(hi)

        ### Do the sum over operators depending on the reconstruction model
        H = self.sum_over_band(self.operator, gain=gain, algo=algo)

        return H

    def get_invntt_operator(self):  # DB and UWB had the same get_invntt_operator except from the return and the det_noise=False in 220 band
        """

        Method to compute the inverse noise covariance matrix in time-domain.

        """

        if self.dict["instrument_type"] == "UWB":
            det_noise = [True, False]
        else:  # doesn't matter for MB, because only the first is used anyway
            det_noise = [True, True]
        invn_list = []
        for iband, band in enumerate(self.used_bands):
            d = self.dict.copy()
            d["filter_nu"] = band * 1e9
            d["effective_duration"] = self.dict["effective_duration{}".format(band)]
            inst = QubicInstrument(d)
            subacq = QubicAcquisition(inst, self.sampling, self.scene, d)
            invn_list.append(subacq.get_invntt_operator(det_noise=det_noise[iband], photon_noise=True))
        self.invn150 = invn_list[0]  # used in PresetAcquisition.get_approx_hth
        if self.dict["instrument_type"] == "UWB":
            self.invN = np.sum(invn_list)
        else:
            self.invN = BlockDiagonalOperator(invn_list, axisout=0)
        return self.invN


class OtherDataParametric:
    def __init__(self, nus, nside, comps, nintegr=2):
        self.nintegr = nintegr
        pkl_file = open(PATH + "AllDataSet_Components_MapMaking.pkl", "rb")
        dataset = pickle.load(pkl_file)
        self.dataset = dataset

        self.nus = nus
        self.nside = nside
        self.npix = 12 * self.nside**2
        self.bw = []
        for _, i in enumerate(self.nus):
            if nintegr == 1:
                self.bw.append(0)
            else:
                self.bw.append(self.dataset["bw{}".format(i)])

        self.fwhm = np.deg2rad(self.create_array("fwhm", self.nus, self.nside) / 60.0)
        self.comps = comps
        self.nc = len(self.comps)

        if nintegr == 1:
            self.allnus = self.nus
        else:
            self.allnus = []
            for inu, nu in enumerate(self.nus):
                self.allnus += list(np.linspace(nu - self.bw[inu] / 2, nu + self.bw[inu] / 2, self.nintegr))
            self.allnus = np.array(self.allnus)

    def create_array(self, name, nus, nside):
        if name == "noise":
            shape = (2, 12 * nside**2, 3)
        else:
            shape = len(nus)
        pkl_file = open(PATH + "AllDataSet_Components_MapMaking.pkl", "rb")
        dataset = pickle.load(pkl_file)

        myarray = np.zeros(shape)

        for ii, i in enumerate(nus):
            myarray[ii] = dataset[name + str(i)]

        return myarray

    def _get_mixing_matrix(self, nus, beta):
        """

        Method to return mixing matrix.

        If beta has shape (ncomp), then the mixing matrix will have shape (nfreq, ncomp).
        If beta has shape (npix, ncomp), the the elements of the mxing matrix vary across the sky, it will have shape (npix, nfreq, ncomp)

        """

        ### Define Mixing Matrix with FGB classes
        mm = MixingMatrix(*self.comps)

        ### Compute them using the eval method at each frequency nus
        mixing_matrix_elements = mm.eval(nus, *beta)

        _sh = mixing_matrix_elements.shape
        if _sh[0] != 1:
            beta = hp.ud_grade(beta, self.nside)
            mixing_matrix_elements = mm.eval(nus, *beta)

            mixing_matrix = np.transpose(mixing_matrix_elements, (1, 0, 2))
        else:
            mixing_matrix = mixing_matrix_elements[0]

        return np.round(mixing_matrix, 6)

    def _get_mixing_operator(self, A):
        """

        Method to define an operator like object for a given frequency nu, the input A should be for one frequency.
        The type of operator depends on the shape of input A.

        """

        if A.ndim == 1:  ### If constant beta across the sky
            r = ReshapeOperator((1, self.npix, 3), (self.npix, 3))
            D = r * DenseOperator(A, broadcast="rightward", shapein=(A.shape[0], self.npix, 3), shapeout=(1, self.npix, 3))

        else:  ### If varying beta across the sky
            r = ReshapeOperator((self.npix, 1, 3), (self.npix, 3))
            _, nc = A.shape

            def reshape_fct(vec, out):
                out[...] = vec.T

            R = Operator(direct=reshape_fct, transpose=reshape_fct, shapein=(nc, self.npix, 3), shapeout=(3, self.npix, nc), flags="linear")

            ### if pixelization of A is lower than the one of components
            if hp.npix2nside(A.shape[0]) != self.nside:
                A = hp.ud_grade(A.T, self.nside).T

            d = DenseBlockDiagonalOperator(A[:, np.newaxis, :], broadcast="rightward", shapein=(self.npix, nc))

            ### Multiply by 3 to create A matrix for I, Q and U
            D = r * BlockDiagonalOperator([d] * 3, new_axisin=0, new_axisout=2) * R

        return D

    def get_invntt_operator(self, fact=None, mask=None):
        # Create an empty array to store the values of sigma
        allsigma = np.array([])

        # Iterate through the frequency values
        for inu, nu in enumerate(self.nus):
            # Determine the scaling factor for the noise
            if fact is None:
                f = 1
            else:
                f = fact[inu]

            # Get the noise value for the current frequency and upsample to the desired nside
            sigma = f * hp.ud_grade(self.dataset["noise{}".format(nu)].T, self.nside).T

            if mask is not None:
                sigma /= np.array([mask, mask, mask]).T

            # Append the noise value to the list of all sigmas
            allsigma = np.append(allsigma, sigma.ravel())

        # Flatten the list of sigmas and create a diagonal operator
        allsigma = allsigma.ravel().copy()
        invN = DiagonalOperator(1 / allsigma**2, broadcast="leftward", shapein=(3 * len(self.nus) * 12 * self.nside**2))

        # Create reshape operator and apply it to the diagonal operator
        R = ReshapeOperator(invN.shapeout, invN.shape[0])
        return R(invN(R.T))

    def get_operator(self, A, convolution, myfwhm=None, nu_co=None, comm=None):
        R2tod = ReshapeOperator((12 * self.nside**2, 3), (3 * 12 * self.nside**2))

        Operator = []

        k = 0
        for ii, _ in enumerate(self.nus):
            ope_i = []
            for _ in range(self.nintegr):
                if convolution:
                    if myfwhm is not None:
                        fwhm = myfwhm[ii]
                    else:
                        fwhm = self.fwhm[ii]
                else:
                    fwhm = 0

                C = HealpixConvolutionGaussianOperator(fwhm=fwhm, lmax=3 * self.nside - 1)

                D = self._get_mixing_operator(A=A[k])

                ope_i += [C * D]

                k += 1

            if comm is not None:
                Operator.append(comm * R2tod(AdditionOperator(ope_i) / self.nintegr))
            else:
                Operator.append(R2tod(AdditionOperator(ope_i) / self.nintegr))

        return BlockColumnOperator(Operator, axisout=0)

    def get_noise(self, seed=None, fact=None, seenpix=None):
        state = np.random.get_state()
        np.random.seed(seed)
        out = np.zeros((len(self.nus), self.npix, 3))
        R2tod = ReshapeOperator((len(self.nus), 12 * self.nside**2, 3), (len(self.nus) * 3 * 12 * self.nside**2))

        for inu, nu in enumerate(self.nus):
            if fact is None:
                f = 1
            else:
                f = fact[inu]
            sigma = f * hp.ud_grade(self.dataset["noise{}".format(nu)].T, self.nside).T
            out[inu] = np.random.standard_normal((self.npix, 3)) * sigma
        if seenpix is not None:
            out[:, seenpix, :] = 0
        np.random.set_state(state)
        return R2tod(out)


class JointAcquisitionFrequencyMapMaking:
    def __init__(self, d, Nrec, Nsub, H=None):
        self.d = d
        self.Nrec = Nrec
        self.Nsub = Nsub

        ### Select the instrument model
        self.qubic = QubicInstrumentType(self.d, self.Nsub, self.Nrec, comps=[], H=H, nu_co=None)

        self.scene = self.qubic.scene
        self.pl143 = PlanckAcquisition(143, self.scene)
        self.pl217 = PlanckAcquisition(217, self.scene)

    def get_operator(self, fwhm=None, seenpix=None):
        ### nstokes is hardcoded to nstokes = 3
        ### We could retrieve it in the shape of H if we want to implement a different nstokes case
        # nstokes = H_qubic.shapein[-1] # might not work for nstokes = 1
        nstokes = 3

        ### The operator that allows the focus on seenpix:
        ### shapein : (self.Nrec, sum(seenpix), nstokes)
        ### shapeout: (self.Nrec, npix, nstokes)
        if seenpix is not None:
            U = (
                ReshapeOperator((self.Nrec * sum(seenpix) * nstokes), (self.Nrec, sum(seenpix), nstokes))
                * PackOperator(np.broadcast_to(seenpix[None, :, None], (self.Nrec, seenpix.size, nstokes)).copy())
            ).T
        else:
            U = IdentityOperator()

        ### Get QUBIC H operator
        H_qubic = self.qubic.get_operator(fwhm=fwhm)

        R_planck = ReshapeOperator((12 * self.qubic.scene.nside**2, nstokes), (12 * self.qubic.scene.nside**2 * nstokes))
        H_planck_ = BlockDiagonalOperator([R_planck] * self.Nrec, new_axisout=0)
        # It is necessary to change the shape of H_planck_ in order to stack it with H_qubic
        R_diag = ReshapeOperator(H_planck_.shapeout, H_planck_.shape[0])
        H_planck = R_diag(H_planck_)

        H_list = [H_qubic]
        ### Doing the BlockDiagonal H_planck line by line in order to stack it with H_qubic in a BlockColumnOperator
        H_list += [H_planck]
        return BlockColumnOperator(H_list, axisout=0) * U

    def get_invntt_operator(  # We stack the invN_qubic and invN_planck on top of eachother
        self, weight_planck=1, beam_correction=None, seenpix=None, mask=None
    ):
        if beam_correction is None:
            beam_correction = [0] * self.Nrec

        invn_q = self.qubic.get_invntt_operator()
        R = ReshapeOperator(invn_q.shapeout, invn_q.shape[0])
        invn_q = [R(invn_q(R.T))]

        invntt_planck143 = weight_planck * self.pl143.get_invntt_operator(beam_correction=beam_correction[0], mask=mask, seenpix=seenpix)
        invntt_planck217 = weight_planck * self.pl217.get_invntt_operator(beam_correction=beam_correction[0], mask=mask, seenpix=seenpix)
        R_planck = ReshapeOperator(invntt_planck143.shapeout, invntt_planck143.shape[0])
        invN_143 = R_planck(invntt_planck143(R_planck.T))
        invN_217 = R_planck(invntt_planck217(R_planck.T))
        if self.Nrec == 1:
            invNe = [invN_143, invN_217]
        else:
            invNe = [invN_143] * int(self.Nrec / 2) + [invN_217] * int(self.Nrec / 2)
        invN = invn_q + invNe
        return BlockDiagonalOperator(invN, axisout=0)


class JointAcquisitionComponentsMapMaking:
    def __init__(self, d, comp, Nsub, nus_external, nintegr, nu_co=None, H=None):
        self.d = d
        self.Nsub = Nsub
        self.comp = comp
        self.nus_external = nus_external
        self.nintegr = nintegr

        ### Select the instrument model
        self.qubic = QubicInstrumentType(self.d, self.Nsub, nrec=2, comps=self.comp, H=H, nu_co=nu_co)

        self.scene = self.qubic.scene
        self.external = OtherDataParametric(self.nus_external, self.scene.nside, self.comp, self.nintegr)
        self.allnus = np.array(list(self.qubic.allnus) + list(self.external.allnus))

    def get_operator(self, A, gain=None, fwhm=None, nu_co=None):
        Aq = A[: self.Nsub]
        Ap = A[self.Nsub :]

        Hq = self.qubic.get_operator(A=Aq, gain=gain, fwhm=fwhm)
        Rq = ReshapeOperator(Hq.shapeout, (Hq.shapeout[0] * Hq.shapeout[1]))

        try:
            mpidist = self.qubic.mpidist
        except Exception:
            mpidist = None

        He = self.external.get_operator(A=Ap, convolution=True, myfwhm=fwhm, comm=mpidist, nu_co=nu_co)

        return BlockColumnOperator([Rq * Hq, He], axisout=0)

    def get_invntt_operator(self, fact=None, mask=None):
        invNq = self.qubic.get_invntt_operator()
        R = ReshapeOperator(invNq.shapeout, invNq.shape[0])
        invNe = self.external.get_invntt_operator(fact=fact, mask=mask)

        return BlockDiagonalOperator([R(invNq(R.T)), invNe], axisout=0)
