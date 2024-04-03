# general imports
import healpy as hp
import numpy as np
import pysm3
import pysm3.units as u
from pysm3 import utils
import pickle
import warnings
warnings.filterwarnings("ignore")

# Qubic imports
from qubic.data import PATH
from qubic.lib.Instrument.Qinstrument import QubicInstrument, QubicMultibandInstrument, compute_freq
from qubic.lib.Qsamplings import get_pointing
from qubic.lib.Qscene import QubicScene
import qubic.lib.Qcomponent_model as c
from fgbuster.mixingmatrix import MixingMatrix

from pysimulators import *
from pyoperators import *
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator


def polarized_I(m, nside, polarization_fraction=0.01):
    
    polangle = hp.ud_grade(hp.read_map(PATH+'psimap_dust90_512.fits'), nside)
    depolmap = hp.ud_grade(hp.read_map(PATH+'gmap_dust90_512.fits'), nside)
    cospolangle = np.cos(2.0 * polangle)
    sinpolangle = np.sin(2.0 * polangle)
    #print(depolmap.shape)
    P_map = polarization_fraction * depolmap * hp.ud_grade(m, nside)
    return P_map * np.array([cospolangle, sinpolangle])
def create_array(name, nus, nside):

    if name == 'noise':
        shape=(2, 12*nside**2, 3)
    else:
        shape=len(nus)
    pkl_file = open(PATH+'AllDataSet_Components_MapMaking.pkl', 'rb')
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
def give_cl_cmb(r=0, Alens=1.):
    power_spectrum = hp.read_cl(PATH+'Cls_Planck2018_lensed_scalar.fits')[:,:4000]
    if Alens != 1.:
        power_spectrum[2] *= Alens
    if r:
        power_spectrum += r * hp.read_cl(PATH+'Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits')[:,:4000]
    return power_spectrum
def rad2arcmin(rad):
    return rad / 0.000290888
def circular_mask(nside, center, radius):
    lon = center[0]
    lat = center[1]
    vec = hp.ang2vec(lon, lat, lonlat=True)
    disc = hp.query_disc(nside, vec, radius=np.deg2rad(radius))
    m = np.zeros(hp.nside2npix(nside))
    m[disc] = 1
    return np.array(m, dtype=bool)
def compute_fwhm_to_convolve(allres, target):
    s = np.sqrt(target**2 - allres**2)
    #if s == np.nan:
    #    s = 0
    return s
def find_co(comp, nus_edge):
    return np.sum(nus_edge < comp[-1].nu) - 1
def parse_addition_operator(operator):

    if isinstance(operator, AdditionOperator):
        for op in operator.operands:
            parse_addition_operator(op)

    else:
        parse_composition_operator(operator)
    return operator
def parse_composition_operator(operator):
    for i, op in enumerate(operator.operands):
        if isinstance(op, HealpixConvolutionGaussianOperator):
            operator.operands[i] = HealpixConvolutionGaussianOperator(fwhm=10)
def insert_inside_list(operator, element, position):

    list = operator.operands
    list.insert(position, element)
    return CompositionOperator(list)
def delete_inside_list(operator, position):

    list = operator.operands
    list.pop(position)
    return CompositionOperator(list)
def fill_hwp_position(nsamples, angle):
    ang = np.zeros(nsamples)
    nangle = len(angle)
    x = int(nsamples/nangle)
    print(x)
    
    for ii, i in enumerate(angle):
        ang[x*ii:x*(ii+1)] = i
        
    return ang
def get_allA(nc, nf, npix, beta, nus, comp, active):
    # Initialize arrays to store mixing matrix values
    allA = np.zeros((beta.shape[0], nf, nc))
    allA_pix = np.zeros((npix, nf, nc))

    # Loop through each element of beta to calculate mixing matrix
    for i in range(beta.shape[0]):
        allA[i] = get_mixingmatrix(beta[i], nus, comp, active)

    # Check if beta and npix are equal
    if beta.shape[0] != npix:
        # Upgrade resolution if not equal
        for i in range(nf):
            for j in range(nc):
                allA_pix[:, i, j] = hp.ud_grade(allA[:, i, j], hp.npix2nside(npix))
        # Return upgraded mixing matrix
        return allA_pix
    else:
        # Return original mixing matrix
        return allA
def get_mixing_operator_verying_beta(nc, nside, A):

    #D = DenseBlockDiagonalOperator(A, broadcast='leftward', shapein=(3, 12*nside**2, nc))
    D = BlockDiagonalOperator([DenseBlockDiagonalOperator(A, broadcast='rightward', shapein=(12*nside**2, nc)),
                           DenseBlockDiagonalOperator(A, broadcast='rightward', shapein=(12*nside**2, nc)),
                           DenseBlockDiagonalOperator(A, broadcast='rightward', shapein=(12*nside**2, nc))], new_axisin=0, new_axisout=2)
    return D
def get_mixingmatrix(beta, nus, comp, active=False):
    A = MixingMatrix(*comp)
    if active:
        i = A.components.index('COLine')
        comp[i] = c.COLine(nu=comp[i].nu, active=True)
        A = MixingMatrix(*comp)
        A_ev = A.evaluator(nus)
        if beta.shape[0] == 0:
            A_ev = A_ev()
        else:
            A_ev = A_ev(beta)
            for ii in range(len(comp)):
                #print('ii : ', ii)
                if ii == i:
                    pass
                else:
                    #print('to zero', ii)
                    A_ev[0, ii] = 0
            #print(i, A, A.shape)    
    
    
    else:
        A_ev = A.evaluator(nus)
        if beta.shape[0] == 0:
            A_ev = A_ev()
        else:
            A_ev = A_ev(beta)
        try:
            
            i = A.components.index('COLine')
            A_ev[0, i] = 0
        except:
            pass
    return A_ev
def get_mixing_operator(beta, nus, comp, nside, Amm=None, active=False):
    
    """
    This function returns a mixing operator based on the input parameters: beta and nus.
    The mixing operator is either a constant operator, or a varying operator depending on the input.
    """

    nc = len(comp)
    if beta.shape[0] != 1 and beta.shape[0] != 2:
        
        nside_fit = hp.npix2nside(beta.shape[0])
    else:
        nside_fit = 0

    # Check if the length of beta is equal to the number of channels minus 1
    if nside_fit == 0: # Constant indice on the sky
        #beta = np.mean(beta)

        # Get the mixing matrix
        if Amm is None:
            A = get_mixingmatrix(beta, nus, comp, active)
        else:
            A = np.array([Amm]).copy()
        
        # Get the shape of the mixing matrix
        _, nc = A.shape
        
        # Create a ReshapeOperator
        R = ReshapeOperator(((1, 12*nside**2, 3)), ((12*nside**2, 3)))
        
        # Create a DenseOperator with the first row of A
        D = DenseOperator(A[0], broadcast='rightward', shapein=(nc, 12*nside**2, 3), shapeout=(1, 12*nside**2, 3))

    else: # Varying indice on the sky
        
        # Get all A matrices nc, nf, npix, beta, nus, comp
        A = get_allA(nc, 1, 12*nside**2, beta, nus, comp, active)
        
        # Get the varying mixing operator
        D = get_mixing_operator_verying_beta(nc, nside, A)

    return D

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

        #print('Acq -> ', self.comm)

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
        nsamplings = self.sampling.comm.allreduce(len(self.sampling))
        #nsamplings = self.comm.allreduce(len(self.sampling))
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
        #self.sampling.comm.Allreduce(MPI.IN_PLACE, as_mpi(hit), op=MPI.SUM)
        self.comm.Allreduce(MPI.IN_PLACE, as_mpi(hit), op=MPI.SUM)
        return hit

    def get_noise(self, det_noise, photon_noise, seed=None, out=None):
        
        np.random.seed(seed)
        out = self.instrument.get_noise(
            self.sampling, self.scene, det_noise, photon_noise, out=out)
        if self.effective_duration is not None:
            #nsamplings = self.comm.allreduce(len(self.sampling))
            nsamplings = self.sampling.comm.allreduce(len(self.sampling))
            
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

    def get_invntt_operator(self, det_noise, photon_noise):

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
        self.sigma = 0
        if det_noise is not None:
            self.sigma = self.instrument.detector.nep / np.sqrt(2 * self.sampling.period)
        #print(self.sigma)
        if photon_noise:
            sigma_photon = self.instrument._get_noise_photon_nep(self.scene) / np.sqrt(2 * self.sampling.period)
            self.sigma = np.sqrt(self.sigma ** 2 + sigma_photon ** 2)
            #print(sigma_photon[0])
        else:
            pass
                # sigma_photon = 0

        if self.bandwidth is None and self.psd is None and self.sigma is None:
            raise ValueError('The noise model is not specified.')


        #print('In acquisition.py: self.forced_sigma={}'.format(self.forced_sigma))
        #print('and self.sigma is:{}'.format(self.sigma))
        if self.forced_sigma is None:
            pass#print('Using theoretical TES noises')
        else:
            #print('Using self.forced_sigma as TES noises')
            self.sigma = self.forced_sigma.copy()

        shapein = (len(self.instrument), len(self.sampling))

        if self.bandwidth is None and self.instrument.detector.fknee == 0:
            #print('diagonal case')

            out = DiagonalOperator(1 / self.sigma ** 2, broadcast='rightward',
                                   shapein=(len(self.instrument), len(self.sampling)))
            #print(out.shape)
            #print(out)

            if self.effective_duration is not None:
                #nsamplings = self.comm.allreduce(len(self.sampling))
                nsamplings = self.sampling.comm.allreduce(len(self.sampling))
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

        #self.warnings(d)
        if d['MultiBand'] and d['nf_sub']>1:
            self.subacqs = [QubicAcquisition(multiinstrument[i], sampling, scene, d) for i in range(len(multiinstrument))]
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

        q = QubicInstrument(d1, FRBW=self[0].instrument.FRBW)
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
class QubicPlanckMultiBandAcquisition:

    def __init__(self, qubic, planck):

        self.qubic = qubic
        #self.Nsamples = self.qubic.Nsamples
        #self.Ndets = self.qubic.Ndets
        self.type = self.qubic.kind
        self.Nsub = self.qubic.Nsub
        self.Nrec = self.qubic.Nrec
        self.scene = self.qubic.scene
        #self.final_fwhm = self.qubic.final_fwhm
        self.planck = planck
        self.nside = self.scene.nside

        #self.nueff = self.qubic.nueff
        self.allnus = self.qubic.allnus

    def sed_for_scaling(self, nus, comp, beta):
        sed = MixingMatrix(comp).evaluator(nus)(beta)
        return sed

    def _loop(self, irec):

        pl = []
        for jrec in range(int(len(self.nueff))):
            print(irec, jrec)
            pl += [self._planck_correction(irec, jrec)]
        
        return pl


    def _planck_correction(self, indice1, indice2):

        R_planck = ReshapeOperator((12*self.qubic.scene.nside**2, 3), (12*self.qubic.scene.nside**2*3))
        if indice1 == indice2:
            return R_planck
        else:
            return R_planck * 0

    def _add_planck_contribution(self, list, ifp):
        
        
        for irec in range(int(len(self.nueff)/2)):
            for jrec in range(int(len(self.nueff))):
                if ifp == 0: f = 0
                else: f = int(len(self.nueff)/2)
                if irec+f == jrec : list += [R_planck]
                else: list += [0*R_planck]

        return list

    def get_operator(self, convolution=False, myfwhm=None):
        
        if self.type == 'QubicIntegrated':   # Classic intrument
            
            # Get QUBIC operator
            H_qubic = self.qubic.get_operator(convolution=convolution, myfwhm=myfwhm)
            R_qubic = ReshapeOperator(H_qubic.operands[0].shapeout, H_qubic.operands[0].shape[0])
            R_planck = ReshapeOperator((12*self.qubic.scene.nside**2, 3), (12*self.qubic.scene.nside**2*3))

            # Create an empty list to hold operators
            full_operator = []

            if self.Nrec == 1:
                Operator = [R_qubic(H_qubic), R_planck]
                return BlockColumnOperator(Operator, axisout=0)
            
            else:
                
                for irec in range(self.Nrec):
                    Operator = [R_qubic(H_qubic.operands[irec])]
                    for jrec in range(self.Nrec):
                        if irec == jrec:
                            Operator += [R_planck]
                        else:
                            Operator += [R_planck*0]
                        
                    full_operator += [BlockColumnOperator(Operator, axisout=0)]
                
                return BlockRowOperator(full_operator, new_axisin=0)

        
        elif self.type == 'Wide':      # WideBand intrument

            # Get QUBIC operator
            H_qubic = self.qubic.get_operator(convolution=convolution, myfwhm=myfwhm)
            R_qubic = ReshapeOperator(H_qubic.operands[0].shapeout, H_qubic.operands[0].shape[0])
            R_planck = ReshapeOperator((12*self.qubic.scene.nside**2, 3), (12*self.qubic.scene.nside**2*3))
            

            if self.Nrec == 1:
                operator = [R_qubic(H_qubic), R_planck, R_planck]
                return BlockColumnOperator(operator, axisout=0)
            
            else:

                full_operator = []
                for irec in range(self.Nrec):
                    operator = [R_qubic(H_qubic.operands[irec])]
                    for jrec in range(self.Nrec):
                        if irec == jrec:
                            operator += [R_planck]
                        else:
                            operator += [R_planck*0]
                    full_operator += [BlockColumnOperator(operator, axisout=0)]
                
                return BlockRowOperator(full_operator, new_axisin=0)


    def get_invntt_operator(self, det_noise, photon_noise, weight_planck=1, beam_correction=None, seenpix=None, mask=None):
        

        if beam_correction is None :
                beam_correction = [0]*self.Nrec

        if self.type == 'Wide':

            photon_noise150, photon_noise220 = photon_noise
            invn_q = self.qubic.get_invntt_operator(det_noise, photon_noise150, photon_noise220)
            R = ReshapeOperator(invn_q.shapeout, invn_q.shape[0])
            invn_q = [R(invn_q(R.T))]


            invntt_planck143 = weight_planck*self.planck[0].get_invntt_operator(beam_correction=beam_correction[0], mask=mask, seenpix=seenpix)
            invntt_planck217 = weight_planck*self.planck[1].get_invntt_operator(beam_correction=beam_correction[0], mask=mask, seenpix=seenpix)
            R_planck = ReshapeOperator(invntt_planck143.shapeout, invntt_planck143.shape[0])
            invN_143 = R_planck(invntt_planck143(R_planck.T))
            invN_217 = R_planck(invntt_planck217(R_planck.T))
            if self.Nrec == 1:
                invNe = [invN_143, invN_217]
            else:
                invNe = [invN_143]*int(self.Nrec/2) + [invN_217]*int(self.Nrec/2)
            invN = invn_q + invNe
            return BlockDiagonalOperator(invN, axisout=0)


            
        
        elif self.type == 'QubicIntegrated':
            if beam_correction is None :
                beam_correction = [0]*self.Nrec
            else:
                if type(beam_correction) is not list:
                    raise TypeError('Beam correction should be a list')
                if len(beam_correction) != self.Nrec:
                    raise TypeError('List of beam correction should have Nrec elements')


            invntt_qubic = self.qubic.get_invntt_operator(det_noise, photon_noise)
            R_qubic = ReshapeOperator(invntt_qubic.shapeout, invntt_qubic.shape[0])
            Operator = [R_qubic(invntt_qubic(R_qubic.T))]

            for i in range(self.Nrec):
                invntt_planck = weight_planck*self.planck.get_invntt_operator(beam_correction=beam_correction[i], mask=mask, seenpix=seenpix)
                R_planck = ReshapeOperator(invntt_planck.shapeout, invntt_planck.shape[0])
                Operator.append(R_planck(invntt_planck(R_planck.T)))

            return BlockDiagonalOperator(Operator, axisout=0)

class PlanckAcquisition:

    def __init__(self, band, scene):
        if band not in (30, 44, 70, 143, 217, 353):
            raise ValueError("Invalid band '{}'.".format(band))
        self.scene = scene
        self.band = band
        self.nside = self.scene.nside
        
        if band == 30:
            filename = 'Variance_Planck30GHz_Kcmb2_ns256.fits'
            var=np.zeros((12*self.scene.nside**2, 3))
            for i in range(3):
                var[:, i] = hp.ud_grade(hp.fitsfunc.read_map(filename, field=i), self.scene.nside)
            sigma = 1e6 * np.sqrt(var)
        elif band == 44:
            filename = 'Variance_Planck44GHz_Kcmb2_ns256.fits'
            var=np.zeros((12*self.scene.nside**2, 3))
            for i in range(3):
                var[:, i] = hp.ud_grade(hp.fitsfunc.read_map(filename, field=i), self.scene.nside)
            sigma = 1e6 * np.sqrt(var)
        elif band == 70:
            filename = 'Variance_Planck70GHz_Kcmb2_ns256.fits'
            var=np.zeros((12*self.scene.nside**2, 3))
            for i in range(3):
                var[:, i] = hp.ud_grade(hp.fitsfunc.read_map(filename, field=i), self.scene.nside)
            sigma = 1e6 * np.sqrt(var)
        elif band == 143:
            filename = 'Variance_Planck143GHz_Kcmb2_ns256.fits'
            self.var = np.array(FitsArray(PATH + filename))
            sigma = 1e6 * np.sqrt(self.var)
        elif band == 217:
            filename = 'Variance_Planck217GHz_Kcmb2_ns256.fits'
            self.var = np.array(FitsArray(PATH + filename))
            sigma = 1e6 * np.sqrt(self.var)
        else:
            filename = 'Variance_Planck353GHz_Kcmb2_ns256.fits'
            var=np.zeros((12*self.scene.nside**2, 3))
            for i in range(3):
                var[:, i] = hp.ud_grade(hp.fitsfunc.read_map(filename, field=i), self.scene.nside)
            sigma = 1e6 * np.sqrt(var)

        


        if scene.kind == 'I':
            sigma = sigma[:, 0]
        elif scene.kind == 'QU':
            sigma = sigma[:, :2]
        if self.nside != 256:
            sigma = np.array(hp.ud_grade(sigma.T, self.nside, power=2),
                             copy=False).T
        self.sigma = sigma

    
    def get_operator(self, nintegr=1):
        Hp = DiagonalOperator(np.ones((12*self.nside**2, 3)), broadcast='rightward',
                                shapein=self.scene.shape, shapeout=np.ones((12*self.nside**2, 3)).ravel().shape)


        if nintegr == 1 :
            return Hp

    def get_invntt_operator(self, beam_correction=0, mask=None, seenpix=None):
        
        if beam_correction != 0:
            factor = (4*np.pi*(np.rad2deg(beam_correction)/2.35/np.degrees(hp.nside2resol(self.scene.nside)))**2)
            #print(f'corrected by {factor}')
            varnew = hp.smoothing(self.var.T, fwhm=beam_correction/np.sqrt(2)) / factor
            self.sigma = 1e6 * np.sqrt(varnew.T)
        
        if mask is not None:
            for i in range(3):
                self.sigma[:, i] /= mask.copy()
                #print(self.sigma[seenpix, i])
                #print(len(mask[seenpix]), mask[seenpix])
        #if seenpix is not None:
        #    myweight = 1 / (self.sigma[seenpix] ** 2)
        #else:
        myweight = 1 / (self.sigma ** 2)
        
        return DiagonalOperator(myweight, broadcast='leftward',
                                shapein=myweight.shape)

    def get_noise(self, seed):
        state = np.random.get_state()
        np.random.seed(seed)
        out = np.random.standard_normal(np.ones((12*self.nside**2, 3)).shape) * self.sigma
        np.random.set_state(state)
        return out
    
    def get_map(self, nu_min, nu_max, Nintegr, sky_config, d, fwhm = None):

        print(f'Integration from {nu_min:.2f} to {nu_max:.2f} GHz with {Nintegr} steps')
        obj = QubicIntegrated(d, Nsub=Nintegr, Nrec=Nintegr)
        if Nintegr == 1:
            allnus = np.array([np.mean([nu_min, nu_max])])
        else:
            allnus = np.linspace(nu_min, nu_max, Nintegr)
        m = obj.get_PySM_maps(sky_config, nus=allnus)
    
        if fwhm is None:
            fwhm = [0]*Nintegr
        
        for i in range(Nintegr):
            C = HealpixConvolutionGaussianOperator(fwhm=fwhm)
            m[i] = C(m[i])
    
        return np.mean(m, axis=0)

class QubicIntegrated(QubicPolyAcquisition):

    def __init__(self, d, Nsub=1, Nrec=1):

        """
        
        The initialization method allows to compute basic parameters such as :

            - self.allnus    : array of all frequency used for the operators
            - self.nueff     : array of the effective frequencies

        """

        self.d = d
        self.d['nf_sub']=Nsub
        self.d['nf_recon']=Nrec
        self.Nsub = Nsub
        self.Nrec = Nrec
        self.fact = int(self.Nsub / self.Nrec)
        self.kind = 'QubicIntegrated'
        if self.Nrec == 1 and self.Nsub == 1:
            self.integration = ''
        else:
            self.integration = 'Trapeze'
        
        self.sampling = get_pointing(self.d)
        self.scene = QubicScene(self.d)

        self.multiinstrument = QubicMultibandInstrument(self.d)

        if self.d['nf_sub'] > 1:
            QubicPolyAcquisition.__init__(self, self.multiinstrument, self.sampling, self.scene, self.d)
        else:
            self.subacqs = [QubicAcquisition(self.multiinstrument[0], self.sampling, self.scene, self.d)]

        if self.integration == 'Trapeze':
            _, _, self.nueff, _, _, _ = compute_freq(self.d['filter_nu'], Nfreq=self.Nrec, relative_bandwidth=self.d['filter_relative_bandwidth'])
            
        else:
            _, self.nus_edge, self.nueff, _, _, _ = compute_freq(self.d['filter_nu'], Nfreq=(self.d['nf_recon']))
        
        self.nside = self.scene.nside
        self.allnus = np.array([q.filter.nu / 1e9 for q in self.multiinstrument])
        #self.subacqs = [QubicAcquisition(self.multiinstrument[i], self.sampling, self.scene, self.d) for i in range(len(self.multiinstrument))]
        
        ############
        ### FWHM ###
        ############
        
        for a in self.subacqs[1:]:
            a.comm = self.subacqs[0].comm
        
        self.allfwhm = np.zeros(len(self.multiinstrument))
        for i in range(len(self.multiinstrument)):
            self.allfwhm[i] = self.subacqs[i].get_convolution_peak_operator().fwhm


        self.final_fwhm = np.zeros(self.d['nf_recon'])
        fact = int(self.d['nf_sub']/self.d['nf_recon'])
        for i in range(self.d['nf_recon']):
            self.final_fwhm[i] = np.mean(self.allfwhm[int(i*fact):int(fact*(i+1))])
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

        q = QubicInstrument(d1, FRBW=q0.FRBW)
        q.detector = q0.detector
        #s_ = self.sampling
        #nsamplings = self.multiinstrument[0].comm.allreduce(len(s_))

        d1['random_pointing'] = True
        d1['sweeping_pointing'] = False
        d1['repeat_pointing'] = False
        d1['RA_center'] = 0.
        d1['DEC_center'] = 0.
        d1['npointings'] = self.d['npointings']
        d1['dtheta'] = 15.
        d1['period'] = self.d['period']

        # s = create_random_pointings([0., 0.], nsamplings, 10., period=s_.period)
        a = QubicAcquisition(q, self.sampling, self.scene, d1)
        return a, q
    def get_noise(self, det_noise, photon_noise, seed=None):

        """
        
        Method which compute the noise of QUBIC.

        """
        np.random.seed(seed)
        a, _ = self._get_average_instrument_acq()
        return a.get_noise(det_noise=det_noise, photon_noise=photon_noise, seed=seed)
    def get_TOD(self, skyconfig, beta, convolution=False, myfwhm=None, noise=False, bandpass_correction=False):

        """
        
        Method which allow to compute QUBIC TOD for a given skyconfig according to a given beta.

        """

        s = Sky(skyconfig, self)
        m_nu = s.scale_component(beta)
        sed = s.get_SED(beta)


        ### Compute operator with Nsub acqusitions
        array = self._get_array_of_operators(convolution=convolution, myfwhm=myfwhm)
        h = BlockRowOperator(array, new_axisin=0)

        if self.Nsub == 1:
            tod = h(m_nu[0])
        else:
            tod = h(m_nu)

        if noise:
            n = self.get_noise()
            tod += n.copy()
 
        if bandpass_correction:
            print('Bandpass correction')
            #print(s.map_ref, beta)
            if s.map_ref is None or beta == None :
                raise TypeError('Check that map_ref or sed are not set to None')
            
            if s.is_cmb:
                sed = np.array([sed[:, 1].T])
                #print(sed.shape)
            tod = self.bandpass_correction(h, tod, s.map_ref[s.i_dust], sed)
        
        return tod
    def bandpass_correction(self, H, tod, map_ref, sed):
        
        fact = int(self.Nsub / self.Nrec)
        k = 0
        modelsky = np.zeros((len(self.allnus), 12*self.nside**2, 3))
        
        for i in range(3):
            #print(sed.shape, np.array([map_ref[:, i]]).shape)
            modelsky[:, :, i] = sed.T @ np.array([map_ref[:, i]])
        for irec in range(self.Nrec):
            delta = modelsky[fact*irec:(irec+1)*fact] - np.mean(modelsky[fact*irec:(irec+1)*fact], axis=0)
            for jfact in range(fact):
                delta_tt = H.operands[k](delta[jfact])
                tod -= delta_tt
                k+=1

        return tod
    def _get_array_of_operators(self, convolution=False, myfwhm=None):
        
        op = []
        
        # Loop through each acquisition in subacqs
        k=0
        for ia, a in enumerate(self.subacqs):
            
            ###################
            ### Convolution ###
            ###################

            if convolution:
                # Calculate the convolution operator for this sub-acquisition
                allfwhm = self.allfwhm.copy()
                target = allfwhm[ia]
                if myfwhm is not None:
                    target = myfwhm[ia]
                C = HealpixConvolutionGaussianOperator(fwhm=target, lmax=2*self.nside)
            else:
                # If convolution is False, set the operator to an identity operator
                C = IdentityOperator()

            k+=1
            
            op.append(a.get_operator() * C)
    
        return op
    def get_operator(self, convolution=False, myfwhm=None):

        # Initialize an empty list to store the sum of operators for each frequency band
        op_sum = []
    
        # Get an array of operators for all sub-arrays
        op = np.array(self._get_array_of_operators(convolution=convolution, myfwhm=myfwhm))
        #print('done')
        # Loop over the frequency bands
        op_sum=[]
        for irec in range(self.Nrec):
            imin = irec*self.fact
            imax = (irec+1)*self.fact - 1

            op_sum += [op[(self.allnus >= self.allnus[imin]) * (self.allnus <= self.allnus[imax])].sum(axis=0)]
            
        return BlockRowOperator(op_sum, new_axisin=0)# * MPIDistributionIdentityOperator(self.d['comm'])
    def get_coverage(self):
        return self.subacqs[0].get_coverage()
    def get_invntt_operator(self, det_noise, photon_noise):
        # Get the inverse noise variance covariance matrix from the first sub-acquisition
        #invN = self.subacqs[0].get_invntt_operator(det_noise, photon_noise)
        #return invN

        # Get the inverse noise variance covariance matrix from the first sub-acquisition
        #invN = self.subacqs[0].get_invntt_operator(det_noise, photon_noise)
        #_, a = self._get_average_instrument_acq()
        sigma = self.subacqs[0].instrument.detector.nep / np.sqrt(self.d['period'] * 2)
        sigma_photon = self.subacqs[0].instrument._get_noise_photon_nep(self.scene) / np.sqrt(self.d['period'] * 2)

        if det_noise == True and photon_noise == True:
            sig = np.sqrt(sigma**2 + sigma_photon**2)
        elif det_noise == True and photon_noise == False:
            sig = sigma
        elif det_noise == False and photon_noise == True:
            sig = sigma_photon.copy()
        else:
            sig = sigma
        
        nsamplings = self.sampling.comm.allreduce(len(self.sampling))
        out = DiagonalOperator(1 / sig**2, broadcast='rightward', shapein=(len(self.subacqs[0].instrument.detector), len(self.sampling)))
        out /= (nsamplings * self.sampling.period / (self.d['effective_duration'] * 31557600))



        return out
class QubicFullBandSystematic(QubicPolyAcquisition):

    def __init__(self, d, Nsub, Nrec=1, comp=[], kind='Two', nu_co=None, H=None, effective_duration150=3, effective_duration220=3):
        
        #if Nsub % 2 != 0:
        #    raise TypeError('Nsub should not be odd')

        if Nrec > 1 and len(comp) > 0:
            raise TypeError('For Components Map-Making, there must be Nrec = 1')
        
        self.d = d
        self.comp = comp
        self.Nsub = int(Nsub/2)
        self.kind = kind
        self.Nrec = Nrec
        self.nu_co = nu_co
        self.effective_duration150=effective_duration150
        self.effective_duration220=effective_duration220
        
        if self.kind == 'Two' and self.Nrec == 1 and len(self.comp) == 0:
            raise TypeError('Dual band instrument can not reconstruct one band')

        if self.kind == 'Two': self.number_FP = 2
        elif self.kind == 'Wide': self.number_FP = 1

        
        #self.relative_bandwidth = relative_bandwidth

        if Nsub < 2:
            raise TypeError('You should use Nsub > 1')
        
        self.d['nf_sub'] = self.Nsub
        self.d['nf_recon'] = 1
        
        self.nu_down = 131.25
        self.nu_up = 247.5

        self.nu_average = np.mean(np.array([self.nu_down, self.nu_up]))
        self.d['filter_nu'] = self.nu_average * 1e9
        
        _, allnus150, _, _, _, _ = compute_freq(150, Nfreq=self.Nsub-1, relative_bandwidth=0.25)
        _, allnus220, _, _, _, _ = compute_freq(220, Nfreq=self.Nsub-1, relative_bandwidth=0.25)
        self.allnus = np.array(list(allnus150) + list(allnus220))
        #print(self.nu_average, self.allnus)

        self.multiinstrument = QubicMultibandInstrument(self.d)
       #print(self.multiinstrument.subinstruments)
       # stop
        self.sampling = get_pointing(self.d)
        self.scene = QubicScene(self.d)

        self.Proj = []
        self.subacqs = []
        
        self.H = []
        self.subacqs = [QubicAcquisition(self.multiinstrument[i], self.sampling, self.scene, self.d) for i in range(len(self.multiinstrument))]
        
        self.allfwhm = np.zeros(len(self.multiinstrument))
        for i in range(len(self.multiinstrument)):
            self.allfwhm[i] = self.subacqs[i].get_convolution_peak_operator().fwhm
        
        if nu_co is not None:
            #dmono = self.d.copy()
            self.d['filter_nu'] = nu_co * 1e9
            sampling = qubic.get_pointing(self.d)
            scene = qubic.QubicScene(self.d)
            instrument_co = QubicInstrument(self.d)
            self.multiinstrument.subinstruments += [instrument_co]
            self.Proj += [QubicAcquisition(self.multiinstrument[-1], sampling, scene, self.d).get_projection_operator()]
            self.subacqs += [QubicAcquisition(self.multiinstrument[-1], sampling, scene, self.d)]
        
        QubicPolyAcquisition.__init__(self, self.multiinstrument, self.sampling, self.scene, self.d)
        
        if H is None:
            self.H = [self.subacqs[i].get_operator() for i in range(len(self.subacqs))]
        else:
            self.H = H
        #print(self.d['nprocs_instrument'])
        #stop
        if self.d['nprocs_instrument'] != 1:
            self.mpidist = self.H[0].operands[-1]

        self.ndets = len(self.subacqs[0].instrument)
        self.nsamples = len(self.sampling)
        
        self.coverage = self.subacqs[0].get_coverage()
        

    def get_hwp_operator(self, angle_hwp):
        """
        Return the rotation matrix for the half-wave plate.

        """
        return Rotation3dOperator('X', -4 * angle_hwp,
                                  degrees=True, shapein=self.Proj[0].shapeout)
    def get_components_operator(self, beta, nu, Amm=None, active=False):
        
        if beta.shape[0] != 0 and beta.shape[0] != 1 and beta.shape[0] != 2:
            r = ReshapeOperator((12*self.scene.nside**2, 1, 3), (12*self.scene.nside**2, 3))
        else:
            r = ReshapeOperator((1, 12*self.scene.nside**2, 3), (12*self.scene.nside**2, 3))
        return  r * get_mixing_operator(beta, nu, self.comp, self.scene.nside, Amm=Amm, active=active)
    def sum_over_band(self, h, gain=None):
        op_sum = []
        f = int(2*self.Nsub / self.Nrec)
        
        
        ### Frequency Map-Making
        if len(self.comp) == 0:
            h = np.array(h)
            for irec in range(self.Nrec):
                imin = irec*f
                imax = (irec+1)*f-1
                op_sum += [h[(self.allnus >= self.allnus[imin]) * (self.allnus <= self.allnus[imax])].sum(axis=0)]
            
            if self.kind == 'wide':
                return BlockRowOperator(op_sum, new_axisin=0)
            else:
                if self.Nrec > 2:
                    return BlockDiagonalOperator([BlockRowOperator(op_sum[:int(self.Nrec/2)], new_axisin=0),
                                                  BlockRowOperator(op_sum[int(self.Nrec/2):int(self.Nrec)], new_axisin=0)], axisout=0)
                else:
                    return ReshapeOperator((2, self.ndets, self.nsamples), (2*self.ndets, self.nsamples)) * \
                           BlockDiagonalOperator([BlockRowOperator(op_sum[:int(self.Nrec/2)], new_axisin=0),
                                                  BlockRowOperator(op_sum[int(self.Nrec/2):int(self.Nrec)], new_axisin=0)], new_axisin=0)

                       
        
        ### Components Map-Making
        else:
            if self.kind == 'wide':
                if gain is None:
                    G = DiagonalOperator(np.ones(self.ndets), broadcast='rightward', shapein=(self.ndets, self.nsamples))
                else:
                    G = DiagonalOperator(gain, broadcast='rightward', shapein=(self.ndets, self.nsamples))
                return G * AdditionOperator(h)
            else:
                if gain is None:
                    G150 = DiagonalOperator(np.ones(self.ndets), broadcast='rightward', shapein=(self.ndets, self.nsamples))
                    G220 = DiagonalOperator(np.ones(self.ndets), broadcast='rightward', shapein=(self.ndets, self.nsamples))
                else:
                    G150 = DiagonalOperator(gain[:, 0], broadcast='rightward', shapein=(self.ndets, self.nsamples))
                    G220 = DiagonalOperator(gain[:, 1], broadcast='rightward', shapein=(self.ndets, self.nsamples))
                return BlockColumnOperator([G150 * AdditionOperator(h[:int(self.Nsub)]), 
                                            G220 * AdditionOperator(h[int(self.Nsub):])], axisout=0)
    def get_operator(self, beta=None, Amm=None, angle_hwp=None, gain=None, fwhm=None):
        
        self.operator = []

        if angle_hwp is None:
            angle_hwp = self.sampling.angle_hwp
        else:
            angle_hwp = fill_hwp_position(self.Proj[0].shapeout[1], angle_hwp)

        #G = DiagonalOperator(g, broadcast='rightward', shapein=(myqubic.Ndets, myqubic.Nsamples))
        for isub in range(self.Nsub*2):

            if beta is None:
                Acomp = IdentityOperator()
            else:
                if Amm is not None:
                    Acomp = self.get_components_operator(beta, np.array([self.allnus[isub]]), Amm=Amm[isub])
                else:
                    Acomp = self.get_components_operator(beta, np.array([self.allnus[isub]]))

            if fwhm is None:
                convolution = IdentityOperator()
            else:
                convolution = HealpixConvolutionGaussianOperator(fwhm=fwhm[isub], lmax=2*self.d['nside'])
            with rule_manager(inplace=True):
                hi = CompositionOperator([
                            self.H[isub], convolution, Acomp])
            
            self.operator.append(hi)

        if self.nu_co is not None:
            
            if beta is None:
                Acomp = IdentityOperator()
            else:
                
                Acomp = self.get_components_operator(beta, np.array([self.nu_co]), active=True)
            distribution = self.subacqs[-1].get_distribution_operator()
            temp = self.subacqs[-1].get_unit_conversion_operator()
            aperture = self.subacqs[-1].get_aperture_integration_operator()
            filter = self.subacqs[-1].get_filter_operator()
            projection = self.Proj[-1]
            #hwp = self.get_hwp_operator(angle_hwp)
            hwp = self.subacqs[-1].get_hwp_operator()
            polarizer = self.subacqs[-1].get_polarizer_operator()
            integ = self.subacqs[-1].get_detector_integration_operator()
            trans = self.multiinstrument[-1].get_transmission_operator()
            trans_atm = self.subacqs[-1].scene.atmosphere.transmission
            response = self.subacqs[-1].get_detector_response_operator()
            if fwhm is None:
                convolution = IdentityOperator()
            else:
                convolution = HealpixConvolutionGaussianOperator(fwhm=fwhm[isub], lmax=2*self.d['nside'])
            with rule_manager(inplace=True):
                hi = CompositionOperator([
                            HomothetyOperator(1 / (2*self.Nsub)), response, trans_atm, trans, integ, polarizer, (hwp * projection),
                            filter, aperture, temp, distribution, convolution, Acomp])
            
            self.operator.append(hi)

            
        H = self.sum_over_band(self.operator, gain=gain)
        
        return H
    def get_invntt_operator(self):
        """
        
        Method to compute the inverse noise covariance matrix in time-domain.

        """
        d150 = self.d.copy()
        d150['filter_nu'] = 150 * 1e9
        d150['effective_duration'] = self.effective_duration150
        ins150 = QubicInstrument(d150)

        d220 = self.d.copy()
        d220['effective_duration'] = self.effective_duration220
        d220['filter_nu'] = 220 * 1e9
        
        ins220 = QubicInstrument(d220)

        subacq150 = QubicAcquisition(ins150, self.sampling, self.scene, d150)
        subacq220 = QubicAcquisition(ins220, self.sampling, self.scene, d220)
        if self.kind == 'two':

            invn150 = subacq150.get_invntt_operator(det_noise=True, photon_noise=True)
            invn220 = subacq220.get_invntt_operator(det_noise=True, photon_noise=True)

            return BlockDiagonalOperator([invn150, invn220], axisout=0)
        
        elif self.kind == 'wide':

            invn150 = subacq150.get_invntt_operator(det_noise=True, photon_noise=True)
            invn220 = subacq220.get_invntt_operator(det_noise=False, photon_noise=True)

            return invn150 + invn220
    def get_PySM_maps(self, config, r=0, Alens=1):

        '''
        
        Read configuration dictionary which contains every components adn the model. 
        
        Example : d = {'cmb':42, 'dust':'d0', 'synchrotron':'s0'}

        The CMB is randomly generated fram specific seed. Astrophysical foregrounds come from PySM 3.
        
        '''

        allmaps = np.zeros((len(config), 12*self.scene.nside**2, 3))
        ell=np.arange(2*self.scene.nside-1)
        mycls = give_cl_cmb(r=r, Alens=Alens)

        for k, kconf in enumerate(config.keys()):
            if kconf == 'cmb':

                np.random.seed(config[kconf])
                cmb = hp.synfast(mycls, self.scene.nside, verbose=False, new=True).T

                allmaps[k] = cmb.copy()
            
            elif kconf == 'dust':

                nu0 = self.comp[k].__dict__['_fixed_params']['nu0']
                sky=pysm3.Sky(nside=self.scene.nside, preset_strings=[config[kconf]], output_unit="uK_CMB")
                #sky.components[0].mbb_index = hp.ud_grade(sky.components[0].mbb_index, 8)
                sky.components[0].mbb_temperature = 20*sky.components[0].mbb_temperature.unit
                #sky.components[0].mbb_index = hp.ud_grade(np.array(sky.components[0].mbb_index), 8)
                mydust=np.array(sky.get_emission(nu0 * u.GHz, None).T * utils.bandpass_unit_conversion(nu0*u.GHz, None, u.uK_CMB))
                    
                allmaps[k] = mydust.copy()
            elif kconf == 'synchrotron':
                nu0 = self.comp[k].__dict__['_fixed_params']['nu0']
                sky=pysm3.Sky(nside=self.scene.nside, preset_strings=[config[kconf]], output_unit="uK_CMB")
                mysync=np.array(sky.get_emission(nu0 * u.GHz, None).T * utils.bandpass_unit_conversion(nu0*u.GHz, None, u.uK_CMB))
                allmaps[k] = mysync.copy()
            elif kconf == 'coline':
                
                #sky = pysm3.Sky(nside=self.nside, preset_strings=['co2'], output_unit="uK_CMB")
                #nu0 = sky.components[0].line_frequency['21'].value
                
                #myco=np.array(sky.get_emission(nu0 * u.GHz, None).T * utils.bandpass_unit_conversion(nu0*u.GHz, None, u.uK_CMB))
                # 10 is for reproduce the PsYM template
                m = hp.ud_grade(hp.read_map(path_to_data+'CO_line.fits') * 10, self.scene.nside)
                #print(self.nside)   
                mP = polarized_I(m, self.scene.nside)
                #print(mP.shape)
                myco = np.zeros((12*self.scene.nside**2, 3))
                myco[:, 0] = m.copy()
                myco[:, 1:] = mP.T.copy()
                allmaps[k] = myco.copy()
            else:
                raise TypeError('Choose right foreground model (d0, s0, ...)')

        #if len(nus) == 1:
        #    allmaps = allmaps[0].copy()
            
        return allmaps

class OtherDataParametric:

    def __init__(self, nus, nside, comp, nintegr=2):
        
        if nintegr == 1:
            raise TypeError('The integration of external data should be greater than 1')
        
        self.nintegr = nintegr
        pkl_file = open(PATH+'AllDataSet_Components_MapMaking.pkl', 'rb')
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
        self.nc = len(self.comp)

        if nintegr == 1:
            self.allnus = self.nus
        else:
            self.allnus = []
            for inu, nu in enumerate(self.nus):
                self.allnus += list(np.linspace(nu-self.bw[inu]/2, nu+self.bw[inu]/2, self.nintegr))
            self.allnus = np.array(self.allnus)
        ### Compute all external nus
    def get_invntt_operator(self, fact=None, mask=None):
        # Create an empty array to store the values of sigma
        allsigma = np.array([])

        # Iterate through the frequency values
        for inu, nu in enumerate(self.nus):
            # Determine the scaling factor for the noise
            if fact is None:
                f=1
            else:
                f=fact[inu]

            # Get the noise value for the current frequency and upsample to the desired nside
            sigma = f * hp.ud_grade(self.dataset['noise{}'.format(nu)].T, self.nside).T

            if mask is not None:
                sigma /= np.array([mask, mask, mask]).T

            # Append the noise value to the list of all sigmas
            allsigma = np.append(allsigma, sigma.ravel())

        # Flatten the list of sigmas and create a diagonal operator
        allsigma = allsigma.ravel().copy()
        invN = DiagonalOperator(1 / allsigma ** 2, broadcast='leftward', shapein=(3*len(self.nus)*12*self.nside**2))

        # Create reshape operator and apply it to the diagonal operator
        R = ReshapeOperator(invN.shapeout, invN.shape[0])
        return R(invN(R.T))
    def get_operator(self, beta, convolution, Amm=None, myfwhm=None, nu_co=None, comm=None):
        R2tod = ReshapeOperator((12*self.nside**2, 3), (3*12*self.nside**2))
        if beta.shape[0] <= 2:
            R = ReshapeOperator((1, 12*self.nside**2, 3), (12*self.nside**2, 3))
        else:
            R = ReshapeOperator((12*self.nside**2, 1, 3), (12*self.nside**2, 3))
        
        Operator=[]
        
        k=0
        for ii, i in enumerate(self.nus):
            ope_i=[]
            for j in range(self.nintegr):
                
                if convolution:
                    if myfwhm is not None:
                        fwhm = myfwhm[ii]
                    else:
                        fwhm = self.fwhm[ii]
                else:
                    fwhm = 0
                #fwhm = fwhm_max if convolution and fwhm_max is not None else (self.fwhm[ii] if convolution else 0)
                C = HealpixConvolutionGaussianOperator(fwhm=fwhm, lmax=2*self.nside)
            
                if Amm is not None:
                    D = get_mixing_operator(beta, np.array([self.allnus[k]]), Amm=Amm[k], comp=self.comp, nside=self.nside, active=False)
                else:
                    D = get_mixing_operator(beta, np.array([self.allnus[k]]), Amm=None, comp=self.comp, nside=self.nside, active=False)
                ope_i += [C * R * D]

                
                k+=1
            
            if i == 217:
                #print('co line')
                if nu_co is not None:
                    Dco = get_mixing_operator(beta, np.array([nu_co]), comp=self.comp, nside=self.nside, active=True)
                    ope_i += [C * R * Dco]

            if comm is not None:
                Operator.append(comm*R2tod(AdditionOperator(ope_i)/self.nintegr))
            else:
                Operator.append(R2tod(AdditionOperator(ope_i)/self.nintegr))
                
        return BlockColumnOperator(Operator, axisout=0)
    def get_noise(self, seed=None, fact=None, seenpix=None):
        state = np.random.get_state()
        np.random.seed(seed)
        out = np.zeros((len(self.nus), self.npix, 3))
        R2tod = ReshapeOperator((len(self.nus), 12*self.nside**2, 3), (len(self.nus)*3*12*self.nside**2))
        for inu, nu in enumerate(self.nus):
            if fact is None:
                f=1
            else:
                f=fact[inu]
            sigma = f * hp.ud_grade(self.dataset['noise{}'.format(nu)].T, self.nside).T
            out[inu] = np.random.standard_normal((self.npix,3)) * sigma
        if seenpix is not None:
            out[:, seenpix, :] = 0
        np.random.set_state(state)
        return R2tod(out)
class JointAcquisitionFrequencyMapMaking:

    def __init__(self, d, kind, Nrec, Nsub):

        self.kind = kind
        self.d = d
        self.Nrec = Nrec
        self.Nsub = Nsub
        #self.qubic = qubic
        self.qubic = QubicFullBandSystematic(self.d, comp=[], Nsub=self.Nsub, Nrec=self.Nrec, kind=self.kind)
        self.scene = self.qubic.scene
        self.pl143 = PlanckAcquisition(143, self.scene)
        self.pl217 = PlanckAcquisition(217, self.scene)


    def get_operator(self, angle_hwp=None, fwhm=None):
        
        if self.kind == 'QubicIntegrated':   # Classic intrument
            
            # Get QUBIC operator
            H_qubic = self.qubic.get_operator(convolution=convolution, myfwhm=myfwhm)
            R_qubic = ReshapeOperator(H_qubic.operands[0].shapeout, H_qubic.operands[0].shape[0])
            R_planck = ReshapeOperator((12*self.qubic.scene.nside**2, 3), (12*self.qubic.scene.nside**2*3))

            # Create an empty list to hold operators
            full_operator = []

            if self.Nrec == 1:
                Operator = [R_qubic(H_qubic), R_planck]
                return BlockColumnOperator(Operator, axisout=0)
            
            else:
                
                for irec in range(self.Nrec):
                    Operator = [R_qubic(H_qubic.operands[irec])]
                    for jrec in range(self.Nrec):
                        if irec == jrec:
                            Operator += [R_planck]
                        else:
                            Operator += [R_planck*0]
                        
                    full_operator += [BlockColumnOperator(Operator, axisout=0)]
                
                return BlockRowOperator(full_operator, new_axisin=0)

        
        elif self.kind == 'wide':      # WideBand intrument

            # Get QUBIC operator
            H_qubic = self.qubic.get_operator(angle_hwp=angle_hwp, fwhm=fwhm)
            R_qubic = ReshapeOperator(H_qubic.operands[0].shapeout, H_qubic.operands[0].shape[0])
            
            R_planck = ReshapeOperator((12*self.qubic.scene.nside**2, 3), (12*self.qubic.scene.nside**2*3))
            

            if self.Nrec == 1:
                operator = [R_qubic(H_qubic), R_planck, R_planck]
                return BlockColumnOperator(operator, axisout=0)
            
            else:

                full_operator = []
                for irec in range(self.Nrec):
                    operator = [R_qubic(H_qubic.operands[irec])]
                    for jrec in range(self.Nrec):
                        if irec == jrec:
                            operator += [R_planck]
                        else:
                            operator += [R_planck*0]
                    full_operator += [BlockColumnOperator(operator, axisout=0)]
                
                return BlockRowOperator(full_operator, new_axisin=0)
            
        elif self.kind == 'two':

            # Get QUBIC operator
            if self.Nrec == 2:
                H_qubic = self.qubic.get_operator(angle_hwp=angle_hwp, fwhm=fwhm).operands[1]
            else:
                H_qubic = self.qubic.get_operator(angle_hwp=angle_hwp, fwhm=fwhm)
            R_qubic = ReshapeOperator(H_qubic.operands[0].shapeout, H_qubic.operands[0].shape[0])
            R_planck = ReshapeOperator((12*self.qubic.scene.nside**2, 3), (12*self.qubic.scene.nside**2*3))
            opefull = []
            for ifp in range(2):
                ope_per_fp = []
                for irec in range(int(self.Nrec/2)):
                    if self.Nrec > 2:
                        operator = [R_qubic * H_qubic.operands[ifp].operands[irec]]
                    else:
                        operator = [R_qubic * H_qubic.operands[ifp]]
                    for jrec in range(int(self.Nrec/2)):
                        if irec == jrec:
                            operator += [R_planck]
                        else:
                            operator += [R_planck*0]
                    ope_per_fp += [BlockColumnOperator(operator, axisout=0)]
                opefull += [BlockRowOperator(ope_per_fp, new_axisin=0)]
            if self.Nrec == 2:
                h = BlockDiagonalOperator(opefull, new_axisin=0)
                _r = ReshapeOperator((h.shapeout[0], h.shapeout[1]), (h.shapeout[0]*h.shapeout[1]))
                return _r * h
            else:
                return BlockDiagonalOperator(opefull, axisout=0)

        else:
            raise TypeError(f'Instrument type {self.kind} is not recognize')
        

    def get_invntt_operator(self, weight_planck=1, beam_correction=None, seenpix=None, mask=None):
        

        if beam_correction is None :
                beam_correction = [0]*self.Nrec

        if self.kind == 'wide':

            invn_q = self.qubic.get_invntt_operator()
            R = ReshapeOperator(invn_q.shapeout, invn_q.shape[0])
            invn_q = [R(invn_q(R.T))]


            invntt_planck143 = weight_planck*self.pl143.get_invntt_operator(beam_correction=beam_correction[0], mask=mask, seenpix=seenpix)
            invntt_planck217 = weight_planck*self.pl217.get_invntt_operator(beam_correction=beam_correction[0], mask=mask, seenpix=seenpix)
            R_planck = ReshapeOperator(invntt_planck143.shapeout, invntt_planck143.shape[0])
            invN_143 = R_planck(invntt_planck143(R_planck.T))
            invN_217 = R_planck(invntt_planck217(R_planck.T))
            if self.Nrec == 1:
                invNe = [invN_143, invN_217]
            else:
                invNe = [invN_143]*int(self.Nrec/2) + [invN_217]*int(self.Nrec/2)
            invN = invn_q + invNe
            return BlockDiagonalOperator(invN, axisout=0)
        
        elif self.kind == 'two':

            invn_q_150 = self.qubic.get_invntt_operator().operands[0]
            invn_q_220 = self.qubic.get_invntt_operator().operands[1]
            R = ReshapeOperator(invn_q_150.shapeout, invn_q_150.shape[0])


            invntt_planck143 = weight_planck*self.pl143.get_invntt_operator(beam_correction=beam_correction[0], mask=mask, seenpix=seenpix)
            invntt_planck217 = weight_planck*self.pl217.get_invntt_operator(beam_correction=beam_correction[0], mask=mask, seenpix=seenpix)
            R_planck = ReshapeOperator(invntt_planck143.shapeout, invntt_planck143.shape[0])
            invN_143 = R_planck(invntt_planck143(R_planck.T))
            invN_217 = R_planck(invntt_planck217(R_planck.T))
            invN = [R(invn_q_150(R.T))]
            for i in range(int(self.Nrec/2)):
                invN += [R_planck(invntt_planck143(R_planck.T))]#, 
                    #R(invn_q_220(R.T)), R_planck(invntt_planck217(R_planck.T))]
            
            invN += [R(invn_q_220(R.T))]

            for i in range(int(self.Nrec/2)):
                invN += [R_planck(invntt_planck217(R_planck.T))]
            
            return BlockDiagonalOperator(invN, axisout=0)

            
        '''
        elif self.kind == 'QubicIntegrated':
            if beam_correction is None :
                beam_correction = [0]*self.Nrec
            else:
                if type(beam_correction) is not list:
                    raise TypeError('Beam correction should be a list')
                if len(beam_correction) != self.Nrec:
                    raise TypeError('List of beam correction should have Nrec elements')


            invntt_qubic = self.qubic.get_invntt_operator(det_noise, photon_noise)
            R_qubic = ReshapeOperator(invntt_qubic.shapeout, invntt_qubic.shape[0])
            Operator = [R_qubic(invntt_qubic(R_qubic.T))]

            for i in range(self.Nrec):
                invntt_planck = weight_planck*self.planck.get_invntt_operator(beam_correction=beam_correction[i], mask=mask, seenpix=seenpix)
                R_planck = ReshapeOperator(invntt_planck.shapeout, invntt_planck.shape[0])
                Operator.append(R_planck(invntt_planck(R_planck.T)))

            return BlockDiagonalOperator(Operator, axisout=0)
        '''

class JointAcquisitionComponentsMapMaking:

    def __init__(self, d, kind, comp, Nsub, nus_external, nintegr, nu_co=None, H=None, ef150=3, ef220=3):

        self.kind = kind
        self.d = d
        self.Nsub = Nsub
        self.comp = comp
        self.nus_external = nus_external
        self.nintegr = nintegr
        #self.qubic = qubic
        self.qubic = QubicFullBandSystematic(self.d, comp=self.comp, Nsub=self.Nsub, Nrec=1, kind=self.kind, nu_co=nu_co, H=H, effective_duration150=ef150, effective_duration220=ef220)
        self.scene = self.qubic.scene
        self.external = OtherDataParametric(self.nus_external, self.scene.nside, self.comp, self.nintegr)

    def get_operator(self, beta, Amm=None, gain=None, fwhm=None, nu_co=None):
        
        if Amm is not None:
            Aq = Amm[:self.Nsub]
            Ap = Amm[self.Nsub:]
        else:
            Aq = None
            Ap = None
        
        Hq = self.qubic.get_operator(beta=beta, gain=gain, fwhm=fwhm, Amm=Aq)
        
        
        Rq = ReshapeOperator(Hq.shapeout, (Hq.shapeout[0]*Hq.shapeout[1]))
        try:
            mpidist = self.qubic.mpidist
        except:
            mpidist = None

        
        He = self.external.get_operator(beta=beta, convolution=False, comm=mpidist, nu_co=nu_co, Amm=Ap)
        
        return BlockColumnOperator([Rq * Hq, He], axisout=0)
    
    def get_invntt_operator(self, fact=None, mask=None):

        invNq = self.qubic.get_invntt_operator()
        R = ReshapeOperator(invNq.shapeout, invNq.shape[0])
        invNe = self.external.get_invntt_operator(fact=fact, mask=mask)

        return BlockDiagonalOperator([R(invNq(R.T)), invNe], axisout=0)


########################################################################################################################################
### code from Mathias Regnier, originally in frequency_acquisition.py

def get_ultrawideband_config():
    
    nu_up = 247.5
    nu_down = 131.25
    nu_ave = np.mean(np.array([nu_up, nu_down]))
    delta = nu_up - nu_ave
    
    return nu_ave, 2*delta/nu_ave

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

class Sky:
    
    def __init__(self, sky_config, qubic, nu0=150):
        
        """
        
        This class allow to compute the sky at different frequency according to a given SED with astrophysical foregrounds.

        """
        self.qubic = qubic
        self.sky_config = sky_config
        self.nside = self.qubic.scene.nside
        self.allnus = self.qubic.allnus
        self.nu0 = nu0

        self.is_cmb = False
        self.is_dust = False
        map_ref = []
        self.comp = []
        k = 0
        for i in self.sky_config.keys():
            if i == 'cmb':
                self.is_cmb = True
                self.cmb = self.get_cmb(self.sky_config[i])
                self.i_cmb = k
                self.comp += [c.CMB()]
                map_ref += [self.cmb]
            elif i == 'dust':
                self.is_dust = True
                self.dust = self.get_dust(self.nu0, self.sky_config[i])
                map_ref += [self.dust]
                self.comp += [c.Dust(nu0=self.nu0, temp=20)]
                self.i_dust = k
            k+=1
        self.map_ref = np.array(map_ref)

        self.A = mm.MixingMatrix(*self.comp).evaluator(self.allnus)

    def get_SED(self, beta=None):

        if beta == None:
            sed = self.A()
        else:
            sed = self.A(beta)

        return sed


        
    def scale_component(self, beta=None):

        m_nu = np.zeros((len(self.allnus), 12*self.nside**2, 3))
        sed = self.get_SED(beta)
        
        if self.is_cmb == True and self.is_dust == True:
            sed = np.array([sed[:, 1]]).T

        if self.is_dust:
            for i in range(3):
                m_nu[:, :, i] = sed @ np.array([self.map_ref[self.i_dust, :, i]])

        if self.is_cmb:
            for i in range(len(self.allnus)):
                m_nu[i] += self.map_ref[self.i_cmb]

        return m_nu
    def get_cmb(self, seed):
        mycls = give_cl_cmb()
        np.random.seed(seed)
        return hp.synfast(mycls, self.nside, verbose=False, new=True).T
    def get_dust(self, nu0, model):
        
        sky=pysm3.Sky(nside=self.nside, preset_strings=[model])
        myfg=np.array(sky.get_emission(nu0 * u.GHz, None).T * utils.bandpass_unit_conversion(nu0*u.GHz, None, u.uK_CMB))
        
        return myfg

class QubicFullBand(QubicPolyAcquisition):


    def __init__(self, d, Nsub, Nrec, relative_bandwidth=0.6138613861386139):

        if Nrec%2 != 0 and Nrec!=1:
            raise TypeError('You should put a number of reconstructed sub-bands divisible by 2')
        self.relative_bandwidth = relative_bandwidth

        if Nsub == 1:
            raise TypeError('You should use Nsub > 1')
        
        self.d = d
        self.Nsub = int(Nsub/2)
        self.Nrec = Nrec
        self.d['nf_sub'] = self.Nsub
        self.d['nf_recon'] = self.Nrec
        self.kind = 'Wide'
        self.number_FP = 1
        
        self.nu_down = 131.25
        self.nu_up = 247.5

        

        #self.nu_average = np.mean(np.array([self.nu_down, self.nu_up]))
        #self.d['filter_nu'] = self.nu_average * 1e9
        if Nsub != 1:
            _, allnus150, _, _, _, _ = compute_freq(150, Nfreq=int(self.Nsub)-1, relative_bandwidth=0.25)
            _, allnus220, _, _, _, _ = compute_freq(220, Nfreq=int(self.Nsub)-1, relative_bandwidth=0.25)
        else:
            _, _, allnus150, _, _, _ = compute_freq(150, Nfreq=int(self.Nsub), relative_bandwidth=0.25)
            _, _, allnus220, _, _, _ = compute_freq(220, Nfreq=int(self.Nsub), relative_bandwidth=0.25)
        self.allnus = np.array(list(allnus150) + list(allnus220))

        self.Nsub *= 2
        
        #self.d['nf_sub'] = 2 * self.d['nf_sub']
        self.multiinstrument = QubicMultibandInstrument(self.d)
        self.sampling = get_pointing(self.d)
        self.scene = QubicScene(self.d)

        self.subacqs = [QubicAcquisition(self.multiinstrument[i], self.sampling, self.scene, self.d) for i in range(len(self.multiinstrument))]
        self.subacqs150 = self.subacqs[:int(self.Nsub/2)]
        self.subacqs220 = self.subacqs[int(self.Nsub/2):self.Nsub]
        
        ### For MPI distribution
        if self.Nsub > 1:
            QubicPolyAcquisition.__init__(self, self.multiinstrument, self.sampling, self.scene, self.d)

        self.allfwhm = np.zeros(len(self.multiinstrument))
        for i in range(len(self.multiinstrument)):
            self.allfwhm[i] = self.subacqs[i].get_convolution_peak_operator().fwhm

        self.nside = self.scene.nside
        self.npix = 12*self.nside**2

        invn = self.get_invntt_operator(True, True, True)
        self.Ndets, self.Nsamples = invn.shapein

    def _get_array_operators(self, convolution=False, myfwhm=None):


        '''
        
        Compute all the Nsub sub-acquisition in one list. Each sub-acquisition contain the instrument specificities and describe the 
        synthetic beam for each frequencies.

        '''

        operator = []
        R = ReshapeOperator((1, 12*self.nside**2, 3), (12*self.nside**2, 3))
        for inu, i in enumerate(self.subacqs):
            if convolution:
                if myfwhm is not None:
                    C = HealpixConvolutionGaussianOperator(fwhm = myfwhm[inu])
                else:
                    C = HealpixConvolutionGaussianOperator(fwhm = self.allfwhm[inu])
            else:
                C = IdentityOperator()

            P = i.get_operator() * C

            operator.append(P)

        self.Ndets, self.Nsamples = operator[0].shapeout

        return operator
    
    def get_operator(self, convolution, myfwhm=None):

        operator = self._get_array_operators(convolution=convolution, myfwhm=myfwhm)
        array_operator = np.array(operator)
        op_sum = []

        f = int(self.Nsub/self.Nrec)

        ### 150 GHz
        for irec in range(self.Nrec):
            imin = irec*f
            imax = (irec+1)*f-1
            op_sum += [array_operator[(self.allnus >= self.allnus[imin]) * (self.allnus <= self.allnus[imax])].sum(axis=0)]

        return BlockRowOperator(op_sum, new_axisin=0)
            
        
    def get_invntt_operator(self, det_noise, photon_noise150, photon_noise220):

        invn150 = self.subacqs150[0].get_invntt_operator(det_noise=False, photon_noise=photon_noise150)
        invn220 = self.subacqs220[0].get_invntt_operator(det_noise=det_noise, photon_noise=photon_noise220)

        return (invn150 + invn220)/2   # factor 2 because it added twice the detector noise
    
    def get_noise(self, det_noise, photon_noise150, photon_noise220, seed=None):
        
        np.random.seed(seed)
        ndet = self.subacqs150[0].get_noise(det_noise, False)
        npho150 = self.subacqs150[0].get_noise(False, photon_noise150)# - ndet
        npho220 = self.subacqs220[0].get_noise(False, photon_noise220)# - ndet
        return ndet, npho150, npho220
    
