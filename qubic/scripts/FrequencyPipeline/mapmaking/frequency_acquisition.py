#########################################################################################################################
#########################################################################################################################
########                                                                                                         ########
########      This file contain the acquisition models to perform the frequency map-making for QUBIC. We can     ########
########      choose to simulate one FP of the instrument, two FP or the WideBand instrument.                    ########
########                                                                                                         ########
#########################################################################################################################
#########################################################################################################################


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
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
path = os.path.dirname(os.getcwd()) + '/data/'
import gc
import time
import warnings
warnings.filterwarnings("ignore")

import mapmaking.instrument as instr

import pysm3.units as u
from importlib import reload
from pysm3 import utils
# FG-Buster packages
import fgb.component_model as c
import fgb.mixing_matrix as mm
import pickle
from scipy.optimize import minimize

# PyOperators stuff
from pysimulators import *
from pyoperators import *
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

def get_ultrawideband_config():
    
    nu_up = 247.5
    nu_down = 131.25
    nu_ave = np.mean(np.array([nu_up, nu_down]))
    delta = nu_up - nu_ave
    
    return nu_ave, 2*delta/nu_ave

def find_folder_recursively(folder_name, start_path=os.path.expanduser("~/Desktop")):
    for root, dirs, files in os.walk(start_path):
        if folder_name in dirs:
            return os.path.join(root, folder_name)
    raise FileNotFoundError(f"{folder_name} not found.")

import pyoperators
from pyoperators import *

__all__ = ['QubicAcquisition',
           'PlanckAcquisition',
           'QubicPlanckAcquisition',
           'QubicPolyAcquisition',
           'QubicMultiBandAcquisition',
           'QubicPlanckMultiBandAcquisition',
           'QubicAcquisitionTwoBands',
           'QubicMultiBandAcquisitionTwoBands',
           'QubicIntegrated',
           'QubicTwoBands',
           'QubicWideBand',
           'QubicOtherIntegrated',
           'PlanckAcquisitionComponentsMapMaking',
           'QubicPlanckAcquisitionComponentsMapMaking',
           'QubicIntegratedComponentsMapMaking',
           'QubicWideBandComponentsMapMaking',
           'QubicTwoBandsComponentsMapMaking']

def create_array(name, nus, nside):

    if name == 'noise':
        shape=(2, 12*nside**2, 3)
    else:
        shape=len(nus)
    pkl_file = open(path+'AllDataSet_Components_MapMaking.pkl', 'rb')
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
    power_spectrum = hp.read_cl(path+'Cls_Planck2018_lensed_scalar.fits')[:,:4000]
    if Alens != 1.:
        power_spectrum[2] *= Alens
    if r:
        power_spectrum += r * hp.read_cl(path+'Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits')[:,:4000]
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
        sed = mm.MixingMatrix(comp).evaluator(nus)(beta)
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
        
        self.sampling = qubic.get_pointing(self.d)

        self.scene = qubic.QubicScene(self.d)

        self.multiinstrument = instr.QubicMultibandInstrument(self.d)

        if self.d['nf_sub'] > 1:
            QubicPolyAcquisition.__init__(self, self.multiinstrument, self.sampling, self.scene, self.d)
        else:
            self.subacqs = [QubicAcquisition(self.multiinstrument[0], self.sampling, self.scene, self.d)]

        if self.integration == 'Trapeze':
            _, _, self.nueff, _, _, _ = qubic.compute_freq(self.d['filter_nu'], Nfreq=self.Nrec, relative_bandwidth=self.d['filter_relative_bandwidth'])
            
        else:
            _, self.nus_edge, self.nueff, _, _, _ = qubic.compute_freq(self.d['filter_nu'], Nfreq=(self.d['nf_recon']))
        
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

        q = instr.QubicInstrument(d1, FRBW=q0.FRBW)
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


'''
class QubicTwoBands:

    def __init__(self, qubic150, qubic220):

        self.qubic150 = qubic150
        self.qubic220 = qubic220
        self.scene = self.qubic150.scene
        self.qubic150.d['noiseless'] = self.qubic220.d['noiseless']
        self.Nsub = self.qubic150.Nsub*2
        self.Nrec = self.qubic150.Nrec*2
        self.type = 'TwoBands'
        self.final_fwhm = np.array([])
        self.final_fwhm = np.append(self.final_fwhm, self.qubic150.final_fwhm)
        self.final_fwhm = np.append(self.final_fwhm, self.qubic220.final_fwhm)
        self.qubic150.d['photon_noise'] = self.qubic220.d['photon_noise']
        self.Nsamples = self.qubic150.Nsamples
        self.Ndets = 2*992

        self.nueff = np.array([])

        self.nueff = np.array(list(self.qubic150.nueff)+list(self.qubic220.nueff))
        self.allnus = np.array(list(self.qubic150.allnus)+list(self.qubic220.allnus))
        self.allfwhm = np.array(list(self.qubic150.allfwhm)+list(self.qubic220.allfwhm))

    def get_TOD(self, skyconfig, beta, convolution=False, myfwhm=None, noise=False, bandpass_correction=False):

        tod150 = self.qubic150.get_TOD(skyconfig, beta, convolution, myfwhm, noise=False, bandpass_correction=bandpass_correction)
        tod220 = self.qubic220.get_TOD(skyconfig, beta, convolution, myfwhm, noise=False, bandpass_correction=bandpass_correction)
        tod = np.r_[tod150, tod220]

        if noise:
            n = self.get_noise()
            tod += n.copy()
        
        return tod
        

    def get_operator(self, convolution):

        """
        
        This method compute the reconstruction operator for the TwoBands instrument.

        """

        self.H150 = self.qubic150.get_operator(convolution=convolution)
        self.H220 = self.qubic220.get_operator(convolution=convolution)

        ope = [self.H150, self.H220]

        if self.qubic150.d['nf_recon'] == 1:
            hh = BlockDiagonalOperator(ope, new_axisout=0)
            R = ReshapeOperator(hh.shapeout, (hh.shapeout[0]*hh.shapeout[1], hh.shapeout[2]))
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

    def __init__(self, qubic150, qubic220):

        self.qubic150 = qubic150
        self.qubic220 = qubic220
        self.scene = self.qubic150.scene
        self.qubic150.d['noiseless'] = self.qubic220.d['noiseless']
        self.Nsub = self.qubic150.Nsub*2
        self.Nrec = self.qubic150.Nrec*2
        self.type = 'WideBand'
        self.final_fwhm = np.array([])
        self.final_fwhm = np.append(self.final_fwhm, self.qubic150.final_fwhm)
        self.final_fwhm = np.append(self.final_fwhm, self.qubic220.final_fwhm)
        self.qubic150.d['photon_noise'] = self.qubic220.d['photon_noise']
        self.photon_noise = self.qubic150.d['photon_noise']
        self.Nsamples = self.qubic150.Nsamples
        self.Ndets = 992

        self.nueff = np.array(list(self.qubic150.nueff)+list(self.qubic220.nueff))
        self.allnus = np.array(list(self.qubic150.allnus)+list(self.qubic220.allnus))
        self.allfwhm = np.array(list(self.qubic150.allfwhm)+list(self.qubic220.allfwhm))
    def get_operator(self, convolution=False, myfwhm=None):

        """
        
        Method to compute the reconstruction operator for the WideBand instrument.

        """
        self.H150 = self.qubic150.get_operator(convolution=convolution, myfwhm=myfwhm)
        self.H220 = self.qubic220.get_operator(convolution=convolution, myfwhm=myfwhm)
        operator = [self.H150, self.H220]

        ### Nrec = 1
        if self.qubic150.Nrec == 1:
            return BlockRowOperator(operator, new_axisin=0)
        ### Nrec > 1
        else:
            return BlockRowOperator(operator, axisin=0)
        
    def get_TOD(self, skyconfig, beta, convolution=False, myfwhm=None, noise=False, bandpass_correction=False):

        tod150 = self.qubic150.get_TOD(skyconfig, beta, convolution, myfwhm, noise=False, bandpass_correction=bandpass_correction)
        tod220 = self.qubic220.get_TOD(skyconfig, beta, convolution, myfwhm, noise=False, bandpass_correction=bandpass_correction)
        tod = tod150 + tod220

        if noise:
            n = self.get_noise()
            tod += n.copy()
        
        return tod
        
    def get_noise(self):

        """
        
        Method to compute the noise vector for the WideBand instrument.
        It assumes only one contribution of the noise detectors but the sum of the photon noise of each separated focal plane.

        """

        detector_noise = self.qubic150.multiinstrument[0].get_noise_detector(self.qubic150.sampling)
        #print(self.qubic150.sampling.period, self.qubic150.d['effective_duration'])
        nsamplings = self.qubic150.sampling.comm.allreduce(len(self.qubic150.sampling))
        fact = np.sqrt(nsamplings * self.qubic150.sampling.period / (self.qubic150.d['effective_duration'] * 31557600))

        if self.photon_noise:
            photon_noise150 = self.qubic150.multiinstrument[0].get_noise_photon(self.qubic150.sampling, self.qubic150.scene)
            photon_noise220 = self.qubic220.multiinstrument[0].get_noise_photon(self.qubic220.sampling, self.qubic220.scene)
            photon_noise = photon_noise150 + photon_noise220

            return fact * (detector_noise + photon_noise)
        else:
            return fact * detector_noise

    def get_invntt_operator(self):

        """
        
        Method to compute the inverse noise covariance matrix in time-domain.

        """

        sigma = self.qubic150.multiinstrument[0].detector.nep / np.sqrt(2 * self.qubic150.sampling.period)
        sigma_photon150 = self.qubic150.multiinstrument[0]._get_noise_photon_nep(self.qubic150.scene) / np.sqrt(2 * self.qubic150.sampling.period)
        sigma_photon220 = self.qubic220.multiinstrument[0]._get_noise_photon_nep(self.qubic220.scene) / np.sqrt(2 * self.qubic220.sampling.period)

        sig = np.sqrt(sigma**2 + sigma_photon150**2 + sigma_photon220**2)

        f = (len(self.qubic150.sampling) * self.qubic150.sampling.period / (self.qubic150.d['effective_duration'] * 31557600))
        out =  DiagonalOperator(1 / sig ** 2, broadcast='rightward',
                                   shapein=(len(self.qubic150.multiinstrument[0]), len(self.qubic150.sampling))) / f


        return out#self.qubic150.get_invntt_operator() + self.qubic220.get_invntt_operator()

    def get_coverage(self):
        return self.qubic150.get_coverage()
class QubicUltraWideBand(QubicIntegrated):

    def __init__(self, d, Nsub, Nrec):


        ### Warnings
        if Nsub == 1:
            raise TypeError('The number of Nsub should be above 1')
        
        if Nsub%2 != 0:
            raise TypeError('The number of bandpass integration should be divisible by an integer')
        
        Nsub = int(Nsub/2)

        self.d150 = d.copy()
        self.d220 = d.copy()
        self.Nsub = Nsub
        self.Nrec = Nrec
        self.fact_sub = int(2*Nsub/Nrec)
        self.d150['filter_nu'] = 150 * 1e9
        self.d220['filter_nu'] = 220 * 1e9
        
        ### Qubic acquisitions
        self.qubic150 = QubicIntegrated(self.d150, Nsub=self.Nsub, Nrec=self.Nrec)
        self.qubic220 = QubicIntegrated(self.d220, Nsub=self.Nsub, Nrec=self.Nrec)
        self.scene = self.qubic150.scene
        
        self.nu_down = 131.25
        self.nu_up = 247.5
        self.nu_ave = np.mean(np.array([self.nu_up, self.nu_down]))

        self.allnus = np.array(list(self.qubic150.allnus) + list(self.qubic220.allnus))
        self.allfwhm = np.array(list(self.qubic150.allfwhm) + list(self.qubic220.allfwhm))

        _, _, self.nueff, _, _, _ = qubic.compute_freq(self.nu_ave, self.Nrec, (self.nu_up - self.nu_down)/self.nu_ave)

        self.subacqs = self.qubic150.subacqs + self.qubic220.subacqs


        """
        
        if self.Nsub == 1 and self.Nrec == 1:
            _, _, self.allnus, _, _, _ = qubic.compute_freq(self.nu_ave, self.Nsub, (self.nu_up - self.nu_down)/self.nu_ave)
        else:
            _, self.allnus, _, _, _, _ = qubic.compute_freq(self.nu_ave, self.Nsub-1, (self.nu_up - self.nu_down)/self.nu_ave)
        ### Scene
        self.scene = self.qubic.scene

        

        if self.bandpass is None:
            self.bandpass = np.ones(self.Nsub)

        if self.Nsub != len(self.bandpass):
            raise TypeError(f'Bandpass should have {len(self.bandpass)} values.')


        ### Effective frequencies
        self.nueff = self.qubic.nueff

        ### All frequencies
        self.allnus = self.qubic.allnus

        ### All FWHM (rad)
        self.allfwhm = self.qubic.allfwhm
        """

    def _get_average_instrument(self):
        
        """
        Create and return a QubicAcquisition instance of a monochromatic
            instrument with frequency correspondent to the mean of the
            frequency range.
        """
        
        #if len(self) == 1:
        #    return self[0]

        ### 150 GHz
        q150 = self.qubic150.multiinstrument[0]
        
        nu150_min = self.qubic150.multiinstrument[0].filter.nu
        nu150_max = self.qubic150.multiinstrument[-1].filter.nu
        
        nep150 = q150.detector.nep
        fknee150 = q150.detector.fknee
        fslope150 = q150.detector.fslope

        d1 = self.qubic150.d.copy()
        d1['filter_nu'] = (nu150_max + nu150_min) / 2.
        d1['filter_relative_bandwidth'] = (nu150_max - nu150_min) / ((nu150_max + nu150_min) / 2.)
        d1['detector_nep'] = nep150
        d1['detector_fknee'] = fknee150
        d1['detector_fslope'] = fslope150
        ins150 = instr.QubicInstrument(d1, FRBW=q150.FRBW)


        ### 220 GHz

        q220 = self.qubic220.multiinstrument[0]
        nu220_min = self.qubic220.multiinstrument[0].filter.nu
        nu220_max = self.qubic220.multiinstrument[-1].filter.nu

        nep220 = q220.detector.nep
        fknee220 = q220.detector.fknee
        fslope220 = q220.detector.fslope

        d1 = self.qubic220.d.copy()
        d1['filter_nu'] = (nu220_max + nu220_min) / 2.
        d1['filter_relative_bandwidth'] = (nu220_max - nu220_min) / ((nu220_max + nu220_min) / 2.)
        d1['detector_nep'] = nep220
        d1['detector_fknee'] = fknee220
        d1['detector_fslope'] = fslope220

        ins220 = instr.QubicInstrument(d1, FRBW=q220.FRBW)

        return ins150, ins220

    def get_TOD(self, skyconfig, beta, convolution=False, myfwhm=None, noise=False, bandpass_correction=False):

        tod150 = self.qubic150.get_TOD(skyconfig, beta, convolution, myfwhm, noise=False, bandpass_correction=bandpass_correction)
        tod220 = self.qubic220.get_TOD(skyconfig, beta, convolution, myfwhm, noise=False, bandpass_correction=bandpass_correction)
        tod = tod150 + tod220

        if noise:
            n = self.get_noise()
            tod += n.copy()
        
        return tod
    
    def get_operator(self, convolution=False, myfwhm=None):

        """
        
        Method to compute the reconstruction operator for the WideBand instrument.

        """
        self.H150 = self.qubic150._get_array_of_operators(convolution=convolution, myfwhm=myfwhm)
        self.H220 = self.qubic220._get_array_of_operators(convolution=convolution, myfwhm=myfwhm)

        op = self.H150 + self.H220

        op_sum = []
        if self.Nrec == 1:
            op_sum = np.sum(np.array(op), axis=0)
            return op_sum
        else:
            for irec in range(self.Nrec):
                imin = irec*self.fact_sub
                imax = (irec+1)*self.fact_sub
                op_sum += [np.sum(np.array(op[imin:imax]), axis=0)]

            return BlockRowOperator(op_sum, new_axisin=0)
    

    def get_noise(self, weights=None):
        
        """
        
        Method to compute the noise vector for the WideBand instrument.
        It assumes only one contribution of the noise detectors but the sum of the photon noise of each separated focal plane.

        """

        if weights is None:
            weights = np.ones(3)

        ave_ins150, ave_ins220 = self._get_average_instrument()
        detector_noise = ave_ins150.get_noise_detector(self.qubic150.sampling)
        nsamplings = self.qubic150.sampling.comm.allreduce(len(self.qubic150.sampling))
        fact = np.sqrt(nsamplings * self.qubic150.sampling.period / (self.qubic150.d['effective_duration'] * 31557600))

        photon_noise150 = ave_ins150.get_noise_photon(self.qubic150.sampling, self.qubic150.scene)
        photon_noise220 = ave_ins220.get_noise_photon(self.qubic220.sampling, self.qubic220.scene)

        return fact * (weights[0] * detector_noise + weights[1] * photon_noise150 + weights[2] * photon_noise220)
    
    def get_invntt_operator(self, weights=None):

        """
        
        Method to compute the inverse noise covariance matrix in time-domain.

        """
        ave_ins150, ave_ins220 = self._get_average_instrument()
        if weights is None:
            weights = np.ones(3)

        sigma = weights[0] * ave_ins150.detector.nep / np.sqrt(2 * self.qubic150.sampling.period)
        sigma_photon150 = weights[1] * ave_ins150._get_noise_photon_nep(self.qubic150.scene) / np.sqrt(2 * self.qubic150.sampling.period)
        sigma_photon220 = weights[2] * ave_ins220._get_noise_photon_nep(self.qubic220.scene) / np.sqrt(2 * self.qubic220.sampling.period)

        sig = np.sqrt(sigma**2 + sigma_photon150**2 + sigma_photon220**2)

        f = (len(self.qubic150.sampling) * self.qubic150.sampling.period / (self.qubic150.d['effective_duration'] * 31557600))
        out =  DiagonalOperator(1 / sig ** 2, broadcast='rightward',
                                   shapein=(len(self.qubic150.multiinstrument[0]), len(self.qubic150.sampling))) / f


        return out

    def get_coverage(self):
        return self.qubic150.get_coverage()
class QubicDualBand(QubicIntegrated):

    def __init__(self, d, Nsub, Nrec):

        if Nrec == 1:
            raise TypeError('You can not reconstruct 1 band with DualBand instrument')
        if Nsub == 1:
            raise TypeError('You should have at least Nsub = 2')
        
        if Nsub%2 != 0:
            raise TypeError('You should have the same number of sub-bands in each bands')
        
        if Nrec%2 != 0:
            raise TypeError('You should reconstruct the same number of band for each focal plane')
        
        self.d150 = d.copy()
        self.d220 = d.copy()
        self.Nrec = int(Nrec/2)
        self.Nsub = int(Nsub/2)

        self.fact_sub = int(Nsub/Nrec)
        self.d150['filter_nu'] = 150 * 1e9
        self.d220['filter_nu'] = 220 * 1e9

        ### Qubic acquisitions
        self.qubic150 = QubicIntegrated(self.d150, Nsub=self.Nsub, Nrec=self.Nrec)
        self.qubic220 = QubicIntegrated(self.d220, Nsub=self.Nsub, Nrec=self.Nrec)
        self.scene = self.qubic150.scene

        self.allnus = np.array(list(self.qubic150.allnus) + list(self.qubic220.allnus))
        self.allfwhm = np.array(list(self.qubic150.allfwhm) + list(self.qubic220.allfwhm))
        self.nueff = np.array(list(self.qubic150.nueff) + list(self.qubic220.nueff))


    def get_TOD(self, skyconfig, beta, convolution=False, myfwhm=None, noise=False, bandpass_correction=False):

        tod150 = self.qubic150.get_TOD(skyconfig, beta, convolution, myfwhm, noise=False, bandpass_correction=bandpass_correction)
        tod220 = self.qubic220.get_TOD(skyconfig, beta, convolution, myfwhm, noise=False, bandpass_correction=bandpass_correction)
        tod = np.r_[tod150, tod220]

        if noise:
            n = self.get_noise()
            tod += n.copy()
        
        return tod

    def get_operator(self, convolution=False, myfwhm=None):

        self.H150 = np.array(self.qubic150._get_array_of_operators(convolution=convolution, myfwhm=myfwhm))
        self.H220 = np.array(self.qubic220._get_array_of_operators(convolution=convolution, myfwhm=myfwhm))

        ### 150 GHz

        op_sum=[]
        for irec in range(self.Nrec):
            imin = irec*self.fact_sub
            imax = (irec+1)*self.fact_sub - 1

            op_sum += [self.H150[(self.qubic150.allnus >= self.qubic150.allnus[imin]) * (self.qubic150.allnus <= self.qubic150.allnus[imax])].sum(axis=0)]
            
        self.H150 = BlockRowOperator(op_sum, new_axisin=0)

        
        ### 220 GHz

        op_sum=[]
        for irec in range(self.Nrec):
            imin = irec*self.fact_sub
            imax = (irec+1)*self.fact_sub - 1

            op_sum += [self.H220[(self.qubic220.allnus >= self.qubic220.allnus[imin]) * (self.qubic220.allnus <= self.qubic220.allnus[imax])].sum(axis=0)]
            
        self.H220 = BlockRowOperator(op_sum, new_axisin=0)

        if self.Nrec == 1:
            hh = BlockDiagonalOperator([self.H150, self.H220], new_axisin=0)
            R = ReshapeOperator(hh.shapeout, (hh.shapeout[0]*hh.shapeout[1], hh.shapeout[2]))
            return R * hh
        else:
            return BlockDiagonalOperator([self.H150, self.H220], axisin=0)
        
    def get_noise(self):

        self.n150 = self.qubic150.get_noise()
        self.n220 = self.qubic220.get_noise()

        return np.r_[self.n150, self.n220]

    def get_invntt_operator(self):

        self.invn150 = self.qubic150.get_invntt_operator()
        self.invn220 = self.qubic220.get_invntt_operator()

        return BlockDiagonalOperator([self.invn150, self.invn220], axisout=0)


'''


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
            _, allnus150, _, _, _, _ = qubic.compute_freq(150, Nfreq=int(self.Nsub)-1, relative_bandwidth=0.25)
            _, allnus220, _, _, _, _ = qubic.compute_freq(220, Nfreq=int(self.Nsub)-1, relative_bandwidth=0.25)
        else:
            _, _, allnus150, _, _, _ = qubic.compute_freq(150, Nfreq=int(self.Nsub), relative_bandwidth=0.25)
            _, _, allnus220, _, _, _ = qubic.compute_freq(220, Nfreq=int(self.Nsub), relative_bandwidth=0.25)
        self.allnus = np.array(list(allnus150) + list(allnus220))

        self.Nsub *= 2
        
        #self.d['nf_sub'] = 2 * self.d['nf_sub']
        self.multiinstrument = instr.QubicMultibandInstrument(self.d)
        self.sampling = qubic.get_pointing(self.d)
        self.scene = qubic.QubicScene(self.d)

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

        






