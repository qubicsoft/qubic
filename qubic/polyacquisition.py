# coding: utf-8
from __future__ import division

import healpy as hp
import numpy as np
from pyoperators import (
    BlockColumnOperator, BlockDiagonalOperator, BlockRowOperator,
    CompositionOperator, DiagonalOperator, I, IdentityOperator,
    MPIDistributionIdentityOperator, MPI, proxy_group, ReshapeOperator,
    rule_manager, pcg)
from pyoperators.utils.mpi import as_mpi
from pysimulators import Acquisition, FitsArray
from pysimulators.interfaces.healpy import (
    HealpixConvolutionGaussianOperator)
from .data import PATH
from .instrument import QubicInstrument
from .acquisition import (QubicAcquisition,
                          PlanckAcquisition,
                          QubicPlanckAcquisition)
from .scene import QubicScene
from .samplings import create_random_pointings

__all__ = ['compute_freq',
           'QubicPolyAcquisition',
           'QubicPolyPlanckAcquisition']

def compute_freq(band, relative_bandwidth=0.25, Nfreq=None):
    '''
    Prepare frequency bands parameters
    band -- int,
        QUBIC frequency band, in GHz.
        Typical values: 150, 220
    relative_bandwidth -- float, optional
        Ratio of the difference between the edges of the
        frequency band over the average frequency of the band:
        2 * (nu_max - nu_min) / (nu_max + nu_min)
        Typical value: 0.25
    Nfreq -- int, optional
        Number of frequencies within the wide band.
        If not specified, then Nfreq = 15 if band == 150
        and Nfreq = 20 if band = 220
    '''
    if Nfreq is None:
        Nfreq = {150: 15, 220: 20}[band]

    nu_min = band * (1 - relative_bandwidth / 2)
    nu_max = band * (1 + relative_bandwidth / 2)
    
    Nfreq_edges = Nfreq + 1
    base = (nu_max / nu_min) ** (1. / Nfreq)

    nus_edge = nu_min * np.logspace(0, Nfreq, Nfreq_edges, endpoint=True, base=base)
    nus = np.array([(nus_edge[i] + nus_edge[i-1]) / 2 for i in range(1, Nfreq_edges)])
    deltas = np.array([(nus_edge[i] - nus_edge[i-1])  for i in range(1, Nfreq_edges)])
    Delta = nu_max - nu_min
    Nbbands = len(nus)
    return Nfreq_edges, nus_edge, nus, deltas, Delta, Nbbands

class QubicPolyAcquisition(object):
    def __init__(self, multiinstrument, sampling, scene, block=None,
                 effective_duration=None,
                 photon_noise=True, max_nbytes=None,
                 nprocs_instrument=None, nprocs_sampling=None,
                 ripples=False, nripples=0,
                 weights=None):
        '''
        acq = QubicPolyAcquisition(QubicMultibandInstrument, sampling, scene)

        Parameters
        ----------
        multiinstrument : QubicMultibandInstrument
            The sub-frequencies are set there
        sampling : 
            QubicSampling instance
        scene :
            QubicScene instance

        For other parameters see the documentation for the QubicAcquisition class

        '''
        self.subacqs = [QubicAcquisition(multiinstrument[i], 
                                 sampling, scene=scene, block=block,
                                 effective_duration=effective_duration,
                                 photon_noise=photon_noise, max_nbytes=max_nbytes,
                                 nprocs_instrument=nprocs_instrument, 
                                 nprocs_sampling=nprocs_sampling,
                                 ripples=ripples, nripples=nripples) for i in range(len(multiinstrument))]
        for a in self[1:]:
            a.comm = self[0].comm
        self.scene = scene
        if weights == None:
            self.weights = np.ones(len(self)) / len(self)
        else:
            self.weights = weights

    def __getitem__(self, i):
        return self.subacqs[i]

    def __len__(self):
        return len(self.subacqs)

    def get_coverage(self):
        '''
        Return an array of monochromatic coverage maps, one for each of subacquisitions
        '''
        if len(self) == 1:
            return self.subacqs[0].get_coverage()
        return np.array([self.subacqs[i].get_coverage() for i in range(len(self))])

    def get_coverage_mask(self, coverages, covlim=0.2):
        '''
        Return a healpix boolean map with True on the pixels where ALL the
            subcoverages are above covlim * subcoverage.max()
        '''
        if coverages.shape[0] != len(self):
            raise ValueError('Use QubicMultibandAcquisition.get_coverage method to create input') 
        if len(self) == 1:
            cov = coverages
            return cov > covlim * np.max(cov)
        observed = [(coverages[i] > covlim * np.max(coverages[i])) for i in range(len(self))]
        obs = reduce(np.logical_and, tuple(observed[i] for i in range(len(self))))
        return obs

    def _get_average_instrument_acq(self):
        '''
        Create and return a QubicAcquisition instance of a monochromatic
            instrument with frequency correspondent to the mean of the
            frequency range.
        '''
        if len(self) == 1:
            return self[0]
        q0 = self[0].instrument
        nu_min = q0.filter.nu
        nu_max = self[-1].instrument.filter.nu
        nep = q0.detector.nep
        fknee = q0.detector.fknee
        fslope = q0.detector.fslope
        q = QubicInstrument(
            filter_nu=(nu_max + nu_min) / 2.,
            filter_relative_bandwidth=(nu_max - nu_min) / ((nu_max + nu_min) / 2.),
            detector_nep=nep, detector_fknee=fknee, detector_fslope=fslope)
        s_ = self[0].sampling
        nsamplings = self[0].comm.allreduce(len(s_))
        s = create_random_pointings([0., 0.], nsamplings, 10., period=s_.period)
        a = QubicAcquisition(
            q, s, self[0].scene, photon_noise=True,  
            effective_duration=self[0].effective_duration)
        return a

    def get_noise(self):
        a = self._get_average_instrument_acq()
        return a.get_noise()

    def _get_array_of_operators(self):
        return [a.get_operator() * w for a, w in zip(self, self.weights)]

    def get_operator_to_make_TOD(self):
        '''
        Return a BlockRowOperator of subacquisition operators
        In polychromatic mode it is only applied to produce the TOD
        To reconstruct maps one should use the get_operator function
        '''
        if len(self) == 1:
            return self.get_operator()
        op = self._get_array_of_operators()
        return BlockRowOperator(op, new_axisin=0)

    def get_operator(self):
        '''
        Return an sum of operators for subacquisitions
        '''
        if len(self) == 1:
            return self[0].get_operator()
        op = np.array(self._get_array_of_operators())
        return np.sum(op, axis=0)

    def get_invntt_operator(self):
        '''
        Return the inverse noise covariance matrix as operator
        '''
        return self[0].get_invntt_operator()

    def get_observation(self, m, convolution=True, noiseless=False):
        '''
        Return TOD for polychromatic synthesised beam

        Parameters
        ----------
        m : np.array((npix, 3)) if self.scene.kind == 'IQU', else np.array((npix))
            Helpix map of CMB
        convolution : boolean, optional [default and recommended = True]
            - if True, convolve the input map with gaussian kernel
            with width specific for each subfrequency and
            return TOD, convolved map,
            where TOD = [H1, H2, H3...] * [m_conv1, m_conv2, m_conv3].T
            and convolved map = average([m_conv1, m_conv2, m_conv3])
            - if False, the input map is considered as already convolved
            and the return is just TOD, which is equal to
            [sum(H1, H2, H3, ...)] * input_map
        noiseless : boolean, optional [default=False]
            if False, add noise to the TOD due to the model
        '''
        if len(self) == 1:
            return self[0].get_observation(m, convolution=convolution, noiseless=noiseless)

        if self.scene.kind == 'IQU':
            shape = (len(self), m.shape[0], m.shape[1])
        else:
            shape = (len(self), m.shape[0])

        if convolution:
            maps_convolved = np.zeros(shape) # array of sky maps, each convolved with its own gaussian
            map_convolved = np.zeros(m.shape) # average convolved map
            for i in range(len(self)):
                C = self[i].get_convolution_peak_operator()
                maps_convolved[i] = C(m)
                map_convolved += maps_convolved[i] * self.weights[i]
            y = self.get_operator_to_make_TOD() * maps_convolved
        else:
           y = self.get_operator() * m

        if not noiseless:
            y += self.get_noise()

        if convolution:
            return y, map_convolved

        return y

    def get_preconditioner(self, cov):
        if cov is not None:
            cov_inv = 1 / cov
            cov_inv[cov == 0.] = 1.
            cov_inv = np.sqrt(cov_inv)
            preconditioner = DiagonalOperator(cov_inv, broadcast='rightward')
        else:
            preconditioner = None
        return preconditioner

    def tod2map(self, tod, cov=None, tol=1e-5, maxiter=1000, verbose=True):
        '''
        Reconstruct map from tod
        '''
        H = self.get_operator()
        invntt = self.get_invntt_operator()

        A = H.T * invntt * H
        b = H.T * invntt * tod

        preconditioner = self.get_preconditioner(cov)
        solution = pcg(A, b, M=preconditioner, 
            disp=verbose, tol=tol, maxiter=maxiter)
        return solution['x']

class QubicPolyPlanckAcquisition(QubicPlanckAcquisition):
    """
    The QubicPolyAcquisition class, which combines the QubicPoly and Planck
    acquisitions.

    """
    def __init__(self, qubic, planck, weights=None):
        """
        acq = QubicPolyPlanckAcquisition(qubic_acquisition, planck_acquisition)

        Parameters
        ----------
        qubic_acquisition : QubicPolyAcquisition
            The QUBIC polychromatic acquisition.
        planck_acquisition : PlanckAcquisition
            The Planck acquisition.

        """
        if not isinstance(qubic, QubicPolyAcquisition):
            raise TypeError('The first argument is not a QubicPolyAcquisition.')
        if not isinstance(planck, PlanckAcquisition):
            raise TypeError('The second argument is not a PlanckAcquisition.')
        if qubic.scene is not planck.scene:
            raise ValueError('The Qubic and Planck scenes are different.')
        self.qubic = qubic
        self.planck = planck
        if weights == None:
            self.weights = np.ones(len(self)) / len(self)
        else:
            self.weights = weights


    def __len__(self):
        return len(self.qubic)

    def get_observation(self, m=None, convolution=True, noiseless=False):
        '''
        Return fusion observation as a sum of monochromatic fusion TODs
        '''
        if convolution and m is None:
            raise ValueError('Define the map, if you want to use convolution option')

        p = self.planck
        if m is None:
            m = p._true_sky
        tod_shape = len(self.qubic[0].instrument) * len(self.qubic[0].sampling) + \
            len(self.qubic.scene.kind) * hp.nside2npix(self.qubic.scene.nside)
        tod = np.zeros(tod_shape)
        if self.qubic.scene.kind == 'IQU':
            maps_convolved = np.empty((len(self), m.shape[0], m.shape[1]))
        else:
            maps_convolved = np.empty((len(self), m.shape[0]))
        for i in range(len(self)):
            q = self.qubic[i]
            if convolution:
                maps_convolved[i] = q.get_convolution_peak_operator() * m
                p._true_sky = maps_convolved[i]
            f = QubicPlanckAcquisition(q, p)
            tod += f.get_observation(convolution=False, noiseless=True) * self.weights[i]

        if not noiseless:
            tod += self.get_noise()

        if convolution:
            return tod, np.average(np.array(maps_convolved), 
                                   axis=0,
                                   weights=self.weights)
        return tod

    def tod2map(self, tod, cov=None, tol=1e-5, maxiter=1000, verbose=True):
        H = self.get_operator()
        invntt = self.get_invntt_operator()
        A = H.T * invntt * H
        b = H.T * invntt * tod

        preconditioner = self.qubic.get_preconditioner(cov)
        solution = pcg(A, b, M=preconditioner, disp=verbose, tol=tol, maxiter=maxiter)
        return solution['x']