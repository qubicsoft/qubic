# coding: utf-8
from __future__ import division

import healpy as hp
import numpy as np
import warnings
from pyoperators import (
    BlockColumnOperator, BlockDiagonalOperator, BlockRowOperator,
    CompositionOperator, DiagonalOperator, I, IdentityOperator,
    MPIDistributionIdentityOperator, MPI, proxy_group, ReshapeOperator,
    rule_manager, pcg)
from pyoperators.utils.mpi import as_mpi
from pysimulators import Acquisition, FitsArray
from pysimulators.interfaces.healpy import (
    HealpixConvolutionGaussianOperator)
import qubic
from .data import PATH
from .acquisition import (QubicAcquisition,
                          PlanckAcquisition,
                          QubicPlanckAcquisition)
from .scene import QubicScene
from .samplings import create_random_pointings, get_pointing

__all__ = ['compute_freq',
           'QubicPolyAcquisition',
           'QubicPolyPlanckAcquisition']


def compute_freq(band, Nfreq=None, relative_bandwidth=0.25):
    """
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
    """

    if Nfreq is None:
        Nfreq = {150: 15, 220: 20}[band]

    nu_min = band * (1 - relative_bandwidth / 2)
    nu_max = band * (1 + relative_bandwidth / 2)

    Nfreq_edges = Nfreq + 1
    base = (nu_max / nu_min) ** (1. / Nfreq)

    nus_edge = nu_min * np.logspace(0, Nfreq, Nfreq_edges, endpoint=True, base=base)
    nus = np.array([(nus_edge[i] + nus_edge[i - 1]) / 2 for i in range(1, Nfreq_edges)])
    deltas = np.array([(nus_edge[i] - nus_edge[i - 1]) for i in range(1, Nfreq_edges)])
    Delta = nu_max - nu_min
    Nbbands = len(nus)
    return Nfreq_edges, nus_edge, nus, deltas, Delta, Nbbands


class QubicPolyAcquisition(object):
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

        if convolution:
            maps_convolved = [np.average(_maps_convolved[(self.nus > mi) * (self.nus < ma)],
                                         axis=0, weights=self.weights[(self.nus > mi) * (self.nus < ma)]) \
                              for (mi, ma) in self.bands]
            return tod, maps_convolved

        return tod

    def get_preconditioner(self, cov):
        if cov is not None:
            cov_inv = 1 / cov
            cov_inv[np.isinf(cov_inv)] = 0.
            preconditioner = DiagonalOperator(cov_inv, broadcast='rightward')
        else:
            preconditioner = None
        return preconditioner

    def tod2map(self, tod, d, cov=None):
        """
        Reconstruct map from tod
        """
        tol = d['tol']
        maxiter = d['maxiter']
        verbose = d['verbose']
        H = self.get_operator()
        invntt = self.get_invntt_operator()

        A = H.T * invntt * H
        b = H.T * invntt * tod

        preconditioner = self.get_preconditioner(cov)
        solution = pcg(A, b, M=preconditioner,
                       disp=verbose, tol=tol, maxiter=maxiter)
        return solution['x'], solution['nit'], solution['error']


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
        if weights is None:
            self.weights = np.ones(len(self))  # / len(self)
        else:
            self.weights = weights

    def __len__(self):
        return len(self.qubic)

    def get_observation(self, m=None, convolution=True, noiseless=False):
        """
        Return fusion observation as a sum of monochromatic fusion TODs
        """
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

    def get_preconditioner(self, H):
        M = (H.T * H * np.ones(H.shapein))[..., 0]
        preconditioner = DiagonalOperator(1 / M, broadcast='rightward')
        return preconditioner

    def tod2map(self, tod, d):
        tol = d['tol']
        maxiter = d['maxiter']
        verbose = d['verbose']
        H = self.get_operator()
        invntt = self.get_invntt_operator()
        A = H.T * invntt * H
        b = H.T * invntt * tod

        preconditioner = self.get_preconditioner(H)
        solution = pcg(A, b, M=preconditioner, disp=verbose, tol=tol, maxiter=maxiter)
        return solution['x']
