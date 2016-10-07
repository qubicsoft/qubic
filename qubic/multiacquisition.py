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
from .scene import QubicScene
from .samplings import create_random_pointings
from .acquisition import (QubicAcquisition,
                          PlanckAcquisition,
                          QubicPlanckAcquisition)
from .polyacquisition import (QubicPolyAcquisition,
                              QubicPolyPlanckAcquisition)
__all__ = ['QubicMultibandAcquisition',
           'QubicMultibandPlanckAcquisition']

class QubicMultibandAcquisition(QubicPolyAcquisition):
    def __init__(self, multiinstrument, sampling, scene, nus, block=None,
                 effective_duration=None,
                 photon_noise=True, max_nbytes=None,
                 nprocs_instrument=None, nprocs_sampling=None,
                 ripples=False, nripples=0,
                 weights=None):
        '''
        Parameters:
        -----------
        nus : array
            edge frequencies for subbands, for example:
            [140, 150, 160] means two bands: one from 140 to 150 GHz and
            one from 150 to 160 GHz
        Note, that number of subbands is not equal to len(self)
        Within each subband there are multiple frequencies
        Documentation for other parameters see in QubicPolyAcquisition
        '''
        QubicPolyAcquisition.__init__(self, multiinstrument, sampling, scene, 
            block=block, effective_duration=effective_duration,
            photon_noise=photon_noise, max_nbytes=max_nbytes,
            nprocs_instrument=nprocs_instrument,
            nprocs_sampling=nprocs_sampling,
            ripples=ripples, nripples=nripples,
            weights=weights)
        if len(nus) > 1:
            self.bands = np.array([[nus[i], nus[i + 1]] for i in xrange(len(nus) - 1)])
        else:
            raise ValueError('The QubicMultibandAcquisition class is designed to'\
                'work with multiple frequencies. For monochromatic case you can use'\
                'the QubicAcquisition class')
        self.nus = np.array([q.filter.nu / 1e9 for q in multiinstrument])

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
        if len(self) == 1:
            return self[0].get_observation(m, convolution=convolution, noiseless=noiseless)

        if self.scene.kind != 'I':
            shape = (len(self), m.shape[1], m.shape[2])
        else:
            shape = m.shape

        if convolution:
            _maps_convolved = np.zeros(shape) # array of sky maps, each convolved with its own gaussian
            for i in range(len(self)):
                C = self[i].get_convolution_peak_operator()
                _maps_convolved[i] = C(m[i])
            y = self.get_operator_to_make_TOD() * _maps_convolved
        else:
            y = self.get_operator() * m

        if not noiseless:
            y += self.get_noise()

        if convolution:
            maps_convolved = [np.average(_maps_convolved[(self.nus > mi) * (self.nus < ma)], 
                              axis=0, weights=self.weights[(self.nus > mi) * (self.nus < ma)]) \
                              for (mi, ma) in self.bands]
            return y, maps_convolved

        return y

    def get_operator(self):
        op = np.array(self._get_array_of_operators())
        op_sum = []
        for band in self.bands:
            op_sum.append(op[(self.nus > band[0]) * (self.nus < band[1])].sum(axis=0))
        return BlockRowOperator(op_sum, new_axisin=0)

    def get_preconditioner(self, cov):
        if cov is not None:
            cov_inv = 1 / cov
            return BlockDiagonalOperator(\
                [DiagonalOperator(cov_inv[(self.nus > mi) * (self.nus < ma)], 
                    broadcast='rightward') for (mi, ma) in self.bands], 
                new_axisin=0)
        else:
            return None

class QubicMultibandPlanckAcquisition(QubicPolyPlanckAcquisition):
    """
    The QubicMultibandPlanckAcquisition class, which combines the QubicMultiband and Planck
    acquisitions.

    """
    def __init__(self, qubic, planck, weights=None):
        """
        acq = QubicPlanckAcquisition(qubic_acquisition, planck_acquisition)

        Parameters
        ----------
        qubic_acquisition : QubicAcquisition
            The QUBIC acquisition.
        planck_acquisition : PlanckAcquisition
            The Planck acquisition.

        """
        if not isinstance(qubic, QubicMultibandAcquisition):
            raise TypeError('The first argument is not a QubicMultibandAcquisition.')
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

    def get_observation(self, maps, noiseless=False, convolution=True):
        """
        Return the fused observation.

        Parameters
        ----------
        maps : numpy array of shape (nbands, npix, 3)
            True input multiband maps
        noiseless : boolean, optional
            If set, return a noiseless observation
        convolution : boolean, optional
            If set, 
        """
        obs_qubic_ = self.qubic.get_observation(
            maps, noiseless=noiseless,
            convolution=convolution)
        obs_qubic = obs_qubic_[0] if convolution else obs_qubic_
        obs_planck = self.planck.get_observation(noiseless=noiseless)
        obs = np.r_[obs_qubic.ravel(), obs_planck.ravel()]
        if convolution:
            return obs, np.array(obs_qubic_[1])
        return obs

    def tod2map(self, tod, cov=None, tol=1e-5, maxiter=1000, verbose=True):
        p = self.planck
        H = []
        for q, w in zip(self.qubic, self.weights):
            H.append(QubicPlanckAcquisition(q, p).get_operator() * w)
        H = np.array(H)
        H = [H[(self.qubic.nus > mi) * (self.qubic.nus < ma)].sum() * \
             self.weights[(self.qubic.nus > mi) * (self.qubic.nus < ma)].sum() \
                for (mi, ma) in self.qubic.bands]
        invntt = self.get_invntt_operator()

        A_columns = []
        for h1 in H:
            c = []
            for h2 in H:
                c.append(h2.T * invntt * h1)
            A_columns.append(BlockColumnOperator(c, axisout=0))
        A = BlockRowOperator(A_columns, axisin=0)

        H = [h.T for h in H]
        b = BlockColumnOperator(H, new_axisout=0) * (invntt * tod)
        sh = b.shape
        if len(sh) == 3:
            b = b.reshape((sh[0] * sh[1], sh[2]))
        else:
            b = b.reshape((sh[0] * sh[1]))

        preconditioner = self.qubic.get_preconditioner(cov)
        solution = pcg(A, b, disp=verbose, tol=tol, maxiter=maxiter)
        if len(sh) == 3:
            maps_recon = solution['x'].reshape(sh[0], sh[1], sh[2])
        else:
            maps_recon = solution['x'].reshape(sh[0], sh[1])
        return maps_recon
