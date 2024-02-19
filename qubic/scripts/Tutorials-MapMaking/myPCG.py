import time

import numpy as np

from pyoperators import (ReshapeOperator)
from scipy.optimize import minimize
from pyoperators.core import IdentityOperator, asoperator
from pyoperators.memory import empty, zeros
from pyoperators.utils.mpi import MPI
from pyoperators.iterative.core import AbnormalStopIteration, IterativeAlgorithm
from pyoperators.iterative.stopconditions import MaxIterationStopCondition
import matplotlib.pyplot as plt
import healpy as hp
import qubic

__all__ = ['pcg']


class PCGAlgorithm(IterativeAlgorithm):
    """
    OpenMP/MPI Preconditioned conjugate gradient iteration to solve A x = b.
    """

    def __init__(
        self,
        A,
        b,
        x0=None,
        tol=1.0e-5,
        maxiter=300,
        M=None,
        truth=None,
        disp=False,
        callback=None,
        reuse_initial_state=False,
    ):
        """
        Parameters
        ----------
        A : {Operator, sparse matrix, dense matrix}
            The real or complex N-by-N matrix of the linear system
            ``A`` must represent a hermitian, positive definite matrix
        b : {array, matrix}
            Right hand side of the linear system. Has shape (N,) or (N,1).
        x0  : {array, matrix}
            Starting guess for the solution.
        tol : float, optional
            Tolerance to achieve. The algorithm terminates when either the
            relative residual is below `tol`.
        maxiter : integer, optional
            Maximum number of iterations.  Iteration will stop after maxiter
            steps even if the specified tolerance has not been achieved.
        M : {Operator, sparse matrix, dense matrix}, optional
            Preconditioner for A.  The preconditioner should approximate the
            inverse of A.  Effective preconditioning dramatically improves the
            rate of convergence, which implies that fewer iterations are needed
            to reach a given error tolerance.
        disp : boolean
            Set to True to display convergence message
        callback : function, optional
            User-supplied function to call after each iteration.  It is called
            as callback(self), where self is an instance of this class.
        reuse_initial_state : boolean, optional
            If set to True, the buffer initial guess (if provided) is reused
            during the iterations. Beware of side effects!
        Returns
        -------
        x : array
            The converged solution.
        Raises
        ------
        pyoperators.AbnormalStopIteration : if the solver reached the maximum
            number of iterations without reaching specified tolerance.
        """
        dtype = A.dtype or np.dtype(float)
        if dtype.kind == 'c':
            raise TypeError('The complex case is not yet implemented.')
        elif dtype.kind != 'f':
            dtype = np.dtype(float)
        b = np.array(b, dtype, copy=False)

        if x0 is None:
            x0 = zeros(b.shape, dtype)

        abnormal_stop_condition = MaxIterationStopCondition(
            maxiter,
            'Solver reached maximum number of iterations without reac'
            'hing specified tolerance.',
        )

        IterativeAlgorithm.__init__(
            self,
            x=x0,
            convergence=np.array([]),
            abnormal_stop_condition=abnormal_stop_condition,
            disp=disp,
            dtype=dtype,
            reuse_initial_state=reuse_initial_state,
            inplace_recursion=True,
            callback=callback,
        )

        A = asoperator(A)
        if A.shapein is None:
            raise ValueError('The operator input shape is not explicit.')
        if A.shapein != b.shape:
            raise ValueError(
                f"The operator input shape '{A.shapein}' is incompatible with that of "
                f"the RHS '{b.shape}'."
            )
        self.A = A
        self.b = b
        self.truth = truth
        self.comm = A.commin
        self.norm = lambda x: _norm2(x, self.comm)
        self.dot = lambda x, y: _dot(x, y, self.comm)

        if M is None:
            M = IdentityOperator()
        self.M = asoperator(M)

        self.tol = tol
        self.b_norm = self.norm(b)
        self.d = empty(b.shape, dtype)
        self.q = empty(b.shape, dtype)
        self.r = empty(b.shape, dtype)
        self.s = empty(b.shape, dtype)

    def initialize(self):
        IterativeAlgorithm.initialize(self)

        if self.b_norm == 0:
            self.error = 0
            self.x[...] = 0
            raise StopIteration('RHS is zero.')

        self.r[...] = self.b
        self.r -= self.A(self.x)
        self.error = np.sqrt(self.norm(self.r) / self.b_norm)
        if self.error < self.tol:
            raise StopIteration('Solver reached maximum tolerance.')
        self.M(self.r, self.d)
        self.delta = self.dot(self.r, self.d)

    def iteration(self):

        '''
        plt.figure(figsize=(5, 5))
        center=qubic.equ2gal(0, -57)

        hp.mollview(self.truth[0, :, 1], cmap='jet', sub=(3, 3, 4), min=-5, max=5, title='Input - Q')
        hp.mollview(self.x[0, :, 1], cmap='jet', sub=(3, 3, 5), min=-5, max=5, title='Output - Q')
        hp.mollview(self.truth[0, :, 1]-self.x[0, :, 1], cmap='jet', sub=(3, 3, 6), min=-5, max=5, title='Residuals - Q')

        plt.savefig('/Users/mregnier/Desktop/PhD Regnier/MapMaking/images/maps_qubicplanck_mono_{:.0f}.png'.format(self.niterations))
        plt.close()
        '''

        self.A(self.d, self.q)
        alpha = self.delta / self.dot(self.d, self.q)
        self.x += alpha * self.d
        self.r -= alpha * self.q
        self.error = np.sqrt(self.norm(self.r) / self.b_norm)
        self.convergence = np.append(self.convergence, self.error)
        if self.error < self.tol:
            raise StopIteration('Solver reached maximum tolerance.')

        self.M(self.r, self.s)
        delta_old = self.delta
        self.delta = self.dot(self.r, self.s)
        beta = self.delta / delta_old
        self.d *= beta
        self.d += self.s

    @staticmethod
    def callback(self):
        if self.disp:
            print(f'{self.niterations:4}: {self.error}')


def pcg(
    A,
    b,
    x0=None,
    tol=1.0e-5,
    maxiter=300,
    M=None,
    truth=None,
    disp=False,
    callback=None,
    reuse_initial_state=False,
):
    """
    output = pcg(A, b, [x0, tol, maxiter, M, disp, callback,
                 reuse_initial_state])
    Parameters
    ----------
    A : {Operator, sparse matrix, dense matrix}
        The real or complex N-by-N matrix of the linear system
        ``A`` must represent a hermitian, positive definite matrix
    b : {array, matrix}
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0  : {array, matrix}
        Starting guess for the solution.
    tol : float, optional
        Tolerance to achieve. The algorithm terminates when either the
        relative residual is below `tol`.
    maxiter : integer, optional
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M : {Operator, sparse matrix, dense matrix}, optional
        Preconditioner for A.  The preconditioner should approximate the
        inverse of A.  Effective preconditioning dramatically improves the
        rate of convergence, which implies that fewer iterations are needed
        to reach a given error tolerance.
    disp : boolean
        Set to True to display convergence message
    callback : function, optional
        User-supplied function to call after each iteration.  It is called
        as callback(self), where self is an instance of this class.
    reuse_initial_state : boolean, optional
        If set to True, the buffer initial guess (if provided) is reused
        during the iterations. Beware of side effects!
    Returns
    -------
    output : dict whose keys are
        'x' : the converged solution.
        'success' : boolean indicating success
        'message' : string indicating cause of failure
        'nit' : number of completed iterations
        'error' : normalized residual ||Ax-b|| / ||b||
        'time' : elapsed time in solver
        'algorithm' : the PCGAlgorithm instance (the callback function has
                      access to it and can store information in it)
    """
    time0 = time.time()
    algo = PCGAlgorithm(
        A,
        b,
        x0=x0,
        tol=tol,
        maxiter=maxiter,
        disp=disp,
        M=M,
        truth=truth,
        callback=callback,
        reuse_initial_state=reuse_initial_state,
    )
    try:
        output = algo.run()
        success = True
        message = ''
    except AbnormalStopIteration as e:
        output = algo.finalize()
        success = False
        message = str(e)
    return {
        'x': output,
        'success': success,
        'message': message,
        'nit': algo.niterations,
        'error': algo.error,
        'time': time.time() - time0,
        'algorithm': algo,
    }


def _norm2(x, comm):
    x = x.ravel()
    n = np.array(np.dot(x, x))
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, n)
    return n
def _dot(x, y, comm):
    d = np.array(np.dot(x.ravel(), y.ravel()))
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, d)
    return d

def infinite_loop(Nite, kmax, tod, invN, acquisition, Hqubic, Hpl, beta0, tol_maps, x0, object_spectral_index, Ndet, Nsamples, center=None, save=False, M=None, disp=True, truth=None, plot=False, **kwargs):

    convergence=[1]
    print('Starting loop : ')
    k=0
    beta = np.array([beta0])

    while convergence[-1] > tol_maps:

        print('k = {}'.format(k+1))
        # PCG
        if k == 0:
            Amm = object_spectral_index.give_mixing_matrix(beta[-1])
            H = acquisition.update_operator(Hqubic, Hpl, NewA=Amm)
            A = H.T * invN * H
            b = H.T * invN * tod

            solution = pcg(A, b, M=M, x0=x0, tol=tol_maps, disp=disp, maxiter=Nite)

        elif k > kmax-1:

            # Spectral index fitting
            beta_i = minimize(object_spectral_index.mychi2, x0=beta[-1], args=(mysolution, tod, Hqubic, Hpl, Ndet, Nsamples), **kwargs).x
            beta = np.append(beta, [beta_i], axis=0)

            break

        else:
            Amm = object_spectral_index.give_mixing_matrix(beta[-1])
            H = acquisition.update_operator(Hqubic, Hpl, NewA=Amm)
            A = H.T * invN * H
            b = H.T * invN * tod

            solution = pcg(A, b, M=M, x0=mysolution, tol=tol_maps, disp=disp, maxiter=Nite)

        mysolution = solution['x']['x']
        convergence = np.append(convergence, solution['x']['convergence'])

        if plot :
            plt.figure(figsize=(15, 5))
            hp.mollview(mysolution[0, :, 0], min=-200, max=200, cmap='jet', sub=(1, 3, 2), title='Solution')
            hp.mollview(truth[0, :, 0], min=-200, max=200, cmap='jet', sub=(1, 3, 1), title='Input')
            hp.mollview(truth[0, :, 0]-mysolution[0, :, 0], min=-2, max=2, cmap='jet', sub=(1, 3, 3), title='Residual')
            plt.suptitle('Iteration = {}'.format(k), y=0.75)
            if save:
                plt.savefig('images/img_I_ite{}.png'.format(k))
            plt.close()

            plt.figure(figsize=(15, 5))
            hp.gnomview(mysolution[0, :, 0], rot=center, reso=20, min=-200, max=200, cmap='jet', sub=(1, 3, 1), title='Solution')
            hp.gnomview(truth[0, :, 0], rot=center, reso=20, min=-200, max=200, cmap='jet', sub=(1, 3, 2), title='Input')
            hp.gnomview(truth[0, :, 0]-mysolution[0, :, 0], rot=center, reso=20, min=-5, max=5, cmap='jet', sub=(1, 3, 3), title='Residual')
            plt.suptitle('Iteration = {}'.format(k), y=1.07)
            if save:
                plt.savefig('images/img_gnom_I_ite{}.png'.format(k))
            plt.close()

            plt.figure(figsize=(15, 5))
            hp.gnomview(mysolution[0, :, 1], rot=center, reso=20, min=-6, max=6, cmap='jet', sub=(1, 3, 2), title='Solution')
            hp.gnomview(truth[0, :, 1], rot=center, reso=20, min=-6, max=6, cmap='jet', sub=(1, 3, 1), title='Input')
            hp.gnomview(truth[0, :, 1]-mysolution[0, :, 1], rot=center, reso=20, min=-5, max=5, cmap='jet', sub=(1, 3, 3), title='Residual')
            plt.suptitle('Iteration = {}'.format(k), y=1.07)
            if save:
                plt.savefig('images/img_gnom_Q_ite{}.png'.format(k))
            plt.close()


        # Spectral index fitting
        beta_i = minimize(object_spectral_index.mychi2, x0=beta[-1], args=(mysolution, tod, Hqubic, Hpl, Ndet, Nsamples), **kwargs).x
        beta = np.append(beta, [beta_i], axis=0)
        k+=1

    return mysolution, beta, convergence
