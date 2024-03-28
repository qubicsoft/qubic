import time

import numpy as np

from pyoperators.core import IdentityOperator, asoperator
from pyoperators.memory import empty, zeros
from pyoperators.utils.mpi import MPI
from pyoperators.iterative.core import AbnormalStopIteration, IterativeAlgorithm
from pyoperators.iterative.stopconditions import MaxIterationStopCondition
from tools.foldertools import *
import matplotlib.pyplot as plt
import healpy as hp
import yaml
import qubic
import mapmaking.systematics as acq
from mapmaking.planck_timeline import *
from mapmaking.noise_timeline import *

 
__all__ = ['pcg']


class PCGAlgorithm(IterativeAlgorithm):
    """
    OpenMP/MPI Preconditioned conjugate gradient iteration to solve A x = b.

    """

    def __init__(
        self,
        A,
        b,
        comm,
        x0=None,
        tol=1.0e-5,
        maxiter=300,
        M=None,
        disp=False,
        callback=None,
        reuse_initial_state=False,
        create_gif=False,
        center=None,
        reso=15,
        figsize=(10, 8),
        seenpix=None,
        jobid=None,
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

        with open('params.yml', "r") as stream:
            self.params = yaml.safe_load(stream)

        self.gif = create_gif
        self.center = center
        self.reso = reso
        self.figsize = figsize
        self.seenpix = seenpix
        self.jobid = jobid
        self.IQU = np.std(x0[:, seenpix, :], axis=1)
        #print(self.IQU, self.IQU.shape)
        #stop
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
        self.comm = comm
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.norm = lambda x: _norm2(x, self.comm)
        self.dot = lambda x, y: _dot(x, y, self.comm)

        if self.gif:
            create_folder_if_not_exists(self.comm, f'gif_convergence_{self.jobid}')
        if M is None:
            M = IdentityOperator()
        self.M = asoperator(M)

        self.tol = tol
        self.b_norm = self.norm(b)
        self.d = empty(b.shape, dtype)
        self.q = empty(b.shape, dtype)
        self.r = empty(b.shape, dtype)
        self.s = empty(b.shape, dtype)

        #self.dict, self.dict_mono = self.get_dict()
        #self.skyconfig = self._get_sky_config()
        #self.joint = acq.JointAcquisitionFrequencyMapMaking(self.dict, self.params['QUBIC']['type'], self.params['QUBIC']['nrec'], self.params['QUBIC']['nsub'])
        #self.fsub = int(self.params['QUBIC']['nsub'] / self.params['QUBIC']['nrec'])

    def initialize(self):
        IterativeAlgorithm.initialize(self)

        if self.b_norm == 0:
            self.error = 0
            self.x[...] = 0
            self.convergence = np.array([])
            raise StopIteration('RHS is zero.')

        self.r[...] = self.b
        self.r -= self.A(self.x)
        self.error = np.sqrt(self.norm(self.r) / self.b_norm)
        if self.error < self.tol:
            raise StopIteration('Solver reached maximum tolerance.')
        self.M(self.r, self.d)
        self.delta = self.dot(self.r, self.d)

    def iteration(self):
        self.t0 = time.time()
        self.A(self.d, self.q)
        alpha = self.delta / self.dot(self.d, self.q)
        self.x += alpha * self.d
        self.IQU = np.std(self.x[:, self.seenpix, :], axis=1)
            
        self.r -= alpha * self.q
        self.error = np.sqrt(self.norm(self.r) / self.b_norm)
        self.convergence = np.append(self.convergence, self.error)
        if self.error < self.tol:
            print('STOP')
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
            print(f'{self.niterations:4}: {self.error:.4e} {time.time() - self.t0:.5f} {self.IQU[:, 0]} {self.IQU[:, 1]} {self.IQU[:, 2]}')


def pcg(
    A,
    b,
    comm,
    x0=None,
    tol=1.0e-5,
    maxiter=300,
    M=None,
    disp=False,
    callback=None,
    reuse_initial_state=False,
    create_gif=False,
    center=None,
    reso=15,
    seenpix=None,
    jobid=None,
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
        comm,
        x0=x0,
        tol=tol,
        maxiter=maxiter,
        disp=disp,
        M=M,
        callback=callback,
        reuse_initial_state=reuse_initial_state,
        create_gif=create_gif,
        center=center,
        reso=reso,
        seenpix=seenpix,
        jobid=jobid,
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
