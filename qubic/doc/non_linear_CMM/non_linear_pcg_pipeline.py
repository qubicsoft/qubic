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


class NonLinearPCGAlgorithm(IterativeAlgorithm):
    
    def __init__(
        self,
        grad_f,
        conjugate_method='polak-ribiere',
        x0=None,
        tol=1.0e-5,
        maxiter=300,
        sigma_0=1.0e-3,
        maxiter_linesearch=10,
        tol_linesearch=1.0e-2,
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
        grad_f : {Non-linear operator}
            The gradient of the function f. Applying grad_f on a vector of shape (N,) 
            should return a vector of the same shape.
        conjugate_method : {string}
            Can be 'polak-ribiere', 'fletcher-reeves', 'hestenes-stiefel', 'dai-yuan' or 'hybrid'
            Different methods are possible for conjugating the successive search directions.
            Here five choices are implemented, many more are possible.
            One should try them to see which one is the fastest in each problem.
            'hybrid' is a combination of 'hestenes-stiefel' and 'dai-yuan'.
            For more details, see: Hager, W. W., & Zhang, H. (2006). A survey of nonlinear 
            conjugate gradient methods. Pacific journal of Optimization, 2(1), 35-58.
        x0  : {array, matrix}
            Starting guess for the solution.
        tol : float, optional
            Tolerance to achieve. The algorithm terminates when either the
            relative residual is below `tol`.
        maxiter : integer, optional
            Maximum number of iterations.  Iteration will stop after maxiter
            steps even if the specified tolerance has not been achieved.
        sigma_0 : float
            Initial guess of the size of the step made at each iteration.
        maxiter_linesearch : integer
            Maximum number of iterations to find the best estimation of the size 
            of the step made at each iteration.
        tol_linesearch : float
            Tolerance to achieve for the size of the step made at each iteration.
        M : {Non-linear operator}, optional
            Preconditioner for the PCG. It should approximate the inverse of the hessian
            matrix of f. M(x) should be easy to cumpute. Effective preconditioning 
            improves the rate of convergence, which implies that fewer iterations 
            are needed to reach a given error tolerance.
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

        Note
        ----
        This code is an implementation of algorithm B5 from:
        Shewchuk, J. R. (1994). An introduction to the conjugate 
        gradient method without the agonizing pain.
        With the addition of different methods for the conjugation of the directions. See:
        Hager, W. W., & Zhang, H. (2006). A survey of nonlinear conjugate gradient methods. 
        Pacific journal of Optimization, 2(1), 35-58.
        """
        if conjugate_method not in ['polak-ribiere', 'fletcher-reeves', 'hestenes-stiefel', 'dai-yuan', 'hybrid']:
            raise ValueError('The conjugate method should be \'polak-ribiere\', '
                             '\'fletcher-reeves\', \'hestenes-stiefel\', \'dai-yuan\' or \'hybrid\'')
        
        dtype = grad_f.dtype or np.dtype(float)
        if dtype.kind == 'c':
            raise TypeError('The complex case is not yet implemented.')
        elif dtype.kind != 'f':
            dtype = np.dtype(float)

        if x0 is None:
            x0 = zeros(grad_f.shapein, dtype)

        with open('params.yml', "r") as stream:
            self.params = yaml.safe_load(stream)

        self.gif = create_gif
        self.center = center
        self.reso = reso
        self.figsize = figsize
        self.seenpix = seenpix
        self.jobid = jobid

        abnormal_stop_condition = MaxIterationStopCondition(
            maxiter,
            'Solver reached maximum number of iterations without reac'
            'hing specified tolerance.',
        )

        IterativeAlgorithm.__init__(
            self,
            x=x0,
            abnormal_stop_condition=abnormal_stop_condition,
            disp=disp,
            dtype=dtype,
            reuse_initial_state=reuse_initial_state,
            inplace_recursion=True,
            callback=callback,
        )

        if grad_f.shapein is None:
            raise ValueError('The operator input shape is not explicit.')
        
        self.grad_f = grad_f
        self.conjugate_method = conjugate_method
        self.comm = grad_f.commin
        self.norm = lambda x: _norm2(x, self.comm)
        self.dot = lambda x, y: _dot(x, y, self.comm)

        if self.gif:
            create_folder_if_not_exists(self.comm, f'gif_convergence_{self.jobid}')
        if M is None:
            M = IdentityOperator()
        self.M = asoperator(M)

        self.tol = tol
        self.sigma_0 = sigma_0
        self.maxiter_linesearch = maxiter_linesearch
        self.tol_linesearch = tol_linesearch
        self.d = empty(grad_f.shapeout, dtype)
        self.r = empty(grad_f.shapeout, dtype)
        self.s = empty(grad_f.shapeout, dtype)

    def initialize(self):
        IterativeAlgorithm.initialize(self)

        self.r[...] = -self.grad_f(self.x)
        self.M(self.r, self.s)
        self.d[...] = self.s
        self.delta_new = self.dot(self.r, self.d)
        self.delta_0 = self.delta_new
        self.error = np.sqrt(self.delta_new / self.delta_0)

    def iteration(self):
        j = 0
        self.delta_d = self.dot(self.d, self.d)
        self.alpha = -self.sigma_0
        self.eta_prev = self.dot(self.grad_f(self.x - self.alpha * self.d), self.d)

        while True: # emulating a do... while loop
            self.eta = self.dot(self.grad_f(self.x), self.d)
            if (self.eta_prev - self.eta) == 0:
                break
            self.alpha *= self.eta / (self.eta_prev - self.eta)
            self.x += self.alpha * self.d
            self.eta_prev = self.eta
            j += 1
            if (j >= self.maxiter_linesearch or self.alpha**2 * self.delta_d <= self.tol_linesearch**2):
                break

        if self.gif:
            if self.comm is not None:
                if self.comm.Get_rank() == 0:
                    plt.figure(figsize=self.figsize)
                    print(self.x.shape)
                    k=1
                    for i in range(self.x.shape[0]):
                        for j in range(self.x.shape[2]):
                            mymap = self.x[i, :, j].copy()
                            mymap[~self.seenpix] = hp.UNSEEN
                            hp.gnomview(mymap, rot=self.center, reso=self.reso, 
                                        cmap='jet', sub=(self.x.shape[0]*2, self.x.shape[2]*2, k), title='Map', notext=True,
                                        min=-3*np.std(self.x[0, self.seenpix, j]), max=3*np.std(self.x[0, self.seenpix, j]))
                            hp.gnomview(self.input_map[i, :, j] - mymap, rot=self.center, 
                                        reso=self.reso, cmap='jet', sub=(self.x.shape[0]*2, self.x.shape[2]*2, k+1), title='Residual', 
                                        notext=True, min=-3*np.std(self.x[0, self.seenpix, j]), max=3*np.std(self.x[0, self.seenpix, j]))
                            k+=2
                    plt.suptitle(f'Iteration : {self.niterations}')
                    plt.savefig(f'gif_convergence_{self.jobid}/maps_{self.niterations}.png')
                    plt.close()
            else:
                plt.figure(figsize=self.figsize)

                k=1
                for i in range(self.x.shape[0]):
                    for j in range(self.x.shape[2]):
                        mymap = self.x[i, :, j].copy()
                        mymap[~self.seenpix] = hp.UNSEEN
                        hp.gnomview(mymap, rot=self.center, reso=self.reso, cmap='jet', 
                                    sub=(self.x.shape[0], self.x.shape[2], k), title='', notext=True,
                                    min=-3*np.std(self.x[0, self.seenpix, j]), max=3*np.std(self.x[0, self.seenpix, j]))
                
                        k+=1
                plt.suptitle(f'Iteration : {self.niterations}')
                plt.savefig(f'gif_convergence_{self.jobid}/maps_{self.niterations}.png')
                plt.close()

        
        if self.conjugate_method in ['hestenes-stiefel', 'dai-yuan', 'hybrid']:
            self.delta_dr = self.dot(self.d, self.r)
        self.r[...] = -self.grad_f(self.x)
        if self.conjugate_method in ['polak-ribiere', 'fletcher-reeves']:
            self.delta_old = self.delta_new
        if self.conjugate_method in ['polak-ribiere', 'hestenes-stiefel', 'hybrid']:
            self.delta_mid = self.dot(self.r, self.s)
        self.M(self.r, self.s)
        self.delta_new = self.dot(self.r, self.s)

        self.error = np.sqrt(self.delta_new / self.delta_0)
        if self.error < self.tol:
            raise StopIteration('Solver reached maximum tolerance.')

        if self.conjugate_method == 'polak-ribiere':
            self.beta = np.max(((self.delta_new - self.delta_mid) / self.delta_old, 0))
        elif self.conjugate_method == 'fletcher-reeves':
            self.beta = self.delta_new/self.delta_old
        elif self.conjugate_method == 'hestenes-stiefel':
            self.beta = np.max(((self.delta_new - self.delta_mid) / (self.delta_dr - self.dot(self.d, self.r)), 0))
        elif self.conjugate_method == 'dai-yuan':
            self.beta = self.delta_new / (self.delta_dr - self.dot(self.d, self.r))
        else: # 'hybrid'
            self.beta = np.max((np.min((self.delta_new - self.delta_mid, self.delta_new)) / (self.delta_dr - self.dot(self.d, self.r)), 0))
        self.d *= self.beta
        self.d += self.s
    
    @staticmethod
    def callback(self):
        if self.disp:
            print(f'{self.niterations:4}: {self.error}')

def non_linear_pcg(
    grad_f,
    conjugate_method='polak-ribiere',
    x0=None,
    tol=1.0e-5,
    maxiter=300,
    sigma_0=1.0e-3,
    maxiter_linesearch=10,
    tol_linesearch=1.0e-2,
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
    grad_f : {Non-linear operator}
        The gradient of the function f. Applying grad_f on a vector of shape (N,) 
        should return a vector of the same shape.
    conjugate_method : {string}
        Can be 'polak-ribiere', 'fletcher-reeves', 'hestenes-stiefel', 'dai-yuan' or 'hybrid'
        Different methods are possible for conjugating the successive search directions.
        Here five choices are implemented, many more are possible.
        One should try them to see which one is the fastest in each problem.
        'hybrid' is a combination of 'hestenes-stiefel' and 'dai-yuan'.
        For more details, see: Hager, W. W., & Zhang, H. (2006). A survey of nonlinear 
        conjugate gradient methods. Pacific journal of Optimization, 2(1), 35-58.
    x0  : {array, matrix}
        Starting guess for the solution.
    tol : float, optional
        Tolerance to achieve. The algorithm terminates when either the
        relative residual is below `tol`.
    maxiter : integer, optional
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    sigma_0 : float
        Initial guess of the size of the step made at each iteration.
    maxiter_linesearch : integer
        Maximum number of iterations to find the best estimation of the size 
        of the step made at each iteration.
    tol_linesearch : float
        Tolerance to achieve for the size of the step made at each iteration.
    M : {Non-linear operator}, optional
        Preconditioner for the PCG. It should approximate the inverse of the hessian
        matrix of f. M(x) should be easy to cumpute. Effective preconditioning 
        improves the rate of convergence, which implies that fewer iterations 
        are needed to reach a given error tolerance.
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
        'algorithm' : the NonLinearPCGAlgorithm instance (the callback function has
                      access to it and can store information in it)

    Note
    ----
    This code is an implementation of algorithm B5 from:
    Shewchuk, J. R. (1994). An introduction to the conjugate 
    gradient method without the agonizing pain.
    With the addition of different methods for the conjugation of the directions. See:
    Hager, W. W., & Zhang, H. (2006). A survey of nonlinear conjugate gradient methods. 
    Pacific journal of Optimization, 2(1), 35-58.
    """
    time0 = time.time()
    algo = NonLinearPCGAlgorithm(
        grad_f,
        conjugate_method,
        x0=x0,
        tol=tol,
        maxiter=maxiter,
        sigma_0=sigma_0,
        maxiter_linesearch=maxiter_linesearch,
        tol_linesearch=tol_linesearch,
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