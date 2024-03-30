import time

import numpy as np

from pyoperators.core import IdentityOperator, asoperator
from pyoperators.memory import empty, zeros
from pyoperators.utils.mpi import MPI
from pyoperators.iterative.core import AbnormalStopIteration, IterativeAlgorithm
from pyoperators.iterative.stopconditions import MaxIterationStopCondition
from simtools.foldertools import *
import matplotlib.pyplot as plt
import healpy as hp
from costfunc.chi2 import * 
from functools import partial

__all__ = ['pcg', 'conjugate_gradient']

def grad(f,x): 
    '''
    CENTRAL FINITE DIFFERENCE CALCULATION
    '''
    h = 0.001#np.cbrt(np.finfo(float).eps)
    d = len(x)
    nabla = np.zeros(d)
    for i in range(d): 
        x_for = np.copy(x) 
        x_back = np.copy(x)
        x_for[i] += h 
        x_back[i] -= h 
        nabla[i] = (f(x_for) - f(x_back))/(2*h) 
    return nabla 

def line_search(f,x,p,nabla, maxiter=20):
    '''
    BACKTRACK LINE SEARCH WITH WOLFE CONDITIONS
    '''
    #print('p = ', p)
    a = 1
    print(a, p)
    c1 = 1e-4 
    c2 = 0.9 
    fx = f(x)
    x_new = x + a * p 
    nabla_new = grad(f,x_new)
    k=1
    while f(x_new) >= fx + (c1*a*nabla.T@p) or nabla_new.T@p <= c2*nabla.T@p : 
        if k > maxiter:
            break
        print(x_new, a)
        a *= 0.5
        x_new = x + a * p 
        nabla_new = grad(f,x_new)
        k+=1
    return a


class ConjugateGradientVaryingBeta:
    
    def __init__(self, pip, x0, comm, solution, allbeta, patch_ids):
        
        self.pip = pip
        self.sims = self.pip.sims
        self.x = x0
        self.comm = comm
        self.solution = solution
        self.allbeta = allbeta
        self.patch_ids = patch_ids
        self.rank = self.comm.Get_rank()
        
    def _gradient(self, beta, patch_id):
        
        beta_map = self.allbeta.copy()
        beta_map[patch_id, 0] = beta.copy()
        
        H_i = self.sims.joint.get_operator(beta_map, gain=self.sims.g_iter, fwhm=self.sims.fwhm_recon, nu_co=self.sims.nu_co)

        _d_sims = H_i(self.solution)

        
        _nabla = _d_sims.T @ self.sims.invN_beta(self.sims.TOD_obs - _d_sims)
        
        
        return self.comm.allreduce(_nabla, op=MPI.SUM)
    
    def _backtracking(self, nabla, x, patch_id):
        _inf = True
        a = 1e-10
        c1 = 1e-4
        c2 = 0.99
        fx = self.f(x)
        x_new = x + a * nabla
        nabla_new = np.array([self._gradient(x_new, patch_id)])
        nabla = np.array([nabla])
        k=0
        while _inf:
            if self.f(x_new) >= fx + (c1 * a * nabla.T @ -nabla):
                break
            elif nabla_new.T @ -nabla <= c2 * nabla.T @ nabla : 
                break
            else:
                print(f'{a}, {x_new}, {self.f(x_new):.3e}, {fx + (c1 * a * nabla.T @ -nabla):.3e}, {nabla_new.T @ -nabla:.3e}, {c2 * nabla.T @ nabla:.3e}')
                a *= 0.5
                x_new = x + a * nabla
                nabla_new = np.array([self._gradient(x_new, patch_id)])
        return a
    
    def run(self, maxiter=20, tol=1e-8):
        
        _inf = True
        k=0
        nabla = np.zeros(len(self.patch_ids))
        alpha = np.zeros(len(self.patch_ids))
        
        self.f = partial(self.pip.chi2.cost_function, solution=self.solution, allbeta=self.allbeta, patch_id=self.patch_ids)
        while _inf:
            k += 1
            
            for i in range(len(self.patch_ids)):
                nabla[i] = self._gradient(self.x[i], self.patch_ids[i])
                alpha[i] = self._backtracking(nabla[i], self.x[i], self.patch_ids[i])
            #print(nabla)
            #print(alpha)
            
            _r = self.x.copy()
            self.x += nabla * alpha
            _r -= self.x.copy()
            
            if self.rank == 0:
                print(f'Iter = {k}    x = {self.x}    tol = {np.sum(abs(_r)):.3e}   alpha = {alpha}   d = {nabla}')
            
            if k+1 > maxiter:
                _inf=False
                return self.x
            
            if np.sum(abs(_r)) < tol:
                _inf=False
                return self.x
        
class ConjugateGradientConstantBeta:
    
    def __init__(self, pip, x0, comm, solution):
        
        self.pip = pip
        self.sims = self.pip.sims
        self.x = x0
        self.comm = comm
        self.solution = solution
        self.rank = self.comm.Get_rank()
    
    
    def _gradient(self, beta):
        
        H_i = self.sims.joint.get_operator(beta, gain=self.sims.g_iter, fwhm=self.sims.fwhm_recon, nu_co=self.sims.nu_co)

        _d_sims = H_i(self.solution)

        
        _nabla = _d_sims.T @ self.sims.invN_beta(self.sims.TOD_obs - _d_sims)
        
        
        return self.comm.allreduce(_nabla, op=MPI.SUM)
    
    def _backtracking(self, nabla, x):
        a = 1e-9
        c1 = 1e-4
        c2 = 0.05
        fx = self.f(x, solution=self.solution)
        x_new = x + a * nabla
        nabla_new = self._gradient(x_new)

        while self.f(x_new, solution=self.solution) >= fx + (c1 * a * nabla.T * nabla) or nabla_new.T * nabla <= c2 * nabla.T * nabla : 
            
            a *= 0.5
            x_new = x + a * nabla
            nabla_new = self._gradient(x_new)
        return a

    def run(self, maxiter=20, tol=1e-8, tau=0.1):
        
        _inf = True
        k=0
        
        self.f = partial(self.pip.chi2.cost_function)
        while _inf:
            k += 1
                
            nabla = self._gradient(self.x)
            alphak = self._backtracking(nabla, self.x)
                
            _r = self.x.copy()
            self.x += nabla * alphak
            _r -= self.x.copy()
            
            if self.rank == 0:
                print(f'Iter = {k}    x = {self.x}    tol = {np.sum(abs(_r)):.3e}   alpha = {alphak}   d = {nabla}')
            
            if k+1 > maxiter:
                _inf=False
                return self.x
            
            if abs(_r) < tol:
                _inf=False
                return self.x
    
    
            
    
    def run_varying(self, patch_ids, maxiter=200, tol=1e-8, tau=0.1):
    
        _inf = True
        k=0
        while _inf:
            
            k += 1
            nabla = np.zeros(len(patch_ids))
            alphak = np.zeros(len(patch_ids))
        
            for i in range(len(patch_ids)):
                beta_map = self.allbeta.copy()
                beta_map[patch_ids[i], 0] = self.x[i]

                nabla[i] = self._gradient_varying(beta_map)
                alphak[i] = self._backtracking(nabla[i], np.array([self.x[i]]))
            
            
            pk = nabla * alphak
            _r = self.x.copy()
            self.x += pk
            _r -= self.x.copy()
            
            self.comm.Barrier()
            
            if self.comm.Get_rank() == 0:
                print(f'Iter = {k}    x = {self.x}    tol = {np.sum(abs(_r)):.6e}   dk = {nabla}')
            
            
            if k+1 > maxiter:
                _inf=False
                return self.x
            
            if np.sum(abs(_r)) < tol:
                _inf=False
                return self.x
        

class CG:
    
    '''
    
    Instance to perform conjugate gradient on cost function.
    
    '''
    
    def __init__(self, chi2, x0, eps, comm):
        
        '''
        
        Arguments :
        -----------
            - fun  :         Cost function to minimize
            - eps  : float - Step size for integration
            - x0   : array - Initial guess 
            - comm : MPI communicator (used only to display messages, fun is already parallelized)
        
        '''
        self.x = x0
        self.chi2 = chi2
        self.eps = eps
        self.comm = comm
        
    def _gradient(self, x):
        
        fx_plus_eps = self.chi2(x+self.eps)
        fx = self.chi2(x)
        fx_plus_eps = self.comm.allreduce(fx_plus_eps, op=MPI.SUM)
        fx = self.comm.allreduce(fx, op=MPI.SUM)
        return (fx_plus_eps - fx) / self.eps
    def __call__(self, maxiter=20, tol=1e-3, verbose=False):
        
        '''
        
        Callable method to run conjugate gradient.
        
        Arguments :
        -----------
            - maxiter : int   - Maximum number of iterations
            - tol     : float - Tolerance
            - verbose : bool  - Display message
        
        '''
        
        _inf = True
        k=0

        if verbose:
            if self.comm.Get_rank() == 0:
                print('Iter       x            Grad                Tol')
        
        while _inf:
            k += 1
            
            _grad = self._gradient(self.x)
            
            _r = self.x[0]
            self.x -= _grad * self.eps
            _r -= self.x[0]
            
            if verbose:
                if self.comm.Get_rank() == 0:
                    print(f'{k}    {self.x[0]:.6e}    {_grad:.6e}     {abs(_r):.6e}')
            
            if k+1 > maxiter:
                _inf=False
                
                return self.x
            
            if abs(_r) < tol:
                _inf=False
                return self.x


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
        disp=True,
        callback=None,
        reuse_initial_state=False,
        create_gif=False,
        center=None,
        reso=15,
        figsize=(10, 8),
        seenpix=None,
        truth=None,
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

        self.gif = create_gif
        self.center = center
        self.reso = reso
        self.figsize = figsize
        self.seenpix = seenpix
        self.truth = truth
        self.stk = ['I', 'Q', 'U']

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
        self.comm = A.commin
        if self.comm is None:
            self.rank = 0
        else:
            self.rank = self.comm.Get_rank()
        self.norm = lambda x: _norm2(x, self.comm)
        self.dot = lambda x, y: _dot(x, y, self.comm)

        if self.gif:
            if self.rank == 0:
                create_folder_if_not_exists('gif_convergence')
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

        if self.gif:
            if self.rank == 0:
                plt.figure(figsize=self.figsize)

                k=1
                for i in range(self.x.shape[0]):
                    for j in range(self.x.shape[2]):
                        mymap = self.x[i, :, j].copy()
                        mymap[~self.seenpix] = hp.UNSEEN
                        residuals = self.x[i, self.seenpix, j] - self.truth[i, self.seenpix, j]
                        hp.gnomview(mymap, rot=self.center, reso=self.reso, cmap='jet', sub=(self.x.shape[0], self.x.shape[2], k), notext=True,
                                min=-3*np.std(self.x[0, self.seenpix, j]), max=3*np.std(self.x[0, self.seenpix, j]), title=r'$\sigma^{}_{}$'.format(self.stk[j], i+1) + f' = {np.std(residuals):.3f}')
                
                        k+=1
                plt.suptitle(f'Iteration : {self.niterations}')
                plt.savefig(f'gif_convergence/maps_{self.niterations}.png')
                plt.close()

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
            print(f'{self.niterations:4}: {self.error:.4e} {time.time() - self.t0:.5f}')


def mypcg(
    A,
    b,
    x0=None,
    tol=1.0e-5,
    maxiter=300,
    M=None,
    disp=True,
    callback=None,
    reuse_initial_state=False,
    create_gif=False,
    center=None,
    reso=15,
    seenpix=None,
    truth=None
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
        callback=callback,
        reuse_initial_state=reuse_initial_state,
        create_gif=create_gif,
        center=center,
        reso=reso,
        seenpix=seenpix,
        truth=truth
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
