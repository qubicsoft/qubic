import os,sys,gc,pickle,yaml, emcee
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from functools import partial
from scipy.optimize import minimize, fmin, fmin_l_bfgs_b
from schwimmbad import MPIPool
from multiprocessing import Pool

import fgbuster.mixing_matrix as mm
import fgbuster.component_model as c

from pyoperators import MPI
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

from qubic import QskySim as qss
from qubic.Qacquisition import *
from qubic.Qutilities import *
from qubic.Qcostfunc import Chi2ConstantBlindJC, Chi2Parametric


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



               
class Plots:

    """
    
    Instance to produce plots on the convergence. 
    
    Arguments : 
    ===========
        - jobid : Int number for saving figures.
        - dogif : Bool to produce GIF.
    
    """
    
    def __init__(self, sims, dogif=True):
        
        self.sims = sims
        self.job_id = self.sims.job_id
        self.dogif = dogif
        self.params = self.sims.params
    
    def plot_beta_2d(self, allbeta, truth, figsize=(8, 6), ki=0):
        
        plt.figure(figsize=figsize)
        
        plt.plot(allbeta[:, 0], allbeta[:, 1], '-or')
        plt.axvline(truth[0], ls='--', color='black')
        plt.axhline(truth[1], ls='--', color='black')
        plt.xlabel(r'$\beta_d$', fontsize=12)
        plt.ylabel(r'$\beta_s$', fontsize=12)        
        plt.savefig(f'jobs/{self.job_id}/beta_2d_iter{ki+1}.png')
        if ki > 0:
            os.remove(f'jobs/{self.job_id}/beta_2d_iter{ki}.png')
        plt.close()    
    def plot_sed(self, nus, A, figsize=(8, 6), truth=None, ki=0):
        
        if self.params['Plots']['conv_beta']:
            
            nf = truth.shape[0]
            plt.figure(figsize=figsize)
            plt.subplot(2, 1, 1)
            allnus=np.linspace(120, 260, 100)
            
            for i in range(A[-1].shape[1]):
                plt.errorbar(nus, truth[:, i], fmt='ob')
                plt.errorbar(nus, A[-1][:, i], fmt='xr')
                #plt.plot(allnus, self.sims.comps[i+1].eval(allnus, np.array([1.54]))[0], '-k')
            
            plt.xlim(120, 260)
            #plt.ylim(1e-1, 5)
            #plt.yscale('log')
            
            plt.subplot(2, 1, 2)
            
            for j in range(A[-1].shape[1]):
                for i in range(nf):
                    _res = abs(truth[i, j] - A[:, i, j])
                    plt.plot(_res, '-r', alpha=0.5)
            #plt.ylim(None, 1)
            plt.xlim(0, self.sims.params['MapMaking']['pcg']['k'])
            plt.yscale('log')
            plt.savefig(f'jobs/{self.job_id}/A_iter{ki+1}.png')
            
            if ki > 0:
                os.remove(f'jobs/{self.job_id}/A_iter{ki}.png')
                
            plt.close()
            
            #do_gif(f'figures_{self.job_id}/', ki+1, 'A_iter', output='Amm.gif')
    def plot_beta_iteration(self, beta, figsize=(8, 6), truth=None, ki=0):

        """
        
        Method to plot beta as function of iteration. beta can have shape (niter) of (niter, nbeta)
        
        """

        if self.params['Plots']['conv_beta']:
            niter = beta.shape[0]
            alliter = np.arange(0, niter, 1)
            
            plt.figure(figsize=figsize)
            plt.subplot(2, 1, 1)
            if np.ndim(beta) == 1:
                plt.plot(alliter[1:]-1, beta[1:])
                if truth is not None:
                    plt.axhline(truth, ls='--', color='red')
            else:
                for i in range(beta.shape[1]):
                   
                    plt.plot(alliter, beta[:, i], '-k', alpha=0.3)
                    if truth is not None:
                        plt.axhline(truth[i], ls='--', color='red')

            plt.subplot(2, 1, 2)
            
            if np.ndim(beta) == 1:
                plt.plot(alliter[1:]-1, abs(truth - beta[1:]))
                #if truth is not None:
                    #plt.axhline(truth, ls='--', color='red')
            else:
                for i in range(beta.shape[1]):
                   
                    plt.plot(alliter, abs(truth[i] - beta[:, i]), '-k', alpha=0.3)
                    #if truth is not None:
                    #    plt.axhline(truth[i], ls='--', color='red')
            plt.yscale('log')
            plt.savefig(f'jobs/{self.job_id}/beta_iter{ki+1}.png')

            if ki > 0:
                os.remove(f'jobs/{self.job_id}/beta_iter{ki}.png')

            plt.close()
    def _display_allcomponents(self, seenpix, figsize=(14, 10), ki=0):
        
        stk = ['I', 'Q', 'U']
        if self.params['Plots']['maps']:
            if self.params['MapMaking']['qubic']['convolution']:
                C = HealpixConvolutionGaussianOperator(fwhm=self.sims.joint_out.qubic.allfwhm[-1])
            else:
                C = HealpixConvolutionGaussianOperator(fwhm=0)
            plt.figure(figsize=figsize)
            k=0
            for istk in range(3):
                for icomp in range(len(self.sims.comps_out)):
                    
                    if self.params['Foregrounds']['nside_fit'] == 0:
                        
                        if self.params['MapMaking']['qubic']['convolution'] or self.params['MapMaking']['qubic']['fake_convolution']:
                            map_in = self.sims.components_conv_out[icomp, :, istk].copy()
                            map_out = self.sims.components_iter[icomp, :, istk].copy()
                        else:
                            map_in = self.sims.components_conv_out[icomp, :, istk].copy()
                            map_out = self.sims.components_iter[icomp, :, istk].copy()
                            
                        sig = np.std(self.sims.components_out[icomp, seenpix, istk])
                        map_in[~seenpix] = hp.UNSEEN
                        map_out[~seenpix] = hp.UNSEEN
                        
                    else:
                        if self.params['MapMaking']['qubic']['convolution'] or self.params['MapMaking']['qubic']['fake_convolution']:
                            map_in = self.sims.components_conv_out[icomp, :, istk].copy()
                            map_out = self.sims.components_iter[istk, :, icomp].copy()
                            sig = np.std(self.sims.components_conv_out[icomp, seenpix, istk])
                        else:
                            map_in = self.sims.components_out[istk, :, icomp].copy()
                            map_out = self.sims.components_iter[istk, :, icomp].copy()
                            sig = np.std(self.sims.components_out[istk, seenpix, icomp])
                        map_in[~seenpix] = hp.UNSEEN
                        map_out[~seenpix] = hp.UNSEEN
                        
                    r = map_in - map_out
                    _reso = 15
                    nsig = 3
                    #r[~seenpix] = hp.UNSEEN
                    hp.gnomview(map_out, rot=self.sims.center, reso=_reso, notext=True, title=f'{self.sims.comps_name_out[icomp]} - {stk[istk]} - Output',
                        cmap='jet', sub=(3, len(self.sims.comps_out)*2, k+1), min=-nsig*sig, max=nsig*sig)
                    k+=1
                    hp.gnomview(r, rot=self.sims.center, reso=_reso, notext=True, title=f'{self.sims.comps_name_out[icomp]} - {stk[istk]} - Residual',
                        cmap='jet', sub=(3, len(self.sims.comps_out)*2, k+1), min=-nsig*np.std(r[seenpix]), max=nsig*np.std(r[seenpix]))
                    
                    k+=1
            
            plt.tight_layout()
            plt.savefig(f'jobs/{self.job_id}/allcomps/allcomps_iter{ki+1}.png')
            
            if self.sims.rank == 0:
                if ki > 0:
                    os.remove(f'jobs/{self.job_id}/allcomps/allcomps_iter{ki}.png')

            plt.close()
    def display_maps(self, seenpix, ngif=0, figsize=(14, 8), nsig=6, ki=0):
        
        """
        
        Method to display maps at given iteration.
        
        Arguments:
        ----------
            - seenpix : array containing the id of seen pixels.
            - ngif    : Int number to create GIF with ngif PNG image.
            - figsize : Tuple to control size of plots.
            - nsig    : Int number to compute errorbars.
        
        """
        
        seenpix = self.sims.coverage/self.sims.coverage.max() > 0.2#self.sims.params['MapMaking']['planck']['thr']
        
        if self.params['Plots']['maps']:
            stk = ['I', 'Q', 'U']
            rms_i = np.zeros((1, 2))
            
            for istk, s in enumerate(stk):
                plt.figure(figsize=figsize)

                k=0
                
                for icomp in range(len(self.sims.comps_out)):
                    
                    if self.params['Foregrounds']['nside_fit'] == 0:
                        if self.params['MapMaking']['qubic']['convolution'] or self.params['MapMaking']['qubic']['fake_convolution']:
                            map_in = self.sims.components_conv_out[icomp, :, istk].copy()
                            map_out = self.sims.components_iter[icomp, :, istk].copy()
                        else:
                            map_in = self.sims.components_out[icomp, :, istk].copy()
                            map_out = self.sims.components_iter[icomp, :, istk].copy()
                            
                    else:
                        if self.params['MapMaking']['qubic']['convolution'] or self.params['MapMaking']['qubic']['fake_convolution']:
                            map_in = self.sims.components_conv_out[icomp, :, istk].copy()
                            map_out = self.sims.components_iter[istk, :, icomp].copy()
                        else:
                            map_in = self.sims.components_out[istk, :, icomp].copy()
                            map_out = self.sims.components_iter[istk, :, icomp].copy()
                    
                    sig = np.std(map_in[seenpix])
                    map_in[~seenpix] = hp.UNSEEN
                    map_out[~seenpix] = hp.UNSEEN
                    r = map_in - map_out
                    r[~seenpix] = hp.UNSEEN
                    if icomp == 0:
                        if istk > 0:
                            rms_i[0, istk-1] = np.std(r[seenpix])
                    
                    _reso = 15
                    nsig = 3
                    hp.gnomview(map_in, rot=self.sims.center, reso=_reso, notext=True, title='',
                        cmap='jet', sub=(len(self.sims.comps_out), 3, k+1), min=-nsig*sig, max=nsig*sig)
                    hp.gnomview(map_out, rot=self.sims.center, reso=_reso, notext=True, title='',
                        cmap='jet', sub=(len(self.sims.comps_out), 3, k+2), min=-nsig*sig, max=nsig*sig)
                    #
                    hp.gnomview(r, rot=self.sims.center, reso=_reso, notext=True, title=f"{np.std(r[seenpix]):.3e}",
                        cmap='jet', sub=(len(self.sims.comps_out), 3, k+3), min=-nsig*sig, max=nsig*sig)
                    
                    
                    
                    #hp.mollview(map_in, notext=True, title='',
                    #    cmap='jet', sub=(len(self.sims.comps), 3, k+1), min=-2*sig, max=2*sig)
                    #hp.mollview(map_out, notext=True, title='',
                    #    cmap='jet', sub=(len(self.sims.comps), 3, k+2), min=-2*sig, max=2*sig)
                     
                    #hp.mollview(r, notext=True, title=f"{np.std(r[seenpix]):.3e}",
                    #    cmap='jet', sub=(len(self.sims.comps), 3, k+3))#, min=-1*sig, max=1*sig)

                    k+=3
                    
                
                #print(rms_i)
                
                plt.tight_layout()
                plt.savefig(f'jobs/{self.job_id}/{s}/maps_iter{ki+1}.png')
                
                if self.sims.rank == 0:
                    if ki > 0:
                        os.remove(f'jobs/{self.job_id}/{s}/maps_iter{ki}.png')

                plt.close()
            self.sims.rms_plot = np.concatenate((self.sims.rms_plot, rms_i), axis=0)
    def plot_gain_iteration(self, gain, alpha, figsize=(8, 6), ki=0):
        
        """
        
        Method to plot convergence of reconstructed gains.
        
        Arguments :
        -----------
            - gain    : Array containing gain number (1 per detectors). It has the shape (Niteration, Ndet, 2) for Two Bands design and (Niteration, Ndet) for Wide Band design
            - alpha   : Transparency for curves.
            - figsize : Tuple to control size of plots.
            
        """
        
        
        if self.params['Plots']['conv_gain']:
            
            plt.figure(figsize=figsize)

            
            
            niter = gain.shape[0]
            ndet = gain.shape[1]
            alliter = np.arange(1, niter+1, 1)

            #plt.hist(gain[:, i, j])
            if self.params['MapMaking']['qubic']['type'] == 'two':
                color = ['red', 'blue']
                for j in range(2):
                    plt.hist(gain[-1, :, j], bins=20, color=color[j])
            #        plt.plot(alliter-1, np.mean(gain, axis=1)[:, j], color[j], alpha=1)
            #        for i in range(ndet):
            #            plt.plot(alliter-1, gain[:, i, j], color[j], alpha=alpha)
                        
            #elif self.params['MapMaking']['qubic']['type'] == 'wide':
            #    color = ['--g']
            #    plt.plot(alliter-1, np.mean(gain, axis=1), color[0], alpha=1)
            #    for i in range(ndet):
            #        plt.plot(alliter-1, gain[:, i], color[0], alpha=alpha)
                        
            #plt.yscale('log')
            #plt.ylabel(r'|$g_{reconstructed} - g_{input}$|', fontsize=12)
            #plt.xlabel('Iterations', fontsize=12)
            plt.xlim(-0.1, 0.1)
            plt.ylim(0, 100)
            plt.axvline(0, ls='--', color='black')
            plt.savefig(f'jobs/{self.job_id}/gain_iter{ki+1}.png')

            if self.sims.rank == 0:
                if ki > 0:
                    os.remove(f'jobs/{self.job_id}/gain_iter{ki}.png')

            plt.close()
    def plot_rms_iteration(self, rms, figsize=(8, 6), ki=0):
        
        if self.params['Plots']['conv_rms']:
            plt.figure(figsize=figsize)
            
            plt.plot(rms[1:, 0], '-b', label='Q')
            plt.plot(rms[1:, 1], '-r', label='U')
            
            #plt.ylim(1e-, None)
            plt.yscale('log')
            
            plt.tight_layout()
            plt.savefig(f'jobs/{self.job_id}/rms_iter{ki+1}.png')
                
            if self.sims.rank == 0:
                if ki > 0:
                    os.remove(f'jobs/{self.job_id}/rms_iter{ki}.png')

            plt.close()
            #rms = np.std(maps[:, seenpix, :], axis=1)     # Can be (Ncomps, Nstk) or (Nstk, Ncomps)
            
            #print(rms.shape)
            #stop



class Pipeline:


    """
    
    Main instance to create End-2-End pipeline for components reconstruction.
    
    Arguments :
    -----------
        - comm    : MPI common communicator (define by MPI.COMM_WORLD).
        - seed    : Int number for CMB realizations.
        - it      : Int number for noise realizations.
        
    """
    
    def __init__(self, comm, seed, seed_noise=None):
        
        if seed_noise == -1:
            if comm.Get_rank() == 0:
                seed_noise = np.random.randint(100000000)
            else:
                seed_noise = None
        seed_noise = comm.bcast(seed_noise, root=0)
        self.sims = PresetSims(comm, seed, seed_noise)
        
        if self.sims.params['Foregrounds']['type'] == 'parametric':
            pass
        elif self.sims.params['Foregrounds']['type'] == 'blind':
            self.chi2 = Chi2ConstantBlindJC(self.sims)
        else:
            raise TypeError(f"{self.sims.params['Foregrounds']['type']} is not yet implemented..")
        self.plots = Plots(self.sims, dogif=True)
        self._rms_noise_qubic_patch_per_ite = np.empty((self.sims.params['MapMaking']['pcg']['ites_to_converge'],len(self.sims.comps_out)))
        self._rms_noise_qubic_patch_per_ite[:] = np.nan
        
    def main(self):
        
        """
        
        Method to run the pipeline by following :
        
            1) Initialize simulation using `PresetSims` instance reading `params.yml`.
            
            2) Solve map-making equation knowing spectral index and gains.
            
            3) Fit spectral index knowing components and gains.
            
            4) Fit gains knowing components and sepctral index.
            
            5) Repeat 2), 3) and 4) until convergence.
        
        """
        
        self._info = True
        self._steps = 0
        
        while self._info:


            self._display_iter()
            
            ### Update self.components_iter^{k} -> self.components_iter^{k+1}
            self._update_components()
            
            ### Update self.beta_iter^{k} -> self.beta_iter^{k+1}
            if self.sims.params['Foregrounds']['fit_spectral_index']:
                self._update_spectral_index()
            else:
                self._index_seenpix_beta = None
                
            #stop
            ### Update self.g_iter^{k} -> self.g_iter^{k+1}
            if self.sims.params['MapMaking']['qubic']['fit_gain']:
                self._update_gain()
            
            ### Wait for all processes and save data inside pickle file
            self.sims.comm.Barrier()
            self._save_data()
            
            ### Compute the rms of the noise per iteration to later analyze its convergence in _stop_condition
            self._compute_maxrms_array()

            ### Stop the loop when self._steps > k
            self._stop_condition()
    def _compute_maps_convolved(self):
        
        """
        
        Method to compute convolved maps for each FWHM of QUBIC.
        
        """
        
        ### We make the convolution before beta estimation to speed up the code, we avoid to make all the convolution at each iteration
        ### Constant spectral index
        if self.sims.params['Foregrounds']['nside_fit'] == 0:
            components_for_beta = np.zeros((self.sims.params['MapMaking']['qubic']['nsub'], len(self.sims.comps), 12*self.sims.params['MapMaking']['qubic']['nside']**2, 3))
            for i in range(self.sims.params['MapMaking']['qubic']['nsub']):

                for jcomp in range(len(self.sims.comps)):
                    if self.sims.params['MapMaking']['qubic']['convolution']:
                        C = HealpixConvolutionGaussianOperator(fwhm = self.fwhm_recon[i], lmax=2*self.sims.params['MapMaking']['qubic']['nside'])
                    else:
                        C = HealpixConvolutionGaussianOperator(fwhm = 0)
                    components_for_beta[i, jcomp] = C(self.sims.components_iter[jcomp])
        else:
            components_for_beta = np.zeros((self.sims.params['MapMaking']['qubic']['nsub'], 3, 12*self.sims.params['MapMaking']['qubic']['nside']**2, len(self.sims.comps)))
            for i in range(self.sims.params['MapMaking']['qubic']['nsub']):
                for jcomp in range(len(self.sims.comps)):
                    if self.sims.params['MapMaking']['qubic']['convolution']:
                        C = HealpixConvolutionGaussianOperator(fwhm = self.sims.fwhm_recon[i], lmax=2*self.sims.params['MapMaking']['qubic']['nside'])
                    else:
                        C = HealpixConvolutionGaussianOperator(fwhm = 0)

                    components_for_beta[i, :, :, jcomp] = C(self.sims.components_iter[:, :, jcomp].T).T
        return components_for_beta
    def _callback(self, x):
        
        """
        
        Method to make callback function readable by `scipy.optimize.minimize`.
        
        """
        
        self.sims.comm.Barrier()
        if self.sims.rank == 0:
            if (self.nfev%10) == 0:
                print(f"Iter = {self.nfev:4d}   beta = {[np.round(x[i], 5) for i in range(len(x))]}")
            else:
                print(f"Iter = {self.nfev:4d}   beta = {[np.round(x[i], 5) for i in range(len(x))]}")
            
            #print(f"{self.nfev:4d}   {x[0]:3.6f}   {self.chi2.chi2_P:3.6e}")
            self.nfev += 1
    def _get_tod_comp(self):
        
        tod_comp = np.zeros((len(self.sims.comps_name_out), self.sims.joint_out.qubic.Nsub*2, self.sims.joint_out.qubic.ndets*self.sims.joint_out.qubic.nsamples))
        
        for i in range(len(self.sims.comps_name_out)):
            for j in range(self.sims.joint_out.qubic.Nsub*2):
                if self.sims.params['MapMaking']['qubic']['convolution']:
                    C = HealpixConvolutionGaussianOperator(fwhm = self.sims.fwhm_recon[j], lmax=2*self.sims.params['MapMaking']['qubic']['nside'])
                else:
                    C = HealpixConvolutionGaussianOperator(fwhm = 0, lmax=2*self.sims.params['MapMaking']['qubic']['nside'])
                tod_comp[i, j] = self.sims.joint_out.qubic.H[j](C(self.sims.components_iter[i])).ravel()
        
        return tod_comp
    def _get_tod_comp_superpixel(self, index):
        if self.sims.rank == 0:
            print('Computing contribution of each super-pixel')
        _index = np.zeros(12*self.sims.params['Foregrounds']['nside_fit']**2)
        _index[index] = index.copy()
        _index_nside = hp.ud_grade(_index, self.sims.joint_out.external.nside)
        tod_comp = np.zeros((len(index), self.sims.joint_out.qubic.Nsub*2, len(self.sims.comps_out), self.sims.joint_out.qubic.ndets*self.sims.joint_out.qubic.nsamples))
        
        maps_conv = self.sims.components_iter.T.copy()

        for j in range(self.sims.params['MapMaking']['qubic']['nsub']):
            for co in range(len(self.sims.comps_out)):
                if self.sims.params['MapMaking']['qubic']['convolution']:
                    C = HealpixConvolutionGaussianOperator(fwhm=self.sims.fwhm_recon[j], lmax=2*self.sims.params['MapMaking']['qubic']['nside'])
                else:
                    C = HealpixConvolutionGaussianOperator(fwhm=0, lmax=2*self.sims.params['MapMaking']['qubic']['nside'])
                maps_conv[co] = C(self.sims.components_iter[:, :, co].T).copy()
                for ii, i in enumerate(index):
        
                    maps_conv_i = maps_conv.copy()
                    _i = _index_nside == i
                    for stk in range(3):
                        maps_conv_i[:, :, stk] *= _i
                    tod_comp[ii, j, co] = self.sims.joint_out.qubic.H[j](maps_conv_i[co]).ravel()

        return tod_comp
    def _update_spectral_index(self):
        
        """
        
        Method that perform step 3) of the pipeline for 2 possible designs : Two Bands and Wide Band
        
        """
        
        if self.sims.params['Foregrounds']['type'] == 'parametric':
            if self.sims.params['Foregrounds']['nside_fit'] == 0:
                self._index_seenpix_beta = 0
                self.nfev = 0
                previous_beta = self.sims.beta_iter.copy()
                
                tod_comp = self._get_tod_comp()
                chi2 = Chi2Parametric(self.sims, tod_comp, self.sims.beta_iter, seenpix_wrap=None)
                
                self.sims.beta_iter = np.array([fmin_l_bfgs_b(chi2, 
                                                              x0=self.sims.beta_iter, callback=self._callback, approx_grad=True, epsilon=1e-6)[0]])
                #fun = partial(self.chi2._qu, tod_comp=self._get_tod_comp(), components=self.sims.components_iter, nus=self.sims.nus_eff[:self.sims.joint.qubic.Nsub*2])
                #self.sims.beta_iter = np.array([fmin_l_bfgs_b(fun, x0=self.sims.beta_iter, callback=self._callback, factr=100, approx_grad=True)[0]])
                
                del tod_comp
                gc.collect()
                
                if self.sims.rank == 0:
                
                    print(f'Iteration k     : {previous_beta}')
                    print(f'Iteration k + 1 : {self.sims.beta_iter.copy()}')
                    print(f'Truth           : {self.sims.beta_in.copy()}')
                    print(f'Residuals       : {self.sims.beta_in - self.sims.beta_iter}')
                    
                    if len(self.sims.comps_out) > 2:
                        self.plots.plot_beta_iteration(self.sims.allbeta[:, 0], truth=self.sims.beta_in[0], ki=self._steps)
                        self.plots.plot_beta_2d(self.sims.allbeta, truth=self.sims.beta_in, ki=self._steps)
                    else:
                        self.plots.plot_beta_iteration(self.sims.allbeta, truth=self.sims.beta_in, ki=self._steps)
            
                self.sims.comm.Barrier()
                print(self.sims.beta_iter.shape)
                print(self.sims.allbeta.shape)
                self.sims.allbeta = np.concatenate((self.sims.allbeta, self.sims.beta_iter), axis=0) 
                #stop
            else:
            
                index_num = hp.ud_grade(self.sims.seenpix_qubic, self.sims.params['Foregrounds']['nside_fit'])    #
                index = np.where(index_num == True)[0]
                index_num2 = hp.ud_grade(self.sims.seenpix_BB, self.sims.params['Foregrounds']['nside_fit'])    #
                index2 = np.where(index_num2 == True)[0]
                
                tod_comp = self._get_tod_comp_superpixel(index)#np.arange(12*self.sims.params['Foregrounds']['nside_fit']**2))
                chi2 = Chi2Parametric(self.sims, tod_comp, self.sims.beta_iter, seenpix_wrap=None)
                self._index_seenpix_beta = index.copy()#chi2._index.copy()
                
                previous_beta = self.sims.beta_iter[self._index_seenpix_beta, 0].copy()
                self.nfev = 0
                
                self.sims.beta_iter[index, 0] = np.array([fmin_l_bfgs_b(chi2, x0=self.sims.beta_iter[index, 0], 
                                                                              callback=self._callback, approx_grad=True, epsilon=1e-6, maxls=5, maxiter=5)[0]])
                
                #self.sims.beta_iter[self._index_seenpix_beta, 0] = minimize(chi2, x0=self.sims.beta_iter[self._index_seenpix_beta, 0] * 0 + 1.53,
                #                                                            callback=self._callback, method='L-BFGS-B', tol=1e-8, options={'eps':1e-5}).x
                del tod_comp
                gc.collect()
                #print(self.sims.beta_iter)
                
                self.sims.allbeta = np.concatenate((self.sims.allbeta, np.array([self.sims.beta_iter])), axis=0)
                
                if self.sims.rank == 0:
                
                    print(f'Iteration k     : {previous_beta}')
                    print(f'Iteration k + 1 : {self.sims.beta_iter[self._index_seenpix_beta, 0].copy()}')
                    print(f'Truth           : {self.sims.beta_in[self._index_seenpix_beta, 0].copy()}')
                    print(f'Residuals       : {self.sims.beta_in[self._index_seenpix_beta, 0] - self.sims.beta_iter[self._index_seenpix_beta, 0]}')
                    self.plots.plot_beta_iteration(self.sims.allbeta[:, self._index_seenpix_beta], 
                                                   truth=self.sims.beta_in[self._index_seenpix_beta, 0], 
                                                   ki=self._steps)
                    
                #stop
                            
        elif self.sims.params['Foregrounds']['type'] == 'blind':
            
            previous_step = self.sims.Amm_iter[:self.sims.joint_out.qubic.Nsub*2, 1:].copy()
            self.nfev = 0
            self._index_seenpix_beta = None
            
            
            for i in range(len(self.sims.comps_out)):
                if self.sims.comps_name_out[i] == 'Dust':
                    #print(self.sims.comps_name_out, i)
                    ### Cost function depending of [Ad, As]
                    tod_comp = self._get_tod_comp()    # (Nc, Nsub, NsNd)
                    #print('tod_comp -> ', tod_comp.shape)
                    #stop
                    fun = partial(self.chi2._qu, tod_comp=tod_comp, A=self.sims.Amm_iter, icomp=i)
            
                    ### Minimization
                    x0 = np.ones(self.sims.params['MapMaking']['qubic']['nrec_blind'])
                    #x0 = self.sims.Amm_iter[:self.sims.joint_out.qubic.Nsub*2, i]
                    bnds = [(0, None) for _ in range(x0.shape[0])]

                    
                    Ai = fmin_l_bfgs_b(fun, x0=x0, approx_grad=True, bounds=bnds, maxiter=30, 
                                   callback=self._callback, epsilon = 1e-6)[0]
                    
                    fsub = int(self.sims.joint_out.qubic.Nsub*2 / self.sims.params['MapMaking']['qubic']['nrec_blind'])
                    for ii in range(self.sims.params['MapMaking']['qubic']['nrec_blind']):
                        self.sims.Amm_iter[ii*fsub:(ii+1)*fsub, i] = np.array([Ai[ii]]*fsub)
                    #print(Ai)
                    #stop
                    #self.sims.Amm_iter[:self.sims.joint_out.qubic.Nsub*2, i] = Ai.copy()
                    
            
            
            self.sims.allAmm_iter = np.concatenate((self.sims.allAmm_iter, np.array([self.sims.Amm_iter])), axis=0)
            
            if self.sims.rank == 0:
                print(f'Iteration k     : {previous_step.ravel()}')
                print(f'Iteration k + 1 : {self.sims.Amm_iter[:self.sims.joint_out.qubic.Nsub*2, 1:].ravel()}')
                print(f'Truth           : {self.sims.Ammtrue[:self.sims.joint_out.qubic.Nsub*2, 1:].ravel()}')
                print(f'Residuals       : {self.sims.Ammtrue[:self.sims.joint_out.qubic.Nsub*2, 1:].ravel() - self.sims.Amm_iter[:self.sims.joint_out.qubic.Nsub*2, 1:].ravel()}')
               
                self.plots.plot_sed(self.sims.joint_out.qubic.allnus, 
                                        self.sims.allAmm_iter[:, :self.sims.joint_out.qubic.Nsub*2, 1:], 
                                        ki=self._steps, truth=self.sims.Ammtrue[:self.sims.joint_out.qubic.Nsub*2, 1:])

                #print('Amm ', self.sims.Amm_out)
                #print('Amm_iter ', self.sims.Amm_iter)
            #stop
        else:
            raise TypeError(f"{self.sims.params['Foregrounds']['type']} is not yet implemented..")          
    def _save_data(self):
        
        """
        
        Method that save data for each iterations. It saves components, gains, spectral index, coverage, seen pixels.
        
        """
        if self.sims.rank == 0:
            if self.sims.params['save'] != 0:
                if (self._steps+1) % self.sims.params['save'] == 0:
                    
                    if self.sims.params['lastite']:
                    
                        if self._steps != 0:
                            os.remove(self.sims.params['foldername'] + '/' + self.sims.params['filename']+f"_seed{str(self.sims.params['CMB']['seed'])}_{str(self.sims.job_id)}_k{self._steps-1}.pkl")
                
                    with open(self.sims.params['foldername'] + '/' + self.sims.params['filename']+f"_seed{str(self.sims.params['CMB']['seed'])}_{str(self.sims.job_id)}_k{self._steps}.pkl", 'wb') as handle:
                        pickle.dump({'components':self.sims.components_in, 
                                 'components_i':self.sims.components_iter,
                                 'beta':self.sims.allbeta,
                                 'beta_true':self.sims.beta_in,
                                 'index_beta':self._index_seenpix_beta,
                                 'g':self.sims.G,
                                 'gi':self.sims.Gi,
                                 'allg':self.sims.allg,
                                 'A':self.sims.Amm_iter,
                                 'Atrue':self.sims.Amm_in,
                                 'allA':self.sims.allAmm_iter,
                                 'G':self.sims.G,
                                 'nus_in':self.sims.nus_eff_in,
                                 'nus_out':self.sims.nus_eff_out,
                                 'center':self.sims.center,
                                 'coverage':self.sims.coverage,
                                 'seenpix':self.sims.seenpix,
                                 'fwhm':self.sims.fwhm}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def _update_components(self, maxiter=None):
        
        """
        
        Method that solve the map-making equation ( H.T * invN * H ) * components = H.T * invN * TOD using OpenMP / MPI solver. 
        
        """
        
        H_i = self.sims.joint_out.get_operator(self.sims.beta_iter, Amm=self.sims.Amm_iter, gain=self.sims.g_iter, fwhm=self.sims.fwhm_recon, nu_co=self.sims.nu_co)
        seenpix_var = self.sims.seenpix_qubic
        
        #print(H_i.shapein, H_i.shapeout)
        #stop
        if self.sims.params['Foregrounds']['nside_fit'] == 0:
            U = (
                ReshapeOperator((len(self.sims.comps_name_out) * sum(seenpix_var) * 3), (len(self.sims.comps_name_out), sum(seenpix_var), 3)) *
                PackOperator(np.broadcast_to(seenpix_var[None, :, None], (len(self.sims.comps_name_out), seenpix_var.size, 3)).copy())
            ).T
        else:
            U = (
                ReshapeOperator((3 * len(self.sims.comps_name_out) * sum(seenpix_var)), (3, sum(seenpix_var), len(self.sims.comps_name_out))) *
                PackOperator(np.broadcast_to(seenpix_var[None, :, None], (3, seenpix_var.size, len(self.sims.comps_name_out))).copy())
            ).T
        
        if self.sims.params['MapMaking']['planck']['fixpixels']:
            self.sims.A = U.T * H_i.T * self.sims.invN * H_i * U
            if self.sims.params['Foregrounds']['nside_fit'] == 0:
                if self.sims.params['MapMaking']['qubic']['convolution']:
                    x_planck = self.sims.components_conv_out * (1 - seenpix_var[None, :, None])
                else:
                    x_planck = self.sims.components_out * (1 - seenpix_var[None, :, None])
            self.sims.b = U.T (  H_i.T * self.sims.invN * (self.sims.TOD_obs - H_i(x_planck)))
        elif self.sims.params['MapMaking']['planck']['fixI']:
            mask = np.ones((len(self.sims.comps_out), 12*self.sims.params['MapMaking']['qubic']['nside']**2, 3))
            mask[:, :, 0] = 0
            P = (
                ReshapeOperator(PackOperator(mask).shapeout, (len(self.sims.comps_out), 12*self.sims.params['MapMaking']['qubic']['nside']**2, 2)) * 
                PackOperator(mask)
                ).T
            
            xI = self.sims.components_conv_out * (1 - mask)
            self.sims.A = P.T * H_i.T * self.sims.invN * H_i * P
            self.sims.b = P.T (  H_i.T * self.sims.invN * (self.sims.TOD_obs - H_i(xI)))
        else:
            self.sims.A = H_i.T * self.sims.invN * H_i
            self.sims.b = H_i.T * self.sims.invN * self.sims.TOD_obs
        
        self._call_pcg(maxiter=maxiter)
    def _call_pcg(self, maxiter=None):

        """
        
        Method that call the PCG in PyOperators.
        
        """
        if maxiter is None:
            maxiter=self.sims.params['MapMaking']['pcg']['maxiter']
        seenpix_var = self.sims.seenpix_qubic
        #self.sims.components_iter_minus_one = self.sims.components_iter.copy()
        
        if self.sims.params['MapMaking']['planck']['fixpixels']:
            mypixels = mypcg(self.sims.A, 
                                    self.sims.b, 
                                    M=self.sims.M, 
                                    tol=self.sims.params['MapMaking']['pcg']['tol'], 
                                    x0=self.sims.components_iter[:, seenpix_var, :], 
                                    maxiter=maxiter, 
                                    disp=True,
                                    create_gif=False,
                                    center=self.sims.center, 
                                    reso=self.sims.params['MapMaking']['qubic']['dtheta'], 
                                    seenpix=self.sims.seenpix, 
                                    truth=self.sims.components_out,
                                    reuse_initial_state=False)['x']['x']  
            self.sims.components_iter[:, seenpix_var, :] = mypixels.copy()
        elif self.sims.params['MapMaking']['planck']['fixI']:
            mypixels = mypcg(self.sims.A, 
                                    self.sims.b, 
                                    M=self.sims.M, 
                                    tol=self.sims.params['MapMaking']['pcg']['tol'], 
                                    x0=self.sims.components_iter[:, :, 1:], 
                                    maxiter=maxiter, 
                                    disp=True,
                                    create_gif=False,
                                    center=self.sims.center, 
                                    reso=self.sims.params['MapMaking']['qubic']['dtheta'], 
                                    seenpix=self.sims.seenpix_qubic, 
                                    truth=self.sims.components_out,
                                    reuse_initial_state=False)['x']['x']  
            self.sims.components_iter[:, :, 1:] = mypixels.copy()
        else:
            mypixels = mypcg(self.sims.A, 
                                    self.sims.b, 
                                    M=self.sims.M, 
                                    tol=self.sims.params['MapMaking']['pcg']['tol'], 
                                    x0=self.sims.components_iter, 
                                    maxiter=maxiter, 
                                    disp=True,
                                    create_gif=False,
                                    center=self.sims.center, 
                                    reso=self.sims.params['MapMaking']['qubic']['dtheta'], 
                                    seenpix=self.sims.seenpix_qubic, 
                                    truth=self.sims.components_out,
                                    reuse_initial_state=False)['x']['x']  
            self.sims.components_iter = mypixels.copy()
        #stop
        if self.sims.rank == 0:
            self.plots.display_maps(self.sims.seenpix_plot, ngif=self._steps+1, ki=self._steps)
            self.plots._display_allcomponents(self.sims.seenpix_plot, ki=self._steps)  
            self.plots.plot_rms_iteration(self.sims.rms_plot, ki=self._steps) 
    def _compute_map_noise_qubic_patch(self):
        
        """
        
        Compute the rms of the noise within the qubic patch.
        
        """
        nbins = 1 #average over the entire qubic patch

        if self.sims.params['Foregrounds']['nside_fit'] == 0:
            if self.sims.params['MapMaking']['qubic']['convolution']:
                residual = self.sims.components_iter - self.sims.components_conv_out
            else:
                residual = self.sims.components_iter - self.sims.components_out
        else:
            if self.sims.params['MapMaking']['qubic']['convolution']:
                residual = self.sims.components_iter.T - self.sims.components_conv_out
            else:
                residual = self.sims.components_iter.T - self.sims.components_out.T
        rms_maxpercomp = np.zeros(len(self.sims.comps_out))

        for i in range(len(self.sims.comps_out)):
            angs,I,Q,U,dI,dQ,dU = get_angular_profile(residual[i],thmax=self.sims.angmax,nbins=nbins,doplot=False,allstokes=True,separate=True,integrated=True,center=self.sims.center)
                
            ### Set dI to 0 to only keep polarization fluctuations 
            dI = 0
            rms_maxpercomp[i] = np.max([dI,dQ,dU])
        return rms_maxpercomp
    def _compute_maxrms_array(self):

        if self._steps <= self.sims.params['MapMaking']['pcg']['ites_to_converge']-1:
            self._rms_noise_qubic_patch_per_ite[self._steps,:] = self._compute_map_noise_qubic_patch()
        elif self._steps > self.sims.params['MapMaking']['pcg']['ites_to_converge']-1:
            self._rms_noise_qubic_patch_per_ite[:-1,:] = self._rms_noise_qubic_patch_per_ite[1:,:]
            self._rms_noise_qubic_patch_per_ite[-1,:] = self._compute_map_noise_qubic_patch()
    def _stop_condition(self):
        
        """
        
        Method that stop the convergence if there are more than k steps.
        
        """
        
        if self._steps >= self.sims.params['MapMaking']['pcg']['ites_to_converge']-1:
            
            deltarms_max_percomp = np.zeros(len(self.sims.comps_out))

            for i in range(len(self.sims.comps_out)):
                deltarms_max_percomp[i] = np.max(np.abs((self._rms_noise_qubic_patch_per_ite[:,i] - self._rms_noise_qubic_patch_per_ite[-1,i]) / self._rms_noise_qubic_patch_per_ite[-1,i]))

            deltarms_max = np.max(deltarms_max_percomp)
            if self.sims.rank == 0:
                print(f'Maximum RMS variation for the last {self.sims.ites_rms_tolerance} iterations: {deltarms_max}')

            if deltarms_max < self.sims.params['MapMaking']['pcg']['noise_rms_variation_tolerance']:
                print(f'RMS variations lower than {self.sims.rms_tolerance} for the last {self.sims.ites_rms_tolerance} iterations.')
                
                ### Update components last time with converged parameters
                #self._update_components(maxiter=100)
                self._info = False        

        if self._steps >= self.sims.params['MapMaking']['pcg']['k']-1:
            
            ### Update components last time with converged parameters
            #self._update_components(maxiter=100)
            
            ### Wait for all processes and save data inside pickle file
            #self.sims.comm.Barrier()
            #self._save_data()
            
            self._info = False
            
        self._steps += 1
    def _display_iter(self):
        
        """
        
        Method that display the number of a specific iteration k.
        
        """
        
        if self.sims.rank == 0:
            print('========== Iter {}/{} =========='.format(self._steps+1, self.sims.params['MapMaking']['pcg']['k']))
    def _update_gain(self):
        
        """
        
        Method that compute gains of each detectors using semi-analytical method g_i = TOD_obs_i / TOD_sim_i
        
        """
        
        self.H_i = self.sims.joint_out.get_operator(self.sims.beta_iter, Amm=self.sims.Amm_iter, gain=np.ones(self.sims.g_iter.shape), fwhm=self.sims.fwhm_recon, nu_co=self.sims.nu_co)
        self.nsampling = self.sims.joint_out.qubic.nsamples
        self.ndets = self.sims.joint_out.qubic.ndets
        if self.sims.params['MapMaking']['qubic']['type'] == 'wide':
            _r = ReshapeOperator(self.sims.joint_out.qubic.ndets*self.sims.joint.qubic.nsamples, (self.sims.joint.qubic.ndets, self.sims.joint.qubic.nsamples))
            #R2det_i = ReshapeOperator(self.sims.joint.qubic.ndets*self.sims.joint.qubic.nsamples, (self.sims.joint.qubic.ndets, self.sims.joint.qubic.nsamples))
            #print(R2det_i.shapein, R2det_i.shapeout)
            #TOD_Q_ALL_i = R2det_i(self.H_i.operands[0](self.sims.components_iter))
            TODi_Q = self.sims.invN.operands[0](self.H_i.operands[0](self.sims.components_iter)[:ndets*nsampling])
            self.sims.g_iter = self._give_me_intercal(TODi_Q, _r(self.sims.TOD_Q))
            self.sims.g_iter /= self.sims.g_iter[0]
            self.sims.allg = np.concatenate((self.sims.allg, np.array([self.sims.g_iter])), axis=0)
            
        elif self.sims.params['MapMaking']['qubic']['type'] == 'two':
            
            

            #R2det_i = ReshapeOperator(2*self.sims.joint.qubic.ndets*self.sims.joint.qubic.nsamples, (2*self.sims.joint.qubic.ndets, self.sims.joint.qubic.nsamples))
            TODi_Q_150 = self.H_i.operands[0](self.sims.components_iter)[:self.ndets*self.nsampling]
            TODi_Q_220 = self.H_i.operands[0](self.sims.components_iter)[self.ndets*self.nsampling:2*self.ndets*self.nsampling]
            
            g150 = self._give_me_intercal(TODi_Q_150, self.sims.TOD_Q[:self.ndets*self.nsampling], self.sims.invN.operands[0].operands[1].operands[0])
            g220 = self._give_me_intercal(TODi_Q_220, self.sims.TOD_Q[self.ndets*self.nsampling:2*self.ndets*self.nsampling], self.sims.invN.operands[0].operands[1].operands[1])
            #g150 /= g150[0]
            #g220 /= g220[0]
            
            self.sims.g_iter = np.array([g150, g220]).T
            self.sims.Gi = join_data(self.sims.comm, self.sims.g_iter)
            self.sims.allg = np.concatenate((self.sims.allg, np.array([self.sims.g_iter])), axis=0)
            #print()
            #stop
            if self.sims.rank == 0:
                print(np.mean(self.sims.g_iter - self.sims.g, axis=0))
                print(np.std(self.sims.g_iter - self.sims.g, axis=0))
            
        #stop
        #### Display convergence of beta
        self.plots.plot_gain_iteration(self.sims.allg - self.sims.g, alpha=0.03, ki=self._steps)
    def _give_me_intercal(self, D, d, _invn):
        
        """
        
        Semi-analytical method for gains estimation.

        """
        
        _r = ReshapeOperator(self.sims.joint_out.qubic.ndets*self.sims.joint_out.qubic.nsamples, (self.sims.joint_out.qubic.ndets, self.sims.joint_out.qubic.nsamples))
        
        return (1/np.sum(_r(D) * _invn(_r(D)), axis=1)) * np.sum(_r(D) * _invn(_r(d)), axis=1)
