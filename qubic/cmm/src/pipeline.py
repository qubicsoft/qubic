import numpy as np
import yaml
import qubic
from qubic import QubicSkySim as qss
import pickle
import gc

import fgb.mixing_matrix as mm
import fgb.component_model as c

from acquisition.systematics import *

from simtools.mpi_tools import *
from simtools.noise_timeline import *
from simtools.foldertools import *
from simtools.analysis import *

import healpy as hp
import matplotlib.pyplot as plt
from functools import partial
from pyoperators import MPI
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
import os
import sys
from scipy.optimize import minimize, fmin, fmin_l_bfgs_b
from solver.cg import (mypcg)
from preset.preset import *
from plots.plots import *
from costfunc.chi2 import Chi2ConstantBlindJC, Chi2Parametric
import emcee
from schwimmbad import MPIPool
from multiprocessing import Pool
    
               
               
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
            seed_noise = np.random.randint(100000000)
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
                                 'g':self.sims.g,
                                 'gi':self.sims.g_iter,
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
            #self.plots.display_maps(self.sims.seenpix_plot, ngif=self._steps+1, ki=self._steps)
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
