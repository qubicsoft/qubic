from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import os
from simtools.foldertools import do_gif

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

            if self._steps > 0:
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
