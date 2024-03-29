import numpy as np
import os.path as op
import healpy as hp
import os
import sys
import pickle
import yaml
import matplotlib.pyplot as plt
from pyoperators import *
import emcee
from multiprocess import Pool
from schwimmbad import MPIPool
from getdist import plots, MCSamples

sys.path.append(os.path.dirname(os.getcwd()))

import fgb.component_model as c
import fgb.mixing_matrix as mm

COMM = MPI.COMM_WORLD

class CMB:
    '''
    Class to define the CMB model
    '''

    def __init__(self, ell):
        
        self.ell = ell
    def cl_to_dl(self, cl):
        '''
        Function to convert the cls into the dls
        '''
        _f = self.ell * (self.ell + 1) / (2 * np.pi)
        return _f * cl
    def get_pw_from_planck(self, r, Alens):
        '''
        Function to compute the CMB power spectrum from the Planck data
        '''
        path = os.path.dirname(os.getcwd()) + '/data/'
        
        power_spectrum = hp.read_cl(path + 'Cls_Planck2018_lensed_scalar.fits')[:,:4000]
        
        if Alens != 1.:
            power_spectrum[2] *= Alens
        
        if r:
            power_spectrum += r * hp.read_cl(path + 'Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits')[:,:4000]
        
        return np.interp(self.ell, np.linspace(1, 4001, 4000), power_spectrum[2])
    def __call__(self, r, Alens):
        
        '''
        Define the CMB model, depending on r and Alens
        '''

        dlBB = self.cl_to_dl(self.get_pw_from_planck(r, Alens))
        return dlBB   
class Foreground:
    '''
    Function to define the Dust model
    '''

    def __init__(self, ell, nus):
        
        self.ell = ell
        self.nus = nus
        self.nrec = len(self.nus)
        self.nspec = int(self.nrec * (self.nrec + 1) / 2)
    def scale_dust(self, nu, nu0_d, betad, temp=20):
        '''
        Function to compute the dust mixing matrix element, depending on the frequency
        '''

        comp = c.Dust(nu0 = nu0_d, temp=temp, beta_d = betad)
        A = mm.MixingMatrix(comp).evaluator(np.array([nu]))()[0]

        return A
    def scale_sync(self, nu, nu0_s, betas):
        '''
        Function to compute the dust mixing matrix element, depending on the frequency
        '''

        comp = c.Synchrotron(nu0 = nu0_s, beta_pl = betas)
        A = mm.MixingMatrix(comp).evaluator(np.array([nu]))()[0]

        return A
    def model_dust_frequency(self, A, alpha, delta, fnu1, fnu2):
        '''
        Function to define the Dust model for two frequencies
        '''

        return abs(A) * delta * fnu1 * fnu2 * (self.ell/80)**alpha
    def model_sync_frequency(self, A, alpha, fnu1, fnu2):
        '''
        Function to define the Dust model for two frequencies
        '''

        return abs(A) * fnu1 * fnu2 * (self.ell/80)**alpha
    def model_dustsync_corr(self, Ad, As, alphad, alphas, fnu1d, fnu2d, fnu1s, fnu2s, eps):
        return eps * np.sqrt(abs(As) * abs(Ad)) * (fnu1d * fnu2s + fnu2d * fnu1s) * (self.ell / 80)**((alphad + alphas)/2)
    def __call__(self, Ad, alphad, betad, deltad, nu0_d, As=None, alphas=None, betas=None, nu0_s=None, eps=None):
        '''
        Function defining the Dust model for all frequencies, depending on Ad, alphad, betad, deltad & nu0_d
        '''

        s = np.zeros((self.nrec, self.nrec))
        
        models = np.zeros((self.nspec, len(self.ell)))
        k=0
        for i in range(self.nrec):
            for j in range(self.nrec):
                if i == j:
                    #print(self.nus[i])
                    fnud = self.scale_dust(self.nus[i], nu0_d, betad)
                    models[k] = self.model_dust_frequency(Ad, alphad, deltad, fnud, fnud)
                    if As is not None or alphas is not None or betas is not None:
                        fnus = self.scale_sync(self.nus[i], nu0_s, betas)
                        models[k] += self.model_sync_frequency(As, alphas, fnus, fnus)
                    
                    if eps is not None:
                            models[k] += self.model_dustsync_corr(Ad, As, alphad, alphas, fnud, fnud, fnus, fnus, eps)
                    k+=1
                else:
                    if s[i, j] == 0:
                        #print(f'Computing X-spectra at {self.nus[i]:.0f} and {self.nus[j]:.0f} GHz')
                        fnu1d = self.scale_dust(self.nus[i], nu0_d, betad)
                        fnu2d = self.scale_dust(self.nus[j], nu0_d, betad)
                        models[k] += self.model_dust_frequency(Ad, alphad, deltad, fnu1d, fnu2d)
                        if As is not None or alphas is not None or betas is not None:
                            fnu1s = self.scale_sync(self.nus[i], nu0_s, betas)
                            fnu2s = self.scale_sync(self.nus[j], nu0_s, betas)
                            models[k] += self.model_sync_frequency(As, alphas, fnu1s, fnu2s)
                        
                        if eps is not None:
                            #print('dustsync')
                            models[k] += self.model_dustsync_corr(Ad, As, alphad, alphas, fnu1d, fnu2d, fnu1s, fnu2s, eps)
                        s[i, j] = 1
                        s[j, i] = 1
                        #stop
                        k+=1
                    else:
                        #print(f'Not {self.nus[i]:.0f} and {self.nus[j]:.0f}')
                        s[i, j] = 1
                        s[j, i] = 1
        
        return models
        
class BBPip:
    
    def __init__(self, path):
    
        with open('sampling_config.yml', "r") as stream:
            self.params = yaml.safe_load(stream)
        
        ### Load general arguments
        self.rank = COMM.Get_rank()
        self.size = COMM.Get_size()
        self.path = path
        #self.path_noise = path_noise
        self.files = os.listdir(self.path)
        #self.files_noise = os.listdir(self.path_noise)
        self.N = len(self.files)
        #self.Nn = len(self.files_noise)
        self.nus = self.open_data(self.path + '/' + self.files[0])['nus']

        self.bandpower = []
        s = np.zeros((len(self.nus), len(self.nus)))
        self.auto_spec = []
        ### Create list of all bands-power
        k=0
        for i in range(len(self.nus)):
            for j in range(i, len(self.nus)):
                if self.nus[i] == self.nus[j]:
                    self.auto_spec += [True]
                else:
                    self.auto_spec += [False]
                self.bandpower += [f'{self.nus[i]:.2f}x{self.nus[j]:.2f}']
                s[i, j] = k
                k+=1

        k=0
        bp_to_rm = []
        for ii, i in enumerate(self.nus):
            if ii < self.params['NUS']['qubic'][1]:
                if self.params['NUS']['qubic'][0]:
                    k += (self.params['NUS']['qubic'][1])
                else:
                    bp_to_rm += [ii]
                    k+=1
                
            else:
                if self.params['NUS'][f'{i:.0f}GHz'] is False:
                    bp_to_rm += [ii]
        #print(bp_to_rm)
        self.nus = np.delete(self.nus, bp_to_rm, 0)
        #print(s)
        s = np.delete(s, bp_to_rm, 0)
        s = np.delete(s, bp_to_rm, 1)
        #print(s)
        bp_to_keep = []
        for i in range(s.shape[0]):
            for j in range(i, s.shape[1]):
                bp_to_keep += [int(s[i, j])]

        self.ell = self.open_data(self.path + '/' + self.files[0])['ell']
        sh = self.open_data(self.path + '/' + self.files[0])['Dls'].shape

        self.Dl = np.zeros((self.N, len(bp_to_keep), sh[1]))
        self.Nl = np.zeros((self.N, len(bp_to_keep), sh[1]))
        
        for i in range(self.N):
            self.Dl[i] = self.open_data(self.path + '/' + self.files[i])['Dls'][bp_to_keep]
            self.Nl[i] = self.open_data(self.path + '/' + self.files[i])['Nl'][bp_to_keep]
            
        
        nbin = -1
        self.ell = self.ell[:nbin].copy()
        self._f = self.ell * (self.ell + 1) / (2 * np.pi)
        self.Dl = self.Dl[:, :, :nbin].copy()
        self.Nl = self.Nl[:, :, :nbin].copy()
        self.DlBB = np.mean(self.Dl, axis=0)
        self.DlBB_err = np.std(self.Dl, axis=0)
        self.NlBB = np.mean(self.Nl, axis=0)
        self.NlBB_err = np.std(self.Nl, axis=0)
        
        self.cmb_model = CMB(self.ell)
        self.fg_model = Foreground(self.ell, self.nus)
        
        #self.DlBB = self.cmb_model(0, 1) + self.fg_model(10, -0.1, 1.54, 1, 353, As=0.5, alphas=0, betas=-3, nu0_s=23)
        self.DlBB -= self.NlBB
        
        model = self.cmb_model(0, 1) + self.fg_model(0, -0.1, 1.54, 1, 353)

        self._make_plots_Dl(self.DlBB, self.NlBB_err, model[:, :])

        ### Define possible components
        self.is_cmb = False
        self.is_dust = False
        self.is_sync = False
        
        self.names = list(self.params['SKY_PARAMETERS'].keys())
        self.free = []

        ### Check free parameters for MCMC
        for name in self.names:
            self.free += [self._check_free_param(self.params['SKY_PARAMETERS'][name])]
        
        ### Check presence of different components
        if self.free[0] is not False or self.free[1] is not False:
            self.is_cmb = True
        
        if self.free[3] is not False or self.free[4] is not False or self.free[5] is not False or self.free[6] is not False:
            self.is_dust = True
        
        if self.free[7] is not False or self.free[8] is not False or self.free[9] is not False:
            self.is_sync = True
    
    
        ### Initiate starting guess for walkers
        np.random.seed(1)
        self.p0 = np.zeros((1, self.params['MCMC']['nwalkers']))
        self.index_notfree_param = []
        for i in range(len(self.names)):
            if self.params['SKY_PARAMETERS'][self.names[i]][0] is True:
                p0 = np.random.normal(self.params['SKY_PARAMETERS'][self.names[i]][3], 
                                      self.params['SKY_PARAMETERS'][self.names[i]][4], 
                                      (1, self.params['MCMC']['nwalkers']))
                self.p0 = np.concatenate((self.p0, p0), axis=0)
            else:
                self.index_notfree_param += [i]
        
        ### Remove wrong parameters in arrays
        self.p0 = np.delete(self.p0, 0, axis=0).T
        self.ndim = self.p0.shape[1]
        self.names = np.delete(self.names, self.index_notfree_param)
        self._f = self.ell * (self.ell + 1) / (2 * np.pi)
    def _make_plots_Dl(self, Dl, Dl_err, model, model2=None, model3=None):
        
        plt.figure(figsize=(12, 12))
        s = np.zeros((len(self.nus), len(self.nus)))
        k=0
        kp=0
        for i in range(len(self.nus)):
            for j in range(len(self.nus)):
                plt.subplot(len(self.nus), len(self.nus), kp+1)
                if i == j:
                    plt.errorbar(self.ell[:], Dl[k, :], yerr=Dl_err[k, :], fmt='or', capsize=5)
                    plt.errorbar(self.ell[:], model[k, :], fmt='-k', capsize=5)
                    if model2 is not None:
                        plt.errorbar(self.ell[:], model2[k, :], fmt='--r', capsize=5)
                    if model3 is not None:
                        plt.errorbar(self.ell[:], model3[k, :], fmt='-m', capsize=5)
                    k+=1
                else:
                    if s[i, j] == 0:
                        plt.errorbar(self.ell[:], Dl[k, :], yerr=Dl_err[k, :], fmt='ob', capsize=5)
                        plt.errorbar(self.ell[:], model[k, :], fmt='-k', capsize=5)
                        if model2 is not None:
                            plt.errorbar(self.ell[:], model2[k, :], fmt='--r', capsize=5)
                        if model3 is not None:
                            plt.errorbar(self.ell[:], model3[k, :], fmt='-m', capsize=5)
                        s[i, j] = 1
                        s[j, i] = 1
                        k+=1
                    else:
                        
                        s[i, j] = 1
                        s[j, i] = 1
                kp+=1
        plt.savefig('Dls.png')
        plt.close()
    def _plot_chains(self, chains):
        
        plt.figure(figsize=(8, 5))
        for i in range(self.ndim):
            plt.subplot(self.ndim, 1, i+1)
            plt.plot(chains[:, :, i], '-k', alpha=0.1)
            
        plt.savefig('chains.png')
        plt.close()
    def _check_free_param(self, line):

        if type(line[0]) is float or type(line[0]) is int:
            return line[0]
        elif line[0] is False:
            return False
        else:
            return True
    def open_data(self, filename):         
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data
    def _fill_params(self, x):
        
        for ii, i in enumerate(self.free):

            if i is True:
                pass
            elif type(i) is float or type(i) is int:
                x = np.insert(x, [ii], i)
            else:
                pass
        return x
    def knox_errors(self, clth):
        return self._f * np.sqrt(2. / ((2 * self.ell + 1) * 0.01 * 30)) * clth
    def knox_covariance(self, clth):
        dcl = self.knox_errors(clth)
        return np.diag(dcl ** 2)
    def log_prob(self, x):
        
        for iparam, param in enumerate(self.names):
            
            if x[iparam] < self.params['SKY_PARAMETERS'][self.names[iparam]][1] or x[iparam] > self.params['SKY_PARAMETERS'][self.names[iparam]][2]:
                return -np.inf
        return 0
    def loglike(self, x):
        
        logprob = self.log_prob(x)
        x = self._fill_params(x)
        
        self.sample_cmb = np.zeros((len(self.ell), len(self.ell)))
        self.sample_dust = np.zeros((len(self.ell), len(self.ell)))
        if self.is_cmb and self.is_dust == False:
            r, Alens = x
            ymodel = self.cmb_model(r, Alens)
        elif self.is_cmb == False and self.is_dust:
            nu0d, A, alpha, beta, Delta = x
            ymodel = self.fg_model(A, alpha, beta, Delta, nu0d)
        elif self.is_cmb and self.is_dust and self.is_sync == False:
            r, Alens, nu0d, A, alpha, beta, Delta = x
            cmbtheo = self.cmb_model(r, Alens)
            fgtheo = self.fg_model(A, alpha, beta, Delta, nu0d)
            ymodel = cmbtheo + fgtheo
        elif self.is_cmb and self.is_dust and self.is_sync:
            r, Alens, nu0d, A, alpha, beta, Delta, nu0s, As, alphas, betas, eps = x
            cmbtheo = self.cmb_model(r, Alens)
            fgtheo = self.fg_model(A, alpha, beta, Delta, nu0d, As, alphas, betas, nu0s, eps)
            ymodel = cmbtheo + fgtheo
        
        
        _r = (self.DlBB - ymodel)
        
        L = logprob
        
        for i in range(self.DlBB.shape[0]):
            #d = self.knox_covariance(cmbtheo/self._f + fgtheo[i]/self._f)
            d = self.knox_covariance(fgtheo[i]/self._f)
            
            cov = np.cov(self.Nl[:, i], rowvar=False) + d
            #cov *= np.eye(len(self.ell))
            invcov = np.linalg.pinv(cov)
            
            L -= 0.5 * (_r[i].T @ invcov @ _r[i])

        return L
    def _merge_data(self, d, d_flat):
        
        if self.rank == 0:
            d_merged = d[0].copy()
            d_flat_merged = d_flat[0].copy()
        
            for i in range(1, self.size):
                d_merged = np.concatenate((d_merged, d[i]), axis=1)
                d_flat_merged = np.concatenate((d_flat_merged, d_flat[i]), axis=0)
        else:
            d_merged = None
            d_flat_merged = None
            
        d_merged = COMM.bcast(d_merged, root=0)
        d_flat_merged = COMM.bcast(d_flat_merged, root=0)
        
        return d_merged, d_flat_merged
    def run(self):
        
        #print(self.p0)
        with MPIPool() as pool:
            sampler = emcee.EnsembleSampler(self.params['MCMC']['nwalkers'], self.ndim, self.loglike, pool=pool)
            sampler.run_mcmc(self.p0, self.params['MCMC']['mcmc_steps'], progress=True)

        #print(self.samp_dust)
        
        #COMM.Barrier()
        chains = sampler.get_chain()
        chains_flat = sampler.get_chain(discard=self.params['MCMC']['discard'], flat=True, thin=15)
        
        #chains_global = COMM.allgather(chains)
        #flatchains_global = COMM.allgather(chains_flat)
        #print(flatchains_global[0].shape)
        #print(self.rank, data_global[0])
        #chains_all, chains_flat_all = self._merge_data(chains_global, flatchains_global)
        #print(self.rank, chains_all.shape, chains_flat_all.shape)
        
        p = np.mean(chains_flat, axis=0)
        p = self._fill_params(p)

        if self.is_cmb and self.is_dust == False:
            r, Alens = p
        elif self.is_cmb == False and self.is_dust:
            A, alpha, beta, Delta = p
        elif self.is_cmb and self.is_dust and self.is_sync == False:
            r, Alens, _, A, alpha, beta, Delta = p
            model1 = self.cmb_model(r, Alens) + self.fg_model(A, alpha, beta, Delta, 353)
            model2 = self.cmb_model(p[0] + np.std(chains_flat, axis=0)[0], Alens) + self.fg_model(A, alpha, beta, Delta, 353)
            model3 = self.cmb_model(0, 1) + self.fg_model(A, alpha, beta, Delta, 353)
        elif self.is_cmb and self.is_dust and self.is_sync:
            r, Alens, _, A, alpha, beta, Delta, _, As, alphas, betas, eps = p
            model1 = self.cmb_model(r, Alens) + self.fg_model(A, alpha, beta, Delta, 353, As, alphas, betas, 23)
            model2 = self.cmb_model(p[0] + np.std(chains_flat, axis=0)[0], Alens) + self.fg_model(A, alpha, beta, Delta, 353, As, alphas, betas, 23, eps)
            model3 = self.cmb_model(0, 1) + self.fg_model(A, alpha, beta, Delta, 353, As, alphas, betas, 23, eps)
            
        self._plot_chains(chains)
        
        self._make_plots_Dl(self.DlBB, self.NlBB_err, model1, model2=model2, model3=model3)
        
        return chains, chains_flat


pip = BBPip(path=os.path.dirname(os.getcwd()) + '/E2E_nrec2/Xspectrum_nrec2_cmbdust/spectrum/')#,
            #path_noise=os.path.dirname(os.getcwd()) + '/E2E_nrec2/cmbdust_noise/spectrum/')

chains, chains_flat = pip.run()




if pip.rank == 0:
    print('Average : ', np.mean(chains_flat, axis=0))
    print('Error   : ', np.std(chains_flat, axis=0))
    filename = pip.params['filename']
    with open(f'{filename}.pkl', 'wb') as handle:
        pickle.dump({'chains':chains, 'chains_flat':chains_flat, 'ell':pip.ell, 
                     'Dls':pip.Dl, 'Nl':pip.Nl, 
                     'nus':pip.nus, 'names':pip.names, 'labels':pip.names}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    s = MCSamples(samples=chains_flat, names=pip.names, labels=pip.names)
    plt.figure()

    # Triangle plot
    g = plots.get_subplot_plotter(width_inch=8)
    #g.settings.alpha_filled_add=0.8
    g.triangle_plot(s, filled=True, markers={'r':0}, title_limit=1)

    plt.savefig('triangle_dist.png')

    plt.close()
