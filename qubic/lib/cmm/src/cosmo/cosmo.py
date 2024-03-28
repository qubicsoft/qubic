######## This file perform a likelihood estimation on the tensor-to-scalar ratio. It take
######## as input a pickle file which have the ell, Dl and errors and return the same file 
######## with the likelihood on r.


import numpy as np
import pickle
import os
from pyoperators import MPI
import healpy as hp
import matplotlib.pyplot as plt
from multiprocess import Pool
from schwimmbad import MPIPool
import emcee
from getdist import plots, MCSamples
import sys

filename = '/pbs/home/m/mregnier/sps1/mapmaking/src/frequency_map_making/Dls_band150_011_ndetFalse_npho150True_npho220True_iteration20.pkl'

comm = MPI.COMM_WORLD
size = comm.Get_size()
class JCHlike(object):

    def __init__(self, model, nside):

        self.model=model
        self.cc=0
        self.dl=30
        self.lmin=40
        self.nside=nside
        self.lmax=2*self.nside-1




    def ana_likelihood(self, rv, leff, fakedata, errors, model, prior,mylikelihood=LogLikelihood, covariance_model_funct=None, otherp=None):
        ll = mylikelihood(xvals=leff, yvals=fakedata, errors=errors,model = model, flatprior=prior,
                                    covariance_model_funct=covariance_model_funct)

        like = np.zeros_like(rv)
        for i in range(len(rv)):
            like[i] = np.exp(ll([rv[i]]))
            maxL = rv[like == np.max(like)]
            cumint = scipy.integrate.cumtrapz(like, x=rv)
            cumint = cumint / np.max(cumint)
            onesigma = np.interp(0.68, cumint, rv[1:])
        if otherp:
            other = np.interp(otherp, cumint, rv[1:])
            return like, cumint, onesigma, other, maxL
        else:
            return like, cumint, onesigma, maxL
    def explore_like(self, leff, cl, errors, rv, Alens=0.1, otherp=None, cov=None, sample_variance=True):

        #     print(lmin, dl, cc)
        #     print(leff)
        #     print(scl_noise[:,2])
        ### Create Namaster Object
        # Unfortunately we need to recalculate fsky for calculating sample variance

        if cov is None:
            Namaster = nam.Namaster(None, lmin=self.lmin, lmax=self.lmax, delta_ell=self.dl)
            Namaster.fsky = 0.03
        else:
            okpix = cov > (np.max(cov) * float(cc))
            maskpix = np.zeros(12*self.nside**2)
            maskpix[okpix] = 1
            Namaster = nam.Namaster(maskpix, lmin=self.lmin, lmax=self.lmax, delta_ell=self.dl)
            Namaster.fsky = 0.03

        #     print('Fsky: {}'.format(Namaster.fsky))
        lbinned, b = Namaster.get_binning(self.nside)

        ### Bibnning CambLib
        #     binned_camblib = qc.bin_camblib(Namaster, '../../scripts/QubicGeneralPaper2020/camblib.pickle',
        #                                     nside, verbose=False)
        #global_dir=os.getcwd()#''/pbs/home/m/mregnier/sps1/QUBIC+/d0/cls'
        #binned_camblib = qc.bin_camblib(Namaster, global_dir+'/camblib.pkl', nside, verbose=False)


        ### Redefine the function for getting binned Cls
        #def myclth(ell,r):
        #    clth = qc.get_Dl_fromlib(ell, r, lib=binned_camblib, unlensed=True)[1]
        #    return clth
        #allfakedata = myclth(leff, 0.)
        #lll, totDL, unlensedCL = qc.get_camb_Dl(lmax=3*256, r=0)
        ### And we need a fast one for BB only as well
        def myBBth(ell, r):
            return self.model(ell, r, Alens=Alens)

        ### Fake data
        fakedata = cl.copy()#myBBth(leff, 0.)


        if sample_variance:
            covariance_model_funct = Namaster.knox_covariance
        else:
            covariance_model_funct = None

        if otherp is None:
            like, cumint, allrlim, maxL = self.ana_likelihood(rv, leff, fakedata, errors*2, myBBth, [[-1,1]],covariance_model_funct=covariance_model_funct)
        else:
            like, cumint, allrlim, other, maxL = self.ana_likelihood(rv, leff, fakedata, errors*2, myBBth, [[-1,1]],covariance_model_funct=covariance_model_funct, otherp=otherp)

        if otherp is None:
            return like, cumint, allrlim, maxL
        else:
            return like, cumint, allrlim, other, maxL

class Forecast:
    
    def __init__(self, ell, Dl, Nl, params=[None, None], mu=[0.01, 1], sig=[0.001, 0.1]):
        
        self.job_id = os.environ.get('SLURM_JOB_ID')
        self.ell = ell
        self._f = self.ell * (self.ell + 1) / (2 * np.pi)
        self.Dl_noisy = Dl
        self.Nl = Nl
        self.params = params
        
        self.nparams = 0
        for i in self.params:
            if i == None:
                self.nparams += 1

        self.Dl = np.mean(self.Dl_noisy - self.Nl, axis=0)
        self.cov = np.cov(self.Dl_noisy - self.Nl, rowvar=False)
        self.invcov = np.linalg.inv(self.cov)
        
        self.mu = mu
        self.sig = sig
    
    def _init_mcmc(self, nwalkers):
        
        x0 = np.zeros((nwalkers, self.nparams))

        for i in range(self.nparams):
            x0[:, i] = np.random.normal(self.mu[i], self.sig[i], (nwalkers))
        return x0
    def give_dl_cmb(self, r=0, Alens=1.):
        
        power_spectrum = hp.read_cl('/home/regnier/work/regnier/CMM-Pipeline/src/data/Cls_Planck2018_lensed_scalar.fits')[:,:4000]
        if Alens != 1.:
            power_spectrum[2] *= Alens
        if r:
            power_spectrum += r * hp.read_cl('/home/regnier/work/regnier/CMM-Pipeline/src/data/Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits')[:,:4000]
        return self._f * np.interp(self.ell, np.arange(1, 4001, 1), power_spectrum[2]) 
    def log_prior(self, x):
        
        if self.params[0] != None: 
            r = self.params[0]
            Alens = x
        if self.params[1] != None: 
            r = x
            Alens = self.params[0]
        if self.params[0] == None and self.params[0] == None: 
            r, Alens = x
            
        if r < -1 or r > 1:
            return -np.inf
        elif Alens < 0 or Alens > 2:
            return -np.inf
        
        return 0     
    def likelihood(self, x):
        
        if self.params[0] != None: 
            r = self.params[0]
            Alens = x
        if self.params[1] != None: 
            r = x
            Alens = self.params[0]
        if self.params[0] == None and self.params[0] == None: 
            r, Alens = x
            
        ysim = self.give_dl_cmb(r=r, Alens=Alens)
        _r = self.Dl - ysim
        
        return self.log_prior(x) - 0.5 * (_r.T @ self.invcov @ _r)
    def _plot_chains(self, chains):
        
        plt.figure(figsize=(8, 6))  
        nsamp, nwalk, ndim = chains.shape
        
        for dim in range(ndim):
            plt.subplot(ndim, 1, dim+1)
            for i in range(nwalk):
                plt.plot(chains[:, i, dim], '-k', alpha=0.1)
        
            plt.plot(np.mean(chains, axis=1)[:, dim], '-k')
            
        plt.savefig(f'chains_{self.job_id}.png')
        plt.close()
    def _get_triangle(self, chainflat, label):
        
        labels = ['r', 'A_{lens}']
        names = ['r', 'Alens']

        s = MCSamples(samples=chainflat, names=names, labels=labels, label=label, ranges={'r':(0, None)})

        plt.figure(figsize=(12, 8))

        # Triangle plot
        g = plots.get_subplot_plotter(width_inch=10)
        g.triangle_plot([s], filled=True, title_limit=1)

        plt.savefig(f'triangle_{self.job_id}.png')
        plt.close()
        
    def run(self, nwalkers, nsteps, dis=0):
        
        x0 = self._init_mcmc(nwalkers)
        
        if size != 1:
            with MPIPool() as pool:
                sampler = emcee.EnsembleSampler(nwalkers, x0.shape[1], self.likelihood, pool=pool)
                sampler.run_mcmc(x0, nsteps, progress=True)
        else:    
            with Pool() as pool:
                sampler = emcee.EnsembleSampler(nwalkers, x0.shape[1], self.likelihood, pool=pool)
                sampler.run_mcmc(x0, nsteps, progress=True)
        
        chainflat = sampler.get_chain(discard=dis, thin=15, flat=True)
        chains = sampler.get_chain()
        
        self._get_triangle(chainflat, label='test')
        self._plot_chains(chains)
        
        return chains, chainflat
            
def open_data(filename):      
    path = '/home/regnier/work/regnier/CMM-Pipeline/src/'
    
    with open(path + '/' + filename, 'rb') as f:
        data = pickle.load(f)
    return data

filename = 'autospectrum_parametric_d6_forecastpaper_dualband_qubic2_lcorr20.pkl'
filename_err = 'autospectrum_parametric_d0_forecastpaper_dualband_qubic2.pkl'

data = open_data(filename)
data_err = open_data(filename_err)


### Forecast
forecast = Forecast(data['ell'], data['Dl'][:5, 0, :], data_err['Dl_1x1'][:5, :], params=[None, None], mu=[0.01, 1], sig=[0.001, 0.1])
    
plt.figure()

plt.errorbar(data['ell'], np.mean(data['Dl'][:5, 0], axis=0), yerr=np.std(data['Dl'][:5, 0], axis=0), fmt='or', capsize=3)
plt.errorbar(data['ell'], np.mean(data_err['Dl_1x1'][:5], axis=0), yerr=np.std(data_err['Dl_1x1'][:5], axis=0), fmt='og', capsize=3)

plt.plot(data['ell'], forecast.give_dl_cmb(r=0, Alens=0.1))

plt.yscale('log')
plt.savefig('Dl.png')
plt.close()
stop
nwalkers = int(sys.argv[1])
nsteps = int(sys.argv[2])

chains, chainflat = forecast.run(nwalkers, nsteps, dis=200)

with open("chains" + filename[12:], 'wb') as handle:
        pickle.dump({'ell':data['ell'], 
                     'Dl':data['Dl'][:, 0, :],
                     'Nl':data['Dl_1x1'],
                     'chains':chains, 
                     'chainflat':chainflat
                     }, handle, protocol=pickle.HIGHEST_PROTOCOL)
        