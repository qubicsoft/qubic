######## This file perform a likelihood estimation on the tensor-to-scalar ratio. It take
######## as input a pickle file which have the ell, Dl and errors and return the same file 
######## with the likelihood on r.


import numpy as np
import scipy
from pyoperators import *
import pickle
import os
import os.path as op
import healpy as hp
import qubic
from qubic import NamasterLib as nam
import matplotlib.pyplot as plt
import time
import sys

filename = str(sys.argv[1])

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

CMB_CL_FILE = op.join(os.getcwd+'/Cls_Planck2018_%s.fits')

def open_pkl(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data

def _get_Cl_cmb(r, Alens=1):

    power_spectrum = hp.read_cl(CMB_CL_FILE%'lensed_scalar')[2,:4000]
    if Alens != 1.:
        power_spectrum[2] *= Alens
    if r:
        power_spectrum += r * hp.read_cl(CMB_CL_FILE%'unlensed_scalar_and_tensor_r1')[2,:4000]
    return power_spectrum
def cl2dl(ell, cl):

    dl=np.zeros(ell.shape[0])
    for i in range(ell.shape[0]):
        dl[i]=(ell[i]*(ell[i]+1)*cl[i])/(2*np.pi)
    return dl
def mymodel(ell, r):
    #print(r)
    cl = _get_Cl_cmb(r)
    cl_binned = cl[ell.astype(int)-1]
    #print(ell, cl_binned)
    return cl2dl(ell, cl_binned)

class LogLikelihood:
    def __init__(self, xvals=None, yvals=None, errors=None, model=None, nbins=None,
                 nsiginit=10, nsigprior=20, flatprior=None, fixedpars=None,
                 covariance_model_funct=None, p0=None, nwalkers=32, chi2=None):
        self.prior = None
        self.model = model
        self.xvals = xvals
        self.yvals = yvals
        if nbins is None:
            self.nbins = len(xvals)
        else:
            self.nbins = nbins
        self.nsiginit = nsiginit
        self.nsigprior = nsigprior
        self.covariance_model_funct = covariance_model_funct
        self.nwalkers = nwalkers
        self.fixedpars = fixedpars
        self.p0 = p0
        self.chi2 = chi2

        if np.ndim(errors) == 1:
            self.covar = np.zeros((np.size(errors), np.size(errors)))
            np.fill_diagonal(self.covar, np.array(errors) ** 2)
        else:
            self.covar = errors

        self.flatprior = flatprior
        if flatprior is None:
            initial_fit = self.minuit(p0=self.p0, chi2=self.chi2)
            self.fitresult = [initial_fit[0], initial_fit[1]]

    def __call__(self, mytheta, extra_args=None, verbose=False):
        if self.fixedpars is not None:
            theta = self.p0.copy()
            theta[self.fixedpars == 0] = mytheta
            #theta[self.fixedpars == 0] = mytheta[self.fixedpars == 0]
        else:
            theta = mytheta
        # theta = mytheta
        self.modelval = self.model(self.xvals[:self.nbins], theta)

        if self.covariance_model_funct is None:
            self.invcov = np.linalg.inv(self.covar)
        else:
            cov_repeat = self.make_covariance_matrix()
            self.invcov = np.linalg.inv(cov_repeat + self.covar)


        lp = self.log_priors(theta)
        if verbose:
            print('Pars')
            print(theta)
            print('Y')
            print(np.shape(self.yvals))
            print(self.yvals[0:10])
            print('Model')
            print(np.shape(self.modelval))
            print(self.modelval[:10])
            print('Diff')
            print(np.shape((self.yvals - self.modelval)))
            print((self.yvals - self.modelval)[0:10])
            print('Diff x invcov')
            print(np.shape((self.yvals - self.modelval).T @ self.invcov))
            print(((self.yvals - self.modelval).T @ self.invcov)[0:10])
        logLLH = lp - 0.5 * (((self.yvals - self.modelval).T @ self.invcov) @ (self.yvals - self.modelval))
        
        if not np.isfinite(logLLH):
            return -np.inf
        else:
            return logLLH

    def make_covariance_matrix(self):
        cov = self.covariance_model_funct(self.modelval[:self.nbins])
        cov_repeat = np.zeros_like(self.covar)
        for i in range(0, len(self.xvals), self.nbins):
            cov_repeat[i:i + self.nbins, i:i + self.nbins] = cov
        return cov_repeat

    def compute_sigma68(self, logLLH, rvalues):
        LLH = [np.exp(logLLH([rvalues[i]])) for i in range(len(rvalues))]

        cumint = cumtrapz(LLH, x=rvalues)  # Cumulative integral
        cumint /= np.max(cumint)
        sigma68 = np.interp(0.68, cumint, rvalues[1:])

        return LLH, sigma68

    def log_priors(self, theta):
        ok = 1
        for i in range(len(theta)):
            if self.flatprior is not None:
                if (theta[i] < self.flatprior[i][0]) or (theta[i] > self.flatprior[i][1]):
                    ok *= 0
            else:
                if np.abs(theta[i] - self.fitresult[0][i]) > (self.nsigprior * np.sqrt(self.fitresult[1][i, i])):
                    ok *= 0
        if ok == 1:
            return 0
        else:
            return -np.inf

    def run(self, nbmc):
        nwalkers = self.nwalkers
        if self.flatprior is not None:
            ndim = len(self.flatprior)
            pos = np.zeros((nwalkers, ndim))
            for d in range(ndim):
                pos[:, d] = np.random.rand(nwalkers) * (self.flatprior[d][1] - self.flatprior[d][0]) + \
                            self.flatprior[d][0]
        else:
            nsigmas = self.nsiginit
            ndim = len(self.fitresult[0])
            pos = np.zeros((nwalkers, ndim))
            for d in range(ndim):
                pos[:, d] = np.random.randn(nwalkers) * np.sqrt(self.fitresult[1][d, d]) * nsigmas + self.fitresult[0][
                    d]
        print('Ndim init:', ndim)
        if self.fixedpars is not None:
            ndim = int(np.sum(self.fixedpars == 0))
        print('New ndim:', ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.__call__)
        if self.fixedpars is not None:
            print('Len(pos):', np.shape(pos))
            print('len(fixedpars):', len(self.fixedpars))
            pos = pos[:, self.fixedpars == 0]
            print('New len(pos):', np.shape(pos))
        sampler.run_mcmc(pos, nbmc, progress=True)
        return sampler

    def fisher_analysis(self, delta_r=1e-7):
        # Model
        modelval_r0 = self.model(self.xvals[:self.nbins], r=0.)
        modelval_deltar = self.model(self.xvals[:self.nbins], r=delta_r)

        # Jacobian, Numerical derivative
        J = (modelval_deltar - modelval_r0) / delta_r

        # Covariance matrix in new basis
        Cov_r = 1 / (J.T @ self.invcov @ J)

        # Sigma at 68 pourcent
        sigma68 = np.sqrt(Cov_r)

        return sigma68

    def call4curvefit(self, x, *pars):
        return self.model(x, pars)

    def curve_fit(self, p0=None):
        if p0 is None:
            p0 = self.p0
        self.fitresult_curvefit = curve_fit(self.call4curvefit, self.xvals, self.yvals,
                                            sigma=np.sqrt(np.diag(self.covar)),
                                            maxfev=1000000, ftol=1e-5, p0=p0)
        return self.fitresult_curvefit[0], self.fitresult_curvefit[1]

    ### This should be modified in order to call the current likelihood instead, not an external one...
    def minuit(self, p0=None, chi2=None, verbose=True, print_level=0, ncallmax=10000, extra_args=None, nsplit=1,
               return_chi2fct=False):
        if p0 is None:
            p0 = self.p0
        if verbose & (print_level > 1):
            print('About to call Minuit with chi2:')
            print(chi2)
            print('Initial parameters, fixed and bounds:')
            for i in range(len(p0)):
                print('Param {0:}: init={1:6.2f} Fixed={2:} Range=[{3:6.3f}, {4:6.3f}]'.format(i, p0[i],
                                                                                               self.fixedpars[i],
                                                                                               self.flatprior[i][0],
                                                                                               self.flatprior[i][1]))
        self.fitresult_minuit = ft.do_minuit(self.xvals, self.yvals, self.covar, p0,
                                             functname=self.model,
                                             fixpars=self.fixedpars, rangepars=self.flatprior,
                                             verbose=verbose, chi2=self.chi2, print_level=print_level,
                                             ncallmax=ncallmax, extra_args=extra_args, nsplit=nsplit)
        if len(self.fitresult_minuit[3]) == 0:
            cov = np.diag(self.fitresult_minuit[2])
        else:
            cov = self.fitresult_minuit[3]
        if return_chi2fct:
            return self.fitresult_minuit[1], cov, self.fitresult_minuit[6]
        else:
            return self.fitresult_minuit[1], cov

    def random_explore_guess(self, ntry=100, fraction=1):
        fit_range_simu = self.flatprior
        fit_fixed_simu = self.fixedpars
        myguess_params = np.zeros((ntry, len(fit_range_simu)))
        for i in range(len(fit_range_simu)):
            if fit_fixed_simu[i] == 0:
                rng = (fit_range_simu[i][1] - fit_range_simu[i][0]) * fraction
                mini = np.max([fit_range_simu[i][0], self.p0[i] - rng / 2])
                maxi = np.min([fit_range_simu[i][0], self.p0[i] + rng / 2])
                myguess_params[:, i] = np.random.rand(ntry) * (maxi - mini) + mini
            else:
                myguess_params[:, i] = self.p0[i]
        return myguess_params
class JCHlike(object):

    def __init__(self, model, nside, dl, lmin, lmax):

        self.model=model
        self.cc=0
        self.dl=dl
        self.lmin=lmin
        self.nside=nside
        self.lmax=lmax

    def ana_likelihood(self, rv, leff, fakedata, errors, model, prior, mylikelihood=LogLikelihood, covariance_model_funct=None, otherp=None, nbins=None):
        ll = mylikelihood(xvals=leff, yvals=fakedata, errors=errors, model = model, flatprior=prior,
                                    covariance_model_funct=covariance_model_funct, nbins=nbins)

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
    def explore_like(self, leff, cl, errors, rv, cov=None, sample_variance=False):

        if cov is None:
            Namaster = nam.Namaster(None, lmin=self.lmin, lmax=self.lmax, delta_ell=self.dl)
            Namaster.fsky = 0.03
        else:
            okpix = cov > (np.max(cov) * float(cc))
            maskpix = np.zeros(12*self.nside**2)
            maskpix[okpix] = 1
            Namaster = nam.Namaster(maskpix, lmin=self.lmin, lmax=self.lmax, delta_ell=self.dl)
            Namaster.fsky = 0.03

        lbinned, b = Namaster.get_binning(self.nside)
        lbinned = lbinned[:-1]

        def myBBth(ell, r):
            return self.model(ell, r)

        ### Fake data
        fakedata = cl.copy()


        if sample_variance:
            Namaster.ell_binned = Namaster.ell_binned[:-1]
            covariance_model_funct = Namaster.knox_covariance
        else:
            covariance_model_funct = None

        like, cumint, allrlim, maxL = self.ana_likelihood(rv, leff, fakedata, errors, myBBth, [[-1,1]],
                                covariance_model_funct=covariance_model_funct, nbins=len(leff))
        
        return like, cumint, allrlim, maxL

def cosmo_like(datafile, rv):
    path = os.getcwd() + '/'
    data = open_pkl(path+datafile)
    print(f'Doing estimation from r = {rv.min()} to r = {rv.max()}')
    like_split, _, _, _ = like.explore_like(data['ell'], data['Dl'], data['error'], rv, cov=None, sample_variance=False)

    return like_split, data


if rank == 0:
    t0 = time.time()

### Namaster parameters
nside=256
dl=35
lmin=40
lmax=2*nside

### Initialization of the likelihood estimator
like = JCHlike(mymodel, nside, dl, lmin, lmax)

### All r for estimation
rv = np.linspace(0, 1, 10)
### Split r in N arrays
rv_split = np.array_split(rv, size)
like_split, data = cosmo_like(filename, rv_split[rank])

like_on_r = comm.gather(like_split,root=0)

if rank == 0:

    allike_on_r = []
    for i in range(size):
        allike_on_r += list(like_on_r[i])
    
    
    plt.figure(figsize=(8, 8))
    #print(rv.shape)
    #print(np.array(allike_on_r))
    plt.plot(rv, np.array(allike_on_r)/np.array(allike_on_r).max())
    plt.xlim(0, 0.1)

    plt.savefig('like.png')
    plt.close()

if rank == 0:
    end = time.time()
    print(f'Simulation done in {end - t0} s')

    a = 'spectrum.pkl'
    newfilename = filename[:-4] + '_likelihood.pkl'
    mydict = {'r':rv, 'like':np.array(allike_on_r)/np.array(allike_on_r).max(), 'ell':data['ell'], 'Dl':data['Dl'], 'error':data['error']}
    output = open(newfilename, 'wb')
    pickle.dump(mydict, output)
    output.close()