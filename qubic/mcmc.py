import numpy as np
import emcee
from scipy.optimize import curve_fit
from scipy.integrate import cumtrapz
from pylab import *
import iminuit
from qubic import fibtools as ft

__all__ = ['LogLikelihood']



class LogLikelihood:
    def __init__(self, xvals=None, yvals=None, errors=None, model=None, nbins=16,
                 nsiginit=10, nsigprior=20, flatprior=None, fixedpars=None,
                 covariance_model_funct=None, p0=None, nwalkers=32, chi2=None,
                 extra_args = None):
        self.prior = None
        self.model = model
        self.xvals = xvals
        self.yvals = yvals
        self.nbins = nbins
        self.nsiginit = nsiginit
        self.nsigprior = nsigprior
        self.covariance_model_funct = covariance_model_funct
        self.nwalkers = nwalkers
        self.fixedpars = fixedpars
        self.p0=p0

        if np.ndim(errors) == 1:
            self.covar = np.zeros((np.size(errors), np.size(errors)))
            np.fill_diagonal(self.covar, np.array(errors) ** 2)
        else:
            self.covar = errors

        self.flatprior = flatprior
        if not flatprior:
            initial_fit = self.minuit(p0=self.p0)
            self.fitresult = [initial_fit[0], initial_fit[1]]
 
    def __call__(self, mytheta, extra_args=None, verbose=False):
        # if self.fixedpars is not None:
        #     theta = self.p0.copy()
        #     #theta[self.fixedpars == 0] = mytheta
        #     theta[self.fixedpars == 0] = mytheta[self.fixedpars == 0]
        # else:
        #     theta = mytheta
        theta = mytheta

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
        # logLLH = lp - 0.5 * np.dot(np.dot(self.yvals - self.modelval, self.invcov), self.yvals - self.modelval)
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

        cumint = cumtrapz(LLH, x=rvalues) # Cumulative integral
        cumint /= np.max(cumint)
        sigma68 = np.interp(0.68, cumint, rvalues[1:])

        return LLH, sigma68

    def log_priors(self, theta):
        ok = 1
        for i in range(len(theta)):
            if self.flatprior:
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
        if self.flatprior:
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
        print('Ndim init:',ndim)
        if self.fixedpars is not None:
            ndim = int(np.sum(self.fixedpars == 0))
        print('New ndim:', ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.__call__)
        if self.fixedpars is not None:
            print('Len(pos):',np.shape(pos))
            print('len(fixedpars):',len(self.fixedpars))
            pos = pos[:,self.fixedpars==0]
            print('New len(pos):',np.shape(pos))
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
            p0=self.p0
        self.fitresult_curvefit = curve_fit(self.call4curvefit, self.xvals, self.yvals, sigma=np.sqrt(np.diag(self.covar)),
                                   maxfev=1000000, ftol=1e-5, p0=p0)
        return self.fitresult_curvefit[0], self.fitresult_curvefit[1]


    ### This should be modified in order to call the current likelihood instead, not an external one...
    def minuit(self, p0=None, chi2=None, verbose=True, print_level=0, ncallmax=10000, extra_args=None, nsplit=1,
                return_chi2fct=False):
        if p0 is None:
            p0=self.p0
        if verbose & (print_level>1):
            print('About to call Minuit with chi2:')
            print(chi2)
            print('Initial parameters, fixed and bounds:')
            for i in range(len(p0)):
                print('Param {0:}: init={1:6.2f} Fixed={2:} Range=[{3:6.3f}, {4:6.3f}]'.format(i, p0[i], 
                    self.fixedpars[i], self.flatprior[i][0], self.flatprior[i][1]))
        self.fitresult_minuit = ft.do_minuit(self.xvals, self.yvals, self.covar, p0,
            functname=self.model, 
            fixpars=self.fixedpars, rangepars=self.flatprior,
            verbose=verbose, chi2=chi2, print_level=print_level, 
            ncallmax=ncallmax, extra_args=extra_args, nsplit=nsplit)
        if len(self.fitresult_minuit[3])==0:
            cov = np.diag(self.fitresult_minuit[2])
        else:
            cov = self.fitresult_minuit[3]
        if return_chi2fct:
            return self.fitresult_minuit[1], cov, self.fitresult_minuit[6]
        else:
            return self.fitresult_minuit[1], cov

    def random_explore_guess(self, ntry=100, fraction=1, verbose = True, extra_args=None):
        fit_range_simu = self.flatprior
        fit_fixed_simu = self.fixedpars
        myguess_params = np.zeros((ntry, len(fit_range_simu)))
        for i in range(len(fit_range_simu)):
            if fit_fixed_simu[i]==0:
                rng = (fit_range_simu[i][1]-fit_range_simu[i][0]) * fraction
                mini = np.max([fit_range_simu[i][0], self.p0[i]-rng/2])
                maxi = np.min([fit_range_simu[i][0], self.p0[i]+rng/2])
                myguess_params[:,i] = np.random.rand(ntry)*(maxi-mini)+mini
            else:
                myguess_params[:,i] = self.p0[i]
        return myguess_params

