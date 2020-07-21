import numpy as np
import emcee
from scipy.optimize import curve_fit


class LogLikelihood:
    def __init__(self, xvals=None, yvals=None, errors=None, model=None,
                 nsiginit=10, nsigprior=20, flatprior=None, covariance_model_funct=None, p0=None):
        self.prior = None
        self.model = model
        self.xvals = xvals
        self.yvals = yvals
        self.nsiginit = nsiginit
        self.nsigprior = nsigprior
        self.covariance_model_funct = covariance_model_funct

        if np.ndim(errors) == 1:
            self.covar = np.zeros((np.size(errors), np.size(errors)))
            np.fill_diagonal(self.covar, np.array(errors) ** 2)
        else:
            self.covar = errors

        self.invcov = np.linalg.inv(self.covar)

        self.flatprior = flatprior
        if not flatprior:
            self.fitresult = curve_fit(model, self.xvals, self.yvals, sigma=np.sqrt(np.diag(self.covar)),
                                       maxfev=1000000, ftol=1e-5, p0=p0)
            print('Initial Fit: ', self.fitresult)

    def __call__(self, theta):
        val = self.model(self.xvals, *theta)
        if self.covariance_model_funct is None:
            invcov = self.invcov
        else:
            cov = self.covariance_model_funct(val)
            invcov = np.linalg.inv(cov + self.covar)

        lp = self.log_priors(theta)
        toreturn = lp - 0.5 * np.dot(np.dot(self.yvals - val, invcov), self.yvals - val)
        if not np.isfinite(toreturn):
            return -np.inf
        else:
            return toreturn

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

    def run(self, nbmc, nwalkers=32):
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

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.__call__)
        sampler.run_mcmc(pos, nbmc, progress=True)
        return sampler
