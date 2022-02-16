import emcee
from multiprocessing import cpu_count, Pool
from scipy.optimize import curve_fit
from scipy.integrate import cumtrapz
from pylab import *
from qubic import fibtools as ft

__all__ = ['LogLikelihood']


class LogLikelihood:
    def __init__(self, xvals=None, yvals=None, errors=None, model=None, nbins=16,
                 nsiginit=10, nsigprior=20, flatprior=None, fixedpars=None,
                 covariance_model_funct=None, p0=None, nwalkers=32, chi2=None):
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
        self.p0 = p0
        self.chi2 = chi2

        if np.ndim(errors) == 1:
            self.covar = np.zeros((np.size(errors), np.size(errors)))
            np.fill_diagonal(self.covar, np.array(errors) ** 2)
        else:
            self.covar = errors

        self.flatprior = flatprior
        if not flatprior:
            initial_fit = self.minuit(p0=self.p0, chi2=self.chi2)
            self.fitresult = [initial_fit[0], initial_fit[1]]

    def __call__(self, mytheta, extra_args=None, verbose=False):
        if self.fixedpars is not None:
            theta = self.p0.copy()
            theta[self.fixedpars == 0] = mytheta
            # theta[self.fixedpars == 0] = mytheta[self.fixedpars == 0]
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


class MCMC:
    def __init__(self, nwalkers, niter, ndim, p0, burnin, axis_names, withpool=False, emcee_filename='emcee.h5'):
        self.nwalkers = nwalkers
        self.niter = niter
        self.ndim = ndim
        self.p0 = p0
        self.burnin = burnin
        self.axis_names = axis_names
        self.withpool = withpool
        self.emcee_filename = emcee_filename

    def run(self, lnprob, args, backend=True):

        with Pool() as pool:
            if not self.withpool:
                pool = None
            if backend:
                backend = emcee.backends.HDFBackend(self.emcee_filename)

                sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, lnprob, args=args,
                                                pool=pool,
                                                backend=backend)
                if backend.iteration > 0:
                    self.p0 = backend.get_last_sample()
                if self.niter - backend.iteration > 0:
                    print("\n =========== Running production... ===========")
                    start = time.time()
                    sampler.run_mcmc(self.p0,
                                     nsteps=max(0, self.niter - backend.iteration),
                                     progress=True)
            else:
                sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, lnprob, args=args,
                                                pool=pool,
                                                backend=None)
                print("\n =========== Running production... ===========")
                start = time.time()
                sampler.run_mcmc(self.p0, nsteps=self.niter, progress=True)

        end = time.time()
        print("MCMC took {0:.1f} seconds".format(end - start))
        return sampler

    def read_sampler(self, sampler, has_blobs=True):
        if has_blobs:
            self.blobs = sampler.get_blobs(flat=False)

        self.chains = sampler.get_chain(discard=0, flat=False, thin=1)
        self.chains_flat = sampler.get_chain(discard=0, flat=True, thin=1)
        self.lnprobs = sampler.get_log_prob(discard=0, flat=False, thin=1)

    def read_backends(self):
        reader = emcee.backends.HDFBackend(self.emcee_filename)
        try:
            tau = reader.get_autocorr_time()
        except emcee.autocorr.AutocorrError:
            tau = -1
        self.tau = tau
        if reader.has_blobs():
            self.blobs = reader.get_blobs(flat=False)
        self.chains = reader.get_chain(discard=0, flat=False, thin=1)
        self.chains_flat = reader.get_chain(discard=0, flat=True, thin=1)
        self.lnprobs = reader.get_log_prob(discard=0, flat=False, thin=1)
        return

    def compute_local_acceptance_rate(self, start_index, last_index, walker_index):
        """Compute the local acceptance rate in a chain.

        Parameters
        ----------
        start_index: int
            Beginning index.
        last_index: int
            End index.
        walker_index: int
            Index of the walker.

        Returns
        -------
        freq: float
            The acceptance rate.

        """
        frequences = []
        test = -2 * self.lnprobs[start_index, walker_index]
        counts = 1
        for index in range(start_index + 1, last_index):
            chi2 = -2 * self.lnprobs[index, walker_index]
            if np.isclose(chi2, test):
                counts += 1
            else:
                frequences.append(float(counts))
                counts = 1
                test = chi2
        frequences.append(counts)
        return 1.0 / np.mean(frequences)

    def set_chain_validity(self):
        """Test the validity of a chain: reject chains whose chi2 is far from the mean of the others.

        Returns
        -------
        valid_chains: list
            List of boolean values, True if the chain is valid, or False if invalid.

        """
        nchains = [k for k in range(self.nwalkers)]
        chisq_averages = []
        chisq_std = []
        for k in nchains:
            chisqs = -2 * self.lnprobs[self.burnin:, k]
            chisq_averages.append(np.mean(chisqs))
            chisq_std.append(np.std(chisqs))
        self.global_average = np.mean(chisq_averages)
        self.global_std = np.mean(chisq_std)
        self.valid_chains = [False] * self.nwalkers
        for k in nchains:
            chisqs = -2 * self.lnprobs[self.burnin:, k]
            chisq_average = np.mean(chisqs)
            chisq_std = np.std(chisqs)
            if 3 * self.global_std + self.global_average < chisq_average:
                self.valid_chains[k] = False
            elif chisq_std < 0.1 * self.global_std:
                self.valid_chains[k] = False
            else:
                self.valid_chains[k] = True
        return self.valid_chains

    def get_reduce_chi2(self, nimages, ndet):
        NDDL = (nimages * ndet - (ndet + nimages + 2))
        return self.global_average / NDDL

    def plot_chains_chi2(self, fontsize=14):
        """Plot chains and chi2."""
        chains = self.chains[self.burnin:, :, :]  # .reshape((-1, self.ndim))
        nchains = [k for k in range(self.nwalkers)]
        steps = np.arange(self.burnin, self.niter)

        fig, ax = plt.subplots(self.ndim + 1, 1, figsize=(10, 7), sharex='all')

        # Chi2 vs Index
        print("Chisq statistics:")
        for k in nchains:
            chisqs = -2 * self.lnprobs[self.burnin:, k]
            text = f"\tWalker {k:d}: {float(np.mean(chisqs)):.3f} +/- {float(np.std(chisqs)):.3f}"
            if not self.valid_chains[k]:
                text += " -> excluded"
                ax[self.ndim].plot(steps, chisqs, c='0.5', linestyle='--')
            else:
                ax[self.ndim].plot(steps, chisqs)
            print(text)

        ax[self.ndim].set_ylim(
            [self.global_average - 5 * self.global_std, self.global_average + 5 * self.global_std])

        # Parameter vs Index
        print("Computing Parameter vs Index plots...")
        for i in range(self.ndim):
            h = ax[i].set_ylabel(self.axis_names[i], fontsize=fontsize)
            h.set_rotation(0)
            for k in nchains:
                if self.valid_chains[k]:
                    ax[i].plot(steps, chains[:, k, i])
                else:
                    ax[i].plot(steps, chains[:, k, i], c='0.5', linestyle='--')
                ax[i].get_yaxis().set_label_coords(-0.05, 0.5)
        h = ax[self.ndim].set_ylabel(r'$\chi^2$', fontsize=fontsize)
        h.set_rotation(0)
        ax[self.ndim].set_xlabel('Steps', fontsize=fontsize)
        ax[self.ndim].get_yaxis().set_label_coords(-0.05, 0.5)

        fig.tight_layout()
        plt.subplots_adjust(hspace=0)
        figure_name = self.emcee_filename.replace('.h5', '_chains.pdf')
        print(f'Save figure: {figure_name}')
        fig.savefig(figure_name, dpi=100)

        return

    def convergence_tests(self, xlim=(None, None), fontsize=14):
        """
        Compute the convergence tests (Gelman-Rubin, acceptance rate).
        """
        chains = self.chains[self.burnin:, :, :]
        nchains = [k for k in range(self.nwalkers)]
        steps = np.arange(self.burnin, self.niter)

        # Acceptance rate vs Index
        print("Computing acceptance rate...")
        min_len = self.niter
        window = 100

        fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex='all')
        print(ax.shape)
        if min_len > window:
            for k in nchains:
                ARs = []
                indices = []
                for pos in range(self.burnin + window, self.niter, window):
                    ARs.append(self.compute_local_acceptance_rate(pos - window, pos, k))
                    indices.append(pos)
                if self.valid_chains[k]:
                    ax[1].plot(indices, ARs, label=f'Walker {k:d}')
                else:
                    ax[1].plot(indices, ARs, label=f'Walker {k:d}', c='gray', linestyle='--')
                ax[1].set_xlabel('Steps', fontsize=fontsize)
                ax[1].set_ylabel('Aceptance rate', fontsize=fontsize)
                ax[1].set_xlim(xlim)

        # Gelman-Rubin test
        if len(nchains) > 1:
            step = max(1, (self.niter - self.burnin) // 20)
            self.gelmans = []
            print(f'Gelman-Rubin tests (burnin={self.burnin:d}, step={step:d}, nsteps={self.niter:d}):')
            for i in range(self.ndim):
                Rs = []
                lens = []
                for pos in range(self.burnin + step, self.niter, step):
                    chain_averages = []
                    chain_variances = []
                    global_average = np.mean(self.chains[self.burnin:pos, self.valid_chains, i])
                    for k in nchains:
                        if not self.valid_chains[k]:
                            continue
                        chain_averages.append(np.mean(self.chains[self.burnin:pos, k, i]))
                        chain_variances.append(np.var(self.chains[self.burnin:pos, k, i], ddof=1))
                    W = np.mean(chain_variances)
                    B = 0
                    for n in range(len(chain_averages)):
                        B += (chain_averages[n] - global_average) ** 2
                    B *= ((pos + 1) / (len(chain_averages) - 1))
                    R = (W * pos / (pos + 1) + B / (pos + 1) * (len(chain_averages) + 1) / len(chain_averages)) / W
                    Rs.append(R - 1)
                    lens.append(pos)
                #                 print(f'\t{self.input_labels[i]}: R-1 = {Rs[-1]:.3f} (l = {lens[-1] - 1:d})')
                self.gelmans.append(Rs[-1])
                ax[0].plot(lens, Rs, lw=1, label=self.axis_names[i])

            ax[0].axhline(0.03, c='k', linestyle='--', label='Threshold 0.03')
            ax[0].set_xlabel('Walker length', fontsize=fontsize)
            ax[0].set_ylabel('$R-1$', fontsize=fontsize)
            ax[0].set_ylim(0, 0.3)
            ax[0].set_yticks(np.arange(0, 0.3, 0.1))
            ax[0].set_xlim(xlim)
            ax[0].legend()
        self.gelmans = np.array(self.gelmans)
        fig.tight_layout()
        plt.subplots_adjust(hspace=0)
        figure_name = self.emcee_filename.replace('.h5', '_convergence.pdf')
        print(f'Save figure: {figure_name}')
        fig.savefig(figure_name, dpi=100)

        return

    def get_params_errors(self):
        chains = self.chains[self.burnin:, self.valid_chains, :]

        self.params = np.mean(chains, axis=(0, 1))
        self.params_std = np.std(chains, axis=(0, 1))

        for i in range(self.ndim):
            print(f'***** {self.axis_names[i]}:')
            print(f'{self.params[i]:.5f} +/- {self.params_std[i]:.5f}')

        s, m, p, = np.shape(chains)
        flat_chains = np.reshape(chains, (s * m, p))
        self.params_cov = np.cov(flat_chains.T)
        print(self.params_cov.shape)
        return
