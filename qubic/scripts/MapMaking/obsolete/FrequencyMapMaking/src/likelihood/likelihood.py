import numpy as np
import emcee
from model.models import *
#from schwimmbad import MPIPool
from multiprocess import Pool

class Sampler:

    def __init__(self, params):
        
        self.params = params

    def initial_guess(self, mu):

        if self.params['Sampler']['ndim'] != len(mu):
            raise TypeError(f"mu arguments don't have {self.params['Sampler']['ndim']} dimensions")

        x0 = np.zeros((self.params['Sampler']['ndim'] * self.params['Sampler']['N'], self.params['Sampler']['ndim']))
        for ii, i in enumerate(mu):
            x0[:, ii] = np.random.normal(i, self.params['Sampler']['sig_initial_guess'], self.params['Sampler']['ndim'] * self.params['Sampler']['N'])
        return x0

    def mcmc(self, mu, likelihood):

        with Pool() as pool:
        #with MPIPool() as pool:
        #    if not pool.is_master():
        #        pool.wait()
        #        sys.exit(0)
            sampler = emcee.EnsembleSampler(self.params['Sampler']['ndim'] * self.params['Sampler']['N'], self.params['Sampler']['ndim'], likelihood, pool=pool)
            sampler.run_mcmc(self.initial_guess(mu), self.params['Sampler']['nsteps'], progress=True)

        return sampler.get_chain(), sampler.get_chain(discard=self.params['Sampler']['discard'], thin=15, flat=True)






    