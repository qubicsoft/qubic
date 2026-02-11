import numpy as np
import multiprocess as mp
from scipy.optimize import minimize
import time

def myChi2(x, beta, patch_id):
    newbeta = beta.copy()
    newbeta[patch_id] = x

    Hi = allexp.update_A(H_sim, newbeta)

    fakedata = Hi(components.T)
    #print('chi2 ', np.sum((fakedata - data)**2))
    return np.sum((fakedata - tod)**2)

'''
def do_fit(self, x):
        # Ajuster les paramètres
        res = minimize(self.chi2, np.ones(1)*1.5, args=(self.beta, self.H, self.x, self.tod, x))
        return res.x

    def parallel_fit(self, x, num_processes=1):
        # Créer une pool de processus
        start = time.time()
        pool = mp.Pool(processes=num_processes)

        # Effectuer l'ajustement sur chaque jeu de paramètres en parallèle
        results = pool.starmap(self.do_fit, [[param_values] for param_values in x])

        # Fermer la pool de processus
        pool.close()
        pool.join()
    
        print(f'Execution time : {time.time() - start:.3f} s', )

        return results
'''
class FitMultiProcess:

    def __init__(self, chi2, Nprocess, x0, method='TNC', tol=1e-20, options={}):
        
        self.chi2 = chi2
        self.Nprocess = Nprocess
        self.method = method
        self.tol = tol
        self.options = options
        self.x0 = x0
        
    def fit(self, args):

        res = minimize(self.chi2, self.x0, args=args, method=self.method, tol=self.tol, options=self.options)
        return res.x
    
    def perform(self, x):
        pool = mp.Pool(processes=self.Nprocess)
        
        results = pool.starmap(self.fit, [[param_values] for param_values in x])
        pool.close()
        pool.join()

        return results
    
