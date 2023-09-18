import numpy as np
import healpy as hp
import fgb.mixing_matrix as mm
import fgb.component_model as c
import matplotlib.pyplot as plt
import pickle

#def separate_dictionaries(input_dict):
#    cmb_dict = input_dict.get('CMB', {})
#    foregrounds_dict = input_dict.get('Foregrounds', {})
#    return cmb_dict, foregrounds_dict

class CMBModel:

    """
    
    CMB description assuming parametrized emission law such as :

        Dl_CMB = r * Dl_tensor_r1 + Alens * Dl_lensed
        
        Parameters
        -----------
            - params : Dictionary coming from `params.yml` file that define every parameters
            - ell    : Multipole used during the analysis
    
    """

    def __init__(self, params, ell):
        
        self.params = params
        self.ell = ell
    
    def give_cl_cmb(self):
        
        """
        
        Method to get theoretical CMB BB power spectrum according to Alens and r.


        """
        power_spectrum = hp.read_cl('data/Cls_Planck2018_lensed_scalar.fits')[:,:4000]
        if self.params['Sky']['CMB']['Alens'][0] != 1.:
            power_spectrum *= self.params['Sky']['CMB']['Alens'][0]
        if self.params['Sky']['CMB']['r'][0]:
            power_spectrum += self.params['Sky']['CMB']['r'][0] * hp.read_cl('data/Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits')[:,:4000]
        return power_spectrum
    def cl2dl(self, ell, cl):

        """
        
        Method to convert Cl to Dl which is Dl = ell * (ell + 1) * Cl / 2 * pi
        
        Arguments :
        -----------
            - ell : Array containing multipoles.
            - cl  : Array containing BB power spectrum. 

        """
        
        dl=np.zeros(ell.shape[0])
        for i in range(ell.shape[0]):
            dl[i]=(ell[i]*(ell[i]+1)*cl[i])/(2*np.pi)
        return dl
    def get_Dl_cmb(self):
        
        """
        
        Method to interpolate theoretical BB power spectrum for effective multipoles.

        """
        allDl = self.cl2dl(np.arange(1, 4001, 1), self.give_cl_cmb()[2])
        Dl_eff = np.interp(self.ell, np.arange(1, 4001, 1), allDl)
        return Dl_eff

class ForeGroundModels(CMBModel):

    """
    
    Foreground models assuming parametrized emission law such as :

        Dl_FG = Ad * Delta_d * fnu1d * fnu2d * (ell/80)**alpha_d +
                As * Delta_s * fnu1s * fnu2s * (ell/80)**alpha_s + 
                eps * sqrt(Ad * As) * Delta_d * (fnu1d * fnu2s + fnu2d * fnu1s) * (ell/80)**((alpha_d + alpha_s)/2)
        
        Parameters
        -----------
            - params : Dictionary coming from `params.yml` file that define every parameters
            - nus    : Array that contain every frequencies for the analysis
            - ell    : Multipole used during the analysis
    
    """

    def __init__(self, params, nus, ell):
        
        ### Heritage of CMB model
        CMBModel.__init__(self, params, ell)
        
        self.nus = nus

    def dust_model(self, ell, fnu1, fnu2):
        return self.params['Sky']['Foregrounds']['Ad'][0] * self.params['Sky']['Foregrounds']['deltad'][0] * fnu1 * fnu2 * (ell/80)**self.params['Sky']['Foregrounds']['alphad'][0]
    def sync_model(self, ell, fnu1, fnu2):
        return self.params['Sky']['Foregrounds']['As'][0] * self.params['Sky']['Foregrounds']['deltas'][0] * fnu1 * fnu2 * (ell/80)**self.params['Sky']['Foregrounds']['alphas'][0]
    def dustsync_model(self, ell, fnu1d, fnu2d, fnu1s, fnu2s):
        m = self.params['Sky']['Foregrounds']['eps'][0] * np.sqrt(abs(self.params['Sky']['Foregrounds']['Ad'][0] * self.params['Sky']['Foregrounds']['As'][0])) * \
            (fnu1d*fnu2s + fnu1s*fnu2d) * (ell/80)**((self.params['Sky']['Foregrounds']['alphad'][0] + self.params['Sky']['Foregrounds']['alphas'][0])/2)
        return m
    def scale_dust(self, nu, temp=20):
        
        """
        
        Frequency scaling of thermal dust according to reference frequency nu0_d. 
        
        Arguments :
        -----------
            - nu   : Int number for frequency in GHz
            - temp : Int number for blackbody temperature
        
        """
        
        comp = c.Dust(nu0=self.params['Sky']['Foregrounds']['nu0_d'], temp=temp, beta_d=self.params['Sky']['Foregrounds']['betad'][0])
    
        A = mm.MixingMatrix(comp).evaluator(np.array([nu]))()[0]
        return A
    def scale_sync(self, nu):
        
        """
        
        Frequency scaling of synchrotron according to reference frequency nu0_s. 
        
        Arguments :
        -----------
            - nu   : Int number for frequency in GHz
            
        """
        
        comp = c.Synchrotron(nu0=self.params['Sky']['Foregrounds']['nu0_s'], beta_pl=self.params['Sky']['Foregrounds']['betas'][0])
    
        A = mm.MixingMatrix(comp).evaluator(np.array([nu]))()[0]
        return A
    def get_Dl_fg(self, ell, fnu1d, fnu2d, fnu1s, fnu2s):
        
        """
        
        Method to compute expected foregrounds cross and auto spectrum according to 2 frequencies.

        Returns
        -------
            - ell   : Array containing measured multipole.
            - fnu1d : Int number for scaling of thermal dust for frequency nu1
            - fnu2d : Int number for scaling of thermal dust for frequency nu2
            - fnu1s : Int number for scaling of synchrotron for frequency nu1
            - fnu2s : Int number for scaling of synchrotron for frequency nu2
        
        """

        m = np.zeros(len(ell))

        if self.params['Sky']['Foregrounds']['Dust']:
            m += self.dust_model(ell, fnu1d, fnu2d)

        if self.params['Sky']['Foregrounds']['Synchrotron']:
            m += self.sync_model(ell, fnu1s, fnu2s)

        if self.params['Sky']['Foregrounds']['DustSync']:
            m += self.dustsync_model(ell, fnu1d, fnu2d, fnu1s, fnu2s)

        return m

class Noise:
    
    """
    
    Instance to compute the noise bias and expected errorbars in power spectra using instrument description performance.
    
    Parameters :
    ------------
        - ell    : Array containing effective multipole
        - depths : Array containing noise description en muK.arcmin (read from `noise.yml`)
        
    """
    
    def __init__(self, ell, depths):
        
        self.ell = ell
        self.nbins = len(self.ell)
        self.depths = depths
        self.f = self.ell * (self.ell + 1) / (2 * np.pi)
        self.nspec = len(self.depths)**2
        self.nfreqs = len(self.depths)
    
    def _get_clnoise(self):
        
        """
        
        Method to compute Clnoise from depths.
        
        """
        
        clnoise = np.zeros((self.nfreqs, self.nbins))
        for i in range(self.nfreqs):
            clnoise[i] = np.radians(self.depths[i]/60)**2
        return clnoise
    def _fact(self, fsky=0.03, dl=30):
        
        """
        
        Method to compute variance for theoretical errorbars.

        """
        twoell = 2 * self.ell + 1
        return np.sqrt(2/(twoell * fsky * dl))
    def _combine(self, Dln1, Dln2):
        return np.sqrt(Dln1**2 + Dln2**2)/np.sqrt(2)
    def _get_errors(self):
        
        """
        
        Method that compute theoretical errorbars.

        """
        
        Dln = np.zeros((self.nfreqs, self.nfreqs, self.nbins))
        f = self.ell * (self.ell + 1) / (2 * np.pi)
        clnoise = self._get_clnoise()
    
        k = 0
        ki = 0
        for i in range(self.nfreqs):
            for j in range(self.nfreqs):
                Dln1 = self.f * clnoise[i] * self._fact()
                Dln2 = self.f * clnoise[j] * self._fact()
                if i != j:
                    f = np.sqrt(2)
                else:
                    f = 1
                Dln[i, j] = self._combine(Dln1, Dln2) / f
                
                k+=1
        
        return Dln.reshape(self.nspec, self.nbins)
    def run(self):
        
        """
        
        Method that return N_ell.
        
        """
        
        clnoise = self._get_clnoise()
        Dln = np.zeros((self.nspec, self.nbins))
        k = 0
        ki = 0
        for i in range(self.nfreqs):
            for j in range(self.nfreqs):
                if i == j: # Auto-spectrum
                    Dln[k] = self.f * clnoise[ki]
                    ki += 1
                else:      # Cross-spectrum
                    pass
                
                k += 1
        return Dln

class Sky(ForeGroundModels):

    """
    
    Sky description for CMB + Foregrouds model assuming parametrized emission law. 
        
        Parameters
        -----------
            - params : Dictionary coming from `params.yml` file that define every parameters
            - nus    : Array that contain every frequencies for the analysis
            - ell    : Multipole used during the analysis
    
    """

    def __init__(self, params, nus, ell):

        ### Heritage of Foregrounds instance (remind that ForeGroundsModels instance herit from CMBModel)
        ForeGroundModels.__init__(self, params, nus, ell)

    def model(self, fnu1d, fnu2d, fnu1s, fnu2s):
        
        """
        
        Method to compute Sky model accroding to 2 frequencies.
        
        Arguments :
        -----------
            - fnu1d : Int number for thermal dust frequency scaling for nu1
            - fnu2d : Int number for thermal dust frequency scaling for nu2
            - fnu1s : Int number for synchrotron frequency scaling for nu1
            - fnu2s : Int number for synchrotron frequency scaling for nu2
        
        """
        
        Dl_cmb = self.get_Dl_cmb() * int(self.params['Sky']['CMB']['cmb'])
        Dl_fg = self.get_Dl_fg(self.ell, fnu1d, fnu2d, fnu1s, fnu2s)

        return Dl_cmb + Dl_fg
    def make_list_free_parameter(self):
        
        """
        
        Method that read `params.yml` file and create list of free parameters.

        """
        
        fp = []
        fp_name = []
        fp_latex = []
        k = 0

        for iname, name in enumerate(self.params['Sky'].keys()):
            try:
                for jname, n in enumerate(self.params['Sky'][name]):
                    if type(self.params['Sky'][name][n]) is list:
                        #print(self.params['Sky'][name][n], n)
                        if self.params['Sky'][name][n][1] == 'f':
                            fp += [self.params['Sky'][name][n][-1]]
                            fp_latex += [self.params['Sky'][name][n][2]]
                            fp_name += [list(self.params['Sky'][name].keys())[k]]
                    k += 1
                k = 0
            except: pass

        return fp, fp_name, fp_latex        
    def update_params(self, new_params):
        
        """
        
        Method that update the value of free parameters. Useful during the MCMC process.

        Arguments :
        -----------
            - new_params : Array containing value for free parameters
            
        """
        k = 0
        for iname, name in enumerate(self.params['Sky'].keys()):
            try:
                for jname, n in enumerate(self.params['Sky'][name]):
                    if type(self.params['Sky'][name][n]) is list:
                        if self.params['Sky'][name][n][1] == 'f':

                            self.params['Sky'][name][n][0] = new_params[k]
                            k+=1
            except: pass
        return self.params
    def get_Dl(self, fsky=0.01):
        
        """
        
        Method that compute Dl for a given set of parameters.

        """
        
        fd = np.zeros(len(self.nus))
        fs = np.zeros(len(self.nus))
        for inu, nu in enumerate(self.nus):
            fd[inu] = self.scale_dust(nu)
            fs[inu] = self.scale_sync(nu)
        
        Dl = np.zeros((len(self.nus)*len(self.nus), len(self.ell)))

        k = 0
        for inu, nu in enumerate(self.nus):
            for jnu, nu in enumerate(self.nus):
                
                Dl[k] = self.model(fd[inu], fd[jnu], fs[inu], fs[jnu])

                k += 1

        return Dl
    def _plot_Dl(self, Dl, Dl_errors, model=None, model_fit=None, model_fit_err=None, figsize=(8, 8), fmt='or', title=None):
        
        """
        
        Method to plot the power spectrum
        
        """
        
        num_dl, num_bin = Dl.shape
        num_nus = int(np.sqrt(num_dl))
        
        ell_min, ell_max = self.ell.min(), self.ell.max()
        ell = np.linspace(ell_min, ell_max, model.shape[1])

        plt.figure(figsize=figsize)

        k = 0
        for _ in range(num_nus):
            for _ in range(num_nus):
                plt.subplot(num_nus, num_nus, k+1)
                plt.errorbar(self.ell, Dl[k], yerr=Dl_errors[k], fmt=fmt, capsize=3)
                if model is not None:
                    plt.plot(ell, model[k], '-k', label='Model')
                if model_fit is not None:
                    plt.plot(ell, model_fit[k], '--b', label='Fit')
                    if model_fit_err is not None:
                        plt.fill_between(ell, model_fit[k] - model_fit_err[k]/2, model_fit[k] + model_fit_err[k]/2, color='blue', alpha=0.2)
                if title is not None:
                    plt.title(title[k])

                k+=1

                if k == 1:
                    plt.legend(frameon=False, fontsize=12)
                plt.xlim(20, 250)
        plt.show()