import fgbuster.mixingmatrix as mm
import healpy as hp
import numpy as np

from qubic.data import PATH as data_dir
from qubic.lib.MapMaking.ComponentMapMaking import Qcomponent_model as c


class CMBModel:
    """

    CMB description assuming parametrized emission law such as :

        Dl_CMB = r * Dl_tensor_r1 + Alens * Dl_lensed

        Parameters
        -----------
            - params : Dictionary coming from `params.yml` file that define every parameters
            - ell    : Multipole used during the analysis

    """

    def __init__(self, ell):
        self.ell = ell

    def give_cl_cmb(self, r, Alens):
        """

        Method to get theoretical CMB BB power spectrum according to Alens and r.


        """
        power_spectrum = hp.read_cl(data_dir + "Cls_Planck2018_lensed_scalar.fits")[:, :4000]
        if Alens != 1.0:
            power_spectrum *= Alens
        if r:
            power_spectrum += r * hp.read_cl(data_dir + "Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits")[:, :4000]
        return power_spectrum

    def cl2dl(self, ell, cl):
        """

        Method to convert Cl to Dl which is Dl = ell * (ell + 1) * Cl / 2 * pi

        Arguments :
        -----------
            - ell : Array containing multipoles.
            - cl  : Array containing BB power spectrum.

        """

        dl = np.zeros(ell.shape[0])
        for i in range(ell.shape[0]):
            dl[i] = (ell[i] * (ell[i] + 1) * cl[i]) / (2 * np.pi)
        return dl

    def get_Dl_cmb(self):
        """

        Method to interpolate theoretical BB power spectrum for effective multipoles.

        """
        allDl = self.cl2dl(np.arange(1, 4001, 1), self.give_cl_cmb()[2])
        Dl_eff = np.interp(self.ell, np.arange(1, 4001, 1), allDl)
        return Dl_eff


class SkySpectra:
    def __init__(self, ell, nus, nu0_d=353, nu0_s=23):
        self.nus = nus
        self.ell = ell
        if self.ell is None:
            self.ell = np.arange(2, 4000)
        self.nbins = len(self.ell)
        self.nfreq = len(self.nus)
        self.nu0_d = nu0_d
        self.nu0_s = nu0_s

    def cl_to_dl(self, cl):
        """
        Function to convert the cls into the dls
        """

        return (self.ell * (self.ell + 1) * cl) / (2 * np.pi)

    def _get_cl_cmb(self, r, Alens):
        """
        Function to compute the CMB power spectrum from the Planck data
        """

        power_spectrum = hp.read_cl(data_dir + "Cls_Planck2018_lensed_scalar.fits")[:, :4000]

        if Alens != 1.0:
            power_spectrum[2] *= Alens
        if r:
            power_spectrum += r * hp.read_cl(data_dir + "Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits")[:, :4000]

        return np.interp(self.ell, np.linspace(1, 4001, 4000), power_spectrum[2])

    def model_cmb(self, r, Alens):
        """
        Define the CMB model, depending on r and Alens
        """

        models = np.zeros((self.nfreq, self.nfreq, len(self.ell)))
        cmb_ps = self._get_cl_cmb(r, Alens)
        models = np.tile(cmb_ps, (self.nfreq, self.nfreq, 1))

        return self.cl_to_dl(models)

    def scale_dust(self, betad, temp=20):
        """
        Function to compute the dust mixing matrix element, depending on the frequency
        """

        comp = c.Dust(nu0=self.nu0_d, temp=temp, beta_d=betad)
        A = mm.MixingMatrix(comp).eval(self.nus)[:, 0]

        return A[None, :] * A[:, None]

    def scale_sync(self, betas):
        """
        Function to compute the dust mixing matrix element, depending on the frequency
        """

        comp = c.Synchrotron(nu0=self.nu0_s, beta_pl=betas)
        A = mm.MixingMatrix(comp).eval(self.nus)[:, 0]

        return A[None, :] * A[:, None]

    def scale_dustsync(self, betad, betas, temp=20):
        comp = c.Dust(nu0=self.nu0_d, temp=temp, beta_d=betad)
        Adust = mm.MixingMatrix(comp).eval(self.nus)[:, 0]

        comp = c.Synchrotron(nu0=self.nu0_s, beta_pl=betas)
        Async = mm.MixingMatrix(comp).eval(self.nus)[:, 0]

        return Adust[None, :] * Async[:, None] + Adust[:, None] * Async[None, :]

    def model(self, r, Alens, Ad, alphad, betad, As, alphas, betas, eps):
        Dl_model = np.zeros((self.nfreq, self.nfreq, self.nbins))

        ### CMB
        Dl_model += self.model_cmb(r, Alens)

        ### Foregrounds
        prod_Anu_d = self.scale_dust(betad=betad)[..., None]
        prod_Anu_s = self.scale_sync(betas=betas)[..., None]

        Dl_model += Ad * prod_Anu_d * (self.ell / 80) ** alphad + As * prod_Anu_s * (self.ell / 80) ** alphas

        prod_Anu_ds = self.scale_dustsync(betad=betad, betas=betas)[..., None]
        Dl_model += eps * np.sqrt(abs(As) * abs(Ad)) * prod_Anu_ds * (self.ell / 80) ** ((alphad + alphas) / 2)

        return Dl_model
