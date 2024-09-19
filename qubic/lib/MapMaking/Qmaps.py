import pickle
import qubic
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pysm3
import pysm3.units as u
import yaml
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
from pysm3 import utils
from qubic.data import PATH as data_dir


# from mapmaking.systematics import give_cl_cmb, arcmin2rad
from qubic.lib.InstrumentModel.Qacquisition import *

from .Qcmbmodel import CMBModel


class Maps:
    
    def __init__(self, skyconfig, nus, nrec, nside=256, corrected_bandpass=True):
        
        self.nus = nus
        self.nside = nside
        self.nrec = nrec
        self.nsub = len(self.nus)
        #self.m_nu = np.zeros((len(self.nus), 12 * self.nside**2, 3))
        self.skyconfig = skyconfig
        
        self.skyconfig_pysm = []
        for key in skyconfig.keys():
            if key == "cmb":
                self.is_cmb = True
            else:
                self.skyconfig_pysm += [skyconfig[key]]
    
    def average_within_band(self, m_nu):

        m_mean = np.zeros((self.nrec, 12*self.nside**2, 3))
        f = int(self.nsub / self.nrec)
        for i in range(self.nrec):
            m_mean[i] = np.mean(m_nu[i * f : (i + 1) * f], axis=0)
        return m_mean
    
    def _get_cmb(self, r, Alens, seed):

        cmbmodel = CMBModel(None)
        mycls = cmbmodel.give_cl_cmb(r, Alens)

        np.random.seed(seed)
        cmb = hp.synfast(mycls, self.nside, verbose=False, new=True).T
        return cmb

    def average_map(self, r, Alens, central_nu, bw, nb=100):

        mysky = np.zeros((12*self.nside**2, 3))

        if len(self.skyconfig_pysm) != 0:
            sky = pysm3.Sky(nside=self.nside, preset_strings=self.skyconfig_pysm)
            edges_min = central_nu - bw / 2
            edges_max = central_nu + bw / 2
            bandpass_frequencies = np.linspace(edges_min, edges_max, nb)
            print(
                f"Integrating bandpass from {edges_min} GHz to {edges_max} GHz with {nb} frequencies."
            )
            mysky += (
                np.array(
                    sky.get_emission(bandpass_frequencies * u.GHz, None)
                    * utils.bandpass_unit_conversion(
                        bandpass_frequencies * u.GHz, None, u.uK_CMB
                    )
                ).T
                / 1.5
            )

        if self.is_cmb:
            cmb = self._get_cmb(r, Alens, self.skyconfig["cmb"])
            mysky += cmb

        return mysky
    def _corrected_maps(self, m_nu, m_nu_fg):

        f = int(self.nsub / self.nrec)

        mean_fg = self.average_within_band(m_nu_fg)

        k = 0
        for i in range(self.nrec):
            delta = m_nu_fg[i * f : (i + 1) * f] - mean_fg[i]
            for j in range(f):
                m_nu[k] -= delta[j]
                k += 1

        return m_nu

class PlanckMaps:

    def __init__(self, nside, r=0, Alens=1):

        #self.params = params

        self.experiments = {'Planck': {'frequency': [30, 44, 70, 100, 143, 217, 353],
                                'depth_i': [150., 162., 210., 77.4, 33., 46.8, 154],
                                'depth_p': [210., 240., 300., 118, 70.2, 105., 439],
                                'fwhm': [32.29, 27.94, 13.08, 9.66, 7.22, 4.90, 4.92],
                                'bw': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]}
                    }

        #self.external_nus = self._read_external_nus()
        # print('external nus : ', self.external_nus)
        self.r = r 
        self.Alens = Alens
        self.nside = nside
        #self.file = file

    
    '''
    def _get_sky_config(self):

        sky = {}

        if self.params["CMB"]["cmb"]:
            if self.params["CMB"]["seed"] == 0:
                seed = np.random.randint(10000000)
            else:
                seed = self.params["CMB"]["seed"]
            sky["cmb"] = seed

        for j in self.params["Foregrounds"]:
            # print(j, self.params['Foregrounds'][j])
            if j == "Dust":
                if self.params["Foregrounds"][j]:
                    sky["dust"] = "d0"
            elif j == "Synchrotron":
                if self.params["Foregrounds"][j]:
                    sky["synchrotron"] = "s0"

        return sky
    '''
    def read_pkl(self, name):

        with open(name, "rb") as f:
            data = pickle.load(f)
        return data
    def _get_cmb(self, r, Alens, seed):

        cmbmodel = CMBModel(None)
        mycls = cmbmodel.give_cl_cmb(r, Alens)

        np.random.seed(seed)
        cmb = hp.synfast(mycls, self.nside, verbose=False, new=True).T
        return cmb
    def _get_ave_map(self, r, Alens, skyconfig, central_nu, bw, nb=100):

        is_cmb = False
        model = []
        for key in skyconfig.keys():
            if key == "cmb":
                is_cmb = True
            else:
                model += [skyconfig[key]]

        mysky = np.zeros((12*self.nside**2, 3))

        if len(model) != 0:
            sky = pysm3.Sky(nside=self.nside, preset_strings=model)
            edges_min = central_nu - bw / 2
            edges_max = central_nu + bw / 2
            bandpass_frequencies = np.linspace(edges_min, edges_max, nb)
            print(
                f"Integrating bandpass from {edges_min} GHz to {edges_max} GHz with {nb} frequencies."
            )
            mysky += (
                np.array(
                    sky.get_emission(bandpass_frequencies * u.GHz, None)
                    * utils.bandpass_unit_conversion(
                        bandpass_frequencies * u.GHz, None, u.uK_CMB
                    )
                ).T
                / 1.5
            )

        if is_cmb:
            cmb = self._get_cmb(r, Alens, skyconfig["cmb"])
            mysky += cmb

        return mysky
    def _get_fwhm(self, nu):
        fwhmi = self.read_pkl(data_dir + f"Planck{nu:.0f}GHz.pkl")[f"fwhm{nu:.0f}"]
        return fwhmi
    def _get_noise(self, nu):
        
        index = self.experiments['Planck']['frequency'].index(nu)
        np.random.seed(None)

        sigma = self.experiments['Planck']['depth_p'][index] / hp.nside2resol(self.nside, arcmin=True)
        
        out = (
            np.random.standard_normal(
                np.ones((12 * self.nside ** 2, 3)).shape
            )
            * sigma
        )
        return out
    def run(self, skyconfig, fwhm=False, number_of_band_integration=100):
        """

        Method that create global variables such as :

            - self.maps : Frequency maps from external data with shape (Nf, Npix, Nstk)
            - self.external_nus  : Frequency array [GHz]

        """

        maps = np.zeros((len(self.experiments['Planck']['frequency']), 12 * self.nside**2, 3))
        maps_noise = np.zeros((len(self.experiments['Planck']['frequency']), 12 * self.nside**2, 3))
        self.fwhm_ext = []
        for inu, nu in enumerate(self.experiments['Planck']['frequency']):
            # print(self.external_nus, inu, nu)
            
            bandwidth = self.experiments['Planck']['bw'][inu]
            maps[inu] = self._get_ave_map(self.r, self.Alens, 
                skyconfig,
                nu,
                nu * bandwidth,
                nb=number_of_band_integration
            )

            maps[inu] *= 0
            
            n = self._get_noise(nu)
            maps[inu] += n
            maps_noise[inu] += n
            
            if fwhm:
                C = HealpixConvolutionGaussianOperator(
                    fwhm=arcmin2rad(self._get_fwhm(nu))
                )
                self.fwhm_ext.append(arcmin2rad(self._get_fwhm(nu)))
                maps[inu] = C(maps[inu])
                maps_noise[inu] = C(maps_noise[inu])
            else:
                self.fwhm_ext.append(0)
        
        return maps, maps_noise

class InputMaps:

    def __init__(self, sky, nus, nrec, nside=256, corrected_bandpass=True):

        self.nus = nus
        self.nside = nside
        self.nrec = nrec
        self.nsub = len(self.nus)
        self.m_nu = np.zeros((len(self.nus), 12 * self.nside**2, 3))
        self.sky = sky

        for i in sky.keys():
            if i == "cmb":
                cmb = self.get_cmb(r=0, Alens=1, seed=self.sky["cmb"])
                self.m_nu += cmb.copy()
            elif i == "dust":
                self.sky_fg = self._separe_cmb_fg()
                self.sky_pysm = pysm3.Sky(self.nside, preset_strings=self.list_fg)
                self.m_nu_fg = self._get_fg_allnu()
                self.m_nu += self.m_nu_fg.copy()

                if corrected_bandpass:
                    self.m_nu = self._corrected_maps(self.m_nu, self.m_nu_fg)

        self.maps = self.average_within_band(self.m_nu)

    def give_cl_cmb(self, r=0, Alens=1.0):

        power_spectrum = hp.read_cl(data_dir+ "Cls_Planck2018_lensed_scalar.fits")[
            :, :4000
        ]
        if Alens != 1.0:
            power_spectrum[2] *= Alens
        if r:
            power_spectrum += (
                r
                * hp.read_cl(
                    data_dir+ "Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits"
                )[:, :4000]
            )
        return power_spectrum

    def get_cmb(self, r=0, Alens=1, seed=None):

        mycls = self.give_cl_cmb(r, Alens)
        np.random.seed(seed)
        return hp.synfast(mycls, self.nside, verbose=False, new=True).T

    def _get_fg_1nu(self, nu):
        return (
            np.array(
                self.sky_pysm.get_emission(nu * u.GHz, None).T
                * utils.bandpass_unit_conversion(nu * u.GHz, None, u.uK_CMB)
            )
            / 1.5
        )

    def _get_fg_allnu(self):

        m = np.zeros((len(self.nus), 12 * self.nside**2, 3))

        for inu, nu in enumerate(self.nus):
            m[inu] = self._get_fg_1nu(nu)

        return m

    def _separe_cmb_fg(self):

        self.list_fg = []
        new_s = {}
        for i in self.sky.keys():
            if i == "cmb":
                pass
            else:
                new_s[i] = self.sky[i]
                self.list_fg += [self.sky[i]]

        return new_s

    def average_within_band(self, m_nu):

        m_mean = np.zeros((self.nrec, 12 * self.nside**2, 3))
        f = int(self.nsub / self.nrec)
        for i in range(self.nrec):
            # print(f'Doing average between {np.min(self.nus[i*f:(i+1)*f])} and {np.max(self.nus[i*f:(i+1)*f])} GHz')
            m_mean[i] = np.mean(m_nu[i * f : (i + 1) * f], axis=0)
        return m_mean

    def _corrected_maps(self, m_nu, m_nu_fg):

        f = int(self.nsub / self.nrec)

        mean_fg = self.average_within_band(m_nu_fg)

        k = 0
        for i in range(self.nrec):
            delta = m_nu_fg[i * f : (i + 1) * f] - mean_fg[i]
            for j in range(f):
                m_nu[k] -= delta[j]
                k += 1

        return m_nu