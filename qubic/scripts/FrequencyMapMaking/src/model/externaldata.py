import numpy as np
import pickle
import yaml
import qubic.Qacquisition as acq
import pysm3
import pysm3.units as u
from pysm3 import utils
import healpy as hp
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
import sys
import os
from qubic import PATH

class PipelineExternalData:

    def __init__(self, file, noise_only=False):
        
        with open('params.yml', "r") as stream:
            self.params = yaml.safe_load(stream)

        with open('noise.yml', "r") as stream:
            self.noise = yaml.safe_load(stream)
        
        self.noise_only = noise_only
        if self.noise_only:
            self.factor = 0
        else:
            self.factor = 1
        self.external_nus = self._read_external_nus()
        #print('external nus : ', self.external_nus)

        self.nside = self.params['SKY']['nside']
        self.skyconfig = self._get_sky_config()
        self.file = file

    def _get_sky_config(self):
        
        sky = {}

        if self.params['CMB']['cmb']:
            if self.params['CMB']['seed'] == 0:
                seed = np.random.randint(10000000)
            else:
                seed = self.params['CMB']['seed'] 
            sky['cmb'] = seed

        for j in self.params['Foregrounds']:
            #print(j, self.params['Foregrounds'][j])
            if j == 'Dust':
                if self.params['Foregrounds'][j]:
                    sky['dust'] = 'd0'
            elif j == 'Synchrotron':
                if self.params['Foregrounds'][j]:
                    sky['synchrotron'] = 's0'

        return sky
    def _get_depth(self, nus):
    
        res = []
    
        for mynu in nus:
            
            
            is_bicep = np.sum(mynu == np.array([95, 150, 220])) != 0
            is_planck = np.sum(mynu == np.array([30, 44, 70, 100, 143, 217, 353])) != 0

            if is_bicep:
                
                index = self.noise['Bicep']['frequency'].index(mynu) if mynu in self.noise['Bicep']['frequency'] else -1
                if index != -1:
                    d = self.noise['Bicep']['depth_p'][index]
                    res.append(d)
                else:
                    res.append(None)  # Fréquence non trouvée, ajout d'une valeur None
    
            elif is_planck:
                index = self.noise['Planck']['frequency'].index(mynu) if mynu in self.noise['Planck']['frequency'] else -1
                if index != -1:
                    d = self.noise['Planck']['depth_p'][index]
                    res.append(d)
                else:
                    res.append(None)  # Fréquence non trouvée, ajout d'une valeur None
    
        return res
    def _update_data(self, maps, nus):

        data = self.read_pkl(self.file)

        data['maps_ext'] = maps
        data['nus_ext'] = nus

        with open(self.file, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def _read_external_nus(self):

        allnus_pl = [30, 44, 70, 100, 143, 217, 353]
        allnus = []

        if self.params['PLANCK']['external_data']:
            allnus += allnus_pl
        #allnus = allnus_bk + allnus_pl
        nus = []

        for inu, nu in enumerate(allnus):
            if inu < 3:
                nus += [allnus[inu]]
            else:
                 nus += [allnus[inu]]
        #print(nus)
        #stop
        return nus
    def read_pkl(self, name):
        
        with open(name, 'rb') as f:
            data = pickle.load(f)
        return data
    def _get_cmb(self, seed, r=0, Alens=1):
        
        mycls = acq.give_cl_cmb(r=r, Alens=Alens)

        np.random.seed(seed)
        cmb = hp.synfast(mycls, self.nside, verbose=False, new=True).T
        return cmb
    def _get_ave_map(self, central_nu, bw, nb=100):

        is_cmb = False
        model = []
        for key in self.skyconfig.keys():
            if key == 'cmb':
                is_cmb = True
            else:
                model += [self.skyconfig[key]]
        
        mysky = np.zeros((12*self.params['SKY']['nside']**2, 3))

        if len(model) != 0:
            sky = pysm3.Sky(nside=self.nside, preset_strings=model)
            edges_min = central_nu - bw/2
            edges_max = central_nu + bw/2
            bandpass_frequencies = np.linspace(edges_min, edges_max, nb)
            print(f'Integrating bandpass from {edges_min} GHz to {edges_max} GHz with {nb} frequencies.')
            mysky += np.array(sky.get_emission(bandpass_frequencies * u.GHz, None) * utils.bandpass_unit_conversion(bandpass_frequencies * u.GHz, None, u.uK_CMB)).T / 1.5


        if is_cmb:
            cmb = self._get_cmb(self.skyconfig['cmb'])
            mysky += cmb
            
        return mysky * self.factor   
    def _get_fwhm(self, nu):
        fwhmi = self.read_pkl(f'data/Planck{nu:.0f}GHz.pkl')[f'fwhm{nu:.0f}']
        return fwhmi
    def _get_noise(self, nu):
        
        np.random.seed(None)
        #print(nu)
        sigma = self._get_depth([nu])[0] / hp.nside2resol(self.nside, arcmin=True)
        #print(sigma)
        #sigma = np.array([hp.ud_grade(self.read_pkl(f'data/Planck{nu:.0f}GHz.pkl')[f'noise{nu:.0f}'][:, i], self.params['SKY']['nside']) for i in range(3)]).T
        out = np.random.standard_normal(np.ones((12*self.params['SKY']['nside']**2, 3)).shape) * sigma
        return out
    def run(self, fwhm=False, noise=True):

        '''

        Method that create global variables such as :

            - self.maps : Frequency maps from external data with shape (Nf, Npix, Nstk)
            - self.external_nus  : Frequency array [GHz]

        '''

        self.maps = np.zeros((len(self.external_nus), 12*self.nside**2, 3))
        self.fwhm_ext = []
        for inu, nu in enumerate(self.external_nus):
            #print(self.external_nus, inu, nu)
            self.maps[inu] = self._get_ave_map(nu, nu*self.params['PLANCK']['bandwidth_planck'], nb=self.params['PLANCK']['nsub_planck'])

            if noise:
                self.maps[inu] += self._get_noise(nu)
            if fwhm:
                C = HealpixConvolutionGaussianOperator(fwhm=acq.arcmin2rad(self._get_fwhm(nu)))
                self.fwhm_ext.append(acq.arcmin2rad(self._get_fwhm(nu)))
                self.maps[inu] = C(self.maps[inu])
            else:
                self.fwhm_ext.append(0)

        #self._update_data(self.maps, self.external_nus)
        
        #with open(self.params['Data']['datafilename'], 'rb') as f:
        #    data = pickle.load(f)
        #self.maps = np.concatenate((data['maps'], self.maps), axis=0)
        #self.nus = np.concatenate((data['nus'], self.external), axis=0)
        #self.maps[:, ~self.seenpix, :] = 0
        #self.maps[:, :, 0] = 0
        #if self.rank == 0:
        #    self.save_pkl(self.params['Data']['datafilename'], {'maps':self.maps, 'nus':self.nus})

class ExternalDataMM:
    
    def __init__(self, sky, nus, nrec, nside=256, corrected_bandpass=True):
        
        self.nus = nus
        self.nside = nside
        self.nrec = nrec
        self.nsub = len(self.nus)
        self.m_nu = np.zeros((len(self.nus), 12*self.nside**2, 3))
        self.sky = sky
        
        for i in sky.keys():
            if i == 'cmb':
                cmb = self.get_cmb(r=0, Alens=1, seed=self.sky['cmb'])
                self.m_nu += cmb.copy()
            elif i == 'dust':
                self.sky_fg = self._separe_cmb_fg()
                self.sky_pysm = pysm3.Sky(self.nside, preset_strings=self.list_fg)
                self.m_nu_fg = self._get_fg_allnu()
                self.m_nu += self.m_nu_fg.copy()
                
                if corrected_bandpass:
                    self.m_nu = self._corrected_maps(self.m_nu, self.m_nu_fg)
            
        self.maps = self.average_within_band(self.m_nu)
                
        
        
    def give_cl_cmb(self, r=0, Alens=1.):
        
        power_spectrum = hp.read_cl(PATH+'Cls_Planck2018_lensed_scalar.fits')[:,:4000]
        if Alens != 1.:
            power_spectrum[2] *= Alens
        if r:
            power_spectrum += r * hp.read_cl(PATH+'Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits')[:,:4000]
        return power_spectrum
    
    def get_cmb(self, r=0, Alens=1, seed=None):
        
        mycls = self.give_cl_cmb(r, Alens)
        np.random.seed(seed)
        return hp.synfast(mycls, self.nside, verbose=False, new=True).T
    
    def _get_fg_1nu(self, nu):
        return np.array(self.sky_pysm.get_emission(nu * u.GHz, None).T * utils.bandpass_unit_conversion(nu * u.GHz, None, u.uK_CMB)) / 1.5
    
    def _get_fg_allnu(self):
        
        m = np.zeros((len(self.nus), 12*self.nside**2, 3))
        
        for inu, nu in enumerate(self.nus):
            m[inu] = self._get_fg_1nu(nu)
            
        return m
    
    def _separe_cmb_fg(self):
        
        self.list_fg = []
        new_s = {}
        for i in self.sky.keys():
            if i == 'cmb':
                pass
            else:
                new_s[i] = self.sky[i]
                self.list_fg += [self.sky[i]]
            
        return new_s
    
    
    def average_within_band(self, m_nu):
        
        m_mean = np.zeros((self.nrec, 12*self.nside**2, 3))
        f = int(self.nsub / self.nrec)
        for i in range(self.nrec):
            #print(f'Doing average between {np.min(self.nus[i*f:(i+1)*f])} and {np.max(self.nus[i*f:(i+1)*f])} GHz')
            m_mean[i] = np.mean(m_nu[i*f : (i+1)*f], axis=0)
        return m_mean
    
    def _corrected_maps(self, m_nu, m_nu_fg):
        
        f = int(self.nsub / self.nrec)
        
        mean_fg = self.average_within_band(m_nu_fg)
        
        k=0
        for i in range(self.nrec):
            delta = m_nu_fg[i*f : (i+1)*f] - mean_fg[i]
            for j in range(f):
                m_nu[k] -= delta[j]
                k+=1
                
        return m_nu
        
        
    

