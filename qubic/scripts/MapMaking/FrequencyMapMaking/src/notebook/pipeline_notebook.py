import numpy as np
import yaml
import pickle
import time
import healpy as hp

from model.models import *
from likelihood.likelihood import *
from plots.plotter import *
import mapmaking.systematics as acq
from mapmaking.frequency_acquisition import get_preconditioner
from mapmaking.planck_timeline import *
from mapmaking.noise_timeline import *
from model.externaldata import *
from tools.foldertools import *
import qubic
import os
from fgb.component_model import *
from qubic import NamasterLib as nam
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
from pyoperators import MPI
from tools.cg import pcg

def save_pkl(name, d):
    with open(name, 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

 
__all__ = ['Spectrum', 
           'FakeFrequencyMapMaking', 
           'PipelineFrequencyMapMaking', 
           'PipelineCrossSpectrum', 
           'PipelineEnd2End']

class Spectrum:

    """

    Method to produce power spectrum from I/Q/U maps stored in 'data.pkl'. Returned power spectrum are DlBB = ell * (ell + 1) / 2 * pi * ClBB

    """

    def __init__(self, file, seenpix=None):
        
        with open('params.yml', "r") as stream:
            self.params = yaml.safe_load(stream)

        with open('noise.yml', "r") as stream:
            self.noise = yaml.safe_load(stream)

        self.seenpix = seenpix
        self.mask_nam = np.ones(12*self.params['Sky']['nside']**2)
        if self.seenpix is not None:
            self.mask_nam[~self.seenpix] = 0
        
        if self.params['Spectrum']['method'] == 'namaster':
            self.Namaster = nam.Namaster(self.mask_nam,
                            lmin=self.params['Spectrum']['lmin'],
                            lmax=self.params['Sky']['nside']*2,
                            delta_ell=self.params['Spectrum']['dl'],
                            aposize=self.params['Spectrum']['aposize'])
            
            self.ell, _ = self.Namaster.get_binning(self.params['Sky']['nside'])
        
        elif self.params['Spectrum']['method'] == 'healpy':
            self.ell = self._binned_ell(np.arange(self.params['Spectrum']['lmin'], 2*self.params['Sky']['nside']+1, 1), self.params['Spectrum']['dl'])
        self.file = file
        self.f = self.ell * (self.ell + 1) / (2 * np.pi)
        self.nbins = len(self.ell)

    def _get_depth(self, nus):
        
        '''

        Method to get depth sensitivity from `noise.yml` file according to a given frequency.

        Arguments :
        -----------

            - nus : list constain frequency

        '''

        with open('noise.yml', "r") as stream:
            noise = yaml.safe_load(stream)
    
        res = []
    
        for mynu in nus:
            index = noise['Planck']['frequency'].index(mynu) if mynu in noise['Planck']['frequency'] else -1
            if index != -1:
                d = noise['Planck']['depth_p'][index]
                res.append(d)
            else:
                res.append(None)
    
        return res
    def _get_Dl(self, map1, map2):
        
        '''

        Method which return cross-Dl for BB spectrum. If seenpix is provided, the map used is partial.

        Arguments :
        -----------

            - map1 and map2 : I/Q/U maps from frequency nu1 and nu2. Expected shape is (Npix, Nstk) for each of them.


        '''
        
        if self.params['Spectrum']['method'] == 'namaster':
            _, Dl, _ = self.Namaster.get_spectra(map=map1, 
                                                 map2=map2, 
                                                 beam_correction=None, 
                                                 pixwin_correction=self.params['Spectrum']['pixwin_correction'],
                                                 verbose=False)

            DlBB = Dl[:, 2].copy()

        elif self.params['Spectrum']['method'] == 'healpy':
            
            DlBB = self.f * self._get_Dl_healpy(map1, map2, lmin=self.params['Spectrum']['lmin'], dl=self.params['Spectrum']['dl'])
                           
        else:
            raise TypeError(f"{self.params['Spectrum']['method']} not recognize method")
        
        return DlBB
    def _get_Dl_healpy(self, map1, map2=None, lmin=20, dl=10):

        '''

        Method to compute ClBB auto and cross spectrum using healpy package. It works onyl with full sky maps.

        Arguments :
        -----------

            - map1 and map2 : Array contains I/Q/U maps with shape (Npix, Nstk).
            - lmin : integer for the minimum multipole
            - dl : integer for the binning of the power spectra

        '''

        if map2 is None:
            clBB = hp.alm2cl(hp.map2alm(map1, lmax=2*self.params['Sky']['nside']))[2]
        
        else:
            clBB = hp.alm2cl(alms1=hp.map2alm(map1, lmax=2*self.params['Sky']['nside']), 
                             alms2=hp.map2alm(map2, lmax=2*self.params['Sky']['nside']))[2]
            
        ClBB_binned = self._binned_spectrum(clBB[lmin:], dl)
        
        return ClBB_binned
    def _binned_spectrum(self, spec, dl):

        '''

        Method that binned the spectrum.

        Arguments :
        -----------
            - spec : array contains spectrum
            - dl : integer for binning

        '''

        clBB_binned = []

        for i in range(len(spec) // dl):
            clBB_binned += [np.mean(spec[i*dl:(i+1)*dl])]

        if len(spec) % dl != 0:
            clBB_binned += [np.mean(spec[-(len(spec) % dl):])]

        clBB_binned = np.array(clBB_binned)
    
        return clBB_binned
    def _binned_ell(self, ell, dl):

        '''

        Method that binned the multipoles.

        Arguments :
        -----------
            - ell : array contains mutipoles
            - dl : integer for binning

        '''

        ell_binned = []

        for i in range(len(ell) // dl):
            ell_binned += [np.mean(ell[i*dl:(i+1)*dl])]

        if len(ell) % dl != 0:
            ell_binned += [np.mean(ell[-(len(ell) % dl):])]

        ell_binned = np.array(ell_binned)
    
        return ell_binned
    def _read_pkl(self):

        '''

        Method that reads pickle file containing data.

        '''

        with open(self.file, 'rb') as f:
            data = pickle.load(f)

        return data
    def _update_data(self, ell, Dl):
        
        '''

        Method that update `data.pkl`.

        Arguments :
        -----------
            - ell : array contains multipoles
            - Dl : array contains spectrum

        '''
        
        data = self._read_pkl()

        data['ell'] = ell
        data['Dl'] = Dl

        with open(self.file, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def run(self):
        
        '''

        Method that run the pipeline for BB power spectrum estimation.

        '''

        print('\n=========== Spectrum ===========\n')

        ### Read external data
        maps_ext = self._read_pkl()['maps_ext']
        maps = self._read_pkl()['maps']
        nus_ext = self._read_pkl()['nus_ext']
        nus = self._read_pkl()['nus']

        depths = np.array(list(np.array(self.noise['QUBIC']['depth_p'])) + list(self._get_depth(nus_ext)))
        
        self.noise_model = Noise(self.ell, depths)
        self.noise_correction = self.noise_model.run()

        m = np.concatenate((maps, maps_ext), axis=0)
        nus = np.concatenate((nus, nus_ext), axis=0)

        self.Dl = np.zeros((m.shape[0]**2, self.nbins))
        
        k=0
        for i in range(m.shape[0]):
            for j in range(m.shape[0]):
                if i == j:
                    if self.seenpix is not None:
                        m[i, ~self.seenpix, :] = 0
                        #m[i, :, 0] = 0
                    self.Dl[k] = self._get_Dl(m[i].T, None)
                else:
                    if self.seenpix is not None:
                        m[i, ~self.seenpix, :] = 0
                        m[j, ~self.seenpix, :] = 0
                        #m[i, :, 0] = 0
                        #m[j, :, 0] = 0
                    self.Dl[k] = self._get_Dl(m[i].T, m[j].T)
                k+=1

        if self.params['Spectrum']['noise_correction']:
            self.Dl -= self.noise_correction
        ### Update data.pkl -> Add spectrum
        self._update_data(self.ell, self.Dl)
        print('\n=========== Spectrum - done ===========\n')

class FakeFrequencyMapMaking(ExternalData2Timeline):
    
    def __init__(self, comm, file, fsky=1):
        
        self.comm = comm
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        

        self.file = file
        self.externaldata = PipelineExternalData(file)
        self.job_id = os.environ.get('SLURM_JOB_ID')
        
        with open('params.yml', "r") as stream:
            self.params = yaml.safe_load(stream)

        self.plots = PlotsMM(self.params)

        _, allnus150, _, _, _, _ = qubic.compute_freq(150, Nfreq=1, relative_bandwidth=0.25)
        _, allnus220, _, _, _, _ = qubic.compute_freq(220, Nfreq=1, relative_bandwidth=0.25)

        self.nus = np.array(list(allnus150) + list(allnus220))
        self.skyconfig = self._get_sky_config()

        ExternalData2Timeline.__init__(self, self.skyconfig, self.nus, self.params['QUBIC']['nrec'], self.params['Sky']['nside'], self.params['QUBIC']['bandpass_correction'])
        self.fsky = fsky
        self.center = qubic.equ2gal(self.params['QUBIC']['RA_center'], self.params['QUBIC']['DEC_center'])
        self.coverage = self.get_coverage([self.params['QUBIC']['RA_center'], self.params['QUBIC']['DEC_center']])
        self.seenpix = self.coverage/self.coverage.max() > self.params['QUBIC']['covcut']
        self.coverage_cut = self.coverage.copy()
        self.coverage_cut[~self.seenpix] = 1
        #self.seenpix = seenpix
        self.nfreq, self.npix, self.nstk = self.maps.shape
        self.nus_eff = self.average_nus()
        
        with open('noise.yml', "r") as stream:
            self.noise = yaml.safe_load(stream)

    def _get_sky_config(self):
        
        sky = {}
        for ii, i in enumerate(self.params['Sky'].keys()):

            if i == 'CMB':
                if self.params['Sky']['CMB']['cmb']:
                    if self.params['QUBIC']['seed'] == 0:
                        if self.rank == 0:
                            seed = np.random.randint(10000000)
                        else:
                            seed = None
                        seed = self.comm.bcast(seed, root=0)
                    else:
                        seed = self.params['QUBIC']['seed']
                    sky['cmb'] = seed
                
            else:
                for jj, j in enumerate(self.params['Sky']['Foregrounds']):
                    if j == 'Dust':
                        if self.params['Sky']['Foregrounds'][j]:
                            sky['dust'] = self.params['QUBIC']['dust_model']
                    elif j == 'Synchrotron':
                        if self.params['Sky']['Foregrounds'][j]:
                            sky['synchrotron'] = self.params['QUBIC']['sync_model']

        return sky    
    def get_coverage(self, center_radec):
        center = qubic.equ2gal(center_radec[0], center_radec[1])
        uvcenter = np.array(hp.ang2vec(center[0], center[1], lonlat=True))
        uvpix = np.array(hp.pix2vec(self.nside, np.arange(12*self.nside**2)))
        ang = np.arccos(np.dot(uvcenter, uvpix))
        indices = np.argsort(ang)
        okpix = ang < -1
        okpix[indices[0:int(self.fsky * 12*self.nside**2)]] = True
        mask = np.zeros(12*self.nside**2)
        mask[okpix] = 1
        return mask   
    def average_nus(self):
        
        nus_eff = []
        f = int(self.nsub / self.nrec)
        for i in range(self.nrec):
            nus_eff += [np.mean(self.nus[i*f : (i+1)*f], axis=0)]
        return np.array(nus_eff)
    def _get_sig(self, depth):
        return depth / hp.nside2resol(self.nside, arcmin=True)
    def _get_realI(self, depth, seed=None):
        
        np.random.seed(seed)
        sig = self._get_sig(depth)
        
        return np.random.normal(0, sig, (self.npix))
    def _get_realQU(self, depth, seed=None):
        
        np.random.seed(seed)
        sig = self._get_sig(depth)
        
        return np.random.normal(0, sig, (self.npix, 2))
    def save_data(self, name, d):

        with open(name, 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def run(self):
        
        if self.rank == 0:

            self.m_nu_in = self.maps.copy()
            n = np.zeros(self.maps.shape)
            for i in range(self.nfreq):
                n[i, :, 0] = self._get_realI(np.array(self.noise['QUBIC']['depth_i'])[i])
                n[i, :, 1:] = self._get_realQU(np.array(self.noise['QUBIC']['depth_p'])[i])
            
            self.maps += n.copy()
        
            self.maps[:, ~self.seenpix, :] = 0
            
            self.save_data(self.file, {'maps':self.maps, 'nus':self.nus_eff, 'coverage':self.coverage, 'center':self.center})
            self.externaldata.run(fwhm=self.params['QUBIC']['convolution'], noise=True)
            self.plots.plot_FMM(self.m_nu_in, self.maps, self.center, self.seenpix, self.nus_eff, istk=0, nsig=3)
            self.plots.plot_FMM(self.m_nu_in, self.maps, self.center, self.seenpix, self.nus_eff, istk=1, nsig=3)
            self.plots.plot_FMM(self.m_nu_in, self.maps, self.center, self.seenpix, self.nus_eff, istk=2, nsig=3) 

class NoiseFromDl:
    
    def __init__(self, ell, Dl, nus, depth, dl=30, fsky=0.03):
        
        self.ell = ell
        self.f = self.ell * (self.ell + 1) / (2 * np.pi)
        self.nbins = len(self.ell)
        
        self.nus = nus
        self.nspec = len(self.nus)**2
        
        self.depth = depth
        self.dl = dl
        self.fsky = fsky
        self.clnoise = self._get_clnoise()
        self.Dl = Dl
        print(self.clnoise)
        
    def _get_clnoise(self):
        return np.radians(self.depth/60)**2
    
    def errorbar_theo(self):
    
        errors_auto = np.zeros((len(self.nus), self.nbins))
        errors = np.zeros((self.nspec, self.nbins))
        
        var = np.sqrt(2/((2 * self.ell + 1) * self.dl * self.fsky))
        cl2dl = self.ell * (self.ell + 1) / (2 * np.pi)
        
        for i in range(len(self.nus)):
            errors_auto[i] = cl2dl * self.clnoise[i]
            
        k = 0
        for i in range(len(self.nus)):
            for j in range(len(self.nus)):
                if i == j:
                    errors[k] = errors_auto[i]**2
                else:
                    errors[k] = errors_auto[i] * errors_auto[j] * 0.5
                
                k += 1
        
        return var * np.sqrt(self.Dl**2 + errors**2)

class MapMakingFromSpec(NoiseFromDl, ExternalData2Timeline):
    
    def __init__(self, ell, file, randomreal=True):
        
        with open('params.yml', "r") as stream:
            self.params = yaml.safe_load(stream)
        
        with open('noise.yml', "r") as stream:
            self.noise = yaml.safe_load(stream)
        
        self.job_id = os.environ.get('SLURM_JOB_ID')
        self.file = file
        self.externaldata = PipelineExternalData(file)
        self.skyconfig = self._get_sky_config()
        _, allnus150, _, _, _, _ = qubic.compute_freq(150, Nfreq=1, relative_bandwidth=0.25)
        _, allnus220, _, _, _, _ = qubic.compute_freq(220, Nfreq=1, relative_bandwidth=0.25)

        self.nus = np.array(list(allnus150) + list(allnus220))
        ExternalData2Timeline.__init__(self, self.skyconfig, self.nus, self.params['QUBIC']['nrec'], self.params['Sky']['nside'], self.params['QUBIC']['bandpass_correction'])
        self.nus_eff = self.average_nus()
        self.save_data(self.file, {'nus':self.nus_eff})
        self.externaldata.run(fwhm=self.params['QUBIC']['convolution'], noise=True)
        
        self.allnus = self._read_data()

        self.nspec = len(self.allnus)**2
        self.ell = ell
        self.randomreal = randomreal
        self.sky = Sky(self.params, self.allnus, self.ell)
        
        self.depth = depths = np.array(list(np.array(self.noise['QUBIC']['depth_p'])) + list(self._get_depth(self.externaldata.external_nus)))
    
    def _read_data(self):

        '''

        Method that read `data.pkl` file.

        '''

        with open(self.file, 'rb') as f:
            data = pickle.load(f)
        return np.concatenate((data['nus'], data['nus_ext']), axis=0)
    def save_data(self, name, d):

        with open(name, 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def average_nus(self):
        
        nus_eff = []
        f = int(self.nsub / self.nrec)
        for i in range(self.nrec):
            #print(f'Doing average between {np.min(self.nus[i*f:(i+1)*f])} and {np.max(self.nus[i*f:(i+1)*f])} GHz')
            nus_eff += [np.mean(self.nus[i*f : (i+1)*f], axis=0)]
        return np.array(nus_eff, dtype=float)
    def _get_sky_config(self):
        
        sky = {}
        for ii, i in enumerate(self.params['Sky'].keys()):
            #print(ii, i)

            if i == 'CMB':
                if self.params['Sky']['CMB']['cmb']:
                    if self.params['QUBIC']['seed'] == 0:
                        seed = np.random.randint(10000000)
                        
                    else:
                        seed = self.params['QUBIC']['seed']
                    sky['cmb'] = seed
                
            else:
                for jj, j in enumerate(self.params['Sky']['Foregrounds']):
                    #print(j, self.params['Foregrounds'][j])
                    if j == 'Dust':
                        if self.params['Sky']['Foregrounds'][j]:
                            sky['dust'] = self.params['QUBIC']['dust_model']
                    elif j == 'Synchrotron':
                        if self.params['Sky']['Foregrounds'][j]:
                            sky['synchrotron'] = self.params['QUBIC']['sync_model']

        return sky   
    def _get_depth(self, nus):
        
        '''

        Method to get depth sensitivity from `noise.yml` file according to a given frequency.

        Arguments :
        -----------

            - nus : list constain frequency

        '''

        with open('noise.yml', "r") as stream:
            noise = yaml.safe_load(stream)
    
        res = []
    
        for mynu in nus:
            index = noise['Planck']['frequency'].index(mynu) if mynu in noise['Planck']['frequency'] else -1
            if index != -1:
                d = noise['Planck']['depth_p'][index]
                res.append(d)
            else:
                res.append(None)
    
        return res
    def foregrounds(self, fnu1, fnu2):
    
        return self.A * fnu1 * fnu2 * (self.ell/self.ell0)**self.alpha
    def _read_pkl(self):

        '''

        Method that reads pickle file containing data.

        '''

        with open(self.file, 'rb') as f:
            data = pickle.load(f)

        return data
    def _update_data(self, ell, Dl):
        
        '''

        Method that update `data.pkl`.

        Arguments :
        -----------
            - ell : array contains multipoles
            - Dl : array contains spectrum

        '''
        
        data = self._read_pkl()

        data['ell'] = ell
        data['Dl'] = Dl

        with open(self.file, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def run(self):
        
        Dl = self.sky.get_Dl()

        noise = NoiseFromDl(self.ell, Dl, self.allnus, self.depth, dl=self.params['Spectrum']['dl'], fsky=self.params['QUBIC']['fsky'])
        err = noise.errorbar_theo()

        if self.randomreal:
            np.random.seed(None)
            for j in range(self.nspec):
                for i in range(len(self.ell)):
                    Dl[j, i] = np.random.normal(Dl[j, i], err[j, i]/2)
        
        #self.save_data(self.file, {'nus':self.allnus})
        self._update_data(self.ell, Dl)
        
class PipelineFrequencyMapMaking:

    """
    
    Instance to reconstruct frequency maps using QUBIC abilities.
    
    Parameters :
    ------------
        - comm : MPI communicator
        - file : str to create folder for data saving
    
    """
    
    def __init__(self, comm, file):

        with open('params.yml', "r") as stream:
            self.params = yaml.safe_load(stream)

        self.file = file
        self.externaldata = PipelineExternalData(file)
        self.job_id = np.random.randint(10000)
        
        ### Initialize plot instance
        self.plots = PlotsMM(self.params)

        
        self.center = qubic.equ2gal(self.params['QUBIC']['RA_center'], self.params['QUBIC']['DEC_center'])
        self.fsub = int(self.params['QUBIC']['nsub'] / self.params['QUBIC']['nrec'])

        ### MPI common arguments
        self.comm = comm
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        ### Sky
        self.dict, self.dict_mono = self.get_dict()
        self.skyconfig = self._get_sky_config()
        
        ### Joint acquisition
        self.joint = acq.JointAcquisitionFrequencyMapMaking(self.dict, self.params['QUBIC']['type'], self.params['QUBIC']['nrec'], self.params['QUBIC']['nsub'])
        self.planck_acquisition143 = acq.PlanckAcquisition(143, self.joint.qubic.scene)
        self.planck_acquisition217 = acq.PlanckAcquisition(217, self.joint.qubic.scene)
        self.nus_Q = self._get_averaged_nus()

        ### Joint acquisition for TOD making
        self.joint_tod = acq.JointAcquisitionFrequencyMapMaking(self.dict, self.params['QUBIC']['type'], self.params['QUBIC']['nsub'], self.params['QUBIC']['nsub'])

        ### Coverage map
        self.coverage = self.joint.qubic.subacqs[0].get_coverage()
        covnorm = self.coverage / self.coverage.max()
        self.seenpix = covnorm > self.params['QUBIC']['covcut']
        self.fsky = self.seenpix.astype(float).sum() / self.seenpix.size
        self.coverage_cut = self.coverage.copy()
        self.coverage_cut[~self.seenpix] = 1

        self.seenpix_for_plot = covnorm > 0
        self.mask = np.ones(12*self.params['Sky']['nside']**2)
        self.mask[self.seenpix] = self.params['QUBIC']['kappa']
        
        
        ### Angular resolutions
        self.targets, self.allfwhm = self._get_convolution()
        
        self.external_timeline = ExternalData2Timeline(self.skyconfig, 
                                                       self.joint.qubic.allnus, 
                                                       self.params['QUBIC']['nrec'], 
                                                       nside=self.params['Sky']['nside'], 
                                                       corrected_bandpass=self.params['QUBIC']['bandpass_correction'])

        ### Define reconstructed and TOD operator
        self._get_H()
        ### Inverse noise covariance matrix
        self.invN = self.joint.get_invntt_operator(mask=self.mask)

        ### Noises
        seed_noise_planck = np.random.randint(10000000)
        print('seed_noise_planck', seed_noise_planck)
        
        self.noise143 = self.planck_acquisition143.get_noise(seed_noise_planck) * self.params['Data']['level_planck_noise']
        self.noise217 = self.planck_acquisition217.get_noise(seed_noise_planck+1) * self.params['Data']['level_planck_noise']

        if self.params['QUBIC']['type'] == 'two':
            qubic_noise = QubicDualBandNoise(self.dict, self.params['QUBIC']['npointings'], self.params['QUBIC']['detector_nep'])
        elif self.params['QUBIC']['type'] == 'wide':
            qubic_noise = QubicWideBandNoise(self.dict, self.params['QUBIC']['npointings'], self.params['QUBIC']['detector_nep'])

        self.noiseq = qubic_noise.total_noise(self.params['QUBIC']['ndet'], 
                                       self.params['QUBIC']['npho150'], 
                                       self.params['QUBIC']['npho220'],
                                       seed=seed_noise_planck).ravel()

    def _get_H(self):
        
        """
        
        Method to compute QUBIC operators.
        
        """
        
        self.H = self.joint.get_operator(fwhm=self.targets)
        self.Htod = self.joint_tod.get_operator(fwhm=self.allfwhm)
        self.Hqtod = self.joint_tod.qubic.get_operator(fwhm=self.allfwhm)  
    def _get_averaged_nus(self):
        
        """
        
        Method to average QUBIC frequencies.

        """
        
        nus_eff = []
        for i in range(self.params['QUBIC']['nrec']):
            nus_eff += [np.mean(self.joint.qubic.allnus[i*self.fsub:(i+1)*self.fsub])]
        
        return np.array(nus_eff)
    def _get_sky_config(self):
        
        """
        
        Method that read `params.yml` file and create dictionary containing sky emission such as :
        
                    d = {'cmb':seed, 'dust':'d0', 'synchrotron':'s0'}
        
        Note that the key denote the emission and the value denote the sky model using PySM convention. For CMB, seed denote the realization.
        
        """
        sky = {}
        for ii, i in enumerate(self.params['Sky'].keys()):
            #print(ii, i)

            if i == 'CMB':
                if self.params['Sky']['CMB']['cmb']:
                    if self.params['QUBIC']['seed'] == 0:
                        if self.rank == 0:
                            seed = np.random.randint(10000000)
                        else:
                            seed = None
                        seed = self.comm.bcast(seed, root=0)
                    else:
                        seed = self.params['QUBIC']['seed']
                    print(f'Seed of the CMB is {seed} for rank {self.rank}')
                    sky['cmb'] = seed
                
            else:
                for jj, j in enumerate(self.params['Sky']['Foregrounds']):
                    #print(j, self.params['Foregrounds'][j])
                    if j == 'Dust':
                        if self.params['Sky']['Foregrounds'][j]:
                            sky['dust'] = self.params['QUBIC']['dust_model']
                    elif j == 'Synchrotron':
                        if self.params['Sky']['Foregrounds'][j]:
                            sky['synchrotron'] = self.params['QUBIC']['sync_model']

        return sky
    def get_ultrawideband_config(self):
        
        """
        
        Method that pre-compute UWB configuration.

        """
        
        nu_up = 247.5
        nu_down = 131.25
        nu_ave = np.mean(np.array([nu_up, nu_down]))
        delta = nu_up - nu_ave
    
        return nu_ave, 2*delta/nu_ave
    def get_dict(self):
    
        """
        
        Method to modify the qubic dictionary.
        
        """

        nu_ave, delta_nu_over_nu = self.get_ultrawideband_config()

        args = {'npointings':self.params['QUBIC']['npointings'], 
                'nf_recon':self.params['QUBIC']['nrec'], 
                'nf_sub':self.params['QUBIC']['nsub'], 
                'nside':self.params['Sky']['nside'], 
                'MultiBand':True, 
                'period':1, 
                'RA_center':self.params['QUBIC']['RA_center'], 
                'DEC_center':self.params['QUBIC']['DEC_center'],
                'filter_nu':nu_ave*1e9, 
                'noiseless':False, 
                'comm':self.comm, 
                'dtheta':self.params['QUBIC']['dtheta'],
                'nprocs_sampling':1, 
                'nprocs_instrument':self.size,
                'photon_noise':True, 
                'nhwp_angles':self.params['QUBIC']['nhwp_angles'], 
                'effective_duration':3, 
                'filter_relative_bandwidth':delta_nu_over_nu, 
                'type_instrument':'wide', 
                'TemperatureAtmosphere150':None, 
                'TemperatureAtmosphere220':None,
                'EmissivityAtmosphere150':None, 
                'EmissivityAtmosphere220':None, 
                'detector_nep':float(self.params['QUBIC']['detector_nep']), 
                'synthbeam_kmax':self.params['QUBIC']['synthbeam_kmax']}
        
        args_mono = args.copy()
        args_mono['nf_recon'] = 1
        args_mono['nf_sub'] = 1

        ### Get the default dictionary
        dictfilename = 'dicts/pipeline_demo.dict'
        d = qubic.qubicdict.qubicDict()
        d.read_from_file(dictfilename)
        dmono = d.copy()
        for i in args.keys():
        
            d[str(i)] = args[i]
            dmono[str(i)] = args_mono[i]

    
        return d, dmono
    def _get_convolution(self):

        """
        
        Method to define expected QUBIC angular resolutions (radians) as function of frequencies.

        """
        
        ### Define FWHMs
        if self.params['QUBIC']['convolution']:
            allfwhm = self.joint.qubic.allfwhm
            targets = np.array([])
            for irec in range(self.params['QUBIC']['nrec']):
                targets = np.append(targets, np.sqrt(allfwhm[irec*self.fsub:(irec+1)*self.fsub]**2 - np.min(allfwhm[irec*self.fsub:(irec+1)*self.fsub])**2))
                #targets = np.sqrt(allfwhm**2 - np.min(allfwhm)**2)
        else:
            targets = None
            allfwhm = None

        return targets, allfwhm
    def get_input_map(self):
        m_nu_in = np.zeros((self.params['QUBIC']['nrec'], 12*self.params['Sky']['nside']**2, 3))

        for i in range(self.params['QUBIC']['nrec']):
            m_nu_in[i] = np.mean(self.external_timeline.m_nu[i*self.fsub:(i+1)*self.fsub], axis=0)
        
        return m_nu_in    
    def _get_tod(self):

        """
        
        Method that compute observed TODs with TOD = H . s + n with H the QUBIC operator, s the sky signal and n the instrumental noise.

        """
        
        if self.params['QUBIC']['type'] == 'wide':
            if self.params['QUBIC']['nrec'] != 1:
                TOD_PLANCK = np.zeros((self.params['QUBIC']['nrec'], 12*self.params['Sky']['nside']**2, 3))
                for irec in range(int(self.params['QUBIC']['nrec']/2)):
                    if self.params['QUBIC']['convolution']:
                        C = HealpixConvolutionGaussianOperator(fwhm=np.min(self.allfwhm[irec*self.fsub:(irec+1)*self.fsub]))
                    else:
                        C = HealpixConvolutionGaussianOperator(fwhm=0)
        
                    TOD_PLANCK[irec] = C(self.external_timeline.maps[irec] + self.noise143)

                for irec in range(int(self.params['QUBIC']['nrec']/2), self.params['QUBIC']['nrec']):
                    if self.params['QUBIC']['convolution']:
                        C = HealpixConvolutionGaussianOperator(fwhm=np.min(self.allfwhm[irec*self.fsub:(irec+1)*self.fsub]))
                    else:
                        C = HealpixConvolutionGaussianOperator(fwhm=0)
        
                    TOD_PLANCK[irec] = C(self.external_timeline.maps[irec] + self.noise217)
            else:
                TOD_PLANCK = np.zeros((2*self.params['QUBIC']['nrec'], 12*self.params['Sky']['nside']**2, 3))
                if self.params['QUBIC']['convolution']:
                    C = HealpixConvolutionGaussianOperator(fwhm=self.allfwhm[-1])
                else:
                    C = HealpixConvolutionGaussianOperator(fwhm=0)

                TOD_PLANCK[0] = C(self.external_timeline.maps[0] + self.noise143)
                TOD_PLANCK[1] = C(self.external_timeline.maps[0] + self.noise217)

            TOD_PLANCK = TOD_PLANCK.ravel()
            TOD_QUBIC = self.Hqtod(self.external_timeline.m_nu).ravel() + self.noiseq
            TOD = np.r_[TOD_QUBIC, TOD_PLANCK]

        else:

            sh_q = self.joint.qubic.ndets * self.joint.qubic.nsamples
            TOD_QUBIC = self.Hqtod(self.external_timeline.m_nu).ravel() + self.noiseq

            TOD_QUBIC150 = TOD_QUBIC[:sh_q].copy()
            TOD_QUBIC220 = TOD_QUBIC[sh_q:].copy()

            TOD = TOD_QUBIC150.copy()
    
            TOD_PLANCK = np.zeros((self.params['QUBIC']['nrec'], 12*self.params['Sky']['nside']**2, 3))
            for irec in range(int(self.params['QUBIC']['nrec']/2)):
                if self.params['QUBIC']['convolution']:
                    C = HealpixConvolutionGaussianOperator(fwhm=np.min(self.allfwhm[irec*self.fsub:(irec+1)*self.fsub]))
                else:
                    C = HealpixConvolutionGaussianOperator(fwhm=0)
        
                TOD = np.r_[TOD, C(self.external_timeline.maps[irec] + self.noise143).ravel()]

            TOD = np.r_[TOD, TOD_QUBIC220.copy()]
            for irec in range(int(self.params['QUBIC']['nrec']/2), self.params['QUBIC']['nrec']):
                if self.params['QUBIC']['convolution']:
                    C = HealpixConvolutionGaussianOperator(fwhm=np.min(self.allfwhm[irec*self.fsub:(irec+1)*self.fsub]))
                else:
                    C = HealpixConvolutionGaussianOperator(fwhm=0)
        
                TOD = np.r_[TOD, C(self.external_timeline.maps[irec] + self.noise217).ravel()]

        self.m_nu_in = self.get_input_map()

        return TOD

    def _barrier(self):

        """
        
        Method to introduce comm.Barrier() function if MPI communicator is detected.
        
        """
        if self.comm is None:
            pass
        else:
            self.comm.Barrier()
    def print_message(self, message):
        
        """
        
        Method to print message only on rank 0 if MPI communicator is detected. It display simple message if not.
        
        """
        
        if self.comm is None:
            print(message)
        else:
            if self.rank == 0:
                print(message)
    def _get_preconditionner(self):
        
        conditionner = np.ones((self.params['QUBIC']['nrec'], 12*self.params['Sky']['nside']**2, 3))
            
        for i in range(conditionner.shape[0]):
            for j in range(conditionner.shape[2]):
                conditionner[i, :, j] = 1/self.coverage_cut
                
        return get_preconditioner(conditionner)
    def _pcg(self, d):

        '''
        
        Solve the map-making equation iteratively :     H^T . N^{-1} . H . x = H^T . N^{-1} . d

        The PCG used for the minimization is intrinsequely parallelized (e.g see PyOperators).
        
        '''


        A = self.H.T * self.invN * self.H
        b = self.H.T * self.invN * d
        print(self.params)
        ### Preconditionning
        M = acq.get_preconditioner(np.ones(12*self.params['Sky']['nside']**2))
        #M = self._get_preconditionner()
        #print("PRECONDITIONNER")

        ### PCG
        start = time.time()
        print(start)
        solution_qubic_planck = pcg(A=A, 
                                    b=b, 
                                    comm=self.comm,
                                    x0=self.m_nu_in, 
                                    M=M, 
                                    tol=self.params['PCG']['tol'], 
                                    disp=True, 
                                    maxiter=self.params['PCG']['maxiter'], 
                                    create_gif=self.params['PCG']['gif'], 
                                    center=self.center, 
                                    reso=self.params['QUBIC']['dtheta'], 
                                    seenpix=self.seenpix,
                                    jobid=self.job_id)

        self._barrier()

        if self.params['PCG']['gif']:
            do_gif(f'gif_convergence_{self.job_id}', solution_qubic_planck['nit'], self.job_id)

        if self.params['QUBIC']['nrec'] == 1:
            solution_qubic_planck['x']['x'] = np.array([solution_qubic_planck['x']['x']])
        end = time.time()
        execution_time = end - start
        self.print_message(f'Simulation done in {execution_time:.3f} s')

        return solution_qubic_planck['x']['x']
    def save_data(self, name, d):

        """
        
        Method to save data using pickle convention.
        
        """
        
        with open(name, 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def run(self):

        """
        
        Method to run the whole pipeline from TOD generation from sky reconstruction by reading `params.yml` file.
        
        """
        
        self.print_message('\n=========== Map-Making ===========\n')

        ### Get simulated data
        self.TOD = self._get_tod()

        ### Wait for all processes
        self._barrier()

        ### Solve map-making equation
        self.s_hat = self._pcg(self.TOD)
        
        ### Plots and saving
        if self.rank == 0:
            
            self.save_data(self.file, {'maps':self.s_hat, 'nus':self.nus_Q, 'coverage':self.coverage, 'center':self.center, 'maps_in':self.m_nu_in, 'parameters':self.params})
            self.externaldata.run(fwhm=self.params['QUBIC']['convolution'], noise=True)
            self.plots.plot_FMM(self.m_nu_in, self.s_hat, self.center, self.seenpix, self.nus_Q, job_id=self.job_id, istk=0, nsig=3, fwhm=0.0048)
            self.plots.plot_FMM(self.m_nu_in, self.s_hat, self.center, self.seenpix, self.nus_Q, job_id=self.job_id, istk=1, nsig=3, fwhm=0.0048)
            self.plots.plot_FMM(self.m_nu_in, self.s_hat, self.center, self.seenpix, self.nus_Q, job_id=self.job_id, istk=2, nsig=3, fwhm=0.0048) 

        self._barrier()   

class PipelineCrossSpectrum:

    """
    
    Instance to preform MCMC on multi-components theory on cross-spectrum.
    
    """

    def __init__(self, file, fsky):

        with open('params.yml', "r") as stream:
            self.params = yaml.safe_load(stream)

        with open('noise.yml', "r") as stream:
            self.noise = yaml.safe_load(stream)

        self.file = file
        self.sampler = Sampler(self.params)
        self.plots = Plots()
        self.fsky = fsky
    
    def _get_depth(self, nus):
        
        '''

        Method to get sensitivity depth from `noise.yml` file.

        Arguments :
        -----------
            - nus : list contains frequencies

        '''

        with open('noise.yml', "r") as stream:
            noise = yaml.safe_load(stream)
    
        res = []
    
        for mynu in nus:
            index = noise['Planck']['frequency'].index(mynu) if mynu in noise['Planck']['frequency'] else -1
            if index != -1:
                d = noise['Planck']['depth_p'][index]
                res.append(d)
            else:
                res.append(None)
    
        return res
    def _read_data(self):

        '''

        Method that read `data.pkl` file.

        '''

        with open(self.file, 'rb') as f:
            data = pickle.load(f)
        return data['Dl'], np.concatenate((data['nus'], data['nus_ext']), axis=0), data['ell'], data['nus_ext']
    def _save_data(self, Dl, Dl_err):

        '''

        Method that save new data into file `data.pkl`.

        Arguments :
        -----------
            - Dl : array containing BB power spectrum.
            - Dl_err : array containing errors on the BB power spectrum.

        '''

        with open(self.file, 'wb') as handle:
            pickle.dump({'nus':self.nus, 
                         'ell':self.ell, 
                         'Dl':Dl, 
                         'Dl_err':Dl_err}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def _get_Dl_model(self, x):
        
        '''

        Method that compute multi-components model from parameters.

        Arguments :
        -----------
            - x : array of free parameters

        '''
        params_true = self.sky.update_params(x)
        model_i = Sky(params_true, self.nus, self.ell)
        return model_i.get_Dl(fsky=self.fsky)
    def log_prior(self, x):
        
        '''

        Method that compute prior on free parameters and stop convergence until parameters are not in the range.
        
        Arguments :
        -----------
            - x : array of free parameters
        
        '''
        for iparam, param in enumerate(x):
            if self.name_free_params[iparam] == 'r':
                if param < -1 or param > 1:
                    return - np.inf
            elif self.name_free_params[iparam] == 'Alens':
                if 0 > param or param > 2:
                    return - np.inf
            elif self.name_free_params[iparam] == 'Ad':
                if 0 > param or param > 1e9:
                    return - np.inf
            elif self.name_free_params[iparam] == 'As':
                if param < -1 or param > 1e9:
                    return - np.inf
            elif self.name_free_params[iparam] == 'betad':
                if param < 1 or param > 2:
                    return - np.inf
            elif self.name_free_params[iparam] == 'betas':
                if param < -4 or param > -2:
                    return - np.inf
            elif self.name_free_params[iparam] == 'alphad':
                if param > 0 or param < -1:
                    return - np.inf
            elif self.name_free_params[iparam] == 'alphas':
                if param > 0 or param < -1:
                    return - np.inf
        return 0
    def chi2(self, x):

        lp = self.log_prior(x)
        Dl_model_i = self._get_Dl_model(x)
        return lp - np.sum(((self.Dl[:self.params['Spectrum']['nbins']] - Dl_model_i[:self.params['Spectrum']['nbins']])/self.Dl_err[:self.params['Spectrum']['nbins']])**2)
    def save_chain(self):

        '''
        
        Method to save resulting chains coming from MCMC in pickle file.
        
        '''

        save_pkl('output_chains.pkl', {'chain':self.chain, 
                                       'chain_flat':self.flat_chain,
                                       'true_values':self.value_free_params, 
                                       'true_names':self.name_free_params, 
                                       'true_names_latex':self.name_latex_free_params})
    def run(self):
        
        '''

        Method to run the cross-spectrum analysis.

        '''

        print('\n=========== MCMC ===========\n')

        self.Dl, self.nus, self.ell, self.nus_ext = self._read_data()
        
        depths = np.array(list(np.array(self.noise['QUBIC']['depth_p'])) + list(self._get_depth(self.nus_ext)))
        
        noise = NoiseFromDl(self.ell, self.Dl, self.nus, depths, dl=self.params['Spectrum']['dl'], fsky=self.params['QUBIC']['fsky'])
        self.Dl_err = noise.errorbar_theo() / 2
        #self.Dl_err = Noise(self.ell, depths)._get_errors()
        
        self.sky = Sky(self.params, self.nus, self.ell)

        self.plots.get_Dl_plot(self.ell[:self.params['Spectrum']['nbins']], self.Dl[:, :self.params['Spectrum']['nbins']], 
                               self.Dl_err[:, :self.params['Spectrum']['nbins']], self.nus, model=self.sky.get_Dl()[:, :self.params['Spectrum']['nbins']])
        
        #stop
        ### Free values, name free parameters, latex name free parameters
        self.value_free_params, self.name_free_params, self.name_latex_free_params = self.sky.make_list_free_parameter()
        print(self.value_free_params, self.name_free_params)
        
        ### MCMC
        self.chain, self.flat_chain = self.sampler.mcmc(self.value_free_params, self.chi2)
        self.fitted_params, self.fitted_params_errors = np.mean(self.flat_chain, axis=0), np.std(self.flat_chain, axis=0)
        
        print(f'Fitted parameters : {self.fitted_params}')
        print(f'Errors            : {self.fitted_params_errors}')
        
        ### Save result chains
        self.save_chain()

        ### Plots
        self.plots.get_triangle(self.flat_chain, self.name_free_params, self.name_latex_free_params)
        self.plots.get_convergence(self.chain)

class PipelineEnd2End:

    """

    Wrapper for End-2-End pipeline. It added class one after the others by running method.run().

    """

    def __init__(self, comm):

        with open('params.yml', "r") as stream:
            self.params = yaml.safe_load(stream)

        self.comm = comm
        self.job_id = np.random.randint(10000)
        
        create_folder_if_not_exists(self.comm, f'allplots_{self.job_id}')

        if not os.path.isdir(self.params['path_out']):
            os.makedirs(self.params['path_out'])
        file = self.params['path_out'] + self.params['Data']['datafilename']+f'_{self.job_id}.pkl'

        ### Initialization
        
        if self.params['QUBIC']['method'] == 'MM':
            self.mapmaking = PipelineFrequencyMapMaking(self.comm, file)
            self.spectrum = Spectrum(file, self.mapmaking.seenpix)
            
        elif self.params['QUBIC']['method'] == 'fake':
            self.mapmaking = FakeFrequencyMapMaking(self.comm, file, fsky=self.params['QUBIC']['fsky'])
            self.spectrum = Spectrum(file, self.mapmaking.seenpix)
            
        elif self.params['QUBIC']['method'] == 'spec':
            self.spectrum = Spectrum(file, None)
            self.mapmaking = MapMakingFromSpec(self.spectrum.ell, file, randomreal=self.params['QUBIC']['randomreal'])
        
        self.cross = PipelineCrossSpectrum(file, fsky=self.params['QUBIC']['fsky'])
        
    
    def main(self):

        ### Execute Frequency Map-Making
        self.mapmaking.run()
        
        ### Execute MCMC sampler
        if self.params['Spectrum']['do_spectrum']:
            if self.comm.Get_rank() == 0:

                ### Run -> compute Dl-cross
                self.spectrum.run()

        self.comm.Barrier()
        
        if self.params['Sampler']['do_sampler']:
            ### Run
            self.cross.run()       

        

        
        

