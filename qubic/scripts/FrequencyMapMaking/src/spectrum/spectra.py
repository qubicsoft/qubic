#### General packages
import numpy as np
import os
import yaml
import pickle
import matplotlib.pyplot as plt
import healpy as hp
#### QUBIC packages
from qubic import compute_freq
from qubic import NamasterLib as nam
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator


class Spectrum:
    '''
    Class to compute the different spectra for our realisations
    '''

    def __init__(self, file, verbose=True):
        
        print('\n=========== Power Spectra ===========\n')
 
        filename = os.path.split(file)
        self.jobid = filename[1].split('_')[1].split('.')[0]
        print(f'Job id found : ', self.jobid)

        self.path_spectrum = os.path.join(os.path.dirname(os.path.dirname(file)), "spectrum")
        if not os.path.isdir(self.path_spectrum):
            os.makedirs(self.path_spectrum)

        with open('params.yml', "r") as stream:
            self.params = yaml.safe_load(stream)
        
        with open(file, 'rb') as f:
            self.dict_file = pickle.load(f)
        
        self.verbose = verbose
        self.sky_maps = self.dict_file['maps'].copy()
        self.noise_maps = self.dict_file['maps_noise'].copy()
        
        self.nus = self.dict_file['nus']
        self.nfreq = len(self.nus)
        self.nrec = self.params['QUBIC']['nrec']
        self.nsub = self.params['QUBIC']['nsub']
        self.fsub = int(self.nsub / self.nrec)
        self.nside = self.params['SKY']['nside']
        self.nsub = int(self.fsub * self.nrec)

        _, nus150, _, _, _, _ = compute_freq(150, Nfreq=self.fsub-1)
        _, nus220, _, _, _, _ = compute_freq(220, Nfreq=self.fsub-1)

        
        self.fwhm150 = self._get_fwhm_during_MM(nus150)
        self.fwhm220 = self._get_fwhm_during_MM(nus220)
        
        self.kernels150 = np.sqrt(self.fwhm150[0]**2 - self.fwhm150[-1]**2)
        self.kernels220 = np.sqrt(self.fwhm220[0]**2 - self.fwhm220[-1]**2)
        self.kernels = np.array([self.kernels150, self.kernels220])
        
        # Define Namaster class
        self.coverage = self.dict_file['coverage']
        self.seenpix = self.coverage/np.max(self.coverage) > 0.2

        self.namaster = nam.Namaster(weight_mask = list(np.array(self.seenpix)),
                                     lmin = self.params['Spectrum']['lmin'],
                                     lmax = self.params['Spectrum']['lmax'],
                                     delta_ell = self.params['Spectrum']['dl'])

        self.ell = self.namaster.get_binning(self.params['SKY']['nside'])[0]

        print(self.ell)
        #stop
        self.allfwhm = self._get_allfwhm()

    def _get_fwhm_during_MM(self, nu):
        return np.deg2rad(0.39268176 * 150 / nu)
    def _get_allfwhm(self):
        '''
        Function to compute the fwhm for all sub bands.

        Return :
            - allfwhm (list [nfreq])
        '''
        allfwhm = np.zeros(self.nfreq)
        kernels = np.zeros(self.nfreq)
        for i in range(self.nfreq):
            
            if self.params['QUBIC']['convolution_in'] is False : #and self.params['QUBIC']['reconvolution_after_MM'] is False:
                allfwhm[i] = 0
            elif self.params['QUBIC']['convolution_in'] is True : #and self.params['QUBIC']['reconvolution_after_MM'] is False:
                allfwhm[i] = self.dict_file['fwhm_rec'][i]

                
        return allfwhm
    def compute_auto_spectrum(self, map, fwhm):
        '''
        Function to compute the auto-spectrum of a given map

        Argument : 
            - map(array) [nrec/ncomp, npix, nstokes] : map to compute the auto-spectrum
            - allfwhm(float) : in radian
        Return : 
            - (list) [len(ell)] : BB auto-spectrum
        '''

        DlBB = self.namaster.get_spectra(map=map.T, map2=None, verbose=False, beam_correction = np.rad2deg(fwhm))[1][:, 2]
        return DlBB
    def compute_cross_spectrum(self, map1, fwhm1, map2, fwhm2):
        '''
        Function to compute cross-spectrum, taking into account the different resolution of each sub-bands

        Arguments :
            - map1 & map2 (array [nrec/ncomp, npix, nstokes]) : the two maps needed to compute the cross spectrum
            - fwhm1 & fwhm2 (float) : the respective fwhm for map1 & map2 in radian

        Return : 
            - (list) [len(ell)] : BB cross-spectrum
        '''

        # Put the map with the highest resolution at the worst one before doing the cross spectrum
        # Important because the two maps had to be at the same resolution and you can't increase the resolution
        if fwhm1<fwhm2 :
            C = HealpixConvolutionGaussianOperator(fwhm=np.sqrt(fwhm2**2 - fwhm1**2))
            convoluted_map = C*map1
            return self.namaster.get_spectra(map=convoluted_map.T, map2=map2.T, verbose=False, beam_correction = np.rad2deg(fwhm2))[1][:, 2]
        else:
            C = HealpixConvolutionGaussianOperator(fwhm=np.sqrt(fwhm1**2 - fwhm2**2))
            convoluted_map = C*map2
            return self.namaster.get_spectra(map=map1.T, map2=convoluted_map.T, verbose=False, beam_correction = np.rad2deg(fwhm1))[1][:, 2]
    def compute_array_power_spectra(self, maps):
        ''' 
        Function to fill an array with all the power spectra computed

        Argument :
            - maps (array [nreal, nrec/ncomp, npix, nstokes]) : all your realisation maps

        Return :
            - power_spectra_array (array [nrec/ncomp, nrec/ncomp]) : element [i, i] is the auto-spectrum for the reconstructed sub-bands i 
                                                                     element [i, j] is the cross-spectrum between the reconstructed sub-band i & j
        '''

        power_spectra_array = np.zeros((self.nfreq, self.nfreq, len(self.ell)))
        
        for i in range(self.nfreq):
            for j in range(i, self.nfreq):
                print(f'====== {self.nus[i]:.0f}x{self.nus[j]:.0f} ======')
                if i==j :
                    # Compute the auto-spectrum
                    power_spectra_array[i,j] = self.compute_auto_spectrum(maps[i], self.allfwhm[i])
                    print(power_spectra_array[i,j, :3])
                    #stop
                else:
                    # Compute the cross-spectrum
                    power_spectra_array[i,j] = self.compute_cross_spectrum(maps[i], self.allfwhm[i], maps[j], self.allfwhm[j])
        return power_spectra_array
    def compute_power_spectra(self):
        '''
        Function to compute the power spectra array for the sky and for the noise realisations

        Return :
            - sky power spectra array (array [nrec/ncomp, nrec/ncomp])
            - noise power spectra array (array [nrec/ncomp, nrec/ncomp])
        '''
        
        sky_power_spectra = self.compute_array_power_spectra(self.sky_maps)
        noise_power_spectra = self.compute_array_power_spectra(self.noise_maps)
        return sky_power_spectra, noise_power_spectra
    def save_data(self, name, d):

        """
        
        Method to save data using pickle convention.
        
        """
        
        with open(name, 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def run(self):
        
        self.Dl, self.Nl = self.compute_power_spectra()
        
        print('Power spectra computed !!!')
        
        self.save_data(self.path_spectrum + '/' + f'spectrum_{self.jobid}.pkl', {'nus':self.nus,
                              'ell':self.ell,
                              'Dls':self.Dl,
                              'Nl':self.Nl,
                              'coverage': self.coverage,
                              'parameters': self.params})
