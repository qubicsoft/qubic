import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import qubic
import pickle
import pysm3
import os
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
from tqdm import tqdm

center = qubic.equ2gal(0, -57)
class DataManagement:
    
    def __init__(self, path, varying=False):
        
        self.varying = varying
        self.path = path
        self.files = os.listdir(self.path)
        self.N = len(self.files) - 190
        #self.ncomps, self.npix, self.nstk = self.open_data(self.files[0])['components_i'].T.shape
        self.components_true = self.open_data(self.files[0])['components']
        #print(path+self.files[0])
        #stop
        if self.varying:
            self.nstk, self.npix, self.ncomps = self.open_data(self.files[0])['components_i'].shape
            self.components_true = self.components_true[:, :, :self.ncomps].T
        else:
            self.ncomps, self.npix, self.nstk = self.open_data(self.files[0])['components_i'].shape
            self.components_true = self.components_true[:self.ncomps]
        
        C = HealpixConvolutionGaussianOperator(fwhm=0.00415369, lmax=2*hp.npix2nside(self.npix))
        nside_fit = 8
        
        self.coverage = self.open_data(self.files[0])['coverage']
        self.seenpix = self.coverage / self.coverage.max() > 0.2
        self.coverage[~self.seenpix] = hp.UNSEEN
        self.seenpix_nside_fit = hp.ud_grade(self.seenpix, nside_fit)
        if self.varying:
            self.beta = np.zeros((self.N, 12*nside_fit**2))
        #print(self.open_data(self.files[0])['beta'].shape)
        #stop
        #print(self.comps_true.shape)
        #stop
        for i in range(self.ncomps):
            self.components_true[i] = C(self.components_true[i])
        
        self.maps = np.zeros((self.N, self.ncomps, self.npix, self.nstk))
        self.residuals = np.zeros((self.N, self.ncomps, self.npix, self.nstk))
        
        for i in tqdm(range(self.N)):
            
            if self.varying:
                self.maps[i] = self.open_data(self.files[i])['components_i'].T.copy()
                self.beta[i] = self.open_data(self.files[0])['beta'][-1, :, 0]
                self.residuals[i] = self.maps[i] - self.components_true
            else:
                self.maps[i] = self.open_data(self.files[i])['components_i'].copy()
                self.residuals[i] = self.maps[i] - self.components_true
                
        #if self.varying:
        #    self.beta[:, ~self.seenpix_nside_fit] = hp.UNSEEN
        
        self.maps[:, :, ~self.seenpix, :] = hp.UNSEEN
        self.components_true[:, ~self.seenpix, :] = hp.UNSEEN
        self.mean_map = np.mean(self.maps, axis=0)
        self.std_map = np.std(self.maps, axis=0)
        
        print(self)
        
    def __repr__(self):
        return f"{self.N} files loaded"
        
    def open_data(self, filename):         
        with open(self.path + '/' + filename, 'rb') as f:
            data = pickle.load(f)
        return data

path = '/pbs/home/m/mregnier/sps1/CMM-Pipeline/src/data_forecast_paper/comparison_DB_vs_UWB/purCMB/'
dm = DataManagement(path + 'parametric_d0_two_inCMB_outCMB_ndet1/',
                    varying=False)
fsky = dm.seenpix.astype(float).sum() / dm.seenpix.size
print(fsky, np.sum(dm.seenpix))
rms_each = np.std(dm.residuals[:, 0, dm.seenpix, :], axis=1)
print(np.mean(rms_each, axis=0))
print(dm.maps.shape)
plt.figure(figsize=(10, 6))

reso = 12
k=1

stk = ['I', 'Q', 'U']
comp = ['CMB', 'Dust @ 150 GHz']
for icomp in range(dm.ncomps):

    hp.gnomview(dm.components_true[icomp, :, 1], rot=center, reso=reso, cmap='jet', sub=(dm.ncomps, 3, k),
    notext=True, title=f'Input - {stk[1]} - {comp[icomp]}', min=-8, max=8)
    hp.gnomview(np.mean(dm.maps, axis=0)[icomp, :, 1], rot=center, reso=reso, cmap='jet', sub=(dm.ncomps, 3, k+1),
    notext=True, title=f'Output - {stk[1]} - {comp[icomp]}', min=-8, max=8)
    _r = dm.components_true[icomp, :, 1] - dm.mean_map[icomp, :, 1]
    _r[~dm.seenpix] = hp.UNSEEN
    hp.gnomview(_r, rot=center, reso=reso, cmap='jet', sub=(dm.ncomps, 3, k+2),
    notext=True, title=f'Residuals - {stk[1]}', min=-8, max=8)
    k+=3

plt.savefig('maps_d0.png')
plt.close()

if dm.varying:
    sky = pysm3.Sky(nside=256, preset_strings=["d1"])
    mbb_index = np.array(sky.components[0].mbb_index)
    mbb_index = hp.ud_grade(mbb_index, 8)
    #mbb_index[~dm.seenpix_nside_fit] = hp.UNSEEN
    plt.figure(figsize=(15, 10))

    hp.mollview(np.mean(dm.beta, axis=0), cmap='jet', sub=(1, 3, 1),
        notext=True, title='', min=1.45, max=1.65)

    hp.mollview(mbb_index, cmap='jet', sub=(1, 3, 2),
        notext=True, title='', min=1.45, max=1.65)

    _r = np.mean(dm.beta, axis=0) - mbb_index
    #_r[~dm.seenpix_nside_fit] = hp.UNSEEN
    hp.mollview(_r, cmap='jet', sub=(1, 3, 3),
        notext=True, title='', min=-0.3, max=0.3)

    plt.savefig('beta_d1.png')
    plt.close()
