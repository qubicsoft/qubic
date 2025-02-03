import numpy as np
import pysm3
import pysm3.units as u
from pysm3 import utils
import healpy as hp
import fgbuster
from fgbuster.component_model import (CMB, Dust, Dust_2b, Synchrotron, AnalyticComponent)
from pylab import *
import os
import qubic
import warnings
import pickle
import sys
from datetime import datetime
from qubic import NamasterLib as nam
from builtins import any
import pandas as pd

warnings.filterwarnings("ignore")
print(fgbuster.__path__)
print(pysm3.__path__)
print(pysm3.__version__)
rcParams['figure.figsize'] = 10, 10

global_dir = './'   # '/sps/qubic/Users/emanzan/libraries/qubic/qubic'
camblib_filename = 'camblib_with_r_from_0.0001.pickle' #'camblib_with_r_from_m0.1_to_p0.1_minstep1e-06.pickle'  #'/doc/CAMB/camblib.pkl' 

def eval_sed(freq_maps, pixok):
    '''
    Function that evaluates the SED as RMS of frequency maps.
    Input: - freq_maps, an array of shape [Nfreq, Nstk, Npix]
    '''
    return np.sqrt( np.std(freq_maps[:,:,pixok], axis=2)**2 + np.mean(freq_maps[:,:,pixok], axis=2)**2 )


def get_freq_maps(freq, sky, npix, pixok):
    '''
    Function that loop over frequency array to get the set of frequency maps
    Input:
    - freq, array of frequency
    - sky, pysm3 sky object
    - npix, number of pixels in the maps
    - coverage, covarage map
    '''
    freq_maps = np.zeros((len(freq),3,npix))
    for f_idx, f in enumerate(freq):
        #np.random.seed(13)
        #freq_maps[f_idx,:,:] = sky.get_emission(f*u.GHz).to(u.uK_CMB, equivalencies=u.cmb_equivalencies(f*u.GHz))
        freq_maps[f_idx,:,:] = sky.get_emission(f*u.GHz)*utils.bandpass_unit_conversion(f*u.GHz,None, u.uK_CMB)
        #.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(f*u.GHz))
        
    #apply mask
    freq_maps[:,:,~pixok] = hp.UNSEEN
    
    return freq_maps


def get_sed_and_maps(freq, sky, npix, coverage):
    '''
    Wrapper function that generates the frequency maps from a pysm3 sky object and then evaluates the RMS SED and returns
    both the maps and the sed
    '''
    pixok = coverage > (np.max(coverage)*0.1)
    freq_maps = get_freq_maps(freq, sky, npix, pixok)
    print('Evaluated maps')
    sed = eval_sed(freq_maps, pixok)
    return sed, freq_maps


def eval_d6_dispersion(freq_vec, n_samples, corr_l, sky, nsided6, verbose=True):
    sky.components[0].correlation_length = corr_l*u.dimensionless_unscaled
    npixd6 = hp.nside2npix(nside=nsided6)
    spectrum = np.zeros((len(freq_vec),n_samples,3,npixd6))

    for f_idx, f in enumerate(freq_vec):
        if verbose:
            print('Doing: {} GHz'.format(f))
        for j in range(n_samples):
            spectrum[f_idx,j,:,:] = sky.get_emission(f*u.GHz).to(u.uK_CMB, equivalencies=u.cmb_equivalencies(f*u.GHz))
    
    sed = np.sqrt( np.std(spectrum, axis=3)**2 + np.mean(spectrum, axis=3)**2 )
    mean_sed = np.mean(sed, axis=1)
    std_sed = np.std(sed, axis=1)
    
    return std_sed, mean_sed, sed, spectrum


def get_coverage(fsky, nside, center_radec=[0., -57.]):
    center = qubic.equ2gal(center_radec[0], center_radec[1])
    uvcenter = np.array(hp.ang2vec(center[0], center[1], lonlat=True))
    uvpix = np.array(hp.pix2vec(nside, np.arange(12*nside**2)))
    ang = np.arccos(np.dot(uvcenter, uvpix))
    indices = np.argsort(ang)
    okpix = ang < -1
    okpix[indices[0:int(fsky * 12*nside**2)]] = True
    mask = np.zeros(12*nside**2)
    mask[okpix] = 1
    return mask


def get_cmb_DlsBB_from_binned_CAMBlib_r_Alens(ell, binned_camblib, r, Alens):
    '''
    This function generates Dls_BB from a given CAMB lib and binning, considering a given r and Alens (amplitude of lensing residual)
    '''  
    #Dls
    DlsBB = qc.get_Dl_fromlib(ell, 0, lib=binned_camblib, unlensed=False, specindex=2)[0] #lensed spectra with r=0
    DlsBB *= Alens #this simulates a lensing residual. Alens = 0 : no lensing. Alens = 1: full lensing 
    DlsBB += qc.get_Dl_fromlib(ell, r, lib=binned_camblib, unlensed=True, specindex=2)[1] #this takes unlensed cmb B-modes for r=1 and scales them to whatever r is given

    return DlsBB[:]


def get_DlsBB_from_CAMB(ell, r, Alens, coverage, nside=256,  cov_cut=0.1, lmin=21, delta_ell=35, lmax = 335):
    '''
    Wrapper function that defines a NaMaster object with a specific mask and ell binning, then uses it to create a camb library and get Dls_BB
    '''     
    #NaMaster obj
    okpix = coverage > (np.max(coverage) * float(cov_cut))
    maskpix = np.zeros(hp.nside2npix(nside))
    maskpix[okpix] = 1
    Namaster = nam.Namaster(maskpix, lmin=lmin, lmax=lmax, delta_ell=delta_ell)#, aposize=2.0)
    
    #binned
    binned_camblib = qc.bin_camblib(Namaster, global_dir + camblib_filename, nside, verbose=False)
    #call Dls generator function
    DlsBB = get_cmb_DlsBB_from_binned_CAMBlib_r_Alens(ell, binned_camblib, r, Alens)

    return DlsBB[:]


def get_spectra(freq_maps, idx_f, coverage, nside,  cov_cut, lmin, delta_ell, lmax, apo_deg):
    '''
    Returns spectra from a given map using namaster and a given ell binning.
    Input:
    - idx_f : a list of 1 or 2 freq. index. Pass only 1 freq in case of auto-Dls, pass 2 freqs in case of cross-Dls
    '''
    
    #def namaster obj from mask and ell binning
    okpix = (coverage == 1) #okpix = coverage > (np.max(coverage) * float(cov_cut))
    '''
    maskpix = np.zeros(hp.nside2npix(nside))
    maskpix[okpix] = 1
    '''
    maskpix = coverage
    Namaster = nam.Namaster(maskpix, lmin=lmin, lmax=lmax, delta_ell=delta_ell, aposize=apo_deg)#, aposize=2.0)
    
    #put to zero all pixel outside the mask
    freq_maps[:,:,~okpix] = 0
    
    #eval auto or cross Dls using namaster and the freq_to_use list. Dls are arrays of [ell,XX]
    w = None
    if len(idx_f) == 1:
        ell, Dls, _ = Namaster.get_spectra(freq_maps[idx_f[0],:,:], map2=freq_maps[idx_f[0],:,:],
                                            purify_e=False, purify_b=True, w=w, verbose=False,
                                            beam_correction=None, pixwin_correction=False)
    
    elif len(idx_f) == 2:
        ell, Dls, _ = Namaster.get_spectra(freq_maps[idx_f[0],:,:], map2=freq_maps[idx_f[1],:,:],
                                            purify_e=False, purify_b=True, w=w, verbose=False,
                                            beam_correction=None, pixwin_correction=False)
        
    else:
        print('ERROR: invalid input frequency index for Dls computation! Pass a list of 1 or 2 frequency index!')
        return    
    
    return ell, Dls


def get_corr_ratio(freq_maps, idx_f, coverage, nside=256,  cov_cut=0.1, lmin=21, delta_ell=35, lmax = 335, apo_deg=0.):
    # eval 
    idx_f_to_use = [idx_f[0]]
    ell, Dls_auto_1 = get_spectra(freq_maps, idx_f_to_use, coverage, nside,  cov_cut, lmin, delta_ell, lmax, apo_deg)
    idx_f_to_use = [idx_f[1]]
    ell, Dls_auto_2 = get_spectra(freq_maps, idx_f_to_use, coverage, nside,  cov_cut, lmin, delta_ell, lmax, apo_deg)
    ell, Dls_cross = get_spectra(freq_maps, idx_f, coverage, nside,  cov_cut, lmin, delta_ell, lmax, apo_deg)

    return ell, Dls_cross[:,2]/np.sqrt(Dls_auto_1[:,2]*Dls_auto_2[:,2]), Dls_auto_1[:,2], Dls_auto_2[:,2], Dls_cross[:,2]
   

def get_planck_spectra(pl_freq_maps, idx_f, coverage, nside,  cov_cut, lmin, delta_ell, lmax, apo_deg):
    '''
    Returns spectra from a given map using namaster and a given ell binning.
    Input:
    - pl_freq_maps : array [N_hm, N_freq, Nstk, Npix]
    - idx_f : a list of 1 or 2 freq. index. Pass only 1 freq in case of auto-Dls, pass 2 freqs in case of cross-Dls
    '''
    
    #def namaster obj from mask and ell binning
    
    okpix = (coverage == 1)#> (np.max(coverage) * 0.) #float(cov_cut)
    '''
    maskpix = np.zeros(hp.nside2npix(nside))
    maskpix[okpix] = 1
    '''
    maskpix = coverage
    Namaster = nam.Namaster(maskpix, lmin=lmin, lmax=lmax, delta_ell=delta_ell, aposize=apo_deg)
    
    #put to zero all pixel outside the mask
    pl_freq_maps[:,:,:,~okpix] = 0
    
    #eval auto or cross Dls using namaster and the freq_to_use list. Dls are arrays of [ell,XX]
    w = None
    if len(idx_f) == 1:
        ell, Dls, _ = Namaster.get_spectra(pl_freq_maps[0,idx_f[0],:,:], map2=pl_freq_maps[1,idx_f[0],:,:],
                                            purify_e=False, purify_b=True, w=w, verbose=False,
                                            beam_correction=None, pixwin_correction=False)
        print(ell.shape, ell)
    
    elif len(idx_f) == 2:
        ell_vec, _ = Namaster.get_binning(nside)
        n_ell = len(ell_vec)
        print(n_ell, ell_vec)
        Dls = np.zeros((n_ell, 4))
        count = 0
        for i in range(2):
            for j in range(2):
                ell, Dlstmp, _ = Namaster.get_spectra(pl_freq_maps[i, idx_f[0],:,:], map2=pl_freq_maps[j, idx_f[1],:,:],
                                            purify_e=False, purify_b=True, w=w, verbose=False,
                                            beam_correction=None, pixwin_correction=False)
                count += 1 
                print('Iter ', count)
                
                Dls += Dlstmp
        Dls = Dls/4
        
    else:            
        print('ERROR: invalid input frequency index for Dls computation! Pass a list of 1 or 2 frequency index!')
        return    
    
    return ell, Dls


def get_planck_corr_ratio(pl_freq_maps, idx_f, coverage, nside=256,  cov_cut=0.1, lmin=21, delta_ell=35, lmax = 335, apo_deg=0.):
    # eval 
    idx_f_to_use = [idx_f[0]]
    print('Auto 1')
    ell, Dls_auto_1 = get_planck_spectra(pl_freq_maps, idx_f_to_use, coverage, nside,  cov_cut, lmin, delta_ell, lmax, apo_deg)
    idx_f_to_use = [idx_f[1]]
    print('Auto 2')
    ell, Dls_auto_2 = get_planck_spectra(pl_freq_maps, idx_f_to_use, coverage, nside,  cov_cut, lmin, delta_ell, lmax, apo_deg)
    print('Cross')
    ell, Dls_cross = get_planck_spectra(pl_freq_maps, idx_f, coverage, nside,  cov_cut, lmin, delta_ell, lmax, apo_deg)

    return ell, Dls_cross[:,2]/np.sqrt(Dls_auto_1[:,2]*Dls_auto_2[:,2]), Dls_auto_1[:,2], Dls_auto_2[:,2], Dls_cross[:,2]


def get_planck_spectra_ave(pl_freq_maps, idx_f, coverage, nside,  cov_cut, lmin, delta_ell, lmax):
    '''
    Returns spectra from a given map using namaster and a given ell binning.
    Input:
    - pl_freq_maps : array [N_hm, N_freq, Nstk, Npix]
    - idx_f : a list of 1 or 2 freq. index. Pass only 1 freq in case of auto-Dls, pass 2 freqs in case of cross-Dls
    '''
    
    #def namaster obj from mask and ell binning
    
    okpix = (coverage == 1)#coverage > (np.max(coverage) * 0.) #float(cov_cut)
    '''
    maskpix = np.zeros(hp.nside2npix(nside))
    maskpix[okpix] = 1
    '''
    maskpix = coverage
    Namaster = nam.Namaster(maskpix, lmin=lmin, lmax=lmax, delta_ell=delta_ell, aposize=0.0)
    
    ell_edge_min = [50,160,320]#,500]
    ell_edge_max = [160,320,500]#,700]
    num_pl_bins = len(ell_edge_min) #4
    ell_binned = [(ell_edge_max[i]+ell_edge_min[i])/2 for i in range(num_pl_bins)]
    
    #put to zero all pixel outside the mask
    pl_freq_maps[:,:,:,~okpix] = 0
    
    #eval auto or cross Dls using namaster and the freq_to_use list. Dls are arrays of [ell,XX]
    Dls_binned = np.zeros((num_pl_bins, 4))
    w = None
    if len(idx_f) == 1:
        ell, Dls, _ = Namaster.get_spectra(pl_freq_maps[0,idx_f[0],:,:], map2=pl_freq_maps[1,idx_f[0],:,:],
                                            purify_e=False, purify_b=True, w=w, verbose=False,
                                            beam_correction=None, pixwin_correction=False)
        print(ell.shape, ell)
        
        for i in range(num_pl_bins):
            ell_to_use = (ell_edge_min[i]<ell) & (ell<ell_edge_max[i])
            Dls_binned[i] = np.mean(Dls[ell_to_use])
    
    elif len(idx_f) == 2:
        #ell_vec, _ = Namaster.get_binning(nside)
        #n_ell = len(ell_vec)
        #print(n_ell, ell_vec)
        count = 0
        for i in range(2):
            for j in range(2):
                ell, Dlstmp, _ = Namaster.get_spectra(pl_freq_maps[i, idx_f[0],:,:], map2=pl_freq_maps[j, idx_f[1],:,:],
                                            purify_e=False, purify_b=True, w=w, verbose=False,
                                            beam_correction=None, pixwin_correction=False)
                
                Dlstmp_binned = np.zeros((num_pl_bins, 4))
                for k in range(num_pl_bins):
                    ell_to_use = (ell_edge_min[k]<ell) & (ell<ell_edge_max[k])
                    Dlstmp_binned[k] = np.mean(Dlstmp[ell_to_use])
                
                count += 1 
                print('Iter ', count)
                
                Dls_binned += Dlstmp_binned
        Dls_binned = Dls_binned/4
        
    else:            
        print('ERROR: invalid input frequency index for Dls computation! Pass a list of 1 or 2 frequency index!')
        return    
    
    return ell_binned, Dls_binned


def get_planck_corr_ratio_ave(pl_freq_maps, idx_f, coverage, nside=256,  cov_cut=0.1, lmin=21, delta_ell=35, lmax = 335):
    # eval 
    idx_f_to_use = [idx_f[0]]
    print('Auto 1')
    ell, Dls_auto_1 = get_planck_spectra_ave(pl_freq_maps, idx_f_to_use, coverage, nside,  cov_cut, lmin, delta_ell, lmax)
    idx_f_to_use = [idx_f[1]]
    print('Auto 2')
    ell, Dls_auto_2 = get_planck_spectra_ave(pl_freq_maps, idx_f_to_use, coverage, nside,  cov_cut, lmin, delta_ell, lmax)
    print('Cross')
    ell, Dls_cross = get_planck_spectra_ave(pl_freq_maps, idx_f, coverage, nside,  cov_cut, lmin, delta_ell, lmax)

    return ell, Dls_cross[:,2]/np.sqrt(Dls_auto_1[:,2]*Dls_auto_2[:,2]), Dls_auto_1[:,2], Dls_auto_2[:,2], Dls_cross[:,2]

    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Nsamples = 500
print('Doing {} realizations'.format(Nsamples))
nside = 256 #512 2048
npix = hp.nside2npix(nside=nside)

fsky = 0.03 #1 #0.03 #CMB-S4 sky fraction
center_ra_dec = [0,-45]#[350,-90] #CMB-S4 sky patch
center = qubic.equ2gal(center_ra_dec[0], center_ra_dec[1])

lmin = 50 #160 #21  #40
lmax = 550 #330 #335 #2*nside-1
delta_ell= 50 #320-160 #200 #35  #30

cov_cut=0.1

path_to_file = './Planck_data/'

#def frequency
freq = np.array([217., 353.005])
idx_freq = [0, 1]

path_to_file = './Planck_data/'
path_to_file_pr2 = './Planck_data/'

#binarized planck PR2 mask
inputfile = path_to_file_pr2+'COM_Mask_Dust-diffuse-and-ps-PIP-L_0{}_R2.00_norm.fits'.format(nside)
covmap = hp.fitsfunc.read_map(inputfile, field=[0], h=False) 
print('Loading : ', inputfile)

data = pd.read_csv(path_to_file+'Corr_ratio_PR3_paper_LR63.csv', skiprows=0, header=None)
ell_paper_pr3 = data[0]
Corr_ratio_paper_pr3 = data[1]

data = pd.read_csv(path_to_file_pr2+'Corr_ratio_PR2_paper_LR63_v2.csv', skiprows=0, header=None)
ell_paper_pr2 = data[0]
Corr_ratio_paper_pr2 = data[1]

data_error = pd.read_csv(path_to_file_pr2+'Error_Corr_ratio_PR2_paper_LR63_v2.csv', skiprows=0, header=None)
Error_Corr_ratio_paper_pr2 = data_error[1] - Corr_ratio_paper_pr2


sky_str=['d0', 'd1', 'd6', 'c1'] #'d4', 'd5', 'd7', 'd6', 'd6', 'd12', 'c1'], 'd6', 'd6', 'd6',
corrl = [10]#,5,1, 4]
sky_obj = []
preset_str = []


#combine cmb + dust model
for i in range(len(sky_str)):
    if sky_str[i]=='c1':
        pass
    else:
        preset_str.append([sky_str[-1],sky_str[i]])  
        print('#####')
        print(preset_str[i])
        sky_obj.append(pysm3.Sky(nside=nside, preset_strings=preset_str[i]))

d6_idx = np.where(np.array(sky_str) == 'd6')[0]
for j in range(len(corrl)):  
    print('setting d6 corr length')
    sky_obj[d6_idx[j]].components[0].correlation_length = corrl[j]*u.dimensionless_unscaled
    print(sky_obj[d6_idx[j]].components[0].correlation_length)  


corr_r = []
for i in range(len(sky_obj)):
    print('#####')
    print(preset_str[i])
    sed, f_maps = get_sed_and_maps(freq, sky_obj[i], npix, covmap)
    ell, corr, _, _, _ = get_corr_ratio(f_maps, idx_freq, covmap, nside=nside,
                                        cov_cut=cov_cut, lmin=lmin, delta_ell=delta_ell, lmax = lmax, apo_deg=5)
    corr_r.append(corr)
    
    
for j in range(Nsamples):
    print('#####')
    print(j+1)
    sed, f_maps = get_sed_and_maps(freq, sky_obj[-1], npix, covmap)
    ell, corr, _, _, _ = get_corr_ratio(f_maps, idx_freq, covmap, nside=nside,
                                        cov_cut=cov_cut, lmin=lmin, delta_ell=delta_ell, lmax = lmax, apo_deg=5)
    corr_r.append(corr)   
       
'''
sed = []
f_maps = []
for i in range(len(sky_obj)):
    print('#####')
    print(preset_str[i])
    sedtmp, mapstmp = get_sed_and_maps(freq, sky_obj[i], npix, covmap)
    sed.append(sedtmp)
    f_maps.append(mapstmp)
    
    
for j in range(Nsamples):
    print('#####')
    sedtmp, mapstmp = get_sed_and_maps(freq, sky_obj[-1], npix, covmap)
    sed.append(sedtmp)
    f_maps.append(mapstmp)  
    
    
corr_r = []
for i in range(len(f_maps)): #sky_obj
    print(i+1) #preset_str[i]
    ell, corr, _, _, _ = get_corr_ratio(f_maps[i], idx_freq, covmap, nside=nside,
                                        cov_cut=cov_cut, lmin=lmin, delta_ell=delta_ell, lmax = lmax, apo_deg=5)
    corr_r.append(corr)
'''

#plot
color = ['b','orange']
fontsize=20   
    
for i in range(len(sky_obj)-1): #-1 #len(sky_obj)-1
    label='CMB + {}'.format(preset_str[i][-1]) #preset_str[i]
    plot(ell, corr_r[i], label=label, marker='o', markersize=9, linestyle='--') #color=color[i],    

corr_ratio_min = np.min(np.array(corr_r[len(sky_obj)-1:]), axis=0)
corr_ratio_max = np.max(np.array(corr_r[len(sky_obj)-1:]), axis=0)
fill_between(ell, corr_ratio_min, corr_ratio_max,
                alpha=0.5, color = 'g', label="CMB + d6; $\ell_{{corr.}}$ = {}".format(corrl[0]))
    
plot(ell_paper_pr3[2:], Corr_ratio_paper_pr3[2:], label='Planck HM Release 3 (2018)', marker='o', markersize=14, color='k', alpha=0.5)
errorbar(ell_paper_pr3[2], 0.995, yerr=0.007, label='300 E2E simulations\nusing Planck HM Release 3 (2018)', marker='o', markersize=14, color='r', capsize=14)
errorbar(ell_paper_pr2[:-1], Corr_ratio_paper_pr2[:-1], yerr = Error_Corr_ratio_paper_pr2[:-1],
         capsize=14,  label='Planck HM Release 2 (2015)', marker='D', markersize=14, color='k')
     
title('$R_{{\ell}}^{{BB}}$ between {:.0f} GHz and {:.0f} GHz'.format(freq[idx_freq[0]], freq[idx_freq[1]]), fontsize=fontsize, y=1.02)
hlines(1, np.min(ell), np.max(ell), linestyle='--', color='k')
xlim([np.min(ell)-5, 450])
ylim([0.8,1.01])
xlabel('$\ell$', fontsize=fontsize)
ylabel('$R_{{\ell}}^{{BB}}$', fontsize=fontsize)
tick_params(axis='both', which='major', labelsize=fontsize)
legend(loc='lower left', fontsize=fontsize, frameon=False)    
    
savefig(path_to_file+'Correlation_ratio_Planck_pysm_models_{}iter.pdf'.format(Nsamples),bbox_inches='tight')

'''
#add planck data
n_hm = 2 #HM 1 and HM2
pl_freq = [217,353]
n_freq = len(pl_freq)
planck_maps = np.zeros((n_hm,n_freq,3,npix)) #[Nhm, Nfreq, Nstk, Npix]
for i in range(n_hm):
    for j in range(n_freq):
        file = path_to_file+'HFI_SkyMap_{}-psb_{}_R3.01_halfmission-{}.fits'.format(pl_freq[j], nside, i+1)
        print('Loading: ', file)
        planck_maps[i,j] = hp.fitsfunc.read_map(file, field=[0,1,2], h=False)
        
        
ell_pl, corr_pl, auto1, auto2, cross = get_planck_corr_ratio(planck_maps, idx_freq, covmap, nside=nside, cov_cut=cov_cut, lmin=lmin, delta_ell=delta_ell, lmax = lmax)

print('Auto spectra 217 GHz: ', auto1)
print('Auto spectra 353 GHz: ', auto2)
print('Cross spectra: ', cross)

print('Correlation ratio: ', corr_pl)
'''
# Save results in pkl file
print('Saving results to file... \n')
pickle.dump([ell, corr_r, corr_ratio_min, corr_ratio_max], open(path_to_file+'CorrRatio_d0d1d6_vs_PlanckPR_nside{}_{}iter.pkl'.format(nside, Nsamples), "wb"))
#pickle.dump([ell_pl, corr_pl, auto1, auto2, cross], open(path_to_file+'CorrRatio_Planck_nside{}.pkl'.format(nside), "wb"))
