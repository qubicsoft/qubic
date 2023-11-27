import numpy as np
import warnings
import pickle
import sys
from pylab import *
import os
from datetime import datetime
import healpy as hp
import json

import pysm3
import pysm3.units as u
from pysm3 import utils

import qubic
from qubic import NamasterLib as nam

import fgbuster
from fgbuster.component_model import (CMB, Dust, Dust_2b, Synchrotron, AnalyticComponent)
warnings.filterwarnings("ignore")

import qubicplus_skygen_lib
import utilitylib
import likelihoodlib
from builtins import any
from importlib.util import find_spec

print()
print('#---- Checking some FGB paths... -----#')
print(fgbuster.__path__)
print(find_spec('cmbdb').submodule_search_locations[0])
print()
#~~~~~~~~~CODE PART~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#def sky and instrument model here : MODIFY THIS PART ACCORDINGLY -----------------------------------

call_number = int(sys.argv[1]) # in case you call the same simulation many times: this is the number of the call. Otherwise, just put 1
name_param_file = str(sys.argv[2])
#apo_size_deg = 10. #5. # float((sys.argv[3]))
print('Call number: {}'.format(call_number))

# load in a dictionary with the parameter of the simulation
with open('./'+name_param_file) as f:
    data = f.read()      
d = json.loads(data)

#save the parameters
N = d['num_of_iter'] #number of iterations, at least 1
save_figs = eval(d['save_fig']) # Bool value. If true, save plots 

r = float(d['r']) #r value
Alens = float(d['Alens']) #amplitude of lensing residual. Alens=0: no lensing. Alens=1: full lensing
use_seed_file = eval(d['use_seed_file']) #Boolean : if True pick cmb seed from an external file, otherwise generate the seeds inside the MC

lmin = int(d['ell_min']) #21
lmax = int(d['ell_max']) #335
delta_l = int(d['delta_ell']) #35
apo_size_deg = float(d['apo_deg']) # float((sys.argv[3])) # Apodization radius in degrees

#def sky map info
center_ra_dec = d['center_ra_dec'] #CMB-S4 sky patch
center = qubic.equ2gal(center_ra_dec[0], center_ra_dec[1])
print('Sky patch is centered at RA,DEC = ', center_ra_dec)
print('WARNING: if you want to change the center of the sky patch, stop the code now!')

fsky = d['fsky'] #CMB-S4 sky fraction
nside = d['nside'] #healpix nside
npix = hp.nside2npix(nside) #number of pixels

n_stk = d['n_stk'] # n_stk = 2 if the compsep is on Q and U, not I. Otherwise n_stk = 3. The input maps (see later) will have 3 stokes components in any case
stk = d['stk']
N_sample_band = d['N_sample_band'] #number of freqs to use in the bandpass integration. Put it to 1 if you want to turn bp integration off

#Specify FGB info
nu0 = d['nu0'] # GHz, This is the FGB nu0 value i.e. the frequency where the component amplitude is 1
print('WARNING: nu0 is fixed at nu0 = {} GHz. If you want to change it, stop the code now!'.format(nu0))
Nside_fit = int(d['Nside_fit']) #If the foregrounds have spatially varying parameters, then FGB will estimate the parameters on sky patches with this Nside. Put it to zero if you want the foreground parameters to be scalar

#Specify which instrument you want to simulate
sim_S4 = eval(d['sim_S4']) #Boolean
sim_BI = eval(d['sim_BI']) #Boolean
n_subbands = d['n_subbands'] #list with number of sub-bands to simulate for each BI case
tot_n_subs_to_sim = len(n_subbands)

#Specify which sky you want to simulate
add_noise = eval(d['add_noise']) #If True, sim is with Noise (added in MC loop)
spectra_type = d['spectra_type'] #cross or auto: if cross, the depth must be increased of sqrt(2)
if eval(d['sim_foregrounds']) is not None:
    skyconfig = {'cmb': 42, 'pysm_fg': d['pysm_fg']} #specify only FG here. CMB will be added later (in the MC loop). 42 is just a reference seed which will be changed later
else:
    skyconfig = {'cmb': 42}
print('There are {} components: {}'.format(len(skyconfig.keys()), skyconfig.keys()))

r = float(d['r']) #r value
Alens = float(d['Alens']) #amplitude of lensing residual. Alens=0: no lensing. Alens=1: full lensing

corr_l = float(d['corr_l']) #dust correlation lenght. Put to zero if you want d0 or d1
#if there is no decorrelation, put corr_l to None
if corr_l == 0.:
    corr_l = None

temp_is_fixed = eval(d['temp_is_fixed']) #if this is True, the temperature is fixed. Else, is a parameter
temp = d['temp']
print('Is temperature fixed? : ', temp_is_fixed)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#define a str with the simulated components to be used in title/name of the saved images/files
file_name = '{}_PipelineFinalConfNoiseNew2_Comp'.format(call_number)
#file_name = '{}_template_apo{}_Comp'.format(call_number, apo_size_deg)
for i in skyconfig.keys():
    file_name += '_{}'.format(skyconfig[i])
file_name += '_{}_{}_{}'.format(r, Alens, corr_l)
#-----------------------------------------------------------------------------------------------
print()
print('#### SETTING INSTRUMENT CONFIGURATION ####')
instrum_config = []

#def CMB-S4 config. Do this by default
print('Configure S4')
freqs = np.array([20., 30., 40., 85., 95., 145., 155., 220., 270.])
bandwidth = np.array([5., 9., 12., 20.4, 22.8, 31.9, 34.1, 48.4, 59.4])
dnu_nu = bandwidth/freqs
edges_min = freqs * (1. - dnu_nu/2)
edges_max = freqs * (1. + dnu_nu/2)
edges = [[edges_min[i], edges_max[i]] for i in range(len(freqs))]
beam_fwhm = np.array([11., 72.8, 72.8, 25.5, 25.5, 22.7, 22.7, 13., 13.])
mukarcmin_TT = np.array([16.5, 9.36, 11.85, 2.02, 1.78, 3.89, 4.16, 10.15, 17.4])
mukarcmin_EE = np.array([10.87, 6.2, 7.85, 1.34, 1.18, 1.8, 1.93, 4.71, 8.08])
mukarcmin_BB = np.array([10.23, 5.85, 7.4, 1.27, 1.12, 1.76, 1.89, 4.6, 7.89])
mukarcmin_P = np.sqrt(mukarcmin_EE**2 + mukarcmin_BB**2) #0.5*(mukarcmin_EE + mukarcmin_BB)
if spectra_type=='cross':
    print('Spectra type is: Cross. Increase noise level by sqrt(2)')
    mukarcmin_TT*= np.sqrt(2)
    mukarcmin_P*= np.sqrt(2)
elif spectra_type=='auto':
    print('Spectra type is: Auto. Noise level is not increased.')
else:
    print('ERROR: Spectra type is unknown! Please set a spectra type')
    exit()
    
ell_min = np.array([30, 30, 30, 30, 30, 30, 30, 30, 30])
Nside = np.array([512, 512, 512, 512, 512, 512, 512, 512, 512])

#CMB-S4 dict config
s4_config = {
    'name': 'S4',
    'center_radec': center_ra_dec,
    'nbands': len(freqs),
    'frequency': freqs,
    'bandwidth': bandwidth,
    'dnu_nu': dnu_nu,
    'edges': edges,
    'fwhm': beam_fwhm,
    'depth_p': mukarcmin_P,
    'depth_i': mukarcmin_TT,
    'depth_e': mukarcmin_EE,
    'depth_b': mukarcmin_BB,
    'ell_min': ell_min,
    'nside': Nside,
    'fsky': fsky,
    'ntubes': 12,
    'nyears': 7.,
    'effective_fraction': np.zeros(len(freqs))+1.}
#if you are simulating S4, save this dict as your instrument
if sim_S4: 
    instrum_config.append(s4_config)
    print(s4_config['depth_p'])#*np.sqrt(2))

#define the pixok set
covmap = utilitylib.get_coverage(fsky, nside, center_radec=center_ra_dec) 
#thr = 0.1
#mymask = (covmap > (np.max(covmap)*thr)).astype(int)
#pixok = mymask > 0
pixok = covmap > (np.max(covmap) * float(0))

#if you are simulating BI, def BI config and save this dict as your instrument
if sim_BI: 
    print('Configure BI')
    for i,nsub in enumerate(n_subbands):
        print('Define BI with {} sub-bands'.format(nsub))
        qp_nsub = np.array([1, 1, 1, nsub, nsub, nsub, nsub, nsub, nsub])
        #qp_nsub = np.array([nsub, nsub, nsub, nsub, nsub, nsub, nsub, nsub, nsub])
        qp_effective_fraction = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        #BI dict config
        qp_config = utilitylib.qubicify(s4_config, qp_nsub, qp_effective_fraction)
        #print(list(np.round(qp_config['frequency'],2)))
        #print(list(np.round(qp_config['depth_i'],2)))
        #print(list(np.round(qp_config['depth_p'],2)))
        #print(list(np.round(qp_config['fwhm'],2)))
        #print()
        print(qp_config['depth_p'])#*np.sqrt(2))
        instrum_config.append(qp_config)

num_of_instrum = len(instrum_config)
print('Number of instruments to simulate = ', num_of_instrum)

utilitylib.plot_all_s4_bi_config_sensitivity(instrum_config)

#---------------------------------------------------
print()
print('#### MAP-MAKING ####')

#def pysm3 sky
if 'pysm_fg' in skyconfig:
    preset_strings = skyconfig['pysm_fg']
    print('Pysm3 models : {}'.format(preset_strings))
                    
    #def sky model
    sky = pysm3.Sky(nside=nside, preset_strings=preset_strings)
    #set decorrelation if present
    if corr_l is not None:
        for m_idx, m in enumerate(preset_strings):
            if 'd' in m: #it's dust
                print('Setting correlation length = ', corr_l)
                sky.components[m_idx].correlation_length = corr_l*u.dimensionless_unscaled
    #if there are spatially varying params, then do a downsize to Nside_fit and then upsize to Nside again
    if Nside_fit != 0: 
        for m_idx, m in enumerate(preset_strings):
            if 'd' in m: #it's dust
                #fix dust temp
                if temp_is_fixed:
                    print('Fix T dust and skip pixelizing dust parameter')
                    sky.components[m_idx].mbb_temperature[:] = temp * sky.components[m_idx].mbb_temperature.unit
                else:
                    print('Skip pixelizing dust parameter')
                    #pass
                '''
                print('Pixelizing dust parameters')
                for spectral_param in [sky.components[m_idx].mbb_index, sky.components[m_idx].mbb_temperature]:
                    spectral_param[:] = hp.ud_grade(hp.ud_grade(spectral_param.value, Nside_fit), nside) * spectral_param.unit
                '''
            if 's' in m: #it's synch
                print('Skip pixelizing synch parameter')
                #print('Pixelizing synch parameter')
                #sky.components[m_idx].pl_index = hp.ud_grade(hp.ud_grade(sky.components[m_idx].pl_index.value, Nside_fit), nside) * sky.components[m_idx].pl_index.unit
else:
    print('No Foregrounds. Passing FG pre-setting.')
    sky = 1


# def. an instrument class starting from the sky components, coverage and instrument type

instrum_class = []
for i in range(num_of_instrum):
    print()
    print('Define instrument class for: ', instrum_config[i]['name'])
    instrum_class.append(qubicplus_skygen_lib.BImaps(skyconfig, sky, covmap, instrum_config[i], r, Alens, corr_l, nside=nside, spectra_type=spectra_type, lmin=lmin, delta_ell=delta_l, save_figs=save_figs)) 
    #skyconfig has only fg at this level

sky_maps = []
if any('d6' in model for model in skyconfig['pysm_fg']):
    print()
    for i in range(num_of_instrum):
        print('Creating null maps...')
        start_time = datetime.now()
        print('Doing ', instrum_config[i]['name'])
        
        sky_maps.append(np.zeros((len(instrum_class[i].nus), 3, instrum_class[i].npix)))
        print('Instrum sky maps have shape: ', sky_maps[i].shape)
        print('...generated maps in {}'.format(datetime.now()-start_time)) 
        print()

else:
    print()
    for i in range(num_of_instrum):
        print('Creating sky noiseless maps...')
        start_time = datetime.now()
        print('Doing ', instrum_config[i]['name'])

        instrum_sky_maps, _ = instrum_class[i].get_fg_maps(N_SAMPLE_BAND=N_sample_band, Nside_patch=Nside_fit)
        print('Instrum sky maps have shape: ', instrum_sky_maps.shape)
        sky_maps.append(instrum_sky_maps)
        '''
        #save sed
        print('SED evaluation and saving to file')
        sed = qubicplus_skygen_lib.eval_sed(instrum_sky_maps)
        pickle.dump([instrum_class[i].nus, sed],
                        open('./results/fg_SED_{}_corrl{}_nsidefit{}_Nbpintegr{}_callnumber{}.pkl'.format(instrum_config[i]['name'], corr_l, Nside_fit, N_sample_band, call_number), "wb"))
        '''
        print('...generated maps in {}'.format(datetime.now()-start_time))  
        print()

#---------------------------------------------------
print()
print('#### BINNING ####')
#define NaMaster object from mask and S4 binning

maskpix = np.zeros(npix)
maskpix[pixok] = 1

if save_figs:
    figure(figsize=(16,10))
    hp.mollview(maskpix, title='mask')
savefig('./Delta_maps/Mask_map_'+file_name+'_apo{}_nsidefit{}_Nbpintegr{}_iter{}.png'.format(apo_size_deg, Nside_fit, N_sample_band, N)) 

print('Apodization degrees : ', apo_size_deg)

Namaster = nam.Namaster(maskpix, lmin=lmin, lmax=lmax, delta_ell=delta_l, aposize=apo_size_deg)
ell_binned, _ = Namaster.get_binning(nside)
num_bin = len(ell_binned)
print('Number of NaMaster bins: ', num_bin)
print('Namaster fsky = ', Namaster.fsky)

if save_figs:
    figure(figsize=(16,10))
    hp.mollview(Namaster.mask_apo, title='mask apodized')
savefig('./Delta_maps/Mask_namaster_map_'+file_name+'_apo{}_nsidefit{}_Nbpintegr{}_iter{}.png'.format(apo_size_deg,  Nside_fit, N_sample_band, N)) 

#---------------------------------------------------
#compsep preprocessing
print()
print('#### COMPSEP ####')
comp = utilitylib.configure_fgb_component(skyconfig, nu0, temp_is_fixed) #sky component list: cmb, dust, synch etc.
n_comp = len(comp)
print('Number of fgb components = ', n_comp)

n_par = utilitylib.get_num_params(skyconfig, temp_is_fixed=temp_is_fixed)

fgb_instrum = []
for i in range(num_of_instrum):
    fgb_instrum.append(utilitylib.configure_fgb_instrument(instrum_config[i], N_sample_band)) #define instrument for fgb

#options={'maxiter': 100000} #additional comp.sep. options

# tol=1e-18

#---------------------------------------------------------------------------------------------------------------------
#for N realizations: gen cmb, add a noise realization and perform component sep
Dls = np.zeros((N, num_of_instrum, num_bin, 4))

if Nside_fit != 0: 
    print('Parameters vary across sky')
    param_maps = np.zeros( (N, num_of_instrum, 2, n_par, hp.nside2npix(Nside_fit)) )
else:
    print('Parameters are scalars')
    param_maps = np.zeros( (N, num_of_instrum, 2, n_par) )
'''
npixok = int(npix*fsky) # -1 
cmb_rec_maps = np.zeros( (N, num_of_instrum, 2, n_stk, npixok) )
cmb_input_maps = np.zeros( (N, num_of_instrum, 2, n_stk, npixok) )

#tot_residual_maps = np.zeros( (N, num_of_instrum, 2, n_stk, npixok) )
#tot_input_maps = np.zeros( (N, num_of_instrum, 2, n_stk, npixok) )
'''

#initialize cmb seed
if use_seed_file:
    print('Using cmb seeds from external file')
    seed_vec = pickle.load(open('./{}seeds_for_cmb.pkl'.format(N), "rb"))[0]
    
else:
    seed_vec = np.zeros(N, dtype=int)
    np.random.seed(None)

#start MC
start_time_iter = datetime.now()
for n in range(N):
    print('Iteration n. :', n+1)
    start_time_iter_n = datetime.now()

    #generate a cmb seed, unless one loads it from an external file
    if use_seed_file == False:
        seed_vec[n] = np.random.randint(1000000)
    if N == 1:
        seed_vec[n] = 42
    print()
    print('CMB seed = ', seed_vec[n])
    
    for i in range(num_of_instrum):
        print()
        print('Doing ', instrum_config[i]['name'])
        print()
        cmb_recon_map = np.zeros((2,3,npix))
        instrum_class[i].seed = seed_vec[n]
        
        #if sky is d6, generate fg realization and then add the cmb. Otherwise add directly the cmb realization.
        if any('d6' in model for model in skyconfig['pysm_fg']):
            instrum_class[i].skyconfig = {'cmb': seed_vec[n], 'pysm_fg': d['pysm_fg']} 
            
            #generate seeds for d6 case
            fg_bp_freqs = np.linspace(1, 320, 100000)
            fg_bp_seed = np.linspace(1, 320, len(fg_bp_freqs))*(n+1+(call_number-1)*N) #np.random.randint(1, 100000, len(allfreqs)) #np.linspace(1, 320, len(allfreqs))*n_ite
            
            skymap, _ = instrum_class[i].get_fg_maps_same_real(fg_bp_seed, fg_bp_freqs, N_SAMPLE_BAND=N_sample_band, Nside_patch=Nside_fit)
            sky_maps[i] = skymap.copy()
            print('Generated fg map with d6 realization.')
            print('Sky maps have shape: ', sky_maps[i].shape)
        else:
            print('Sky maps have shape: ', sky_maps[i].shape)
    
        #generate cmb realization
        instrum_class[i].skyconfig = {'cmb': seed_vec[n]}
        cmb_maps, _, _ = instrum_class[i].get_sky_maps(same_resol=0, verbose=True, coverage=False, noise=False)
        for j in range(2):
            #def input map: cmb, fg, noise
            input_map = sky_maps[i].copy() #fg maps
            input_map += cmb_maps
            #save map image
            if save_figs:
                figure(figsize=(16,10))
                for i_stk in range(n_stk):
                    hp.gnomview(input_map[0,i_stk+1,:], rot=center,
                                reso=15, title=instrum_config[i]['name']+': input map '+stk[i_stk],
                                norm='hist', sub=(1,n_stk,i_stk+1))
                savefig('./Delta_maps/Input_map_'+file_name+'_run{}_seed{}_{}_nsidefit{}_Nbpintegr{}_iter{}of{}.png'.format(j+1, seed_vec[n], instrum_config[i]['name'],  Nside_fit, N_sample_band, n+1, N)) 
            
                masked_maps = input_map[0].copy()
                masked_maps[:,~pixok] = hp.UNSEEN
                figure(figsize=(16,10))
                for i_stk in range(n_stk):
                    hp.gnomview(masked_maps[i_stk+1,:], rot=center,
                                reso=15, title=instrum_config[i]['name']+': input map '+stk[i_stk],
                                norm='hist', sub=(1,n_stk,i_stk+1))
                savefig('./Delta_maps/Input_maskedmap_'+file_name+'_run{}_seed{}_{}_nsidefit{}_Nbpintegr{}_iter{}of{}.png'.format(j+1, seed_vec[n], instrum_config[i]['name'],  Nside_fit, N_sample_band, n+1, N)) 
            
            #create noise realization
            if add_noise:
                print('Adding noise to the input maps...')
                noise_map = instrum_class[i].gen_noise_maps(coverage=False)
                input_map += noise_map
            #apply mask
            input_map[:,:,~pixok] = hp.UNSEEN
            
            #apply compsep only on the Q,U maps
            start_time = datetime.now()
            if Nside_fit != 0:
                data = input_map[:, 1:, :] #must take all pixels otherwise can't divide into patches
            else:
                data = input_map[:, 1:, pixok]
            result = fgbuster.basic_comp_sep(comp, fgb_instrum[i], data, nside=Nside_fit, tol=1e-18, method='TNC') #options=options, 
            print('Compsep duration: {}'.format(datetime.now()-start_time))
            
            #save reconstructed comps maps
            if Nside_fit != 0:
                recons_maps = result.s.copy() 
            else:
                recons_maps = np.zeros((n_comp, n_stk, npix))
                recons_maps[:,:,pixok] = result.s
            '''                
            cmb_rec_maps[n,i,j,:,:] = recons_maps[0,:,pixok].T
            cmb_input_maps[n,i,j,:,:] = cmb_maps[0,1:,pixok].T
            '''
            #save global residual map image
            if save_figs:
                residual_maps_baseline = np.zeros( (n_stk, npix) )
                residual_maps_baseline[:,~pixok] = hp.UNSEEN
                residual_maps_baseline[:, pixok] = recons_maps[0,:,pixok].T - cmb_maps[0,1:,pixok].T #tot_residual_maps[n,i,j,:,:]
                figure(figsize=(16,10))
                for i_stk in range(n_stk):
                    hp.gnomview(residual_maps_baseline[i_stk,:], rot=center,
                                reso=15, title=instrum_config[i]['name']+': CMB residual '+stk[i_stk],
                                norm='hist', sub=(1,n_stk,i_stk+1))
                savefig('./Delta_maps/Delta_cmb_map_'+file_name+'_run{}_apo{}_seed{}_{}_nsidefit{}_Nbpintegr{}_iter{}of{}.png'.format(j+1, apo_size_deg, seed_vec[n], instrum_config[i]['name'],  Nside_fit, N_sample_band, n+1, N))          
            
            #save params
            if n_par > 0 :
                print(result.x) # N, num_of_instrum, 2, n_par, npix/1
                if Nside_fit != 0: 
                    param_maps[n,i,j,:,:] = result.x
                else:
                    param_maps[n,i,j,:] = result.x
            
            
            #save reconstructed component maps for Dls estimation
            comp_map = recons_maps.copy() #result.s.copy()
            #convert cmb map to I,Q,U with I=0
            cmb_IQU = utilitylib.get_maps_for_namaster_QU(comp_map[0], npix)
            cmb_IQU[:,~pixok] = 0
            cmb_recon_map[j,:,:] = cmb_IQU.copy()
            '''
            map_test = input_map[0,:,:].copy()
            map_test[0,:] = 0
            map_test[:,~pixok] = 0
            cmb_recon_map[j,:,:] =map_test
            '''
            #save map image
            if save_figs:
                figure(figsize=(16,10))
                stk_full = ['I','Q','U']
                for i_stk in range(3):
                    hp.gnomview(cmb_recon_map[j,i_stk,:], rot=center,
                                reso=15, title=instrum_config[i]['name']+': CMB map for spectra '+stk_full[i_stk],
                                norm='hist', sub=(1,3,i_stk+1))
                savefig('./Delta_maps/Recon{}_cmb_map_'.format(j)+file_name+'_run{}_apo{}_seed{}_{}_nsidefit{}_Nbpintegr{}_iter{}of{}.png'.format(j+1, apo_size_deg, seed_vec[n], instrum_config[i]['name'],  Nside_fit, N_sample_band, n+1, N))  
            
        #get cross-spectra
        w = None
        leff, Dls[n,i,:,:], _ = Namaster.get_spectra(cmb_recon_map[0,:,:], map2=cmb_recon_map[1,:,:],
                                         purify_e=False,
                                         purify_b=True,
                                         w=w,
                                         verbose=False,
                                         beam_correction=None,
                                         pixwin_correction=False)

                                         
        print('ell shape: ', leff.shape)
        print('Estimated Dls:\n ', Dls[n,i,:,:])
        #print('Noise Dls: ', Dls_noise[n,:,:])
        
        # Save results in pkl file
        print('Saving results to file... \n')
        pickle.dump([leff, Dls[:,i,:,:], seed_vec], open('./results/cls_VaryingCMBseed_'+file_name+'_{}_nsidefit{}_Nbpintegr{}_iter{}.pkl'.format(instrum_config[i]['name'], Nside_fit, N_sample_band, N), "wb"))
        #pickle.dump([cmb_rec_maps[:,i,:,:,:]], open('./results/CMB_outmap_VaryingCMBseed_'+file_name+'_{}_nsidefit{}_Nbpintegr{}_iter{}.pkl'.format(instrum_config[i]['name'], Nside_fit, N_sample_band, N), "wb"))
        #pickle.dump([cmb_input_maps[:,i,:,:,:]], open('./results/CMB_inmap_VaryingCMBseed_'+file_name+'_{}_nsidefit{}_Nbpintegr{}_iter{}.pkl'.format(instrum_config[i]['name'], Nside_fit, N_sample_band, N), "wb"))
        pickle.dump([param_maps[:,i]], open('./results/params_VaryingCMBseed_'+file_name+'_{}_nsidefit{}_Nbpintegr{}_iter{}.pkl'.format(instrum_config[i]['name'], Nside_fit, N_sample_band, N), "wb"))
    print('Iteration duration: {}'.format(datetime.now()-start_time_iter_n))

print('Iteration number {0} is over. Total duration: {1}'.format(n+1, datetime.now()-start_time_iter))
print('Code ended!')
#-----------------------------------------------------------------------------------------------------------------

'''
    #plot Dls from all instruments compared with theo. one 
    DlsBB_theoretic_r = likelihoodlib.get_DlsBB_from_CAMB(leff[1:-1], r, Alens, nside=nside, coverage=covmap, cov_cut=0.1, lmin=lmin, delta_ell=delta_l, lmax = lmax) 
    print('Theor. Dls BB with r={}'.format(r))
    print(DlsBB_theoretic_r)
    #compare average Dls with the theoretical one
    figure(figsize=(10,10))
    plot(leff[1:-1], DlsBB_theoretic_r, label='Theo. BB spectrum')
    for i in range(num_of_instrum): 
        plot(leff[1:-1], Dls[n,i,1:-1,2], label='{}, sim. BB spectrum'.format(instrum_config[i]['name']))
    title_str='ite. {} of {},  integration points = {}, r = {}, Alens = {}, corr. l. = {}'.format(n+1, N, N_sample_band, r, Alens, corr_l)
    title(title_str, fontsize=9)
    xlabel('$\ell$')
    ylabel('$D_{{\ell}}$')
    legend(fontsize=12, loc='upper right')
    #yscale('log')
    savefig('./Cls_Likelihood_images/Sim_vs_theo_Dls_testnoise6_'+file_name+'_seed{}_{}_nsidefit{}_Nbpintegr{}_iter{}of{}.png'.format(seed_vec[n], instrum_config[i]['name'], Nside_fit, N_sample_band, n+1, N))
'''

'''
            #save input cmb map image
            figure(figsize=(16,10))
            for i_stk in range(n_stk):
                hp.gnomview(cmb_maps[freq_idx,1+i_stk,:], rot=center, reso=15, title=instrum_config[i]['name']+': Input CMB '+stk[i_stk], norm='hist', sub=(1,n_stk,i_stk+1))
            savefig('./Delta_maps/Input_CMB_map_testnoise6_'+file_name+'_seed{}_{}_nsidefit{}_Nbpintegr{}_iter{}of{}.png'.format(seed_vec[n], instrum_config[i]['name'],
                                                                                                                                 Nside_fit, N_sample_band, n+1, N))
            
            #save recon cmb map image
            figure(figsize=(16,10))
            for i_stk in range(n_stk):
                hp.gnomview(recons_maps[0,i_stk,:], rot=center, reso=15, title=instrum_config[i]['name']+': Recon CMB '+stk[i_stk], norm='hist', sub=(1,n_stk,i_stk+1))
            savefig('./Delta_maps/Recon_CMB_map_testnoise6_'+file_name+'_seed{}_{}_nsidefit{}_Nbpintegr{}_iter{}of{}.png'.format(seed_vec[n], instrum_config[i]['name'], 
                                                                                                                                 Nside_fit, N_sample_band, n+1, N))
            
                                                                                                                                 
            
            #save cmb residual map image
            delta_cmb_maps = recons_maps[0,:,:].copy()
            delta_cmb_maps -= cmb_maps[freq_idx,1:,:] #[n,i,j,0,:,:] for cmb take the residual
            delta_cmb_maps[:,~pixok] = hp.UNSEEN #[n,i,j,0,:,~pixok]
        
            figure(figsize=(16,10))
            for i_stk in range(n_stk):
                hp.gnomview(delta_cmb_maps[i_stk,:], rot=center, reso=15, title=instrum_config[i]['name']+': Delta CMB '+stk[i_stk], norm='hist', sub=(1,n_stk,i_stk+1))
            savefig('./Delta_maps/Delta_CMB_map_testnoise6_'+file_name+'_seed{}_{}_nsidefit{}_Nbpintegr{}_iter{}of{}.png'.format(seed_vec[n], instrum_config[i]['name'], Nside_fit, N_sample_band, n+1, N))
'''