import numpy as np
import pysm3
import pysm3.units as u
from pysm3 import utils
import healpy as hp
import fgbuster
from pylab import *
import os
import qubic
import warnings
import pickle
import sys
from datetime import datetime
from qubic import NamasterLib as nam
warnings.filterwarnings("ignore")
print(fgbuster.__path__)

import qubicplus_skygen_lib
import utilitylib


#FUNCTION DEFINITION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def _rj2cmb(freqs):
    return (np.ones_like(freqs) * u.K_RJ).to(
        u.K_CMB, equivalencies=u.cmb_equivalencies(freqs * u.GHz)).value


def _cmb2rj(freqs):
    return (np.ones_like(freqs) * u.K_CMB).to(
        u.K_RJ, equivalencies=u.cmb_equivalencies(freqs * u.GHz)).value

def _rj2jysr(freqs):
    return (np.ones_like(freqs) * u.K_RJ).to(
        u.Jy / u.sr, equivalencies=u.cmb_equivalencies(freqs * u.GHz)).value


def _jysr2rj(freqs):
    return (np.ones_like(freqs) * u.Jy / u.sr).to(
        u.K_RJ, equivalencies=u.cmb_equivalencies(freqs * u.GHz)).value


def _cmb2jysr(freqs):
    return (np.ones_like(freqs) * u.K_CMB).to(
        u.Jy / u.sr, equivalencies=u.cmb_equivalencies(freqs * u.GHz)).value


def _jysr2cmb(freqs):
    return (np.ones_like(freqs) * u.Jy / u.sr).to(
        u.K_CMB, equivalencies=u.cmb_equivalencies(freqs * u.GHz)).value


def get_instr(config, N_SAMPLE_BAND):
    
    freq_maps=config['frequency']
    
    if len(freq_maps) == 9:
        name='CMBS4'
        n_freq=9
    elif len(freq_maps)==33:
        name='QubicPlus'
        n_freq=33
    else:
        name='CMBS4BI'
        n_freq=42
    
    print('###################')
    print('Instrument is: {}'.format(name))
    print('Number of samples for bandpass integration: {}'.format(N_SAMPLE_BAND))
    print('###################')
    instrument = fgbuster.get_instrument(name)
    #instrument.frequency=freq_maps
    
    bandpasses = config['bandwidth'] 
    
    freq_maps_bp_integrated = np.zeros_like(freq_maps)
    new_list_of_freqs_flat = []
    new_list_of_freqs = []
    #freqs_init = instrument.frequency*1.0
    
    for f in range(freq_maps_bp_integrated.shape[0]):

        fmin = freq_maps[f]-bandpasses[f]/2
        fmax = freq_maps[f]+bandpasses[f]/2
        #### bandpass_frequencies = np.linspace(fmin, fmax, fsteps) * u.GHz
        freqs = np.linspace(fmin, fmax, N_SAMPLE_BAND)
        weights_flat = np.ones(N_SAMPLE_BAND)
        weights = weights_flat.copy()/ _jysr2rj(freqs)
        weights /= _rj2cmb(freqs)
        weights /= np.trapz(weights, freqs * 1e9)
        new_list_of_freqs.append((freqs, weights))

    instrument.frequency = new_list_of_freqs

    return instrument


def ParamCompSep(config, prop, nside, skyconfig, model, foreg_maps, coverage, iib, noise=True, fix_temp=None,
                 x0=[], break_width=1., fixsync=True, nu0=85, Nsample=100):

    covmap = coverage
    pixok = covmap>0
    

    if prop[0] == 1 : #s4
        conf=config[0]
        fgmaps = foreg_maps[0]
        print('Has fg_maps the same freq. length of the instrument? ', (len(conf['frequency'])==fgmaps.shape[0]))
        inputs, inputs_noiseless, _ = qubicplus_skygen_lib.BImaps(skyconfig, conf, nside=nside).getskymaps(fg_maps=fgmaps,
                                      same_resol=0,
                                      verbose=True,
                                      coverage=covmap,
                                      noise=True)

    elif prop[1] == 1: #bi
        conf=config[1]
        fgmaps = foreg_maps[1]
        print('Has fg_maps the same freq. length of the instrument? ', (len(conf['frequency'])==fgmaps.shape[0]))
        inputs, inputs_noiseless, _ = qubicplus_skygen_lib.BImaps(skyconfig, conf, nside=nside).getskymaps(fg_maps=fgmaps,
                                      same_resol=0,
                                      verbose=True,
                                      coverage=covmap,
                                      noise=True)
    '''
    else:
        inputs, inputs_noiseless, _ = qubicplus_skygen_lib.combinedmaps(skyconfig, config, nside=nside, prop=prop).getskymaps(
                                      same_resol=0,
                                      verbose=False,
                                      coverage=covmap,
                                      noise=True,
                                      beta=[1.54-delta_beta, 1.54+delta_beta, nubreak, break_width],
                                      fix_temp=fix_temp,
                                      iib=iib)'''




    ###### Preparation for COMPSEP ######
    nus, depth_i, depth_p, fwhm = utilitylib.get_good_config(config, prop)

    if len(nus) == 9:
        name='CMBS4'
        n_freq=9
    elif len(nus)==33:
        name='QubicPlus'
        n_freq=33
    else:
        name='CMBS4BI'
        n_freq=42

    print()
    print('Define instrument taking into account bandpass integration')
    #def instrument and instrument.frequency taking bandpass integration into account
    if iib==1:
        instr = fgbuster.get_instrument(name)
        instr.frequency=nus
        print('###################')
        print('Instrument is: {}'.format(name))
        print('Number of samples for bandpass integration: {}'.format(iib))
        print('###################')
    elif iib>1:
        instr = get_instr(conf, Nsample)
    else:
        print('ERROR: Set correct number of freq. for bandpass integration! At least 1')

    #set depths and fwhm equal to the corresponding configuration
    instr.depth_i=depth_i
    instr.depth_p=depth_p
    instr.fwhm=fwhm

    print()
    print('Define components')
    comp = utilitylib.get_comp_for_fgb(nu0=nu0, model=model, fix_temp=fix_temp, x0=x0, bw=break_width, fixsync=fixsync)

    print()
    print('##### COMP SEP #####')

    options={'maxiter': 100000}
    #tol=1e-18
    
    #apply compsep only on the Q,U maps
    data = inputs[:, 1:, pixok] #take only pixok to speed up the code

    #cov=utilitylib.get_cov_for_weighted(len(nus), depth_i, depth_p, covmap, nside=nside)

    r=fgbuster.basic_comp_sep(comp, instr, data, options=options, tol=1e-18, method='TNC')#, bounds=bnds)
    #r=fgbuster.weighted_comp_sep(comp, instr, data, cov, options=options, tol=1e-18, method='TNC')

    print('\nMessage -> ', r.message)
    print('# of function evaluation -> ', r.nfev)
    print()

    print('...compsep done! \n')
    print('Estimated params: ', r.x)

    components = utilitylib.get_comp_from_MixingMatrix(r, comp, instr, inputs[:, 1:, :], nus, covmap, noise, nside)
    components_noiseless = utilitylib.get_comp_from_MixingMatrix(r, comp, instr, inputs_noiseless[:, 1:, :], nus, covmap, False, nside)
    return components, components_noiseless, r
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~CODE PART~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
center_ra_dec = [-30,-30] #CMB-S4 sky patch
center = qubic.equ2gal(center_ra_dec[0], center_ra_dec[1])
print('Sky patch is centered at RA,DEC = ', center_ra_dec)
print('WARNING: if you want to change the center of the sky patch, stop the code now!')

fsky=0.03
nside=256

#def CMB-S4 config
freqs = np.array([20., 30., 40., 85., 95., 145., 155., 220., 270.])
bandwidth = np.array([5., 9., 12., 20.4, 22.8, 31.9, 34.1, 48.4, 59.4])
dnu_nu = bandwidth/freqs
beam_fwhm = np.array([11., 72.8, 72.8, 25.5, 25.5, 22.7, 22.7, 13., 13.])
mukarcmin_TT = np.array([16.5, 9.36, 11.85, 2.02, 1.78, 3.89, 4.16, 10.15, 17.4])
mukarcmin_EE = np.array([10.87, 6.2, 7.85, 1.34, 1.18, 1.8, 1.93, 4.71, 8.08])
mukarcmin_BB = np.array([10.23, 5.85, 7.4, 1.27, 1.12, 1.76, 1.89, 4.6, 7.89])
ell_min = np.array([30, 30, 30, 30, 30, 30, 30, 30, 30])
Nside = np.array([512, 512, 512, 512, 512, 512, 512, 512, 512])
edges_min = freqs * (1. - dnu_nu/2)
edges_max = freqs * (1. + dnu_nu/2)
edges = [[edges_min[i], edges_max[i]] for i in range(len(freqs))]
#CMB-S4 config
s4_config = {
    'nbands': len(freqs),
    'frequency': freqs,
    'depth_p': 0.5*(mukarcmin_EE + mukarcmin_BB),
    'depth_i': mukarcmin_TT,
    'depth_e': mukarcmin_EE,
    'depth_b': mukarcmin_BB,
    'fwhm': beam_fwhm,
    'bandwidth': bandwidth,
    'dnu_nu': dnu_nu,
    'ell_min': ell_min,
    'nside': Nside,
    'fsky': fsky,
    'ntubes': 12,
    'nyears': 7.,
    'edges': edges,
    'effective_fraction': np.zeros(len(freqs))+1.
            }

#define the pixok set
covmap = utilitylib.get_coverage(s4_config['fsky'], nside, center_radec=center_ra_dec) 
thr = 0
mymask = (covmap > (np.max(covmap)*thr)).astype(int)
pixok = mymask > 0

#def BI config
qp_nsub = np.array([1, 1, 1, 5, 5, 5, 5, 5, 5])
qp_effective_fraction = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
qp_config = utilitylib.qubicify(s4_config, qp_nsub, qp_effective_fraction)

#def sky and instrument model
nb_param=1
noise=True

nu0=85 #def nu0 value
print('WARNING: nu0 is fixed at nu0 = {} GHz. If you want to change it, stop the code now!'.format(nu0))

#variables from command line~~~~~~
N=int(sys.argv[1])           # Number of iterations
print('Total number of iterations: ', N)
ite=int(sys.argv[2])         # Number of times this code is executed simultaneously
print('Ite value = ', ite)
nubreak=int(sys.argv[3])     # True value of nubreak
iib=int(sys.argv[6]) #number of freqs to use in the band integration
additional_info = str(sys.argv[7]) #contains any additional info for the filename

if int(sys.argv[4]) == 1:
    T=20
    name_T='_fixtemp'
else:
    T=None
    name_T=''
    nb_param+=1

if int(sys.argv[5]) == 1:
    fix_sync=True
    name_s='_fixsync'
else:
    fix_sync=False
    name_s=''
    nb_param+=1

N_sample_band=iib

print('T = {}'.format(T))
print('Fix sync -> {}'.format(fix_sync))
print('# of params -> {}'.format(nb_param))
print('N_sample_band = ', N_sample_band)
fix_temp=T
fix_sync=fix_sync
#~~~~~~~~~~~~~~~~~~~~~
'''
N=2       # Number of iterations
ite=1         # To save
nubreak=150 # True value of nubreak
iib=1
temp_conf = 1
synch_conf = 1

if temp_conf == 1:
    T=20
    name_T='_fixtemp'
else:
    T=None
    name_T=''
    nb_param+=1
    
if synch_conf == 1:
    fix_sync=True
    name_s='_fixsync'
else:
    fix_sync=False
    name_s=''
    nb_param+=1
    
print('T = {}'.format(T))
print('Fix sync -> {}'.format(fix_sync))
print('# of params -> {}'.format(nb_param))
fix_temp=T
fix_sync=fix_sync
'''

model='1b' #fitting dust model
dust_config='d02b' #dust model for map creation
bw=0.3 #steepness of the frequency break in the 2-beta dust model

npix=hp.nside2npix(nside)

maskpix = np.zeros(npix)
maskpix[pixok] = 1

lmin=21
lmax=355
dl=35

Namaster = nam.Namaster(maskpix, lmin=lmin, lmax=lmax, delta_ell=dl)
ell_binned, _=Namaster.get_binning(nside)
num_bin = len(ell_binned)
print('Number of Namaster bins: ', num_bin)

props=[0,1] #instrumental configurations
db=[-0.1, -0.05, -0.03, 0.0, 0.03, 0.05, 0.1] #np.linspace(0.0, 0.05, 6) #[0.0, 0.05] #delta beta cases
print('###################')
print('BEWARE: you are simulating... props = {0}, db = {1}'.format(props, db))
print('###################')
print()

#def fg maps
nus_s4=len(s4_config['frequency'])
nus_bi=len(qp_config['frequency'])

fg_maps_s4 = np.empty((len(db),nus_s4,3,npix))
fg_maps_bi = np.empty((len(db),nus_bi,3,npix))

'''
#define number of freq to use for bi and s4
if iib==1: #this part is not physically correct, but returns the correct result
    iib_bi=iib
    iib_s4=iib
else: #the number of s4 freqs is equal to the number of bi freqs in the overall PHYSICAl band
    n_sub_bands_bi=qp_nsub[-1]
    iib_bi=iib
    iib_s4=iib_bi*n_sub_bands_bi
'''

#fill in the fg maps
print('Creating foreground maps...')
start_time_fg = datetime.now()

for k, delta_beta in enumerate(db):
    print('###################')
    print(' Delta beta : {}'.format(db[k]))
    print('###################')

    if delta_beta==0.0:
        skyconfig={'dust':'d0', 'synchrotron':'s0'}
    else:
        skyconfig={'dust':'d02b', 'synchrotron':'s0'}

    for conf in props:
        if conf==0: #s4
            start_time = datetime.now()
            print()
            print('S4')
            fg_maps_s4[k,:,:,:] = qubicplus_skygen_lib.BImaps(skyconfig, s4_config, nside=nside).get_fg_freq_integrated_maps(
                verbose=True, N_SAMPLE_BAND=N_sample_band, beta=[1.54-delta_beta, 1.54, nubreak, bw], temp=20)
            print('Duration: {}'.format(datetime.now()-start_time))
        elif conf==1: #bi
            start_time = datetime.now()
            print()
            print('BI')
            #for bi below take iib=iib_bi
            fg_maps_bi[k,:,:,:] = qubicplus_skygen_lib.BImaps(skyconfig, qp_config, nside=nside).get_fg_freq_integrated_maps(
                verbose=True, N_SAMPLE_BAND=N_sample_band, beta=[1.54-delta_beta, 1.54, nubreak, bw], temp=20)
            print('Duration: {}'.format(datetime.now()-start_time))
    
            #set first 3 frequency maps for bi equal to s4 ones
            #fg_maps_bi[k,0:3,:,:]=fg_maps_s4[k,0:3,:,:]
        else:
            print('ERROR: Hybrid configurations have not been implemented yet! Exit code now!')
            sys.exit()

print('...generated maps in {}'.format(datetime.now()-start_time_fg))


#def arrays to store results
param_comb = np.zeros(((((2, N, len(props), len(db), nb_param)))))
rms_est_cmb = np.zeros((((N, len(props), len(db), 1))))
rms_est_dust = np.zeros((((N, len(props), len(db), 1))))
cl=np.zeros(((((N, len(props), len(db), num_bin, 4)))))
tabseed=np.zeros(N)

#gen cmb seed
seed=42 #np.random.randint(1000000)

#start iteration: generate cmb and noise, then do compsep
print('Entering the iteration loop...')
for j in range(N):
    start_time_iter = datetime.now()
    print('Start iteration number {}'.format(j+1))
    
    #store cmb seed
    tabseed[j]=seed
    print("seed is {}".format(seed))

    for i in range(len(props)):
        for k in range(len(db)):
            start_time_db = datetime.now()
            #set initial guess values
            if fix_temp is not None:
                x0=[1.54]
            else:
                x0=[1.54, 20]
                
            print('###################')
            print(' BI fration : {}%'.format(props[i]*100))
            print(' Delta beta : {}'.format(db[k]))
            print(' Init : ', x0)
            print('###################')
            
            #set instrument configuration
            BIprop=props[i]
            S4prop=1-BIprop
            frac=[S4prop, BIprop]
            foreground_map=[fg_maps_s4[k,:,:,:], fg_maps_bi[k,:,:,:]]

            comp1, comp1_noiseless, r_comb_1 = ParamCompSep([s4_config, qp_config],
                                    prop=frac,
                                    skyconfig={'cmb':seed},
                                    model=model,
                                    foreg_maps=foreground_map,
                                    coverage=covmap,
                                    iib=iib,
                                    noise=noise,
                                    fix_temp=fix_temp,
                                    nside=nside,
                                    x0=x0,
                                    break_width=bw,
                                    fixsync=fix_sync,
                                    nu0=nu0, 
                                    Nsample=N_sample_band)

            comp2, comp2_noiseless, r_comb_2 = ParamCompSep([s4_config, qp_config],
                                    prop=frac,
                                    skyconfig={'cmb':seed},
                                    model=model,
                                    foreg_maps=foreground_map,
                                    coverage=covmap,
                                    iib=iib,
                                    noise=noise,
                                    fix_temp=fix_temp,
                                    nside=nside,
                                    x0=x0,
                                    break_width=bw,
                                    fixsync=fix_sync,
                                    nu0=nu0, 
                                    Nsample=N_sample_band)
            
            #store estimated params
            param_comb[0, j, i, k]=r_comb_1.x
            param_comb[1, j, i, k]=r_comb_2.x

            #store reconstructed cmb map. Place QU stokes parameters and 0 for I
            maps1 = utilitylib.get_maps_for_namaster_QU(comp1, nside=nside)
            maps2 = utilitylib.get_maps_for_namaster_QU(comp2, nside=nside)

            # To be sure that not seen pixels are 0 and not hp.UNSEEN
            maps1[:, ~pixok]=0
            maps2[:, ~pixok]=0
            
            #eval and store cls
            w=None
            leff, cl[j, i, k], _ = Namaster.get_spectra(maps1, map2=maps2,
                                         purify_e=False,
                                         purify_b=True,
                                         w=w,
                                         verbose=False,
                                         beam_correction=None,
                                         pixwin_correction=False)

            print('Estimated cross-Cl BB: ', cl[j, i, k, :, 2])
            # Save results in pkl file
            print('Saving results to file... \n')
            pickle.dump([leff, cl, param_comb, tabseed, db, props, bw, sys.argv], open('./results/cls{}_iib{:.0f}_QU{}{}_truenub{:.0f}_{}reals_{}.pkl'.format(additional_info, iib, name_T, name_s, nubreak, N, ite), "wb"))
            print('Duration: {}'.format(datetime.now()-start_time_db))
            
    print('Iteration number {0} is over. Duration: {1}'.format(j+1, datetime.now()-start_time_iter))
print('Code ended!')
