# -*- coding: utf-8 -*-
from pylab import *
import pysm3
import pysm3.units as u
import numpy as np
import numpy.ma as ma
import healpy as hp
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pickle
from astropy.io import fits
import pandas as pd
from importlib import reload

from qubic import camb_interface as qc
import fgbuster as fgb

import warnings
warnings.filterwarnings("ignore")

import qubic
from qubic import NamasterLib as nam

rc('figure', figsize=(16, 10))
rc('font', size=15)


# ############ Notebook for COmponent Separation using FGBuster
# ### Function mostly written by Mathias Régnier and Simon Biquard (2021)
# ### And then a bit adapted by JCH (sept. 2021)
# ### This is not intended to be the final library on this... just some playground...

stk = ['I', 'Q', 'U']

### Some usefull functions (see FGB-FullSky-NoNoise.Rmd)
def separate(comp, instr, maps_to_separate, tol=1e-12, print_option=True):
    solver_options = {}
    solver_options['disp'] = print_option
    fg_args = comp, instr, maps_to_separate
    fg_kwargs = {'method': 'BFGS', 'tol': 1e-12, 'options': solver_options}
    try:
        res = fgb.basic_comp_sep(*fg_args, **fg_kwargs)
    except KeyError:
        fg_kwargs['options']['disp'] = False
        res = fgb.basic_comp_sep(*fg_args, **fg_kwargs)
    if print_option:
        print()
        print("message:", res.message)
        print("success:", res.success)
        print("result:", res.x)
    return res


def reconvolve(maps, fwhms, ref_fwhm, verbose=False):
    if verbose: 
        print('Reconvolution to common FWHM')
    sig_conv = np.sqrt(ref_fwhm**2 - fwhms**2)
    maps_out = np.zeros_like(maps)
    for i in range(len(fwhms)):
        if sig_conv[i] == 0:
            if verbose:
                print('Map {0:} fwhmin={1:6.3f} fwhmout={2:6.3f} => We do not reconvolve'.format(i, 
                                                                                             fwhms[i], ref_fwhm))
            maps_out[i,:] = maps[i,:]
        else:
            if verbose:
                print('Map {0:} fwhmin={1:6.3f} fwhmout={2:6.3f} => We reconvolve with {3:6.3f}'.format(i, 
                                                                                                    fwhms[i], 
                                                                                                    ref_fwhm, 
                                                                                                    sig_conv[i]))
            maps_out[i,:] = hp.smoothing(maps[i,:], fwhm=np.deg2rad(sig_conv[i]), pol=True, verbose=False)
    return maps_out

# this function is from Simon Biquard
def get_alm_maps(pixel_maps, fwhms, resol_correction=False, ref_fwhm=0, pixwin_correction=False, verbose=False):
    """
    Compute alm maps from pixel maps and format them for FgBuster.
    """
    sh = np.shape(pixel_maps)
    nside = hp.npix2nside(sh[2])
    n = sh[0]
    lmax = 2*nside+1
    ell = np.arange(start=0, stop= lmax+1)

    ref_sigma_rad = np.deg2rad(ref_fwhm) / 2.355
    #ref_fl = np.exp(- 0.5 * np.square(ref_sigma_rad * ell))
    ref_fl = hp.gauss_beam(np.deg2rad(ref_fwhm), lmax=lmax)
    
    if verbose: 
        print('In get_alm_maps: FWHM = ', fwhms)
    beam_sigmas_rad = np.deg2rad(fwhms) / (2*np.sqrt(2*np.log(2)))
    pixwin = hp.pixwin(nside, lmax=lmax) if pixwin_correction else np.ones(lmax + 1)

    # compute maps
    #figure()
    alm_maps = None
    for f in range(n):
        alms = hp.map2alm(pixel_maps[f], lmax=lmax, pol=True)
        correction = None
        if f == 0:
            sh = np.shape(alms)
            alm_maps = np.empty((n, sh[0], 2 * sh[1]))
        if resol_correction:
            print('Applying Resol Correction')
            #gauss_fl = np.exp(- 0.5 * np.square(beam_sigmas_rad[f] * ell))
            gauss_fl = hp.gauss_beam(np.deg2rad(fwhms[f]), lmax=lmax)
            correction = ref_fl / gauss_fl / pixwin
            #plot(correction, label='freq {}'.format(f))
        else:
            print('No Resol Correction applied')
        for i, t in enumerate(alms):
            alm_maps[f, i] = format_alms(hp.almxfl(t, correction) if resol_correction else t)
    #legend()
    #title('Bl ratio in get_alm_maps')
    return alm_maps

# credits to J. Errard for these two functions
def intersect_mask(maps):
    if hp.pixelfunc.is_ma(maps):
        mask = maps.mask
    else:
        mask = maps == hp.UNSEEN

    # Mask entire pixel if any of the frequencies in the pixel is masked
    return np.any(mask, axis=tuple(range(maps.ndim - 1)))

def format_alms(alms, lmin=0, nulling_option=True):
    lmax = hp.Alm.getlmax(alms.shape[-1])
    alms = np.asarray(alms, order='C')
    alms = alms.view(np.float64)
    em = hp.Alm.getlm(lmax)[1]
    em = np.stack((em, em), axis=-1).reshape(-1)
    mask_em = [m != 0 for m in em]
    #alms[..., mask_em] *= np.sqrt(2)   ## Commented by JCH following Slack discussion with Josquin 14/09/2021
    if nulling_option:
        alms[..., np.arange(1, lmax + 1, 2)] = hp.UNSEEN  # mask imaginary m = 0
        mask_alms = intersect_mask(alms)
        alms[..., mask_alms] = 0  # thus no contribution to the spectral likelihood
    alms = np.swapaxes(alms, 0, -1)
    if lmin != 0:
        ell = hp.Alm.getlm(lmax)[0]
        ell = np.stack((ell, ell), axis=-1).reshape(-1)
        mask_lmin = [ll < lmin for ll in ell]
        if nulling_option:
            alms[mask_lmin, ...] = hp.UNSEEN
    return alms

def convolve_maps(inmaps, fwhms):
    sh = np.shape(inmaps)
    if len(sh)==2:
        maps = np.reshape(inmaps, (1,sh[0], sh[1]))
    else:
        maps = inmaps
    all_fwhms = np.zeros(sh[0]) + fwhms
    maps_conv = np.array([hp.smoothing(m.copy(), fwhm=np.deg2rad(fw), pol=True, verbose=False) for m,fw in zip(maps, all_fwhms)])
    if len(sh)==2:
        maps_conv = np.reshape(maps_conv, sh)
    return maps_conv


def display_maps(inmaps, bigtitle=None, mytitle='', figsize=(16,10), nsig=3, 
                 rot=None, reso=15, moll=False, add_rms=False, force_rng=None, unseen=None, freqs=None):
    rc('figure', figsize=figsize)
    figure()
    if bigtitle is not None:
        suptitle(bigtitle, fontsize=30, y=1.05)
    sh = np.shape(inmaps)
    if len(sh)==2:
        maps = np.reshape(inmaps.copy(), (1,sh[0], sh[1]))
    else:
        maps = inmaps.copy()
    if unseen is not None:
        maps[:,:,unseen] = hp.UNSEEN
    nf = maps.shape[0]
    nstk = maps.shape[1]
    mypixok = (maps[0,0,:] !=hp.UNSEEN) & (maps[0,0,:] !=0)
    for i in range(nf):
        for j in range(nstk):
            ss = np.std(maps[0,j,mypixok])
            if freqs is None:
                nuprint = i
            else:
                nuprint = freqs[i]
            thetitle = mytitle+' {} nu={:5.1f}'.format(stk[j], nuprint)
            if force_rng is None:
                mini = -nsig*ss
                maxi = nsig*ss
            else:
                mini = -force_rng[j]
                maxi = force_rng[j]
            if add_rms:
                thetitle += ' RMS={0:5.2g}'.format(ss)
            if moll:
                hp.mollview(maps[i,j,:], sub=(nf,3,3*i+j+1), min = mini, max=maxi,
                       title=thetitle)
            else:
                hp.gnomview(maps[i,j,:], sub=(nf,3,3*i+j+1), min = mini, max=maxi,
                       title=thetitle, rot=rot, reso=reso)
    tight_layout()        



def apply_fgb(inmaps, freqs, fwhms, verbose=True, 
              apodize=0, plot_apo=False, apocut=False, apotype='C1',
              coverage_recut=None, coverage=None,
              resol_correction=True, ref_fwhm=0.5,
              alm_space=False,
              plot_separated=False, center=None, add_title='',
              plot_residuals=False, truth=None, alm_maps=False, apply_to_unconvolved=False):
    ### FGB Configuration
    instrument = fgb.get_instrument('Qubic')
    instrument.frequency = freqs
    instrument.fwhm = fwhms
    components = [fgb.Dust(150., temp=20.), fgb.CMB()]

    ### Check good pixels
    pixok = inmaps[0,0,:] != hp.UNSEEN
    nside = hp.npix2nside(len(inmaps[0,0,:]))

    if apodize != 0:
        mymask = pixok.astype(float)
        nmt = nam.Namaster(mymask, 40, 400, 30, aposize=apodize, apotype=apotype)
        apodized_mask = nmt.get_apodized_mask()
        if plot_apo: 
            hp.gnomview(apodized_mask, title='Apodized Mask {} deg.'.format(apodize), reso=15, rot=center)
        maps = inmaps * apodized_mask
        maps[:,:,~pixok] = hp.UNSEEN
    else:
        maps = inmaps.copy()
        apodized_mask = np.ones(12*nside**2)
        

    ### Data to feed FGB:
    if alm_space:
        if verbose:
            print('\nFBG in alm-space with resol_correction={} and ref_resol={}'.format(resol_correction, ref_fwhm))
        mydata = get_alm_maps(maps, fwhms, 
                              ref_fwhm=ref_fwhm, resol_correction=resol_correction, 
                              verbose=verbose)

        if (truth is not None) & ~apply_to_unconvolved:
            if verbose:
                print('\nNow reconvolving truth to ref_fwhm = {}'.format(ref_fwhm))
            mytruth = [hp.smoothing(truth[0], fwhm=np.deg2rad(ref_fwhm), pol=True, verbose=False), 
                       hp.smoothing(truth[1], fwhm=np.deg2rad(ref_fwhm), pol=True, verbose=False)]
        space = ' (alm based)'
    else:
        if verbose:
            print('\nFBG in pixel-space with resol_correction={} and ref_resol={}'.format(resol_correction, ref_fwhm))

        if resol_correction:
            if verbose:
                print('\nNow reconvolving input maps to ref_fwhm = {}'.format(ref_fwhm))
            mydata = reconvolve(maps, fwhms, ref_fwhm, verbose=verbose)

            if (truth is not None):
                if verbose:
                    print('\nNow reconvolving truth (dust and CMB) to ref_fwhm = {}'.format(ref_fwhm))
                mytruth = [hp.smoothing(truth[0], fwhm=np.deg2rad(ref_fwhm), pol=True, verbose=False), 
                           hp.smoothing(truth[1], fwhm=np.deg2rad(ref_fwhm), pol=True, verbose=False)]
            space = ' (Pixel based - Reconv.)'
        else:
            mydata = maps
            if (truth is not None):
                mytruth = truth.copy()
            space = ' (Pixel based - No Reconv.)'

        if coverage_recut is not None:
            if verbose:
                print('Applying coverage recut to {}'.format(coverage_recut))
            fidregion = (coverage > (coverage_recut*np.max(coverage)))
            mydata[...,~fidregion] = hp.UNSEEN
            mapregions = np.zeros(12*nside**2) + hp.UNSEEN
            mapregions[pixok] = 1
            mapregions[fidregion] = 2
            if verbose:
                hp.gnomview(mapregions, rot=center, reso=15, title='Fiducial region: {}'.format(coverage_recut))
                show()

        if (apodize !=0) & (apocut==True):
            fidregion = apodized_mask == 1
            mydata[...,~fidregion] = hp.UNSEEN
            
                 
    ### FGB itself
    if verbose: print('Starting FGBuster in s')
    r = separate(components, instrument, mydata, print_option=verbose)
    if verbose: print('Resulting beta: {}'.format(r.x[0]))

        
    ### Resulting separated maps
    if alm_space:
        if alm_maps:
            ### Directly use the output from FGB => not very accurate because of alm-transform near the edges of the patch
            almdustrec = r.s[0,:,:]
            print('ALM', np.shape(almdustrec))
            dustrec = hp.alm2map(almdustrec[..., ::2] + almdustrec[..., 1::2]*1j, nside)
            dustrec[:,~pixok] = hp.UNSEEN
            almcmbrec = r.s[1,:,:]
            cmbrec = hp.alm2map(almcmbrec[..., ::2] + almcmbrec[..., 1::2]*1j, nside)
            cmbrec[:,~pixok] = hp.UNSEEN
        else:
            ### Instead we use the fitted beta and recalculate the component maps in pixel space
            ### This avoids inaccuracy of the alm transformation near the edges of the patch
            A = fgb.MixingMatrix(*components)
            print('          >>> building s = Wd in pixel space')
            A_ev = A.evaluator(instrument.frequency)
            A_maxL = A_ev(r.x)
            print('A_maxl', np.shape(A_maxL))
    
            if apply_to_unconvolved:
                # We apply the mixing matrix to maps each at its own resolution... 
                # this allows to keep the effecgive resolution as good as possible, but surely the bell is no
                # longer Gaussian. It is the combnation of various resolutions with weird weights.
                themaps = maps.copy()
                # Now we calculate the effective Bell
                Bl_each = []
                for fw in fwhms:
                    Bl_gauss_fwhm = hp.gauss_beam( np.radians(fw), lmax=2*nside+1)
                    Bl_each.append( Bl_gauss_fwhm )
                    plot(Bl_gauss_fwhm, label='FWHM {:4.2}'.format(fw))
                Bl_each = np.array(Bl_each)
                ### Compute W matrix (from Josquin)
                invN = np.diag(np.ones(len(fwhms)))
                inv_AtNA = np.linalg.inv(A_maxL.T.dot(invN).dot(A_maxL))
                W = inv_AtNA.dot( A_maxL.T ).dot(invN)
                Bl_eff_Dust =  W.dot(Bl_each)[0]    #1 is for the Dust
                Bl_eff_CMB =  W.dot(Bl_each)[1]    #1 is for the CMB component

                # print('W matrix')
                # print(W)
                # print('A_maxL matrix')
                # print(A_maxL)
                # print('Bl_each:',np.shape(Bl_each))

                Bl_eff_Dust_new = np.zeros(2*nside+1)
                Bl_eff_CMB_new = np.zeros(2*nside+1)
                ones_comp = np.ones(2) #number of components
                for i in range(2*nside+1):
                    blunknown = W.dot(Bl_each[:,i] * np.dot(A_maxL,ones_comp))
                    Bl_eff_Dust_new[i] = blunknown[0]
                    Bl_eff_CMB_new[i] = blunknown[1]


                plot(Bl_eff_CMB, ':', label='Effective Bl CMB: W.Bl', lw=3)
                plot(Bl_eff_Dust, ':', label='Effective Bl Dust: W.Bl', lw=3)
                plot(Bl_eff_CMB_new, '--', label='Effective Bl CMB NEW: W.B.A.1', lw=3)
                plot(Bl_eff_Dust_new, '--', label='Effective Bl Dust NEW: W.B.A.1', lw=3)
                legend()
                # We need to smooth the truth with this Bell
                if (truth is not None):
                    if verbose:
                        print('\nNow reconvolving truth (dust and CMB) with effective Bell')
                    mytruth = [hp.smoothing(truth[0], beam_window=Bl_eff_Dust_new, pol=True, verbose=False), 
                               hp.smoothing(truth[1], beam_window=Bl_eff_CMB_new, pol=True, verbose=False)]
            else:
                # We will apply the mixing matrix to maps at a chosen reeference resolution
                # this might not bee the most optimal although it is simple in the sens that the components maps resolution
                # is what we have chosen and is gaussian.
                themaps = reconvolve(maps, fwhms, ref_fwhm, verbose=verbose)
                components_bell = hp.gauss_beam( np.radians(ref_fwhm), lmax=2*nside+1)
            themaps[themaps == hp.UNSEEN] = 0
            ### Needed to comment the two lines below as prewhiten_factors was None
            #prewhiten_factors = fgb.separation_recipes._get_prewhiten_factors(instrument, themaps.shape, nside)
            #invN = np.zeros(prewhiten_factors.shape+prewhiten_factors.shape[-1:])
            r.s = fgb.algebra.Wd(A_maxL, themaps.T)#, invN=invN)     
            r.s = np.swapaxes(r.s,-1,0)
            dustrec = r.s[0,:,:]
            dustrec[:,~pixok] = hp.UNSEEN
            cmbrec = r.s[1,:,:]        
            cmbrec[:,~pixok] = hp.UNSEEN
    else:
        dustrec = r.s[0,:,:]
        cmbrec = r.s[1,:,:]
            
    if plot_separated:
        display_maps(dustrec, bigtitle=r'$\beta=${0:7.6f} - Reconstructed Dust'.format(r.x[0])+space, rot=center, figsize=(16, 7))
        display_maps(cmbrec, bigtitle=r'$\beta=${0:7.6f} - Reconstructed CMB'.format(r.x[0])+space, rot=center, figsize=(16, 7))

    if truth:
        resid_dust = dustrec - mytruth[0] * apodized_mask
        resid_dust[:,~pixok] = hp.UNSEEN
        resid_cmb = cmbrec - mytruth[1] * apodized_mask
        resid_cmb[:,~pixok] = hp.UNSEEN
        if coverage_recut:
            resid_cmb[:,~fidregion] = hp.UNSEEN
            resid_dust[:,~fidregion] = hp.UNSEEN
            pixok = fidregion.copy()
        if (apodize != 0) & (apocut==True):
            resid_cmb[:,~fidregion] = hp.UNSEEN
            resid_dust[:,~fidregion] = hp.UNSEEN
            pixok = fidregion.copy()
        sigs_dust = np.std(resid_dust[:, pixok], axis=1)
        sigs_cmb = np.std(resid_cmb[:, pixok], axis=1)
        if plot_residuals:
#             display_maps(mytruth[0], bigtitle=r'$\beta=${0:7.6f} Input Dust'.format(r.x[0])+space, 
#                          rot=center, figsize=(16, 7), add_rms=True)
#             display_maps(mytruth[1], bigtitle=r'$\beta=${0:7.6f} - Input CMB'.format(r.x[0])+space, 
#                          rot=center, figsize=(16, 7), add_rms=True)

            display_maps(mytruth[0], bigtitle='Truth Dust Reconvolved', 
                         rot=center, figsize=(16, 7), add_rms=True, unseen=~pixok)
            display_maps(mytruth[1], bigtitle='Truth CMB Reconvolved', 
                         rot=center, figsize=(16, 7), add_rms=True, unseen=~pixok)

            display_maps(resid_dust, bigtitle=r'$\beta=${0:7.6f} Residuals Dust'.format(r.x[0])+space, 
                         rot=center, figsize=(16, 7), add_rms=True)
            display_maps(resid_cmb, bigtitle=r'$\beta=${0:7.6f} - Residuals CMB'.format(r.x[0])+space, 
                         rot=center, figsize=(16, 7), add_rms=True)
            
            figure()
            suptitle(r'$\beta=${0:7.6f} - Residuals:'.format(r.x[0])+space, fontsize=30, y=1.05)
            for i in range(3):
                subplot(1,3,i+1)
                hist(resid_dust[i, pixok], range=[-5*sigs_dust[i], 5*sigs_dust[i]], 
                     bins=100, alpha=0.5, color='b', label='Dust: RMS={:4.2g}'.format(sigs_dust[i]), density=True)
                hist(resid_cmb[i, pixok], range=[-5*sigs_cmb[i], 5*sigs_cmb[i]], 
                    bins=100, alpha=0.5, color='r', label='CMB: RMS={:4.2g}'.format(sigs_cmb[i]), density=True)
                title('Residuals Stokes {}'.format(stk[i]))
                legend()
            tight_layout()
    if truth:
        return r.x[0], dustrec, cmbrec, sigs_dust, sigs_cmb, resid_dust, resid_cmb, mytruth[0], mytruth[1]
    else:
        return r.x[0], dustrec, cmbrec
