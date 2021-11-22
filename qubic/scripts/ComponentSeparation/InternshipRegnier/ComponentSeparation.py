import random
import healpy as hp
import glob
from scipy.optimize import curve_fit
import pickle
from importlib import reload
import time
import scipy
<<<<<<< HEAD
<<<<<<< HEAD
=======
import pandas as pd
>>>>>>> 817389f4cc3163541fa042c883a3919ba9169a19
=======
import pandas as pd
>>>>>>> master
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import pylab
from pylab import arange, show, cm
from astropy import units as uq
import gc
from qubic import QubicSkySim as qss
from qubic import NamasterLib as nam
<<<<<<< HEAD
<<<<<<< HEAD


### FGBuster functions module
=======
import fgbuster as fgb
import qubic
from comp_tools import format_alms, same_resolution


### FGBuster functions module
import fgbuster
>>>>>>> 817389f4cc3163541fa042c883a3919ba9169a19
=======
import fgbuster as fgb
import qubic
from comp_tools import format_alms, same_resolution


### FGBuster functions module
import fgbuster
>>>>>>> master
from fgbuster import get_instrument, get_sky, get_observation, ilc, basic_comp_sep, harmonic_ilc, weighted_comp_sep, multi_res_comp_sep  # Predefined instrumental and sky-creation configurations
from fgbuster.visualization import corner_norm, plot_component
from fgbuster.mixingmatrix import MixingMatrix
from fgbuster.observation_helpers import _rj2cmb, _jysr2rj, get_noise_realization

# Imports needed for component separation
from fgbuster import (separation_recipes, xForecast, CMB, Dust, Synchrotron, FreeFree, PowerLaw,  # sky-fitting model
<<<<<<< HEAD
<<<<<<< HEAD
                      basic_comp_sep)  # separation routine
                                           

# Function to get all maps at the same resolution

=======
                      basic_comp_sep)



def open_picklefile(directory, name, var) :

    """
    Open a pickle file from saved data
    """

    tab = pickle.load(open(directory + name, 'rb'))
    variable = tab[var]
    return variable


# Function to get all maps at the same resolution

=======
                      basic_comp_sep)



def open_picklefile(directory, name, var) :

    """
    Open a pickle file from saved data
    """

    tab = pickle.load(open(directory + name, 'rb'))
    variable = tab[var]
    return variable


# Function to get all maps at the same resolution

>>>>>>> master
def get_alm_maps(pixel_maps, fwhm, lmax = 512, resol_correction=False, ref_arcmin=None):
    """
    Compute alm maps from pixel maps and format them for FgBuster.
    """
    
    ell = np.arange(start=0, stop=lmax+1)
    if ref_arcmin is None:
        # if not specified take the lowest
        ref_arcmin = 60 * fwhm[0]  # in degrees
    ref_sigma_rad = np.deg2rad(ref_arcmin / 60.) / 2.355
    ref_fl = np.exp(- 0.5 * np.square(ref_sigma_rad * ell))
    beam_sigmas_rad = np.deg2rad(fwhm) / 2.355
    # pixwin = hp.pixwin(nside, lmax=lmax) if pixwin_correction else np.ones(lmax + 1)
    # compute maps
    alm_maps = None
    for f in range(pixel_maps.shape[0]):
        alms = hp.map2alm(pixel_maps[f], lmax=lmax, pol=True)
        correction = None
        if f == 0:
            sh = np.shape(alms)
            alm_maps = np.empty((pixel_maps.shape[0], sh[0], 2 * sh[1]))
        if resol_correction:
            gauss_fl = np.exp(- 0.5 * np.square(beam_sigmas_rad[f] * ell))
            correction = 1 / gauss_fl# / pixwin
        for i, t in enumerate(alms):
            alm_maps[f, i] = format_alms(hp.almxfl(t, correction) if resol_correction else t)
    return alm_maps

<<<<<<< HEAD
>>>>>>> 817389f4cc3163541fa042c883a3919ba9169a19
=======
>>>>>>> master
def same_resol(map1, fwhm, fwhm_target=None, verbose=False) :
    sh = np.shape(map1)
    nb_bands = sh[0]
    delta_fwhm = np.zeros(nb_bands)
    if fwhm_target is None:
        myfwhm = np.max(fwhm)
    else:
        myfwhm = fwhm_target
<<<<<<< HEAD
<<<<<<< HEAD
    print(myfwhm, np.max(fwhm))
=======
    #print(myfwhm, np.max(fwhm))
>>>>>>> 817389f4cc3163541fa042c883a3919ba9169a19
=======
    #print(myfwhm, np.max(fwhm))
>>>>>>> master
    maps_out = np.zeros_like(map1)
    fwhm_out = np.zeros(nb_bands)
    for i in range(map1.shape[0]):
        delta_fwhm_deg = np.sqrt(myfwhm**2-fwhm[i]**2)
<<<<<<< HEAD
<<<<<<< HEAD
        if verbose:
            print('Sub {0:}: Going from {1:5.12f} to {2:5.12f}'.format(i, fwhm[i], myfwhm))
=======
        #if verbose:
            #print('Sub {0:}: Going from {1:5.12f} to {2:5.12f}'.format(i, fwhm[i], myfwhm))
>>>>>>> master
        if delta_fwhm_deg != 0:
            delta_fwhm[i] = delta_fwhm_deg
            #print('   -> Convolution with {0:5.2f}'.format(delta_fwhm_deg))
            maps_out[i,:,:] = hp.sphtfunc.smoothing(map1[i,:,:],
                                                    fwhm=np.radians(delta_fwhm_deg),
                                                    verbose=False)
        else:
            #print('   -> No Convolution'.format(delta_fwhm_deg))
            maps_out[i,:,:] = map1[i,:,:]

        fwhm_out[i] = np.sqrt(fwhm[i]**2 + delta_fwhm_deg**2)

    return maps_out, fwhm_out, delta_fwhm

def separate(comp, instr, maps_to_separate, tol=1e-12, print_option=True):
    solver_options = {}
    solver_options['disp'] = True
    fg_args = comp, instr, maps_to_separate
    fg_kwargs = {'method': 'BFGS', 'tol': tol, 'options': solver_options}
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

def depth(Nb, Ny) :

    depth_p = np.ones(Nb) * (np.sqrt(2)*np.sqrt(3)*2/np.sqrt(Ny/3))
    depth_i = np.ones(Nb) * (np.sqrt(3)*2/np.sqrt(Ny/3))

    return depth_i, depth_p

def give_me_rms_I(X, nside) :
    rms = np.std(X[:, 0, :], axis=1)
    rms *= np.sqrt(hp.nside2pixarea(nside, degrees=True)) * 60
    return rms

def give_me_freqs_fwhm(dic, Nb) :
    band = dic['filter_nu'] / 1e9
    filter_relative_bandwidth = dic['filter_relative_bandwidth']
    a, nus_edge, nus_in, df, e, Nbbands_in = qubic.compute_freq(band, Nb, filter_relative_bandwidth)
    return nus_in, dic['synthbeam_peak150_fwhm'] * 150 / nus_in

def SED_eval(comp, nu0 = None, temp = None, beta_d = None, nu = None) :

    if comp == 'Dust' :
        SED = fgbuster.component_model.Dust(nu0 = nu0, temp = temp, beta_d = beta_d).eval(nu = nu)
    elif comp == 'CMB' :
        SED = fgbuster.component_model.CMB().eval(nu = nu)
    else :
        raise TypeError('Give the good component please !')

    return SED

def scaling(R, okpix, nu, beta_d, npix, Stkp) :

    if Stkp == 'I' :

        maps_scale_dust = np.zeros(((nu.shape[0], 1, npix)))
        maps_scale_cmb = np.zeros(((nu.shape[0], 1, npix)))

        sed_dust = SED_eval('Dust', nu0 = 150., temp = 20., beta_d = beta_d, nu = nu)
        sed_cmb = SED_eval('CMB', nu0 = 150., temp = 20., beta_d = beta_d, nu = nu)
        for i in range(nu.shape[0]) :
            maps_scale_dust[i, 0, okpix] = sed_dust[i] * R.s[1]
            maps_scale_cmb[i, 0, okpix] = sed_cmb[i] * R.s[0]

    elif Stkp == 'QU' :

        maps_scale_dust = np.zeros(((nu.shape[0], 2, npix)))
        maps_scale_cmb = np.zeros(((nu.shape[0], 2, npix)))

        sed_dust = SED_eval('Dust', nu0 = 150., temp = 20., beta_d = beta_d, nu = nu)
        sed_cmb = SED_eval('CMB', nu0 = 150., temp = 20., beta_d = beta_d, nu = nu)
        for i in range(len(nu)) :
            maps_scale_dust[i, :, okpix] = sed_dust[i] * R.s[1].T
            maps_scale_cmb[i, :, okpix] = sed_cmb[i] * R.s[0].T

    elif Stkp == 'IQU' :

        maps_scale_dust = np.zeros(((len(nu), 3, npix)))
        maps_scale_cmb = np.zeros(((len(nu), 3, npix)))

        sed_dust = SED_eval('Dust', nu0 = 150., temp = 20., beta_d = beta_d, nu = nu)
        sed_cmb = SED_eval('CMB', nu0 = 150., temp = 20., beta_d = beta_d, nu = nu)
        for i in range(len(nu)) :
            maps_scale_dust[i, :, :] = sed_dust[i] * R.s[1]
            maps_scale_cmb[i, :, :] = sed_cmb[i] * R.s[0]

    else :

        raise TypeError('Incorrect Stokes parameter')


    return maps_scale_cmb, maps_scale_dust

def depthi_test(dic, nu, fwhm) :
    Tobs = 3*60*60*24*365
    Ndet = 992*2
    fsky = 9776 * hp.pixelfunc.nside2pixarea(256, degrees = True) * 60**2
    nep = 4e-17/10
    d_i = np.zeros([nu.shape[0]])

    def nep2net(nep, freq, bandwidth, temp):
        h = 6.62607004e-34
        k = 1.38064852e-23
        x = h*freq/k/temp
        dPdT = (2*k*bandwidth) * (x**2*np.exp(x)) /(np.exp(x)-1)**2
        n = nep / dPdT
        return n

    def give_me_di(NET, fs, Ndet, Tobs) :
        return (NET**2 * fs)/(Ndet * Tobs)

    for i in range(nu.shape[0]) :
        nu0 = nu[i]*1e9
        bw = nu0 * fwhm[i]
        temp = dic['temperature']
        net = nep2net(nep, nu0, bw, temp)
        net = net * 1e6
        d_i[i] = give_me_di(net, fsky, Ndet, Tobs)

    return np.sqrt(d_i)

class CompSep(object) :

    """

    Class that brings together different methods of component separations. Currently, there is only 'fg_buster' definition which work with 2 components (CMB and Dust).

    """

    def __init__(self, d) :

        self.nside = d['nside']
        self.npix = 12 * self.nside**2
        self.Nfout = int(d['nf_recon'])
        self.Nfin = int(4*d['nf_sub'])
        self.lmin = 20
        self.lmax = 2 * self.nside - 1
        self.delta_ell = 16

    def fg_buster(self, map1=None, map_noise = None, comp = [CMB(), Dust(150., temp = 20)], freq=None, fwhmdeg=None, target = None, okpix = None) :

        """
        --------
        inputs :
            - map1 : map of which we want to separate the components -> array type & (nb_bands, Nstokes, Npix)
            - comp : list type which contains components that we want to separe. Dust must have nu0 in input and we can fixe the temperature.
            - freq : list type of maps frequency
            - fwhmdeg : list type of full width at half maximum (in degrees). It can be different values.
            - target : if target is not None, "same_resol" definition is applied and put all the maps at the same resolution. If target is None, make sure that all the resolution are the same.
            - okpix : boolean array type which exclude the edges of the map.

        --------
        output :
            - r : Dictionary which contains the amplitude of each components, the estimated parameter beta_d and dust temperature.

        """
        ins = get_instrument('Qubic')


        #print('depthp', ins.depth_p)
        #print('depthi', ins.depth_i)


        print("Put the maps at the same angular resolution..")
        # Change resolution of each map if it's necessary
        if target is not None :
            map1, tab_fwhm, delta_fwhm = same_resol(map1, fwhmdeg, fwhm_target = target, verbose = True)
            if map_noise is not None :
                map_noise, tab_fwhm, delta_fwhm = same_resol(map_noise, fwhmdeg, fwhm_target = target, verbose = True)
            else :
                pass
        else :
            pass

        # Change good frequency and FWHM
        ins.frequency = freq
        ins.fwhm = fwhmdeg

        print("Give the rms of noise..")

        if map_noise is not None :
            ins.depth_i = give_me_rms_I(map_noise, self.nside)
            ins.depth_p = give_me_rms_I(map_noise, self.nside)*np.sqrt(2)

        else :
            pass

        print("Perform component separation..")

        if okpix is not None :
            map1[:, :, ~okpix] = hp.UNSEEN
        r = basic_comp_sep(comp, ins, map1, method = 'TNC', tol = 1e-5)


        return r

    def weighted(self, map1=None, cov = None, freq=None, fwhmdeg = None, target = None, okpix = None, Stokesparameter = 'IQU', nside = None, dust_only = False) :

<<<<<<< HEAD
             raise TypeError('Incorrect Stokes parameter')
    
        
                
        return X_array   
    
    def fg_buster(self, map1=None, comp=None, freq=None, fwhmdeg=None, target = None, okpix = None, Stokesparameter = 'IQU') :
        
=======
        #if verbose:
            #print('Sub {0:}: Going from {1:5.12f} to {2:5.12f}'.format(i, fwhm[i], myfwhm))
        if delta_fwhm_deg != 0:
            delta_fwhm[i] = delta_fwhm_deg
            #print('   -> Convolution with {0:5.2f}'.format(delta_fwhm_deg))
            maps_out[i,:,:] = hp.sphtfunc.smoothing(map1[i,:,:],
                                                    fwhm=np.radians(delta_fwhm_deg),
                                                    verbose=False)
        else:
            #print('   -> No Convolution'.format(delta_fwhm_deg))
            maps_out[i,:,:] = map1[i,:,:]

        fwhm_out[i] = np.sqrt(fwhm[i]**2 + delta_fwhm_deg**2)

    return maps_out, fwhm_out, delta_fwhm

def separate(comp, instr, maps_to_separate, tol=1e-12, print_option=True):
    solver_options = {}
    solver_options['disp'] = True
    fg_args = comp, instr, maps_to_separate
    fg_kwargs = {'method': 'BFGS', 'tol': tol, 'options': solver_options}
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

def depth(Nb, Ny) :

    depth_p = np.ones(Nb) * (np.sqrt(2)*np.sqrt(3)*2/np.sqrt(Ny/3))
    depth_i = np.ones(Nb) * (np.sqrt(3)*2/np.sqrt(Ny/3))

    return depth_i, depth_p

def give_me_rms_I(X, nside) :
    rms = np.std(X[:, 0, :], axis=1)
    rms *= np.sqrt(hp.nside2pixarea(nside, degrees=True)) * 60
    return rms

def give_me_freqs_fwhm(dic, Nb) :
    band = dic['filter_nu'] / 1e9
    filter_relative_bandwidth = dic['filter_relative_bandwidth']
    a, nus_edge, nus_in, df, e, Nbbands_in = qubic.compute_freq(band, Nb, filter_relative_bandwidth)
    return nus_in, dic['synthbeam_peak150_fwhm'] * 150 / nus_in

def SED_eval(comp, nu0 = None, temp = None, beta_d = None, nu = None) :

    if comp == 'Dust' :
        SED = fgbuster.component_model.Dust(nu0 = nu0, temp = temp, beta_d = beta_d).eval(nu = nu)
    elif comp == 'CMB' :
        SED = fgbuster.component_model.CMB().eval(nu = nu)
    else :
        raise TypeError('Give the good component please !')

    return SED

def scaling(R, okpix, nu, beta_d, npix, Stkp) :

    if Stkp == 'I' :

        maps_scale_dust = np.zeros(((nu.shape[0], 1, npix)))
        maps_scale_cmb = np.zeros(((nu.shape[0], 1, npix)))

        sed_dust = SED_eval('Dust', nu0 = 150., temp = 20., beta_d = beta_d, nu = nu)
        sed_cmb = SED_eval('CMB', nu0 = 150., temp = 20., beta_d = beta_d, nu = nu)
        for i in range(nu.shape[0]) :
            maps_scale_dust[i, 0, okpix] = sed_dust[i] * R.s[1]
            maps_scale_cmb[i, 0, okpix] = sed_cmb[i] * R.s[0]

    elif Stkp == 'QU' :

        maps_scale_dust = np.zeros(((nu.shape[0], 2, npix)))
        maps_scale_cmb = np.zeros(((nu.shape[0], 2, npix)))

        sed_dust = SED_eval('Dust', nu0 = 150., temp = 20., beta_d = beta_d, nu = nu)
        sed_cmb = SED_eval('CMB', nu0 = 150., temp = 20., beta_d = beta_d, nu = nu)
        for i in range(len(nu)) :
            maps_scale_dust[i, :, okpix] = sed_dust[i] * R.s[1].T
            maps_scale_cmb[i, :, okpix] = sed_cmb[i] * R.s[0].T

    elif Stkp == 'IQU' :

        maps_scale_dust = np.zeros(((len(nu), 3, npix)))
        maps_scale_cmb = np.zeros(((len(nu), 3, npix)))

        sed_dust = SED_eval('Dust', nu0 = 150., temp = 20., beta_d = beta_d, nu = nu)
        sed_cmb = SED_eval('CMB', nu0 = 150., temp = 20., beta_d = beta_d, nu = nu)
        for i in range(len(nu)) :
            maps_scale_dust[i, :, :] = sed_dust[i] * R.s[1]
            maps_scale_cmb[i, :, :] = sed_cmb[i] * R.s[0]

    else :

        raise TypeError('Incorrect Stokes parameter')


    return maps_scale_cmb, maps_scale_dust

def depthi_test(dic, nu, fwhm) :
    Tobs = 3*60*60*24*365
    Ndet = 992*2
    fsky = 9776 * hp.pixelfunc.nside2pixarea(256, degrees = True) * 60**2
    nep = 4e-17/10
    d_i = np.zeros([nu.shape[0]])

    def nep2net(nep, freq, bandwidth, temp):
        h = 6.62607004e-34
        k = 1.38064852e-23
        x = h*freq/k/temp
        dPdT = (2*k*bandwidth) * (x**2*np.exp(x)) /(np.exp(x)-1)**2
        n = nep / dPdT
        return n

    def give_me_di(NET, fs, Ndet, Tobs) :
        return (NET**2 * fs)/(Ndet * Tobs)

    for i in range(nu.shape[0]) :
        nu0 = nu[i]*1e9
        bw = nu0 * fwhm[i]
        temp = dic['temperature']
        net = nep2net(nep, nu0, bw, temp)
        net = net * 1e6
        d_i[i] = give_me_di(net, fsky, Ndet, Tobs)

    return np.sqrt(d_i)

class CompSep(object) :

    """

    Class that brings together different methods of component separations. Currently, there is only 'fg_buster' definition which work with 2 components (CMB and Dust).

    """

    def __init__(self, d) :

        self.nside = d['nside']
        self.npix = 12 * self.nside**2
        self.Nfout = int(d['nf_recon'])
        self.Nfin = int(4*d['nf_sub'])
        self.lmin = 20
        self.lmax = 2 * self.nside - 1
        self.delta_ell = 16

    def fg_buster(self, map1=None, map_noise = None, comp = [CMB(), Dust(150., temp = 20)], freq=None, fwhmdeg=None, target = None, okpix = None) :

        """
        --------
        inputs :
            - map1 : map of which we want to separate the components -> array type & (nb_bands, Nstokes, Npix)
            - comp : list type which contains components that we want to separe. Dust must have nu0 in input and we can fixe the temperature.
            - freq : list type of maps frequency
            - fwhmdeg : list type of full width at half maximum (in degrees). It can be different values.
            - target : if target is not None, "same_resol" definition is applied and put all the maps at the same resolution. If target is None, make sure that all the resolution are the same.
            - okpix : boolean array type which exclude the edges of the map.

        --------
        output :
            - r : Dictionary which contains the amplitude of each components, the estimated parameter beta_d and dust temperature.

        """
        ins = get_instrument('Qubic')


        #print('depthp', ins.depth_p)
        #print('depthi', ins.depth_i)


        print("Put the maps at the same angular resolution..")
        # Change resolution of each map if it's necessary
        if target is not None :
            map1, tab_fwhm, delta_fwhm = same_resol(map1, fwhmdeg, fwhm_target = target, verbose = True)
            if map_noise is not None :
                map_noise, tab_fwhm, delta_fwhm = same_resol(map_noise, fwhmdeg, fwhm_target = target, verbose = True)
            else :
                pass
        else :
            pass

        # Change good frequency and FWHM
        ins.frequency = freq
        ins.fwhm = fwhmdeg

        print("Give the rms of noise..")

        if map_noise is not None :
            ins.depth_i = give_me_rms_I(map_noise, self.nside)
            ins.depth_p = give_me_rms_I(map_noise, self.nside)*np.sqrt(2)

        else :
            pass

        print("Perform component separation..")

        if okpix is not None :
            map1[:, :, ~okpix] = hp.UNSEEN
        r = basic_comp_sep(comp, ins, map1, method = 'TNC', tol = 1e-5)


        return r

    def weighted(self, map1=None, cov = None, freq=None, fwhmdeg = None, target = None, okpix = None, Stokesparameter = 'IQU', nside = None, dust_only = False) :

>>>>>>> 817389f4cc3163541fa042c883a3919ba9169a19
=======
>>>>>>> master
        """
        --------
        inputs :
            - map1 : map of which we want to separate the components -> array type & (nb_bands, Nstokes, Npix)
            - comp : list type which contains components that we want to separe. Dust must have nu0 in input and we can fixe the temperature.
            - freq : list type of maps frequency
<<<<<<< HEAD
<<<<<<< HEAD
            - fwhmdeg : list type of full width at half maximum (in degrees). It can be different values.
=======
            - cov : Covariance matrix (it have to be broadcastable with map1 shape)
>>>>>>> master
            - target : if target is not None, "same_resol" definition is applied and put all the maps at the same resolution. If target is None, make sure that all the resolution are the same.
            - okpix : boolean array type which exclude the edges of the map.

        --------
        output :
            - r : Dictionary which contains the amplitude of each components, the estimated parameter beta_d and dust temperature.

        """

        print("Importation of Qubic instrument..")
        ins = get_instrument('Qubic')


        #print('depthp', ins.depth_p)
        #print('depthi', ins.depth_i)


        print("Put the maps at the same angular resolution..")
        # Change resolution of each map if it's necessary
        if target is not None :
            map1, tab_fwhm, delta_fwhm = same_resol(map1, fwhmdeg, fwhm_target = target, verbose = False)
            cov_all = np.zeros(((map1.shape[0], 3, self.npix)))
            cov_all[:, :, okpix] = cov

            cov_all, tab_fwhm, delta_fwhm = same_resol(cov_all, fwhmdeg, fwhm_target = target, verbose = False)
        else :
            cov_all = np.zeros(((map1.shape[0], 3, self.npix)))
            cov_all[:, :, okpix] = cov


        print("Give the right frequencies..")
        # Change good frequency and FWHM
        ins.frequency = freq
        ins.fwhm = fwhmdeg


        # Apply FG Buster
        print("Perform component separation..")

        map1[:, :, ~okpix] = hp.UNSEEN
        cov_all[:, :, ~okpix] = hp.UNSEEN

        if dust_only == True :
            comp = [Dust(150., temp = 20)]
        else :
            comp = [CMB(), Dust(150., temp = 20)]

        if Stokesparameter == 'IQU' :

            r = weighted_comp_sep(comp, ins, map1[:, :, :], cov = cov_all[:, :, :], nside = nside, method = 'BFGS', tol = 1e-5)

        elif Stokesparameter == 'QU' :

            r = weighted_comp_sep(comp, ins, map1[:, 1:, :], cov = cov_all[:, 1:, :], nside = nside, method = 'BFGS', options = {}, tol = 1e-5)

        elif Stokesparameter == 'I' :

<<<<<<< HEAD
            r = basic_comp_sep(comp, ins, map1[:, 0, okpix])
=======
            - cov : Covariance matrix (it have to be broadcastable with map1 shape)
            - target : if target is not None, "same_resol" definition is applied and put all the maps at the same resolution. If target is None, make sure that all the resolution are the same.
            - okpix : boolean array type which exclude the edges of the map.

        --------
        output :
            - r : Dictionary which contains the amplitude of each components, the estimated parameter beta_d and dust temperature.

        """

        print("Importation of Qubic instrument..")
        ins = get_instrument('Qubic')


        #print('depthp', ins.depth_p)
        #print('depthi', ins.depth_i)


        print("Put the maps at the same angular resolution..")
        # Change resolution of each map if it's necessary
        if target is not None :
            map1, tab_fwhm, delta_fwhm = same_resol(map1, fwhmdeg, fwhm_target = target, verbose = False)
            cov_all = np.zeros(((map1.shape[0], 3, self.npix)))
            cov_all[:, :, okpix] = cov

            cov_all, tab_fwhm, delta_fwhm = same_resol(cov_all, fwhmdeg, fwhm_target = target, verbose = False)
        else :
            cov_all = np.zeros(((map1.shape[0], 3, self.npix)))
            cov_all[:, :, okpix] = cov


        print("Give the right frequencies..")
        # Change good frequency and FWHM
        ins.frequency = freq
        ins.fwhm = fwhmdeg


        # Apply FG Buster
        print("Perform component separation..")

        map1[:, :, ~okpix] = hp.UNSEEN
        cov_all[:, :, ~okpix] = hp.UNSEEN

        if dust_only == True :
            comp = [Dust(150., temp = 20)]
        else :
            comp = [CMB(), Dust(150., temp = 20)]

        if Stokesparameter == 'IQU' :

            r = weighted_comp_sep(comp, ins, map1[:, :, :], cov = cov_all[:, :, :], nside = nside, method = 'BFGS', tol = 1e-5)

        elif Stokesparameter == 'QU' :

            r = weighted_comp_sep(comp, ins, map1[:, 1:, :], cov = cov_all[:, 1:, :], nside = nside, method = 'BFGS', options = {}, tol = 1e-5)

        elif Stokesparameter == 'I' :

            r = weighted_comp_sep(comp, ins, map1[:, 0, :], cov = cov_all[:, 0, :], nside = nside, method = 'BFGS', options = {}, tol = 1e-5)
>>>>>>> 817389f4cc3163541fa042c883a3919ba9169a19
=======
            r = weighted_comp_sep(comp, ins, map1[:, 0, :], cov = cov_all[:, 0, :], nside = nside, method = 'BFGS', options = {}, tol = 1e-5)
>>>>>>> master

        else :

             raise TypeError('Incorrect Stokes parameter')

        return r
<<<<<<< HEAD
<<<<<<< HEAD
=======


class Apply(object):

    def __init__(self, d) :

        self.nside = d['nside']
        self.npix = 12 * self.nside**2
        self.Nfout = int(d['nf_recon'])
        self.Nfin = int(4*d['nf_sub'])
        self.lmax = 512

    def do_same_resol(self, dic, nsub, noise, coverage, seed, dust_only = False, iib = False, spatial_noise = False, nunu_correlation = False, noise_profile = False) :

        """

        That function perform the component separation of nsub sub-bands at noise level. We suppose now that all the maps are at the same angular resolution. 

        """

>>>>>>> master
        
        if dust_only is True :
            sky_config = {'dust' : 'd0'}
            comp = [Dust(150, temp = 20)]
        else :
            sky_config = {'cmb' : seed, 'dust' : 'd0'}
            comp = [CMB(), Dust(150, temp = 20)]

        Qubic_sky = qss.Qubic_sky(sky_config, dic)
        freqs, fwhm = give_me_freqs_fwhm(dic, nsub)
        fwhmdegs = [np.max(fwhm)]*nsub
        dic['nf_sub'] = 4*nsub
        dic['nf_recon'] = nsub


        maps, maps_noiseless, noise, _ = Qubic_sky.get_partial_sky_maps_withnoise(coverage=coverage,
                                       Nyears=noise, verbose=False, FWHMdeg=fwhmdegs, seed = seed,
                                       spatial_noise=spatial_noise,
                                       nunu_correlation=nunu_correlation, noise_profile = noise_profile,
                                       integrate_into_band=iib)
        maps = np.transpose(maps, (0, 2, 1))
        maps_noiseless = np.transpose(maps_noiseless, (0, 2, 1))
        noise = np.transpose(noise, (0, 2, 1))



        r_noisy = CompSep(dic).fg_buster(map1=maps, map_noise = noise, comp = comp, freq=freqs, fwhmdeg=fwhmdegs, target = None)
        r_noiseless = CompSep(dic).fg_buster(map1=maps_noiseless, map_noise = noise, comp = comp, freq=freqs, fwhmdeg=fwhmdegs, target = None)

        return r_noisy, r_noiseless


    def do_not_same_resol(self, dic, nsub, noise, coverage, okpix_to_fgb, seed, dust_only = False, iib = False, spatial_noise = False, nunu_correlation = False, noise_profile = False) :

        """

        That function perform the component separation of nsub sub-bands at noise level. We suppose now that all the maps are at the same angular resolution. 

        """

        def give_me_mask_apo(okpix = None, lmin = 2, lmax = 512, delta_ell = 16) :
            mask = np.zeros(12 * dic['nside']**2)
            mask[okpix] = 1
            # Namaster object
            Namaster = nam.Namaster(mask, lmin=2, lmax=512, delta_ell=16)
            apo = Namaster.mask_apo
            return apo

        # Define okpix
        cov150 = open_picklefile('/home/mathias/Bureau/FG-Buster/', 'Coverage_dtheta_15_pointing_3000_256.pkl', 'coverage')
        okpix = (cov150 > (0.1*np.max(cov150)))

        apo = give_me_mask_apo(okpix = okpix)

        
        if dust_only is True :
            sky_config = {'dust' : 'd0'}
            comp = [Dust(150, temp = 20)]
        else :
            sky_config = {'cmb' : seed, 'dust' : 'd0'}
            comp = [CMB(), Dust(150, temp = 20)]

        Qubic_sky = qss.Qubic_sky(sky_config, dic)
        freqs, fwhmdegs = give_me_freqs_fwhm(dic, nsub)
        
        dic['nf_sub'] = 4*nsub
        dic['nf_recon'] = nsub


        maps, maps_noiseless, noise, _ = Qubic_sky.get_partial_sky_maps_withnoise(coverage=coverage,
                                       Nyears=noise, verbose=False, FWHMdeg=fwhmdegs, seed = seed,
                                       spatial_noise=spatial_noise,
                                       nunu_correlation=nunu_correlation, noise_profile = noise_profile,
                                       integrate_into_band=iib)
        maps = np.transpose(maps, (0, 2, 1))*apo
        maps_noiseless = np.transpose(maps_noiseless, (0, 2, 1))*apo
        noise = np.transpose(noise, (0, 2, 1))*apo





        r_noisy = CompSep(dic).fg_buster(map1=maps, map_noise = noise, comp = comp, freq=freqs, fwhmdeg=fwhmdegs, target = np.max(fwhmdegs) + 1e-8, okpix = okpix_to_fgb)
        r_noiseless = CompSep(dic).fg_buster(map1=maps_noiseless, map_noise = noise, comp = comp, freq=freqs, fwhmdeg=fwhmdegs, target = np.max(fwhmdegs) + 1e-8, okpix = okpix_to_fgb)

        return r_noisy, r_noiseless

    def do_not_same_resol_cut_fullsky(self, dic, nsub, noise, okpix_to_fgb, seed, dust_only = False, iib = False, spatial_noise = False, nunu_correlation = False, noise_profile = False) :

        """

        That function perform the component separation of nsub sub-bands at noise level. We suppose now that all the maps are at the same angular resolution. 

        """

        
        if dust_only is True :
            sky_config = {'dust' : 'd0'}
            comp = [Dust(150, temp = 20)]
        else :
            sky_config = {'cmb' : seed, 'dust' : 'd0'}
            comp = [CMB(), Dust(150, temp = 20)]

        Qubic_sky = qss.Qubic_sky(sky_config, dic)
        freqs, fwhmdegs = give_me_freqs_fwhm(dic, nsub)
        
        dic['nf_sub'] = 4*nsub
        dic['nf_recon'] = nsub

        okpix_fullsky = np.ones(12*self.nside**2)


        maps, _, noise, _ = Qubic_sky.get_partial_sky_maps_withnoise(coverage=okpix_fullsky,
                                       Nyears=noise, verbose=False, FWHMdeg=fwhmdegs, seed = seed,
                                       spatial_noise=spatial_noise,
                                       nunu_correlation=nunu_correlation, noise_profile = noise_profile,
                                       integrate_into_band=iib)
        maps = np.transpose(maps, (0, 2, 1))
        noise = np.transpose(noise, (0, 2, 1))

        maps_same_resol, _, _ = same_resol(maps, fwhmdegs, np.max(fwhmdegs) + 1e-8)
        maps_noise_same_resol, _, _ = same_resol(noise, fwhmdegs, np.max(fwhmdegs) + 1e-8)

        # Noisy maps
        qubic_sky = np.zeros(((self.Nfout, 3, 12*self.nside**2)))
        qubic_sky[:, :, okpix_to_fgb] = maps_same_resol[:, :, okpix_to_fgb]

        # Noise maps
        qubic_sky_noise = np.zeros(((self.Nfout, 3, 12*self.nside**2)))
        qubic_sky_noise[:, :, okpix_to_fgb] = maps_noise_same_resol[:, :, okpix_to_fgb]



        r_noisy = CompSep(dic).fg_buster(map1=qubic_sky, map_noise = qubic_sky_noise, comp = comp, freq=freqs, fwhmdeg=fwhmdegs, target = None, okpix = okpix_to_fgb)

        return r_noisy 

    def do_comparison_pixels_alms_same_resol(self, dic, nsub, noise, okpix, method = None, dust_only = False) :


        seed = np.random.randint(100000000)
        lmax = self.lmax

        if dust_only is True :
            sky_config = {'dust' : 'd0'}
            comp = [Dust(150, temp = 20)]
        else :
            sky_config = {'cmb' : seed, 'dust' : 'd0'}
            comp = [CMB(), Dust(150, temp = 20)]


        Qubic_sky = qss.Qubic_sky(sky_config, dic)
        freqs, fwhmdegs = give_me_freqs_fwhm(dic, nsub)
        fwhmdegs = [np.max(fwhmdegs)]*nsub
        
        dic['nf_sub'] = 4*nsub
        dic['nf_recon'] = nsub

        okpix_fullsky = np.ones(12*self.nside**2)


        maps, _, N, _ = Qubic_sky.get_partial_sky_maps_withnoise(coverage=okpix,
                                       Nyears=noise, verbose=False, FWHMdeg=fwhmdegs, seed = seed,
                                       spatial_noise=False,
                                       nunu_correlation=False, noise_profile = False,
                                       integrate_into_band=False)
        maps = np.transpose(maps, (0, 2, 1))
        N = np.transpose(N, (0, 2, 1))

        ins = get_instrument('Qubic')
        ins.frequency = freqs
        ins.fwhm = fwhmdegs
        ins.depth_i = give_me_rms_I(N, self.nside)
        ins.depth_p = give_me_rms_I(N, self.nside)*np.sqrt(2)

        if method == 'pixels' :
            r = separate(comp, ins, maps, tol=1e-5, print_option=True)
        elif method == 'alms' :
            alms = get_alm_maps(maps, fwhmdegs, lmax = 512, resol_correction=False, ref_arcmin=None)
            r = separate(comp, ins, alms, tol=1e-5, print_option=True)
        else :
            raise TypeError("Choose the good method bewteen map-based or alms-based")

        return r


    def do_comparison_pixels_alms(self, dic, nsub, noise, okpix, okpix_to_fgb = None, seed = None, map1 = None, map_noise = None, do_apo = False, noiseless = False, method = None, dust_only = False) :

        """
        That function do the separation for pixels or alms absed method. 
        ---------
        inputs : dic -> Dictionary
                 nsub -> Number of sub-bands
                 noise -> Noise level (in years)
                 okpix -> Create maps for special seen pixels. We provide these pixels to FG-Buster
                 seed -> If seed is provided, we have the same realisation
                 map -> If map is porvided, we use this map, if not we generate a random map
                 map_noise -> Same that map but for the noise maps
                 do_apo -> If True, we apply the apodizing on map to reduce edge effects
                 noiseless -> If True, we use a noiseless map
                 method -> Write pixels to use pixels separation and alms to use the alms of maps provided
                 dust_only -> If True, we generate only maps with dust and non CMB, If False we use CMB+Dust maps
        """

        if seed is None :
            seed = np.random.randint(100000000)

        lmax = self.lmax
        freqs, fwhmdegs = give_me_freqs_fwhm(dic, nsub)

        if map1 is not None :
            maps = map1
        if map_noise is not None :
            N = map_noise
        if map1 is None :
            if dust_only is True :
                sky_config = {'dust' : 'd0'}
                comp = [Dust(150, temp = 20)]
            else :
                sky_config = {'cmb' : seed, 'dust' : 'd0'}
                comp = [CMB(), Dust(150, temp = 20)]
            
            Qubic_sky = qss.Qubic_sky(sky_config, dic)
        
            dic['nf_sub'] = 4*nsub
            dic['nf_recon'] = nsub

            if noiseless :
                _, maps, N, _ = Qubic_sky.get_partial_sky_maps_withnoise(coverage=okpix,
                                       Nyears=noise, verbose=False, FWHMdeg=fwhmdegs, seed = seed,
                                       spatial_noise=False,
                                       nunu_correlation=False, noise_profile = False,
                                       integrate_into_band=False)
            else :
                maps, _, N, _ = Qubic_sky.get_partial_sky_maps_withnoise(coverage=okpix,
                                       Nyears=noise, verbose=False, FWHMdeg=fwhmdegs, seed = seed,
                                       spatial_noise=False,
                                       nunu_correlation=False, noise_profile = False,
                                       integrate_into_band=False)
            

        ins = get_instrument('Qubic')

        if do_apo :
            mask = np.zeros(12 * dic['nside']**2)
            mask[okpix] = 1
            # Namaster object
            Namaster = nam.Namaster(mask, lmin=2, lmax=lmax, delta_ell=16)
            apo = Namaster.mask_apo

            maps = np.transpose(maps, (0, 2, 1))*apo
            N = np.transpose(N, (0, 2, 1))*apo
        else :
            maps = np.transpose(maps, (0, 2, 1))
            N = np.transpose(N, (0, 2, 1))

        ins.frequency = freqs
        ins.fwhm = fwhmdegs

        if noiseless is False :
            N, _, _ = same_resol(N, fwhmdegs, fwhm_target=np.max(fwhmdegs) + 1e-8, verbose=False)
            ins.depth_i = give_me_rms_I(N[:, :, okpix], self.nside)
            ins.depth_p = give_me_rms_I(N[:, :, okpix], self.nside)*np.sqrt(2)

        if method == 'alms' :
            alms = get_alm_maps(maps, fwhmdegs, lmax = 512, resol_correction=True, ref_arcmin=None)
            r = separate(comp, ins, alms, tol=1e-18, print_option=False)

        elif method == 'pixels' :
            maps_same_resol, _, _ = same_resol(maps, fwhmdegs, fwhm_target=np.max(fwhmdegs) + 1e-8, verbose=False)
            if okpix_to_fgb is not None :
                maps_same_resol[:, :, ~okpix_to_fgb] = hp.UNSEEN
            else :
                maps_same_resol[:, :, ~okpix] = hp.UNSEEN
            r = separate(comp, ins, maps_same_resol, tol=1e-5, print_option=False)

        else :
            raise TypeError("Choose the good method bewteen map-based or alms-based")
        

        return r, maps, N





        
<<<<<<< HEAD
        return tab_cmb
                     
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
=======


class Apply(object):

    def __init__(self, d) :

        self.nside = d['nside']
        self.npix = 12 * self.nside**2
        self.Nfout = int(d['nf_recon'])
        self.Nfin = int(4*d['nf_sub'])
        self.lmax = 512

    def do_same_resol(self, dic, nsub, noise, coverage, seed, dust_only = False, iib = False, spatial_noise = False, nunu_correlation = False, noise_profile = False) :

        """

        That function perform the component separation of nsub sub-bands at noise level. We suppose now that all the maps are at the same angular resolution. 

        """

        
        if dust_only is True :
            sky_config = {'dust' : 'd0'}
            comp = [Dust(150, temp = 20)]
        else :
            sky_config = {'cmb' : seed, 'dust' : 'd0'}
            comp = [CMB(), Dust(150, temp = 20)]

        Qubic_sky = qss.Qubic_sky(sky_config, dic)
        freqs, fwhm = give_me_freqs_fwhm(dic, nsub)
        fwhmdegs = [np.max(fwhm)]*nsub
        dic['nf_sub'] = 4*nsub
        dic['nf_recon'] = nsub


        maps, maps_noiseless, noise, _ = Qubic_sky.get_partial_sky_maps_withnoise(coverage=coverage,
                                       Nyears=noise, verbose=False, FWHMdeg=fwhmdegs, seed = seed,
                                       spatial_noise=spatial_noise,
                                       nunu_correlation=nunu_correlation, noise_profile = noise_profile,
                                       integrate_into_band=iib)
        maps = np.transpose(maps, (0, 2, 1))
        maps_noiseless = np.transpose(maps_noiseless, (0, 2, 1))
        noise = np.transpose(noise, (0, 2, 1))



        r_noisy = CompSep(dic).fg_buster(map1=maps, map_noise = noise, comp = comp, freq=freqs, fwhmdeg=fwhmdegs, target = None)
        r_noiseless = CompSep(dic).fg_buster(map1=maps_noiseless, map_noise = noise, comp = comp, freq=freqs, fwhmdeg=fwhmdegs, target = None)

        return r_noisy, r_noiseless


    def do_not_same_resol(self, dic, nsub, noise, coverage, okpix_to_fgb, seed, dust_only = False, iib = False, spatial_noise = False, nunu_correlation = False, noise_profile = False) :

        """

        That function perform the component separation of nsub sub-bands at noise level. We suppose now that all the maps are at the same angular resolution. 

        """

        def give_me_mask_apo(okpix = None, lmin = 2, lmax = 512, delta_ell = 16) :
            mask = np.zeros(12 * dic['nside']**2)
            mask[okpix] = 1
            # Namaster object
            Namaster = nam.Namaster(mask, lmin=2, lmax=512, delta_ell=16)
            apo = Namaster.mask_apo
            return apo

        # Define okpix
        cov150 = open_picklefile('/home/mathias/Bureau/FG-Buster/', 'Coverage_dtheta_15_pointing_3000_256.pkl', 'coverage')
        okpix = (cov150 > (0.1*np.max(cov150)))

        apo = give_me_mask_apo(okpix = okpix)

        
        if dust_only is True :
            sky_config = {'dust' : 'd0'}
            comp = [Dust(150, temp = 20)]
        else :
            sky_config = {'cmb' : seed, 'dust' : 'd0'}
            comp = [CMB(), Dust(150, temp = 20)]

        Qubic_sky = qss.Qubic_sky(sky_config, dic)
        freqs, fwhmdegs = give_me_freqs_fwhm(dic, nsub)
        
        dic['nf_sub'] = 4*nsub
        dic['nf_recon'] = nsub


        maps, maps_noiseless, noise, _ = Qubic_sky.get_partial_sky_maps_withnoise(coverage=coverage,
                                       Nyears=noise, verbose=False, FWHMdeg=fwhmdegs, seed = seed,
                                       spatial_noise=spatial_noise,
                                       nunu_correlation=nunu_correlation, noise_profile = noise_profile,
                                       integrate_into_band=iib)
        maps = np.transpose(maps, (0, 2, 1))*apo
        maps_noiseless = np.transpose(maps_noiseless, (0, 2, 1))*apo
        noise = np.transpose(noise, (0, 2, 1))*apo





        r_noisy = CompSep(dic).fg_buster(map1=maps, map_noise = noise, comp = comp, freq=freqs, fwhmdeg=fwhmdegs, target = np.max(fwhmdegs) + 1e-8, okpix = okpix_to_fgb)
        r_noiseless = CompSep(dic).fg_buster(map1=maps_noiseless, map_noise = noise, comp = comp, freq=freqs, fwhmdeg=fwhmdegs, target = np.max(fwhmdegs) + 1e-8, okpix = okpix_to_fgb)

        return r_noisy, r_noiseless

    def do_not_same_resol_cut_fullsky(self, dic, nsub, noise, okpix_to_fgb, seed, dust_only = False, iib = False, spatial_noise = False, nunu_correlation = False, noise_profile = False) :

        """

        That function perform the component separation of nsub sub-bands at noise level. We suppose now that all the maps are at the same angular resolution. 

        """

        
        if dust_only is True :
            sky_config = {'dust' : 'd0'}
            comp = [Dust(150, temp = 20)]
        else :
            sky_config = {'cmb' : seed, 'dust' : 'd0'}
            comp = [CMB(), Dust(150, temp = 20)]

        Qubic_sky = qss.Qubic_sky(sky_config, dic)
        freqs, fwhmdegs = give_me_freqs_fwhm(dic, nsub)
        
        dic['nf_sub'] = 4*nsub
        dic['nf_recon'] = nsub

        okpix_fullsky = np.ones(12*self.nside**2)


        maps, _, noise, _ = Qubic_sky.get_partial_sky_maps_withnoise(coverage=okpix_fullsky,
                                       Nyears=noise, verbose=False, FWHMdeg=fwhmdegs, seed = seed,
                                       spatial_noise=spatial_noise,
                                       nunu_correlation=nunu_correlation, noise_profile = noise_profile,
                                       integrate_into_band=iib)
        maps = np.transpose(maps, (0, 2, 1))
        noise = np.transpose(noise, (0, 2, 1))

        maps_same_resol, _, _ = same_resol(maps, fwhmdegs, np.max(fwhmdegs) + 1e-8)
        maps_noise_same_resol, _, _ = same_resol(noise, fwhmdegs, np.max(fwhmdegs) + 1e-8)

        # Noisy maps
        qubic_sky = np.zeros(((self.Nfout, 3, 12*self.nside**2)))
        qubic_sky[:, :, okpix_to_fgb] = maps_same_resol[:, :, okpix_to_fgb]

        # Noise maps
        qubic_sky_noise = np.zeros(((self.Nfout, 3, 12*self.nside**2)))
        qubic_sky_noise[:, :, okpix_to_fgb] = maps_noise_same_resol[:, :, okpix_to_fgb]



        r_noisy = CompSep(dic).fg_buster(map1=qubic_sky, map_noise = qubic_sky_noise, comp = comp, freq=freqs, fwhmdeg=fwhmdegs, target = None, okpix = okpix_to_fgb)

        return r_noisy 

    def do_comparison_pixels_alms_same_resol(self, dic, nsub, noise, okpix, method = None, dust_only = False) :


        seed = np.random.randint(100000000)
        lmax = self.lmax

        if dust_only is True :
            sky_config = {'dust' : 'd0'}
            comp = [Dust(150, temp = 20)]
        else :
            sky_config = {'cmb' : seed, 'dust' : 'd0'}
            comp = [CMB(), Dust(150, temp = 20)]


        Qubic_sky = qss.Qubic_sky(sky_config, dic)
        freqs, fwhmdegs = give_me_freqs_fwhm(dic, nsub)
        fwhmdegs = [np.max(fwhmdegs)]*nsub
        
        dic['nf_sub'] = 4*nsub
        dic['nf_recon'] = nsub

        okpix_fullsky = np.ones(12*self.nside**2)


        maps, _, N, _ = Qubic_sky.get_partial_sky_maps_withnoise(coverage=okpix,
                                       Nyears=noise, verbose=False, FWHMdeg=fwhmdegs, seed = seed,
                                       spatial_noise=False,
                                       nunu_correlation=False, noise_profile = False,
                                       integrate_into_band=False)
        maps = np.transpose(maps, (0, 2, 1))
        N = np.transpose(N, (0, 2, 1))

        ins = get_instrument('Qubic')
        ins.frequency = freqs
        ins.fwhm = fwhmdegs
        ins.depth_i = give_me_rms_I(N, self.nside)
        ins.depth_p = give_me_rms_I(N, self.nside)*np.sqrt(2)

        if method == 'pixels' :
            r = separate(comp, ins, maps, tol=1e-5, print_option=True)
        elif method == 'alms' :
            alms = get_alm_maps(maps, fwhmdegs, lmax = 512, resol_correction=False, ref_arcmin=None)
            r = separate(comp, ins, alms, tol=1e-5, print_option=True)
        else :
            raise TypeError("Choose the good method bewteen map-based or alms-based")

        return r


    def do_comparison_pixels_alms(self, dic, nsub, noise, okpix, okpix_to_fgb = None, seed = None, map1 = None, map_noise = None, do_apo = False, noiseless = False, method = None, dust_only = False) :

        """
        That function do the separation for pixels or alms absed method. 
        ---------
        inputs : dic -> Dictionary
                 nsub -> Number of sub-bands
                 noise -> Noise level (in years)
                 okpix -> Create maps for special seen pixels. We provide these pixels to FG-Buster
                 seed -> If seed is provided, we have the same realisation
                 map -> If map is porvided, we use this map, if not we generate a random map
                 map_noise -> Same that map but for the noise maps
                 do_apo -> If True, we apply the apodizing on map to reduce edge effects
                 noiseless -> If True, we use a noiseless map
                 method -> Write pixels to use pixels separation and alms to use the alms of maps provided
                 dust_only -> If True, we generate only maps with dust and non CMB, If False we use CMB+Dust maps
        """

        if seed is None :
            seed = np.random.randint(100000000)

        lmax = self.lmax
        freqs, fwhmdegs = give_me_freqs_fwhm(dic, nsub)

        if map1 is not None :
            maps = map1
        if map_noise is not None :
            N = map_noise
        if map1 is None :
            if dust_only is True :
                sky_config = {'dust' : 'd0'}
                comp = [Dust(150, temp = 20)]
            else :
                sky_config = {'cmb' : seed, 'dust' : 'd0'}
                comp = [CMB(), Dust(150, temp = 20)]
            
            Qubic_sky = qss.Qubic_sky(sky_config, dic)
        
            dic['nf_sub'] = 4*nsub
            dic['nf_recon'] = nsub

            if noiseless :
                _, maps, N, _ = Qubic_sky.get_partial_sky_maps_withnoise(coverage=okpix,
                                       Nyears=noise, verbose=False, FWHMdeg=fwhmdegs, seed = seed,
                                       spatial_noise=False,
                                       nunu_correlation=False, noise_profile = False,
                                       integrate_into_band=False)
            else :
                maps, _, N, _ = Qubic_sky.get_partial_sky_maps_withnoise(coverage=okpix,
                                       Nyears=noise, verbose=False, FWHMdeg=fwhmdegs, seed = seed,
                                       spatial_noise=False,
                                       nunu_correlation=False, noise_profile = False,
                                       integrate_into_band=False)
            

        ins = get_instrument('Qubic')

        if do_apo :
            mask = np.zeros(12 * dic['nside']**2)
            mask[okpix] = 1
            # Namaster object
            Namaster = nam.Namaster(mask, lmin=2, lmax=lmax, delta_ell=16)
            apo = Namaster.mask_apo

            maps = np.transpose(maps, (0, 2, 1))*apo
            N = np.transpose(N, (0, 2, 1))*apo
        else :
            maps = np.transpose(maps, (0, 2, 1))
            N = np.transpose(N, (0, 2, 1))

        ins.frequency = freqs
        ins.fwhm = fwhmdegs

        if noiseless is False :
            N, _, _ = same_resol(N, fwhmdegs, fwhm_target=np.max(fwhmdegs) + 1e-8, verbose=False)
            ins.depth_i = give_me_rms_I(N[:, :, okpix], self.nside)
            ins.depth_p = give_me_rms_I(N[:, :, okpix], self.nside)*np.sqrt(2)

        if method == 'alms' :
            alms = get_alm_maps(maps, fwhmdegs, lmax = 512, resol_correction=True, ref_arcmin=None)
            r = separate(comp, ins, alms, tol=1e-18, print_option=False)

        elif method == 'pixels' :
            maps_same_resol, _, _ = same_resol(maps, fwhmdegs, fwhm_target=np.max(fwhmdegs) + 1e-8, verbose=False)
            if okpix_to_fgb is not None :
                maps_same_resol[:, :, ~okpix_to_fgb] = hp.UNSEEN
            else :
                maps_same_resol[:, :, ~okpix] = hp.UNSEEN
            r = separate(comp, ins, maps_same_resol, tol=1e-5, print_option=False)

        else :
            raise TypeError("Choose the good method bewteen map-based or alms-based")
        

        return r, maps, N





        



=======


        
>>>>>>> master
















<<<<<<< HEAD
>>>>>>> 817389f4cc3163541fa042c883a3919ba9169a19
=======
>>>>>>> master
