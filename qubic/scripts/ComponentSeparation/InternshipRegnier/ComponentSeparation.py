import random
import healpy as hp
import glob
from scipy.optimize import curve_fit
import pickle
from importlib import reload
import time
import scipy
import pandas as pd
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


### FGBuster functions module
import fgbuster
from fgbuster import get_instrument, get_sky, get_observation, ilc, basic_comp_sep, harmonic_ilc, weighted_comp_sep, multi_res_comp_sep  # Predefined instrumental and sky-creation configurations
from fgbuster.visualization import corner_norm, plot_component
from fgbuster.mixingmatrix import MixingMatrix
from fgbuster.observation_helpers import _rj2cmb, _jysr2rj, get_noise_realization

# Imports needed for component separation
from fgbuster import (separation_recipes, xForecast, CMB, Dust, Synchrotron, FreeFree, PowerLaw,  # sky-fitting model
                      basic_comp_sep)


# Function to get all maps at the same resolution

def same_resol(map1, fwhm, fwhm_target=None, verbose=False) :
    sh = np.shape(map1)
    nb_bands = sh[0]
    delta_fwhm = np.zeros(nb_bands)
    if fwhm_target is None:
        myfwhm = np.max(fwhm)
    else:
        myfwhm = fwhm_target
    #print(myfwhm, np.max(fwhm))
    maps_out = np.zeros_like(map1)
    fwhm_out = np.zeros(nb_bands)
    for i in range(map1.shape[0]):
        delta_fwhm_deg = np.sqrt(myfwhm**2-fwhm[i]**2)
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

def depth(Nb, Ny) :

    depth_p = np.ones(Nb) * (np.sqrt(2)*np.sqrt(3)*2/np.sqrt(Ny/3))
    depth_i = np.ones(Nb) * (np.sqrt(3)*2/np.sqrt(Ny/3))

    return depth_i, depth_p

def give_me_rms_I(X, nside) :
    rms = np.std(X[:, 0, :], axis=1)
    rms *= np.sqrt(hp.nside2pixarea(nside, degrees=True)) * 60
    return rms

def give_me_rms_P(X, nside) :
    rms = np.std(X[:, 1:, :], axis=(1, 2)) * np.sqrt(hp.nside2pixarea(nside, degrees=True)) * 60
    return rms

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

    def fg_buster(self, map1=None, map_noise = None, comp=[CMB(), Dust(150., temp = 20)], freq=None, fwhmdeg=None, target = None, okpix = None, Stokesparameter = 'IQU', nside = None, dust_only = False) :

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

        print("Importation of Qubic instrument..")
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


        print("Give the right frequencies and fwhm..")
        # Change good frequency and FWHM
        ins.frequency = freq
        ins.fwhm = fwhmdeg

        print("Give the rms of noise..")

        if map_noise is not None :
            ins.depth_i = give_me_rms_I(map_noise, self.nside)
            ins.depth_p = give_me_rms_I(map_noise, self.nside)*np.sqrt(2)

        else :
            pass

        map1[:, :, ~okpix] = hp.UNSEEN

        if dust_only == True :
            comp = [Dust(150., temp = 20)]
        else :
            comp = [CMB(), Dust(150., temp = 20)]

        # Apply FG Buster
        print("Perform component separation..")

        if Stokesparameter == 'IQU' :

            r = basic_comp_sep(comp, ins, map1[:, :, :], nside = nside, method = 'BFGS', tol = 1e-5)

        elif Stokesparameter == 'QU' :

            r = basic_comp_sep(comp, ins, map1[:, 1:, :], nside = nside, method = 'BFGS', tol = 1e-5)

        elif Stokesparameter == 'I' :

            r = basic_comp_sep(comp, ins, map1[:, 0, :], nside = nside, method = 'BFGS', tol = 1e-5)

        else :

             raise TypeError('Incorrect Stokes parameter')


        return r

    def weighted(self, map1=None, comp=[CMB(), Dust(150., temp = 20)], cov = None, freq=None, fwhmdeg = None, target = None, okpix = None, Stokesparameter = 'IQU', nside = None, dust_only = False) :

        """
        --------
        inputs :
            - map1 : map of which we want to separate the components -> array type & (nb_bands, Nstokes, Npix)
            - comp : list type which contains components that we want to separe. Dust must have nu0 in input and we can fixe the temperature.
            - freq : list type of maps frequency
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

        print("Shape : ", map1[:, :, okpix].shape)
        print("Shape : ", cov_all[:, :, okpix].shape)

        if Stokesparameter == 'IQU' :

            r = weighted_comp_sep(comp, ins, map1[:, :, :], cov = cov_all[:, :, :], nside = nside, method = 'BFGS', tol = 1e-5)

        elif Stokesparameter == 'QU' :

            r = weighted_comp_sep(comp, ins, map1[:, 1:, :], cov = cov_all[:, 1:, :], nside = nside, method = 'BFGS', options = {}, tol = 1e-5)

        elif Stokesparameter == 'I' :

            r = weighted_comp_sep(comp, ins, map1[:, 0, :], cov = cov_all[:, 0, :], nside = nside, method = 'BFGS', options = {}, tol = 1e-5)

        else :

             raise TypeError('Incorrect Stokes parameter')

        return r
