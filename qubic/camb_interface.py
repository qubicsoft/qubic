from __future__ import division

import numpy as np
import scipy.interpolate as interp

import healpy as hp
import camb
import camb.correlations as cc
import pickle

from qubic.utils import progress_bar


def get_camb_Dl(lmax=2500, H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06, As=2e-9, ns=0.965, r=0.):
    """
    Inspired from: https://camb.readthedocs.io/en/latest/CAMBdemo.html
    NB: this returns Dl = l(l+1)Cl/2pi
    Python CL arrays are all zero based (starting at l=0), Note l=0,1 entries will be zero by default.
    The different DL are always in the order TT, EE, BB, TE (with BB=0 for unlensed scalar results).
    """

    # Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()
    # This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.WantTensors = True
    pars.InitPower.set_params(As=As, ns=ns, r=r)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    # calculate results for these parameters
    results = camb.get_results(pars)
    # get dictionary of CAMB power spectra
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    totDL = powers['total']
    unlensedDL = powers['unlensed_total']
    # Python CL arrays are all zero based (starting at L=0), Note L=0,1 entries will be zero by default.
    # The different CL are always in the order TT, EE, BB, TE (with BB=0 for unlensed scalar results).
    ls = np.arange(totDL.shape[0])
    return ls, totDL, unlensedDL


def Dl2Cl_without_monopole(ls, totDL):
    """
    Go from Dls to Cls.
    """
    cls = np.zeros_like(totDL)
    for i in range(4):
        cls[2:, i] = 2 * np.pi * totDL[2:, i] / (ls[2:] * (ls[2:] + 1))
    return cls


def rcamblib(rvalues, lmax=3 * 256, save=None):
    """
    Make CAMB library
    """
    lll, totDL, unlensedDL = get_camb_Dl(lmax=lmax, r=0)
    spec = np.zeros((len(lll), 4, len(rvalues)))
    specunlensed = np.zeros((len(lll), 4, len(rvalues)))
    i = 0
    bar = progress_bar(len(rvalues), 'CAMB Spectra')
    for r in rvalues:
        bar.update()
        ls, spec[:, :, i], specunlensed[:, :, i] = get_camb_Dl(lmax=lmax, r=r)
        i += 1

    camblib = [lll, rvalues, spec, specunlensed]
    if save:
        with open(save, 'wb') as handle:
            pickle.dump(camblib, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return camblib


def bin_camblib(Namaster, filename, nside, verbose=True, return_unbinned=False):
    """
    Bin the spectra using Namaster.
    Parameters
    ----------
    Namaster
    filename
    nside
    verbose
    return_unbinned

    Returns
    -------

    """
    lll, rvalues, spec, specunlensed = read_camblib(filename)
    ellb, b = Namaster.get_binning(nside)
    nbins = len(ellb)
    nr = len(rvalues)
    binned_spec = np.zeros((nbins, 4, nr))
    binned_specunlensed = np.zeros((nbins, 4, nr))
    fact = 2 * np.pi / (ellb * (ellb + 1))
    if verbose:
        bar = progress_bar(nr, 'Binning CAMB Librairy')
    for ir in range(nr):
        for j in range(4):
            binned_spec[:, j, ir] = fact * b.bin_cell(
                np.reshape(spec[:Namaster.lmax + 1, j, ir], (1, Namaster.lmax + 1)))
            binned_specunlensed[:, j, ir] = fact * b.bin_cell(
                np.reshape(specunlensed[:Namaster.lmax + 1, j, ir], (1, Namaster.lmax + 1)))
        if verbose:
            bar.update()
    if return_unbinned:
        return [ellb, rvalues, binned_spec, binned_specunlensed, [lll, rvalues, spec, specunlensed]]
    else:
        return [ellb, rvalues, binned_spec, binned_specunlensed]


def read_camblib(file):
    with open(file, 'rb') as handle:
        camblib = pickle.load(handle)
    return camblib


def get_Dl_fromlib(lvals, r, lib=None, specindex=None, unlensed=False):
    if lib is None:
        ### If not library was provided, we recalculated one
        rmin = 0.001
        rmax = 1
        nb = 10
        lmaxcamb = 3 * 256
        rvalues = np.concatenate((np.zeros(1), np.logspace(np.log10(rmin), np.log10(rmax), nb)))
        camblib = rcamblib(rvalues, lmaxcamb)
    elif isinstance(lib, str):
        ### If the library provided is a filename we read it
        camblib = read_camblib(lib)
    else:
        ### then the provided library should be a instance of a camb library: a list
        camblib = lib

    lll = camblib[0]

    ### If specindex is not specified we do all four of them
    if specindex is None:
        myspec = np.zeros((len(lvals), 4))
        if unlensed:
            myspecunlensed = np.zeros((len(lvals), 4))
        for i in range(4):
            interpolant = interp.RectBivariateSpline(lll, camblib[1], camblib[2][:, i, :])
            myspec[:, i] = np.ravel(interpolant(lvals, r))
            if unlensed:
                interpolant = interp.RectBivariateSpline(lll, camblib[1], camblib[3][:, i, :])
                myspecunlensed[:, i] = np.ravel(interpolant(lvals, r))
            else:
                myspecunlensed = None
    ### if an index has been specified (eg. 2 for BB) then we only compute this one (useful to speed-up MCMC)
    else:
        interpolant = interp.RectBivariateSpline(lll, camblib[1], camblib[2][:, specindex, :])
        myspec = np.ravel(interpolant(lvals, r))
        if unlensed:
            interpolant = interp.RectBivariateSpline(lll, camblib[1], camblib[3][:, specindex, :])
            myspecunlensed = np.ravel(interpolant(lvals, r))
        else:
            myspecunlensed = None

    return myspec, myspecunlensed


def ctheta_2_cell(theta_deg, ctheta, lmax, normalization=1.):
    ### this is how camb recommends to prepare the x = cos(theta) values for integration
    ### These x values do not contain x=1 so we have. to do this case separately
    x, w = np.polynomial.legendre.leggauss(lmax + 1)
    xdeg = np.degrees(np.arccos(x))

    ### We first replace theta=0 by 0 and do that case separately
    myctheta = ctheta.copy()
    myctheta[0] = 0
    ### And now we fill the array that should include polarization (we put zeros there)
    ### with the values of our imput c(theta) interpolated at the x locations
    allctheta = np.zeros((len(x), 4))
    allctheta[:, 0] = np.interp(xdeg, theta_deg, myctheta)

    ### Here we call the camb function that does the transform to Cl
    clth = cc.corr2cl(allctheta, x, w, lmax)
    lll = np.arange(lmax + 1)

    ### the special case x=1 corresponds to theta=0 and add 2pi times c(theta=0) to the Cell
    return lll, clth[:, 0] + ctheta[0] * normalization


def cell_2_ctheta(cell, theta_deg=None, normalization=1.):
    lmax = len(cell) - 1
    x, w = np.polynomial.legendre.leggauss(lmax + 1)

    allcell = np.zeros((len(cell), 4))
    allcell[:, 0] = cell - cell[0]
    ctheta = cc.cl2corr(allcell, x, lmax=lmax)[:, 0]

    ### Case x = 1
    x = np.append(x, 1)
    ctheta = np.append(ctheta, cell[0] / normalization)
    xdeg = np.degrees(np.arccos(x))

    if theta_deg is None:
        #### put x and ctheta in reverse order to have increasing theta
        return xdeg[::-1], ctheta[::-1]
    else:
        return theta_deg, np.interp(theta_deg, xdeg[::-1], ctheta[::-1])


def simulate_correlated_map(nside, signoise, clin=None,
                            nside_fact=1, lmax_nside=2.,
                            generate_alm=False, verbose=True,
                            myiter=3, use_weights=False, seed=None, synfast=True):
    #### Define the seed
    if seed is not None:
        np.random.seed(42)

    #### We can work at the planned nside
    # normal maps
    lmax = int(lmax_nside * nside)
    ell = np.arange(lmax + 1)
    npix = 12 * nside ** 2

    #### Or with higher resolution maps in order to reduce the effect of aliasing on the RMS of the maps
    #### However this does not change the Cl spectrum so is likely to be worthless
    ### higher resolution maps
    nside_big = nside_fact * nside
    lmax_big = int(lmax_nside * nside_big)
    ell_big = np.arange(lmax_big + 1)
    npix_big = 12 * nside_big ** 2

    #### We also need to account for the pixel window function
    pixwin = hp.pixwin(nside_big)[:lmax_big + 1] * 0 + 1
    if clin is None:
        clth = 1. / pixwin ** 2
        return np.random.randn(12 * nside ** 2) * signoise
    else:
        clth = clin[0:lmax_big + 1] / clin[0] / pixwin ** 2

    #### There are three options here
    # 1. use ssynfast to directly generate the map with the correct spectrum (fastest)
    # 2. generate alms by hand and go back to map-sapce (essentially equivalent to the previous)
    # 3. generate a map in pixel space, smooth it with hp.smoothing() => slower by ~ factor 5

    if synfast:
        ### Case 1.
        fact = signoise * np.sqrt(4 * np.pi / npix_big) * nside_fact
        map_back = hp.synfast(clth, nside_big, lmax=lmax_big, verbose=False) * fact
    # print('Simulated a correlated map')
    else:
        ### Cases 2 and 3 Genereate the alms be it in harmonic space or pixel-space
        if generate_alm:
            ### Case 2
            if verbose:
                print('simulate alms in harmonic space')
            alm_size = hp.sphtfunc.Alm.getsize(lmax_big)
            alm_rms = 1. / np.sqrt(2) * signoise * nside_fact * np.sqrt(4 * np.pi / npix_big)
            alms = (np.random.randn(alm_size) + np.random.randn(alm_size) * 1.0j) * alm_rms
        else:
            ### Case 3
            if verbose:
                print('Simulate in pixel-space an uncorrelated map')
            ### Map realization with large nside
            map_uncorr_big = np.random.randn(npix_big) * signoise * nside_fact
            rms_uncorr_big = np.std(map_uncorr_big)
            ### Now alms
            alms = hp.map2alm(map_uncorr_big, lmax=lmax_big, iter=myiter, use_weights=use_weights)

        ### Apply filter:
        alms = hp.almxfl(alms, np.sqrt(clth))

        ### Now go back to pixel-space
        map_back = hp.alm2map(alms, nside_big, lmax=lmax_big, verbose=verbose)

    if nside_fact == 1:
        return map_back
    else:
        return hp.ud_grade(map_back, nside)
