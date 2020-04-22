from __future__ import division

import numpy as np
import random
import string
import os
import scipy.interpolate as interp


import healpy as hp
import pysm
import pysm.units as u
from pysm import utils
import camb
import pickle

import qubic
from qubic.utils import progress_bar
from qubic import NamasterLib as nam

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
    cls = np.zeros_like(totDL)
    for i in range(4):
        cls[2:, i] = 2 * np.pi * totDL[2:, i] / (ls[2:] * (ls[2:] + 1))
    return cls


######### Make CAMB library
def rcamblib(rvalues,lmax=3*256, save=None):
	lll, totDL, unlensedDL = get_camb_Dl(lmax=lmax, r=0)
	spec = np.zeros((len(lll),4, len(rvalues)))
	specunlensed = np.zeros((len(lll),4,len(rvalues)))
	i=0
	bar = progress_bar(len(rvalues), 'CAMB Spectra')
	for r in rvalues:	
		bar.update()	
		ls, spec[:,:,i], specunlensed[:,:,i] = get_camb_Dl(lmax=lmax, r=r)
		i = i + 1

	camblib = [lll, rvalues, spec, specunlensed]
	if save:
		with open('./camblib.pickle', 'wb') as handle:
			pickle.dump(camblib, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return camblib

def bin_camblib(Namaster, filename, nside, verbose=True, return_unbinned=False):
	lll, rvalues, spec, specunlensed = read_camblib(filename)
	ellb, b = Namaster.get_binning(nside)
	nbins = len(ellb)
	nr = len(rvalues)
	binned_spec = np.zeros((nbins, 4, nr))
	binned_specunlensed = np.zeros((nbins, 4, nr))
	fact = 2 * np.pi / (ellb * (ellb + 1))
	if  verbose: bar = progress_bar(nr, 'Binning CAMB Librairy')
	for ir in range(nr):
		for j in range(4):
			binned_spec[:, j, ir] = fact * b.bin_cell(np.reshape(spec[:Namaster.lmax + 1,j,ir], (1, Namaster.lmax + 1)))
			binned_specunlensed[:, j, ir] = fact * b.bin_cell(np.reshape(specunlensed[:Namaster.lmax + 1,j,ir], (1, Namaster.lmax + 1)))
		if verbose: bar.update()	
	if return_unbinned:
		return [ellb, rvalues, binned_spec, binned_specunlensed, [lll, rvalues, spec, specunlensed]]
	else:
		return [ellb, rvalues, binned_spec, binned_specunlensed]



def read_camblib(file):
	with open(file, 'rb') as handle: camblib = pickle.load(handle)
	return camblib

def get_Dl_fromlib(lvals, r, lib=None, specindex=None, unlensed=False):
	if lib is None:
		### If not library was provided, we recalculated one
		rmin = 0.001
		rmax = 1
		nb =10
		lmaxcamb = 3*256
		rvalues = np.concatenate((np.zeros(1),np.logspace(np.log10(rmin),np.log10(rmax),nb)))
		camblib = rcamblib(rvalues, lmaxcamb)
	elif isinstance(lib,str):
		### If the library provided is a filename we read it
		camblib = read_camblib(lib)
	else:
		### then the provided library should be a instance of a camb library: a list
		camblib = lib

	lll = camblib[0]

	### If specindex is not specified we do all four of them
	if specindex is None:
		myspec = np.zeros((len(lvals),4)) 
		if unlensed: myspecunlensed = np.zeros((len(lvals),4)) 
		for i in range(4):
			interpolant = interp.RectBivariateSpline(lll,camblib[1],camblib[2][:,i,:])
			myspec[:,i] = np.ravel(interpolant(lvals, r))
			if unlensed:
				interpolant = interp.RectBivariateSpline(lll,camblib[1],camblib[3][:,i,:])
				myspecunlensed[:,i] = np.ravel(interpolant(lvals, r))
			else:
				myspecunlensed=None
	### if an index has been specified (eg. 2 for BB) then we only compute this one (useful to speed-up MCMC)
	else:
		interpolant = interp.RectBivariateSpline(lll,camblib[1],camblib[2][:,specindex,:])
		myspec = np.ravel(interpolant(lvals, r))
		if unlensed: 
			interpolant = interp.RectBivariateSpline(lll,camblib[1],camblib[3][:,specindex,:])
			myspecunlensed = np.ravel(interpolant(lvals, r))
		else:
			myspecunlensed = None


	return myspec, myspecunlensed

