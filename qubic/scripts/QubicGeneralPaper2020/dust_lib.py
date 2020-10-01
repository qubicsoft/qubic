
from pylab import *
import healpy as hp
from matplotlib.pyplot import *
import numpy as np
import os
import sys
import string
import random
from scipy.constants import c
import scipy.integrate
import scipy.constants
from scipy import interpolate
from scipy import integrate



def Bnu(nuGHz, temp):
	h = scipy.constants.h
	c = scipy.constants.c
	k = scipy.constants.k
	nu = nuGHz*1e9
	return 2 * h * nu**3 / c**2 / (np.exp(h * nu / k / temp) - 1)

def dBnu_dT(nuGHz, temp):
	h = scipy.constants.h
	c = scipy.constants.c
	k = scipy.constants.k
	nu = nuGHz*1e9
	theBnu = Bnu(nuGHz, temp)
	return (theBnu * c / nu / temp)**2 / 2 * np.exp(h * nu / k / temp) / k

def mbb(nuGHz, beta, temp):
	return (nuGHz/353)**beta * Bnu(nuGHz, temp)

def KCMB2MJy_sr(nuGHz):
	h = scipy.constants.h
	c = scipy.constants.c
	k = scipy.constants.k
	T = 2.725
	nu = nuGHz*1e9
	x = h * nu / k / T
	ex = np.exp(x)
	fac_in = dBnu_dT(nuGHz, T)
	fac_out = 1e20
	return fac_in * fac_out

def freq_conversion(nuGHz_in, nuGHz_out, betadust, Tdust):
	val_in = KCMB2MJy_sr(nuGHz_in) / mbb(nuGHz_in, betadust, Tdust)
	val_out = KCMB2MJy_sr(nuGHz_out) / mbb(nuGHz_out, betadust, Tdust)
	return val_in / val_out


def Dl_BB_dust(ell, freqGHz1, freqGHz2=None, params = None):
	if params is None: params = [13.4 * 0.45, -2.42, 1.59, 19.6]
	if freqGHz2 is None: freqGHz2=freqGHz1
	Dl_353_ell80 = params[0]
	alpha_bb = params[1]
	betadust = params[2]
	Tdust = params[3]
	return Dl_353_ell80 * (freq_conversion(353, freqGHz1, betadust, Tdust) * freq_conversion(353, freqGHz2, betadust, Tdust)) * (ell/80)**(alpha_bb+2)

