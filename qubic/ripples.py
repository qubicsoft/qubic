# coding: utf-8
from __future__ import division

import healpy as hp
import numpy as np
from pyoperators.utils import reshape_broadcast
import numexpr as ne
from pyoperators import Operator, Rotation3dOperator
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
from pysimulators import BeamGaussian
from pdb import set_trace
from scipy.interpolate import splrep, splev
from cPickle import load
from .data import PATH

__all__ = ['ConvolutionRippledGaussianOperator',
           'BeamGaussianRippled']

class ConvolutionRippledGaussianOperator(HealpixConvolutionGaussianOperator):
    """
    Convolve a Healpix map by a gaussian kernel, modulated by ripples.

    """
    def __init__(self, freq,
                 nripples=2,
                 **keywords):
        nripples_max = 2
        if nripples not in range(nripples_max + 1):
            raise ValueError(
                'Input nripples is not a non-negative integer less than {}'.
                format(nripples_max + 1))
        self.nripples = nripples
        with open(PATH + 'sb_peak_plus_two_ripples_150HGz.pkl', 'r') as f:
            fl = load(f)
        fl /= fl.max()
        if freq == 150e9:
            fl_ = fl
        else:
            corr1 = [  1.65327594e-02, -2.24216210e-04, 9.70939946e-07, -1.40191824e-09]
            corr2 = [ -3.80559542e-01, 4.76370274e-03, -1.84237511e-05, 2.37962542e-08]
            def f(x, p):
                return p[0] + p[1] * x + p[2] * x**2 + p[3] * x**3
            ell = np.arange(len(fl)) + 1
            spl = splrep(ell * freq / 150e9, fl)
            if freq > 150e9:
                fl_ = splev(ell, spl) * (1 + f(freq / 1e9, corr2) + ell * f(freq / 1e9, corr1))
            else:
                fl_ = np.zeros(len(ell))
                fl_ = splev(ell[ell < ell.max() * freq / 150e9], spl) * \
                       (1 + f(freq / 1e9, corr2) + ell[ell < ell.max() * freq / 150e9] * f(freq / 1e9, corr1))
        self.fl = np.sqrt(fl_)
        self.fl[np.isnan(self.fl)] = 0.

    def direct(self, input, output):
        if input.ndim == 1:
            input = input[:, None]
            output = output[:, None]
        for i, o in zip(input.T, output.T):
            ialm = hp.map2alm(i)
            alm_smoothed = hp.almxfl(ialm, self.fl)
            o[...] = hp.alm2map(alm_smoothed, hp.npix2nside(len(i)))# * \
#                     2.2196409083134503 ## A map convolved with ripples is
#                                        ## this factor lower than the map conv. with gaussian

class ConvolutionRingOperator(ConvolutionRippledGaussianOperator):
    def __init__(self, freq,
                 nripple=1,
                 **keywords):
        nripple_max = 2
        if nripple not in np.arange(1, nripple_max + 1):
            raise ValueError(
                'Input nripple is not a positive integer '
                'less or equal {}'.format(nripples_max))
        self.nripple = nripple
        fl = hp.mrdfits(PATH + 'sb_peak_ripple{}_150HGz.fits'.
                               format(nripple))[0]
#        fl /= fl.max()
        fl = np.sqrt(fl)
        if freq == 150e9:
            self.fl = fl
        else:
            ell = np.arange(len(fl)) + 1
            spl = splrep(ell * freq / 150e9, fl)
            self.fl = splev(ell, spl)
        
class BeamGaussianRippled(BeamGaussian):
    """
    Axisymmetric gaussian beam with ripples.

    """
    def __init__(self, fwhm, backward=False, nripples=0):
        """
        Parameters
        ----------
        fwhm : float
            The Full-Width-Half-Maximum of the beam, in radians.
        backward : boolean, optional
            If true, the maximum of the beam is at theta=pi.

        """
        BeamGaussian.__init__(self, fwhm, backward=backward)
        self.nripples = nripples
        
    def __call__(self, theta, phi):
        if self.backward:
            theta = np.pi - theta
            
        s_peak = self.sigma
        s_ripple = self.sigma / 1.96

        coef = -0.5 / s_peak**2
        out = ne.evaluate('exp(coef * theta**2)')

        h = [0.01687, 0.00404] # relative heights of the first two ripples
        add = np.zeros(out.shape)
        for r in range(self.nripples):
            coef = -0.5 / s_ripple**2
            rh = h[r]
            m = s_peak * 4.014 + s_peak * r * 2.308
            add += ne.evaluate('rh * exp(coef * (theta - m)**2)')
            set_trace()
        out += add
            
        return reshape_broadcast(out, np.broadcast(theta, phi).shape)


