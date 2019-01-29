#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 09:44:35 2019

@author: james
"""

import numpy as np
import fibtools as ft
import matplotlib.pyplot as plt
from pysimulators import FitsArray
import matplotlib.mlab as mlab
import scipy.ndimage.filters as f

def FreqResp(theTES, frange, fff, filt):
	figure()
	#setup plot params
	
	spectrum, freq = mlab.psd(dd[theTES,:], Fs=FREQ_SAMPLING, NFFT=nsamples, window=mlab.window_hanning)
	filtered_spec = f.gaussian_filter1d(spectrum, filt)
	rng = (freq > frange[0]) & (freq < frange[1])
	loglog(freq[rng], filtered_spec[rng], label='Data')
	
	#do plot

	xlim(frange[0], frange[1])
	title('Tes #{}'.format(theTES+1))
	ylim(np.min(filtered_spec[rng])*0.8, np.max(filtered_spec[rng])*1.2)
	xlabel('Freq [Hz]')
	ylabel('Power Spectrum [$nA^2.Hz^{-1}$]')
	#### Show where the signal is expected
	for ii in xrange(10): plt.plot(np.array([fff,fff])*(ii+1),[1e-20,1e-10],'r--', alpha=0.3)
	#### PT frequencies
	fpt = 1.724
	for ii in xrange(10): plt.plot(np.array([fpt,fpt])*(ii+1),[1e-20,1e-10],'k--', alpha=0.3)

	return

def FiltFreqResp(theTES, frange, fff, filt, freqs_pt, bw_0):
	plt.figure()
	#set up data
	
	for i in xrange(len(freqs_pt)):
		notch.append([freqs_pt[i], bw_0*(1+i)])

	sigfilt = dd[theTES,:]
	for i in xrange(len(notch)):
		sigfilt = ft.notch_filter(sigfilt, notch[i][0], notch[i][1], FREQ_SAMPLING)
	
	spectrum_f, freq_f = mlab.psd(sigfilt, Fs=FREQ_SAMPLING, NFFT=nsamples, window=mlab.window_hanning)

	xlim(frange[0], frange[1])
	rng = (freq > frange[0]) & (freq < frange[1])
	loglog(freq[rng], filtered_spec[rng], label='Data')
	loglog(freq[rng], f.gaussian_filter1d(spectrum_f,filt)[rng], label='Filt')
	title('Tes #{}'.format(theTES+1))
	ylim(np.min(filtered_spec[rng])*0.8, np.max(filtered_spec[rng])*1.2)
	xlabel('Freq [Hz]')
	ylabel('Power Spectrum [$nA^2.Hz^{-1}$]')
	#### Show where the signal is expected
	for ii in xrange(10): plot(np.array([fff,fff])*(ii+1),[1e-20,1e-10],'r--', alpha=0.3)
	#### PT frequencies
	fpt = 1.724
	for ii in xrange(10): plot(np.array([fpt,fpt])*(ii+1),[1e-20,1e-10],'k--', alpha=0.3)

	return