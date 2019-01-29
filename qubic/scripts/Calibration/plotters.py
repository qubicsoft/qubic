#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 09:44:35 2019

@author: james
"""

import matplotlib.pyplot as plt
import numpy as np

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
