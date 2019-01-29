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

def FreqResp(theTES, frange, fff, filt, dd, FREQ_SAMPLING,nsamples):
	plt.figure()
	#setup plot params
	
	spectrum, freq = mlab.psd(dd[theTES,:], Fs=FREQ_SAMPLING, NFFT=nsamples, window=mlab.window_hanning)
	filtered_spec = f.gaussian_filter1d(spectrum, filt)
	rng = (freq > frange[0]) & (freq < frange[1])
	plt.loglog(freq[rng], filtered_spec[rng], label='Data')
	
	#do plot

	plt.xlim(frange[0], frange[1])
	plt.title('Tes #{}'.format(theTES+1))
	plt.ylim(np.min(filtered_spec[rng])*0.8, np.max(filtered_spec[rng])*1.2)
	plt.xlabel('Freq [Hz]')
	plt.ylabel('Power Spectrum [$nA^2.Hz^{-1}$]')
	#### Show where the signal is expected
	for ii in xrange(10): plt.plot(np.array([fff,fff])*(ii+1),[1e-20,1e-10],'r--', alpha=0.3)
	#### PT frequencies
	fpt = 1.724
	for ii in xrange(10): plt.plot(np.array([fpt,fpt])*(ii+1),[1e-20,1e-10],'k--', alpha=0.3)

	return

def FiltFreqResp(theTES, frange, fff, filt, freqs_pt, bw_0, dd, 
				 FREQ_SAMPLING, nsamples, freq, spectrum, filtered_spec):
	plt.figure()
	#set up data
	notch = []
	
	for i in xrange(len(freqs_pt)):
		notch.append([freqs_pt[i], bw_0*(1+i)])

	sigfilt = dd[theTES,:]
	for i in xrange(len(notch)):
		sigfilt = ft.notch_filter(sigfilt, notch[i][0], notch[i][1], FREQ_SAMPLING)
	
	spectrum_f, freq_f = mlab.psd(sigfilt, Fs=FREQ_SAMPLING, NFFT=nsamples, window=mlab.window_hanning)

	plt.xlim(frange[0], frange[1])
	rng = (freq > frange[0]) & (freq < frange[1])
	plt.loglog(freq[rng], filtered_spec[rng], label='Data')
	plt.loglog(freq[rng], f.gaussian_filter1d(spectrum_f,filt)[rng], label='Filt')
	plt.title('Tes #{}'.format(theTES+1))
	plt.ylim(np.min(filtered_spec[rng])*0.8, np.max(filtered_spec[rng])*1.2)
	plt.xlabel('Freq [Hz]')
	plt.ylabel('Power Spectrum [$nA^2.Hz^{-1}$]')
	#### Show where the signal is expected
	for ii in xrange(10): plt.plot(np.array([fff,fff])*(ii+1),[1e-20,1e-10],'r--', alpha=0.3)
	#### PT frequencies
	fpt = 1.724
	for ii in xrange(10): plt.plot(np.array([fpt,fpt])*(ii+1),[1e-20,1e-10],'k--', alpha=0.3)

	return

def FoldedFiltTES(tt, pars, theTES, folded, folded_notch):
	plt.figure()
	### Plot it along with a guess for fiber signal
	plt.plot(tt, folded[theTES,:], label='Data TES #{}'.format(theTES))
	plt.plot(tt, folded_notch[theTES,:], label='Data TES #{} (with Notch filter)'.format(theTES))
		  #for simsig, we should pass in 'pars' values
	plt.plot(tt, ft.simsig(tt, pars), label='Expected')
	plt.legend()
	plt.ylabel('Current [nA]')
	plt.xlabel('time [s]')
	
	return

def FoldedTESFreeFit(tt, bla, theTES, folded):
	plt.figure()
	#takes in free fit result as 'bla'
	params =  bla[1]
	err = bla[2]
	
	plt.plot(tt, folded[theTES,:], label='Data TES #{}'.format(theTES))
	plt.plot(tt, ft.simsig(tt, bla[1]), label='Fitted: \n cycle={0:8.3f}+/-{1:8.3f} \n tau = {2:8.3f}+/-{3:8.3f}s \n t0 = {4:8.3f}+/-{5:8.3f}s \n amp = {6:8.3f}+/-{7:8.3f}'.format(params[0], err[0], params[1], err[1], params[2], err[2], params[3], err[3]))
	plt.legend()
	plt.ylabel('Current [nA]')
	plt.xlabel('time [s]')
	plt.title('TES {} folded with simsig params'.format(theTES))
	
	return


def Allplots(fib, allparams, allparams1, allparams2, okfinal, okfinal1, okfinal2, asic, med=False, rng=[0,0.4]):
	plt.figure()
	
	plt.subplot(2,2,1)
	plt.hist(allparams[okfinal, 1],range=rng,bins=30,label='All ({}) '.format(okfinal.sum())+ft.statstr(allparams[okfinal,1]*1000, median=med)+' ms', alpha=0.5)
	plt.hist(allparams1[okfinal1, 1],range=rng,bins=30,label='Asic1 ({})'.format(okfinal1.sum())+ft.statstr(allparams1[okfinal1,1]*1000, median=med)+' ms', alpha=0.5)
	plt.hist(allparams1[okfinal2, 1],range=rng,bins=30,label='Asic2 ({})'.format(okfinal2.sum())+ft.statstr(allparams2[okfinal2,1]*1000, median=med)+' ms', alpha=0.5)
	plt.xlabel('Tau [sec]')
	plt.legend(fontsize=7, frameon=False)
	plt.title('Fib {} - Tau [s]'.format(fib))

	plt.subplot(2,2,2)
	plt.hist(allparams[okfinal, 3],range=[0,1],bins=15,label='All ({}) '.format(okfinal.sum())+ft.statstr(allparams[okfinal,3], median=med)+' nA', alpha=0.5)
	plt.hist(allparams1[okfinal1, 3],range=[0,1],bins=15,label='Asic1 ({}) '.format(okfinal1.sum())+ft.statstr(allparams1[okfinal1,3], median=med)+' nA', alpha=0.5)
	plt.hist(allparams1[okfinal2, 3],range=[0,1],bins=15,label='Asic2 ({}) '.format(okfinal2.sum())+ft.statstr(allparams2[okfinal2,3], median=med)+' nA', alpha=0.5)
	plt.xlabel('Amp [nA]')
	plt.legend(fontsize=7, frameon=False)
	plt.title('Fib {} - Amp [nA]'.format(fib))

	plt.subplot(2,2,3)
	imtau = ft.image_asics(data1=allparams1[:,1], data2=allparams2[:,1])	
	plt.imshow(imtau,vmin=0,vmax=0.5)
	plt.title('Tau [s] - Fiber {}'.format(fib,asic))
	plt.colorbar()

	plt.subplot(2,2,4)
	imamp = ft.image_asics(data1=allparams1[:,3], data2=allparams2[:,3])	
	plt.imshow(imamp,vmin=0,vmax=1)
	plt.title('Amp [nA] - Fiber {}'.format(fib,asic))
	plt.colorbar()
	plt.tight_layout()
	plt.savefig('fib{}_summary.png'.format(fib))

	return


def TESvsThermo(fib, tt, folded1, folded2, okfinal1, okfinal2, thermos):
	plt.figure()
	plt.subplot(2,1,1)
	plt.plot(tt, np.mean(folded1[okfinal1 * ~thermos,:], axis=0), 'b', lw=2, label='Valid TES average')
	plt.plot(tt, np.mean(folded1[thermos,:],axis=0), 'r', lw=2, label='Thermometers')
	plt.title('Fib = {} - ASIC 1'.format(fib))
	plt.legend(loc='upper left', fontsize=8)

	plt.subplot(2,1,2)
	plt.plot(tt, np.mean(folded2[okfinal2 * ~thermos,:], axis=0), 'b', lw=2, label='Valid TES average')
	plt.plot(tt, np.mean(folded2[thermos,:],axis=0), 'r', lw=2, label='Thermometers')
	plt.title('Fib = {} - ASIC 2'.format(fib))
	plt.savefig('fib{}_thermoVsTES.png'.format(fib))

	return