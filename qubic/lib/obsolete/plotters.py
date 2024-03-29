#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 09:44:35 2019

@author: james, louise, JCH
"""

from __future__ import division, print_function
import numpy as np
import qubic.fibtools as ft
from matplotlib.pyplot import *
from pysimulators import FitsArray
import matplotlib.mlab as mlab
import scipy.ndimage.filters as f


def TimeSigPlot(time, dd, theTES):
    figure()
    clf()
    plot(time, dd[theTES, :])
    xlabel('Time [s]')
    ylabel('Current [nA]')

    return


def FreqResp(freq, frange, filtered_spec, theTES, fff):
    figure()
    # setup plot params
    rng = (freq > frange[0]) & (freq < frange[1])
    loglog(freq[rng], filtered_spec[rng], label='Data')

    # do plot
    xlim(frange[0], frange[1])
    title('Tes #{}'.format(theTES + 1))
    ylim(np.min(filtered_spec[rng]) * 0.8, np.max(filtered_spec[rng]) * 1.2)
    xlabel('Freq [Hz]')
    ylabel('Power Spectrum [$nA^2.Hz^{-1}$]')
    #### Show where the signal is expected
    for ii in range(10): plot(np.array([fff, fff]) * (ii + 1), [1e-20, 1e-10], 'r--', alpha=0.3)
    #### PT frequencies
    fpt = 1.724
    for ii in range(10): plot(np.array([fpt, fpt]) * (ii + 1), [1e-20, 1e-10], 'k--', alpha=0.3)

    return



def FiltFreqResp(theTES, frange, fff, filt, dd, notch, FREQ_SAMPLING, nsamples, freq, spectrum, filtered_spec):
    """
    Plot original and notch filtered frequency.
    sigfilt requires you select for which TES you would like to
    apply the notch filter - single TES analysis

    """
    # notch filter according to notch - must select TES

    sigfilt = dd[theTES, :]
    for i in range(len(notch)):
        sigfilt = ft.notch_filter(sigfilt, notch[i][0], notch[i][1], FREQ_SAMPLING)

    # get new spectrum with notch filter applied
    spectrum_f, freq_f = mlab.psd(sigfilt, Fs=FREQ_SAMPLING, NFFT=nsamples, window=mlab.window_hanning)

    # start plotting
    figure()
    xlim(frange[0], frange[1])
    rng = (freq > frange[0]) & (freq < frange[1])
    loglog(freq[rng], filtered_spec[rng], label='Data')
    loglog(freq[rng], f.gaussian_filter1d(spectrum_f, filt)[rng], label='Filt')
    title('Tes #{}'.format(theTES + 1))
    ylim(np.min(filtered_spec[rng]) * 0.8, np.max(filtered_spec[rng]) * 1.2)
    xlabel('Freq [Hz]')
    ylabel('Power Spectrum [$nA^2.Hz^{-1}$]')
    #### Show where the signal is expected
    for ii in range(10): plot(np.array([fff, fff]) * (ii + 1), [1e-20, 1e-10], 'r--', alpha=0.3)
    #### PT frequencies
    fpt = 1.724
    for ii in range(10): plot(np.array([fpt, fpt]) * (ii + 1), [1e-20, 1e-10], 'k--', alpha=0.3)

    return


def FoldedFiltTES(tt, pars, theTES, folded, folded_notch):
    figure()
    ### Plot it along with a guess for fiber signal
    plot(tt, np.mean(folded, axis=0), label='Median of Folding')
    plot(tt, folded[theTES, :], label='Data TES #{}'.format(theTES))
    plot(tt, folded_notch[theTES, :], label='Data TES #{} (with Notch filter)'.format(theTES))
    # for simsig, we should pass in 'pars' values
    plot(tt, ft.simsig(tt, pars), label='Expected')
    legend()
    ylabel('Current [nA]')
    xlabel('time [s]')

    return


def FoldedTESFreeFit(tt, bla, theTES, folded):
    figure()
    # takes in free fit result as 'bla'
    params = bla[1]
    err = bla[2]

    plot(tt, folded[theTES, :], label='Data TES #{}'.format(theTES))
    plot(tt, ft.simsig(tt, bla[1]),
             label='Fitted: \n cycle={0:8.3f}+/-{1:8.3f} \n tau = {2:8.3f}+/-{3:8.3f}s \n t0 = {4:8.3f}+/-{5:8.3f}s \n amp = {6:8.3f}+/-{7:8.3f}'.format(
                 params[0], err[0], params[1], err[1], params[2], err[2], params[3], err[3]))
    legend()
    ylabel('Current [nA]')
    xlabel('time [s]')
    title('TES {} folded with simsig params'.format(theTES))

    return


def Allplots(fib, allparams, allparams1, allparams2, okfinal, okfinal1, okfinal2, asic, med=False, rng=None,
             cmap='viridis'):
    figure()

    subplot(2, 2, 1)
    mmt, sst = ft.meancut(allparams[okfinal, 1], 3)
    hist(allparams[okfinal, 1], range=[0, mmt + 4 * sst], bins=15,
             label='All ({}) '.format(okfinal.sum()) + ft.statstr(allparams[okfinal, 1] * 1000, median=med) + ' ms',
             alpha=0.5)
    hist(allparams1[okfinal1, 1], range=[0, mmt + 4 * sst], bins=15,
             label='Asic1 ({})'.format(okfinal1.sum()) + ft.statstr(allparams1[okfinal1, 1] * 1000, median=med) + ' ms',
             alpha=0.5)
    hist(allparams1[okfinal2, 1], range=[0, mmt + 4 * sst], bins=15,
             label='Asic2 ({})'.format(okfinal2.sum()) + ft.statstr(allparams2[okfinal2, 1] * 1000, median=med) + ' ms',
             alpha=0.5)
    xlabel('Tau [sec]')
    legend(fontsize=7, frameon=False)
    title('Fib {} - Tau [s]'.format(fib))

    subplot(2, 2, 2)
    mma, ssa = ft.meancut(allparams[okfinal, 3], 3)
    hist(allparams[okfinal, 3], range=[0, mma + 4 * ssa], bins=15,
             label='All ({}) '.format(okfinal.sum()) + ft.statstr(allparams[okfinal, 3], median=med) + ' nA', alpha=0.5)
    hist(allparams1[okfinal1, 3], range=[0, mma + 4 * ssa], bins=15,
             label='Asic1 ({}) '.format(okfinal1.sum()) + ft.statstr(allparams1[okfinal1, 3], median=med) + ' nA',
             alpha=0.5)
    hist(allparams1[okfinal2, 3], range=[0, mma + 4 * ssa], bins=15,
             label='Asic2 ({}) '.format(okfinal2.sum()) + ft.statstr(allparams2[okfinal2, 3], median=med) + ' nA',
             alpha=0.5)
    xlabel('Amp [nA]')
    legend(fontsize=7, frameon=False)
    title('Fib {} - Amp [nA]'.format(fib))

    subplot(2, 2, 3)
    imtau = ft.image_asics(data1=allparams1[:, 1], data2=allparams2[:, 1])
    imshow(imtau, vmin=0, vmax=mmt + 4 * sst, interpolation='nearest', cmap=cmap)
    title('Tau [s] - Fiber {}'.format(fib, asic))
    colorbar()

    subplot(2, 2, 4)
    imamp = ft.image_asics(data1=allparams1[:, 3], data2=allparams2[:, 3])
    imshow(imamp, vmin=0, vmax=mma + 4 * ssa, interpolation='nearest', cmap=cmap)
    title('Amp [nA] - Fiber {}'.format(fib, asic))
    colorbar()
    tight_layout()
    return


def TESvsThermo(fib, tt, folded1, folded2, okfinal1, okfinal2, thermos):
    figure()
    subplot(2, 1, 1)
    plot(tt, np.mean(folded1[okfinal1 * ~thermos, :], axis=0), 'b', lw=2, label='Valid TES average')
    plot(tt, np.mean(folded1[thermos, :], axis=0), 'r', lw=2, label='Thermometers')
    title('Fib = {} - ASIC 1'.format(fib))
    legend(loc='upper left', fontsize=8)
    subplot(2, 1, 2)
    plot(tt, np.mean(folded2[okfinal2 * ~thermos, :], axis=0), 'b', lw=2, label='Valid TES average')
    plot(tt, np.mean(folded2[thermos, :], axis=0), 'r', lw=2, label='Thermometers')
    title('Fib = {} - ASIC 2'.format(fib))

    return
