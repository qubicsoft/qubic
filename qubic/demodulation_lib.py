from __future__ import division, print_function
from qubicpack import qubicpack as qp
import qubic.fibtools as ft
import qubic.plotters as p
import qubic.lin_lib as ll
import qubic.sb_fitting as sbfit
import qubic
from pysimulators import FitsArray
from qubic.utils import progress_bar

import numpy as np
from matplotlib.pyplot import *
import scipy.ndimage.filters as f
import glob
import scipy.signal as scsig
from scipy import interpolate
import datetime as dt
import sys
import healpy as hp


def printnow(truc):
    print(truc)
    sys.stdout.flush()


def read_cal_src_data(file_list, time_offset=7200):
    ttsrc_i = []
    ddsrc_i = []
    for ff in file_list:
        thett, thedd = np.loadtxt(ff).T
        ttsrc_i.append(thett + time_offset)
        ddsrc_i.append(thedd)

    t_src = np.concatenate(ttsrc_i)
    data_src = np.concatenate(ddsrc_i)
    return t_src, data_src


def bin_per_period(period, time, invec):
    # Bins the vecors in in_list (assumed to be sampled with vector time) per period
    # we label each data sample with a period
    period_index = ((time - time[0]) / period).astype(int)
    # We loop on periods to measure their respective amplitude and azimuth
    allperiods = np.unique(period_index)
    tper = np.zeros(len(allperiods))
    nvec = np.shape(invec)[0]
    nsamples = len(time)
    newvecs = np.zeros((nvec, len(allperiods)))
    for i in range(len(allperiods)):
        ok = (period_index == allperiods[i])
        tper[i] = np.mean(time[ok])
        newvecs[:, i] = np.mean(invec[:, ok], axis=1)
    return tper, newvecs


def hf_noise_estimate(tt, dd):
    sh = np.shape(dd)
    if len(sh) == 1:
        dd = np.reshape(dd, (1, len(dd)))
        ndet = 1
    else:
        ndet = sh[0]
    estimate = np.zeros(ndet)
    for i in range(ndet):
        spectrum_f, freq_f = ft.power_spectrum(tt, dd[i, :], rebin=True)
        mean_level = np.mean(spectrum_f[np.abs(freq_f) > (np.max(freq_f) / 2)])
        samplefreq = 1. / (tt[1] - tt[0])
        estimate[i] = (np.sqrt(mean_level * samplefreq / 2))

    return estimate


def return_rms_period(period, indata, others=None, verbose=False, remove_noise=False):
    """
    Returns the RMS in each period - not such a good proxy for demondulation but robust and does not need the
    modulation signal as an input.
    Main drawback: you end up with a combination of the noise + signal => more an upper-limit to the signal...
    the parameter called others contains a (n, nsamples) array of vectors to be averaged in each new bin (convenient
    if you also have azimuth, elevation or any other stuff...).
    The option remove_noise, if true measures the HF noise in the TODs and removes it from the RMS in order to
    attempt to debias the result from HF noise
    """

    if verbose: print('return_rms_period: indata length=',len(indata))
    time = indata[0]
    data = indata[1]

    if verbose:
        printnow('Entering RMS/period')
    if data.ndim == 1:
        nTES = 1
    else:
        sh = np.shape(data)
        nTES = sh[0]
    if verbose: print('return_rms_period: nTES=',nTES)
    # We label each data sample with a period
    period_index = ((time - time[0]) / period).astype(int)
    # We loop on periods to measure their respective amplitude and azimuth
    allperiods = np.unique(period_index)
    tper = np.zeros(len(allperiods))
    ampdata = np.zeros((nTES, len(allperiods)))
    err_ampdata = np.zeros((nTES, len(allperiods)))
    if others is not None:
        newothers = bin_per_period(period, time, others)
    if verbose:
        printnow('Calculating RMS per period for {} periods and {} TES'.format(len(allperiods), nTES))
    for i in range(len(allperiods)):
        ok = (period_index == allperiods[i])
        tper[i] = np.mean(time[ok])
        if nTES == 1:
            mm, ss = ft.meancut(data[ok], 3)
            ampdata[0, i] = ss
            err_ampdata[0, i] = 1
        else:
            for j in range(nTES):
                mm, ss = ft.meancut(data[j, ok], 3)
                ampdata[j, i] = ss
                err_ampdata[j, i] = 1

    if remove_noise:
        hf_noise = hf_noise_estimate(time, data)
        var_diff = np.zeros((nTES, len(tper)))
        for k in range(nTES):
            var_diff[k, :] = ampdata[k, :] ** 2 - hf_noise[k] ** 2
        ampdata = np.sqrt(np.abs(var_diff)) * np.sign(var_diff)

    if others is None:
        return tper, ampdata, err_ampdata
    else:
        return tper, ampdata, err_ampdata, newothers


def scan2ang_RMS(period, indata, median=True, lowcut=None, highcut=None, verbose=False):
    new_az = np.interp(indata['t_data'], indata['t_azel'], indata['az'])

    # ## Check if filtering is requested
    if (lowcut is None) & (highcut is None):
        dataf = indata['data'].copy()
    else:
        if verbose:
            printnow('Filtering data')
        dataf = ft.filter_data(indata['t_data'], indata['data'], lowcut, highcut)
    if verbose: print('scan2ang_RMS:  dataf.shape=',dataf.shape)
    # ## First get the RMS per period
    if verbose:
        printnow('Resampling Azimuth')
    az = np.interp(indata['t_data'], indata['t_azel'], indata['az'])
    if verbose:
        printnow('Resampling Elevation')
    el = np.interp(indata['t_data'], indata['t_azel'], indata['el'])
    others = np.array([az, el])
    tper, ampdata, err_ampdata, newothers = return_rms_period(period, (indata['t_data'], dataf),
                                                              verbose=verbose, others=others)
    azper = newothers[0]
    elper = newothers[1]
    # ## Convert azimuth to angle
    angle = azper * np.cos(np.radians(elper))
    # ## Fill the return variable for unbinned
    unbinned = {}
    unbinned['t'] = tper
    unbinned['az'] = azper
    unbinned['el'] = elper
    unbinned['az_ang'] = angle
    unbinned['sb'] = ampdata
    unbinned['dsb'] = err_ampdata
    return unbinned


def scan2ang_demod(period, indata, lowcut=None, highcut=None, verbose=False):
    print('scan2ang_demod: type(indata)=',type(indata))
    print('scan2ang_demod: indata.keys()=',indata.keys())
    print("scan2ang_demod: indata['data'].shape=",indata['data'].shape)
    if indata['data'].ndim == 1:
        nTES = 1
    else:
        sh = np.shape(indata['data'])
        nTES = sh[0]

    # ## First demodulate
    demodulated = demodulate_old(indata, 1. / period, verbose=verbose, lowcut=lowcut, highcut=highcut)
    # ## Resample az and el similarly as data
    azd = np.interp(indata['t_data'], indata['t_azel'], indata['az'])
    eld = np.interp(indata['t_data'], indata['t_azel'], indata['el'])

    # ## Resample to one value per modulation period
    if verbose:
        printnow('Resampling to one value per modulation period')
    period_index = ((indata['t_data'] - indata['t_data'][0]) / period).astype(int)
    allperiods = np.unique(period_index)
    newt = np.zeros(len(allperiods))
    newaz = np.zeros(len(allperiods))
    newel = np.zeros(len(allperiods))
    newsb = np.zeros((nTES, len(allperiods)))
    newdsb = np.zeros((nTES, len(allperiods)))
    for i in range(len(allperiods)):
        ok = period_index == allperiods[i]
        newt[i] = np.mean(indata['t_data'][ok])
        newaz[i] = np.mean(azd[ok])
        newel[i] = np.mean(eld[ok])
        newsb[:, i] = np.mean(demodulated[:, ok], axis=1)
        newdsb[:, i] = np.std(demodulated[:, ok], axis=1) / ok.sum()

    unbinned = {}
    unbinned['t'] = newt
    unbinned['az'] = newaz
    unbinned['el'] = newel
    unbinned['az_ang'] = newaz * np.cos(np.radians(newel))
    unbinned['sb'] = newsb
    unbinned['dsb'] = newdsb
    return unbinned


class MySpl:
    def __init__(self, xin, nbspl):
        self.xin = xin
        self.nbspl = nbspl
        self.xspl = np.linspace(np.min(self.xin), np.max(self.xin), self.nbspl)
        F = np.zeros((np.size(xin), self.nbspl))
        self.F = F
        for i in np.arange(self.nbspl):
            self.F[:, i] = self.get_spline(self.xin, i)

    def __call__(self, x, pars):
        theF = np.zeros((np.size(x), self.nbspl))
        for i in np.arange(self.nbspl):
            theF[:, i] = self.get_spline(x, i)
        return np.dot(theF, pars)


class interp_template:
    """
    Allows to interpolate a template applying an amplitude, offset and phase
    This si to be used in order to fit this template in each epriod of some pseudo-periodic signal
    For instance for demodulation
    """
    def __init__(self, xx, yy):
        self.period = xx[-1] - xx[0]
        self.xx = xx
        self.yy = yy

    def __call__(self, x, pars, extra_args=None):
        amp = pars[0]
        off = pars[1]
        dx = pars[2]
        return np.interp(x, self.xx - dx, self.yy, period=self.period) * amp + off


def fitperiod(x, y, fct):
    guess = np.array([np.std(y), np.mean(y), 0.])
    res = ft.do_minuit(x, y, y * 0 + 1, guess, functname=fct, verbose=False, nohesse=True,
                       force_chi2_ndf=True)
    return res


def return_fit_period(period, indata, others=None, verbose=False, template=None):
    """
    Returns the amplitude, offset and phase of a template fit to the data in each period
    the parameter called others contains a (n,nsamples) array of vectors to be averaged in each new bin (convenient
    if you also have azimuth, elevation or any other stuff...).
    """

    time = indata[0]
    data = indata[1]

    if verbose:
        printnow('Entering Fit/period')
    if data.ndim == 1:
        nTES = 1
    else:
        sh = np.shape(data)
        nTES = sh[0]

    if template is None:
        xxtemplate = np.linspace(0, period, 100)
        yytemplate = -np.sin(xxtemplate / period * 2 * np.pi)
        yytemplate /= np.std(yytemplate)
    else:
        xxtemplate = template[0]
        yytemplate = template[1]
        yytemplate /= np.std(yytemplate)
    fct = interp_template(xxtemplate, yytemplate)

    # ## we label each data sample with a period
    period_index = ((time - time[0]) / period).astype(int)
    # ## We loop on periods to measure their respective amplitude and azimuth
    allperiods = np.unique(period_index)
    tper = np.zeros(len(allperiods))
    ampdata = np.zeros((nTES, len(allperiods)))
    err_ampdata = np.zeros((nTES, len(allperiods)))
    if others is not None:
        newothers = bin_per_period(period, time, others)
    if verbose:
        printnow('Performing fit per period for {} periods and {} TES'.format(len(allperiods), nTES))
    for i in range(len(allperiods)):
        ok = (period_index == allperiods[i])
        tper[i] = 0.5 * (np.min(time[ok]) + np.max(time[ok]))  # np.mean(time[ok])
        if nTES == 1:
            res = fitperiod(time[ok], data[ok], fct)
            ampdata[0, i] = res[1][0]
            err_ampdata[0, i] = res[2][0]
        else:
            for j in range(nTES):
                res = fitperiod(time[ok], data[j, ok], fct)
                ampdata[j, i] = res[1][0]
                err_ampdata[j, i] = res[2][0]
    if others is None:
        return tper, ampdata, err_ampdata
    else:
        return tper, ampdata, err_ampdata, newothers


# def demodulate_JC(period, indata, indata_src, others=None, verbose=False, template=None, quadrature=False,
#     remove_noise=False, doplot=False):
#     ### Proper demodulation with quadrature methoid as an option: http://web.mit.edu/6.02/www/s2012/handouts/14.pdf
#     ### In the case of quadrature demodulation, the HF noise RMS/sqrt(2) adds to the demodulated. 
#     ### The option remove_noise=True
#     ### estimates the HF noise in the TODs and removes it from the estimate in order to attempt to debias.
#     time = indata[0]
#     data = indata[1]
#     sh = data.shape
#     if len(sh)==1:
#         data = np.reshape(data, (1,sh[0]))
#     time_src = indata_src[0]
#     data_src = indata_src[1]
#     #print(quadrature)
#     if quadrature==True:
#         ### Shift src data by 1/2 period
#         data_src_shift = np.interp(time_src-period/2, time_src, data_src, period=period)
#         demod = (np.sqrt((data*data_src)**2 + (data*data_src_shift)**2))/np.sqrt(2)
#     else:
#         demod = data*data_src

#     ### Now smooth over a period
#     import scipy.signal as scsig
#     FREQ_SAMPLING = 1./(time[1]-time[0])
#     size_period = int(FREQ_SAMPLING * period) + 1
#     filter_period = np.ones((size_period,)) / size_period
#     demodulated = np.zeros_like(demod)
#     sh=np.shape(demod)
#     bar=progress_bar(sh[0], 'Filtering Detectors: ')
#     for i in range(sh[0]):
#         bar.update()
#         demodulated[i,:] = scsig.fftconvolve(demod[i,:], filter_period, mode='same')

#     # Remove First and last periods
#     nper = 4.
#     nsamples = int(nper * period / (time[1]-time[0]))
#     timereturn = time[nsamples:-nsamples]
#     demodulated = demodulated[:, nsamples:-nsamples]

#     if remove_noise:
#         hf_noise = hf_noise_estimate(time, data)/np.sqrt(2)
#         var_diff = np.zeros((sh[0], len(timereturn)))
#         for k in range(sh[0]):
#             var_diff[k,:] = demodulated[k,:]**2 - hf_noise[k]**2
#         demodulated = np.sqrt(np.abs(var_diff))*np.sign(var_diff)


#     if doplot:
#         sh = np.shape(data)
#         if sh[0] > 1:
#             thetes = 95
#         else:
#             thetes = 0

#         clf()
#         subplot(2,1,1)
#         plot(time-time[0], renorm(data[thetes,:]), label='Data TES {}'.format(thetes+1))
#         plot(time-time[0], renorm(data_src),label='Src used for demod')
#         plot(time-time[0], renorm(demod[thetes,:]),label='Demod signal')
#         plot(timereturn-time[0], renorm(demodulated[thetes,:]),label='Demod Low-passed')
#         legend(loc='lower right')
#         subplot(2,1,2)
#         plot(time-time[0], renorm(data[thetes,:])+5, label='Data TES {}'.format(thetes+1))
#         plot(time-time[0], renorm(data_src)+5,label='Src used for demod')
#         plot(time-time[0], renorm(demod[thetes,:]),label='Demod signal')
#         plot(timereturn-time[0], renorm(demodulated[thetes,:]),label='Demod Low-passed')
#         xlim(30,70)
#         legend(loc='lower right')
#         show()
#         # stop

#     if sh[0]==1:
#         demodulated = demodulated[0,:]
#     return timereturn, demodulated, demodulated*0+1

def demodulate_JC(period, indata, indata_src, others=None, verbose=False, template=None, quadrature=False,
                  remove_noise=False, doplot=False):
    """
    Proper demodulation with quadrature method as an option: http://web.mit.edu/6.02/www/s2012/handouts/14.pdf
    In the case of quadrature demodulation, the HF noise RMS/sqrt(2) adds to the demodulated.
    The option remove_noise=True
    estimates the HF noise in the TODs and removes it from the estimate in order to attempt to debias.
    """
    time = indata[0]
    data = indata[1]
    sh = data.shape
    if len(sh) == 1:
        data = np.reshape(data, (1, sh[0]))
    time_src = indata_src[0]
    data_src = indata_src[1]
    # print(quadrature)
    if quadrature:
        # ## Shift src data by 1/2 period
        data_src_shift = np.interp(time_src - period / 2, time_src, data_src, period=period)

    # ## Now smooth over a period
    import scipy.signal as scsig
    FREQ_SAMPLING = 1. / (time[1] - time[0])
    size_period = int(FREQ_SAMPLING * period) + 1
    filter_period = np.ones((size_period,)) / size_period
    demodulated = np.zeros_like(data)
    sh = np.shape(data)
    # bar=progress_bar(sh[0], 'Filtering Detectors: ')
    for i in range(sh[0]):
        # bar.update()
        if quadrature:
            demodulated[i, :] = scsig.fftconvolve(
                (np.sqrt((data[i, :] * data_src) ** 2 + (data[i, :] * data_src_shift) ** 2)) / np.sqrt(2),
                filter_period, mode='same')
        else:
            demodulated[i, :] = scsig.fftconvolve(data[i, :] * data_src,
                                                  filter_period, mode='same')

    # Remove First and last periods
    nper = 4.
    nsamples = int(nper * period / (time[1] - time[0]))
    timereturn = time[nsamples:-nsamples]
    demodulated = demodulated[:, nsamples:-nsamples]

    if remove_noise:
        hf_noise = hf_noise_estimate(time, data) / np.sqrt(2)
        var_diff = np.zeros((sh[0], len(timereturn)))
        for k in range(sh[0]):
            var_diff[k, :] = demodulated[k, :] ** 2 - hf_noise[k] ** 2
        demodulated = np.sqrt(np.abs(var_diff)) * np.sign(var_diff)

    if doplot:
        sh = np.shape(data)
        if sh[0] > 1:
            thetes = 95
        else:
            thetes = 0

        clf()
        subplot(2, 1, 1)
        plot(time - time[0], renorm(data[thetes, :]), label='Data TES {}'.format(thetes + 1))
        plot(time - time[0], renorm(data_src), label='Src used for demod')
        plot(time - time[0], renorm(demod[thetes, :]), label='Demod signal')
        plot(timereturn - time[0], renorm(demodulated[thetes, :]), label='Demod Low-passed')
        legend(loc='lower right')
        subplot(2, 1, 2)
        plot(time - time[0], renorm(data[thetes, :]) + 5, label='Data TES {}'.format(thetes + 1))
        plot(time - time[0], renorm(data_src) + 5, label='Src used for demod')
        plot(time - time[0], renorm(demod[thetes, :]), label='Demod signal')
        plot(timereturn - time[0], renorm(demodulated[thetes, :]), label='Demod Low-passed')
        xlim(30, 70)
        legend(loc='lower right')
        show()
        # stop

    if sh[0] == 1:
        demodulated = demodulated[0, :]
    return timereturn, demodulated, demodulated * 0 + 1


def demodulate_methods(data_in, fmod, fourier_cuts=None, verbose=False, src_data_in=None, method='demod',
                       others=None, template=None, remove_noise=False):
    # Various demodulation methods
    # Others is a list of other vectors (with similar time sampling as the data to demodulate)
    # that we need to sample the same way as the data
    if fourier_cuts is None:
        # Duplicate the input data
        data = data_in.copy()
        if src_data_in is None:
            src_data = None
        else:
            src_data = src_data_in.copy()
    else:
        # Filter data and source accordingly
        lowcut = fourier_cuts[0]
        highcut = fourier_cuts[1]
        notch = fourier_cuts[2]
        newtod = ft.filter_data(data_in[0], data_in[1], lowcut, highcut,
                                notch=notch, rebin=True, verbose=verbose)
        data = [data_in[0], newtod]
        if src_data_in is None:
            src_data = None
        else:
            new_src_tod = ft.filter_data(src_data_in[0], src_data_in[1], lowcut, highcut,
                                         notch=notch, rebin=True, verbose=verbose)
            src_data = [src_data_in[0], new_src_tod]

            # Now we have the "input" data, we can start demodulation
    period = 1. / fmod
    if method == 'rms':
        # RMS method: calculate the RMS in each period (beware ! it returns noise+signal !)
        return return_rms_period(period, data, others=others, verbose=verbose,
                                 remove_noise=remove_noise)
    elif method == 'fit':
        return return_fit_period(period, data, others=others, verbose=verbose, template=template)
    elif method == 'demod':
        return demodulate_JC(period, data, src_data, others=others, verbose=verbose, template=None)
    elif method == 'demod_quad':
        return demodulate_JC(period, data, src_data, others=others, verbose=verbose, template=None,
                             quadrature=True, remove_noise=remove_noise)
    elif method == 'absolute_value':
        return np.abs(data)


def demodulate_old(indata, fmod, lowcut=None, highcut=None, verbose=False):
    printnow('Starting Demodulation')
    if indata['data'].ndim == 1:
        nTES = 1
    else:
        sh = np.shape(indata['data'])
        nTES = sh[0]

    if np.array(indata['t_src']).ndim == 0:
        printnow('Source Data is not there: output is zero')
        demodulated = np.zeros((nTES, len(indata['t_data'])))
    else:
        # Check if filtering is requested
        if (lowcut is None) & (highcut is None):
            dataf = indata['data'].copy()
            new_src = np.interp(indata['t_data'], indata['t_src'], indata['data_src'])
        else:
            if verbose: printnow('Filtering data and Src Signal')
            dataf = ft.filter_data(indata['t_data'], indata['data'], lowcut, highcut)
            new_src = ft.filter_data(indata['t_data'], np.interp(indata['t_data'], indata['t_src'], indata['data_src']),
                                     lowcut, highcut)

        if nTES == 1:
            dataf = np.reshape(dataf, (1, len(indata['data'])))

        # Make the product for demodulation with changing sign for data
        product = -dataf * new_src
        # Smooth it over a period
        ppp = 1. / fmod
        FREQ_SAMPLING = 1. / ((np.max(indata['t_data']) - np.min(indata['t_data'])) / len(indata['t_data']))
        size_period = int(FREQ_SAMPLING * ppp) + 1
        filter_period = np.ones((nTES, size_period,)) / size_period
        demodulated = scsig.fftconvolve(product, filter_period, mode='same', axes=1)
    return demodulated


def fit_ang_gauss(ang, data, errdata=None, cut=None):
    # Guess for the peak location
    datar = (data - np.min(data))
    ok = np.isfinite(datar)
    datar = datar / np.sum(datar[ok])
    peak_guess = np.sum(ang[ok] * (datar[ok]))
    guess = np.array([peak_guess, 1.5, np.max(data[ok]) - np.min(data[ok]), np.median(data[ok])])
    # If no errors were provided then use ones
    if errdata is None:
        errdata = np.ones(len(data))
    # if a cut is needed for the top of the peak
    if cut is None:
        ok = (ang > -1e100) & (errdata > 0)
    else:
        ok = (data < cut) & (errdata > 0)
    # Do the fit now
    res = ft.do_minuit(ang[ok], data[ok], errdata[ok], guess, functname=gauss, verbose=False, nohesse=True,
                       force_chi2_ndf=True)
    return res[1], res[2]


def gauss(x, par):
    return par[3] + par[2] * np.exp(-0.5 * (x - par[0]) ** 2 / (par[1] / 2.35) ** 2)


def gauss_line(x, par):
    return par[4] * x + par[3] + par[2] * np.exp(-0.5 * (x - par[0]) ** 2 / (par[1] / 2.35) ** 2)


class MySpl:
    """
    A spline class to be used for fitting
    Example:
    npts = 55
    xx = np.random.rand(npts)*10
    yy = 2*xx**3 - 1*xx**2 +7*xx - 1
    dy = np.zeros(npts)+30
    errorbar(xx, yy,yerr=dy, fmt='ro')

    xxx = np.linspace(0,10,1000)
    nx = 10
    myspl = MySpl(xxx,nx)

    guess = guess=np.zeros(myspl.nbspl)
    res = ft.do_minuit(xx, yy, dy, guess, 
                   functname=myspl, verbose=False,nohesse=True, force_chi2_ndf=False)

    plot(xxx, myspl(xxx, res[1]))
    """

    def __init__(self, xin, nbspl):
        self.xin = xin
        self.nbspl = nbspl
        self.xspl = np.linspace(np.min(self.xin), np.max(self.xin), self.nbspl)
        F = np.zeros((np.size(xin), self.nbspl))
        self.F = F
        for i in np.arange(self.nbspl):
            self.F[:, i] = self.get_spline(self.xin, i)

    def __call__(self, x, pars):
        theF = np.zeros((np.size(x), self.nbspl))
        for i in np.arange(self.nbspl): theF[:, i] = self.get_spline(x, i)
        return (np.dot(theF, pars))

    def get_spline(self, xx, index):
        yspl = np.zeros(np.size(self.xspl))
        yspl[index] = 1.
        tck = interpolate.splrep(self.xspl, yspl)
        yy = interpolate.splev(xx, tck, der=0)
        return yy


class SimSrcTOD:
    """
    Class to simulate the TOD signal when modulating with the source
    The main signal will be that of the source (simulated)
    It is modulated by a slowly varying spline for amplitude, offset and phase 
    each with a given number of spline nodes
    """

    def __init__(self, xin, pars_src, nbspl_amp, nbspl_offset, nbspl_phase):
        self.xin = xin
        self.nbspl_amp = nbspl_amp
        self.nbspl_offset = nbspl_offset
        self.nbspl_phase = nbspl_phase
        ### Splines for each of amplitude, offset and phase
        self.spl_amp = MySpl(xin, nbspl_amp)
        self.spl_offset = MySpl(xin, nbspl_offset)
        self.spl_phase = MySpl(xin, nbspl_phase)
        ### Source parameters: 0=amp, 1=mod_freq, 2=offset
        self.pars_src = pars_src

    def amplitude(self, x, pars):
        # ## Amplitude function
        pars_amp = pars[0:self.nbspl_amp]
        amp = self.spl_amp(x, pars_amp)
        return amp

    def offset(self, x, pars):
        # ## Offset function
        pars_offset = pars[self.nbspl_amp:self.nbspl_amp + self.nbspl_offset]
        offset = self.spl_offset(x, pars_offset)
        return offset

    def phase(self, x, pars):
        # ## Phase function
        pars_phase = pars[self.nbspl_amp + self.nbspl_offset:self.nbspl_amp + self.nbspl_offset + self.nbspl_phase]
        phase = self.spl_phase(x, pars_phase)
        return phase

    def __call__(self, x, pars):
        amp = self.amplitude(x, pars)
        offset = self.offset(x, pars)
        phase = self.phase(x, pars)
        ### Source input signal: 0=amp, 1=mod_freq, 2=offset
        input_src = ll.sim_generator_power(x, self.pars_src[0], self.pars_src[2], self.pars_src[1], phase) - 0.5

        ### Now modulate with amplitude and offset
        return amp * input_src + offset


def scan2ang_splfit(period, time, data, t_src, src, t_az, az, lowcut, highcut, elevation, nbins=150, superbinning=1.,
                    doplot=False):
    # ## Filter Data and Source Signal the same way + change sign of data
    new_data = -ft.filter_data(time, data, lowcut, highcut)
    new_src = ft.filter_data(time, np.interp(time, t_src, src), lowcut, highcut)

    # ## Now resample data into bins such that the modulation period is well sampled
    # ## We want bins with size < period/4
    approx_binsize = period / 4 / superbinning
    nbins_new = int((time[-1] - time[0]) / approx_binsize)
    print('Number of initial bins in data: {}'.format(len(data)))
    print('Number of new bins in data: {}'.format(nbins_new))
    x_data, newdata, dx, dy = ft.profile(time, new_data, nbins=nbins_new, plot=False)
    new_az = np.interp(x_data, t_az, az)

    # ### Source parameters
    src_amp = 5.  # Volts
    src_period = period  # seconds
    src_phase = 0  # Radians
    src_offset = 2.5  # Volts
    pars_src = np.array([src_amp, 1. / src_period, src_offset])

    # nbspl_amp = 20
    # nbspl_offset = 20
    # nbspl_phase = 4
    delta_az = np.max(az) - np.min(az)
    nbspl_amp = int(delta_az * 2)
    nbspl_offset = int(delta_az * 2)
    nbspl_phase = 4
    print('Number of Splines for Amplitude: {}'.format(nbspl_amp))
    print('Number of Splines for Offset   : {}'.format(nbspl_offset))
    print('Number of Splines for Phase    : {}'.format(nbspl_phase))

    simsrc = SimSrcTOD(x_data, pars_src, nbspl_amp, nbspl_offset, nbspl_phase)

    guess = np.concatenate((np.ones(nbspl_amp), np.zeros(nbspl_offset), np.zeros(nbspl_phase)))

    res = ft.do_minuit(x_data, newdata, np.ones(len(newdata)), guess, functname=simsrc, verbose=False, nohesse=True,
                       force_chi2_ndf=False)

    if doplot:
        subplot(4, 1, 1)
        plot(new_az * np.cos(np.radians(elevation)), newdata, label='Data')
        plot(new_az * np.cos(np.radians(elevation)), simsrc(x_data, res[1]), label='Fit')
        plot(new_az * np.cos(np.radians(elevation)), new_az * 0, 'k--')
        legend()
        subplot(4, 1, 2)
        plot(new_az * np.cos(np.radians(elevation)), simsrc.amplitude(x_data, res[1]))
        plot(new_az * np.cos(np.radians(elevation)), new_az * 0, 'k--')
        title('Amplitude')
        subplot(4, 1, 3)
        plot(new_az * np.cos(np.radians(elevation)), simsrc.offset(x_data, res[1]))
        plot(new_az * np.cos(np.radians(elevation)), new_az * 0, 'k--')
        title('Offset')
        subplot(4, 1, 4)
        plot(new_az * np.cos(np.radians(elevation)), simsrc.phase(x_data, res[1]))
        plot(new_az * np.cos(np.radians(elevation)), new_az * 0, 'k--')
        ylim(-np.pi, np.pi)
        title('Phase')
        tight_layout()

    azvals = np.linspace(np.min(az), np.max(az), nbins)
    ang = azvals * np.cos(np.radians(elevation))
    sb = np.interp(ang, new_az * np.cos(np.radians(elevation)), simsrc.amplitude(x_data, res[1]))
    dsb = np.ones(nbins)
    return azvals, ang, sb, dsb


def general_demodulate(period, indata, lowcut, highcut, nbins=150, median=True, method='demod', verbose=False,
                       doplot=True, rebin=False, cut=None, label=None, renormalize_plot=True):
    # Call one of the methods
    if method == 'demod':
        if verbose:
            printnow('Demodulation Method')
        unbinned = scan2ang_demod(period, indata, verbose=verbose, lowcut=lowcut, highcut=highcut)
    elif method == 'rms':
        if verbose:
            printnow('RMS Method')
        unbinned = scan2ang_RMS(period, indata, verbose=verbose, median=median, lowcut=lowcut, highcut=highcut)
    # elif method == 'splfit':
    #     azbins, elbins, angle, sb, dsb = scan2ang_splfit(period, time, data, t_src, src, 
    #                                      t_az, az, lowcut, highcut, elevation, 
    #                                     nbins=nbins, superbinning=1., doplot=False)

    if rebin:
        # Now rebin the data
        if verbose:
            printnow('Now rebin the data')
        if indata['data'].ndim == 1:
            sh = [1, len(indata['data'])]
        else:
            sh = np.shape(indata['data'])
        ang = np.zeros(nbins)
        sb = np.zeros((sh[0], nbins))
        dsb = np.zeros((sh[0], nbins))
        others = np.zeros((nbins, 2))
        for i in range(sh[0]):
            if verbose:
                if (16 * (i / 16)) == i:
                    printnow('Rebinning TES {} over {}'.format(i, sh[0]))
            ang, sb[i, :], dang, dsb[i, :], others = ft.profile(unbinned['az_ang'],
                                                                unbinned['sb'][i, :], nbins=nbins, plot=False,
                                                                dispersion=True, log=False, median=median,
                                                                cutbad=False, rebin_as_well=[unbinned['az'],
                                                                                             unbinned['el']])
        binned = {}
        binned['az'] = others[:, 0]
        binned['el'] = others[:, 1]
        binned['az_ang'] = ang
        binned['sb'] = sb
        binned['dsb'] = dsb
    else:
        binned = None

    # If requested plot the result
    if doplot:
        if verbose:
            printnow('Plot')
        toplot = unbinned
        if rebin:
            toplot = binned
        if label is None:
            label = method
        if indata['data'].ndim == 1:
            sh = [1, len(indata['data'])]
        else:
            sh = np.shape(indata['data'])
        for i in range(sh[0]):
            errorbar(toplot['az'], toplot['sb'][i, :], yerr=toplot['dsb'][i, :], fmt='.-',
                     label=label + ' {}'.format(i))
        legend()

    return unbinned, binned


def array_info(ar):
    mini = np.min(ar)
    maxi = np.max(ar)
    return 'Shape = {0:15} Min={1:6.3g} Max={2:6.3g} Rng={3:6.3g} Mean={4:6.3g} Std={5:6.3g}'.format(str(np.shape(ar)),
                                                                                                     mini, maxi,
                                                                                                     maxi - mini,
                                                                                                     np.mean(ar),
                                                                                                     np.std(ar))


def read_data_azel_src(dirfile, AsicNum, TESNum=None, calsource_dir='/qubic/Data/Calib-TD/calsource/', verbose=False):
    a = qp()
    a.read_qubicstudio_dataset(dirfile, asic=AsicNum)
    # ############################## TES DATA ###################################
    # Get TES data: if TESNum is None, will return the whole ASIC array (128)
    if TESNum:
        data = a.timeline(TES=TESNum)
    else:
        data = a.timeline_array()
    # Get TES timestamps
    t_data = a.timeline_timeaxis(axistype='pps')
    # ###########################################################################

    # ############################## Az and El data #############################
    az = a.azimuth()
    el = a.elevation()
    t_azel = a.timeaxis(datatype='hk', axistype='pps')
    # ###########################################################################

    # ############################## Cal Src Data ###############################
    # ### First need to check wether there is a calsrouce data file or not
    # ### and then read the file if it exists
    data_time = dt.datetime.utcfromtimestamp(a.hk['ASIC_SUMS']['ComputerDate'][0])
    glob_pattern = data_time.strftime('calsource_%Y%m%dT%H%M*.dat')
    bla = glob.glob(calsource_dir + glob_pattern)
    if bla == []:
        if verbose:
            print('No CalSource file found corresponding to this dataset: ' + dirfile)
        t_src = -1
        data_src = -1
    else:
        if verbose:
            print('Found Calibration Source date in: ' + bla[0])
        t_src, data_src = read_cal_src_data(bla)

    # ### Now return everything
    if verbose:
        print('Returning:')
        print('   t_data  : ', array_info(t_data))
        print('   data    : ', array_info(data))
        print('   t_azel  : ', array_info(t_azel))
        print('   az      : ', array_info(az))
        print('   el      : ', array_info(el))
        print('   t_src   : ', array_info(t_src))
        print('   data_src: ', array_info(data_src))

    retval = {}
    retval['t_data'] = t_data
    retval['data'] = data
    retval['t_azel'] = t_azel
    retval['az'] = az
    retval['el'] = el
    retval['t_src'] = t_src
    retval['data_src'] = data_src
    return retval


def renorm(ar):
    return (ar - np.mean(ar)) / np.std(ar)


def bin_image_elscans(x, y, data, xr, nx, TESIndex):
    ny = len(y)
    mapsum = np.zeros((nx, ny))
    mapcount = np.zeros((nx, ny))
    for i in range(ny):
        thex = x[i]
        dd = data[i] - np.mean(data[i], axis=0)
        idx = ((thex - xr[0]) / (xr[1] - xr[0]) * nx).astype(int)
        for j in range(len(thex)):
            if ((idx[j] >= 0) & (idx[j] < nx)):
                mapsum[idx[j], i] += dd[TESIndex, j]
                mapcount[idx[j], i] += 1.
    mapout = np.zeros((nx, ny))
    ok = mapcount > 0
    mapout[ok] = mapsum[ok] / mapcount[ok]
    xx = np.linspace(xr[0], xr[1], nx + 1)
    xx = 0.5 * (xx[1:] + xx[:-1])
    mm, ss = ft.meancut(mapout[ok], 3)
    mapout[ok] -= mm
    return np.flip(mapout.T, axis=(0, 1)), xx, y


def scan2hpmap(ns, azdeg, eldeg, data):
    coadd = np.zeros(12 * ns ** 2)
    count = np.zeros(12 * ns ** 2)
    ip = hp.ang2pix(ns, np.pi / 2 - np.radians(eldeg), np.radians(azdeg))
    for i in range(len(azdeg)):
        coadd[ip[i]] += data[i]
        count[ip[i]] += 1
    ok = count != 0
    sbmap = np.zeros(12 * ns ** 2)
    sbmap[ok] = coadd[ok] / count[ok]
    mm, ss = ft.meancut(sbmap[ok], 3)
    sbmap[ok] -= mm
    sbmap[~ok] = 0
    return sbmap


def make_tod(scans, axis=1):
    tod = scans[0]
    for i in np.arange(len(scans) - 1) + 1:
        tod = np.concatenate((tod, scans[i]), axis=axis)
    return tod


def CalSrcPower_Vs_Nu(freq):
    # ## Taken from Tx 263 130-170GHz User Guide available on Atrium
    ff = np.array([129.93109059, 130.93364949, 131.93662559, 132.93897589,
                   133.94132618, 134.94430228, 135.94686118, 136.95004589,
                   137.90587712, 138.90927044, 139.91203794, 140.91501405,
                   141.91882458, 142.92326092, 143.92686285, 144.92942175,
                   145.93218925, 146.93537396, 147.93876728, 148.99034851,
                   149.94597113, 150.94853003, 151.95171474, 153.00350458,
                   153.9593358, 154.96335494, 155.96695686, 156.97076739,
                   157.9259728, 158.92790588, 159.97948711, 160.98371485,
                   161.98794259, 162.94419103, 163.99702389, 165.00292048,
                   165.91431869, 167.01429644, 167.92068812, 168.92387283,
                   169.97399383])
    power_mW = np.array([12.83207659, 13.22505174, 13.35869985, 13.90693552, 14.47767056,
                         14.62397723, 15.07182833, 15.07182833, 14.77176243, 14.62397723,
                         14.9210411, 15.07182833, 14.62397723, 13.76780255, 13.49369857,
                         13.90693552, 14.18943378, 14.18943378, 14.04747453, 13.63006154,
                         13.49369857, 13.90693552, 13.90693552, 13.35869985, 13.09274073,
                         12.57660204, 12.32621374, 11.95994707, 12.08081044, 12.83207659,
                         12.45077849, 11.84029289, 11.25974057, 10.81586195, 9.88006554,
                         8.66944441, 7.23416893, 6.81063762, 7.23416893, 7.23416893,
                         7.53105632])
    return np.interp(freq, ff, power_mW)


def CalSrcPowerMeterResponse_Vs_Nu(freq):
    # ## Taken from Tx 263 130-170GHz User Guide available on Atrium
    # ## It ic=ncludes both output response and power meter response
    ff = np.array([129.96075353, 130.90266876, 131.89167975, 133.02197802,
                   133.96389325, 134.90580848, 135.89481947, 136.97802198,
                   137.96703297, 139.00313972, 139.89795918, 141.02825746,
                   141.97017268, 143.00627943, 144.04238619, 144.93720565,
                   145.9733124, 147.00941915, 147.95133438, 148.98744113,
                   150.02354788, 150.96546311, 151.90737834, 153.03767661,
                   153.97959184, 154.92150706, 155.95761381, 156.99372057,
                   157.93563579, 158.97174254, 159.91365777, 160.99686028,
                   161.98587127, 162.9277865, 163.96389325, 164.95290424,
                   165.89481947, 166.97802198, 167.91993721, 168.95604396,
                   169.94505495])
    response = np.array([1.48598131, 1.56133453, 1.67410027, 1.80564562, 1.91838202,
                         1.97504365, 2.0691178, 2.16325063, 2.0704089, 2.22058716,
                         2.07161197, 2.18446573, 2.07290306, 2.07354861, 2.07419416,
                         2.22428439, 2.33707948, 2.46856615, 2.52522778, 2.54456491,
                         2.63866841, 2.88224592, 3.0323655, 3.03306973, 3.2018809,
                         3.16508458, 3.16573013, 3.27852521, 3.63425226, 4.10218753,
                         4.70090524, 4.85111284, 4.47789727, 4.68409161, 4.62866239,
                         4.34890477, 3.88220191, 3.80811045, 4.1638375, 4.4074737,
                         4.66977215])
    return np.interp(freq, ff, response)


def dB(y):
    negative = y <= 0
    bla = 10 * np.log10(y / np.max(y))
    bla[negative] = -50
    return bla


def get_spectral_response(name, freqs, allmm, allss, nsig=3, method='demod', TESNum=None,
                          directory='/Users/hamilton/Qubic/Calib-TD/SpectralResponse/'):
    # Restore the data already treated
    allmm = FitsArray(directory + '/allmm_' + method + '_' + name + '.fits')
    allss = FitsArray(directory + '/allss_' + method + '_' + name + '.fits')
    freqs = FitsArray(directory + '/freqs_' + method + '_' + name + '.fits')

    # Correct for Source Characteristics
    if method == 'rms':
        # Then the analysis does not use the power meter data and we only need to correct for the output power
        allmm /= (CalSrcPower_Vs_Nu(freqs))
        allss /= (CalSrcPower_Vs_Nu(freqs))
    else:
        # In the case of demod we need to correct for both the power_meter response and the output power
        # This is done using the function below
        allmm /= (CalSrcPowerMeterResponse_Vs_Nu(freqs))
        allss /= (CalSrcPowerMeterResponse_Vs_Nu(freqs))

    sh = np.shape(allmm)
    nTES = sh[0]

    # Normalize all TES to the same integral
    allfnorm = np.zeros((256, len(freqs)))
    allsnorm = np.zeros((256, len(freqs)))
    infilter = (freqs >= 124) & (freqs <= 182)
    outfilter = ~infilter
    for tesindex in range(256):
        baseline = np.mean(allmm[tesindex, outfilter])
        integ = np.sum(allmm[tesindex, infilter] - baseline)
        allfnorm[tesindex, :] = (allmm[tesindex, :] - baseline) / integ
        allsnorm[tesindex, :] = allss[tesindex, :] / integ

    # If only one TES is requested we return it. Otherwise we need to do an average over nicely
    # Behaving TES...
    if TESNum is not None:
        return freqs, allfnorm[TESNum - 1, :] - np.min(allfnorm[TESNum - 1, :]), allsnorm[TESNum - 1, :]
    else:
        # Discriminant Variable: a chi2 we want it to be bad in the sense that
        # we want the spectrum to be inconsistent with a straight line inside the QUBIC band
        discrim = np.nansum(allfnorm[:, infilter] ** 2 / allsnorm[:, infilter] ** 2, axis=1)
        mr, sr = ft.meancut(discrim, 3)
        threshold = mr + nsig * sr
        ok = (discrim > threshold)
        print('Spectral Response calculated over {} TES'.format(ok.sum()))
        filtershape = np.zeros(len(freqs))
        errfiltershape = np.zeros(len(freqs))
        for i in range(len(freqs)):
            filtershape[i], errfiltershape[i] = ft.meancut(allfnorm[ok, i], 2)
        # errfiltershape /= np.sqrt(ok.sum())
        # Then remove the smallest value in order to avoid negative values
        filtershape -= np.min(filtershape)
        return freqs, filtershape, errfiltershape


def qubic_sb_model(x, pars, return_peaks=False):
    x2d = x[0]
    y2d = x[1]
    xc = pars[0]
    yc = pars[1]
    dist = pars[2]
    angle = pars[3]
    distx = pars[4]
    disty = pars[5]
    ampgauss = pars[6]
    xcgauss = pars[7]
    ycgauss = pars[8]
    fwhmgauss = pars[9]
    fwhmpeaks = pars[10]
    nrings = 2
    amps = np.zeros(9)
    npeaks_tot = len(amps)
    npeaks_line = int(np.sqrt(npeaks_tot))
    nrings = (npeaks_line - 1) / 2 + 1
    x = (np.arange(npeaks_line) - (nrings - 1)) * dist
    xx, yy = np.meshgrid(x, x)
    xxyy = np.array([np.ravel(xx), np.ravel(yy)])
    cosang = np.cos(np.radians(angle))
    sinang = np.sin(np.radians(angle))
    rotmat = np.array([[cosang, -sinang], [sinang, cosang]])
    newxxyy = np.zeros((4, 9))
    for i in range(npeaks_tot):
        thexxyy = np.dot(rotmat, xxyy[:, i])
        newxxyy[0, i] = thexxyy[0]
        newxxyy[1, i] = thexxyy[1]
        # newxxyy.append(thexxyy)
    # newxxyy =  np.array(newxxyy).T

    newxxyy[0, :] += distx * (newxxyy[1, :]) ** 2
    newxxyy[1, :] += disty * (newxxyy[0, :]) ** 2
    newxxyy[0, :] += xc
    newxxyy[1, :] += yc

    themap = np.zeros_like(x2d)
    for i in range(npeaks_tot):
        amps[i] = ampgauss * np.exp(
            -0.5 * ((xcgauss - newxxyy[0, i]) ** 2 + (ycgauss - newxxyy[1, i]) ** 2) / (fwhmgauss / 2.35) ** 2)
        themap += amps[i] * np.exp(
            -((x2d - newxxyy[0, i]) ** 2 + (y2d - newxxyy[1, i]) ** 2) / (2 * (fwhmpeaks / 2.35) ** 2))
    newxxyy[2, :] = amps
    newxxyy[3, :] = fwhmpeaks

    # satpix = themap >= saturation
    # themap[satpix] = saturation

    if return_peaks:
        return themap, newxxyy
    else:
        return themap


def flattened_qubic_sb_model(x, pars):
    return np.ravel(qubic_sb_model(x, pars))


def fit_sb(TESNum, dirfiles, scaling=140e3, newsize=70, dmax=5., az_center=0., el_center=50., doplot=False,
           vmin=None, vmax=None, resample=True):
    # Read flat maps
    flatmap_init, az_init, el_init = sbfit.get_flatmap(TESNum, dirfiles)

    if resample:
        # Resample input map to have less pixels to deal with for fitting
        flatmap = scsig.resample(scsig.resample(flatmap_init, newsize, axis=0), newsize, axis=1)
        delta_az = np.median(az_init - np.roll(az_init, 1))
        delta_el = np.median(el_init - np.roll(el_init, 1))
        az = np.linspace(np.min(az_init) - delta_az / 2, np.max(az_init) + delta_az / 2, newsize)
        el = np.linspace(np.min(el_init) - delta_el / 2, np.max(el_init) + delta_el / 2, newsize)
    else:
        flatmap = flatmap_init
        az = az_init
        el = el_init
    az2d, el2d = np.meshgrid(az * np.cos(np.radians(50)), np.flip(el))

    # ## First find the location of the maximum close to the center
    distance_max = dmax
    mask = (np.sqrt((az2d - az_center) ** 2 + (el2d - el_center) ** 2) < distance_max).astype(int)
    wmax = np.where((flatmap * mask) == np.max(flatmap * mask))
    maxval = flatmap[wmax][0]
    print('Maximum of map is {0:5.2g} and was found at: az={1:5.2f}, el={2:5.2f}'.format(maxval,
                                                                                         az2d[wmax][0], el2d[wmax][0]))

    # ## Now fit all parameters
    x = [az2d, el2d]
    parsinit = np.array([az2d[wmax][0], el2d[wmax][0], 8.3, 44., 0., 0.009, maxval / scaling, 0., 50., 13., 1.])
    rng = [[az2d[wmax][0] - 1., az2d[wmax][0] + 1.],
           [el2d[wmax][0] - 1., el2d[wmax][0] + 1.],
           [8., 8.75],
           [43., 47.],
           [-0.02, 0.02],
           [-0.02, 0.02],
           [0, 1000],
           [-3, 3],
           [47., 53],
           [10., 16.],
           [0.5, 1.5]]
    fit = ft.do_minuit(x, np.ravel(flatmap / scaling), np.ones_like(np.ravel(flatmap)), parsinit,
                       functname=flattened_qubic_sb_model, chi2=ft.MyChi2_nocov, rangepars=rng,
                       force_chi2_ndf=True)
    themap, newxxyy = qubic_sb_model(x, fit[1], return_peaks=True)

    if doplot:
        rc('figure', figsize=(18, 4))
        parfit = fit[1]
        sh = np.shape(newxxyy)
        print(sh)
        subplot(1, 2, 1)
        imshow(flatmap / scaling, extent=[np.min(az) * np.cos(np.radians(50)),
                                          np.max(az) * np.cos(np.radians(50)),
                                          np.min(el), np.max(el)],
               vmin=vmin, vmax=vmax)
        colorbar()
        for i in range(sh[1]):
            ax = plot(newxxyy[0, i], newxxyy[1, i], 'r.')
        title('Input Map - TES #{}'.format(TESNum))
        xlabel('Angle in Az direction [deg.]')
        ylabel('Elevation [deg.]')

        subplot(1, 2, 2)
        imshow(flatmap / scaling - themap, extent=[np.min(az) * np.cos(np.radians(50)),
                                                   np.max(az) * np.cos(np.radians(50)),
                                                   np.min(el), np.max(el)],
               vmin=vmin, vmax=vmax)
        colorbar()
        title('Residual Map - TES #{}'.format(TESNum))
        xlabel('Angle in Az direction [deg.]')
        ylabel('Elevation [deg.]')

    fit[1][6] *= scaling
    fit[2][6] *= scaling
    # fit[1][11] *= scaling
    # fit[2][11] *= scaling
    return flatmap_init, az_init, el_init, fit, newxxyy


def mygauss2d(x2d, y2d, center, sx, sy, rho):
    sh = np.shape(x2d)
    xx = np.ravel(x2d) - center[0]
    yy = np.ravel(y2d) - center[1]
    z = (xx / sx) ** 2 - 2 * rho * xx * yy / sx / sy + (yy / sy) ** 2
    return np.reshape(np.exp(-z / (2 * (1 - rho ** 2))), sh)


def qubic_sb_model_asym(x, pars, return_peaks=False):
    x2d = x[0]
    y2d = x[1]
    nrings = 2
    xc = pars[0]
    yc = pars[1]
    dist = pars[2]
    angle = pars[3]
    distx = pars[4]
    disty = pars[5]
    ampgauss = pars[6]
    xcgauss = pars[7]
    ycgauss = pars[8]
    fwhmgauss = pars[9]
    fwhmxpeaks = pars[10:19]
    fwhmypeaks = pars[19:28]
    rhopeaks = pars[28:37]

    amps = np.zeros(9)
    npeaks_tot = len(amps)
    npeaks_line = int(np.sqrt(npeaks_tot))
    nrings = (npeaks_line - 1) / 2 + 1
    x = (np.arange(npeaks_line) - (nrings - 1)) * dist
    xx, yy = np.meshgrid(x, x)
    xxyy = np.array([np.ravel(xx), np.ravel(yy)])
    cosang = np.cos(np.radians(angle))
    sinang = np.sin(np.radians(angle))
    rotmat = np.array([[cosang, -sinang], [sinang, cosang]])
    newxxyy = []
    for i in range(npeaks_tot):
        thexxyy = np.dot(rotmat, xxyy[:, i])
        newxxyy.append(thexxyy)
    newxxyy = np.array(newxxyy).T

    newxxyy[0, :] += distx * (newxxyy[1, :]) ** 2
    newxxyy[1, :] += disty * (newxxyy[0, :]) ** 2

    newxxyy[0, :] += xc
    newxxyy[1, :] += yc

    themap = np.zeros_like(x2d)
    for i in range(npeaks_tot):
        amps[i] = ampgauss * np.exp(
            -0.5 * ((xcgauss - newxxyy[0, i]) ** 2 + (ycgauss - newxxyy[1, i]) ** 2) / (fwhmgauss / 2.35) ** 2)
        themap += amps[i] * mygauss2d(x2d, y2d, newxxyy[:, i], fwhmxpeaks[i] / 2.35, fwhmypeaks[i] / 2.35, rhopeaks[i])

    #     satpix = themap >= saturation
    #     themap[satpix] = saturation

    if return_peaks:
        return themap, newxxyy
    else:
        return themap


def flattened_qubic_sb_model_asym(x, pars):
    return np.ravel(qubic_sb_model_asym(x, pars))


def fit_sb_asym(TESNum, dirfiles, scaling=140e3, newsize=70, dmax=5., az_center=0., el_center=50., doplot=False):
    # Read flat maps
    flatmap_init, az_init, el_init = sbfit.get_flatmap(TESNum, dirfiles)

    # ## Resample input map to have less pixels to deal with for fitting
    flatmap = scsig.resample(scsig.resample(flatmap_init, newsize, axis=0), newsize, axis=1)
    delta_az = np.median(az_init - np.roll(az_init, 1))
    delta_el = np.median(el_init - np.roll(el_init, 1))
    az = np.linspace(np.min(az_init) - delta_az / 2, np.max(az_init) + delta_az / 2, newsize)
    el = np.linspace(np.min(el_init) - delta_el / 2, np.max(el_init) + delta_el / 2, newsize)
    az2d, el2d = np.meshgrid(az * np.cos(np.radians(50)), np.flip(el))

    # ## First find the location of the maximum close to the center
    distance_max = dmax
    mask = (np.sqrt((az2d - az_center) ** 2 + (el2d - el_center) ** 2) < distance_max).astype(int)
    wmax = np.where((flatmap * mask) == np.max(flatmap * mask))
    maxval = flatmap[wmax][0]
    print('Maximum of map is {0:5.2g} and was found at: az={1:5.2f}, el={2:5.2f}'.format(maxval,
                                                                                         az2d[wmax][0], el2d[wmax][0]))

    # ## Now fit all parameters
    x = [az2d, el2d]
    parsinit = np.array([az2d[wmax][0], el2d[wmax][0], 8.3, 44., 0., 0.01, maxval / scaling, 0., 50., 13.])
    fwhmxinit = np.zeros(9) + 1
    fwhmyinit = np.zeros(9) + 1
    rhosinit = np.zeros(9)
    parsinit = np.append(np.append(np.append(parsinit, fwhmxinit), fwhmyinit), rhosinit)
    rng = [[az2d[wmax][0] - 1., az2d[wmax][0] + 1.],
           [el2d[wmax][0] - 1., el2d[wmax][0] + 1.],
           [8., 8.75],
           [43., 47.],
           [-0.1, 0.1],
           [-0.1, 0.1],
           [0, 1000],
           [-3, 3],
           [47., 53],
           [10., 16.]]
    for i in range(9):
        rng.append([0.5, 1.5])
    for i in range(9):
        rng.append([0.5, 1.5])
    for i in range(9):
        rng.append([-1, 1])

    fit = ft.do_minuit(x, np.ravel(flatmap / scaling), np.ones_like(np.ravel(flatmap)), parsinit,
                       functname=flattened_qubic_sb_model_asym, chi2=ft.MyChi2_nocov, rangepars=rng,
                       force_chi2_ndf=True)
    themap, newxxyy = qubic_sb_model(x, fit[1], return_peaks=True)

    if doplot:
        rc('figure', figsize=(18, 4))
        parfit = fit[1]
        sh = np.shape(newxxyy)
        print(sh)
        subplot(1, 2, 1)
        imshow(flatmap / scaling, extent=[np.min(az) * np.cos(np.radians(50)),
                                          np.max(az) * np.cos(np.radians(50)),
                                          np.min(el), np.max(el)])
        colorbar()
        for i in range(sh[1]):
            ax = plot(newxxyy[0, i], newxxyy[1, i], 'r.')
        title('Input Map - TES #{}'.format(TESNum))
        xlabel('Angle in Az direction [deg.]')
        ylabel('Elevation [deg.]')

        subplot(1, 2, 2)
        imshow(flatmap / scaling - themap, extent=[np.min(az) * np.cos(np.radians(50)),
                                                   np.max(az) * np.cos(np.radians(50)),
                                                   np.min(el), np.max(el)])
        colorbar()
        title('Residual Map - TES #{}'.format(TESNum))
        xlabel('Angle in Az direction [deg.]')
        ylabel('Elevation [deg.]')

    fit[1][6] *= scaling
    fit[2][6] *= scaling
    fit[1][10] *= scaling
    fit[2][10] *= scaling
    return flatmap_init, az_init, el_init, fit, newxxyy


# def demodulate_directory(thedir, ppp, lowcut=0.3, highcut=10., nbins=250, method='rms', rebin=True):
#     print ''
#     print '##############################################################'
#     print 'Directory {} '.format(thedir)
#     print '##############################################################'
#     allsb = []
#     all_az_el_azang = []
#     for iasic in [0,1]:
#         print '======== ASIC {} ====================='.format(iasic)
#         AsicNum = iasic+1
#         a = qp()
#         a.read_qubicstudio_dataset(thedir, asic=AsicNum)
#         data=a.azel_etc(TES=None)
#         data['t_src'] += 7200
#         stop
#         unbinned, binned = general_demodulate(ppp, data, 
#                                                 lowcut, highcut,
#                                                 nbins=nbins, median=True, method=method, 
#                                                 doplot=False, rebin=rebin, verbose=False)
#         # all_az_el_azang.append(np.array([unbinned['az'], unbinned['el'], unbinned['az_ang']]))
#         # allsb.append(unbinned['sb'])
#         all_az_el_azang.append(np.array([binned['az'], binned['el'], binned['az_ang']]))
#         allsb.append(binned['sb'])
#     sh0 = allsb[0].shape
#     sh1 = allsb[1].shape
#     mini = np.min([sh0[1], sh1[1]])
#     sb = np.append(allsb[0][:,:mini], allsb[1][:,:mini], axis=0)
#     az_el_azang = np.array(all_az_el_azang[0][:,:mini])
#     print az_el_azang.shape
#     print sb.shape
#     return sb, az_el_azang

def demodulate_directory(thedir, ppp, TESmask=None, srcshift=0.,
                         lowcut=0.3, highcut=10., nbins=250, method='rms', rebin=True):
    print('')
    print('##############################################################')
    print('Directory {} '.format(thedir))
    print('##############################################################')
    datas = []
    for iasic in [0, 1]:
        print('======== ASIC {} ====================='.format(iasic))
        AsicNum = iasic + 1
        a = qp()
        a.verbosity = 0
        # print('Verbosity:',a.verbosity)
        a.read_qubicstudio_dataset(thedir, asic=AsicNum)
        data = a.azel_etc(TES=None)
        ndata = len(data['t_data'])
        data['t_src'] = np.array(data['t_src'])
        data['data_src'] = np.array(data['data_src'])
        data['t_src'] -= srcshift
        # data['t_src'] += 7200
        datas.append(data)
    # Concatenate the data from the two asics in a single one
    data = datas[0].copy()
    data['data'] = np.concatenate((datas[0]['data'], datas[1]['data']), axis=0)
    # for k in data.keys(): print(k, data[k].shape)

    if TESmask is not None:
        data['data'] = data['data'][TESmask, :]

    unbinned, binned = general_demodulate(ppp, data,
                                          lowcut, highcut,
                                          nbins=nbins, median=True, method=method,
                                          doplot=False, rebin=rebin, verbose=False)
    if rebin:
        az_el_azang = np.array([binned['az'], binned['el'], binned['az_ang']])
        sb = binned['sb']
    else:
        az_el_azang = np.array([unbinned['az'], unbinned['el'], unbinned['az_ang'], unbinned['t']])
        sb = unbinned['sb']

    print(az_el_azang.shape)
    # print sb.shape
    return sb, az_el_azang


def sigmoid_saturation(x, l):
    '''
    This si the common sigmoid function modified to have a slope equals to 1 at zero whatever the value
    of the lambda parameter. Then if lambda =
    '''
    if l == 0:
        return x
    else:
        return 4. / l * (1. / (1 + np.exp(-x * l)) - 0.5)


def hwp_sin(x, pars, extra_args=None):
    amplitude = pars[0]
    XPol = 1 - pars[1]
    phase = pars[2]
    return (amplitude * 0.5 * (1 - np.abs(XPol) * np.sin(4 * np.radians(x + phase))))


def hwp_sin_sat(x, pars, extra_args=None):
    amplitude = pars[0]
    XPol = 1 - pars[1]
    phase = pars[2]
    return amplitude * sigmoid_saturation(0.5 * (1 - np.abs(XPol) * np.sin(4 * np.radians(x + phase))), pars[3])


def hwp_fitpol(thvals, ampvals, ampvals_err, doplot=False, str_title=None, saturation=False, myguess=None,
               force_chi2_ndf=True):
    okdata = ampvals_err != 0

    if not saturation:
        print('Using Simple SIN')
        fct_name = hwp_sin
        guess = np.array([np.max(np.abs(ampvals)), 0, 0.])
        rangepars = [[0, np.max(np.abs(ampvals)) * 10], [0, 1], [-180, 180]]
    else:
        print('Using Sin with Saturation')
        fct_name = hwp_sin_sat
        guess = np.array([np.max(np.abs(ampvals)), 0, 0., 1.])
        rangepars = [[0, np.max(np.abs(ampvals)) * 10], [0, 1], [-180, 180], [0., 100.]]

    if myguess is not None:
        guess = myguess

        print('Guess: ', guess)
        print('Range: ', rangepars)
    fithwp = ft.do_minuit(thvals[okdata], np.abs(ampvals[okdata]), ampvals_err[okdata], guess, functname=fct_name,
                          force_chi2_ndf=force_chi2_ndf, verbose=False, rangepars=rangepars)
    print(fithwp[1])
    if doplot:
        errorbar(thvals[okdata], np.abs(ampvals[okdata]) / fithwp[1][0], yerr=ampvals_err[okdata] / fithwp[1][0],
                 fmt='r.')
        angs = np.linspace(0, 90, 90)
        if saturation is False:
            lab = 'XPol = {0:5.2f}% +/- {1:5.2f}% '.format(fithwp[1][1] * 100, fithwp[2][1] * 100)
        else:
            lab = 'XPol = {0:5.2f}% +/- {1:5.2f}% \n Saturation = {2:5.2f} +/- {3:5.2f}'.format(fithwp[1][1] * 100,
                                                                                                fithwp[2][1] * 100,
                                                                                                fithwp[1][3],
                                                                                                fithwp[2][3])

        plot(angs, fct_name(angs, fithwp[1]) / fithwp[1][0],
             label=lab)
        ylim(-0.1, 1.1)
        plot(angs, angs * 0, 'k--')
        plot(angs, angs * 0 + 1, 'k--')
        legend(loc='upper left')
        xlabel('HWP Angle [Deg.]')
        ylabel('Normalized signal')
        if str_title:
            title(str_title)
    return fithwp


def coadd_flatmap(datain, az, el,
                  azmin=None, azmax=None, elmin=None, elmax=None, naz=50, nel=50,
                  filtering=None, silent=False, remove_eltrend=True):
    if azmin is None:
        azmin = np.min(az)
    if azmax is None:
        azmax = np.max(az)
    if elmin is None:
        elmin = np.min(el)
    if elmax is None:
        elmax = np.max(el)

    sh = np.shape(datain)
    if len(sh) == 1:
        nTES = 1
        nsamples = sh[0]
        data = np.reshape(datain, (1, len(datain)))
    else:
        nTES = sh[0]
        nsamples = sh[1]
        data = datain

    if filtering:
        if not silent:
            bar = progress_bar(nTES, 'Filtering Detectors: ')
        for i in range(nTES):
            data[i, :] = ft.filter_data(filtering[0], data[i, :], filtering[1], filtering[2], notch=filtering[3],
                                        rebin=True, verbose=False, order=5)
            if not silent:
                bar.update()

    map_az_lims = np.linspace(azmin, azmax, naz + 1)
    map_az = 0.5 * (map_az_lims[1:] + map_az_lims[0:-1])
    map_el_lims = np.linspace(elmin, elmax, nel + 1)
    map_el = 0.5 * (map_el_lims[1:] + map_el_lims[0:-1])

    azindex = (naz * (az - azmin) / (azmax - azmin)).astype(int)
    elindex = (nel * (el - elmin) / (elmax - elmin)).astype(int)

    # ## Keeping only the inner part
    inside = (azindex >= 0) & (azindex < naz) & (elindex >= 0) & (elindex < nel)
    data = data[:, inside]
    azindex = azindex[inside]
    elindex = elindex[inside]
    nsamples = inside.sum()

    if not silent:
        print('Making maps')
    mapdata = np.zeros((nTES, nel, naz))
    mapcount = np.zeros((nTES, nel, naz))

    for i in range(nsamples):
        mapdata[:, elindex[i], azindex[i]] += data[:, i]
        mapcount[:, elindex[i], azindex[i]] += 1

    themap = np.zeros((nTES, nel, naz))
    ok = mapcount != 0
    themap[ok] = mapdata[ok] / mapcount[ok]

    if remove_eltrend:
        for k in range(nTES):
            for iel in range(nel):
                themap[k, iel, :] -= np.median(themap[k, iel, :])

    if nTES == 1:
        return themap[0, :, :], map_az, map_el
    else:
        return themap, map_az, map_el
