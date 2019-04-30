from qubicpack import qubicpack as qp
import fibtools as ft
import plotters as p
import lin_lib as ll
import qubic
from pysimulators import FitsArray

import numpy as np
from matplotlib.pyplot import *
import matplotlib.mlab as mlab
import scipy.ndimage.filters as f
import glob
import string
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


def return_rms_period(period, time, azimuth, elevation, data, verbose=False):
    if verbose:
        printnow('Entering RMS/period')
    if data.ndim == 1:
        nTES = 1
    else:
        sh = np.shape(data)
        nTES = sh[0]
    ### we label each data sample with a period
    period_index = ((time - time[0]) / period).astype(int)
    ### We loop on periods to measure their respective amplitude and azimuth
    allperiods = np.unique(period_index)
    tper = np.zeros(len(allperiods))
    azper = np.zeros(len(allperiods))
    elper = np.zeros(len(allperiods))
    ampdata = np.zeros((nTES, len(allperiods)))
    err_ampdata = np.zeros((nTES, len(allperiods)))
    if verbose:
        printnow('Calculating RMS per period for {} periods and {} TES'.format(len(allperiods), nTES))
    for i in xrange(len(allperiods)):
        ok = (period_index == allperiods[i])
        azper[i] = np.mean(azimuth[ok])
        elper[i] = np.mean(elevation[ok])
        tper[i] = np.mean(time[ok])
        if nTES == 1:
            mm, ss = ft.meancut(data[ok], 3)
            ampdata[0, i] = ss
            err_ampdata[0, i] = 1
        else:
            for j in xrange(nTES):
                mm, ss = ft.meancut(data[j, ok], 3)
                ampdata[j, i] = ss
                err_ampdata[j, i] = 1
    return tper, azper, elper, ampdata, err_ampdata


def scan2ang_RMS(period, indata, median=True, lowcut=None, highcut=None, verbose=False):
    new_az = np.interp(indata['t_data'], indata['t_azel'], indata['az'])

    ### Check if filtering is requested
    if (lowcut is None) & (highcut is None):
        dataf = indata['data'].copy()
    else:
        if verbose: printnow('Filtering data')
        dataf = filter_data(indata['t_data'], indata['data'], lowcut, highcut)

    ### First get the RMS per period
    if verbose: printnow('Resampling Azimuth')
    az = np.interp(indata['t_data'], indata['t_azel'], indata['az'])
    if verbose: printnow('Resampling Elevation')
    el = np.interp(indata['t_data'], indata['t_azel'], indata['el'])
    tper, azper, elper, ampdata, err_ampdata = return_rms_period(period, indata['t_data'], az, el, dataf,
                                                                 verbose=verbose)
    ### Convert azimuth to angle
    angle = azper * np.cos(np.radians(elper))
    ### Fill the return variable for unbinned
    unbinned = {}
    unbinned['t'] = tper
    unbinned['az'] = azper
    unbinned['el'] = elper
    unbinned['az_ang'] = angle
    unbinned['sb'] = ampdata
    unbinned['dsb'] = err_ampdata
    return unbinned


def scan2ang_demod(period, indata, median=True, lowcut=None, highcut=None, verbose=False):
    if indata['data'].ndim == 1:
        nTES = 1
    else:
        sh = np.shape(indata['data'])
        nTES = sh[0]

    ### First demodulate
    demodulated = demodulate(indata, 1. / period, verbose=verbose, lowcut=lowcut, highcut=highcut)

    ### Resample az and el similarly as data
    azd = np.interp(indata['t_data'], indata['t_azel'], indata['az'])
    eld = np.interp(indata['t_data'], indata['t_azel'], indata['el'])

    ### Resample to one value per modulation period
    if verbose: printnow('Resampling to one value per modulation period')
    period_index = ((indata['t_data'] - indata['t_data'][0]) / period).astype(int)
    allperiods = np.unique(period_index)
    newt = np.zeros(len(allperiods))
    newaz = np.zeros(len(allperiods))
    newel = np.zeros(len(allperiods))
    newsb = np.zeros((nTES, len(allperiods)))
    newdsb = np.zeros((nTES, len(allperiods)))
    for i in xrange(len(allperiods)):
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


def demodulate(indata, fmod, lowcut=None, highcut=None, verbose=False):
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
        ### Check if filtering is requested
        if (lowcut is None) & (highcut is None):
            dataf = indata['data'].copy()
            new_src = np.interp(indata['t_data'], indata['t_src'], indata['data_src'])
        else:
            if verbose: printnow('Filtering data and Src Signal')
            dataf = filter_data(indata['t_data'], indata['data'], lowcut, highcut)
            new_src = filter_data(indata['t_data'], np.interp(indata['t_data'], indata['t_src'], indata['data_src']),
                                  lowcut, highcut)

        if nTES == 1: dataf = np.reshape(dataf, (1, len(indata['data'])))

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
        return (yy)


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
        ### Amplitude function
        pars_amp = pars[0:self.nbspl_amp]
        amp = self.spl_amp(x, pars_amp)
        return amp

    def offset(self, x, pars):
        ### Offset function
        pars_offset = pars[self.nbspl_amp:self.nbspl_amp + self.nbspl_offset]
        offset = self.spl_offset(x, pars_offset)
        return offset

    def phase(self, x, pars):
        ### Phase function
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
    ### Filter Data and Source Signal the same way + change sign of data
    new_data = -filter_data(time, data, lowcut, highcut)
    new_src = filter_data(time, np.interp(time, t_src, src), lowcut, highcut)

    ### Now resample data into bins such that the modulation period is well sampled
    ### We want bins with size < period/4
    approx_binsize = period / 4 / superbinning
    nbins_new = int((time[-1] - time[0]) / approx_binsize)
    print('Number of initial bins in data: {}'.format(len(data)))
    print('Number of new bins in data: {}'.format(nbins_new))
    x_data, newdata, dx, dy = ft.profile(time, new_data, nbins=nbins_new, plot=False)
    new_az = np.interp(x_data, t_az, az)

    #### Source parameters
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

    if doplot == True:
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
    ### Call one of the methods
    if method == 'demod':
        if verbose: printnow('Demodulation Method')
        unbinned = scan2ang_demod(period, indata, verbose=verbose, median=median, lowcut=lowcut, highcut=highcut)
    elif method == 'rms':
        if verbose: printnow('RMS Method')
        unbinned = scan2ang_RMS(period, indata, verbose=verbose, median=median, lowcut=lowcut, highcut=highcut)
    # elif method == 'splfit':
    #     azbins, elbins, angle, sb, dsb = scan2ang_splfit(period, time, data, t_src, src, 
    #                                      t_az, az, lowcut, highcut, elevation, 
    #                                     nbins=nbins, superbinning=1., doplot=False)

    if rebin:
        ### Now rebin the data
        if verbose: printnow('Now rebin the data')
        if indata['data'].ndim == 1:
            sh = [1, len(indata['data'])]
        else:
            sh = np.shape(indata['data'])
        ang = np.zeros(nbins)
        sb = np.zeros((sh[0], nbins))
        dsb = np.zeros((sh[0], nbins))
        others = np.zeros((nbins, 2))
        for i in xrange(sh[0]):
            if verbose: 
                if (16*(i/16))==i: 
                    printnow('Rebinning TES {} over {}'.format(i,sh[0]))
            ang, sb[i,:], dang, dsb[i,:], others = ft.profile(unbinned['az_ang'], 
                                                              unbinned['sb'][i,:], nbins=nbins, plot=False, 
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

    ### If requested plot the result
    if doplot == True:
        if verbose: printnow('Plot')
        toplot = unbinned
        if rebin: toplot = binned
        if label == None:
            label = method
        if indata['data'].ndim == 1:
            sh = [1, len(indata['data'])]
        else:
            sh = np.shape(indata['data'])
        for i in xrange(sh[0]):
            errorbar(toplot['az_ang'], toplot['sb'][i, :], yerr=toplot['dsb'][i, :], fmt='.-',
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
    ############################### TES DATA ###################################
    #### Get TES data: if TESNum is None, will return the whole ASIC array (128)
    if TESNum:
        data = a.timeline(TES=TESNum)
    else:
        data = a.timeline_array()
    #### Get TES timestamps
    t_data = a.timeline_timeaxis(axistype='pps')
    ############################################################################

    ############################### Az and El data #############################
    az = a.azimuth()
    el = a.elevation()
    t_azel = a.timeaxis(datatype='hk', axistype='pps')
    ############################################################################

    ############################### Cal Src Data ###############################
    #### First need to check wether there is a calsrouce data file or not
    #### and then read the file if it exists
    data_time = dt.datetime.utcfromtimestamp(a.hk['ASIC_SUMS']['ComputerDate'][0])
    glob_pattern = data_time.strftime('calsource_%Y%m%dT%H%M*.dat')
    bla = glob.glob(calsource_dir + glob_pattern)
    if bla == []:
        if verbose: print 'No CalSource file found corresponding to this dataset: ' + dirfile
        t_src = -1
        data_src = -1
    else:
        if verbose: print 'Found Calibration Source date in: ' + bla[0]
        t_src, data_src = read_cal_src_data(bla)

    #### Now return everything
    if verbose:
        print 'Returning:'
        print '   t_data  : ', array_info(t_data)
        print '   data    : ', array_info(data)
        print '   t_azel  : ', array_info(t_azel)
        print '   az      : ', array_info(az)
        print '   el      : ', array_info(el)
        print '   t_src   : ', array_info(t_src)
        print '   data_src: ', array_info(data_src)

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


def power_spectrum(time_in, data_in, rebin=True):
    if rebin:
        ### Resample the data on a reguklar grid
        time = np.linspace(time_in[0], time_in[-1], len(time_in))
        data = np.interp(time, time_in, data_in)
    else:
        time = time_in
        data = data_in

    spectrum_f, freq_f = mlab.psd(data, Fs=1. / (time[1] - time[0]), NFFT=len(data), window=mlab.window_hanning)
    return spectrum_f, freq_f


def filter_data(time_in, data_in, lowcut, highcut, rebin=True, verbose=False):
    sh = np.shape(data_in)
    if rebin:
        if verbose: printnow('Rebinning before Filtering')
        ### Resample the data on a regular grid
        time = np.linspace(time_in[0], time_in[-1], len(time_in))
        if len(sh) == 1:
            data = np.interp(time, time_in, data_in)
        else:
            data = vec_interp(time, time_in, data_in)
    else:
        if verbose: printnow('No rebinning before Filtering')
        time = time_in
        data = data_in

    FREQ_SAMPLING = 1. / ((np.max(time) - np.min(time)) / len(time))
    filt = scsig.butter(5, [lowcut / FREQ_SAMPLING, highcut / FREQ_SAMPLING], btype='bandpass', output='sos')
    if len(sh) == 1:
        dataf = scsig.sosfilt(filt, data)
    else:
        dataf = scsig.sosfilt(filt, data, axis=1)
    return dataf


def vec_interp(x, xin, yin):
    sh = np.shape(yin)
    nvec = sh[0]
    yout = np.zeros_like(yin)
    for i in xrange(nvec):
        yout[i, :] = np.interp(x, xin, yin[i, :])
    return yout


def bin_image_elscans(x, y, data, xr, nx, TESIndex):
    ny = len(y)
    mapsum = np.zeros((nx, ny))
    mapcount = np.zeros((nx, ny))
    for i in xrange(ny):
        thex = x[i]
        dd = data[i] - np.mean(data[i], axis=0)
        idx = ((thex - xr[0]) / (xr[1] - xr[0]) * nx).astype(int)
        for j in xrange(len(thex)):
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
    for i in xrange(len(azdeg)):
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


def get_hpmap(TESNum, directory):
    return qubic.io.read_map(directory+'/Healpix/healpix_TESNum_{}.fits'.format(TESNum))    


def get_lines(lines, directory):
    nn = len(lines)
    hpmaps = np.zeros((nn, 4, 12*256**2))
    nums = np.zeros((nn, 4),dtype=int)
    for l in xrange(nn):
        for i in xrange(4):
            if lines[l] < 33:
                nums[l,i] = int(lines[l]+32*i)
            else:
                nums[l,i] = int(lines[l]-32+32*i)+128
            hpmaps[l,i,:] = get_hpmap(nums[l,i],directory)
    return hpmaps, nums


def show_lines(maps, nums, min=None, max=None):
    sh = np.shape(maps)
    nl = sh[0]
    for l in xrange(nl):
        for i in xrange(4):
            hp.gnomview(maps[l,i,:], reso=10, min=min, max=max, sub=(nl, 4, l*4+i+1), title=nums[l,i])
    tight_layout()


def get_flatmap(TESNum, directory):
    themap = FitsArray(directory+'/Flat/imgflat_TESNum_{}.fits'.format(TESNum))
    az = FitsArray(directory+'/Flat/azimuth.fits'.format(TESNum))    
    el = FitsArray(directory+'/Flat/elevation.fits'.format(TESNum))
    return themap, az, el


def CalSrcPower_Vs_Nu(freq):
    ### Taken from Tx 263 130-170GHz User Guide available on Atrium
    ff = np.array([129.93109059, 130.93364949, 131.93662559, 132.93897589,
       133.94132618, 134.94430228, 135.94686118, 136.95004589,
       137.90587712, 138.90927044, 139.91203794, 140.91501405,
       141.91882458, 142.92326092, 143.92686285, 144.92942175,
       145.93218925, 146.93537396, 147.93876728, 148.99034851,
       149.94597113, 150.94853003, 151.95171474, 153.00350458,
       153.9593358 , 154.96335494, 155.96695686, 156.97076739,
       157.9259728 , 158.92790588, 159.97948711, 160.98371485,
       161.98794259, 162.94419103, 163.99702389, 165.00292048,
       165.91431869, 167.01429644, 167.92068812, 168.92387283,
       169.97399383])
    power_mW = np.array([12.83207659, 13.22505174, 13.35869985, 13.90693552, 14.47767056,
       14.62397723, 15.07182833, 15.07182833, 14.77176243, 14.62397723,
       14.9210411 , 15.07182833, 14.62397723, 13.76780255, 13.49369857,
       13.90693552, 14.18943378, 14.18943378, 14.04747453, 13.63006154,
       13.49369857, 13.90693552, 13.90693552, 13.35869985, 13.09274073,
       12.57660204, 12.32621374, 11.95994707, 12.08081044, 12.83207659,
       12.45077849, 11.84029289, 11.25974057, 10.81586195,  9.88006554,
        8.66944441,  7.23416893,  6.81063762,  7.23416893,  7.23416893,
        7.53105632])
    return np.interp(freq, ff, power_mW)


def CalSrcPowerMeterResponse_Vs_Nu(freq):
    ### Taken from Tx 263 130-170GHz User Guide available on Atrium
    ### It ic=ncludes both output response and power meter response
    ff = np.array([129.96075353, 130.90266876, 131.89167975, 133.02197802,
       133.96389325, 134.90580848, 135.89481947, 136.97802198,
       137.96703297, 139.00313972, 139.89795918, 141.02825746,
       141.97017268, 143.00627943, 144.04238619, 144.93720565,
       145.9733124 , 147.00941915, 147.95133438, 148.98744113,
       150.02354788, 150.96546311, 151.90737834, 153.03767661,
       153.97959184, 154.92150706, 155.95761381, 156.99372057,
       157.93563579, 158.97174254, 159.91365777, 160.99686028,
       161.98587127, 162.9277865 , 163.96389325, 164.95290424,
       165.89481947, 166.97802198, 167.91993721, 168.95604396,
       169.94505495])
    response = np.array([1.48598131, 1.56133453, 1.67410027, 1.80564562, 1.91838202,
       1.97504365, 2.0691178 , 2.16325063, 2.0704089 , 2.22058716,
       2.07161197, 2.18446573, 2.07290306, 2.07354861, 2.07419416,
       2.22428439, 2.33707948, 2.46856615, 2.52522778, 2.54456491,
       2.63866841, 2.88224592, 3.0323655 , 3.03306973, 3.2018809 ,
       3.16508458, 3.16573013, 3.27852521, 3.63425226, 4.10218753,
       4.70090524, 4.85111284, 4.47789727, 4.68409161, 4.62866239,
       4.34890477, 3.88220191, 3.80811045, 4.1638375 , 4.4074737 ,
       4.66977215])
    return np.interp(freq, ff, response)


def dB(y):
    negative = y <= 0
    bla = 10*np.log10(y/np.max(y))
    bla[negative] = -50
    return bla


def get_spectral_response(name, freqs, allmm, allss, nsig=3, method='demod', TESNum = None,
                          directory='/Users/hamilton/Qubic/Calib-TD/SpectralResponse/'):
    #### Restore the data already treated
    allmm = FitsArray(directory+'/allmm_'+method+'_'+name+'.fits')
    allss = FitsArray(directory+'/allss_'+method+'_'+name+'.fits')
    freqs = FitsArray(directory+'/freqs_'+method+'_'+name+'.fits')

    #### Correct for Source Characteristics
    if method == 'rms':
        ### Then the analysis does not use the power meter data and we only need to correct for the output power
        allmm /= (CalSrcPower_Vs_Nu(freqs))
        allss /= (CalSrcPower_Vs_Nu(freqs))
    else:
        ### In the case of demod we need to correct for both the power_meter response and the output power
        ### This is done using the function below
        allmm /= (CalSrcPowerMeterResponse_Vs_Nu(freqs))
        allss /= (CalSrcPowerMeterResponse_Vs_Nu(freqs))

    sh = np.shape(allmm)
    nTES = sh[0]
    
    #### Normalize all TES to the same integral
    allfnorm = np.zeros((256, len(freqs)))
    allsnorm = np.zeros((256, len(freqs)))
    infilter = (freqs >= 124) & (freqs <= 182)
    outfilter =  ~infilter
    for tesindex in xrange(256):
        baseline = np.mean(allmm[tesindex,outfilter])
        integ = np.sum(allmm[tesindex, infilter]-baseline)
        allfnorm[tesindex,:] = (allmm[tesindex,:]-baseline)/integ
        allsnorm[tesindex,:] = allss[tesindex,:]/integ

    ### If only one TES is requested we return it. Otherwise we need to do an average over nicely
    ### Behaving TES...
    if TESNum is not None:
        return freqs, allfnorm[TESNum-1,:]-np.min(allfnorm[TESNum-1,:]), allsnorm[TESNum-1,:]
    else:
        ### Discriminant Variable: a chi2 we want it to be bad in the sense that 
        ### we want the spectrum to be inconsistent with a straight line inside the QUBIC band
        discrim = np.nansum(allfnorm[:,infilter]**2/allsnorm[:,infilter]**2, axis=1)
        mr, sr = ft.meancut(discrim,3)
        threshold = mr+nsig*sr
        ok = (discrim > threshold)
        print 'Spectral Response calculated over {} TES'.format(ok.sum())
        filtershape = np.zeros(len(freqs))
        errfiltershape = np.zeros(len(freqs))
        for i in xrange(len(freqs)):
            filtershape[i], errfiltershape[i] = ft.meancut(allfnorm[ok,i],2)
        #errfiltershape /= np.sqrt(ok.sum())
        # Then remove the smallest value in order to avoid negative values
        filtershape -= np.min(filtershape)
        return freqs, filtershape, errfiltershape
    
