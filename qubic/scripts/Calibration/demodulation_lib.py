from qubicpack import qubicpack as qp
import fibtools as ft
import plotters as p
import lin_lib as ll

import numpy as np
from matplotlib.pyplot import *
import matplotlib.mlab as mlab
import scipy.ndimage.filters as f
import glob
import string
import scipy.signal as scsig
from scipy import interpolate

def read_cal_src_data(file_list):
    ttsrc_i = []
    ddsrc_i = []
    for ff in file_list:
        thett, thedd = np.loadtxt(ff).T
        ttsrc_i.append(thett+3600)
        ddsrc_i.append(thedd)
    
    t_src = np.concatenate(ttsrc_i)
    data_src = np.concatenate(ddsrc_i)
    return t_src, data_src

def return_rms_period(period, time, azimuth, data):
    ### we label each data sample with a period
    period_index = ((time-time[0])/period).astype(int)
    ### We loop on periods to measure their respective amplitude and azimuth
    allperiods = np.unique(period_index)
    tper = np.zeros(len(allperiods))
    azper = np.zeros(len(allperiods))
    ampdata = np.zeros(len(allperiods))
    err_ampdata = np.zeros(len(allperiods))
    for i in xrange(len(allperiods)):
        ok = (period_index == allperiods[i])
        azper[i] = np.mean(azimuth[ok])
        tper[i] = np.mean(time[ok])
        ampdata[i] = np.std(data[ok])
        err_ampdata[i] = np.std(data[ok])/np.sqrt(2*ok.sum())
    return tper, azper, ampdata, err_ampdata


def scan2ang_RMS(period, time, azimuth, data, elevation, nbins=150, return_unbinned=False, median=True):
    ### First get the RMS per period
    tper, azper, ampdata, err_ampdata = return_rms_period(period, time, azimuth, data)
    ### Convert azimuth to angle
    angle = azper * np.cos(np.radians(elevation))
    if return_unbinned:
        return angle, ampdata, err_ampdata
    else:
        ### Rebin the result
        ang, sb, dang, dsb = ft.profile(angle, ampdata, nbins=nbins, 
                                    plot=False, dispersion=True, log=False, median=median, cutbad=False)
        return ang, sb, dsb

def fit_ang_gauss(ang, data, errdata=None, cut=None):
    # Guess for the peak location
    datar = (data-np.min(data))
    datar = datar / np.sum(datar)
    peak_guess = np.sum(ang * (datar))
    guess = np.array([peak_guess, 1., np.max(data)-np.min(data), np.median(data)])
    # If no errors were provided then use ones
    if errdata is None:
        errdata = np.ones(len(data))
    # if a cut is needed for the top of the peak
    if cut is None:
        ok = (ang > -1e100) & (errdata > 0)
    else:
        ok = (data < cut) & (errdata > 0)
    # Do the fit now
    res = ft.do_minuit(ang[ok], data[ok], errdata[ok], guess, 
                   functname=gauss, verbose=False,nohesse=True, force_chi2_ndf=True)
    return res[1], res[2]


def gauss(x,par):
    return par[3]+par[2]*np.exp(-0.5 * (x-par[0])**2 / (par[1]/2.35)**2)

def gauss_line(x,par):
    return par[4]*x+par[3]+par[2]*np.exp(-0.5 * (x-par[0])**2 / (par[1]/2.35)**2)


def demodulate(time, data, t_src, src, lowcut, highcut, fmod):
    import scipy.signal as scsig
    
    ### Filter Data and Source Signal the same way
    FREQ_SAMPLING = 1./(time[1]-time[0])
    filt = scsig.butter(5, [lowcut / FREQ_SAMPLING, highcut / FREQ_SAMPLING], btype='bandpass', output='sos')
    # Filter Data and change its sign to be in the same as Src
    new_data = -scsig.sosfilt(filt, data)
    # Interpolate Src on data times and filter it
    new_src = scsig.sosfilt(filt, np.interp(time, t_src, src))

    # Make the product for demodulation
    product = new_data * new_src
    
    # Smooth it over a period
    ppp = 1./fmod
    size_period = int(FREQ_SAMPLING * ppp)+1
    filter_period = np.ones((size_period,))/size_period
    demodulated = np.convolve(product, filter_period, mode='same')
    
    return demodulated

def scan2ang_demod(period, time, data, t_src, src, t_az, az, lowcut, highcut, elevation, 
                   nbins=150, median=True):
    ### First demodulate
    demodulated = demodulate(time, data, t_src, src, lowcut, highcut, 1./period)
    
    # Rebin this demodulated data as a function of azimuth corrected for elevation
    ang, sb, dang, dsb = ft.profile(np.interp(time, t_az, az)*np.cos(np.radians(elevation)), 
                                    demodulated, nbins=nbins,
                                    dispersion=True, plot=False, median=median)
    
    return ang, sb, dsb


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
        F=np.zeros((np.size(xin),self.nbspl))
        self.F=F
        for i in np.arange(self.nbspl):
            self.F[:,i]=self.get_spline(self.xin, i)

    def __call__(self, x, pars):
        theF=np.zeros((np.size(x),self.nbspl))
        for i in np.arange(self.nbspl): theF[:,i]=self.get_spline(x,i)
        return(np.dot(theF,pars))
        
    def get_spline(self, xx, index):
        yspl=np.zeros(np.size(self.xspl))
        yspl[index]=1.
        tck=interpolate.splrep(self.xspl,yspl)
        yy=interpolate.splev(xx,tck,der=0)
        return(yy)


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
        
    def offset (self, x, pars):
        ### Offset function
        pars_offset = pars[self.nbspl_amp:self.nbspl_amp+self.nbspl_offset]
        offset = self.spl_offset(x, pars_offset)
        return offset
        
    def phase(self, x, pars):
        ### Phase function
        pars_phase = pars[self.nbspl_amp+self.nbspl_offset:self.nbspl_amp+self.nbspl_offset+self.nbspl_phase]
        phase = self.spl_phase(x, pars_phase)
        return phase
        
    def __call__(self, x, pars):
        amp = self.amplitude(x, pars)
        offset = self.offset(x, pars)
        phase = self.phase(x,pars)
        ### Source input signal: 0=amp, 1=mod_freq, 2=offset
        input_src = ll.sim_generator_power(x, self.pars_src[0], self.pars_src[2], self.pars_src[1], phase)-0.5
        
        ### Now modulate with amplitude and offset
        return amp * input_src + offset

def scan2ang_splfit(period, time, data, t_src, src, t_az, az, lowcut, highcut, elevation, 
                   nbins=150, superbinning=1., doplot=False):
    ### Filter Data and Source Signal the same way
    FREQ_SAMPLING = 1./(time[1]-time[0])
    filt = scsig.butter(5, [lowcut / FREQ_SAMPLING, highcut / FREQ_SAMPLING], btype='bandpass', output='sos')
    # Filter Data and change its sign to be in the same as Src
    new_data = -scsig.sosfilt(filt, data)
    # Interpolate Src on data times and filter it
    new_src = scsig.sosfilt(filt, np.interp(time, t_src, src))

    ### Now resample data into bins such that the modulation period is well sampled
    ### We want bins with size < period/4
    approx_binsize = period/4/superbinning
    nbins_new = int((time[-1]-time[0])/approx_binsize)
    print('Number of initial bins in data: {}'.format(len(data)))
    print('Number of new bins in data: {}'.format(nbins_new))
    x_data, newdata, dx, dy = ft.profile(time, new_data, nbins=nbins_new, plot=False)
    new_az = np.interp(x_data, t_az, az)
    
    #### Source parameters
    src_amp = 5.          # Volts
    src_period = period   # seconds
    src_phase = 0        # Radians
    src_offset = 2.5      # Volts
    pars_src = np.array([src_amp, 1./src_period, src_offset])
    
    # nbspl_amp = 20
    # nbspl_offset = 20
    # nbspl_phase = 4
    delta_az = np.max(az)-np.min(az)
    nbspl_amp = int(delta_az * 2)
    nbspl_offset = int(delta_az * 2)
    nbspl_phase = 4
    print('Number of Splines for Amplitude: {}'.format(nbspl_amp))
    print('Number of Splines for Offset   : {}'.format(nbspl_offset))
    print('Number of Splines for Phase    : {}'.format(nbspl_phase))
    
    simsrc = SimSrcTOD(x_data, pars_src, nbspl_amp, nbspl_offset, nbspl_phase)

    guess = np.concatenate((np.ones(nbspl_amp),np.zeros(nbspl_offset),np.zeros(nbspl_phase)))

    res = ft.do_minuit(x_data, newdata, np.ones(len(newdata)), guess,
                   functname=simsrc, verbose=False,nohesse=True, force_chi2_ndf=False)
    
    if doplot==True:
        subplot(4,1,1)
        plot(new_az*np.cos(np.radians(elevation)), newdata, label='Data')
        plot(new_az*np.cos(np.radians(elevation)), simsrc(x_data, res[1]), label='Fit')
        plot(new_az*np.cos(np.radians(elevation)), new_az*0,'k--')
        legend()
        subplot(4,1,2)
        plot(new_az*np.cos(np.radians(elevation)), simsrc.amplitude(x_data, res[1]))
        plot(new_az*np.cos(np.radians(elevation)), new_az*0,'k--')
        title('Amplitude')
        subplot(4,1,3)
        plot(new_az*np.cos(np.radians(elevation)), simsrc.offset(x_data, res[1]))
        plot(new_az*np.cos(np.radians(elevation)), new_az*0,'k--')
        title('Offset')
        subplot(4,1,4)
        plot(new_az*np.cos(np.radians(elevation)), simsrc.phase(x_data, res[1]))
        plot(new_az*np.cos(np.radians(elevation)), new_az*0,'k--')
        ylim(-np.pi,np.pi)
        title('Phase')
        tight_layout()
    
    azvals = np.linspace(np.min(az), np.max(az), nbins)
    ang = azvals * np.cos(np.radians(elevation))
    sb = np.interp(ang, new_az*np.cos(np.radians(elevation)), simsrc.amplitude(x_data, res[1]))
    dsb = np.ones(nbins)
    return ang, sb, dsb



def general_demodulate(period, time, data, t_src, src, t_az, az, lowcut, highcut, elevation, 
                   nbins=150, median=True, method='demod', 
                   doplot=True, unbinned=False, cut=None, label=None, renormalize_plot=True):
    ### Call one of the methods
    if method == 'demod':
        angle, sb, dsb = scan2ang_demod(period, time, data, t_src, src, 
                                        t_az, az, lowcut, highcut, elevation, 
                                        nbins=nbins, median=median)
    elif method == 'rms':
        new_az = np.interp(time, t_az, az)
        angle, sb, dsb = scan2ang_RMS(period, time, new_az, data, elevation, 
                                 nbins=nbins, return_unbinned=unbinned, median=median)
    elif method == 'splfit':
        angle, sb, dsb = scan2ang_splfit(period, time, data, t_src, src, 
                                         t_az, az, lowcut, highcut, elevation, 
                                        nbins=nbins, superbinning=1., doplot=False)
    
    ### Fit the result
    pars, err_pars = fit_ang_gauss(angle, sb, errdata=dsb, cut=cut)
    
    if renormalize_plot:
        nn = pars[2]
        oo = pars[3]
    else:
        nn = 1.
        oo = 0.

    ### If requested plot the result
    if doplot==True:
        if label==None:
            label=method
        errorbar(angle, (sb-oo)/nn, yerr=dsb/nn, fmt='.-', label=label)
        xxx2 = np.linspace(np.min(angle), np.max(angle), 1000)
        plot(xxx2, (gauss(xxx2, pars)-oo)/nn, lw=2,
             label='Fit on '+label+': FWHM = {0:5.2f} +/- {1:5.2f} deg.'.format(pars[1], err_pars[1]))
        legend()

    return angle, sb, dsb, pars, err_pars







