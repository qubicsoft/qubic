from qubicpack.qubicfp import qubicfp
import qubic.fibtools as ft
import qubic.demodulation_lib as dl
from pysimulators import FitsArray

import numpy as np
from matplotlib.pyplot import *
import scipy.ndimage.filters as f
import glob
import string
import scipy.signal as scsig
from scipy import interpolate
import datetime as dt
from importlib import reload
import scipy.misc
import pprint
from scipy.signal import chirp, find_peaks, peak_widths
import qubic.sb_fitting as sbfit
import healpy as hp
from qubic.io import write_map
import os


class BeamMapsAnalysis(object):

    """
    This class allows to make the same things that in Sample_demodulation.Rmd. It generates raw data and make all the analysis to
    obtain beam maps.
    """

    def __init__(self, a, lc, hc, notch, TESNum, asic):

        '''

        a is the object from qubicpack

        '''
        self.a = a
        self.lowcut=lc
        self.highcut=hc
        self.notch=notch
        self.TESNum=TESNum
        self.asic=asic

    def get_raw_data(self):

        tod=self.a.timeline(TES=self.TESNum, asic=self.asic)
        tt=self.a.timeaxis(axistype='pps', asic=self.asic)

        return tt, tod
    def filter_data(self, tt, tod, doplot=False):

        filtered_data = ft.filter_data(tt, tod, self.lowcut, self.highcut, notch=self.notch, rebin=True, verbose=True, order=5)

        if doplot:

            #plot limits
            thefreqmod = abs(self.a.hk['CALSOURCE-CONF']['Mod_freq'])
            #number of harmonics
            nharm = 10
            #filtering parameters
            period = 1./ thefreqmod
            xmin = 0.1
            xmax = 90.
            ymin = 1e0
            ymax = 1e13
            figure(figsize=(16, 8))
            ############ Power spectrum RAW plot
            spectrum_f, freq_f = ft.power_spectrum(tt, tod, rebin=True)
            plot(freq_f, f.gaussian_filter1d(spectrum_f,1), label='Raw Data')
            spectrum_f2, freq_f2 = ft.power_spectrum(tt, filtered_data, rebin=True)
            plot(freq_f2, f.gaussian_filter1d(spectrum_f2,1), label='Filtered Data')
            plot([self.lowcut, self.lowcut],[ymin,ymax],'k', lw=3, label='Bandpass')
            plot([self.highcut, self.highcut],[ymin,ymax],'k', lw=3)
            plot([1./period,1./period],[ymin,ymax],'k--', lw=3, alpha=0.3, label='Calsource Harmonics')
            for i in range(10):
                plot([1./period*i,1./period*i],[ymin,ymax],'k--', lw=3, alpha=0.3)
            #plot the pulse tube harmoncs
            plot([self.notch[0,0],self.notch[0,0]], [ymin,ymax],'m:', lw=3, label='Pulse Tube Harmonics')
            for i in range(nharm):
                plot([self.notch[0,0]*(i+1),self.notch[0,0]*(i+1)], [ymin,ymax],'m:', lw=3)
            legend(loc='center left')
            yscale('log')
            xscale('log')
            xlabel('Frequency [Hz]')
            ylabel('Power Spectrum')
            xlim(xmin, xmax)
            ylim(ymin, ymax)
            tight_layout()
            show()

        return filtered_data
    def demodulation(self, tt, data, remove_noise):

        t_src = self.a.calsource()[0]
        data_src = self.a.calsource()[1]
        fourier_cuts = [self.lowcut, self.highcut, self.notch]
        freq_mod = abs(a.hk['CALSOURCE-CONF']['Mod_freq'])
        # internpolate
        src = [tt, np.interp(tt, t_src, data_src)]
        #demod in quadrature, should have no time dependance but increased RMS noise
        newt_demod, amp_demod, err = dl.demodulate_methods([tt, data],
                                                            freq_mod,
                                                            src_data_in=src,
                                                            method='demod_quad', remove_noise=remove_noise,
                                                            fourier_cuts=fourier_cuts)

        return newt_demod, amp_demod
    def make_flat_map(self, tt, data, doplot=False):

        time_azel = self.a.timeaxis(datatype='hk',axistype='pps')
        az = self.a.azimuth()
        el = self.a.elevation()

        #for quad demod
        newaz = np.interp(tt, time_azel, az)
        newel = np.interp(tt, time_azel, el)
        azmin = min(az)
        azmax = max(az)
        elmin = min(el)
        elmax = max(el)
        naz = 101
        nel = 101
        #map for quad demod
        mymap, azmap, elmap = dl.coadd_flatmap(data,
                                                newaz,
                                                newel,
                                                filtering=None,
                                                azmin=azmin, azmax=azmax,
                                                elmin=elmin, elmax=elmax,
                                                naz=naz,nel=nel)

        if doplot:
            figure(figsize=(16,8))
            imshow(-mymap, aspect='equal', origin='lower',
                        extent=[azmin*np.cos(np.radians(50)), azmax*np.cos(np.radians(50)), elmin, elmax],)
            colorbar()
            show()


        return -mymap

    def fullanalysis(self, filter=False, demod=False, remove_noise=True, doplot=True, save=False):

        # Generate TOD from qubicpack
        print('Get Raw data')
        tt, tod = self.get_raw_data()
        data=tod.copy()

        # Filtering
        if filter:
            print('Filtering')
            filtered_data=self.filter_data(tt, tod, doplot=doplot)
            data=filtered_data.copy()

        # Demodulation
        if demod:
            print('Demodulation')
            newt, data=self.demodulation(tt, data, remove_noise=remove_noise)

        print('Make Flat Maps')
        mymap=self.make_flat_map(tt, data, doplot=doplot)

        if save:
            self.save_azel_mymap(mymap)

        return mymap






    def save_maps_allTES(self, mymap):

        repository=os.getcwd()+'/Fits/Flat'
        try:
            os.makedirs(repository)
        except OSError:
            if not os.path.isdir(repository):
                raise

        FitsArray(mymap).save(repository+'/imgflat_allTES.fits')
    def save_healpix_allTES(self, mymap):

        repository=os.getcwd()+'/Fits/Healpix'
        try:
            os.makedirs(repository)
        except OSError:
            if not os.path.isdir(repository):
                raise

        FitsArray(mymap).save(repository+'/healpix_allTES.fits')
    def save_azel_mymap(self, mymap):

        repository=os.getcwd()+'/Fits/Flat'
        try:
            os.makedirs(repository)
        except OSError:
            if not os.path.isdir(repository):
                raise

        if self.asic == 2:
            factor=128
        else:
            factor=0

        FitsArray(mymap).save(repository+'/imgflat_TESNum_{}.fits'.format(self.TESNum+factor))

        #"""save the az el files for flats..."""
        #FitsArray(az).save(repository+'/azimuth.fits')
        #FitsArray(el).save(repository+'/elevation.fits')
    def save_healpixmap(self, healpixmap, TESNum):

        repository=os.getcwd()+'/Fits/Healpix'
        try:
            os.makedirs(repository)
        except OSError:
            if not os.path.isdir(repository):
                raise

        FitsArray(healpixmap).save(repository+'/healpix_'+'TESNum_'+str(TESNum)+'.fits')





print()
