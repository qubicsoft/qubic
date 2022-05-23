import qubic
from qubic import selfcal_lib as scal
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
from alive_progress import alive_bar
import os


class BeamMapsAnalysis(object):

    """
    This class allows to make the same things that in Sample_demodulation.Rmd.
    It generates raw data and make all the analysis to obtain beam maps.
    """

    def __init__(self, a, TESNum, asic):

        """
        Parameters
        ----------
        a: Object from qubicpack
            Contains data
        TESNum: int
            Number of seen TES.
        asic: int
            Number of seen asic.
        """

        self.a = a
        self.mod_freq=self.a.hk['CALSOURCE-CONF']['Mod_freq'][0]
        if self.mod_freq < 0 :
            self.demod=False
        else:
            self.demod=True
        self.nharm=10
        self.lowcut=self.a.hk['CALSOURCE-CONF']['Amp_hfreq'][0]
        self.highcut=self.a.hk['CALSOURCE-CONF']['Amp_lfreq'][0]
        self.notch=np.array([[1.724, 0.005, self.nharm]])
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


def plot_data_on_FP(datain, q, lim=None, savepdf=None, **kwargs):


    """

    Parameters :

        - datain : array -> The data that you want to plot on the focal plane. The data must have the shape (N_tes x N_data)
        for 1D plot or (N_tes x N_data x N_data) for 2D plot.
        - q : object -> object of qubic computing with qubic package
        - x : array -> for 1D plot, you can give x axis for the plot
        - lim : array -> have the shape [x_min, x_max, y_min, y_max] if you want to put limit on axis
        - savepdf : str -> Put the name of the file if you want to save the plot
        - **kwargs : -> You can put severals arguments to modify the plot (color, linestyle, ...)

    """

    if len(datain.shape)==3:
        dimension=2
    elif len(datain.shape)==2:
        dimension=1

    x=np.linspace(-0.0504, -0.0024, 17)
    y=np.linspace(-0.0024, -0.0504, 17)

    X, Y = np.meshgrid(x, y)

    allTES=np.arange(1, 129, 1)

    #delete thermometers tes
    good_tes=np.delete(allTES, np.array([4,36,68,100])-1, axis=0)

    fig, axs = subplots(nrows=17, ncols=17, figsize=(50, 50))
    k=0
    for j in [1, 2]:
        for ites, tes in enumerate(good_tes):
            if j > 1:
                newtes=tes+128
            else:
                newtes=tes
            #print(ites, tes, j)

            xtes, ytes, FP_index, index_q= scal.TES_Instru2coord(TES=tes, ASIC=j, q=q, frame='ONAFP', verbose=False)
            ind=np.where((np.round(xtes, 4) == np.round(X, 4)) & (np.round(ytes, 4) == np.round(Y, 4)))


            if dimension == 1:

                axs[ind[0][0], ind[1][0]].plot(datain[k], **kwargs)

                if lim != None:
                    axs[ind[0][0], ind[1][0]].set_xlim(lim[0], lim[1])
                    axs[ind[0][0], ind[1][0]].set_ylim(lim[2], lim[3])

            elif dimension == 2:
                #beam=_read_fits_beam_maps(newtes)
                axs[ind[0][0], ind[1][0]].imshow(datain[k], **kwargs)

            axs[ind[0][0], ind[1][0]].set_title('TES = {:.0f}'.format(tes))


            k+=1
    if savepdf != None:
        savefig(savepdf, format="pdf", bbox_inches="tight")
    show()
