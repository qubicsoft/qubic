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

class HandlingPip(object):

    """
    This class allows to make the same things that in Sample_demodulation.Rmd
    """

    def __init__(self, file):

        self.file=file
        self.a = qubicfp()
        self.a.read_qubicstudio_dataset(self.file)

        self.calconf=self.a.hk['CALSOURCE-CONF']
        self.mod_freq=self.a.hk['CALSOURCE-CONF']['Mod_freq']
        #print(self.calconf)
        self.t_src = self.a.calsource()[0]
        self.data_src = self.a.calsource()[1]
        self.nharm = 10
        self.notch = np.array([[1.724, 0.004, self.nharm]])
    def get_azel(self):
        return self.a.azimuth(), self.a.elevation()
    def get_source_data(self):
        return self.a.calsource()[0], self.a.calsource()[1]
    def get_timeline_timeaxis(self, TESNum, asic):
        tod = self.a.timeline(TES=TESNum, asic=asic)
        tt = self.a.timeaxis(axistype='pps', asic=asic)
        return tt, tod
    def get_spectrum_data(self, TESNum, asic, filter=True, lowcut=0.5, highcut=20):

        #print('    -> Get timeline axis')
        tt, data = self.get_timeline_timeaxis(TESNum, asic)
        if filter:
            #print('    -> Filtering')
            data = ft.filter_data(tt, data, lowcut, highcut, notch=self.notch, rebin=True, verbose=True, order=5)
        #print('    -> Done \n')
        #print('    -> Get power spectrum')
        spectrum, freq = ft.power_spectrum(tt, data, rebin=True)
        #print('    -> Done \n')

        return tt, data, freq, spectrum
    def demodulation_simple(self, tt, newdata, TESNum, asic, makeflatmaps = True, lowcut=0.5, highcut=20):

        time_azel, _=self.get_timeline_timeaxis(TESNum=TESNum, asic=asic)
        az, el=self.get_azel()
        src = [tt, np.interp(tt, self.t_src, self.data_src)]
        fourier_cuts = [lowcut, highcut, self.notch]
        #print('    -> Demodulation')

        newt_demod, amp_demod, errors_demod = dl.demodulate_methods([tt, newdata],
                                                            self.mod_freq,
                                                            src_data_in=src,
                                                            method='demod_quad', remove_noise=False,
                                                            fourier_cuts=fourier_cuts)
        #print('    -> Done \n')

        time_azel=self.a.timeaxis(datatype='hk',axistype='pps')
        if makeflatmaps:
            #for quad demod
            newaz = np.interp(newt_demod, time_azel, az)
            newel = np.interp(newt_demod, time_azel, el)

            azmin = min(az)
            azmax = max(az)
            elmin = min(el)
            elmax = max(el)
            naz = 101
            nel = 101
            #map for quad demod
            mymap, azmap, elmap = dl.coadd_flatmap(amp_demod, newaz, newel,
                                    filtering=None,
                                    azmin=azmin, azmax=azmax,
                                    elmin=elmin, elmax=elmax,
                                    naz=naz,nel=nel)

            return newt_demod, amp_demod, errors_demod, mymap, azmap, elmap, newaz, newel
        return newt_demod, amp_demod, errors_demod

    def from_file_to_demodulation(self, TESNum, asic, filter=True, makeflatmaps=True, lowcut=0.5, highcut=20, save=False, verbose=False):

        time_azel = self.a.timeaxis(datatype='hk',axistype='pps')
        if verbose:
            print('    ->Get Az El \n')
        az, el = self.get_azel()
        if verbose:
            print('    -> Get source data \n')
        t_src, data_src = self.get_source_data()
        if verbose:
            print('    -> Get TOD \n')
        tt, tod = self.get_timeline_timeaxis(TESNum=TESNum, asic=asic)
        if verbose:
            print('    -> Get spectrum \n')
        tt, tod, freq, spectrum=self.get_spectrum_data(TESNum, asic, filter=filter, lowcut=lowcut, highcut=highcut)

        if verbose:
            print('    -> Demodulation \n')
        newt_demod, amp_demod, errors_demod, mymap, azmap, elmap, newaz, newel = self.demodulation_simple(tt, tod,
                                                    TESNum=TESNum, asic=asic, makeflatmaps=makeflatmaps)
        if verbose:
            print('    -> Saving Fits files \n')
        hpmapa = dl.scan2hpmap(128, newaz*np.cos(np.radians(50)), newel-50, amp_demod)
        if save:
            self.save_azel_mymap(mymap, az, el, TESNum)
            self.save_healpixmap(hpmapa, TESNum)

        return newt_demod, amp_demod, errors_demod, mymap, azmap, elmap, hpmapa



    def from_file_to_demodulation_allTES(self, makeflatmaps=True):

        time_azel = self.a.timeaxis(datatype='hk',axistype='pps')
        print('    -> Get Az El \n')
        az, el = self.get_azel()
        print('    -> Get source data \n')
        t_src, data_src = self.get_source_data()
        print('    -> Get TOD \n')
        tt, tod = self.get_timeline_timeaxis(TESNum=94, asic=1)
        newt_demod, amp_demod, errors_demod, mymap, azmap, elmap, newaz, newel = self.demodulation_simple(tt, tod,
                                                    TESNum=94, asic=1, makeflatmaps=makeflatmaps)

        naz = 101
        nel = 101
        mymaps = np.zeros(((256, naz, nel)))
        hpmaps=np.zeros((256, 12*128**2))
        # Loop over 2 asic
        k=0
        for i in range(2):
            # Loop over 128 TES
            for j in range(128):
                print('{}/256'.format(k+1))
                tod = self.a.timeline(TES=j, asic=i)
                _, amp, _, mymaps[k], _, _, _, _ = self.demodulation_simple(tt, tod,
                                                            TESNum=j, asic=i, makeflatmaps=makeflatmaps)

                hpmaps[k] = dl.scan2hpmap(128, newaz*np.cos(np.radians(50)), newel-50, amp)

                k+=1

        self.save_maps_allTES(self, mymaps)
        self.save_healpix_allTES(self, hpmaps)
        return mymaps, hpmaps



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
    def save_azel_mymap(self, mymap, az, el, TESNum):

        repository=os.getcwd()+'/Fits/Flat'
        try:
            os.makedirs(repository)
        except OSError:
            if not os.path.isdir(repository):
                raise

        FitsArray(mymap).save(repository+'/imgflat_TESNum_{}.fits'.format(TESNum))

        """save the az el files for flats..."""
        FitsArray(az).save(repository+'/azimuth.fits')
        FitsArray(el).save(repository+'/elevation.fits')
    def save_healpixmap(self, healpixmap, TESNum):

        repository=os.getcwd()+'/Fits/Healpix'
        try:
            os.makedirs(repository)
        except OSError:
            if not os.path.isdir(repository):
                raise

        FitsArray(healpixmap).save(repository+'/healpix_'+'TESNum_'+str(TESNum)+'.fits')





print()
