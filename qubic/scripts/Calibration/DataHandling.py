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
from scipy.interpolate import interp1d
from qubic.io import write_map
from alive_progress import alive_bar
from scipy.signal import medfilt
import os

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
def identify_scans(thk, az, el, tt=None, median_size=101, thr_speedmin=0.1, doplot=False, plotrange=[0,1000]):
    """
    This function identifies and assign numbers the various regions of a back-and-forth scanning using the housepkeeping time, az, el
        - a numbering for each back & forth scan
        - a region to remove at the end of each scan (bad data due to FLL reset, slowingg down of the moiunt, possibly HWP rotation
        - is the scan back or forth ?
    It optionnaly iinterpolate this information to the TOD sampling iif provided.
    Parameters
    ----------
    input
    thk : np.array()
            time samples (seconds) for az and el at the housekeeeping sampling rate
    az : np.array()
            azimuth in degrees at the housekeeping sampling rate
    el : np.array()
            elevation in degrees at the housekeeping sampling rate
    tt : Optional : np.array()
            None buy default, if not None:
            time samples (seconds) at the TOD sampling rate
            Then. the output will also containe az,el and scantype interpolated at TOD sampling rate
    thr_speedmin : Optional : float
            Threshold for angular velocity to be considered as slow
    doplot : [Optional] : Boolean
            If True displays some useeful plot
    output :
    scantype_hk: np.array(int)
            type of scan for each sample at housekeeping sampling rate:
            * 0 = speed to slow - Bad data
            * n = scanning towards positive azimuth
            * -n = scanning towards negative azimuth
            where n is the scan number (starting at 1)
    azt : [optional] np.array()
            azimuth (degrees) at the TOD sampling rate
    elt : [optional] np.array()
            elevation (degrees) at the TOD sampling rate
    scantype : [optiona] np.array()
            same as scantype_hk, but interpolated at TOD sampling rate
    """

    ### Sampling for HK data
    timesample = np.median(thk[1:]-thk[:-1])
    ### angular velocity
    medaz_dt = medfilt(np.gradient(az, timesample), median_size)
    ### Identify regions of change
    # Low velocity -> Bad
    c0 = np.abs(medaz_dt) < thr_speedmin
    # Positive velicity => Good
    cpos = (~c0) * (medaz_dt > 0)
    # Negative velocity => Good
    cneg = (~c0) * (medaz_dt < 0)

    ### Scan identification at HK sampling
    scantype_hk = np.zeros(len(thk), dtype='int')-10
    scantype_hk[c0] =0
    scantype_hk[cpos] = 1
    scantype_hk[cneg] = -1
    # check that we have them all
    count_them = np.sum(scantype_hk==0) + np.sum(scantype_hk==-1) + np.sum(scantype_hk==1)
    if count_them != len(scantype_hk):
        print('Identify_scans: Bad Scan counting at HK sampling level - Error')
        stop

    ### Now give a number to each back and forth scan
    num = 0
    previous = 0
    for i in range(len(scantype_hk)):
        if scantype_hk[i] == 0:
            previous = 0
        else:
            if (previous == 0) & (scantype_hk[i] > 0):
                # we have a change
                num += 1
            previous = 1
            scantype_hk[i] *= num

    dead_time = np.sum(c0) / len(thk)

    if doplot:
        ### Some plotting
        plt.subplot(2,2,1)
        plt.title('Angular Velocity Vs. Azimuth - Dead time = {0:4.1f}%'.format(dead_time*100))
        plt.plot(az, medaz_dt)
        plt.plot(az[c0], medaz_dt[c0], 'ro', label='Slow speed')
        plt.plot(az[cpos], medaz_dt[cpos], '.', label='Scan +')
        plt.plot(az[cneg], medaz_dt[cneg], '.', label='Scan -')
        plt.xlabel('Azimuth [deg]')
        plt.ylabel('Ang. velocity [deg/s]')
        plt.legend(loc='upper left')
        plt.subplot(2,2,2)
        plt.title('Angular Velocity Vs. time - Dead time = {0:4.1f}%'.format(dead_time*100))
        plt.plot(thk, medaz_dt)
        plt.plot(thk[c0], medaz_dt[c0], 'ro', label='speed=0')
        plt.plot(thk[cpos], medaz_dt[cpos], '.', label='Scan +')
        plt.plot(thk[cneg], medaz_dt[cneg], '.', label='Scan -')
        plt.legend(loc='upper left')
        plt.xlabel('Time [s]')
        plt.ylabel('Ang. velocity [deg/s]')
        plt.xlim(plotrange[0],plotrange[1])
        plt.subplot(2,3,4)
        plt.title('Azimuth Vs. time - Dead time = {0:4.1f}%'.format(dead_time*100))
        plt.plot(thk, az)
        plt.plot(thk[c0], az[c0], 'ro', label='speed=0')
        plt.plot(thk[cpos], az[cpos], '.', label='Scan +')
        plt.plot(thk[cneg], az[cneg], '.', label='Scan -')
        plt.legend(loc='upper left')
        plt.xlabel('Time [s]')
        plt.ylabel('Azimuth [deg]')
        plt.xlim(plotrange[0],plotrange[1])
        plt.subplot(2,3,5)
        plt.title('Elevation Vs. time - Dead time = {0:4.1f}%'.format(dead_time*100))
        plt.plot(thk, el)
        plt.plot(thk[c0], el[c0], 'ro', label='speed=0')
        plt.plot(thk[cpos], el[cpos], '.', label='Scan +')
        plt.plot(thk[cneg], el[cneg], '.', label='Scan -')
        plt.legend(loc='lower left')
        plt.xlabel('Time [s]')
        plt.ylabel('Elevtion [deg]')
        plt.xlim(plotrange[0],plotrange[1])
        elvals = el[(thk > plotrange[0]) & (thk < plotrange[1])]
        deltael = np.max(elvals) - np.min(elvals)
        plt.ylim(np.min(elvals) - deltael/5, np.max(elvals) + deltael/5)

        allnums = np.unique(np.abs(scantype_hk))
        for n in allnums[allnums > 0]:
            ok = np.abs(scantype_hk) == n
            xx = np.mean(thk[ok])
            yy = np.mean(el[ok])
            if (xx > plotrange[0])  & (xx < plotrange[1]):
                plt.text(xx, yy+deltael/20, str(n))

        plt.subplot(2,3,6)
        plt.title('Elevation Vs. time - Dead time = {0:4.1f}%'.format(dead_time*100))
        thecol = (np.arange(len(allnums))*256/(len(allnums)-1)).astype(int)
        for i in range(len(allnums)):
            ok = np.abs(scantype_hk) == allnums[i]
            plt.plot(az[ok], el[ok], color=plt.get_cmap(plt.rcParams['image.cmap'])(thecol[i]))
        plt.ylim(np.min(el), np.max(el))
        plt.xlabel('Azimuth [deg]')
        plt.ylabel('Elevtion [deg]')
        plt.scatter(-allnums*0, -allnums*0-10,c=allnums)
        aa=plt.colorbar()
        aa.ax.set_ylabel('Scan number')
        #plt.tight_layout()
    if tt is not None:
        ### We propagate these at TOD sampling rate  (this is an "step interpolation": we do not want intermediatee values")
        scantype = interp1d(thk, scantype_hk, kind='previous', fill_value='extrapolate')(tt)
        scantype = scantype.astype(int)
        count_them = np.sum(scantype==0) + np.sum(scantype<=-1) + np.sum(scantype>=1)
        if count_them != len(scantype):
            print('Bad Scan counting at data sampling level - Error')
            stop
        ### Interpolate azimuth and elevation to TOD sampling
        azt = np.interp(tt, thk, az)
        elt = np.interp(tt, thk, el)
        ### Return evereything
        return scantype_hk, azt, elt, scantype
    else:
        ### Return scantype at HK sampling only
        return scantype_hk
def get_mode(y, nbinsmin=51):
    mm, ss = ft.meancut(y, 4)
    hh = np.histogram(y, bins=int(np.min([len(y) / 30, nbinsmin])), range=[mm - 5 * ss, mm + 5 * ss])
    idmax = np.argmax(hh[0])
    mymode = 0.5 * (hh[1][idmax + 1] + hh[1][idmax])
    return mymode

class BeamMapsAnalysis(object):

    """
    This class allows to make the same things that in Sample_demodulation.Rmd.
    It generates raw data and make all the analysis to obtain beam maps.
    """

    def __init__(self, a):

        """
        Parameters
        ----------
        a: Object from qubicpack
        """

        self.a = a
        self.thk = self.a.timeaxis(datatype='hk')
        self.nside=256#128
        self.mod_freq=self.a.hk['CALSOURCE-CONF']['Mod_freq'][0]
        if self.mod_freq < 0 :
            self.demod=False
        else:
            self.demod=True
        self.nharm=10
        self.lowcut=self.a.hk['CALSOURCE-CONF']['Amp_hfreq'][0]
        self.highcut=self.a.hk['CALSOURCE-CONF']['Amp_lfreq'][0]
        self.notch=np.array([[1.724, 0.005, self.nharm]])
        print('\n Loading data...\n')
        self.tt, self.tod = self.a.tod()
        print('Data loaded\n')
        # Interpolate tt
        self.azt = np.interp(self.tt, self.thk, self.a.azimuth())
        self.elt = np.interp(self.tt, self.thk, self.a.elevation())
        self.scantype_hk, self.azt, self.elt, self.scantype = identify_scans(self.thk, self.a.azimuth(), self.a.elevation(), tt=self.tt, doplot=False,
                                                                                                                                plotrange=[0,2000], thr_speedmin=0.1)

    def filter_data(self, tt, data, lowcut, highcut, doplot=False):

        filtered_data = ft.filter_data(tt, data, lowcut, highcut, notch=self.notch, rebin=True, verbose=True, order=5)

        if doplot:

            #plot limits
            thefreqmod = abs(self.a.hk['CALSOURCE-CONF']['Mod_freq'])
            #number of harmonics
            nharm = 10
            #filtering parameters
            period = 1./ thefreqmod
            xmin = 0.001
            xmax = 90.
            ymin = 1e0
            ymax = 1e13
            figure(figsize=(16, 8))
            ############ Power spectrum RAW plot
            spectrum_f, freq_f = ft.power_spectrum(tt, data, rebin=True)
            plot(freq_f, f.gaussian_filter1d(spectrum_f,1), label='Raw Data')
            spectrum_f2, freq_f2 = ft.power_spectrum(tt, filtered_data, rebin=True)
            plot(freq_f2, f.gaussian_filter1d(spectrum_f2,1), label='Filtered Data')
            #plot([self.lowcut, self.lowcut],[ymin,ymax],'k', lw=3, label='Bandpass')
            #plot([self.highcut, self.highcut],[ymin,ymax],'k', lw=3)
            #plot([1./period,1./period],[ymin,ymax],'k--', lw=3, alpha=0.3, label='Calsource Harmonics')
            #for i in range(10):
            #    plot([1./period*i,1./period*i],[ymin,ymax],'k--', lw=3, alpha=0.3)
            #plot the pulse tube harmoncs
            #plot([self.notch[0,0],self.notch[0,0]], [ymin,ymax],'m:', lw=3, label='Pulse Tube Harmonics')
            #for i in range(nharm):
            #    plot([self.notch[0,0]*(i+1),self.notch[0,0]*(i+1)], [ymin,ymax],'m:', lw=3)
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
    def make_healpix_map(self, azt, elt, tod, countcut=0):
        unseen_val=hp.UNSEEN
        ips = hp.ang2pix(self.nside, azt, elt, lonlat=True)
        mymap = np.zeros(12*self.nside**2)
        mapcount = np.zeros(12*self.nside**2)
        for i in range(len(azt)):
            mymap[ips[i]] -= tod[i]
            mapcount[ips[i]] += 1
        unseen = mapcount <= countcut
        mymap[unseen] = unseen_val
        mapcount[unseen] = unseen_val
        mymap[~unseen] = mymap[~unseen] / mapcount[~unseen]
        return mymap
    def remove_noise(self, tt, tod):

        hf_noise = hf_noise_estimate(tt, tod) / np.sqrt(2)
        var_diff = tod ** 2 - hf_noise ** 2
        tod = np.sqrt(np.abs(var_diff)) * np.sign(var_diff)
        return tod
    def fullanalysis(self, number_of_tes=None, filter=None, demod=False, remove_noise=True, make_maps='healpix', doplot=True, save=False):

        # Generate TOD from qubicpack
        #print('Get Raw data')
        #tt, tod = self.get_raw_data()


        if number_of_tes == None:
            sh=np.arange(1, self.tod.shape[0]+1, 1)
        else:
            sh=[number_of_tes]

        mymaps_flat=np.zeros((len(sh), 101, 101))
        mymaps_hp=np.zeros((len(sh), 12*self.nside**2))

        for ish, i in enumerate(sh) :
            print('Analyse TES {:.0f} / {:.0f}'.format(i, len(sh)))
            data=self.tod[i-1].copy()

            if remove_noise:
                data=self.remove_noise(self.tt, data)
            # Filtering
            if filter is not None:
                print('Filtering')
                lowcut=filter[0]
                highcut=filter[1]
                filtered_data=self.filter_data(self.tt, data, doplot=doplot, lowcut=lowcut, highcut=highcut)
                data=filtered_data.copy()

            # Demodulation
            if demod:
                print('Demodulation')
                newt, data=self.demodulation(self.tt, data, remove_noise=remove_noise)

            if make_maps=='healpix':
                print('Making healpix maps')
                mymaps_hp[ish]=self.make_healpix_map(self.azt, self.elt, data)
            elif make_maps=='flat':
                print('Make Flat Maps')
                mymaps_flat[ish]=self.make_flat_map(self.tt, data, doplot=doplot)

            if save:
                if make_maps=='healpix':
                    self.save_azel_mymap_hp(mymaps_hp[ish], TESNum=i)
                elif make_maps=='flat':
                    self.save_azel_mymap(mymaps_flat[ish], TESNum=i)

        if make_maps=='healpix':
            print('Making healpix maps')
            mymaps=mymaps_hp.copy()
        elif make_maps=='flat':
            print('Make Flat Maps')
            mymaps=mymaps_flat.copy()

        return mymaps



    def save_azel_mymap(self, mymap, TESNum):

        repository=os.getcwd()+'/Fits/Flat'
        try:
            os.makedirs(repository)
        except OSError:
            if not os.path.isdir(repository):
                raise

        FitsArray(mymap).save(repository+'/imgflat_TESNum_{}.fits'.format(TESNum))

    def save_azel_mymap_hp(self, mymap, TESNum):

        repository=os.getcwd()+'/Fits/Healpix'
        try:
            os.makedirs(repository)
        except OSError:
            if not os.path.isdir(repository):
                raise

        FitsArray(mymap).save(repository+'/imgHealpix_TESNum_{}.fits'.format(TESNum))


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

def _read_fits_beam_maps(TESNum):
    from astropy.io import fits as pyfits
    path=os.getcwd()
    folder='/Fits/Flat/'
    name='imgflat_TESNum_{:.0f}.fits'.format(TESNum)
    allpath=path+folder+name
    #print(allpath)

    hdulist = pyfits.open(allpath)
    header = hdulist[0].header

    return hdulist[0].data
