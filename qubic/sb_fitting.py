from __future__ import division, print_function

from qubicpack import qubicpack as qp
import qubic.fibtools as ft
import qubic.plotters as p
import qubic.lin_lib as ll
import qubic
# import demodulation_lib as dl

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
import time
import numexpr as ne
import pickle


def get_hpmap(TESNum, directory):
    return np.array(qubic.io.read_map(directory + '/Healpix/healpix_TESNum_{}.fits'.format(TESNum)))


def get_lines(lines, directory):
    nn = len(lines)
    hpmaps = np.zeros((nn, 4, 12 * 256 ** 2))
    nums = np.zeros((nn, 4), dtype=int)
    for l in range(nn):
        for i in range(4):
            if lines[l] < 33:
                nums[l, i] = int(lines[l] + 32 * i)
            else:
                nums[l, i] = int(lines[l] - 32 + 32 * i) + 128
            hpmaps[l, i, :] = get_hpmap(nums[l, i], directory)
    return hpmaps, nums


def show_lines(maps, nums, min=None, max=None):
    sh = np.shape(maps)
    nl = sh[0]
    for l in range(nl):
        for i in range(4):
            hp.gnomview(maps[l, i, :], reso=10, min=min, max=max, sub=(nl, 4, l * 4 + i + 1), title=nums[l, i])
    tight_layout()


def get_flatmap(TESNum, directory, azmin=None, azmax=None, elmin=None, elmax=None, remove=None,
                fitted_directory=None):
    themap = np.array(FitsArray(directory + '/Flat/imgflat_TESNum_{}.fits'.format(TESNum)))
    az = np.array(FitsArray(directory + '/Flat/azimuth.fits'.format(TESNum)))
    el = np.array(FitsArray(directory + '/Flat/elevation.fits'.format(TESNum)))
    if azmin is None:
        azmin = np.min(az)
    if azmax is None:
        azmax = np.max(az)
    if elmin is None:
        elmin = np.min(el)
    if elmax is None:
        elmax = np.max(el)
    okaz = (az >= azmin) & (az <= azmax)
    az = az[okaz]
    okel = (el >= elmin) & (el <= elmax)
    el = el[okel]
    themap = themap[:, okaz][okel, :]
    if remove is not None:
        mm = np.mean(remove)
        ss = np.std(remove)
        xc, yval, dx, dy, others = ft.profile(remove, themap, rng=[mm - 2 * ss, mm + 2 * ss], mode=True,
                                              nbins=20, cutbad=True, plot=False, dispersion=False, clip=3)
        bla = np.polyfit(xc, yval, 1, w=1. / dy ** 2)
        pp = np.poly1d(bla)
        themap -= pp(remove)

    if fitted_directory is None:
        return themap, az, el
    else:
        ### Read fitted synthesized beam
        thefile = open(fitted_directory + '/fit-TES{}.pk'.format(TESNum), 'rb')
        fitpars = pickle.load(thefile, encoding='latin1')
        thefile.close()
        sbfitmodel = SbModelIndepPeaks(nrings=2, common_fwhm=True, no_xy_shift=False, distortion=False)
        x = np.meshgrid(az * np.cos(np.radians(50)), np.flip(el))
        fitmap, newxxyy = sbfitmodel(x, fitpars, return_peaks=True)
        return themap, az, el, fitmap, newxxyy


def show_flatmap(directory, TESNum, vmin=None, vmax=None, cbar=False):
    flatmap, az, el = get_flatmap(TESNum, directory)
    imshow(flatmap, extent=[np.min(az) * np.cos(np.radians(50)),
                            np.max(az) * np.cos(np.radians(50)),
                            np.min(el), np.max(el)], vmin=vmin, vmax=vmax)
    title('TES #{}'.format(TESNum))
    xlabel('Az angle')
    ylabel('El')
    if cbar:
        colorbar()


def show_flatmaps_list(directory, TESNums, vmin=None, vmax=None, cbar=False, nx=5, tight=True):
    nn = len(TESNums)
    ny = nn / nx + 1
    ii = 1
    for nn in TESNums:
        subplot(ny, nx, ii)
        show_flatmap(directory, nn, vmin=vmin, vmax=vmax, cbar=cbar)
        ii += 1
    if tight:
        tight_layout()


def beeps(nbeeps):
    import os
    beep = lambda x: os.system("echo -n '\a';sleep 0.2;" * x)
    beep(nbeeps)


def thph2uv(th, ph):
    sth = np.sin(th)
    cth = np.cos(th)
    sph = np.sin(ph)
    cph = np.cos(ph)
    return np.array([sth * cph, sth * sph, cth])


def uv2thph(uv):
    r = np.sum(uv ** 2, axis=0)
    th = np.nan_to_num(np.arccos(uv[2] / r))
    ph = np.arctan2(uv[1], uv[0])
    return np.array([th, ph])


def rotmatX(th):
    cth = np.cos(th)
    sth = np.sin(th)
    rotmat = np.array([[1, 0, 0], [0, cth, -sth], [0, sth, cth]])
    return rotmat


def rotmatY(th):
    cth = np.cos(th)
    sth = np.sin(th)
    rotmat = np.array([[cth, 0, sth], [0, 1, 0], [-sth, 0, cth]])
    return rotmat


def rotmatZ(th):
    cth = np.cos(th)
    sth = np.sin(th)
    rotmat = np.array([[cth, -sth, 0], [sth, cth, 0], [0, 0, 1]])
    return rotmat


### Rotation from QUBIC reference frame to the measurement one
def rotate_q2m(thin, phin, angs=np.radians(np.array([0., 90., 0.])), inverse=False):
    #### All Angles in Radians
    uvecin = thph2uv(thin, phin)
    ### First rotation: pitch angle around optical axis
    r0 = rotmatZ(angs[0])
    ### Second rotation: down to actual elevation
    r1 = rotmatY(angs[1])
    ### Third rotation: go to actual azimuth
    r2 = rotmatZ(angs[2])
    # uvecout = np.dot(r2, np.dot(r1, np.dot(r0, uvecin)))
    R = np.dot(r2, np.dot(r1, r0))
    if inverse:
        Rinv = np.linalg.inv(R)
        uvecout = np.dot(Rinv, uvecin)
    else:
        uvecout = np.dot(R, uvecin)
    return uv2thph(uvecout)


#######################################################################################################################################
# ######################################################################################################################################
class SimpleSbModel:
    """
    Class defining the simplest Synthesized Beam model for QUBIC:
    - A square grid of n Gaussian Peaks with the same symmetric FWHM.
    - n is defined by nrings (1=>1 peak, 2=>9 peaks, ...)
    - The square grid has a rotation angle
    - The amplitude of the peaks is given by a common Primary Gaussian Beam whose FWHM and center is fit
    - A distorsion to the position of the peaks is allowed

    So Finally parameters are:
    [0]: Az center of the square [deg]
    [1]: El center of the square [deg]
    [2]: Interpeak distance [deg]
    [3]: Orientation angle [deg]
    [4]: X distorsion magnitude
    [5]: X distosion power
    [6]: Y distorsion magnitude
    [7]: Y distorsion power
    [8]: FWHM of the peaks [deg]
    [9]: Amplitude of primary beam
    [10]: Az center of primary beam [deg]
    [11]: El center of primary beam [deg]
    [12]: FWHM of primary beam [deg]
    """

    def __init__(self, startpars=None, ranges=None, fixpars=None, nrings=2, extra_args=None):
        ### Preparing the grid of peaks
        self.name = 'SimpleSbModel'
        self.nrings = nrings
        self.npeaks_line = 2 * nrings - 1
        self.npeaks = self.npeaks_line ** 2
        x = (np.arange(self.npeaks_line) - (self.nrings - 1))
        xx, yy = np.meshgrid(x, x)
        self.xxyy = np.array([np.ravel(xx), np.ravel(yy)])

        ### Parameters names
        self.npars = 13
        self.parnames = ['AzCenter', 'ElCenter', 'PeakDist', 'Angle', 'XDistAmp', 'XDistPow', 'YDistAmp', 'YDistPow',
                         'FWHMPeaks',
                         'PrimAmp', 'PrimAzCenter', 'PrimElCenter', 'PrimFWHM']
        ### Default parameters
        if startpars is None:
            self.startpars = np.array([0., 50., 8., 45., 0.0, 2., 0.0, 2., 0.9, 1e5, 0., 50., 13.])
        else:
            self.startpars = startpars

        ### Fixed parameters
        if fixpars is None:
            self.fixpars = np.zeros(len(self.startpars))
        else:
            self.fixpars = fixpars

        ### Range allowed for fitting
        if ranges is None:
            self.ranges = np.array([[-5., 45., 7., 40., 0.0, 0., 0.0, 0., 0.5, 0.0, -3., 47., 10.],
                                    [5., 55., 9., 50., 0.1, 4., 0.1, 4., 1.5, 1e6, 3., 53., 16.]])
        else:
            self.ranges = ranges

        ### Possible extra-arguments
        self.extra_args = extra_args

    def __call__(self, x, pars, return_peaks=False):
        x2d = x[0]  # The Azimuth values of the pixels
        y2d = x[1]  # The elevation values of the pixels
        # The parameters ##########################################
        xc = pars[0]
        yc = pars[1]
        dist = pars[2]
        angle = pars[3]
        distx = pars[4]
        distpowerX = pars[5]
        disty = pars[6]
        distpowerY = pars[7]
        fwhmpeaks = pars[8]
        ampgauss = pars[9]
        xcgauss = pars[10]
        ycgauss = pars[11]
        fwhmgauss = pars[12]

        # Peaks positions #########################################
        # Rotate initial Grid centered on (0,0)
        cosang = np.cos(np.radians(angle))
        sinang = np.sin(np.radians(angle))
        rotmat = np.array([[cosang, -sinang], [sinang, cosang]])
        newxxyy = np.zeros((4, self.npeaks))
        for i in range(self.npeaks):
            newxxyy[0:2, i] = np.dot(rotmat, self.xxyy[:, i])
        # Scale it wit interpeak distance
        newxxyy *= dist
        # Apply Distorsions
        undistxxyy = newxxyy.copy()
        newxxyy[0, :] += distx * np.abs(undistxxyy[1, :]) ** distpowerX
        newxxyy[1, :] += disty * np.abs(undistxxyy[0, :]) ** distpowerY
        # Move to actual center
        newxxyy[0, :] += xc
        newxxyy[1, :] += yc

        ### Peak amplitudes and resulting map #######################
        themap = np.zeros(x2d.shape)
        amps = np.zeros(self.npeaks)
        for i in range(self.npeaks):
            amp = ampgauss * np.exp(
                -0.5 * ((xcgauss - newxxyy[0, i]) ** 2 + (ycgauss - newxxyy[1, i]) ** 2) / (fwhmgauss / 2.35) ** 2)
            x0 = newxxyy[0, i]
            y0 = newxxyy[1, i]
            sig = fwhmpeaks / 2.35
            themap += ne.evaluate("amp * exp(-0.5*((x2d-x0)**2 + (y2d-y0)**2)/sig**2)")

        newxxyy[2, :] = amps
        newxxyy[3, :] = fwhmpeaks

        if return_peaks:
            return themap, newxxyy
        else:
            return np.ravel(themap)

    def print_start(self):
        print('|---------------------------------------------------------------------|')
        print('|-------------------- Initial Parameters -----------------------------|')
        print('|---------------------------------------------------------------------|')
        print('|Parameter        | init-value | range0      | range1      |  fixed   |')
        print('|---------------------------------------------------------------------|')
        for i in range(self.npars):
            print('|{0:<16} | {1:>10.3f} |{2:>13.3f}|{3:>13.3f}|    {4:^3}   |'.format(self.parnames[i],
                                                                                       self.startpars[i],
                                                                                       self.ranges[0, i],
                                                                                       self.ranges[1, i],
                                                                                       self.fixpars[i]))
        print('|---------------------------------------------------------------------|')


#######################################################################################################################################
# ######################################################################################################################################


#######################################################################################################################################
# ######################################################################################################################################
class SbModelIndepPeaksAmpFWHM:
    """
    Class defining the simplest Synthesized Beam model for QUBIC:
    - A square grid of n Gaussian Peaks with the same symmetric FWHM.
    - n is defined by nrings (1=>1 peak, 2=>9 peaks, ...)
    - The square grid has a rotation angle
    - The amplitude of the peaks are independent
    - The FWHM of the peaks can be all the same or different depending on the keyword common_fwhm
      If common_fwhm is True then this model is equivalent to SbModelIndepPeaksAmp
    - A distorsion to the position of the peaks is allowed

    So Finally parameters are:
    [0]: Az center of the square [deg]
    [1]: El center of the square [deg]
    [2]: Interpeak distance [deg]
    [3]: Orientation angle [deg]
    [4]: X distorsion magnitude
    [5]: X distosion power
    [6]: Y distorsion magnitude
    [7]: Y distorsion power
    [8]: Average FWHM of the peaks
    [9+2*ipeak]: Amplitudes of the peaks
    [9+2*ipeak+1]: Delta FWHM of each peak to be added to the average one [deg]
    """

    def __init__(self, startpars=None, ranges=None, fixpars=None, nrings=2, extra_args=None, verbose=False,
                 common_fwhm=False):
        ### Preparing the grid of peaks
        self.name = 'SbModelIndepPeaksAmpFWHM'
        self.common_fwhm = common_fwhm
        self.nrings = nrings
        self.npeaks_line = 2 * nrings - 1
        self.npeaks = self.npeaks_line ** 2
        x = (np.arange(self.npeaks_line) - (self.nrings - 1))
        xx, yy = np.meshgrid(x, x)
        self.xxyy = np.array([np.ravel(xx), np.ravel(yy)])

        npars = 9 + self.npeaks * 2
        self.npars = npars
        ### Parameters names
        self.parnames = np.repeat('              ', npars)
        firstparnames = ['AzCenter', 'ElCenter', 'PeakDist', 'Angle', 'XDistAmp', 'XDistPow', 'YDistAmp', 'YDistPow',
                         'FWHMPeaksAv']
        for i in range(9):
            self.parnames[i] = firstparnames[i]
        for i in range(self.npeaks):
            self.parnames[9 + 2 * i] = 'AmpPeak{}'.format(i)
            self.parnames[9 + 2 * i + 1] = 'DeltaFWHMPeak{}'.format(i)

        ### Default parameters
        if startpars is None:
            self.startpars = np.zeros(npars) * 1.
            self.startpars[0:9] = np.array([0., 50., 8., 45., 0.0, 2., 0.0, 2., 0.9])
            for i in range(self.npeaks):
                self.startpars[9 + 2 * i] = 1e5
                self.startpars[9 + 2 * i + 1] = 0.
        else:
            self.startpars = startpars

        ### Fixed parameters
        if fixpars is None:
            self.fixpars = np.zeros(len(self.startpars))
            ### A subtlety here: if we ask for a fit with common FWHM for all peaks, then we need to fix the individual
            ### FWHMs to zero (as at the end the FWHM used is the sum of the two). In the other case, when we want to fit
            ### diffferent FWHM for each, then the average fwhm is fixed while the others can var around zero
            if self.common_fwhm:
                for i in range(self.npeaks):
                    self.fixpars[9 + 2 * i + 1] = 1
            else:
                self.fixpars[8] = 1
        else:
            self.fixpars = fixpars

        ### Range allowed for fitting
        if ranges is None:
            self.ranges = np.zeros((2, npars))
            self.ranges[:, 0:9] = np.array([[-5., 45., 7., 40., 0.0, 0., 0.0, 0., 0.5],
                                            [5., 55., 9., 50., 0.1, 4., 0.1, 4., 1.5]])
            for i in range(self.npeaks):
                self.ranges[0, 9 + 2 * i] = 0.
                self.ranges[1, 9 + 2 * i] = 1e6
                self.ranges[0, 9 + 2 * i + 1] = -0.5
                self.ranges[1, 9 + 2 * i + 1] = 0.5
        else:
            self.ranges = ranges

        ### Possible extra-arguments
        self.extra_args = extra_args

        if verbose: self.print_start()

    def __call__(self, x, pars, return_peaks=False):
        x2d = x[0]  # The Azimuth values of the pixels
        xmin = np.min(x2d)
        xmax = np.max(x2d)
        y2d = x[1]  # The elevation values of the pixels
        ymin = np.min(y2d)
        ymax = np.max(y2d)
        # The parameters ##########################################
        xc = pars[0]
        yc = pars[1]
        dist = pars[2]
        angle = pars[3]
        distx = pars[4]
        distpowerX = pars[5]
        disty = pars[6]
        distpowerY = pars[7]
        fwhmpeaks_av = pars[8]
        amps = np.zeros(self.npeaks)
        fwhmpeaks = np.zeros(self.npeaks)
        for i in range(self.npeaks):
            amps[i] = pars[9 + 2 * i]
            fwhmpeaks[i] = fwhmpeaks_av + pars[9 + 2 * i + 1]

        ### Peaks positions #########################################
        # Rotate initial Grid centered on (0,0)
        cosang = np.cos(np.radians(angle))
        sinang = np.sin(np.radians(angle))
        rotmat = np.array([[cosang, -sinang], [sinang, cosang]])
        newxxyy = np.zeros((4, self.npeaks))
        for i in range(self.npeaks): newxxyy[0:2, i] = np.dot(rotmat, self.xxyy[:, i])
        # Scale it wit interpeak distance
        newxxyy *= dist
        # Apply Distorsions
        undistxxyy = newxxyy.copy()
        yok = undistxxyy[1, :] != 0
        newxxyy[0, yok] += distx * np.abs(undistxxyy[1, yok]) ** distpowerX
        xok = undistxxyy[0, :] != 0
        newxxyy[1, xok] += disty * np.abs(undistxxyy[0, xok]) ** distpowerY
        # Move to actual center
        newxxyy[0, :] += xc
        newxxyy[1, :] += yc
        if ~(np.product(np.isfinite(newxxyy)).astype(bool)):
            stop

        ### Peak amplitudes and resulting map #######################
        # themap = np.zeros(x2d.shape)
        # for i in range(self.npeaks):
        # 	themap += amps[i]*np.exp(-((x2d-newxxyy[0,i])**2 +(y2d-newxxyy[1,i])**2)/(2*(fwhmpeaks[i]/2.35)**2) )
        themap = np.zeros(x2d.shape)
        for i in range(self.npeaks):
            amp = amps[i]
            x0 = newxxyy[0, i]
            y0 = newxxyy[1, i]
            sig = fwhmpeaks[i] / 2.35
            themap += ne.evaluate("amp * exp(-0.5*((x2d-x0)**2 + (y2d-y0)**2)/sig**2)")

        newxxyy[2, :] = amps
        newxxyy[3, :] = fwhmpeaks

        if return_peaks:
            return themap, newxxyy
        else:
            return np.ravel(themap)

    def print_start(self):
        print('|---------------------------------------------------------------------|')
        print('|-------------------- Initial Parameters -----------------------------|')
        print('|            Important: common_fwhm = {0:}                            |'.format(self.common_fwhm))
        print('|---------------------------------------------------------------------|')
        print('|Parameter        | init-value | range0      | range1      |  fixed   |')
        print('|---------------------------------------------------------------------|')
        for i in range(self.npars):
            print('|{0:<16} | {1:>10.3f} |{2:>13.3f}|{3:>13.3f}|    {4:^3}   |'.format(self.parnames[i],
                                                                                       self.startpars[i],
                                                                                       self.ranges[0, i],
                                                                                       self.ranges[1, i],
                                                                                       self.fixpars[i].astype(int)))
        print('|---------------------------------------------------------------------|')


#######################################################################################################################################
# ######################################################################################################################################


#######################################################################################################################################
# ######################################################################################################################################
class SbModelIndepPeaks:
    """
    Class defining the simplest Synthesized Beam model for QUBIC:
    - A square grid of n Gaussian Peaks with the same symmetric FWHM.
    - n is defined by nrings (1=>1 peak, 2=>9 peaks, ...)
    - The square grid has a rotation angle
    - The amplitude of the peaks are independent
    - The FWHM of the peaks can be all the same or different depending on the keyword common_fwhm
      If common_fwhm is True then this model is equivalent to SbModelIndepPeaksAmp
    - A global distorsion to the position of the peaks is allowed
    - A small shift of each peak position is allowed around the initial position

    So if common_fwhm=True and no_xy_shift=True and distrotion=True this model is equivalent to SbModelIndepPeaksAmp
    So if common_fwhm=False and no_xy_shift=True and distortion=True this model is equivalent to SbModelIndepPeaksAmpFWHM

    So Finally parameters are:
    [0]: Az center of the square [deg]
    [1]: El center of the square [deg]
    [2]: Interpeak distance [deg]
    [3]: Orientation angle [deg]
    [4]: X distorsion magnitude
    [5]: X distosion power
    [6]: Y distorsion magnitude
    [7]: Y distorsion power
    [8]: Average FWHM of the peaks
    [9+4*ipeak]: Amplitudes of the peaks
    [9+4*ipeak+1]: Delta FWHM of each peak to be added to the average one [deg]
    [9+4*ipeak+2]: X shift of the peak [deg]
    [9+4*ipeak+3]: Y shift of the peak [deg]
    """

    def __init__(self, startpars=None, ranges=None, fixpars=None, nrings=2, extra_args=None, verbose=False,
                 common_fwhm=False, no_xy_shift=False, distortion=True):
        ### Preparing the grid of peaks
        self.name = 'SbModelIndepPeaks'
        self.common_fwhm = common_fwhm
        self.no_xy_shift = no_xy_shift
        self.distortion = distortion
        self.nrings = nrings
        self.npeaks_line = 2 * nrings - 1
        self.npeaks = self.npeaks_line ** 2
        x = (np.arange(self.npeaks_line) - (self.nrings - 1))
        xx, yy = np.meshgrid(x, x)
        self.xxyy = np.array([np.ravel(xx), np.ravel(yy)])

        npars = 9 + self.npeaks * 4
        self.npars = npars
        ### Parameters names
        self.parnames = np.repeat('              ', npars)
        firstparnames = ['AzCenter', 'ElCenter', 'PeakDist', 'Angle', 'XDistAmp', 'XDistPow', 'YDistAmp', 'YDistPow',
                         'FWHMPeaksAv']
        for i in range(9):
            self.parnames[i] = firstparnames[i]
        for i in range(self.npeaks):
            self.parnames[9 + 4 * i] = 'AmpPeak{}'.format(i)
            self.parnames[9 + 4 * i + 1] = 'DeltaFWHMPeak{}'.format(i)
            self.parnames[9 + 4 * i + 2] = 'DeltaXPeak{}'.format(i)
            self.parnames[9 + 4 * i + 3] = 'DeltaYPeak{}'.format(i)

        ### Default parameters
        if startpars is None:
            self.startpars = np.zeros(npars) * 1.
            self.startpars[0:9] = np.array([0., 50., 8., 45., 0.0, 2., 0.0, 2., 0.9])
            for i in range(self.npeaks):
                self.startpars[9 + 4 * i] = 1e5
                self.startpars[9 + 4 * i + 1] = 0.
                self.startpars[9 + 4 * i + 2] = 0.
                self.startpars[9 + 4 * i + 3] = 0.
        else:
            self.startpars = startpars

        ### Fixed parameters
        if fixpars is None:
            self.fixpars = np.zeros(len(self.startpars))
            ### A subtlety here: if we ask for a fit with common FWHM for all peaks, then we need to fix the individual
            ### FWHMs to zero (as at the end the FWHM used is the sum of the two). In the other case, when we want to fit
            ### diffferent FWHM for each, then the average fwhm is fixed while the others can var around zero
            if self.common_fwhm:
                for i in range(self.npeaks):
                    self.fixpars[9 + 4 * i + 1] = 1
            else:
                self.fixpars[8] = 1

            if self.no_xy_shift:
                for i in range(self.npeaks):
                    self.fixpars[9 + 4 * i + 2] = 1
                    self.fixpars[9 + 4 * i + 3] = 1
            if self.distortion is False:
                self.fixpars[4] = 1
                self.fixpars[5] = 1
                self.fixpars[6] = 1
                self.fixpars[7] = 1
        else:
            self.fixpars = fixpars

        ### Range allowed for fitting
        if ranges is None:
            self.ranges = np.zeros((2, npars))
            self.ranges[:, 0:9] = np.array([[-5., 45., 7., 40., 0.0, 0., 0.0, 0., 0.5],
                                            [5., 55., 9., 50., 0.1, 4., 0.1, 4., 1.5]])
            for i in range(self.npeaks):
                self.ranges[0, 9 + 4 * i] = 0.
                self.ranges[1, 9 + 4 * i] = 1e6
                self.ranges[0, 9 + 4 * i + 1] = -0.5
                self.ranges[1, 9 + 4 * i + 1] = 0.5
                self.ranges[0, 9 + 4 * i + 2] = -2
                self.ranges[1, 9 + 4 * i + 2] = 2
                self.ranges[0, 9 + 4 * i + 3] = -2
                self.ranges[1, 9 + 4 * i + 3] = 2
        else:
            self.ranges = ranges

        ### Possible extra-arguments
        self.extra_args = extra_args
        self.time = time.time()
        self.ncalls = 1

        if verbose:
            self.print_start()

    def __call__(self, x, pars, return_peaks=False):
        # t0 = time.time()
        # self.ncalls += 1
        # print('Call #{0} - {1:5.2f} ms '.format(self.ncalls, 1000*(t0-self.time)))
        # self.time = t0
        x2d, y2d = x  # The Azimuth and elevation values of the pixels

        # The parameters ##########################################
        xc = pars[0]
        yc = pars[1]
        dist = pars[2]
        angle = pars[3]
        distx = pars[4]
        distpowerX = pars[5]
        disty = pars[6]
        distpowerY = pars[7]
        fwhmpeaks_av = pars[8]
        amps = np.zeros(self.npeaks)
        fwhmpeaks = np.zeros(self.npeaks)
        dx = np.zeros(self.npeaks)
        dy = np.zeros(self.npeaks)
        for i in range(self.npeaks):
            amps[i] = pars[9 + 4 * i]
            fwhmpeaks[i] = fwhmpeaks_av + pars[9 + 4 * i + 1]
            dx[i] = pars[9 + 4 * i + 2]
            dy[i] = pars[9 + 4 * i + 3]

        ### Peaks positions #########################################
        # Rotate initial Grid centered on (0,0)
        cosang = np.cos(np.radians(angle))
        sinang = np.sin(np.radians(angle))
        rotmat = np.array([[cosang, -sinang], [sinang, cosang]])
        newxxyy = np.zeros((4, self.npeaks))
        for i in range(self.npeaks): newxxyy[0:2, i] = np.dot(rotmat, self.xxyy[:, i])
        # Scale it wit interpeak distance
        newxxyy *= dist
        # Apply Distorsions
        undistxxyy = newxxyy.copy()
        yok = undistxxyy[1, :] != 0
        newxxyy[0, yok] += distx * np.abs(undistxxyy[1, yok]) ** distpowerX
        xok = undistxxyy[0, :] != 0
        newxxyy[1, xok] += disty * np.abs(undistxxyy[0, xok]) ** distpowerY
        # Move to actual center
        newxxyy[0, :] += xc + dx
        newxxyy[1, :] += yc + dy
        if ~(np.product(np.isfinite(newxxyy)).astype(bool)):
            stop

        ### Peak amplitudes and resulting map #######################
        # themap = np.zeros(np.shape(x2d))
        # for i in range(self.npeaks):
        # 	themap += amps[i]*np.exp(-((x2d-newxxyy[0,i])**2 +(y2d-newxxyy[1,i])**2)/(2*(fwhmpeaks[i]/2.35)**2) )

        ### Peak amplitudes and resulting map #######################
        ### Much faster version (gain ~3.5)
        themap = np.zeros(x2d.shape)
        for i in range(self.npeaks):
            amp = amps[i]
            x0 = newxxyy[0, i]
            y0 = newxxyy[1, i]
            sig = fwhmpeaks[i] / 2.35
            themap += ne.evaluate("amp * exp(-0.5*((x2d-x0)**2 + (y2d-y0)**2)/sig**2)")

        newxxyy[2, :] = amps
        newxxyy[3, :] = fwhmpeaks

        if return_peaks:
            return themap, newxxyy
        else:
            return np.ravel(themap)

    def print_start(self):
        print('|---------------------------------------------------------------------|')
        print('|-------------------- Initial Parameters -----------------------------|')
        print('|            Important: common_fwhm = {0:}                            |'.format(self.common_fwhm))
        print('|            Important: no_xy_shift = {0:}                            |'.format(self.no_xy_shift))
        print('|---------------------------------------------------------------------|')
        print('|Parameter        | init-value | range0      | range1      |  fixed   |')
        print('|---------------------------------------------------------------------|')
        for i in range(self.npars):
            print('|{0:<16} | {1:>10.3f} |{2:>13.3f}|{3:>13.3f}|    {4:^3}   |'.format(self.parnames[i],
                                                                                       self.startpars[i],
                                                                                       self.ranges[0, i],
                                                                                       self.ranges[1, i],
                                                                                       self.fixpars[i].astype(int)))
        print('|---------------------------------------------------------------------|')


#######################################################################################################################################
# ######################################################################################################################################


def fit_sb(flatmap_init, az_init, el_init, model, newsize=70, dmax=5., az_center=0., el_center=50.,
           doplot=False,
           resample=False, verbose=False, extra_title='', return_fitted=False, precision=None, nsiglo=1., nsighi=3.,
           figsave=None):
    #### If requested, resample the iage in order to speedup the fitting
    if resample:
        ### Resample input map to have less pixels to deal with for fitting
        flatmap = scsig.resample(scsig.resample(flatmap_init, newsize, axis=0), newsize, axis=1)
        az = np.linspace(np.min(az_init), np.max(az_init), newsize)
        el = np.linspace(np.min(el_init), np.max(el_init), newsize)
    else:
        flatmap = flatmap_init
        az = np.array(az_init)
        el = np.array(el_init)
    az2d, el2d = np.meshgrid(az * np.cos(np.radians(el_center)), np.flip(el))

    # if verbose:
    # 	print('fit_sb: Model Name = ',model.name)
    # 	print('Initial Parameters:')
    # 	model.print_start()

    ### First find the location of the maximum closest to the center
    distance_max = dmax
    mask = np.array(np.sqrt((az2d - az_center) ** 2 + (el2d - el_center) ** 2) < distance_max).astype(int)
    ### We use a smoothed version of the map in order to avoid glitches
    smoothed_flatmap = f.gaussian_filter(flatmap, 2.)
    wmax = np.where((smoothed_flatmap * mask) == np.max(smoothed_flatmap * mask))
    maxval = flatmap[wmax][0]
    # print('Maximum of map is {0:5.2g} and was found at: az={1:5.2f}, el={2:5.2f}'.format(maxval,az2d[wmax][0], el2d[wmax][0]))

    ### Get the fixed parameters
    fixpars = model.fixpars

    ### Create range for the fitting around the initial values
    ranges = model.ranges.T

    ### Update initial parameters with the values found on the map
    parsinit = model.startpars
    parsinit[0] = az2d[wmax][0]
    parsinit[1] = el2d[wmax][0]
    parsinit[9] = maxval
    if model.name == 'SbModelIndepPeaksAmp':
        parsinit[9:] = maxval
        ranges[9:, 1] = maxval * 2
    elif model.name == 'SbModelIndepPeaksAmpFWHM':
        for i in range(model.npeaks):
            parsinit[9 + 2 * i] = maxval
            ranges[9 + 2 * i, 1] = maxval * 2
    elif model.name == 'SbModelIndepPeaks':
        for i in range(model.npeaks):
            parsinit[9 + 4 * i] = maxval
            ranges[9 + 4 * i, 1] = np.abs(maxval) * 2

    ### Run the fitting
    x = [az2d, el2d]
    mm, ss = ft.meancut(flatmap, 3)
    if verbose:
        print('Running Minuit with model: {}'.format(model.name))
        model.print_start()
    fit = ft.do_minuit(x, np.ravel(flatmap), np.zeros_like(np.ravel(flatmap)) + ss, parsinit,
                       functname=model, chi2=ft.MyChi2_nocov, rangepars=ranges, fixpars=fixpars,
                       force_chi2_ndf=False, verbose=False, nohesse=True, precision=precision)
    fitpars = fit[1]
    fiterrs = fit[2]

    ### Get the peaks positions and amplitudes
    themap, newxxyy = model(x, fitpars, return_peaks=True)
    ### Put the fitted amplitude values to zero when needed (when peak ouside the input image)
    if model.name == 'SbModelIndepPeaksAmp':
        xmin = np.min(az2d)
        xmax = np.max(az2d)
        ymin = np.min(el2d)
        ymax = np.max(el2d)
        for i in range(model.npeaks):
            if (((newxxyy[0, i] < xmin) or (newxxyy[0, i] > xmax)) or (
                    (newxxyy[1, i] < ymin) or (newxxyy[1, i] > ymax))):
                fitpars[9 + i] = 0
                fiterrs[9 + i] = 0
        ### Now get the map again with updated parameters
        themap, newxxyy = model(x, fitpars, return_peaks=True)
    elif model.name == 'SbModelIndepPeaksAmpFWHM':
        xmin = np.min(az2d)
        xmax = np.max(az2d)
        ymin = np.min(el2d)
        ymax = np.max(el2d)
        for i in range(model.npeaks):
            if (((newxxyy[0, i] < xmin) or (newxxyy[0, i] > xmax)) or (
                    (newxxyy[1, i] < ymin) or (newxxyy[1, i] > ymax))):
                fitpars[9 + 2 * i] = 0
                fiterrs[9 + 2 * i] = 0
        ### Now get the map again with updated parameters
        themap, newxxyy = model(x, fitpars, return_peaks=True)
    elif model.name == 'SbModelIndepPeaks':
        xmin = np.min(az2d)
        xmax = np.max(az2d)
        ymin = np.min(el2d)
        ymax = np.max(el2d)
        for i in range(model.npeaks):
            if (((newxxyy[0, i] < xmin) or (newxxyy[0, i] > xmax)) or (
                    (newxxyy[1, i] < ymin) or (newxxyy[1, i] > ymax))):
                fitpars[9 + 4 * i] = 0
                fiterrs[9 + 4 * i] = 0

    if verbose:
        print('===========================================================')
        print('Fitted values:')
        print('-----------------------------------------------------------')
        for i in range(len(parsinit)):
            print('{0:<20}: {1:>12.6f} +/- {2:>12.6f}'.format(model.parnames[i], fitpars[i], fiterrs[i]))
        print('-----------------------------------------------------------')
        print('Residuals**2/pix : {0:^8.5g}'.format(np.sum((flatmap - themap) ** 2) / np.size(flatmap)))
        print('===========================================================')

    if doplot:
        rc('figure', figsize=(18, 4))
        sh = np.shape(newxxyy)
        mm, ss = ft.meancut(flatmap, 3)

        subplot(1, 4, 1)
        imshow(themap, extent=[np.min(az) * np.cos(np.radians(50)),
                               np.max(az) * np.cos(np.radians(50)),
                               np.min(el), np.max(el)],
               vmin=mm - ss, vmax=mm + 10 * ss)
        colorbar()
        title('Fitted Map ' + extra_title)
        xlabel('Angle in Az direction [deg.]')
        ylabel('Elevation [deg.]')

        mm, ss = ft.meancut(flatmap - themap, 3)
        subplot(1, 4, 2)
        imshow(flatmap, extent=[np.min(az) * np.cos(np.radians(50)),
                                np.max(az) * np.cos(np.radians(50)),
                                np.min(el), np.max(el)],
               vmin=mm - nsiglo * ss, vmax=mm + nsighi * ss)
        xlim(np.min(az) * np.cos(np.radians(50)), np.max(az) * np.cos(np.radians(50)))
        ylim(np.min(el), np.max(el))
        colorbar()

        for i in range(sh[1]):
            ax = plot(newxxyy[0, i], newxxyy[1, i], 'r.')
        title('Input Map ' + extra_title)
        xlabel('Angle in Az direction [deg.]')
        ylabel('Elevation [deg.]')

        subplot(1, 4, 3)
        imshow(themap, extent=[np.min(az) * np.cos(np.radians(50)),
                               np.max(az) * np.cos(np.radians(50)),
                               np.min(el), np.max(el)],
               vmin=mm - nsiglo * ss, vmax=mm + nsighi * ss)
        colorbar()
        title('Fitted Map ' + extra_title)
        xlabel('Angle in Az direction [deg.]')
        ylabel('Elevation [deg.]')

        subplot(1, 4, 4)
        imshow(flatmap - themap, extent=[np.min(az) * np.cos(np.radians(50)),
                                         np.max(az) * np.cos(np.radians(50)),
                                         np.min(el), np.max(el)],
               vmin=mm - nsiglo * ss, vmax=mm + nsighi * ss)
        colorbar()
        title('Residual Map ' + extra_title)
        xlabel('Angle in Az direction [deg.]')
        ylabel('Elevation [deg.]')
        tight_layout()

        if figsave:
            savefig(figsave)
        show()

    if return_fitted:
        return fit, newxxyy, themap
    else:
        return fit, newxxyy
