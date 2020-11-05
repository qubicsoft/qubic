from __future__ import division, print_function

import glob

import numpy as np
import healpy as hp
import pandas as pd
from scipy import ndimage

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.ticker as plticker
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patches as patches

from astropy.io import fits

import qubic
from qubicpack.utilities import Qubic_DataDir
from qubicpack.pixel_translation import make_id_focalplane, tes2index

__all__ = ['SelfCalibration']


class SelfCalibration:
    """
    Get power on the focal plane with or without optical aberrations
    and on the sky for a given horn configuration.

    """

    def __init__(self, baseline, d, dead_switches=None):
        """

        Parameters
        ----------
        baseline : list
            Baseline formed with 2 horns, index between 1 and 64 as on the instrument.
        d : dictionary
        dead_switches : list of int
            Broken switches, always closed between 1 and 64. By default is None. 
        """
        self.baseline = baseline
        self.dead_switches = dead_switches
        self.d = d
        # Replace CC by TD or FI
        d['detarray'] = d['detarray'].replace(d['detarray'][-7:-5], d['config'])

        if len(self.baseline) != 2:
            raise ValueError('The baseline should contain 2 horns.')
        for i in self.baseline:
            if i < 1 or i > 64:
                raise ValueError('Horns indices must be in [1, 64].')
        if self.dead_switches is not None:
            for i in self.dead_switches:
                if i < 1 or i > 64:
                    raise ValueError('Horns indices must be in [1, 64].')

    def get_dead_detectors_mask(self, quadrant=3):
        """
        Build masks for the FP where bad detectors are NAN and good detectors are 1., one of shape (34x34)
        and one of shape (17x17) for one quadrant.
        We use the ONAFP frame.

        Parameters
        ----------
        quadrant : int
            Quadrant of the focal plane in [1, 2, 3, 4]
            By default is 3 for the TD

        Returns
        -------
        full_mask : array of shape (34x34)
            mask for the full FP.
        quart_mask = array of shape (17x17)
            mask for one quadrant

        """
        FPidentity = make_id_focalplane()
        quad = np.rot90(np.reshape(FPidentity.quadrant, (34, 34)), k=-1, axes=(0, 1))

        calfile_path = Qubic_DataDir(datafile=self.d['detarray'])
        calfile = fits.open(calfile_path + '/' + self.d['detarray'])

        if self.d['detarray'] == 'CalQubic_DetArray_P87_TD.fits':
            full_mask = np.rot90(calfile['removed'].data, k=-1, axes=(0, 1))
            full_mask = np.where(full_mask == 1, np.nan, full_mask)
            full_mask = np.where(full_mask == 0, 1, full_mask)

            quart = full_mask[np.where(quad != quadrant, 6, full_mask) != 6]
            quart_mask = np.reshape(quart, (17, 17))

            return full_mask, quart_mask

        else:
            print('There is no dead detectors in this calfile')

    def get_power_combinations(self, q, theta=np.array([0.]), phi=np.array([0.]), nu=150e9,
                               spectral_irradiance=1.,
                               reso=34, xmin=-0.06, xmax=0.06, doplot=True):
        """
            Returns the power on the focal plane for each pointing, for different configurations
            of the horn array: all open, all open except i, except j, except i and j, only i open,
             only j open, only i and j open.
        Parameters
        ----------
        q : a qubic monochromatic instrument
        theta : array-like
            The source zenith angle [rad].
        phi : array-like
            The source azimuthal angle [rad].
        nu : float
            Source frequency in Hz.
        spectral_irradiance : array-like
            The source spectral_irradiance [W/m^2/Hz].
        reso : int
            Pixel number on one side on the focal plane image
        xmin : float
            Position of the border of the focal plane to the center [m]
        xmax : float
            Position of the opposite border of the focal plane to the center [m]
        doplot : bool
            If True, do the plots for the first pointing.

        Returns
        -------
        S, Cminus_i, Cminus_j, Sminus_ij, Ci, Cj, Sij : arrays of shape (reso, reso, #pointings)
            Power on the focal plane for each configuration, for each pointing.

        """

        # All open
        q.horn.open = True
        if self.dead_switches is not None:
            for i in self.dead_switches:
                q.horn.open[i - 1] = False
        S = get_power_on_array(q, theta, phi, nu, spectral_irradiance, reso, xmin, xmax)
        if doplot:
            plt.figure()
            plt.subplot(4, 4, 1)
            q.horn.plot()
            plt.axis('off')
            plt.subplot(4, 4, 2)
            plt.imshow(S[:, :, 0], origin='lower')
            plt.colorbar()
            plt.title('$S$')

        # All open except i
        q.horn.open = True
        if self.dead_switches is not None:
            for i in self.dead_switches:
                q.horn.open[i - 1] = False
        q.horn.open[self.baseline[0] - 1] = False
        Cminus_i = get_power_on_array(q, theta, phi, nu, spectral_irradiance, reso, xmin, xmax)
        if doplot:
            plt.subplot(4, 4, 3)
            q.horn.plot()
            plt.axis('off')
            plt.subplot(4, 4, 4)
            plt.imshow(Cminus_i[:, :, 0], origin='lower')
            plt.colorbar()
            plt.title('$C_{-i}$')

        # All open except j
        q.horn.open = True
        if self.dead_switches is not None:
            for i in self.dead_switches:
                q.horn.open[i - 1] = False
        q.horn.open[self.baseline[1] - 1] = False
        Cminus_j = get_power_on_array(q, theta, phi, nu, spectral_irradiance, reso, xmin, xmax)
        if doplot:
            plt.subplot(4, 4, 5)
            q.horn.plot()
            plt.axis('off')
            plt.subplot(4, 4, 6)
            plt.imshow(Cminus_j[:, :, 0], origin='lower')
            plt.colorbar()
            plt.title('$C_{-j}$')

        # All open except baseline [i, j]
        q.horn.open = True
        if self.dead_switches is not None:
            for i in self.dead_switches:
                q.horn.open[i - 1] = False
        q.horn.open[self.baseline[0] - 1] = False
        q.horn.open[self.baseline[1] - 1] = False
        Sminus_ij = get_power_on_array(q, theta, phi, nu, spectral_irradiance, reso, xmin, xmax)
        if doplot:
            plt.subplot(4, 4, 7)
            q.horn.plot()
            plt.axis('off')
            plt.subplot(4, 4, 8)
            plt.imshow(Sminus_ij[:, :, 0], origin='lower')
            plt.colorbar()
            plt.title('$S_{-ij}$')

        # Only i open (not a realistic observable)
        q.horn.open = False
        if self.dead_switches is not None:
            for i in self.dead_switches:
                q.horn.open[i - 1] = False
        q.horn.open[self.baseline[0] - 1] = True
        Ci = get_power_on_array(q, theta, phi, nu, spectral_irradiance, reso, xmin, xmax)
        if doplot:
            plt.subplot(4, 4, 9)
            q.horn.plot()
            plt.axis('off')
            plt.subplot(4, 4, 10)
            plt.imshow(Ci[:, :, 0], origin='lower')
            plt.colorbar()
            plt.title('$C_i$')

        # Only j open (not a realistic observable)
        q.horn.open = False
        q.horn.open[self.baseline[1] - 1] = True
        Cj = get_power_on_array(q, theta, phi, nu, spectral_irradiance, reso, xmin, xmax)
        if doplot:
            plt.subplot(4, 4, 11)
            q.horn.plot()
            plt.axis('off')
            plt.subplot(4, 4, 12)
            plt.imshow(Cj[:, :, 0], origin='lower')
            plt.colorbar()
            plt.title('$C_j$')

        # Only baseline [i, j] open (not a realistic observable)
        q.horn.open = False
        q.horn.open[self.baseline[0] - 1] = True
        q.horn.open[self.baseline[1] - 1] = True
        Sij = get_power_on_array(q, theta, phi, nu, spectral_irradiance, reso, xmin, xmax)
        if doplot:
            plt.subplot(4, 4, 13)
            q.horn.plot()
            plt.axis('off')
            plt.subplot(4, 4, 14)
            plt.imshow(Sij[:, :, 0], origin='lower')
            plt.colorbar()
            plt.title('$S_{ij}$')

        return S, Cminus_i, Cminus_j, Sminus_ij, Ci, Cj, Sij

    def compute_fringes(self, q, theta=np.array([0.]), phi=np.array([0.]), nu=150e9, spectral_irradiance=1., reso=34,
                        xmin=-0.06, xmax=0.06, doplot=False):
        """
        Return the fringes on the FP by making the computation
        fringes =(S_tot - Cminus_i - Cminus_j + Sminus_ij) / Ci
        q : a qubic monochromatic instrument
        """

        S_tot, Cminus_i, Cminus_j, Sminus_ij, Ci, Cj, Sij = \
            SelfCalibration.get_power_combinations(self, q, theta=theta, phi=phi, nu=nu,
                                                   spectral_irradiance=spectral_irradiance, reso=reso,
                                                   xmin=xmin, xmax=xmax, doplot=doplot)

        fringes = (S_tot - Cminus_i - Cminus_j + Sminus_ij) / Ci

        return fringes

    def get_power_fp_aberration(self, rep, doplot=True, indep_config=None):
        """
        Compute power on the focal plane for a given horn configuration taking
        into account optical aberrations given by Maynooth simulations. The source
        is on the optical axis emitting at 150GHz.

        Parameters
        ----------
        rep : str
            Path of the repository for the simulated files, can be download at :
            https://drive.google.com/open?id=19dPHw_CeuFZ068b-VRT7N-LWzOL1fmfG
        doplot : bool
            If True, make a plot with the intensity in the focal plane.
        indep_config : list of int
            By default it is None and in this case, it will use the baseline
            defined in your object on which you call the method.
            If you want an other configuration (all open for example), you can
            put here a list with the horns you want to open.

        Returns
        -------
        power : array of shape (nn, nn)
            Power on the focal plane at high resolution (sampling used in simulations).

        """
        if self.d['config'] != 'TD':
            raise ValueError('The instrument in the dictionary must be the TD')

        q = qubic.QubicInstrument(self.d)

        # Get simulation files
        files = sorted(glob.glob(rep + '/*.dat'))

        nhorns = len(files)
        if nhorns != 64:
            raise ValueError('You should have 64 .dat files')

        # This is done to get the right file for each horn
        horn_transpose = np.arange(64)
        horn_transpose = np.reshape(horn_transpose, (8, 8))
        horn_transpose = np.ravel(horn_transpose.T)

        # Get the sample number from the first file
        data0 = pd.read_csv(files[0], sep='\t', skiprows=0)
        nn = data0['X_Index'].iloc[-1] + 1
        print('Sampling number = {}'.format(nn))

        # Get all amplitudes and phases for each open horn
        if indep_config is None:
            open_horns = self.baseline
            nopen_horns = len(self.baseline)
        else:
            open_horns = indep_config
            nopen_horns = len(indep_config)

        q.horn.open = False
        q.horn.open[np.asarray(open_horns) - 1] = True

        allampX = np.empty((nopen_horns, nn, nn))
        allphiX = np.empty((nopen_horns, nn, nn))
        allampY = np.empty((nopen_horns, nn, nn))
        allphiY = np.empty((nopen_horns, nn, nn))
        for i, swi in enumerate(open_horns):
            if swi < 1 or swi > 64:
                raise ValueError('The switch indices must be between 1 and 64 ')

            thefile = files[horn_transpose[swi - 1]]
            # print('Horn ', swi, ': ', thefile[98:104])
            data = pd.read_csv(thefile, sep='\t', skiprows=0)
            
            allampX[i, :, :] = np.reshape(np.asarray(data['MagX']), (nn, nn)).T
            allampY[i, :, :] = np.reshape(np.asarray(data['MagY']), (nn, nn)).T

            allphiX[i, :, :] = np.reshape(np.asarray(data['PhaseX']), (nn, nn)).T
            allphiY[i, :, :] = np.reshape(np.asarray(data['PhaseY']), (nn, nn)).T

        # Electric field for each open horn
        Ax = allampX * (np.cos(allphiX) + 1j * np.sin(allphiX))
        Ay = allampY * (np.cos(allphiY) + 1j * np.sin(allphiY))

        # Sum of the electric fields
        sumampx = np.sum(Ax, axis=0)
        sumampy = np.sum(Ay, axis=0)

        # Power on the focal plane
        power = np.abs(sumampx) ** 2 + np.abs(sumampy) ** 2

        if doplot:
            plt.figure()
            plt.subplot(121)
            q.horn.plot()
            plt.axis('off')

            plt.subplot(122)
            plt.imshow(power, origin='lower')
            plt.title('Power at the sampling resolution')
            plt.colorbar()

        return power


    def get_fringes_aberration_combination(self, rep):
        """
        Return the fringes on the FP (power) with aberrations using Creidhe files
        by doing the computation :
        fringes = (S_tot - Cminus_i - Cminus_j + Sminus_ij) / Ci

        Parameters
        ----------
        rep : str
            Path of the repository for the simulated files, can be download at :
            https://drive.google.com/open?id=19dPHw_CeuFZ068b-VRT7N-LWzOL1fmfG

        Returns
        -------
        fringes_aber : array of shape (nn, nn)
            Fringes in the focal plane at high resolution (sampling used in simulations).

        """
        i = self.baseline[0]
        j = self.baseline[1]
        all_open = np.arange(1, 65)

        S_tot_aber = SelfCalibration.get_power_fp_aberration(self, rep,
                                                             doplot=False,
                                                             indep_config=all_open)
        Cminus_i_aber = SelfCalibration.get_power_fp_aberration(self, rep,
                                                                doplot=False,
                                                                indep_config=np.delete(all_open, i - 1))
        Cminus_j_aber = SelfCalibration.get_power_fp_aberration(self, rep,
                                                                doplot=False,
                                                                indep_config=np.delete(all_open, j - 1))
        Sminus_ij_aber = SelfCalibration.get_power_fp_aberration(self, rep,
                                                                 doplot=False,
                                                                 indep_config=np.delete(all_open, [i - 1, j - 1]))
        Ci_aber = SelfCalibration.get_power_fp_aberration(self, rep,
                                                          doplot=True,
                                                          indep_config=[i])

        fringes_aber = (S_tot_aber - Cminus_i_aber - Cminus_j_aber + Sminus_ij_aber) / Ci_aber

        return fringes_aber


def make_external_A(rep, open_horns):
    """
    Compute external_A from simulated files with aberrations.
    This can be used in get_synthbeam method that returns the synthetic beam on the sky.
    Parameters
    ----------
    rep : str
        Path of the repository for the simulated files, can be download at :
        https://drive.google.com/open?id=19dPHw_CeuFZ068b-VRT7N-LWzOL1fmfG
    open_horns : list
        Indices of the open horns between 0 and 63.

    Returns
    -------
    external_A : list of tables describing the phase and amplitude at each point of the focal
        plane for each of the horns:
        [0] : array of nn with x values in meters
        [1] : array of nn with y values in meters
        [2] : array of [nhorns, nn, nn] with amplitude on X
        [3] : array of [nhorns, nn, nn] with amplitude on Y
        [4] : array of [nhorns, nn, nn] with phase on X in degrees
        [5] : array of [nhorns, nn, nn] with phase on Y in degrees

    """
    # Get simulation files
    files = sorted(glob.glob(rep + '/*.dat'))

    nhorns = len(files)
    if nhorns != 64:
        raise ValueError('You should have 64 .dat files')

    # Get the sample number from the first file
    data0 = pd.read_csv(files[0], sep='\t', skiprows=0)
    nn = data0['X_Index'].iloc[-1] + 1
    print('Sampling number = ', nn)

    xmin = data0['X'].iloc[0] * 1e-3
    xmax = data0['X'].iloc[-1] * 1e-3
    ymin = data0['Y'].iloc[0] * 1e-3
    ymax = data0['Y'].iloc[-1] * 1e-3
    print('xmin={}m, xmax={}m, ymin={}m, ymax={}m'.format(xmin, xmax, ymin, ymax))

    xx = np.linspace(xmin, xmax, nn)
    yy = np.linspace(ymin, ymax, nn)

    # Get all amplitudes and phases for each open horn
    nopen_horns = len(open_horns)

    allampX = np.empty((nopen_horns, nn, nn))
    allphiX = np.empty((nopen_horns, nn, nn))
    allampY = np.empty((nopen_horns, nn, nn))
    allphiY = np.empty((nopen_horns, nn, nn))
    for i, horn in enumerate(open_horns):
        print('horn ', horn)
        if horn < 0 or horn > 63:
            raise ValueError('The horn indices must be between 0 and 63 ')

        data = pd.read_csv(files[horn], sep='\t', skiprows=0)
        allampX[i, :, :] = np.reshape(np.asarray(data['MagX']), (nn, nn))
        allampY[i, :, :] = np.reshape(np.asarray(data['MagY']), (nn, nn))

        allphiX[i, :, :] = np.reshape(np.asarray(data['PhaseX']), (nn, nn))
        allphiY[i, :, :] = np.reshape(np.asarray(data['PhaseY']), (nn, nn))

    external_A = [-xx, -yy, allampX, allampY, allphiX, allphiY]

    return external_A


def get_power_on_array(q, theta=np.array([0.]), phi=np.array([0.]), nu=150e9, spectral_irradiance=1.,
                       reso=34, xmin=-0.06, xmax=0.06):
    """
    Compute power on the focal plane in the ONAFP frame for different positions of the source
    with respect to the instrument.

    Parameters
    ----------
    q : a qubic monochromatic instrument
    theta : array-like
        The source zenith angle [rad].
    phi : array-like
        The source azimuthal angle [rad].
    nu : float
        Source frequency in Hz.
    spectral_irradiance : array-like
        The source spectral_irradiance [W/m^2/Hz].
    reso : int
        Pixel number on one side on the focal plane image
    xmin : float
        Position of the border of the focal plane to the center [m]
    xmax : float
        Position of the opposite border of the focal plane to the center [m]

    Returns
    ----------
    power : array of shape (reso, reso, #pointings)
        The power on the focal plane for each pointing.
    """
    nptg = len(theta)
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, reso), np.linspace(xmin, xmax, reso))
    x1d = np.ravel(xx)
    y1d = np.ravel(yy)
    z1d = x1d * 0 - q.optics.focal_length
    position = np.array([x1d, y1d, z1d]).T

    # Electric field on the FP in the GRF frame
    field = q._get_response(theta, phi, spectral_irradiance, position, q.detector.area,
                            nu, q.horn, q.primary_beam, q.secondary_beam)
    power_GRF = np.reshape(np.abs(field) ** 2, (reso, reso, nptg))

    # Go to the ONAFP frame
    power_ONAFP = np.rot90(power_GRF, k=-1, axes=(0, 1))

    return power_ONAFP


def index2TESandASIC(index):
    """
    Convert an index on the FP to the corresponding TES and ASICS.
    Parameters
    ----------
    index : int
        index on the FP between 0 and 1155.

    Returns
    -------
    TES: int between 1 and 128 if the given index corresponds to a TES,
        0 if not.
    ASIC: int between 1 and 8 if the given index corresponds to a TES,
        0 if not.

    """
    if index < 0 or index > 1155:
        raise ValueError('index must be between 0 and 1155')
    else:
        FPidentity = make_id_focalplane()
        TES = FPidentity[index].TES
        ASIC = FPidentity[index].ASIC

    return TES, ASIC


def image_fp2tes_signal(full_real_fp):
    """
    Convert an image of the FP to an array with the signal
    of each TES using the TES indices of the real FP.
    Make sure to use the ONAFP frame.
    Parameters
    ----------
    full_real_fp : array of shape (34, 34)
        Image on the full FP.

    Returns
    -------
    tes_signal : array of shape (128, 8)
        Signal on each TES, for each ASIC.

    """
    if np.shape(full_real_fp) != (34, 34):
        raise ValueError('The focal plane image should have for shape (34, 34).')

    else:
        tes_signal = np.empty((128, 8))
        index = 0
        for i in range(34):
            for j in range(34):
                TES, ASIC = index2TESandASIC(index)
                if TES != 0:
                    tes_signal[TES - 1, ASIC - 1] = full_real_fp[i, j]
                index += 1
        return tes_signal


def tes_signal2image_fp(tes_signal, asics):
    """
    tes_signal : array of shape (128, #ASICS)
        Signal on each TES, for each ASIC.
    asics : list
        Indices of the asics used between 1 and 8.
    """
    thermos = [4, 36, 68, 100]
    image_fp = np.empty((34, 34))
    image_fp[:] = np.nan
    FPidentity = make_id_focalplane()
    for ASIC in asics:
        for TES in range(128):
            if TES + 1 not in thermos:
                index = tes2index(TES + 1, ASIC)
                image_fp[index // 34, index % 34] = tes_signal[TES, ASIC - 1]
    return image_fp


def make_labels(rep, nn=241, ndet=992, img_size=0.12, doplot=True):
    """
    Get averaged signal in each TES (real FP). Based on Creidhe code.

    Parameters
    ----------
    rep : str
        Path of the repository for the simulated files, can be download at :
        https://drive.google.com/open?id=19dPHw_CeuFZ068b-VRT7N-LWzOL1fmfG
    nn : int
        Resolution of the image.
    ndet : int
        Number of TES.
    img_size : float
        Dimension in m of one side of the image.
    doplot : bool

    Returns
    -------
    readv : array of shape(992, 4, 2)
        Corners positions.
    labels : array_like of ints
            Assign labels to the values of the img. Has to have the same shape as img.
    """

    # Get TES positions of the 4 corners
    vertices = pd.read_csv(rep + '/vertices.txt', sep='\ ', header=None, engine='python')
    readv = np.zeros((ndet, 4, 2))
    for i in range(4):
        readv[:, i, :] = vertices.iloc[i::4, :]

    labels = np.zeros((nn, nn))

    xx = np.linspace(- img_size / 2., img_size / 2., nn)
    yy = np.linspace(- img_size / 2., img_size / 2., nn)
    XX, YY = np.meshgrid(xx, yy)

    for i in range(nn):
        for j in range(nn):
            for d in range(0, ndet):
                if readv[d, 2, 0] <= XX[i, j] <= readv[d, 3, 0] and \
                        readv[d, 2, 1] <= YY[i, j] <= readv[d, 1, 1]:
                    labels[i, j] = d+1
    if doplot:
        plt.imshow(labels, origin='lower')
        plt.colorbar()

    return readv, labels


def fulldef2tespixels(img, labels, ndet=992):
    """
        Get signal in each TES (real FP).

        Parameters
        ----------
        img : array of shape (nn, nn)
            Signal on the focal plane at high resolution, can be larger than the real FP.
        labels : array_like of ints
            Assign labels to the values of the img. Has to have the same shape as img.
        ndet : int
            Number of TES.

        Returns
        -------
        3 lists with the number of points averaged in each TES, the sum of the signal and the mean.
        """
    unique, counts_perTES = np.unique(labels, return_counts=True)
    mean_perTES = ndimage.mean(img, labels, index=np.arange(1, ndet + 1))
    sum_perTES = ndimage.sum(img, labels, index=np.arange(1, ndet + 1))

    return counts_perTES, sum_perTES, mean_perTES


def make_plot_real_fp(readv, sig_perTES, vmin=0., vmax=1.):
    """
    Plot real FP using TES locations.
    """
    # All this to get colormap
    cm = plt.get_cmap('viridis')
    # plot scale from average
    cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    scalarMap.set_array(sig_perTES)

    fig, ax7 = plt.subplots()
    fig.figsize = (12, 12)
    for i in range(0, 992):
        rect = patches.Rectangle((readv[i, 2, 0], readv[i, 2, 1]),
                                 (readv[i, 0, 0] - readv[i, 1, 0]),
                                 (readv[i, 0, 1] - readv[i, 3, 1]),
                                 linewidth=1,
                                 edgecolor='none',
                                 facecolor=scalarMap.to_rgba(sig_perTES[i]))
        ax7.add_patch(rect)

    plt.xlim(-.055, .055)  # to see the focal plane
    plt.ylim(-.055, .055)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    ax7.set_aspect('equal')
    plt.colorbar(scalarMap, shrink=0.8)

    return fig


def get_quadrant3(q, signal_perTES, doplot=False):
    """

    Parameters
    ----------
    q : a qubic instrument
    signal_perTES : bytearray
        Signal in each TES (992 detectors).
    doplot : bool
        If True, make a plot.

    Returns
    -------
    An image (17x17) of quadrant 3.

    """
    quadrant3 = signal_perTES[496:744]
    indice = -(q.detector.center // 0.003)

    img = np.zeros((17, 17))
    for k in range(248):
        i = int(indice[k, 0])
        j = int(indice[k, 1])
        img[i - 1, j - 1] = quadrant3[k]
    img[img == 0.] = np.nan
    img = np.rot90(img)

    if doplot:
        plt.figure()
        plt.imshow(img)

    return img


def get_real_fp(full_fp, quadrant=None):
    """
    Return the real focal plane, one pixel for each TES.
    Parameters
    ----------
    full_fp : 2D array of shape (34, 34)
        Image of the focal plane.
    quadrant : int
        If you only want one quadrant of the focal plane,
        you can choose one in [1, 2, 3, 4]

    Returns
    -------
    full_real_fp : full fp (34x34)
    quart_fp : one quadrant (17x17)

    """
    if np.shape(full_fp) != (34, 34):
        raise ValueError('The focal plane shape should be (34, 34).')
    else:
        FPidentity = make_id_focalplane()
        tes = np.reshape(FPidentity.TES, (34, 34))
        # The rotation is needed to be in the ONAFP frame
        quad = np.rot90(np.reshape(FPidentity.quadrant, (34, 34)), k=-1, axes=(0, 1))

        # Put the pixels that are not TES to NAN
        full_real_fp = np.where(tes == 0, np.nan, full_fp)
        if quadrant is None:
            return full_real_fp

        else:
            if quadrant not in [1, 2, 3, 4]:
                raise ValueError('quadrant must be 1, 2, 3 or 4')
            else:
                # Get only one quadrant
                quart = full_real_fp[np.where(quad != quadrant+1, 6, full_real_fp) != 6]
                quart_fp = np.reshape(quart, (17, 17))

                return full_real_fp, quart_fp


def add_fp_simu_aber(image_aber, vmin, vmax, alpha=0.3, diameter_simu=120):
    """
    Over plot the real FP on a simulation with aberrations.
    Should be improved because space between quadrants is not taken into account.
    Parameters
    ----------
    image_aber : 2D array
        Image larger than the real focal plane.
    vmin, vmax : float
        Color scale for imshow.
    alpha : float
        Transparency for the FP circle.
    diameter_simu : float
        Diameter of the simulation image in mm.

    Returns
    -------
    fig : the figure

    """
    nn = np.shape(image_aber)[0]  # Sampling used in the simu
    fp_radius = 51 * nn / diameter_simu  # Radius in pixels
    tes_size = 3 * nn / diameter_simu
    print('TES size in pixels :', tes_size)
    print('FP radius in pixels :', fp_radius)

    fig, ax = plt.subplots()
    ax.imshow(image_aber, origin='lower', vmin=vmin, vmax=vmax)

    # Add a circle of the FP size
    circ = Circle((nn / 2., nn / 2.), fp_radius, alpha=alpha, color='w')
    ax.add_patch(circ)

    # Add a grid where each square is a TES
    loc = plticker.MultipleLocator(base=tes_size)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    ax.grid(color='w', linestyle='-', linewidth=1)

    # Add 2 lines to see the quadrants
    x = range(nn)
    y = np.ones(nn) * nn / 2.
    ax.plot(x, y, '-', linewidth=3, color='w')
    ax.plot(y, x, '-', linewidth=3, color='w')

    return fig


def get_simulation(param, q, baseline, files, labels, nn=241, doplot=True, verbose=True):
    """
    Get a simulation from Maynooth on quadrant 3 to fit the fringe measurement.
    Parameters
    ----------
    param : list
        Calibration source theta angle and frequency. Parameter that you will fit.
    q : a qubic instrument
    baseline : list of int
        The 2 horns open, between 1 and 64.
    files : list
         the simulations
    labels : array_like of ints
        See function make_labels.
    nn : int
        Sampling resolution for the simulations.
    doplot : bool
    verbose : bool

    Returns
    -------
    img : quadrant 3 with the signal in each TES.

    """
    theta_source = param[0]
    freq_source = param[1]

    # This is done to get the right file for each horn
    horn_transpose = np.arange(64)
    horn_transpose = np.reshape(horn_transpose, (8, 8))
    horn_transpose = np.ravel(horn_transpose.T)

    allampX = np.empty((2, nn, nn))
    allphiX = np.empty((2, nn, nn))
    allampY = np.empty((2, nn, nn))
    allphiY = np.empty((2, nn, nn))
    for i, swi in enumerate(baseline):
        # Phase calculation
        horn_x = q.horn.center[swi - 1, 0]
        horn_y = q.horn.center[swi - 1, 1]
        dist = np.sqrt(horn_x ** 2 + horn_y ** 2)  # distance between the horn and the center
        phi = - 2 * np.pi / 3e8 * freq_source * 1e9 * dist * np.sin(np.deg2rad(theta_source))

        thefile = files[horn_transpose[swi - 1]]
        if verbose:
            print('Horn ', swi, ': ', thefile[98:104])
        data = pd.read_csv(thefile, sep='\t', skiprows=0)

        allampX[i, :, :] = np.reshape(np.asarray(data['MagX']), (nn, nn)).T
        allampY[i, :, :] = np.reshape(np.asarray(data['MagY']), (nn, nn)).T

        allphiX[i, :, :] = np.reshape(np.asarray(data['PhaseX']), (nn, nn)).T + phi
        allphiY[i, :, :] = np.reshape(np.asarray(data['PhaseY']), (nn, nn)).T + phi

    # Electric field for each open horn
    Ax = allampX * (np.cos(allphiX) + 1j * np.sin(allphiX))
    Ay = allampY * (np.cos(allphiY) + 1j * np.sin(allphiY))

    # Sum of the electric fields
    sumampx = np.sum(Ax, axis=0)
    sumampy = np.sum(Ay, axis=0)

    # Power on the focal plane
    power = np.abs(sumampx) ** 2 + np.abs(sumampy) ** 2

    if doplot:
        plt.figure()
        plt.subplot(121)
        q.horn.plot()
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(power, origin='lower')
        plt.title('Power at the sampling resolution')
        plt.colorbar()

    counts_perTES, sum_perTES, mean_perTES = fulldef2tespixels(power, labels)

    img = get_quadrant3(q, mean_perTES, doplot=doplot)

    return img

def plot_horns(q):
    hcenters = q.horn.center[:,0:2]
    plt.plot(hcenters[:,0], hcenters[:,1], 'ro')
    
def plot_baseline(q,bs):
    hcenters = q.horn.center[:,0:2]
    plt.plot(hcenters[np.array(bs)-1,0], hcenters[np.array(bs)-1,1], lw=4, label=bs)

def give_bs_pars(q,bs):
    hc = q.horn.center[:,0:2]
    hc0 = hc[np.array(bs[0])-1,:]
    hc1 = hc[np.array(bs[1])-1,:]
    bsxy = hc1-hc0
    theta = np.degrees(np.arctan2(bsxy[1], bsxy[0]))
    length = np.sqrt(np.sum(bsxy**2))
    return theta, length



def check_equiv(vecbs1, vecbs2, tol=1e-5):
    norm1 = np.dot(vecbs1, vecbs1)
    norm2 = np.dot(vecbs2, vecbs2)
    cross12 = np.cross(vecbs1, vecbs2)
    if (np.abs(norm1-norm2) < tol) & (np.abs(cross12) < tol):
        return True
    else:
        return False
    
def find_equivalent_baselines(all_bs, q):
    ### Convert to array
    all_bs = np.array(all_bs)
    ### centers
    hcenters = q.horn.center[:,0:2]
    ### Baselines vectors
    all_vecs = np.zeros((len(all_bs), 2))
    for ib in range(len(all_bs)):
        coordsA = hcenters[all_bs[ib][0],:]
        coordsB = hcenters[all_bs[ib][1],:]
        all_vecs[ib,:] = coordsB-coordsA

    ### List of types of equivalence for each baseline: initially = -1
    all_eqtype = np.zeros(len(all_bs), dtype=int)-1

    ### First type is zero and is associated to first baseline
    eqnum = 0
    all_eqtype[0] = eqnum
    
    ### Indices of baselines
    index_bs = np.arange(len(all_bs))
    
    ### Loop over baselines
    for ib in range(0, len(all_bs)):
        ### Identify those that have no type
        wnotype = all_eqtype==-1
        bsnotype = all_bs[wnotype]
        vecsnotype = all_vecs[wnotype,:]
        indexnotype = index_bs[wnotype]
        ### Loop over those with no type
        for jb in range(len(bsnotype)):
            ### Check if equivalent
            status = check_equiv(all_vecs[ib,:], vecsnotype[jb,:])
            ### If so: give it the current type
            if status:
                all_eqtype[indexnotype[jb]] = eqnum
        ### We have gone through all possibilities for this type so we increment the type by 1
        eqnum = np.max(all_eqtype)+1
    
    alltypes = np.unique(all_eqtype)
    bseq = []
    for i in range(len(alltypes)):
        bseq.append(index_bs[all_eqtype==i])

    return bseq, all_eqtype



