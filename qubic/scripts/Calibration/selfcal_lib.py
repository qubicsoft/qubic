from __future__ import division, print_function

import glob

import cv2
import numpy as np
import healpy as hp
import pandas as pd
import qubic
import matplotlib.pyplot as plt
from astropy.io import fits

from qubicpack.utilities import Qubic_DataDir
from qubicpack.pixel_translation import make_id_focalplane, tes2index

__all__ = ['SelfCalibration']


class SelfCalibration:
    """
    Get power on the focal plane with or without optical aberrations
    and on the sky for a given horn configuration.

    """

    def __init__(self, baseline, dead_switches, d):
        """

        Parameters
        ----------
        baseline : list
            Baseline formed with 2 horns, index between 1 and 64 as on the instrument.
        dead_switches : int or list of int
            Broken switches, always closed.
        d : dictionary
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
        for i in self.dead_switches:
            if i < 1 or i > 64:
                raise ValueError('Horns indices must be in [1, 64].')

    def get_dead_detectors_mask(self, quadrant=3):
        """
        Build masks for the FP where bad detectors are NAN and good detectors are 1., one of shape (34x34)
        and one of shape (17x17) for one quadrant.

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
        quad = np.reshape(FPidentity.quadrant, (34, 34))

        calfile_path = Qubic_DataDir(datafile=self.d['detarray'])
        calfile = fits.open(calfile_path + '/' + self.d['detarray'])

        if self.d['detarray'] == 'CalQubic_DetArray_P87_TD.fits':
            full_mask = calfile['removed'].data
            full_mask = np.where(full_mask == 1, np.nan, full_mask)
            full_mask = np.where(full_mask == 0, 1, full_mask)

            quart = full_mask[np.where(quad != quadrant, 6, full_mask) != 6]
            quart_mask = np.reshape(quart, (17, 17))

            return full_mask, quart_mask

        else:
            print('There is no dead detectors in this calfile')

    def get_power_combinations(self, q, theta=np.array([0.]), phi=np.array([0.]),
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
            q.horn.open[self.dead_switches] = False
        S = get_power_on_array(q, theta, phi, spectral_irradiance, reso, xmin, xmax)
        if doplot:
            plt.figure()
            plt.subplot(4, 4, 1)
            q.horn.plot()
            plt.axis('off')
            plt.subplot(4, 4, 2)
            plt.imshow(S[:, :, 0])
            plt.colorbar()
            plt.title('$S$')

        # All open except i
        q.horn.open = True
        if self.dead_switches is not None:
            q.horn.open[self.dead_switches] = False
        q.horn.open[self.baseline[0] - 1] = False
        Cminus_i = get_power_on_array(q, theta, phi, spectral_irradiance, reso, xmin, xmax)
        if doplot:
            plt.subplot(4, 4, 3)
            q.horn.plot()
            plt.axis('off')
            plt.subplot(4, 4, 4)
            plt.imshow(Cminus_i[:, :, 0])
            plt.colorbar()
            plt.title('$C_{-i}$')

        # All open except j
        q.horn.open = True
        if self.dead_switches is not None:
            q.horn.open[self.dead_switches] = False
        q.horn.open[self.baseline[1] - 1] = False
        Cminus_j = get_power_on_array(q, theta, phi, spectral_irradiance, reso, xmin, xmax)
        if doplot:
            plt.subplot(4, 4, 5)
            q.horn.plot()
            plt.axis('off')
            plt.subplot(4, 4, 6)
            plt.imshow(Cminus_j[:, :, 0])
            plt.colorbar()
            plt.title('$C_{-j}$')

        # All open except baseline [i, j]
        q.horn.open = True
        if self.dead_switches is not None:
            q.horn.open[self.dead_switches] = False
        q.horn.open[self.baseline[0] - 1] = False
        q.horn.open[self.baseline[1] - 1] = False
        Sminus_ij = get_power_on_array(q, theta, phi, spectral_irradiance, reso, xmin, xmax)
        if doplot:
            plt.subplot(4, 4, 7)
            q.horn.plot()
            plt.axis('off')
            plt.subplot(4, 4, 8)
            plt.imshow(Sminus_ij[:, :, 0])
            plt.colorbar()
            plt.title('$S_{-ij}$')

        # Only i open (not a realistic observable)
        q.horn.open = False
        if self.dead_switches is not None:
            q.horn.open[self.dead_switches] = False
        q.horn.open[self.baseline[0] - 1] = True
        Ci = get_power_on_array(q, theta, phi, spectral_irradiance, reso, xmin, xmax)
        if doplot:
            plt.subplot(4, 4, 9)
            q.horn.plot()
            plt.axis('off')
            plt.subplot(4, 4, 10)
            plt.imshow(Ci[:, :, 0])
            plt.colorbar()
            plt.title('$C_i$')

        # Only j open (not a realistic observable)
        q.horn.open = False
        q.horn.open[self.baseline[1] - 1] = True
        Cj = get_power_on_array(q, theta, phi, spectral_irradiance, reso, xmin, xmax)
        if doplot:
            plt.subplot(4, 4, 11)
            q.horn.plot()
            plt.axis('off')
            plt.subplot(4, 4, 12)
            plt.imshow(Cj[:, :, 0])
            plt.colorbar()
            plt.title('$C_j$')

        # Only baseline [i, j] open (not a realistic observable)
        q.horn.open = False
        q.horn.open[self.baseline[0] - 1] = True
        q.horn.open[self.baseline[1] - 1] = True
        Sij = get_power_on_array(q, theta, phi, spectral_irradiance, reso, xmin, xmax)
        if doplot:
            plt.subplot(4, 4, 13)
            q.horn.plot()
            plt.axis('off')
            plt.subplot(4, 4, 14)
            plt.imshow(Sij[:, :, 0])
            plt.colorbar()
            plt.title('$S_{ij}$')

        return S, Cminus_i, Cminus_j, Sminus_ij, Ci, Cj, Sij

    def compute_fringes(self, theta=np.array([0.]), phi=np.array([0.]), reso=34, spectral_irradiance=1.):
        """
        Return the fringes on the FP by making the computation
        fringes =(S_tot - Cminus_i - Cminus_j + Sminus_ij) / Ci
        """
        q = qubic.QubicMultibandInstrument(self.d)

        S_tot, Cminus_i, Cminus_j, Sminus_ij, Ci, Cj, Sij = \
            SelfCalibration.get_power_combinations(self, q[0], theta=theta, phi=phi, reso=reso,
                                                   spectral_irradiance=spectral_irradiance,
                                                   doplot=False)
        fringes = (S_tot - Cminus_i - Cminus_j + Sminus_ij) / Ci

        return fringes

    def get_power_fp_aberration(self, rep, doplot=True, theta_source=0., freq_source=150., indep_config=None):
        """
        Compute power in the focal plane for a given horn configuration taking
        into account optical aberrations given in Creidhe simulations.

        Parameters
        ----------
        rep : str
            Path of the repository for the simulated files, can be download at :
            https://drive.google.com/open?id=19dPHw_CeuFZ068b-VRT7N-LWzOL1fmfG
        doplot : bool
            If True, make a plot with the intensity in the focal plane.
        theta_source : float
            Angle in degree between the optical axis of Qubic and the source.
        freq_source : float
            Frequency of the source in GHz
        indep_config : list of int
            By default it is None and in this case, it will use the baseline
            defined in your object on which you call the method.
            If you want an other configuration (all open for example), you can
            put here a list with the horns you want to open.

        Returns
        -------
        power : array of shape (nn, nn)
            Power in the focal plane at high resolution (sampling used in simulations).

        """
        if self.d['config'] != 'TD':
            raise ValueError('The instrument in the dictionary must be the TD')

        q = qubic.QubicInstrument(self.d)

        # Get simulation files
        files = sorted(glob.glob(rep + '/*.dat'))

        nhorns = len(files)
        if nhorns != 64:
            raise ValueError('You should have 64 .dat files')

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

            # Phase calculation
            horn_x = q.horn.center[swi - 1, 0]
            horn_y = q.horn.center[swi - 1, 1]
            d = np.sqrt(horn_x ** 2 + horn_y ** 2)  # distance between the horn and the center
            phi = - 2 * np.pi / 3e8 * freq_source * 1e9 * d * np.sin(np.deg2rad(theta_source))

            data = pd.read_csv(files[swi - 1], sep='\t', skiprows=0)
            allampX[i, :, :] = np.reshape(np.asarray(data['MagX']), (nn, nn))
            allampY[i, :, :] = np.reshape(np.asarray(data['MagY']), (nn, nn))

            allphiX[i, :, :] = np.reshape(np.asarray(data['PhaseX']), (nn, nn)) + phi
            allphiY[i, :, :] = np.reshape(np.asarray(data['PhaseY']), (nn, nn)) + phi

        # Electric field for each open horn
        Ax = allampX * (np.cos(allphiX) + 1j * np.sin(allphiX))
        Ay = allampY * (np.cos(allphiY) + 1j * np.sin(allphiY))

        # Sum of the electric fields
        sumampx = np.sum(Ax, axis=0)
        sumampy = np.sum(Ay, axis=0)

        # Intensity in the focal plane with high resolution
        # and with the focal plane resolution
        power = np.abs(sumampx) ** 2 + np.abs(sumampy) ** 2

        if doplot:
            plt.figure()
            plt.subplot(121)
            q.horn.plot()
            plt.axis('off')

            plt.subplot(122)
            plt.imshow(power)
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
                                                          doplot=False,
                                                          indep_config=[i])

        fringes_aber = (S_tot_aber - Cminus_i_aber - Cminus_j_aber + Sminus_ij_aber) / Ci_aber

        return fringes_aber

    def get_synthetic_beam_sky(self, q, scene, tes, default_open=True, with_baseline=True):
        """
        Return the synthetic beam projected on the sky for a given TES.
        Plot the horn matrix and the synthetic beam.

        Parameters
        ----------
        q : Qubic monochromatic instrument
        scene : a Qubic scene
        tes : int
            TES number for which you reconstruct the synthetic beam.
        default_open : bool
            If True, all switches are open except the ones in baseline.
            If False, all switches are close except the one in baseline.
            True by default.
        with_baseline : bool
            If true, the baseline is closed. If false, it is not close and
            you can have the full synthetic beam on the sky.

        Returns
        -------
        The synthetic beam on the sky.

        """

        if default_open:
            q.horn.open = True
            if with_baseline:
                for i in self.baseline:
                    q.horn.open[i - 1] = False
            for i in self.dead_switches:
                q.horn.open[i - 1] = False
        else:
            q.horn.open = False
            for i in self.baseline:
                q.horn.open[i - 1] = True
        sb = q.get_synthbeam(scene, idet=tes)

        plt.subplot(121)
        q.horn.plot()
        plt.axis('off')
        hp.gnomview(sb, sub=122, rot=(0, 90), reso=5, xsize=350, ysize=350,
                    title='Synthetic beam on the sky for TES {}'.format(tes),
                    cbar=True, notext=True)
        return sb


def get_power_on_array(q, theta=np.array([0.]), phi=np.array([0.]), spectral_irradiance=1.,
                       reso=34, xmin=-0.06, xmax=0.06):
    """
    Compute power on the focal plane for different positions of the source
    with respect to the instrument.

    Parameters
    ----------
    q : a qubic monochromatic instrument
    theta : array-like
        The source zenith angle [rad].
    phi : array-like
        The source azimuthal angle [rad].
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
    z1d = x1d * 0 - 0.3
    position = np.array([x1d, y1d, z1d]).T

    field = q._get_response(theta, phi, spectral_irradiance, position, q.detector.area,
                            q.filter.nu, q.horn, q.primary_beam, q.secondary_beam)
    power = np.reshape(np.abs(field) ** 2, (reso, reso, nptg))

    return power


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


def tes_signal2image_fp(tes_signal):
    """
    Inverse function of image_fp2tes_signal.
    """
    thermos = [4, 36, 68, 100]
    image_fp = np.empty((34, 34))
    image_fp[:] = np.nan
    for ASIC in range(8):
        for TES in range(128):
            if TES + 1 not in thermos:
                index = tes2index(TES + 1, ASIC + 1)
                image_fp[index // 34, index % 34] = tes_signal[TES, ASIC]
    return image_fp


def get_real_fp(full_fp, quadrant=None):
    """
    Return the real focal plane, one pixel for each TES.
    Parameters
    ----------
    full_fp : 2D array
        Image of the focal plane with a resolution.
    quadrant : int
        If you only want one quadrant of the focal plane,
        you can choose one in [1, 2, 3, 4]

    Returns
    -------
    full_real_fp : full fp (34x34)
    quart_fp : one quadrant (17x17)

    """
    # Decrease the resolution to the one of the FP
    if np.shape(full_fp) != (34, 34):
        full_real_fp = cv2.resize(full_fp, (34, 34))  # Not sure this function is the best
    else:
        full_real_fp = np.copy(full_fp)

    FPidentity = make_id_focalplane()
    tes = np.reshape(FPidentity.TES, (34, 34))
    quad = np.reshape(FPidentity.quadrant, (34, 34))

    # Put the pixels that are not TES to NAN
    full_real_fp = np.where(tes == 0, np.nan, full_real_fp)
    if quadrant is None:
        return full_real_fp

    else:
        if quadrant not in [1, 2, 3, 4]:
            raise ValueError('quadrant must be 1, 2, 3 or 4')
        else:
            # Get only one quadrant
            quart = full_real_fp[np.where(quad != quadrant, 6, full_real_fp) != 6]
            quart_fp = np.reshape(quart, (17, 17))

            return full_real_fp, quart_fp
