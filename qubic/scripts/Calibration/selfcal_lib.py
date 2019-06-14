import glob
import healpy as hp
from matplotlib.pyplot import *
import pandas as pd
import cv2

from qubicpack import qubicpack as qp
from qubicpack import pix2tes
from qubicpack.utilities import ASIC_index

import qubic


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

        if len(self.baseline) != 2:
            raise ValueError('The baseline should contain 2 horns.')
        for i in self.baseline:
            if i<1 or i>64:
                raise ValueError('Horns indices must be in [1, 64].')
        for i in self.dead_switches:
            if i < 1 or i > 64:
                raise ValueError('Horns indices must be in [1, 64].')

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
        S = SelfCalibration.get_power_on_array(q, theta, phi, spectral_irradiance, reso, xmin, xmax)
        if doplot:
            figure()
            subplot(4, 4, 1)
            q.horn.plot()
            axis('off')
            subplot(4, 4, 2)
            imshow(S[:, :, 0])
            colorbar()
            title('$S$')

        # All open except i
        q.horn.open = True
        if self.dead_switches is not None:
            q.horn.open[self.dead_switches] = False
        q.horn.open[self.baseline[0] - 1] = False
        Cminus_i = SelfCalibration.get_power_on_array(q, theta, phi, spectral_irradiance, reso, xmin, xmax)
        if doplot:
            subplot(4, 4, 3)
            q.horn.plot()
            axis('off')
            subplot(4, 4, 4)
            imshow(Cminus_i[:, :, 0])
            colorbar()
            title('$C_{-i}$')

        # All open except j
        q.horn.open = True
        if self.dead_switches is not None:
            q.horn.open[self.dead_switches] = False
        q.horn.open[self.baseline[1] - 1] = False
        Cminus_j = SelfCalibration.get_power_on_array(q, theta, phi, spectral_irradiance, reso, xmin, xmax)
        if doplot:
            subplot(4, 4, 5)
            q.horn.plot()
            axis('off')
            subplot(4, 4, 6)
            imshow(Cminus_j[:, :, 0])
            colorbar()
            title('$C_{-j}$')

        # All open except baseline [i, j]
        q.horn.open = True
        if self.dead_switches is not None:
            q.horn.open[self.dead_switches] = False
        q.horn.open[self.baseline[0] - 1] = False
        q.horn.open[self.baseline[1] - 1] = False
        Sminus_ij = SelfCalibration.get_power_on_array(q, theta, phi, spectral_irradiance, reso, xmin, xmax)
        if doplot:
            subplot(4, 4, 7)
            q.horn.plot()
            axis('off')
            subplot(4, 4, 8)
            imshow(Sminus_ij[:, :, 0])
            colorbar()
            title('$S_{-ij}$')

        # Only i open (not a realistic observable)
        q.horn.open = False
        if self.dead_switches is not None:
            q.horn.open[self.dead_switches] = False
        q.horn.open[self.baseline[0] - 1] = True
        Ci = SelfCalibration.get_power_on_array(q, theta, phi, spectral_irradiance, reso, xmin, xmax)
        if doplot:
            subplot(4, 4, 9)
            q.horn.plot()
            axis('off')
            subplot(4, 4, 10)
            imshow(Ci[:, :, 0])
            colorbar()
            title('$C_i$')

        # Only j open (not a realistic observable)
        q.horn.open = False
        q.horn.open[self.baseline[1] - 1] = True
        Cj = SelfCalibration.get_power_on_array(q, theta, phi, spectral_irradiance, reso, xmin, xmax)
        if doplot:
            subplot(4, 4, 11)
            q.horn.plot()
            axis('off')
            subplot(4, 4, 12)
            imshow(Cj[:, :, 0])
            colorbar()
            title('$C_j$')

        # Only baseline [i, j] open (not a realistic observable)
        q.horn.open = False
        q.horn.open[self.baseline[0] - 1] = True
        q.horn.open[self.baseline[1] - 1] = True
        Sij = SelfCalibration.get_power_on_array(q, theta, phi, spectral_irradiance, reso, xmin, xmax)
        if doplot:
            subplot(4, 4, 13)
            q.horn.plot()
            axis('off')
            subplot(4, 4, 14)
            imshow(Sij[:, :, 0])
            colorbar()
            title('$S_{ij}$')

        return S, Cminus_i, Cminus_j, Sminus_ij, Ci, Cj, Sij

    def get_fringes_fp_TD(self, theta=np.array([0.]), phi=np.array([0.]), spectral_irradiance=1.):
        """
        Computes the fringe signals in each TES for point source.
        The sources moves and we compute the fringes for each pointing
        that corresponds to a position  of the source.

        Parameters
        ----------
        theta : array-like of shape (#pointings,)
            The source zenith angle [rad].
        phi : array-like of shape (#pointings,)
            The source azimuthal angle [rad].
        irradiance : array-like
            The source spectral_irradiance [W/m^2/Hz].

        Returns
        -------
        tes_fringes_signal : array of shape (256, #pointings)
            Fringe signal (power) in each TES.

        """
        q = qubic.QubicMultibandInstrument(self.d)

        S_tot, Cminus_i, Cminus_j, Sminus_ij, Ci, Cj, Sij = \
            SelfCalibration.get_power_combinations(self, q[0], theta=theta, phi=phi,
                                                   spectral_irradiance=spectral_irradiance,
                                                   doplot=False)
        nptg = np.shape(S_tot)[2]
        fringes = np.empty((17, 17, nptg))
        tes_fringes_signal = np.empty((256, nptg))
        for ptg in range(nptg):
            S_tot4 = SelfCalibration.full2quarter(S_tot[:, :, ptg])
            Cminus_i4 = SelfCalibration.full2quarter(Cminus_i[:, :, ptg])
            Cminus_j4 = SelfCalibration.full2quarter(Cminus_j[:, :, ptg])
            Sminus_ij4 = SelfCalibration.full2quarter(Sminus_ij[:, :, ptg])
            Ci4 = SelfCalibration.full2quarter(Ci[:, :, ptg])

            fringes[:, :, ptg] = (S_tot4 - Cminus_i4 - Cminus_j4 + Sminus_ij4) / Ci4

            tes_fringes_signal[:, ptg] = SelfCalibration.image_fp2tes_signal(fringes[:, :, ptg])

        return tes_fringes_signal

    def get_power_fp_aberration(self, rep, doplot=True, theta_source=0., freq_source=150., indep_config=None):
        """
        Compute power in the focal plane for a given horn configuration taking
        into account optical aberrations given in Creidhe simulations.

        Parameters
        ----------
        rep : str
            Path of the repository for the simulated files, can be download at :
            https://drive.google.com/drive/folders/1sC7-DrdsTigL0d7Z8KzPQ3uoWSy0Phxh?usp=sharing
        doplot : bool
            If True, make a plot with the intensity in the focal plane.
        theta_source : float
            Angle in degree between the optical axis of Qubic and the source.
        freq_source : float
            Frequency of the source in GHz
        indep_config : list of int
            By default it is None and in this case, it will use the baseline
            defined i your object on which you call the method.
            If you want an other configuration (all open for example), you can
            put here a list with the horns you want to open.

        Returns
        -------
        int_sampling_reso : array of shape (nn, nn)
            Power in the focal plane at high resolution (sampling used in simulations.
        int_fp_reso : array of shape (34, 34)
            Power in the focal plane at the TES resolution.

        """
        if self.d['config'] != 'TD':
            raise ValueError('The instrument in the dictionary must be the TD')

        q = qubic.QubicInstrument(self.d)

        # Get simulation files
        files = sorted(glob.glob(rep + '*.dat'))

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
            horn_x = q.horn.center[swi-1, 0]
            horn_y = q.horn.center[swi-1, 1]
            d = np.sqrt(horn_x**2 + horn_y**2) # distance between the horn and the center
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
        int_sampling_reso = np.abs(sumampx) ** 2 + np.abs(sumampy) ** 2
        int_fp_reso = cv2.resize(int_sampling_reso, (34, 34))

        if doplot:
            figure()
            subplot(131)
            q.horn.plot()
            axis('off')
            subplot(132)
            imshow(int_sampling_reso)
            title('Power at the sampling resolution')
            colorbar()

            subplot(133)
            imshow(int_fp_reso)
            title('Power at the TES resolution')
            colorbar()

        return int_sampling_reso, int_fp_reso

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

        subplot(121)
        q.horn.plot()
        axis('off')
        hp.gnomview(sb, sub=122, rot=(0, 90), reso=5, xsize=350, ysize=350,
                    title='Synthetic beam on the sky for TES {}'.format(tes),
                    cbar=True, notext=True)
        return sb

    @staticmethod
    def image_fp2tes_signal(image_fp):
        """
        Go from an image of one quarter of the focal plane to the signal of each TES.

        Parameters
        ----------
        image_fp : array of shape (17, 17)
            Image of one quarter of the focal plane with the signal for each TES.

        Returns
        -------
        tes_signal : array of shape (256,)
            Signals in each TES, the 128 first elements are for asic 1
            and the 128 next are for asic 2.
        """

        tes_signal = np.zeros(256)
        pix_grid = pix2tes.assign_pix_grid()

        TES2PIX = pix2tes.assign_pix2tes()
        for l in range(17):
            for c in range(17):
                pix = pix_grid[l, c]
                if pix != 0.:
                    if pix in TES2PIX[ASIC_index(1), :]:
                        tes = pix2tes.pix2tes(pix, 1)
                        tes_signal[tes - 1] = image_fp[l, c]

                    else:
                        tes = pix2tes.pix2tes(pix, 2)
                        tes_signal[tes - 1 + 128] = image_fp[l, c]

        return tes_signal

    @staticmethod
    def tes_signal2image_fp(tes_signal):
        """
        Go from the signal of each TES to an image of one quarter of the focal plane.

        Parameters
        ----------
        tes_signal : array of shape (256,)
            Signals in each TES, the 128 first elements are for asic 1
            and the 128 next are for asic 2.

        Returns
        -------
        image_fp : array of shape (17, 17)
            Image of one quarter of the focal plane with the signal for each TES.

        """

        image_fp = np.zeros((17, 17))

        pix_grid = pix2tes.assign_pix_grid()

        for i, signal in enumerate(tes_signal):
            tes_index = i + 1  # TES indices start at 1 and not 0

            # We split between asic1 and asic2
            if tes_index < 129:
                pix = pix2tes.tes2pix(tes_index, 1)
            else:
                pix = pix2tes.tes2pix(tes_index - 128, 2)

            # This condition avoids thermometers
            if pix < 1000:
                coord = np.reshape(np.where(pix_grid == pix), 2)
                image_fp[coord[0], coord[1]] = signal

        return image_fp

    @staticmethod
    def full2quarter(full_fp):
        """
        Reduce a complete focal plane to a quarter.
        We also make a rotation to be coherent with QubicStudio..
        Parameters
        ----------
        full_fp : array of shape (34, 34)
            Power on the total focal plane for each pointing.

        Returns
        quart_fp : array of shape (17, 17)
            The power on a quarter of the focal plane

        """
        if np.shape(full_fp) != (34, 34):
            raise ValueError('The complete focal plane must be 34*34')

        pix_grid = pix2tes.assign_pix_grid()
        focal_plan = np.where(pix_grid > 0, 1, pix_grid)
        quart_fp = np.rot90(full_fp[:17, :17], 3) * focal_plan
        quart_fp[quart_fp == 0.] = np.nan

        return quart_fp

    @staticmethod
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
        step = (xmax - xmin) / reso
        xx, yy = np.meshgrid(np.arange(xmin, xmax, step), np.arange(xmin, xmax, step))
        x1d = np.ravel(xx)
        y1d = np.ravel(yy)
        z1d = x1d * 0 - 0.3
        position = np.array([x1d, y1d, z1d]).T

        field = q._get_response(theta, phi, spectral_irradiance, position, q.detector.area, q.filter.nu, q.horn,
                                q.primary_beam, q.secondary_beam)
        power = np.reshape(np.abs(field) ** 2, (reso, reso, nptg))
        power = np.fliplr(power) # There is a symmetry bug, need to flip it

        return power

# ============= JC functions old
def tes2imgpix(tesnum, extra_args=None):
    if extra_args is None:
        a1 = qp()
        a1.assign_asic(1)
        a2 = qp()
        a2.assign_asic(2)
    else:
        a1 = extra_args[0]
        a2 = extra_args[1]

    ij = np.zeros((len(tesnum), 2))
    for i in xrange(len(tesnum)):
        if i < 128:
            pixnum = a1.tes2pix(tesnum[i])
            ww = np.where(a1.pix_grid == pixnum)
        else:
            pixnum = a2.tes2pix(tesnum[i] - 128)
            ww = np.where(a2.pix_grid == pixnum)
        if len(ww[0]) > 0:
            ij[i, :] = ww
        else:
            ij[i, :] = [17, 17]
    return ij


def fringe_focalplane(x, pars, extra_args=None):
    baseline = pars[0]
    alpha = pars[1]
    phase = pars[2]
    amplitude = pars[3]
    nu = 150e9
    lam = 3e8 / nu
    f = 300e-3  # Focal Length in mm
    freq_fringe = baseline / lam
    TESsize = 3.e-3

    ijtes = tes2imgpix(np.arange(256) + 1, extra_args=extra_args)

    fringe = amplitude * np.cos(2. * np.pi * freq_fringe * (
            ijtes[:, 0] * np.cos(alpha * np.pi / 180) + ijtes[:, 1] * np.sin(
        alpha * np.pi / 180)) * TESsize / f + phase * np.pi / 180)
    thermos = [4 - 1, 36 - 1, 68 - 1, 100 - 1, 4 - 1 + 128, 36 - 1 + 128, 68 - 1 + 128, 100 - 1 + 128]
    fringe[thermos] = 0
    mask = x > 0
    fringe[~mask] = 0
    return fringe
