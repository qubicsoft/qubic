from qubicpack import qubicpack as qp
from qubicpack import pix2tes
from qubicpack.utilities import ASIC_index

import qubic

from matplotlib.pyplot import *


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

    image_fp = np.zeros((17, 17)) + np.nan

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
            coord = np.reshape(np.where(pix_grid == pix), (2))
            image_fp[coord[0], coord[1]] = signal

    return image_fp


def get_power_on_array(q, theta, phi, spectral_irradiance, reso=34, xmin=-0.06, xmax=0.06):
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
    return power


def get_power_combinations(q, theta, phi, spectral_irradiance, baseline, reso=34,
                 xmin=-0.06, xmax=0.06, dead_switch=None, doplot=False):
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
    baseline : array
        Baseline formed with 2 horns, index between 1 and 64 as in the manip.
    reso : int
        Pixel number on one side on the focal plane image
    xmin : float
        Position of the border of the focal plane to the center [m]
    xmax : float
        Position of the opposite border of the focal plane to the center [m]
    dead_switch : int or list of int
        Broken switches, always closed.
    doplot : bool
        If True, do the plots for the first pointing.

    Returns
    -------
    S, Cminus_i, Cminus_j, Sminus_ij, Ci, Cj, Sij : arrays of shape (reso, reso, #pointings)
        Power on the focal plane for each configuration, for each pointing.

    """

    # All open
    q.horn.open = True
    if dead_switch is not None:
        q.horn.open[dead_switch] = False
    S = get_power_on_array(q, theta, phi, spectral_irradiance, reso, xmin, xmax)
    if doplot:
        figure()
        subplot(4, 4, 1)
        q.horn.plot()
        axis('off')
        subplot(4, 4, 2)
        imshow(S[:, :, 0], interpolation='nearest')
        print(S[:, :, 0].shape)
        colorbar()
        title('$S$')

    # All open except i
    q.horn.open = True
    if dead_switch is not None:
        q.horn.open[dead_switch] = False
    q.horn.open[baseline[0] - 1] = False
    Cminus_i = get_power_on_array(q, theta, phi, spectral_irradiance, reso, xmin, xmax)
    if doplot:
        subplot(4, 4, 3)
        q.horn.plot()
        axis('off')
        subplot(4, 4, 4)
        imshow(Cminus_i[:, :, 0], interpolation='nearest')
        colorbar()
        title('$C_{-i}$')

    # All open except j
    q.horn.open = True
    if dead_switch is not None:
        q.horn.open[dead_switch] = False
    q.horn.open[baseline[1] - 1] = False
    Cminus_j = get_power_on_array(q, theta, phi, spectral_irradiance, reso, xmin, xmax)
    if doplot:
        subplot(4, 4, 5)
        q.horn.plot()
        axis('off')
        subplot(4, 4, 6)
        imshow(Cminus_j[:, :, 0], interpolation='nearest')
        colorbar()
        title('$C_{-j}$')

    # All open except baseline [i, j]
    q.horn.open = True
    if dead_switch is not None:
        q.horn.open[dead_switch] = False
    q.horn.open[baseline[0] - 1] = False
    q.horn.open[baseline[1] - 1] = False
    Sminus_ij = get_power_on_array(q, theta, phi, spectral_irradiance, reso, xmin, xmax)
    if doplot:
        subplot(4, 4, 7)
        q.horn.plot()
        axis('off')
        subplot(4, 4, 8)
        imshow(Sminus_ij[:, :, 0], interpolation='nearest')
        colorbar()
        title('$S_{-ij}$')

    # Only i open (not a realistic observable)
    q.horn.open = False
    if dead_switch is not None:
        q.horn.open[dead_switch] = False
    q.horn.open[baseline[0] - 1] = True
    Ci = get_power_on_array(q, theta, phi, spectral_irradiance, reso, xmin, xmax)
    if doplot:
        subplot(4, 4, 9)
        q.horn.plot()
        axis('off')
        subplot(4, 4, 10)
        imshow(Ci[:, :, 0], interpolation='nearest')
        colorbar()
        title('$C_i$')

    # Only j open (not a realistic observable)
    q.horn.open = False
    q.horn.open[baseline[1] - 1] = True
    Cj = get_power_on_array(q, theta, phi, spectral_irradiance, reso, xmin, xmax)
    if doplot:
        subplot(4, 4, 11)
        q.horn.plot()
        axis('off')
        subplot(4, 4, 12)
        imshow(Cj[:, :, 0], interpolation='nearest')
        colorbar()
        title('$C_j$')

    # Only baseline [i, j] open (not a realistic observable)
    q.horn.open = False
    q.horn.open[baseline[0] - 1] = True
    q.horn.open[baseline[1] - 1] = True
    Sij = get_power_on_array(q, theta, phi, spectral_irradiance, reso, xmin, xmax)
    if doplot:
        subplot(4, 4, 13)
        q.horn.plot()
        axis('off')
        subplot(4, 4, 14)
        imshow(Sij[:, :, 0], interpolation='nearest')
        colorbar()
        title('$S_{ij}$')

    return S, Cminus_i, Cminus_j, Sminus_ij, Ci, Cj, Sij


def full2quarter(signal):
    """
    Reduce a complete focal plane to a quarter.
    Parameters
    ----------
    signal : array of shape (34, 34, #pointings)
        Power on the total focal plane for each pointing.

    Returns
        s_quarter : array of shape (17, 17, #pointings)
            The power on a quarter of the focal plane

    """
    if np.shape(signal)[0] != 34:
        raise ValueError('The complete focal plane must be 34*34')

    pix_grid = pix2tes.assign_pix_grid()
    focal_plan = np.where(pix_grid > 0, 1, pix_grid)

    nptg = np.shape(signal)[2]
    s_quarter = np.empty((17, 17, nptg))

    for ptg in xrange(nptg):
        s_quarter[:, :, ptg] = signal[:17, 17:, ptg] * focal_plan

    return s_quarter


def get_fringes_fp_TD(baseline, basedir='../', theta=np.array([0.]), phi=np.array([0.]), irradiance=1.):
    """
    Computes the fringe signals in each TES for point source.
    The sources moves and we compute the fringes for each pointing
    that corresponds to a positio  of the source.

    Parameters
    ----------
    baseline : array
        Baseline formed with 2 horns, index between 1 and 64 as in the manip.
    basedir : str
        Path of the dictionary.
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
    dictfilename = basedir + 'global_source.dict'
    d = qubic.qubicdict.qubicDict()
    d.read_from_file(dictfilename)
    q = qubic.QubicMultibandInstrument(d)

    S_tot, Cminus_i, Cminus_j, Sminus_ij, Ci, Cj, Sij = get_power_combinations(q[0], theta, phi,
                                                                     irradiance, baseline,
                                                                     dead_switch=None, doplot=False)

    S_tot = full2quarter(S_tot)
    Cminus_i = full2quarter(Cminus_i)
    Cminus_j = full2quarter(Cminus_j)
    Sminus_ij = full2quarter(Sminus_ij)
    Ci = full2quarter(Ci)

    fringes = (S_tot - Cminus_i - Cminus_j + Sminus_ij) / Ci

    nptg = np.shape(fringes)[2]
    tes_fringes_signal = np.empty((256, nptg))
    for ptg in range(nptg):
        tes_fringes_signal[:, ptg] = image_fp2tes_signal(fringes[:, :, ptg])

    return tes_fringes_signal
