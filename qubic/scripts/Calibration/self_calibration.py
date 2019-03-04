#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib.pyplot import *
import healpy as hp
from astropy.io import fits
import qubic

basedir = '/home/louisemousset/QUBIC/MyGitQUBIC'
dictfilename = basedir + '/qubic/qubic/scripts/global_source.dict'

d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)

q = qubic.QubicMultibandInstrument(d)

s = qubic.QubicScene(d)


def get_power_on_array(q, nn, xmin, xmax, theta, phi, spectral_irradiance):
    """
    Compute power on the focal plane for different position of the source
    with respect to the instrument.

    Parameters
    ----------
    q : a qubic monochromatic instrument
    nn : int
        Pixel number on one side on the focal plane image
    xmin : float
    xmax : float
    theta : array-like
        The source zenith angle [rad].
    phi : array-like
        The source azimuthal angle [rad].
    spectral_irradiance : array-like
        The source spectral_irradiance [W/m^2/Hz].

    Returns
    ----------
    power : array of shape (nn, nn, #pointings)
        The power on the focal plane for each pointing.
    """
    nptg = len(theta)
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, nn), np.linspace(xmin, xmax, nn))
    x1d = np.ravel(xx)
    y1d = np.ravel(yy)
    z1d = x1d * 0 - 0.3
    position = np.array([x1d, y1d, z1d]).T

    field = q._get_response(theta, phi, spectral_irradiance, position, q.detector.area,
                                      q.filter.nu, q.horn, q.primary_beam, q.secondary_beam)
    power = np.reshape(np.abs(field) ** 2, (nn, nn, nptg))
    return power


def selfcal_data(q, nn, xmin, xmax, theta, phi, spectral_irradiance, baseline, doplot=False):
    """
        Returns the power on the focal plane for each pointing, for different configurations
        of the horn array: all open, all open except i, except j, except i and j, only i open,
         only j open, only i and j open.
    Parameters
    ----------
    q : a qubic monochromatic instrument
    nn : int
        Pixel number on one side on the focal plane image
    xmin : float
    xmax : float
    theta : array-like
        The source zenith angle [rad].
    phi : array-like
        The source azimuthal angle [rad].
    spectral_irradiance : array-like
        The source spectral_irradiance [W/m^2/Hz].
    baseline : array
        Baseline formed with 2 horns.
    doplot : bool
        If True, do the plots for the first pointing.

    Returns
    -------
    S, Cminus_i, Cminus_j, Sminus_ij, Ci, Cj, Sij : arrays of shape (nn, nn, #pointings)
        Power on the focal plane for each configuration, for each pointing.

    """
    indices = q.horn.index
    q.horn.open = True
    S_config = q.horn.open

    # All open
    q.horn.open = S_config
    S = get_power_on_array(q, nn, xmin, xmax, theta, phi, spectral_irradiance)
    if doplot:
        figure()
        subplot(4, 4, 1)
        q.horn.plot()
        subplot(4, 4, 2)
        imshow(S[:,:,0])
        title('$S$')

    # All open except i
    Cminus_i_config = S_config.copy()
    Cminus_i_config[np.where(indices == baseline[0])] = False
    q.horn.open = Cminus_i_config
    Cminus_i = get_power_on_array(q, nn, xmin, xmax, theta, phi, spectral_irradiance)
    if doplot:
        subplot(4, 4, 3)
        q.horn.plot()
        subplot(4, 4, 4)
        imshow(Cminus_i[:,:,0])
        title('$C_{-i}$')

    # All open except j
    Cminus_j_config = S_config.copy()
    Cminus_j_config[np.where(indices == baseline[1])] = False
    q.horn.open = Cminus_j_config
    Cminus_j = get_power_on_array(q, nn, xmin, xmax, theta, phi, spectral_irradiance)
    if doplot:
        subplot(4, 4, 5)
        q.horn.plot()
        subplot(4, 4, 6)
        imshow(Cminus_j[:,:,0])
        title('$C_{-j}$')

    # All open except baseline [i, j]
    Sminus_ij_config = S_config.copy()
    Sminus_ij_config[np.where(indices == baseline[0])] = False
    Sminus_ij_config[np.where(indices == baseline[1])] = False
    q.horn.open = Sminus_ij_config
    Sminus_ij = get_power_on_array(q, nn, xmin, xmax, theta, phi, spectral_irradiance)
    if doplot:
        subplot(4, 4, 7)
        q.horn.plot()
        subplot(4, 4, 8)
        imshow(Sminus_ij[:,:,0])
        title('$S_{-ij}$')

    # Only i open (not a realistic observable)
    Ci_config = ~S_config.copy()
    Ci_config[np.where(indices == baseline[0])] = True
    q.horn.open = Ci_config
    Ci = get_power_on_array(q, nn, xmin, xmax, theta, phi, spectral_irradiance)
    if doplot:
        subplot(4, 4, 9)
        q.horn.plot()
        subplot(4, 4, 10)
        imshow(Ci[:,:,0])
        title('$C_i$')

    # Only j open (not a realistic observable)
    Cj_config = ~S_config.copy()
    Cj_config[np.where(indices == baseline[1])] = True
    q.horn.open = Cj_config
    Cj = get_power_on_array(q, nn, xmin, xmax, theta, phi, spectral_irradiance)
    if doplot:
        subplot(4, 4, 11)
        q.horn.plot()
        subplot(4, 4, 12)
        imshow(Cj[:,:,0])
        title('$C_j$')

    # Only baseline [i, j] open (not a realistic observable)
    Sij_config = ~S_config.copy()
    Sij_config[np.where(indices == baseline[0])] = True
    Sij_config[np.where(indices == baseline[1])] = True
    q.horn.open = Sij_config
    Sij = get_power_on_array(q, nn, xmin, xmax, theta, phi, spectral_irradiance)
    if doplot:
        subplot(4, 4, 13)
        q.horn.plot()
        subplot(4, 4, 14)
        imshow(Sij[:,:,0])
        title('$S_{ij}$')

    return S, Cminus_i, Cminus_j, Sminus_ij, Ci, Cj, Sij


nn = 100
xmin = -0.06
xmax = 0.06
phi = np.linspace(0.2, 0.4, 4)
theta = np.linspace(0.2, 0.4, 4)
spectral_irradiance = 0.5
baseline = [161, 165]
nptg = len(theta)

power = get_power_on_array(q[0], nn, xmin, xmax, theta, phi, spectral_irradiance)
S, Cminus_i, Cminus_j, Sminus_ij, Ci, Cj, Sij = selfcal_data(q[0], nn, xmin, xmax, theta, phi,
                                                    spectral_irradiance, baseline, doplot=True)


figure()
subplot(2, 2, 1)
imshow(Sij[:,:,0], interpolation='nearest', origin='lower')
colorbar()
title('True Baseline $S_{ij}$')

subplot(2, 2, 2)
imshow((S - Cminus_i - Cminus_j + Sminus_ij)[:,:,0], interpolation='nearest', origin='lower')
colorbar()
title('$S - C_{-i} - C_{-j} + S_{-ij}$ (Old JC)')

subplot(2, 2, 3)
imshow((Cminus_i + Cminus_j - 2 * Sminus_ij)[:,:,0], interpolation='nearest', origin='lower')
colorbar()
title('$C_{-i} + C_{-j} - 2*S_{-ij}$ (Paper - Wrong !)')

subplot(2, 2, 4)
imshow((Ci + Cj - Cminus_i - Cminus_j + Sminus_ij + S)[:,:,0], interpolation='nearest', origin='lower')
colorbar()
title('$C_i + C_j - C_{-i} - C_{-j} + S_{-ij} + S$ (Creidhe)')


figure()
for ptg in xrange(nptg):
    imshow((S - Cminus_i - Cminus_j + Sminus_ij)[:, :, ptg], interpolation='nearest', origin='lower')
    title('$S - C_{-i} - C_{-j} + S_{-ij}$\n' + 'ptg={}'.format(ptg))
    pause(1)
    if s==0: colorbar()