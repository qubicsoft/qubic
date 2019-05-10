#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from matplotlib.pyplot import *
import qubic
from qubicpack import qubicpack as qp

from selfcal_lib import *

basedir = '/home/louisemousset/QUBIC/MyGitQUBIC'
dictfilename = basedir + '/qubic/qubic/scripts/global_source.dict'

d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)

q = qubic.QubicMultibandInstrument(d)

s = qubic.QubicScene(d)

# Focal plane parameters
reso = 34
xmin = - 0.06
xmax = 0.06

# Source parameters : positions and spectral irradiance
phi = np.arange(0., 0.8, 0.2)
theta = np.arange(0., 0.8, 0.2)
irradiance = 1.

# number of pointings (we move the source instead of the instrument)
nptg = len(theta)

# Horns
baseline_manip = [19, 1]

baseline_manip_1 = [46, 64]

baseline_manip_2 = [19, 1]

# Test with 2 pairs of horns
S_tot1, Cminus_i1, Cminus_j1, Sminus_ij1, Ci1, Cj1, Sij1 = selfcal_data(q[0], reso, xmin, xmax, theta, phi,
                                                                        irradiance, baseline_manip_1,
                                                                        dead_switch=None, doplot=True)

S_tot2, Cminus_i2, Cminus_j2, Sminus_ij2, Ci2, Cj2, Sij2 = selfcal_data(q[0], reso, xmin, xmax, theta, phi,
                                                                        irradiance, baseline_manip_2,
                                                                        dead_switch=None, doplot=True)

# Test with a close horn
S_tot, Cminus_i, Cminus_j, Sminus_ij, Ci, Cj, Sij = selfcal_data(q[0], theta, phi,
                                                                 irradiance, baseline_manip,
                                                                 dead_switch=None, doplot=True)

S_totc, Cminus_ic, Cminus_jc, Sminus_ijc, Cic, Cjc, Sijc = selfcal_data(q[0], reso, xmin, xmax, theta, phi,
                                                                        irradiance, baseline_manip,
                                                                        dead_switch=[2, 9], doplot=True)

# Reduce to the quarter of the focal plane
S_tot = full2quarter(S_tot)
Cminus_i = full2quarter(Cminus_i)
Cminus_j = full2quarter(Cminus_j)
Sminus_ij = full2quarter(Sminus_ij)
Ci = full2quarter(Ci)
Cj = full2quarter(Cj)
Sij = full2quarter(Sij)

# Figure with the fringes
figure('combination_reso={0}_baseline{1}_{2}'.format(reso, baseline_manip[0], baseline_manip[1]))
subplot(2, 2, 1)
imshow(Sij[:, :, 0], interpolation='nearest')
colorbar()
title('True Baseline $S_{ij}$')

subplot(2, 2, 2)
imshow((S_tot + Ci + Cj - Cminus_i - Cminus_j + Sminus_ij)[:, :, 0], interpolation='nearest')
colorbar()
title('$S_{-ij} + C_i + C_j - C_{-i} - C_{-j} + S_{tot}$')

subplot(2, 2, 3)
imshow((S_tot - Cminus_i - Cminus_j + Sminus_ij)[:, :, 0], interpolation='nearest')
colorbar()
title('$S_{-ij} - C_{-i} - C_{-j} + S_{tot}$')

# Animation with the fringes reconstructed for each pointing
figure()
for ptg in xrange(nptg):
    imshow((S_tot - Cminus_i - Cminus_j + Sminus_ij)[:, :, ptg],
           interpolation='nearest')
    title('$S - C_{-i} - C_{-j} + S_{-ij}$\n' + 'ptg={}'.format(ptg))
    pause(1)
    if s == 0:
        colorbar()

# Test the TES mapping
focal_plan = np.where(qp().pix_grid > 0, 1, qp().pix_grid)

S_test = np.arange(1, 290)
S_test = np.reshape(S_test, (17, 17)) * focal_plan
S_test = np.reshape(S_test, (17, 17, 1))
imshow(S_test[:, :, 0], interpolation='nearest')

tes_signal = get_tes_signal(S_tot)
imshow(ft.image_asics(all1=tes_signal[:, 0]), interpolation='nearest')

# Test============
figure()
subplot(2, 2, 1)
imshow((S_tot - Cminus_i - Cminus_j + Sminus_ij)[reso / 2:, :reso / 2, 0], interpolation='nearest')
# imshow((Sminus_ij)[reso / 2:, :reso / 2, 0], interpolation='nearest', origin='lower')

colorbar()

subplot(2, 2, 2)
imshow((S_totc - Cminus_ic - Cminus_jc + Sminus_ijc)[reso / 2:, :reso / 2, 0], interpolation='nearest')
# imshow((Sminus_ijc)[reso / 2:, :reso / 2, 0], interpolation='nearest', origin='lower')
colorbar()

subplot(2, 2, 3)
imshow(((S_tot - Cminus_i - Cminus_j + Sminus_ij)
        - (S_totc - Cminus_ic - Cminus_jc + Sminus_ijc))
       [reso / 2:, :reso / 2, 0], interpolation='nearest')
# imshow(((Sminus_ij) - (Sminus_ijc))
#        [reso / 2:, :reso / 2, 0], interpolation='nearest', origin='lower')
colorbar()



# ============== Old functions, not used anymore ====================
# These 2 functions  were used before, they use the index of the horns in simulations.


def horn_manip2simu(ihorn):
    """

    Parameters
    ----------
    ihorn : integer
        Index of a horn on the TD, between 1 and 64.

    Returns
    -------
        Index of the horn in simulations for the TD.

    """
    if ihorn < 1 or ihorn > 64:
        print ('wrong horn index')
    else:
        if ihorn % 8 == 0:
            row = ihorn // 8 - 1
            col = 7
        else:
            row = ihorn // 8
            col = ihorn % 8 - 1
        return 161 + row * 22 + col


def selfcal_data_old(q, reso, xmin, xmax, theta, phi, spectral_irradiance, baseline, doplot=False):
    """
        Returns the power on the focal plane for each pointing, for different configurations
        of the horn array: all open, all open except i, except j, except i and j, only i open,
         only j open, only i and j open.
    Parameters
    ----------
    q : a qubic monochromatic instrument
    reso : int
        Pixel number on one side on the focal plane image
    xmin : float
        Position of the border of the focal plane to the center [m]
    xmax : float
        Position of the opposite border of the focal plane to the center [m]
    theta : array-like
        The source zenith angle [rad].
    phi : array-like
        The source azimuthal angle [rad].
    spectral_irradiance : array-like
        The source spectral_irradiance [W/m^2/Hz].
    baseline : array
        Baseline formed with 2 horns, index must be like in simu.
    doplot : bool
        If True, do the plots for the first pointing.

    Returns
    -------
    S, Cminus_i, Cminus_j, Sminus_ij, Ci, Cj, Sij : arrays of shape (reso, reso, #pointings)
        Power on the focal plane for each configuration, for each pointing.

    """
    indices = q.horn.index
    q.horn.open = True
    S_config = q.horn.open

    # All open
    q.horn.open = S_config
    S = get_power_on_array(q, reso, xmin, xmax, theta, phi, spectral_irradiance)
    if doplot:
        figure()
        subplot(4, 4, 1)
        q.horn.plot()
        axis('off')
        subplot(4, 4, 2)
        imshow(S[reso / 2:, :reso / 2, 0], interpolation='nearest', origin='lower')
        print(S[:, :, 0].shape)
        colorbar()
        title('$S$')

    # All open except i
    Cminus_i_config = S_config.copy()
    Cminus_i_config[np.where(indices == baseline[0])] = False
    q.horn.open = Cminus_i_config
    Cminus_i = get_power_on_array(q, reso, xmin, xmax, theta, phi, spectral_irradiance)
    if doplot:
        subplot(4, 4, 3)
        q.horn.plot()
        axis('off')
        subplot(4, 4, 4)
        imshow(Cminus_i[reso / 2:, :reso / 2, 0], interpolation='nearest', origin='lower')
        colorbar()
        title('$C_{-i}$')

    # All open except j
    Cminus_j_config = S_config.copy()
    Cminus_j_config[np.where(indices == baseline[1])] = False
    q.horn.open = Cminus_j_config
    Cminus_j = get_power_on_array(q, reso, xmin, xmax, theta, phi, spectral_irradiance)
    if doplot:
        subplot(4, 4, 5)
        q.horn.plot()
        axis('off')
        subplot(4, 4, 6)
        imshow(Cminus_j[reso / 2:, :reso / 2, 0], interpolation='nearest', origin='lower')
        colorbar()
        title('$C_{-j}$')

    # All open except baseline [i, j]
    Sminus_ij_config = S_config.copy()
    Sminus_ij_config[np.where(indices == baseline[0])] = False
    Sminus_ij_config[np.where(indices == baseline[1])] = False
    q.horn.open = Sminus_ij_config
    Sminus_ij = get_power_on_array(q, reso, xmin, xmax, theta, phi, spectral_irradiance)
    if doplot:
        subplot(4, 4, 7)
        q.horn.plot()
        axis('off')
        subplot(4, 4, 8)
        imshow(Sminus_ij[reso / 2:, :reso / 2, 0], interpolation='nearest', origin='lower')
        colorbar()
        title('$S_{-ij}$')

    # Only i open (not a realistic observable)
    Ci_config = ~S_config.copy()
    Ci_config[np.where(indices == baseline[0])] = True
    q.horn.open = Ci_config
    Ci = get_power_on_array(q, reso, xmin, xmax, theta, phi, spectral_irradiance)
    if doplot:
        subplot(4, 4, 9)
        q.horn.plot()
        axis('off')
        subplot(4, 4, 10)
        imshow(Ci[reso / 2:, :reso / 2, 0], interpolation='nearest', origin='lower')
        colorbar()
        title('$C_i$')

    # Only j open (not a realistic observable)
    Cj_config = ~S_config.copy()
    Cj_config[np.where(indices == baseline[1])] = True
    q.horn.open = Cj_config
    Cj = get_power_on_array(q, reso, xmin, xmax, theta, phi, spectral_irradiance)
    if doplot:
        subplot(4, 4, 11)
        q.horn.plot()
        axis('off')
        subplot(4, 4, 12)
        imshow(Cj[reso / 2:, :reso / 2, 0], interpolation='nearest', origin='lower')
        colorbar()
        title('$C_j$')

    # Only baseline [i, j] open (not a realistic observable)
    Sij_config = ~S_config.copy()
    Sij_config[np.where(indices == baseline[0])] = True
    Sij_config[np.where(indices == baseline[1])] = True
    q.horn.open = Sij_config
    Sij = get_power_on_array(q, reso, xmin, xmax, theta, phi, spectral_irradiance)
    if doplot:
        subplot(4, 4, 13)
        q.horn.plot()
        axis('off')
        subplot(4, 4, 14)
        imshow(Sij[reso / 2:, :reso / 2, 0], interpolation='nearest', origin='lower')
        colorbar()
        title('$S_{ij}$')

    return S, Cminus_i, Cminus_j, Sminus_ij, Ci, Cj,
