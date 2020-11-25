from __future__ import division, print_function

import glob
import numpy as np
import pandas as pd

from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import dblquad

import matplotlib.pyplot as plt

from qubicpack.pixel_translation import make_id_focalplane, tes2index

__all__ = ['Fringes']


# ========== Plot functions =============
def plot_horns(q):
    xhorns = q.horn.center[:, 0]
    yhorns = q.horn.center[:, 1]
    plt.plot(xhorns, yhorns, 'ro')


def plot_baseline(q, bs):
    hcenters = q.horn.center[:, 0:2]
    plt.plot(hcenters[np.array(bs) - 1, 0], hcenters[np.array(bs) - 1, 1], lw=4, label=bs)


def scatter_plot_FP(q, x, y, FP_signal, frame, s=None, title=None, unit='[W / Hz]', **kwargs):
    if s is None:
        if q.config == 'TD':
            s = 180
        else:
            s = 40
    plt.scatter(x, y, c=FP_signal, marker='s', s=s, **kwargs)
    clb = plt.colorbar()
    clb.ax.set_title(unit)
    plt.xlabel(f'X_{frame} [m]', fontsize=14)
    plt.ylabel(f'Y_{frame} [m]', fontsize=14)
    plt.axis('square')
    plt.title(title, fontsize=14)


def pcolor_plot_FP(q, x, y, FP_signal, frame, title=None, unit='[W / Hz]', **kwargs):
    x2D = q.detector.unpack(x)
    y2D = q.detector.unpack(y)
    FP_signal2D = q.detector.unpack(FP_signal)

    plt.pcolor(x2D, y2D, FP_signal2D, **kwargs)
    clb = plt.colorbar()
    clb.ax.set_title(unit)
    plt.xlabel(f'X_{frame} [m]', fontsize=14)
    plt.ylabel(f'Y_{frame} [m]', fontsize=14)
    plt.axis('square')
    plt.title(title, fontsize=14)


def plot_horn_and_FP(q, x, y, FP_signal, frame, title=None, s=None, unit='[W / Hz]', **kwargs):
    plt.subplots(1, 2)
    plt.suptitle(title, fontsize=18)
    plt.subplots_adjust(wspace=0.3)

    plt.subplot(121)
    q.horn.plot()
    plt.axis('square')
    plt.xlabel('X_GRF [m]', fontsize=14)
    plt.ylabel('Y_GRF [m]', fontsize=14)

    plt.subplot(122)
    if s is None:
        if q.config == 'TD':
            s = 180
        else:
            s = 40
    plt.scatter(x, y, c=FP_signal, marker='s', s=s, **kwargs)
    clb = plt.colorbar()
    clb.ax.set_title(unit)
    plt.xlabel(f'X_{frame} [m]', fontsize=14)
    plt.ylabel(f'Y_{frame} [m]', fontsize=14)
    plt.axis('square')


# ========== Tool functions =============
def close_switches(q, switches):
    q.horn.open = True
    for i in switches:
        q.horn.open[i - 1] = False


def open_switches(q, switches):
    q.horn.open = False
    for i in switches:
        q.horn.open[i - 1] = True


def get_TEScoordinates_ONAFP(q):
    # TES centers in the ONAFP frame
    xGRF_TES = q.detector.center[:, 0]
    yGRF_TES = q.detector.center[:, 1]
    xONAFP_TES = - yGRF_TES
    yONAFP_TES = xGRF_TES

    # TES vertex in the ONAFP frame
    vGRF_TES = q.detector.vertex
    vONAFP_TES = vGRF_TES[..., [1, 0, 2]]
    vONAFP_TES[..., 0] *= - 1

    return xONAFP_TES, yONAFP_TES, vONAFP_TES


def TES_Instru2coord(TES, ASIC, q, frame='ONAFP'):
    """
    From (TES, ASIC) numbering on the instrument to (x,y) coordinates in ONAFP or GRF frame.
    Returns also the focal plane index.
    !!! If q is a TD instrument, only ASIC 1 and 2 are acceptable.
    Parameters
    ----------
    TES: TES number as defined on the instrument
    ASIC: ASIC number
    q: QubicInstrument()
    frame: str
        'GRF' or 'ONAFP' only

    Returns
    -------

    """
    if TES in [4, 36, 68, 100]:
        raise ValueError('This is a thermometer !')
    FP_index = tes2index(TES, ASIC)
    print('FP_index', FP_index)

    centerGRF = q.detector.center[q.detector.index == FP_index][0]
    xGRF = centerGRF[0]
    yGRF = centerGRF[1]

    if frame == 'GRF':
        print('X_GRF = {:.3f} mm, Y_GRF = {:.3f} mm'.format(xGRF * 1e3, yGRF * 1e3))
        return xGRF, yGRF, FP_index

    elif frame == 'ONAFP':
        xONAFP = - yGRF
        yONAFP = xGRF
        print('X_ONAFP = {:.3f} mm, Y_ONAFP = {:.3f} mm'.format(xONAFP * 1e3, yONAFP * 1e3))
        return xONAFP, yONAFP, FP_index
    else:
        raise ValueError('The frame is not valid.')


def get_TES_Instru_coords(q, frame='ONAFP'):
    thermos = [4, 36, 68, 100]
    if q.config == 'TD':
        nASICS = 2
    else:
        nASICS = 8

    nTES = nASICS * 128
    x = np.zeros(nTES)
    y = np.zeros(nTES)
    FP_index = np.zeros(nTES)

    for ASIC in range(1, nASICS + 1):
        for TES in range(1, 129):
            print(f'\n ASIC {ASIC} - TES {TES}')
            if TES not in thermos:
                index = (TES - 1) + 128 * (ASIC - 1)
                x[index], y[index], FP_index[index] = TES_Instru2coord(TES, ASIC, q, frame=frame)
            else:
                print('Thermometer !')

    return x, y, FP_index


def get_TESvertices_FromMaynoothFiles(rep, ndet=992):
    """
    Get TES vertices from Maynooth files.
    Parameters
    ----------
    rep : str
        Path of the repository for the simulated files, can be download at :
        https://drive.google.com/open?id=19dPHw_CeuFZ068b-VRT7N-LWzOL1fmfG
    ndet : int
        Number of TES.

    Returns
    -------
    A 3D array containing the TES vertices coordinates on the focal plane.
    Shape=(992, 4, 2)

    """
    # Get a 2D array from the file
    vertices2D = pd.read_csv(rep + '/vertices.txt', sep='\ ', header=None, engine='python')

    # Make a 3D array of shape (992, 4, 2)
    vertices = np.zeros((ndet, 4, 2))
    for i in range(4):
        vertices[:, i, :] = vertices2D.iloc[i::4, :]
    return vertices


def make_position(xmin, xmax, reso, focal_length):
    """Position on the focal plane in the GRF frame."""
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, reso), np.linspace(xmin, xmax, reso))
    x1d = np.ravel(xx)
    y1d = np.ravel(yy)
    z1d = x1d * 0 - focal_length
    position = np.array([x1d, y1d, z1d]).T

    return position


def give_bs_pars(q, bs):
    hc = q.horn.center[:, 0:2]
    hc0 = hc[np.array(bs[0]) - 1, :]
    hc1 = hc[np.array(bs[1]) - 1, :]
    bsxy = hc1 - hc0
    theta = np.degrees(np.arctan2(bsxy[1], bsxy[0]))
    length = np.sqrt(np.sum(bsxy ** 2))
    return theta, length


def check_equiv(vecbs1, vecbs2, tol=1e-5):
    norm1 = np.dot(vecbs1, vecbs1)
    norm2 = np.dot(vecbs2, vecbs2)
    cross12 = np.cross(vecbs1, vecbs2)
    if (np.abs(norm1 - norm2) < tol) & (np.abs(cross12) < tol):
        return True
    else:
        return False


def find_equivalent_baselines(all_bs, q):
    ### Convert to array
    all_bs = np.array(all_bs)
    ### centers
    hcenters = q.horn.center[:, 0:2]
    ### Baselines vectors
    all_vecs = np.zeros((len(all_bs), 2))
    for ib in range(len(all_bs)):
        coordsA = hcenters[all_bs[ib][0], :]
        coordsB = hcenters[all_bs[ib][1], :]
        all_vecs[ib, :] = coordsB - coordsA

    ### List of types of equivalence for each baseline: initially = -1
    all_eqtype = np.zeros(len(all_bs), dtype=int) - 1

    ### First type is zero and is associated to first baseline
    eqnum = 0
    all_eqtype[0] = eqnum

    ### Indices of baselines
    index_bs = np.arange(len(all_bs))

    ### Loop over baselines
    for ib in range(0, len(all_bs)):
        ### Identify those that have no type
        wnotype = all_eqtype == -1
        bsnotype = all_bs[wnotype]
        vecsnotype = all_vecs[wnotype, :]
        indexnotype = index_bs[wnotype]
        ### Loop over those with no type
        for jb in range(len(bsnotype)):
            ### Check if equivalent
            status = check_equiv(all_vecs[ib, :], vecsnotype[jb, :])
            ### If so: give it the current type
            if status:
                all_eqtype[indexnotype[jb]] = eqnum
        ### We have gone through all possibilities for this type so we increment the type by 1
        eqnum = np.max(all_eqtype) + 1

    alltypes = np.unique(all_eqtype)
    BLs_sort = []
    for i in range(len(alltypes)):
        BLs_sort.append(index_bs[all_eqtype == i])
    return BLs_sort, all_eqtype


def plot_BLs_eq(allBLs, BLs_sort, q):
    nclass_eq = len(BLs_sort)

    plt.subplots(1, nclass_eq, figsize=(16, 6))
    for i in range(nclass_eq):
        dataset_eq = BLs_sort[i]
        ax = plt.subplot(1, nclass_eq, i + 1)
        ax.set_aspect('equal')
        plot_horns(q)
        plt.title(f'Type {i}', fontsize=14)
        print(f'Type {i}:')
        for j in dataset_eq:
            print(f'  - {allBLs[j]}')
            plot_baseline(q, allBLs[j])
        plt.legend()

    return


# ========== Compute power on the focal plane =============
def make_external_A(rep, open_horns):
    """
    Compute external_A from simulated files with aberrations.
    This can be used in get_response_power method that returns the synthetic beam on the sky.
    Parameters
    ----------
    rep : str
        Path of the repository for the simulated files, can be download at :
        https://drive.google.com/open?id=19dPHw_CeuFZ068b-VRT7N-LWzOL1fmfG
    open_horns : list
        Indices of the open horns between 1 and 64.

    Returns
    -------
    external_A : list of tables describing the phase and amplitude at each point of the focal
            plane for each of the horns:
            [0] : array, X coordinates with shape (n) in GRF [m]
            [1] : array, Y coordinates with shape (n) in GRF [m]
            [2] : array, amplitude on X with shape (n, nhorns)
            [3] : array, amplitude on Y with shape (n, nhorns)
            [4] : array, phase on X with shape (n, nhorns) [rad]
            [5] : array, phase on Y with shape (n, nhorns) [rad]

    """
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
    nn = (data0['X_Index'].iloc[-1] + 1)
    print('Sampling number = ', nn)

    n = len(data0.index)

    # X and Y positions in the GRF frame
    xONAFP = data0['X'] * 1e-3
    yONAFP = data0['Y'] * 1e-3
    xGRF = yONAFP
    yGRF = - xONAFP
    print(xGRF.shape)

    # Get all amplitudes and phases for each open horn
    nopen_horns = len(open_horns)

    allampX = np.empty((n, nopen_horns))
    allphiX = np.empty((n, nopen_horns))
    allampY = np.empty((n, nopen_horns))
    allphiY = np.empty((n, nopen_horns))
    print(allphiY.shape)
    for i, swi in enumerate(open_horns):
        print('horn ', swi)
        if swi < 1 or swi > 64:
            raise ValueError('The horn indices must be between 1 and 64 ')

        thefile = files[horn_transpose[swi - 1]]
        print('Horn ', swi, ': ', thefile[-10:])
        data = pd.read_csv(thefile, sep='\t', skiprows=0)

        print(data['MagX'].shape)
        allampX[:, i] = data['MagX']
        allampY[:, i] = data['MagX']

        allphiX[:, i] = data['PhaseX']
        allphiY[:, i] = data['PhaseY']

    external_A = [xGRF, yGRF, allampX, allampY, allphiX, allphiY]

    return external_A


def get_response_power(q,
                       theta, phi, nu, spectral_irradiance,
                       frame='ONAFP', external_A=None, hwp_position=0,
                       verbose=False):
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

    if external_A is None:
        position = q.detector.center  # GRF
    else:
        x1d = external_A[0]
        y1d = external_A[1]
        z1d = x1d * 0 - q.optics.focal_length
        position = np.array([x1d, y1d, z1d]).T
    xGRF = position[:, 0]
    yGRF = position[:, 1]

    # Electric field on the FP in the GRF frame
    E = q._get_response(theta, phi, spectral_irradiance, position, q.detector.area,
                        nu, q.horn, q.primary_beam, q.secondary_beam,
                        external_A=external_A, hwp_position=hwp_position)
    power = np.abs(E) ** 2
    # power *= q.filter.bandwidth  # [W/Hz] to [W]

    if verbose:
        print(f'# pointings = {nptg}')
        print(q.detector.center.shape)
        print(E.shape)
        print(power.shape)
        print(xGRF.shape)

    if frame == 'GRF':
        return xGRF, yGRF, power
    elif frame == 'ONAFP':
        # Make a pi/2 rotation from GRF -> ONAFP referential frame
        xONAFP = - yGRF
        yONAFP = xGRF
        return xONAFP, yONAFP, power
    else:
        raise ValueError('The frame is not valid. It must be GRF or ONAFP.')


def get_power_Maynooth(open_horns, theta, nu, horn_center, rep, hwp_position=0, verbose=True):
    # Get simulation files
    files = sorted(glob.glob(rep + '/*.dat'))
    if len(files) != 64:
        raise ValueError('You should have 64 .dat files')

    nhorns = len(open_horns)
    # Get the sample number from the first file and the coordinates X, Y
    data0 = pd.read_csv(files[0], sep='\t', skiprows=0)
    nn = data0['X_Index'].iloc[-1] + 1
    xONAFP = data0['X'] * 1e-3  # convert from mm to m
    yONAFP = data0['Y'] * 1e-3

    if verbose:
        print(f'# open horns = {nhorns}')
        print(f'Sampling number = {nn}')
        print(f'Number of lines = {nn ** 2}')

    # This is done to get the right file for each horn
    horn_transpose = np.arange(64)
    horn_transpose = np.reshape(horn_transpose, (8, 8))
    horn_transpose = np.ravel(horn_transpose.T)

    Ax = np.zeros((nhorns, nn ** 2))
    Ay = np.zeros_like(Ax)
    Phasex = np.zeros_like(Ax)
    Phasey = np.zeros_like(Ax)
    for i, swi in enumerate(open_horns):
        if swi < 1 or swi > 64:
            raise ValueError('The switch indices must be between 1 and 64 ')

        thefile = files[horn_transpose[swi - 1]]
        if verbose:
            print('Horn ', swi, ': ', thefile[-10:])
        data = pd.read_csv(thefile, sep='\t', skiprows=0)

        # Phase calculation
        horn_x = horn_center[swi - 1, 0]
        horn_y = horn_center[swi - 1, 1]
        dist = np.sqrt(horn_x ** 2 + horn_y ** 2)  # distance between the horn and the center
        additional_phase = - 2 * np.pi / 3e8 * nu * 1e9 * dist * np.sin(np.deg2rad(theta))

        Ax[i, :] = data['MagX']
        Ay[i, :] = data['MagY']

        Phasex[i, :] = data['PhaseX'] + additional_phase
        Phasey[i, :] = data['PhaseY'] + additional_phase

    # Electric field for each open horn
    Ex = Ax * (np.cos(Phasex) + 1j * np.sin(Phasex))
    Ey = Ay * (np.cos(Phasey) + 1j * np.sin(Phasey))

    # Sum of the electric fields
    sumEx = np.sum(Ex, axis=0)
    sumEy = np.sum(Ey, axis=0)

    # HWP modulation
    phi_hwp = np.arange(0, 8) * np.pi / 16
    sumEx *= np.cos(2 * phi_hwp[hwp_position])
    sumEy *= np.sin(2 * phi_hwp[hwp_position])

    # Power on the focal plane
    # power = np.abs(sumEx) ** 2 + np.abs(sumEy) ** 2
    power = np.abs(sumEx + sumEy) ** 2

    return xONAFP, yONAFP, power


def fullreso2TESreso(x, y, power, TESvertex, TESarea, interp=False, verbose=True):
    ndet = np.shape(TESvertex)[0]
    powerTES = np.zeros(ndet)
    print('ndet:', ndet)

    if interp:
        print('********** Begin interpolation **********')
        reso = int(np.sqrt(x.shape[0]))
        print('Reso:', reso)
        power_interp = RegularGridInterpolator((np.unique(x), np.unique(y)), power.reshape((reso, reso)), method='linear',
                                               bounds_error=False, fill_value=0.)
        power_interp_function = lambda x, y: power_interp(np.array([x, y]))

        print('********** Begin integration in the TES era **********')
        for TES in range(ndet):
            # Boundaries for the integral
            x1 = np.min(TESvertex[TES, :, 0])
            x2 = np.max(TESvertex[TES, :, 0])
            y1 = np.min(TESvertex[TES, :, 1])
            y2 = np.max(TESvertex[TES, :, 1])
            gfun = lambda x: y1
            hfun = lambda x: y2
            if verbose:
                xTES = (x1 + x2) / 2
                yTES = (y1 + y2) / 2
                print('\n Power at the TES center:', power_interp_function(xTES, yTES))
                print('x boundaries: {:.2f} to {:.2f} mm'.format(x1 * 1e3, x2 * 1e3))
                print('Delta x = {:.2f} mm'.format((x2 - x1) * 1e3))
                print('y boundaries: {:.2f} to {:.2f} mm'.format(y1 * 1e3, y2 * 1e3))
                print('Delta y = {:.2f} mm'.format((y2 - y1) * 1e3))

            power, _ = dblquad(power_interp_function, x1, x2, gfun, hfun)
            powerTES[TES] = power

        powerTES /= TESarea

    else:
        for TES in range(ndet):
            x1 = np.min(TESvertex[TES, :, 0])
            x2 = np.max(TESvertex[TES, :, 0])
            y1 = np.min(TESvertex[TES, :, 1])
            y2 = np.max(TESvertex[TES, :, 1])

            insideTEScondition = ((x >= x1) & (x <= x2) & (y >= y1) & (y <= y2))
            indices = np.where(insideTEScondition)[0]
            count = indices.shape[0]
            powerTES[TES] = np.sum(power[indices])
            powerTES[TES] /= count

    return powerTES

# ========== Fringe simulations =============
class Fringes:
    def __init__(self, baseline):
        """
        Parameters
        ----------
        baseline: list
            Baseline formed with 2 horns, index between 1 and 64 as on the instrument.
        """
        for i in baseline:
            if i < 1 or i > 400:
                raise ValueError('Horns indices must be in [1, 400].')

        self.baseline = baseline

    def get_fringes(self, q,
                    theta=np.array([0.]), phi=np.array([0.]),
                    nu=150e9, spectral_irradiance=1.,
                    frame='ONAFP',
                    doplot=True, verbose=True, **kwargs):

        q.horn.open = False
        q.horn.open[self.baseline[0] - 1] = True
        q.horn.open[self.baseline[1] - 1] = True
        x, y, fringes = get_response_power(q, theta, phi, nu, spectral_irradiance,
                                           frame=frame, verbose=verbose)

        if doplot:
            nptg = np.shape(theta)[0]
            for i in range(nptg):
                plot_horn_and_FP(q, x, y, fringes[:, i], frame=frame,
                                 title='Baseline {} - Theta={}deg - Phi={}deg'.format(self.baseline,
                                                                                      np.rad2deg(theta[i]),
                                                                                      np.rad2deg(phi[i])), **kwargs)
        return x, y, fringes

    def get_all_combinations_power(self, q,
                                   theta=np.array([0.]), phi=np.array([0.]),
                                   nu=150e9, spectral_irradiance=1.,
                                   frame='ONAFP',
                                   doplot=True, verbose=True, **kwargs):
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

        q.horn.open = True

        # All open
        x, y, S = get_response_power(q, theta, phi, nu, spectral_irradiance, frame=frame, verbose=verbose)
        if doplot:
            plot_horn_and_FP(q, x, y, S[:, 0], frame=frame, title='$S$ - All open', **kwargs)

        # All open except i
        q.horn.open[self.baseline[0] - 1] = False
        _, _, Cminus_i = get_response_power(q, theta, phi, nu, spectral_irradiance, frame=frame, verbose=verbose)
        if doplot:
            plot_horn_and_FP(q, x, y, Cminus_i[:, 0], frame=frame,
                             title='$C_{-i}$' + f' - Horn {self.baseline[0]} close', **kwargs)

        # All open except baseline [i, j]
        q.horn.open[self.baseline[1] - 1] = False
        _, _, Sminus_ij = get_response_power(q, theta, phi, nu, spectral_irradiance, frame=frame, verbose=verbose)
        if doplot:
            plot_horn_and_FP(q, x, y, Sminus_ij[:, 0], frame=frame,
                             title='$S_{-ij}$' + f' - Baseline {self.baseline} close', **kwargs)

        # All open except j
        q.horn.open[self.baseline[0] - 1] = True
        _, _, Cminus_j = get_response_power(q, theta, phi, nu, spectral_irradiance, frame=frame, verbose=verbose)
        if doplot:
            plot_horn_and_FP(q, x, y, Cminus_j[:, 0], frame=frame,
                             title='$C_{-j}$' + f' - Horn {self.baseline[1]} close', **kwargs)

        # Only i open (not a realistic observable)
        q.horn.open = False
        q.horn.open[self.baseline[0] - 1] = True
        _, _, Ci = get_response_power(q, theta, phi, nu, spectral_irradiance, frame=frame, verbose=verbose)
        if doplot:
            plot_horn_and_FP(q, x, y, Ci[:, 0], frame=frame,
                             title='$C_i$' + f' - Only horn {self.baseline[0]} open', **kwargs)

        # Only j open (not a realistic observable)
        q.horn.open[self.baseline[0] - 1] = False
        q.horn.open[self.baseline[1] - 1] = True
        _, _, Cj = get_response_power(q, theta, phi, nu, spectral_irradiance, frame=frame, verbose=verbose)
        if doplot:
            plot_horn_and_FP(q, x, y, Cj[:, 0], frame=frame,
                             title='$C_j$' + f' - Only horn {self.baseline[1]} open', **kwargs)

        # Only baseline [i, j] open (not a realistic observable)
        q.horn.open[self.baseline[0] - 1] = True
        _, _, Sij = get_response_power(q, theta, phi, nu, spectral_irradiance, frame=frame, verbose=verbose)
        if doplot:
            plot_horn_and_FP(q, x, y, Sij[:, 0], frame=frame,
                             title='$S_{ij}$' + f' - Only baseline {self.baseline} open', **kwargs)

        return x, y, S, Cminus_i, Sminus_ij, Cminus_j, Ci, Cj, Sij

    def get_fringes_from_combination(self, q, measured_comb=True,
                                     theta=np.array([0.]), phi=np.array([0.]),
                                     nu=150e9, spectral_irradiance=1.,
                                     frame='ONAFP',
                                     doplot=True, verbose=True, **kwargs):
        """
        Return the fringes on the FP by making the computation
        fringes =(S_tot - Cminus_i - Cminus_j + Sminus_ij)
        q : a qubic monochromatic instrument
        """

        x, y, S_tot, Cminus_i, Sminus_ij, Cminus_j, Ci, Cj, Sij = \
            Fringes.get_all_combinations_power(self, q,
                                               theta=theta, phi=phi,
                                               nu=nu, spectral_irradiance=spectral_irradiance,
                                               frame=frame,
                                               doplot=False, verbose=verbose, **kwargs)
        if measured_comb:
            fringes_comb = S_tot - Cminus_i - Cminus_j + Sminus_ij
        else:
            fringes_comb = S_tot - Cminus_i - Cminus_j + Sminus_ij + Ci + Cj

        if doplot:
            nptg = np.shape(theta)[0]
            for i in range(nptg):
                plot_horn_and_FP(q, x, y, fringes_comb[:, 0], frame=frame,
                                 title='Baseline {} - Theta={}deg - Phi={}deg'.format(self.baseline,
                                                                                      np.rad2deg(theta[i]),
                                                                                      np.rad2deg(phi[i])), **kwargs)
        return x, y, fringes_comb

    def get_fringes_Maynooth(self, q, rep,
                             theta=np.array([0.]), nu=150e9,
                             interp=False,
                             verbose=True):
        if q.config != 'TD':
            raise ValueError('Maynooth simulations are for the TD only.')

        xONAFP, yONAFP, power = get_power_Maynooth(self.baseline, theta, nu, q.horn.center, rep, verbose=verbose)

        # TES centers and TES vertex in the ONAFP frame
        xONAFP_TES, yONAFP_TES, vONAFP_TES = get_TEScoordinates_ONAFP(q)

        powerTES = fullreso2TESreso(xONAFP, yONAFP, power,
                                    vONAFP_TES, q.detector.area,
                                    interp=interp,
                                    verbose=verbose)

        # power_TES *= q.filter.bandwidth  # W/Hz to W

        return xONAFP_TES, yONAFP_TES, powerTES

    def get_fringes_Maynooth_combination(self, q, rep,
                                         measured_comb=True,
                                         theta=np.array([0.]), nu=150e9,
                                         interp=False,
                                         verbose=True):
        i = self.baseline[0]
        j = self.baseline[1]
        all_open = np.arange(1, 65)
        first_close = np.delete(all_open, i - 1)
        second_close = np.delete(all_open, j - 1)
        both_close = np.delete(all_open, [i - 1, j - 1])
        xONAFP, yONAFP, S_tot = get_power_Maynooth(all_open, theta, nu, q.horn.center, rep, verbose=verbose)
        _, _, Cminus_i = get_power_Maynooth(first_close, theta, nu, q.horn.center, rep, verbose=verbose)
        _, _, Cminus_j = get_power_Maynooth(second_close, theta, nu, q.horn.center, rep, verbose=verbose)
        _, _, Sminus_ij = get_power_Maynooth(both_close, theta, nu, q.horn.center, rep, verbose=verbose)

        if measured_comb:
            fringes_comb = S_tot - Cminus_i - Cminus_j + Sminus_ij
        else:
            _, _, Ci = get_power_Maynooth([i - 1], theta, nu, q.horn.center, rep, verbose=verbose)
            _, _, Cj = get_power_Maynooth([j - 1], theta, nu, q.horn.center, rep, verbose=verbose)
            fringes_comb = S_tot - Cminus_i - Cminus_j + Sminus_ij + Ci + Cj

        # TES centers and TES vertex in the ONAFP frame
        xONAFP_TES, yONAFP_TES, vONAFP_TES = get_TEScoordinates_ONAFP(q)

        fringes_comb_TES = fullreso2TESreso(xONAFP, yONAFP, fringes_comb,
                                            vONAFP_TES, q.detector.area,
                                            interp=interp,
                                            verbose=verbose)

        return xONAFP_TES, yONAFP_TES, fringes_comb_TES


# ========== Old functions =============
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

