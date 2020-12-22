from __future__ import division, print_function

import glob
import numpy as np
import pandas as pd

from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import dblquad

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from qubicpack.pixel_translation import tes2index

__all__ = ['Model_Fringes_QubicSoft', 'Model_Fringes_Maynooth']


# ========== Plot functions =============
def plot_horns(q, simple=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    if simple:
        xhorns = q.horn.center[:, 0]
        yhorns = q.horn.center[:, 1]
        ax.plot(xhorns, yhorns, 'ro')
    else:
        q.horn.plot()
    ax.set_xlabel('X_GRF [m]', fontsize=14)
    ax.set_ylabel('Y_GRF [m]', fontsize=14)
    ax.axis('square')
    return


def plot_baseline(q, bs, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    hcenters = q.horn.center[:, 0:2]
    ax.plot(hcenters[np.array(bs) - 1, 0], hcenters[np.array(bs) - 1, 1], lw=4, label=bs)
    return


def scatter_plot_FP(q, x, y, FP_signal, frame, fig=None, ax=None,
                    s=None, title=None, unit='[W / Hz]', cbar=True, **kwargs):
    """
    Make a scatter plot of the focal plane.
    Parameters
    ----------
    q: QubicInstrument()
    x: array
        X coordinates for the TES.
    y: array
        Y coordinates for the TES.
    FP_signal: array
        Signal to plot in 1D, should be ordered as x and y
    frame: str
        'GRF' or 'ONAFP', the frame used for x and y
    s: int
        Marker size on the plot
    title: str
        Plot title
    unit: str
        Unit of the signal to plot.
    kwargs: any kwarg for plt.scatter()

    """
    if fig is None:
        fig, ax = plt.subplots()
    if s is None:
        if q.config == 'TD':
            s = ((fig.get_figwidth() / 35 * fig.dpi) ** 2)
        else:
            s = ((fig.get_figwidth() / 70 * fig.dpi) ** 2)
    img = ax.scatter(x, y, c=FP_signal, marker='s', s=s, **kwargs)
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        clb = fig.colorbar(img, cax=cax)
        clb.ax.set_title(unit)
    ax.set_xlabel(f'X_{frame} [m]', fontsize=14)
    ax.set_ylabel(f'Y_{frame} [m]', fontsize=14)
    ax.axis('square')
    ax.set_title(title, fontsize=14)
    return


def pcolor_plot_FP(q, x, y, FP_signal, frame, title=None, unit='[W / Hz]', **kwargs):
    """
    Make a pcolor plot of the focal plane.
    !!! x, y, FP_signal must be ordered as defined in q.detector.
    Parameters
    ----------
    q: QubicInstrument()
    x: array
        X coordinates for the TES.
    y: array
        Y coordinates for the TES.
    FP_signal: array
        Signal to plot in 1D, should be ordered as x and y
    frame: str
        'GRF' or 'ONAFP', the frame used for x and y
    title: str
        Plot title
    unit: str
        Unit of the signal to plot.
    kwargs: any kwarg for plt.pcolor()

    """
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
    return


def plot_horn_and_FP(q, x, y, FP_signal, frame, s=None, title=None, unit='[W / Hz]', **kwargs):
    """
    Plot the horn array in GRF and a scatter plot of the focal plane in GRF or ONAFP.
    See scatter_plot_FP()
    """
    fig, axs = plt.subplots(1, 2)
    ax0, ax1 = np.ravel(axs)
    fig.suptitle(title, fontsize=18)
    fig.subplots_adjust(wspace=0.3)

    plot_horns(q, ax=ax0)

    scatter_plot_FP(q, x, y, FP_signal, frame, fig=fig, ax=ax1, marker='s', s=s, **kwargs)
    return


def plot_BLs_eq(allBLs, BLs_sort, q, simple=True, figsize=(12, 6)):
    """
    Plot the horn array and the observed baselines for each type (class of equivalence).
    Parameters
    ----------
    allBLs: list
        List containing all the baselines in the dataset.
    BLs_sort: list
        Indices of the baselines in the dataset for each type
    q: QubicInstrument
    """
    nclass_eq = len(BLs_sort)

    fig, axs = plt.subplots(1, nclass_eq, figsize=figsize)
    axs = np.ravel(axs)
    for i in range(nclass_eq):
        ax = axs[i]
        dataset_eq = BLs_sort[i]
        plot_horns(q, simple=simple, ax=ax)
        ax.set_title(f'Type {i}', fontsize=14)
        print(f'Type {i}:')
        for j in dataset_eq:
            print(f'  - {allBLs[j]}')
            plot_baseline(q, allBLs[j], ax=ax)
        ax.legend()
    return

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
    """
    Get coordinates of the TES, x and y for the centers
    and the 4 corners (vertex) in the ONAFP frame.
    """
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

def get_horn_coordinates_ONAFP(q):
    """
    Get coordinates of the horn center (x, y, z),
    in the ONAFP frame.
    """
    # Horn centers in the ONAFP frame
    center_GRF = q.horn.center
    center_ONAFP = np.zeros_like(center_GRF)
    center_ONAFP[:, 0] = - center_GRF[:, 1] # xONAFP = -yGRF
    center_ONAFP[:, 1] = center_GRF[:, 0]   # yONAFP = xGRF
    center_ONAFP[:, 2] = center_GRF[:, 2]   # zONAFP = zGRF

    return center_ONAFP


def TES_Instru2coord(TES, ASIC, q, frame='ONAFP', verbose=True):
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
    x, y: TES center coordinates.
    FP_index: Focal Plane index, as used in Qubic soft.
    index_q: position index of the FP_index in q.detector.index()

    """
    if TES in [4, 36, 68, 100]:
        raise ValueError('This is a thermometer !')
    FP_index = tes2index(TES, ASIC)
    if verbose:
        print('FP_index =', FP_index)

    index_q = np.where(q.detector.index == FP_index)[0][0]
    if verbose:
        print('Index_q =', index_q)

    centerGRF = q.detector.center[q.detector.index == FP_index][0]
    xGRF = centerGRF[0]
    yGRF = centerGRF[1]

    if frame not in ['GRF', 'ONAFP']:
         raise ValueError('The frame is not valid.')
    elif frame == 'GRF':
        if verbose:
            print('X_GRF = {:.3f} mm, Y_GRF = {:.3f} mm'.format(xGRF * 1e3, yGRF * 1e3))
        return xGRF, yGRF, FP_index, index_q
    elif frame == 'ONAFP':
        xONAFP = - yGRF
        yONAFP = xGRF
        if verbose:
            print('X_ONAFP = {:.3f} mm, Y_ONAFP = {:.3f} mm'.format(xONAFP * 1e3, yONAFP * 1e3))
        return xONAFP, yONAFP, FP_index, index_q


def get_TES_Instru_coords(q, frame='ONAFP', verbose=True):
    """
    Same as TES_Instru2coord() but loop on all TES.
    """
    thermos = [4, 36, 68, 100]
    if q.config == 'TD':
        nASICS = 2
    else:
        nASICS = 8

    nTES = nASICS * 128
    x = np.zeros(nTES)
    y = np.zeros(nTES)
    FP_index = np.zeros(nTES, dtype=int)
    index_q = np.zeros(nTES, dtype=int)

    for ASIC in range(1, nASICS + 1):
        for TES in range(1, 129):
            if verbose:
                print(f'\n ASIC {ASIC} - TES {TES}')
            if TES not in thermos:
                i = (TES - 1) + 128 * (ASIC - 1)
                x[i], y[i], FP_index[i], index_q[i]= TES_Instru2coord(TES, ASIC, q, frame=frame, verbose=verbose)
            else:
                if verbose:
                    print('Thermometer !')

    return x, y, FP_index, index_q


def get_TESvertices_FromMaynoothFiles(rep, ndet=992):
    """
    Get TES vertices coordinates from Maynooth files.
    Not very useful because `q.detector.vertex` gives the same.
    Parameters
    ----------
    rep : str
        Path of the repository for the simulated files, can be download at :
        https://drive.google.com/open?id=19dPHw_CeuFZ068b-VRT7N-LWzOL1fmfG
    ndet : int
        Number of TES.
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


def give_bs_pars(q, bs, frame='GRF'):
    """Find orientation angle and length for a baseline."""

    # X, Y coordinates of the 2 horns in GRF or ONAFP.
    if frame == 'ONAFP':
        hc = get_horn_coordinates_ONAFP(q)
        hc = hc[:, :2]
    elif frame == 'GRF':
        hc = q.horn.center[:, :2]
    hc0 = hc[bs[0] - 1, :]
    hc1 = hc[bs[1] - 1, :]

    bsxy = hc1 - hc0
    theta = np.degrees(np.arctan2(bsxy[1], bsxy[0]))
    length = np.sqrt(np.sum(bsxy ** 2))
    xycenter = (hc0 + hc1) / 2.
    return theta, length, xycenter


def check_equiv(vecbs1, vecbs2, tol=1e-5):
    """Check if 2 baselines are equivalent."""
    norm1 = np.dot(vecbs1, vecbs1)
    norm2 = np.dot(vecbs2, vecbs2)
    cross12 = np.cross(vecbs1, vecbs2)
    if (np.abs(norm1 - norm2) < tol) & (np.abs(cross12) < tol):
        return True
    else:
        return False


def find_equivalent_baselines(all_bs, q):
    """
    Find the equivalent baselines in a list.
    Parameters
    ----------
    all_bs: list
        List of baselines.
    q: QubicInstrument

    Returns
    -------
    BLs_sort: List with baseline indices sorted according to equivalence.
        ex: BLs_sort = [[1, 3], [0, 2, 4]] means that you have 2 different classes
        of equivalence with 2 and 3 baselines respectively.
    all_eqtype: List of integers with the type of each baseline.
        ex: In the example above, you have 2 types (0 or 1)
        so you will get all_eqtype = [1, 0, 1, 0, 1]

    """
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


# ========== Compute power on the focal plane =============
def make_external_A(rep, open_horns):
    """
    Compute external_A from simulated files with aberrations.
    This can be used in get_response_power method that returns the synthetic beam on the sky
    or in get_response() to have the signal on the focal plane.
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
    Compute power on the focal plane in the ONAFP frame for one position of the source
    with respect to the instrument.

    Parameters
    ----------
    q: a qubic monochromatic instrument
    theta: float
        The source zenith angle [rad].
    phi: float
        The source azimuthal angle [rad].
    nu: float
        Source frequency in Hz.
    spectral_irradiance : array-like
        The source spectral_irradiance [W/m^2/Hz].
    frame: str
        Referential frame you want to use: 'GRF' or 'ONAFP'
    external_A: list of tables describing the phase and amplitude at
    each point of the focal plane for each of the horns, see make_external_A()
    hwp_position : int
        HWP position from 0 to 7.

    Returns
    ----------
    x, y: 1D arrays with the coordinates on the focal plane in GRF or ONAFP.
    power: array with the power on the focal plane for each posiion (x, y) and each pointing.
    """
    if frame not in ['GRF', 'ONAFP']:
        raise ValueError('The frame is not valid. It must be GRF or ONAFP.')

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
        print('Detector centers shape:', q.detector.center.shape)
        print('Power shape:', power.shape)
        print('X_GRF shape:', xGRF.shape)

    if frame == 'GRF':
        x = xGRF
        y = yGRF
    else:
        # Make a pi/2 rotation from GRF -> ONAFP referential frame
        xONAFP = - yGRF
        yONAFP = xGRF
        x = xONAFP
        y = yONAFP
    return x, y, power


def get_power_Maynooth(rep, open_horns, theta, nu, horn_center, hwp_position=0, verbose=True):
    """
    Get power on the focal plane from Maynooth simulations.
    Parameters
    ----------
    rep: str
        Repository with the simulation files.
    open_horns: list
        List of open horns.
    theta: float
        The source zenith angle [rad].
    nu: float
        Frequency of the calibration source [Hz]
    horn_center: array
        Coordinates of the horns.
    hwp_position: int
        HWP position from 0 to 7.
    verbose: bool

    Returns
    -------
    (x, y) coordinates on the focal plane in the ONAFP frame and the power at each coordinate.

    """
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
        additional_phase = - 2 * np.pi / 3e8 * nu * dist * np.sin(theta)

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
    """
    Decrease the resolution to have the power in each TES.
    Parameters
    ----------
    x, y: array
        Coordinates on the FP
    power: array
        Power on the FP.
    TESvertex: array
        Coordinates of the 4 corners of each TES.
    TESarea: float
        Area of the detectors [m²]
    interp: bool
        If True, interpolate and integrate in each TES (takes time).
        If False, make the mean in each TES (faster).
    verbose: bool

    Returns
    -------
    The power in each TES.
    """
    ndet = np.shape(TESvertex)[0]
    powerTES = np.zeros(ndet)
    print('ndet:', ndet)

    if interp:
        print('********** Begin interpolation **********')
        reso = int(np.sqrt(x.shape[0]))
        print('Reso:', reso)
        power_interp = RegularGridInterpolator((np.unique(x),
                                                np.unique(y)),
                                               power.reshape((reso, reso)),
                                               method='linear',
                                               bounds_error=False,
                                               fill_value=0.)
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
class Model_Fringes_Ana:
    def __init__(self, q, baseline, theta_source=0., nu_source=150e9, fwhm=20., amp=1., frame='ONAFP'):
        """

        Parameters
        ----------
        q: QubicInstrument
        baseline: list
            Baseline formed with 2 horns, index between 1 and 64 as on the instrument.
        theta_source: float
            The source zenith angle [rad].
        nu_source: float
            Source frequency [Hz].
        fwhm: float
        amp: float
            Global amplitude for the fringes.
        """
        self.BL = baseline
        self.q = q
        self.focal = q.optics.focal_length
        self.theta_source = theta_source
        self.nu_source = nu_source
        self.lam = 3e8 / self.nu_source
        self.fwhm = fwhm
        self.amp = amp
        self.frame = frame

        # Detector centers
        if self.frame == 'ONAFP':
            xONAFP, yONAFP, _ = get_TEScoordinates_ONAFP(self.q)
            self.x = xONAFP
            self.y = yONAFP
        elif self.frame == 'GRF':
            self.x = self.q.detector.center[:, 0]
            self.y = self.q.detector.center[:, 1]

        # Angle and length of the baseline:
        BL_angle, BL_length, BL_center = give_bs_pars(self.q, self.BL, frame='ONAFP')
        self.BL_angle = np.deg2rad(BL_angle)
        self.BL_length = BL_length
        self.BL_xc = BL_center[0]
        self.BL_yc = BL_center[1]

        # Additional phase
        dist = np.sqrt(self.BL_xc ** 2 + self.BL_yc ** 2)
        phase = - 2 * np.pi / 3e8 * self.nu_source * dist * np.sin(self.theta_source)
        self.phase = phase

    def get_fringes(self, times_gaussian=True):
        if times_gaussian:
            sigma = np.deg2rad(self.fwhm / 2.355 * self.focal)
            gaussian = np.exp(- 0.5 * ((self.x - self.BL_xc) ** 2 + (self.y - self.BL_yc) ** 2) / sigma ** 2)
        else:
            gaussian = 1.
        xprime = (self.x * np.cos(self.BL_angle) + self.y * np.sin(self.BL_angle))
        interfrange = self.lam * self.focal / self.BL_length
        self.fringes = self.amp * np.cos((2. * np.pi / interfrange * xprime) + self.phase) * gaussian

        return self.x, self.y, self.fringes


class Model_Fringes_QubicSoft:
    def __init__(self, q, baseline,
                 theta_source=0.,
                 phi_source=0.,
                 nu_source=150e9,
                 spec_irrad_source=1.,
                 frame='ONAFP',
                 external_A=None,
                 hwp_position=0):
        """
        Parameters
        ----------
        q: QubicInstrument
        baseline: list
            Baseline formed with 2 horns, index between 1 and 64 as on the instrument.
        theta_source: float
            The source zenith angle [rad].
        phi_source: float
            The source azimuthal angle [rad].
        nu_source: float
            Source frequency [Hz].
        spec_irrad: array-like
            The source spectral_irradiance [W/m^2/Hz].
        frame: str
            'GRF' or 'ONAFP'.
        """
        self.q = q
        self.baseline = baseline
        self.theta_source = theta_source
        self.phi_source = phi_source
        self.nu_source = nu_source
        self.spec_irrad_source = spec_irrad_source
        self.frame = frame
        self.external_A = external_A
        self.hwp_position = hwp_position

    def get_fringes(self, doplot=True, verbose=True, **kwargs):
        """
        Compute the fringes on the focal plane directly opening the baseline.
        see get_response_power() for the arguments.
        Returns (x, y) coordinates and the power.
        """
        open_switches(self.q, self.baseline)

        self.x, self.y, self.fringes = get_response_power(self.q, self.theta_source, self.phi_source, self.nu_source,
                                           self.spec_irrad_source,
                                           frame=self.frame,
                                           external_A=self.external_A,
                                           hwp_position=self.hwp_position,
                                           verbose=verbose)

        if doplot:
            plot_horn_and_FP(self.q, self.x, self.y, self.fringes, frame=self.frame,
                             title='Baseline {} - Theta={}deg - Phi={}deg'.format(self.baseline,
                                                                                  np.rad2deg(self.theta_source),
                                                                                  np.rad2deg(self.phi_source)),
                             **kwargs)
        return self.x, self.y, self.fringes

    def get_all_combinations_power(self, doplot=True, verbose=True, **kwargs):
        """
            Returns the power on the focal plane at each pointing (each position of the source),
            for different configurations of the horn array: all open, all open except i, except j,
            except i and j, only i open, only j open, only i and j open.

        Returns
        -------
        x, y: The coordinates on the FP.
        S, Cminus_i, Cminus_j, Sminus_ij, Ci, Cj, Sij : arrays
            Power on the focal plane for each configuration, at each pointing.

        """

        self.q.horn.open = True

        # All open
        self.x, self.y, S = get_response_power(self.q, self.theta_source, self.phi_source, self.nu_source,
                                     self.spec_irrad_source, frame=self.frame, verbose=verbose)
        if doplot:
            plot_horn_and_FP(self.q, self.x, self.y, S, frame=self.frame, title='$S$ - All open', **kwargs)

        # All open except i
        self.q.horn.open[self.baseline[0] - 1] = False
        _, _, Cminus_i = get_response_power(self.q, self.theta_source, self.phi_source, self.nu_source,
                                            self.spec_irrad_source, frame=self.frame, verbose=verbose)
        if doplot:
            plot_horn_and_FP(self.q, self.x, self.y, Cminus_i, frame=self.frame,
                             title='$C_{-i}$' + f' - Horn {self.baseline[0]} close', **kwargs)

        # All open except baseline [i, j]
        self.q.horn.open[self.baseline[1] - 1] = False
        _, _, Sminus_ij = get_response_power(self.q, self.theta_source, self.phi_source, self.nu_source,
                                             self.spec_irrad_source, frame=self.frame, verbose=verbose)
        if doplot:
            plot_horn_and_FP(self.q, self.x, self.y, Sminus_ij, frame=self.frame,
                             title='$S_{-ij}$' + f' - Baseline {self.baseline} close', **kwargs)

        # All open except j
        self.q.horn.open[self.baseline[0] - 1] = True
        _, _, Cminus_j = get_response_power(self.q, self.theta_source, self.phi_source, self.nu_source,
                                            self.spec_irrad_source, frame=self.frame, verbose=verbose)
        if doplot:
            plot_horn_and_FP(self.q, self.x, self.y, Cminus_j, frame=self.frame,
                             title='$C_{-j}$' + f' - Horn {self.baseline[1]} close', **kwargs)

        # Only i open (not a realistic observable)
        self.q.horn.open = False
        self.q.horn.open[self.baseline[0] - 1] = True
        _, _, Ci = get_response_power(self.q, self.theta_source, self.phi_source, self.nu_source,
                                      self.spec_irrad_source, frame=self.frame, verbose=verbose)
        if doplot:
            plot_horn_and_FP(self.q, self.x, self.y, Ci, frame=self.frame,
                             title='$C_i$' + f' - Only horn {self.baseline[0]} open', **kwargs)

        # Only j open (not a realistic observable)
        self.q.horn.open[self.baseline[0] - 1] = False
        self.q.horn.open[self.baseline[1] - 1] = True
        _, _, Cj = get_response_power(self.q, self.theta_source, self.phi_source, self.nu_source,
                                      self.spec_irrad_source, frame=self.frame, verbose=verbose)
        if doplot:
            plot_horn_and_FP(self.q, self.x, self.y, Cj, frame=self.frame,
                             title='$C_j$' + f' - Only horn {self.baseline[1]} open', **kwargs)

        # Only baseline [i, j] open (not a realistic observable)
        self.q.horn.open[self.baseline[0] - 1] = True
        _, _, Sij = get_response_power(self.q, self.theta_source, self.phi_source, self.nu_source,
                                       self.spec_irrad_source, frame=self.frame, verbose=verbose)
        if doplot:
            plot_horn_and_FP(self.q, self.x, self.y, Sij, frame=self.frame,
                             title='$S_{ij}$' + f' - Only baseline {self.baseline} open', **kwargs)

        return self.x, self.y, S, Cminus_i, Sminus_ij, Cminus_j, Ci, Cj, Sij

    def get_fringes_from_combination(self, measured_comb=True, doplot=True, verbose=True, **kwargs):
        """
        Return the fringes on the FP by making the combination

        Returns
        -------
        x, y: The coordinates on the FP.
        fringes : Fringes on the FP, for each coordinate, at each pointing.
        """

        self.x, self.y, S_tot, Cminus_i, Sminus_ij, Cminus_j, Ci, Cj, Sij = \
            self.get_all_combinations_power(doplot=doplot, verbose=verbose, **kwargs)
        if measured_comb:
            self.fringes_comb = S_tot - Cminus_i - Cminus_j + Sminus_ij
        else:
            self.fringes_comb = S_tot - Cminus_i - Cminus_j + Sminus_ij + Ci + Cj

        if doplot:
            plot_horn_and_FP(self.q, self.x, self.y, self.fringes_comb, frame=self.frame,
                             title='Baseline {} - Theta={}deg - Phi={}deg'.format(self.baseline,
                                                                                  np.rad2deg(self.theta_source),
                                                                                  np.rad2deg(self.phi_source)),
                             **kwargs)
        return self.x, self.y, self.fringes_comb


class Model_Fringes_Maynooth:
    def __init__(self, q, baseline, rep,
                 theta_source=0., nu_source=150e9,
                 frame='ONAFP', interp=False):
        """
        Parameters
        ----------
        q: QubicInstrument
        baseline: list
            Baseline formed with 2 horns, index between 1 and 64 as on the instrument.
        rep: str
            Repository with the simulation files.
        theta: float
            The source zenith angle [rad].
        nu: float
            Frequency of the calibration source [Hz]
        frame: str
            'GRF' or 'ONAFP'.
        interp: bool
            If True, interpolate and integrate in each TES (takes time).
            If False, make the mean in each TES (faster).
        """
        self.q = q
        self.baseline = baseline
        self.rep = rep
        self.theta_source = theta_source
        self.nu_source = nu_source
        self.frame = frame
        self.interp = interp

    def get_fringes(self, verbose=True):
        """
        Compute fringes on the focal plane from Maynooth simulations.
        Returns
        -------
        x, y coordinates on the FP in the ONAFP frame and the corresponding power.

        """
        if self.q.config != 'TD':
            raise ValueError('Maynooth simulations are for the TD only.')

        xONAFP, yONAFP, fringes_fullreso = get_power_Maynooth(self.rep, self.baseline,
                                                   self.theta_source, self.nu_source,
                                                   self.q.horn.center, verbose=verbose)

        # TES centers and TES vertex in the ONAFP frame
        xONAFP_TES, yONAFP_TES, vONAFP_TES = get_TEScoordinates_ONAFP(self.q)

        self.fringes = fullreso2TESreso(xONAFP, yONAFP, fringes_fullreso,
                                        vONAFP_TES, self.q.detector.area,
                                        interp=self.interp,
                                        verbose=verbose)

        # power_TES *= q.filter.bandwidth  # W/Hz to W

        if self.frame == 'ONAFP':
            self.x = xONAFP_TES
            self.y = yONAFP_TES
        elif self.frame == 'GRF':
            self.x = yONAFP_TES
            self.y = - xONAFP_TES
        return self.x, self.y, self.fringes

    def get_fringes_from_combination(self, measured_comb=True, verbose=True):
        """
        Compute fringes on the focal plane from Maynooth simulations doing the combination with
        S_tot, Cminus_i, Cminus_j, Sminus_ij, Ci and Cj.
        Parameters
        ----------
        measured_comb: bool
            If True, returns the measured combination: S_tot - Cminus_i - Cminus_j + Sminus_ij.
            If False, returns the complete combination: S_tot - Cminus_i - Cminus_j + Sminus_ij + Ci + Cj.

        verbose: bool

        Returns
        -------
        x, y coordinates on the FP in the ONAFP frame and the corresponding power.
        """

        i = self.baseline[0]
        j = self.baseline[1]
        all_open = np.arange(1, 65)
        first_close = np.delete(all_open, i - 1)
        second_close = np.delete(all_open, j - 1)
        both_close = np.delete(all_open, [i - 1, j - 1])
        xONAFP, yONAFP, S_tot = get_power_Maynooth(self.rep, all_open, self.theta_source, self.nu_source,
                                                   self.q.horn.center, verbose=verbose)
        _, _, Cminus_i = get_power_Maynooth(self.rep, first_close, self.theta_source, self.nu_source,
                                            self.q.horn.center, verbose=verbose)
        _, _, Cminus_j = get_power_Maynooth(self.rep, second_close, self.theta_source, self.nu_source,
                                            self.q.horn.center, verbose=verbose)
        _, _, Sminus_ij = get_power_Maynooth(self.rep, both_close, self.theta_source, self.nu_source,
                                             self.q.horn.center, verbose=verbose)

        if measured_comb:
            fringes_comb_fullreso = S_tot - Cminus_i - Cminus_j + Sminus_ij
        else:
            _, _, Ci = get_power_Maynooth(self.rep, [i - 1], self.theta_source, self.nu_source,
                                          self.q.horn.center, verbose=verbose)
            _, _, Cj = get_power_Maynooth(self.rep, [j - 1], self.theta_source, self.nu_source,
                                          self.q.horn.center, verbose=verbose)
            fringes_comb_fullreso = S_tot - Cminus_i - Cminus_j + Sminus_ij + Ci + Cj

        # TES centers and TES vertex in the ONAFP frame
        xONAFP_TES, yONAFP_TES, vONAFP_TES = get_TEScoordinates_ONAFP(self.q)

        self.fringes_comb = fullreso2TESreso(xONAFP, yONAFP, fringes_comb_fullreso,
                                            vONAFP_TES, self.q.detector.area,
                                            interp=self.interp,
                                            verbose=verbose)
        if self.frame == 'ONAFP':
            self.x = xONAFP_TES
            self.y = yONAFP_TES
        elif self.frame == 'GRF':
            self.x = yONAFP_TES
            self.y = - xONAFP_TES

        return self.x, self.y, self.fringes_comb
