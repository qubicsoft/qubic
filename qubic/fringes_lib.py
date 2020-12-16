from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits as pyfits
from matplotlib.backends.backend_pdf import PdfPages
import scipy.optimize as sop

from qubicpack.qubicfp import qubicfp
import qubic.fibtools as ft
from qubic import selfcal_lib as scal


# ============== Get data ==============
def get_data(datafolder, ASIC, TES=28, doplot=True, src_data=False, 
    subtract_t0=True):
    """
    Get the TODs for one ASIC.
    Parameters
    ----------
    datafolder: str
        Folder containing the data.
    ASIC: int
        ASIC number.
    TES: int
        TES number as defined on the instrument (for the plot only)
    doplot: bool
    src_data: if True,will return the srouce data as well
    subtract_t0: if True, will remove to the time the first time element

    Returns
    -------
    Time and signal for all TES in one ASIC. If src_data is True: will also return the  source time and signal
    """
    ASIC = int(ASIC)

    # Qubicpack object
    a = qubicfp()
    a.verbosity = 0
    a.read_qubicstudio_dataset(datafolder)

    # Data form the object
    data  = a.timeline_array(asic=ASIC)
    t_data = a.timeaxis(datatype='science',asic=ASIC)
    t0 = t_data[0]
    if subtract_t0:
        t_data -= t0
    if src_data:
        t_src = a.calsource()[0]
        if subtract_t0:
            t_src -= t0
        d_src = a.calsource()[1]

    if doplot:
        fig, axs = plt.subplots(1, 2, figsize=(15, 3))
        plt.subplots_adjust(wspace=0.5)

        axs[0].plot(t_data, data[TES - 1, :])
        axs[0].set_title(datafolder[-5:])

        axs[1].plot(t_data, data[TES - 1, :])
        axs[1].set_title(datafolder[-5:])
        axs[1].set_xlim(0, 40)
        plt.show()

    if src_data:
        return t_data, data, t_src, d_src
    else:
        return t_data, data

def cut_data(t0, tf, t_data, data):
    """
    Cut the TODs from t0 to tf.
    They can be None if you do not want to cut the beginning or the end.
    """
    if t0 is None:
        t0 = t_data[0]
    if tf is None:
        tf = t_data[-1]

    ok = (t_data >= t0) & (t_data <= tf)
    t_data_cut = t_data[ok]
    data_cut = data[:, ok]

    return t_data_cut, data_cut


def cut_data_Nperiods(t0, tf, t_data, data, period):
    """
    Cut the TODs from t0 to tf with an integer number of periods
    They can be None if you do not want to cut the beginning or the end.
    """
    if t0 is None:
        t0 = t_data[0]
    if tf is None:
        tf = t_data[-1]

    nper = np.floor((tf - t0) / period).astype(int)
    tend = t0 + nper * period
    ok = (t_data >= t0) & (t_data <= tend)
    t_data_cut = t_data[ok]
    data_cut = data[:, ok]

    return t_data_cut, data_cut, nper


def find_right_period(guess, t_data, data_oneTES, delta=1.5, nb=250):
    ppp = np.linspace(guess - delta, guess + delta, nb)
    rms = np.zeros(len(ppp))
    for i in range(len(ppp)):
        xin = t_data % ppp[i]
        yin = data_oneTES
        xx, yy, dx, dy, o = ft.profile(xin, yin, nbins=100, plot=False)
        rms[i] = np.std(yy)
    period = ppp[np.argmax(rms)]

    return ppp, rms, period


def make_diff_sig(params, t, wt, data):
    """
    Make the difference between the TODs and the simulation.
    Parameters
    ----------
    params: list
        ctime, starting time, the 6 amplitudes.
    t : array
        Time sampling.
    wt: float
        Waiting time [s], number of second the signal keep constant.
    data: array with TODs

    """

    thesim = ft.simsig_fringes(t, wt, params)
    diff = data - thesim
    return diff


def make_combination(param_est, verbose=0):
    """ Make the combination to get the fringes:
        S_tot - Cminus_i - Cminus_j + Sminus_ij using the amplitudes found in the fit."""
    amps = param_est[2:8]
    if verbose>0:
        print('Check:', amps[2], amps[4])
    return (amps[0]+amps[3]+amps[5])/3 + amps[2] - amps[1] - amps[4]


def weighted_sum(vals, errs, coeffs):
    thesum = np.sum(coeffs * vals)
    thesigma = np.sqrt(np.sum(coeffs**2 * errs**2))
    return thesum, thesigma


def analyse_fringesLouise(datafolder, t0=None, tf=None, wt=5.,
                          lowcut=0.001, highcut=10., nbins=120,
                          notch=np.array([[1.724, 0.005, 10]]),
                          tes_check=28, param_guess=[0.1, 0., 1, 1, 1, 1, 1, 1],
                          median=False, read_data=None, verbose=True, ):
    """
    Parameters
    ----------
    datafolder: str
        Folder containing the data.
    t0, tf: float
        Start and end time to cut the TODs.
    wt: float
        Waiting time [s] on each step.
    lowcut, highcut: float
        Low and high cut for filtering
    nbins: int
        Number of bins for filtering
    notch: array
        Defined a notch filter.
    tes_check: int
        One TES to check the period.
    param_guess: list
        ctime, starting time, the 6 amplitudes.
    median: bool
        Parameter for folding.
    read_data: array
        If it is None, it will read the data,
        else, it will use the one you pass here (saves time).
    verbose: bool
    Returns
    -------
    Time, folded signal, the 8 parameters estimated with the fit,
    the combination of the amplitudes, the period and the residuals
    between the fit and the signal. They are computed for each TES.
    """

    combination = np.zeros(256)
    param_est = np.zeros((256, 8))
    folded_bothasics = np.zeros((256, nbins))
    residuals = np.zeros_like(folded_bothasics)

    for ASIC in [1, 2]:
        if read_data is None:
            # Read the data
            t_data, data = get_data(datafolder, ASIC, doplot=False)
        else:
            t_data, data = read_data[ASIC - 1]

        # Cut the data
        t_data_cut, data_cut = cut_data(t0, tf, t_data, data)

        # Find the true period
        if ASIC == 1:
            ppp, rms, period = find_right_period(6 * wt, t_data_cut, data_cut[tes_check - 1, :])
            if verbose:
                print('period:', period)
                print('Expected : ', 6 * wt)

        # Fold and filter the data
        folded, t, folded_nonorm, newdata = ft.fold_data(t_data_cut,
                                                         data_cut,
                                                         period,
                                                         lowcut,
                                                         highcut,
                                                         nbins,
                                                         notch=notch,
                                                         median=median,
                                                         silent=verbose,
                                                         )
        if ASIC == 1:
            folded_bothasics[:128, :] = folded
        else:
            folded_bothasics[128:, :] = folded

        # Fit (Louise method)
        for TES in range(1, 129):
            index = (TES - 1) + 128 * (ASIC - 1)
            fit = sop.least_squares(make_diff_sig,
                                    param_guess,
                                    args=(t,
                                          period / 6.,
                                          folded[TES - 1, :]),
                                    bounds=([0., -2, -2, -2, -2, -2, -2, -2],
                                            [1., 2, 2, 2, 2, 2, 2, 2]),
                                    verbose=verbose
                                    )
            param_est[index, :] = fit.x
            combination[index] = make_combination(param_est[index, :])

            residuals[index, :] = folded_bothasics[index, :] - ft.simsig_fringes(t, period / 6., param_est[index, :])

    return t, folded_bothasics, param_est, combination, period, residuals


def make_w_Michel(t, tm1=12, tm2=2, ph=5):
    # w is made to make the combination to see fringes with Michel's method
    w = np.zeros_like(t)
    wcheck = np.zeros_like(t)
    period = len(w) / 6
    for i in range(len(w)):
        if (((i - ph) % period) >= tm1) and (((i - ph) % period) < period - tm2):
            if (((i - ph) // period) == 0) | (((i - ph) // period) == 3):
                w[i] = 1.
            if (((i - ph) // period) == 1) | (((i - ph) // period) == 2):
                w[i] = -1.

    return w, wcheck


def analyse_fringes_Michel(datafolder, w, t0=None, tf=None, wt=5.,
                            lowcut=0.001, highcut=10, nbins=120,
                            notch=np.array([[1.724, 0.005, 10]]),
                            tes_check=28,
                            verbose=True, median=False, read_data = None, silent=False):
    """
    Compute the fringes with Michel's method.
    """

    res_michel = np.zeros(256)
    folded_bothasics = np.zeros((256, nbins))

    for ASIC in [1, 2]:
        if read_data is None:
            # Read the data
            t_data, data = get_data(datafolder, ASIC, doplot=False)
        else:
            t_data, data = read_data[ASIC - 1]

        # Cut the data
        t_data_cut, data_cut = cut_data(t0, tf, t_data, data)

        # Find the true period
        if ASIC == 1:
            ppp, rms, period = find_right_period(6 * wt, t_data_cut, data_cut[tes_check - 1, :])
            if verbose:
                print('period:', period)
                print('Expected : ', 6 * wt)

        # Fold and filter the data
        folded, t, folded_nonorm, newdata = ft.fold_data(t_data_cut,
                                                         data_cut,
                                                         period,
                                                         lowcut,
                                                         highcut,
                                                         nbins,
                                                         notch=notch,
                                                         median=median,
                                                         silent=silent,
                                                         )
        if ASIC == 1:
            folded_bothasics[:128, :] = folded
        else:
            folded_bothasics[128:, :] = folded

        # Michel method
        for TES in range(1, 129):
            index = (TES - 1) + 128 * (ASIC - 1)
            res_michel[index] = np.sum(folded[TES - 1, :] * w)

    return t, folded_bothasics, res_michel, period


def make_keyvals(date, type_eq, neq, Vtes, nstep=6, ecosorb='yes', frame='ONAFP'):
    '''
    Make a dictionary with relevant information on the measurement.
    Assign the FITS keyword values for the primary header
    '''
    keyvals = {}
    keyvals['DATE-OBS'] = (date, 'Date of the measurement')
    keyvals['TYPE-EQ'] = (type_eq, 'Equivalence type for this dataset')
    keyvals['NBLS'] = (neq, 'Number of equivalent baselines')
    keyvals['NSTEP'] = (nstep, 'Number of stable steps per cycle')
    keyvals['V_TES'] = (Vtes, 'TES voltage [V]')
    keyvals['ECOSORD'] = (ecosorb, 'Ecosorb on the source')
    keyvals['FRAME'] = (frame, 'Referential frame for (X, Y) TES')

    return keyvals


def make_fdict(BLs_eq, wt_eq, Ncycles_eq, xTES, yTES, t,
               folded, params, combination, periods, residuals, images):
    """ Make a dictionary with all relevant data."""
    fdict = {}
    fdict['BLS-EQ'] = BLs_eq
    fdict['WT'] = wt_eq
    fdict['NCYCLES'] = Ncycles_eq
    fdict['X_TES'] = xTES
    fdict['Y_TES'] = yTES
    fdict['TIME'] = t
    fdict['FOLDED'] = folded
    fdict['PARAMS'] = params
    fdict['COMBINATION'] = combination
    fdict['PERIODS'] = periods
    fdict['RESIDUALS'] = residuals
    fdict['IMAGES'] = images

    return fdict


def write_fits_fringes(out_dir, save_name, keyvals, fdict):
    """ Save a .fits with the fringes data."""
    if out_dir[-1] != '/':
        out_dir += '/'

    # Header creation
    hdr = pyfits.Header()
    for key in keyvals.keys():
        hdr[key] = (keyvals[key])

    hdu_prim = pyfits.PrimaryHDU(header=hdr)
    allhdu = [hdu_prim]
    for key in fdict.keys():
        hdu = pyfits.ImageHDU(data=fdict[key], name=key)
        allhdu.append(hdu)

    thdulist = pyfits.HDUList(allhdu)
    thdulist.writeto(out_dir + save_name, 'warn')

    return

def read_fits_fringes(file):
    """
    Read a .fits where you saved the data and returns two dictionaries with
    the header content and the data themselves.
    """
    hdulist = pyfits.open(file)
    header = hdulist[0].header
    print(header.keys)

    fringes_dict = {}
    for i in range(1, len(hdulist)):
        extname = hdulist[i].header['EXTNAME']
        data = hdulist[i].data
        fringes_dict[extname] = data

    return header, fringes_dict


def make_mask2D_thermometers_TD():
    mask_thermos = np.ones((17, 17))
    mask_thermos[0, 12:] = np.nan
    mask_thermos[1:5, 16] = np.nan
    return mask_thermos


def remove_thermometers(x, y, combi):
    """Remove the 8 thermometers.
    Returns AD arrays with 248 values and not 256."""
    combi = combi[x != 0.]
    x = x[x != 0.]
    y = y[y != 0.]
    return x, y, combi

def plot_fringes_onFP(q, BL_index, keyvals, fdict, mask=None):
    """Plot fringes on the FP with imshow and with a scatter plot."""
    if type(keyvals['NSTEP']) is tuple:
        for i in keyvals:
            keyvals[i] = keyvals[i][0]

    BL = fdict['BLS-EQ'][BL_index]
    date = keyvals['DATE-OBS']
    x = fdict['X_TES']
    y = fdict['Y_TES']
    combi = fdict['COMBINATION'][BL_index]
    frame = keyvals['FRAME']
    image = fdict['IMAGES'][BL_index]

    x, y, combi = remove_thermometers(x, y, combi)

    if mask is None:
        mask = make_mask2D_thermometers_TD()

    plt.subplots(1, 2, figsize=(14, 7))
    plt.suptitle(f'Baseline {BL} - ' + date, fontsize=14)
    lim = 2
    cmap = 'bwr'
    plt.subplot(121)
    plt.imshow(np.nan_to_num(image * mask),
               vmin=-lim, vmax=lim,
               cmap=cmap,
               interpolation='Gaussian')
    ft.qgrid()
    plt.title('Imshow', fontsize=14)
    plt.colorbar()

    plt.subplot(122)
    scal.scatter_plot_FP(q, x, y, combi, frame,
                         title='Scatter plot',
                         unit=None,
                         cmap=cmap,
                         vmin=-lim, vmax=lim
                         )
    return


def save_fringes_pdf_plots(out_dir, q, keyvals, fdict, mask=None):
    """Save all the fringe plots (all baselines) in a pdf file."""
    if type(keyvals['NSTEP']) is tuple:
        for i in keyvals:
            keyvals[i] = keyvals[i][0]

    neq = keyvals['NBLS']
    date = keyvals['DATE-OBS']
    type_eq = keyvals['TYPE-EQ']
    myname = 'Fringes_' + date + f'_TypeEq{type_eq}_with_{neq}BLs.pdf'

    with PdfPages(out_dir + myname) as pp:
        for i in range(neq):
            plot_fringes_onFP(q, i, keyvals, fdict, mask=mask)
            pp.savefig()
    return


def plot_folded_fit(TES, BL_index, keyvals, fdict):
    """Plot one folded signal for one TES with the fit and the residuals."""
    if type(keyvals['NSTEP']) is tuple:
        for i in keyvals:
            keyvals[i] = keyvals[i][0]

    BL = fdict['BLS-EQ'][BL_index]  # Baseline
    date = keyvals['DATE-OBS']  # Date of observation
    params = fdict['PARAMS'][BL_index, TES - 1, :]  # Fit parameters
    t0 = params[1]  # Starting time
    amps = params[2:8]  # Amplitudes
    t = fdict['TIME']  # Time
    folded = fdict['FOLDED'][BL_index, TES - 1, :]  # Folded signal
    period = fdict['PERIODS'][BL_index]  # Period
    nstep = keyvals['NSTEP']
    stable_time = period / nstep
    resid = fdict['RESIDUALS'][BL_index, TES - 1, :]  # Rediduals

    # Plot
    plt.figure(figsize=(6, 6))
    plt.plot(t, folded, label='folded signal')
    plt.plot(t, ft.simsig_fringes(t, stable_time, params), label='Fit')
    plt.plot(np.arange(0, period, stable_time) + t0, amps, 'ro', label='Amplitudes')
    plt.plot(t, resid, label='Residuals: RMS={0:6.4f}'.format(np.std(resid)))

    for k in range(nstep):
        plt.axvline(x=stable_time * k + t0, color='k', ls=':', alpha=0.3)
    plt.title(f'TES {TES} - Baseline {BL} - {date}', fontsize=14)
    plt.legend(loc='upper right')
    plt.ylim(-2.5, 2.5)

    return


def save_folded_fit_pdf_plots(out_dir, keyvals, fdict):
    """Save all the plots (folded signal, fit and residuals)
    for all TES in a .pdf."""
    if type(keyvals['NSTEP']) is tuple:
        for i in keyvals:
            keyvals[i] = keyvals[i][0]

    neq = keyvals['NBLS']
    date = keyvals['DATE-OBS']
    type_eq = keyvals['TYPE-EQ']
    myname = 'Folded_fit_' + date + f'_TypeEq{type_eq}_with_{neq}BLs.pdf'

    with PdfPages(out_dir + myname) as pp:
        for BL_index in range(neq):
            BL = fdict['BLS-EQ'][BL_index]
            plt.figure()
            plt.text(-1, 0, f'Baseline {BL}', fontsize=40)
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            plt.axis('off')
            pp.savefig()
            for TES in range(256):
                plot_folded_fit(TES, BL_index, keyvals, fdict)
                pp.savefig()
    return


def plot_sum_diff_fringes(keyvals, fdict, mask=None, lim=2, cmap='bwr'):
    """Plot the sum and the difference of all equivalent baselines."""
    if type(keyvals['NSTEP']) is tuple:
        for i in keyvals:
            keyvals[i] = keyvals[i][0]

    images = fdict['IMAGES']
    neq = keyvals['NBLS']
    date = keyvals['DATE-OBS']
    type_eq = keyvals['TYPE-EQ']

    if mask is None:
        mask = make_mask2D_thermometers_TD()

    sgns = np.ones((neq, 17, 17))
    for i in range(neq):
        sgns[i, :, :] *= (-1) ** i

    av_fringe = np.sum(images, axis=0) / neq
    diff_fringe = np.sum(images * sgns, axis=0) / neq

    plt.subplots(1, 2, figsize=(14, 7))
    plt.suptitle(f'Class equi {type_eq} with {neq} BLs - {date}', fontsize=14)

    plt.subplot(121)
    plt.imshow(np.nan_to_num(av_fringe * mask),
               vmin=-lim, vmax=lim,
               cmap=cmap,
               interpolation='Gaussian')
    ft.qgrid()
    plt.title(f'Imshow - Sum / {neq}', fontsize=14)
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(np.nan_to_num(diff_fringe * mask),
               vmin=-lim, vmax=lim,
               cmap=cmap,
               interpolation='Gaussian')
    ft.qgrid()
    plt.title(f'Imshow - Diff / {neq}', fontsize=14)
    plt.colorbar()
    plt.tight_layout()

    return


def find_t0(tfold, dfold, period, nconfigs=6, doplot=False):
    """
    Find time where configuration change in the square modulation.
    """

    # Average the signal over all periods
    msignal = np.mean(dfold, axis=0)
    # calculate the derivative and find where it is high
    dsignal = np.abs(np.gradient(msignal))
    md, sd = ft.meancut(dsignal, 3)
    thr = np.abs(dsignal - md) > (3 * sd)

    # Let's find clusters of high derivatives:
    # each time we take the first high derivative element
    t_change = tfold[thr]
    expected_stable_time = period / nconfigs
    start_times = []
    incluster = 0
    for i in range(len(t_change)):
        if incluster == 0:
            start_times.append(t_change[i])
            incluster = 1
        if i > 0:
            if (t_change[i] - t_change[i - 1]) > (expected_stable_time * 0.6):
                incluster = 0
    start_times = np.array(start_times)

    # Now we take the median of all start_times modulo period/nconfigs
    t0 = np.median(start_times % (period / nconfigs))

    if doplot:
        plt.figure(figsize=(10, 8))
        plt.plot(tfold, msignal, label='Mean over periods')
        plt.plot(tfold, dsignal, label='Derivative')
        plt.plot(tfold[thr], dsignal[thr], 'ro', label='High Derivative (>3sig)')
        for i in range(len(start_times)):
            if i == 0:
                lab = 'Found Start times'
            else:
                lab = None
            plt.axvline(x=start_times[i], ls='--', label=lab, alpha=0.5)
        for i in range(6):
            if i == 0:
                lab = 'Median Start Time (modulo period/6)'
            else:
                lab = None
            plt.axvline(x=t0 + i * period / nconfigs, color='r', ls='--', label=lab)
        plt.legend(framealpha=0.2)
        plt.title('t0 determination on Reference TES')
        plt.xlabel('Time in Period')
        plt.ylabel('Signal averaged over periods')
        plt.tight_layout()

    return t0

