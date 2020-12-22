from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits as pyfits
from matplotlib.backends.backend_pdf import PdfPages
import scipy.optimize as sop

from qubicpack.qubicfp import qubicfp
import qubic.fibtools as ft
from qubic import selfcal_lib as scal

__all__ = ['Fringes_Analysis']

# ============== Get data ==============
def get_data(datafolder, asics, src_data=False, subtract_t0=True):
    """
    Get the TODs for one ASIC.
    Parameters
    ----------
    datafolder: str
        Folder containing the data.
    asics: list
        ASIC numbers.
    doplot: bool
    src_data: if True,will return the srouce data as well
    subtract_t0: if True, will remove to the time the first time element

    Returns
    -------
    Time and signal for all TES in one ASIC.
    If src_data is True: will also return the  source time and signal
    """

    # Qubicpack object
    a = qubicfp()
    a.verbosity = 0
    a.read_qubicstudio_dataset(datafolder)

    # TOD from all ASICS
    data = []
    tdata = []
    for i, ASIC in enumerate(asics):
        ASIC = int(ASIC)
        data_oneASIC  = a.timeline_array(asic=ASIC)
        data.append(data_oneASIC)
        tdata_oneASIC = a.timeaxis(datatype='science', asic=ASIC)
        if subtract_t0:
            tdata_oneASIC -= tdata_oneASIC[0]
        tdata.append(tdata_oneASIC)
    tdata = np.array(tdata)
    data = np.array(data)

    if src_data: # Get calibration source data
        tsrc = a.calsource()[0]
        if subtract_t0:
            tsrc -= tsrc[0]
        dsrc = a.calsource()[1]

        return tdata, data, tsrc, dsrc
    else:
        return tdata, data


def plot_TOD(tdata, data, TES, tsrc=None, dsrc=None, xlim=None, figsize=(12, 6)):
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.plot(tdata, data[TES - 1, :])
    ax.set_title(f'TOD for TES {TES}')
    if xlim is not None:
        ax.set_xlim(0, xlim)
    plt.show()

    if dsrc is not None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
        ax.plot(tsrc, dsrc[TES - 1, :])
        ax.set_title('Calibration source data')
        if xlim is not None:
            ax.set_xlim(0, xlim)
        plt.show()
    return


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
    data_cut = data[ok]

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


def analyse_fringesLouise(datafolder, asics, t0=None, tf=None, wt=5.,
                          lowcut=1e-5, highcut=2., nbins=120,
                          notch=np.array([[1.724, 0.005, 10]]),
                          tes_check=28, param_guess=[0.1, 0., 1, 1, 1, 1, 1, 1],
                          median=False, verbose=True, ):
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


    # Read the data
    t_data, data = get_data(datafolder, asics)
    nasics, ndet, _ = data.shape
    ndet_tot = nasics * ndet

    fringes1D = np.zeros(ndet_tot)
    param_est = np.zeros((ndet_tot, 8))
    dfold = np.zeros((ndet_tot, nbins))
    residuals_time = np.zeros_like(dfold)

    for i, ASIC in enumerate(asics):
        # Cut the data
        t_data_cut, data_cut = cut_data(t0, tf, t_data[i], data[i])

        # Find the true period
        if i == 0:
            ppp, rms, period = find_right_period(6 * wt, t_data_cut, data_cut[tes_check - 1, :])
            if verbose:
                print('period:', period)
                print('Expected : ', 6 * wt)

        # Fold and filter the data
        fold, tfold, _, _ = ft.fold_data(t_data_cut,
                                         data_cut,
                                         period,
                                         lowcut,
                                         highcut,
                                         nbins,
                                         notch=notch,
                                         median=median,
                                         silent=verbose,
                                         )
        dfold[ndet*i:ndet*(i+1), :] = fold

        # Fit (Louise method)
        for j in range(ndet):
            index = ndet * i + j
            fit = sop.least_squares(make_diff_sig,
                                    param_guess,
                                    args=(tfold,
                                          period / 6.,
                                          fold[j, :]),
                                    bounds=([0., -2, -2, -2, -2, -2, -2, -2],
                                            [1., 2, 2, 2, 2, 2, 2, 2]),
                                    verbose=verbose
                                    )
            param_est[index, :] = fit.x
            fringes1D[index] = make_combination(param_est[index, :])

            residuals_time[index, :] = dfold[index, :] - ft.simsig_fringes(tfold, period / 6., param_est[index, :])

    return tfold, dfold, param_est, fringes1D, period, residuals_time


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
        folded, t, _, newdata = ft.fold_data(t_data_cut,
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


def make_keyvals(date, nBLs, Vtes, nstep=6, ecosorb='yes', frame='ONAFP'):
    '''
    Make a dictionary with relevant information on the measurement.
    Assign the FITS keyword values for the primary header
    '''
    keyvals = {}
    keyvals['DATE-OBS'] = (date, 'Date of the measurement')
    keyvals['NBLS'] = (nBLs, 'Number of baselines')
    keyvals['NSTEP'] = (nstep, 'Number of stable steps per cycle')
    keyvals['V_TES'] = (Vtes, 'TES voltage [V]')
    keyvals['ECOSORD'] = (ecosorb, 'Ecosorb on the source')
    keyvals['FRAME'] = (frame, 'Referential frame for (X, Y) TES')

    return keyvals


def make_fdict(allBLs, allwt, allNcycles, xTES, yTES, t,
               allfolded, allparams, allfringes1D, allperiods, allresiduals):
    """ Make a dictionary with all relevant data."""
    fdict = {}
    fdict['BLS'] = allBLs
    fdict['WT'] = allwt
    fdict['NCYCLES'] = allNcycles
    fdict['X_TES'] = xTES
    fdict['Y_TES'] = yTES
    fdict['TIME'] = t
    fdict['FOLDED'] = allfolded
    fdict['PARAMS'] = allparams
    fdict['FRINGES_1D'] = allfringes1D
    fdict['PERIODS'] = allperiods
    fdict['RESIDUALS'] = allresiduals

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


# def plot_fringes_onFP(q, BL_index, keyvals, fdict, mask=None, lim=2, cmap='bwr', cbar=True, s=None):
#     """Plot fringes on the FP with imshow and with a scatter plot."""
#     if type(keyvals['NSTEP']) is tuple:
#         for i in keyvals:
#             keyvals[i] = keyvals[i][0]
#
#     BL = fdict['BLS'][BL_index]
#     date = keyvals['DATE-OBS']
#     x = fdict['X_TES']
#     y = fdict['Y_TES']
#     fringes1D = fdict['FRINGES_1D'][BL_index]
#     frame = keyvals['FRAME']
#
#     x, y, fringes1D = remove_thermometers(x, y, fringes1D)
#
#     if mask is None:
#         mask = make_mask2D_thermometers_TD()
#
#     fig = plt.figure()
#     fig.suptitle(f'Baseline {BL} - ' + date, fontsize=14)
#     ax0 = plt.subplot(121)
#     img = ax0.imshow(np.nan_to_num(fringes2D * mask),
#                        vmin=-lim, vmax=lim,
#                        cmap=cmap,
#                        interpolation='Gaussian')
#     ft.qgrid()
#     ax0.set_title('Imshow', fontsize=14)
#     if cbar:
#         divider = make_axes_locatable(ax0)
#         cax = divider.append_axes('right', size='5%', pad=0.05)
#         fig.colorbar(img, cax=cax)
#
#     ax1 = plt.subplot(122)
#     scal.scatter_plot_FP(q, x, y, fringes1D, frame,
#                          fig=fig, ax=ax1,
#                          s=s,
#                          title='Scatter plot',
#                          unit=None,
#                          cmap=cmap,
#                          vmin=-lim, vmax=lim,
#                          cbar=cbar
#                          )
#     return


def save_fringes_pdf_plots(out_dir, q, keyvals, fdict, mask=None, **kwargs):
    """Save all the fringe plots (all baselines) in a pdf file."""
    if type(keyvals['NSTEP']) is tuple:
        for i in keyvals:
            keyvals[i] = keyvals[i][0]

    neq = keyvals['NBLS']
    date = keyvals['DATE-OBS']
    myname = 'Fringes_' + date + f'_{neq}BLs.pdf'

    with PdfPages(out_dir + myname) as pp:
        for i in range(neq):
            plot_fringes_onFP(q, i, keyvals, fdict, mask=mask, **kwargs)
            pp.savefig()
    return


def plot_folded_fit(TES, BL_index, keyvals, fdict, ax=None, legend=True):
    """Plot one folded signal for one TES with the fit and the residuals."""
    if type(keyvals['NSTEP']) is tuple:
        for i in keyvals:
            keyvals[i] = keyvals[i][0]

    params = fdict['PARAMS'][BL_index][TES - 1, :]  # Fit parameters
    t0 = params[1]  # Starting time
    amps = params[2:8]  # Amplitudes
    t = fdict['TIME']  # Time
    folded = fdict['FOLDED'][BL_index][TES - 1, :]  # Folded signal
    period = fdict['PERIODS'][BL_index]  # Period
    nstep = keyvals['NSTEP']
    stable_time = period / nstep
    resid = fdict['RESIDUALS'][BL_index][TES - 1, :]  # Residuals

    # Plot
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(t, folded, label='folded signal')
    ax.plot(t, ft.simsig_fringes(t, stable_time, params), label='Fit')
    ax.plot(np.arange(0, period, stable_time) + t0, amps, 'ro', label='Amplitudes')
    ax.plot(t, resid, label='Residuals: RMS={0:6.4f}'.format(np.std(resid)))

    for k in range(nstep):
        ax.axvline(x=stable_time * k + t0, color='k', ls=':', alpha=0.3)
    ax.set_title(f'TES {TES}', fontsize=14)
    ax.set_ylim(-2.5, 2.5)
    if legend:
        ax.legend(loc='upper right')
    return


def save_folded_fit_pdf_plots(out_dir, keyvals, fdict):
    """Save all the plots (folded signal, fit and residuals)
    for all TES in a .pdf."""
    if type(keyvals['NSTEP']) is tuple:
        for i in keyvals:
            keyvals[i] = keyvals[i][0]

    nBLs = keyvals['NBLS']
    date = keyvals['DATE-OBS']
    myname = 'Folded_fit_' + date + f'_{nBLs}BLs.pdf'

    with PdfPages(out_dir + myname) as pp:
        plt.figure()
        plt.text(-1, 0, f'Data from {date}', fontsize=40)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.axis('off')
        pp.savefig()
        for BL_index in range(nBLs):
            BL = fdict['BLS'][BL_index]
            plt.figure()
            plt.text(-1, 0, f'Baseline {BL}', fontsize=40)
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            plt.axis('off')
            pp.savefig()
            for page in range(11):
                fig, axs = plt.subplots(6, 4, figsize=(15, 25))
                axs = np.ravel(axs)
                for t in range(24):
                    ax = axs[t]
                    TES = page * 24 + t
                    if TES < 256:
                        plot_folded_fit(TES, BL_index, keyvals, fdict, ax=ax, legend=False)
                pp.savefig()
    return


def plot_sum_diff_fringes(q, keyvals, fdict, mask=None, lim=2, cmap='bwr'):
    """Plot the sum and the difference of all equivalent baselines."""
    if type(keyvals['NSTEP']) is tuple:
        for i in keyvals:
            keyvals[i] = keyvals[i][0]

    fringes2D = fdict['FRINGES_2D']
    allBLs = fdict['BLS']
    date = keyvals['DATE-OBS']

    if mask is None:
        mask = make_mask2D_thermometers_TD()

    BLs_sort, BLs_type = scal.find_equivalent_baselines(allBLs, q)
    ntype = np.max(BLs_type) + 1 # Number of equivalency types

    for j in range(ntype):
        images = np.array(fringes2D)[BLs_type == j]
        neq = len(BLs_sort[j]) # Number of equivalent baselines for that type
        sgns = np.ones((neq, 17, 17))
        for i in range(neq):
            sgns[i, :, :] *= (-1) ** i

        av_fringe = np.sum(images, axis=0) / neq
        diff_fringe = np.sum(images * sgns, axis=0) / neq

        plt.subplots(1, 2)
        plt.suptitle(f'{neq} BLs - {date}', fontsize=14)

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


def reorder_data(data, xdata, ydata, xqsoft, yqsoft):
    ndata = len(data)
    ndet = xdata.shape[0]
    data_ordered = []
    for k in range(ndata):
        olddata = data[k]
        newdata = np.zeros_like(olddata)
        for det in range(ndet):
            index_simu = np.where((xqsoft == xdata[det]) & (yqsoft == ydata[det]))[0][0]
            newdata[index_simu] = olddata[det]
        data_ordered.append(newdata)
    return data_ordered


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

