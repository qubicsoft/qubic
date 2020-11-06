from __future__ import division, print_function

import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.optimize as sop
from matplotlib.backends.backend_pdf import PdfPages

import qubic
from qubic import selfcal_lib as sc
from qubicpack.utilities import Qubic_DataDir

from qubicpack import qubicpack as qp
from qubicpack.qubicfp import qubicfp
import qubic.fibtools as ft




# ============== Functions ==============
def get_data(dirs, nf, asic, tes=28, doplot=True):
    asic = str(asic)
    thedir = dirs[nf]

    # Qubicpack object
    a = qubicfp()
    a.verbosity = 0
    a.read_qubicstudio_dataset(thedir)
    data = a.azel_etc(TES=None)

    # Signal for one TES
    t0 = data['t_data ' + asic][0]
    t_data = data['t_data ' + asic] - t0
    data = data['data ' + asic]

    if doplot:
        fig, axs = plt.subplots(1, 2, figsize=(15, 3))
        plt.subplots_adjust(wspace=0.5)

        axs[0].plot(t_data, data[tes - 1, :])
        axs[0].set_title(thedir[-5:])

        axs[1].plot(t_data, data[tes - 1, :])
        axs[1].set_title(thedir[-5:])
        axs[1].set_xlim(0, 40)
        plt.show()

    return thedir, t_data, data

def cut_data(t0, tf, t_data, data):
    if t0 is None:
        t0 = 0

    if tf is None:
        tf = np.max(t_data)

    ok = (t_data > t0) & (t_data < tf)
    t_data_cut = t_data[ok] - t0
    data_cut = data[:, ok]

    return t_data_cut, data_cut

def find_right_period(guess, t_data, data_oneTES):
    ppp = np.linspace(guess - 1.5, guess + 1.5, 250)
    rms = np.zeros(len(ppp))
    for i in range(len(ppp)):
        xin = t_data % ppp[i]
        yin = data_oneTES
        xx, yy, dx, dy, o = ft.profile(xin, yin, nbins=100, plot=False)
        rms[i] = np.std(yy)
    period = ppp[np.argmax(rms)]

    return ppp, rms, period

def make_diff_sig(params, x, stable_time, data):
    thesim = ft.simsig_fringes(x, stable_time, params)
    diff = data - thesim
    return diff

def make_combination(param_est, verbose=0):
    amps = param_est[2:8]
    if verbose>0:
        print('Check:', amps[2], amps[4])
    #return amps[0] + amps[3] - amps[1] - amps[2]
    return (amps[0]+amps[3]+amps[5])/3 + amps[2] - amps[1] - amps[4]

def analyse_fringes(dirs, m, w=None, t0=None, tf=None, stable_time=3.,
                    lowcut=0.001, highcut=10, nbins=120,
                    notch=np.array([[1.724, 0.005, 10]]),
                    tes_check=28, param_guess=[0.1, 0., 1, 1, 1, 1, 1, 1], 
                    verbose=0, median=False, read_data = None, silent=False):
    res_w = np.zeros(256)
    res_fit = np.zeros(256)
    param_est = np.zeros((256, 8))
    folded_bothasics = np.zeros((256, nbins))
    for asic in [1, 2]:
        if read_data is None:
            #print('Reading data')
            _, t_data, data = get_data(dirs, m, asic, doplot=False)
        else:
            t_data, data = read_data[asic-1]

        # Cut the data
        t_data_cut, data_cut = cut_data(t0, tf, t_data, data)

        # Find the true period
        if asic == 1:
            ppp, rms, period = find_right_period(6 * stable_time, t_data_cut, data_cut[tes_check - 1, :])
            if verbose:
                print('period:', period)
                print('Expected : ', 6 * stable_time)

        # Fold and filter the data
        folded, t, folded_nonorm, newdata = ft.fold_data(t_data_cut,
                                                         data_cut,
                                                         period,
                                                         lowcut,
                                                         highcut,
                                                         nbins,
                                                         notch=notch,
                                                         median=median,
                                                         silent=silent
                                                         )
        if asic == 1:
            folded_bothasics[:128, :] = folded
        else:
            folded_bothasics[128:, :] = folded

        # Fit (Louise method) and Michel method
        for tes in range(1, 129):
            TESindex = (tes - 1) + 128 * (asic - 1)
            if w is not None:
                res_w[TESindex] = np.sum(folded[tes - 1, :] * w)

            fit = sop.least_squares(make_diff_sig,
                                    param_guess,
                                    args=(t,
                                          period / 6.,
                                          folded[tes - 1, :]),
                                    bounds=([0., -2, -2, -2, -2, -2, -2, -2],
                                            [1., 2, 2, 2, 2, 2, 2, 2]),
                                    verbose=verbose
                                    )
            param_est[TESindex, :] = fit.x
            res_fit[TESindex] = make_combination(param_est[TESindex, :])

    return t, folded_bothasics, param_est, res_w, res_fit, period

