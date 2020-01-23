from __future__ import division, print_function

import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sop
import pandas as pd
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

    return thedir, t_data, data


def cut_data(t0, tf, t_data, data):
    ok = (t_data > t0) & (t_data < tf)
    t_data_cut = t_data[ok] - t0
    data_cut = data[:, ok]

    return t_data_cut, data_cut


def make_spectrum(t_data, data_oneTES, period):
    # Sampling frequency
    npoints = len(t_data)
    t0, tf = t_data[0], t_data[-1]
    f_sampling = npoints / (tf - t0)

    # Spectrum
    spectrum_f, freq_f = mlab.psd(data_oneTES,
                                  Fs=f_sampling,
                                  NFFT=2 ** int(np.log(len(data_oneTES)) / np.log(2)),
                                  window=mlab.window_hanning)
    plt.plot(freq_f, spectrum_f)
    plt.loglog()
    plt.xlim(0.1, 10)
    for i in range(1, 10):
        plt.axvline(x=i / period, color='orange')
    plt.grid()

    return spectrum_f, freq_f


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


def make_combination(param_est):
    amps = param_est[2:8]
    print('Check:', amps[2], amps[4])
    return amps[0] + amps[3] - amps[1] - amps[2]


def analyse_fringes(dirs, m, w, t0=4, tf=400, stable_time=3.,
                    lowcut=0.001, highcut=10, nbins=120,
                    notch=np.array([[1.724, 0.005, 10]]),
                    tes_check=28, param_guess=[0.1, 0., 1, 1, 1, 1, 1, 1]):
    res_w = np.zeros(256)
    res_fit = np.zeros(256)
    param_est = np.zeros((256, 8))
    folded_bothasics = np.zeros((256, nbins))
    for asic in [1, 2]:
        _, t_data, data = get_data(dirs, m, asic, doplot=False)
        t_data_cut, data_cut = cut_data(t0, tf, t_data, data)
        if asic == 1:
            ppp, rms, period = find_right_period(6 * stable_time, t_data_cut, data_cut[tes_check - 1, :])
            print('period:', period)

        folded, t, folded_nonorm, newdata = ft.fold_data(t_data_cut,
                                                         data_cut,
                                                         period,
                                                         lowcut,
                                                         highcut,
                                                         nbins,
                                                         notch=notch
                                                         )
        if asic == 1:
            folded_bothasics[:128, :] = folded
        else:
            folded_bothasics[128:, :] = folded

        for tes in range(1, 129):
            TESindex = (tes - 1) + 128 * (asic - 1)
            res_w[TESindex] = np.sum(folded[tes - 1, :] * w)

            fit = sop.least_squares(make_diff_sig,
                                    param_guess,
                                    args=(t,
                                          stable_time,
                                          folded[tes - 1, :]),
                                    bounds=([0., -2, -2, -2, -2, -2, -2, -2],
                                            [1., 2, 2, 2, 2, 2, 2, 2]),
                                    verbose=1
                                    )
            param_est[TESindex, :] = fit.x
            res_fit[TESindex] = make_combination(param_est[TESindex, :])

    return t, folded_bothasics, param_est, res_w, res_fit


# =============== Fringe simulations =================
rep = Qubic_DataDir(datafile='detcentres.txt')
print('rep:', rep)

# Get simulation files
files = sorted(glob.glob(rep + '/*.dat'))

# Get a dictionary
basedir = Qubic_DataDir(datafile='instrument.py', )
print('basedir : ', basedir)
dictfilename = basedir + '/dicts/global_source_oneDet.dict'
d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)

# Make an instrument
q = qubic.QubicInstrument(d)

horn1 = [49, 57, 52, 60, 25, 1, 57, 60, 40, 40, 39, 39]
horn2 = [25, 25, 28, 28, 28, 4, 43, 63, 64, 63, 64, 63]
param = [0., 150.]
readv, lab = sc.make_labels(rep)

plt.figure()
plt.subplots_adjust(wspace=0.5, hspace=0.5)
for i in range(12):
    baseline = [horn1[i], horn2[i]]
    q.horn.open = False
    q.horn.open[np.asarray(baseline) - 1] = True
    img = sc.get_simulation(param, q, baseline, files, lab, doplot=False)
    plt.subplot(4, 3, i + 1)
    plt.title('Baseline {}'.format(baseline))
    plt.imshow(img)

# ============== Get data ==============
global_dir = '/home/louisemousset/QUBIC/Qubic_work/Calibration/datas/'
# June measurement
# data_dir = global_dir + '2019-06-07/'

# December measurement
# data_dir = global_dir + 'fringes2019-12-19/'

# January measurement
data_dir = global_dir + '2020-01-13/'

dirs = np.sort(glob.glob(data_dir + '*switch*'))
print('# simu:', len(dirs))

labels = []
for i, d in enumerate(dirs):
    bla = str.split(d, '/')
    labels.append(bla[-1])
    print(i, labels[i])

# Select a simulation
nf = 2
tes = 28
asic = 1

thedir, t_data, data = get_data(dirs, nf, asic, tes)

# Cut the data
tstart = 4
tend = 400

t_data_cut, data_cut = cut_data(tstart, tend, t_data, data)

# Find the right period
ppp, rms, period = find_right_period(18, t_data_cut, data_cut[tes - 1, :])
print('period : ', ppp[np.argmax(rms)])

plt.figure()
rc('figure', figsize=(9, 4.5))
plt.subplots_adjust(wspace=2)

plt.subplot(211)
plot(ppp, rms, '.')
plt.axvline(x=period, color='orange')

plt.subplot(212)
plt.plot(t_data % period, data[tes - 1, :], '.')
plt.xlim(0, period)

# Filter the data (just to give an idea because it is done when folding)
lowcut = 0.001
highcut = 10.
nharm = 10
notch = np.array([[1.724, 0.005, nharm]])

newdata = ft.filter_data(t_data_cut, data_cut[tes-1, :], lowcut, highcut, notch=notch,
                         rebin=True, verbose=True, order=5)

spectrum_f, freq_f = make_spectrum(t_data_cut, data_cut[tes-1, :], period)

spectrum_f2, freq_f2 = make_spectrum(t_data_cut, newdata, period)

# compute spectrum with fibtools
spectrum_f3, freq_f3 = ft.power_spectrum(t_data_cut, newdata, rebin=True)

plt.figure()
plt.subplot(211)
plt.plot(freq_f, spectrum_f, label='Original')
plt.plot(freq_f2, spectrum_f2, label='filtered')
plt.plot(freq_f3, spectrum_f3, label='filtered2')
plt.legend()
plt.loglog()
plt.ylim(1e1, 1e17)

plt.subplot(212)
plt.plot(t_data_cut, data_cut[tes-1, :], label='Original')
plt.plot(t_data_cut, newdata, label='Filtered')
plt.legend()

# Fold and filter the data
nbins = 120
folded, t, folded_nonorm, newdata = ft.fold_data(t_data_cut,
                                                 data_cut,
                                                 period,
                                                 lowcut,
                                                 highcut,
                                                 nbins,
                                                 notch=notch,
                                                 )
plt.figure()
plt.subplot(211)
plt.plot(t_data_cut, data_cut[tes - 1, :])
plt.title('Data cut')
plt.xlim(0, period)

plt.subplot(212)
plt.plot(t, folded[tes - 1, :])
plt.title('Folded data')
plt.xlim(0, period)

# ========== Fit folded signal ================
param_guess = [0.1, 0., 1, 1, 1, 1, 1, 1]
stable_time = 3.
fit = sop.least_squares(make_diff_sig,
                        param_guess,
                        args=(t,
                              stable_time,
                              folded[tes - 1, :]),
                        bounds=([0., -2, -2, -2, -2, -2, -2, -2],
                                [1., 2, 2, 2, 2, 2, 2, 2]),
                        verbose=1
                        )
param_est = fit.x
print('Param_est :', param_est)

x0_est = param_est[1]
amps_est = param_est[2:8]

plt.figure()
plt.plot(t, folded[tes - 1, :], label='folded signal')
plt.plot(t, ft.simsig_fringes(t, stable_time, param_est), label='fit')
plt.plot(np.arange(0, 6 * stable_time, stable_time) + x0_est, amps_est, 'ro', label='amplitudes')
plt.title('ASIC {}, TES {}'.format(asic, tes))
plt.legend()
plt.grid()

comb = make_combination(param_est)
print(comb)

# ========= Michel's method ===================
# w is made to make the combination to see fringes
tm1 = 12
tm2 = 2
ph = 5
w = np.zeros_like(t)
wcheck = np.zeros_like(t)
print(len(w))
per = len(w) / 6
for i in range(len(w)):
    if (((i - ph) % per) >= tm1) and (((i - ph) % per) < per - tm2):
        if (((i - ph) // per) == 0) | (((i - ph) // per) == 3):
            w[i] = 1.
        if (((i - ph) // per) == 1) | (((i - ph) // per) == 2):
            w[i] = -1.

npts = np.sum(w != 0.) / 4.

print(npts)
print(np.sum(np.abs(w[int(per + ph):int(2 * per + ph)])))
print(np.sum(w))

themax = np.max(folded[tes - 1, :])

plt.figure()
plt.plot(t, folded[tes - 1, :])
plt.plot(t, w * themax, 'o')
plt.plot(t, wcheck * themax, 'x')
plt.xlim(0, period)
plt.grid()


# ============ Analysis for both ASICs and all measurements ==================
t, folded_bothasics, param_est, res_w, res_fit = analyse_fringes(dirs, nf, w, t0=4, tf=200, stable_time=3.)
fringe = ft.image_asics(all1=res_w)

# =================== Make a mask ==============
# Mask to remove the 8 thermometer pixels
mask = np.ones_like(fringe)
mask[0, 12:] = np.nan
mask[1:5, 16] = np.nan

# Mask to remove bad pixels
# By looking at the signals
bad1 = np.array([1, 4, 5, 11, 12, 18, 19, 21, 29, 30, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                 46, 47, 48, 49, 50, 51, 63, 65, 66, 68, 69, 70, 78, 80, 83, 89, 90, 91, 93, 97, 99, 100,
                 101, 102, 104, 108, 114, 115, 116, 119, 121, 122, 124, 126]) - 1
bad2 = np.array([2, 4, 8, 11, 12, 16, 21, 23, 26, 27, 28, 29, 36, 37, 40, 46, 49, 51, 55, 57, 58, 62,
                 63, 64, 68, 69, 71, 74, 76, 78, 79, 83, 89, 92, 94, 95, 96, 98, 99, 100, 101, 102, 103, 104, 106,
                 107, 108, 109, 110, 111, 112, 115, 117, 119, 121, 126, 127]) + 127

# From Michel
bad = [4, 7, 11, 12, 19, 35, 36, 37, 41, 42, 43, 45, 47, 48, 53, 55, 65, 68, 70, 89, 92, 97, 100, 102, 104,
       114, 119, 121, 126, 132, 144, 149, 150, 151, 152, 154,
       155, 156, 157, 164, 169, 170, 174, 179, 183, 190, 191, 192, 196, 203, 208, 217, 222, 226, 227, 228,
       229, 230, 231, 233, 235, 236, 237, 238, 240, 241, 243, 244, 245, 246, 247, 248, 249, 253, 254, 256]

bad = np.array(bad)
maskres = np.ones_like(res_w)
maskres[bad - 1] = np.nan
# maskres[bad1] = np.nan
# maskres[bad2] = np.nan

mask2 = ft.image_asics(all1=maskres)

# Mask to remove max values from check
lim = 10000
mask3 = np.ones_like(fringe_fit)
mask3[np.abs(fringe_fit) < lim] = np.nan

plt.figure()
plt.imshow(mask3, vmin=-1e5, vmax=1e5)
plt.title('mask3')
plt.colorbar()

# ============== Plots =============
# Look at one fit
x0_est = param_est[tes - 1, 1]
amps_est = param_est[tes - 1, 2:8]
plt.figure()
plt.plot(t, folded_bothasics[tes - 1, :], label='folded signal')
plt.plot(t, ft.simsig_fringes(t, stable_time, param_est[tes - 1, :]), label='fit')
plt.plot(np.arange(0, 6 * stable_time, stable_time) + x0_est, amps_est, 'ro', label='amplitudes')
plt.plot(t, w, 'g+', label='w')

# Plot fringes measured and simu
baseline = [57, 25]
plt.figure()
plt.suptitle('Baseline {}'.format(baseline))

plt.subplot(121)
lim = 3
plt.imshow(fringe * mask, vmin=-lim, vmax=lim)
plt.title('Measurement 2020-01-13')
# plt.colorbar()

plt.subplot(122)
q.horn.open = False
q.horn.open[np.asarray(baseline) - 1] = True
img = sc.get_simulation(param, q, baseline, files, lab, doplot=False, verbose=False)
plt.title('Simulation')
plt.imshow(img)

# ================ Loop on all set of measurements ===============
allfolded_bothasics = []
allparam_est = []
allres_w = []
allres_fit = []
for m in range(12):
    t, folded_bothasics, param_est, res_w, res_fit = analyse_fringes(dirs, m, w)
    allfolded_bothasics.append(folded_bothasics)
    allparam_est.append(param_est)
    allres_w.append(res_w)
    allres_fit.append(res_fit)

allfolded_bothasics = np.array(allfolded_bothasics)
allparam_est = np.array(allparam_est)
allres_w = np.array(allres_w)
allres_fit = np.array(allres_fit)

# np.save(data_dir + 'allfolded_bothasics_inv', allfolded_bothasics)
# np.save(data_dir + 'allres_w_inv', allres_w)
# np.save(data_dir + 'allres_fit_inv', allres_fit)
# np.save(data_dir + 'allparam_est_inv', allparam_est)

# Plot all the fringes
plt.figure()
plt.suptitle('Fit method')
plt.subplots_adjust(hspace=0.6)
lim = 8
for i in range(12):
    fringe = ft.image_asics(all1=allres_w[i])
    plt.subplot(4, 3, i + 1)
    plt.imshow(fringe * mask * mask2, vmin=-lim, vmax=lim)
    plt.title('Bl {}'.format(labels[i][-5:]))
    plt.colorbar()

allres_fit = np.load(data_dir + 'allres_fit_inv.npy')
allres_w = np.load(data_dir + 'allres_w_inv.npy')
allfolded_bothasics = np.load(data_dir + 'allfolded_bothasics_inv.npy')
allparam_est = np.load(data_dir + 'allparam_est_inv.npy')

# Make a PDF with all the TES signals
with PdfPages(data_dir + '/allTES_fit2.pdf') as pdf:
    for tes in range(1, 257):
        asic = 1
        if tes > 128:
            asic = 2
        plt.figure(figsize=(21, 30))
        plt.suptitle('ASIC {}, TES {}'.format(asic, tes % 128), fontsize=35)
        plt.subplots_adjust(wspace=0.2, hspace=0.5)
        for i in range(12):
            plt.subplot(4, 3, i + 1)
            x0_est = allparam_est[i, tes - 1, 1]
            amps_est = allparam_est[i, tes - 1, 2:8]
            plt.plot(t, allfolded_bothasics[i, tes - 1, :], label='folded signal')
            plt.plot(t, ft.simsig_fringes(t, stable_time, allparam_est[i, tes - 1, :]), label='fit')
            plt.plot(np.arange(0, 6 * stable_time, stable_time) + x0_est, amps_est, 'ro', label='amplitudes')
            plt.plot(t, w, 'g+', label='w')
            plt.title('Bl {}'.format(labels[i][-5:]), fontsize=15)
            if i == 0:
                plt.legend()
            plt.grid()

        pdf.savefig()
        plt.close()
