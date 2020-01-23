from __future__ import division, print_function

import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sop
import pandas as pd

# from matplotlib import rc
# rc('figure', figsize=(9, 4.5))
# rc('font', size=12)
# rc('text', usetex=False)

import qubic
from qubic import selfcal_lib as sc
from qubicpack.utilities import Qubic_DataDir

from qubicpack import qubicpack as qp
from qubicpack.qubicfp import qubicfp
import qubic.fibtools as ft

from pysimulators import FitsArray
from qubicpack.pix2tes import assign_tes_grid

tes_grid = assign_tes_grid()


# ============== Functions ==============
def get_data(dirs, nf, asic, tes=28, doplot=True):
    asic = str(asic)
    thedir = dirs[nf]
    # print(thedir)

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


def cut_data(tstart, tend, t_data, data):
    ok = (t_data > tstart) & (t_data < tend)
    t_data_cut = t_data[ok] - tstart
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


def simsig_fringes(x, stable_time, params):
    dx = x[1] - x[0]
    npoints = len(x)
    tf = x[-1]

    ctime = params[0]
    x0 = params[1]
    amp = params[2:8]
    #     print(amp)

    sim_init = np.zeros(len(x))

    for i in range(6):
        a = int(npoints / tf * stable_time * i)
        b = int((stable_time * i + stable_time) * npoints / tf)
        #         print(a, b)
        sim_init[a: b] = amp[i]

    # Add a phase
    sim_init_shift = np.interp((x - x0) % max(x), x, sim_init)

    # Convolved by an exponential filter
    thesim = ft.exponential_filter1d(sim_init_shift, ctime / dx, mode='wrap')

    return np.array(thesim).astype(np.float64)


def make_diff_sig(params, x, stable_time, data):
    thesim = simsig_fringes(x, stable_time, params)
    diff = data - thesim
    return diff


def make_combination(param_est):
    amps = param_est[2:8]
    print('Check:', amps[2], amps[4])
    return amps[0] + amps[3] - amps[1] - amps[2]


def get_quadrant3(q, signal_perTES, doplot=False):
    quadrant3 = signal_perTES[496:744]
    indice = -(q.detector.center // 0.003)

    img = np.zeros((17, 17))
    for k in range(248):
        i = int(indice[k, 0])
        j = int(indice[k, 1])
        img[i - 1, j - 1] = quadrant3[k]
    img[img == 0.] = np.nan
    img = np.rot90(img)

    if doplot:
        plt.figure()
        plt.imshow(img)

    return img


def get_simulation(param, q, baseline, horn_transpose, files, labels, nn=241, doplot=True):
    theta_source = param[0]
    freq_source = param[1]

    allampX = np.empty((2, nn, nn))
    allphiX = np.empty((2, nn, nn))
    allampY = np.empty((2, nn, nn))
    allphiY = np.empty((2, nn, nn))
    for i, swi in enumerate(baseline):
        # Phase calculation
        horn_x = q.horn.center[swi - 1, 0]
        horn_y = q.horn.center[swi - 1, 1]
        dist = np.sqrt(horn_x ** 2 + horn_y ** 2)  # distance between the horn and the center
        phi = - 2 * np.pi / 3e8 * freq_source * 1e9 * dist * np.sin(np.deg2rad(theta_source))

        thefile = files[horn_transpose[swi - 1]]
        print('Horn ', swi, ': ', thefile[98:104])
        data = pd.read_csv(thefile, sep='\t', skiprows=0)

        allampX[i, :, :] = np.reshape(np.asarray(data['MagX']), (nn, nn)).T
        allampY[i, :, :] = np.reshape(np.asarray(data['MagY']), (nn, nn)).T

        allphiX[i, :, :] = np.reshape(np.asarray(data['PhaseX']), (nn, nn)).T + phi
        allphiY[i, :, :] = np.reshape(np.asarray(data['PhaseY']), (nn, nn)).T + phi

    # Electric field for each open horn
    Ax = allampX * (np.cos(allphiX) + 1j * np.sin(allphiX))
    Ay = allampY * (np.cos(allphiY) + 1j * np.sin(allphiY))

    # Sum of the electric fields
    sumampx = np.sum(Ax, axis=0)
    sumampy = np.sum(Ay, axis=0)

    # Power on the focal plane
    power = np.abs(sumampx) ** 2 + np.abs(sumampy) ** 2

    if doplot:
        plt.figure()
        plt.subplot(121)
        q.horn.plot()
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(power, origin='lower')
        plt.title('Power at the sampling resolution')
        plt.colorbar()

    counts_perTES, sum_perTES, mean_perTES = sc.fulldef2tespixels(power, labels)

    img = get_quadrant3(q, mean_perTES, doplot=doplot)

    return img


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
nf = 1
tes = 28  # 105#39
asic = 1

thedir, t_data, data = get_data(dirs, nf, asic, tes)

# Cut the data
tstart = 4
tend = 200

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

# Fold and filter the data
lowcut = 0.001
highcut = 10.
nharm = 10
notch = np.array([[1.724, 0.005, nharm]])

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
plt.plot(t, simsig_fringes(t, stable_time, param_est), label='fit')
plt.plot(np.arange(0, 6 * stable_time, stable_time) + x0_est, amps_est, 'ro', label='amplitudes')
plt.title('ASIC {}, TES {}'.format(asic, tes))
plt.legend()
plt.grid()

comb = make_combination(param_est)
print(comb)

# ========= Michel's method ===================
# w is made to make the combination to see fringes
tm1 = 10
tm2 = 4
ph = 11
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
def analyse_fringes(dirs, m, w, t0=4, tf=400, stable_time=3.,
                    lowcut=0.001, highcut=10, nbins=120,
                    notch=np.array([[1.724, 0.005, 10]]),
                    tes_check=28, param_guess=[0.1, 0., 1, 1, 1, 1, 1, 1]):
    res_w = np.zeros(256)
    res_fit = np.zeros(256)
    param_est = np.zeros((256, 8))
    folded_bothasics = np.zeros((256, nbins))
    for asic in [1, 2]:
        thedir, t_data, data = get_data(dirs, m, asic, doplot=False)
        t_data_cut, data_cut = cut_data(t0, tf, t_data, data)
        if asic==1:
            ppp, rms, period = find_right_period(6*stable_time, t_data_cut, data_cut[tes_check - 1, :])
        print('period:', period)

        folded, t, folded_nonorm, newdata = ft.fold_data(t_data_cut,
                                                      data_cut,
                                                      period,
                                                      lowcut,
                                                      highcut,
                                                      nbins,
                                                      notch=notch
                                                      )
        if asic ==1:
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

t, folded_bothasics, param_est, res_w, res_fit = analyse_fringes(dirs, nf, w, t0=4, tf=200, stable_time=3.)

x0_est = param_est[tes - 1, 1]
amps_est = param_est[tes - 1, 2:8]
plt.figure()
plt.plot(t, folded_bothasics[tes - 1, :], label='folded signal')
plt.plot(t, simsig_fringes(t, stable_time, param_est[tes - 1, :]), label='fit')
plt.plot(np.arange(0, 6*stable_time, stable_time) + x0_est, amps_est, 'ro', label='amplitudes')
plt.plot(t, w, 'g+', label='w')

plt.figure()
baseline = [57, 25]
plt.suptitle('Baseline {}'.format(baseline))
plt.subplot(121)
lim=3
fringe = ft.image_asics(all1=res_w)
plt.imshow(fringe * mask, vmin=-lim, vmax=lim)
plt.title('Measurement 2020-01-13')
# plt.colorbar()
plt.subplot(122)

q.horn.open = False
q.horn.open[np.asarray(baseline) - 1] = True

img = get_simulation(param, q, baseline, horn_transpose, files, lab, doplot=False)

plt.title('Simulation')
plt.imshow(img)



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
lim=8
for i in range(12):
    fringe = ft.image_asics(all1=allres_w[i])
    plt.subplot(4, 3, i+1)
    plt.imshow(fringe * mask * mask2, vmin=-lim, vmax=lim)
    plt.title('Bl {}'.format(labels[i][-5:]))
    plt.colorbar()

allres_fit = np.load(data_dir + 'allres_fit_inv.npy')
allres_w = np.load(data_dir + 'allres_w_inv.npy')
allfolded_bothasics = np.load(data_dir + 'allfolded_bothasics_inv.npy')
allparam_est = np.load(data_dir + 'allparam_est_inv.npy')
from matplotlib.backends.backend_pdf import PdfPages
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
            plt.plot(t, simsig_fringes(t, stable_time, allparam_est[i, tes - 1, :]), label='fit')
            plt.plot(np.arange(0, 6*stable_time, stable_time) + x0_est, amps_est, 'ro', label='amplitudes')
            plt.plot(t, w, 'g+', label='w')
            plt.title('Bl {}'.format(labels[i][-5:]), fontsize=15)
            if i == 0:
                plt.legend()
            plt.grid()

        pdf.savefig()
        plt.close()



# =================== Make a mask ==============
# Mask to remove the 8 thermometer pixels
mask = np.ones_like(fringe)
mask[0, 12:] = np.nan
mask[1:5, 16] = np.nan

# Mask to remove bad pixels
bad1 = np.array([1, 4, 5, 11, 12, 18, 19, 21, 29, 30, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                 46, 47, 48, 49, 50, 51, 63, 65, 66, 68, 69, 70, 78, 80, 83, 89, 90, 91, 93, 97, 99, 100,
                 101, 102, 104, 108, 114, 115, 116, 119, 121, 122, 124, 126]) - 1
bad2 = np.array([2, 4, 8, 11, 12, 16, 21, 23, 26, 27, 28, 29, 36, 37, 40, 46, 49, 51, 55, 57, 58, 62,
                 63, 64, 68, 69, 71, 74, 76, 78, 79, 83, 89, 92, 94, 95, 96, 98, 99, 100, 101, 102, 103, 104, 106,
                 107, 108, 109, 110, 111, 112, 115, 117, 119, 121, 126, 127]) + 127

bad = [4, 7, 11, 12, 19, 35, 36, 37, 41, 42, 43, 45, 47, 48, 53, 55, 65, 68, 70, 89, 92, 97, 100, 102, 104,
       114, 119, 121, 126, 132, 144, 149, 150, 151, 152, 154,
       155, 156, 157, 164, 169, 170, 174, 179, 183, 190, 191, 192, 196, 203, 208, 217, 222, 226, 227, 228,
 229, 230, 231, 233, 235, 236, 237, 238, 240, 241, 243, 244, 245, 246, 247, 248, 249, 253, 254, 256]
bad = np.array(bad)
maskres = np.ones_like(allres_w[0,:])
maskres[bad-1] = np.nan
# maskres[bad1] = np.nan
# maskres[bad2] = np.nan

mask2 = ft.image_asics(all1=maskres)

# Mask to remove max values from check
lim = 10000
mask3 = np.ones_like(fringe_fit)
mask3[np.abs(fringe_fit) < lim] = np.nan
plt.imshow(mask3, vmin=-1e5, vmax=1e5)
plt.title('mask3')
colorbar()

# Apply masks on fringes
plt.figure()
plt.imshow(fringe * mask * mask2, vmin=-0.5, vmax=0.5)

# =============== Fringes simulation =================
from qubic import selfcal_lib as sc
from qubicpack.utilities import Qubic_DataDir
rep = Qubic_DataDir(datafile='detcentres.txt')
print('rep:', rep)

# Get simulation files
files = sorted(glob.glob(rep + '/*.dat'))

# This is done to get the right file for each horn
horn_transpose = np.arange(64)
horn_transpose = np.reshape(horn_transpose, (8, 8))
horn_transpose = np.ravel(horn_transpose.T)


# Use a tool from qubicpack to get a path
basedir = Qubic_DataDir(datafile='instrument.py', )
print('basedir : ', basedir)
dictfilename = basedir + '/dicts/global_source_oneDet.dict'

# Get a dictionary
d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)

# Make an instrument
horn1 = [49, 57, 52, 60, 25, 1, 57, 60, 40, 40, 39, 39]
horn2 = [25, 25, 28, 28, 28, 4, 43, 63, 64, 63, 64, 63]

param = [0., 150.]
readv, lab = sc.make_labels(rep)
q = qubic.QubicInstrument(d)
plt.figure()
plt.subplots_adjust(wspace=0.5, hspace=0.5)
for i in range(12):
    baseline = [horn1[i], horn2[i]]
    q.horn.open = False
    q.horn.open[np.asarray(baseline) - 1] = True

    img = get_simulation(param, q, baseline, horn_transpose, files, lab, doplot=False)
    plt.subplot(4, 3, i+1)
    plt.title('Baseline {}'.format(baseline))
    plt.imshow(img)