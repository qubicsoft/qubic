import glob

import numpy as np
from matplotlib.pyplot import *
import matplotlib.mlab as mlab

from qubicpack import qubicpack as qp
import fibtools as ft

import demodulation_lib as dl
from qubic.utils import progress_bar

# ========== Get data ==================
day = '2019-04-04'
data_dir = '/home/louisemousset/QUBIC/Qubic_work/Calibration/datas/' + day + '/'
dirs = np.sort(glob.glob(data_dir + '*RFswitch*'))
print (dirs)

# Take the first file
thedir = dirs[0]
print(thedir)

# Look at the file labels
labels = []
for d in dirs:
    bla = str.split(d, '__')
    labels.append(bla[1])
print (labels)

# =============================================


def get_amp_first_harmonic(a, tes, freq_mod):
    """
    This function takes the signal of a given TES, makes its spectrum
    and get the amplitude of the first harmonic that should be around
    the modulation frequency of the source.

    Parameters
    ----------
    a : qubicpack object
    tes : int
        tes number
    freq_mod : float
        modulation frequency of the external source

    Returns
    -------

    """
    data = a.timeline(TES=tes)
    t_data = a.timeline_timeaxis(axistype='pps')
    FREQ_SAMPLING = 1. / (t_data[1] - t_data[0])
    spectrum_f, freq_f = mlab.psd(data, Fs=FREQ_SAMPLING, NFFT=len(data), window=mlab.window_hanning)
    amp_peak = np.interp(freq_mod, freq_f, spectrum_f)
    okfit = np.abs(freq_f - freq_mod) < 0.1

    guess = np.array([freq_mod, 0.01, np.max(spectrum_f[okfit]), np.median(spectrum_f[okfit])])

    res = ft.do_minuit(freq_f[okfit], spectrum_f[okfit], np.ones(np.sum(okfit)), guess, functname=dl.gauss,
                       fixpars=[1, 0, 0, 0, 0], nohesse=True, force_chi2_ndf=True)

    return t_data, data, FREQ_SAMPLING, freq_f, spectrum_f, amp_peak, okfit, res

# =========== Test with one TES ==============
# Read data from a given ASIC
AsicNum = 1
a = qp()
a.read_qubicstudio_dataset(thedir, asic=AsicNum)

freq_mod = 1.
tes_num = 44
t_data, data, FREQ_SAMPLING, freq_f, spectrum_f, amp_peak, okfit, res = get_amp_first_harmonic(a, tes_num, freq_mod)
print('FREQ_SAMPLING = {}'.format(FREQ_SAMPLING))

# Signal as a function of time
plot(t_data, (data - np.mean(data)) / np.std(data), label='Data')

# Look at the amplitude of the peak
print('Amplitude = {}'.format(res[1][2]))

figure('TES spectrum')
plot(freq_f, spectrum_f, label='Data')
plot(freq_f[okfit], dl.gauss(freq_f[okfit], res[1]), label='Gaussian: amplitude = {0:5.3g}'.format(res[1][2]))
plot([freq_mod, freq_mod], [1e4, 1e13], label='Modulation Frequency: {}Hz'.format(freq_mod))
yscale('log')
xscale('log')
xlim(0.01, 3.)
ylim(1e4, 1e13)
legend(loc='best')


# =============== Loop on all TES an both ASICS ================
allres = np.zeros((256, 4))
allerr = np.zeros((256, 4))
allamp_peak = np.zeros(256)

for asic in [1, 2]:
    a.read_qubicstudio_dataset(thedir, asic=asic)
    bar = progress_bar(128, 'ASIC #{}'.format(AsicNum))
    for tes in np.arange(128) + 1:
        t_data, data, FREQ_SAMPLING, freq_f, spectrum_f, amp_peak, okfit, res = get_amp_first_harmonic(a, tes, freq_mod)
        bar.update()
        tes_index = (tes - 1) + 128 * (asic - 1)
        allres[tes_index, :] = res[1]
        allerr[tes_index, :] = res[2]
        allamp_peak[tes_index] = amp_peak

amps = allres[:, 2]
# amps = allamp_peak
img_one = ft.image_asics(all1=amps)
mm, ss = ft.meancut(amps, 3)
imshow(img_one, vmin=0, vmax=mm + 3 * ss, interpolation='nearest')
colorbar()

# ============== Loop on all files =================
allres_tot = np.zeros((len(dirs), 256, 4))
allerr_tot = np.zeros((len(dirs), 256, 4))
allamp_peak_tot = np.zeros((len(dirs), 256))

for idir in xrange(len(dirs)):
    thedir = dirs[idir]
    for asic in [1, 2]:
        a.read_qubicstudio_dataset(thedir, asic=asic)
        bar = progress_bar(128, 'file #{0} ASIC #{1}'.format(idir, AsicNum))
        for tes in np.arange(128) + 1:
            t_data, data, FREQ_SAMPLING, freq_f, spectrum_f, amp_peak, okfit, res = get_amp_first_harmonic(a, tes, freq_mod)
            bar.update()
            tes_index = (tes - 1) + 128 * (asic - 1)
            allres_tot[idir, tes_index, :] = res[1]
            allerr_tot[idir, tes_index, :] = res[2]
            allamp_peak_tot[idir, tes_index] = amp_peak

# ============= Look at signal for each file ===========
amplitudes = allres_tot[:, :, 2]
# amplitudes = allamp_peak_tot

mm, ss = ft.meancut(amplitudes, 3)

allimg = np.empty((len(dirs), 17, 17))

figure('Each measurement')
for i in xrange(len(dirs)):
    subplot(3, 3, i + 1)
    amps = amplitudes[i, :]
    img = ft.image_asics(all1=amps)
    allimg[i, :, :] = img
    imshow(img, vmin=0, vmax=50*ss, interpolation='nearest')
    colorbar()
    title(labels[i])

# ============== Try to make the fringes ================

# This is Stot
index_tot = 1
Stot = amplitudes[index_tot, :]

# These are C-i, C-j and S-ij
index_21_35 = ['21_35', 3, 4, 0]
index_21_39 = ['21_39', 3, 5, 2]
index_39_54 = ['39_54', 5, 8, 6]
index_21_54 = ['21_54', 3, 8, 7]

allsets = [index_21_35, index_21_39, index_39_54, index_21_54]

allfringe = np.zeros((len(allsets), 17, 17))

for iset in xrange(len(allsets)):
    theset = allsets[iset]
    C_i = amplitudes[theset[1], :]
    C_j = amplitudes[theset[2], :]
    S_ij = amplitudes[theset[3], :]
    fringe = Stot + S_ij - C_i - C_j
    allfringe[iset, :, :] = ft.image_asics(all1=fringe)

mm_fringe, ss_fringe = ft.meancut(np.isfinite(allfringe), 3)
rng = ss_fringe

figure('fringes')
for i in xrange(len(allsets)):
    subplot(2, 2, i + 1)
    imshow(allfringe[i, :, :], vmin=0., vmax=1e13, interpolation='nearest')
    title(allsets[i][0])
    colorbar()
