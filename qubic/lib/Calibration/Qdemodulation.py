import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as scsig

import qubic.lib.Calibration.Qfiber as ft
from qubic.lib.Calibration.Qfiber import printnow


class interp_template:
    """
    Allows to interpolate a template applying an amplitude, offset and phase
    This si to be used in order to fit this template in each epriod of some pseudo-periodic signal
    For instance for demodulation
    """

    def __init__(self, xx, yy):
        self.period = xx[-1] - xx[0]
        self.xx = xx
        self.yy = yy

    def __call__(self, x, pars, extra_args=None):
        amp = pars[0]
        off = pars[1]
        dx = pars[2]
        return np.interp(x, self.xx - dx, self.yy, period=self.period) * amp + off


class Demodulation:
    def read_cal_src_data(self, file_list, time_offset=7200):
        ttsrc_i = []
        ddsrc_i = []
        for ff in file_list:
            thett, thedd = np.loadtxt(ff).T
            ttsrc_i.append(thett + time_offset)
            ddsrc_i.append(thedd)
        t_src = np.concatenate(ttsrc_i)
        data_src = np.concatenate(ddsrc_i)

        return t_src, data_src

    def bin_per_period(self, period, time, invec):
        period_index = ((time - time[0]) / period).astype(int)
        allperiods = np.unique(period_index)
        tper = np.zeros(len(allperiods))
        nvec = np.shape(invec)[0]
        newvecs = np.zeros((nvec, len(allperiods)))
        for i in range(len(allperiods)):
            ok = period_index == allperiods[i]
            tper[i] = np.mean(time[ok])
            newvecs[:, i] = np.mean(invec[:, ok], axis=1)

        return tper, newvecs

    def hf_noise_estimate(self, tt, dd):
        sh = np.shape(dd)
        if len(sh) == 1:
            dd = np.reshape(dd, (1, len(dd)))
            ndet = 1
        else:
            ndet = sh[0]
        estimate = np.zeros(ndet)
        for i in range(ndet):
            spectrum_f, freq_f = ft.power_spectrum(tt, dd[i, :], rebin=True)
            mean_level = np.mean(spectrum_f[np.abs(freq_f) > (np.max(freq_f) / 2)])
            samplefreq = 1.0 / (tt[1] - tt[0])
            estimate[i] = np.sqrt(mean_level * samplefreq / 2)

        return estimate

    def return_rms_period(self, period, indata, others=None, verbose=False, remove_noise=False):
        if verbose:
            print("return_rms_period: indata length=", len(indata))
        time = indata[0]
        data = indata[1]

        if verbose:
            printnow("Entering RMS/period")
        if data.ndim == 1:
            nTES = 1
        else:
            sh = np.shape(data)
            nTES = sh[0]

        if verbose:
            print("return_rms_period: nTES=", nTES)

        # We label each data sample with a period
        period_index = ((time - time[0]) / period).astype(int)
        # We loop on periods to measure their respective amplitude and azimuth
        allperiods = np.unique(period_index)
        tper = np.zeros(len(allperiods))
        ampdata = np.zeros((nTES, len(allperiods)))
        err_ampdata = np.zeros((nTES, len(allperiods)))

        if others is not None:
            newothers = plt.bin_per_period(period, time, others)
        if verbose:
            printnow("Calculating RMS per period for {} periods and {} TES".format(len(allperiods), nTES))

        for i in range(len(allperiods)):
            ok = period_index == allperiods[i]
            tper[i] = np.mean(time[ok])
            if nTES == 1:
                mm, ss = ft.meancut(data[ok], 3)
                ampdata[0, i] = ss
                err_ampdata[0, i] = 1
            else:
                for j in range(nTES):
                    mm, ss = ft.meancut(data[j, ok], 3)
                    ampdata[j, i] = ss
                    err_ampdata[j, i] = 1

        if remove_noise:
            hf_noise = self.hf_noise_estimate(time, data)
            var_diff = np.zeros((nTES, len(tper)))
            for k in range(nTES):
                var_diff[k, :] = ampdata[k, :] ** 2 - hf_noise[k] ** 2
            ampdata = np.sqrt(np.abs(var_diff)) * np.sign(var_diff)

        if others is None:
            return tper, ampdata, err_ampdata
        else:
            return tper, ampdata, err_ampdata, newothers

    def fitperiod(self, x, y, fct):
        guess = np.array([np.std(y), np.mean(y), 0.0])
        res = ft.do_minuit(x, y, y * 0 + 1, guess, functname=fct, verbose=False, nohesse=True, force_chi2_ndf=True)
        return res

    def return_fit_period(self, period, indata, others=None, verbose=False, template=None):
        time = indata[0]
        data = indata[1]

        if verbose:
            printnow("Entering Fit/period")
        if data.ndim == 1:
            nTES = 1
        else:
            sh = np.shape(data)
            nTES = sh[0]

        if template is None:
            xxtemplate = np.linspace(0, period, 100)
            yytemplate = -np.sin(xxtemplate / period * 2 * np.pi)
            yytemplate /= np.std(yytemplate)
        else:
            xxtemplate = template[0]
            yytemplate = template[1]
            yytemplate /= np.std(yytemplate)

        fct = interp_template(xxtemplate, yytemplate)  # class of interpolation

        period_index = ((time - time[0]) / period).astype(int)
        allperiods = np.unique(period_index)
        tper = np.zeros(len(allperiods))
        ampdata = np.zeros((nTES, len(allperiods)))
        err_ampdata = np.zeros((nTES, len(allperiods)))

        if others is not None:
            newothers = self.bin_per_period(period, time, others)
        if verbose:
            printnow("Performing fit per period for {} periods and {} TES".format(len(allperiods), nTES))

        for i in range(len(allperiods)):
            ok = period_index == allperiods[i]
            tper[i] = 0.5 * (np.min(time[ok]) + np.max(time[ok]))  # np.mean(time[ok])
            if nTES == 1:
                res = self.fitperiod(time[ok], data[ok], fct)
                ampdata[0, i] = res[1][0]
                err_ampdata[0, i] = res[2][0]
            else:
                for j in range(nTES):
                    res = self.fitperiod(time[ok], data[j, ok], fct)
                    ampdata[j, i] = res[1][0]
                    err_ampdata[j, i] = res[2][0]

        if others is None:
            return tper, ampdata, err_ampdata
        else:
            return tper, ampdata, err_ampdata, newothers

    def demodulate_JC(self, period, indata, indata_src, others=None, verbose=False, template=None, quadrature=False, remove_noise=False):
        time = indata[0]
        data = indata[1]
        sh = data.shape

        if len(sh) == 1:
            data = np.reshape(data, (1, sh[0]))

        time_src = indata_src[0]
        data_src = indata_src[1]
        # print(quadrature)
        if quadrature:
            # ## Shift src data by 1/2 period
            data_src_shift = np.interp(time_src - period / 2, time_src, data_src, period=period)

        # ## Now smooth over a period
        FREQ_SAMPLING = 1.0 / (time[1] - time[0])
        size_period = int(FREQ_SAMPLING * period) + 1
        filter_period = np.ones((size_period,)) / size_period
        demodulated = np.zeros_like(data)
        sh = np.shape(data)
        for i in range(sh[0]):
            if quadrature:
                demodulated[i, :] = scsig.fftconvolve((np.sqrt((data[i, :] * data_src) ** 2 + (data[i, :] * data_src_shift) ** 2)) / np.sqrt(2), filter_period, mode="same")
            else:
                demodulated[i, :] = scsig.fftconvolve(data[i, :] * data_src, filter_period, mode="same")

        # Remove First and last periods
        nper = 4.0
        nsamples = int(nper * period / (time[1] - time[0]))
        timereturn = time[nsamples:-nsamples]
        demodulated = demodulated[:, nsamples:-nsamples]

        if remove_noise:
            hf_noise = self.hf_noise_estimate(time, data) / np.sqrt(2)
            var_diff = np.zeros((sh[0], len(timereturn)))
            for k in range(sh[0]):
                var_diff[k, :] = demodulated[k, :] ** 2 - hf_noise[k] ** 2
            demodulated = np.sqrt(np.abs(var_diff)) * np.sign(var_diff)

        if sh[0] == 1:
            demodulated = demodulated[0, :]

        return timereturn, demodulated, demodulated * 0 + 1

    def demodulate_methods(self, data_in, fmod, fourier_cuts=None, verbose=False, src_data_in=None, method="demod", others=None, template=None, remove_noise=False):
        # Various demodulation methods
        # Others is a list of other vectors (with similar time sampling as the data to demodulate)
        # that we need to sample the same way as the data.

        # To filter the data before demodulation give the array fourier_cuts = [lowcut, highcut, notch].
        # If both lowcut and highcut are given, a bandpass filter is applied.
        # If lowcut is given but highcut = None, a highpass filter at f_cut = lowcut is applied.
        # If highcut is given but lowcut = None, a lowpass filter at f_cut = highcut is applied.
        # If none of them is given, no cut frequency filter is applied.
        # In any case notch filter can still be used. If notch = None, notch filter is not applied.
        #
        if fourier_cuts is None:
            # Duplicate the input data
            data = data_in.copy()
            if src_data_in is None:
                src_data = None
            else:
                src_data = src_data_in.copy()

        else:
            # Filter data and source accordingly
            lowcut = fourier_cuts[0]
            highcut = fourier_cuts[1]
            notch = fourier_cuts[2]
            newtod = ft.filter_data(data_in[0], data_in[1], lowcut, highcut, notch=notch, rebin=True, verbose=verbose)
            data = [data_in[0], newtod]
            if src_data_in is None:
                src_data = None
            else:
                new_src_tod = ft.filter_data(src_data_in[0], src_data_in[1], lowcut, highcut, notch=notch, rebin=True, verbose=verbose)
                src_data = [src_data_in[0], new_src_tod]

            # Now we have the "input" data, we can start demodulation
        period = 1.0 / fmod

        if method == "rms":
            # RMS method: calculate the RMS in each period (beware ! it returns noise+signal !)
            return self.return_rms_period(period, data, others=others, verbose=verbose, remove_noise=remove_noise)
        elif method == "fit":
            return self.return_fit_period(period, data, others=others, verbose=verbose, template=template)
        elif method == "demod":
            return self.demodulate_JC(period, data, src_data, others=others, verbose=verbose, template=None)
        elif method == "demod_quad":
            return self.demodulate_JC(period, data, src_data, others=others, verbose=verbose, template=None, quadrature=True, remove_noise=remove_noise)
        elif method == "absolute_value":
            return np.abs(data)
