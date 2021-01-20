from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits as pyfits
from matplotlib.backends.backend_pdf import PdfPages
import scipy.optimize as sop
from scipy.signal import resample

from qubicpack.qubicfp import qubicfp
import qubic.fibtools as ft
from qubic import selfcal_lib as scal

__all__ = ['FringesAnalysis', 'SaveFringesFitsPdf']


class FringesAnalysis:
    def __init__(self, datafolder, date, q, baseline, ncycles=20, stable_time=5.,
                 asics=[1, 2], src_data=False, subtract_t0=True, cut=False, t0cut=None, tfcut=None,
                 refTESnum=95, refASICnum=1, allh=[True, False, False, True, False, False], nsp_per=240,
                 lowcut=1e-5, highcut=2., nbins=120, notch=np.array([[1.724, 0.005, 10]]),
                 verbose=False):
        """
        Parameters
        ----------
        datafolder: str
            Folder containing the data.
        date: str
            Date of the measurement.
        q: a QubicInstrument
        baseline: list
            The 2 horns.
        ncycles: int
            Number of cycles run during the data taking.
        stable_time: float
            Waiting time [s] on each step.
        asics: list
            ASIC numbers.
        src_data: bool
            If True, will return the source data as well
        subtract_t0: bool,
            If True, will remove to the time the first time element when getting data.
        cut: bool
            If True, it will cut the data at t0cut, tfcut.
        t0cut, tfcut: float
            Start and end time to cut the TODs.
        refTESnum, refASICnum: int
            One reference TES to check the period.
        allh: list
            True on the steps where all horns are open, False elsewhere.
        nsp_per: int
            Number of sample per period for resampling.
        lowcut, highcut: float
            Low and high cut for filtering.
        nbins: int
            Number of bins for filtering.
        notch: array
            Define a notch filter.
        verbose: bool
        """
        self.datafolder = datafolder
        self.date = date
        self.q = q
        self.baseline = baseline
        self.ncycles = ncycles
        self.asics = asics
        self.src_data = src_data
        self.subtract_t0 = subtract_t0
        self.refTESnum = refTESnum
        self.refASICnum = refASICnum
        self.stable_time = stable_time
        self.allh = allh
        self.nsp_per = nsp_per
        self.lowcut = lowcut
        self.highcut = highcut
        self.nbins = nbins
        self.notch = notch
        self.verbose = verbose

        # Get data
        self.tdata, self.data, self.tsrc, self.dsrc = self._get_data()

        self.nasics, self.ndet_oneASIC, _ = self.data.shape
        self.ndet = self.nasics * self.ndet_oneASIC
        self.nsteps = len(self.allh)
        self.expected_period = self.stable_time * self.nsteps

        self.cut = cut
        self.t0cut = t0cut
        self.tfcut = tfcut
        if cut:
            print('Cutting the data from t0cut to tfcut.')
            self.tdata, self.data = cut_data(self.t0cut, self.tfcut, self.tdata, self.data)
            # The number of cycles need to be updated
            self.ncycles = self.tdata // (self.stable_time * self.nsteps)

        # Reference period determine on reference TES
        _, _, self.refperiod = self.find_right_period(self.refTESnum, self.refASICnum)
        self.refstable_time = self.refperiod / self.nsteps

        # Get coordinates for all TES, order as on the instrument
        self.xTES, self.yTES, _, self.allindex_q = scal.get_TES_Instru_coords(self.q, frame='ONAFP', verbose=False)

    def _get_data(self):
        """
        Get the TODs for one ASIC.
        Returns
        -------
        Time and signal for all TES in one ASIC.
        If src_data is True: will also return the  source time and signal
        """

        # Qubicpack object
        a = qubicfp()
        a.verbosity = 0
        a.read_qubicstudio_dataset(datadir=self.datafolder)

        # TOD from all ASICS
        data = []
        tdata = []
        for i, ASIC in enumerate(self.asics):
            ASIC = int(ASIC)
            data_oneASIC = a.timeline_array(asic=ASIC)
            data.append(data_oneASIC)
            tdata_oneASIC = a.timeaxis(datatype='science', asic=ASIC)
            if self.subtract_t0:
                tdata_oneASIC -= tdata_oneASIC[0]
            tdata.append(tdata_oneASIC)
        tdata = np.array(tdata)
        data = np.array(data)

        if self.src_data:  # Get calibration source data
            tsrc = a.calsource()[0]
            if self.subtract_t0:
                tsrc -= tsrc[0]
            dsrc = a.calsource()[1]
        else:
            tsrc = None
            dsrc = None

        return tdata, data, tsrc, dsrc

    def plot_TOD(self, ASIC, TES, xlim=None, figsize=(12, 6)):
        idx = (ASIC - 1) * 128 + (TES - 1)
        plt.figure(figsize=figsize)
        ax = plt.gca()
        ax.plot(self.tdata, self.data[idx, :])
        ax.set_title(f'TOD for TES {TES} - ASIC {ASIC}')
        if xlim is not None:
            ax.set_xlim(0, xlim)
        plt.show()

        if self.dsrc is not None:
            plt.figure(figsize=figsize)
            ax = plt.gca()
            ax.plot(self.tsrc, self.dsrc[idx, :])
            ax.set_title('Calibration source data')
            if xlim is not None:
                ax.set_xlim(0, xlim)
            plt.show()
        return

    def find_right_period(self, TES, ASIC, filtering=True, delta=0.5, nb=100):
        """

        Parameters
        ----------
        TES, ASIC: int
            TES and ASIC numbers.
        filtering: bool
            If True, data are filtered.
        delta
        nb

        Returns
        -------

        """
        if filtering:
            thedata = ft.filter_data(self.tdata[ASIC - 1, :],
                                     self.data[ASIC - 1, TES - 1, :],
                                     self.lowcut, self.highcut, notch=self.notch, rebin=True)
        else:
            thedata = self.data[ASIC - 1, TES - 1, :]
        ppp = np.linspace(self.expected_period - delta, self.expected_period + delta, nb)
        rms = np.zeros(len(ppp))
        for i in range(len(ppp)):
            xin = self.tdata[ASIC - 1, :] % ppp[i]
            xx, yy, dx, dy, o = ft.profile(xin, thedata, nbins=100, plot=False)
            rms[i] = np.std(yy)
        period = ppp[np.argmax(rms)]

        if self.verbose:
            print('Found period {0:5.3f}s on TES {1:} ASIC {1:}'.format(period, TES, ASIC))
            print('Expected : ', self.expected_period)

        return ppp, rms, period

    def analyse_fringesLouise(self, return_folding=False, median=False, doplot=True):
        """
        Parameters
        ----------

        Returns
        -------
        Time, folded signal, the 8 parameters estimated with the fit,
        the combination of the amplitudes, the period and the residuals
        between the fit and the signal. They are computed for each TES.
        """

        fringes1D = np.zeros(self.ndet)
        err_fringes1D = np.zeros(self.ndet)
        params = np.zeros((self.ndet, self.nsteps + 2))
        err_params = np.zeros_like(params)
        datafold = np.zeros((self.ndet, self.nbins))
        err_datafold = np.zeros_like(datafold)
        residuals_time = np.zeros_like(datafold)

        weights = np.array([1. / 3, -1., 1., 1. / 3, -1., 1. / 3])
        for i, ASIC in enumerate(self.asics):
            # Filter, fold and normalize the data
            dfold, tfold, _, errfold, _, _ = ft.fold_data(self.tdata[i, :],
                                                          self.data[i, :, :],
                                                          self.refperiod,
                                                          self.lowcut,
                                                          self.highcut,
                                                          self.nbins,
                                                          notch=self.notch,
                                                          median=median,
                                                          return_error=True,
                                                          silent=self.verbose
                                                          )

            datafold[self.ndet_oneASIC * i:self.ndet_oneASIC * (i + 1), :] = dfold
            err_datafold[self.ndet_oneASIC * i:self.ndet_oneASIC * (i + 1), :] = errfold

            # Fit (Louise method)
            param_guess = [0.1, 0.] + [1.] * self.nsteps
            if self.verbose:
                print('Guess parameters:', param_guess)
            for j in range(self.ndet_oneASIC):
                index = self.ndet_oneASIC * i + j
                # With curve_fit, it is not possible to have 'args'
                popt, pcov = sop.curve_fit(lambda tfold, ctime, tstart, a0, a1, a2, a3, a4, a5:
                                           ft.simsig_fringes(tfold,
                                                             self.refstable_time,
                                                             [ctime, tstart, a0, a1, a2, a3, a4, a5]),
                                           tfold,
                                           dfold[j, :],
                                           p0=param_guess,
                                           sigma=errfold[j, :],
                                           absolute_sigma=True,
                                           bounds=([0., -2, -2, -2, -2, -2, -2, -2],
                                                   [1., 2, 2, 2, 2, 2, 2, 2])
                                           )
                params[index, :] = popt
                err_params[index, :] = np.sqrt(np.diag(pcov))

                # Combination to get fringes
                fringes1D[index], err_fringes1D[index] = weighted_sum(params[index, 2:], err_params[index, 2:], weights)

                residuals_time[index, :] = dfold[j, :] - ft.simsig_fringes(tfold, self.refstable_time, popt)
        if doplot:
            # Fringes
            fig, axs = plt.subplots(1, 2, figsize=(13, 7))
            fig.subplots_adjust(wspace=0.5)
            fig.suptitle(f'Fringes with Louise method - BL {self.baseline} - {self.date}')
            ax0, ax1 = axs.ravel()
            plot_fringes_imshow_interp(fringes1D, fig=fig, ax=ax0)
            plot_fringes_scatter(self.q, self.xTES, self.yTES, fringes1D, s=80, fig=fig, ax=ax1)

            # Errors
            plot_fringes_errors(self.q, fringes1D, err_fringes1D, self.xTES, self.yTES, s=80,
                                normalize=True, lim=1., suptitle=f'Errors - BL {self.baseline} - {self.date}')

            # Folding
            plot_folding_fit(self.refTESnum, self.refASICnum, tfold, datafold, residuals_time,
                             self.refperiod, params, err_params)

        if return_folding:
            return params, err_params, fringes1D, err_fringes1D, tfold, datafold, err_datafold, residuals_time
        else:
            return params, err_params, fringes1D, err_fringes1D

    def _make_w_Michel(self, t, tm1=12, tm2=2, ph=5):
        # w is made to make the combination to see fringes with Michel's method
        w = np.zeros_like(t)
        wcheck = np.zeros_like(t)
        for i in range(len(w)):
            if (((i - ph) % self.refperiod) >= tm1) and (((i - ph) % self.refperiod) < self.refperiod - tm2):
                if (((i - ph) // self.refperiod) == 0) | (((i - ph) // self.refperiod) == 3):
                    w[i] = 1.
                if (((i - ph) // self.refperiod) == 1) | (((i - ph) // self.refperiod) == 2):
                    w[i] = -1.

        return w, wcheck

    def analyse_fringes_Michel(self, median=False, verbose=True, doplot=True):
        """
        Compute the fringes with Michel's method.
        """
        fringes1D = np.zeros(self.ndet)
        dfold = np.zeros((self.ndet, self.nbins))

        for i, ASIC in enumerate(self.asics):

            # Fold and filter the data
            fold, tfold, _, _ = ft.fold_data(self.tdata[i, :],
                                             self.data[i, :, :],
                                             self.refperiod,
                                             self.lowcut,
                                             self.highcut,
                                             self.nbins,
                                             notch=self.notch,
                                             median=median,
                                             silent=verbose,
                                             )
            dfold[self.ndet_oneASIC * i:self.ndet_oneASIC * (i + 1), :] = fold

            # Michel method
            w, _ = self._make_w_Michel(tfold)
            for j in range(self.ndet_oneASIC):
                index = self.ndet_oneASIC * i + j
                fringes1D[index] = np.sum(fold[j, :] * w)

        if doplot:
            # Fringes
            fig, axs = plt.subplots(1, 2, figsize=(13, 7))
            fig.subplots_adjust(wspace=0.5)
            fig.suptitle(f'Fringes with Michel method - BL {self.baseline} - {self.date}')
            ax0, ax1 = axs.ravel()
            plot_fringes_imshow_interp(fringes1D, fig=fig, ax=ax0)
            plot_fringes_scatter(self.q, self.xTES, self.yTES, fringes1D, s=80, fig=fig, ax=ax1)

        return tfold, dfold, fringes1D

    def find_high_derivative(self, signal):
        """Calculate the derivative of a signal and find where it is high."""
        dsignal = np.abs(np.gradient(signal))
        md, sd = ft.meancut(dsignal, 3)
        thr = np.abs(dsignal - md) > (3 * sd)
        return dsignal, thr

    def find_high_derivative_clusters(self, tfold, thr, period):
        """ Find clusters of high derivatives: each time we take the first high derivative element."""

        t_change = tfold[thr]
        expected_stable_time = period / self.nsteps
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
        return start_times

    def find_t0(self, period, doplot=False):
        """
        Find time where configuration change in the square modulation using the reference TES.
        """
        # Resample and fold
        timeTES = self.tdata[self.refASICnum - 1, :]
        dataTES = self.data[self.refASICnum - 1, self.refTESnum - 1, :]
        tfold, dfold = self.resample_fold_oneTES(timeTES, dataTES, period, doplot=False)

        # Average the signal over all periods
        msignal = np.mean(dfold, axis=0)

        # Calculate the derivative and find where it is high
        dsignal, thr = self.find_high_derivative(msignal)

        # Find clusters of high derivatives
        start_times = self.find_high_derivative_clusters(tfold, thr, period)

        # Now we take the median of all start_times modulo period/nsteps
        expected_stable_time = period / self.nsteps
        t0 = np.median(start_times % expected_stable_time)

        if doplot:
            self._plot_finding_t0(tfold, msignal, dsignal, thr, start_times, t0)

        return t0

    def resample_fold_oneTES(self, timeTES, dataTES, period, doplot=True):

        # First Step: Data Filtering
        dfilter = ft.filter_data(timeTES, dataTES, self.lowcut, self.highcut, notch=self.notch, rebin=True)

        # Crop the data in order to have an integer number of periods
        tcrop, dcrop, _ = cut_data_Nperiods(None, None, timeTES, dfilter, period)

        # Resample the signal
        newtime = np.linspace(tcrop[0], tcrop[-1], self.ncycles * self.nsp_per)
        newdata = resample(dcrop, self.ncycles * self.nsp_per)

        # Fold the data
        tfold = np.linspace(0, period, self.nsp_per)
        dfold = np.reshape(newdata, (self.ncycles, self.nsp_per))

        if doplot:
            plt.figure(figsize=(8, 6))
            plt.plot(newtime, newdata)
            plt.xlabel('Time')
            plt.ylabel('ADU')
            plt.title('TODs crop and resample')

            self._plot_foldingJC(tfold, dfold, period, suptitle='Data resample and fold')

        return tfold, dfold

    def roll_oneTES(self, dfold, t0, period=None):
        """Shift the folded data in order to have t0=0."""

        droll = np.roll(dfold, -int(t0 / period * self.nsp_per), axis=1)
        return droll

    def remove_median_allh(self, tfold, droll, period, skip_rise=0., skip_fall=0., doplot=True):
        # Roughly remove the median of the all_h configurations
        ok_all_horns = np.zeros_like(tfold, dtype=bool)
        for i in range(self.nsteps):
            if self.allh[i]:
                tmini = i * period / self.nsteps + skip_rise * period / self.nsteps
                tmaxi = (i + 1) * period / self.nsteps - skip_fall * period / self.nsteps
                ok = (tfold >= tmini) & (tfold < tmaxi)
                ok_all_horns[ok] = True
        droll_rm_median = droll - np.median(droll[:, ok_all_horns])

        if doplot:
            self._plot_foldingJC(tfold, droll_rm_median, period, skip_rise=skip_rise, skip_fall=skip_fall,
                                 suptitle='Data fold after rolling and removing allh average')

        return droll_rm_median

    def remove_slope(self, m_points, std_points):
        """Fit a slope between the "all horns open" configurations and remove it."""
        xx = np.arange(self.nsteps)
        for i in range(self.ncycles):
            # Linear fit
            pars = np.polyfit(x=np.arange(self.nsteps)[self.allh],
                              y=m_points[i, self.allh],
                              deg=1,
                              w=1. / std_points[i, self.allh] ** 2,
                              full=False,
                              cov=False)
            m_points[i, :] = m_points[i, :] - (pars[0] * xx + pars[1])

        return m_points

    def average_each_period_oneTES(self, tfold, droll, period, skip_rise=0., skip_fall=0.,
                                   median=True, remove_slope=False):

        # We assume that the array has been np.rolled so that the t0 is in time sample 0
        stable_time = period / self.nsteps

        # Remove the average of each period
        droll = (droll.T - np.mean(droll, axis=1)).T

        # Perform first an average/median in each step of each cycle over the points
        # (possibly skipping beginning and end)
        m_points = np.zeros((self.ncycles, self.nsteps))
        std_points = np.zeros((self.ncycles, self.nsteps))
        for i in range(self.nsteps):
            # Cut the data
            tstart = i * stable_time + skip_rise * stable_time
            tend = (i + 1) * stable_time - skip_fall * stable_time
            ok = (tfold >= tstart) & (tfold < tend)
            for j in range(self.ncycles):
                if median:
                    m_points[j, i] = np.median(droll[j, ok])
                else:
                    m_points[j, i], _ = ft.meancut(droll[j, ok], 3)
                std_points[j, i] = np.std(droll[j, ok])

        if remove_slope:
            # Fit a slope between the "all horns open" configurations and remove it.
            m_points = self.remove_slope(m_points, std_points)
        else:
            # Remove the average off "all horns open configurations" for each period
            for i in range(self.ncycles):
                m_points[i, :] -= np.mean(m_points[i, self.allh])

        return m_points, std_points

    def average_all_periods_oneTES(self, droll, m_points, std_points, median=True,
                                   speak=False, doplot=False):
        """
        Calculating the average in each bin over periods in various ways:
            1/ We can use the whole flat section or cut a bit at the beginning and at the end
            2/ Simple average
            3/ more fancy stuff: removing a slope determined by asking the 3 measurements of "all horns" to be equal
        Parameters
        ----------
        droll:
        m_points, std_points:
        median: bool
            If True, takes the median and not the mean on each folded step.
        remove_slope: bool
            If True, fit a slope between the "all horns open" configurations and remove it
        speak: bool
        doplot: bool

        Returns
        -------

        """

        # Average or median over all periods
        vals = np.zeros(self.nsteps)
        errs = np.zeros(self.nsteps)
        for i in range(self.nsteps):
            if median:
                vals[i] = np.median(m_points[:, i])
            else:
                vals[i] = np.mean(m_points[:, i])
            # errs[i] = 1./self.ncycles * np.sqrt(self.nsteps / self.nsp_per) \
            #           * np.sqrt(np.sum(m_points[:, i]**2))
            errs[i] = np.std(m_points[:, i])

        # Residuals in time domain (not too relevant as some baselines were removed
        # as a result, large fluctuations in time-domain are usually well removed)
        newdroll = np.zeros_like(droll)
        for i in range(self.nsteps):
            newdroll[:, i * self.nsp_per // self.nsteps:(i + 1) * self.nsp_per // self.nsteps] = vals[i]
        residuals_time = droll - newdroll

        # We would rather calculate the relevant residuals in the binned domain
        # between the final values and those after levelling
        residuals_bin = np.ravel(m_points - vals)
        _, sigres = ft.meancut(residuals_bin, 3)

        if speak:
            for i in range(self.nsteps):
                print('############')
                print('Step {}'.format(i+1))
                for j in range(self.ncycles):
                    print('per {}: {} +/- {}'.format(j, m_points[j, i], std_points[j, i]))
                print('============')
                print('Value {} +/- {}'.format(vals[i], errs[i]))
                print('============')

        if doplot:
            self._plot_average_foldedTES(m_points, std_points, droll, newdroll, residuals_time,
                                         vals, errs, residuals_bin)

        return vals, errs, sigres

    def analyse_fringesJC(self, skip_rise=0.2, skip_fall=0.1, median=True, remove_slope=False,
                          force_period=None, force_t0=None, doplot=True):

        # ================= Determine the correct period reference TES ========
        if force_period is None:
            period = self.refperiod
            if self.verbose:
                print('Using reference period {0:5.3f}s'.format(self.refperiod))
        else:
            period = force_period
            if self.verbose:
                print('Using Forced period {0:5.3f}s'.format(period))

        # =============== Determine t0 on reference TES ======================
        if force_t0 is None:
            t0 = self.find_t0(period, doplot=doplot)
            if self.verbose:
                print('Found t0 {0:5.3f}s on TES {1:}, ASIC {1:}'.format(t0, self.refTESnum, self.refASICnum))
        else:
            t0 = force_t0
            if self.verbose:
                print('Using forced t0 {0:5.3f}s'.format(t0))

        # =============== Loop on ASICs and TES ======================
        m_points = np.zeros((self.ndet, self.ncycles, self.nsteps))
        std_points = np.zeros((self.ndet, self.ncycles, self.nsteps))
        vals = np.zeros((self.ndet, self.nsteps))
        errs = np.zeros((self.ndet, self.nsteps))
        sigres = np.zeros(self.ndet)
        fringes1D = np.zeros(self.ndet)
        err_fringes1D = np.zeros(self.ndet)
        weights = np.array([1. / 3, -1, 1, 1. / 3, -1, 1. / 3])
        for i, ASIC in enumerate(self.asics):
            print(f'*********** Starting ASIC {ASIC} **************')
            for j, TES in enumerate(np.arange(1, 129)):

                index = i * self.ndet_oneASIC + j
                if (i == (self.refASICnum - 1)) & (j == (self.refTESnum - 1)):
                    speak = True
                    thedoplot = True * doplot
                else:
                    speak = False
                    thedoplot = False

                timeTES = self.tdata[i, :]
                dataTES = self.data[i, j, :]
                # Fold, roll and remove the median of allh
                tfold, dfold = self.resample_fold_oneTES(timeTES, dataTES, period=period, doplot=thedoplot)
                droll = self.roll_oneTES(dfold, t0, period=period)
                droll_rm_median = self.remove_median_allh(tfold, droll, period,
                                                          skip_rise=skip_rise, skip_fall=skip_fall,
                                                          doplot=thedoplot)

                # Average each period
                m_points[index, :, :], std_points[index, :, :] = self.average_each_period_oneTES(tfold, droll_rm_median, period,
                                                                     skip_rise=skip_rise, skip_fall=skip_fall,
                                                                     median=median, remove_slope=remove_slope)
                # Average over all periods
                vals[index, :], errs[index, :], sigres[index] = self.average_all_periods_oneTES(droll_rm_median,
                                                                                                m_points[index, :, :],
                                                                                                std_points[index, :, :],
                                                                                                median=median,
                                                                                                speak=speak,
                                                                                                doplot=thedoplot)
                # Combination and get fringes
                fringes1D[index], err_fringes1D[index] = weighted_sum(vals[index, :], errs[index, :], weights)

        # Get the good TES from a cut on residuals
        mm, ss = ft.meancut(np.log10(sigres), 3)
        oktes = np.ones(self.ndet)
        oktes[np.abs(np.log10(sigres) - mm) > 2 * ss] = np.nan

        if doplot:
            # Fringes
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            fig.subplots_adjust(wspace=0.5)
            fig.suptitle(f'Fringes with JC method - BL {self.baseline} - {self.date}')
            ax0, ax1 = axs.ravel()
            plot_fringes_imshow_interp(fringes1D, fig=fig, ax=ax0)
            plot_fringes_scatter(self.q, self.xTES, self.yTES, fringes1D, s=80, fig=fig, ax=ax1)

            # Error
            plot_fringes_errors(self.q, fringes1D, err_fringes1D, self.xTES, self.yTES,
                                s=80, normalize=True, lim=1., frame='ONAFP')

            # TOD Residuals
            plot_residuals(self.q, sigres, oktes, self.xTES, self.yTES)

        return vals, errs, sigres, fringes1D, err_fringes1D, oktes, m_points, std_points

    # ================ Plot functions very specific to JC analysis
    def _plot_finding_t0(self, tfold, msignal, dsignal, thr, start_times, t0, figsize=(12, 6)):
        plt.figure(figsize=figsize)
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
            plt.axvline(x=t0 + i * self.stable_time, color='r', ls='--', label=lab)
        plt.legend(framealpha=0.2)
        plt.title('t0 determination on Reference TES')
        plt.xlabel('Time in Period')
        plt.ylabel('Signal averaged over periods')
        plt.tight_layout()

        return

    def _plot_foldingJC(self, tfold, dfold, period, skip_rise=None, skip_fall=None, suptitle='', figsize=(12, 6)):

        fig, axs = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(suptitle)
        ax1, ax2 = np.ravel(axs)

        ax1.imshow(dfold,
                   origin='lower',
                   aspect='auto',
                   extent=[0, np.max(tfold) + (tfold[1] - tfold[0]) / 2, 0, self.ncycles + 0.5])
        for i in range(self.nsteps):
            ax1.axvline(x=i * (period / self.nsteps), color='k', lw=3)
        ax1.set_xlabel('Time in period')
        ax1.set_ylabel('Period #')

        for i in range(self.ncycles):
            ax2.plot(tfold, dfold[i, :], alpha=0.5)
        for i in range(self.nsteps):
            ax2.axvline(x=i * (period / self.nsteps), color='k', lw=3)
            if skip_rise is not None:
                ax2.axvspan(i * (period / self.nsteps),
                            (i + skip_rise) * (period / self.nsteps),
                            alpha=0.1, color='red')
                ax2.axvspan((i + (1. - skip_fall)) * (period / self.nsteps),
                            (i + 1) * (period / self.nsteps),
                            alpha=0.1, color='red')
        ax2.set_xlabel('Time in period')
        ax2.set_ylabel('ADU')

        return

    def _plot_average_foldedTES(self, m_points, std_points,
                                dfold, newdfold, residuals_time,
                                vals, errs, residuals_bin,
                                suptitle=None, figsize=(12, 10)):
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(suptitle)
        ax1, ax2, ax3, ax4 = np.ravel(axs)

        ttt = np.arange(self.nsteps) * self.stable_time + self.stable_time / 2  # Mean time of each step

        for i in range(self.ncycles):
            if i == 0:
                lab = 'Raw'
            else:
                lab = None
            ax1.errorbar(ttt, m_points[i, :],
                         yerr=std_points[i, :],
                         xerr=self.stable_time / 2, fmt='o', label=lab)
        ax1.set_title('Configuration bins before levelling per period')
        ax1.set_xlabel('Time in period')
        ax1.set_ylabel('Value for each period')
        ax1.legend()

        ax2.plot(np.ravel(dfold), label='Input signal')
        ax2.plot(np.ravel(newdfold), label='Reconstructed')
        ax2.plot(np.ravel(residuals_time), label='Residuals')
        ax2.set_xlabel('time samples')
        ax2.set_ylabel('Time domain signal')
        ax2.set_title('Time domain \n[large drift is actually removed]')
        ax2.legend()

        ax3.plot(np.ravel(m_points), ls='solid', label='Per Period')
        ax3.plot(np.ravel(m_points * 0. + vals), ls='solid', label='Values')
        ax3.plot(residuals_bin, ls='solid', label='Residuals')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Values')
        ax3.set_title('Final Residuals')
        ax3.legend()

        for i in range(self.ncycles):
            if i == 0:
                lab = 'Values for each period'
            else:
                lab = None
            ax4.errorbar(ttt, m_points[i, :], yerr=std_points[i, :],
                         xerr=self.stable_time / 2, fmt='x', alpha=0.3, color='orange', label=lab)
        ax4.errorbar(ttt, vals, yerr=errs, xerr=self.stable_time / 2, color='r',
                     label='Final Points', fmt='rx')
        ax4.set_title('Final Configurations (after levelling)')
        ax4.set_xlabel('Time in period')
        ax4.set_ylabel('Value')
        ax4.legend()

        return


# =========================================
class SaveFringesFitsPdf:
    def __init__(self, q, date_obs, allBLs, allstable_time, allNcycles, xTES, yTES, allfringes1D, allerr_fringes1D,
                 allokTES=None, allsigres=None, nsteps=6, ecosorb='yes', frame='ONAFP'):
        self.q = q
        self.date_obs = date_obs
        self.allBLs = allBLs
        self.nBLs = len(allBLs)
        self.BLs_sort, self.BLs_type = scal.find_equivalent_baselines(self.allBLs, self.q)
        self.allstable_time = allstable_time
        self.allNcycles = allNcycles
        self.xTES = xTES
        self.yTES = yTES
        self.allfringes1D = allfringes1D
        self.allerr_fringes1D = allerr_fringes1D
        self.allokTES = allokTES
        self.allsigres = allsigres
        self.nsteps = nsteps
        self.ecosorb = ecosorb
        self.frame = frame
        self.keyvals = self._make_keyvals()
        self.fdict = self._make_fdict()

    def _make_keyvals(self):
        """
        Make a dictionary with relevant information on the measurement.
        Assign the FITS keyword values for the primary header
        """
        keyvals = {'DATE-OBS': (self.date_obs, 'Date of the measurement'),
                   'NBLS': (self.nBLs, 'Number of baselines'),
                   'NSTEP': (self.nsteps, 'Number of stable steps per cycle'),
                   'ECOSORD': (self.ecosorb, 'Ecosorb on the source'),
                   'FRAME': (self.frame, 'Referential frame for (X, Y) TES')}

        return keyvals

    def _make_fdict(self):
        """ Make a dictionary with all relevant data."""
        fdict = {'BLS': self.allBLs, 'Stable_time': self.allstable_time, 'NCYCLES': self.allNcycles,
                 'X_TES': self.xTES, 'Y_TES': self.yTES, 'FRINGES_1D': self.allfringes1D,
                 'ERRORS': self.allerr_fringes1D, 'OK_TES': self.allokTES, 'SIGMA_RES': self.allsigres}

        return fdict

    def save_fringes_pdf_plots(self, out_dir, save_name=None, mask=None):
        """Save all the fringe plots (all baselines) in a pdf file."""
        if type(self.nsteps) is tuple:
            for i in self.keyvals:
                self.keyvals[i] = self.keyvals[i][0]
        if save_name is None:
            save_name = 'Fringes_' + self.date_obs + f'_{self.nBLs}BLs.pdf'

        with PdfPages(out_dir + save_name) as pp:
            for i in range(self.nBLs):
                fig, axs = plt.subplots(1, 2, figsize=(13, 7))
                fig.subplots_adjust(wspace=0.5)
                fig.suptitle(f'Fringes - BL {self.allBLs[i]} - {self.date_obs} - type {self.BLs_type[i]}')
                ax0, ax1 = axs.ravel()
                plot_fringes_imshow_interp(self.allfringes1D[i], fig=fig, ax=ax0, mask=mask)
                plot_fringes_scatter(self.q, self.xTES, self.yTES, self.allfringes1D[i], s=80, fig=fig, ax=ax1)
                pp.savefig()
        return

    def write_fits_fringes(self, out_dir, save_name=None):
        """ Save a .fits with the fringes data."""
        if out_dir[-1] != '/':
            out_dir += '/'

        if save_name is None:
            save_name = 'Fringes_' + self.date_obs + f'_{self.nBLs}BLs.fits'

        # Header creation
        hdr = pyfits.Header()
        for key in self.keyvals.keys():
            hdr[key] = (self.keyvals[key])

        hdu_prim = pyfits.PrimaryHDU(header=hdr)
        allhdu = [hdu_prim]
        for key in self.fdict.keys():
            hdu = pyfits.ImageHDU(data=self.fdict[key], name=key)
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


# ============== Tool functions ==============
def cut_data(t0, tf, tdata, data):
    """
    Cut the TODs from t0 to tf.
    They can be None if you do not want to cut the beginning or the end.
    """
    if t0 is None:
        t0 = tdata[0]
    if tf is None:
        tf = tdata[-1]

    ok = (tdata >= t0) & (tdata <= tf)
    tdata_cut = tdata[ok]
    data_cut = data[:, ok]

    return tdata_cut, data_cut


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


def weighted_sum(vals, errs, weights):
    """Weighted sum and its error"""
    sum = np.sum(weights * vals)
    sigma = np.sqrt(np.sum(weights ** 2 * errs ** 2))
    return sum, sigma


def make_mask2D_thermometers_TD():
    mask_thermos = np.ones((17, 17))
    mask_thermos[0, 12:] = np.nan
    mask_thermos[1:5, 16] = np.nan
    return mask_thermos


def remove_thermometers(x, y, fringes1D):
    """Remove the 8 thermometers.
    Returns 1D arrays with 248 values and not 256."""
    fringes1D = fringes1D[x != 0.]
    x = x[x != 0.]
    y = y[y != 0.]
    return x, y, fringes1D


def reorder_data(data, xdata, ydata, xqsoft, yqsoft):
    """Reorder data (TES signal) as ordered in the QUBIC soft.
    The number of TES should be 248 (without thermometers)."""
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


# ============== Plot functions ==============
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
    ntype = np.max(BLs_type) + 1  # Number of equivalency types

    for j in range(ntype):
        images = np.array(fringes2D)[BLs_type == j]
        neq = len(BLs_sort[j])  # Number of equivalent baselines for that type
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


def plot_fringes_scatter(q, xTES, yTES, fringes1D, normalize=True, frame='ONAFP', fig=None, ax=None,
                         cbar=True, lim=1., cmap='bwr', s=None, title='Scatter plot'):
    x, y, fringes = remove_thermometers(xTES, yTES, fringes1D)

    if normalize:
        fringes /= np.nanstd(fringes)

    if ax is None:
        fig, ax = plt.subplots()
    scal.scatter_plot_FP(q, x, y, fringes, frame,
                         fig=fig, ax=ax,
                         s=s,
                         title=title,
                         unit='',
                         cmap=cmap,
                         vmin=-lim, vmax=lim,
                         cbar=cbar
                         )
    return


def plot_fringes_imshow_interp(fringes1D, normalize=True, interp='Gaussian', mask=None,
                               fig=None, ax=None, cbar=True, lim=1., cmap='bwr', title='Imshow'):
    # Make the 2D fringes
    fringes2D = ft.image_asics(all1=fringes1D)
    if normalize:
        fringes2D /= np.nanstd(fringes2D)

    if mask is None:
        mask = make_mask2D_thermometers_TD()

    if ax is None:
        fig, ax = plt.subplots()
    img = ax.imshow(np.nan_to_num(fringes2D * mask),
                    vmin=-lim, vmax=lim,
                    cmap=cmap,
                    interpolation=interp)
    # ft.qgrid()
    ax.set_title(title, fontsize=14)
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(img, cax=cax)
    return


def plot_folding_fit(TES, ASIC, tfold, datafold, residuals_time, period,
                     params, errs, allh=[True, False, False, True, False, True]):
    idx = (ASIC - 1) * 128 + (TES - 1)
    nsteps = len(params[idx, 2:])
    amps = params[idx, 2:]
    t0 = params[idx, 1]
    stable_time = period / nsteps
    print(stable_time)
    mean_allh = np.mean(amps[allh])

    plt.figure()
    plt.plot(tfold, datafold[idx, :], label='Folded signal')
    plt.errorbar(np.arange(0, period, period / nsteps), amps, yerr=errs[idx, 2:],
                 fmt='o', color='r', label='Fit Amplitudes')
    plt.plot(tfold, ft.simsig_fringes(tfold, period / nsteps, params[idx, :]),
             label='Fit')
    plt.plot(tfold, residuals_time[idx, :],
             label='Residuals: RMS={0:6.4f}'.format(np.std(residuals_time[idx, :])))
    for k in range(nsteps):
        plt.axvline(x=stable_time * k + t0, color='k', ls=':', alpha=0.3)
    plt.axhline(mean_allh, color='k', linestyle='--', label='Mean all open')
    plt.legend(loc='upper right')
    plt.xlabel('Time [s]')
    plt.ylabel('TOD')
    plt.title(f'TES {TES} - ASIC {ASIC}')
    plt.grid()
    plt.ylim(-2.5, 2.5)
    return


def plot_residuals(q, sigres, oktes, xTES, yTES, frame='ONAFP', suptitle=None):
    _, _, sigres = remove_thermometers(xTES, yTES, sigres)
    xTES, yTES, oktes = remove_thermometers(xTES, yTES, oktes)

    mm, ss = ft.meancut(np.log10(sigres), 3)

    fig, axs = plt.subplots(1, 3)
    fig.suptitle(suptitle)
    fig.subplots_adjust(wspace=0.5)
    ax0, ax1, ax2 = axs

    ax0.hist(np.log10(sigres), bins=20, label='{0:5.2f} +/- {1:5.2f}'.format(mm, ss))
    ax0.axvline(x=mm, color='r', ls='-', label='Mean')
    ax0.axvline(x=mm - ss, color='r', ls='--', label='1 sigma')
    ax0.axvline(x=mm + ss, color='r', ls='--')
    ax0.axvline(x=mm - 2 * ss, color='r', ls=':', label='2 sigma')
    ax0.axvline(x=mm + 2 * ss, color='r', ls=':')
    ax0.set_xlabel('np.log10(TOD Residuals)')
    ax0.set_title('Histogram residuals')
    ax0.legend()

    scal.scatter_plot_FP(q, xTES, yTES, oktes, frame=frame,
                         fig=fig, ax=ax1, cmap='bwr', cbar=False, s=60, title='TES OK (2sig)')

    scal.scatter_plot_FP(q, xTES, yTES, sigres * oktes, frame=frame,
                         fig=fig, ax=ax2, cmap='bwr', cbar=True, unit='', s=60, title='TOD Residuals')
    return


def plot_fringes_errors(q, fringes1D, err_fringes1D, xTES, yTES, frame='ONAFP',
                        s=None, normalize=True, lim=1., suptitle=None):
    _, _, fringes1D = remove_thermometers(xTES, yTES, fringes1D)
    xTES, yTES, err_fringes1D = remove_thermometers(xTES, yTES, err_fringes1D)

    if normalize:
        fringes1D /= np.nanstd(fringes1D)
        err_fringes1D /= np.nanstd(err_fringes1D)

    fig, axs = plt.subplots(1, 2)
    fig.suptitle(suptitle)
    fig.subplots_adjust(wspace=0.5)
    ax0, ax1 = axs
    scal.scatter_plot_FP(q, xTES, yTES, err_fringes1D, frame=frame,
                         fig=fig, ax=ax0, cmap='bwr', cbar=True, unit='', s=s, title='Errors',
                         vmin=-lim, vmax=lim)

    scal.scatter_plot_FP(q, xTES, yTES, np.abs(fringes1D / err_fringes1D), frame=frame,
                         fig=fig, ax=ax1, cmap='bwr', cbar=True, unit='', s=s, title='|Values / Errors|',
                         vmin=0, vmax=3)
    return
