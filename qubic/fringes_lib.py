from __future__ import division, print_function
import numpy as np
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits as pyfits
from astropy.convolution import convolve, Gaussian2DKernel
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import cm
import scipy.optimize as sop
from scipy.signal import resample

from qubicpack.qubicfp import qubicfp
import qubic.fibtools as ft
from qubic import selfcal_lib as scal

__all__ = ['FringesAnalysis', 'SaveFringesFitsPdf']


class FringesAnalysis:
    def __init__(self, datafolder, date, q, baseline, ncycles=20, stable_time=5.,
                 asics=[1, 2], src_data=False, subtract_t0=True,
                 refTESnum=None, refASICnum=None, allh=[True, False, False, True, False, True], nsp_per=240,
                 lowcut=1e-5, highcut=2., nbins=120, notch=np.array([[1.724, 0.005, 10]]),
                 fraction_bad_TES=0.75, sigma_conv_astropy=0.7, sort_TES=True,
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
        fraction_bad_TES: float
            Fraction between 0 and 1 to set a threshold for bad TES.
            For example, if fraction=0.75, 1/4 of the detectors will be masked.
        sigma_conv_astropy: float
            Sigma of the Gaussian for the Astropy convolution for plots, in numbers of detectors.
        sort_TES: bool
            If False, do not sort the TES from good to bad. It can be useful in some cases, for example,
            when you do not want to make the full analysis but simply look at some TOD.
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
        self.stable_time = stable_time
        self.allh = allh
        self.nsp_per = nsp_per
        self.lowcut = lowcut
        self.highcut = highcut
        self.nbins = nbins
        self.notch = notch
        self.fraction_bad_TES = fraction_bad_TES
        self.sigma_conv_astropy = sigma_conv_astropy
        self.verbose = verbose

        # Get data
        self.tdata, self.data, self.tsrc, self.dsrc = self._get_data()

        self.nasics, self.ndet_oneASIC, _ = self.data.shape
        self.ndet = self.nasics * self.ndet_oneASIC
        self.nsteps = len(self.allh)
        self.expected_period = self.stable_time * self.nsteps

        if sort_TES:
            self.detectors_sort, self.ctimes = self.sort_TES_good2bad()
            self.mask_bad_TES = self.make_mask_bad_tes(self.detectors_sort, fraction=self.fraction_bad_TES)

            if self.verbose:
                print('******** 10 best TES found:', self.detectors_sort[:10, 0], self.detectors_sort[:10, 1])

                nbad_TES = int(self.ndet - self.fraction_bad_TES * self.ndet)
                print(f'******** With fraction_bad_TES = {self.fraction_bad_TES}, '
                      f'{nbad_TES}/{self.ndet} are considered as bad.')
        if refTESnum is not None:
            self.refTESnum = refTESnum
            self.refASICnum = refASICnum
        elif refTESnum is None and sort_TES:
            self.refTESnum = self.detectors_sort[0, 0]
            self.refASICnum = self.detectors_sort[0, 1]
        else:
            raise ValueError('You should either give a reference TES and ASIC or set sort_TES to True.')

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
        data = - np.array(data)  # Minus because when the current goes down it means more power.

        if self.src_data:  # Get calibration source data
            tsrc = a.calsource()[0]
            if self.subtract_t0:
                tsrc -= tsrc[0]
            dsrc = a.calsource()[1]
        else:
            tsrc = None
            dsrc = None

        return tdata, data, tsrc, dsrc

    def plot_TOD(self, ASIC=None, TES=None, xlim=None, figsize=(12, 6)):
        """
        Plot the TODs for a given TES. if TES is None, the plot is done for the reference TES.
        Parameters
        ----------
        ASIC: int
            ASIC number, 1 or 2 for the TD.
        TES: int
            TES index, between 1 and 128.
        xlim: float
            Upper abscissa limit.
        figsize: bytearray
            Size of the figure, as in matplotlib.

        """
        if ASIC is None:
            ASIC = self.refASICnum
            TES = self.refTESnum
        time = self.tdata[ASIC - 1, :]
        TOD = self.data[ASIC - 1, TES - 1, :]
        plt.figure(figsize=figsize)
        ax = plt.gca()
        ax.plot(time, TOD)
        ax.set_title(f'TOD for TES {TES} - ASIC {ASIC}')
        if xlim is not None:
            ax.set_xlim(0, xlim)
        plt.show()
        return

    def find_right_period(self, TES, ASIC, filtering=True, delta=0.5, nb=100):
        """
        Determine the true period of one cycle.
        Parameters
        ----------
        TES, ASIC: int
            TES and ASIC numbers.
        filtering: bool
            If True, data are filtered.
        delta: float
        nb: int

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
            print('Found period {0:5.3f}s on TES {1:} ASIC {2:}'.format(period, TES, ASIC))
            print('Expected : ', self.expected_period)

        return ppp, rms, period

    def sort_TES_good2bad(self, median=True, doplot=True):
        """
        Make a fit on the TODs, fit the steps of the cycle with time constants.
        The residuals are used to order the TES from the best to the worst.
        Parameters
        ----------
        median: bool
            If True, a median and not a mean is done when folding the data using a function from fibtools.
        doplot: bool
            If True, a plot is made for the 5 best TES.

        Returns
        -------
        Time, folded signal, the 8 parameters estimated with the fit,
        the combination of the amplitudes, the period and the residuals
        between the fit and the signal. They are computed for each TES.
        """

        ctimes = np.zeros((self.nasics, self.ndet_oneASIC))
        params = np.zeros((self.ndet, self.nsteps + 2))
        err_params = np.zeros_like(params)
        datafold = np.zeros((self.ndet, self.nbins))
        err_datafold = np.zeros_like(datafold)
        residuals_time = np.zeros_like(datafold)
        if self.verbose:
            print('******** Start sort TES from good to bad.***********')
        for i, ASIC in enumerate(self.asics):
            # Filter, fold and normalize the data
            dfold, tfold, _, errfold, _, _ = ft.fold_data(self.tdata[i, :],
                                                          self.data[i, :, :],
                                                          self.expected_period,
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

            # Fit
            param_guess = [0.1, 0.] + [1.] * self.nsteps
            for j in range(self.ndet_oneASIC):
                index = self.ndet_oneASIC * i + j
                # With curve_fit, it is not possible to have 'args' so we use a lambda function
                popt, pcov = sop.curve_fit(lambda tfold, ctime, tstart, a0, a1, a2, a3, a4, a5:
                                           ft.simsig_fringes(tfold,
                                                             self.stable_time,
                                                             [ctime, tstart, a0, a1, a2, a3, a4, a5]),
                                           tfold,
                                           dfold[j, :],
                                           p0=param_guess,
                                           sigma=errfold[j, :],
                                           absolute_sigma=True,
                                           bounds=([0., -2, -2, -2, -2, -2, -2, -2],
                                                   [1., 2, 2, 2, 2, 2, 2, 2]),
                                           maxfev=10000
                                           )
                params[index, :] = popt
                # Take the time constant of the detectors
                ctimes[i, j] = popt[0]
                err_params[index, :] = np.sqrt(np.diag(pcov))

                residuals_time[index, :] = dfold[j, :] - ft.simsig_fringes(tfold, self.stable_time, popt)

        # Order the TES as function of the residual on the fit
        std = np.std(residuals_time, axis=1)
        std_argsort = np.argsort(std)
        detectors_sort = np.zeros((self.ndet, 2), dtype=int)
        detectors_sort[:, 0] = (std_argsort % self.ndet_oneASIC) + 1  # TES
        detectors_sort[:, 1] = (std_argsort // self.ndet_oneASIC) + 1  # ASIC

        # Plot the 5 best TES
        if doplot:
            for p in range(5):
                plot_folding_fit(detectors_sort[p, 0], detectors_sort[p, 1], tfold, datafold, residuals_time,
                                 self.expected_period, params, err_params)

        return detectors_sort, ctimes

    def make_mask_bad_tes(self, detectors_sort, fraction=0.75):
        """

        Parameters
        ----------
        detectors_sort: array
            TES numbers ordered from the best to the worst.
            It is a 2D array containing the TES and the ASIC index.
        fraction: float
            Fraction between 0 and 1 to set a threshold for bad TES.
            For example, if fraction=0.75, 1/4 of the detectors will be masked.

        Returns
        -------
        mask: 1 for good TES and NAN for bad TES.
        """

        mask = np.ones(self.ndet)
        # All TES in a 1D array from 1 to 128*nasics
        alldet = detectors_sort[:, 0] + self.ndet_oneASIC * (detectors_sort[:, 1] - 1)
        mask[alldet[int(fraction * self.ndet):] - 1] = np.nan
        return mask

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
        dsignal, thr = find_high_derivative(msignal)

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

        return tfold, dfold

    def roll_oneTES(self, dfold, t0, period=None):
        """Shift the folded data in order to have t0=0."""

        droll = np.roll(dfold, -int(t0 / period * self.nsp_per), axis=1)

        return droll

    def remove_median_allh(self, tfold, droll, period, skip_rise=0., skip_fall=0.):
        # Roughly remove the median of the all_h configurations
        ok_all_horns = np.zeros_like(tfold, dtype=bool)
        for i in range(self.nsteps):
            if self.allh[i]:
                tmini = i * period / self.nsteps + skip_rise
                tmaxi = (i + 1) * period / self.nsteps - skip_fall
                ok = (tfold >= tmini) & (tfold < tmaxi)
                ok_all_horns[ok] = True
        droll_rm_median = droll - np.median(droll[:, ok_all_horns])

        return droll_rm_median

    def remove_slope_percycle(self, m_points, err_m_points, doplot=True):
        """Fit a slope between the "all horns open" configurations (3 points) and remove it.
        Do it for each cycle."""
        xx = np.arange(self.nsteps)
        m_points_rm = np.zeros_like(m_points)
        for j in range(self.ncycles):

            # Linear fit
            pars = np.polyfit(x=xx[self.allh],
                              y=m_points[j, self.allh],
                              deg=1,
                              w=1. / err_m_points[j, self.allh],
                              full=False,
                              cov=False)
            # Subtract the slope to the 6 points, the points with all horns open will be around 0 by construction
            m_points_rm[j, :] = m_points[j, :] - (pars[0] * xx + pars[1])

            if j < 5:
                if doplot:
                    ttt = xx * self.stable_time + self.stable_time / 2
                    plt.figure()
                    plt.title(f'Fit the slope on all horn open - Cycle {j + 1}')
                    plt.plot(ttt, (pars[0] * xx + pars[1]), 'k--', label='Linear fit all open')
                    plt.step(ttt, m_points[j, :], color='b', where='mid', label='Original points')
                    plt.step(ttt, m_points_rm[j, :], color='r', where='mid', label='After correction')
                    plt.xlabel('Time[s]')
                    plt.ylabel('Amplitude')
                    plt.legend()
        return m_points_rm

    def remove_slope_allcycles(self, m_points, err_m_points, doplot=True):
        """Fit a slope between the "all horns open" configurations and remove it.
        Finally this function is not used for the fringes analysis, I prefer removing the slope cycle by cycle. """
        xx = np.arange(self.nsteps * self.ncycles)

        # Linear fit
        pars = np.polyfit(x=xx[self.allh * self.ncycles],
                          y=np.ravel(m_points[:, self.allh]),
                          deg=1,
                          w=1. / np.ravel(err_m_points[:, self.allh]),
                          full=False,
                          cov=False)
        # Subtract the slope to the points, take the first point as the reference
        ref, _ = ft.meancut(m_points[:, self.allh], nsig=3)
        m_points_rm = np.ravel(m_points) - (pars[0] * xx + pars[1]) + ref

        m_points_rm = np.reshape(m_points_rm, (self.ncycles, self.nsteps))

        if doplot:
            ttt = xx * self.stable_time + self.stable_time / 2
            plt.figure()
            plt.title('Fit the slope on all horn open - All cycles')
            plt.plot(ttt, (pars[0] * xx + pars[1]), 'k--', label='fit')
            plt.plot(ttt, np.ravel(m_points), 'bo', label='Original points')
            plt.plot(ttt, np.ravel(m_points_rm), 'ro', label='After correction')
            plt.xlabel('Time[s]')
            plt.ylabel('Amplitude')
            plt.legend()

        return m_points_rm

    def average_over_points_oneTES(self, tfold, droll, period, skip_rise=0., skip_fall=0.,
                                   median=True, rm_slope_percycle=False, doplot=True):
        """Average on each step in the cycle."""

        # We assume that the array has been np.rolled so that the t0 is in time sample 0
        stable_time = period / self.nsteps

        # Remove the average of each cycle
        droll = (droll.T - np.mean(droll, axis=1)).T

        # Perform first an average/median in each step of each cycle over the points
        # (possibly skipping beginning and end)
        m_points = np.zeros((self.ncycles, self.nsteps))
        err_m_points = np.zeros((self.ncycles, self.nsteps))
        for i in range(self.nsteps):
            # Cut the data
            tstart = i * stable_time + skip_rise
            tend = (i + 1) * stable_time - skip_fall
            ok = (tfold >= tstart) & (tfold < tend)
            for j in range(self.ncycles):
                m_points[j, i], err_m_points[j, i] = ft.meancut(droll[j, ok],
                                                                nsig=3,
                                                                med=median,
                                                                disp=False)

        if rm_slope_percycle:
            m_points = self.remove_slope_percycle(m_points, err_m_points, doplot=doplot)
        return m_points, err_m_points

    def average_over_cycles_oneTES(self, m_points, err_m_points, median=True,
                                   Ncycles_to_use=None,
                                   speak=False, doplot=False):
        """
        Parameters
        ----------
        m_points, err_m_points: float
            Mean and errors on each step for one TES.
        median: bool
            If True, takes the median and not the mean on each folded step.
        Ncycles_to_use: int
            Number of cycles to use. If None, all cycles will be used.
        speak: bool
        doplot: bool

        Returns
        -------

        """
        if Ncycles_to_use is None:
            Ncycles_to_use = self.ncycles

        # Average or median over all cycles
        Mcycles = np.zeros(self.nsteps)
        err_Mcycles = np.zeros(self.nsteps)
        for i in range(self.nsteps):
            Mcycles[i], err_Mcycles[i] = ft.meancut(m_points[:Ncycles_to_use, i], nsig=3, med=median, disp=False)

        if speak:
            for i in range(self.nsteps):
                print('############')
                print('Step {}'.format(i + 1))
                for j in range(Ncycles_to_use):
                    print('cycle {}: {} +/- {}'.format(j, m_points[j, i], err_m_points[j, i]))
                print('============')
                print('Mean/Median: {} +/- {}'.format(Mcycles[i], err_Mcycles[i]))
                print('============')

        if doplot:
            self.plot_average_over_steps(m_points, err_m_points, Mcycles, err_Mcycles)

        return Mcycles, err_Mcycles

    def analyse_fringes(self, median=True, remove_median_allh=False,
                        Ncycles_to_use=None,
                        rm_slope_percycle=False,
                        force_period=None, force_t0=None, doplotTESsort=0):
        """
        Full analysis to get fringes from the TODs.
        Parameters
        ----------
        median: bool
        remove_median_allh: bool
        Ncycles_to_use: int
            Number of cycles to use. If None, all cycles will be used.
        rm_slope_percycle: bool
            If True, for each cycle, a slope on the steps where all horns are open is fit
            and removed.
        force_period: float
            If None, the period is determined automatically but the period can be forced using this argument.
        force_t0: float
            If None, t0 is determined on the best TES but it can be given using this argument.
        doplotTESsort: int
            TES index once they are ordered from the best to the worst for which plots are made.
            By default, it is 0, so plots are made for the best TES.

        Returns
        -------

        """

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
            t0 = self.find_t0(period, doplot=True)
            if self.verbose:
                print('Found t0 {0:5.3f}s on TES {1:}, ASIC {2:}'.format(t0, self.refTESnum, self.refASICnum))
        else:
            t0 = force_t0
            if self.verbose:
                print('Using forced t0 {0:5.3f}s'.format(t0))

        # =============== Loop on ASICs and TES ======================
        m_points = np.zeros((self.ndet, self.ncycles, self.nsteps))
        err_m_points = np.zeros((self.ndet, self.ncycles, self.nsteps))
        Mcycles = np.zeros((self.ndet, self.nsteps))
        err_Mcycles = np.zeros((self.ndet, self.nsteps))
        fringes1D = np.zeros(self.ndet)
        err_fringes1D = np.zeros(self.ndet)
        fringes1D_percycle = np.zeros((self.ndet, self.ncycles))
        err_fringes1D_percycle = np.zeros((self.ndet, self.ncycles))
        weights = np.array([1. / 3, -1, 1, 1. / 3, -1, 1. / 3])

        for i, ASIC in enumerate(self.asics):
            print(f'*********** Starting ASIC {ASIC} **************')
            for j, TES in enumerate(np.arange(1, 129)):
                # Use the time constant for skip rise if it is reasonable.
                if self.ctimes[i, j] < 0.6:
                    skip_rise = self.ctimes[i, j] * 5.
                else:
                    skip_rise = 1.
                skip_fall = 0.5
                # print(skip_rise, skip_fall)

                # If TES in doplotTESsort, active the plot option
                want_plot = np.any((self.detectors_sort[doplotTESsort, 0] == TES) &
                                   (self.detectors_sort[doplotTESsort, 1] == ASIC))
                if want_plot:
                    rank = np.where((self.detectors_sort[:, 0] == TES) &
                                    (self.detectors_sort[:, 1] == ASIC))[0][0]
                    print(f'\n ===== Making plots for TES {TES} - ASIC {ASIC} - Goodness rank {rank}')
                    doplot = True
                    speak = True
                else:
                    doplot = False
                    speak = False

                index = i * self.ndet_oneASIC + j
                timeTES = self.tdata[i, :]
                dataTES = self.data[i, j, :]
                # Fold, roll and remove the median of allh
                tfold, dfold = self.resample_fold_oneTES(timeTES, dataTES, period=period, doplot=doplot)
                droll = self.roll_oneTES(dfold, t0, period=period)
                if doplot:
                    self._plot_folding(tfold, droll, period, skip_rise=skip_rise, skip_fall=skip_fall,
                                       suptitle='Data fold and roll')
                if remove_median_allh:
                    droll = self.remove_median_allh(tfold, droll, period,
                                                    skip_rise=skip_rise, skip_fall=skip_fall)
                    if doplot:
                        self._plot_folding(tfold, droll, period, skip_rise=skip_rise, skip_fall=skip_fall,
                                           suptitle='Data fold, roll and remove median all horn open')

                # Average each step on each cycle
                m_points[index, :, :], err_m_points[index, :, :] = self.average_over_points_oneTES(
                    tfold,
                    droll,
                    period,
                    skip_rise=skip_rise,
                    skip_fall=skip_fall,
                    median=median,
                    rm_slope_percycle=rm_slope_percycle,
                    doplot=doplot)
                # Make the combination for each cycle
                for k in range(self.ncycles):
                    fringes1D_percycle[index, k], err_fringes1D_percycle[index, k] = weighted_sum(m_points[index, k, :],
                                                                                                  err_m_points[index, k,
                                                                                                  :],
                                                                                                  weights)

                # Average over cycles and make the combination on the mean to get fringes
                Mcycles[index, :], err_Mcycles[index, :] = self.average_over_cycles_oneTES(
                    m_points[index, :, :],
                    err_m_points[index, :, :],
                    median=median,
                    Ncycles_to_use=Ncycles_to_use,
                    speak=speak,
                    doplot=doplot)
                fringes1D[index], err_fringes1D[index] = weighted_sum(Mcycles[index, :], err_Mcycles[index, :], weights)
                if doplot:
                    self._plot_TOD_reconstruction(droll, Mcycles[index, :])

        # Final plots
        # Fringes
        fig, axs = plt.subplots(2, 2, figsize=(10, 12))
        fig.subplots_adjust(wspace=0.5)
        fig.suptitle(f'Fringes and errors - BL {self.baseline} - {self.date}')
        ax0, ax1, ax2, ax3 = axs.ravel()
        # Make a convolution with Astropy (just to see better the fringes)
        fringes2D = make2Dfringes_data(fringes1D * self.mask_bad_TES)
        fringes2D_conv = astropy_convolution(fringes2D, sigma=self.sigma_conv_astropy)
        cmap_bwr = make_cmap_nan_black('bwr')
        plot_fringes_imshow(fringes2D_conv, fig=fig, ax=ax0, cmap=cmap_bwr,
                            title='Astropy convolution', mask=make_mask2D_thermometers_TD())
        plot_fringes_scatter(self.q, self.xTES, self.yTES, fringes1D * self.mask_bad_TES, s=100,
                             fig=fig, ax=ax1, cmap=cmap_bwr)

        # Errors
        cmap_viridis = make_cmap_nan_black('viridis')
        plot_fringes_scatter(self.q, self.xTES, self.yTES, err_fringes1D * self.mask_bad_TES, s=100, fig=fig, ax=ax2,
                             cmap=cmap_viridis, normalize=False, vmin=0., vmax=400, title='Errors')
        plot_fringes_scatter(self.q, self.xTES, self.yTES, np.abs(fringes1D / err_fringes1D) * self.mask_bad_TES,
                             s=100, fig=fig, ax=ax3, cmap=cmap_viridis, normalize=False, vmin=0., vmax=3.,
                             title='|Values/Errors|')

        return m_points, err_m_points, Mcycles, err_Mcycles, fringes1D, err_fringes1D, \
               fringes1D_percycle, err_fringes1D_percycle

    # ================ Plot functions very specific to the fringes analysis
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

    def _plot_folding(self, tfold, dfold, period, skip_rise=None, skip_fall=None, suptitle='', figsize=(12, 6)):

        fig, axs = plt.subplots(1, 2, figsize=figsize)
        fig.subplots_adjust(wspace=0.5)
        fig.suptitle(suptitle)
        ax1, ax2 = np.ravel(axs)

        ax1.imshow(dfold,
                   origin='lower',
                   aspect='auto',
                   extent=[0, np.max(tfold) + (tfold[1] - tfold[0]) / 2, 0, self.ncycles + 0.5])
        for i in range(self.nsteps):
            ax1.axvline(x=i * (period / self.nsteps), color='k', lw=3)
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Cycle number')

        for i in range(self.ncycles):
            ax2.plot(tfold, dfold[i, :], alpha=0.5)
        for i in range(self.nsteps):
            ax2.axvline(x=i * (period / self.nsteps), color='k', lw=3)
            if skip_rise is not None:
                if i == 0:
                    lab1 = 'Skip rise {:.2f}s'.format(skip_rise)
                    lab2 = 'Skip fall {:.2f}s'.format(skip_fall)
                else:
                    lab1 = None
                    lab2 = None

                ax2.axvspan(i * (period / self.nsteps),
                            (i * (period / self.nsteps) + skip_rise),
                            alpha=0.2, color='red', label=lab1)
                ax2.axvspan(((i + 1) * (period / self.nsteps) - skip_fall),
                            (i + 1) * (period / self.nsteps),
                            alpha=0.2, color='b', label=lab2)
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('ADU')
        ax2.legend()

        return

    def plot_average_over_steps(self, m_points, err_m_points, Mcycles, err_Mcycles, figsize=(12, 10)):
        plt.figure(figsize=figsize)
        ax = plt.gca()

        ttt = np.arange(self.nsteps) * self.stable_time + self.stable_time / 2  # Mean time of each step

        for i in range(self.ncycles):
            if i == 0:
                lab = 'Mean on each step'
            else:
                lab = None
            ax.errorbar(ttt, m_points[i, :],
                        yerr=err_m_points[i, :],
                        xerr=self.stable_time / 2, fmt='o', alpha=0.1 + i * 0.025, color='blue', label=lab)
        ax.errorbar(ttt, Mcycles, yerr=err_Mcycles, xerr=self.stable_time / 2, color='r',
                    label='Mean over cycles', fmt='rx')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Value for each cycle')
        ax.legend()
        return

    def _plot_TOD_reconstruction(self, dfold, Mcycles, figsize=(12, 10)):
        tf = self.ncycles * self.nsteps * self.stable_time
        tt = np.arange(0, tf, self.stable_time)
        time = np.linspace(0., tf, self.ncycles * self.nsp_per)
        TOD_reconstructed = np.tile(Mcycles, self.ncycles)
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        ax.plot(time, np.ravel(dfold), label='Input signal')
        ax.step(tt, TOD_reconstructed, where='post', label='Reconstructed TOD with Mcycles')
        ax.set_xlabel('Time samples')
        ax.set_ylabel('TOD')
        ax.legend()
        return


# =========================================
class SaveFringesFitsPdf:
    def __init__(self, q, date_obs, allBLs, allstable_time, allNcycles, xTES, yTES, allfringes1D, allerr_fringes1D,
                 allmask_bad_TES=None, nsteps=6, ecosorb='yes', frame='ONAFP'):
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
        self.allmask_bad_TES = allmask_bad_TES
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
                 'ERRORS': self.allerr_fringes1D, 'MASK_BAD_TES': self.allmask_bad_TES}

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
                fringes2D = make2Dfringes_data(self.allfringes1D[i] * self.allmask_bad_TES[i])
                fringes2D_conv = astropy_convolution(fringes2D)
                plot_fringes_imshow(fringes2D_conv, fig=fig, ax=ax0, mask=mask, title='Astropy convolution',
                                    cmap=make_cmap_nan_black('bwr'))
                plot_fringes_scatter(self.q, self.xTES, self.yTES, self.allfringes1D[i] * self.allmask_bad_TES[i],
                                     s=80, fig=fig, ax=ax1)
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
def decide_bad_TES(allmask_bad_TES, condition=2):
    """
    Make a mask for bad detectors, common to all images.
    Parameters
    ----------
    allmask_bad_TES: list
        List with the mask of the worst TES for each image.
    condition: int
        Number of images where the detector must be NAN to consider it as bad.

    """
    nimages = len(allmask_bad_TES)
    ndet = allmask_bad_TES[0].shape[0]

    the_mask = np.zeros(ndet)
    for k in range(nimages):
        if k == 0:  # replace NAN by 1
            the_mask[np.isnan(allmask_bad_TES[k])] = 1
        else:  # Add 1 if it is NAN
            the_mask[np.isnan(allmask_bad_TES[k])] += 1

    the_mask[the_mask >= condition] = np.nan
    the_mask[the_mask < condition] = 1

    print(f'Number of bad detectors: {int(ndet - np.nansum(the_mask))}/{ndet}')
    return the_mask


def give_index_bad_TES(the_mask):
    """Give the index of the bad TES from a mask."""
    ndet = the_mask.shape[0]
    nbadtes = int(ndet - np.nansum(the_mask))
    badTES = np.zeros((nbadtes, 2), dtype=int)
    badTES[:, 0] = np.argwhere(np.isnan(the_mask))[:, 0] % 128 + 1  # TES
    badTES[:, 1] = np.argwhere(np.isnan(the_mask))[:, 0] // 128 + 1  # ASIC

    return badTES


def find_high_derivative(signal):
    """Calculate the derivative of a signal and find where it is high."""
    dsignal = np.abs(np.gradient(signal))
    md, sd = ft.meancut(dsignal, 3)
    thr = np.abs(dsignal - md) > (3 * sd)
    return dsignal, thr


def cut_data(t0, tf, tdata, data):
    """
    Cut the TODs from t0 to tf.
    They can be None if you do not want to cut the beginning or the end.
    data shape: [nTES, nsamples]
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
    w_sum = np.sum(weights * vals)
    sigma = np.sqrt(np.sum(weights ** 2 * errs ** 2))
    return w_sum, sigma


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


def make_cmap_nan_black(cmap):
    cmap_bwr = copy.copy(cm.get_cmap(cmap))
    cmap_bwr.set_bad(color='k')
    return cmap_bwr


def plot_fringes_scatter(q, xTES, yTES, fringes1D, normalize=True, frame='ONAFP', fig=None, ax=None,
                         cbar=True, vmin=-1., vmax=1., cmap=make_cmap_nan_black('bwr'), s=None, title='Scatter plot'):
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
                         vmin=vmin,
                         vmax=vmax,
                         cbar=cbar,
                         plotnonfinite=True
                         )
    return


def make2Dfringes_QubicSoft(fringes1D, q, nan2zero=False):
    """fringes1D must have 248 elements ordered as in Qubic soft."""
    fringes2D = q.detector.unpack(fringes1D)[17:, :17]
    if nan2zero:
        fringes2D[np.isnan(fringes2D)] = 0.
    return fringes2D


def make2Dfringes_data(fringes1D, nan2zero=False):
    fringes2D = ft.image_asics(all1=fringes1D)
    if nan2zero:
        fringes2D[np.isnan(fringes2D)] = 0.
    return fringes2D


def astropy_convolution(fringes2D, sigma=0.7, nan_treatment='interpolate', preserve_nan=True):
    kernel = Gaussian2DKernel(sigma)
    image_convolved = convolve(fringes2D,
                               kernel,
                               nan_treatment=nan_treatment,
                               preserve_nan=preserve_nan)
    return image_convolved


def plot_fringes_imshow(fringes2D, normalize=True, interp=None, mask=None,
                        fig=None, ax=None, cbar=True, vmin=-1, vmax=1.,
                        cmap='bwr', title='Imshow'):
    if normalize:
        fringes2D /= np.nanstd(fringes2D)

    if mask is not None:
        fringes2D *= mask

    if ax is None:
        fig, ax = plt.subplots()
    img = ax.imshow(np.rot90(fringes2D, k=2),
                    vmin=vmin, vmax=vmax,
                    cmap=cmap,
                    interpolation=interp)
    ax.axis('off')
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
