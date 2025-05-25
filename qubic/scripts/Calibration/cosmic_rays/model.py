import numpy as np
import scipy as sp
from typing import Optional
from scipy.stats import pearsonr, linregress
from scipy.optimize import curve_fit

import multiprocessing as mp

from results import FitResult


class Model:
    """
    Class containing static methods for:
    - Filtering candidate signals using various criteria and parallel processing.
    - Fitting an exponential decay model to signal data.
    - Deconvolving the instrument's transfer function from the Time-Ordered Data (TOD).
    """

    @staticmethod
    def get_initial_params(x: np.ndarray, y: np.ndarray) -> tuple:
        """
         Calculate initial parameters for the exponential decay fit used in scipy.optimize.curve_fit.
        Uses numerical integration and least squares to estimate parameters a, b, and c in the model:
            f(t) = a + b * exp(c * t)

        Thanks to:
        https://it.scribd.com/doc/14674814/Regressions-et-equations-integrales
        https://stackoverflow.com/questions/77822770/exponential-fit-is-failing-in-some-cases/77840735#77840735


        Parameters
        ----------
        x: np.ndarray
            Time array corresponding to the exponential decay segment.
        y: np.ndarray
            Signal array corresponding to the exponential decay segment.

        Returns
        -------
        tuple
            A tuple of initial parameters (a, b, c) for the exponential decay function
        """
        # Initialize an array 's' with zeros having the same shape as y
        s = np.zeros_like(y)
        # Compute cumulative trapezoidal integration of y with respect to x
        s[1:] = s[:-1] + 0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])

        # Compute elements for the 2x2 coefficient matrix used to solve for c
        c11 = np.sum((x - x[0]) ** 2)  # Sum of squared differences of x from its first element.
        c12 = np.sum((x - x[0]) * s)  # Sum of products between time differences and integrated values.
        c21 = c12  # Symmetric entry: c21 equals c12.
        c22 = np.sum(s ** 2)  # Sum of squared integrated signal values.

        # Compute elements for the right-hand side vector
        cf11 = np.sum((y - y[0]) * (x - x[0]))  # Sum of products between y differences and x differences.
        cf21 = np.sum((y - y[0]) * s)  # Sum of products between y differences and integrated values.

        # Form the coefficient matrix and the corresponding factor vector
        c_matrix = np.array([[c11, c12],
                             [c21, c22]])

        c_factor = np.array([[cf11],
                             [cf21]])

        # Solve for parameter c; extract the second element from the solution vector
        c = (np.linalg.inv(c_matrix) @ c_factor)[1]

        # Set up a second linear system to solve for parameters a and b
        ab11 = len(y)  # Number of data points.
        ab12 = np.sum(np.exp(c * x))  # Sum of exponentials scaled by c.
        ab21 = ab12  # Symmetric entry: ab21 equals ab12.
        ab22 = np.sum(np.exp(2 * c * x))  # Sum of squared exponentials.
        abf11 = np.sum(y)  # Sum of signal values.
        abf21 = np.sum(y * np.exp(c * x))  # Sum of signal values weighted by exponential factor.

        # Form the matrix and the factor vector for parameters a and b
        ab_matrix = np.array([[ab11, ab12],
                              [ab21, ab22]])

        ab_factor = np.array([[abf11],
                              [abf21]])

        # Replace any zero entries in ab_matrix with a small value to prevent division by zero
        ab_matrix = np.where(ab_matrix == 0, 1e-10, ab_matrix)

        # Solve the linear system to obtain a and b.
        a, b = np.linalg.inv(ab_matrix) @ ab_factor

        return a, b, c

    @staticmethod
    def exp_model_uncertainty_jacobian(t, popt, pcov):
        """
        Calcola l'incertezza sigma_y(t) usando il jacobiano e il prodotto matriciale.

        Parameters:
          t : array-like
              I valori temporali.
          popt : list or array
              I parametri ottimali [a, b, c].
          pcov : 2D array
              La matrice di covarianza dei parametri.

        Returns:
          sigma_y : array
              L'incertezza calcolata per ogni valore di t.
        """
        a, b, c = popt
        # Costruisci il jacobiano per ogni t; ogni riga corrisponde a [dy/da, dy/db, dy/dc]
        J = np.empty((len(t), 3))
        J[:, 0] = 1
        J[:, 1] = np.exp(c * t)
        J[:, 2] = b * t * np.exp(c * t)
        # Calcola l'incertezza come sigma_y = sqrt(J * pcov * J^T) per ogni riga di J
        sigma_y = np.sqrt(np.einsum('ij,jk,ik->i', J, pcov, J))

        return sigma_y

    @staticmethod
    def exp_decay(t: np.ndarray, a: int, b: int, c: float) -> np.ndarray:
        """
        Exponential decay model function.

        Models the signal as: f(t) = a + b * exp(c * t)

        Parameters
        ----------
        t: np.ndarray
            Time array at which to evaluate the function.
        a: int
            Baseline (steady state) value.
        b: int
            Amplitude of the exponential component.
        c: float
            Decay rate, equal to -1/tau (where tau is the time constant).

        Returns
        -------
        np.ndarray
            Computed values of the exponential decay model.
        """
        return a + b * np.exp(c * t)

    @staticmethod
    def get_fit_candidate(time_raw: np.ndarray, s_clean: np.ndarray, std: np.ndarray, eps=0.001) -> Optional[FitResult]:
        """
        Perform an exponential decay fit on a valid candidate signal segment.

        Parameters
        ---------
        time_raw: np.ndarray
            Array of time values.
        s_clean: np.ndarray
            Array of signal values.
        std: np.ndarray
            Standard deviation of the signal, used for weighting in the fit.

        eps: float


        Returns
        ------
        np.array
           If the fit is acceptable (p-value > 0.05), returns an array containing:
             - Optimal fit parameters (popt)
             - Uncertainty on the time constant (sigma_tau)
             - Degrees of freedom (nu)
             - Reduced chi-square (chi_square_reduced)
             - p-value (p_value)
             - Normalized residuals (residuals divided by y)
           Otherwise, returns np.nan.
        """

        # Identify the index of the maximum value in the signal, marking the start of the decay.
        sig_max_idx = s_clean.argmax()

        # linear fit over the vertical linear trend
        lin_fit = linregress(time_raw[:sig_max_idx + 1], s_clean[:sig_max_idx + 1])

        # Define the time array for the decay phase, shifting it so that t=0 corresponds to the maximum.
        x = time_raw[sig_max_idx:] - time_raw[sig_max_idx]
        # Extract the corresponding signal values for the decay phase.
        y = s_clean[sig_max_idx:]

        try:
            while True:
                # Estimate initial parameters for the exponential decay fit
                p0 = Model.get_initial_params(x, y)
                # absolute_sigma is set to False because the uncertainties are not very precise.
                # In other words, sigma is scaled to match the sample variance of the residuals after the fit:
                # pcov(absolute_sigma=False) = pcov(absolute_sigma=True) * chisq(popt)/(M-N)
                popt, pcov = curve_fit(Model.exp_decay, x, y, p0=p0, sigma=[std] * y.shape[0], absolute_sigma=False)

                # Compute the residuals between the model and the observed data
                residuals = Model.exp_decay(x, *popt) - y

                # Calculate the degrees of freedom: number of data points minus number of parameters
                nu = len(y) - len(popt)

                # estimate the standard deviation of the residuals
                new_std = residuals.std()
                if np.fabs(new_std - std) / std < eps:
                    break
                else:
                    std = new_std

            # Compute the chi-square statistic: sum of squared residuals normalized by the variance
            chi_square = np.sum((residuals ** 2) / std ** 2)

            # Calculate the reduced chi-square (chi_square per degree of freedom)
            chi_square_reduced = chi_square / nu

            # Significance: if the p-value is less than the significance threshold of 0.05,
            # we reject the null hypothesis H_0 (the model fits the data).
            # If the p-value is greater than 0.05, we accept the null hypothesis
            p_value = 1 - sp.stats.chi2.cdf(chi_square, nu)

            # Calculate the uncertainty on the time constant parameter
            sigma_tau = np.sqrt(pcov[2][2]) / popt[2] ** 2

            # saving fit information over a candidate in a fit_res object
            fit_res = FitResult(
                a=popt[0],
                b=popt[1],
                c=popt[2],
                sigma_tau=sigma_tau,
                nu=nu,
                chi_square_reduced=chi_square_reduced,
                p_value=p_value,
                slope=lin_fit.slope,
                slope_sigma=lin_fit.stderr,
                slope_p_value=lin_fit.pvalue,
                residuals=(residuals / y).tolist(),
                pcov=pcov.tolist(),
                index=None)

            return fit_res if fit_res.is_valid() else None

        except:
            return None

    @staticmethod
    def emg_reconstruction(sig: np.ndarray, tau: float, dt: float, window: int) -> np.ndarray:
        """
          Perform EMG (Exponential Modified Gaussian) reconstruction of a signal.
          Deconvolves a signal with an EMG shape by adding its derivative multiplied by the time constant.

          Parameters
          ----------
          sig: np.ndarray
              Input signal of the TES.
          tau: float
              Average time constant of the TES.
          dt: float
              Time step between successive samples.

          window: int
              Window used in savgol filter

          Returns
          -------
          np.ndarray
              Deconvolved signal
          """

        gradient = sp.signal.savgol_filter(np.gradient(sig, dt),
                                           window_length=window,
                                           polyorder=3,
                                           mode='nearest')

        return sig + gradient * tau

    @staticmethod
    def deconvolve(conv_signal: np.ndarray,
                   M: int,
                   dt: float,
                   tau: float) -> np.ndarray:
        """
        Deconvolve the signal from the instrument's transfer function using EMG reconstruction.
        Then attenuate high-frequency noise with a moving average filter.

        Parameters
        ----------
        conv_signal: np.ndarray
            Signal of the TES convolved with the instrument's transfer function.
        M: int
            Number of points for the moving average window.
        dt: float
            Time step between samples.
        tau: float
            Average time constant for deconvolution.

        Returns
        -------
        np.ndarray
            Deconvolved signal with high-frequency noise reduced.
        """
        # Apply EMG reconstruction to the convolved signal
        deconv_signal = Model.emg_reconstruction(conv_signal, tau, dt, M)

        # Create a normalized moving average window of length M.
        # A normalization of the window is performed in order to NOT modify the amplitude of the signal
        win = np.ones(M) / M

        # Convolve the deconvolved signal with the window; mode='same' preserves the original signal length.
        return sp.signal.convolve(deconv_signal, win, mode='same')

    @staticmethod
    def candidate_filter(tm_candidate: np.ndarray[float],
                         s_candidate: np.ndarray[float],
                         pts_vertical_trend: int,
                         r_threshold: float = -0.9,
                         p_threshold: float = 0.05) -> bool:
        """
        Filter a candidate signal based on several criteria.

        A valid candidate should exhibit an initial vertical linear growth followed by an exponential decay.
        The candidate is rejected if:
          - Any signal value is non-positive.
          - The relative difference between the first and last signal values exceeds 1.5%.
          - The number of points in the vertical growth phase is less than the specified threshold.
          - The log-transformed decay does not show consistent differences.
          - The Pearson correlation of the decay phase does not meet the specified thresholds.

        Parameters
        ---------
        tm_candidate: np.ndarray
            Time values corresponding to the candidate.
        s_candidate: np.ndarray
            Signal values of the candidate.
        pts_vertical_trend: int
            Minimum number of points required in the vertical growth phase.
        r_threshold: float, optional
            Correlation coefficient threshold; candidates with r above this (less negative) are rejected.
        p_threshold: float, optional
            p-value threshold; candidates with p-value above this are rejected.

        Returns
        ------
        bool:
            True if the candidate passes all the criteria; False otherwise.
        """

        # Reject the candidate if any signal value is zero or negative.
        if np.any(s_candidate <= 0.):
            return False

        # Compute the relative difference between the first and last signal values
        ratio = abs(s_candidate[-1] - s_candidate[0]) / max(s_candidate[0], s_candidate[-1])
        # Reject if the relative difference is greater than 1.5%.
        if ratio > 0.015:
            return False

        # Determine the number of points up to the maximum signal value
        if (sig_max_idx := s_candidate.argmax() + 1) < pts_vertical_trend:
            return False

        # Select the exponential decay part of the candidate signal
        s_exp_decay = s_candidate[sig_max_idx:]

        # Compute the natural logarithm of the decay segment
        log_s = np.log(s_exp_decay)

        # Calculate differences between consecutive log-transformed values
        diff_s = np.diff(log_s)
        # Compute the mean and standard deviation of these differences
        mu, sigma = np.mean(diff_s), np.std(diff_s)

        # Reject the candidate if differences deviate by more than 2 sigma from the mean
        if not np.all(np.abs(diff_s - mu) < 2 * sigma):
            return False

        # Calculate Pearson correlation coefficient and p-value between time and log(signal).
        r, p_value = pearsonr(tm_candidate[sig_max_idx:], log_s)

        # Reject candidate if the correlation or p-value does not meet the specified thresholds.
        if r >= r_threshold or p_value >= p_threshold:
            return False

        return True

    @staticmethod
    def multiprocess_filter(pool: mp.Pool,
                            candidates: list[list[int]],
                            function: callable,
                            args: tuple,
                            chunksize: int) -> list:

        """
        Apply a filtering function to a list of candidates using parallel processing.

        Parameters
        ----------
        pool: multiprocessing.Pool
            The multiprocessing pool managing worker processes.
        candidates: list[list[int]]
            List of candidate entries to be filtered.
        function: callable
            Filtering function that returns True for valid candidates and False otherwise.
        args: tuple
            Additional arguments to be passed to the filtering function.
        chunksize: int
            Number of candidates each worker should process in one batch.

        Returns
        ------
        list
            List of candidates that pass the filtering function.
        """
        # Parallelize the filtering operation using starmap; then select only candidates that return True.
        return [candidate for candidate, keep in zip(candidates,
                                                     pool.starmap(function,
                                                                  args,
                                                                  chunksize)) if keep]
