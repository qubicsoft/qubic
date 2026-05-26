import numpy as np
import scipy as sp
from typing import Optional
from scipy.stats import pearsonr, linregress
from scipy.optimize import curve_fit

import multiprocessing as mp

from cosmic_rays.results import FitResult


class Model:
    """
    A module for handling exponential decay modeling, fitting with optimization, and assessing fit quality.

    This module provides tools for estimating parameters of an exponential decay model, performing fits on
    signal data, calculating uncertainties, and analyzing fit results. The main goal is to enable robust
    and accurate parameter estimation in scenarios where the signal adheres to an exponential decay profile.

    Classes
    -------
    Model
        Contains static methods for modeling and fitting exponential decay functions.
    """

    @staticmethod
    def get_initial_params(x: np.ndarray, y: np.ndarray) -> tuple:
        """
        Calculates initial parameters for the exponential decay function used in curve fitting.

        This function estimates the parameters `a`, `b`, and `c` for the exponential decay model:
            f(t) = a + b * exp(c * t)

        The approach leverages numerical integration and least squares to approximate the initial
        values for the parameters, which are useful for initializing optimization routines.

        Thanks to:
        https://it.scribd.com/doc/14674814/Regressions-et-equations-integrales
        https://stackoverflow.com/questions/77822770/exponential-fit-is-failing-in-some-cases/77840735#77840735

        :param x: Array of time points corresponding to the exponential decay segment.
        :type: np.ndarray

        :param y:  Array of signal values corresponding to the exponential decay segment.
        :type: np.ndarray

        :return: A tuple (a, b, c) representing the estimated initial parameters for the exponential
            decay function, where:
            - a (float): Offset or baseline of the exponential function.
            - b (float): Amplitude or scale of the exponential term.
            - c (float): Rate constant of the exponential term.
        :rtype: tuple
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
        ab11 = len(y)  # Number of data points
        ab12 = np.sum(np.exp(c * x))  # Sum of exponentials scaled by c
        ab21 = ab12  # Symmetric entry: ab21 equals ab12
        ab22 = np.sum(np.exp(2 * c * x))  # Sum of squared exponentials
        abf11 = np.sum(y)  # Sum of signal values
        abf21 = np.sum(y * np.exp(c * x))  # Sum of signal values weighted by exponential factor

        # Form the matrix and the factor vector for parameters a and b
        ab_matrix = np.array([[ab11, ab12],
                              [ab21, ab22]])

        ab_factor = np.array([[abf11],
                              [abf21]])

        # Replace any zero entries in ab_matrix with a small value to prevent division by zero
        ab_matrix = np.where(ab_matrix == 0, 1e-10, ab_matrix)

        # Solve the linear system to get a and b.
        a, b = np.linalg.inv(ab_matrix) @ ab_factor

        return a, b, c

    @staticmethod
    def exp_model_uncertainty_jacobian(t, popt, pcov):
        """
        Calculates the uncertainty of the model output using the Jacobian matrix and
        matrix multiplication.

        The method computes the uncertainties associated with the model prediction
        for given time values `t`, based on the optimal parameters and the covariance
        matrix of the parameters.

        :param t: Time values
        :type: array-like

        :param popt: Optimal parameters of the model [a, b, c]
        :type: list or array

        :param pcov: Covariance matrix associated with the parameters
        :type: 2D array

        :return: Calculated uncertainties for each value in `t`
        :rtype: array
        """

        a, b, c = popt
        # Construct the Jacobian for each t; each row corresponds to [dy/da, dy/db, dy/dc]
        J = np.empty((len(t), 3))
        J[:, 0] = 1
        J[:, 1] = np.exp(c * t)
        J[:, 2] = b * t * np.exp(c * t)
        # Calculate the uncertainty as sigma_y = sqrt(J * pcov * J^T) for each row of J
        sigma_y = np.sqrt(np.einsum('ij,jk,ik->i', J, pcov, J))

        return sigma_y

    @staticmethod
    def exp_decay(t: np.ndarray, a: int, b: int, c: float) -> np.ndarray:
        """
        Exponential decay model function.

        Models the signal as: f(t) = a + b * exp(c * t)

        :param t: Time array at which to evaluate the function
        :type: np.ndarray

        :param a: Baseline (steady state) value
        :type: int

        :param b: Amplitude of the exponential component
        :type: int

        :param c: Decay rate, equal to -1/tau (where tau is the time constant)
        :type: float

        :return: Computed values of the exponential decay model
        :rtype: np.ndarray
        """
        return a + b * np.exp(c * t)

    @staticmethod
    def get_fit_candidate(time_raw: np.ndarray, s_clean: np.ndarray, std: np.ndarray, eps=0.001) -> Optional[FitResult]:

        """
        Perform an exponential decay fit on a valid candidate signal segment.

        :param time_raw: Array of candidate time values
        :type: np.ndarray

        :param s_clean: Array of candidate signal values
        :type: np.ndarray

        :param std: Standard deviation of the signal, used for weighting in the fit. Computed as the
            standard deviation of extra points before and after
        :type: np.ndarray

        :param eps: A small value added for numerical stability (default is 0.001)
        :type: float, optional


        :return: If the fit is acceptable (p-value > 0.05), returns a FitResult object containing:
              - Optimal fit parameters (a, b, c)
              - Uncertainty on the time constant (sigma_tau)
              - Degrees of freedom (nu)
              - Reduced chi-square statistic (chi_square_reduced)
              - P-value (p_value)
              - Slope and intercept of the linear fit (slope, intercept)
              - Standard error of the slope (slope_sigma)
              - P-value of the slope (slope_p_value)
              - Normalized residuals (residuals divided by y)
              - Covariance matrix of the fit (pcov)
            If the fit is invalid or maximum iteration is reached, returns None.

        :rtype: Optional[FitResult]
        """

        # Identify the index of the maximum value in the candidate signal, marking the start of the decay
        sig_max_idx = s_clean.argmax()

        # Perform a linear regression on the signal segment forming the rising part before the exponential decrease
        lin_fit = linregress(time_raw[:sig_max_idx + 1], s_clean[:sig_max_idx + 1])

        # Define the time array for the decay phase, shifting it so that t=0 corresponds to the maximum
        x = time_raw[sig_max_idx:] - time_raw[sig_max_idx]
        # Extract the corresponding signal values for the decay phase
        y = s_clean[sig_max_idx:]

        eps = 1e-12
        try:
            n_iter = 0
            while True:
                # Estimate initial parameters for the exponential decay fit
                p0 = Model.get_initial_params(x, y)
                # absolute_sigma is set to False because the uncertainties are not very precise.
                # In other words, sigma is scaled to match the sample variance of the residuals after the fit:
                # pcov(absolute_sigma=False) = pcov(absolute_sigma=True) * chisq(popt)/(M-N)
                # M: number of data points, N: number of parameters
                # Since the noise level is only approximately known, sigma is used here as a relative weight
                # rather than as an absolute measurement uncertainty. By setting absolute_sigma=False,
                # the parameter covariance matrix is rescaled according to the actual scatter of the residuals,
                # leading to more realistic uncertainties on the fitted parameters when the noise model is imperfect.
                popt, pcov = curve_fit(Model.exp_decay, x, y, p0=p0, sigma=[std + eps] * y.shape[0], absolute_sigma=False)

                # Compute the residuals between the model and the observed data
                residuals = Model.exp_decay(x, *popt) - y

                # Calculate the degrees of freedom: number of data points minus number of parameters
                nu = len(y) - len(popt)

                # estimate the standard deviation of the residuals
                new_std = residuals.std()
                if np.fabs(new_std - std) / (std + eps) < eps or (n_iter := n_iter + 1) >= 20:
                    break
                else:
                    std = new_std

            if n_iter >= 20:
                return None
            # Compute the chi-square statistic: sum of squared residuals normalized by the variance
            chi_square = np.sum((residuals ** 2) / (std + eps) ** 2)

            # Calculate the reduced chi-square (chi_square per degree of freedom)
            chi_square_reduced = chi_square / nu

            # Significance: if the p-value is less than the significance threshold of 0.05,
            # we reject the null hypothesis H_0 (the model fits the data).
            # If the p-value is greater than 0.05, we accept the null hypothesis
            p_value = 1 - sp.stats.chi2.cdf(chi_square, nu)

            # Calculate the uncertainty on the time constant parameter
            sigma_tau = np.sqrt(pcov[2][2]) / popt[2] ** 2

            fit_res = FitResult(
                a=popt[0],
                b=popt[1],
                c=popt[2],
                sigma_tau=sigma_tau,
                nu=nu,
                chi_square_reduced=chi_square_reduced,
                p_value=p_value,
                slope=lin_fit.slope,
                intercept=lin_fit.intercept,
                slope_sigma=lin_fit.stderr,
                slope_p_value=lin_fit.pvalue,
                residuals=(residuals / y).tolist(),
                pcov=pcov.tolist(),
                index=None)
            return fit_res if fit_res.is_valid() else None

        except:
            return None


    @staticmethod
    def candidate_filter(tm_candidate: np.ndarray,
                         s_candidate: np.ndarray,
                         noise_before: np.ndarray,
                         noise_after: np.ndarray,
                         pts_vertical_trend: int,
                         r_threshold: float = -0.9,
                         p_threshold: float = 0.05,
                         rmse_threshold: float = 0.02) -> bool:

        """
        Filter a candidate signal based on various criteria to determine its validity.

        A valid candidate must demonstrate an initial vertical linear growth phase followed by
        exponential decay. Various checks are performed, including positivity constraints, relative
        difference limits, correlation thresholds, statistical consistency, and a fitted decay validation.

        :param tm_candidate:  Time values correspond to the candidate
        :type: np.ndarray

        :param s_candidate: Signal values of the candidate
        :type: np.ndarray

        :param noise_before: Noise values observed before the candidate signal
        :type: np.ndarray

        :param noise_after: Noise values observed after the candidate signal
        :type: np.ndarray

        :param pts_vertical_trend: Minimum number of points required in the vertical growth phase
        :type: int

        :param r_threshold: Correlation coefficient threshold;
            candidates with values above this (less negative) are rejected. Defaults to -0.9
        :type: float, optional

        :param p_threshold: P-value threshold; candidates with values above this are rejected. Defaults to 0.05
        :type: float, optional

        :param rmse_threshold:  Maximum root-mean-square error accepted for the decay model fit. Defaults to 0.02
        :type: float, optional

        :return: True if the candidate satisfies all criteria; False otherwise
        :rtype: bool
        """

        # rejects the candidate if the maximum value of noise_before points before or the maximum value of noise_after
        # points after the candidate itself are greater than the average value of the candidate
        if noise_before is not None and noise_before.max() > s_candidate.mean():
            return False

        if noise_after is not None and noise_after.max() > s_candidate.mean():
            return False

        # Reject the candidate if any signal value is zero or negative
        if np.any(s_candidate <= 0.):
            return False

        # Compute the relative difference between the first and last signal values
        ratio = abs(s_candidate[-1] - s_candidate[0]) / max(s_candidate[0], s_candidate[-1])
        # Reject if the relative difference is greater than 1.5%
        if ratio > 0.015:
            return False

        # Find the peak index and require at least pts_vertical_trend samples before the peak
        # to estimate the pre-peak trend reliably
        if (sig_max_idx := s_candidate.argmax()) + 1 < pts_vertical_trend:
            return False

        # Select the exponential decay part of the candidate signal
        s_exp_decay = s_candidate[sig_max_idx:]
        t_exp_decay = tm_candidate[sig_max_idx:]

        # Compute the natural logarithm of the decay segment to apply the pearson correlation test
        log_s = np.log(s_exp_decay)

        # Calculate differences between consecutive log-transformed values
        diff_s = np.diff(log_s)
        # Compute the mean and standard deviation of these differences
        mu, sigma = diff_s.mean(), diff_s.std()

        # Reject the candidate if differences deviate by more than 2 sigma from the mean
        if not np.all(np.abs(diff_s - mu) < 2 * sigma):
            return False

        # Calculate Pearson correlation coefficient and p-value between time and log(signal)
        r, p_value = pearsonr(t_exp_decay, log_s)

        # Reject a candidate if the correlation or p-value does not meet the specified thresholds
        if r > r_threshold or p_value > p_threshold:
            return False

        # Fit log_s = m * t + q
        slope, intercept, _r, _p, _stderr = linregress(t_exp_decay, log_s)
        pred_log_s = slope * t_exp_decay + intercept
        rmse = np.sqrt(np.mean((log_s - pred_log_s) ** 2))

        # Reject if the decay is not well described by an exponential (RMSE too high)
        if rmse > rmse_threshold or slope >= 0:
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

        This method allows efficient parallel execution of a filtering function over
        a list of candidates using a multiprocessing pool. It distributes tasks across
        worker processes and collects only the candidates that pass the filtering
        condition.

        :param pool: The multiprocessing pool managing worker processes
        :type: multiprocessing.Pool

        :param candidates: List of candidate entries to be filtered
        :type: list[list[int]]

        :param function: Filtering function that returns True for valid candidates and False otherwise
        :type: callable

        :param args: Additional arguments to be passed to the filtering function.
        :type: tuple

        :param chunksize: Number of candidates each worker should process in one batch.
        :type: int

        :return: List of candidates that pass the filtering function.
        :rtype: list
        """

        # Parallelize the filtering operation using starmap; then select only candidates that return True
        return [candidate for candidate, keep in zip(candidates,
                                                     pool.starmap(function,
                                                                  args,
                                                                  chunksize)) if keep]
