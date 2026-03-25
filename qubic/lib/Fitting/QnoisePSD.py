"""
QnoisePSD.py
============

Tools for computing and fitting noise power spectral densities (PSD).

This module provides utilities to estimate power spectra from time-domain
data and to fit common noise models such as white noise, 1/f noise, and
generalized power-law spectra.

Author
------
Tom Laclavère
"""

import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit

import qubic.lib.Calibration.Qfiber as ft

# ========== Compute Power Spectrum ==========


def compute_ps(data, timestep):
    """
    Compute the two-sided Power Spectral Density (PSD) of a time-domain signal.

    Parameters
    ----------
    data : array_like
        Input time-domain signal. Must be one-dimensional.
    timestep : float
        Sampling interval of the signal (seconds per sample).

    Returns
    -------
    freq : ndarray
        Frequencies corresponding to the PSD values (Hz).
        Ranges from -Nyquist to +Nyquist.
    ps : ndarray
        Two-sided power spectral density of the signal.
        Units are [signal units]^2 / Hz.
    """

    N = data.size

    freq = np.fft.fftfreq(N, d=timestep)
    ps = (timestep / N) * np.abs(np.fft.fft(data)) ** 2

    return freq, ps


def compute_real_ps(data, timestep):
    """
    Compute the one-sided Power Spectral Density (PSD) of a real-valued signal.

    Parameters
    ----------
    data : array_like
        Real-valued time-domain signal. Must be one-dimensional.
    timestep : float
        Sampling interval of the signal (seconds per sample).

    Returns
    -------
    freq : ndarray
        Positive frequencies corresponding to the PSD values (Hz), from 0 to Nyquist.
    ps : ndarray
        One-sided power spectral density of the signal.
        Units are [signal units]^2 / Hz.

    Notes
    -----
    - One-sided correction is applied: all bins except the first and last are multiplied by 2.
    """

    N = data.size

    freq = np.fft.rfftfreq(N, d=timestep)
    ps = (timestep / N) * np.abs(np.fft.rfft(data)) ** 2

    # One-sided correction
    if N > 2:
        ps[1:-1] *= 2

    return freq, ps


# ========== Noise Models ==========


def white_noise(f, A_white):
    """
    White noise PSD.

    Parameters
    ----------
    f : array_like
        Frequencies.
    A_white : float
        Amplitude of the white noise (rms).

    Returns
    -------
    psd : ndarray
        Power spectral density of white noise.
    """

    return A_white**2 * np.ones_like(f)


def one_over_f_noise(f, A_f, alpha):
    """
    1/f^alpha noise PSD.

    Parameters
    ----------
    f : array_like
        Frequencies (should be >0 to avoid division by zero).
    A_f : float
        Amplitude of 1/f noise at f=1 Hz.
    alpha : float
        Exponent of 1/f noise.

    Returns
    -------
    psd : ndarray
        Power spectral density of 1/f^alpha noise.
    """

    return A_f**2 / f**alpha


def combined_noise(f, A_white, f_knee, alpha):
    """
    Combination of white noise + 1/f^alpha noise.

    PSD = white + 1/f^alpha

    Parameters
    ----------
    f : array_like
        Frequencies.
    A_white : float
        Amplitude of white noise.
    f_knee : float
        "Knee" frequency where the 1/f^alpha noise equals the white noise level (Hz).
    alpha : float
        Exponent of 1/f noise.

    Returns
    -------
    psd : ndarray
        Combined PSD.
    """

    return A_white**2 * (1 + np.abs(f_knee / f) ** alpha)


def noise_model(f, *args):
    """
    Dispatch noise model based on number of parameters.

    Parameters
    ----------
    f : array_like
        Frequencies.
    args : tuple
        Parameters of the noise model:
        - 1 param: white noise amplitude
        - 2 params: A_f, alpha for 1/f^alpha
        - 3 params:*args for combined

    Returns
    -------
    psd : ndarray
        Power spectral density evaluated at f.
    """
    print(args)
    print(args[0])
    if len(args) == 1:
        return white_noise(f, *args)

    elif len(args) == 2:
        return one_over_f_noise(f, *args)

    elif len(args) == 3:
        return combined_noise(f, *args)

    else:
        raise ValueError("Undefined noise model. Please adjust your parameters.")


# ========== Fitting Utilities ==========


def nll_gauss(f, psd, sigma, *args):
    """
    Gaussian negative log-likelihood for PSD fitting.

    Parameters
    ----------
    A : float
        White noise amplitude.
    f_knee : float
        Knee frequency of 1/f noise.
    alpha : float
        Exponent of 1/f noise.
    freq : ndarray
        Frequency array.
    ps : ndarray
        Observed PSD.
    sigma : float or ndarray
        Standard deviation of each PSD point.

    Returns
    -------
    nll : float
        Negative log-likelihood.
    """

    psd_model = noise_model(f, *args)

    return np.sum(
        0.5 * ((psd - psd_model) / sigma) ** 2 + 0.5 * np.log(2 * np.pi * sigma**2)
    )


def nll_gauss_wrapper(freq, ps, sigma, n_params):
    """
    Return a function with explicit signature for Minuit
    """
    if n_params == 1:

        def f(A_white):
            return np.sum(0.5 * ((ps - noise_model(freq, A_white)) / sigma) ** 2)
    elif n_params == 2:

        def f(A_f, alpha):
            return np.sum(0.5 * ((ps - noise_model(freq, A_f, alpha)) / sigma) ** 2)
    elif n_params == 3:

        def f(A_white, f_knee, alpha):
            return np.sum(
                0.5 * ((ps - noise_model(freq, A_white, f_knee, alpha)) / sigma) ** 2
            )
    else:
        raise ValueError("Only 1, 2, or 3 parameters supported")
    return f


def fit_minuit_ll_from_ps(freq, ps, noise_model_x0, nbins=300, plot=False, log=True):
    """
    Fit a binned power spectral density using Minuit with Gaussian likelihood.
    Automatically selects the noise model (white, 1/f, or combined) based on the
    length of `noise_model_x0`.

    Parameters
    ----------
    freq : ndarray
        Frequencies corresponding to the PSD values (Hz).
        Ranges from -Nyquist to +Nyquist.
    ps : ndarray
        Two-sided power spectral density of the signal.
        Units are [signal units]^2 / Hz.
    noise_model_x0 : list or tuple
        Initial guesses for the model parameters:
        - 1 element: [A_white] (white noise)
        - 2 elements: [A_f, alpha] (1/f^alpha)
        - 3 elements: [A_white, f_knee, alpha] (combined)
    nbins : int, optional
        Number of bins for PSD averaging.
    plot : bool, optional
        If True, plot the PSD and fit.
    data_name : str, optional
        Dataset name for plotting.
    index : int or None, optional
        Index label for plotting.

    Returns
    -------
    values : dict
        Best-fit parameter values.
    errors : dict
        Parameter uncertainties.
    reduced_chi2 : float
        Reduced chi-squared of the fit.
    """
    freq, ps = freq[1:], ps[1:]  # skip DC
    ps /= ps[0]

    # Bin PSD
    if plot:
        plt.figure(dpi=150)
    binned_freq, binned_ps, _, binned_ps_error, _ = ft.profile(
        freq, ps, nbins=nbins, plot=plot, log=log
    )

    binned_ps_error = np.maximum(binned_ps_error, 1e-20)

    # Log Likelihood for Noise model
    n_params = len(noise_model_x0)
    nll = nll_gauss_wrapper(binned_freq, binned_ps, binned_ps_error, n_params)

    # Initialize Minuit with initial guesses
    m = Minuit(nll, *noise_model_x0)

    # Set limits (all positive)
    for i, val in enumerate(noise_model_x0):
        m.limits[i] = (0, None)

    # Run fit
    m.migrad()
    m.hesse()

    # Plot
    if plot:
        plt.plot(freq, ps, label="Data")
        plt.plot(freq, noise_model(freq, *m.values), "k", label="Fit")

        ndof = len(binned_ps) - len(noise_model_x0)
        print("binned_ps :", len(binned_ps))
        print("noise_model : ", len(noise_model_x0))
        print("ndof : ", ndof)
        chi2 = np.sum(
            ((binned_ps - noise_model(binned_freq, *m.values)) / binned_ps_error) ** 2
        )
        print("chi2 : ", chi2)
        reduced_chi2 = chi2 / ndof

        # fit_info = [
        #     f"$\\chi^2$/ndof = {m.fval:.2f} / {m.ndof} = {m.fmin.reduced_chi2:.2f}"
        # ]

        fit_info = [f"$\\chi^2$/ndof = {chi2:.2f} / {ndof} = {reduced_chi2:.2f}"]
        for p, v, e in zip(m.parameters, m.values, m.errors):
            fit_info.append(f"{p} = ${v:.2e} \\pm {e:.2e}$")

        plt.title("PSD Fit")
        plt.legend(title="\n".join(fit_info))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel(r"Power Spectrum ($[signal]^2/Hz$)")
        plt.xscale("log")
        plt.yscale("log")
        plt.show()

    return m.values, m.errors, m.fmin.reduced_chi2


def fit_minuit_ll(data, timestep, noise_model_x0, nbins=300, plot=False, log=True):
    """
    Fit a binned power spectral density using Minuit with Gaussian likelihood.
    Automatically selects the noise model (white, 1/f, or combined) based on the
    length of `noise_model_x0`.

    Parameters
    ----------
    data : array_like
        1D time-domain signal.
    timestep : float
        Sampling interval of the signal (s).
    noise_model_x0 : list or tuple
        Initial guesses for the model parameters:
        - 1 element: [A_white] (white noise)
        - 2 elements: [A_f, alpha] (1/f^alpha)
        - 3 elements: [A_white, f_knee, alpha] (combined)
    nbins : int, optional
        Number of bins for PSD averaging.
    plot : bool, optional
        If True, plot the PSD and fit.
    data_name : str, optional
        Dataset name for plotting.
    index : int or None, optional
        Index label for plotting.

    Returns
    -------
    values : dict
        Best-fit parameter values.
    errors : dict
        Parameter uncertainties.
    reduced_chi2 : float
        Reduced chi-squared of the fit.
    """

    # Compute PSD
    freq, ps = compute_real_ps(data, timestep=timestep)
    freq, ps = freq[1:], ps[1:]  # skip DC
    ps /= ps[0]

    # Bin PSD
    if plot:
        plt.figure(dpi=150)
    binned_freq, binned_ps, _, binned_ps_error, _ = ft.profile(
        freq, ps, nbins=nbins, plot=plot, log=log
    )
    binned_ps_error = np.maximum(binned_ps_error, 1e-20)

    # Log Likelihood for Noise model
    n_params = len(noise_model_x0)
    nll = nll_gauss_wrapper(binned_freq, binned_ps, binned_ps_error, n_params)

    # Initialize Minuit with initial guesses
    m = Minuit(nll, *noise_model_x0)

    # Set limits (all positive)
    for i, val in enumerate(noise_model_x0):
        m.limits[i] = (0, None)

    # Run fit
    m.migrad()
    m.hesse()

    # Plot
    if plot:
        plt.plot(freq, ps, label="Data")
        plt.plot(freq, noise_model(freq, *m.values), "k", label="Fit")

        ndof = len(binned_ps) - len(noise_model_x0)
        print("binned_ps :", len(binned_ps))
        print("noise_model : ", len(noise_model_x0))
        print("ndof : ", ndof)
        chi2 = np.sum(
            ((binned_ps - noise_model(binned_freq, *m.values)) / binned_ps_error) ** 2
        )
        print("chi2 : ", chi2)
        reduced_chi2 = chi2 / ndof

        # fit_info = [
        #     f"$\\chi^2$/ndof = {m.fval:.2f} / {m.ndof} = {m.fmin.reduced_chi2:.2f}"
        # ]

        fit_info = [f"$\\chi^2$/ndof = {chi2:.2f} / {ndof} = {reduced_chi2:.2f}"]

        for p, v, e in zip(m.parameters, m.values, m.errors):
            fit_info.append(f"{p} = ${v:.2e} \\pm {e:.2e}$")

        plt.title("PSD Fit")
        plt.legend(title="\n".join(fit_info))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel(r"Power Spectrum ($[signal]^2/Hz$)")
        plt.xscale("log")
        plt.yscale("log")
        plt.show()

    return m.values, m.errors, m.fmin.reduced_chi2
