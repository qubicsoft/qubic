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


# ========== Fitting Utilities ==========


def nll_gauss_model(model, freq, ps, sigma):
    """
    Return a Minuit-compatible function for any model.

    model : callable
        model(freq, *params)
    """

    def nll(*params):
        psd_model = model(freq, *params)
        return np.sum(0.5 * ((ps - psd_model) / sigma) ** 2)

    return nll


def make_minuit(nll, param_names, x0):
    """
    Create Minuit object with named parameters.
    """

    kwargs = dict(zip(param_names, x0))
    m = Minuit(nll, **kwargs)

    for name in param_names:
        m.limits[name] = (0, None)

    return m


def fit_minuit_ll_from_ps(
    freq,
    ps,
    model,
    x0,
    param_names,
    nbins=300,
    plot=False,
    log=True,
):
    freq, ps = freq[1:], ps[1:]
    ps /= np.mean(ps[-10:])

    if plot:
        plt.figure(dpi=150)

    binned_freq, binned_ps, _, binned_ps_error, _ = ft.profile(
        freq, ps, nbins=nbins, plot=plot, log=log
    )

    binned_ps_error = np.maximum(binned_ps_error, 1e-20)

    # Build NLL
    nll = nll_gauss_model(model, binned_freq, binned_ps, binned_ps_error)

    # Minuit
    m = make_minuit(nll, param_names, x0)

    m.migrad()
    m.hesse()

    # Plot
    if plot:
        plt.plot(freq, ps, label="Data")
        plt.plot(freq, model(freq, *m.values), "k", label="Fit")

        ndof = len(binned_ps) - len(x0)
        chi2 = np.sum(
            ((binned_ps - model(binned_freq, *m.values)) / binned_ps_error) ** 2
        )
        reduced_chi2 = chi2 / ndof

        fit_info = [f"$\\chi^2$/ndof = {chi2:.2f} / {ndof} = {reduced_chi2:.2f}"]

        for p in param_names:
            v = m.values[p]
            e = m.errors[p]
            fit_info.append(f"{p} = ${v:.2e} \\pm {e:.2e}$")

        plt.legend(title="\n".join(fit_info))
        plt.xscale("log")
        plt.yscale("log")
        plt.show()

    return m.values, m.errors, m.fmin.reduced_chi2


def fit_minuit_ll(
    data,
    timestep,
    model,
    x0,
    param_names,
    nbins=300,
    plot=False,
    log=True,
):
    # Compute PSD
    freq, ps = compute_real_ps(data, timestep=timestep)
    freq, ps = freq[1:], ps[1:]
    ps /= np.mean(ps[-10:])

    if plot:
        plt.figure(dpi=150)

    binned_freq, binned_ps, _, binned_ps_error, _ = ft.profile(
        freq, ps, nbins=nbins, plot=plot, log=log
    )

    binned_ps_error = np.maximum(binned_ps_error, 1e-20)

    # Generic NLL
    nll = nll_gauss_model(model, binned_freq, binned_ps, binned_ps_error)

    # Minuit
    m = make_minuit(nll, param_names, x0)

    m.migrad()
    m.hesse()

    if plot:
        plt.plot(freq, ps, label="Data")
        plt.plot(freq, model(freq, *m.values), "k", label="Fit")

        ndof = len(binned_ps) - len(x0)
        chi2 = np.sum(
            ((binned_ps - model(binned_freq, *m.values)) / binned_ps_error) ** 2
        )
        reduced_chi2 = chi2 / ndof

        fit_info = [f"$\\chi^2$/ndof = {chi2:.2f} / {ndof} = {reduced_chi2:.2f}"]

        for p in param_names:
            v = m.values[p]
            e = m.errors[p]
            fit_info.append(f"{p} = ${v:.2e} \\pm {e:.2e}$")

        plt.legend(title="\n".join(fit_info))
        plt.xscale("log")
        plt.yscale("log")
        plt.show()

    return m.values, m.errors, m.fmin.reduced_chi2
