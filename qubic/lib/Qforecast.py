"""
renamed from analytical_forecast_lib.py
"""

import healpy as hp
import numpy as np

from qubic.lib.Instrument.Qinstrument import compute_freq
from qubic.lib.MapMaking.FrequencyMapMaking.Qspectra_component import CMBModel


class NoiseEquivalentTemperature:
    def __init__(self, NEPs, band, relative_bandwidth=0.25):
        self.band = band
        self.NEPs = NEPs
        self.T = 2.7255
        self.h = 6.62e-34
        self.k = 1.38e-23
        self.c = 3e8
        _, _, _, _, self.bw, _ = compute_freq(self.band, Nfreq=1, relative_bandwidth=relative_bandwidth)

        self.NETs = self._NEP2NET_db(np.sqrt(np.sum(self.NEPs**2)), self.band)

    def _get_derivative_Bnu_db(self, band):
        dnu = 0.5 * self.bw * 1e9
        nu = band * 1e9
        x = (self.h * nu) / (self.k * self.T)
        dIdT = ((2 * (self.h**2) * nu**4) / ((self.c**2) * self.k * self.T**2)) * (np.exp(x) / (np.exp(x) - 1) ** 2) * dnu

        return dIdT

    def _NEP2NET_db(self, NEP, band):
        dIdT = self._get_derivative_Bnu_db(band)

        return np.array([NEP / (np.sqrt(2) * (dIdT * 1e-12))])


class AnalyticalForecast:
    """

    Instance to produce analytical forecast

    Arguments :

        - nus : list
        - NEPdet : list
        - NEPpho : list

    """

    def __init__(self, nus, NEPdet, NEPpho, Nyrs=3, Nh=400, fsky=0.0182, nside=256, instr="DB"):
        ### Check type of inputs
        if instr == "DB":
            if type(NEPdet) is not list or type(NEPpho) is not list:
                raise TypeError("NEP type should be a list")

        ### Check length of inputs
        if instr == "DB":
            if len(NEPdet) != len(NEPpho):
                raise TypeError("NEPdet and NEPpho should have the same length")

        self.nside = nside  # Map pixelization
        self.Nyrs = Nyrs  # Nyrs
        self.Tobs = 3600 * 24 * 365 * self.Nyrs  # Observation time [s]
        self.Nh = Nh  # Number of horns (detectors for an imager)
        self.fsky = fsky  # Observed sky fraction
        self.nus = nus  # Physical bands
        self.nfreqs = len(NEPdet)  # Number of frequencies
        self.intr = instr

        ### Store NEPs
        if self.intr == "DB":
            self.NEPs = np.zeros((self.nfreqs, 2))
            for i in range(self.nfreqs):
                self.NEPs[i, 0] = NEPdet[i] * 2  # factor 2 because sig^2 = NEP^2 / (2 * Ts) ???
                self.NEPs[i, 1] = NEPpho[i]
        elif self.intr == "UWB":
            self.Nyrs *= 2  # Multiply Nyrs by two have the same effect than having twice less pointings
            self.Tobs = 3600 * 24 * 365 * self.Nyrs  # Observation time [s]
            self.NEPs = np.zeros((self.nfreqs, 3))
            for i in range(self.nfreqs):
                self.NEPs[i, 0] = NEPdet[0] * 2 * np.sqrt(2)
                self.NEPs[i, 1] = NEPpho[0]  # / np.sqrt(2)
                self.NEPs[i, 2] = NEPpho[1]  # / np.sqrt(2)

    def _get_effective_depths(self, NETs):
        """

        Convert Noise Equivalent Temperature in sensitivity depths | muK.sqrt(s) -> muK.arcmin

        """

        Omega = (4 * np.pi * self.fsky) / ((np.pi / (180 * 60)) ** 2)
        depths = 4 * np.sqrt((Omega * np.power(NETs, 2.0)) / (self.Tobs * self.Nh))  # * 1/np.sqrt(2)

        return depths

    def _get_power_spectra(self, depths, A, correlation=False):
        """

        Compute noise power spectrum in each components depending on size of A, assumed to be white.

                    Nl = (A.T . N^{-1} . A)^{-1}

        """

        fwhm = np.array([0.0041, 0.0041])
        bl = np.array([hp.gauss_beam(b, lmax=2 * self.nside) for b in fwhm])
        nl = (bl / np.radians(depths / 60.0)[:, np.newaxis]) ** 2
        AtNA = np.einsum("fi, fl, fj -> lij", A, nl, A)

        sig2_00 = np.linalg.pinv(AtNA) / hp.nside2resol(self.nside, arcmin=True)

        Nl = sig2_00[0, 0, 0]
        if correlation:
            Nl += np.sqrt(2) * sig2_00[0, 0, 1]
            # Nl += sig2_00[0, 0, 1]

        return Nl

    def _fisher(self, ell, Nl):
        """

        Fisher to compute sigma(r) for a given noise power spectrum.

        """

        ClBB = CMBModel(ell).give_cl_cmb(ell, r=1, Alens=0.0)
        s = np.sum((ell + 0.5) * self.fsky * (ClBB / Nl) ** 2) ** (-1 / 2)

        return s

    def main(self, A, ell, correlation=False):
        """

        Method to convert NEPs to sigma(r).

        """

        ### NEPs [W/sqrt(Hz)] -> NETs [muK.sqrt(s)]
        NETs = np.zeros(self.nfreqs)
        for i in range(self.nfreqs):
            NETs[i] = NoiseEquivalentTemperature(self.NEPs[i], self.nus[i]).NETs

        ### NETs [muK.sqrt(s)] -> depths [muK.arcmin]
        depths = self._get_effective_depths(NETs)

        ### depths [muK.arcmin] -> Nl [muK^2]
        Nl = self._get_power_spectra(depths, A, correlation=correlation)

        sigr = self._fisher(ell, Nl)

        return sigr
