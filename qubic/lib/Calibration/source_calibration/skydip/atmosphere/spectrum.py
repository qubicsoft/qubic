import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# nomi delle colonne attese nel CSV prodotto da am_runner.py.

AM_FREQUENCY_COLUMN = "frequency_GHz"
AM_TAU_COLUMN = "tau"
AM_TX_COLUMN = "tx"
AM_TRJ_COLUMN = "Trj_K"
AM_TB_COLUMN = "Tb_K"




@dataclass
class AtmosphericSpectrum:
    """
    Spettro atmosferico finale prodotto da AM.
    """

    frequency_GHz: np.ndarray
    tau: np.ndarray
    Tb_K: np.ndarray
    data: np.ndarray
    # CSV da cui è stato letto lo spettro
    source_path: Path


    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "AtmosphericSpectrum":

        csv_path = csv_path.expanduser().resolve()
        data = np.genfromtxt(csv_path, delimiter=",", names=True)

        if data.size == 0:
            raise ValueError(f"CSV file {csv_path} is empty.")

        if data.dtype.names is None:
            raise ValueError(f"CSV file {csv_path} does not contain a valid header.")

        required_columns = {AM_FREQUENCY_COLUMN, AM_TAU_COLUMN, AM_TB_COLUMN}
        missing_columns = required_columns - set(data.dtype.names)

        if missing_columns:
            raise ValueError(
                f"Missing required columns in {csv_path}: {sorted(missing_columns)}. "
                f"Available columns are: {data.dtype.names}.")

        order = np.argsort(data["frequency_GHz"])
        data = data[order]

        frequency_GHz = np.asarray(data[AM_FREQUENCY_COLUMN], dtype=np.float64)
        tau = np.asarray(data[AM_TAU_COLUMN], dtype=np.float64)
        Tb_K = np.asarray(data[AM_TB_COLUMN], dtype=np.float64)

       
        return cls(frequency_GHz=frequency_GHz,
                   tau=tau,
                   Tb_K=Tb_K,
                   data=data,
                   source_path=csv_path)

    @property
    def f_min_GHz(self) -> float:
        return float(self.frequency_GHz[0])

    @property
    def f_max_GHz(self) -> float:
        return float(self.frequency_GHz[-1])

    @property
    def bandwidth_GHz(self) -> float:
        return self.f_max_GHz - self.f_min_GHz

    @property
    def tau_area_GHz(self) -> float:
        """
        Area under tau(nu), using frequency in GHz.

        This is not an effective optical depth. The effective band-averaged
        optical depth is `band_averaged_tau`.
        """
        return float(np.trapz(self.tau, self.frequency_GHz))

    @property
    def band_averaged_tau(self) -> float:

        if self.bandwidth_GHz <= 0:
            raise ValueError("Bandwidth must be strictly positive.")

        return self.tau_area_GHz / self.bandwidth_GHz

    @property
    def Tb_area_KGHz(self) -> float:
        """
        Area under Tb(nu), using frequency in GHz.

        This is not an effective atmospheric brightness temperature. The
        effective band-averaged value is `band_averaged_Tb_K`.
        """
        return float(np.trapz(self.Tb_K, self.frequency_GHz))

    @property
    def band_averaged_Tb_K(self) -> float:

        if self.bandwidth_GHz <= 0:
            raise ValueError("Bandwidth must be strictly positive.")

        return self.Tb_area_KGHz / self.bandwidth_GHz

    def get_effective_atmospheric_parameters(
            self,
            strategy: str = "band_average",
    ) -> tuple[float, float]:
        """
        Return the effective atmospheric parameters used by the calibration.

        strategy="band_average"
            tau_eff = 1 / Δν ∫ tau(ν) dν
            Tb_eff  = 1 / Δν ∫ Tb(ν) dν

        strategy="max"
            tau_eff = max(tau)
            Tb_eff  = max(Tb_K)

        strategy="median"
            tau_eff = median(tau)
            Tb_eff  = median(Tb_K)

        strategy="min"
            tau_eff = min(tau)
            Tb_eff  = min(Tb_K)
        """
        if strategy == "band_average":
            return self.band_averaged_tau, self.band_averaged_Tb_K

        if strategy == "max":
            return float(np.nanmax(self.tau)), float(np.nanmax(self.Tb_K))

        if strategy == "median":
            return float(np.nanmedian(self.tau)), float(np.nanmedian(self.Tb_K))

        if strategy == "min":
            return float(np.nanmin(self.tau)), float(np.nanmin(self.Tb_K))

        raise ValueError(
            f"Unknown atmospheric parameter strategy {strategy!r}. "
            "Allowed values are: 'band_average', 'max', 'median', 'min'."
        )

    def atmospheric_temperature_vs_elevation(
        self,
        elevation_deg: np.ndarray,
        bandpass_response: np.ndarray | None = None,
        min_elevation_deg: float = 1.0,
    ) -> np.ndarray:
        elevation_deg = np.asarray(elevation_deg, dtype=np.float64)
        elevation_clipped = np.clip(elevation_deg, min_elevation_deg, None)
        airmass = 1.0 / np.sin(np.deg2rad(elevation_clipped))

        tau = np.asarray(self.tau, dtype=np.float64)
        tb_zenith = np.asarray(self.Tb_K, dtype=np.float64)
        zenith_emissivity = 1.0 - np.exp(-tau)

        valid = (
            np.isfinite(self.frequency_GHz)
            & np.isfinite(tau)
            & np.isfinite(tb_zenith)
            & np.isfinite(zenith_emissivity)
            & (zenith_emissivity > 0)
        )

        if np.count_nonzero(valid) < 2:
            raise ValueError("At least two valid atmospheric samples are required.")

        frequency = self.frequency_GHz[valid]
        tau_valid = tau[valid]
        tb_valid = tb_zenith[valid]

        t_eff = tb_valid / (1.0 - np.exp(-tau_valid))

        if bandpass_response is None:
            weights = np.ones_like(frequency, dtype=np.float64)
        else:
            weights_full = np.asarray(bandpass_response, dtype=np.float64)

            if weights_full.shape != self.frequency_GHz.shape:
                raise ValueError("bandpass_response must have the same shape as frequency_GHz.")

            weights = weights_full[valid]

        denominator = np.trapz(weights, frequency)

        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("The integrated bandpass response must be strictly positive.")

        t_atm = []

        for am in np.ravel(airmass):
            t_nu = t_eff * (1.0 - np.exp(-tau_valid * am))
            t_atm.append(float(np.trapz(t_nu * weights, frequency) / denominator))

        return np.asarray(t_atm, dtype=np.float64).reshape(elevation_deg.shape)

    def summary(self) -> dict[str, float | str]:
        return {
            "source_path": str(self.source_path),
            "f_min_GHz": self.f_min_GHz,
            "f_max_GHz": self.f_max_GHz,
            "bandwidth_GHz": self.bandwidth_GHz,
            "tau_area_GHz": self.tau_area_GHz,
            "Tb_area_KGHz": self.Tb_area_KGHz,
            "band_averaged_tau": self.band_averaged_tau,
            "band_averaged_Tb_K": self.band_averaged_Tb_K,
            "tau_min": float(np.nanmin(self.tau)),
            "tau_max": float(np.nanmax(self.tau)),
            "Tb_K_min": float(np.nanmin(self.Tb_K)),
            "Tb_K_max": float(np.nanmax(self.Tb_K)),
        }