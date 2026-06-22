import numpy as np
from pathlib import Path
from dataclasses import dataclass

# Fixed value for tau in AtmosphericSpectrum
TAU_FIXED = 0.1

from qubic.lib.Calibration.source_calibration.common.plotting import plot_atmospheric_spectrum


@dataclass
class AtmosphericSpectrum:
    """
    Atmospheric spectrum sampled over frequency.

    Parameters
    ----------
    frequency_GHz
        Frequency axis in GHz.
    tau
        Atmospheric opacity spectrum.
    Tb_K
        Planck brightness temperature spectrum in Kelvin.
    data
        Structured numpy array loaded from the CSV file.
    source_path
        Path of the CSV file used to build the object.
    """
    frequency_GHz: np.ndarray
    tau: np.ndarray
    Tb_K: np.ndarray
    data: np.ndarray
    source_path: Path

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "AtmosphericSpectrum":
        """
        Build an AtmosphericSpectrum from a CSV file.

        The CSV file is expected to contain at least the columns:
        - frequency_GHz
        - tau
        - Tb_K
        """
        csv_path = Path(csv_path)
        data = np.genfromtxt(csv_path, delimiter=",", names=True)

        if data.size == 0:
            raise ValueError(f"CSV file {csv_path} is empty.")

        if data.dtype.names is None:
            raise ValueError(f"CSV file {csv_path} does not contain a valid header.")

        required_columns = {"frequency_GHz", "tau", "Tb_K"}
        missing_columns = required_columns - set(data.dtype.names)
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {csv_path}: {sorted(missing_columns)}"
            )

        order = np.argsort(data["frequency_GHz"])
        data = data[order]

        frequency = np.asarray(data["frequency_GHz"], dtype=float)
        tau = np.full_like(frequency, TAU_FIXED, dtype=float)
        tb_k = np.asarray(data["Tb_K"], dtype=float)

        if frequency.ndim != 1 or tau.ndim != 1 or tb_k.ndim != 1:
            raise ValueError("frequency_GHz, tau and Tb_K must be 1D arrays.")
        if frequency.size < 2:
            raise ValueError("At least two frequency samples are required for integration.")
        if frequency.size != tau.size or frequency.size != tb_k.size:
            raise ValueError("frequency_GHz, tau and Tb_K must have the same length.")

        return cls(
            frequency_GHz=frequency,
            tau=tau,
            Tb_K=tb_k,
            data=data,
            source_path=csv_path,
        )

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
    def integrated_tau(self) -> float:
        """
        Integral of tau over the full frequency band.

        Units: GHz, since tau is dimensionless.
        """
        return float(np.trapz(self.tau, self.frequency_GHz))

    @property
    def integrated_Tb_KGHz(self) -> float:
        """
        Integral of Tb_K over the full frequency band.

        Units: K * GHz.
        """
        return float(np.trapz(self.Tb_K, self.frequency_GHz))

    @property
    def mean_tau_in_band(self) -> float:
        """
        Band-averaged tau over the full frequency interval.
        """
        if self.bandwidth_GHz <= 0:
            raise ValueError("Bandwidth must be strictly positive.")
        return self.integrated_tau / self.bandwidth_GHz

    @property
    def mean_Tb_K_in_band(self) -> float:
        """
        Band-averaged Tb_K over the full frequency interval.
        """
        if self.bandwidth_GHz <= 0:
            raise ValueError("Bandwidth must be strictly positive.")
        return self.integrated_Tb_KGHz / self.bandwidth_GHz

    def summary(self) -> dict[str, float | str]:
        """
        Return a compact summary of the loaded spectrum and of the band integrals.
        """
        return {
            "source_path": str(self.source_path),
            "f_min_GHz": self.f_min_GHz,
            "f_max_GHz": self.f_max_GHz,
            "bandwidth_GHz": self.bandwidth_GHz,
            "integrated_tau_GHz": self.integrated_tau,
            "integrated_Tb_KGHz": self.integrated_Tb_KGHz,
            "mean_tau_in_band": self.mean_tau_in_band,
            "mean_Tb_K_in_band": self.mean_Tb_K_in_band,
        }


if __name__ == "__main__":

    csv_path = Path(
        "/qubic/scripts/Calibration/source_calibration/skydip/data/weather/2026-03-11/ALMA_MAM_50_130_170GHz_z0deg_T288K_scale1.0.csv")
    spectrum = AtmosphericSpectrum.from_csv(csv_path)

    print("Atmospheric spectrum summary:")
    for key, value in spectrum.summary().items():
        print(f"  {key}: {value}")

    plot_atmospheric_spectrum(spectrum, y_key="tau", show=True)
    plot_atmospheric_spectrum(spectrum, y_key="Tb_K", show=True)

