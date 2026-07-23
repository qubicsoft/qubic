import numpy as np
from scipy.signal import welch
from scipy.optimize import curve_fit
from dataclasses import dataclass, field

from qubic.lib.Calibration.source_calibration.skydip.config_calibration import SkydipCalibrationConfig



@dataclass(frozen=True)
class SkydipNoiseSpectrum:
    skydip_id: int
    direction: str
    start_idx: int
    stop_idx: int
    azimuth_mean: float

    frequency_hz: np.ndarray
    asd_adu_per_sqrt_hz: np.ndarray
    asd_k_per_sqrt_hz: np.ndarray | None = None


    @classmethod
    def from_segment(cls,
                     segment,
                     config: SkydipCalibrationConfig,
                     conversion_factor_adu_per_k: float | None = None):

        if isinstance(segment, dict):
            skydip_id = int(segment["skydip_id"])
            direction = str(segment["direction"])
            start_idx = int(segment["start_idx"])
            stop_idx = int(segment["stop_idx"])
            azimuth_mean = float(segment["azimuth_mean"])
            time = np.asarray(segment["time"], dtype=np.float64)
            signal = np.asarray(segment["signal"], dtype=np.float64)


        else:
            skydip_id = int(segment.skydip_id)
            direction = str(segment.direction)
            start_idx = int(segment.start_idx)
            stop_idx = int(segment.stop_idx)
            azimuth_mean = float(segment.azimuth_mean)
            time = np.asarray(segment.time, dtype=np.float64)
            signal = np.asarray(segment.signal, dtype=np.float64)

        noverlap = config.noise.noverlap

        if noverlap in (None, "None", "none", ""):
            noverlap = None

        else:
            noverlap = min(int(noverlap), config.noise.nperseg - 1)


        sampling_frequency = 1 / float(np.median(np.diff(time)))
        frequency_hz, psd_adu2_per_hz = (welch(
            signal,
            fs=sampling_frequency,
            window=config.noise.window,
            noverlap=noverlap,
            scaling=config.noise.scaling,
            nperseg=config.noise.nperseg,
            detrend=config.noise.detrend))

        asd_adu_per_sqrt_hz = np.sqrt(psd_adu2_per_hz)

        if conversion_factor_adu_per_k is None:
            asd_k_per_sqrt_hz = None
        else:
            asd_k_per_sqrt_hz = asd_adu_per_sqrt_hz / abs(conversion_factor_adu_per_k)

        return cls(
            skydip_id=skydip_id,
            direction=direction,
            start_idx=start_idx,
            stop_idx=stop_idx,
            azimuth_mean=azimuth_mean,
            frequency_hz=frequency_hz,
            asd_adu_per_sqrt_hz=asd_adu_per_sqrt_hz,
            asd_k_per_sqrt_hz=asd_k_per_sqrt_hz,
        )

def compute_all_skydip_noise_spectra(segments: list,
                                     config: SkydipCalibrationConfig,
                                     conversion_factor_adu_per_k: float | None = None) -> list[SkydipNoiseSpectrum]:

    noise_spectra: list[SkydipNoiseSpectrum] = []

    for segment in segments:
        spectrum = SkydipNoiseSpectrum.from_segment(
            segment=segment,
            config=config,
            conversion_factor_adu_per_k=conversion_factor_adu_per_k,
        )

        noise_spectra.append(spectrum)

    return noise_spectra



@dataclass(frozen=True)
class DatasetNoiseSummary:
    tau_eff: float
    asd_unit: str

    mean_knee_frequency_hz: float
    std_knee_frequency_hz: float

    mean_white_noise_asd: float
    std_white_noise_asd: float

    mean_alpha: float
    std_alpha: float

    mean_cutoff_frequency_hz: float
    std_cutoff_frequency_hz: float

    n_successful_fits: int
    n_failed_fits: int


    @staticmethod
    def _asd_model(
            frequency_hz: np.ndarray,
            white_noise_asd_k_per_sqrt_hz: float,
            knee_frequency_hz: float,
            alpha: float,
            cutoff_frequency_hz: float) -> np.ndarray:
        """
        ASD model:

        ASD(f) = white_noise_asd
                 * sqrt(1 + (f_knee / f)^alpha)
                 * exp(-f / f_cutoff)
        """

        frequency_hz = np.asarray(frequency_hz, dtype=np.float64)

        return (
            white_noise_asd_k_per_sqrt_hz
            * np.sqrt(1.0 + (knee_frequency_hz / frequency_hz) ** alpha)
            * np.exp(-frequency_hz / cutoff_frequency_hz)
        )

    @classmethod
    def from_spectra(
            cls,
            noise_spectra: list[SkydipNoiseSpectrum],
            tau_eff: float,
            plateau_fraction: float = 0.25,
            asd_unit: str = "K") -> "DatasetNoiseSummary":

        if asd_unit not in ("K", "ADU"):
            raise ValueError(
                "asd_unit must be either 'K' or 'ADU'. "
                f"Got {asd_unit!r}."
            )

        selected_spectra = []

        for spectrum in noise_spectra:
            if asd_unit == "K":
                if spectrum.asd_k_per_sqrt_hz is None:
                    raise ValueError(
                        "Expected calibrated spectra when asd_unit='K'."
                    )

            selected_spectra.append(spectrum)

        if len(selected_spectra) == 0:
            raise ValueError("No noise spectra were provided.")

        fit_knee_frequencies_hz = []
        fit_white_noise_asds = []
        fit_alphas = []
        fit_cutoff_frequencies_hz = []

        n_failed_fits = 0

        for spectrum in selected_spectra:

            frequency_hz = np.asarray(spectrum.frequency_hz, dtype=np.float64)

            if asd_unit == "K":
                asd = np.asarray(spectrum.asd_k_per_sqrt_hz, dtype=np.float64)
            else:
                asd = np.asarray(spectrum.asd_adu_per_sqrt_hz, dtype=np.float64)

            valid_fit = (
                    np.isfinite(frequency_hz)
                    & np.isfinite(asd)
                    & (frequency_hz > 0.0)
                    & (asd > 0.0)
            )

            fit_frequency = frequency_hz[valid_fit]
            fit_asd = asd[valid_fit]

            if fit_frequency.size < 4:
                n_failed_fits += 1
                continue

            n_freq = fit_frequency.size
            plateau_start_idx = int((1.0 - plateau_fraction) * n_freq)
            plateau_start_idx = max(0, min(plateau_start_idx, n_freq - 1))

            high_frequency_asd = fit_asd[plateau_start_idx:]
            initial_white_noise_asd = float(np.nanmedian(high_frequency_asd))

            if not np.isfinite(initial_white_noise_asd) or initial_white_noise_asd <= 0.0:
                initial_white_noise_asd = float(np.nanmedian(fit_asd))

            if not np.isfinite(initial_white_noise_asd) or initial_white_noise_asd <= 0.0:
                n_failed_fits += 1
                continue

            initial_knee_frequency_hz = float(np.nanmedian(fit_frequency))
            initial_alpha = 1.0
            initial_cutoff_frequency_hz = float(fit_frequency[-1])

            lower_bounds = [
                0.0,
                fit_frequency[0],
                0.0,
                fit_frequency[0],
            ]

            upper_bounds = [
                np.inf,
                fit_frequency[-1],
                10.0,
                np.inf,
            ]

            try:
                best_fit_parameters, _ = curve_fit(
                    cls._asd_model,
                    fit_frequency,
                    fit_asd,
                    p0=[
                        initial_white_noise_asd,
                        initial_knee_frequency_hz,
                        initial_alpha,
                        initial_cutoff_frequency_hz,
                    ],
                    bounds=(lower_bounds, upper_bounds),
                    maxfev=20000,
                )

            except (RuntimeError, ValueError):
                n_failed_fits += 1
                continue

            white_noise_asd = float(best_fit_parameters[0])
            knee_frequency_hz = float(best_fit_parameters[1])
            alpha = float(best_fit_parameters[2])
            cutoff_frequency_hz = float(best_fit_parameters[3])

            if not (
                    np.isfinite(white_noise_asd)
                    and np.isfinite(knee_frequency_hz)
                    and np.isfinite(alpha)
                    and np.isfinite(cutoff_frequency_hz)
            ):
                n_failed_fits += 1
                continue

            fit_white_noise_asds.append(white_noise_asd)
            fit_knee_frequencies_hz.append(knee_frequency_hz)
            fit_alphas.append(alpha)
            fit_cutoff_frequencies_hz.append(cutoff_frequency_hz)

        fit_knee_frequencies_hz = np.asarray(
            fit_knee_frequencies_hz,
            dtype=np.float64,
        )

        fit_white_noise_asds = np.asarray(
            fit_white_noise_asds,
            dtype=np.float64,
        )

        fit_alphas = np.asarray(
            fit_alphas,
            dtype=np.float64,
        )

        fit_cutoff_frequencies_hz = np.asarray(
            fit_cutoff_frequencies_hz,
            dtype=np.float64,
        )

        if fit_knee_frequencies_hz.size == 0:
            raise ValueError("All skydip noise spectrum fits failed.")

        return cls(
            tau_eff=float(tau_eff),

            mean_knee_frequency_hz=float(np.nanmean(fit_knee_frequencies_hz)),
            std_knee_frequency_hz=float(np.nanstd(fit_knee_frequencies_hz)),

            asd_unit=("K/√Hz" if asd_unit == "K" else "ADU/√Hz"),

            mean_white_noise_asd=float(
                np.nanmean(fit_white_noise_asds)
            ),
            std_white_noise_asd=float(
                np.nanstd(fit_white_noise_asds)
            ),

            mean_alpha=float(np.nanmean(fit_alphas)),
            std_alpha=float(np.nanstd(fit_alphas)),

            mean_cutoff_frequency_hz=float(
                np.nanmean(fit_cutoff_frequencies_hz)
            ),
            std_cutoff_frequency_hz=float(
                np.nanstd(fit_cutoff_frequencies_hz)
            ),

            n_successful_fits=int(fit_knee_frequencies_hz.size),
            n_failed_fits=int(n_failed_fits),
        )





@dataclass(frozen=True)
class DatasetNoiseInFrequencyRange:
    tes_idx: int
    frequency_min_hz: float
    frequency_max_hz: float
    mean_asd_k_per_sqrt_hz: float
    std_asd_k_per_sqrt_hz: float
    n_spectra: int

    @classmethod
    def from_spectra_frequency_range(cls,
                                     noise_spectra: list[SkydipNoiseSpectrum],
                                     config: SkydipCalibrationConfig,
                                     tes_idx: int) -> "DatasetNoiseInFrequencyRange":
        """
        Compute the mean calibrated ASD in a frequency interval.

        The frequency interval is read from config.noise.frequency_range.
        The method averages ASD values inside the selected frequency band
        for each skydip spectrum, then averages the resulting values across
        all valid skydips.
        """

        if len(config.noise.frequency_range) != 2:
            raise ValueError(
                "config.noise.frequency_range must contain exactly two values: "
                "[fmin, fmax]."
            )

        fmin_hz, fmax_hz = (float(value) for value in config.noise.frequency_range)

        if fmin_hz >= fmax_hz:
            raise ValueError(
                "config.noise.frequency_range must satisfy fmin < fmax. "
                f"Got fmin={fmin_hz} Hz and fmax={fmax_hz} Hz."
            )

        mean_asd_per_spectrum = []

        for spectrum in noise_spectra:

            frequency_hz = np.asarray(spectrum.frequency_hz, dtype=np.float64)
            asd_k = spectrum.asd_k_per_sqrt_hz

            if asd_k is None:
                continue

            asd_k = np.asarray(asd_k, dtype=np.float64)

            valid = (
                np.isfinite(frequency_hz)
                & np.isfinite(asd_k)
                & (frequency_hz >= fmin_hz)
                & (frequency_hz <= fmax_hz)
            )

            if not np.any(valid):
                continue

            mean_asd_per_spectrum.append(
                float(np.nanmean(asd_k[valid]))
            )

        if not mean_asd_per_spectrum:
            raise ValueError(
                "No valid ASD values were found in the selected frequency range "
                f"[{fmin_hz}, {fmax_hz}] Hz for TES {tes_idx}."
            )

        mean_asd_per_spectrum = np.asarray(
            mean_asd_per_spectrum,
            dtype=np.float64,
        )

        return cls(
            tes_idx=tes_idx,
            frequency_min_hz=float(fmin_hz),
            frequency_max_hz=float(fmax_hz),
            mean_asd_k_per_sqrt_hz=float(np.nanmean(mean_asd_per_spectrum)),
            std_asd_k_per_sqrt_hz=float(np.nanstd(mean_asd_per_spectrum)),
            n_spectra=int(mean_asd_per_spectrum.size),
        )
