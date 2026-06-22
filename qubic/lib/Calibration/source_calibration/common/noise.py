import numpy as np
from scipy.signal import welch
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

def compute_all_skydip_noise_spectra(
    segments: list,
    config: SkydipCalibrationConfig,
    conversion_factor_adu_per_k: float | None = None,
) -> list[SkydipNoiseSpectrum]:

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

    knee_frequency_hz: float
    plateau_asd_k_per_sqrt_hz: float
    plateau_mad_k_per_sqrt_hz: float

    frequency_hz: np.ndarray
    median_asd_k_per_sqrt_hz: np.ndarray

    @classmethod
    def from_spectra(
            cls,
            noise_spectra: list[SkydipNoiseSpectrum],
            tau_eff: float,
            plateau_fraction: float = 0.25,
            tolerance: float = 0.10):

        calibrated_spectra = []

        for spectrum in noise_spectra:
            if spectrum.asd_k_per_sqrt_hz is None:
                raise ValueError("Expected calibrated spectra.")

            calibrated_spectra.append(spectrum)

        #  se gli skydip hanno lunghezze diverse, welch può restituire
        #  griglie di frequenza diverse. Quindi per combinarle devi
        #  scegliere una griglia comune
        reference_frequency = calibrated_spectra[0].frequency_hz

        # nterpoli tutte le ASD su quella griglia
        asd_matrix = []

        for spectrum in calibrated_spectra:
            asd_interp = np.interp(
                reference_frequency,
                spectrum.frequency_hz,
                spectrum.asd_k_per_sqrt_hz,
            )
            asd_matrix.append(asd_interp)

        # shape = (n_skydips, n_frequencies)
        asd_matrix = np.asarray(asd_matrix)

        #  ASD rappresentativa del dataset/TES
        median_asd = np.nanmedian(asd_matrix, axis=0)

        #  Prendi l’ultimo pezzo dello spettro
        n_freq = reference_frequency.size
        plateau_start_idx = int((1.0 - plateau_fraction) * n_freq)

        plateau_values = median_asd[plateau_start_idx:]
        plateau_asd = float(np.nanmedian(plateau_values))
        plateau_mad = float(np.nanmedian(np.abs(plateau_values - plateau_asd)))

        #  Trovare la frequenza di ginocchio
        lower = plateau_asd * (1.0 - tolerance)
        upper = plateau_asd * (1.0 + tolerance)
        inside_plateau = (median_asd >= lower) & (median_asd <= upper)

        knee_idx = plateau_start_idx

        for i in range(n_freq):
            fraction_inside = np.mean(inside_plateau[i:])
            if fraction_inside > 0.8:
                knee_idx = i
                break

        knee_frequency_hz = float(reference_frequency[knee_idx])

        return cls(
            tau_eff=float(tau_eff),
            knee_frequency_hz=knee_frequency_hz,
            plateau_asd_k_per_sqrt_hz=plateau_asd,
            plateau_mad_k_per_sqrt_hz=plateau_mad,
            frequency_hz=reference_frequency,
            median_asd_k_per_sqrt_hz=median_asd,
        )







# def compute_skydip_noise_spectrum(
#     segment: "SkydipCalibrationSegment",
#     conversion_factor_adu_per_k: float,
#     sampling_frequency_hz: float,
#     nperseg: int | None = None,
#     noverlap: int | None = None,
#     detrend: str = "constant",
#     window: str = "hann",
# ) -> dict:
#     """
#     Compute the Welch noise spectrum for one skydip segment.
#
#     Returns a dictionary containing:
#     - frequency_hz
#     - psd_adu2_per_hz
#     - asd_adu_per_sqrt_hz
#     - asd_k_per_sqrt_hz
#
#     Notes
#     -----
#     The Welch estimator returns a PSD in ADU^2/Hz.
#     The corresponding ASD is:
#
#         ASD_ADU = sqrt(PSD_ADU^2/Hz)
#
#     and the calibrated ASD is obtained by dividing by the gain in ADU/K:
#
#         ASD_K = ASD_ADU / (ADU/K)
#     """
#     signal = np.asarray(segment.signal, dtype=np.float64)
#
#     if signal.ndim != 1:
#         raise ValueError("segment.signal must be a 1D array.")
#     if signal.size < 2:
#         raise ValueError("segment.signal must contain at least two samples.")
#     if not np.isfinite(conversion_factor_adu_per_k) or conversion_factor_adu_per_k == 0:
#         raise ValueError("conversion_factor_adu_per_k must be finite and non-zero.")
#     if not np.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0:
#         raise ValueError("sampling_frequency_hz must be finite and positive.")
#
#     valid = np.isfinite(signal)
#     if np.count_nonzero(valid) < 2:
#         raise ValueError("segment.signal must contain at least two finite samples.")
#
#     signal = signal[valid]
#
#     if nperseg is None:
#         nperseg = min(1024, signal.size)
#     else:
#         nperseg = int(min(nperseg, signal.size))
#
#     if nperseg < 2:
#         raise ValueError("nperseg must be at least 2 after clipping to the signal length.")
#
#     if noverlap is None:
#         noverlap = nperseg // 2
#     else:
#         noverlap = int(noverlap)
#
#     frequency_hz, psd_adu2_per_hz = welch(
#         signal,
#         fs=sampling_frequency_hz,
#         window=window,
#         nperseg=nperseg,
#         noverlap=noverlap,
#         detrend=detrend,
#         scaling="density",
#     )
#
#     asd_adu_per_sqrt_hz = np.sqrt(psd_adu2_per_hz)
#     asd_k_per_sqrt_hz = asd_adu_per_sqrt_hz / conversion_factor_adu_per_k
#
#     return {
#         "skydip_id": int(segment.skydip_id),
#         "direction": str(segment.direction),
#         "azimuth_mean_deg": float(segment.azimuth_mean),
#         "start_idx": int(segment.start_idx),
#         "stop_idx": int(segment.stop_idx),
#         "conversion_factor_adu_per_k": float(conversion_factor_adu_per_k),
#         "frequency_hz": frequency_hz,
#         "psd_adu2_per_hz": psd_adu2_per_hz,
#         "asd_adu_per_sqrt_hz": asd_adu_per_sqrt_hz,
#         "asd_k_per_sqrt_hz": asd_k_per_sqrt_hz,
#     }
#
#
#
# def compute_all_skydip_noise_spectra(
#     tm: np.ndarray,
#     tod: np.ndarray,
#     elevation: np.ndarray,
#     skydip_intervals: SkydipIntervals,
#     tau_eff: float,
#     tb_eff_k: float,
#     conversion_factors_csv: str | Path,
#     min_elevation_deg: float = 1.0,
#     nperseg: int | None = None,
#     noverlap: int | None = None,
#     detrend: str = "constant",
#     window: str = "hann",
# ) -> list[dict]:
#     """
#     Compute Welch noise spectra for all detected skydips.
#
#     The gain for each skydip is read from the CSV file generated by
#     `save_skydip_conversion_factors(...)` and matched by skydip_id.
#     """
#     tm = np.asarray(tm, dtype=np.float64)
#     tod = np.asarray(tod, dtype=np.float64)
#     elevation = np.asarray(elevation, dtype=np.float64)
#
#
#     dt = np.diff(tm)
#     dt = dt[np.isfinite(dt) & (dt > 0)]
#     if dt.size == 0:
#         raise ValueError("Could not infer a valid sampling interval from tm.")
#
#     sampling_frequency_hz = 1.0 / float(np.median(dt))
#
#     from qubic.lib.Calibration.source_calibration.skydip.calibration import extract_skydip_tod_segments
#     segments = extract_skydip_tod_segments(
#         tm=tm,
#         tod=tod,
#         elevation=elevation,
#         skydip_intervals=skydip_intervals,
#         tau_eff=tau_eff,
#         tb_eff_k=tb_eff_k,
#         min_elevation_deg=min_elevation_deg,
#     )
#
#     conversion_factors_csv = Path(conversion_factors_csv)
#     with open(conversion_factors_csv, "r", newline="") as f:
#         reader = csv.DictReader(f)
#         rows = list(reader)
#
#     if not rows:
#         raise ValueError(f"Conversion-factor CSV {conversion_factors_csv} is empty.")
#
#     gain_by_skydip_id: dict[int, float] = {}
#     for row in rows:
#         value = row.get("conversion_factor_adu_per_k", "")
#         if value in ("", None):
#             continue
#         gain = float(value)
#         if np.isfinite(gain):
#             gain_by_skydip_id[int(row["skydip_id"])] = gain
#
#     results: list[dict] = []
#     for segment in segments:
#         if segment.skydip_id not in gain_by_skydip_id:
#             continue
#
#         result = compute_skydip_noise_spectrum(
#             segment=segment,
#             conversion_factor_adu_per_k=gain_by_skydip_id[segment.skydip_id],
#             sampling_frequency_hz=sampling_frequency_hz,
#             nperseg=nperseg,
#             noverlap=noverlap,
#             detrend=detrend,
#             window=window,
#         )
#         results.append(result)
#
#     return results
#
# def compute_skydip_raw_noise_spectrum(segment,
#                                       nperseg: int = 1024) -> dict:
#     """
#     Compute the raw Welch noise spectrum for a single skydip segment without
#     applying any ADU/K conversion factor.
#
#     Returns both PSD in ADU^2/Hz and ASD in ADU/sqrt(Hz).
#     """
#     time = np.asarray(segment.time, dtype=np.float64)
#     signal = np.asarray(segment.signal, dtype=np.float64)
#
#
#     dt = np.diff(time)
#     valid_dt = dt[np.isfinite(dt) & (dt > 0)]
#     if valid_dt.size == 0:
#         raise ValueError(f"Skydip {segment.skydip_id} does not contain a valid time sampling.")
#
#     fs = 1.0 / np.nanmedian(valid_dt)
#
#     valid = np.isfinite(signal)
#     if np.count_nonzero(valid) < 2:
#         raise ValueError(f"Skydip {segment.skydip_id} does not contain enough finite samples.")
#
#     signal_valid = signal[valid]
#     nperseg_eff = min(int(nperseg), signal_valid.size)
#     if nperseg_eff < 2:
#         raise ValueError(f"Skydip {segment.skydip_id} does not contain enough samples for Welch.")
#
#     frequency_hz, psd_adu2_per_hz = welch(
#         signal_valid,
#         fs=fs,
#         nperseg=nperseg_eff,
#         scaling="density",
#     )
#     asd_adu_per_sqrt_hz = np.sqrt(psd_adu2_per_hz)
#
#     return {
#         "skydip_id": int(segment.skydip_id),
#         "direction": str(segment.direction),
#         "azimuth_mean_deg": float(segment.azimuth_mean),
#         "start_idx": int(segment.start_idx),
#         "stop_idx": int(segment.stop_idx),
#         "frequency_hz": np.asarray(frequency_hz, dtype=np.float64),
#         "psd_adu2_per_hz": np.asarray(psd_adu2_per_hz, dtype=np.float64),
#         "asd_adu_per_sqrt_hz": np.asarray(asd_adu_per_sqrt_hz, dtype=np.float64),
#     }
#
#
# def compute_all_skydip_raw_noise_spectra(tm: np.ndarray,
#                                          tod: np.ndarray,
#                                          elevation: np.ndarray,
#                                          skydip_intervals,
#                                          min_elevation_deg: float = 1.0,
#                                          nperseg: int = 1024) -> list[dict]:
#     """
#     Compute the raw Welch noise spectra for all detected skydips without applying
#     any ADU/K conversion factor.
#     """
#     tm = np.asarray(tm, dtype=np.float64)
#     tod = np.asarray(tod, dtype=np.float64)
#     elevation = np.asarray(elevation, dtype=np.float64)
#
#     from qubic.lib.Calibration.source_calibration.skydip.calibration import extract_skydip_tod_segments
#     # We only need the segmentation here. T_atm is irrelevant for the raw PSD,
#     # so use dummy atmospheric values.
#     segments = extract_skydip_tod_segments(
#         tm=tm,
#         tod=tod,
#         elevation=elevation,
#         skydip_intervals=skydip_intervals,
#         tau_eff=1.0,
#         tb_eff_k=1.0,
#         min_elevation_deg=min_elevation_deg,
#     )
#
#     results: list[dict] = []
#     for segment in segments:
#         results.append(
#             compute_skydip_raw_noise_spectrum(
#                 segment=segment,
#                 nperseg=nperseg,
#             )
#         )
#
#     return results
