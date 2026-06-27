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

        if len(calibrated_spectra) == 0:
            raise ValueError("No calibrated noise spectra were provided.")

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

