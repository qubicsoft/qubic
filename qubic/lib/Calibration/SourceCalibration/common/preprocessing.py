import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter

from qubic.lib.Calibration.SourceCalibration.common.io import read_qubic_dataset
from qubic.lib.Calibration.SourceCalibration.common.plotting import plot_focal_plane_tods


@dataclass(frozen=True, slots=True)
class ScanIntervals:
    subscans: list[np.ndarray]
    sweeps: list[np.ndarray]


@dataclass(frozen=True, slots=True)
class PointingInterpolation:
    azimuth: np.ndarray
    elevation: np.ndarray
    top_peakset: np.ndarray
    bottom_peakset: np.ndarray


@dataclass(frozen=True, slots=True)
class ScanMask:
    mask_scan: np.ndarray


@dataclass(frozen=True, slots=True)
class ScanTurningPoints:
    az_smooth: np.ndarray
    y0: float
    cross_up: np.ndarray
    cross_down: np.ndarray
    bottoms_peakset: list[dict[str, int]]
    tops_peakset: list[dict[str, int]]


def get_az_peaks_idx(tm_tod: np.ndarray,
                     tm_hk: np.ndarray,
                     peaks: np.ndarray) -> np.ndarray:

    tm_tod = np.asarray(tm_tod, dtype=np.float64)
    tm_hk = np.asarray(tm_hk, dtype=np.float64)
    peaks = np.asarray(peaks, dtype=int)

    if tm_tod.ndim != 1 or tm_hk.ndim != 1:
        raise ValueError("tm_tod and tm_hk must be 1D arrays.")
    if peaks.ndim != 1:
        raise ValueError("peaks must be a 1D array.")
    if tm_tod.size < 2:
        raise ValueError("tm_tod must contain at least two samples.")
    if peaks.size == 0:
        return np.array([], dtype=int)

    thk_target = tm_hk[peaks]

    pos = np.searchsorted(tm_tod, thk_target, side="right")
    pos = np.clip(pos, 1, len(tm_tod) - 1)

    left = pos - 1
    right = pos

    idx_interp = np.where(
        np.abs(thk_target - tm_tod[left]) <= np.abs(thk_target - tm_tod[right]),
        left,
        right,
    )
    return idx_interp.astype(int)


def get_top_bottom_azimuth(cross_up: np.ndarray,
                           cross_down: np.ndarray,
                           azimuth: np.ndarray) -> tuple[list[dict[str, int]], list[dict[str, int]]]:

    cross_up = np.asarray(cross_up, dtype=int)
    cross_down = np.asarray(cross_down, dtype=int)
    azimuth = np.asarray(azimuth, dtype=np.float64)

    tops: list[dict[str, int]] = []
    for i in range(len(cross_up)):
        start = int(cross_up[i])
        stop = int(cross_down[i]) if i < len(cross_down) - 1 else len(azimuth)

        split = azimuth[start:stop]
        if split.size == 0:
            continue

        left = start + int(split.argmax())
        right = start + int(split.size - 1 - split[::-1].argmax())

        tops.append(
            {
                "start": start,
                "stop": stop,
                "left_max": left,
                "right_max": right,
            }
        )

    bottoms: list[dict[str, int]] = []
    if len(cross_up) > 0:
        initial_split = azimuth[: cross_up[0]]
        if initial_split.size > 0:
            bottoms.append(
                {
                    "start": 0,
                    "stop": int(cross_up[0]),
                    "left_min": 0,
                    "right_min": int(cross_up[0] - initial_split[::-1].argmin()),
                }
            )

    for i in range(len(cross_down)):
        start = int(cross_down[i])
        stop = int(cross_up[i + 1]) if i < len(cross_up) - 1 else len(azimuth)

        split = azimuth[start:stop]
        if split.size == 0:
            continue

        left = start + int(split.argmin())
        right = start + int(split.size - 1 - split[::-1].argmin())

        bottoms.append(
            {
                "start": start,
                "stop": stop,
                "left_min": left,
                "right_min": right,
            }
        )

    return tops, bottoms


def extract_scan_turning_points(azimuth: np.ndarray,
                                smooth_window: int = 51,
                                polyorder: int = 3) -> ScanTurningPoints:

    azimuth = np.asarray(azimuth, dtype=np.float64)
    if azimuth.ndim != 1:
        raise ValueError("azimuth must be a 1D array.")
    if azimuth.size < 5:
        raise ValueError("azimuth must contain at least 5 samples.")

    window = min(smooth_window, azimuth.size if azimuth.size % 2 == 1 else azimuth.size - 1)
    if window < 3:
        window = 3
    if window % 2 == 0:
        window -= 1
    if polyorder >= window:
        polyorder = window - 1

    az_smooth = savgol_filter(azimuth, window_length=window, polyorder=polyorder)
    y0 = float(az_smooth.mean())

    sign_above = az_smooth >= y0
    cross_up = np.where((sign_above[1:] == True) & (sign_above[:-1] == False))[0]
    cross_down = np.where((sign_above[1:] == False) & (sign_above[:-1] == True))[0]

    tops_peakset, bottoms_peakset = get_top_bottom_azimuth(cross_up, cross_down, azimuth)

    return ScanTurningPoints(
        az_smooth=az_smooth,
        y0=y0,
        cross_up=cross_up,
        cross_down=cross_down,
        bottoms_peakset=bottoms_peakset,
        tops_peakset=tops_peakset,
    )


def interpolate_pointing_to_tod_time(tm_hk: np.ndarray,
                                     tm_tod: np.ndarray,
                                     azimuth: np.ndarray,
                                     elevation: np.ndarray,
                                     top_peakset: np.ndarray,
                                     bottom_peakset: np.ndarray) -> PointingInterpolation:

    tm_hk = np.asarray(tm_hk, dtype=np.float64)
    tm_tod = np.asarray(tm_tod, dtype=np.float64)
    azimuth = np.asarray(azimuth, dtype=np.float64)
    elevation = np.asarray(elevation, dtype=np.float64)
    top_peakset = np.asarray(top_peakset, dtype=int)
    bottom_peakset = np.asarray(bottom_peakset, dtype=int)

    az_interp = np.interp(tm_tod, tm_hk, azimuth)
    el_interp = np.interp(tm_tod, tm_hk, elevation)

    top_arr_interp = get_az_peaks_idx(tm_tod, tm_hk, top_peakset)
    bottom_arr_interp = get_az_peaks_idx(tm_tod, tm_hk, bottom_peakset)

    top_arr_interp = top_arr_interp.reshape(-1, 2) if top_arr_interp.size != 0 else np.empty((0, 2), dtype=int)
    bottom_arr_interp = (
        bottom_arr_interp.reshape(-1, 2) if bottom_arr_interp.size != 0 else np.empty((0, 2), dtype=int)
    )

    return PointingInterpolation(
        azimuth=az_interp,
        elevation=el_interp,
        top_peakset=top_arr_interp,
        bottom_peakset=bottom_arr_interp,
    )


def build_scan_mask(top_arr_interp: np.ndarray,
                    bottom_arr_interp: np.ndarray,
                    n_samples: int) -> ScanMask:

    top_arr_interp = np.asarray(top_arr_interp, dtype=int)
    bottom_arr_interp = np.asarray(bottom_arr_interp, dtype=int)

    mask_scan = np.ones(n_samples, dtype=bool)

    intervals = []
    if top_arr_interp.size != 0:
        intervals.append(top_arr_interp)
    if bottom_arr_interp.size != 0:
        intervals.append(bottom_arr_interp)

    if intervals:
        for row in np.vstack(intervals):
            start = max(int(row[0]), 0)
            stop = min(int(row[1]) - 1, n_samples)
            if stop > start:
                mask_scan[start:stop] = False

    return ScanMask(mask_scan=mask_scan)


def split_scan(mask_scan: np.ndarray) -> ScanIntervals:

    mask_scan = np.asarray(mask_scan, dtype=bool)
    idx_scan = np.where(mask_scan)[0]

    if idx_scan.size == 0:
        return ScanIntervals(subscans=[], sweeps=[])

    gaps = np.where(np.diff(idx_scan) > 1)[0]
    subscans = [arr.astype(int) for arr in np.split(idx_scan, gaps + 1)]

    sweeps = [
        np.concatenate((subscans[i], subscans[i + 1])).astype(int)
        for i in range(0, len(subscans) - 1, 2)
    ]

    return ScanIntervals(subscans=subscans, sweeps=sweeps)


def polynomial_trend(time_: Optional[np.ndarray],
                     tods: np.ndarray,
                     deg: int) -> np.ndarray:

    tods = np.asarray(tods, dtype=np.float64)
    n_samples = tods.shape[1 if tods.ndim == 2 else 0]
    time_ = time_ if time_ is not None else np.arange(n_samples, dtype=float)
    time_ = np.asarray(time_, dtype=np.float64)

    t0 = time_.mean()
    dt = time_ - t0
    std = dt.std() or 1.0
    x = dt / std

    V = np.vander(x, N=deg + 1, increasing=True)
    C, *_ = np.linalg.lstsq(V, tods.T, rcond=None)
    trend = (V @ C).T

    return trend


def linear_detrend(time_: Optional[np.ndarray],
                   tods: np.ndarray) -> np.ndarray:

    tods = np.asarray(tods, dtype=np.float64)
    return tods - polynomial_trend(time_, tods, deg=1)


def polynomial_detrend(time_: Optional[np.ndarray],
                       tods: np.ndarray,
                       deg: int = 2) -> np.ndarray:

    tods = np.asarray(tods, dtype=np.float64)
    return tods - polynomial_trend(time_, tods, deg=deg)


def remove_dc_offset(tods: np.ndarray) -> np.ndarray:

    tods = np.asarray(tods, dtype=np.float64)
    return tods - np.median(tods, axis=1, keepdims=True)


def normalize_tods(tods: np.ndarray) -> np.ndarray:

    tods = np.asarray(tods, dtype=np.float64)
    scale = np.max(np.abs(tods), axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    return tods / scale


def smooth_tods(tods: np.ndarray,
                window_length: int = 21,
                polyorder: int = 3) -> np.ndarray:

    tods = np.asarray(tods, dtype=np.float64)
    if tods.ndim != 2:
        raise ValueError("tods must be a 2D array.")

    n_samples = tods.shape[1]
    window = min(window_length, n_samples if n_samples % 2 == 1 else n_samples - 1)
    if window < 3:
        return tods.copy()
    if window % 2 == 0:
        window -= 1
    if polyorder >= window:
        polyorder = window - 1

    return savgol_filter(tods, window_length=window, polyorder=polyorder, axis=1)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d/%m/%Y | %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    dataset_path = Path("/Volumes/Data/PycharmProjects/calibration/qubic/scripts/Calibration/skydip_calibration/data/input/2026-03-11/2026-03-11_16.48.15__SkyDip")
    output_dir = Path("/Volumes/Data/PycharmProjects/calibration/qubic/scripts/Calibration/skydip_calibration/data/output") / dataset_path.name / "plots"

    dataset = read_qubic_dataset(
        dataset_path=dataset_path,
        logger=logger,
        log_label="PREPROCESSING_TEST",
    )

    if dataset.elevation is None:
        raise ValueError("Elevation is required to test preprocessing on skydips.")

    if dataset.interp_elevation is None:
        raise ValueError("Interpolated elevation is required to test preprocessing on skydips.")

    filtered_tods = remove_dc_offset(dataset.signals)
    linear_detrended = linear_detrend(dataset.time, filtered_tods)
    polynomial_detrended = polynomial_detrend(dataset.time, filtered_tods, deg=2)
    centered_tods = remove_dc_offset(dataset.signals)
    normalized_tods = normalize_tods(centered_tods)
    smoothed_tods = smooth_tods(centered_tods)

    z_rad = np.pi / 2.0 - np.deg2rad(dataset.interp_elevation)
    airmass = 1.0 / np.cos(z_rad)

    logger.info("Skydip TOD shape: %s", dataset.signals.shape)
    logger.info("Time shape: %s", dataset.time.shape)
    logger.info("Elevation shape: %s", dataset.elevation.shape)
    logger.info("Interpolated elevation shape: %s", dataset.interp_elevation.shape)
    logger.info("Airmass shape: %s", airmass.shape)
    logger.info("Airmass range: %.4f -> %.4f", np.min(airmass), np.max(airmass))
    logger.info("DC-offset removed TOD shape: %s", filtered_tods.shape)
    logger.info("Linear detrended TOD shape: %s", linear_detrended.shape)
    logger.info("Polynomial detrended TOD shape: %s", polynomial_detrended.shape)
    logger.info("Centered TOD shape: %s", centered_tods.shape)
    logger.info("Normalized TOD shape: %s", normalized_tods.shape)
    logger.info("Smoothed TOD shape: %s", smoothed_tods.shape)

    plot_focal_plane_tods(
        tods=smoothed_tods,
        output_path=output_dir / "skydip_tods_centered.png",
        title="Skydip TODs on focal plane after DC-offset removal"
    )