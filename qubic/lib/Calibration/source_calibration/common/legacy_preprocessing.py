import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter

from qubic.lib.Calibration.source_calibration.common.io import read_qubic_dataset




@dataclass(frozen=True, slots=True)
class ScanIntervals:
    """
    Represents a structure to store scan intervals.

    This class is designed to hold information about subscan and sweep intervals.
    It ensures that the stored data is immutable and that
    memory is efficiently managed through the use of frozen dataclass and slot
    mechanisms.

    Attributes:
        subscans:
            A list of numpy.ndarray objects representing the intervals for
            subscans in the scanning process.
        sweeps:
            A list of numpy.ndarray objects representing the intervals for
            sweeps in the scanning process.
    """
    subscans: list[np.ndarray]
    sweeps: list[np.ndarray]


@dataclass(frozen=True, slots=True)
class AzimuthSegments:
    """
    Stores azimuth-domain segments separated into fixed-azimuth plateaus and
    scan ramps.

    Attributes:
        flat_segments:
            List of index arrays identifying plateau intervals around azimuth
            turning points.
        rising_ramps:
            List of index arrays identifying subscans with increasing azimuth.
        falling_ramps:
            List of index arrays identifying subscans with decreasing azimuth.
    """
    flat_segments: list[np.ndarray]
    rising_ramps: list[np.ndarray]
    falling_ramps: list[np.ndarray]


@dataclass(frozen=True, slots=True)
class PointingInterpolation:
    """
    Represents a data structure for handling pointing interpolation data.

    This class is designed to store and manage the data required for pointing
    interpolation. It provides a frozen and memory-efficient implementation using
    dataclasses with frozen=True and slots=True. The main purpose of this class is to
    store az and el pointing data and their associated extrema indices on
    the time axis for the Time-Ordered Data (TOD).

    Attributes:
        azimuth (np.ndarray): The array of az values reported on the time axis of
            the TOD.
        elevation (np.ndarray): The array of el values reported on the time axis
            of the TOD.
        top_peakset (np.ndarray): Indices of the maxima of the scan reported on the time
            axis of the TOD.
        bottom_peakset (np.ndarray): Indices of the minima reported on the time axis of
            the TOD.
    """

    azimuth: np.ndarray
    elevation: np.ndarray
    # indici dei massimi di scansione riportati sul time axis TOD
    top_peakset: np.ndarray
    # indici dei minimi riportati sul time axis TOD
    bottom_peakset: np.ndarray


@dataclass(frozen=True, slots=True)
class ScanMask:
    """
    Represents a scan mask used in various applications.

    This class encapsulates a mask represented as a numpy array, which is used
    for scanning-related processes. It is designed to be immutable (`frozen=True`)
    and lightweight (`slots=True`), ensuring robust and efficient usage.

    Attributes:
        mask_scan (np.ndarray): A numpy array representing the mask used for
        scanning.
    """
    mask_scan: np.ndarray


@dataclass(frozen=True, slots=True)
class ScanTurningPoints:
    """
    Represents the turning points of an az scan.

    This class encapsulates the details of the turning points within an az
    scan. It includes smoothed az data, crossing points, and information
    on peaks and valleys detected in the scan. The class is immutable and uses
    slots for memory efficiency.

    Attributes:
        az_smooth (np.ndarray): The smoothed az data from the scan.
        y0 (float): The baseline level in the scan used for reference.
        cross_up (np.ndarray): Indices of the points where the scan crosses
            the baseline from below.
        cross_down (np.ndarray): Indices of the points where the scan crosses
            the baseline from above.
        bottoms_peakset (list[dict[str, int]]): A list of dictionaries representing
            detected valleys/low points in the scan. Each dictionary contains
            details like the index of the low point and associated metadata.
        tops_peakset (list[dict[str, int]]): A list of dictionaries representing
            detected peaks/high points in the scan. Each dictionary contains
            details like the index of the high point and associated metadata.
    """
    # informazioni legate ai turning points di una scansione in az

    az_smooth: np.ndarray
    el_smooth: np.ndarray
    y0: float
    cross_up: np.ndarray
    cross_down: np.ndarray
    bottoms_peakset: list[dict[str, int]]
    tops_peakset: list[dict[str, int]]


def get_az_peaks_idx(tm_tod: np.ndarray,
                     tm_hk: np.ndarray,
                     peaks: np.ndarray) -> np.ndarray:
    """
    Find the closest indices in a time-ordered dataset corresponding to specified peaks.

    This function determines the indices in the time-ordered data (`tm_tod`) that most closely
    align with peak positions identified in another time array (`tm_hk`). The dataset `peaks`
    provides the indices of these peaks within `tm_hk`. The function uses a search-and-compare
    approach to map these peak positions to `tm_tod` while ensuring the result is consistent
    with the provided arrays.

    Parameters:
    tm_tod : numpy.ndarray
        A 1D array containing time-ordered data samples.
    tm_hk : numpy.ndarray
        A 1D array containing time data where the peaks are located.
    peaks : numpy.ndarray
        A 1D array of integer indices indicating positions of peaks in `tm_hk`.

    Returns:
    numpy.ndarray
        A 1D array of integer indices that represent positions in `tm_tod` corresponding
        to the peaks in `tm_hk`.

    Raises:
    ValueError
        If `tm_tod` or `tm_hk` are not 1D arrays.
    ValueError
        If `peaks` is not a 1D array.
    ValueError
        If `tm_tod` contains fewer than two samples.
    """


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
    """
    Determines the top and bottom az ranges from provided crossing indices and az values.

    This function computes the top ranges (peaks) and bottom ranges (valleys) in an az data array,
    based on given crossing-up and crossing-down indices. It identifies the start and stop indices of each range
    and locates the lexftmost and rightmost maximum or minimum values within each range.

    Parameters:
        cross_up (np.ndarray): Array of indices representing the crossing-up points.
        cross_down (np.ndarray): Array of indices representing the crossing-down points.
        azimuth (np.ndarray): Array of az values.

    Returns:
        tuple[list[dict[str, int]], list[dict[str, int]]]: A tuple containing two lists:
            - The first list represents the tops (peaks) and contains dictionaries, each with keys:
                - start (int): Starting index of the range.
                - stop (int): Stopping index of the range.
                - left_max (int): Index of the leftmost maximum value within the range.
                - right_max (int): Index of the rightmost maximum value within the range.
            - The second list represents the bottoms (valleys) and contains dictionaries, each with keys:
                - start (int): Starting index of the range.
                - stop (int): Stopping index of the range.
                - left_min (int): Index of the leftmost minimum value within the range.
                - right_min (int): Index of the rightmost minimum value within the range.
    """


    tops: list[dict[str, int]] = []
    for i in range(len(cross_up)):
        start = int(cross_up[i])
        stop = int(cross_down[i]) if i < len(cross_down) else len(azimuth)
        mid = (stop - start) // 2
        print(f"TOP: start: {start}, stop: {stop}, mid: {mid}")

        split = azimuth[start:stop]
        if split.size == 0:
            continue

        left = start + split[:mid].argmax()
        right = start + split.size - 1 - split[:mid-1:-1].argmax()

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
        # gestisce i primi due bottom che non verrebbero considerati
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
        mid = (stop - start) // 2
        print(f"BOTTOM: start: {start}, stop: {stop}, mid: {mid}")

        split = azimuth[start:stop]
        if split.size == 0:
            continue

        # left = start + split[:mid].argmin()
        # right = start + split.size - 1 - split[:mid-1:-1].argmin()
        left = start + split.argmin()
        right = start + split.size - 1 - split[::-1].argmin()


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
                                elevation: np.ndarray,
                                smooth_window: int = 51,
                                polyorder: int = 3) -> ScanTurningPoints:
    """
    Extracts turning points from a scan's az data. The function identifies the key transition
    points in the az signal and smooths it using the Savitzky-Golay filter to improve the
    accuracy of the turning point detection.

    Parameters:
    az : np.ndarray
        A 1D array of az data points to process.
    smooth_window : int, optional
        The window length for the Savitzky-Golay filter. Must be a positive odd integer greater
        than or equal to 3. Defaults to 51.
    polyorder : int, optional
        The polynomial order for the Savitzky-Golay filter. Must be less than `smooth_window`.
        Defaults to 3.

    Returns:
    ScanTurningPoints
        An object representing details of the processed az signal, including the smoothed
        az array, reference level (mean), turning point indices for both upward and downward
        crossings, and peak sets for top and bottom positions.

    Raises:
    ValueError
        If the `az` array is not a 1D array, contains fewer than 5 samples, or invalid
        configurations for `smooth_window` and `polyorder` are provided.
    """


    window = min(smooth_window, azimuth.size if azimuth.size % 2 == 1 else azimuth.size - 1)
    if window < 3:
        window = 3
    if window % 2 == 0:
        window -= 1
    if polyorder >= window:
        polyorder = window - 1

    az_smooth = azimuth #savgol_filter(azimuth, window_length=window, polyorder=polyorder)
    y0 = savgol_filter(azimuth, window_length=window, polyorder=polyorder).mean()

    sign_above = az_smooth >= y0
    cross_up = np.where((sign_above[1:] == True) & (sign_above[:-1] == False))[0]
    cross_down = np.where((sign_above[1:] == False) & (sign_above[:-1] == True))[0]

    tops_peakset, bottoms_peakset = get_top_bottom_azimuth(cross_up, cross_down, az_smooth)

    return ScanTurningPoints(
        az_smooth=az_smooth,
        el_smooth=elevation,
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


# --- Inserted functions for azimuth segment extraction ---

def intervals_to_index_segments(intervals: np.ndarray,
                                n_samples: int) -> list[np.ndarray]:
    """
    Convert a (N, 2) array of [start, stop] intervals into explicit index arrays.

    The `stop` value is treated as exclusive.
    """
    intervals = np.asarray(intervals, dtype=int)

    if intervals.size == 0:
        return []
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError("intervals must have shape (N, 2).")

    segments: list[np.ndarray] = []
    for start, stop in intervals:
        start = max(int(start), 0)
        stop = min(int(stop), n_samples)
        if stop > start:
            segments.append(np.arange(start, stop, dtype=int))

    return segments


def classify_subscans_by_az_direction(subscans: list[np.ndarray],
                                      azimuth: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Split subscans into rising and falling azimuth ramps.
    """
    azimuth = np.asarray(azimuth, dtype=np.float64)

    rising_ramps: list[np.ndarray] = []
    falling_ramps: list[np.ndarray] = []

    for subscan in subscans:
        subscan = np.asarray(subscan, dtype=int)
        if subscan.size < 2:
            continue

        delta_az = float(azimuth[subscan[-1]] - azimuth[subscan[0]])
        if delta_az > 0:
            rising_ramps.append(subscan)
        elif delta_az < 0:
            falling_ramps.append(subscan)

    return rising_ramps, falling_ramps


def extract_azimuth_segments(pointing: PointingInterpolation,
                             scan_intervals: ScanIntervals) -> AzimuthSegments:
    """
    Build flat and ramp segments from interpolated pointing and subscan intervals.
    """
    n_samples = len(pointing.azimuth)

    flat_segments = intervals_to_index_segments(pointing.top_peakset, n_samples)
    flat_segments.extend(intervals_to_index_segments(pointing.bottom_peakset, n_samples))
    flat_segments.sort(key=lambda segment: int(segment[0]) if segment.size > 0 else -1)

    rising_ramps, falling_ramps = classify_subscans_by_az_direction(
        scan_intervals.subscans,
        pointing.azimuth,
    )

    return AzimuthSegments(
        flat_segments=flat_segments,
        rising_ramps=rising_ramps,
        falling_ramps=falling_ramps,
    )


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

    dataset_path = Path("/qubic/scripts/Calibration/skydip/data/dataset/2026-03-11/2026-03-11_16.48.15__SkyDip")
    output_dir = Path("/qubic/scripts/Calibration/skydip/data/output") / dataset_path.name / "plots"

    dataset = read_qubic_dataset(
        dataset_path=dataset_path,
        logger=logger,
        log_label="PREPROCESSING_TEST",
    )

    if dataset.elevation is None:
        raise ValueError("Elevation is required to test preprocessing on skydips.")

    if dataset.interp_elevation is None:
        raise ValueError("Interpolated el is required to test preprocessing on skydips.")

    filtered_tods = remove_dc_offset(dataset.signals)
    linear_detrended = linear_detrend(dataset.time, filtered_tods)
    polynomial_detrended = polynomial_detrend(dataset.time, filtered_tods, deg=2)
    centered_tods = remove_dc_offset(dataset.signals)
    normalized_tods = normalize_tods(centered_tods)
    smoothed_tods = smooth_tods(centered_tods)

    z_rad = np.pi / 2.0 - np.deg2rad(dataset.interp_elevation)
    airmass = 1.0 / np.cos(z_rad)

    logger.info("skydip TOD shape: %s", dataset.signals.shape)
    logger.info("Time shape: %s", dataset.time.shape)
    logger.info("Elevation shape: %s", dataset.elevation.shape)
    logger.info("Interpolated el shape: %s", dataset.interp_elevation.shape)
    logger.info("Airmass shape: %s", airmass.shape)
    logger.info("Airmass range: %.4f -> %.4f", np.min(airmass), np.max(airmass))
    logger.info("DC-offset removed TOD shape: %s", filtered_tods.shape)
    logger.info("Linear detrended TOD shape: %s", linear_detrended.shape)
    logger.info("Polynomial detrended TOD shape: %s", polynomial_detrended.shape)
    logger.info("Centered TOD shape: %s", centered_tods.shape)
    logger.info("Normalized TOD shape: %s", normalized_tods.shape)
    logger.info("Smoothed TOD shape: %s", smoothed_tods.shape)

    from qubic.lib.Calibration.source_calibration.common.plotting import plot_focal_plane_tods
    plot_focal_plane_tods(
        tods=smoothed_tods,
        output_path=output_dir / "skydip_tods_centered.pdf",
        title="skydip TODs on focal plane after DC-offset removal"
    )