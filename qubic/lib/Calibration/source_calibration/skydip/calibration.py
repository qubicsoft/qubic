import csv
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

from qubic.lib.Calibration.source_calibration.common.io import QubicDataset
from qubic.lib.Calibration.source_calibration.common.preprocessing import SkydipIntervals
from qubic.lib.Calibration.source_calibration.skydip.config_calibration import SkydipCalibrationConfig


def linear_fit(x: np.ndarray,
               y: np.ndarray) -> tuple[float, float]:

    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept)


@dataclass
class SkydipCalibrationSegment:
    skydip_id: int
    start_idx: int
    stop_idx: int
    direction: str
    azimuth_mean: float

    time: np.ndarray
    signal: np.ndarray
    elevation: np.ndarray
    airmass: np.ndarray
    t_atm: np.ndarray

    slope_adu_per_k: float | None = None
    intercept_adu: float | None = None

    def __post_init__(self):
        self.slope_adu_per_k, self.intercept_adu = linear_fit(x=self.t_atm,
                                                              y=self.signal)

    @classmethod
    def from_tod_segments(
            cls,
            tod_segments: list[dict],
            dataset: QubicDataset,
            tau_eff: float,
            tb_eff_k: float,
    ):
        """
        Build calibration segments from pre-extracted skydip TOD segments.
        """

        segments: list[SkydipCalibrationSegment] = []

        for tod_segment in tod_segments:
            skydip_id = int(tod_segment["skydip_id"])
            start_idx = int(tod_segment["start_idx"])
            stop_idx = int(tod_segment["stop_idx"])

            if start_idx < 0 or stop_idx >= dataset.time.size or stop_idx < start_idx:
                raise ValueError(
                    f"Invalid skydip interval for skydip_id={skydip_id}: "
                    f"start={start_idx}, stop={stop_idx}."
                )

            elevation_seg = dataset.interp_elevation[start_idx:stop_idx + 1]

            zenith_angle_rad = np.deg2rad(90.0 - elevation_seg)
            cos_zenith = np.cos(zenith_angle_rad)

            airmass_seg = 1.0 / cos_zenith
            t_atm_seg = (tb_eff_k / tau_eff) * (
                    1.0 - np.exp(-tau_eff / cos_zenith)
            )

            segments.append(
                cls(
                    skydip_id=skydip_id,
                    start_idx=start_idx,
                    stop_idx=stop_idx,
                    direction=str(tod_segment["direction"]),
                    azimuth_mean=float(tod_segment["azimuth_mean"]),
                    time=np.asarray(tod_segment["time"], dtype=np.float64),
                    signal=np.asarray(tod_segment["signal"], dtype=np.float64),
                    elevation=elevation_seg,
                    airmass=airmass_seg,
                    t_atm=t_atm_seg,
                )
            )

        return segments



@dataclass()
class DatasetConversionFactorSummary:
    """
    Summary of the skydip conversion factors for one dataset and one TES.

    The operational conversion factor used to calibrate the full TOD is
    median_adu_per_k.
    """
    segments: list[SkydipCalibrationSegment]

    slopes_adu_per_k: np.ndarray = field(init=False)
    intercepts_adu: np.ndarray = field(init=False)
    median_adu_per_k: float= field(init=False)
    mad_adu_per_k: float= field(init=False)

    def __post_init__(self):

        self.slopes_adu_per_k = np.asarray([
            segment.slope_adu_per_k for segment in self.segments
        ])

        self.intercepts_adu = np.asarray([segment.intercept_adu for segment in self.segments])

        self.median_adu_per_k = float(np.median(self.slopes_adu_per_k))
        self.mad_adu_per_k = float(np.median(np.abs(self.slopes_adu_per_k - self.median_adu_per_k)))

    @property
    def dataset_conversion_factor_adu_per_k(self) -> float:
        return self.median_adu_per_k

    def save_to_csv(self,
                    tes_idx: int,
                    dataset: QubicDataset):

        output_file = (dataset.calibration_plots_dir / f"tes_{tes_idx}_conversion_factors.csv")
        output_file.parent.mkdir(parents=True, exist_ok=True)


        fieldnames = ["skydip_id",
                      "direction",
                      "azimuth_mean_deg",
                      "start_idx",
                      "stop_idx",
                      "slope_adu_per_k",
                      "intercept_adu"]

        with output_file.open("w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for segment in self.segments:
                writer.writerow({
                    "skydip_id": segment.skydip_id,
                    "direction": segment.direction,
                    "azimuth_mean_deg": segment.azimuth_mean,
                    "start_idx": segment.start_idx,
                    "stop_idx": segment.stop_idx,
                    "slope_adu_per_k": segment.slope_adu_per_k,
                    "intercept_adu": segment.intercept_adu,
                })

        return output_file
