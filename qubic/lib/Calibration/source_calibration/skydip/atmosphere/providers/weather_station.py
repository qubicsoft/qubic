import csv
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from qubic.lib.Calibration.source_calibration.common.utils import (parse_ground_temperature_datetime,
                                                                   parse_utc_datetime)
KELVIN_OFFSET = 273.15

@dataclass
class WeatherStationTimeSeries:
    """
    Weather-station time series for one observing day.
    For now this class reads the ground-temperature column only.

    Expected file format:
    - one column with timestamp, e.g. "16:46:07 11-03-2026"
    - one column with ground temperature in Celsius
    """

    time_utc: np.ndarray
    temperature_C: np.ndarray
    source_path: Path

    @classmethod
    def from_csv(cls,
                 csv_path: Path,
                 datetime_column_index: int = 0,
                 temperature_column_index: int = 1,
                 delimiter: str = ",") -> "WeatherStationTimeSeries":

        times: list[datetime] = []
        temperatures_C: list[float] = []

        with open(csv_path, newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)

            for row in reader:
                if len(row) <= max(datetime_column_index, temperature_column_index):
                    continue

                time_text = row[datetime_column_index].strip()
                temperature_text = row[temperature_column_index].strip()

                try:
                    time_utc = parse_ground_temperature_datetime(time_text)
                    temperature_C = float(temperature_text)

                except ValueError:
                    continue

                if np.isfinite(temperature_C):
                    times.append(time_utc)
                    temperatures_C.append(temperature_C)

        if len(times) == 0:
            raise ValueError(f"No valid ground-temperature samples found in {csv_path}.")

        order = np.argsort(np.asarray([t.timestamp() for t in times], dtype=np.float64))

        return cls(time_utc=np.asarray(times, dtype=object)[order],
                   temperature_C=np.asarray(temperatures_C, dtype=np.float64)[order],
                   source_path=csv_path)

    def select_range(self,
                     start_time_utc: str | datetime,
                     stop_time_utc: str | datetime) -> tuple[np.ndarray, np.ndarray]:

        """
        Selezione temporale dello skydip
        """

        start = parse_utc_datetime(start_time_utc)
        stop = parse_utc_datetime(stop_time_utc)

        if stop < start:
            raise ValueError("stop_time_utc must be greater than or equal to start_time_utc.")

        mask = np.asarray([(start <= t <= stop) for t in self.time_utc], dtype=bool,)

        return self.time_utc[mask], self.temperature_C[mask]

    def mean_temperature_C_between(self,
                                   start_time_utc: str | datetime,
                                   stop_time_utc: str | datetime) -> float:

        """Temperatura ground media in corrsiponderenza dello skydip"""
        _, temperature_window = self.select_range(start_time_utc=start_time_utc,
                                                  stop_time_utc=stop_time_utc)

        if temperature_window.size == 0:
            raise ValueError(
                "No ground-temperature samples found between "
                f"{parse_utc_datetime(start_time_utc).isoformat()} and "
                f"{parse_utc_datetime(stop_time_utc).isoformat()}."
            )

        return float(np.nanmean(temperature_window))

    def mean_temperature_K_between(self,
                                   start_time_utc: str | datetime,
                                   stop_time_utc: str | datetime) -> float:

        return (self.mean_temperature_C_between(
                start_time_utc=start_time_utc,
                stop_time_utc=stop_time_utc) + KELVIN_OFFSET)