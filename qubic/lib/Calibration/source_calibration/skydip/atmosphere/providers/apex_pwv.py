import csv
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from qubic.lib.Calibration.source_calibration.common.utils import parse_utc_datetime

APEX_PWV_TIME_COLUMN = "Date time"
APEX_PWV_COLUMN = "Precipitable Water Vapor [mm]"

@dataclass
class ApexPWVTimeSeries:
    """
    PWV time series downloaded from the APEX/ESO weather archive.
    """

    time_utc: np.ndarray
    pwv_mm: np.ndarray
    source_path: Path

    @classmethod
    def from_csv(cls,
                 csv_path: Path,
                 time_column: str = APEX_PWV_TIME_COLUMN,
                 pwv_column: str = APEX_PWV_COLUMN) -> "ApexPWVTimeSeries":
        """
        1. Legge il CSV APEX
        2. cerca le colonne Date Time e Precipitable Water Vapor [mm]
        3. converte le date in UTC;
        4. converte il PWV in float;
        5. ordina tutto temporalmente.
        """

        csv_path = csv_path.expanduser().resolve()

        times: list[datetime] = []
        pwv_values: list[float] = []

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames is None:
                raise ValueError(f"CSV file {csv_path} does not contain a valid header.")

            missing_columns = {time_column, pwv_column} - set(reader.fieldnames)
            if missing_columns:
                raise ValueError(
                    f"Missing required columns in {csv_path}: {sorted(missing_columns)}. "
                    f"Available columns are: {reader.fieldnames}"
                )

            for row in reader:
                try:
                    time_utc = parse_utc_datetime(row[time_column])
                    pwv_mm = float(row[pwv_column])
                except (ValueError, KeyError):
                    continue

                if np.isfinite(pwv_mm):
                    times.append(time_utc)
                    pwv_values.append(pwv_mm)

        if len(times) == 0:
            raise ValueError(f"No valid PWV samples found in {csv_path}.")

        order = np.argsort(np.asarray([t.timestamp() for t in times], dtype=np.float64))

        return cls(time_utc=np.asarray(times, dtype=object)[order],
                   pwv_mm=np.asarray(pwv_values, dtype=np.float64)[order],
                   source_path=csv_path)


    def select_range(self,
                     start_time_utc: str | datetime,
                     stop_time_utc: str | datetime) -> tuple[np.ndarray, np.ndarray]:
        """
        Return time and PWV samples in the inclusive interval
        [start_time_utc, stop_time_utc].
        prende solo i campioni APEX dentro lo skydip
        """
        start = parse_utc_datetime(start_time_utc)
        stop = parse_utc_datetime(stop_time_utc)

        if stop < start:
            raise ValueError("stop_time_utc must be greater than or equal to start_time_utc.")

        mask = np.asarray([(start <= t <= stop) for t in self.time_utc], dtype=bool)

        return self.time_utc[mask], self.pwv_mm[mask]


    def median_pwv_between(self,
                           start_time_utc: str | datetime,
                           stop_time_utc: str | datetime) -> float:
        """
        PWV estimate over the actual skydip time interval.
        """
        _, pwv_window = self.select_range(start_time_utc=start_time_utc,
                                          stop_time_utc=stop_time_utc)

        if pwv_window.size == 0:
            raise ValueError(
                "No APEX PWV samples found between "
                f"{parse_utc_datetime(start_time_utc).isoformat()} and "
                f"{parse_utc_datetime(stop_time_utc).isoformat()}."
            )

        return float(np.nanmedian(pwv_window))

if __name__ == "__main__":

    from qubic.lib.Calibration.source_calibration.common import io
    from qubic.lib.Calibration.source_calibration.skydip.atmosphere.config_atmosphere import load_atmosphere_config

    config_path = Path(sys.argv[1]).expanduser().resolve()
    config = load_atmosphere_config(config_path)

    dataset = io.read_qubic_dataset(dataset_path=config.dataset_path)

    skydip_start_time_utc = dataset.start_time_utc
    skydip_stop_time_utc = dataset.stop_time_utc
    skydip_duration_s = dataset.duration_s

    apex = ApexPWVTimeSeries.from_csv(config.apex_pwv_csv)

    times, pwv = apex.select_range(start_time_utc=skydip_start_time_utc,
                                   stop_time_utc=skydip_stop_time_utc)

    print("APEX PWV time series")
    print(f"  config_path: {config_path}")
    print(f"  source_path: {apex.source_path}")
    print(f"  n_samples: {apex.pwv_mm.size}")
    print(f"  first_time_utc: {apex.time_utc[0].isoformat()}")
    print(f"  last_time_utc: {apex.time_utc[-1].isoformat()}")
    print(f"  pwv_min_mm: {np.nanmin(apex.pwv_mm):.4f}")
    print(f"  pwv_max_mm: {np.nanmax(apex.pwv_mm):.4f}")
    print(f"  dataset_name: {config.dataset_path.name}")
    print(f"  dataset_duration_s: {skydip_duration_s:.3f}")
    print(f"  skydip_start_time_utc: {skydip_start_time_utc.isoformat()}")
    print(f"  skydip_stop_time_utc: {skydip_stop_time_utc.isoformat()}")
    print(f"  selected_samples_skydip: {pwv.size}")

    if pwv.size > 0:
        print(f"  selected_first_time_utc: {times[0].isoformat()}")
        print(f"  selected_last_time_utc: {times[-1].isoformat()}")
        print(f"  median_pwv_skydip_mm: {np.nanmedian(pwv):.4f}")