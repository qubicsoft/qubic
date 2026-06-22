import sys
from dataclasses import dataclass
# cached_property serve a non rileggere più volte dataset, APEX CSV, weather station CSV, ecc.
from functools import cached_property
from pathlib import Path

from qubic.lib.Calibration.source_calibration.common import io

from qubic.lib.Calibration.source_calibration.skydip.atmosphere.config_atmosphere import AtmosphereConfig, load_atmosphere_config
from qubic.lib.Calibration.source_calibration.skydip.atmosphere.providers.am_runner import AMRunConfig, write_am_spectrum_to_csv
from qubic.lib.Calibration.source_calibration.skydip.atmosphere.providers.am_template import AmTemplate, AmTemplateLibrary
from qubic.lib.Calibration.source_calibration.skydip.atmosphere.providers.apex_pwv import ApexPWVTimeSeries
from qubic.lib.Calibration.source_calibration.skydip.atmosphere.providers.weather_station import WeatherStationTimeSeries
from qubic.lib.Calibration.source_calibration.skydip.atmosphere.spectrum import AtmosphericSpectrum

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cached_property
from pathlib import Path

import numpy as np

from qubic.lib.Calibration.source_calibration.common.utils import parse_utc_datetime

@dataclass(frozen=True)
class AtmosphericDatasetMetadata:

    dataset_path: Path
    time: np.ndarray

    @property
    def dataset_name(self) -> str:
        return self.dataset_path.name

    @property
    def start_time_utc(self) -> datetime:
        return parse_utc_datetime(self.dataset_name)

    @property
    def duration_s(self) -> float:
        return float(np.nanmax(self.time) - np.nanmin(self.time))

    @property
    def stop_time_utc(self) -> datetime:
        return self.start_time_utc + timedelta(seconds=self.duration_s)

@dataclass(frozen=True)
class AtmosphericModel:
    """
    High-level atmospheric-model orchestrator for one skydip.
    The model connects:
    - the QUBIC dataset time interval;
    - APEX PWV;
    - weather-station ground temperature;
    - the AM atmospheric template;
    - the AM executable;
    - the final AtmosphericSpectrum.
    """
    config: AtmosphereConfig

    @classmethod
    def from_yaml(cls, config_path: Path) -> "AtmosphericModel":
        return cls(config=load_atmosphere_config(config_path))

    @cached_property
    def dataset(self) -> AtmosphericDatasetMetadata:
        saved_input_dir = self.config.output_dir.parent / "dataset"
        time_path = saved_input_dir / f"time_{self.config.dataset_path.name}.npy"

        if not time_path.exists():
            raise FileNotFoundError(
                f"Could not find saved TOD time array: {time_path}. "
                "Run io.py first to create the saved .npy/.npz dataset files, "
                "or check that the atmosphere output path points to the same dataset output directory."
            )

        time = np.load(time_path)

        return AtmosphericDatasetMetadata(
            dataset_path=self.config.dataset_path,
            time=time,
        )

    @property
    def skydip_start_time_utc(self):
        return self.dataset.start_time_utc

    @property
    def skydip_stop_time_utc(self):
        return self.dataset.stop_time_utc

    @property
    def skydip_duration_s(self) -> float:
        return self.dataset.duration_s

    @cached_property
    def template_library(self) -> AmTemplateLibrary:
        return AmTemplateLibrary.default()

    @cached_property
    def template(self) -> AmTemplate:
        return self.template_library.get(site=self.config.am.site,
                                         season=self.config.am.season,
                                         h2o_percentile=self.config.am.h2o_percentile)

    @cached_property
    def apex(self) -> ApexPWVTimeSeries:
        return ApexPWVTimeSeries.from_csv(self.config.apex_pwv_csv)

    @cached_property
    def pwv_apex_mm(self) -> float:
        return self.apex.median_pwv_between(start_time_utc=self.skydip_start_time_utc,
                                            stop_time_utc=self.skydip_stop_time_utc)

    @cached_property
    def weather_station(self) -> WeatherStationTimeSeries:
        return WeatherStationTimeSeries.from_csv(csv_path=self.config.weather_station_csv,
                                                 datetime_column_index=self.config.weather_station.datetime_column_index,
                                                 temperature_column_index=self.config.weather_station.temperature_column_index,
                                                 delimiter=self.config.weather_station.delimiter)

    @cached_property
    def tground_k(self) -> float:
        return self.weather_station.mean_temperature_K_between(start_time_utc=self.skydip_start_time_utc,
                                                               stop_time_utc=self.skydip_stop_time_utc)

    @cached_property
    def water_vapor_scale(self) -> float:
        return self.template.water_vapor_scale_from_pwv(self.pwv_apex_mm)

    @cached_property
    def am_config(self) -> AMRunConfig:
        return AMRunConfig(am_executable=self.config.am.executable,
                           cookbook_dir=self.config.am.cookbook_dir,
                           template=self.template,
                           f_min_ghz=self.config.am.freq_min_ghz,
                           f_max_ghz=self.config.am.freq_max_ghz,
                           step_mhz=self.config.am.step_mhz,
                           zenith_angle_deg=self.config.am.zenith_angle_deg,
                           tground_k=self.tground_k,
                           water_vapor_scale=self.water_vapor_scale)

    @property
    def default_output_csv(self) -> Path:
        return self.config.output_dir / f"{self.am_config.output_stem()}.csv"

    def run(self,
            output_csv: Path | None = None,
            cwd: Path | None = None) -> AtmosphericSpectrum:

        output_csv = self.default_output_csv if output_csv is None else output_csv

        generated_csv = write_am_spectrum_to_csv(config=self.am_config,
                                                 output_csv=output_csv,
                                                 cwd=cwd)

        return AtmosphericSpectrum.from_csv(generated_csv)

    def load_existing_spectrum(self, csv_path: Path | None = None) -> AtmosphericSpectrum:
        csv_path = self.default_output_csv if csv_path is None else csv_path

        return AtmosphericSpectrum.from_csv(csv_path)

    def summary(self) -> dict[str, float | str]:
        return {

            "dataset": str(self.config.dataset_path),
            "dataset_name": self.config.dataset_path.name,
            "skydip_start_time_utc": self.skydip_start_time_utc.isoformat(),
            "skydip_stop_time_utc": self.skydip_stop_time_utc.isoformat(),
            "skydip_duration_s": self.skydip_duration_s,
            "apex_pwv_csv": str(self.config.apex_pwv_csv),
            "weather_station_csv": str(self.config.weather_station_csv),
            "template": self.template.name,
            "template_path": str(self.am_config.amc_path),
            "template_pwv_scale1_mm": self.template.pwv_scale1_mm,
            "pwv_apex_mm": self.pwv_apex_mm,
            "water_vapor_scale": self.water_vapor_scale,
            "tground_k": self.tground_k,
            "frequency_min_GHz": self.config.am.freq_min_ghz,
            "frequency_max_GHz": self.config.am.freq_max_ghz,
            "step_MHz": self.config.am.step_mhz,
            "zenith_angle_deg": self.config.am.zenith_angle_deg,
            "output_csv": str(self.default_output_csv)}


if __name__ == "__main__":

    config_path = Path(sys.argv[1]).expanduser().resolve()
    model = AtmosphericModel.from_yaml(config_path)

    print("Atmospheric model summary")
    for key, value in model.summary().items():
        print(f"  {key}: {value}")

    print("\nAM command")
    print("  " + " ".join(model.am_config.command()))

    try:
        spectrum = model.run()

    except RuntimeError as exc:
        print("\nAM execution failed. Full diagnostic follows:\n")
        print(exc)
        raise

    print("\nGenerated atmospheric spectrum")
    for key, value in spectrum.summary().items():
        print(f"  {key}: {value}")