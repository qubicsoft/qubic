import sys
import yaml
import dacite
from pathlib import Path
from dataclasses import dataclass, field
from astropy import units as u
from astropy.units import Quantity

from qubic.lib.Calibration.source_calibration.common.utils import _resolve_path

@dataclass(frozen=True)
class AtmospherePathsConfig:

    root: Path
    input: Path

@dataclass(frozen=True)
class ApexConfig:

    pwv_csv: Path

@dataclass(frozen=True)
class WeatherStationConfig:

    weather_csv: Path
    datetime_column_index: int
    temperature_column_index: int
    delimiter: str

@dataclass(frozen=True)
class ParameterEstimationConfig:

    parameter_strategy: str = "band_average"

@dataclass(frozen=True)
class AMConfig:

    executable: Path
    cookbook_dir: Path
    season: str
    site: str
    h2o_percentile: int
    frequency_min: u.Quantity
    frequency_max: u.Quantity
    step: u.Quantity
    zenith_angle: u.Quantity

@dataclass
class AtmosphereConfig:

    paths: AtmospherePathsConfig
    apex: ApexConfig
    parameters_estimation: ParameterEstimationConfig
    weather_station: WeatherStationConfig
    am: AMConfig

    root: Path = field(init=False)

    weather_dir: Path = field(init=False)
    apex_pwv_csv: Path = field(init=False)
    weather_station_csv: Path = field(init=False)

    am_executable: Path = field(init=False)
    am_cookbook_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        """
        Resolve all atmosphere-related paths once when the config is created.

        Dataset and run output paths are not stored here because they are
        defined per run in the skydip calibration config.
        """

        self.root = self.paths.root.expanduser().resolve()

        self.input_dir = _resolve_path(self.paths.input, self.root)
        self.apex_pwv_csv = _resolve_path(self.apex.pwv_csv, self.root)
        self.weather_station_csv = _resolve_path(self.weather_station.weather_csv, self.root)

        self.am_executable = self.am.executable.expanduser().resolve()
        self.am_cookbook_dir = self.am.cookbook_dir.expanduser().resolve()



def load_atmosphere_config(path: str | Path) -> AtmosphereConfig:

    path = Path(path).expanduser().resolve()
    data = yaml.safe_load(path.read_text())

    type_hooks = {
        Path: lambda p: Path(p) if isinstance(p, str) else p,
        Quantity: u.Quantity,
    }

    config = dacite.from_dict(
        data_class=AtmosphereConfig,
        data=data,
        config=dacite.Config(type_hooks=type_hooks),
    )

    return config

if __name__ == "__main__":

    # Path(__file__).parents[5] : /Volumes/Data/PycharmProjects/calibration/qubic
    config_path = Path(__file__).parents[5] / "scripts" / "Calibration" / "source_calibration" / "skydip" / "configs" / sys.argv[1]

    if not config_path.exists() or config_path.suffix not in [".yaml", ".yml"]:
        raise ValueError(f"Config file `{config_path}` does not exist or is not a YAML file")

    config = load_atmosphere_config(config_path.expanduser().resolve())

    print("Atmosphere config")
    print(f"  weather_dir: {config.weather_dir}")
    print(f"  apex_pwv_csv: {config.apex_pwv_csv}")
    print(f"  weather_station_csv: {config.weather_station_csv}")
    print(f"  am_executable: {config.am.executable}")
    print(f"  cookbook_dir: {config.am.cookbook_dir}")