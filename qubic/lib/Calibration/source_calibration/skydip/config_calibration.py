import yaml
import dacite
from pathlib import Path
from typing import Literal
from dataclasses import dataclass

from astropy import units as u
from qubic.lib.Calibration.source_calibration.common.utils import _resolve_path

@dataclass(frozen=True)
class DatasetPathsConfig:
    """
    Dataclass per singola run
    """
    dataset: Path
    output: Path
    atmospheric_config: Path

@dataclass
class PathsConfig:
    """
    Paths block of the skydip calibration config.
    The YAML can contain paths relative to paths.root.
    During __post_init__, all run paths are resolved once and stored as
    absolute paths inside self.runs.
    """

    root: Path
    runs: list[DatasetPathsConfig]

    def __post_init__(self) -> None:
        """
        Resolve root and all run paths once when the config is created.
        """

        self.root = self.root.expanduser().resolve()

        self.runs = [DatasetPathsConfig(dataset=_resolve_path(run.dataset, self.root),
                                        output=_resolve_path(run.output, self.root),
                                        atmospheric_config=_resolve_path(run.atmospheric_config, self.root))
                     for run in self.runs]


@dataclass(frozen=True)
class TodProcessingConfig:
    centered: bool
    normalized: bool

@dataclass(frozen=True)
class CalibrationConfig:
    tes_indices: int | list[int] | Literal["all"]
    run_am_if_missing: bool


@dataclass(frozen=True)
class PreprocessingConfig:
    az_velocity_threshold: u.Quantity
    el_velocity_threshold: u.Quantity
    min_block_duration: u.Quantity
    el_smooth_window: int
    el_polyorder: int
    up_left_extension: u.Quantity
    down_right_extension: u.Quantity


@dataclass(frozen=True)
class PlotConfig:
    linewidth: float
    alpha: float
    fit_linewidth: float
    fit_alpha: float
    show: bool


@dataclass(frozen=True)
class NoiseConfig:
    nperseg: int
    window: str
    noverlap: str | None
    scaling: str
    detrend: str
    selected_frequency: u.Quantity
    enabled: bool


@dataclass(frozen=True)
class SkydipCalibrationConfig:
    paths: PathsConfig
    calibration: CalibrationConfig
    preprocessing: PreprocessingConfig
    tod_processing: TodProcessingConfig
    plots: PlotConfig
    noise: NoiseConfig



def load_skydip_calibration_config(config_path: Path) -> SkydipCalibrationConfig:

    config_path = config_path.expanduser().resolve()
    data = yaml.safe_load(config_path.read_text())

    type_hooks = {u.Quantity: u.Quantity,
                  Path: lambda p: Path(p) if isinstance(p, str) else p}

    config = dacite.from_dict(SkydipCalibrationConfig,
                              data,
                              config=dacite.Config(type_hooks=type_hooks))

    if not config.paths.runs:
        raise ValueError("paths.runs must contain at least one dataset configuration.")

    return config