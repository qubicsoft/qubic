from pathlib import Path
from typing import Optional
from dataclasses import dataclass


# def _resolve_path(path: str) -> Path:
#     p = Path(__file__).expanduser().resolve().parents[3] / path
#
#     if p.is_file():
#         p.parent.mkdir(parents=True, exist_ok=True)
#
#     if p.is_dir():
#         p.mkdir(parents=True, exist_ok=True)
#
#     return p


def _resolve_path(path: str) -> Path:
    return Path(path).expanduser()

type_hooks = {Path: _resolve_path}


@dataclass(frozen=True, slots=True)
class ConstraintConfig:
    type: str
    min: Optional[float] = None
    max: Optional[float] = None


@dataclass(frozen=True, slots=True)
class ObservatoryConfig:
    name: str
    lon: float
    lat: float
    height: float
    pressure: float
    relative_humidity: float
    timezone: str
    description: str


@dataclass(frozen=True, slots=True)
class SourcesConfig:
    regions_file: Path
    point_sources_file: Path


@dataclass(frozen=True, slots=True)
class RunConfig:
    year: int
    month: int
    radius: float
    radial_samples: int
    polar_angle_samples: int
    trajectory_time_resolution: str
    monthly_visibility_start: str
    monthly_visibility_end: str
    monthly_heatmap_resolution: str


@dataclass(frozen=True, slots=True)
class TasksConfig:
    load_point_sources: bool
    load_extended_sources: bool
    process_moon: bool
    run_constraints: bool
    run_trajectory_plots: bool
    write_trajectories: bool
    run_monthly_visibility: bool
    run_monthly_heatmaps: bool
    plot_sidereal: bool


@dataclass(frozen=True, slots=True)
class GeneralConfig:
    output_dir: Path
    time_resolution: str
    mpl_style: Path


@dataclass(frozen=True, slots=True)
class ObservationConfig:
    general: GeneralConfig
    observatory: ObservatoryConfig
    constraints: dict[str, list[ConstraintConfig]]
    sources: SourcesConfig
    run: RunConfig
    tasks: TasksConfig
