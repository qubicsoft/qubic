from pathlib import Path
from typing import Optional
from dataclasses import dataclass

def _resolve_path(path: str) -> Path:
    """
    Resolve a given relative path to an absolute path within the project structure.

    The function resolves the provided relative path by appending it to a base
    location tied to the parent's directory of the script's root. If the resolved
    path points to a file, its parent directory is created if it does not exist.
    If the resolved path points to a directory, the directory itself is created
    if it does not exist.

    Parameters
    ----------
    path : str
        A relative file or directory path to resolve and ensure its existence.

    Returns
    -------
    Path
        An absolute path object corresponding to the resolved location.
    """

    # __file__ is the absolute path to the current file
    p = Path(__file__).expanduser().resolve().parents[3] / path

    if p.is_file():
        p.parent.mkdir(parents=True, exist_ok=True)

    if p.is_dir():
        p.mkdir(parents=True, exist_ok=True)

    return p


type_hooks = {
    Path: _resolve_path
}

# the parameter frozen=True avoids to modify the attributes of the dataclass
# after creating the object.
# slots=True avoids adding attributes during runtime.
# once the configuration file is read, the dataclass is frozen and cannot be modified anymore
@dataclass(frozen=True, slots=True)
class ConstraintConfig:
    """
    Configuration for defining constraints.

    This class is used to specify constraints for various entities
    with a defined type and optional minimum and maximum bounds.

    Attributes
    ----------
    type : str
        The type of the constraint, specifying the category or
        characteristic of the constraint.
    min : Optional[float], optional
        The minimum value for the constraint, default is None.
    max : Optional[float], optional
        The maximum value for the constraint, default is None.
    """
    type: str
    min: Optional[float] = None
    max: Optional[float] = None


@dataclass(frozen=True, slots=True)
class ObservatoryConfig:
    """
    Configuration detailing an astronomical observatory's location and environmental parameters.

    This class is used to hold metadata about an observatory, including its geographic location,
    environmental conditions, and general descriptive information. The data captured in this
    class can be utilized for observational calculations or serving as part of a larger
    astronomical survey toolset.

    Attributes
    ----------
    name : str
        The name of the observatory.
    lon : float
        Longitude of the observatory in degrees.
    lat : float
        Latitude of the observatory in degrees.
    height : float
        Elevation of the observatory in meters above sea level.
    pressure : float
        Air pressure at the observatory in hPa (hectopascals).
    relative_humidity : float
        The relative humidity at the observatory as a percentage (0-100%).
    timezone : str
        Timezone of the observatory in IANA format.
    description : str
        A textual description of the observatory.
    """
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
    """
    Configuration class for managing source file paths.

    This class is designed to handle the configuration of file paths related to
    regions and point sources. It encapsulates file path information in a
    structured and organized way, which can be used for further processing
    or analysis.

    Attributes
    ----------
    regions_file : str
        Path to the file containing region-specific data.
    point_sources_file : str
        Path to the file containing point source-specific data.
    """
    regions_file: Path
    point_sources_file: Path


@dataclass(frozen=True, slots=True)
class RunConfig:
    """
    Configuration class for running simulations and analyses.

    This class is used to define all the required parameters and attributes
    for running certain simulations or analyses involving geometrical and
    temporal configurations. It encapsulates information such as the year
    and month of the simulation, spatial parameters like radius and sample
    density, and temporal resolution for the trajectory and visibility
    calculations. Additionally, it holds the settings for heatmap resolutions
    and visibility periods.

    Attributes
    ----------
    year : int
        The year for which the simulation or analysis is conducted.
    month : int
        The month for which the simulation or analysis is conducted.
    radius : float
        The radius defining the spatial boundary for the simulation.
    radial_samples : int
        The number of samples to be taken along the radial dimension.
    polar_angle_samples : int
        The number of samples to be taken along the polar angle dimension.
    trajectory_time_resolution : str
        The temporal resolution for trajectory calculations, specified as
        a string in a time duration format (e.g., "1H" for one hour).
    monthly_visibility_start : str
        The start date or time of the monthly visibility window, as a string.
    monthly_visibility_end : str
        The end date or time of the monthly visibility window, as a string.
    monthly_heatmap_resolution : str
        The resolution of the generated monthly heatmap, specified as a string
        format.
    """
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
    """
    Configuration for task management in an astronomical analysis system.

    This class defines configurations used to manage and specify various tasks
    related to astronomical data processing and visualization. Tasks included
    address loading different source types, processing lunar data, applying
    constraints, generating plots and heatmaps, and other related tasks.

    Attributes
    ----------
    load_point_sources : bool
        Indicates if point sources should be loaded.
    load_extended_sources : bool
        Indicates if extended sources should be loaded.
    process_moon : bool
        Specifies whether to process data related to the Moon.
    run_constraints : bool
        Determines if constraints should be executed.
    run_trajectory_plots : bool
        Indicates if trajectory plots should be generated.
    write_trajectories : bool
        Determines if trajectories should be written to output.
    run_monthly_visibility : bool
        Specifies whether to calculate monthly visibility.
    run_monthly_heatmaps : bool
        Indicates if monthly heatmaps should be generated.
    plot_sidereal : bool
        Determines whether sidereal plots should be created.
    """
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
    """
    Represents the general configuration for an application or system.

    Provides foundational configurations that dictate the behavior and location
    settings of the application, such as the output directory and time resolution.
    This class serves as a centralized container for general settings.

    Attributes
    ----------
    output_dir : str
        The output directory for the application's file storage or operations.
    time_resolution : str
        The time resolution format setting, which determines the granularity
        of time-related operations or data.
    """
    output_dir: Path
    time_resolution: str
    mpl_style: Path


@dataclass(frozen=True, slots=True)
class ObservationConfig:
    """
    Configuration class for observation setup.

    This class serves as a unified configuration container for defining observation
    parameters. It encapsulates various sub-configurations related to general
    settings, the observatory setup, constraints, sources, runtime configuration,
    and task details. This allows for structured and modular handling of
    observation data.

    Attributes
    ----------
    general : GeneralConfig
        General configuration settings for the observation.
    observatory : ObservatoryConfig
        Configuration details specific to the observatory.
    constraints : dict[str, list[ConstraintConfig]]
        A mapping between constraint names and their corresponding configuration
        settings. Each constraint name is mapped to a list of `ConstraintConfig`
        objects.
    sources : SourcesConfig
        Configuration information about the sources to be observed.
    run : RunConfig
        Runtime configuration parameters for the observation process.
    tasks : TasksConfig
        Details about the tasks or operations to be performed during the
        observation.
    """
    general: GeneralConfig
    observatory: ObservatoryConfig
    constraints: dict[str, list[ConstraintConfig]]
    sources: SourcesConfig
    run: RunConfig
    tasks: TasksConfig
