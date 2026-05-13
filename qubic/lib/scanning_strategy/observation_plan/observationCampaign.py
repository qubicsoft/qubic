import sys
import logging
from pathlib import Path

import numpy as np
import toml
import tqdm
import dacite
from astropy.time import Time
import calendar as cal
from datetime import datetime
from datetime import timezone as dt_timezone
from typing import List
import astropy.units as u
from astroplan import (Observer,
                       Constraint,
                       AltitudeConstraint,
                       AirmassConstraint)

from matplotlib import pyplot as plt

from .constraint import SunSeparationConstraint, MoonSeparationConstraint
from astropy.coordinates import EarthLocation
from pytz import timezone

from .schema.types import type_hooks
from .schema.types import ObservationConfig
from .source import PointSource, ExtendedSource, load_sources_from_file, Source


logging.basicConfig(filemode="w",
                    encoding='utf8',
                    format="%(asctime)s - %(funcName)s - %(message)s",
                    datefmt="%d/%m/%Y | %H:%M:%S",
                    level=logging.INFO)

def plot_sidereal_overall(celestial_regions, outpath: Path, ang_offset: float = 15.0):
    """
    Plots the sidereal elevation tracks for specified celestial regions and saves the generated
    figure to the given output path.

    This function visualizes the sidereal elevation tracks of various celestial regions over the
    local sidereal time (LST). It highlights elevation window bands if defined for the regions,
    and generates labeled plots of visibility and elevation. The resulting plot is stored as a
    high-resolution image.

    :param celestial_regions: A dictionary where keys are region names (str) and values are objects
        containing sidereal tracking data such as local sidereal time (LST), elevation, visibility,
        minimum altitude, and maximum altitude.
    :type celestial_regions: dict
    :param outpath: A path to the directory where the output plot image should be saved.
    :type outpath: Path
    :param ang_offset: Optional angular offset used to define inner bounding bands inside the
        elevation window. Default is 15.0 degrees.
    :type ang_offset: float
    :return: A tuple containing the generated Matplotlib figure and axis objects.
    :rtype: tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]
    """
    fig, ax = plt.subplots(figsize=(12, 6), tight_layout=True)

    bands_drawn = False
    band_label = r"${} ^ \circ \leq \mathrm{{{}}} \leq {} ^ \circ$"

    for name, region in celestial_regions.items():
        lst, elev, alt_min, alt_max, vis = region.sidereal_track()

        if (not bands_drawn
            and (alt_min is not None)
            and (alt_max is not None)):
            ax.axhspan(
                alt_min, alt_max,
                color="lightgray",
                alpha=0.6,
                label=band_label.format(int(alt_min), r"Elevation \, Window", int(alt_max)))

            inner_min = alt_min + ang_offset
            inner_max = alt_max - ang_offset

            if inner_min < inner_max:
                ax.axhspan(
                    inner_min, inner_max,
                    color="silver",
                    alpha=0.6,
                    label=band_label.format(int(inner_min), r"Elevation \, Center", int(inner_max)))

            bands_drawn = True

        ax.plot(lst, np.where(vis, elev, np.nan), linewidth=2.0, label=name)

    first = next(iter(celestial_regions.values()))
    # date_str = first.obs_time.strftime("%Y-%m-%d")
    date_str = first.obs_time.to_datetime(timezone=first.qubic_site.timezone).strftime("%Y-%m-%d")
    title = rf"$\bf{{Sidereal \, plot}}$ : all sources on {date_str}".strip()
    ax.set_title(title, fontsize=11, pad=10)

    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 3))
    ax.set_ylim(0, 90)
    ax.set(xlabel="Local Sidereal Time (LST) [h]", ylabel=r"Elevation [$^\circ$]")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    plt.savefig(outpath / "all_extended_sidereal.png", dpi=600)

    return fig, ax


class ObservationCampaign:
    """
    Manages observation campaigns by handling the observatory configuration,
    constraints, celestial region observability, and daily observation planning.

    The class is designed to load and use the provided configuration for
    various operations such as analyzing extended source observability, applying
    constraints, and scheduling daily observation plans. Essential components
    like the observatory setup and constraints are constructed from the
    configuration, ensuring flexibility and modularity for different campaigns.

    :ivar config: Loaded campaign configuration that contains observatory details,
        constraints, tasks, and operational context.
    :type config: ObservationConfig
    :ivar site: An astronomical observer set up based on the configuration
        parameters.
    :type site: Observer
    :ivar point_constraints: Constraints applicable to point targets, constructed
        from the configuration.
    :type point_constraints: List[Constraint]
    :ivar extended_constraints: Constraints applicable to extended celestial
        regions, constructed from the configuration.
    :type extended_constraints: List[Constraint]
    """

    def __init__(self, config_path: Path):

        self.config_path = Path(config_path).expanduser().resolve()
        self.config = self.load_config(self.config_path)

        self.output_dir = self._resolve_config_path(self.config.general.output_dir)
        self.mpl_style_path = self._resolve_config_path(self.config.general.mpl_style)
        self.regions_file = self._resolve_config_path(self.config.sources.regions_file)
        self.point_sources_file = self._resolve_config_path(self.config.sources.point_sources_file)

        self.site = self.build_observatory()
        self.point_constraints = self.build_constraints('point')
        self.extended_constraints = self.build_constraints('extended')

        plt.style.use(self.mpl_style_path)

    @staticmethod
    def load_config(config_path: Path) -> ObservationConfig:
        """
        Loads the observation configuration from a TOML file.

        :param config_path: The path to the configuration file.
        :type config_path: str

        :raises FileNotFoundError: If the configuration file is not found at the
            specified `config_path`.

        :return: An instance of `ObservationConfig` populated with values read from
            the TOML file.
        :rtype: ObservationConfig
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file '{config_path}' not found")
        config_dict = toml.load(config_path)

        return dacite.from_dict(data_class=ObservationConfig,
                                data=config_dict,
                                config=dacite.Config(type_hooks=type_hooks))

    def _resolve_config_path(self, path_value: Path) -> Path:
        """
        Resolve a path from the TOML configuration relative to the directory that
        contains the configuration file.
        """
        path_obj = Path(path_value).expanduser()
        if not path_obj.is_absolute():
            path_obj = self.config_path.parent / path_obj
        return path_obj.resolve()

    def build_observatory(self) -> Observer:
        """
        Builds and returns an astronomical observer based on the provided configuration.
        The observer object is created using geodetic location information, pressure, relative
        humidity, timezone, and other observatory-specific details.

        :raises KeyError: If any expected keys are missing from the configuration.
        :raises AttributeError: If configuration objects are incorrectly defined or
            attributes are improperly accessed.

        :return: An Observer object with the specified configuration.
        :rtype: Observer
        """
        location = EarthLocation.from_geodetic(
            lon=self.config.observatory.lon * u.deg,
            lat=self.config.observatory.lat * u.deg,
            height=self.config.observatory.height * u.m)

        return Observer(
            name=self.config.observatory.name,
            location=location,
            pressure=self.config.observatory.pressure * u.bar,
            relative_humidity=self.config.observatory.relative_humidity,
            timezone=timezone(self.config.observatory.timezone),
            description=self.config.observatory.description)

    def build_constraints(self, constraint_type: str) -> List[Constraint]:
        """
        Builds a list of constraints based on the given constraint type and configuration.

        This method interprets the configuration data for constraints and instantiates
        the corresponding constraint objects. The constraint type is used to filter
        and process the appropriate constraints from the configuration.

        :param constraint_type: A string representing the type of constraints to build.
        :return: A list containing instantiated Constraint objects based on the provided
                 configuration and constraint type.
        :rtype: List[Constraint]
        :raises ValueError: If an unsupported constraint type is encountered in the
                            configuration data.
        """

        constraints = []
        for constraint_cfg in self.config.constraints[constraint_type]:

            if constraint_cfg.type == 'AltitudeConstraint':
                constraints.append(AltitudeConstraint(
                    min=constraint_cfg.min * u.deg if constraint_cfg.min else None,
                    max=constraint_cfg.max * u.deg if constraint_cfg.max else None
                ))
            elif constraint_cfg.type == 'AirmassConstraint':
                constraints.append(AirmassConstraint(max=constraint_cfg.max))
            elif constraint_cfg.type == 'SunSeparationConstraint':
                constraints.append(SunSeparationConstraint(min=constraint_cfg.min * u.deg))
            elif constraint_cfg.type == 'MoonSeparationConstraint':
                constraints.append(MoonSeparationConstraint(min=constraint_cfg.min * u.deg))
            else:
                raise ValueError(f"Unsupported constraint type '{constraint_cfg.type}'")
        return constraints

    def analyze_observability(self):
        """
        Analyzes the observability of celestial regions based on provided configurations.

        This method evaluates the configuration settings to determine if monthly visibility
        plots or monthly heatmaps need to be generated. It processes celestial regions,
        applies constraints, and computes observability metrics, including creating visual
        representations like visibility plots or heatmaps for specified time ranges.
        The celestial regions are extended sources loaded from a file, and their observability
        characteristics are computed using specific parameters such as time resolution, region
        radius, and sample counts.

        :param self: The instance of the class containing configuration and
            operational context.
        :return: None
        """
        if not (self.config.tasks.run_monthly_visibility or self.config.tasks.run_monthly_heatmaps):
            return

        celestial_regions = ExtendedSource.load_sources(
            self.site,
            Time(self.config.run.monthly_visibility_start),
            self.extended_constraints,
            u.Quantity(self.config.general.time_resolution),
            source_file=self.regions_file,
            results_dir=self.output_dir / "panoramic",
            radius=self.config.run.radius * u.deg,
            radial_samples=self.config.run.radial_samples,
            polar_angle_samples=self.config.run.polar_angle_samples)

        with tqdm.tqdm(total=len(celestial_regions),
                       iterable=celestial_regions.items(),
                       desc="Region Analysis",
                       colour="green",
                       dynamic_ncols=True,
                       file=sys.stdout) as pbar:

            for region_name, region in pbar:

                pbar.set_description(f"Region `{region_name}`")

                region.configure()

                if self.config.tasks.plot_sidereal:
                    region.plot_sidereal()

                if self.config.tasks.run_monthly_visibility:
                    region.plot_months_visible(
                        Time(self.config.run.monthly_visibility_start),
                        Time(self.config.run.monthly_visibility_end),
                        time_resolution=u.Quantity(self.config.run.monthly_heatmap_resolution),
                        plot_path=self.output_dir /
                                  "panoramic" / region_name / f"{region_name}_month_visibility.png")

                if self.config.tasks.run_monthly_heatmaps:
                    for month in range(1, 13):
                        region.plot_monthly_heatmap(
                            self.config.run.year,
                            month,
                            time_resolution=u.Quantity(self.config.run.monthly_heatmap_resolution))

        # plot siderale complessivo
        plot_sidereal_overall(celestial_regions, self.output_dir / "panoramic")

    def plan_observations(self):
        """
        Generates and prepares observation plans for an entire month.

        This method calculates the total number of days in the specified month and
        iteratively schedules and prepares observation plans for each day of the
        month. It uses the year and month configuration supplied to create
        appropriate observation times for each day within the specified month.

        :raises AttributeError: Raised if mandatory configuration attributes are
            missing or improperly set.

        :return: None
        """
        n_days = cal.monthrange(self.config.run.year, self.config.run.month)[1]

        for day in tqdm.tqdm(
                range(1, n_days + 1),
                desc=f"Planning observations for {self.config.run.year}-{cal.month_name[self.config.run.month]}",
                file=sys.stdout,
                colour="green",
                dynamic_ncols=True):

            local_tz = self.site.timezone
            obs_time_local = local_tz.localize(datetime(self.config.run.year, self.config.run.month, day, 0, 0, 0))
            obs_time = Time(obs_time_local.astimezone(dt_timezone.utc))
            #obs_time = Time(datetime(self.config.run.year, self.config.run.month, day), scale='utc')
            self.prepare_observation_plan(obs_time)

    def prepare_observation_plan(self, obs_time: Time):
        """
        Prepare the directory structure and observation plan.

        This method creates a directory specific to the observation date within the
        base directory defined in the configuration. It retrieves observation targets
        for the specified day and processes each target.

        :param obs_time: The date and time for the observation session.
        :type obs_time: Time
        :return: None
        """

        obs_local = obs_time.to_datetime(timezone=self.site.timezone)
        obs_dir = self.output_dir / obs_local.strftime("%B_%Y") / obs_local.strftime("%Y_%m_%d")

        # obs_dir = self.output_dir / obs_time.strftime("%B_%Y") / obs_time.strftime("%Y_%m_%d")

        obs_dir.mkdir(parents=True, exist_ok=True)

        targets = self.select_observation_targets(obs_time, obs_dir)

        for target in targets:
            self.process_observation_target(target)

    def select_observation_targets(self, obs_time: Time, obs_dir: Path) -> List[Source]:
        """
        Selects and processes observation targets based on the provided observation
        time and output directory for the results. This function manages different
        kinds of sources such as point sources, extended sources applying the necessary
        processing steps for each type.

        :param obs_time: Observation time for which the targets are to be selected
                         and processed.
        :type obs_time: Time
        :param obs_dir: Directory where the results of the observation and
                          processing will be stored.
        :type obs_dir: str
        :return: List of observation targets generated after processing all applicable
                 sources.
        :rtype: List[Source]
        """
        targets = []

        if self.config.tasks.load_point_sources:
            point_sources, extended_sources = load_sources_from_file(
                path=self.point_sources_file,
                qubic_site=self.site,
                obs_time=obs_time,
                point_constraints=self.point_constraints,
                extended_constraints=self.extended_constraints,
                time_resolution=u.Quantity(self.config.general.time_resolution),
                results_dir=obs_dir,
                radius=self.config.run.radius * u.deg,
                radial_samples=self.config.run.radial_samples,
                polar_angle_samples=self.config.run.polar_angle_samples
            )
            targets.extend(point_sources + extended_sources)

        if self.config.tasks.load_extended_sources:
            sky_regions = ExtendedSource.load_sources(
                self.site,
                obs_time,
                self.extended_constraints,
                u.Quantity(self.config.general.time_resolution),
                source_file=self.regions_file,
                results_dir=obs_dir,
                radius=self.config.run.radius * u.deg,
                radial_samples=self.config.run.radial_samples,
                polar_angle_samples=self.config.run.polar_angle_samples)

            targets.extend(sky_regions.values())

        if self.config.tasks.process_moon:
            targets.append(self.create_moon_target(obs_time, obs_dir))

        return targets

    def create_moon_target(self, obs_time: Time, obs_dir: Path) -> PointSource:
        """
        Creates a moon target as a PointSource object initialized with the provided
        observation time and results directory. The method applies the given configuration
        parameters and site specifications for creating the moon target.

        :param obs_time: The observation time for the moon target.
        :param obs_dir: The directory where observation results will be stored.
        :return: A PointSource object representing the moon target.
        """
        return PointSource(
            name="moon",
            qubic_site=self.site,
            obs_time=obs_time,
            constraints=self.point_constraints,
            time_resolution=u.Quantity(self.config.general.time_resolution),
            is_fixed=False,
            results_dir=obs_dir)

    def process_observation_target(self, target: Source):
        """
        Processes the given observation target by configuring it and optionally
        performing tasks such as evaluating constraints, plotting trajectories,
        and writing trajectory data. The specific tasks are determined by the
        configuration settings.

        :param target: The observation target to process.
        :type target: Source
        :return: None
        """
        target.configure()

        if self.config.tasks.run_constraints:
            target.evaluate_constraints(make_plot=True)

        if self.config.tasks.run_trajectory_plots:
            target.plot_trajectory(
                loc_time_resolution=u.Quantity(self.config.run.trajectory_time_resolution),
                make_plot=True)

        if self.config.tasks.write_trajectories:
            target.write_trajectory()

        if self.config.tasks.plot_sidereal and target.is_fixed:
            target.plot_sidereal()


def main():

    if len(sys.argv) > 1:
        config_file = Path(sys.argv[1])
    else:
        config_file = Path(__file__).parents[2] / "configs" / "conf.toml"

    if not config_file.exists() or config_file.suffix != ".toml":
        raise FileNotFoundError(f"Configuration file '{config_file}' not found")

    campaign = ObservationCampaign(config_file)
    campaign.analyze_observability()
    campaign.plan_observations()


if __name__ == '__main__':
    main()
