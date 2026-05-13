import os
import csv
import logging
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap

import calendar as cal
from datetime import datetime as dt, timezone as dt_timezone, timedelta

import astropy.units as u
from astropy.time import Time
from astropy.table import Table, QTable
from astropy.coordinates import SkyCoord, get_body, AltAz, ICRS

from astroquery.simbad import Simbad

from astroplan.plots import plot_sky
from astroplan import Constraint, Observer
from astroplan.utils import time_grid_from_range
from astroplan import AltitudeConstraint

import warnings

plt.ioff()

NO_SIMBAD_SOURCE = False


def load_sources_from_file(
        path: Path,
        qubic_site,
        obs_time,
        point_constraints: list,
        extended_constraints: list,
        time_resolution: u.Quantity,
        results_dir: Path = Path('.'),
        radius: u.Quantity = 14 * u.deg,
        radial_samples: int = 50,
        polar_angle_samples: int = 100):

    """
    Loads sources from a file and categorizes them into point sources and extended sources
    based on their properties. Each source name in the specified file is queried to identify
    its characteristics. If the source has a major axis dimension greater than one arcminute,
    it is categorized as an extended source; otherwise, it is considered a point source.

    :param path: Path to the file containing names of celestial sources.
    :type path: str
    :param qubic_site: Observing site information for QUBIC.
    :param obs_time: Observation time information for the sources.
    :param point_constraints: List of constraints applied to point sources.
    :type point_constraints: list
    :param extended_constraints: List of constraints applied to extended sources.
    :type extended_constraints: list
    :param time_resolution: Time resolution of the observation in appropriate units.
    :type time_resolution: u.Quantity
    :param results_dir: Directory where results are stored. Default is the current directory.
    :type results_dir: str
    :param radius: Radius around the source for extended source calculations, provided in units.
    :type radius: u.Quantity
    :param radial_samples: Number of radial samples for extended sources.
    :type radial_samples: int
    :param polar_angle_samples: Number of polar angle samples for extended sources.
    :type polar_angle_samples: int
    :return: A tuple containing two lists - one of point sources and the other of extended sources.
    :rtype: tuple
    """
    # Aggiungo i campi necessari a SIMBAD
    Simbad.add_votable_fields("ra", "dec", "coo_err_maj", "coo_err_min", "dim")

    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        sources = {row["source"]: bool(int(row["is_fixed"])) for row in reader}

    point_sources = []
    extended_sources = []

    for name in sources:
        unit = (u.hourangle, u.deg)
        coord = None
        is_fixed = sources[name]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tbl = Simbad.query_object(name)


        if not tbl:
            global NO_SIMBAD_SOURCE


            if not NO_SIMBAD_SOURCE:
                print(f" - `{name}` not found in SIMBAD. Trying astropy.coordinates.get_body() ...")
                NO_SIMBAD_SOURCE = True
            try:
                get_body(name, time=obs_time, location=qubic_site.location)
                tbl = {"galdim_majaxis": [0]}
                coord = None
                is_fixed = False
            except Exception as e:
                print(f"Error getting coordinates for {name}: {e}")
                continue


        # Coordinate ICRS for fixed sources only.
        # Solar-system bodies are kept dynamic and will be evaluated on the full time grid.
        if coord is None and "ra" in tbl.colnames if hasattr(tbl, "colnames") else "ra" in tbl:
            ra, dec = tbl["ra"][0], tbl["dec"][0]
            coord = SkyCoord(ra, dec, unit=unit, frame="icrs")

        # Major axis in arcmin – if > 1 it is an extended source
        maj = tbl["galdim_majaxis"][0]
        is_extended = (maj is not None and maj > 1)

        if is_extended:
            src = ExtendedSource(
                name=name,
                coord=coord,
                qubic_site=qubic_site,
                obs_time=obs_time,
                constraints=extended_constraints,
                time_resolution=time_resolution,
                results_dir=results_dir,
                radius=radius,
                radial_samples=radial_samples,
                polar_angle_samples=polar_angle_samples)

            extended_sources.append(src)
        else:
            src = PointSource(
                name=name,
                qubic_site=qubic_site,
                obs_time=obs_time,
                constraints=point_constraints,
                time_resolution=time_resolution,
                coord=coord,
                is_fixed=is_fixed,
                results_dir=results_dir)

            point_sources.append(src)

    return point_sources, extended_sources


class Source(ABC):
    """
    Abstract base class for point or extended sources.
    It contains common methods and delegates the specific logic to the derived classes

    Args
    ----------
    name                (str): source name
    qubic_site:         (Observer): it contains information about an observer’s location and environment
    obs_time:           (Time): observation start time
    constraints:        (list): list of constraints on the source's trajectory
    time_resolution:    (u.Quantity): time resolution for the observation grid
    results_dir:        (str): directory to save the results. Defaults to current directory

    Attributes
    -------------
    (all the above)

    coord:               (SkyCoord):  SkyCoord object (object providing a flexible interface for celestial coordinate
                                      representation, manipulation, and transformation between systems) for a solar system
                                      body as observed from a location on Earth in the Geocentric Celestial Reference
                                      System (GCRS). GCRS is distinct form ICRS (International Celestial Reference System)
                                      mainly in that it is relative to the Earth’s center-of-mass rather than the
                                      solar system Barycenter.
                                      That means this frame includes the effects of aberration (unlike ICRS).

    time_grid:           (list[Time]): Linearly spaced time intervals, each with a duration of time_resolution,
                                       associated with the constraints grid.

    constraints_grid:    (np.ndarray): The grid contains boolean values (True or False) indicating whether the observation
                                       of the source is possible, based on the constraints, for each time interval in
                                       time_grid.

    valid_time_intervals: (list): Time intervals during which the observation of the source is possible,
                                  based on the constraints grid

    plots_dir:            (str): directory to save the plots. Defaults to current directory
    """

    def __init__(
            self,
            name: str,
            qubic_site: Observer,
            obs_time: Time,
            constraints: list,
            time_resolution: u.Quantity,
            is_fixed: bool = True,
            coord: SkyCoord | None = None,
            results_dir: Path = Path('.'),
            logger: logging.Logger = None):

        self.name = name
        self.qubic_site = qubic_site
        self.obs_time = obs_time
        self.is_fixed = is_fixed
        self.time_resolution = time_resolution
        self.constraints = constraints
        self.coord = coord
        self.results_dir = results_dir

        self.time_grid: list[Time] = []
        self.constraints_grid: np.ndarray = np.array([])
        self.valid_time_intervals: list = []
        self.plots_dir: str = ''

        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logger or self.setup_logger()

    def setup_logger(self):

        base = logging.getLogger()
        logger = logging.getLogger(self.name)

        logger.setLevel(base.level)

        fh = logging.FileHandler(self.results_dir / f'{self.name}.log',
                                 mode="w",
                                 encoding="utf-8")

        if base.handlers and base.handlers[0].formatter:
            fh.setFormatter(base.handlers[0].formatter)

        logger.addHandler(fh)
        logger.propagate = False

        return logger

    def configure(self):
        """
        Configuration of the directory where the plots will be saved.
        """

        self.plots_dir = os.path.join(self.results_dir, f'{self.name}_plots')
        os.makedirs(self.plots_dir, exist_ok=True)

    def compute_time_grid(self):
        """
        Creation of the time intervals of the constraints grid (each with a duration of time_resolution).
        It goes from local midnight at the observatory on the date containing obs_time to 1 microsecond
        before the following local midnight.
        """

        if self.time_grid:
            return

        local_tz = self.qubic_site.timezone
        obs_dt_local = self.obs_time.to_datetime(timezone=local_tz)


        start_local = local_tz.localize(dt(obs_dt_local.year, obs_dt_local.month, obs_dt_local.day, 0, 0, 0))
        end_local = start_local + timedelta(days=1) - timedelta(microseconds=1)

        start = Time(start_local.astimezone(dt_timezone.utc))
        end = Time(end_local.astimezone(dt_timezone.utc))

        self.time_grid = time_grid_from_range([start, end], time_resolution=self.time_resolution)

    def compute_valid_times_from_constraints(self):
        """
        Get continuous time intervals during which the observation of the source is possible, i.e.
        where all the constraints are satisfied
        """

        if self.constraints_grid.size == 0:
            self.evaluate_constraints()

        if not self.constraints:
            self.valid_time_intervals = [[self.time_grid[0], self.time_grid[-1]]]
            return

        # Mask that checks whether the constraints are satisfied for each time interval in time_grid.
        mask = np.all(self.constraints_grid == 1, axis=0)
        idx_true = np.where(mask)[0]

        if idx_true.size == 0:
            self.valid_time_intervals = []
            return

        gaps = np.where(np.diff(idx_true) > 1)[0]
        runs = np.split(idx_true, gaps + 1)

        # Each run is one continuous observability window. The indices in a run
        # identify valid grid samples. The stop time is moved to the beginning of
        # the first non-valid slot after the run, so every output file represents
        # one continuous interval for which the constraints remain valid.
        self.valid_time_intervals = []
        for r in runs:
            start = self.time_grid[r[0]]
            stop_idx = min(r[-1] + 1, len(self.time_grid) - 1)
            stop = self.time_grid[stop_idx]

            if stop == start:
                stop = start + self.time_resolution

            self.valid_time_intervals.append([start, stop])

    def ecsv_time_intervals(self):
        """
        Return continuous observation windows for ECSV output.

        The constraint grid is built on one local day, from local midnight
        to the following local midnight. A real observing window can therefore
        be split into two pieces by that artificial boundary: one piece at the
        beginning of the local day and one piece at the end of the same local
        day. For ECSV files only, join those two edge pieces into one continuous
        local-night window crossing midnight.
        """

        if not self.valid_time_intervals:
            self.compute_valid_times_from_constraints()

        # Work on a copy: this method must not modify
        # `self.valid_time_intervals`, because that list is also used by the
        # constraint-grid logic and by the polar plots.
        intervals = list(self.valid_time_intervals)

        # If there are zero or one valid intervals, there is nothing to merge.
        # In that case the ECSV intervals are identical to the standard valid
        # intervals.
        if len(intervals) < 2:
            return intervals

        # Because the time grid is ordered from local midnight to local
        # midnight, a window crossing midnight can only be split into the first
        # and last intervals of the list.
        first_start, first_stop = intervals[0]
        last_start, last_stop = intervals[-1]

        # Check whether the first valid interval starts exactly at the first
        # sample of the grid, i.e. at local midnight of the selected observing
        # day. A small tolerance is used to avoid floating-point precision
        # issues in Astropy Time differences.
        starts_at_local_midnight = abs((first_start - self.time_grid[0]).to_value(u.s)) < 0.5

        # Check whether the last valid interval reaches the end of the local
        # day covered by the grid. The tolerance is one time-resolution step,
        # because the stop value can be either the last grid sample or the first
        # boundary after the last valid sample, depending on how the interval was
        # constructed.
        reaches_end_of_local_day = abs((last_stop - self.time_grid[-1]).to_value(u.s)) <= self.time_resolution.to_value(u.s)

        # If both conditions are true, the first and last intervals are not two
        # independent observing windows. They are the same physical observing
        # window, split only because the grid stops at local midnight.
        if starts_at_local_midnight and reaches_end_of_local_day:

            # Move the early-morning block to the following local day by adding
            # one day to its stop time, and use the evening block as the start.
            # Example for a grid built on local date 2026-05-12:
            #   first interval: [2026-05-12 00:00, 2026-05-12 07:22]
            #   last interval:  [2026-05-12 21:33, 2026-05-12 23:58]
            # These two blocks are on the same local grid day only because the
            # grid is cut at midnight. For the ECSV file, the early-morning
            # block is interpreted as the continuation of the evening block into
            # the following local date, so they become:
            #   [2026-05-12 21:33, 2026-05-13 07:22]
            merged = [last_start, first_stop + 1 * u.day]

            # Keep any intermediate valid windows unchanged, and append the
            # merged night-crossing interval as one single ECSV window.
            intervals = intervals[1:-1] + [merged]

        return intervals

    def _parse_ylabels(self):

        def fmt_val(v):
            if hasattr(v, 'unit') and v.unit.is_equivalent(u.deg):
                return rf"{v.value} ^ \circ"
            return str(v)

        labels = []
        for c in self.constraints:
            minv, maxv = c.min, c.max

            name = c.__class__.__name__.replace('Constraint', '')

            if minv is not None and maxv is None:

                label = rf"${name} \ge {fmt_val(minv)}$"
            elif minv is None and maxv is not None:

                label = rf"${name} \le {fmt_val(maxv)}$"
            else:

                label = rf"${fmt_val(minv)} \le {name} \le {fmt_val(maxv)}$"

            labels.append(label)
        return labels

    def _plot_constraint_grid(self, grid: np.ndarray, make_plot: bool = False):
        """
        Heat-map of constraints, reusable by all data sources

        Parameters
        ----------
        grid: np.ndarray
            constraint grid containing True or False values depending on whether the constraints
            are satisfied or not in a given time interval

        make_plot: bool
            if true, make the constraint grid plot
        """

        if not self.constraints:
            return

        if not make_plot:
            return

        extent = (-0.5, len(self.time_grid) - 0.5,
                  -0.5, len(self.constraints) - 0.5)

        fig, ax = plt.subplots(figsize=(14, 6), tight_layout=True)

        # two-color map: invalid (orange), valid (teal)
        cmap = ListedColormap(["#D95F02", "#1B9E77"])
        ax.imshow(grid, extent=extent, origin="lower", cmap=cmap, vmin=0, vmax=1)

        # Compute percentage of observable hours
        valid_per_slot = np.all(grid, axis=0)
        n_valid = valid_per_slot.sum()
        slot_hours = self.time_resolution.to_value(u.hour)
        tot_hours = n_valid * slot_hours
        percent = tot_hours / 24.0 * 100

        # Set title including percentage
        safe_name = self.name.replace("-", r"\,")
        title = (
                r"$\bf{Constraint\ Grid:}$ " +
                f"${safe_name}$ on {self.obs_time.strftime('%Y-%m-%d')}\n" +
                rf"${percent:.1f} \% \ observable$ - " +
                rf"Time Resolution: ${round(self.time_resolution.value)} \, {self.time_resolution.unit}$"
        )

        fig.suptitle(title)

        n_times = len(self.time_grid)
        desired_n_labels = 12
        major_step = max(1, int(np.ceil(n_times / desired_n_labels)))
        major_ticks = np.arange(0, n_times, major_step)

        if major_ticks[-1] != n_times - 1:
            major_ticks = np.append(major_ticks, n_times - 1)

        times_utc = self.time_grid.to_datetime(timezone=dt_timezone.utc)
        ax.set_xticks(major_ticks)
        ax.set_xticklabels([times_utc[i].strftime("%H:%M") for i in major_ticks], rotation=30, ha="right")
        ax.set_xlabel('UTC Time')

        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(major_ticks)

        times_local = self.time_grid.to_datetime(timezone=self.qubic_site.timezone)
        ax_top.set_xticklabels([times_local[i].strftime("%H:%M") for i in major_ticks], rotation=30, ha="left")
        ax_top.set_xlabel('Local Time')

        ax.tick_params(axis='x', pad=2)
        ax_top.tick_params(axis='x', pad=6)

        ytick_labels = self._parse_ylabels()

        # y-ticks for each constraint class names
        ax.set_yticks(range(len(self.constraints)))
        ax.set_yticklabels(ytick_labels)

        minor_step = max(1, int(np.ceil(len(self.time_grid) / 24)))
        ax.set_xticks(np.arange(-0.5, len(self.time_grid), minor_step), minor=True)
        ax.set_yticks(np.arange(-0.5, len(self.constraints)), minor=True)

        sunset = self.qubic_site.sun_set_time(self.obs_time)
        sunrise = self.qubic_site.sun_rise_time(self.obs_time)
        sunrise_utc = sunrise.to_datetime(timezone=dt_timezone.utc)
        sunset_utc = sunset.to_datetime(timezone=dt_timezone.utc)
        sunrise_local = sunrise.to_datetime(timezone=self.qubic_site.timezone)
        sunset_local = sunset.to_datetime(timezone=self.qubic_site.timezone)
        ax.annotate(
            "Sunrise: "
            f"{sunrise_utc.strftime('%H:%M')} UTC / {sunrise_local.strftime('%H:%M')} LT\n"
            "Sunset:  "
            f"{sunset_utc.strftime('%H:%M')} UTC / {sunset_local.strftime('%H:%M')} LT",
            xy=(len(self.time_grid) - 3.5, len(self.constraints) - 0.5),
            xytext=(len(self.time_grid) - 3.5, len(self.constraints) + 0.5),
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            bbox=dict(boxstyle="round", fc="black", alpha=0.5),
        )

        ax.grid(which='minor', color='#333333', linewidth=1)
        ax.set_aspect("auto")

        ecsv_intervals = self.ecsv_time_intervals()
        if ecsv_intervals:
            # Use the UTC start time of the first real ECSV observation window,
            # so the constraint-grid plot name is consistent with the ECSV and
            # polar-plot filenames when the source is observable.
            plot_time = ecsv_intervals[0][0]
        else:
            # If the source is never observable on this grid, there is no ECSV
            # window to use. Fall back to the UTC start time of the planned
            # observation day, so the constraint grid can still be saved.
            plot_time = self.obs_time

        plot_time_utc = plot_time.to_datetime(timezone=dt_timezone.utc)
        plot_name = f"{self.name}-{plot_time_utc.strftime('%Y%m%d_%H_%M')}_constraint_grid"
        plt.savefig(os.path.join(self.plots_dir, plot_name), dpi=600)
        plt.close()

    def plot_monthly_heatmap(self, year, month, time_resolution=None, cmap='Blues'):
        """
        Plot a monthly heatmap of valid observation windows for the given source.

        Parameters
        ----------
        year: int
            year of interest

        month: int
            month of interest

        time_resolution: u.Quantity
            time resolution for the observation grid

        cmap: str
            colormap to use for plotting
        """

        # use provided time_resolution or fall back to the object's default
        time_resolution = time_resolution if time_resolution is not None else self.time_resolution

        # determine number of days in the specified month
        n_days = cal.monthrange(year, month)[1]
        # Calculate how many time steps fit into a full day at the given resolution
        # note: u.dimensionless_unscaled in order to avoid dimensional warnings with Astropy.Quantity
        steps_per_day = int((24 * u.h / time_resolution).to(u.dimensionless_unscaled).value)

        # Initialize availability matrix: days x time slots
        availability = np.zeros((n_days, steps_per_day), dtype=int)
        # Get the local timezone for the observatory site
        local_tz = self.qubic_site.timezone

        # Build a reference grid for the first day in local time to label x-axis
        # calculate start day
        ref_start_local = local_tz.localize(dt(year, month, 1, 0, 0, 0))
        # conversion to UTC
        ref_start = Time(ref_start_local.astimezone(dt_timezone.utc))
        # calculate end day
        ref_end = ref_start + 1 * u.day - 1 * u.microsecond
        # x-axis build on time resolution steps
        ref_grid = time_grid_from_range([ref_start, ref_end], time_resolution=time_resolution)
        # grid conversion to Local Time
        ref_grid_local = ref_grid.to_datetime(timezone=local_tz)

        # Determine tick positions and labels for each hour
        steps_per_hour = int((1 * u.h / time_resolution).to(u.dimensionless_unscaled).value)
        xticks = np.arange(0, steps_per_day, steps_per_hour)
        xticklabels = [ref_grid_local[i].strftime("%H:%M") for i in xticks]

        # Loop over each day of the month
        for day in range(1, n_days + 1):

            # Local midnight for this day: convert to UTC Astropy Time
            start_dt_local = local_tz.localize(dt(year, month, day, 0, 0, 0))
            start = Time(start_dt_local.astimezone(dt_timezone.utc))
            # Define end of the day interval in UTC
            end = start + 1 * u.day - 1 * u.microsecond

            # Generate the time grid for this day
            day_grid = time_grid_from_range([start, end],
                                            time_resolution=time_resolution)

            # Choose target coordinates: either region_coord or compute via get_body or use self.coord
            if hasattr(self, "region_coord"):
                target = self.region_coord
            else:
                try:
                    target = get_body(self.name, time=day_grid,
                                      location=self.qubic_site.location)
                except Exception:
                    if self.coord is None:
                        raise RuntimeError("Coordinates not initialised; "
                                           "load or compute the source first.")
                    target = self.coord

            # Start with all times valid, then apply each constraint
            mask = np.ones(len(day_grid), dtype=bool)

            for c in self.constraints:
                try:
                    # point source
                    cond = c(self.qubic_site, target, day_grid)
                except Exception:
                    # Extended‑source path (grid_times_targets=True)
                    cond = c(self.qubic_site, target,
                             times=day_grid, grid_times_targets=True)

                # If constraint returns 2D (e.g. multi-axis), reduce to a single boolean per time slot
                if cond.ndim == 2:
                    cond = np.all(cond, axis=0)

                # update the mask retaining only time indeces verifying the constraints
                mask &= cond

            # Store the result for this day as integers (0=invalid, 1=valid)
            availability[day - 1, :] = mask.astype(int)

        # If there is at least one valid slot, plot the heatmap
        if availability.any():

            fig, ax = plt.subplots(figsize=(14, max(3, 0.4 * n_days)), tight_layout=True)
            im = ax.imshow(availability, aspect="auto", cmap=cmap, vmin=0, vmax=1)

            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=45, ha="right")

            ax.set_yticks(np.arange(n_days))
            ax.set_yticklabels([f"{d:02d}" for d in range(1, n_days + 1)])

            ax.set_xlabel("Local time (HH:MM)")
            ax.set_ylabel("Day of month")
            ax.set_title(f"Valid observation windows for '{self.name}' – "
                         f"{cal.month_name[month]} {year}")

            # minor grid for readability
            ax.set_xticks(np.arange(-0.5, steps_per_day, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, n_days, 1), minor=True)
            ax.grid(which="minor", color="#333333", linewidth=0.4)

            os.makedirs(self.plots_dir, exist_ok=True)
            out_fname = os.path.join(self.plots_dir,
                                     f"{self.name}_{year}-{month:02d}_heatmap")

            plt.savefig(out_fname, dpi=600)
            plt.close()

            self.logger.info(f"Saved monthly heat‑map to %s", out_fname)
        else:
            self.logger.info(f"No valid observation windows for '%s' and %s",
                             self.name, cal.month_name[month])

    def months_visible(self,
                       start: Time,
                       end: Time,
                       time_resolution: u.Quantity = 1 * u.h):
        """
        Return a list of (year, month) pairs in which the source
        is observable at least once within the interval [start, end],
        satisfying all constraints simultaneously

        Parameters
        ----------
        start, end : Time
            UTC time bounds for the search interval

        time_resolution : u.Quantity`
            Time step resolution for the grid

        Returns
        -------
        visible : list[tuple[int, int]]
            Chronologically ordered list of (YYYY, MM) pairs
        """

        visible: list[tuple[int, int]] = []

        # Determine target coordinates array: extended region or single point
        if hasattr(self, "region_coord"):
            target_coords = self.region_coord
        else:
            if self.coord is None:
                raise RuntimeError("coord not initialized: load the source first")

            # SkyCoord array of length 1
            target_coords = np.atleast_1d(self.coord)

        # Start from the first day of the month containing start
        current = Time(f"{start.datetime.year}-{start.datetime.month:02d}-01 00:00:00", scale='utc')

        # Iterate month by month until we pass end
        while current < end:

            year = current.datetime.year
            month = current.datetime.month
            n_days = cal.monthrange(year, month)[1]

            month_start = current
            month_end = Time(f"{year}-{month:02d}-{n_days:02d} 23:59:59", scale='utc')

            # If the month starts after end, exit
            if month_start > end:
                break

            # Build a uniform time grid for the entire month
            month_grid = time_grid_from_range([month_start, month_end], time_resolution=time_resolution)

            # Initialize boolean mask: True for each time step until proven otherwise
            all_valid = np.ones(len(month_grid), dtype=bool)

            # Evaluate each constraint on the time grid
            for c in self.constraints:
                try:

                    # cond shape: N_targets x N_times
                    cond = c(self.qubic_site, target_coords,
                             times=month_grid, grid_times_targets=True)

                    if cond.ndim == 2:
                        cond = np.all(cond, axis=0)

                # due to fact that astroplan doesn't manage well extended sources, in case of error
                # evaluate each target separately
                except TypeError:

                    cond_stack = np.vstack([c(self.qubic_site, tc, month_grid) for tc in target_coords])
                    cond = np.all(cond_stack, axis=0)

                all_valid &= cond

                # if no times remain valid, no need to test the other constraints for this month
                if not np.any(all_valid):
                    break

            # If at least one time step survives, the whole month is flagged
            if np.any(all_valid):
                visible.append((year, month))

            # Advance to the first day of next month
            if month == 12:
                current = Time(f"{year + 1}-01-01 00:00:00", scale='utc')
            else:
                current = Time(f"{year}-{month + 1:02d}-01 00:00:00", scale='utc')

        return visible

    def plot_months_visible(self,
                            start: Time,
                            end: Time,
                            time_resolution: u.Quantity = 1 * u.h,
                            plot_path: str | None = None):
        """
        Plot a heatmap (years x months) showing which months the source
        is observable for at least `time_resolution`, given all constraints.

        Parameters
        ----------
        start, end : Time
            UTC time interval to analyze

        time_resolution : u.Quantity
            Time step resolution used by months_visible when sampling each day

        plot_path : str or None
            If provided, save the figure to this path
        """

        # Get list of (year, month) pairs where the source is visible
        vis = self.months_visible(start, end, time_resolution)

        # If no months are observable, exit
        if not vis:
            self.logger.warning("No observable months in the requested period")
            return

        # Build an array of all years in the interval
        years = list(range(start.datetime.year, end.datetime.year + 1))
        # Initialize availability matrix: years x months
        avail = np.zeros((len(years), 12), dtype=int)

        # Mark months with visibility
        for year, month in vis:
            avail[years.index(year), month - 1] = 1

        fig, ax = plt.subplots(figsize=(10, max(3, 0.6 * len(years))),
                               tight_layout=True)

        # two-color map: light gray for not visible, green for visible
        cmap = ListedColormap(["#ECECEC", "#1B9E77"])
        ax.imshow(avail, aspect="auto", cmap=cmap, vmin=0, vmax=1)

        # x axis: months
        ax.set_xticks(range(12))
        ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                           rotation=45, ha="right")

        # y axis: years
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlabel("Month")
        ax.set_ylabel("Year")
        ax.set_title(f"Months observable for '{self.name}'")

        ax.set_xticks(np.arange(-.5, 12, 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(years), 1), minor=True)
        ax.grid(which="minor", color="#333333", linewidth=0.5)

        if plot_path:
            fname = f"{self.name}_months_visible"
            plt.savefig(os.path.join(self.plots_dir, fname), dpi=600)

        plt.close()

    def sidereal_track(self, *, n_samples=400):
        """
        Calculates the sidereal track of the object's center position over time, using a specified
        number of time samples. The function determines the local sidereal time (LST), elevation
        track, and also applies any altitude constraints defined in the observation conditions.

        :param n_samples: Number of time samples to evaluate the sidereal track. Defaults to 400.
        :type n_samples: int, optional
        :return: A tuple containing:
            - lst: Local sidereal time values (in hours), sorted in ascending order.
            - elevations: Elevations (in degrees) of the center position at the corresponding LST.
            - alt_min: Minimum altitude constraint (in degrees), if an AltitudeConstraint is defined.
              Otherwise, None.
            - alt_max: Maximum altitude constraint (in degrees), if an AltitudeConstraint is defined.
              Otherwise, None.
            - visible_mask: A boolean mask indicating whether the center's elevation satisfies the
              altitude constraints at each sampled LST.
        :rtype: tuple
        """
        center = self.coord if self.coord.isscalar else self.coord[0]

        time_grid = self.obs_time + np.linspace(0, 25, n_samples) * u.hour
        lst = self.qubic_site.local_sidereal_time(time_grid).hour

        sort_idx = np.argsort(lst)
        time_grid = time_grid[sort_idx]
        lst = lst[sort_idx]

        # Altitude track of the region center
        altaz = AltAz(obstime=time_grid, location=self.qubic_site.location)
        elevations = center.transform_to(altaz).alt.to_value(u.deg)

        alt_min, alt_max = None, None
        for c in self.constraints or []:
            if isinstance(c, AltitudeConstraint):
                alt_min = c.min.to_value(u.deg) if c.min is not None else 0.0
                alt_max = c.max.to_value(u.deg) if c.max is not None else 90.0
                break

        if alt_min is None or alt_max is None:
            visible_mask = np.ones_like(elevations, dtype=bool)
        else:
            visible_mask = (elevations >= alt_min) & (elevations <= alt_max)

        return lst, elevations, alt_min, alt_max, visible_mask


    def plot_sidereal(self, ang_offset: float = 15.0, n_samples: int = 400):

        if not self.time_grid:
            self.compute_time_grid()

        lst, elev, alt_min, alt_max, vis = self.sidereal_track(n_samples=n_samples)

        frac_visible = vis.mean() * 100.0

        fig, ax = plt.subplots(figsize=(12, 6), tight_layout=True)

        # Gray band for the instrumental elevation window
        band_label = r"${} ^ \circ \leq \mathrm{{{}}} \leq {} ^ \circ$"
        ax.axhspan(alt_min, alt_max,
                   color="lightgray",
                   alpha=0.6,
                   label=band_label.format(int(alt_min), r"Elevation \, Window", int(alt_max)))

        ax.axhspan(alt_min + ang_offset, alt_max - ang_offset,
                   color="silver",
                   alpha=0.6,
                   label=band_label.format(int(alt_min + ang_offset),
                                           r"Elevation \, Center",
                                           int(alt_max - ang_offset)))

        # Full center-elevation curve
        ax.plot(lst, elev, linestyle=(0, (4, 4)), color="gray", linewidth=1.5, label="Not Visible")

        # Overplot only the visible segments in solid blue
        if vis.any():
            ax.plot(lst, np.where(vis, elev, np.nan), linewidth=2.5, label='Visible (Source Center)')

        ax.set_xlim(0,  24)
        ax.set_ylim(0, 90)
        ax.set_xticks(np.arange(0,  25, 3))
        ax.set(xlabel=r"Local Sidereal Time (LST) [h]", ylabel=r"Elevation [$ ^ \circ$]")
        ax.grid(True, alpha=0.3, linestyle=":", color="gray")

        safe_name = self.name # .replace("-", r"\-")
        obs_local_date = self.obs_time.to_datetime(timezone=self.qubic_site.timezone).strftime("%Y-%m-%d")
        title = rf"$\bf{{Observation \, window}}$ : ${safe_name}$ on {obs_local_date}" + "\n"
        # title = rf"$\bf{{Observation \, window}}$ : ${safe_name}$ on {self.obs_time.strftime('%Y-%m-%d')}" + "\n"

        if hasattr(self, 'radius'):
            title += rf"Ring of radius ${self.radius.to_value(u.deg):.1f}^\circ$"
        else:
            title += ""
        title += rf" fully in $[{int(alt_min)}^\circ, {int(alt_max)}^\circ]$ for {frac_visible:.1f} $\%$"

        ax.set_title(title, fontsize=11, pad=10)

        ax.legend()

        save_path = os.path.join(self.plots_dir, f"{self.name}_sidereal_time")
        plt.savefig(save_path)
        plt.close()

        self.logger.info(f"Saved sidereal time plot to %s", save_path)

    def write_trajectory(self, dest: str | None = None):
        """
        Write one ECSV file per continuous valid observation window.

        Parameters
        ----------
        dest : str or None
            If provided, save the trajectory to this path
        """

        # Ensure valid intervals are computed
        if not self.valid_time_intervals:
            self.compute_valid_times_from_constraints()

        if not self.valid_time_intervals:
            return

        # Ensure time grid and coordinates are available
        if not self.time_grid:
            self.compute_time_grid()

        if self.coord is None:
            raise RuntimeError("Coordinates not initialized: load the source first")

        dest = dest or self.plots_dir

        # Loop over each valid time interval
        for start, stop in self.ecsv_time_intervals():

            times_local, times_utc, ras, decs, elevations, azimuths = [], [], [], [], [], []

            start_utc = start.to_datetime(timezone=dt_timezone.utc)

            # Name ECSV files by the UTC start time of the observation window.
            name = f"{self.name}-{start_utc.strftime('%Y%m%d_%H_%M')}"

            # sample times in the interval
            tw = time_grid_from_range([start, stop], time_resolution=self.time_resolution)

            # saving central coords of the extended region
            if self.coord.isscalar:

                # project the central coords over time_grid in altaz coords
                altaz = AltAz(obstime=tw, location=self.qubic_site.location)
                sc_altaz = self.coord.transform_to(altaz)

                # Collect times and coordinates
                times_local.extend(tw.to_datetime(timezone=self.qubic_site.timezone))
                times_utc.extend(tw.to_datetime(timezone=dt_timezone.utc))

                ras.extend(np.full(len(tw), self.coord.ra.deg))
                decs.extend(np.full(len(tw), self.coord.dec.deg))

                elevations.extend(sc_altaz.alt.deg)
                azimuths.extend(sc_altaz.az.deg)
            else:
                # Mask the global time grid to this interval
                mask = (self.time_grid >= start) & (self.time_grid <= stop)
                times_sel = self.time_grid[mask]

                if len(times_sel) == 0:
                    continue

                altaz = AltAz(obstime=times_sel, location=self.qubic_site.location)
                coords_sel = self.coord[mask]

                altaz_coords = coords_sel.transform_to(altaz)
                icrs_coords = coords_sel.transform_to('icrs')

                # Collect the masked times and corresponding RA/DEC
                times_local.extend(times_sel.to_datetime(timezone=self.qubic_site.timezone))
                times_utc.extend(times_sel.to_datetime(timezone=dt_timezone.utc))

                ras.extend(icrs_coords.ra.deg)
                decs.extend(icrs_coords.dec.deg)

                elevations.extend(altaz_coords.alt.deg)
                azimuths.extend(altaz_coords.az.deg)

            # Convert datetime objects to ISO strings
            times_local = [t.strftime('%Y-%m-%d %H:%M:%S') for t in times_local]
            times_utc = [t.strftime('%Y-%m-%d %H:%M:%S') for t in times_utc]

            # Build QTable and write to ECSV
            table = QTable([times_utc, times_local, ras, decs, elevations, azimuths],
                           names=("time_utc", "time_local", "ra", "dec", "elevation", "azimuth"))

            savepath = os.path.join(dest, f"{name}.ecsv")

            table.write(savepath, format="ascii.ecsv", overwrite=True)
            self.logger.info(f"Wrote valid observation times with RA/DEC to %s - %s",
                             dest, start.strftime('%Y/%m/%d %H:%S'))

    @classmethod
    @abstractmethod
    def load_sources(
            cls,
            qubic_site,
            obstime,
            constraints,
            time_resolution,
            source_file: str,
            radius: u.Quantity = 10 * u.deg,
            radial_samples: int = 50,
            polar_angle_samples: int = 100,
    ):
        ...

    @abstractmethod
    def evaluate_constraints(self, make_plot: bool = False):
        ...

    @abstractmethod
    def plot_trajectory(self, loc_time_resolution: u.Quantity, make_plot: bool = False):
        ...


class PointSource(Source):

    def load_point_source(self):
        if not self.time_grid:
            self.compute_time_grid()

        if not self.coord:
            self.coord = get_body(self.name, time=self.time_grid,
                                  location=self.qubic_site.location)

    @classmethod
    def load_sources(cls, *args, **kwargs):
        raise NotImplementedError("Not implemented yet")

    def evaluate_constraints(self, make_plot: bool = False):
        """
        Evaluate each constraint for the source over its time grid and
        optionally display a heatmap of constraint satisfaction.

        Parameters
        ----------
        make_plot : bool
            If True, generate and display a plot of the constraints grid
        """

        if not self.time_grid:
            self.compute_time_grid()

        if self.coord is None:
            self.load_point_source()

        grid = np.zeros((len(self.constraints), len(self.time_grid)))

        for i, c in enumerate(self.constraints):
            grid[i, :] = c(self.qubic_site, self.coord, self.time_grid)

        self.constraints_grid = grid
        self._plot_constraint_grid(grid, make_plot)

    def plot_trajectory(self, loc_time_resolution: u.Quantity, make_plot: bool = False):
        """
        Plot the sky trajectory of the source and the Sun during valid observation intervals.

        Parameters
        ----------
        loc_time_resolution : u.Quantity
            Time step resolution for sampling each valid interval

        make_plot : bool
            If True, save the resulting plot to disk
        """

        if self.coord is None:
            self.load_point_source()
        if not self.valid_time_intervals:
            self.compute_valid_times_from_constraints()

        if not self.valid_time_intervals:
            return

        add_label = True

        for start, stop in self.valid_time_intervals:
            tw = time_grid_from_range([start, stop], time_resolution=loc_time_resolution)
            altaz = AltAz(obstime=tw, location=self.qubic_site.location)

            if self.coord.isscalar:
                source = self.coord.transform_to(altaz)
            else:
                source = get_body(self.name, time=tw, location=self.qubic_site.location)

            sun = get_body("sun", time=tw, location=self.qubic_site.location)

            # altaz_frame = AltAz(obstime=tw, location=self.qubic_site.location)
            # disk_altaz = source.transform_to(altaz_frame)
            # print(disk_altaz.alt.deg, tw.strftime('%H:%M'))

            plot_sky(
                source, self.qubic_site, tw,
                style_kwargs={
                    "marker": "o",
                    "label": f"{self.name}: {start.datetime.strftime('%H:%M')}"
                             f"-{stop.datetime.strftime('%H:%M')}"
                },
                north_to_east_ccw=False)

            plot_sky(
                sun, self.qubic_site, tw,
                style_kwargs={
                    "marker": "*",
                    "color": "gold",
                    "label": "sun" if add_label else ""
                },
                north_to_east_ccw=False)

            add_label = False

        safe_name = self.name.replace("-", r"\,")
        obs_local_date = self.obs_time.to_datetime(timezone=self.qubic_site.timezone).strftime("%Y-%m-%d")
        title = (r"$\bf{Polar\ plot:}$ " + f"${safe_name}$ on {obs_local_date}\n" +
                 rf"Time Resolution: ${round(loc_time_resolution.value)} \, {loc_time_resolution.unit}$")

        plt.legend(loc="upper right", bbox_to_anchor=(1.50, 1))
        fig = plt.gcf()
        fig.suptitle(title)
        fig.tight_layout()
        plt.gca().set_facecolor("whitesmoke")
        if make_plot:
            first_start = self.ecsv_time_intervals()[0][0]
            first_start_utc = first_start.to_datetime(timezone=dt_timezone.utc)
            plot_name = f"{self.name}-{first_start_utc.strftime('%Y%m%d_%H_%M')}_polar_plot"
            plt.savefig(os.path.join(self.plots_dir, plot_name), dpi=600)
        plt.close()


class ExtendedSource(Source):

    def __init__(
            self,
            name: str,
            coord: SkyCoord,
            qubic_site: Observer,
            obs_time: Time,
            constraints: list,
            region_coord: SkyCoord | None = None,
            results_dir: Path = '.',
            time_resolution: u.Quantity = 1 * u.h,
            radius: u.Quantity = 10 * u.deg,
            radial_samples: int = 50,
            polar_angle_samples: int = 100):

        super().__init__(name, qubic_site, obs_time,
                         constraints, time_resolution, True, coord, results_dir)

        self.radius = radius
        self.radial_samples = radial_samples
        self.polar_angle_samples = polar_angle_samples

        if not region_coord:
            # Generate an array of equally spaced position angles [0, 360) degrees
            pas = np.linspace(0, 360, polar_angle_samples) * u.deg
            # Create an array of constant separations equal to the specified radius
            seps = np.full_like(pas, radius)

            # Sample points around the central coordinate to define the disk
            disk_icrs = self.coord.directional_offset_by(
                position_angle=pas,
                separation=seps)

            self.region_coord = disk_icrs

    @classmethod
    def load_sources(
            cls,
            qubic_site,
            obstime,
            constraints,
            time_resolution,
            source_file: Path,
            results_dir: Path = '.',
            radius: u.Quantity = 14 * u.deg,
            radial_samples: int = 50,
            polar_angle_samples: int = 100):

        """
        Load source definitions from an ECSV file and create Source instances
        for each region, sampling a disk around the central coordinate.

        Parameters
        ----------
        qubic_site : Site
            Observatory site providing location and timezone

        obstime : Time
            Observation time for computing geocentric transformations

        constraints : list[Constraint]
            List of constraint callables to apply to each source

        time_resolution : u.Quantity
            Time step used when sampling observation intervals

        source_file : str
            Path to an ECSV file with columns ['name', 'l_deg', 'b_deg'] in galactic coords

        results_dir : str, optional
            Directory in which to store results (default: current directory)

        radius : u.Quantity, optional
            Angular radius of the sampled disk around each source center (default: 10 deg)

        radial_samples : int, optional
            Number of radial samples for disk (unused in this implementation)

        polar_angle_samples : int, optional
            Number of position-angle samples around the disk (default: 100)

        Returns
        -------
        dict[str, cls]
            Mapping from source name to an instantiated Source object.
        """

        if not os.path.isfile(source_file):
            raise FileNotFoundError(source_file)

        # Read the ECSV table of sky regions
        sky_regions = Table.read(source_file, format="ascii.ecsv")

        # # Generate an array of equally spaced position angles [0, 360) degrees
        # pas = np.linspace(0, 360, polar_angle_samples) * u.deg
        # # Create an array of constant separations equal to the specified radius
        # seps = np.full_like(pas, radius)

        result = {}

        # Loop over each row in the table to build source instances
        for row in sky_regions:
            name = row["name"]
            # Construct ICRS coordinate from galactic (l, b) values
            center_icrs = SkyCoord(
                row["l_deg"], row["b_deg"],
                unit=u.deg, frame="galactic"
            ).transform_to("icrs")

            result[name] = cls(
                name, center_icrs,
                qubic_site, obstime, constraints, None, results_dir,
                time_resolution, radius, radial_samples, polar_angle_samples)

        return result

    def evaluate_constraints(self, make_plot: bool = False):
        """
        Evaluate each constraint on the predefined time grid for an extended source
        (using region_coord) and optionally display a heatmap of the results.

        Parameters
        ----------
        make_plot : bool
            If True, generate and show a plot of the constraints grid
        """

        if not self.time_grid:
            self.compute_time_grid()

        if not self.constraints:
            return

        grid = np.zeros((len(self.constraints), len(self.time_grid)))

        for i, c in enumerate(self.constraints):
            grid[i, :] = np.all(
                c(times=self.time_grid,
                  observer=self.qubic_site,
                  targets=self.region_coord,
                  grid_times_targets=True),
                axis=0)

        self.constraints_grid = grid
        self._plot_constraint_grid(grid, make_plot)

    def plot_trajectory(self, loc_time_resolution: u.Quantity, make_plot: bool = False):
        """
        Plot the altitude trajectories of the disk center and the two edge points
        ("top" = center + radius, "bottom" = center – radius), alongside the Sun and Moon.

        Parameters
        ----------
        loc_time_resolution : u.Quantity
            Time step resolution for sampling each valid interval

        make_plot : bool
            If True, save the resulting polar plot to disk
        """

        if not self.valid_time_intervals:
            self.compute_valid_times_from_constraints()

        if not self.valid_time_intervals:
            return

        add_label = True

        for start, stop in self.valid_time_intervals:
            tw = time_grid_from_range([start, stop], time_resolution=loc_time_resolution)
            altaz = AltAz(obstime=tw, location=self.qubic_site.location)

            # center
            center_altaz = self.coord.transform_to(altaz)

            #  shift +\- radius
            # offset = self.radius
            # top_altaz = SkyCoord(
            #     alt=np.minimum(center_altaz.alt + offset, 90 * u.deg),
            #     az=center_altaz.az, frame=altaz
            # )
            # bottom_altaz = SkyCoord(
            #     alt=np.maximum(center_altaz.alt - offset, -90 * u.deg),
            #     az=center_altaz.az, frame=altaz
            # )

            sun = get_body("sun", time=tw, location=self.qubic_site.location).transform_to(altaz)
            moon = get_body("moon", time=tw, location=self.qubic_site.location).transform_to(altaz)

            plot_sky(center_altaz, self.qubic_site, tw,
                     style_kwargs={"marker": "o",
                                   "label": f'Center: {start.datetime.strftime("%H:%M")} - {stop.datetime.strftime("%H:%M")} UTC'},
                     north_to_east_ccw=False)

            # plot_sky(top_altaz, self.qubic_site, tw,
            #          style_kwargs={"marker": "^",
            #                        "label": f"Top (+{offset})" if add_label else ""},
            #          north_to_east_ccw=False)
            #
            # plot_sky(bottom_altaz, self.qubic_site, tw,
            #          style_kwargs={"marker": "v",
            #                        "label": f"Bottom (–{offset})" if add_label else ""},
            #          north_to_east_ccw=False)

            plot_sky(sun, self.qubic_site, tw,
                     style_kwargs={"marker": "*", "color": "gold",
                                   "label": "Sun" if add_label else ""},
                     north_to_east_ccw=False)

            plot_sky(moon, self.qubic_site, tw,
                     style_kwargs={"marker": "+", "color": "red",
                                   "label": "Moon" if add_label else ""},
                     north_to_east_ccw=False)
            add_label = False

        plt.legend(loc="upper right", bbox_to_anchor=(1.60, 1))
        fig = plt.gcf()

        safe_name = self.name.replace("-", r"\,")
        obs_local_date = self.obs_time.to_datetime(timezone=self.qubic_site.timezone).strftime("%Y-%m-%d")
        title = (
                r"$\bf{Polar\ plot:}$ " + f"${safe_name}$ on {obs_local_date}\n"
                                          r"QUBIC Patch Radius " + f"{self.radius.value:.1f}" + r"$ ^ \circ$ - " +
                rf"Time Resolution: ${round(loc_time_resolution.value)} \, {loc_time_resolution.unit}$")

        fig.suptitle(title)
        fig.tight_layout()
        plt.gca().set_facecolor("whitesmoke")

        if make_plot:
            first_start = self.ecsv_time_intervals()[0][0]
            first_start_utc = first_start.to_datetime(timezone=dt_timezone.utc)
            plot_name = f"{self.name}-{first_start_utc.strftime('%Y%m%d_%H_%M')}_polar_plot"
            plt.savefig(os.path.join(self.plots_dir, plot_name), dpi=600)
        plt.close()
