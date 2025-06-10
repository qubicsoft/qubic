from abc import ABC, abstractmethod  # abc: abstract base class

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap

import calendar as cal
from datetime import datetime as dt, timezone as dt_timezone

import astropy.units as u
from astropy.time import Time
from astropy.table import Table, QTable
from astropy.coordinates import SkyCoord, get_body, EarthLocation, AltAz, GCRS

from astroquery.simbad import Simbad

from pytz import timezone
from astroplan.plots import plot_sky
from astroplan import FixedTarget, Constraint, Observer
from astroplan.utils import time_grid_from_range
from astroplan import AltitudeConstraint, AirmassConstraint

from constraint import SunSeparationConstraint, MoonSeparationConstraint

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'


def load_sources_from_file(
        path: str,
        qubic_site,
        obs_time,
        point_constraints: list,
        extended_constraints: list,
        time_resolution: u.Quantity,
        results_dir: str = '.',
        radius: u.Quantity = 14 * u.deg,
        radial_samples: int = 50,
        polar_angle_samples: int = 100):
    """
    Carica da un file di testo una lista di sorgenti e le classifica
    in PointSource ed ExtendedSource.

    Restituisce una tupla: (point_sources, extended_sources).
    """
    # Aggiungo i campi necessari a SIMBAD
    Simbad.add_votable_fields("ra", "dec", "coo_err_maj", "coo_err_min", "dim")

    # Leggo i nomi dal file
    with open(path) as f:
        names = [line.strip() for line in f if line.strip()]

    point_sources = []
    extended_sources = []

    for name in names:
        tbl = Simbad.query_object(name)
        if tbl is None:
            print(f"{name}: non trovato")
            continue

        # Coordinate ICRS
        ra, dec = tbl["ra"][0], tbl["dec"][0]
        coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame="icrs")

        # Major axis in arcmin – se > 1 consideriamo estesa
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
                polar_angle_samples=polar_angle_samples
            )
            extended_sources.append(src)
        else:
            src = PointSource(
                name=name,
                qubic_site=qubic_site,
                obs_time=obs_time,
                constraints=point_constraints,
                time_resolution=time_resolution,
                coord=coord,
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
            coord: SkyCoord | None = None,
            results_dir: str = '.'):

        self.name = name
        self.qubic_site = qubic_site
        self.obs_time = obs_time
        self.time_resolution = time_resolution
        self.constraints = constraints
        self.coord = coord
        self.results_dir = results_dir

        self.time_grid: list[Time] = []
        self.constraints_grid: np.ndarray = np.array([])
        self.valid_time_intervals: list = []
        self.plots_dir: str = '..'

    def configure(self):
        """
        Configuration of the directory where the plots will be saved.
        """

        os.makedirs(self.results_dir, exist_ok=True)

        self.plots_dir = os.path.join(self.results_dir, f'{self.name}_plots')
        os.makedirs(self.plots_dir, exist_ok=True)

    def compute_time_grid(self):
        """
        Creation of the time intervals of the constraints grid (each with a duration of time_resolution).
        It goes from the midnight of the observation start time (obs_time) to 1 microsecond
        before the midnight of the following day.
        """

        if self.time_grid:
            return

        # start and end observation times in UTC
        start = self.obs_time
        end = start + 1 * u.day - 1 * u.microsecond
        # Get linearly-spaced sequence of times, each with a duration of time_resolution
        self.time_grid = time_grid_from_range([start, end],
                                              time_resolution=self.time_resolution)

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
        # It contains True or False depending on whether the constraints are met
        mask = np.all(self.constraints_grid == 1, axis=0)
        # Indices of time_grid that correspond to True values in the mask
        idx_true = np.where(mask)[0]

        if idx_true.size == 0:
            self.valid_time_intervals = []
            return

        # Starting indices of uninterrupted time interval blocks
        gaps = np.where(np.diff(idx_true) > 1)[0]
        # Arrays of indices corresponding to uninterrupted time interval blocks
        runs = np.split(idx_true, gaps + 1)

        # list of lists containing the start and stop time values of uninterrupted time interval blocks
        # note: the indices in runs refer to the temporal beginning of each time interval block (which constitute the
        # time_grid), NOT to the end of the block.
        # So when the last block of an uninterrupted time interval blocks is reached
        # we have to retrieve the index of the end of the block by summing 1 to r[-1] (the end index of a block is the
        # start index of the following one)
        # (if we the last block correspond to the end of time_grid, the index of the block end will be set to
        # len(self.time_grid) - 1)
        # self.valid_time_intervals = [[self.time_grid[r[0]],
        #                               self.time_grid[min(r[-1] + 1, len(self.time_grid) - 1)]] for r in runs]

        self.valid_time_intervals = [[self.time_grid[r[0]],
                                      self.time_grid[r[-1]]] for r in runs]

        # Check if the start time of the last time interval block is equal to its end time.
        # If so, add one time_resolution to properly account for the final time interval block

        for idx, (start, stop) in enumerate(self.valid_time_intervals):
            if start == stop:
                self.valid_time_intervals[idx] = [start, stop + self.time_resolution - 1 * u.s]

        # last_valid = self.valid_time_intervals[-1]
        # if last_valid[0] == last_valid[-1]:
        #     self.valid_time_intervals[-1][1] += self.time_resolution - 1 * u.s

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

        # Compute time resolution from the first two entries of time_grid
        dt = (self.time_grid[1] - self.time_grid[0]).to(u.hour)
        # Number of time steps in a full 24h period at this resolution
        N = int((24 * u.hour / dt).decompose().value)

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
        tot_hours = n_valid * dt.value  # dt is in hours
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

        ax.set_xticks(range(len(self.time_grid)))
        ax.set_xticklabels([t.datetime.strftime("%H:%M") for t in self.time_grid], rotation=30, ha="right")
        ax.set_xlabel('UTC Time')

        ytick_labels = self._parse_ylabels()

        # y-ticks for each constraint class names
        ax.set_yticks(range(len(self.constraints)))
        ax.set_yticklabels(ytick_labels)

        ax.set_xticks(np.arange(-0.5, N), minor=True)
        ax.set_yticks(np.arange(-0.5, len(self.constraints)), minor=True)

        ax.grid(which='minor', color='#333333', linewidth=1)
        ax.set_aspect("equal")

        plt.savefig(os.path.join(self.plots_dir, f"{self.name}_constraint_grid.png"), dpi=400)
        plt.show()

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

            # Local midnight for this day → convert to UTC Astropy Time
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

            # save & show
            os.makedirs(self.plots_dir, exist_ok=True)
            out_fname = os.path.join(self.plots_dir,
                                     f"{self.name}_{year}-{month:02d}_heatmap.png")

            plt.savefig(out_fname, dpi=400)
            plt.show()
            print(f"Saved monthly heat‑map to {out_fname}")
        else:
            print(f"No valid observation windows for '{self.name}' and {cal.month_name[month]} ")

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

                # due to fact that astroplan doesn't manage well extended sources, in cas of error
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
        Plot a heatmap (years × months) showing which months the source
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
            print("No observable months in the requested period")
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
            fname = f"{self.name}_months_visible.png"
            plt.savefig(os.path.join(self.plots_dir, fname), dpi=400)

        plt.show()

    def write_trajectory(self, dest: str | None = None):
        """
        Write valid observation times with RA and DEC to an ECSV file.

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

        # Default filename path in plots_dir
        dest = dest or self.plots_dir

        # Loop over each valid time interval
        for start, stop in self.valid_time_intervals:

            times_local, times_utc, ras, decs, elevations, azimuths = [], [], [], [], [], []
            name = f"{self.name}-{start.strftime('%Y%m%d_%H_%M')}-{stop.strftime('%H_%M')}"

            # sample times in the interval
            tw = time_grid_from_range([start, stop], time_resolution=self.time_resolution)
            # Transform to AltAz frame at each sample time
            altaz = AltAz(obstime=tw, location=self.qubic_site.location)

            # saving central coords of the extended region
            if self.coord.isscalar:

                # project the central coords over time_grid in altaz coords
                sc_altaz = self.coord.transform_to(altaz)
                # Convert back to ICRS to get RA/DEC
                sc_icrs = sc_altaz.transform_to('icrs')

                # Collect times and coordinates
                times_local.extend(tw.to_datetime(timezone=self.qubic_site.timezone))
                times_utc.extend(tw.to_datetime(timezone=dt_timezone.utc))

                ras.extend(sc_icrs.ra.deg)
                decs.extend(sc_icrs.dec.deg)

                elevations.extend(sc_altaz.alt.deg)
                azimuths.extend(sc_altaz.az.deg)
            else:
                mask = (self.time_grid >= start) & (self.time_grid <= stop)

                altaz_coords = self.coord[mask].transform_to(altaz)
                # For point sources, use the precomputed ICRS coordinates array
                icrs_coords = self.coord[mask].transform_to('icrs')
                # Mask the global time grid to this interval

                # Collect the masked times and corresponding RA/DEC
                times_local.extend(self.time_grid[mask].to_datetime(timezone=self.qubic_site.timezone))
                times_utc.extend(self.time_grid[mask].to_datetime(timezone=dt_timezone.utc))

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
            print(f"Wrote valid observation times with RA/DEC to {dest} - {start.strftime('%Y/%m/%d %H:%S')}")

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
    def load_sources(*args, **kwargs):
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
                north_to_east_ccw=False,
            )
            plot_sky(
                sun, self.qubic_site, tw,
                style_kwargs={
                    "marker": "*",
                    "color": "gold",
                    "label": "sun" if add_label else ""
                },
                north_to_east_ccw=False,
            )

            add_label = False

        safe_name = self.name.replace("-", r"\,")
        title = (r"$\bf{Polar\ plot:}$ " + f"${safe_name}$ on {self.obs_time.strftime('%Y-%m-%d')}\n" +
                 rf"Time Resolution: ${round(loc_time_resolution.value)} \, {loc_time_resolution.unit}$")

        plt.legend(loc="upper right", bbox_to_anchor=(1.50, 1))
        fig = plt.gcf()
        fig.suptitle(title)
        fig.tight_layout()
        plt.gca().set_facecolor("whitesmoke")
        if make_plot:
            plt.savefig(os.path.join(self.plots_dir, f"{self.name}_polar_plot"),
                        dpi=400)
        plt.show()


class ExtendedSource(Source):

    def __init__(
            self,
            name: str,
            coord: SkyCoord,
            qubic_site: Observer,
            obs_time: Time,
            constraints: list,
            region_coord: SkyCoord | None = None,
            results_dir: str = '.',
            time_resolution: u.Quantity = 1 * u.h,
            radius: u.Quantity = 10 * u.deg,
            radial_samples: int = 50,
            polar_angle_samples: int = 100):

        super().__init__(name, qubic_site, obs_time,
                         constraints, time_resolution, coord, results_dir)

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
            source_file: str,
            results_dir: str = '.',
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
                                   "label": f'Center: {start.datetime.strftime("%H:%M")} - {stop.datetime.strftime("%H:%M")}'},
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
        title = (
                r"$\bf{Polar\ plot:}$ " + f"${safe_name}$ on {self.obs_time.strftime('%Y-%m-%d')}\n"
                                          r"QUBIC Patch Radius " + f"{self.radius.value:.1f}" + r"$^\circ$ - " +
                rf"Time Resolution: ${round(loc_time_resolution.value)} \, {loc_time_resolution.unit}$"
        )

        fig.suptitle(title)
        fig.tight_layout()
        plt.gca().set_facecolor("whitesmoke")

        if make_plot:
            plt.savefig(os.path.join(self.plots_dir,
                                     f"{self.name}_polar_plot"), dpi=400)
        plt.show()


if __name__ == "__main__":
    obs_time = Time('2025-06-12T00:00:00')  # right region
    region_path = "data/sky_regions.ecsv"
    # sources_path = "data/sources.txt"

    time_resolution = 30 * u.min
    point_source_constraints = [
        AltitudeConstraint(35 * u.deg, 85 * u.deg),
        AirmassConstraint(3),
        SunSeparationConstraint(min=50 * u.deg)
    ]

    extended_source_constraints = point_source_constraints + [MoonSeparationConstraint(min=30 * u.deg)]

    location = EarthLocation.from_geodetic(
        lon=-66.8714 * u.deg,
        lat=-24.1844 * u.deg,
        height=4820 * u.m,
    )
    qubic_site = Observer(
        name="Qubic",
        location=location,
        pressure=0.5533 * u.bar,
        relative_humidity=0.20,
        timezone=timezone("America/Argentina/Salta"),
        description="Qubic telescope on Alto Chorrillos, Salta",
    )

    month = 6
    year = 2025
    base_dir = "plots_trajectories"
    n_days = cal.monthrange(year, month)[1]

    for day in range(1, n_days + 1):
        obs_time = Time(f'{year}-{month}-{day}T00:00:00')
        result_dir = os.path.join(base_dir, obs_time.strftime("%Y%m%d"))

        # points, extended = load_sources_from_file(
        #     path=sources_path,
        #     obs_time=obs_time,
        #     point_constraints=point_source_constraints,
        #     time_resolution=time_resolution,
        #     results_dir=result_dir,
        #     qubic_site=qubic_site,
        #     extended_constraints=extended_source_constraints,
        #     radial_samples=50,
        #     polar_angle_samples=100)
        #
        # for source in points + extended:
        #     source.configure()
        #     source.evaluate_constraints(make_plot=True)
        #     source.plot_trajectory(loc_time_resolution=20 * u.min, make_plot=True)
        #     source.write_trajectory()

        moon_source = PointSource("moon",
                                  qubic_site,
                                  obs_time,
                                  point_source_constraints,
                                  time_resolution=time_resolution,
                                  results_dir=result_dir)
        #
        moon_source.configure()
        moon_source.evaluate_constraints(make_plot=True)
        moon_source.plot_trajectory(loc_time_resolution=time_resolution / 2, make_plot=True)
        moon_source.write_trajectory()

        regions = ExtendedSource.load_sources(
            qubic_site, obs_time, extended_source_constraints,
            time_resolution=time_resolution,
            radius=14 * u.deg,
            source_file=region_path,
            results_dir=result_dir
        )

        for region in regions:
            region = regions[region]
            region.configure()
            region.evaluate_constraints(make_plot=True)
            region.plot_trajectory(loc_time_resolution=time_resolution / 2, make_plot=True)
            region.write_trajectory()

        #     region.plot_months_visible(Time("2025-01-01"),
        #                                Time("2026-12-31"),
        #                                time_resolution=15 * u.min)

        #     for month in range(1, 13):
        #         region.plot_monthly_heatmap(2025, month, time_resolution=15 * u.min)
        # # #
        # break
