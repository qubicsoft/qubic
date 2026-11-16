
"""
This merged script integrates:
- Simulation of telescope azimuth/altitude sweeps with ramps and pauses
- Observation planning 
- Diagnostic plotting of pointing data, including azimuth vs. time, elevation vs. time, azimuth vs. elevation, azimuth speed vs. time, and RA/DEC

This final version uses the user‐specified sample_params for all QubicObservation calls,
and it saves all plots and tables to disk (under per‐source subdirectories) without interactive display.
"""

from __future__ import annotations
import os
import calendar as cal
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import astropy.units as u
from astropy.time import Time
from astropy.table import Table, QTable
from astropy.coordinates import (
    SkyCoord,
    get_body,
    EarthLocation,
    AltAz,
    ICRS,
)

from pytz import timezone
from astroplan.plots import plot_sky
from astroplan import (
    Observer,
    time_grid_from_range,
    AltitudeConstraint,
    AirmassConstraint,
)

from constraint import SunSeparationConstraint, MoonSeparationConstraint


# =============================================================================
# User‐specified parameters for all QubicObservation instances:
# =============================================================================
sample_params = {
    "latitude": -24.1844,            # degrees
    "longitude": -66.8714,           # degrees
    "delta_az": 20.0,                # degrees half‐span
    "nsweeps_per_elevation": 25,     # sweeps per elevation block
    "period": 1,                     # seconds (read‐out period)
    # 'angspeed' is optional; default to 1.0 deg/s if not provided
}


# =============================================================================
# Section 1: QubicObservation (telescope motion simulator)
# =============================================================================

class QubicObservation:
    def __init__(self, d: dict, hor_down=30, hor_up=70):
        # --- Site & horizon cuts ---
        self.earth_location = EarthLocation(lat=d["latitude"]*u.deg,
                                            lon=d["longitude"]*u.deg)
        self.utc_offset    = -3 * u.hour
        self.hor_down      = hor_down * u.deg
        self.hor_up        = hor_up   * u.deg

        # --- Pointing center (galactic) ---
        self.eq  = SkyCoord(ra=d["RA_center"]*u.deg,
                            dec=d["DEC_center"]*u.deg,
                            frame="icrs")
        self.gal = self.eq.galactic

        # --- Observation params ---
        self.date_obs = Time(d["date_obs"])
        self.duration = d["duration"] * u.hour

        # ** Use the user‐supplied speed for BOTH azimuth and elevation **
        self.ang_speed = d.get("angspeed", 1.0) * (u.deg/u.s)      # deg/s
        self.max_speed = self.ang_speed.to_value(u.deg/u.s)       # numeric

        self.delta_az  = d["delta_az"] * u.deg
        self.nsweeps_per_elevation = d["nsweeps_per_elevation"]
        self.period    = d["period"] * u.s
        self.nsweep_even = (self.nsweeps_per_elevation % 2 == 0)

        # --- Sweep‐shape parameters ---
        self.step_s        = 0.1    # sampling dt in seconds
        self.t_ramp        = 1.0    # not used for elevation slew
        self.pause_duration= 3.0    # pause at end of each azimuth sweep

        # build **one** speed profile for azimuth sweeps, using self.max_speed
        n_ramp = int(self.t_ramp / self.step_s)
        x = np.linspace(-3, 3, n_ramp)
        ramp = 1/(np.cosh(x)**2)
        ramp *= self.max_speed/np.max(ramp)
        ramp_distance = np.sum(ramp)*self.step_s

        flat_dist = 2*self.delta_az.to_value(u.deg) - 2*ramp_distance
        if flat_dist < 0:
            raise ValueError("Δaz too small for given accel ramp")
        t_flat = flat_dist/self.max_speed
        n_flat = int(t_flat/self.step_s)
        n_pause = int(self.pause_duration/self.step_s)

        flat = np.ones(n_flat)*self.max_speed
        self.speed_prof = np.concatenate([ramp, flat, ramp[::-1], np.zeros(n_pause)])
        # per‐step azimuth change in deg
        az_steps = self.speed_prof * self.step_s
        az_steps *= (2*self.delta_az.to_value(u.deg))/np.sum(az_steps)
        self.az_step = az_steps

        # total time of *one* azimuth sweep at constant elevation
        self.t_sweep = self.speed_prof.size * self.step_s * u.s

    def get_centers(self):
        # build a dense 1-second grid and transform the galactic center into AltAz
        start = self.date_obs - self.utc_offset
        tgrid = np.arange(0, self.duration.to_value(u.s), 1)*u.s
        frame = AltAz(obstime=start + tgrid, location=self.earth_location)
        path  = self.gal.transform_to(frame)
        # apply horizon cuts
        visible = path[(path.alt>self.hor_down)&(path.alt<self.hor_up)]
        if len(visible)==0:
            raise ValueError("Source never visible in the given elevation range")

        # break into contiguous intervals and pick centers every t_sweep*nsweeps
        unix = visible.obstime.unix
        gaps = np.where(np.diff(unix)>1)[0]
        ends = np.append(gaps, len(unix)-1)
        starts = np.append(0, gaps+1)
        centers = []
        for s,e in zip(starts, ends):
            dur = unix[e]-unix[s]
            dt_centers = (self.t_sweep * self.nsweeps_per_elevation).to_value(u.s)
            offs = np.arange(0, dur, dt_centers)*u.s
            base_time = visible.obstime[s]
            for off in offs:
                centers.append(AltAz(az=0*u.deg,   # placeholder
                                      alt=visible.alt[s],
                                      obstime=base_time+off,
                                      location=self.earth_location))
        # now fill the correct az/alt from the galactic track
        # by re‐transforming each center time
        return SkyCoord(self.gal.l, self.gal.b, frame="galactic").transform_to(
            AltAz(obstime=[c.obstime for c in centers],
                  location=self.earth_location)
        )

    def AzimuthSweep(self, center: AltAz, idx: int) -> np.ndarray:
        # build the zig-zag of azimuth angles
        seqs = []
        half = np.cumsum(self.az_step)
        L = self.delta_az.to_value(u.deg)
        for i in range(self.nsweeps_per_elevation):
            if self.nsweep_even or (i%2==0):
                seq = center.az.value - L + half
            else:
                seq = center.az.value + L - half
            seqs.append(seq)
        AZ = np.concatenate(seqs)

        # build the timestamp array for that sweep
        t0 = center.obstime - self.t_sweep/2
        times = t0 + np.arange(len(AZ))*self.step_s*u.s

        # assemble and down‐sample by period
        data = np.empty((len(AZ),3),object)
        data[:,0]=times; data[:,1]=AZ; data[:,2]=center.alt.value
        keep = np.arange(len(AZ))%(int(self.period.to_value(u.s)/self.step_s))==0
        return data[keep]

    def get_pointing(self) -> AltAz | None:
        try:
            centers = self.get_centers()
        except ValueError:
            return None

        segments = []
        shift = 0*u.s

        for i, C in enumerate(centers):
            # time‐shifted center
            C0 = AltAz(az=C.az, alt=C.alt,
                       obstime=C.obstime+shift,
                       location=self.earth_location)

            # 1) azimuth block
            blk = self.AzimuthSweep(C0, i)
            segments.append(blk)

            # 2) insert elevation slew to next center
            
    # … inside get_pointing(), after you build `blk` …

        if i < len(centers) - 1:
            # get current & next elevation in degrees
            alt0 = C.alt.to_value(u.deg)
            alt1 = centers[i+1].alt.to_value(u.deg)
            delta_alt = alt1 - alt0

            # total slew time at self.ang_speed (deg/s)
            slew_time = abs(delta_alt) * u.deg / self.ang_speed  # → seconds

            # compute ideal number of samples at step_s
            raw_steps = (slew_time / self.step_s).decompose().value
            n_steps = int(raw_steps)

            # if there’s any delta, force at least 2 steps
            if delta_alt != 0:
                n_steps = max(n_steps, 2)

            if n_steps > 0:
                last_time = blk[-1, 0]  # timestamp of last az‐block point
                last_az   = blk[-1, 1]  # its azimuth

                # build n_steps+1 evenly spaced times from 0→slew_time, drop t=0
                slew_offsets = np.linspace(0, slew_time.to_value(u.s), n_steps+1)[1:] * u.s
                slew_times   = last_time + slew_offsets

                # linearly interpolate elevation from alt0→alt1
                slew_alts    = np.linspace(alt0, alt1, n_steps)

                # hold az fixed during elevation slew
                slew_azs     = np.full(n_steps, last_az)

                ramp_arr = np.empty((n_steps, 3), object)
                ramp_arr[:,0] = slew_times
                ramp_arr[:,1] = slew_azs
                ramp_arr[:,2] = slew_alts
                segments.append(ramp_arr)

            # push all subsequent blocks forward by the slew duration
            cumulative_shift += slew_time


        P = np.vstack(segments)
        return AltAz(az=P[:,1]*u.deg,
                     alt=P[:,2]*u.deg,
                     obstime=P[:,0],
                     location=self.earth_location)












# =============================================================================
# Section 2: Observation Planning (sofia_plan) + Diagnostic plotting
# =============================================================================

class Source(ABC):
    """
    Abstract base class for point or extended sources. Builds a time grid,
    evaluates constraints, plots heatmaps, and (in write_trajectory) calls
    QubicObservation to generate real telescope motion trajectories.
    """

    def __init__(
        self,
        name: str,
        qubic_site: Observer,
        obs_time: Time,
        constraints: list,
        time_resolution: u.Quantity,
        results_dir: str = ".",
    ):
        self.name = name
        self.qubic_site = qubic_site
        self.obs_time = obs_time
        self.time_resolution = time_resolution
        self.constraints = constraints
        self.results_dir = results_dir

        self.coord: SkyCoord | None = None
        self.time_grid: list[Time] = []
        self.constraints_grid: np.ndarray = np.array([])
        self.valid_time_intervals: list[list[Time]] = []
        self.plots_dir: str = "."

    def configure(self):
        """Ensure result directories exist."""
        os.makedirs(self.results_dir, exist_ok=True)
        self.plots_dir = os.path.join(self.results_dir, f"{self.name}_plots")
        os.makedirs(self.plots_dir, exist_ok=True)

    def compute_time_grid(self):
        """
        Build a linearly spaced time grid from midnight of obs_time (UTC)
        to 23:59:59 of that UTC day, stepping by time_resolution.
        """
        if self.time_grid:
            return

        start = self.obs_time
        end = start + 1 * u.day - 1 * u.microsecond
        self.time_grid = time_grid_from_range([start, end], time_resolution=self.time_resolution)

    def compute_valid_times_from_constraints(self):
        """
        From constraints_grid (n_constraints × n_time_slots), find continuous intervals 
        where all constraints == True. Populate valid_time_intervals with [start, end] pairs.
        """
        if self.constraints_grid.size == 0:
            self.evaluate_constraints()

        mask = np.all(self.constraints_grid == 1, axis=0)
        idx_true = np.where(mask)[0]
        if idx_true.size == 0:
            self.valid_time_intervals = []
            return

        gaps = np.where(np.diff(idx_true) > 1)[0]
        runs = np.split(idx_true, gaps + 1)

        intervals: list[list[Time]] = []
        for r in runs:
            start_idx = r[0]
            end_idx = min(r[-1] + 1, len(self.time_grid) - 1)
            intervals.append([self.time_grid[start_idx], self.time_grid[end_idx]])

        # If last interval’s start == end, extend it by one resolution step
        if intervals and (intervals[-1][0] == intervals[-1][1]):
            intervals[-1][1] += self.time_resolution - 1 * u.s

        self.valid_time_intervals = intervals

    def _plot_constraint_grid(self, grid: np.ndarray, make_plot: bool = False):
        """
        Display a heatmap (constraints × time_grid) showing True/False.
        If make_plot=True, save to disk (under self.plots_dir).
        """
        if not make_plot:
            return

        extent = (-0.5, len(self.time_grid) - 0.5, -0.5, len(self.constraints) - 0.5)
        fig, ax = plt.subplots(figsize=(14, 6), tight_layout=True)
        fig.suptitle(f"Constraint grid: {self.name} – {self.obs_time.iso}")

        cmap = ListedColormap(["#D95F02", "#1B9E77"])
        ax.imshow(grid, extent=extent, origin="lower", cmap=cmap, vmin=0, vmax=1)

        ax.set_xticks(range(len(self.time_grid)))
        ax.set_xticklabels(
            [t.datetime.strftime("%H:%M") for t in self.time_grid],
            rotation=30,
            ha="right",
        )
        ax.set_xlabel("UTC Time")

        ax.set_yticks(range(len(self.constraints)))
        ax.set_yticklabels([c.__class__.__name__ for c in self.constraints])

        ax.set_xticks(np.arange(-0.5, len(self.time_grid)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(self.constraints)), minor=True)
        ax.grid(which="minor", color="#333333", linewidth=1)
        ax.set_aspect("equal")

        out_path = os.path.join(self.plots_dir, f"{self.name}_constraint_grid.png")
        plt.savefig(out_path, dpi=600)
        plt.close(fig)

    @abstractmethod
    def evaluate_constraints(self, make_plot: bool = False):
        ...

    @abstractmethod
    def plot_trajectory(self, loc_time_resolution: u.Quantity, make_plot: bool = False):
        ...

    @classmethod
    @abstractmethod
    def load_sources(
        cls,
        qubic_site: Observer,
        obstime: Time,
        constraints: list,
        time_resolution: u.Quantity,
        source_file: str,
        radius: u.Quantity = 10 * u.deg,
        radial_samples: int = 50,
        polar_angle_samples: int = 100,
        results_dir: str = ".",
    ) -> list[Source]:
        ...

    def write_trajectory(self, dest: Optional[str] = None) -> str:
        """
        For each valid time interval, instantiate QubicObservation (using sample_params)
        to simulate the telescope’s actual pointing (Az/Alt) over that interval.
        Transform to ICRS to get RA/DEC, then write out a QTable
        (time_utc, time_local, ra, dec, elevation, azimuth) to an ECSV file.
        Returns the last filename.
        """
        if not self.valid_time_intervals:
            self.compute_valid_times_from_constraints()
        if not self.valid_time_intervals:
            return ""

        if not self.time_grid:
            self.compute_time_grid()
        if self.coord is None:
            raise RuntimeError("Coordinates not initialized: load the source first")

        # Determine a single, static RA/DEC for the patch center:
        if self.coord.isscalar:
            ra0 = self.coord.icrs.ra.to_value(u.deg)
            dec0 = self.coord.icrs.dec.to_value(u.deg)
        else:
            # take the first element if coord is an array
            ra0 = self.coord.icrs.ra[0].to_value(u.deg)
            dec0 = self.coord.icrs.dec[0].to_value(u.deg)

        folder = dest or self.plots_dir
        os.makedirs(folder, exist_ok=True)

        last_saved = ""
        for start, stop in self.valid_time_intervals:
            # Build dictionary `d` for QubicObservation using sample_params:
            d = {
                "latitude": sample_params["latitude"],
                "longitude": sample_params["longitude"],
                "RA_center": ra0,
                "DEC_center": dec0,
                "date_obs": start.iso,
                "duration": (stop - start).to_value(u.hour),
                "angspeed": sample_params.get("angspeed", 0.22),
                "delta_az": sample_params["delta_az"],
                "nsweeps_per_elevation": sample_params["nsweeps_per_elevation"],
                "period": sample_params["period"],
            }

            qubic_obs = QubicObservation(d)
            pointing = qubic_obs.get_pointing()

            if pointing is None:
                print(f"  → skipping {self.name} for {start.iso}–{stop.iso} (never in elevation range)")
                continue


            tw = pointing.obstime
            # Convert time_local → ISO‐8601 strings to avoid JSON serialization errors
            times_local_iso = [
                t.to_datetime(timezone=self.qubic_site.timezone).isoformat() for t in tw
            ]

            elevations = pointing.alt.to_value(u.deg)
            azimuths = pointing.az.to_value(u.deg)

            icrs_coords = pointing.transform_to(ICRS())
            ras = icrs_coords.ra.to_value(u.deg)
            decs = icrs_coords.dec.to_value(u.deg)

            # Build a QTable with 6 columns:
            # time_utc (Astropy Time), time_local (ISO string), ra, dec, elevation, azimuth
            table = QTable(
                [tw, times_local_iso, ras, decs, elevations, azimuths],
                names=("time_utc", "time_local", "ra", "dec", "elevation", "azimuth"),
            )
            fname = f"{self.name}-{start.strftime('%Y%m%d_%H%M')}-{stop.strftime('%H%M')}.ecsv"
            savepath = os.path.join(folder, fname)
            table.write(savepath, format="ascii.ecsv", overwrite=True)
            print(f"Wrote trajectory for {self.name} from {start.iso} to {stop.iso} → {savepath}")
            last_saved = savepath

        return last_saved


class PointSource(Source):
    @classmethod
    def load_sources(*args, **kwargs):
        raise NotImplementedError("PointSource.load_sources not implemented")

    def load_point_source(self):
        if not self.time_grid:
            self.compute_time_grid()
        self.coord = get_body(self.name, time=self.time_grid, location=self.qubic_site.location)

    def evaluate_constraints(self, make_plot: bool = False):
        """
        Evaluate each constraint on the time_grid for a point source.
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
        Plot the sky trajectory of the point source and the Sun during valid intervals.
        Saved as PNG in each source’s `plots_dir`.
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
            source_altaz = get_body(self.name, time=tw, location=self.qubic_site.location)
            sun_altaz = get_body("sun", time=tw, location=self.qubic_site.location)

            fig = plt.figure(figsize=(6, 6))
            plot_sky(
                source_altaz,
                self.qubic_site,
                tw,
                style_kwargs={
                    "marker": "o",
                    "label": f"{self.name}: {start.datetime.strftime('%H:%M')}-{stop.datetime.strftime('%H:%M')}",
                },
                north_to_east_ccw=False,
            )
            plot_sky(
                sun_altaz,
                self.qubic_site,
                tw,
                style_kwargs={"marker": "*", "color": "gold", "label": "Sun" if add_label else ""},
                north_to_east_ccw=False,
            )
            add_label = False

            plt.legend(loc="upper right", bbox_to_anchor=(1.50, 1))
            fig.suptitle(f"Polar plot {self.name} – {self.obs_time.iso}")
            fig.tight_layout()
            plt.gca().set_facecolor("whitesmoke")

            if make_plot:
                out_path = os.path.join(
                    self.plots_dir, f"{self.name}_polar_{start.strftime('%H%M')}_{stop.strftime('%H%M')}.png"
                )
                plt.savefig(out_path, dpi=600)
            plt.close(fig)


class ExtendedSource(Source):
    def __init__(
        self,
        name: str,
        coord: SkyCoord,
        region_coord: SkyCoord,
        qubic_site: Observer,
        obs_time: Time,
        constraints: list,
        results_dir: str = ".",
        time_resolution: u.Quantity = 1 * u.h,
        radius: u.Quantity = 10 * u.deg,
        radial_samples: int = 50,
        polar_angle_samples: int = 100,
    ):
        super().__init__(name, qubic_site, obs_time, constraints, time_resolution, results_dir)
        self.coord = coord
        self.region_coord = region_coord
        self.radius = radius
        self.radial_samples = radial_samples
        self.polar_angle_samples = polar_angle_samples

    @classmethod
    def load_sources(
        cls,
        qubic_site: Observer,
        obstime: Time,
        constraints: list,
        time_resolution: u.Quantity,
        source_file: str,
        radius: u.Quantity = 10 * u.deg,
        radial_samples: int = 50,
        polar_angle_samples: int = 100,
        results_dir: str = ".",
    ) -> list[ExtendedSource]:
        """
        Read sky_regions.ecsv (with columns 'name','l_deg','b_deg') and return
        a list of ExtendedSource instances, each sampling a disk of radius
        around the center coordinate.
        """
        if not os.path.isfile(source_file):
            raise FileNotFoundError(source_file)

        sky_regions = Table.read(source_file, format="ascii.ecsv")
        pas = np.linspace(0, 360, polar_angle_samples) * u.deg
        seps = np.full_like(pas, radius)

        result: list[ExtendedSource] = []
        for row in sky_regions:
            name = row["name"]
            center_icrs = (
                SkyCoord(row["l_deg"], row["b_deg"], unit=u.deg, frame="galactic")
                .transform_to("icrs")
            )
            disk_icrs = center_icrs.directional_offset_by(position_angle=pas, separation=seps)
            result.append(
                cls(
                    name,
                    center_icrs,
                    disk_icrs,
                    qubic_site,
                    obstime,
                    constraints,
                    results_dir,
                    time_resolution,
                    radius,
                    radial_samples,
                    polar_angle_samples,
                )
            )
        return result

    def evaluate_constraints(self, make_plot: bool = False):
        """
        Evaluate each constraint for the extended source over its time grid.
        The constraint functions accept (observer, targets, times, grid_times_targets=True).
        """
        if not self.time_grid:
            self.compute_time_grid()

        grid = np.zeros((len(self.constraints), len(self.time_grid)))
        for i, c in enumerate(self.constraints):
            cond = c(
                times=self.time_grid,
                observer=self.qubic_site,
                targets=self.region_coord,
                grid_times_targets=True,
            )
            if cond.ndim == 2:
                cond = np.all(cond, axis=0)
            grid[i, :] = cond
        self.constraints_grid = grid
        self._plot_constraint_grid(grid, make_plot)

    def plot_trajectory(self, loc_time_resolution: u.Quantity, make_plot: bool = False):
        """
        Plot altitudes of region center, "top" (+radius), "bottom" (−radius), Sun, Moon
        during valid intervals. Saved as PNG in self.plots_dir.
        """
        if not self.valid_time_intervals:
            self.compute_valid_times_from_constraints()
        if not self.valid_time_intervals:
            return

        add_label = True
        for start, stop in self.valid_time_intervals:
            tw = time_grid_from_range([start, stop], time_resolution=loc_time_resolution)
            altaz = AltAz(obstime=tw, location=self.qubic_site.location)

            center_altaz = self.coord.transform_to(altaz)
            offset = self.radius
            top_altaz = SkyCoord(
                alt=np.minimum(center_altaz.alt + offset, 90 * u.deg),
                az=center_altaz.az,
                frame=altaz,
            )
            bottom_altaz = SkyCoord(
                alt=np.maximum(center_altaz.alt - offset, -90 * u.deg),
                az=center_altaz.az,
                frame=altaz,
            )

            sun = get_body("sun", time=tw, location=self.qubic_site.location).transform_to(altaz)
            moon = get_body("moon", time=tw, location=self.qubic_site.location).transform_to(altaz)

            fig = plt.figure(figsize=(6, 6))
            plot_sky(
                center_altaz,
                self.qubic_site,
                tw,
                style_kwargs={"marker": "o", "label": f"Center: {start.datetime.strftime('%H:%M')}-{stop.datetime.strftime('%H:%M')}"},
                north_to_east_ccw=False,
            )
            plot_sky(
                top_altaz, self.qubic_site, tw, style_kwargs={"marker": "^", "label": f"Top (+{offset})" if add_label else ""}, north_to_east_ccw=False
            )
            plot_sky(
                bottom_altaz, self.qubic_site, tw, style_kwargs={"marker": "v", "label": f"Bottom (–{offset})" if add_label else ""}, north_to_east_ccw=False
            )
            plot_sky(
                sun, self.qubic_site, tw, style_kwargs={"marker": "*", "color": "gold", "label": "Sun" if add_label else ""}, north_to_east_ccw=False
            )
            plot_sky(
                moon, self.qubic_site, tw, style_kwargs={"marker": "+", "color": "red", "label": "Moon" if add_label else ""}, north_to_east_ccw=False
            )

            add_label = False

            plt.legend(loc="upper right", bbox_to_anchor=(1.60, 1))
            fig.suptitle(f"Polar plot {self.name} – {self.obs_time.iso}")
            fig.tight_layout()
            plt.gca().set_facecolor("whitesmoke")

            if make_plot:
                out_path = os.path.join(
                    self.plots_dir,
                    f"{self.name}_polar_{start.strftime('%H%M')}_{stop.strftime('%H%M')}.png",
                )
                plt.savefig(out_path, dpi=600)
            plt.close(fig)


# =============================================================================
# Section 3: Diagnostic plotting utility
# =============================================================================

def plot_trajectory_diagnostics(traj_file: str, site_location: EarthLocation):
    """
    Load a trajectory ECSV file and plot:
      1) Azimuth vs Time
      2) Elevation vs Time
      3) Azimuth vs Elevation (colored by time)
      4) Azimuth Speed vs Time
      5) RA vs DEC

    All plots are saved as: <traj_basename>_diagnostics.png
    """
    print(f"Generating diagnostic plots for {traj_file}")
    table = QTable.read(traj_file, format="ascii.ecsv")
    time = Time(table["time_utc"])
    az_deg = table["azimuth"]
    el_deg = table["elevation"]

    altaz = AltAz(
        az=az_deg * u.deg,
        alt=el_deg * u.deg,
        obstime=time,
        location=site_location,
    )

    # Time axes
    t_h = (time - time[0]).to_value(u.hour)
    t_s = (time - time[0]).sec

    # Az speed (handle wrap at 360°)
    daz = np.diff(az_deg)
    daz = np.where(daz > 180, daz - 360, daz)
    daz = np.where(daz < -180, daz + 360, daz)
    dt = np.diff(t_s)
    az_speed = np.abs(daz / dt)
    az_speed = np.insert(az_speed, 0, 0.0)

    # Compute RA/DEC by transforming AltAz → ICRS
    icrs = altaz.transform_to(ICRS())
    ra_deg = icrs.ra.to_value(u.deg)
    dec_deg = icrs.dec.to_value(u.deg)

    fig = plt.figure(figsize=(12, 10))

    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(t_h, az_deg, "b-")
    ax1.set_xlabel("Time (h)")
    ax1.set_ylabel("Azimuth (°)")
    ax1.set_title("Azimuth vs Time")
    ax1.grid()

    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(t_h, el_deg, "r-")
    ax2.set_xlabel("Time (h)")
    ax2.set_ylabel("Elevation (°)")
    ax2.set_title("Elevation vs Time")
    ax2.grid()

    ax3 = fig.add_subplot(3, 2, 3)
    sc = ax3.scatter(az_deg, el_deg, c=t_h, cmap="viridis", s=10)
    plt.colorbar(sc, ax=ax3, label="Time (h)")
    ax3.set_xlabel("Azimuth (°)")
    ax3.set_ylabel("Elevation (°)")
    ax3.set_title("Az vs El")
    ax3.grid()

    ax4 = fig.add_subplot(3, 2, 4)
    ax4.plot(t_h, az_speed, "g-")
    ax4.set_xlabel("Time (h)")
    ax4.set_ylabel("Az speed (°/s)")
    ax4.set_title("Azimuth Speed vs Time")
    ax4.grid()

    ax5 = fig.add_subplot(3, 2, 5)
    ax5.plot(ra_deg, dec_deg, ".", color="purple", markersize=2)
    ax5.set_xlabel("RA (°)")
    ax5.set_ylabel("DEC (°)")
    ax5.set_title("RA vs DEC")
    ax5.grid()

    plt.tight_layout()
    base = os.path.splitext(traj_file)[0]
    out_path = base + "_diagnostics.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


# =============================================================================
# Section 4: Execution in __main__
# =============================================================================

if __name__ == "__main__":
    # 1) Planning parameters
    time_resolution = 30 * u.min
    point_source_constraints = [
        AltitudeConstraint(35 * u.deg, 85 * u.deg),
        AirmassConstraint(3),
        SunSeparationConstraint(min=50 * u.deg),
    ]
    extended_source_constraints = point_source_constraints + [
        MoonSeparationConstraint(min=30 * u.deg)
    ]

    # 2) Define the observatory site (astroplan.Observer)
    location = EarthLocation.from_geodetic(
        lon=sample_params["longitude"] * u.deg,
        lat=sample_params["latitude"] * u.deg,
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

    # 3) Loop over a Month (days 24 → end)
    month = 6
    year = 2025
    n_days = cal.monthrange(year, month)[1]
    extended_source_path = "sky_regions.ecsv"  # must contain 4 regions

    # Change the range for more days 

    for day in range(29, n_days + 1):
        obs_time = Time(f"{year}-{month:02d}-{day:02d}T00:00:00", scale="utc")
        result_dir = os.path.join("plots_trajectories", obs_time.strftime("%Y%m%d"))

        # a) Moon as a PointSource
        moon_source = PointSource(
            name="moon",
            qubic_site=qubic_site,
            obs_time=obs_time,
            constraints=point_source_constraints,
            time_resolution=time_resolution,
            results_dir=result_dir,
        )

        # b) Extended sources from sky_regions.ecsv
        regions = ExtendedSource.load_sources(
            qubic_site=qubic_site,
            obstime=obs_time,
            constraints=extended_source_constraints,
            time_resolution=time_resolution,
            source_file=extended_source_path,
            radius=14 * u.deg,
            results_dir=result_dir,
        )

        # c) For each source: plan, plot, write trajectory, diagnostics
        for source in regions + [moon_source]:
            source.configure()
            source.evaluate_constraints(make_plot=True)
            source.plot_trajectory(loc_time_resolution=20 * u.min, make_plot=True)
            traj_file = source.write_trajectory()

            if traj_file:
                plot_trajectory_diagnostics(traj_file, qubic_site.location)
