import numpy as np
from astropy.time import Time
from astropy.table import Table
import matplotlib.pyplot as plt
import healpy as hp
import matplotlib.gridspec as gridspec
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
import astropy.units as u
from collections import Counter

from qubic.lib.Instrument.Qacquisition import QubicAcquisition
from qubic.lib.Instrument.Qinstrument import QubicInstrument
from qubic.lib.Qdictionary import qubicDict
from qubic.lib.Qscene import QubicScene
from qubic.lib.Qsamplings import QubicSampling
from qubic.lib.Qhdf5 import HDF5Dict


class QubicScanScheduler:
    """
    Generates QUBIC telescope pointing schedules and command files.

    Usage follows three steps:
        1. Instantiate with observation metadata.
        2. Load the source track (from equatorial coordinates or from file).
        3. Run a scanning strategy to produce a QubicSampling and command file.

    Two scanning strategies are available for any source:

    ``schedule_elevation_tracking``
        Follows the source across the sky, binning its track into elevation
        strips and filling each strip with back-and-forth azimuth sweeps.

    ``schedule_block_scan``
        Divides the observation into fixed-duration constant-elevation blocks,
        stepping in elevation by a fixed amount between blocks.

    Parameters
    ----------
    date : str
        UTC start of the observation.
    source_name : str
        Label used in command files and plot titles.
    az_range : float, optional
        Full peak-to-peak azimuth sweep width [deg]. Default 40.0.

    Examples
    --------
    Galactic plane with elevation-tracking strategy::

        sched = QubicScanScheduler('2025-06-01', 'GalCenter', az_range=40.0)
        sched.load_source_from_coordinates(ra=266.4, dec=-29.0)
        sched.schedule_elevation_tracking(delta_el=2.0)
        sched.plot_schedule(title='GalCenter')
        sched.compute_coverage(plot=True, title='GalCenter')

    Moon with block-scan strategy::

        sched = QubicScanScheduler('2025-06-01', 'Moon', az_range=40.0)
        sched.load_source_from_file('moon_ephemeris.ecsv')
        sched.schedule_block_scan(block_duration=600, el_start=40.0, el_step=2.0)
    """

    # QUBIC site: Alto Chorrillos, Argentina
    _SITE = EarthLocation(lat=-24.183333333 * u.deg, lon=-66.466666667 * u.deg)

    # Fixed scan parameters
    _MAX_SPEED    = 1.0   # maximum azimuth scan speed [deg/s]
    _T_RAMP       = 1.0   # sech² ramp duration at each end of a half-sweep [s]
    _PAUSE        = 5.0   # pause at each turnaround [s]
    _EL_MIN       = 30.0  # minimum allowed telescope elevation [deg]
    _EL_MAX       = 70.0  # maximum allowed telescope elevation [deg]
    _DT_OUT       = 1.0   # output sampling cadence [s]
    _STEP_S       = 0.1   # internal integration step [s]
    _HWP_STEP     = 15    # HWP angular step size [deg]

    # ------------------------------------------------------------------ 
    #  1. Initialisation                                                   
    # ------------------------------------------------------------------ 	
    def __init__(self, date, source_name, az_range=40.0):
        self.date        = date
        self.source_name = source_name
        self.az_range    = az_range
        self.start_utc   = Time(date)

        # Set by load_source_from_* methods
        self._t_obs    = None   # time grid relative to obs start [s]
        self._az_track = None   # source azimuth at each sample [deg]
        self._el_track = None   # source elevation at each sample [deg]

        # Set by schedule_* methods
        self.samplings = None   # QubicSampling object
        self.commands  = []     # list of command strings


    # ------------------------------------------------------------------ 
    #  2. Source track loading                                             
    # ------------------------------------------------------------------ 
    def load_source_from_coordinates(self, ra, dec):
        """
        Compute the source Alt/Az track for a full day from equatorial coordinates.

        Parameters
        ----------
        ra : float
            Right ascension [deg, ICRS].
        dec : float
            Declination [deg, ICRS].
        """
        tgrid  = np.arange(0, 24 * 3600, 1.0) * u.s
        frame  = AltAz(obstime=self.start_utc + tgrid, location=self._SITE)
        source = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
        path   = source.transform_to(frame)

        self._t_obs    = tgrid.to_value(u.s)
        self._az_track = path.az.deg
        self._el_track = path.alt.deg

    def load_source_from_file(self, filepath):
        """
        Load a pre-computed Alt/Az ephemeris from an ECSV file.

        The file must contain columns ``time_utc``, ``azimuth``, and
        ``elevation``.

        Parameters
        ----------
        filepath : str
            Path to the ``.ecsv`` file.
        """
        table = Table.read(filepath, format='ascii.ecsv')

        self._t_obs    = (Time(table['time_utc']) - self.start_utc).sec
        self._az_track = np.array(table['azimuth'])
        self._el_track = np.array(table['elevation'])


    # ------------------------------------------------------------------ 
    #  3 Scanning strategy                        
    # ------------------------------------------------------------------ 
    def schedule_elevation_tracking(self, delta_el=2.0):
        """
        Schedule pointings using the elevation-tracking strategy.

        The source track is binned into elevation strips of height
        ``delta_el``.  Each strip is filled with back-and-forth azimuth
        sweeps centred on the mean source azimuth within that strip.
        If the source azimuth drifts too far within a strip, the strip
        is split into independent chunks.

        A command file named ``{date}_{source_name}_commands.txt`` is
        written automatically.

        Requires a source track to have been loaded first via
        ``load_source_from_coordinates`` or ``load_source_from_file``.

        Parameters
        ----------
        delta_el : float, optional
            Elevation strip height [deg]. Default 2.0.

        Returns
        -------
        samplings : QubicSampling
        """
        self._check_source_loaded()
        self.samplings = self._build_elevation_tracking(delta_el)
        return self.samplings


    def schedule_block_scan(self, block_duration, el_start, el_step):
        """
        Schedule pointings using the block-scan strategy.

        The observation window is divided into consecutive fixed-duration
        blocks.  Each block scans at a constant elevation centred on the
        mean source azimuth during that block.  The elevation advances by
        ``el_step`` after every block.

        Use a positive ``el_step`` to scan from low to high elevation,
        negative to scan from high to low.

        Requires a source track to have been loaded first via
        ``load_source_from_coordinates`` or ``load_source_from_file``.

        Parameters
        ----------
        block_duration : float
            Duration of each scan block [s].
        el_start : float
            Elevation of the first block [deg].
        el_step : float
            Elevation increment between blocks [deg].  Signed: positive
            steps upward, negative steps downward.
        output_file : str, optional
            If provided, command strings are written to this path.

        Returns
        -------
        samplings : QubicSampling
        """
        self._check_source_loaded()
        self.samplings = self._build_block_scan(block_duration, el_start, el_step)
        return self.samplings


    # ------------------------------------------------------------------ 
    #  4. Diagnostics                                                      
    # ------------------------------------------------------------------ 
    def plot_schedule(self, title=None):
        """
        Four-panel diagnostic plot of the pointing schedule.

        Panels
        ------
        1. Elevation vs. time — source track overlaid with telescope pointings.
        2. Azimuth vs. time   — source track overlaid with telescope pointings.
        3. Az vs El scatter
        4. Bar chart of samples per elevation bin.

        Requires ``schedule_elevation_tracking`` or ``schedule_block_scan``
        to have been called first.

        Parameters
        ----------
        title : str, optional
            File name prefix and figure title.  Defaults to ``source_name``.
        """
        self._check_scheduled()
        label = title or self.source_name

        t_out  = self.samplings.time
        az_out = self.samplings.azimuth
        el_out = self.samplings.elevation

        fig = plt.figure(figsize=(14, 10))
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

        # Panel 1 — elevation vs. time
        ax1 = fig.add_subplot(gs[0, 0])
        if self._t_obs is not None:
            ax1.plot(self._t_obs / 3600, self._el_track,
                     color='steelblue', lw=1.5, label='Source')
        ax1.scatter(t_out / 3600, el_out, s=1, color='tomato',
                    alpha=0.4, label='Telescope')
        ax1.set_xlabel('Time [hours]')
        ax1.set_ylabel('Elevation [deg]')
        ax1.set_title('Elevation vs. time')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Panel 2 — azimuth vs. time
        ax2 = fig.add_subplot(gs[0, 1])
        if self._t_obs is not None:
            ax2.plot(self._t_obs / 3600, self._az_track,
                     color='steelblue', lw=1.5, label='Source')
        ax2.scatter(t_out / 3600, az_out, s=1, color='tomato',
                    alpha=0.4, label='Telescope')
        ax2.set_xlabel('Time [hours]')
        ax2.set_ylabel('Azimuth [deg]')
        ax2.set_title('Azimuth vs. time')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Panel 3 — Az vs El, coloured by time
        ax3 = fig.add_subplot(gs[1, 0])
        sc  = ax3.scatter(az_out, el_out, c=t_out / 3600, s=1,
                          cmap='plasma', alpha=0.6)
        plt.colorbar(sc, ax=ax3, label='Time [hours]')
        ax3.set_xlabel('Azimuth [deg]')
        ax3.set_ylabel('Elevation [deg]')
        ax3.set_title('Telescope footprint (Az, El)')
        ax3.grid(True, alpha=0.3)

        # Panel 4 — samples per elevation bin
        ax4 = fig.add_subplot(gs[1, 1])
        counts    = Counter(np.round(el_out, 3))
        el_sorted = sorted(counts.keys())
        n_sorted  = [counts[e] for e in el_sorted]
        ax4.bar(el_sorted, n_sorted, width=0.7,
                color='steelblue', edgecolor='k', linewidth=0.4)
        ax4.set_xlabel('Elevation bin center [deg]')
        ax4.set_ylabel('Number of samples')
        ax4.set_title('Samples per elevation bin')
        ax4.grid(True, alpha=0.3, axis='y')

        plt.savefig(f'{label}_pointings.pdf', bbox_inches='tight')


    def compute_coverage(self, plot=False, title=None):
        """
        Compute the QUBIC sky coverage map and optionally plot it.

        Requires ``schedule_elevation_tracking`` or ``schedule_block_scan``
        to have been called first.

        Parameters
        ----------
        plot : bool, optional
            If True, plot a Mollweide coverage map in galactic coordinates
        title : str, optional
            File name prefix and figure title.  Defaults to ``source_name``.

        Returns
        -------
        coverage : numpy.ndarray
            HEALPix coverage map.
        seenpix : numpy.ndarray of bool
            Pixels with coverage > 10 % of the maximum.
        """
        self._check_scheduled()
        label = title or self.source_name

        d = qubicDict()
        d.read_from_file("td.dict")

        qubic    = QubicInstrument(d)
        scene    = QubicScene(d)
        acq      = QubicAcquisition(qubic, self.samplings, scene, d)
        coverage = acq.get_coverage()
        seenpix  = coverage > 0.1 * np.max(coverage)

        if plot:
            hp.mollview(coverage, title=f'Coverage {label} (Galactic)', cmap='Spectral_r')

            # Overlay celestial equator (Dec = 0°) in galactic projection
            ra_grid  = np.linspace(0, 360, 1000)
            eq       = SkyCoord(ra=ra_grid * u.deg,
                                dec=np.zeros_like(ra_grid) * u.deg,
                                frame='icrs').galactic
            l_eq     = np.radians(eq.l.wrap_at(360 * u.deg).deg)
            theta_eq = np.pi / 2 - np.radians(eq.b.deg)
            hp.projplot(theta_eq, l_eq, color='black', linewidth=1.5)

            # Overlay QUBIC zenith line (~Dec = -23°)
            zen      = SkyCoord(ra=ra_grid * u.deg,
                                dec=np.full_like(ra_grid, -23.0) * u.deg,
                                frame='icrs').galactic
            l_zen    = np.radians(zen.l.wrap_at(360 * u.deg).deg)
            theta_zen = np.pi / 2 - np.radians(zen.b.deg)
            hp.projplot(theta_zen, l_zen, color='white', linewidth=1.5)

            plt.savefig(f'{label}_coverage.pdf', bbox_inches='tight')

        return coverage, seenpix

    # ------------------------------------------------------------------ 
    #  5. Saving and loading samplings
    # ------------------------------------------------------------------ 
    def save_sampling(self, filename):
        """
        Save the current samplings to an HDF5 file.

        Parameters
        ----------
        filename : str
            Path to the output HDF5 file.
       """
        self._check_scheduled()

        data = {
            "azimuth":   np.array(self.samplings.azimuth),
            "elevation": np.array(self.samplings.elevation),
            "angle_hwp": np.array(self.samplings.angle_hwp),
            "time":      np.array(self.samplings.time),
            "fix_az":    bool(self.samplings.fix_az),
            "latitude":  float(self.samplings.latitude),
            "longitude": float(self.samplings.longitude),
            "date_obs":  self.samplings.date_obs.iso[0],
        }
        HDF5Dict().save_dict(filename, data)

    @staticmethod
    def load_sampling(filename):
        """
        Load samplings from an HDF5 file.

        Parameters
        ----------
        filename : str
            Path to the HDF5 file.

        Returns
        -------
        QubicSampling
        """
        data = HDF5Dict().load_dict(filename)

        return QubicSampling(
            azimuth   = data["azimuth"],
            elevation = data["elevation"],
            angle_hwp = data["angle_hwp"],
            time      = data["time"],
            date_obs  = data["date_obs"],
            longitude = data["longitude"],
            latitude  = data["latitude"],
            fix_az    = data["fix_az"],
        )


    # ------------------------------------------------------------------ 
    #  Private helpers                                                     
    # ------------------------------------------------------------------ 
    def _check_source_loaded(self):
        """Raise if no source track has been loaded yet."""
        if self._t_obs is None:
            raise RuntimeError(
                "No source track loaded. Call load_source_from_coordinates "
                "or load_source_from_file first.")

    def _check_scheduled(self):
        """Raise if no scheduling has been run yet."""
        if self.samplings is None:
            raise RuntimeError(
                "No schedule computed yet. Call schedule_elevation_tracking "
                "or schedule_block_scan first.")

    def _make_sweep_profile(self):
        """
        Build a single az-sweep displacement profile centred on zero.

        The profile consists of a forward half-sweep and a backward
        half-sweep, each shaped with sech² ramps at start and end for
        smooth acceleration / deceleration, separated by short pauses
        at the turnaround points.

        Returns
        -------
        sweep : ndarray
            Az offset from the sweep centre [deg], length = one full sweep.
        t_sweep : float
            Duration of one full sweep [s].
        """
        n_ramp = int(self._T_RAMP / self._STEP_S)
        x      = np.linspace(-3, 3, n_ramp)
        ramp   = 1.0 / np.cosh(x) ** 2
        ramp  *= self._MAX_SPEED / np.max(ramp)   # normalise to max speed

        ramp_dist = np.sum(ramp) * self._STEP_S
        flat_dist = self.az_range - 2 * ramp_dist
        if flat_dist < 0:
            raise ValueError(
                f"Ramp distance ({2*ramp_dist:.2f} deg) exceeds az_range "
                f"({self.az_range:.2f} deg). Reduce _T_RAMP or increase az_range.")

        n_flat  = int(flat_dist / self._MAX_SPEED / self._STEP_S)
        n_pause = int(self._PAUSE / self._STEP_S)

        # Half-sweep speed profile: ramp up → flat → ramp down
        half_speed = np.concatenate([ramp, np.full(n_flat, self._MAX_SPEED), ramp[::-1]])
        half_disp  = half_speed * self._STEP_S
        # Rescale so total displacement equals exactly az_range
        half_disp *= self.az_range / half_disp.sum()

        fwd       = np.cumsum(half_disp)
        bwd       = fwd[-1] - np.cumsum(half_disp)
        fwd_pause = np.full(n_pause, fwd[-1])
        bwd_pause = np.full(n_pause, bwd[-1])

        sweep   = np.concatenate([fwd, fwd_pause, bwd, bwd_pause])
        sweep  -= sweep.mean()           # centre on zero
        t_sweep = sweep.size * self._STEP_S

        return sweep, t_sweep

    def _generate_block_samples(self, az_center, el_center, t_start,
                                 n_sweeps, sweep, hwp_index):
        """
        Generate pointing time series for a single constant-elevation block.

        Parameters
        ----------
        az_center : float
            Central azimuth [deg].
        el_center : float
            Fixed elevation for the block [deg].
        t_start : float
            Block start time relative to the observation start [s].
        n_sweeps : int
            Number of complete sweeps to generate.
        sweep : ndarray
            Single-sweep displacement profile from ``_make_sweep_profile``.
        hwp_index : int
            Running HWP step counter (continues across blocks).

        Returns
        -------
        t_arr, az_arr, el_arr, hwp_arr : ndarray
        new_hwp_index : int
            Updated HWP counter after this block.
        """
        n_total = n_sweeps * sweep.size

        t_arr   = t_start + np.arange(n_total) * self._STEP_S
        az_arr  = np.tile(sweep, n_sweeps) + az_center
        el_arr  = np.full(n_total, el_center)

        # HWP angle: advances by _HWP_STEP after every complete sweep
        n_positions = int(90 / self._HWP_STEP) + 1
        hwp_arr = self._HWP_STEP * np.mod(
            np.arange(n_total) // sweep.size + hwp_index,
            n_positions
        )

        return t_arr, az_arr, el_arr, hwp_arr, hwp_index + n_sweeps

    def _assemble_sampling(self, all_t, all_az, all_el, all_hwp):
        """
        Downsample concatenated pointing arrays and return a QubicSampling.

        Parameters
        ----------
        all_t, all_az, all_el, all_hwp : list of ndarray
            Per-block arrays to concatenate.

        Returns
        -------
        samplings : QubicSampling
        """
        downsample = max(1, int(round(self._DT_OUT / self._STEP_S)))
        date_str   = self.start_utc.to_datetime().strftime('%Y-%m-%d')

        return QubicSampling(
            azimuth   = np.concatenate(all_az)[::downsample],
            elevation = np.concatenate(all_el)[::downsample],
            time      = np.concatenate(all_t)[::downsample],
            date_obs  = date_str,
            angle_hwp = np.concatenate(all_hwp)[::downsample].astype(int),
            longitude = self._SITE.lon.deg,
            latitude  = self._SITE.lat.deg,
            fix_az    = False,
        )


    def _build_elevation_tracking(self, delta_el):
        """
        Core implementation of the elevation-tracking strategy.

        Bins the source track into elevation strips and fills each strip
        with back-and-forth azimuth sweeps.  Within each strip, the
        source track is further split into chunks if the azimuth drifts
        by more than half the scan width minus 15, ensuring that the center 
        of the patch is visibile with the secondary peaks of the synthesized beam

        Parameters
        ----------
        delta_el : float
            Elevation strip height [deg].

        Returns
        -------
        samplings : QubicSampling
        """
        sweep, t_sweep = self._make_sweep_profile()

        bin_edges    = np.arange(self._EL_MIN, self._EL_MAX + delta_el, delta_el)
        max_az_drift = self.az_range / 2 - 15.0
        date_str     = self.start_utc.to_datetime().strftime('%Y-%m-%d')

        all_t, all_az, all_el, all_hwp = [], [], [], []
        self.commands = []
        hwp_index     = 0

        for i in range(len(bin_edges) - 1):
            el_low, el_high = bin_edges[i], bin_edges[i + 1]
            el_center       = 0.5 * (el_low + el_high)

            in_bin = (self._el_track >= el_low) & (self._el_track < el_high)
            if not np.any(in_bin):
                continue

            # Split continuous index runs into separate segments
            idx      = np.where(in_bin)[0]
            segments = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)

            for seg in segments:
                # Further split if azimuth drift within the segment is too large
                az_drift = np.abs(self._az_track[seg] - self._az_track[seg[0]])
                split_at = np.where(np.diff(az_drift // max_az_drift) > 0)[0] + 1
                chunks   = np.split(seg, split_at)

                for chunk in chunks:
                    t_enter  = self._t_obs[chunk[0]]
                    duration = self._t_obs[chunk[-1]] - t_enter
                    n_sweeps = int(duration / t_sweep)
                    if n_sweeps == 0:
                        continue

                    az_center = np.mean(self._az_track[chunk])
                    t_arr, az_arr, el_arr, hwp_arr, hwp_index = self._generate_block_samples(az_center, el_center, t_enter, n_sweeps, sweep, hwp_index)

                    # Build command string for this block
                    t_abs   = self.start_utc + t_enter * u.s
                    t_str   = t_abs.to_datetime().strftime('%Y-%m-%dT%H:%M:%S')
                    el_label = f"{el_low:.0f}_{el_high:.0f}"
                    self.commands.append((t_enter,
                        f"# {t_str} UTC\n"
                        f"do_constant_elevation_scanning.py "
                        f"el={el_center:.1f} "
                        f"azmin={az_center - self.az_range / 2:.1f} "
                        f"azmax={az_center + self.az_range / 2:.1f} "
                        f"tstart={t_str} "
                        f"duration={int(round(duration))} "
                        f"title='{self.source_name}_el{el_label}'\n"
                    ))

                    all_t.append(t_arr)
                    all_az.append(az_arr)
                    all_el.append(el_arr)
                    all_hwp.append(hwp_arr)

        if not all_t:
            raise RuntimeError(
                "No complete sweeps could be scheduled. Check that the source "
                "track crosses the elevation range ")

        # Write command file sorted by start time
        fname = f"{date_str}_{self.source_name}_commands.txt"
        with open(fname, 'w') as f:
            for _, cmd in sorted(self.commands):
                f.write(cmd)
        print(f"{len(self.commands)} commands written to '{fname}'")

        return self._assemble_sampling(all_t, all_az, all_el, all_hwp)


    def _build_block_scan(self, block_duration, el_start, el_step):
        """
        Core implementation of the block-scan strategy.

        Divides the source track window into consecutive fixed-duration blocks.
        Each block scans at a constant elevation equal to ``el_start +
        n * el_step``, centred on the mean source azimuth in that block.

        Parameters
        ----------
        block_duration : float
            Duration of each block [s].
        el_start : float
            Elevation of the first block [deg].
        el_step : float
            Signed elevation increment per block [deg].

        Returns
        -------
        samplings : QubicSampling
        """
        sweep, t_sweep = self._make_sweep_profile()
        
        date_str     = self.start_utc.to_datetime().strftime('%Y-%m-%d')
        t_obs_unix   = self.start_utc.unix + self._t_obs
        t_unix_start = self.start_utc.unix + self._t_obs[0]
        t_unix_end   = self.start_utc.unix + self._t_obs[-1]

        t_block_start = t_unix_start
        el_current    = el_start

        all_t, all_az, all_el, all_hwp = [], [], [], []
        self.commands = []
        hwp_index     = 0

        while t_block_start <= t_unix_end:
            t_block_end  = t_block_start + block_duration
            block_mask   = (t_obs_unix >= t_block_start) & (t_obs_unix < t_block_end)


            az_center = np.mean(self._az_track[block_mask])
            n_sweeps  = int(block_duration / t_sweep)

            if n_sweeps > 0:
                # Convert block start to seconds relative to obs start
                t_rel = t_block_start - self.start_utc.unix
                t_arr, az_arr, el_arr, hwp_arr, hwp_index = self._generate_block_samples(az_center, el_current,
                t_rel, n_sweeps, sweep, hwp_index)
                all_t.append(t_arr); all_az.append(az_arr)
                all_el.append(el_arr); all_hwp.append(hwp_arr)

            # Build command string
            tstart_str = Time(t_block_start, format='unix').isot.split('.')[0]
            self.commands.append(
				f"do_constant_elevation_scanning.py"
				f" Tbath=0.318"
				f" tstart={tstart_str}"
				f" duration={int(block_duration)}"
				f" azmin={az_center - self.az_range / 2:.1f}"
				f" azmax={az_center + self.az_range / 2:.1f}"
				f" el={el_current:.1f}"
				f" title={self.source_name}_scans\n"
			)

            el_current += el_step
            t_block_start = t_block_end

        if not all_t:
            raise RuntimeError(
                "No complete sweeps could be scheduled for block scan.")

        fname = f"{date_str}_{self.source_name}_commands.txt"
        with open(fname, 'w') as f:
            f.writelines(self.commands)
        print(f"{len(self.commands)} scan blocks written to '{fname}'")

        return self._assemble_sampling(all_t, all_az, all_el, all_hwp)
