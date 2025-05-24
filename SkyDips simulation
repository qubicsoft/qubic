import numpy as np
import pandas as pd
import warnings
from astropy import units as u
from astropy.coordinates import SkyCoord, AltAz, EarthLocation, get_sun
from astropy.time import Time
import matplotlib.pyplot as plt


class QubicObservation:

    def __init__(self, d, hor_down=30, hor_up=70, sun_sep=20):
        self.earth_location = EarthLocation(lat=d['latitude'] * u.deg,
                                            lon=d['longitude'] * u.deg)
        self.utc_offset = -3 * u.hour
        self.hor_down = hor_down * u.deg
        self.hor_up = hor_up * u.deg
        self.sun_sep = sun_sep * u.deg

        self.eq = SkyCoord(ra=d['RA_center'] * u.deg,
                           dec=d['DEC_center'] * u.deg, frame='icrs')
        self.gal = self.eq.galactic

        self.date_obs = Time(d['date_obs'])
        self.duration = d['duration'] * u.hour
        self.ang_speed = d['angspeed'] * (u.deg / u.s)
        self.delta_az = d['delta_az'] * u.deg
        self.nsweeps_per_elevation = d['nsweeps_per_elevation']
        self.period = d['period'] * u.s

        self.nsweep_even = self.nsweeps_per_elevation % 2 == 0
        self.dtsweepOneAz = (2 * self.delta_az / self.ang_speed).to(u.s)
        self.dt_centers = self.dtsweepOneAz * self.nsweeps_per_elevation

    def change_hor_down(self, hor_down): self.hor_down = hor_down * u.deg
    def change_hor_up(self, hor_up): self.hor_up = hor_up * u.deg
    def change_sun_sep(self, sun_sep): self.sun_sep = sun_sep * u.deg

    def get_pointing(self):
        self.centers = self.get_centers()
        print(f"*********** {self.centers.obstime.size-1} recenterings performed ***********")
        pointing = None
        for i in range(len(self.centers)):
            new_pointing = self.AzimuthSweep(self.centers[i], i)
            pointing = new_pointing if pointing is None else np.append(pointing, new_pointing, axis=0)
        return AltAz(az=pointing[:, 1] * u.deg, alt=pointing[:, 2] * u.deg,
                     obstime=pointing[:, 0], location=self.earth_location)

    def get_centers(self):
        start = self.date_obs - self.utc_offset
        time = np.arange(0, self.duration.to(u.s).value, 1) * u.s
        hor_frame = AltAz(obstime=start + time, location=self.earth_location)
        altaz = self.gal.transform_to(hor_frame)

        hor_mask = (altaz.alt > self.hor_down) & (altaz.alt < self.hor_up)
        altaz = altaz[hor_mask]

        if not altaz.obstime.size:
            raise ValueError('Source never visible')

        unix_times = altaz.obstime.unix
        end_index = np.append(np.where(np.diff(unix_times) > 1)[0], altaz.obstime.size - 1)
        start_index = np.append(0, end_index[:-1] + 1)
        durations = unix_times[end_index] - unix_times[start_index]

        if np.any(durations < self.dt_centers.to_value(u.s)):
            warnings.warn('Insufficient time for elevation changes')

        recenter_time = []
        for i in range(len(durations)):
            start_time = unix_times[start_index[i]] - unix_times[0]
            steps = np.arange(0, durations[i], self.nsweeps_per_elevation * self.dtsweepOneAz.to_value(u.s))
            recenter_time.append(start_time + steps)
        recenter_time = np.concatenate(recenter_time) * u.s
        return self.gal.transform_to(AltAz(obstime=altaz.obstime[0] + recenter_time,
                                           location=self.earth_location))

    def AzimuthSweep(self, center, idx):

        direction = [1, -1] if (self.nsweep_even or idx % 2 == 0) else [-1, 1]
        Sweep_Azimuth = np.array([])

        # Parameters

        step_s = 0.1  # seconds
        max_speed = 1.0  # deg/s
        t_ramp = 1.0  # seconds
        pause_duration = 3.0  # seconds
        delta_az = self.delta_az.to_value(u.deg)

        # Time samples for acc/dec ramps
        n_ramp = int(t_ramp / step_s)
        x_ramp = np.linspace(-3, 3, n_ramp)
        ramp_profile = 1 / (np.cosh(x_ramp) ** 2) # Put this ramp as an external method
        ramp_profile *= max_speed / np.max(ramp_profile)
        ramp_distance = np.sum(ramp_profile) * step_s

        # Flat segment
        flat_distance = 2 * delta_az - 2 * ramp_distance
        if flat_distance < 0:
            raise ValueError("Ramp too wide or speed too high for given delta_az.")
        
        t_flat = flat_distance / max_speed
        n_flat = int(t_flat / step_s)
        flat_profile = np.ones(n_flat) * max_speed

        # Pause segment at end (zero speed)
        n_pause = int(pause_duration / step_s)
        pause_profile = np.zeros(n_pause)

        # Total speed profile for one sweep
        full_speed_profile = np.concatenate([
            ramp_profile,
            flat_profile,
            ramp_profile[::-1],
            pause_profile
        ])

        # Convert speed to azimuth steps
        az_step = full_speed_profile * step_s
        az_step *= (2 * delta_az) / np.sum(az_step)  # Normalize to match exact sweep range

        for i in range(self.nsweeps_per_elevation):
            sweep_dir = direction[i % 2]
            if sweep_dir > 0:
                azimuths = center.az.value - delta_az + np.cumsum(az_step)
            else:
                azimuths = center.az.value + delta_az - np.cumsum(az_step)
            Sweep_Azimuth = np.append(Sweep_Azimuth, azimuths)

        # Build full sweep array
        Sweep = np.empty((len(Sweep_Azimuth), 3), dtype=object)
        start_time = center.obstime - self.dt_centers * 0.5
        sweep_duration = len(Sweep_Azimuth) * step_s * u.s
        start_time = center.obstime + idx * sweep_duration
        total_time = len(Sweep_Azimuth) * step_s
        time_steps = np.arange(0, total_time, step_s)[:len(Sweep_Azimuth)] * u.s
        Sweep[:, 0] = start_time + time_steps
        Sweep[:, 1] = Sweep_Azimuth
        Sweep[:, 2] = center.alt.value

        # Sun avoidance
        sep = self.SunSeparation(Sweep).deg
        Sweep = Sweep[sep > self.sun_sep.value]
        if Sweep.size == 0:
            raise ValueError('Sun obscures observation path')

        keep_idx = np.arange(len(Sweep)) % int(self.period.to_value(u.s) / step_s) == 0
        return Sweep[keep_idx]

    def SunSeparation(self, Sweep):
        pointing = AltAz(az=Sweep[:, 1] * u.deg, alt=Sweep[:, 2] * u.deg,
                         obstime=Sweep[:, 0], location=self.earth_location)
        sun_pos = get_sun(pointing.obstime)
        return pointing.separation(sun_pos)

    def SkyDips(self, azimuth, elevation, delta_elevation, ang_speed_elevation, dead_time=0.):
        az = azimuth * u.deg
        alt = elevation * u.deg
        delta_alt = delta_elevation * u.deg
        

        min_tsampling = 0.1  # sec

        if alt < self.hor_down:
            warnings.warn('Elevation is below the QUBIC field of view.')
        elif (alt + delta_alt) > self.hor_up:
            warnings.warn(f'The elevation sweep exceeds field of view: [{self.hor_down}, {self.hor_up}]')
        elif alt > self.hor_up:
            raise ValueError('Starting elevation above QUBIC field of view.')

        one_sweep_time = delta_alt.to_value(u.deg) / ang_speed_elevation + dead_time
        nsweeps = int((self.duration.to_value(u.s)) / one_sweep_time)

        alt_step = min_tsampling * ang_speed_elevation
        dead_samples = round(dead_time / min_tsampling)

        Up = lambda s, e, st: np.arange(s, e, st)
        Dn = lambda s, e, st: np.flip(np.arange(s, e, st))

        Sweep_Elevation = np.array([])
        for i in range(nsweeps):
            if i % 2 == 0:
                sweep_el = Up(alt.value, alt.value + delta_alt.value, alt_step)
                sweep_el = np.append(sweep_el, np.ones(dead_samples) * (alt.value + delta_alt.value))
            else:
                sweep_el = Dn(alt.value, alt.value + delta_alt.value, alt_step)
                sweep_el = np.append(sweep_el, np.ones(dead_samples) * alt.value)
            Sweep_Elevation = np.append(Sweep_Elevation, sweep_el)

        az_array = np.full_like(Sweep_Elevation, az.value)
        time = self.date_obs + np.linspace(0, self.duration.to_value(u.s),
                                           len(Sweep_Elevation), endpoint=False) * u.s

        pointing = AltAz(az=az_array * u.deg, alt=Sweep_Elevation * u.deg,
                         obstime=time, location=self.earth_location)

        ratio = int(self.period.to_value(u.s) / min_tsampling)
        indexes = np.arange(pointing.obstime.size)
        return pointing[indexes[indexes % ratio == 0]]
    
    def DeltaTime(self,time):
		
        '''
        INPUT:
        - time vector array --> each item is an Time object from astropy

        Return a vector of float --> each item is the time elapsed from the first item of array (in seconds)
		'''
	    
        Time = np.array( pd.to_datetime(time.value), dtype='datetime64[us]' ) #np.array datetime in microseconds
        delta_time = np.cumsum(np.diff(Time)) #compute the deltatime from beginning
        delta_time = delta_time / np.timedelta64(1,'s') #transform to seconds keeping the decimals
        delta_time = np.insert(delta_time,0,0) #add the beginning time (0.)

        return delta_time
    
    def SunTrajectory(self, date="2025-04-14", npoints=500):


        time_start = Time(f"{date} 00:00:00")
        times = time_start + np.linspace(0, 24, npoints) * u.hour / 24

        frame = AltAz(obstime=times, location=self.earth_location)
        sun = get_sun(times).transform_to(frame)

        data = np.empty((npoints, 3), dtype=object)
        data[:, 0] = times
        data[:, 1] = sun.az.to_value(u.deg)
        data[:, 2] = sun.alt.to_value(u.deg)
        return data



def plot_observation(d):

    qubic = QubicObservation(d)
    pointing = qubic.get_pointing()

    #Time arrays
    time_rel = (pointing.obstime - pointing.obstime[0]).to(u.hour).value
    time_sec = (pointing.obstime - pointing.obstime[0]).sec
    az = pointing.az.deg

    #Scan speed
    dt = np.diff(time_sec)
    daz = np.diff(az)
    speed = np.abs(np.insert(daz / dt, 0, 0))

    #Compute Sun trajectory over the observation time range
    times = pointing.obstime
    sun_altaz = get_sun(times).transform_to(AltAz(obstime=times, location=qubic.earth_location))
    sun_az = sun_altaz.az.deg
    sun_alt = sun_altaz.alt.deg

    
    plt.figure(figsize=(12, 8))


    # Azimuth vs Time
    plt.subplot(2, 2, 1)
    plt.plot(time_rel, az, 'b-', alpha=0.7, label='Telescope')
    plt.plot(time_rel, sun_az, 'orange', linestyle='--', label='Sun')
    plt.xlabel('Time (hours)')
    plt.ylabel('Azimuth (deg)')
    plt.title('Azimuth vs Time')
    plt.legend()
    plt.grid(True)
    
    # Elevation vs Time
    plt.subplot(2, 2, 2)
    plt.plot(time_rel, pointing.alt.deg, 'ro', alpha=0.7)
    #plt.plot(time_rel, sun_alt, 'goldenrod', linestyle='--', label='Sun')
    plt.xlabel('Time (hours)')
    plt.ylabel('Elevation (deg)')
    plt.title('Elevation vs Time')
    
    plt.grid(True)

    # Elevation vs Azimuth
    plt.subplot(2, 2, 3)
    sc = plt.scatter(az, pointing.alt.deg, c=time_rel, cmap='viridis', s=10)
    #plt.scatter(sun_az, sun_alt, c='orange', s=10, alpha=0.5, label='Sun')
    plt.colorbar(sc, label='Time (hours)')
    plt.xlabel('Azimuth (deg)')
    plt.ylabel('Elevation (deg)')
    plt.title('Elevation vs Azimuth')
    plt.grid(True)

    # Compute azimuth speed in deg/s
    az_speed = np.gradient(az, time_sec)
    s= np.abs(az_speed)

    # Time arrays
    time_rel = (pointing.obstime - pointing.obstime[0]).to(u.hour).value
    time_sec = (pointing.obstime - pointing.obstime[0]).sec
    az = pointing.az.deg

    # Scan speed
    dt = np.diff(time_sec)
    daz = np.diff(az)
    speed = np.insert(daz / dt, 0, 0)
    s=np.abs(speed)
    
    # Azimuth Speed vs Time
    plt.subplot(2, 2, 4)
    plt.plot(time_rel, s, 'g-', alpha=0.7)
    plt.xlabel('Time (hours)')
    plt.ylabel('Azimuth Speed (deg/s)')
    plt.title('Azimuth Speed vs Time')
    plt.grid(True)

    plt.tight_layout()
    plt.show()




# Parameters
if __name__ == "__main__":
    sample_params = {
        'latitude': -24.1844,
        'longitude': -66.8714,
        'RA_center': 266.41683708,
        'DEC_center': -29.00781056,
        'date_obs': '2025-04-20 00:00:00',
        'duration': 4,  # in hours
        'angspeed': 1,  # deg/s
        'delta_az': 20.0,  # deg
        'nsweeps_per_elevation': 25,
        'period': 1,  # s
        }

    plot_observation(sample_params)
