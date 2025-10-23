import numpy as np
import pandas as pd
import warnings
from astropy import units as u
from astropy.coordinates import SkyCoord, AltAz, EarthLocation, get_sun
from astropy.time import Time
import matplotlib.pyplot as plt


class QubicObservation:

    def __init__(self, d, hor_down=30, hor_up=70, sun_sep=20):
        
        # Location 
        self.earth_location = EarthLocation(lat=d['latitude'] * u.deg,
                                           lon=d['longitude'] * u.deg)
        self.utc_offset = -3 * u.hour
        self.hor_down = hor_down * u.deg
        self.hor_up = hor_up * u.deg

        # Sky coordinates
        self.eq = SkyCoord(ra=d['RA_center'] * u.deg,
                           dec=d['DEC_center'] * u.deg, frame='icrs')
        self.gal = self.eq.galactic

        # Observation parameters
        self.date_obs = Time(d['date_obs'])
        self.duration = d['duration'] * u.hour
        self.ang_speed = d['angspeed'] * (u.deg / u.s)
        self.delta_az = d['delta_az'] * u.deg
        self.nsweeps_per_elevation = d['nsweeps_per_elevation']
        self.period = d['period'] * u.s
        self.nsweep_even = (self.nsweeps_per_elevation % 2 == 0)

        # Sweep shape parameters
        self.step_s = 0.1              # sampling interval [s]
        self.max_speed = 1.0           # [deg/s]
        self.t_ramp = 1.0              # ramp duration [s]
        self.pause_duration = 3.0      # pause at end [s]

        # Compute segment counts
        self.n_ramp = int(self.t_ramp / self.step_s)
        x = np.linspace(-3, 3, self.n_ramp)
        ramp = 1 / (np.cosh(x)**2)
        ramp *= self.max_speed / np.max(ramp)
        ramp_distance = np.sum(ramp) * self.step_s

        flat_dist = 2 * self.delta_az.to_value(u.deg) - 2 * ramp_distance
        if flat_dist < 0:
            raise ValueError("Ramp parameters incompatible with delta_az")
        t_flat = flat_dist / self.max_speed
        self.n_flat = int(t_flat / self.step_s)
        self.n_pause = int(self.pause_duration / self.step_s)

        # Combine into speed profile 
        
        flat = np.ones(self.n_flat) * self.max_speed
        self.speed_prof = np.concatenate([ramp, flat, ramp[::-1], np.zeros(self.n_pause)])
        
        # Precompute azimuth step sizes for one full sweep
        
        az_range = 2 * self.delta_az.to_value(u.deg)
        self.az_step = self.speed_prof * self.step_s
        self.az_step *= az_range / np.sum(self.az_step)

        # Timing
        total_samples = self.speed_prof.size
        self.t_sweep = total_samples * self.step_s * u.s
        self.dt_centers = self.t_sweep * self.nsweeps_per_elevation

    def change_hor_down(self, hor_down):  self.hor_down = hor_down * u.deg
    def change_hor_up(self, hor_up):      self.hor_up = hor_up * u.deg

    def get_pointing(self):
        self.centers = self.get_centers()
        print(f"*********** {len(self.centers)-1} recenterings performed ***********")
        all_pts = [self.AzimuthSweep(cen, idx)
                   for idx, cen in enumerate(self.centers)]
        P = np.vstack(all_pts)
        return AltAz(az=P[:,1]*u.deg,
                     alt=P[:,2]*u.deg,
                     obstime=P[:,0],
                     location=self.earth_location)

    def get_centers(self):
        start = self.date_obs - self.utc_offset
        tgrid = np.arange(0, self.duration.to(u.s).value, 1) * u.s
        frame = AltAz(obstime=start + tgrid, location=self.earth_location)
        path = self.gal.transform_to(frame)
        mask = (path.alt > self.hor_down) & (path.alt < self.hor_up)
        vis = path[mask]
        if len(vis) == 0:
            raise ValueError('Source never visible')

        unix = vis.obstime.unix
        ends = np.append(np.where(np.diff(unix) > 1)[0], len(vis)-1)
        starts = np.append(0, ends[:-1]+1)
        durations = unix[ends] - unix[starts]

        rec_times = []
        for s, dur in zip(starts, durations):
            offset = unix[s] - unix[0]
            steps = np.arange(0, dur, self.dt_centers.to_value(u.s))
            rec_times.append(offset + steps)
        rec = np.concatenate(rec_times) * u.s

        return self.gal.transform_to(
            AltAz(obstime=vis.obstime[0] + rec,
                  location=self.earth_location))

    def AzimuthSweep(self, center, idx):
        
        # Build back-and-forth azimuths sweeps
        
        sequences = []
        base = center.az.value
        half = self.az_step.cumsum()
        L = self.delta_az.to_value(u.deg)
        for i in range(self.nsweeps_per_elevation):
            if (self.nsweep_even or i % 2 == 0):  # forward on even
                seq = base - L + half
            else:                                 # backward on odd
                seq = base + L - half
            sequences.append(seq)
        AZ = np.concatenate(sequences)

        # Timestamp around center
        
        t0 = center.obstime - self.t_sweep/2
        times = t0 + np.arange(len(AZ)) * self.step_s * u.s

        # Assembling the data
        
        data = np.empty((len(AZ),3),object)
        data[:,0] = times
        data[:,1] = AZ
        data[:,2] = center.alt.value

        # Down-sample by period
        idxs = np.arange(data.shape[0]) % int(self.period.to_value(u.s)/self.step_s) == 0
        return data[idxs]

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
        min_tsampling = 0.1 * u.s
        nt = len(Sweep_Elevation)
        time = self.date_obs + np.arange(nt) * min_tsampling
        
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


    
    plt.figure(figsize=(12, 8))


    # Azimuth vs Time
    plt.subplot(2, 2, 1)
    plt.plot(time_rel, az, 'b-', alpha=0.7, label='Telescope')
    plt.xlabel('Time (hours)')
    plt.ylabel('Azimuth (deg)')
    plt.title('Azimuth vs Time')
    plt.legend()
    plt.grid(True)
    
    # Elevation vs Time
    plt.subplot(2, 2, 2)
    plt.plot(time_rel, pointing.alt.deg, 'r', alpha=0.7)
    plt.xlabel('Time (hours)')
    plt.ylabel('Elevation (deg)')
    plt.title('Elevation vs Time')
    
    plt.grid(True)

    # Elevation vs Azimuth
    plt.subplot(2, 2, 3)
    sc = plt.scatter(az, pointing.alt.deg, c=time_rel, cmap='viridis', s=10)
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
