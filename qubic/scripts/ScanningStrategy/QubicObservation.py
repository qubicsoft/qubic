##########################
# QubicObservation class #
##########################


'''

Project: Qubic Observation class - Object to model a realistic scanning strategy for QUBIC simulation.
Author: Nicola brancadori - nicola.brancadori@studenti.unimi.it
Creation Date: 10-Jan-2023
Last edit: 28-Nov-2023

'''
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, AltAz, EarthLocation, Galactic
import astropy.coordinates as coord
from astropy.time import Time
import matplotlib.pyplot as plt
import warnings

class QubicObservation:


	def __init__(self,d,hor_down=30,hor_up=70,sun_sep=40):
	#Alto Chorillos location and time zone
		self.earth_location = EarthLocation(lat=d['latitude'],lon=d['longitude'])
		self.utc_offset = -3*u.hour
		self.hor_down = hor_down*u.deg
		self.hor_up = hor_up*u.deg
		self.sun_sep = sun_sep*u.deg
		
	#Point in the sky coordinates (galactic and equatorial)
		self.eq = SkyCoord(d['RA_center'],d['DEC_center'],frame='icrs',unit='deg') #(x1=ra,x2=dec)
		self.gal = self.eq.galactic
		
	#Observation features
		self.date_obs = Time(d['date_obs'])
		self.duration = d['duration']
		self.ang_speed = d['angspeed']
		self.delta_az = d['delta_az']
		self.nsweeps_per_elevation = d['nsweeps_per_elevation']
		self.period = d['period']

	#Set some useful variables, as attributes, to perform the get_pointing method.
		
		#boolean variable to True if the nsweeps_per_elevation is even.
		self.nsweep_even=True if self.nsweeps_per_elevation%2 == 0 else False
		#Time per OneAzimuth Sweep
		self.dtsweepOneAz = (self.delta_az / self.ang_speed)*2*u.second
		#Time per an entire Sweep
		self.dt_centers = self.dtsweepOneAz * self.nsweeps_per_elevation


	#To change the three QUBIC parameters that are not in the dictionary
	def change_hor_down(self,hor_down):
		self.hor_down = hor_down*u.deg

	def change_hor_up(self,hor_up):
		self.hor_up = hor_up*u.deg

	def change_sun_sep(self,sun_sep):
		self.sun_sep = sun_sep*u.deg
	
	def get_pointing(self):
		
		"""
		Return the pointing astropy.coordinates.builtin_frames.altaz.AltAz object:
		- pointing.obstime is the array of time at which each point was taken. 
		- pointing.az is the array of azimuth
		- pointing.alt is the array of elevation
		"""


		self.centers = self.get_centers()

		print("******* With these parameters ",self.centers.obstime.size-1," recentering are performed ***********")

		#perform the azimuth sweeps around each center and build the pointing array
		for i in range(0,np.size(self.centers.alt)):
			if (i==0):
				pointing = self.AzimuthSweep(self.centers[0],0)
			else:
				pointing = np.append(pointing,self.AzimuthSweep(self.centers[i],i),axis=0)

		#Pointing array as astropy object
		pointing = AltAz(az=pointing[:,1]*u.deg,alt=pointing[:,2]*u.deg,obstime=pointing[:,0],location=self.earth_location)
		
		return pointing

	def get_centers(self):
		
		"""
		Return the array of centers (coordinates.AltAz astropy object) in horizontal frame.
		The centers are the positions around which the sweeping in azimuth is performed.
		"""
		
		#Compute the path of the patch (source) center.
		start = self.date_obs - self.utc_offset
		time = np.arange(0, self.duration*3600)*u.second
		hor_frame = AltAz(obstime = start + time, location = self.earth_location)
		altaz = self.gal.transform_to(hor_frame)

		#Apply the Horizon effect
		#Horizon mask
		hor_mask = (altaz.alt.value > self.hor_down.value) & (altaz.alt.value < self.hor_up.value)
		altaz = altaz[hor_mask]
		

		#Compute the actual duration and start of the observation.
		#Actual means with horizon effect
		if (np.any(altaz.obstime)==False):
			raise ValueError('The source is never visible')
		else:
			#The source may not be always visible during the observation period (e.g multi-day observations). 
			#I formalize this by considering each interval in which the source is visible as a separate observation 
			#with a beginning and an end:
			
			end_index = np.where(np.diff(altaz.obstime.unix) > 1)
			end_index = np.append(end_index,altaz.obstime.size - 1)

			start_index = np.zeros(1,dtype=int)
			start_index = np.append(start_index,end_index[:-1] + 1)

			durations = altaz.obstime[end_index].unix - altaz.obstime[start_index].unix

		if( np.any(durations < self.dt_centers.value) ):
			warnings.warn('You stay always on the same elevation with these parameters')

		#Build the array of timecenter cycling through the different observations (durations)
		for i in range(durations.size):
			if (i==0):
				recenter_time = np.arange(0,durations[0],self.nsweeps_per_elevation*self.dtsweepOneAz.value)*u.second
			else:
				start_time = altaz.obstime[start_index[i]].unix - altaz.obstime[0].unix
				recenter_time = np.append(recenter_time, np.arange(start_time,durations[i]+start_time,self.nsweeps_per_elevation*self.dtsweepOneAz.value)*u.second)
		
		#Build the array of center as an astropy object
		hor_frame = AltAz(obstime = altaz.obstime[0] + recenter_time, location = self.earth_location)
		altaz = self.gal.transform_to(hor_frame) #--> This is the centers vector

		
		return altaz



	def AzimuthSweep(self, center, idx):
		"""
		INPUT:
	 	- center (coordinates.AltAz astropy object) the center around which perform the azimuth
	 	- idx (int) the index of the center referring to the centers array (maybe I can generalize this)

	 	Return an array after azimuthsweep called Sweep:
	 	- sweep[:,0] obstime time array
	 	- sweep[:,1] azimuth array
	 	- sweep[:,2] altitude array
	 	"""

		# Define the forward sweep azimuth function
		FAz = lambda c,d,n : np.arange(c-d,c+d,n) #forward sweep (c,d,n) --> (center azimuth,delta azimuth,azimuth step)
		BAz = lambda c,d,n : np.flip(np.arange(c+d,c-d,n)) #backwards sweep (c,d,n) --> (center azimuth,delta azimuth,azimuth step)

		# Determine the direction of azimuth sweep based on nsweeps_even(even or odd) and idx(even or odd) of the center
		if self.nsweep_even or idx % 2 == 0:
			direction = [1,-1]
		else:
			direction = [-1,1]

		#Compute the number of points per sweep in azimuth
		Azstep = 1.*self.ang_speed

		#Create the sweep array to fill with azimuth values
		Sweep_Azimuth = np.array([])

		for i in range(self.nsweeps_per_elevation):
			#select the direction item (0 if i is even; 1 if i is odd) that will be the sweep direction
			sweep_direction = direction[i%2]
			
			# Perform the appropriate (forward or backward parametrized by sweep_direction variable) azimuth sweep
			if (sweep_direction == 1):
				sweep_azimuth = FAz(center.az.value, self.delta_az * sweep_direction, Azstep)
			else:
				sweep_azimuth = BAz(center.az.value, self.delta_az * sweep_direction, Azstep)

			Sweep_Azimuth = np.append(Sweep_Azimuth,sweep_azimuth)
		

		#Create the pointing vector
		Sweep = np.empty( shape=(Sweep_Azimuth.size,3) , dtype=object)

		#Create the time vector with sampling time of 1 second (that is why the minimum period of the observation is 1 sec --> to update)
		start_time = center.obstime - self.dt_centers*0.5
		step_time = np.linspace(0,self.dt_centers,Sweep_Azimuth.size,endpoint=False)
		time =  start_time + step_time
		
		#Create the elevation vector
		elevation = np.full(Sweep_Azimuth.size, center.alt.value)

		Sweep[:,0] = time
		Sweep[:,2] = elevation
		Sweep[:,1] = Sweep_Azimuth
		
		
		#Sun mask
		sep = self.SunSeparation(Sweep).deg
		sun_mask = sep > self.sun_sep.value
		Sweep = Sweep[sun_mask]
		if (Sweep.size == 0):
			raise ValueError('The sun obscure all the source path')

		#masked the array with the period of measure
		indexes = np.arange(Sweep[:,0].size)
		indexes = indexes[indexes%self.period==0]
		Sweep = Sweep[indexes]


		return Sweep


	def SkyDips(self,azimuth,elevation,delta_elevation,ang_speed_elevation,dead_time = 0.):

		"""
		INPUT:
	 	- azimuth (float) the azimuth at which perform the azimuth
	 	- elevation (float) the initial elevation value
	 	- delta_elevation (float) the elevation interval of the sky dips
	 	- ang_speed_elevation (float) the angular velocity of the telescope motion
	 	- dead_time (float) the time spent by the telescope at the same elevation 

	 	Return the poiniting of the sky dips as astropy.coordinates.builtin_frames.altaz.AltAz object:
		- pointing.obstime is the array of time at which each point was taken. 
		- pointing.az is the array of azimuth
		- pointing.alt is the array of elevation
	 	"""

		az = azimuth*u.deg
		alt = elevation*u.deg
		delta_alt = delta_elevation*u.deg

		min_tsampling = 0.1

		if (alt.value < self.hor_down.value):
			warnings.warn('Elevation is below the QUBIC field of view: <',self.hor_down)
		elif ((alt + delta_alt).value > self.hor_up.value):
			warnings.warn('The elevation sweep is out of QUBIC field of view:', '[',self.hor_down,', ',self.hor_up,']')
		elif (alt.value > self.hor_up.value):
			raise ValueError('The starting elevation is above the QUBIC field of view: >',self.hor_up)

		#Time to finish a sweep in Elevation plus teh deadtime
		one_sweep_time = (delta_alt.value/ang_speed_elevation) + dead_time
		#Compute the number of sweeps
		nsweeps = round(self.duration*3600/one_sweep_time)
		#Step in elevation for each measure
		alt_step = min_tsampling * ang_speed_elevation #0.1 sec is the minimum time sampling
		#Number of measures in the dead time
		dead_samples = round(dead_time/min_tsampling)

		#Define the upward and downward sweeps in elevation
		Up = lambda s,e,st : np.arange(s,e,st) #upward sweep (s,e,st) --> (elevation start, elevation end, elevation step)
		Dn = lambda s,e,st : np.flip(np.arange(s,e,st)) #downward sweep (s,e,st) --> (elevation start, elevation end, elevation step)

		#Create the all sweep array to fill with elevation values
		Sweep_Elevation = np.array([])
		
		for i in range(nsweeps):	
			if (i%2 == 0):
				sweep_el = Up(alt.value, alt.value + delta_alt.value, alt_step)
				#Append the value at the same elevation taken in the deadtime 
				sweep_el = np.append(sweep_el,np.ones(dead_samples)*(alt.value + delta_alt.value))
			else:
				sweep_el = Dn(alt.value, alt.value + delta_alt.value, alt_step)
				#Append the value at the same elevation taken in the deadtime 
				sweep_el = np.append(sweep_el,np.ones(dead_samples)*alt.value)

			Sweep_Elevation = np.append(Sweep_Elevation,sweep_el)
		
		az = np.ones(Sweep_Elevation.size)*az

		#Build the time array
		start_time = self.date_obs
		step_time = np.linspace(0,self.duration*3600,Sweep_Elevation.size,endpoint=False)*u.second
		time =  start_time + step_time
		#step_time = step_time.round(decimals = 6)
		#print('step_time: ',np.diff(step_time))
		#print('step_time: ',step_time[:50])
		#print('start_time: ',start_time)
		#print('step_time: ',np.diff( self.DeltaTime(time) ) )

		pointing = AltAz(az,Sweep_Elevation*u.deg,obstime=time,location=self.earth_location)

		#Masked the array with the period of measure
		ratio = self.period/min_tsampling
		indexes = np.arange(pointing.obstime.size)
		indexes = indexes[ indexes%ratio == 0 ]
		pointing = pointing[indexes]
		
		return pointing


	def SunSeparation(self, Sweep):

		'''
		INPUT:
		- pointing array --> (time,az,alt)

		Return the sun separation array, each item is the the angular separation 
		between the point and the sun.

		'''
		pointing = AltAz(az=Sweep[:,1]*u.deg,alt=Sweep[:,2]*u.deg,obstime=Sweep[:,0],location=self.earth_location)
		
		sun_pos = coord.get_sun(pointing.obstime)

		sep = pointing.separation(sun_pos)
		
		
		return sep


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












