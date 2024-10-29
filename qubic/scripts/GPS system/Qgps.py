import os
import struct

import numpy as np
import matplotlib.pyplot as plt

import datetime as dt

#TODO : Add case where the cal source is not aligned with the antenna vector
#TODO : Add case where the base antenna is not located at the center of the QUBIC window

class GPS:
    
    def __init__(self, gps_data_file_path, base_antenna_position, distance_between_antennas, distance_calsource, observation_date):
        
        ### Fixed parameters
        #! Need to decide in which coordinate system I should express it
        #! Maybe in a cartesian system centered on the center of the QUBIC window/horn array
        self.base_antenna_position = base_antenna_position
        self.distance_between_antennas = distance_between_antennas
        self.distance_calsource = distance_calsource
        self.observation_date = observation_date
        
        ### Convert the data from GPS system into a dictionary
        self.gps_data = self.read_gps_bindat(gps_data_file_path)
        
        ### Import all the GPS data from the dictionary and convert them in proper units
        self.timestamp = np.array(self.gps_data['timestamp'])
        
        ### Build datetime array
        self.datetime = self.create_datetime_array(self.timestamp)
        
        ### 
        self.observation_indices = self.get_observation_indices(self.datetime, self.observation_date)
        self.observation_time = self.timestamp[self.observation_indices]
        self.observation_datetime = self.datetime[self.observation_indices]
        
        # rpN, rpE, rpD give the relative position of the antenna 2 wrt base antenna in North, East, Down coordinates
        self.rpN = np.array(self.gps_data['rpN'])[self.observation_indices] / 10000                       # in m
        self.rpE = np.array(self.gps_data['rpE'])[self.observation_indices] / 10000                       # in m
        self.rpD = np.array(self.gps_data['rpD'])[self.observation_indices] / 10000                       # in m
        
        # roll give the angle between antenna 2 - antenna 1 vector and the North axis
        self.roll = np.radians(np.array(self.gps_data['roll'])[self.observation_indices] / 1000)          # in rad
        
        # yaw give the angle between antenna 2 - antenna 1 vector and the horizontal plane
        #! Need to double check that
        self.yaw = np.radians(np.array(self.gps_data['yaw'])[self.observation_indices] / 1000)            # in rad
        
        self.pitchIMU = np.radians(np.array(self.gps_data['pitchIMU'])[self.observation_indices] / 1000)  # in rad
        self.rollIMU = np.radians(np.array(self.gps_data['rollIMU'])[self.observation_indices] / 1000)    # in rad
        self.temperature = np.array(self.gps_data['temperature'])[self.observation_indices] / 10          # in Celsius
        self.checksum = np.array(self.gps_data['checksum'])[self.observation_indices]
        
        ### Compute position of antennas 1 & 2 and calibration source in North, East, Down cooridnates
        #! Be careful: as it is now, the calibration source needs to be on the straight line formed by antenna 1 and antenna 2
        self.position_antenna2 = self.position_antenna_2(self.base_antenna_position)
        self.position_antenna1 = self.position_wrt_antenna_2(self.distance_between_antennas) + self.position_antenna2
        self.position_calsource = self.position_wrt_antenna_2(self.distance_calsource) + self.position_antenna2

        ### Compute the vectors between the calibration source and QUBIC, and the vector between the antennas in NED coordinates
        self.vector_1_2 = self.position_antenna2 - self.position_antenna1
        self.vector_calsource = self.position_calsource 
        
        ### Compute the calibration source orientation angles
        self.calsource_orientation_angles = np.degrees(self.calsource_orientation(self.vector_1_2, self.vector_calsource))        
    
    def read_gps_bindat(self, path_file):
        """GPS binary data.
        
        Method to convert the binary data acquired from by RTK simple broadcast into readable format and store them in a dictionary.

        Parameters
        ----------
        path_file : str
            GPS file path.

        Returns
        -------
        dat: dict
            Dictionary containing the GPS data.

        Raises
        ------
        ValueError
            If the file is not found.
        """   
        
        if not os.path.isfile(path_file):
            print('ERROR!  File not found: %s' % path_file)
            return

        ### read the data
        h = open(path_file,'rb')
        bindat = h.read()
        h.close()

        ### interpret the binary data
        fmt = '<Bdiiiiiiifi'
        nbytes = 45
        names = "STX,timestamp,rpN,rpE,rpD,roll,yaw,pitchIMU,rollIMU,temperature,checksum".split(',')
        data = {}
        for name in names:
            data[name] = []    

        idx = 0
        while idx+nbytes<len(bindat):
            packet = bindat[idx:idx+nbytes]
            dat_list = struct.unpack(fmt,packet)

            if len(dat_list)!=len(names):
                raise ValueError('ERROR:  Incompatible data.')

            for datidx,name in enumerate(names):
                data[name].append(dat_list[datidx])

            idx += nbytes

        return data
    
    def create_datetime_array(self, timestamp, utc_offset=0):
        """Datetime array.

        Build datetime array from timestamp data. We can convert them to any time zone using utc_offset parameter.

        Parameters
        ----------
        timestamp : array_like
            Timestamp data.
        utc_offset : int, optional
            UTC offset for any time zone in hours, by default 0

        Returns
        -------
        datetime : array_like
            Datetime array.
        """        
        
        return np.array([dt.datetime.utcfromtimestamp(tstamp) for tstamp in timestamp]) + dt.timedelta(hours=utc_offset)
    
    def datetime_to_index(self, datetime, observation_date):
        """Datetime to index.
        
        Method to find the data index associated to a chosen date.

        Parameters
        ----------
        datetime : datetime
            Datetime array.
        date : str or datetime
            Chosen date, either in the form of a string ('year-month-dayThour:minute:second') or a datetime isoformat object.

        Returns
        -------
        date_index : int
            Index associated to the chosen date.

        Raises
        ------
        TypeError
            Raise TypeError if the chosen date is not in the form of a string or a datetime object.
        IndexError
            Raise ValueError if the chosen date is not in the data.
        """            

        if isinstance(observation_date, str):
            date = dt.datetime.fromisoformat(observation_date)
        elif isinstance(observation_date, dt.datetime):
            date = observation_date
        else:
            raise TypeError('ERROR! Please choose a date in the form of a string or a datetime object.')

        try:
            return np.where(datetime == date)[0][0]
        except :
            raise IndexError('ERROR! The date you chose is not in the data.')
        
    def get_observation_indices(self, datetime, observation_date):
        """Observation indices.
        
        Return the indices associated to the chosen date. 
        If date.shape is 1, the method returns a unique index associated to the chosen date. 
        If date.shape is 2, the method returns the indices associated to the chosen date range.

        Parameters
        ----------
        datetime : datetime
            Datetime array.
        date : array_like
            Array of dates, either a single date or a date range, in str or datetime format.

        Returns
        -------
        observation_indices : array_like
            Array containing the indices associated to the observation date(s).

        Raises
        ------
        ValueError
            Raise ValueError if the date does not have the proper shape, 1 or 2.
        """        
         
        ### If we give an unique observation date
        if len(observation_date.shape) == 1 and observation_date.shape[0] == 1:
            return np.array([self.datetime_to_index(datetime, observation_date[0])])
        
        ### If we give a starting and stoping dates
        if len(observation_date.shape) == 1 and observation_date.shape[0] == 2:
            start_index = self.datetime_to_index(datetime, observation_date[0])
            end_index = self.datetime_to_index(datetime, observation_date[1])
            return np.arange(start_index, end_index, 1, dtype=int)
        
        else:
            raise ValueError('ERROR! Please choose a correct shape for the date: 1 or 2.')

    def position_antenna_2(self, base_antenna_position):
        """Position antenna 2.
        
        Method to compute the position of the antenna 2 in North, East, Down coordinates.

        Parameters
        ----------
        base_antenna_position : array_like
            The base antenna position in North, East, Down coordinates.
        gps_data : dict
            Dictionary containing the GPS data.
        date : array_like
            Array of dates, either a single date or a date range, in str or datetime format.

        Returns
        -------
        antenna_2_position : array_like
            Position of the antenna 2 in North, East, Down coordinates.
        """        
        
        ### Call the base antenna position in North, East, Down coordinates
        rpN_base, rpE_base, rpD_base = base_antenna_position
        
        ### Compute the position of the antenna 2 in North, East, Down coordinates
        rpN_antenna_2 = self.rpN + rpN_base
        rpE_antenna_2 = self.rpE + rpE_base
        rpD_antenna_2 = self.rpD + rpD_base
        
        return np.array([rpN_antenna_2, rpE_antenna_2, rpD_antenna_2])
    
    def position_wrt_antenna_2(self, distance_w_antenna_2):
        """Position wrt antenna 2.
        
        General fonction to compute the position of any point located on the straight line formed by the antenna 1 - anntenna 2 vector, wrt antenna 2 in North, East, Down coordinates.
        
        Be careful, yaw is not the usual theta angle in sphercial cooridinates (i.e. the latitude angle): it corresponds to the elevation angle. 
        It is why we need to add np.pi/2 in the conversion formulas.

        Parameters
        ----------
        distance_w_antenna_2 : float
            Distance between a point located  on the straight line formed by the antenna 1 - anntenna 2 vector and antenna 2.
        gps_data : dict
            Dictionary containing the GPS data.
        date : array_like
            Array of dates, either a single date or a date range, in str or datetime format.

        Returns
        -------
        position_wrt_antenna_2 : array_like
            Position of the point wrt antenna 2 in North, East, Down coordinates.
        """        
        
        ### Compute the position of the antenna 1 wrt antenna 2 in North, East, Down coordinates
        _rpN = distance_w_antenna_2 * np.cos(self.roll) * np.sin(np.pi/2 - self.yaw)
        _rpE = distance_w_antenna_2 * np.sin(self.roll) * np.sin(np.pi/2 - self.yaw)
        _rpD = - distance_w_antenna_2 * np.cos(np.pi/2 - self.yaw)
        
        return np.array([_rpN, _rpE, _rpD])
    
    def calsource_orientation(self, vector_1_2, vector_cal):
        """Calsource orientation.
        
        Method to compute the orientation of the calsource.

        Parameters
        ----------
        vector_1_2 : array_like
            Vector between antenna 1 and 2.
        vector_cal : array_like
            Vector QUBIC and the calibration source.

        Returns
        -------
        angles : array_like
            Orientation angles of the calsource.
        """
        
        angles = np.zeros(vector_1_2.shape)        
        
        ### Down vector
        ed = np.array([0, 0, 1])
        
        ### Direction of calsource vector
        n_cal = vector_cal / np.linalg.norm(vector_cal)
        
        ### Projections of vector_1_2 on planes ortho to down axis and vector_cal
        vector_1_2_ortho_n_cal = vector_1_2 - (np.sum(vector_1_2 * n_cal, axis=0) / np.sum(n_cal * n_cal, axis=0))[None, :] * n_cal
        vector_1_2_ortho_ed = vector_1_2 - (np.sum(vector_1_2 * ed[:, None], axis=0) / np.dot(ed, ed))[None, :] * ed[:, None]
        
        ### Build orthogonal vector to vector_cal and down axis
        vector_ortho = np.cross(n_cal, ed, axisa=0).T
        
        ### Compute the angles
        angles[0] = np.arccos(np.sum(vector_1_2_ortho_n_cal * vector_ortho, axis=0) / (np.linalg.norm(vector_1_2_ortho_n_cal, axis=0) * np.linalg.norm(vector_ortho, axis=0)))
        angles[2] = np.arccos(np.sum(vector_1_2_ortho_ed * vector_ortho, axis=0) / (np.linalg.norm(vector_1_2_ortho_ed, axis=0) * np.linalg.norm(vector_ortho, axis=0)))

        return angles