import os
import struct

import numpy as np
import matplotlib.pyplot as plt

import datetime as dt


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
        self.position_antenna2_ned = self.position_antenna_2(self.base_antenna_position)
        self.position_antenna1_ned = self.position_wrt_antenna_2(self.distance_between_antennas) + self.position_antenna2_ned
        self.position_calsource_ned = self.position_wrt_antenna_2(self.distance_calsource) + self.position_antenna2_ned

        ### Compute the vectors between the calibration source and QUBIC, and the vector between the antennas in NED coordinates
        self.vector_vector_1_2_ned = self.position_antenna2_ned - self.position_antenna1_ned
        self.vector_calsource_ned = self.position_calsource_ned 

        ### Compute the vectors in the XYZ coordinates system, define such that ex is the direction of the calsource vector, ey is the orthogonal vector on the North-East plane and ez is the Down axis
        self.vector_vector_1_2_xyz = self.ned_to_xyz(self.vector_vector_1_2_ned, self.vector_calsource_ned)
        self.vector_calsource_xyz = self.ned_to_xyz(self.vector_calsource_ned, self.vector_calsource_ned)
        
        ### Compute the calibration source orientation angles
        self.calsource_orientation_angles = self.calsource_orientation(self.vector_vector_1_2_xyz)        
    
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
        try :
            observation_date.shape
        except:
            return self.datetime_to_index(datetime, observation_date)
        
        ### If we give a starting and stoping dates
        if observation_date.shape[0] == 2:
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
        _rpN = distance_w_antenna_2 * np.cos(self.roll) * np.sin(self.yaw + np.pi/2)
        _rpE = distance_w_antenna_2 * np.sin(self.roll) * np.sin(self.yaw + np.pi/2)
        _rpD = distance_w_antenna_2 * np.cos(self.yaw + np.pi/2)
        
        return np.array([_rpN, _rpE, _rpD])
    
    def ned_to_xyz(self, ned_vector, calsource_vector):
        """NED to XYZ.
        
        Method to compute the XYZ coordinates of a vector expressed in NED coordinates. The XYZ coordinates are defined such that ex is oriented in calsource direction,
        ey is the orthogonal vector to ex in the plane defined by the North and East axis, and ez is simply the Down axis.

        Parameters
        ----------
        ned_vector : array_like
            Vector expressed in NED coordinates.
        calsource_vector : array_like
            Vector used to define the XYZ coordinates.

        Returns
        -------
        xyz_vector : array_like
            ned_vector expressed in XYZ coordinates.
        """        
        
        ### Compute the rotation matrix
        north_vector = np.array([1, 0, 0])
        theta = np.arccos(np.dot(north_vector, calsource_vector) / (np.linalg.norm(north_vector) * np.linalg.norm(calsource_vector)))
        print('theta', theta.shape)
        
        rotation_matrix = np.zeros((theta.shape[0], 3, 3))
        rotation_matrix[:, 0, 0] = np.cos(theta)
        rotation_matrix[:, 0, 1] = -np.sin(theta)
        rotation_matrix[:, 1, 0] = np.sin(theta)
        rotation_matrix[:, 1, 1] = np.cos(theta)
        rotation_matrix[:, 2, 2] = 1
        
        ### Compute ned_vector in xyz coordinates
        xyz_vectors = np.zeros((3, theta.shape[0]))
        for itime in range(theta.shape[0]):
            xyz_vectors[:, itime] = np.dot(rotation_matrix[itime], ned_vector[:, itime])

        return xyz_vectors
        
    def calsource_orientation(self, vector_1_2_xyz):
        """Calsource orientation.
        
        Method to compute the orientation of the calsource in the XYZ coordinates.

        Parameters
        ----------
        vector_1_2_xyz : array_like
            Vector used to compute the orientation of the calsource in the XYZ coordinates.

        Returns
        -------
        angles : array_like
            Orientation angles of the calsource in the XYZ coordinates.
        """        
        
        ex, ey, ez = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
        
        angles = np.zeros(vector_1_2_xyz.shape)

        for itime_index in range(vector_1_2_xyz.shape[1]):
            # Rotation around the x-axis
            angles[0, itime_index] = np.arccos(np.dot(vector_1_2_xyz[:, itime_index], ey) / np.sqrt((np.dot(vector_1_2_xyz[:, itime_index], ey))**2 + np.dot(vector_1_2_xyz[:, itime_index], ez)**2))
            # Rotation around the y-axis
            #! I don't think that we can have this information from the GPS data.
            angles[1, itime_index] = 0
            # Rotation around the z-axis
            angles[2, itime_index] = np.arccos(np.dot(vector_1_2_xyz[:, itime_index], ex) / np.sqrt((np.dot(vector_1_2_xyz[:, itime_index], ex))**2 + np.dot(vector_1_2_xyz[:, itime_index], ey)**2))
            
        return angles