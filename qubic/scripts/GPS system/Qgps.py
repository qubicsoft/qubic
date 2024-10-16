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
        
        ### Convert the data from GPS system into a dictionary
        self.gps_data = self.read_gps_bindat(gps_data_file_path)
        
        ### Import all the GPS data from the dictionary and convert them in proper units
        self.timestamp = self.gps_data['timestamp']
        
        # rpN, rpE, rpD give the relative position of the antenna 2 wrt base antenna in North, East, Down coordinates
        self.rpN = np.array(self.gps_data['rpN']) / 10000                       # in m
        self.rpE = np.array(self.gps_data['rpE']) / 10000                       # in m
        self.rpD = np.array(self.gps_data['rpD']) / 10000                       # in m
        
        # roll give the angle between antenna 2 - antenna 1 vector and the North axis
        self.roll = np.radians(np.array(self.gps_data['roll']) / 1000)          # in rad
        
        # yaw give the angle between antenna 2 - antenna 1 vector and the horizontal plane
        #! Need to double check that
        self.yaw = np.radians(np.array(self.gps_data['yaw']) / 1000)            # in rad
        
        self.pitchIMU = np.radians(np.array(self.gps_data['pitchIMU']) / 1000)  # in rad
        self.rollIMU = np.radians(np.array(self.gps_data['rollIMU']) / 1000)    # in rad
        self.temperature = np.array(self.gps_data['temperature']) / 10          # in Celsius
        self.checksum = np.array(self.gps_data['checksum'])
        
        ### Build datetime array
        self.datetime = self.create_datetime_array(self.timestamp)
        
        ### 
        observation_indices = self.observation_indices(self.datetime, observation_date)
        
        ### Compute position of antennas 1 & 2 and calibration source in North, East, Down cooridnates
        self.position_antenna2 = np.array([self.position_antenna_2(self.base_antenna_position, self.datetime[index]) for index in observation_indices])
        self.position_antenna1 = np.array([self.position_wrt_antenna_2(self.distance_between_antennas, self.datetime[index]) for index in observation_indices]) + self.position_antenna2
        self.position_calsource = np.array([self.position_wrt_antenna_2(self.distance_calsource, self.datetime[index]) for index in observation_indices]) + self.position_antenna2
    
    def read_gps_bindat(self, filename):
        '''
        read the binary data acquired from by RTK simple broadcast
        '''
        if not os.path.isfile(filename):
            print('ERROR!  File not found: %s' % filename)
            return

        # read the data
        h = open(filename,'rb')
        bindat = h.read()
        h.close()

        # interpret the binary data
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
        ValueError
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
        except IndexError:
            raise ValueError('ERROR! The date you chose is not in the data.')
        
    def observation_indices(self, datetime, observation_date):
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
         
        # If we give an unique observation date
        try :
            observation_date.shape
        except:
            return self.datetime_to_index(datetime, observation_date)
        
        # If we give a starting and stoping date
        if observation_date.shape[0] == 2:
            start_index = self.datetime_to_index(datetime, observation_date[0])
            end_index = self.datetime_to_index(datetime, observation_date[1])
            return np.arange(start_index, end_index, 1)
        
        else:
            raise ValueError('ERROR! Please choose a correct shape for the date: 1 or 2.')

    def position_antenna_2(self, base_antenna_position, observation_date):
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
        
        ### Find the index associated to the chosen date
        date_index = self.observation_indices(self.datetime, observation_date)
        
        ### Call the base antenna position in North, East, Down coordinates
        rpN_base, rpE_base, rpD_base = base_antenna_position
        
        ### Compute the position of the antenna 2 in North, East, Down coordinates
        rpN_antenna_2 = self.rpN[date_index] + rpN_base
        rpE_antenna_2 = self.rpE[date_index] + rpE_base
        rpD_antenna_2 = self.rpD[date_index] + rpD_base
        
        return np.array([rpN_antenna_2, rpE_antenna_2, rpD_antenna_2])
    
    def position_wrt_antenna_2(self, distance_w_antenna_2, observation_date):
        """Position wrt antenna 2.
        
        General fonction to compute the position of any point located on the straight line formed by the antenna 1 - anntenna 2 vector, wrt antenna 2 in North, East, Down coordinates.

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
        
        ### Find the index associated to the chosen date
        date_index = self.observation_indices(self.datetime, observation_date)
        
        ### Call the angles associated with antenna 1 - antenna 2 vector
        roll = self.roll[date_index]
        yaw = self.yaw[date_index]
        
        ### Compute the position of the antenna 1 wrt antenna 2 in North, East, Down coordinates
        _rpN = distance_w_antenna_2 * np.cos(roll) * np.sin(yaw + np.pi/2)
        _rpE = distance_w_antenna_2 * np.sin(roll) * np.sin(yaw + np.pi/2)
        _rpD = distance_w_antenna_2 * np.cos(yaw + np.pi/2)
        
        return np.array([_rpN, _rpE, _rpD])
        
    