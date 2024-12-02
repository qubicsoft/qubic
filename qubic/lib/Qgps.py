import os
import struct

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go

import datetime as dt

class GPStools:
    
    def __init__(self, gps_data_path):
        """GPSTools class.
        
        Class to handle the GPS data and to perform the operations neccessary to use them.

        Parameters
        ----------
        gps_data_path : string or dict
            Path of the GPS binary file or dictionary containing the GPS data.
        """

        ### Convert the data from GPS system into a dictionary
        self.gps_data = self.get_gps_data(gps_data_path)
        
       ### Extract all the GPS data from the dictionary and convert them in proper units
        self.extract_gps_data(self.gps_data)

    def get_gps_data(self, gps_data_path):
        """GPS Data.
        
        Method used to build the dictionnary that contains the GPS Data, either by converting the binary file or by using the given dictionary.

        Parameters
        ----------
        gps_data_path : string or dict
            Path of the GPS binary file or dictionary containing the GPS data.

        Returns
        -------
        gps_data: dict
            Dictionary containing the GPS data.

        Raises
        ------
        ValueError
            If the GPS data file does not exist.
        """
        
        ### Convert the data from GPS system into a dictionary
        if type(gps_data_path) == str:
            if os.path.isfile(gps_data_path):
                return self.read_gps_bindat(gps_data_path)
            else:
                raise ValueError("The GPS data file does not exist")
        else:
            return gps_data_path

    def read_gps_bindat(self, gps_data_path):
        """GPS binary data.
        
        Method to convert the binary data acquired from by RTK simple broadcast into readable format and store them in a dictionary.

        Parameters
        ----------
        gps_data_path : str
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
        
        if not os.path.isfile(gps_data_path):
            print('ERROR!  File not found: %s' % gps_data_path)
            return

        ### read the data
        h = open(gps_data_path,'rb')
        bindat = h.read()
        h.close()

        ### interpret the binary data
        fmt = '<Bdiiiiiiifi'
        nbytes = 45
        names = "STX,timestamp,rpN,rpE,rpD,roll,yaw,pitchIMU,rollIMU,temperature,checksum".split(',')
        data = {}
        for name in names:
            data[name] = []    

        index = 0
        while index+nbytes<len(bindat):
            packet = bindat[index:index+nbytes]
            dat_list = struct.unpack(fmt,packet)

            if len(dat_list)!=len(names):
                raise ValueError('ERROR:  Incompatible data.')

            for datindex,name in enumerate(names):
                data[name].append(dat_list[datindex])

            index += nbytes

        return data
    
    def extract_gps_data(self, gps_data):
        """Extract GPS data.
        
        Method to extract the GPS data from the dictionary and convert them in proper units.
        It also converts the Down data into Up coordinates (usual vectical axis in cartesian coordinates).

        Parameters
        ----------
        gps_data : dict
            Dictionary containing the GPS data.
        """
        
        ### Build datetime array
        self._timestamp = np.array(gps_data['timestamp'])
        self._datetime = self.create_datetime_array(self._timestamp)
        
        ### rpN, rpE, rpD give the relative position of the antenna 2 wrt base antenna in North, East, Down coordinates
        self.rpN = np.array(gps_data['rpN']) / 10000                           # in m
        self.rpE = np.array(gps_data['rpE']) / 10000                           # in m
        #! - sign to switch from Down to Up axis, which is more usual
        self.rpD = - np.array(gps_data['rpD']) / 10000                         # in m
        
        ### roll give the angle between antenna 2 - antenna 1 vector and the North axis
        self.roll = np.radians(np.array(gps_data['roll'])) / 1000              # in rad
        
        ### yaw give the angle between antenna 2 - antenna 1 vector and the horizontal plane
        self.yaw = np.radians(np.array(gps_data['yaw'])) / 1000                # in rad
        
        ### Other GPS parameters, not used yet
        self._pitchIMU = np.radians(np.array(gps_data['pitchIMU'])) / 1000      # in rad
        self.rollIMU = np.radians(np.array(gps_data['rollIMU'])) / 1000        # in rad
        self._temperature = np.array(gps_data['temperature']) / 10              # in Celsius
        self._checksum = np.array(gps_data['checksum'])
    
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
        
    def plot_gps_position_vector(self, index_start=0, index_stop=-1):
        """Plot GPS position vector.
    
        Plot the position vector of the antenna in the North, East and Up directions.

        Parameters
        ----------
        index_start : int, optional
            First observation index, by default 0
        index_stop : int, optional
            Last observation index, by default -1
        """
        
        fig, ax = plt.subplots(figsize = (15,5))

        ax.set_xlabel('Date')
        ax.set_ylabel('Position (m)')
        ax.plot(self._datetime[index_start:index_stop], self.rpN[index_start:index_stop], color = 'red', label = 'North component')
        ax.plot(self._datetime[index_start:index_stop], self.rpE[index_start:index_stop], color = 'blue', label = 'East component')
        ax.plot(self._datetime[index_start:index_stop], self.rpD[index_start:index_stop], color = 'green', label = 'Up component')

        fig.tight_layout() 
        ax.set_title("Position Vector Components")
        fig.legend()
        plt.show()

    def plot_gps_angles(self, index_start=0, index_stop=-1):
        """Plot GPS angles.
        
        Plot the roll and yaw angles.

        Parameters
        ----------
        index_start : int, optional
            First observation index, by default 0
        index_stop : int, optional
            Last observation index, by default -1
        """        

        fig, ax = plt.subplots(figsize = (15,5))
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Angles (rad)')

        ax.plot(self._datetime[index_start:index_stop], self.roll[index_start:index_stop], color = 'pink', label = 'Roll angle')
        ax.plot(self._datetime[index_start:index_stop], self.yaw[index_start:index_stop], color = 'brown', label = 'Yaw angle')

        fig.tight_layout() 
        ax.set_title("Angles")
        fig.legend()
        plt.show()
        
    def plot_gps_data(self, index_start=0, index_stop=-1):
        """Plot GPS data.
        
        Plot the position vector and the angles.

        Parameters
        ----------
        index_start : int, optional
            First observation index, by default 0
        index_stop : int, optional
            Last observation index, by default -1
        """

        fig, ax1 = plt.subplots(figsize = (15,5))

        color_a = 'tab:pink'
        color_r = 'tab:red'
        color_b = 'tab:blue'
        color_d = 'tab:green'
        color_c = 'tab:brown'
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Position Vector Components (m)', color = color_r)
        ax1.plot(self._datetime[index_start:index_stop], self.rpN[index_start:index_stop], color = color_a, label = 'North component')
        ax1.plot(self._datetime[index_start:index_stop], self.rpE[index_start:index_stop], color = color_b, label = 'East component')
        ax1.plot(self._datetime[index_start:index_stop], self.rpD[index_start:index_stop], color = color_d, label = 'Up component')
        ax1.axvline(x=self._datetime[index_start:index_stop], ymin=np.min(self.rpN[index_start:index_stop]), ymax=np.max(self.rpN[index_start:index_stop]), color='grey', linestyle='--', linewidth=1, label='Start')

        ax2 = ax1.twinx()

        ax2.plot(self._datetime[index_start:index_stop], self.roll[index_start:index_stop], color = color_r, label = 'Roll angle')
        ax2.plot(self._datetime[index_start:index_stop], self.yaw[index_start:index_stop], color = color_c, label = 'Yaw angle')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Angles (rad)', color = color_a)


        fig.tight_layout()
        ax1.set_title("Position vector components")
        fig.legend()
        plt.show()
    
class GPSAntenna(GPStools):
    
    def __init__(self, gps_data_path, distance_between_antennas):
        """GPSAntenna class.
        
        Class to compute the position of the two GPS antennas.

        Parameters
        ----------
        gps_data_path : string or dict
            Path of the GPS binary file or dictionary containing the GPS data.
        distance_between_antennas : float
            Distance between the two antennas, it's necessary to compute the position of antenna 1.
        """
        
        ### Initialize the GPSTools class
        GPStools.__init__(self, gps_data_path)
        
        ### Fixed parameters
        self.base_antenna_position = np.array([0, 0, 0])
        self.distance_between_antennas = distance_between_antennas
        
        ### Compute position of antennas 1 & 2 and calibration source in North, East, Down cooridnates
        self.position_antenna2 = self.get_position_antenna_2(self.base_antenna_position)
        self.position_antenna1 = self.get_position_antenna_1(self.distance_between_antennas)     
        
    def get_position_antenna_2(self, base_antenna_position = np.array([0, 0, 0])):
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
    
    def get_position_antenna_1(self, distance_between_antennas):
        """Position wrt antenna 2.
        
        General fonction to compute the position of any point located on the straight line formed by the antenna 1 - anntenna 2 vector.
        In the code, we used it only to compute the position of the antenna 1.
        
        Be careful, yaw is not the usual theta angle in sphercial cooridinates (i.e. the latitude angle): it corresponds to the elevation angle. 
        It is why we need to add np.pi/2 in the conversion formulas.

        Parameters
        ----------
        distance_between_antennas : float
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
        _rpN = distance_between_antennas * np.cos(self.roll) * np.sin(np.pi/2 - self.yaw)
        _rpE = distance_between_antennas * np.sin(self.roll) * np.sin(np.pi/2 - self.yaw)
        _rpD = distance_between_antennas * np.cos(np.pi/2 - self.yaw)
        
        return np.array([_rpN, _rpE, _rpD]) + self.position_antenna2

class GPSCalsource(GPSAntenna):
    
    def __init__(self, gps_data, position_ini_antenna1, position_ini_antenna2, position_ini_calsource, distance_between_antennas, observation_date, position_qubic = np.array([0, 0, 0]), observation_only = False):
        
        GPSAntenna.__init__(self, gps_data, distance_between_antennas)
        
        ### Fixed parameters
        self.base_antenna_position = np.array([0, 0, 0])
        self.position_ini_antenna1 = position_ini_antenna1
        self.position_ini_antenna2 = position_ini_antenna2
        self.position_ini_calsource = position_ini_calsource
        self.observation_date = observation_date
        self.position_qubic = position_qubic
        
        ### Import all the GPS data from the dictionary and convert them in proper units
        self.timestamp = self._timestamp.reshape(-1)
        # Build datetime array
        self.datetime = self.create_datetime_array(self.timestamp)
        # Build observation variables : index, time, datetime
        self.observation_indices = self.get_observation_indices(self.datetime, self.observation_date).reshape(-1)
        print('The observation indices are : ', self.observation_indices)
        self.observation_time = self.timestamp[self.observation_indices].reshape(-1)
        self.observation_datetime = self.datetime[self.observation_indices].reshape(-1)
        # Keep only data during observatin time
        if observation_only:
            self._get_observation_data(self.observation_indices)
        
        ### Compute position of antennas 1 & 2 and calibration source in North, East, Down cooridnates
        self.distance_between_antennas = np.linalg.norm(self.position_ini_antenna2 - self.position_ini_antenna1)
        self.position_antenna2 = self.get_position_antenna_2(self.base_antenna_position)
        self.position_antenna1 = self.get_position_antenna_1(self.distance_between_antennas) 
        
        ### Compute the vectors between the calibration source and QUBIC, and the vector between the antennas in NED coordinates
        self.vector_1_2_ini =  self.position_ini_antenna2 - self.position_ini_antenna1
        self.vector_1_2 = self.position_antenna2 - self.position_antenna1
        self.vector_calsource_qubic_ini = self.position_qubic - self.position_ini_calsource

        ### Compute the calibration source orientation vector
        self.rotation_instance = self.compute_rotation(self.vector_1_2, self.vector_1_2_ini[:, None])
        self.vector_calsource_orientation = self.apply_rotation(self.vector_calsource_qubic_ini, self.rotation_instance)
        
        ### Compute the position of the calibration source in cartesian and azimutal coordinates
        self.position_calsource = self.get_calsource_position(self.position_ini_antenna2[:, None], self.position_ini_calsource[:, None], self.position_antenna2) 
        self.position_calsource_azel = self.cartesian_to_azel(self.position_calsource)
                
    def _get_observation_data(self, observation_indices):

        ### rpN, rpE, rpD give the relative position of the antenna 2 wrt base antenna in North, East, Down coordinates
        self.rpN = self.rpN[observation_indices].reshape(-1)                                      # in m
        self.rpE = self.rpE[observation_indices].reshape(-1)                                      # in m
        self.rpD = self.rpD[observation_indices].reshape(-1)                                      # in m

        ### roll give the angle between antenna 2 - antenna 1 vector and the North axis
        self.roll = self.roll[observation_indices].reshape(-1)                                    # in rad

        ### yaw give the angle between antenna 2 - antenna 1 vector and the horizontal plane
        self.yaw = self.yaw[observation_indices].reshape(-1)                                      # in rad

        ### Other GPS parameters, not used yet
        self.pitchIMU = self._pitchIMU[observation_indices].reshape(-1)                           # in rad
        self.rollIMU = self.rollIMU[observation_indices].reshape(-1)                              # in rad
        self.temperature = self._temperature[observation_indices].reshape(-1)                     # in Celsius
        self.checksum = self._checksum[observation_indices].reshape(-1)
        
    def compute_rotation(self, v1, v2):
        """Rotation.
        
        Compute the rotation instance from Spipy.spatial.transform.Rotation, that transforms v1 to v2.

        Parameters
        ----------
        v1 : array_like
            Fisrt vector.
        v2 : array_like
            Second vector.

        Returns
        -------
        rotationon_instance : Rotation
            Rotation instance from Spipy.spatial.transform.Rotation.
        """
        
        ### Normalize the vectors and compute the dot product and cross product
        v1_normalized = v1 / np.linalg.norm(v1, axis=0)
        v2_normalized = v2 / np.linalg.norm(v2, axis=0)
        dot_product = np.sum(v1_normalized * v2_normalized, axis=0)
        cross_product = np.cross(v2_normalized.T, v1_normalized.T).T
        
        ### Define the rotation axis and angle between the vectors
        rotation_axis = cross_product / np.linalg.norm(cross_product, axis=0)
        angle = np.arctan2(cross_product, dot_product)
        
        ### Build the scipy Rotation instance
        rotation_instance = R.from_rotvec((angle * rotation_axis).T)
        
        return rotation_instance  
          
    def apply_rotation(self, v, rotation_instance):
        """Apply rotation.
        
        Apply the rotation instance to the vector v.

        Parameters
        ----------
        v : array_like
            Vector to rotate.
        rotation_instance : Rotation
            Rotation instance from Spipy.spatial.transform.Rotation.
            
        Returns
        -------
        rotated_vector : array_like
            Rotated vector.
        """
        
        ### Rotate the vector using the rotation instance
        rotated_vector = rotation_instance.apply(v)
        return rotated_vector.T
    
    def get_calsource_position(self, position_ini_antenna, position_ini_calsource, position_antenna):
        """Calsource position.
        
        Compute the position of the calibration source, using the translation between the initial and current position of one antenna.
        This translation is then applied on the initial position of the calibration source.
        
        Parameters
        ----------
        position_ini_antenna : array_like
            Initial position of the antenna.
        position_ini_calsource : array_like
            Initial position of the calibration source.
        position_antenna : array_like
            Position of the antenna.

        Returns
        -------
        position_calsource : array_like
            Position of the calibration source.
        """
        
        ### Compute the translation between the initial and current position of the antenna
        translation = position_antenna - position_ini_antenna
        
        ### Apply the translation to the initial position of the calibration source
        return position_ini_calsource + translation
        
    def cartesian_to_azel(self, cartesian_position):
        
        x, y, z = cartesian_position

        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)
        phi = np.arccos(z / r)

        azimuth = np.degrees(theta)
        elevation = np.degrees(phi)

        return np.array([azimuth, elevation])
        
    def plot_vector_plotly(self, fig, pos, vector, color='blue', name='vector', show_arrow=True, arrow_size = 0.2):
        start = pos
        end = pos + vector
        # Vecteur normalisé pour l'effet pointe
        norm = np.linalg.norm(vector)

        if norm == 0:
            return
        vector_unit = vector / norm

        # Ajouter le vecteur principal
        fig.add_trace(go.Scatter3d(
                x=[start[0], end[0]], 
                y=[start[1], end[1]], 
                z=[start[2], end[2]],
                mode='lines',
                line=dict(color=color, width=2),
                name=name,
                text=[name, name],
                hovertemplate=(
                        "<b>%{text}</b><br>"
                        "X: %{x:.2f}<br>"
                        "Y: %{y:.2f}<br>"
                        "Z: %{z:.2f}<extra></extra>" 
                        )
            ))

        if show_arrow:
            if vector_unit[0] != 0 or vector_unit[1] != 0:
                # General case: construct perpendicular vectors
                ortho1 = np.cross(vector_unit, [0, 0, 1])
            else:
                # Special case: vector is along z-axis
                ortho1 = np.cross(vector_unit, [1, 0, 0])
            
            ortho1 /= np.linalg.norm(ortho1)  # Normalize the first orthogonal vector
            ortho2 = np.cross(vector_unit, ortho1)  # Compute the second orthogonal vector
            ortho2 /= np.linalg.norm(ortho2)  # Normalize the second orthogonal vector

            # Base of the arrowhead
            tip_base = np.array(end) - arrow_size * vector_unit

            # Compute the points for the arrowhead
            point1 = tip_base + arrow_size * 0.5 * ortho1
            point2 = tip_base - arrow_size * 0.5 * ortho1
            # point3 = tip_base + arrow_size * 0.5 * ortho2
            # point4 = tip_base - arrow_size * 0.5 * ortho2

            # Add the arrowhead segments
            for point in [point1, point2]: #, point3, point4]:
                fig.add_trace(go.Scatter3d(
                    x=[end[0], point[0]],
                    y=[end[1], point[1]],
                    z=[end[2], point[2]],
                    mode='lines',
                    line=dict(color=color, width=5),
                    showlegend=False,
                    hovertemplate=(
                        "X: %{x:.2f}<br>"
                        "Y: %{y:.2f}<br>"
                        "Z: %{z:.2f}<extra></extra>" 
                        )
                ))
    
    def plot_calsource_deviation_plotly(self, index):

        fig = go.Figure()

        points_data = [
            (self.position_antenna1[:, index], 'darkblue', 'square', 'Antenna 1'),
            (self.position_antenna2[:, index], 'darkblue', 'diamond', 'Antenna 2'),
            (self.position_calsource[:, index], 'darkred', 'x', 'Calibration Source'),
            (self.position_ini_antenna1, 'blue', 'square', 'Initial Antenna 1'),
            (self.position_ini_antenna2, 'blue', 'diamond', 'Initial Antenna 2'),
            (self.position_ini_calsource, 'red', 'x', 'Initial Calibration Source'),
            (self.base_antenna_position, 'pink', 'circle', 'Base Antenna'),
            (self.position_qubic, 'black', 'circle', 'QUBIC')
        ]
        
        vectors_data = [
            (self.position_antenna1[:, index], self.vector_1_2[:, index], 'darkblue', 'Vector Antenna 1 to 2'),
            (self.position_ini_antenna1, self.vector_1_2_ini, 'blue', 'Initial Vector Antenna 1 to 2'),
            (self.position_calsource[:, index], self.vector_calsource_orientation[:, index], 'darkred', 'Vector Calibration Source'),
            (self.position_ini_calsource, self.vector_calsource_qubic_ini, 'red', 'Initial Vector Calibration Source')
        ]

        ### Plot points as 3D scatter
        for pos, color, symbol, name in points_data:
            fig.add_trace(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode='markers',
                marker=dict(size=5, color=color, symbol=symbol),
                text=[name],
                name=name,
                hovertemplate=(
                        "<b>%{text}</b><br>"
                        "X: %{x:.2f}<br>"
                        "Y: %{y:.2f}<br>"
                        "Z: %{z:.2f}<extra></extra>" 
                        )
            ))

        ### Plot vectors
        for pos, vector, color, name in vectors_data:
            self.plot_vector_plotly(fig, pos, vector, color=color, name=name, show_arrow=True)
            
        # Get min/max coordinates of all points and vectors
        all_points = np.vstack([
            self.position_antenna1[:, index],
            self.position_antenna2[:, index],
            self.position_calsource[:, index],
            self.position_ini_antenna1,
            self.position_ini_antenna2,
            self.position_ini_calsource,
            self.base_antenna_position,
            self.position_qubic
        ])

        # Add vector endpoints
        vector_endpoints = np.vstack([
            self.position_antenna1[:, index] + self.vector_1_2[:, index],
            self.position_ini_antenna1 + self.vector_1_2_ini,
            self.position_calsource[:, index] + self.vector_calsource_orientation[:, index], 
            self.position_ini_calsource - self.vector_calsource_qubic_ini
        ])

        all_points = np.vstack([all_points, vector_endpoints])

        # Calculate characteristic scale
        margin = 0.2
        min_coords = np.min(all_points, axis=0)
        max_coords = np.max(all_points, axis=0)
        plot_range = max_coords - min_coords
        char_scale = np.max(plot_range)  # Use largest range as characteristic scale
        
        # Calculate center points
        center = (max_coords + min_coords) / 2

        # Set consistent limits using characteristic scale
        limits_min = center - (1 + margin) * char_scale/2
        limits_max = center + (1 + margin) * char_scale/2

        # Update layout with calculated limits
        fig.update_layout(
            width=1200,
            height=800,
            scene=dict(
                xaxis=dict(range=[limits_min[0], limits_max[0]], title='North'),
                yaxis=dict(range=[limits_min[1], limits_max[1]], title='East'),
                zaxis=dict(range=[limits_min[2], limits_max[2]], title='Down'),
                aspectmode='cube',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            title=f'Calibration source - Position and Orientation - {self.datetime[index]}',
            showlegend=True,
            legend=dict(x=1.1, y=0.5)
        )


        return fig
        
    def plot_calsource_deviation(self, index):
        
        ### Initialize the 3d figure
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        ### Plot antenna1, antenna 2, base antenna, qubic and the calibration source
        ax.scatter(self.position_antenna1[0, index], self.position_antenna1[1, index], self.position_antenna1[2, index], color='darkblue', marker='s', s=100, label='Antenna 1')
        ax.scatter(self.position_antenna2[0, index], self.position_antenna2[1, index], self.position_antenna2[2, index], color='darkblue', marker='^', s=100, label='Antenna 2')
        ax.scatter(self.position_calsource[0, index], self.position_calsource[1, index], self.position_calsource[2, index], color='darkred', marker='*', s=100, label='Calibration Source')
        ax.scatter(self.position_ini_antenna1[0], self.position_ini_antenna1[1], self.position_ini_antenna1[2], color='b', marker='s', s=100, label='Initial Antenna 1')
        ax.scatter(self.position_ini_antenna2[0], self.position_ini_antenna2[1], self.position_ini_antenna2[2], color='b', marker='^', s=100, label='Initial Antenna 2')
        ax.scatter(self.position_ini_calsource[0], self.position_ini_calsource[1], self.position_ini_calsource[2], color='r', marker='*', s=100, label='Initial Calibration Source')
        ax.scatter(self.base_antenna_position[0], self.base_antenna_position[1], self.base_antenna_position[2], color='k', s=100, label='Base Antenna')
        ax.scatter(self.position_qubic[0], self.position_qubic[1], self.position_qubic[2], color='pink', marker='o', s=100, label='QUBIC')

        ### Plot the vector between antenna 1 and antenna 2
        ax.quiver(self.position_antenna1[0, index], self.position_antenna1[1, index], self.position_antenna1[2, index], 
                    self.vector_1_2[0, index],
                    self.vector_1_2[1, index],
                    self.vector_1_2[2, index],
                    color='darkblue', arrow_length_ratio=0.1, linewidth=2, label='Vector Antenna 1 to 2')
        ax.quiver(self.position_ini_antenna1[0], self.position_ini_antenna1[1], self.position_ini_antenna1[2], 
                    self.vector_1_2_ini[0],
                    self.vector_1_2_ini[1],
                    self.vector_1_2_ini[2],
                    color='b', arrow_length_ratio=0.1, linewidth=2, label='Initial Vector Antenna 1 to 2')
        
        ### Plot the vector between QUBIC and the calibration source
        ax.quiver(self.position_calsource[0, index], self.position_calsource[1, index], self.position_calsource[2, index],
                  self.vector_calsource_orientation[0, index], self.vector_calsource_orientation[1, index], self.vector_calsource_orientation[2, index],
                  color='darkred', arrow_length_ratio=0.1, linewidth=2, label='Vector Calibration Source Deviation')
        ax.quiver(self.position_ini_calsource[0], self.position_ini_calsource[1], self.position_ini_calsource[2], 
                    -self.vector_calsource_qubic_ini[0], 
                    -self.vector_calsource_qubic_ini[1],
                    -self.vector_calsource_qubic_ini[2],
                    color='red', arrow_length_ratio=0.1, linewidth=2, label='Vector QUBIC to Calibration Source')
        
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5]) 
        ax.set_zlim([-1, 1])
        ax.set_xlabel('North', fontsize=12, labelpad=10)
        ax.set_ylabel('East', fontsize=12, labelpad=10)
        ax.set_zlabel('Down', fontsize=12, labelpad=10)

        ax.set_title(f'Calibration source - Position and Orientation - {self.datetime[index]}', fontsize=16, pad=20)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.legend() 
 
    def plot_angle_3d(self, ax, origin, v1, v2, angle, num_points=1000, radius=0.5, **kwargs):
        """Plot angle 3d.
        
        General function to plot a 3D angle between two vectors v1 and v2.

        Parameters
        ----------
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D
            Matplotlib 3D axis
        origin : array_like
            Position of the origin of the angle.
        v1 : array_like
            Vector 1.
        v2 : array_like
            Vector 2.
        angle : float
            Angle between v1 and v2, in radians.
        num_points : int, optional
            Number of points used to plot the angle, by default 100
        radius : float, optional
            Radiius from the origin at which the angle is plotted, by default 0.5
            
        Other Parameters
        ----------------
        kwargs : optional
            Any kwarg for plt.plot()
        """       
        
        #! This function can be moved in a more genral repostitory
        
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        # Create orthonormal basis for the plane containing v1 and v2
        normal = np.cross(v1_norm, v2_norm)
        if np.allclose(normal, 0):
            # Vectors are parallel, choose an arbitrary perpendicular vector
            normal = np.array([1, 0, 0]) if np.allclose(v1_norm, [0, 1, 0]) else np.cross(v1_norm, [0, 1, 0])
        normal = normal / np.linalg.norm(normal)
        
        angles = np.linspace(0, angle, num_points)
        arc_points = np.zeros((num_points, 3))
        
        for i, theta in enumerate(angles):
            rotated = v1_norm * np.cos(theta) + \
                    np.cross(normal, v1_norm) * np.sin(theta) + \
                    normal * np.dot(normal, v1_norm) * (1 - np.cos(theta))
            arc_points[i] = origin + radius * rotated
            
        ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], **kwargs)
               
    def get_calsource_orientation(self, vector_cal, vector_cal_qubic):
        r"""Calsource orientation.
        
        Method to compute the orientation of the calsource.
        - angles[0] is the angle around vector_cal. To compute it, let's define :
        
        .. math::
            \vec{n_{cal}} = \frac{\vec{V_{base \rightarrow cal}}}{||\vec{V_{base \rightarrow cal}}||}, \vec{e_D} = (0, 0, 1) and \vec{V_{ortho}} = \vec{V_{base \rightarrow cal}} \times \vec{e_D}
            
            Projection of the self.vector_cal on the orthogonal plane to vector_cal and down axis :
        
        .. math::
            \vec{V_{1 \rightarrow 2, cal}} = \vec{V_{1 \rightarrow 2}} - \frac{\vec{V_{1 \rightarrow 2}} \cdot \vec{n_{cal}}}{\vec{n_{cal}}\cdot \vec{n_{cal}}} \cdot \vec{n_{cal}}
            
            Then : 
        .. math ::
            \cos(angle[0]) = \frac{\vec{V_{base \rightarrow cal} \cdot \vec{V_{ortho}}}}{||\vec{V_{base \rightarrow cal}}||\cdot||\vec{V_{ortho}}||}
        
        - angles[1] is the angle around the axis orthogonal to vector_cal and down axis. It is fixed to 0 as we can't compute it with the system.
        
        - angles[2] is the angle around the down axis. To compute it, let's define :

            Projection of the self.vector_cal on the down axis :
        
        .. math::
            \vec{V_{1 \rightarrow 2, D}} = \vec{V_{1 \rightarrow 2}} - \frac{\vec{V_{1 \rightarrow 2}} \cdot \vec{e_{D}}}{\vec{e_{D}} \cdot \vec{e_{D}}} \cdot \vec{e_{D}}
            
            Then : 
        .. math ::
            \cos(angle[0]) = \frac{\vec{V_{1 \rightarrow 2, D}} \cdot \vec{V_{ortho}}}}{||\vec{V_{1 \rightarrow 2, D}}||\cdot||\vec{V_{ortho}}||}

        Parameters
        ----------
        self.vector_cal : array_like
            Vector between antenna 1 and 2.
        vector_cal : array_like
            Vector between QUBIC and the calibration source.

        Returns
        -------
        angles : array_like
            Orientation angles of the calsource.
        """
        
        angles = np.zeros(vector_cal.shape)    
        
        ### Down vector
        ed = np.array([0, 0, 1])
        
        ### Direction of calsource vector
        self.n_cal_qubic = vector_cal_qubic / np.linalg.norm(vector_cal_qubic)
        
        ### Build orthogonal vector to vector_cal
        self.vector_ortho_vert = np.cross(self.n_cal_qubic, ed, axisa=0).T
        self.vector_ortho_horiz = np.cross(self.n_cal_qubic, self.vector_ortho_vert, axisa=0, axisb=0).T
        
        ### Projections of self.vector_1_2 on planes ortho to vector_cal
        self.vector_1_2_calsource_proj = vector_cal - (np.sum(vector_cal * self.n_cal_qubic, axis=0) / np.sum(self.n_cal_qubic * self.n_cal_qubic, axis=0))[None, :] * self.n_cal_qubic
        self.vector_1_2_vertical_proj = vector_cal - (np.sum(vector_cal * self.vector_ortho_vert, axis=0) / np.sum(self.vector_ortho_vert * self.vector_ortho_vert, axis=0))[None, :] * self.vector_ortho_vert
        self.vector_1_2_horizontal_proj = vector_cal - (np.sum(vector_cal * self.vector_ortho_horiz, axis=0) / np.sum(self.vector_ortho_horiz * self.vector_ortho_horiz, axis=0))[None, :] * self.vector_ortho_horiz

        ### Compute the angles
        angles[0] = np.arccos(np.sum(self.vector_1_2_calsource_proj * self.vector_ortho_vert, axis=0) / (np.linalg.norm(self.vector_1_2_calsource_proj, axis=0) * np.linalg.norm(self.vector_ortho_vert, axis=0)))
        angles[1] = np.arccos(np.sum(self.vector_1_2_vertical_proj * self.vector_ortho_horiz, axis=0) / (np.linalg.norm(self.vector_1_2_vertical_proj, axis=0) * np.linalg.norm(self.vector_ortho_horiz, axis=0)))
        angles[2] = np.arccos(np.sum(self.vector_1_2_horizontal_proj * self.vector_ortho_vert, axis=0) / (np.linalg.norm(self.vector_1_2_horizontal_proj, axis=0) * np.linalg.norm(self.vector_ortho_vert, axis=0)))

        return angles
        
    def plot_system(self, index):
        """Plot the system.
        
        Function the plot the all system : antenna1, antenna 2, base antenna and the calibration source, positions, the associated vectors and the calsource orientation angles.

        Parameters
        ----------
        index : int, optional
            Time index at which you want to make the plot, by default 0
        """              
        
        #* I don't want to add too arguments now, as it is a very specific function, associated with the GPS instance

        ### Initialize the 3d figure
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ### Get min/max coordinates of all points and vectors
        all_points = np.vstack([
            self.position_antenna1[:, index],
            self.position_antenna2[:, index], 
            self.base_antenna_position,
            self.position_qubic,
            self.position_calsource[:, index]
        ])

        min_coords = np.min(all_points, axis=0)
        max_coords = np.max(all_points, axis=0)

        # Add some margin (e.g. 20%)
        margin = 0.2
        plot_range = max_coords - min_coords
        limits_min = min_coords - margin * plot_range
        limits_max = max_coords + margin * plot_range
        
        # Scale all vectors consistently
        vector_scale = np.max(plot_range) * 0.5
        def scale_factor(vector):
            return vector * vector_scale / np.linalg.norm(vector)

        ### Plot antenna1, antenna 2, base antenna, qubic and the calibration source
        ax.scatter(self.position_antenna1[0, index], self.position_antenna1[1, index], self.position_antenna1[2, index], color='b', marker='s', s=100)
        ax.scatter(self.position_antenna2[0, index], self.position_antenna2[1, index], self.position_antenna2[2, index], color='b', marker='^', s=100)
        ax.scatter(self.base_antenna_position[0], self.base_antenna_position[1], self.base_antenna_position[2], color='k', s=100)
        ax.scatter(self.position_qubic[0], self.position_qubic[1], self.position_qubic[2], color='pink', marker='o', s=100)
        ax.scatter(self.position_calsource[0, index], self.position_calsource[1, index], self.position_calsource[2, index], color='r', marker='*', s=100)
        
        ### Plot the vector between antenna 1 and antenna 2
        ax.quiver(self.position_antenna1[0, index], self.position_antenna1[1, index], self.position_antenna1[2, index], 
                    self.vector_1_2[0, index],
                    self.vector_1_2[1, index],
                    self.vector_1_2[2, index],
                    color='b', arrow_length_ratio=0.1, linewidth=2)
        
        ### Projection of the vector between antenna 1 and antenna 2 on the plane orthogonal to Qubic-Calsource
        scale_vector_1_2_calesource = scale_factor(self.vector_1_2_calsource_proj[:, index])
        ax.quiver(self.position_calsource[0, index], self.position_calsource[1, index], self.position_calsource[2, index],
                  scale_vector_1_2_calesource[0], 
                  scale_vector_1_2_calesource[1], 
                  scale_vector_1_2_calesource[2],
                  color='darkblue', arrow_length_ratio=0., linewidth=1, linestyle='--', alpha=0.7)
        ax.quiver(self.position_calsource[0, index], self.position_calsource[1, index], self.position_calsource[2, index],
                  -scale_vector_1_2_calesource[0], 
                  -scale_vector_1_2_calesource[1], 
                  -scale_vector_1_2_calesource[2],
                  color='darkblue', arrow_length_ratio=0., linewidth=1, linestyle='--', alpha=0.7)
        
        ### Projection of the vector between antenna 1 and antenna 2 on Qubic-Calsource/ortho vertical plane
        scale_vector_1_2_vertical_proj = scale_factor(self.vector_1_2_vertical_proj[:, index])
        ax.quiver(self.position_calsource[0, index], self.position_calsource[1, index],  self.position_calsource[2, index],
                  scale_vector_1_2_vertical_proj[0],
                  scale_vector_1_2_vertical_proj[1],
                  scale_vector_1_2_vertical_proj[2],
                  color='royalblue', arrow_length_ratio=0., linewidth=1, linestyle='--', alpha=0.7)
        ax.quiver(self.position_calsource[0, index], self.position_calsource[1, index],  self.position_calsource[2, index],
                  -scale_vector_1_2_vertical_proj[0],
                  -scale_vector_1_2_vertical_proj[1],
                  -scale_vector_1_2_vertical_proj[2],
                  color='royalblue', arrow_length_ratio=0., linewidth=1, linestyle='--', alpha=0.7)
        
        ### Projection of the vector between antenna 1 and antenna 2 on Qubic-Calsource/ortho horizontal plane
        scale_vector_1_2_horizontal_proj = scale_factor(self.vector_1_2_horizontal_proj[:, index])
        ax.quiver(self.position_calsource[0, index], self.position_calsource[1, index],  self.position_calsource[2, index],
                  scale_vector_1_2_horizontal_proj[0], 
                  scale_vector_1_2_horizontal_proj[1], 
                  scale_vector_1_2_horizontal_proj[2],
                  color='turquoise', arrow_length_ratio=0., linewidth=1, linestyle='--', alpha=0.7)
        ax.quiver(self.position_calsource[0, index], self.position_calsource[1, index],  self.position_calsource[2, index],
                  -scale_vector_1_2_horizontal_proj[0], 
                  -scale_vector_1_2_horizontal_proj[1], 
                  -scale_vector_1_2_horizontal_proj[2],
                  color='turquoise', arrow_length_ratio=0., linewidth=1, linestyle='--', alpha=0.7)
        
        ### Plot the vector between QUBIC and the calibration source
        ax.quiver(self.position_qubic[0], self.position_qubic[1], self.position_qubic[2], 
                    self.vector_calsource_qubic[0, index], 
                    self.vector_calsource_qubic[1, index],
                    self.vector_calsource_qubic[2, index],
                    color='r', arrow_length_ratio=0.1, linewidth=2)
        
        ### Plot the 3 rotation axis
        # Plot the QUBIC-calsource axis
        scale_vector_calsource_qubic = scale_factor(self.vector_calsource_qubic[:, index])
        ax.quiver(self.position_calsource[0, index], self.position_calsource[1, index], self.position_calsource[2, index], 
                    scale_vector_calsource_qubic[0],
                    scale_vector_calsource_qubic[1],
                    scale_vector_calsource_qubic[2],
                    color='orange', arrow_length_ratio=0., linewidth=3, linestyle='--', alpha=0.7)

        # Plot the orthogonal vertical axis
        scale_vector_ortho_vert = scale_factor(self.vector_ortho_vert[:, index])
        ax.quiver(self.position_calsource[0, index], self.position_calsource[1, index], self.position_calsource[2, index], 
                    scale_vector_ortho_vert[0], 
                    scale_vector_ortho_vert[1], 
                    scale_vector_ortho_vert[2], 
                    color='grey', arrow_length_ratio=0., linewidth=3, linestyle='--', alpha=0.7)
        ax.quiver(self.position_calsource[0,index], self.position_calsource[1,index], self.position_calsource[2, index], 
                    -scale_vector_ortho_vert[0], 
                    -scale_vector_ortho_vert[1], 
                    -scale_vector_ortho_vert[2], 
                    color='grey', arrow_length_ratio=0., linewidth=3, linestyle='--', alpha=0.7)
        # Plot the orthogonal horizontal axis
        scale_vector_ortho_horiz = scale_factor(self.vector_ortho_horiz[:, index])
        ax.quiver(self.position_calsource[0, index], self.position_calsource[1, index], self.position_calsource[2, index], 
                    scale_vector_ortho_horiz[0], 
                    scale_vector_ortho_horiz[1], 
                    scale_vector_ortho_horiz[2], 
                    color='green', arrow_length_ratio=0., linewidth=3, linestyle='--', alpha=0.7)
        ax.quiver(self.position_calsource[0,index], self.position_calsource[1,index], self.position_calsource[2, index], 
                    -scale_vector_ortho_horiz[0], 
                    -scale_vector_ortho_horiz[1], 
                    -scale_vector_ortho_horiz[2], 
                    color='green', arrow_length_ratio=0., linewidth=3, linestyle='--', alpha=0.7)
        
        
        ### Plot the angle around the QUBIC-calsource axis
        self.plot_angle_3d(ax, origin=self.position_calsource[:, index], v2=self.vector_1_2_calsource_proj[:, index],
                    v1=self.vector_ortho_vert[:, index], angle=np.radians(self.calsource_orientation_angles[0, index]), 
                    radius=0.25,
                    color='orange', linewidth=3)
        
        ### Plot the angle around horizontal orthognoal axis
        self.plot_angle_3d(ax, origin=self.position_calsource[:, index], v2=self.vector_1_2_vertical_proj[:, index],
                           v1=self.vector_ortho_horiz[:, index], angle=np.radians(self.calsource_orientation_angles[1, index]),
                           radius=0.25,
                           color='grey', linewidth=3)
        
        ### Plot the angle around vertical orthognoal axis
        self.plot_angle_3d(ax, origin=self.position_calsource[:, index], v2=self.vector_1_2_horizontal_proj[:, index], 
                    v1=self.vector_ortho_vert[:, index], angle=np.radians(self.calsource_orientation_angles[2, index]), 
                    radius=0.25,
                    color='green', linewidth=3)
        
        ax.set_xlim([limits_min[0], limits_max[0]])
        ax.set_ylim([limits_min[1], limits_max[1]]) 
        ax.set_zlim([limits_min[2], limits_max[2]])
        ax.set_xlabel('North', fontsize=12, labelpad=10)
        ax.set_ylabel('East', fontsize=12, labelpad=10)
        ax.set_zlabel('Down', fontsize=12, labelpad=10)

        ax.set_title(f'Calibration source - Position and Orientation - {self.datetime[index]}', fontsize=16, pad=20)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # First legend for positions
        legend1 = ax.legend(
            [ax.get_children()[i] for i in [0, 1, 2, 3, 4]], 
            ['Antenna 1', 'Antenna 2', 'Base antenna', 'Qubic', 'Calibration Source'],
            loc='upper left', bbox_to_anchor=(1, 1)
        )
        ax.add_artist(legend1)

        # Second legend for vectors
        legend2 = ax.legend(
            [ax.get_children()[i] for i in [5, 7, 9, 11, 12]], 
            ['Vector 1-2', 
             'Vector 1-2 - Orthogonal vertical projection', 
             'Vector 1-2 - Orthogonal horizontal projection',
             'Vector 1-2 - Calsource projection',
             'Vector QUBIC-Calibration Source'],
             loc='upper left', bbox_to_anchor=(1, 0.8)
        )
        ax.add_artist(legend2)
        
        # Third legend for axis
        legend3 = ax.legend(
            [ax.get_children()[i] for i in [13, 15, 16]], 
            ['Calsource axis',
             'Orthogonal vertical axis',
             'Orthogonal horizontal axis'],
             loc='upper left', bbox_to_anchor=(1, 0.6)
        )
        ax.add_artist(legend3)

        # Fourth legend for angles
        legend4 = ax.legend(
            [ax.get_children()[i] for i in [18, 19, 20]], 
            [f'Angle around calsource axis : {self.calsource_orientation_angles[0, index]:.2f} °',
             f'Angle around orthogonal vertical axis : {self.calsource_orientation_angles[1, index]:.2f} °',
            f'Angle around orthogonal horizontal axis : {self.calsource_orientation_angles[2, index]:.2f} °'],
            loc='upper left', bbox_to_anchor=(1, 0.4)
        )
        ax.add_artist(legend4)
        
        plt.tight_layout(rect=[0, 0, 0.85, 0.85])
        
    def plot_system_plotly(self, index):
        """
        Interactive 3D visualization of the GPS system using plotly
        """
        import plotly.graph_objects as go
        
        # Get min/max coordinates for plot range
        all_points = np.vstack([
            self.position_antenna1[:, index],
            self.position_antenna2[:, index], 
            self.base_antenna_position,
            self.position_qubic,
            self.position_calsource[:, index]
        ])
        
        min_coords = np.min(all_points, axis=0)
        max_coords = np.max(all_points, axis=0)
        margin = 0.2
        plot_range = max_coords - min_coords
        limits_min = min_coords - margin * plot_range
        limits_max = max_coords + margin * plot_range
        
        # Create figure
        fig = go.Figure()
        
        # Add points
        points_data = [
            (self.position_antenna1[:, index], 'blue', 'square', 'Antenna 1'),
            (self.position_antenna2[:, index], 'blue', 'triangle-up', 'Antenna 2'),
            (self.base_antenna_position, 'black', 'circle', 'Base Antenna'),
            (self.position_qubic, 'pink', 'circle', 'QUBIC'),
            (self.position_calsource[:, index], 'red', 'star', 'Calibration Source')
        ]
        
        for pos, color, symbol, name in points_data:
            fig.add_trace(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode='markers',
                marker=dict(size=10, color=color, symbol=symbol),
                name=name
            ))
        
        # Add vectors
        def add_vector(start, vector, color, name, dash=None):
            end = start + vector
            fig.add_trace(go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode='lines',
                line=dict(color=color, width=4, dash=dash),
                name=name
            ))
        
        # Main vectors
        add_vector(self.position_antenna1[:, index], self.vector_1_2[:, index], 'blue', 'Vector 1-2')
        add_vector(self.position_qubic, self.vector_calsource_qubic[:, index], 'red', 'Vector QUBIC-Calibration Source')
        
        # Projections
        add_vector(self.position_calsource[:, index], self.vector_1_2_calsource_proj[:, index], 'darkblue', 'Vector 1-2 Calsource proj', 'dash')
        add_vector(self.position_calsource[:, index], self.vector_1_2_vertical_proj[:, index], 'royalblue', 'Vector 1-2 Vertical proj', 'dash')
        add_vector(self.position_calsource[:, index], self.vector_1_2_horizontal_proj[:, index], 'turquoise', 'Vector 1-2 Horizontal proj', 'dash')
        
        # Rotation axes
        add_vector(self.position_calsource[:, index], self.vector_ortho_vert[:, index], 'grey', 'Orthogonal vertical axis', 'dot')
        add_vector(self.position_calsource[:, index], self.vector_ortho_horiz[:, index], 'green', 'Orthogonal horizontal axis', 'dot')
        
        # Add angles as text annotations
        fig.add_trace(go.Scatter3d(
            x=[self.position_calsource[0, index]],
            y=[self.position_calsource[1, index]],
            z=[self.position_calsource[2, index]],
            mode='text',
            text=[f'Angles:<br>Around calsource: {self.calsource_orientation_angles[0, index]:.2f}°<br>' + 
                f'Around vertical: {self.calsource_orientation_angles[1, index]:.2f}°<br>' +
                f'Around horizontal: {self.calsource_orientation_angles[2, index]:.2f}°'],
            name='Orientation angles'
        ))
        
        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis_title='North',
                yaxis_title='East', 
                zaxis_title='Down',
                xaxis=dict(range=[limits_min[0], limits_max[0]]),
                yaxis=dict(range=[limits_min[1], limits_max[1]]),
                zaxis=dict(range=[limits_min[2], limits_max[2]]),
                aspectmode='cube'
            ),
            title=f'Calibration source - Position and Orientation - {self.datetime[index]}',
            showlegend=True,
            legend=dict(x=1.1, y=0.5)
        )
        
        return fig

    # def _get_observation_data(self, observation_indices):
        
    #     if observation_indices.shape == 1:
    #         observation_indices = observation_indices[0]
        
    #     ### rpN, rpE, rpD give the relative position of the antenna 2 wrt base antenna in North, East, Down coordinates
    #     self.rpN = self.rpN[observation_indices].reshape(-1)                                      # in m
    #     self.rpE = self.rpE[observation_indices].reshape(-1)                                      # in m
    #     self.rpD = self.rpD[observation_indices].reshape(-1)                                      # in m
        
    #     ### roll give the angle between antenna 2 - antenna 1 vector and the North axis
    #     self.roll = self.roll[observation_indices].reshape(-1)                                    # in rad
        
    #     ### yaw give the angle between antenna 2 - antenna 1 vector and the horizontal plane
    #     self.yaw = self.yaw[observation_indices].reshape(-1)                                      # in rad
        
    #     ### Other GPS parameters, not used yet
    #     self.pitchIMU = self._pitchIMU[observation_indices].reshape(-1)                            # in rad
    #     self.rollIMU = self.rollIMU[observation_indices].reshape(-1)                              # in rad
    #     self.temperature = self._temperature[observation_indices].reshape(-1)                      # in Celsius
    #     self.checksum = self._checksum[observation_indices].reshape(-1)
    
    # def get_calsource_position_rigid(self, position_ini_antenna1, position_ini_antenna2, position_ini_calsource, position_antenna1, position_antenna2):
    #     """Calsource position.
        
    #     Compute the position of the calibration source using the rigid transformation law. The algorithm follow different steps :
        
    #     1. Compute the barycenters of the initial and current vectors between the two antennas. 
    #     2. Compute the rotation matrix between these two vectors using the singular value decomposition. 
    #     3. Compute the new position of the calibration source using the rigid transformation law. 

    #     Parameters
    #     ----------
    #     position_ini_antenna1 : array_like
    #         Initial position of the antenna 1. 
    #     position_ini_antenna2 : array_like
    #         Initial position of the antenna 2.
    #     position_ini_calsource : array_like
    #         Initial position of the calibration source.
    #     position_antenna1 : array_like
    #         Current position of the antenna 1.
    #     position_antenna2 : array_like
    #         Current position of the antenna 2.

    #     Returns
    #     -------
    #     position_calsource : array_like
    #         Current position of the calibration source.
            
    #     """  
    #     #! This method does not work well when the vector 1_2 is close to initial vector 1_2.

    #     ### Define vectors
    #     vector_antennas_ini = position_ini_antenna2 - position_ini_antenna1
    #     vector_antennas = position_antenna2 - position_antenna1
    #     vec_ini_norm = vector_antennas_ini / np.linalg.norm(vector_antennas_ini)
    #     vec_norm = vector_antennas / np.linalg.norm(vector_antennas)
        
    #     ### Compute barycenters
    #     B_ini = (position_ini_antenna1 + position_ini_antenna2) / 2
    #     B = (position_antenna1 + position_antenna2) / 2
        
    #     H = np.outer(vec_ini_norm, vec_norm)
    #     U, _, Vt = np.linalg.svd(H)
    #     R = Vt.T @ U.T

    #     position_calsource = R @ (position_ini_calsource - B_ini) + B
            
    #     return position_calsource 
