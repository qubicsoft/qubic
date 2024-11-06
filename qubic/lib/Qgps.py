import os
import struct

import numpy as np
import matplotlib.pyplot as plt

import datetime as dt

#TODO : Add case where the cal source is not aligned with the antenna vector
#TODO : Add case where the base antenna is not located at the center of the QUBIC window

class GPS:
    
    def __init__(self, gps_data, position_ini_antenna1, position_ini_antenna2, position_ini_calsource, observation_date, position_qubic = np.array([0, 0, 0])):
        
        ### Fixed parameters
        self.base_antenna_position = np.array([0, 0, 0])
        self.position_ini_antenna1 = position_ini_antenna1
        self.position_ini_antenna2 = position_ini_antenna2
        self.position_ini_calsource = position_ini_calsource
        self.observation_date = observation_date
        self.position_qubic = position_qubic
        
        ### Convert the data from GPS system into a dictionary
        if type(gps_data) == str:
            if os.path.isfile(gps_data):
                self.gps_data = self.read_gps_bindat(gps_data)
            else:
                raise ValueError("The GPS data file does not exist")
        else:
            self.gps_data = gps_data
        
        ### Import all the GPS data from the dictionary and convert them in proper units
        self.timestamp = np.array(self.gps_data['timestamp']).reshape(-1)
        
        ### Build datetime array
        self.datetime = self.create_datetime_array(self.timestamp)
        
        ### Build observation variables : index, time, datetime
        self.observation_indices = self.get_observation_indices(self.datetime, self.observation_date).reshape(-1)
        self.observation_time = self.timestamp[self.observation_indices].reshape(-1)
        self.observation_datetime = self.datetime[self.observation_indices].reshape(-1)
        
        # rpN, rpE, rpD give the relative position of the antenna 2 wrt base antenna in North, East, Down coordinates
        self.rpN = np.array(self.gps_data['rpN'])[self.observation_indices].reshape(-1) / 10000                     # in m
        self.rpE = np.array(self.gps_data['rpE'])[self.observation_indices].reshape(-1) / 10000                     # in m
        self.rpD = np.array(self.gps_data['rpD'])[self.observation_indices].reshape(-1) / 10000                     # in m
        
        # roll give the angle between antenna 2 - antenna 1 vector and the North axis
        self.roll = np.radians(np.array(self.gps_data['roll'])[self.observation_indices].reshape(-1) / 1000)          # in rad
        
        # yaw give the angle between antenna 2 - antenna 1 vector and the horizontal plane
        self.yaw = np.radians(np.array(self.gps_data['yaw'])[self.observation_indices].reshape(-1) / 1000)            # in rad
        
        # Other GPS parameters, not used yet
        self.pitchIMU = np.radians(np.array(self.gps_data['pitchIMU'])[self.observation_indices].reshape(-1) / 1000)  # in rad
        self.rollIMU = np.radians(np.array(self.gps_data['rollIMU'])[self.observation_indices].reshape(-1) / 1000)    # in rad
        self.temperature = np.array(self.gps_data['temperature'])[self.observation_indices].reshape(-1) / 10          # in Celsius
        self.checksum = np.array(self.gps_data['checksum'])[self.observation_indices].reshape(-1)
        
        ### Compute position of antennas 1 & 2 and calibration source in North, East, Down cooridnates
        #! Be careful: as it is now, the calibration source needs to be on the straight line formed by antenna 1 and antenna 2
        self.distance_between_antennas = np.linalg.norm(self.position_ini_antenna2 - self.position_ini_antenna1)
        self.position_antenna2 = self.get_position_antenna_2(self.base_antenna_position)
        self.position_antenna1 = self.get_position_antenna_1(self.distance_between_antennas) 
        self.position_calsource = self.get_calsource_position(self.position_ini_antenna1, self.position_ini_antenna2, self.position_ini_calsource, self.position_antenna1, self.position_antenna2) 
        
        ### Compute the vectors between the calibration source and QUBIC, and the vector between the antennas in NED coordinates
        self.vector_1_2 = self.position_antenna2 - self.position_antenna1
        self.vector_calsource = self.position_calsource - self.position_qubic[:, None]
        
        ### Compute the calibration source orientation angles
        self.calsource_orientation_angles = np.degrees(self.get_calsource_orientation(self.vector_1_2, self.vector_calsource))        
    
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
        
        return np.array([dt.datetime.utcfromtimestamp(float(tstamp)) for tstamp in timestamp]) + dt.timedelta(hours=utc_offset) 
       
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
            return np.array(self.datetime_to_index(datetime, observation_date[0]))
        
        ### If we give a starting and stoping dates
        if len(observation_date.shape) == 1 and observation_date.shape[0] == 2:
            start_index = self.datetime_to_index(datetime, observation_date[0])
            end_index = self.datetime_to_index(datetime, observation_date[1])
            return np.arange(start_index, end_index, 1, dtype=int)
        
        else:
            raise ValueError('ERROR! Please choose a correct shape for the date: 1 or 2.')

    def get_position_antenna_2(self, base_antenna_position):
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
          
        return base_antenna_position[:, None] + np.array([self.rpN, self.rpE, self.rpD])
    
    def get_position_antenna_1(self, distance_w_antenna_2):
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
        
        return np.array([_rpN, _rpE, _rpD]) + self.position_antenna2
    
    def get_calsource_position(self, position_ini_antenna1, position_ini_antenna2, position_ini_calsource, position_antenna1, position_antenna2):
        """Calsource position.
        
        Compute the position of the calibration source using the rigid transformation law. The algorithm follow different steps :
        
        1. Compute the barycenters of the initial and current vectors between the two antennas. 
        2. Compute the rotation matrix between these two vectors using the singular value decomposition. 
        3. Compute the new position of the calibration source using the rigid transformation law.

        Parameters
        ----------
        position_ini_antenna1 : array_like
            Initial position of the antenna 1. 
        position_ini_antenna2 : array_like
            Initial position of the antenna 2.
        position_ini_calsource : array_like
            Initial position of the calibration source.
        position_antenna1 : array_like
            Current position of the antenna 1.
        position_antenna2 : array_like
            Current position of the antenna 2.

        Returns
        -------
        position_calsource : array_like
            Current position of the calibration source.
            
        """        
        
        ### Define vectors
        vector_antennas_ini = position_ini_antenna2 - position_ini_antenna1
        vector_antennas = position_antenna2 - position_antenna1
        vec_ini_norm = vector_antennas_ini / np.linalg.norm(vector_antennas_ini)
        vec_norm = vector_antennas / np.linalg.norm(vector_antennas)
        
        ### Compute barycenters
        C = (position_ini_antenna1 + position_ini_antenna2) / 2
        C_ = (position_antenna1 + position_antenna2) / 2
        
        ### Compute rotation through singular value decomposition
        position_calsource = np.zeros_like(position_antenna1)
    
        for i in range(position_antenna1.shape[1]):
            H = np.outer(vec_ini_norm, vec_norm[:, i])
            U, _, Vt = np.linalg.svd(H)
            d = np.linalg.det(Vt.T @ U.T)
            R = Vt.T @ np.diag([1, 1, d]) @ U.T
        
            ### Compute new position using the rigid transformation
            position_calsource[:, i] = R @ (position_ini_calsource - C) + C_[:, i]
            
        return position_calsource      
       
    def get_calsource_orientation(self, vector_1_2, vector_cal):
        r"""Calsource orientation.
        
        Method to compute the orientation of the calsource.
        - angles[0] is the angle around vector_cal. To compute it, let's define :
        
        .. math::
            \vec{n_{cal}} = \frac{\vec{V_{base \rightarrow cal}}}{||\vec{V_{base \rightarrow cal}}||}, \vec{e_D} = (0, 0, 1) and \vec{V_{ortho}} = \vec{V_{base \rightarrow cal}} \times \vec{e_D}
            
            Projection of the self.vector_1_2 on the orthogonal plane to vector_cal and down axis :
        
        .. math::
            \vec{V_{1 \rightarrow 2, cal}} = \vec{V_{1 \rightarrow 2}} - \frac{\vec{V_{1 \rightarrow 2}} \cdot \vec{n_{cal}}}{\vec{n_{cal}}\cdot \vec{n_{cal}}} \cdot \vec{n_{cal}}
            
            Then : 
        .. math ::
            \cos(angle[0]) = \frac{\vec{V_{base \rightarrow cal} \cdot \vec{V_{ortho}}}}{||\vec{V_{base \rightarrow cal}}||\cdot||\vec{V_{ortho}}||}
        
        - angles[1] is the angle around the axis orthogonal to vector_cal and down axis. It is fixed to 0 as we can't compute it with the system.
        
        - angles[2] is the angle around the down axis. To compute it, let's define :

            Projection of the self.vector_1_2 on the down axis :
        
        .. math::
            \vec{V_{1 \rightarrow 2, D}} = \vec{V_{1 \rightarrow 2}} - \frac{\vec{V_{1 \rightarrow 2}} \cdot \vec{e_{D}}}{\vec{e_{D}} \cdot \vec{e_{D}}} \cdot \vec{e_{D}}
            
            Then : 
        .. math ::
            \cos(angle[0]) = \frac{\vec{V_{1 \rightarrow 2, D}} \cdot \vec{V_{ortho}}}}{||\vec{V_{1 \rightarrow 2, D}}||\cdot||\vec{V_{ortho}}||}

        Parameters
        ----------
        self.vector_1_2 : array_like
            Vector between antenna 1 and 2.
        vector_cal : array_like
            Vector between QUBIC and the calibration source.

        Returns
        -------
        angles : array_like
            Orientation angles of the calsource.
        """
        
        angles = np.zeros(vector_1_2.shape)    
        
        ### Down vector
        ed = np.array([0, 0, 1])
        
        ### Direction of calsource vector
        self.n_cal = vector_cal / np.linalg.norm(vector_cal)
        
        ### Build orthogonal vector to vector_cal
        self.vector_ortho_vert = np.cross(self.n_cal, ed, axisa=0).T
        self.vector_ortho_horiz = np.cross(self.n_cal, self.vector_ortho_vert, axisa=0, axisb=0).T
        
        ### Projections of self.vector_1_2 on planes ortho to vector_cal
        self.vector_1_2_ortho = vector_1_2 - (np.sum(vector_1_2 * self.n_cal, axis=0) / np.sum(self.n_cal * self.n_cal, axis=0))[None, :] * self.n_cal
        self.vector_1_2_proj = vector_1_2 - (np.sum(vector_1_2 * self.vector_ortho_horiz, axis=0) / np.sum(self.vector_ortho_horiz * self.vector_ortho_horiz, axis=0))[None, :] * self.vector_ortho_horiz

        ### Compute the angles
        angles[0] = np.arccos(np.sum(self.vector_1_2_ortho * self.vector_ortho_vert, axis=0) / (np.linalg.norm(self.vector_1_2_ortho, axis=0) * np.linalg.norm(self.vector_ortho_vert, axis=0)))
        angles[2] = np.arccos(np.sum(self.vector_1_2_proj * self.vector_ortho_vert, axis=0) / (np.linalg.norm(self.vector_1_2_proj, axis=0) * np.linalg.norm(self.vector_ortho_vert, axis=0)))

        return angles
    
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
        
    def plot_system(self, index=0):
        """Plot the system.
        
        Function the plot the all system : antenna1, antenna 2, base antenna and the calibration source, positions, the associated vectors and the calsource orientation angles.

        Parameters
        ----------
        index : int, optional
            Time index at which you want to make the plot, by default 0
        """              
        
        #! I will have to write a more general code for the dimension and the ranges of the plot
        #* I don't want to add too arguments now, as it is a very specific function, associated with the GPS instance

        ### Initialize the 3d figure
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ### Compute a standard size for the plot
        standard_size = np.maximum(np.linalg.norm(self.vector_1_2[:, index], axis=0), np.linalg.norm(self.vector_calsource[:, index], axis=0))


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
        
        ### Projection of the vector between antenna 1 and antenna 2 on the North/East plane
        ax.quiver(self.position_calsource[0, index], self.position_calsource[1, index], self.position_calsource[2, index],
                  self.vector_1_2_ortho[0, index] * standard_size, 
                  self.vector_1_2_ortho[1, index] * standard_size, 
                  self.vector_1_2_ortho[2, index] * standard_size,
                  color='darkblue', arrow_length_ratio=0., linewidth=2, linestyle='--', alpha=0.7)
        ax.quiver(self.position_calsource[0, index], self.position_calsource[1, index], self.position_calsource[2, index],
                  -self.vector_1_2_ortho[0, index] * standard_size, 
                  -self.vector_1_2_ortho[1, index] * standard_size, 
                  -self.vector_1_2_ortho[2, index] * standard_size,
                  color='darkblue', arrow_length_ratio=0., linewidth=2, linestyle='--', alpha=0.7)
        
        ### Projection on Qubic-Calsource plane
        ax.quiver(self.position_calsource[0, index], self.position_calsource[1, index],  self.position_calsource[2, index],
                  self.vector_1_2_proj[0, index] * standard_size, 
                  self.vector_1_2_proj[1, index] * standard_size, 
                  self.vector_1_2_proj[2, index] * standard_size,
                  color='turquoise', arrow_length_ratio=0., linewidth=2, linestyle='--', alpha=0.7)
        ax.quiver(self.position_calsource[0, index], self.position_calsource[1, index],  self.position_calsource[2, index],
                  -self.vector_1_2_proj[0, index] * standard_size, 
                  -self.vector_1_2_proj[1, index] * standard_size, 
                  -self.vector_1_2_proj[2, index] * standard_size,
                  color='turquoise', arrow_length_ratio=0., linewidth=2, linestyle='--', alpha=0.7)
        
        ### Plot the vector between QUBIC and the calibration source
        ax.quiver(self.position_qubic[0], self.position_qubic[1], self.position_qubic[2], 
                    self.vector_calsource[0, index], 
                    self.vector_calsource[1, index],
                    self.vector_calsource[2, index],
                    color='r', arrow_length_ratio=0.1, linewidth=2)
        
        ### Plot the 3 rotation axis
        # Plot the QUBIC-calsource axis
        vector_calsource_norm = np.linalg.norm(self.vector_calsource[:, index])
        ax.quiver(self.position_qubic[0], self.position_qubic[1], self.position_qubic[2], 
                    self.vector_calsource[0, index] / vector_calsource_norm * standard_size,
                    self.vector_calsource[1, index] / vector_calsource_norm * standard_size,
                    self.vector_calsource[2, index] / vector_calsource_norm * standard_size,
                    color='orange', arrow_length_ratio=0., linewidth=1, linestyle='--', alpha=0.7)

        # Plot the orthogonal vertical axis
        orth_vert_norm = np.linalg.norm(self.vector_ortho_vert[:, index])
        ax.quiver(self.position_calsource[0, index], self.position_calsource[1, index], self.position_calsource[2, index], 
                    (self.vector_ortho_vert[0, index]) / orth_vert_norm * standard_size, 
                    (self.vector_ortho_vert[1, index]) / orth_vert_norm * standard_size, 
                    (self.vector_ortho_vert[2, index]) / orth_vert_norm * standard_size, 
                    color='grey', arrow_length_ratio=0., linewidth=1, linestyle='--', alpha=0.7)
        ax.quiver(self.position_calsource[0,index], self.position_calsource[1,index], self.position_calsource[2, index], 
                    (self.vector_ortho_vert[0, index]) / - orth_vert_norm * standard_size, 
                    (self.vector_ortho_vert[1, index]) / - orth_vert_norm * standard_size, 
                    (self.vector_ortho_vert[2, index]) / - orth_vert_norm * standard_size, 
                    color='grey', arrow_length_ratio=0., linewidth=1, linestyle='--', alpha=0.7)
        # Plot the orthogonal horizontal axis
        orth_horiz_norm = np.linalg.norm(self.vector_ortho_horiz[:, index])
        ax.quiver(self.position_calsource[0, index], self.position_calsource[1, index], self.position_calsource[2, index], 
                    (self.vector_ortho_horiz[0, index]) / orth_horiz_norm * standard_size, 
                    (self.vector_ortho_horiz[1, index]) / orth_horiz_norm * standard_size, 
                    (self.vector_ortho_horiz[2, index]) / orth_horiz_norm * standard_size, 
                    color='green', arrow_length_ratio=0., linewidth=1, linestyle='--', alpha=0.7)
        ax.quiver(self.position_calsource[0,index], self.position_calsource[1,index], self.position_calsource[2, index], 
                    (self.vector_ortho_horiz[0, index]) / - orth_horiz_norm * standard_size, 
                    (self.vector_ortho_horiz[1, index]) / - orth_horiz_norm * standard_size, 
                    (self.vector_ortho_horiz[2, index]) / - orth_horiz_norm * standard_size, 
                    color='green', arrow_length_ratio=0., linewidth=1, linestyle='--', alpha=0.7)
        ### Plot the angle around the QUBIC-calsource axis
        self.plot_angle_3d(ax, origin=self.position_calsource[:, index], v2=self.vector_1_2_ortho[:, index],
                    v1=self.vector_ortho_vert[:, index], angle=np.radians(self.calsource_orientation_angles[0, index]), 
                    radius=0.25 * standard_size,
                    color='orange', linewidth=3)
        
        ### Plot the angle around Down axis
        self.plot_angle_3d(ax, origin=self.position_calsource[:, index], v2=self.vector_1_2_proj[:, index], 
                    v1=self.vector_ortho_vert[:, index], angle=np.radians(self.calsource_orientation_angles[2, index]), 
                    radius=0.25 * standard_size,
                    color='green', linewidth=3)
        
        ax.set_xlim([-standard_size, standard_size])
        ax.set_ylim([-standard_size, standard_size])
        ax.set_zlim([-standard_size, standard_size])
        ax.set_xlabel('North', fontsize=12, labelpad=10)
        ax.set_ylabel('East', fontsize=12, labelpad=10)
        ax.set_zlabel('Down', fontsize=12, labelpad=10)

        ax.set_title(f'Calibration source - Position and Orientation - {self.observation_datetime[index]}', fontsize=16, pad=20)
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
            [ax.get_children()[i] for i in [5, 7, 9, 10]], 
            ['Vector 1-2', 
             'Vector 1-2 - Orthogonal projection', 
             'Vector 1-2 - Calsource projection',
             'Vector QUBIC-Calibration Source'],
             loc='upper left', bbox_to_anchor=(1, 0.8)
        )
        ax.add_artist(legend2)
        
        # Third legend for axis
        legend3 = ax.legend(
            [ax.get_children()[i] for i in [11, 13, 14]], 
            ['Calsource axis',
             'Orthogonal vertical axis',
             'Orthogonal horizontal axis'],
             loc='upper left', bbox_to_anchor=(1, 0.6)
        )
        ax.add_artist(legend3)

        # Fourth legend for angles
        legend4 = ax.legend(
            [ax.get_children()[i] for i in [16, 17]], 
            [f'Angle around calsource axis : {self.calsource_orientation_angles[0, index]:.2f} °',
            f'Angle around orthogonal horizontal axis : {self.calsource_orientation_angles[2, index]:.2f} °'],
            loc='upper left', bbox_to_anchor=(1, 0.4)
        )
        ax.add_artist(legend4)
        
        plt.tight_layout(rect=[0, 0, 0.85, 0.85])
        
class GPS_old:
    
    def __init__(self, gps_data_file_path, distance_between_antennas, distance_calsource, observation_date, position_qubic = np.array([0, 0, 0])):
        
        ### Fixed parameters
        self.base_antenna_position = np.array([0, 0, 0])
        self.position_qubic = position_qubic
        self.distance_between_antennas = distance_between_antennas
        self.distance_calsource = distance_calsource
        self.observation_date = observation_date
        
        ### Convert the data from GPS system into a dictionary
        self.gps_data = self.read_gps_bindat(gps_data_file_path)
        
        ### Import all the GPS data from the dictionary and convert them in proper units
        self.timestamp = np.array(self.gps_data['timestamp'])
        
        ### Build datetime array
        self.datetime = self.create_datetime_array(self.timestamp)
        
        ### Build observation variables : index, time, datetime
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
        self.yaw = np.radians(np.array(self.gps_data['yaw'])[self.observation_indices] / 1000)            # in rad
        
        # Other GPS parameters, not used yet
        self.pitchIMU = np.radians(np.array(self.gps_data['pitchIMU'])[self.observation_indices] / 1000)  # in rad
        self.rollIMU = np.radians(np.array(self.gps_data['rollIMU'])[self.observation_indices] / 1000)    # in rad
        self.temperature = np.array(self.gps_data['temperature'])[self.observation_indices] / 10          # in Celsius
        self.checksum = np.array(self.gps_data['checksum'])[self.observation_indices]
        
        ### Compute position of antennas 1 & 2 and calibration source in North, East, Down cooridnates
        #! Be careful: as it is now, the calibration source needs to be on the straight line formed by antenna 1 and antenna 2
        self.position_antenna2 = self.get_position_antenna_2(self.base_antenna_position)
        self.position_antenna1 = self.get_position_wrt_antenna_2(self.distance_between_antennas) 
        self.position_calsource = self.get_position_wrt_antenna_2(self.distance_calsource) 
        print(self.position_antenna1)
        print(self.position_antenna2)
        print(self.position_calsource)

        ### Compute the vectors between the calibration source and QUBIC, and the vector between the antennas in NED coordinates
        self.vector_1_2 = self.position_antenna2 - self.position_antenna1
        self.vector_calsource = self.position_calsource - self.position_qubic[:, None]
        
        ### Compute the calibration source orientation angles
        self.calsource_orientation_angles = np.degrees(self.get_calsource_orientation(self.vector_1_2, self.vector_calsource))        
    
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

    def get_position_antenna_2(self, base_antenna_position):
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
    
    def get_position_wrt_antenna_2(self, distance_w_antenna_2):
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
        
        return np.array([_rpN, _rpE, _rpD]) + self.position_antenna2
    
    def get_calsource_orientation(self, vector_1_2, vector_cal):
        r"""Calsource orientation.
        
        Method to compute the orientation of the calsource.
        - angles[0] is the angle around vector_cal. To compute it, let's define :
        
        .. math::
            \vec{n_{cal}} = \frac{\vec{V_{base \rightarrow cal}}}{||\vec{V_{base \rightarrow cal}}||}, \vec{e_D} = (0, 0, 1) and \vec{V_{ortho}} = \vec{V_{base \rightarrow cal}} \times \vec{e_D}
            
            Projection of the vector_1_2 on the orthogonal plane to vector_cal and down axis :
        
        .. math::
            \vec{V_{1 \rightarrow 2, cal}} = \vec{V_{1 \rightarrow 2}} - \frac{\vec{V_{1 \rightarrow 2}} \cdot \vec{n_{cal}}}{\vec{n_{cal}}\cdot \vec{n_{cal}}} \cdot \vec{n_{cal}}
            
            Then : 
        .. math ::
            \cos(angle[0]) = \frac{\vec{V_{base \rightarrow cal} \cdot \vec{V_{ortho}}}}{||\vec{V_{base \rightarrow cal}}||\cdot||\vec{V_{ortho}}||}
        
        - angles[1] is the angle around the axis orthogonal to vector_cal and down axis. It is fixed to 0 as we can't compute it with the system.
        
        - angles[2] is the angle around the down axis. To compute it, let's define :

            Projection of the vector_1_2 on the down axis :
        
        .. math::
            \vec{V_{1 \rightarrow 2, D}} = \vec{V_{1 \rightarrow 2}} - \frac{\vec{V_{1 \rightarrow 2}} \cdot \vec{e_{D}}}{\vec{e_{D}} \cdot \vec{e_{D}}} \cdot \vec{e_{D}}
            
            Then : 
        .. math ::
            \cos(angle[0]) = \frac{\vec{V_{1 \rightarrow 2, D}} \cdot \vec{V_{ortho}}}}{||\vec{V_{1 \rightarrow 2, D}}||\cdot||\vec{V_{ortho}}||}

        Parameters
        ----------
        vector_1_2 : array_like
            Vector between antenna 1 and 2.
        vector_cal : array_like
            Vector between QUBIC and the calibration source.

        Returns
        -------
        angles : array_like
            Orientation angles of the calsource.
        """
        
        angles = np.zeros(vector_1_2.shape)        
        
        ### Down vector
        ed = np.array([0, 0, 1])
        
        ### Direction of calsource vector
        self.n_cal = vector_cal / np.linalg.norm(vector_cal)
        
        ### Build orthogonal vector to vector_cal
        self.vector_ortho_vert = np.cross(self.n_cal, ed, axisa=0).T
        self.vector_ortho_horiz = np.cross(self.n_cal, self.vector_ortho_vert, axisa=0, axisb=0).T
        
        ### Projections of vector_1_2 on planes ortho to vector_cal
        self.vector_1_2_ortho = vector_1_2 - (np.sum(vector_1_2 * self.n_cal, axis=0) / np.sum(self.n_cal * self.n_cal, axis=0))[None, :] * self.n_cal
        self.vector_1_2_proj = vector_1_2 - (np.sum(vector_1_2 * self.vector_ortho_horiz, axis=0) / np.sum(self.vector_ortho_horiz * self.vector_ortho_horiz, axis=0))[None, :] * self.vector_ortho_horiz
        
        ### Compute the angles
        angles[0] = np.arccos(np.sum(self.vector_1_2_ortho * self.vector_ortho_vert, axis=0) / (np.linalg.norm(self.vector_1_2_ortho, axis=0) * np.linalg.norm(self.vector_ortho_vert, axis=0)))
        angles[2] = np.arccos(np.sum(self.vector_1_2_proj * self.vector_ortho_vert, axis=0) / (np.linalg.norm(self.vector_1_2_proj, axis=0) * np.linalg.norm(self.vector_ortho_vert, axis=0)))

        return angles
    
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
        
    def plot_system(self, index=0):
        """Plot the system.
        
        Function the plot the all system : antenna1, antenna 2, base antenna and the calibration source, positions, the associated vectors and the calsource orientation angles.

        Parameters
        ----------
        index : int, optional
            Time index at which you want to make the plot, by default 0
        """              
        
        #! I will have to write a more general code for the dimension and the ranges of the plot
        #* I don't want to add too arguments now, as it is a very specific function, associated with the GPS instance

        ### Initialize the 3d figure
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ### Compute a standard size for the plot
        standard_size = np.max([np.linalg.norm(self.vector_1_2[:, index]), np.linalg.norm(self.vector_calsource[:, index])])

        ### Plot antenna1, antenna 2, base antenna, qubic and the calibration source
        ax.scatter(self.position_antenna1[0, index], self.position_antenna1[1, index], self.position_antenna1[2, index], color='b', marker='s', s=100)
        ax.scatter(self.position_antenna2[0, index], self.position_antenna2[1, index], self.position_antenna2[2, index], color='b', marker='^', s=100)
        ax.scatter(self.base_antenna_position[0], self.base_antenna_position[1], self.base_antenna_position[2], color='k', s=100)
        ax.scatter(self.position_qubic[0], self.position_qubic[1], self.position_qubic[2], color='pink', marker='o', s=100)
        ax.scatter(self.position_calsource[0, index], self.position_calsource[1, index], self.position_calsource[2, index], color='r', marker='*', s=100)
        
        ### Plot the vector between antenna 1 and antenna 2
        ax.quiver(self.position_antenna1[0, index], self.position_antenna1[1, index], self.position_antenna1[2, index], 
                    self.position_antenna2[0, index] - self.position_antenna1[0, index], 
                    self.position_antenna2[1, index] - self.position_antenna1[1, index], 
                    self.position_antenna2[2, index] - self.position_antenna1[2, index],
                    color='b', arrow_length_ratio=0.1, linewidth=2)
        
        ### Projection of the vector between antenna 1 and antenna 2 on the North/East plane
        ax.quiver(self.position_calsource[0, index], self.position_calsource[1, index], self.position_calsource[2, index],
                  self.vector_1_2_ortho[0, index] * standard_size, 
                  self.vector_1_2_ortho[1, index] * standard_size, 
                  self.vector_1_2_ortho[2, index] * standard_size,
                  color='darkblue', arrow_length_ratio=0., linewidth=2, linestyle='--', alpha=0.7)
        ax.quiver(self.position_calsource[0, index], self.position_calsource[1, index], self.position_calsource[2, index],
                  -self.vector_1_2_ortho[0, index] * standard_size, 
                  -self.vector_1_2_ortho[1, index] * standard_size, 
                  -self.vector_1_2_ortho[2, index] * standard_size,
                  color='darkblue', arrow_length_ratio=0., linewidth=2, linestyle='--', alpha=0.7)
        
        ### Projection on Qubic-Calsource plane
        ax.quiver(self.position_calsource[0, index], self.position_calsource[1, index],  self.position_calsource[2, index],
                  self.vector_1_2_proj[0, index] * standard_size, 
                  self.vector_1_2_proj[1, index] * standard_size, 
                  self.vector_1_2_proj[2, index] * standard_size,
                  color='turquoise', arrow_length_ratio=0., linewidth=2, linestyle='--', alpha=0.7)
        ax.quiver(self.position_calsource[0, index], self.position_calsource[1, index],  self.position_calsource[2, index],
                  -self.vector_1_2_proj[0, index] * standard_size, 
                  -self.vector_1_2_proj[1, index] * standard_size, 
                  -self.vector_1_2_proj[2, index] * standard_size,
                  color='turquoise', arrow_length_ratio=0., linewidth=2, linestyle='--', alpha=0.7)
        
        ### Plot the vector between QUBIC and the calibration source
        ax.quiver(self.position_qubic[0], self.position_qubic[1], self.position_qubic[2], 
                    self.position_calsource[0, index] - self.position_qubic[0], 
                    self.position_calsource[1, index] - self.position_qubic[1], 
                    self.position_calsource[2, index] - self.position_qubic[2], 
                    color='r', arrow_length_ratio=0.1, linewidth=2)
        
        ### Plot the 3 rotation axis
        # Plot the QUBIC-calsource axis
        ax.quiver(self.position_qubic[0], self.position_qubic[1], self.position_qubic[2], 
                    (self.position_calsource[0, index] - self.position_qubic[0]) * standard_size, 
                    (self.position_calsource[1, index] - self.position_qubic[1]) * standard_size, 
                    (self.position_calsource[2, index] - self.position_qubic[2]) * standard_size, 
                    color='orange', arrow_length_ratio=0., linewidth=1, linestyle='--', alpha=0.7)

        # Plot the orthogonal vertical axis
        orth_vert_norm = np.linalg.norm(self.vector_ortho_vert[:, index])
        ax.quiver(self.position_calsource[0, index], self.position_calsource[1, index], self.position_calsource[2, index], 
                    (self.vector_ortho_vert[0, index]) / orth_vert_norm * standard_size, 
                    (self.vector_ortho_vert[1, index]) / orth_vert_norm * standard_size, 
                    (self.vector_ortho_vert[2, index]) / orth_vert_norm * standard_size, 
                    color='grey', arrow_length_ratio=0., linewidth=1, linestyle='--', alpha=0.7)
        ax.quiver(self.position_calsource[0,index], self.position_calsource[1,index], self.position_calsource[2, index], 
                    (self.vector_ortho_vert[0, index]) / - orth_vert_norm * standard_size, 
                    (self.vector_ortho_vert[1, index]) / - orth_vert_norm * standard_size, 
                    (self.vector_ortho_vert[2, index]) / - orth_vert_norm * standard_size, 
                    color='grey', arrow_length_ratio=0., linewidth=1, linestyle='--', alpha=0.7)
        # Plot the orthogonal horizontal axis
        orth_horiz_norm = np.linalg.norm(self.vector_ortho_horiz[:, index])
        ax.quiver(self.position_calsource[0, index], self.position_calsource[1, index], self.position_calsource[2, index], 
                    (self.vector_ortho_horiz[0, index]) / orth_horiz_norm * standard_size, 
                    (self.vector_ortho_horiz[1, index]) / orth_horiz_norm * standard_size, 
                    (self.vector_ortho_horiz[2, index]) / orth_horiz_norm * standard_size, 
                    color='green', arrow_length_ratio=0., linewidth=1, linestyle='--', alpha=0.7)
        ax.quiver(self.position_calsource[0,index], self.position_calsource[1,index], self.position_calsource[2, index], 
                    (self.vector_ortho_horiz[0, index]) / - orth_horiz_norm * standard_size, 
                    (self.vector_ortho_horiz[1, index]) / - orth_horiz_norm * standard_size, 
                    (self.vector_ortho_horiz[2, index]) / - orth_horiz_norm * standard_size, 
                    color='green', arrow_length_ratio=0., linewidth=1, linestyle='--', alpha=0.7)
        ### Plot the angle around the QUBIC-calsource axis
        self.plot_angle_3d(ax, origin=self.position_calsource[:, index], v2=self.vector_1_2_ortho[:, index],
                    v1=self.vector_ortho_vert[:, index], angle=np.radians(self.calsource_orientation_angles[0, index]), 
                    radius=0.25 * standard_size,
                    color='orange', linewidth=3)
        
        ### Plot the angle around Down axis
        self.plot_angle_3d(ax, origin=self.position_calsource[:, index], v2=self.vector_1_2_proj[:, index], 
                    v1=self.vector_ortho_vert[:, index], angle=np.radians(self.calsource_orientation_angles[2, index]), 
                    radius=0.25 * standard_size,
                    color='green', linewidth=3)
        
        ax.set_xlim([-standard_size, standard_size])
        ax.set_ylim([-standard_size, standard_size])
        ax.set_zlim([-standard_size, standard_size])
        ax.set_xlabel('North', fontsize=12, labelpad=10)
        ax.set_ylabel('East', fontsize=12, labelpad=10)
        ax.set_zlabel('Down', fontsize=12, labelpad=10)

        ax.set_title(f'Calibration source - Position and Orientation - {self.observation_datetime[index]}', fontsize=16, pad=20)
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
            [ax.get_children()[i] for i in [5, 7, 9, 10]], 
            ['Vector 1-2', 
             'Vector 1-2 - Orthogonal projection', 
             'Vector 1-2 - Calsource projection',
             'Vector QUBIC-Calibration Source'],
             loc='upper left', bbox_to_anchor=(1, 0.8)
        )
        ax.add_artist(legend2)
        
        # Third legend for vectors
        legend3 = ax.legend(
            [ax.get_children()[i] for i in [11, 13, 14]], 
            ['Calsource axis',
             'Orthogonal vertical axis',
             'Orthogonal horizontal axis'],
             loc='upper left', bbox_to_anchor=(1, 0.6)
        )
        ax.add_artist(legend3)

        # Fourth legend for angles
        legend4 = ax.legend(
            [ax.get_children()[i] for i in [16, 17]], 
            [f'Angle around calsource axis : {self.calsource_orientation_angles[0, index]:.2f} °',
            f'Angle around orthogonal horizontal axis : {self.calsource_orientation_angles[2, index]:.2f} °'],
            loc='upper left', bbox_to_anchor=(1, 0.4)
        )
        ax.add_artist(legend4)
        
        plt.tight_layout(rect=[0, 0, 0.85, 0.85])