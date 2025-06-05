import datetime as dt
import os
import struct

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation as R


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
        if type(gps_data_path) is str:
            if os.path.isfile(gps_data_path):
                return self.read_gps_bindat(gps_data_path)
            else:
                raise ValueError("The GPS data file does not exist")
        else:
            return gps_data_path

    def read_gps_bindat(self, gps_data_path: str, verbosity=0):
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
            ValueError("ERROR!  File not found: %s" % gps_data_path)

        # get date of the file.  We had a format change 2025-01-15 10:00 = 1736931600
        file_date = os.path.getatime(gps_data_path)
        if file_date < 1736931600:
            fmt = "<Bdiiiiiiifi"
            print("using old format with integer roll and yaw!")
        else:
            fmt = "<Bdiiiiifffi"

        # read the data
        h = open(gps_data_path, "rb")
        bindat = h.read()
        h.close()

        # interpret the binary data
        nbytes = 45
        names = "STX,timestamp,rpN,rpE,rpD,roll,yaw,pitchIMU,rollIMU,temperature,checksum".split(",")
        data = {}
        for name in names:
            data[name] = []

        idx = 0
        while idx + nbytes < len(bindat):
            packet = bindat[idx : idx + nbytes]
            dat_list = struct.unpack(fmt, packet)

            if len(dat_list) != len(names):
                print("ERROR:  Incompatible data at byte %i" % idx)
                if verbosity > 1:
                    input("enter to continue ")
                idx += 1
                continue

            if dat_list[0] != 0xAA:
                print("ERROR: Incorrect data at byte %i" % idx)
                if verbosity > 1:
                    input("enter to continue ")
                idx += 1
                continue

            for datidx, name in enumerate(names):
                data[name].append(dat_list[datidx])
                if verbosity > 0:
                    print(dat_list)

            idx += nbytes

        for name in data.keys():
            data[name] = np.array(data[name])

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
        self._timestamp = np.array(gps_data["timestamp"])
        self._datetime = self.create_datetime_array(self._timestamp)

        ### rpN, rpE, rpD give the relative position of the antenna 2 wrt base antenna in North, East, Down coordinates
        self.rpN = np.array(gps_data["rpN"]) / 10000  # in m
        self.rpE = np.array(gps_data["rpE"]) / 10000  # in m
        #! - sign to switch from Down to Up axis, which is more usual
        self.rpD = -np.array(gps_data["rpD"]) / 10000  # in m

        ### roll give the angle between antenna 2 - antenna 1 vector and the North axis
        self.roll = np.radians(np.array(gps_data["roll"])) / 1000  # in rad

        ### yaw give the angle between antenna 2 - antenna 1 vector and the horizontal plane
        self.yaw = np.radians(np.array(gps_data["yaw"])) / 1000  # in rad

        ### Other GPS parameters, not used yet
        self._pitchIMU = np.radians(np.array(gps_data["pitchIMU"])) / 1000  # in rad
        self.rollIMU = np.radians(np.array(gps_data["rollIMU"])) / 1000  # in rad
        self._temperature = np.array(gps_data["temperature"]) / 10  # in Celsius
        self._checksum = np.array(gps_data["checksum"])

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
            raise TypeError("ERROR! Please choose a date in the form of a string or a datetime object.")

        try:
            return int(np.where(datetime == date)[0][0])
        except Exception:
            raise IndexError("ERROR! The date you chose is not in the data.")

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
        if len(observation_date.shape) == 1 and observation_date.size == 1:
            return np.array([self.datetime_to_index(datetime, observation_date[0])], dtype=int)

        ### If we give a starting and stoping dates
        if len(observation_date.shape) == 1 and observation_date.size == 2:
            start_index = self.datetime_to_index(datetime, observation_date[0])
            end_index = self.datetime_to_index(datetime, observation_date[1])
            return np.arange(start_index, end_index, 1, dtype=int)

        else:
            raise ValueError("ERROR! Please choose a correct shape for the date: 1 or 2.")

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

        fig, ax = plt.subplots(figsize=(15, 5))

        ax.set_xlabel("Date")
        ax.set_ylabel("Position (m)")
        ax.plot(self._datetime[index_start:index_stop], self.rpN[index_start:index_stop], color="red", label="North component")
        ax.plot(self._datetime[index_start:index_stop], self.rpE[index_start:index_stop], color="blue", label="East component")
        ax.plot(self._datetime[index_start:index_stop], self.rpD[index_start:index_stop], color="green", label="Up component")

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

        fig, ax = plt.subplots(figsize=(15, 5))

        ax.set_xlabel("Date")
        ax.set_ylabel("Angles (rad)")

        ax.plot(self._datetime[index_start:index_stop], self.roll[index_start:index_stop], color="pink", label="Roll angle")
        ax.plot(self._datetime[index_start:index_stop], self.yaw[index_start:index_stop], color="brown", label="Yaw angle")

        fig.tight_layout()
        ax.set_title("Angles")
        fig.legend()
        plt.show()

    def plot_gps_data(self, index_start=0, index_stop=-1, position_limit=None, angle_limit=None):
        """Plot GPS data.

        Plot the position vector and the angles.

        Parameters
        ----------
        index_start : int, optional
            First observation index, by default 0
        index_stop : int, optional
            Last observation index, by default -1
        """

        fig, ax1 = plt.subplots(figsize=(15, 5))

        color_a = "tab:pink"
        color_r = "tab:red"
        color_b = "tab:blue"
        color_d = "tab:green"
        color_c = "tab:brown"

        ax1.set_xlabel("Date")
        ax1.set_ylabel("Position Vector Components (m)", color=color_r)
        ax1.plot(self._datetime[index_start:index_stop], self.rpN[index_start:index_stop], color=color_r, label="North component")
        ax1.plot(self._datetime[index_start:index_stop], self.rpE[index_start:index_stop], color=color_b, label="East component")
        ax1.plot(self._datetime[index_start:index_stop], self.rpD[index_start:index_stop], color=color_d, label="Up component")

        ax2 = ax1.twinx()

        ax2.plot(self._datetime[index_start:index_stop], self.roll[index_start:index_stop], color=color_a, label="Roll angle")
        ax2.plot(self._datetime[index_start:index_stop], self.yaw[index_start:index_stop], color=color_c, label="Yaw angle")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Angles (rad)", color=color_a)

        fig.tight_layout()
        ax1.set_title("GPS Data")
        fig.legend(bbox_to_anchor=(1, 1), loc="upper left")
        if position_limit is not None:
            ax1.set_ylim(position_limit)
        if angle_limit is not None:
            ax2.set_ylim(angle_limit)
        plt.show()

    def plot_gps_data_plotly(self, index_start=0, index_stop=-1, position_limit=None, angle_limit=None):
        """
        Plot GPS data using Plotly.

        This function creates an interactive plot with two y-axes:
        - Primary y-axis: Position vector components (North, East, Up).
        - Secondary y-axis: Angles (Roll, Yaw).

        Parameters
        ----------
        index_start : int, optional
            First observation index (default is 0)
        index_stop : int, optional
            Last observation index (default is -1)
        position_limit : tuple of float, optional
            Y-axis limits for position components (min, max)
        angle_limit : tuple of float, optional
            Y-axis limits for angles (min, max)
        """
        # Slice data
        x_data = self._datetime[index_start:index_stop]
        rpN_data = self.rpN[index_start:index_stop]
        rpE_data = self.rpE[index_start:index_stop]
        rpD_data = self.rpD[index_start:index_stop]
        roll_data = self.roll[index_start:index_stop]
        yaw_data = self.yaw[index_start:index_stop]

        # Define colors
        color_a = "pink"
        color_r = "red"
        color_b = "blue"
        color_d = "green"
        color_c = "brown"

        # Create a figure with a secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces for position vector components on the primary y-axis
        fig.add_trace(go.Scatter(x=x_data, y=rpN_data, mode="lines", name="North component", line=dict(color=color_r)), secondary_y=False)
        fig.add_trace(go.Scatter(x=x_data, y=rpE_data, mode="lines", name="East component", line=dict(color=color_b)), secondary_y=False)
        fig.add_trace(go.Scatter(x=x_data, y=rpD_data, mode="lines", name="Up component", line=dict(color=color_d)), secondary_y=False)

        # Add traces for angles on the secondary y-axis
        fig.add_trace(go.Scatter(x=x_data, y=roll_data, mode="lines", name="Roll angle", line=dict(color=color_a)), secondary_y=True)
        fig.add_trace(go.Scatter(x=x_data, y=yaw_data, mode="lines", name="Yaw angle", line=dict(color=color_c)), secondary_y=True)

        # Update figure layout
        fig.update_layout(title="GPS Data", xaxis_title="Date", legend=dict(x=1.05, y=1), width=900, height=500, template="plotly_white")
        fig.update_yaxes(title_text="Position Vector Components (m)", secondary_y=False)
        fig.update_yaxes(title_text="Angles (rad)", secondary_y=True)

        # Set y-axis limits if provided
        if position_limit is not None:
            fig.update_yaxes(range=position_limit, secondary_y=False)
        if angle_limit is not None:
            fig.update_yaxes(range=angle_limit, secondary_y=True)

        fig.show()


class GPSAntenna(GPStools):
    def __init__(self, gps_data_path, distance_antennas):
        """GPSAntenna class.

        Class to compute the position of the two GPS antennas.

        Parameters
        ----------
        gps_data_path : string or dict
            Path of the GPS binary file or dictionary containing the GPS data.
        distance_antennas : float
            Distance between the two antennas, it's necessary to compute the position of antenna 1.
        """

        ### Initialize the GPSTools class
        GPStools.__init__(self, gps_data_path)

        ### Fixed parameters
        self.base_antenna_position = np.array([0, 0, 0])
        self.distance_antennas = distance_antennas

        ### Compute position of antennas 1 & 2 and calibration source in North, East, Down cooridnates
        self.position_antenna2 = self.get_position_antenna_2(self.base_antenna_position)
        self.position_antenna1 = self.get_position_antenna_1(self.distance_antennas)

    def get_position_antenna_2(self, base_antenna_position=np.array([0, 0, 0])):
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

    def get_position_antenna_1(self, distance_antennas):
        """Position wrt antenna 2.

        General fonction to compute the position of any point located on the straight line formed by the antenna 1 - anntenna 2 vector.
        In the code, we used it only to compute the position of the antenna 1.

        Be careful, yaw is not the usual theta angle in sphercial cooridinates (i.e. the latitude angle): it corresponds to the elevation angle.
        It is why we need to add np.pi/2 in the conversion formulas.

        Parameters
        ----------
        distance_antennas : float
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
        _rpN = distance_antennas * np.cos(self.roll) * np.sin(np.pi / 2 - self.yaw)
        _rpE = distance_antennas * np.sin(self.roll) * np.sin(np.pi / 2 - self.yaw)
        _rpD = distance_antennas * np.cos(np.pi / 2 - self.yaw)

        return np.array([_rpN, _rpE, _rpD]) + self.position_antenna2


class GPSCalsource(GPSAntenna):
    def __init__(
        self, gps_data, position_ini_antenna1, position_ini_antenna2, position_ini_calsource, observation_date, distance_antennas=False, position_qubic=np.array([0, 0, 0]), observation_only=False
    ):
        ### Fixed parameters
        self.base_antenna_position = np.array([0, 0, 0])
        self.position_ini_antenna1 = position_ini_antenna1
        self.position_ini_antenna2 = position_ini_antenna2
        self.position_ini_calsource = position_ini_calsource
        self.observation_date = observation_date
        self.position_qubic = position_qubic

        ### Distance between antennas to initialize GPSAntenna
        if not distance_antennas:
            distance_antennas = np.linalg.norm(self.position_ini_antenna2 - self.position_ini_antenna1)
        else:
            distance_antennas = distance_antennas
        GPSAntenna.__init__(self, gps_data, distance_antennas)

        ### Import all the GPS data from the dictionary and convert them in proper units
        self.timestamp = self._timestamp.reshape(-1)
        # Build datetime array
        self.datetime = self.create_datetime_array(self.timestamp)
        # Build observation variables : index, time, datetime
        self.observation_indices = self.get_observation_indices(self.datetime, self.observation_date).reshape(-1)
        print("The observation indices are : ", self.observation_indices)
        # Keep only data during observatin time
        if observation_only:
            self._get_observation_data(self.observation_indices)

        ### Compute the vectors between the calibration source and QUBIC, and the vector between the antennas in NED coordinates
        self.vector_1_2_ini = self.position_ini_antenna2 - self.position_ini_antenna1
        self.vector_1_2 = self.position_antenna2 - self.position_antenna1
        self.vector_calsource_qubic_ini = self.position_qubic - self.position_ini_calsource

        ### Compute the calibration source orientation vector
        self.deviation_angle = self.compute_angle(self.vector_1_2, self.vector_1_2_ini[:, None])
        self.rotation_instance = self.compute_rotation(self.vector_1_2, self.vector_1_2_ini[:, None])
        self.vector_calsource_orientation = self.apply_rotation(self.vector_calsource_qubic_ini, self.rotation_instance)

        ### Compute the position of the calibration source in cartesian and azimutal coordinates
        self.position_calsource = self.get_calsource_position(self.position_ini_antenna2[:, None], self.position_ini_calsource[:, None], self.position_antenna2)
        self.position_calsource_azel = self.cartesian_to_azel(self.position_calsource)

    def _get_observation_data(self, observation_indices):
        ### Time and datetime during observation period
        self.observation_time = self.timestamp[self.observation_indices].reshape(-1)
        self.observation_datetime = self.datetime[self.observation_indices].reshape(-1)

        ### rpN, rpE, rpD give the relative position of the antenna 2 wrt base antenna in North, East, Down coordinates
        self.rpN = self.rpN[observation_indices].reshape(-1)  # in m
        self.rpE = self.rpE[observation_indices].reshape(-1)  # in m
        self.rpD = self.rpD[observation_indices].reshape(-1)  # in m

        ### roll give the angle between antenna 2 - antenna 1 vector and the North axis
        self.roll = self.roll[observation_indices].reshape(-1)  # in rad

        ### yaw give the angle between antenna 2 - antenna 1 vector and the horizontal plane
        self.yaw = self.yaw[observation_indices].reshape(-1)  # in rad

        ### Other GPS parameters, not used yet
        self.pitchIMU = self._pitchIMU[observation_indices].reshape(-1)  # in rad
        self.rollIMU = self.rollIMU[observation_indices].reshape(-1)  # in rad
        self.temperature = self._temperature[observation_indices].reshape(-1)  # in Celsius
        self.checksum = self._checksum[observation_indices].reshape(-1)

    def _compute_dot_product(self, v1, v2):
        v1_normalized = v1 / np.linalg.norm(v1, axis=0)
        v2_normalized = v2 / np.linalg.norm(v2, axis=0)
        dot_product = np.sum(v1_normalized * v2_normalized, axis=0)

        return dot_product

    def _compute_cross_product(self, v1, V2):
        v1_normalized = v1 / np.linalg.norm(v1, axis=0)
        v2_normalized = V2 / np.linalg.norm(V2, axis=0)
        cross_product = np.cross(v2_normalized.T, v1_normalized.T).T

        return cross_product

    def compute_angle(self, v1, v2):
        dot_product = self._compute_dot_product(v1, v2)
        cross_product = self._compute_cross_product(v1, v2)

        angle = np.arctan2(cross_product, dot_product)
        return angle

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

        cross_product = self._compute_cross_product(v1, v2)

        ### Define the rotation axis and angle between the vectors
        rotation_axis = cross_product / np.linalg.norm(cross_product, axis=0)
        angle = self.compute_angle(v1, v2)

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
        """Cartesian to AzEl.

        Convert cartesian coordinates to AzEl coordinates.

        Parameters
        ----------
        cartesian_position : array_like
            Position in cartesian coordinates.

        Returns
        -------
        azel_position : array_like
            Position in AzEl coordinates.
        """

        x, y, z = cartesian_position

        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)
        phi = np.arccos(z / r)

        azimuth = np.degrees(theta)
        elevation = np.degrees(phi)

        return np.array([azimuth, elevation])

    def plot_vector(self, fig, pos, vector, color="blue", name="vector", show_arrow=True, arrow_size=0.2):
        """Plot vector with plotly.

        General method to plot a vector with arrow using plotly.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Matplotlib figure.
        pos : array_like
            Position of the vector.
        vector : array_like
            Vector to plot.
        color : str, optional
            Color of the vector, by default 'blue'
        name : str, optional
            Name of the vector, by default 'vector'
        show_arrow : bool, optional
            Show or not the arrow, by default True
        arrow_size : float, optional
            Vector's arrow size, by default 0.2
        """

        ### Coordiantes of the two points defining the vector
        start = pos
        end = pos + vector

        ### Build unitary vector
        vector_unit = vector / np.linalg.norm(vector)

        ### Plot the segment between the two points
        fig.add_trace(
            go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode="lines",
                line=dict(color=color, width=2),
                name=name,
                text=[name, name],
                hovertemplate=("<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>"),
            )
        )

        ### Plot the arrowhead
        if show_arrow:
            if vector_unit[0] != 0 or vector_unit[1] != 0:
                # General case: construct perpendicular vectors
                ortho1 = np.cross(vector_unit, [0, 0, 1])
            else:
                # Special case: vector is along z-axis
                ortho1 = np.cross(vector_unit, [1, 0, 0])

            ortho1 /= np.linalg.norm(ortho1)
            # Compute the second orthogonal vector
            ortho2 = np.cross(vector_unit, ortho1)
            ortho2 /= np.linalg.norm(ortho2)
            # Base of the arrowhead
            tip_base = np.array(end) - arrow_size * vector_unit

            # Compute the points for the arrowhead
            point1 = tip_base + arrow_size * 0.5 * ortho1
            point2 = tip_base - arrow_size * 0.5 * ortho1

            # Add the arrowhead segments
            for point in [point1, point2]:
                fig.add_trace(
                    go.Scatter3d(
                        x=[end[0], point[0]],
                        y=[end[1], point[1]],
                        z=[end[2], point[2]],
                        mode="lines",
                        line=dict(color=color, width=5),
                        showlegend=False,
                        hovertemplate=("X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>"),
                    )
                )

    def plot_calsource_deviation(self, index):
        """Plot Calsource deviation.

        Plot the deviation of the calibration source orientation at a given observation time, compared to the initial position.

        Parameters
        ----------
        index : int
            Index of the observation time.

        """

        fig = go.Figure()

        ### Store the points and vectors and thir associated informations
        points_data = [
            (self.position_antenna1[:, index], "darkblue", "square", "Antenna 1"),
            (self.position_antenna2[:, index], "darkblue", "diamond", "Antenna 2"),
            (self.position_calsource[:, index], "darkred", "x", "Calibration Source"),
            (self.position_ini_antenna1, "blue", "square", "Initial Antenna 1"),
            (self.position_ini_antenna2, "blue", "diamond", "Initial Antenna 2"),
            (self.position_ini_calsource, "red", "x", "Initial Calibration Source"),
            (self.base_antenna_position, "pink", "circle", "Base Antenna"),
            (self.position_qubic, "black", "circle", "QUBIC"),
        ]
        vectors_data = [
            (self.position_antenna1[:, index], self.vector_1_2[:, index], "darkblue", "Vector Antenna 1 to 2"),
            (self.position_ini_antenna1, self.vector_1_2_ini, "blue", "Initial Vector Antenna 1 to 2"),
            (self.position_calsource[:, index], self.vector_calsource_orientation[:, index], "darkred", "Vector Calibration Source"),
            (self.position_ini_calsource, self.vector_calsource_qubic_ini, "red", "Initial Vector Calibration Source"),
        ]

        ### Plot points as 3D scatter
        for pos, color, symbol, name in points_data:
            fig.add_trace(
                go.Scatter3d(
                    x=[pos[0]],
                    y=[pos[1]],
                    z=[pos[2]],
                    mode="markers",
                    marker=dict(size=5, color=color, symbol=symbol),
                    text=[name],
                    name=name,
                    hovertemplate=("<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>"),
                )
            )

        ### Plot vectors
        for pos, vector, color, name in vectors_data:
            self.plot_vector(fig, pos, vector, color=color, name=name, show_arrow=True)

        ### Store points and vectors positions, to build a reference scale for the limit of the plot
        all_points = np.vstack(
            [
                self.position_antenna1[:, index],
                self.position_antenna2[:, index],
                self.position_calsource[:, index],
                self.position_ini_antenna1,
                self.position_ini_antenna2,
                self.position_ini_calsource,
                self.base_antenna_position,
                self.position_qubic,
            ]
        )
        vector_endpoints = np.vstack(
            [
                self.position_antenna1[:, index] + self.vector_1_2[:, index],
                self.position_ini_antenna1 + self.vector_1_2_ini,
                self.position_calsource[:, index] + self.vector_calsource_orientation[:, index],
                self.position_ini_calsource - self.vector_calsource_qubic_ini,
            ]
        )
        all_points = np.vstack([all_points, vector_endpoints])

        ### Calculate characteristic scale
        margin = 0.2
        min_coords = np.min(all_points, axis=0)
        max_coords = np.max(all_points, axis=0)
        plot_range = max_coords - min_coords

        ### Use largest range as characteristic scale
        char_scale = np.max(plot_range)
        center = (max_coords + min_coords) / 2
        limits_min = center - (1 + margin) * char_scale / 2
        limits_max = center + (1 + margin) * char_scale / 2

        ### Update layout with calculated limits
        fig.update_layout(
            width=1200,
            height=800,
            scene=dict(
                xaxis=dict(range=[limits_min[0], limits_max[0]], title="North"),
                yaxis=dict(range=[limits_min[1], limits_max[1]], title="East"),
                zaxis=dict(range=[limits_min[2], limits_max[2]], title="Down"),
                aspectmode="cube",
                camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            title=f"Calibration source - Position and Orientation at time : {self.datetime[index]}",
            showlegend=True,
            legend=dict(x=1.1, y=0.5),
        )

        fig.show()

    def plot_calsource_deviation_alt(self, index):
        """Plot calsource deviation alt.

        Alternative to plot the deviation of the calsource using matplotlib only.

        Parameters
        ----------
        index : int
            Index of the observation time.
        """

        ### Initialize the 3d figure
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection="3d")

        ### Plot antenna1, antenna 2, base antenna, qubic and the calibration source
        ax.scatter(self.position_antenna1[0, index], self.position_antenna1[1, index], self.position_antenna1[2, index], color="darkblue", marker="s", s=100, label="Antenna 1")
        ax.scatter(self.position_antenna2[0, index], self.position_antenna2[1, index], self.position_antenna2[2, index], color="darkblue", marker="^", s=100, label="Antenna 2")
        ax.scatter(self.position_calsource[0, index], self.position_calsource[1, index], self.position_calsource[2, index], color="darkred", marker="*", s=100, label="Calibration Source")
        ax.scatter(self.position_ini_antenna1[0], self.position_ini_antenna1[1], self.position_ini_antenna1[2], color="b", marker="s", s=100, label="Initial Antenna 1")
        ax.scatter(self.position_ini_antenna2[0], self.position_ini_antenna2[1], self.position_ini_antenna2[2], color="b", marker="^", s=100, label="Initial Antenna 2")
        ax.scatter(self.position_ini_calsource[0], self.position_ini_calsource[1], self.position_ini_calsource[2], color="r", marker="*", s=100, label="Initial Calibration Source")
        ax.scatter(self.base_antenna_position[0], self.base_antenna_position[1], self.base_antenna_position[2], color="k", s=100, label="Base Antenna")
        ax.scatter(self.position_qubic[0], self.position_qubic[1], self.position_qubic[2], color="pink", marker="o", s=100, label="QUBIC")

        ### Plot the vector between antenna 1 and antenna 2
        ax.quiver(
            self.position_antenna1[0, index],
            self.position_antenna1[1, index],
            self.position_antenna1[2, index],
            self.vector_1_2[0, index],
            self.vector_1_2[1, index],
            self.vector_1_2[2, index],
            color="darkblue",
            arrow_length_ratio=0.1,
            linewidth=2,
            label="Vector Antenna 1 to 2",
        )
        ax.quiver(
            self.position_ini_antenna1[0],
            self.position_ini_antenna1[1],
            self.position_ini_antenna1[2],
            self.vector_1_2_ini[0],
            self.vector_1_2_ini[1],
            self.vector_1_2_ini[2],
            color="b",
            arrow_length_ratio=0.1,
            linewidth=2,
            label="Initial Vector Antenna 1 to 2",
        )

        ### Plot the vector between QUBIC and the calibration source
        ax.quiver(
            self.position_calsource[0, index],
            self.position_calsource[1, index],
            self.position_calsource[2, index],
            self.vector_calsource_orientation[0, index],
            self.vector_calsource_orientation[1, index],
            self.vector_calsource_orientation[2, index],
            color="darkred",
            arrow_length_ratio=0.1,
            linewidth=2,
            label="Vector Calibration Source Deviation",
        )
        ax.quiver(
            self.position_ini_calsource[0],
            self.position_ini_calsource[1],
            self.position_ini_calsource[2],
            self.vector_calsource_qubic_ini[0],
            self.vector_calsource_qubic_ini[1],
            self.vector_calsource_qubic_ini[2],
            color="red",
            arrow_length_ratio=0.1,
            linewidth=2,
            label="Vector QUBIC to Calibration Source",
        )

        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1, 1])
        ax.set_xlabel("North", fontsize=12, labelpad=10)
        ax.set_ylabel("East", fontsize=12, labelpad=10)
        ax.set_zlabel("Down", fontsize=12, labelpad=10)

        ax.set_title(f"Calibration source - Position and Orientation - {self.datetime[index]}", fontsize=16, pad=20)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.legend()

    def plot_position_calsource_azel(self, start_index=0, end_index=-1):
        """Plot position calsource azel.

        Function to plot the evolution of the position of the calibration source in azel.
        """

        plt.figure()

        az, el = self.position_calsource_azel[:, start_index:end_index]

        plt.plot(az, el, ".", label="Calibration Source")
        plt.xlabel("Azimuth [deg]")
        plt.ylabel("Elevation [deg]")
        plt.title("Calibration Source Position in Azimuth and Elevation")
        plt.grid(True)
        plt.show()
