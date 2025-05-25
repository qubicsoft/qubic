import os
import re
import sys
import glob

import toml
import logging
import time as tm
import numba as nb
import numpy as np
from tqdm import tqdm
import shutil, atexit, tempfile
from datetime import datetime as dt

from scipy.interpolate import LSQUnivariateSpline
from scipy.stats import landau, expon, norm, cauchy

import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager

from model import Model as md
from utils import write_results
from plots import Plots as my_plt
from utils import get_tes_asic_from_index
from mixedDistribution import MixedDistribution

from qubicpack.qubicfp import qubicfp
from qubicpack.utilities import NPIXELS

import matplotlib.pyplot as plt



class Crd:
    """
    The class performs cosmic ray detection and signal deconvolution of TES data,
    removing the instrument's transfer function (characterized by the behavior of a low-pass filter).
    This transfer function exhibits a response similar to that induced by cosmic rays in the TOD.
    As a result, the signal is effectively cleaned of cosmic ray artifacts.

    Args
    ----------
    source                    (str | list[str]): path to the datasets
    dest                      (str | list[str]): path to the saving point folder
    points_vertical_trend                 (int): minimum number of points required to define a vertical trend
    points_exp_decrease                   (int): minimum number of points required to define an exponential decrease
    mask                (str | list[list[int]]): path to the mask file
    analysis_type                         (str): TES analysis mode
    analysis_type_args               (list[int]): specific arguments for TES analysis mode
    deconvolution                         (bool): if True, it performs signal deconvolution of TES data
    remove_files                          (bool): if True, it removes the npy and npz files which are not necessary
    remove_drift                          (bool): if True, it removes the atmospheric drift
    save_results                          (bool): if True, it saves the results to a json file

    Attributes
    ----------
    (All of the above)

    logger   (logging.Logger): logger
    logfname            (str): log file name
    fs                (float): sampling frequency

    __observation_dir           (str): path to the result folder, it is specific to each dataset analyzed
    __input_dir                 (str): path to the input folder, it contains the npy and npz files
                                       (it is contained in the relative observation folder)
    __output_dir                (str): path to the output folder, it contains the plots of the cosmic rays and
                                       the information regarding the cosmic rays found
                                       (it is contained int the relative observation folder)
    __dataset  (dict[str, list[str]]): a dictionary where the keys represent folders designated to store the results
                                       of specific scanning strategies, and the values specify the paths to the
                                       corresponding datasets to be analyzed
    __plots_dir                 (str): path to the plots folder, it is contained into the relative observation folder
    __taus_fname                (str): path to the taus file, it contains the information about the time constants
    __time_fname                (str): path to the times file (npy)
    __signals_fname             (str): path to the TES signals files (npz)
    __signals_clean_fname       (str): path to the TES signals files (npz) cleaned from the atmospheric drift
    __interp_elev_fname         (str): path to the elevation files (npz) interpolated over the time of the electronics
    __thermometers            (tuple): list of focal plane thermometers
    __n_knots                   (int): number of knots of the spline fit (used to fit the atmospheric drift)
    __t_knots            (np.ndarray): times values of the knots of the spline fit
    __std_coeff                 (int): multiplicative factor used to set the threshold from which to search cosmic rays.
                                       The threshold is: __std_coeff * std(signal) + mean(signal)
    __taus                     (dict): dictionary which contains the information about the cosmic rays found
    """

    # Class attribute representing the TES analysis mode (refer to the `check_mode` method)
    MODE = ('all', 'range', 'sequence')

    def __init__(self,
                 source: str | list[str],
                 dest: str | list[str],
                 points_vertical_trend: int = 3,
                 points_exp_decrease: int = 6,
                 mask: str | list[list[int]] = '',
                 analysis_type: str = 'all',
                 analysis_type_args: list[int] = None,
                 deconvolution: bool = True,
                 remove_files: bool = False,
                 remove_drift: bool = False,
                 save_results: bool = True):

        # attributes
        self.source = source
        self.dest = dest
        self.points_exp_decrease = points_exp_decrease
        self.points_vertical_trend = points_vertical_trend
        self.mask = mask
        self.analysis_type = analysis_type
        self.analysis_type_args = analysis_type_args
        self.deconvolution = deconvolution
        self.remove_files = remove_files
        self.remove_drift = remove_drift
        self.save_results = save_results

        self.logger: logging.Logger = None
        self.logfname: str = ''
        self.fs = 0.

        self.__observation_dir: str = ''
        self.__input_dir: str = ''
        self.__output_dir: str = ''
        self.__datasets: dict[str, list[str]] = dict()
        self.__plots_dir = ''
        self.__taus_fname = ''
        self.__time_fname = ''
        self.__signals_fname = ''
        self.__signals_clean_fname = ''
        self.__interp_elev_fname = ''
        self.__thermometers = (4, 36, 68, 100)
        self.__n_knots = 7
        self.__t_knots: np.ndarray = None
        self.__std_coeff = 5
        self.__taus = dict()

    # Getter/Setter method to retrieve/modify the path of private attributes.
    # `property` is a decorator that provides getter/setter methods.
    # Here, it is used to access private attributes as if they were public attributes.
    # Used only for testing purposes (should not be used for other purposes).
    @property
    def observation_dir(self):
        return self.__observation_dir

    @property
    def signals_fname(self):
        return self.__signals_fname

    @property
    def signals_clean_fname(self):
        return self.__signals_clean_fname

    def get_dest(self):
        return self.dest

    @classmethod
    def read_config(cls, path: str):
        """
        Reads a TOML configuration file and returns a Crd object configured accordingly.

        Parameters
        ----------
        path : str
            File system path to the TOML configuration file.

        Returns
        -------
        Crd
            A new instance of the Crd class, with its attributes
            set based on the contents of the TOML configuration file.
        """

        # Load the TOML configuration from the specified path
        config = toml.load(path)

        # Retrieve mandatory source path from the config
        source = config["path"]["source"]

        # Retrieve optional destination path or default to the current working directory
        dest = config["optional"].get("destination", os.getcwd())

        # Retrieve optional mask information or None if not specified
        mask = config["optional"].get("mask", None)

        # Retrieve optional number of points for vertical trend; defaults to 3
        vertical_points = config["optional"].get("vertical_points", 3)

        # Retrieve optional number of points for exponential decay; defaults to 6
        exponential_points = config["optional"].get("exponential_points", 6)

        # Retrieve optional analysis type; defaults to 'all' (all TES to be analyzed)
        analysis_type = config["optional"].get("analysis_type", "all")

        # Retrieve TES analysis argument, which may be a list; defaults to [-1] (all TES to be analyzed)
        tes = config['optional'].get('tes', [-1])

        # Retrieve optional boolean for drift removal; defaults to False
        remove_drift = config["optional"].get("remove_drift", False)

        # Retrieve optional boolean for file removal; defaults to True
        remove_files = config["optional"].get("remove_files", True)

        # Retrieve optional boolean for signal deconvolution; defaults to True
        deconvolution = config["optional"].get("deconvolution", True)

        # Retrieve optional boolean for saving results; defaults to True
        save_results = config["optional"].get("save_results", True)

        # Retrieve optional dpi resolution
        my_plt.dpi = config["optional"].get("dpi", 500)

        # Instantiate and return a Crd object with the extracted configuration
        return cls(source=source,
                   dest=dest,
                   points_vertical_trend=vertical_points,
                   points_exp_decrease=exponential_points,
                   mask=mask,
                   analysis_type=analysis_type,
                   analysis_type_args=tes,
                   deconvolution=deconvolution,
                   remove_drift=remove_drift,
                   remove_files=remove_files,
                   save_results=save_results)

    def get_datasets(self):
        """
        Searches for dataset paths from the user-specified source(s) and organizes them
        into a dictionary for further analysis

        Returns
        -------
        dict[str, list[str]]:
             A dictionary in which each key is a folder designated for storing the results of
            a specific scanning strategy, and each value is a list of paths to the corresponding
            datasets to be analyzed
        """

        # If 'source' is a single path, convert it to a list; otherwise, use the original list
        source = self.source if isinstance(self.source, list) else [self.source]

        # Iterate over each path in 'source'
        for path in source:

            # Prepare a list to hold the dataset paths for each scanning strategy
            datasets = []
            # 'tokens' is a list of scanning strategies or the single destination path
            tokens = [self.dest]

            # If the path contains parentheses, it indicates multiple scanning strategies
            if '(' in path:
                # Split the path into the main (root) part and the parenthetical part (other)
                root, other = path.split('(')

                # Extract tokens (scanning strategies) within the parentheses.
                # Each token is trimmed of whitespace, and the trailing ')*' is removed
                tokens = [token.strip() for token in other.rstrip(')*').split(',')]

                # For each strategy, append the root path plus token (with a wildcard) to 'datasets'
                datasets.extend([root + token + '*' for token in tokens])
            else:
                # If there's no parenthesis, treat the path as a single dataset
                datasets.append(path)

            # Match each scanning strategy with the corresponding dataset path
            for dt_category, dt_path in zip(tokens, datasets):

                # If the strategy token differs from 'self.dest', join them into one path
                dt_category = dt_category if dt_category == self.dest else os.path.join(self.dest, dt_category)
                # Initialize or retrieve the existing list of dataset paths for this strategy
                self.__datasets[dt_category] = self.__datasets.get(dt_category, [])

                # Look for valid dataset folders by searching through all matching directories
                for folder in glob.glob(dt_path):
                    # Recursively walk through each folder, checking subdirectories and files
                    for root, dirs, files in os.walk(folder):
                        # A valid dataset folder must contain these specific subdirectories
                        if all(d in dirs for d in ['Hks', 'Raws', 'Sums']):
                            self.__datasets[dt_category] = self.__datasets[dt_category] + [root]

                # Sort the dataset paths for this strategy in descending (reverse) order
                self.__datasets[dt_category] = sorted(self.__datasets[dt_category], reverse=True)

        # Return the dictionary of scanning strategies and their corresponding dataset paths
        return self.__datasets

    @staticmethod
    def cmp_mask_by_date(target: str, selected: str) -> float:
        """
        Compares two mask filenames based on their date/timestamp portion and returns
        the time difference in seconds.

        Parameters
        ----------
        target : str
            Filename (or string) representing the first mask. Must contain a datetime string
            at the start, formatted as 'YYYY-MM-DD_HH.MM.SS...'.
        selected : str
            Filename (or string) representing the second mask. Must contain a datetime string
            at the start, formatted as 'YYYY-MM-DD_HH.MM.SS...'.

        Returns
        -------
        float
            The difference in seconds between the two extracted timestamps (target - selected)
        """

        # Extract and parse the datetime portion (first two tokens separated by '_') in the 'target' string
        t_dtime = dt.strptime('_'.join(target.split('_')[:2]), '%Y-%m-%d_%H.%M.%S')
        # Extract and parse the datetime portion (first two tokens separated by '_') in the 'selected' string
        s_dtime = dt.strptime('_'.join(selected.split('_')[:2]), '%Y-%m-%d_%H.%M.%S')

        # Return the difference in total seconds between the two parsed datetimes
        return abs(t_dtime - s_dtime).total_seconds()

    @staticmethod
    def check_mask(mask_fname: str) -> list[list[int]] | None:
        """
        Validates a mask file that contains TES indices for two different signs (positive or negative).
        The file can have at most two rows:
          • The first row for positive TES indices, or -1 if none.
          • The second row for negative TES indices, or -1 if none.

        Examples of valid mask files:
        1) No positive TES, only negative TES:
             -1
             2 4 5 7 13 14 15 18 19 25 26 28 29 30 31 32 33 50
        -----------------------------------------------------
        2) Only positive TES, no negative TES:
             2 4 5 7 13 14 15 18 19 25 26 28 29 30 31 32 33 50
             -1
        -----------------------------------------------------
        3) Both positive and negative TES:
             2 4 5 7 13 14 15 18 19 25 26 28 29 30 31 32 33 50
             52 60 71 65 74 79
        -----------------------------------------------------

        Parameters
        ----------
        mask_fname : str
            Path to the file containing the TES mask. Each row must consist of
            integers (digit strings), either multiple non-negative integers or
            a single -1.

        Returns
        -------
        list[list[int]] | None
            A list of rows, where each row contains the integer-converted TES indices.
            Returns None if the file is invalid or does not adhere to the required
            structure.
        """

        # Check if the mask file exists
        if not os.path.isfile(mask_fname):
            return None

        # Read the entire file into a list of lines
        with open(mask_fname, 'r') as fin:
            mask = fin.readlines()

        # Verify that the file contains 1 or 2 lines only, and all tokens are either digits or negative integers
        if not len(mask) or len(mask) > 2 or not all(
                i.isdigit() or int(i) < 0 for r in mask for i in r.strip().split()):
            return None

        # Convert the lines into a nested list of integers
        mask = [list(map(int, row.strip().split())) for row in mask]

        # In rows that contain multiple values, ensure all values are non-negative.
        # A single -1 is allowed to signal "no valid TES" for that row
        if any(i < 0 for r in mask for i in r if len(r) != 1):
            return None

        # If all checks pass, return the parsed mask data
        return mask

    def select_mask(self, temperature: int) -> list[list[int]] | None:
        """
        Selects a valid TES mask file (closest in date, then in temperature) for a dataset.

        The method proceeds as follows:
          1. If the 'self.mask' attribute is a file path and the file is valid, it returns that mask.
          2. Otherwise, it searches within a directory of potential masks, sorting them by date
             (relative to the dataset) and then by temperature (closest to the provided 'temperature').
          3. Once a valid mask is found, it sets 'self.mask' to that mask and returns it.

        Parameters
        ----------
        temperature : int
            The temperature (in mK) of the current dataset.

        Returns
        -------
        list[list[int]] | None
            The selected TES mask as a nested list of integers, or None if no valid mask is found.
        """

        # 1. If 'self.mask' is a single file path, attempt to validate and return it
        if os.path.isfile(self.mask):
            self.mask = self.check_mask(self.mask)
            self.logger.info('Mask selected: %s', self.mask)
            return self.mask

        # 2. If 'self.mask' is a folder, gather all potential mask files within it.
        #    Use a wildcard if not already present to match all files in the folder.
        masks = glob.glob(self.mask + (os.sep + '*' if '*' not in self.mask else ''))

        # Sort the retrieved mask files by their date difference from the dataset.
        # The comparison uses 'cmp_mask_by_date', which returns a signed difference in seconds.
        masks = sorted(masks,
                       key=lambda x: self.cmp_mask_by_date(os.path.basename(self.source),
                                                           os.path.basename(x).replace('mask_', '')))

        # Regex pattern to extract the temperature in mK from the mask filename
        temp_pattern = re.compile(r'(\d+)mK', flags=re.IGNORECASE)

        # Extract the dataset date/time from the 'self.source' filename
        dtime_ref = dt.strptime('_'.join(os.path.basename(self.source).split('_')[:2]), '%Y-%m-%d_%H.%M.%S')

        # Start with the year of the dataset, and then refine the filter by month and day
        mask_pattern = str(dtime_ref.year)

        # Filter mask files by matching them with the dataset's year, month, and day.
        # Each iteration refines the pattern, ensuring we only retain time-compatible masks.
        # f'{x:02}': this notation indicates that if the number occupies less than two characters,
        # the remaining character is set equal to zero
        for token in list(map(lambda x: f'{x:02}', [dtime_ref.month, dtime_ref.day])):

            # Filter the list for masks that contain 'mask_pattern'
            if not (result := list(filter(lambda x: mask_pattern in x, masks))):
                # If no masks match in the current iteration, retain those from the previous step
                break

            # Restrict 'masks' to those that passed the filter
            masks = result
            # Extend 'mask_pattern' with the month (and later the day) for a more precise match
            mask_pattern = '-'.join([mask_pattern, token])

        # 3. At this stage, 'masks' should contain only time-compatible options.
        #    Now sort them by temperature proximity to the dataset's temperature and pick the first valid one
        for mask in sorted(masks, key=lambda x: abs(temperature - int(temp_pattern.search(x).group(1)))):
            # Use 'check_mask' to verify the validity of the selected mask file.
            if selected_mask := self.check_mask(mask):
                self.logger.info('Dataset temp: %s mK | Mask selected: %s', temperature, mask)
                self.mask = selected_mask
                return self.mask

    def make_result_dir(self):
        """
        Creates a results directory for each dataset under analysis and initializes the logger.

        The method does the following:
        1. Builds a folder name for storing dataset results.
        2. Creates the folder if it doesn't already exist.
        3. Sets up the logger, ensuring a unique log file is created for each dataset.
        """

        # Build the folder name using a prefix plus the base name of the source path
        folder = "crd_" + os.path.split(self.source)[-1]

        # Combine the destination path and the folder name to get the full results directory path
        self.__observation_dir = os.path.join(self.dest, folder)

        # Create the results directory (if it exists, no exception is raised)
        os.makedirs(self.__observation_dir, exist_ok=True)

        # Construct the full path to the log file
        self.logfname = os.path.join(self.__observation_dir, f"{folder}.log")

        # Configure the logging module so that a new log file is created per dataset
        # force = True ensures that one logfile is created for each dataset analysed
        logging.basicConfig(filename=self.logfname,
                            filemode="w",
                            encoding='utf8',
                            format="%(asctime)s - %(funcName)s - %(message)s",
                            datefmt="%d/%m/%Y | %H:%M:%S",
                            level=logging.INFO,
                            force=True)

        # Create a logger for this module
        self.logger = logging.getLogger(__name__)

    def export_data(self):
        """
        Selects a valid mask for the current dataset and exports TES signals according to
        the sign specified in the mask. Computes the sampling frequency, saves the housekeeping
        time array, elevation values, and the time of the electronics.

        Steps:
          1. If the time and signals files already exist, load them and set the sampling frequency.
          2. Otherwise, read the dataset from the QubicFP, compute and store the time axis,
             select the appropriate mask, and export the masked TES signals to an .npz file.

        Returns
        -------
        list[str]
            The list of dataset keys (TES indices) if the files exist; otherwise, returns
            an empty list after a new export is performed.
        """

        # If both the time file and signals file exist, load and reuse them
        if os.path.isfile(self.__time_fname) and os.path.isfile(self.__signals_fname):
            # Load the time axis from the existing .npy file
            tm = np.load(self.__time_fname)
            # Compute the sampling frequency (assume uniform sampling, so fs = 1 / delta_t)
            self.fs = 1 / tm[1]
            # Return the list of keys (TES indices) from the loaded signals .npz file
            return np.load(self.__signals_fname).files

        # Otherwise, export data from QubicFP
        print("Exporting data via qubicfp...", end="")

        # Create a QubicFP instance and suppress verbosity for minimal console output
        qubic = qubicfp()
        qubic.verbosity = 0

        # Read QubicStudio dataset from the directory specified by self.source
        qubic.read_qubicstudio_dataset(datadir=self.source)
        # Retrieve the time axis (tm) and the corresponding TES signals
        tm, signals = qubic.tod()

        # Compute the sampling frequency from the time axis
        self.fs = 1 / (tm[1] - tm[0])
        # Convert the measured temperature from mK to K and select the appropriate mask
        # (qubic.temperature is expressed in mK)
        self.mask = self.select_mask(temperature=round(qubic.temperature * 1000))

        # Attempt to read housekeeping (HK) time data and elevation
        if not (t_hk := qubic.timeaxis(datatype='platform')).size:
            # If HK data is not available, issue a warning
            self.logger.warning("No time axis housekeeping data")

        else:
            # Interpolate elevation over the electronics time axis and save it
            np.save(self.__interp_elev_fname, np.interp(tm - tm[0], t_hk - t_hk[0], qubic.elevation()))

        # Save the shifted time array to .npy
        np.save(self.__time_fname, tm - tm[0])

        # Prepare lists to store TES indices (keys) and their corresponding signals (masked)
        keys = []
        masked = []

        # If no valid mask is specified, consider all TES indices (positive) by default
        self.mask = self.mask or [list(range(signals.shape[0]))]

        # For each row in the mask:
        #   • Row 0 corresponds to positive TES
        #   • Row 1 corresponds to negative TES
        for row in range(len(self.mask)):
            # If this row contains only -1, skip because it indicates no valid TES for that sign
            if len(self.mask[row]) == 1 and self.mask[row][0] == -1:
                continue

            # Iterate over the TES indices in the current row
            for tes in self.mask[row]:
                # If the TES index is out of range, stop processing
                if tes >= len(signals):
                    break
                # Record the TES index as a string key
                keys.append(str(tes))
                # Multiply the signal by (-1)^row to account for sign
                masked.append((-1) ** row * (signals[tes] / max(np.fabs(signals[tes].min()), signals[tes].max())))

        # Save the TES signals in an .npz file, using TES indices as keys
        np.savez(self.__signals_fname, **dict(zip(keys, masked)))

    def configure_data(self):
        """
        Creates all necessary directories for input/output data, sets up the paths
        used during analysis, and exports the required data files for processing.

        Specifically, this method performs the following steps:
          1. Logs details about the input/output folder configuration and dataset source.
          2. Builds paths for input, output, and plot directories, then creates them if needed.
          3. Defines filenames for time arrays, raw signals, optionally drift-cleaned signals,
             interpolated elevation, and cosmic-ray time constants (taus).
          4. Exports the raw TES data (and any required additional data) for analysis.
        """

        self.logger.info('input and output folders of preprocessed data (npy, npz files + taus, plots): %s',
                         self.dest)

        self.logger.info('folder containing observation data in .fits format: %s',
                         self.source)

        # Build the path for input and output directories within the results folder
        self.__input_dir = os.path.join(self.__observation_dir, "input")
        self.__output_dir = os.path.join(self.__observation_dir, "output")

        self.logger.info('input dir: %s', self.__input_dir)
        self.logger.info('output dir: %s', self.__output_dir)

        # Create the input and output directories if they do not exist
        os.makedirs(self.__input_dir, exist_ok=True)
        os.makedirs(self.__output_dir, exist_ok=True)

        self.logger.info('input dir created')
        self.logger.info('output dir created')

        # Define file paths for the time array, raw signals, optionally drift-cleaned signals,
        # and interpolated elevation data
        self.__time_fname = os.path.join(self.__input_dir, "times_raw.npy")
        self.__signals_fname = os.path.join(self.__input_dir, "signals_raw.npz")
        self.__signals_clean_fname = os.path.join(self.__input_dir,
                                                  "signals_clean.npz") if self.remove_drift else self.__signals_fname
        self.__interp_elev_fname = os.path.join(self.__input_dir, "interp_elev.npy")

        # Define the path for cosmic-ray time constants (taus) output
        self.__taus_fname = os.path.join(self.__output_dir,
                                         f"crd_taus__{os.path.split(self.__observation_dir)[-1]}.json")

        self.logger.info('time file path: %s', self.__time_fname)
        self.logger.info('raw signal file path: %s', self.__signals_fname)
        self.logger.info('signal clean file path: %s', self.__signals_clean_fname)
        self.logger.info('interp elevation file path: %s', self.__interp_elev_fname)
        self.logger.info('taus file path: %s', self.__taus_fname)

        # Build and log the plots directory path
        self.__plots_dir = os.path.join(self.__output_dir, "plots")
        self.logger.info('plots dir: %s', self.__plots_dir)

        # Create the plots directory if it does not already exist
        os.makedirs(self.__plots_dir, exist_ok=True)
        self.logger.info('plots dir created')

        # Finally, export the data (TES signals, housekeeping info, etc.) for further analysis
        self.export_data()

    @staticmethod
    def remove_lsql(n_tes: int, time_shm: dict, t_knots: np.ndarray, signals_fname: str) -> np.ndarray:
        """
        Calculates and removes the LSQUnivariateSpline fit for a given TES signal, effectively
        subtracting the atmospheric drift component.

        Parameters
        ----------
        n_tes : int
            TES index (valid range 0 to 255).
        time_shm : dict
            Shared memory block identifier for time data. Expected keys:
              • 'shape' for the shape of the time array
              • 'dtype' for the data type
              • 'shm' for the shared memory object
        t_knots : np.ndarray
            The time values of the knots for the spline fit.
        signals_fname : str
            File path to the saved raw signals (in an .npz file).

        Returns
        -------
        np.ndarray
            The TES signal after subtracting the LSQUnivariateSpline fit (i.e., without atmospheric drift).
        """

        # Create a NumPy array connected to the shared memory block
        # time_shm['shm'].buf points to the shared memory region
        sig_tm = np.ndarray(shape=time_shm['shape'], dtype=time_shm['dtype'], buffer=time_shm['shm'].buf)

        # Load the raw TES signal from the specified .npz file
        signal_raw = np.load(signals_fname)[n_tes]

        # Create an LSQUnivariateSpline object (least squares spline fit) for the TES signal
        fit = signal_raw - LSQUnivariateSpline(sig_tm, signal_raw, t_knots)(sig_tm)

        # Subtract the spline fit from the raw TES signal and return it

        # filtered = pp.apply_wiener_filter_stft(fit, sig_tm[1])
        return fit  # filtered[:len(sig_tm)]

    def remove_atmospheric_drift(self) -> list[int]:
        """
        Creates an .npz file containing signals that have been cleaned of atmospheric drift.
        This method uses shared memory to parallelize the spline fitting process for each TES.

        Returns
        -------
        list[int]
            List of TES indices that are available for further analysis.
        """

        # Load the file containing the raw signals
        signals = np.load(self.__signals_fname)
        self.logger.info("%s signals will be exported in .npz format", len(signals))

        # If the drift-cleaned signal file already exists, skip regeneration
        if os.path.isfile(self.__signals_clean_fname):
            self.logger.info("the file '%s' already exist. Skipping", self.__signals_clean_fname)
            return list(map(int, np.load(self.__signals_clean_fname).files))

        print('Saving signals cleaned from the atmospheric drift...', end="")

        # Use a SharedMemoryManager context so that all shared memory is released automatically afterward
        with SharedMemoryManager() as smm:
            # Load the time array used for analysis
            sig_time = np.load(self.__time_fname)

            # the Schoenberg-Whitney conditions require taking the second value of the time array
            # and the second-to-last one
            self.__t_knots = np.linspace(sig_time[1], sig_time[-2], self.__n_knots)
            self.logger.info('t_knots: %s', self.__t_knots)

            # (1) Create a shared memory block sized for the time array
            shm = smm.SharedMemory(size=sig_time.nbytes)

            # (2) Create a NumPy array that maps onto the shared memory buffer
            dst = np.ndarray(shape=sig_time.shape, dtype=sig_time.dtype, buffer=shm.buf)

            # (3) Copy the original time data into the shared memory array
            dst[:] = sig_time[:]

            # (4) Prepare data for multiprocessing, including the shared memory reference
            time_shm = {'shm': shm, 'shape': sig_time.shape, 'dtype': sig_time.dtype}

            # Number of signals (i.e., number of TES)
            n_sigs = len(signals)

            # Launch a multiprocessing pool to call `get_interp_signal` for each TES in parallel
            with mp.Pool() as pool:
                res = pool.starmap(self.remove_lsql,
                                   zip(signals.files,  # TES indices (as strings) in the .npz file
                                       [time_shm] * n_sigs,  # Repeated shared memory info
                                       [self.__t_knots] * n_sigs,  # Repeated knot array
                                       [self.__signals_fname] * n_sigs),  # Repeated raw signals filename
                                   chunksize=max(1, n_sigs // mp.cpu_count()))

                # Save the drift-cleaned signals to a new .npz file, pairing TES keys with processed data
                np.savez(self.__signals_clean_fname, **dict(zip(signals.files, res)))

        # Return the list of TES indices (as integers) available in the new .npz file
        return list(map(int, signals.files))

    ## Caching of compiled functions has several known limitations:
    #
    # The caching of compiled functions is not performed on a function-by-function basis.
    # The cached function is the main jit function, and all secondary functions (those called by the main function)
    # are incorporated in the cache of the main function.
    #
    # Cache invalidation fails to recognize changes in functions defined in a different file.
    # This means that when a main jit function calls functions that were imported from a different module,
    # a change in those other modules will not be detected and the cache will not be updated.
    # This carries the risk that “old” function code might be used in the calculations.
    #
    # Global variables are treated as constants. The cache will remember the value of the global variable at
    # compilation time. On cache load, the cached function will not rebind to the new value of the global variable.

    @staticmethod
    @nb.njit(nogil=True, cache=True)
    def get_candidates(signal: np.ndarray,
                       std_mask: np.ndarray,
                       offset: int = 0,
                       lower: float = 0,
                       points_exp_decrease: float = 0) -> list[list[int]]:
        """
        Identifies potential cosmic ray candidates in the TES signal.
        Each candidate must show a vertical increase (linear growth) followed by
        an exponential decrease. The method checks whether segments of the
        signal exceed a specified threshold and then tracks their indices.

        Parameters
        ----------
        signal : np.ndarray
            Array containing the TES signal values.
        std_mask : np.ndarray
            Indices of signal points exceeding a certain threshold
            (e.g., 5 * std(signal) + mean(signal)).
        offset : int, optional
            An offset added to candidate indices to map them back to
            the larger time-domain (default is 0).
        lower : float, optional
            A lower-bound threshold (default is mean(signal) + std(signal)).
            If the signal goes below this threshold, the candidate ends.
        points_exp_decrease : float, optional
            The minimum number of points required to confirm an exponential
            decreasing segment (default is 0).

        Returns
        -------
        list[list[int]]
            A list of [start_index, end_index] pairs (with offset applied)
            representing each detected cosmic-ray candidate.
        """

        # Tracks the index used to skip already processed points in std_mask
        j = 0
        # Holds the list of valid candidate segments
        candidates = list()

        # Iterate over all threshold-exceeding points
        for i in std_mask:
            # Skip points already part of a processed candidate
            if i < j:
                continue

            # Temporary storage for consecutive indices in a candidate
            points = []

            # iterate over the points above and below the threshold starting
            # from the index point "i"
            for j in range(i, signal.shape[0] - 1):

                # add indices to the candidate until the condition is verified:
                # i.e., when the value of the signal at index "j" is less than
                # the value of the signal at index "j+1"
                if signal[j] < signal[j + 1] or signal[j + 1] < lower:
                    # Add the last valid index to 'points'
                    points.append(j)
                    # If we have enough points to qualify as exponential decrease
                    if len(points) >= points_exp_decrease:
                        # Determine where linear vertical growth started
                        start = points[0]

                        # find the number of points of linear growth.
                        # this while loop stops one index BEFORE the last value
                        # that satisfies the condition, so when I save the candidate
                        # I have to subtract 1 from start
                        while signal[max(start - 1, 0)] > signal[max(start - 2, 0)] > lower:
                            start -= 1

                        # candidate is vertical growth followed by decreasing exponential.
                        # Contains the initial index of linear growth and the final index
                        # of exponential decreasing trend
                        candidates.append([max(offset + start - 1, 0), offset + points[-1] + 1])

                    break

                # If the signal is still steadily decreasing (>= next points), keep collecting
                points.append(j)

        return candidates

    def check_mode(self, tes_keys: list[int]) -> list[int]:
        """
        Checks and sets the TES analysis mode. The possible modes are:
          • 'all': Analyze all TES.
          • 'range': Analyze a continuous range of TES indices (e.g., [10, 20]).
          • 'sequence': Analyze a sequence of TES indices (e.g., [1, 5, 80, 100]).

        Parameters
        ----------
        tes_keys : list[int]
            The keys corresponding to the TES signals present in the .npz files.

        Returns
        -------
        list[int]
            A list of TES indices to be analyzed according to the chosen mode.
        """

        # If the analysis mode indicates 'all' (i.e., a single value -1), analyze all available TES
        if len(self.analysis_type_args) == 1 and self.analysis_type_args[0] == -1:
            self.analysis_type_args = tes_keys

        # Otherwise, if no arguments exist or there is an invalid TES in the list, raise an error
        elif not self.analysis_type_args or (
                isinstance(self.analysis_type_args, list) and any(
            tes not in tes_keys for tes in self.analysis_type_args)):
            error = (f"Some selected TES are invalid\n{self.analysis_type =} | {self.analysis_type_args=}\n"
                     f"Available tes index: {tes_keys}")
            raise ValueError(error)

        # If the mode is 'range', reduce self.analysis_type_args to the specified interval
        if self.analysis_type == 'range':
            self.analysis_type_args = tes_keys[tes_keys.index(self.analysis_type_args[0]): tes_keys.index(
                self.analysis_type_args[1]) + 1]

        # If the mode is 'sequence', use self.analysis_type_args as-is (already a sequence)
        elif self.analysis_type == 'sequence':
            # No changes needed; self.analysis_type_args is already the sequence of TES
            pass

        # If the mode is neither 'range' nor 'sequence', default to analyzing all valid TES
        else:
            self.analysis_type_args = tes_keys

        # Log the mode and the final list of TES to be analyzed
        self.logger.info("Mode: %s | Tes: %s", self.analysis_type, self.analysis_type_args)

        return self.analysis_type_args

    def get_taus(self, tes_idx: list[int]) -> dict[int, dict[str, list[float]]]:
        """
        Computes and returns the time constants for each TES (Transition Edge Sensor)
        by processing their cleaned signals and fitting an exponential decay model.

        Parameters
        ----------
        tes_idx : list[int]
            List of all available TES indices for analysis.

        Returns
        -------
        dict[int, dict[str, list[float]]]
            A dictionary where each key is a TES index and its value is another dictionary
            containing details about the time constant (tau) and associated fit parameters.
            The inner dictionary includes keys such as:
              - 'taus': list of time constants (tau)
              - 'tes': the TES and asic identifiers
              - 'exp fit params': exponential fit parameters
              - 'sigma': standard deviation from the fit
              - 'chi square': chi-square statistics of the fit
              - 'residuals': residuals from the fit
              - 'indexes': indexes corresponding to the candidate region in the signal
        """

        # Counter to track total number of time constants found
        all_taus_found = 0

        # Determine which TES to analyze based on the user-selected mode
        iter_ = self.check_mode(tes_idx)

        # Load the raw time array and the drift-cleaned signals from file
        time_raw = np.load(self.__time_fname)
        signals = np.load(self.__signals_clean_fname)

        # Create a multiprocessing pool to parallelize candidate processing and fitting
        with mp.Pool() as pool:

            # Iterate over each TES index to process its signal, with a progress bar
            for n_tes in tqdm(iter_, ncols=100, file=sys.stdout, desc="Progress", unit='tes', leave=False):
                # Retrieve the signal for the current TES (using string key)
                signal = signals[str(n_tes)]
                # Compute the standard deviation of the signal
                s_std = signal.std()
                # Compute the mean of the signal
                s_mean = signal.mean()

                # Define lower and upper thresholds for candidate search:
                # lower: baseline threshold; upper: threshold for significant events
                lower = s_std + s_mean
                upper = self.__std_coeff * s_std + s_mean

                # Identify indices where the signal exceeds the upper threshold
                std_mask = np.where(signal > upper)[0]

                # Process only if there are indices above the threshold
                if std_mask.size:

                    # Determine a "before" window to capture pre-event dynamics, ensuring it doesn't go negative
                    before = max(std_mask[0] - 3 * self.points_vertical_trend, 0)
                    # Determine an "after" window to capture the complete candidate, adding a margin for search
                    # (the "+1" is needed because, by default, the right endpoint is excluded in slicing.
                    # Therefore, writing `signal[mask[0]:mask[-1]]` excludes the last starting point
                    # for searching for a candidate)
                    after = min(std_mask[-1] + 1 + 5 * self.points_exp_decrease, len(signal))

                    # Identify candidate regions using the get_candidates method over a slice of the signal
                    # (the parameter offset is necessary because, when I execute the candidate filter,
                    # I retrieve the data for each candidate starting from the entire signal and time arrays)
                    candidates = Crd.get_candidates(
                        signal=signal[std_mask[0] - before: std_mask[-1] + after],
                        std_mask=std_mask - std_mask[0] + before,
                        points_exp_decrease=self.points_exp_decrease,
                        offset=abs(std_mask[0] - before),
                        lower=lower)

                    # If candidates are found, further process them
                    if candidates:

                        # Determine chunk size for multiprocessing based on number of candidates and CPU cores
                        chunksize = max(1, len(candidates) // mp.cpu_count())

                        # Prepare time segments for each candidate from the raw time array
                        tm_candidates = map(lambda c: time_raw[c[0]: c[1]], candidates)
                        # Prepare corresponding signal segments for each candidate
                        sig_candidates = map(lambda c: signal[c[0]: c[1]], candidates)

                        # Apply a multiprocessing filter to refine candidates using the candidate_filter function
                        candidates = md.multiprocess_filter(pool=pool,
                                                            candidates=candidates,
                                                            function=md.candidate_filter,
                                                            args=zip(tm_candidates,
                                                                     sig_candidates,
                                                                     [self.points_vertical_trend] * len(candidates)),
                                                            chunksize=chunksize)

                        # Recalculate chunksize for the filtered candidate list
                        chunksize = max(1, len(candidates) // mp.cpu_count())

                        # Update time and signal segments for the valid candidates
                        tm_candidates = map(lambda c: time_raw[c[0]:c[1]], candidates)
                        sig_candidates = map(lambda c: signal[c[0]:c[1]], candidates)

                        # Define an offset to capture additional points around candidates
                        # for standard deviation calculation
                        std_offset = 20
                        # Compute the standard deviation for each candidate by considering extra points before and after
                        std_candidates = map(lambda c:
                                             np.array([*signal[max(0, c[0] - std_offset): c[0]],
                                                       *signal[c[1]: min(len(signal), c[1] + std_offset)]]).std(),
                                             candidates)

                        # Apply a fitting function to each candidate to obtain a list of FitResult objects
                        taus_matrix = pool.starmap(md.get_fit_candidate,
                                                   zip(tm_candidates, sig_candidates, std_candidates),
                                                   chunksize)

                        # filtering to obtain valid fits only
                        valid_results = [(fit_res, idxs) for fit_res, idxs in zip(taus_matrix, candidates) if fit_res]

                        if valid_results:

                            all_taus_found += len(valid_results)
                            tes, asic = get_tes_asic_from_index(n_tes)

                            # For a given TES, iterates over each candidate's FitResult object
                            # and saves the fit information
                            taus, params, pcov, sigma, chi_sq, res, slopes, slope_errs, slope_p_values, idxs = zip(
                                *[
                                    (
                                        fit_res.tau,
                                        [fit_res.a, fit_res.b, fit_res.c],
                                        fit_res.pcov,
                                        fit_res.sigma_tau,
                                        [fit_res.nu, fit_res.chi_square_reduced, fit_res.p_value],
                                        fit_res.residuals,
                                        fit_res.slope,
                                        fit_res.slope_sigma,
                                        fit_res.slope_p_value,
                                        idx
                                    )
                                    for fit_res, idx in valid_results
                                ]
                            )

                            # For a given TES, save the fit information into the private `_taus` dictionary
                            self.__taus[n_tes] = {
                                'taus': taus,
                                'tes': [tes, asic],
                                'exp fit params': params,
                                'covariance matrix': pcov,
                                'sigma': sigma,
                                'chi square': chi_sq,
                                'residuals': res,
                                'slopes': slopes,
                                'slope errs': slope_errs,
                                'slope p values': slope_p_values,
                                'indexes': idxs
                            }
                            self.logger.info("#Tes (index): %s | #Time constants: %s", n_tes, len(taus))

            # Log the total number of time constants found across all TESs
            self.logger.info("Number of time constants for all TESs: %s", all_taus_found)

            # Return the dictionary containing time constants and fit parameters for each TES
            return self.__taus

    def multidataset_report(self,
                            fname: str,
                            mdt_report: dict,
                            make_plot: bool = False,
                            global_dest: str = None,
                            title: str = None):
        """
        Provides general insights based on the analysis results from individual datasets.
        It visualizes cosmic ray occurrences as a function of amplitude, event duration,
        and instrument elevation. For each TES, it also displays the estimated time constant
        along with its associated uncertainty.

        Parameters
        ----------
        fname : str
            Base filename or data path used for saving outputs.
        mdt_report : dict
            Dictionary containing all necessary information to generate the plots.
        make_plot : bool, optional
            If True, generate plots of the analysis results (default is False).
        global_dest : str, optional
            Destination folder for saving the analysis outputs; if not provided, self.dest is used.
        title : str, optional
            Title to be used for the generated plot (if any).
        """

        # Initialize an empty list for elevation values
        elev = []
        # Load the cleaned TES signals from file
        signals = np.load(self.__signals_clean_fname)
        # Load the time array corresponding to the signals
        tm_signals = np.load(self.__time_fname)

        # Check if an interpolated elevation file exists
        if os.path.isfile(self.__interp_elev_fname):
            # Load elevation data if available
            elev = np.load(self.__interp_elev_fname)

        # Loop over each TES for which time constants have been computed
        for tes in self.__taus:
            # Retrieve the signal for the current TES using its string key
            sig = signals[str(tes)]

            # If the TES is not yet in the multidataset report, initialize its structure
            if tes not in mdt_report:
                mdt_report[tes] = dict()  # Create a new dictionary for this TES
                mdt_report[tes]['taus'] = list()  # Initialize list for time constants
                mdt_report[tes]['energy'] = list()  # Initialize list for cosmic ray event amplitudes
                mdt_report[tes]['elevation'] = list()  # Initialize list for elevations at event occurrences
                mdt_report[tes]['time'] = list()  # Initialize list for event durations
                mdt_report[tes]['residuals'] = list()  # Initialize list for fit residuals

            energy = []  # List to store energy (amplitude) for current TES
            taus_tm = []  # List to store event duration (time differences) for current TES
            tes_taus = []  # List to store time constant and its uncertainty for current TES
            elevation = []  # List to store elevation data corresponding to events
            residuals = []  # List to collect residuals from candidate fits

            # Iterate over computed taus for current TES
            for idx, tau in enumerate(self.__taus[tes]['taus']):
                # Only consider tau values that are of the order of 10^-2
                if round(np.log10(tau)) == -2:
                    # Retrieve start and stop indices for the candidate
                    start, stop = self.__taus[tes]['indexes'][idx]

                    # Calculate and store the duration of the candidate event
                    taus_tm.append(tm_signals[stop - 1] - tm_signals[start])

                    # Append the time constant along with its uncertainty for this candidate
                    tes_taus.append([tau, self.__taus[tes]['sigma'][idx]])
                    # Calculate event energy as the difference between max and min signal values in the candidate range
                    energy.append(sig[start:stop].max() - sig[start:stop].min())

                    # Extend the residuals list with residuals from the candidate fit
                    residuals.extend(self.__taus[tes]['residuals'][idx])
                    # If elevation data is available, calculate and append the mean elevation for the candidate event
                    if len(elev):
                        elevation.append(
                            elev[start:stop].mean() if elev[0] > elev[-1] else (90 - elev[start:stop]).mean())

            # Update the multidataset report for the current TES with the collected data
            mdt_report[tes]['taus'].extend(tes_taus)
            mdt_report[tes]['energy'].extend(energy)
            mdt_report[tes]['elevation'].extend(elevation)
            mdt_report[tes]['time'].extend(taus_tm)
            mdt_report[tes]['residuals'].extend(residuals)

        # If plotting is requested, generate the multidataset report plots.
        # When the code analyzes the last dataset of a given scanning strategy
        # or the last dataset among all those to be analyzed (regardless of the scanning strategy),
        # the multi dataset plot is generated
        if make_plot:
            energy = []  # Reinitialize energy list for aggregated plotting.
            taus_tm = []  # Reinitialize event duration list for plotting.
            average_taus_per_tes = []  # List to collect averaged tau values from each TES.
            elevation = []  # List to collect elevation data for all TES.
            residuals = []  # List to collect all residuals for plotting.
            tes_with_taus = []  # List to record TES indices that have estimated taus.
            all_taus_sigma = []  # list to collect taus from all dataset, from all TES
            fig_kw = dict(figsize=(12, 6), tight_layout=True)  # Figure keyword arguments for plot styling

            # Loop over each TES in the multidataset report
            for tes in mdt_report:
                # If the TES has computed time constants, calculate its average tau and uncertainty
                if mdt_report[tes]['taus']:
                    all_taus_sigma.extend(mdt_report[tes]['taus'])

                    mean_tau, mean_sigma = np.mean(mdt_report[tes]['taus'], axis=0)
                    mdt_report[tes]['tau'] = [mean_tau, mean_sigma / len(mdt_report[tes]['taus']) ** 0.5]
                    tes_with_taus.append(tes)  # Add TES index to the list
                    average_taus_per_tes.append(mdt_report[tes]['tau'])  # Append the averaged tau and sigma
                    energy.extend(mdt_report[tes]['energy'])  # Aggregate energy values
                    elevation.extend(mdt_report[tes]['elevation'])  # Aggregate elevation data
                    taus_tm.extend(mdt_report[tes]['time'])  # Aggregate event duration values
                    residuals.extend(mdt_report[tes]['residuals'])  # Aggregate fit residuals

            global_average_tau, global_average_sigma = np.mean(all_taus_sigma, axis=0)
            global_average_sigma /= len(all_taus_sigma) ** 0.5

            # Create a figure and subplots for the multidataset report using a custom plotting function
            hist_fig, (ax_amp, ax_elev, ax_tm, ax_res, ax_tau) = my_plt.custom_subplots(2, [3, 2], **fig_kw)

            # Set the overall title of the figure, with spaces replaced by non-breaking spaces for formatting
            hist_fig.suptitle(rf'${title.replace(' ', '~')}$')

            # Plot histogram of cosmic ray amplitudes with a Landau distribution fit
            my_plt.plot_hist(ax_amp,
                             r'$\tau ~ vs ~ Amplitudes$',
                             r'$ Amplitude \ [ADU]$',
                             None,
                             '',
                             '',
                             to_plot=energy,
                             fit_func=landau,
                             fit_name_args=("loc = {:.4f},", "scale = {:.4f}"))

            # Plot histogram of elevation values
            my_plt.plot_hist(ax_elev,
                             r'$ \tau ~ vs ~ Elevation$',
                             r'$ Elevation ~ [degrees]$',
                             None,
                             '',
                             '',
                             to_plot=elevation)

            # Plot histogram of event durations with an exponential fit
            my_plt.plot_hist(ax_tm,
                             title=rf'$\tau ~ vs ~ Time ~ Frame: ~ {global_average_tau:.3g} \pm {global_average_sigma:.3g} $',
                             xlabel=r'$Time ~ [s]$',
                             taus=None,
                             data_path='',
                             datatype='time',
                             to_plot=[len(taus_tm) / (tm_signals[1] * len(tm_signals))] + taus_tm,
                             fit_func=expon,
                             fit_name_args=("loc = {:.2f},", "scale = {:.2f}"))

            # Plot histogram of fit residuals with a normal distribution fit
            normCauchy = MixedDistribution(norm, cauchy)
            my_plt.plot_hist(ax_res,
                             title=r'$Cosmic ~ Rays ~ Residual ~ Fit$',
                             xlabel=r'$Residuals$',
                             taus=None,
                             data_path='',
                             to_plot=residuals,
                             fit_func=normCauchy,
                             fit_name_args=(r'{:.2f} \% ~ outliers',),
                             datatype="residuals")

            # Configure the subplot for displaying estimated time constants for each TES
            ax_tau.set(facecolor='whitesmoke',
                       title=r"$Estimation~of~\tau$",
                       xlabel=r'$ \# TES$',
                       ylabel=r'$\tau$')

            # Plot error bars for the time constants with their uncertainties
            ax_tau.errorbar(tes_with_taus, [t[0] for t in average_taus_per_tes], [t[1] for t in average_taus_per_tes],
                            fmt='o', color='steelblue')

            ax_tau.grid(True)

            # Compute percentiles for the estimated tau values
            tau_values = [t[0] for t in average_taus_per_tes]

            percentiles = np.percentile(tau_values, [25, 50, 75])
            labels = [rf'${p}\%~=~{val:.5f}$' for p, val in zip([25, 50, 75], percentiles)]

            for val, label, color in zip(percentiles, labels, ['silver', 'gray', 'black']):
                ax_tau.axhline(val, label=label, linestyle='--', color=color)

            # Add a legend indicating the coverage percentage (based on number of taus and total TES pixels)
            ax_tau.legend(title=rf'$Coverage: ~ {len(average_taus_per_tes) / (2 * NPIXELS) * 100: .2f} \% $')

            # Determine the destination folder to save analysis results
            dest = global_dest or self.dest
            # Save the residuals to a npy file
            np.save(os.path.join(dest, f"{fname}_residuals"), residuals)
            self.logger.info(f"Saving {fname} residuals: # samples {len(residuals)}")

            # Define the file path for saving the multidataset report figur
            mdreport_filepath = os.path.join(dest, fname)
            self.logger.info("Saving multidataset report to %s", mdreport_filepath)
            hist_fig.savefig(mdreport_filepath, dpi=my_plt.dpi)
            # Close the figure to free up resources
            plt.close(hist_fig)

            # Define the file path for saving the estimated taus as a CSV file
            est_taus_filepath = os.path.join(dest, f'{fname}_estimated_tau.csv')
            self.logger.info("Saving estimated taus to %s", est_taus_filepath)
            # Write the estimated taus to CSV using the write_results utility function
            write_results(est_taus_filepath, dict(zip(tes_with_taus, average_taus_per_tes)), fmt="csv",
                          header=["ASIC", "TES", "average tau", "sigma"])

    @staticmethod
    def _init_worker():
        """
        # Sets a separate matplotlib cache for each process to avoid conflicts when using multiprocessing
        """

        # creates a temporary cache directory
        tmpdir = tempfile.mkdtemp(prefix="mplcache_")
        # Sets the temporary directory as Matplotlib’s cache directory
        os.environ["MPLCONFIGDIR"] = tmpdir
        # Removes the temporary directory as the final step after saving all plots
        atexit.register(shutil.rmtree, tmpdir, ignore_errors=True)

    def plot_results(self):
        """
        Generates and saves all analysis plots.

        This method performs two main tasks:
          1. It creates individual plots for each TES by splitting the available tau data into groups
             of three, then parallelizes the plotting process.
          2. It produces a summary plot for the entire dataset.
        """

        # List to hold the global TES indices, repeated according to the number of plots needed
        # (each plot shows up to 3 taus)
        tes_global_idx = []

        # List to hold the tau information dictionary for each TES, repeated as needed for plotting
        taus_per_tes_list = []

        # List to hold the starting index for plotting taus (each plot displays 3 taus, so indices are 0, 3, 6, etc.)
        start_tau_idx = []

        # List to hold copies of the paths list for each plotting subprocess
        paths_list = []

        # Create a multiprocessing pool to parallelize the creation of individual TES plots
        with mp.Pool(initializer=self._init_worker) as pool:
            # Define a list of paths used in plotting:
            # - Observation directory, plots directory, cleaned signals, interpolated elevation, and time file
            paths = [
                self.__observation_dir,
                self.__plots_dir,
                self.__signals_clean_fname,
                self.__interp_elev_fname,
                self.__time_fname
            ]

            # Loop over each TES for which tau data is available
            for tes in self.__taus:
                tes_taus = len(self.__taus[tes]['taus'])  # Number of tau entries for the current TES
                n_plots = round(tes_taus / 3)  # Estimate the number of plots needed (each plot shows 3 taus)
                # Calculate the total iterations: n_plots plus an extra plot if there are remaining taus
                n_iter = n_plots + (tes_taus - 3 * n_plots) * (3 * n_plots < tes_taus)
                # Extend the list of global TES indices with the current TES, repeated n_iter times
                tes_global_idx.extend([tes] * n_iter)
                # Extend the list with the tau data for the current TES, repeated n_iter times
                taus_per_tes_list.extend([self.__taus[tes]] * n_iter)
                # Extend the list with copies of the paths list for each plot to be generated
                paths_list.extend([paths] * n_iter)
                # Extend the list with starting indices for each plot (0, 3, 6, etc.)
                start_tau_idx.extend([3 * i for i in range(n_iter)])

            # Parallelize the creation of individual TES plots using the starmap function
            pool.starmap(my_plt.plot_taus,
                         zip(tes_global_idx,
                             taus_per_tes_list,
                             paths_list,
                             start_tau_idx),
                         chunksize=max(1, len(tes_global_idx) // mp.cpu_count()))

        # After parallel plotting, generate a summary plot for the entire dataset
        my_plt.plot_taus(0, self.__taus, paths, 0, self.__thermometers, summary_plot=True)

    def _find_cosmic_rays(self) -> (dict, int):
        """
        Searches for cosmic ray events based on computed time constants from TES signals.

        This method configures the necessary data directories, exports cleaned signals,
        computes time constants for each TES, saves the residuals, writes the results to a file,
        and generates the required plots for the analysis.

        Returns
        -------
        (dict, int)
            A tuple where the first element is a dictionary with cosmic ray information
            for each TES and the second element is the total number of cosmic ray events found.
        """

        # Log initial configuration info
        self.logger.info("configuration of all the necessary variables")

        # Configure directories and paths for exported files (.npy and .npz formats)
        self.configure_data()

        # Log the start of signal export
        if self.remove_drift:
            self.logger.info("export clean signals from atmospheric drift")

        # If remove_drift is False, the clean signal path equals the raw signal path.
        # Otherwise, save_interp_signals corrects for atmospheric drift and returns the TES available for analysis.
        tes_keys = self.remove_atmospheric_drift()
        # Log the start of time constant extraction
        self.logger.info("search for the time constants of all TESs")

        # Start a timer to measure processing time
        start = tm.perf_counter()
        # Compute time constants for the available TES
        taus_per_tes = self.get_taus(tes_keys)

        # Initialize a list to gather residuals from candidate fits
        residuals = []
        res: list[float]
        for tes in taus_per_tes:
            # Flatten residuals from each TES into a single list
            residuals.extend([value for res in taus_per_tes[tes]['residuals'] for value in res])

        if residuals:
            # Save residuals to a file if any residuals were collected
            np.save(os.path.join(self.__output_dir, "residuals"), residuals)
            self.logger.info(f"Saving single dataset residuals: # samples {len(residuals)}")

        end = tm.perf_counter()
        execution_time = f"Time to process signals: {end - start:.2f}[s]"
        # Log the execution time
        self.logger.info(execution_time)

        print("\033[96m" + execution_time + "...\033[00m", end="")

        self.logger.info("write time constants to the file '%s'", self.__taus_fname)

        # Write the computed time constants to file
        write_results(self.__taus_fname, taus_per_tes)

        self.logger.info("save the time constant plots for single TES and the time constant plot of all TES")
        print("\033[0;33m" + "Saving plots..." + "\033[00m", end="")

        # Generate and save the individual and summary plots for the time constants
        self.plot_results()

        end = "Analysis completed successfully"
        print("\033[32m" + end + "\033[00m", end='\n\n')
        self.logger.info(end)

        # Return the dictionary of cosmic ray events per TES and the total number of cosmic ray events found
        return taus_per_tes, len([tau for tes in taus_per_tes for tau in taus_per_tes[tes]['taus']])

    def find_cosmic_rays(self):
        """
        Manages the process of cosmic ray detection by coordinating various methods.
        It verifies the analysis mode and available datasets, processes each dataset for cosmic ray detection,
        generates multi-dataset reports if applicable, and optionally removes temporary files.
        """

        # Check if the analysis type is valid according to the allowed modes in Crd.MODE
        if self.analysis_type not in Crd.MODE:
            raise ValueError(f"mode must be one of {Crd.MODE}")

        # Retrieve the datasets to be analyzed; if none are found, raise an error
        if not (datasets := self.get_datasets()):
            raise ValueError("no datasets found")

        # If there are at least two datasets and file removal is not enabled,
        # warn the user about large disk space usage and prompt whether to remove files
        if sum(map(len, datasets.values())) >= 2 and not self.remove_files:
            warning = """\033[91mWarning!
                    You have selected more than two datasets to analyze, and you haven't requested 
                    the removal of .npy and .npz files. Keep in mind that analyzing a single dataset generates .npy and .npz files 
                    that will take up approximately 9 GB of space in total! Do you want to MAINTAIN all the .npy and .npz files 
                    that will be created (NOT recommended action)? [y/n]\033[0m """

            self.remove_files = input(warning).lower() != 'y'

        # Initialize a list to store Crd objects for datasets with detected cosmic rays
        crds = list()
        # Initialize a dictionary to collect the results.
        results = dict()
        # Get the path of the last dataset (used later for global multi-dataset report plotting)
        last_path = list(datasets.values())[-1][-1]

        # Iterate over each scanning strategy in the datasets
        for strategy in datasets:

            # List to collect datasets for which no cosmic rays were found
            no_crd = list()
            # Create an entry in the results dictionary for this strategy
            results[os.path.basename(strategy)] = dict()
            # Identify the last dataset path for the current scanning strategy
            scan_strat_last_path = datasets[strategy][-1]

            # If more than one dataset exists for this strategy, initialize a Crd object for multi-dataset reporting
            if len(datasets[strategy]) > 1:
                scan_strat_crd = Crd(source=f'scan_strat_{os.path.basename(strategy)}',
                                     dest=strategy,
                                     points_vertical_trend=self.points_vertical_trend,
                                     points_exp_decrease=self.points_exp_decrease)

            # Process each dataset (source) within the current scanning strategy
            for src in datasets[strategy]:
                # Create a Crd object for the current dataset with the specified analysis parameters
                crd = Crd(source=src,
                          dest=strategy,
                          points_vertical_trend=self.points_vertical_trend,
                          points_exp_decrease=self.points_exp_decrease,
                          mask=self.mask,
                          analysis_type=self.analysis_type,
                          analysis_type_args=self.analysis_type_args,
                          deconvolution=self.deconvolution,
                          remove_files=self.remove_files,
                          remove_drift='moonscan' in src.lower(),
                          save_results=self.save_results)

                # Create all necessary directories to store analysis results
                crd.make_result_dir()

                crd.logger.info('observation dir: %s', crd.__observation_dir)
                print('\033[45m' + f'Dataset under analysis: {src}' + '\033[00m')

                try:
                    # Execute cosmic ray detection on the dataset and retrieve tau results and count
                    taus, n_taus = crd._find_cosmic_rays()

                    # If multiple datasets exist for the scanning strategy, generate a multi-dataset report for this
                    # strategy.
                    if len(datasets[strategy]) > 1:
                        crd.multidataset_report(fname=os.path.basename(strategy),
                                                mdt_report=scan_strat_crd.__taus,
                                                make_plot=src == scan_strat_last_path,
                                                title=f'Multi-Dataset Analysis - Scanning Strategy: '
                                                      f'{os.path.basename(strategy)}')

                    # If more than one scanning strategy is being analyzed, generate a global multi-dataset report
                    if len(datasets) > 1:
                        # the plot is created only when the provided dataset is the final one to be analyzed
                        crd.multidataset_report(fname='all_datasets',
                                                mdt_report=self.__taus,
                                                make_plot=src == last_path,
                                                global_dest=self.dest,
                                                title="Multi-Dataset Analysis - All Scanning Strategies")

                    # If file removal is enabled, delete temporary .npy and .npz files (except for the necessary
                    # cleaned signals file)
                    if self.remove_files:
                        for file in glob.glob(os.path.join(crd.__input_dir, '*')):
                            # Determine which file to keep based on whether drift removal is applied.
                            clean_signals = 'signals_raw.npz' if 'moonscan' not in src.lower() else 'signals_clean.npz'

                            if clean_signals not in file:
                                os.remove(file)

                        # If the plots directory is empty after removals, remove it along with the tau file and
                        # output directory.
                        if not os.listdir(crd.__plots_dir):
                            os.rmdir(crd.__plots_dir)
                            os.remove(crd.__taus_fname)
                            os.rmdir(crd.__output_dir)
                            crd.logger.info('No time constants found. Plots dir removed')

                    # If tau candidates are found, store the count in the results and add the Crd object to the list;
                    # otherwise, mark the dataset as free of cosmic rays
                    if n_taus:
                        results[os.path.basename(strategy)][os.path.basename(src)] = n_taus
                        crds.append(crd)
                    else:
                        no_crd.append([crd.logfname, crd.__observation_dir])

                except Exception as e:
                    print(f"\033[31mERROR: {e} \033[0m")
                    crd.logger.exception("Exception while executing find_cosmic_rays")

                # If deconvolution is enabled, remove cosmic ray artifacts from the signal
                if self.deconvolution:
                    crd.remove_cr()

            # Shutdown logging for the current strategy to close all log files
            logging.shutdown()

            # For datasets with no detected cosmic rays, move log files and remove associated directories if file removal is enabled
            for dataset in no_crd:

                if self.remove_files:
                    try:
                        shutil.move(dataset[0], strategy)
                        shutil.rmtree(dataset[1])
                    except Exception as e:
                        print(f"\033[31mERROR: {e} \033[0m")

        # If saving results is enabled, write the cosmic ray detection results to a JSON file
        if self.save_results:
            write_results(os.path.join(self.dest, 'cosmic_rays_detected.json'), results, indent=2)

        print('\033[42m' + 'ALL DATASET ANALYZED' + '\033[00m')

        return crds

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute the cosine similarity between two vectors.

        Parameters
        ----------
        a : np.ndarray
            First vector.
        b : np.ndarray
            Second vector.

        Returns
        -------
        float
            The cosine similarity value between vectors a and b
        """

        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    @staticmethod
    def check_similarity(tes: int,
                         taus: list[float],
                         signal: str | np.ndarray,
                         fs: float) -> tuple[int, float, np.ndarray] | tuple[int, None, np.ndarray]:

        """
        Evaluate the similarity between the original signal and its deconvolved version using cosine similarity.
        If the similarity is below a defined threshold, return the TES identifier, the similarity value,
        and the deconvolved signal; otherwise, return a tuple with None values.

        Parameters
        ----------
        tes : int
            TES identifier.
        taus : list[float]
            List of tau values for the TES.
        signal : str or np.ndarray
            Either the file path to the signal data (.npz) or the signal array.
        fs : float
            Sampling frequency of the signal.

        Returns
        -------
        tuple[int, float, np.ndarray] or None
            A tuple containing the TES identifier, the cosine similarity (if below threshold),
            and the deconvolved signal; otherwise, (tes, None, None).
        """

        # If 'signal' is provided as a file path, load the signal data for the given TES.
        if isinstance(signal, str):
            signal = np.load(signal)[str(tes)]

        elif not isinstance(signal, np.ndarray):
            raise ValueError('Signal must be either .npz file path or np.ndarray')

        # Define the similarity; only signals with similarity below this are processed
        threshold = 0.8

        # Estimate the average tau from the provided list
        tau = np.mean(taus)
        deconvolved_signal = md.deconvolve(conv_signal=signal, tau=tau, M=1 * round(fs), dt=1 / fs)

        # Calculate the cosine similarity between the original and deconvolved signals
        if (similarity := Crd.cosine_similarity(signal, deconvolved_signal)) < threshold:
            # If similarity is below the threshold, return the TES identifier, similarity value, and deconvolved signal
            return tes, similarity, deconvolved_signal
        else:
            # Convergence reached. No more deconvolution needed for specific tes.
            # return the TES identifier with None values for similarity and signal
            return tes, None, deconvolved_signal

    def remove_cr(self):
        """
        Removes cosmic rays from the TES signals via deconvolution.

        This method iteratively applies the check_similarity function in parallel
        across all available TES signals (using multiprocessing) to deconvolve the
        cosmic ray artifacts. It updates the signal for each TES until no further
        deconvolution is required, then saves the final deconvolved signals.
        """

        # Create a multiprocessing pool to parallelize the check_similarity function
        with mp.Pool() as pool:

            # Log the start of deconvolution for the current observation directory
            self.logger.info("Deconvolution of %s", os.path.basename(self.__observation_dir))

            # Create a list of TES keys (as strings) from the keys of the tau dictionary
            tes_keys = list(map(str, self.__taus.keys()))
            # Initialize a list to store similarity metrics for each TES
            dataset_similarity = list()

            # For each available TES, initialize a dictionary mapping the TES key to its signal path
            similarity = dict(zip(tes_keys, [self.__signals_clean_fname] * len(tes_keys)))

            # Continue processing until all TES have been deconvolved
            while tes_keys:

                # Apply check_similarity in parallel to each TES with their corresponding
                # tau data, signal, and sampling frequency
                result = pool.starmap(self.check_similarity,
                                      zip(tes_keys,
                                          [self.__taus[int(tes)]['taus'] for tes in tes_keys],
                                          [similarity[tes] for tes in tes_keys],
                                          [self.fs] * len(tes_keys))
                                      )

                # Process the results from check_similarity for each TES
                for tes, sim, signal in result:
                    # Update the signal for the current TES with the deconvolved signal
                    similarity[tes] = signal

                    if sim is not None:
                        # Append similarity metric if returned
                        dataset_similarity.append(sim)
                    else:
                        # Remove the TES key if no further deconvolution is needed
                        tes_keys.remove(tes)

            # Save the updated deconvolved signals into an .npz file
            np.savez(self.__signals_clean_fname, **similarity)
            self.logger.info("Deconvolution completed for %s", os.path.basename(self.__observation_dir))


if __name__ == '__main__':

    mp.set_start_method("spawn")
    # Check if the configuration file (passed as the first command-line argument) exists
    if not os.path.isfile(conf := sys.argv[1]):
        # Raise an error if the file is not found
        raise ValueError(f'File `{conf}` not found')

    # Create a Crd object by reading the configuration from the specified file
    crd = Crd.read_config(conf)

    # Run the cosmic ray detection analysis
    crd.find_cosmic_rays()
