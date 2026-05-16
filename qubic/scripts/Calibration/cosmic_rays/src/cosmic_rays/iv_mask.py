import os
import sys
import glob
import atexit
import shutil
import logging
import tempfile
import time

import numpy as np
import toml
from tqdm import tqdm
import multiprocessing as mp

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from scipy.signal import savgol_filter

from qubicpack.qubicfp import qubicfp
from qubicpack.utilities import NPIXELS

mpl.use('Agg')


class IVMask:
    """
    Class to process IV (current-voltage) data for TES (Transition Edge Sensor) analysis.

    This class estimates the TES sign, generates mask files, and optionally produces IV plots.
    The mask files contain:
    - First line: global indices of positive TES.
    - Second line: global indices of negative TES.
    If there are no positive (negative) TESs (invalid IV curve), the row will contain only the value -1.

    The analysis outputs and mask files are saved in a specified directory, or a default "iv_analysis" directory
    is created by the class.

    Args
    ---------
    iv_dir : str
        path for input IV datasets

    output_dir: (str, optional)
        path where analysis outputs will be saved. Default is `iv_analysis`

    density: int, optional
        number of uniformly sampled points along the IV curve. Default is 30.

    make_iv_plots : bool, optional
        if True, generates IV plots for each dataset. Default is `False`

    Attributes
    ----------
    (All of the above)

    datasets : list
        List of paths to IV datasets

    masks_output : str
        Path where mask files will be stored

    logger : logging.Logger
        Logger instance for logging information

    logfname : str
        Name of the log file

    self.iv_npz_dir : str
         directory where cached IV curves (.npz) will live

    Class Attributes
    ----------------
    sign_matrix : np.ndarray
        A 2D array representing the sign matrix for TES determination
    """

    # class attribute to store the sign matrix as a NumPy array
    sign_matrix: np.ndarray = np.array([])

    def __init__(self,
                 iv_dir: str,
                 output_dir: str = "iv_analysis",
                 density: int = 30,
                 make_iv_plots: bool = False,
                 save_sign_matrix: bool = True, ):

        # attributes
        self.iv_dir = iv_dir
        self.output_dir = output_dir
        self.density = density
        self.make_iv_plots = make_iv_plots

        self.datasets: list = []
        self.masks_output: str = ''

        self.logger: logging.Logger = None
        self.logfname: str = ''
        self.save_sign_matrix = save_sign_matrix
        self.iv_npz_dir: str = ''

    @classmethod
    def load_conf(cls, path):
        """
        Loads configuration from a TOML file and initializes the class.

        The method reads a TOML configuration file from the specified path, extracts relevant information,
        and uses it to initialize a new instance of the class. The configuration file is expected to
        contain keys for source directory, output directory, IV plots flag, density, and save sign
        matrix flag.

        Args:
            path (str): Path to the TOML configuration file.

        Returns:
            cls: An instance of the class initialized with parameters from the configuration file.
        """

        toml_dict = toml.load(path)
        iv_dir = toml_dict['source']
        output_dir = toml_dict['output']
        make_iv_plots = toml_dict['iv_plots']
        density = toml_dict['density']
        save_sign_matrix = toml_dict['save_sign_matrix']

        return cls(iv_dir,
                   output_dir,
                   density,
                   make_iv_plots,
                   save_sign_matrix)

    def configure(self):
        """
        Configures the IVMask instance by setting up directories, logging, and dataset retrieval.

        This method creates the output directory and masks subdirectory, sets up the logging configuration,
        retrieves and sorts the dataset files from the input directory, logs the process, and initializes
        the sign_matrix whose number of rows is equal to the number of IV datasets and the number of columns is equal
        to 2 * NPIXELS, i.e., the total number of TES (256).
        """

        os.makedirs(self.output_dir, exist_ok=True)
        self.logfname = os.path.join(self.output_dir, "masks.log")

        self.masks_output = os.path.join(self.output_dir, "masks")
        os.makedirs(self.masks_output, exist_ok=True)

        # create directory for .npz cache
        self.iv_npz_dir = os.path.join(self.output_dir, "iv_npz")
        os.makedirs(self.iv_npz_dir, exist_ok=True)

        logging.basicConfig(filename=self.logfname,
                            filemode='w',
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            datefmt='%d/%b/%y - %H:%M:%S',
                            level=logging.INFO)

        self.logger = logging.getLogger(__name__)

        self.datasets = sorted(glob.glob(self.iv_dir))

        self.logger.info(f"Found {len(self.datasets)} datasets")
        self.logger.info(self.datasets)

        self.logger.info(f"Writing masks to {self.masks_output}")

        # initialize the sign matrix with zeros; dimensions: number of datasets x (2 * NPIXELS)
        self.sign_matrix = np.zeros((len(self.datasets), 2 * NPIXELS), dtype=int)

        self.logger.info("Sample density: %s", self.density)

    @staticmethod
    def get_tes_sign(v: np.ndarray,
                     i: np.ndarray,
                     density: int = 30,
                     get_points: bool = False,
                     ratio=0.2) -> int | tuple[int, np.ndarray]:

        """
        Determines the TES sign for a given IV curve.

        This static method analyzes the voltage (v) and current (i) arrays to determine if the TES
        is positive, negative, or not valid. Additionally, it can return specific data points that
        highlight the IV curve trend if requested.

        :param v: Array of voltage values
        :type: np.ndarray

        :param i: Array of current values
        :type: np.ndarray

        :param density: Number of uniformly sampled points along the IV curve. Default is 30. A Savitzky–Golay
            filter is applied with a window proportional to `density` before sampling
        :type: int, optional

        :param get_points: Flag to indicate whether to return the points used to highlight the IV curve trend.
            Default is False.
        :type: bool, optional

       :param ratio: Parameter that defines the start and end intervals of the I-V curve within which the
            minimum should not be located. Default is 0.2.
       :type: float, optional

        :return: 1 if TES is positive, -1 if negative, or 0 if not valid. If `get_points` is True,
            a tuple is returned containing the TES sign and the points array.
        :rtype: int or tuple[int, np.ndarray]
        """

        # check if the voltage values are strictly increasing.
        # If not, invert the voltage and current arrays.
        if v[-1] < v[0]:
            v = v[::-1]
            i = i[::-1]

        # Apply Savitzky–Golay smoothing to the input array
        # The window length is chosen to be odd and roughly proportional to the local data density
        # Compute an odd window length: take half of 'density', force it to be odd, and require a minimum size of 5
        win = max(5, (density // 2) * 2 + 1)

        # If the window length is too large compared to the data size,
        # reduce it to the largest valid odd value smaller than the array length
        if win >= i.shape[0]:
            win = i.shape[0] - 1 if (i.shape[0] % 2 == 0) else i.shape[0] - 2

        # Choose the polynomial order for the Savitzky–Golay filter:
        # use order 3 when possible, otherwise fall back to order 2,
        # ensuring that polyorder < window_length
        poly = 3 if win > 3 else 2
        # Smooth the data using the Savitzky–Golay filter with the selected window length and polynomial order
        i_smooth = savgol_filter(i, window_length=win, polyorder=poly)

        # assume the TES sign is positive initially
        tes_sign = 'positive'
        not_valid = 0

        # determine the TES sign based on the change in smoothed current values; flip if necessary
        if i_smooth[i_smooth.shape[0] // 2] - i_smooth[0] > 0 > i_smooth[-1] - i_smooth[i_smooth.shape[0] // 2]:
            tes_sign = 'negative'
            i_smooth = np.fabs(i_smooth - i_smooth.max())

        # find the index of the minimum current value
        min_idx = np.argmin(i_smooth)

        # if the minimum is too close to the edges of the data, consider the TES not valid
        # and return zero (or tuple if get_points is True)
        if 0 <= min_idx <= int(len(i_smooth) * ratio) or min_idx >= int(len(i_smooth) * (1 - ratio)):
            # TES not valid, return 0 or tuple with an empty array
            return (not_valid if not get_points
                    else (not_valid, np.array([])))

        # uniformly sample the whole IV curve with `density` points,
        # then split the sample in two subsets: before the minimum
        # and sufficiently after the minimum (skip the immediate region
        # around the minimum defined by `ratio`)
        points_all = np.linspace(0, i_smooth.shape[0] - 1, density, dtype=int)

        points_before = points_all[points_all < min_idx]
        points_after = points_all[points_all >= int(min_idx * (1 + ratio))]

        points = np.hstack((points_before, points_after))

        # count the number of points that are before the minimum index
        points_sx_min = np.sum(points < min_idx)

        # check if the segments before and after the minimum have the expected monotonic behavior
        if np.any(np.diff(i_smooth[points[:points_sx_min]]) >= 0) or np.any(np.diff(i_smooth[points[points_sx_min:]]) <= 0):
            # TES not valid, return 0 or tuple with an empty array
            return (not_valid if not get_points
                    else (not_valid, np.array([])))

        # Return the sign of the TES: 1 for positive, -1 for negative; include points if requested
        if not get_points:
            return 1 if tes_sign == 'positive' else -1
        else:
            return (1 if tes_sign == 'positive' else -1), points


    def plot_iv_curve(self, v_tes, i_tes, points, tes, asic, dt_path, tes_sign, iv_plots_dir):

        """
        Generates a plot for the I-V (current-voltage) curve of a specific TES (Transition Edge Sensor)
        and saves it to the specified directory. The plot includes the data points and highlights
        specific points if provided.

        :param v_tes:  Array of voltage values for the TES
        :type: numpy.ndarray

        :param i_tes:  Array of current values corresponding to the voltage array
        :type: numpy.ndarray

        :param points: Indices of specific points to highlight on the plot, if any
        :type: numpy.ndarray

        :param tes: The index of the TES being analyzed
        :type: int

        :param asic: The index of the ASIC associated with the TES
        :type: int

        :param dt_path: The path to acquire plot-related metadata, such as title information
        :type: str

        :param tes_sign: Indicates the sign of the TES data being plotted (e.g., 1 for "Positive", -1
            for "Negative", 0 as "Invalid")
        :type: int

        :param iv_plots_dir: Directory path where the I-V plot will be saved
        :type: str

        :return: None

        Raises:
        -------
        Exception
            Raised if the plot cannot be successfully saved within the prescribed number of
            retries. The error is logged, and retries are attempted with exponential backoff.
        """

        max_retries = 3
        sign_dict = {1: "Positive", -1: "Negative", 0: "Invalid"}

        tes_disp = tes + 1
        asic_disp = asic + 1
        title = os.path.basename(dt_path).replace("__", "~").replace("_", "~")
        plot_path = os.path.join(iv_plots_dir, f"iv_curve_{asic_disp}_{tes_disp}_{sign_dict[tes_sign]}.png")

        for attempt in range(max_retries):
            fig = None
            try:
                fig, ax = plt.subplots(figsize=(15, 5), tight_layout=True)
                ax.set(
                    xlabel="V [Volt]",
                    ylabel="I [Ampere]",
                    facecolor='whitesmoke',
                    title=rf"$ASIC ~ {asic_disp} ~ - ~ TES ~ {tes_disp}$"
                )
                ax.plot(v_tes, i_tes, label=rf"${sign_dict[tes_sign]}$")

                if points.size:
                    ax.scatter(v_tes[points], i_tes[points], color='red')

                fig.suptitle(rf"$I-V ~ curve ~ - ~ {title}$")
                ax.legend(title=rf"$Sign: ~$")

                fig.savefig(plot_path)
                plt.close(fig)
                return

            except Exception as e:
                if attempt < max_retries - 1:
                    self.clean_temp_caches()
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                else:
                    self.logger.error(
                        f"Failed to save plot for ASIC {asic_disp} TES {tes_disp} "
                        f"after {max_retries} attempts: {str(e)}"
                    )
                    if fig:
                        plt.close(fig)
                    return

    def _write_sign_matrix(self, dataset: str, sign_row: np.ndarray):
        """
        Writes the sign matrix for a dataset to a mask file.

        This private method creates a mask file for the given dataset that contains the indices
        of positive and negative TES values. It logs the process and writes the indices to a text file, whose rows
        are divided as follows:
        - First row: global indices of POSITIVE TESs;
        - Second row: global indices of NEGATIVE TESs;
        If there are no positive (negative) TESs, the row will contain only the value -1.

        :param dataset: the dataset name
        :type: str

        :param sign_row: A row from the sign matrix corresponding to the dataset, containing TES sign values
        :type: np.ndarray
        """

        # construct the mask filename based on the dataset name (e.g., mask_<dataset>.txt)
        mask_fname = os.path.join(self.masks_output, 'mask_' + dataset + '.txt')

        self.logger.info(f"Writing {mask_fname}")

        # retrieve indices for positive and negative TES based on the sign values in the sign_row
        positives = np.where(sign_row == 1)[0]
        negatives = np.where(sign_row == -1)[0]

        # create strings for positive and negative indices; if no indices are found, default to '-1'
        positives = ' '.join(map(str, positives)) + '\n' if positives.size else '-1\n'
        negatives = ' '.join(map(str, negatives)) + '\n' if negatives.size else '-1\n'

        # write the positive and negative indices to the mask file
        with open(mask_fname, 'w', newline='', encoding='utf8') as fout:
            fout.writelines([positives, negatives])

    @staticmethod
    def _init_worker():
        """
        Initializes the worker process by creating and configuring unique temporary directories
        required for its operation. Sets up the environment variables and Matplotlib configuration
        to support the use of TeX and other LaTeX packages. Ensures proper cleanup of the temporary
        resources upon process termination.

        Raises
        ------
        No direct exceptions are raised by this method, but issues may arise if temporary directories
        cannot be created or if dependencies like Matplotlib or TeX are misconfigured.
        """


        # Create a unique temporary folder for each worker
        base_dir = tempfile.mkdtemp(prefix=f"worker_{os.getpid()}_")
        mpl_dir = os.path.join(base_dir, "mpl_cache")
        os.makedirs(mpl_dir, exist_ok=True)

        # set the environment variables required for Matplotlib and TeX
        os.environ['MPLCONFIGDIR'] = mpl_dir
        os.environ['TEXMFVAR'] = base_dir
        atexit.register(shutil.rmtree, base_dir, ignore_errors=True)

        # Configure Matplotlib
        mpl.use('Agg')
        plt.rcParams.update({
            'text.usetex': True,
            'font.family': 'serif',
            # 'text.latex.preamble': r'''
            #     \usepackage{amsmath}
            #     \usepackage{amssymb}
            # '''
        })

    def _load_iv_curves(self, dt_path: str):
        """
        Load or create the IV‑curve cache for a dataset.

        This method loads IV‑curve data from a cache if it exists, or creates the cache
        by reading from a FITS file and processing the data. The cache, if created, is
        saved as a compressed NumPy `.npz` file. The method ensures that the IV‑curve
        data is efficiently stored and avoids unnecessary recalculation.

        :param dt_path:  Path to the dataset FITS file from which to derive or load the IV‑curve data.
        :type: str


        :return: v_all : ndarray[object]
                A 1‑D array storing the voltage data per ASIC. Each element is a
                ndarray corresponding to pixels within an ASIC.

            i_all : ndarray[object]
                A 1‑D array storing the current data per ASIC. Each element is a
                ndarray corresponding to pixels within an ASIC.

            temp : float | None
                The sample temperature in Kelvin, if available. Returns None if the
                temperature is not available.
        :rtype: tuple[ndarray, ndarray, float | None]
        """

        cache_fname = os.path.join(self.iv_npz_dir,
                                   os.path.basename(dt_path) + ".npz")

        if os.path.exists(cache_fname):
            self.logger.info(f"Loading IV curves from cache: {cache_fname}")
            data = np.load(cache_fname, allow_pickle=True)
            v_all = data["vtes"]
            i_all = data["ites"]
            temp = data["temperature"].item()

            if isinstance(temp, float) and np.isnan(temp):
                temp = None
            return v_all, i_all, temp

        # Cache miss: need to read the FITS file
        qubic = qubicfp()
        qubic.verbosity = 0
        qubic.read_qubicstudio_dataset(dt_path)

        v_list = []
        i_list = []
        for asic in filter(lambda x: x, qubic.asic_list):
            v_tes, i_tes = asic.best_iv_curve()    # v_tes shape (NPIXELS, n_points)
            v_list.append(v_tes)                   # keep as ndarray; row‑wise per TES
            i_list.append(i_tes)

        # Build 1‑D object arrays to avoid broadcasting issues
        v_all = np.empty(len(v_list), dtype=object)
        i_all = np.empty(len(i_list), dtype=object)
        for k in range(len(v_list)):
            v_all[k] = v_list[k]
            i_all[k] = i_list[k]

        temp = qubic.temperature if qubic.temperature else 0.32

        # Save to cache
        np.savez_compressed(
            cache_fname,
            vtes=v_all,
            ites=i_all,
            temperature=(temp if temp is not None else np.nan)
        )

        del qubic
        return v_all, i_all, temp

    def clean_temp_caches(self):
        """
        Removes temporary caches in directories specific to the current process.

        This method cleans up temporary files from directories specified by
        the current process's environment variables, ensuring that it does
        not delete the directories themselves, only their contents. Errors
        related to permissions or other issues are logged appropriately.

        Raises:
            This method directly raises no specific exceptions. Errors
            encountered during file or directory removal are caught and logged
            without interrupting the execution flow.
        """

        # Directory of our process
        our_dirs = [
            os.environ.get('MPLCONFIGDIR', ''),
            os.environ.get('TEXMFVAR', '')
        ]

        for dir_path in our_dirs:
            if os.path.isdir(dir_path):
                try:
                    # not delete the directories themselves, only their contents
                    for filename in os.listdir(dir_path):
                        file_path = os.path.join(dir_path, filename)
                        try:
                            if os.path.isfile(file_path) or os.path.islink(file_path):
                                os.unlink(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                        except Exception as e:
                            # Ignore errors on system files/directories
                            if "Permission denied" not in str(e):
                                self.logger.warning(f"Could not delete {file_path}: {str(e)}")
                except Exception as e:
                    self.logger.error(f"Error cleaning {dir_path}: {str(e)}")

    def sign_estimate(self):
        """
        Processes and analyzes datasets to estimate signs, generate IV curves, and optionally
        create plots for temperature-sensitive datasets. This method handles multiprocessing
        to improve performance while processing large numbers of pixels per dataset.

        Raises:
            FileNotFoundError: Raised if any dependent data file is missing.
            ValueError: Raised if input data is inconsistent or invalid.
            RuntimeError: Raised in case of any failure during multiprocessing or file
                          operations.
        """

        ctx = mp.get_context('spawn')
        with ctx.Pool(initializer=self._init_worker) as pool:

            for dt_row, dt_path in tqdm(enumerate(self.datasets),
                                        ncols=100,
                                        total=len(self.datasets),
                                        file=sys.stdout,
                                        desc="progress",
                                        unit="dataset"):

                dt = os.path.basename(dt_path)

                # Load (or build) IV curves and temperature
                v_all, i_all, temp = self._load_iv_curves(dt_path)

                # Append temperature tag if available and not already present
                if temp is not None and "mk" not in dt.lower():
                    temp_rounded = round(temp * 1000)
                    self.datasets[dt_row] += f"_{temp_rounded}mk"
                    dt += f"_{temp_rounded}mk"

                # Plot folders
                if self.make_iv_plots:
                    output_iv = os.path.join(self.output_dir, f"iv_{dt}")
                    os.makedirs(output_iv, exist_ok=True)
                    iv_plots_dir = os.path.join(output_iv, "iv_plots")
                    os.makedirs(iv_plots_dir, exist_ok=True)

                n_asics = v_all.shape[0]

                for idx_asic in range(n_asics):
                    v_tes = v_all[idx_asic]
                    i_tes = i_all[idx_asic]

                    chunksize = max(1, NPIXELS // mp.cpu_count())
                    start, stop = idx_asic * NPIXELS, (idx_asic + 1) * NPIXELS

                    results = pool.starmap(
                        self.get_tes_sign,
                        zip(v_tes,
                            i_tes,
                            [self.density] * NPIXELS,
                            [self.make_iv_plots] * NPIXELS),
                        chunksize=chunksize
                    )

                    if not self.make_iv_plots:
                        self.sign_matrix[dt_row][start:stop] = results
                    else:
                        self.sign_matrix[dt_row][start:stop] = [row[0] for row in results]
                        points = [row[1] for row in results]

                        pool.starmap(
                            self.plot_iv_curve,
                            zip(v_tes,
                                i_tes,
                                points,
                                range(NPIXELS),
                                [idx_asic] * NPIXELS,
                                [dt_path] * NPIXELS,
                                self.sign_matrix[dt_row][start:stop],
                                [iv_plots_dir] * NPIXELS),
                            chunksize=chunksize
                        )

                # Write the mask file for the dataset
                self._write_sign_matrix(dt, self.sign_matrix[dt_row])

        if self.save_sign_matrix:
            path = os.path.join(self.output_dir, "sign_matrix.npz")
            np.savez_compressed(path,
                                sign_matrix=self.sign_matrix,
                                datasets=np.array(self.datasets, dtype="U"))

        self.logger.info("Analysis completed successfully")
        print("\033[32mAnalysis completed successfully\033[00m\n")

    def plot_sign(self):
        """
        Generates a summary plot showing the validity and sign of TES across all datasets.

        This method creates an image plot of the sign matrix where each row corresponds to a dataset and
        each column corresponds to a TES. The plot includes a colorbar to indicate positive, not valid, and negative
        TES values, and saves the figure to the output directory.
        """

        # create a list of dataset names from the dataset file paths
        datasets = list(map(os.path.basename, self.datasets))

        fig, ax = plt.subplots(figsize=(15, 6), tight_layout=True)
        fig.subplots_adjust(left=0.35)

        ax.tick_params(axis='y', labelsize=6)
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets)

        # number of tes
        total = self.sign_matrix.shape[1]
        always_pos = np.sum(np.all(self.sign_matrix == 1, axis=0))
        always_neg = np.sum(np.all(self.sign_matrix == -1, axis=0))
        always_invalid = np.sum(np.all(self.sign_matrix == 0, axis=0))

        always_pos_pct = always_pos / total * 100
        always_neg_pct = always_neg / total * 100
        always_invalid_pct = always_invalid / total * 100

        ax.tick_params(axis='y', labelsize=7)
        ax.set_title(r'Validity and  Sign of a tes  vs. IV datasets', fontsize=16)
        ax.set_ylabel(r'$ IV \, datasets$')
        ax.set_xlabel(r'$ \# TES $')
        ax.grid(which='minor', color='#333333', linewidth=1)

        cmap = ListedColormap(['#E15759', '#F28E2B', '#4E79A7'])
        im = ax.imshow(self.sign_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

        cbar = fig.colorbar(im)
        cbar.set_ticks([1, 0, -1])
        cbar.set_ticklabels([
            rf'Positive (always +: {always_pos_pct:.1f} $\%$)',
            rf'Not Valid (always 0: {always_invalid_pct:.1f} $\%$)',
            rf'Negative (always -: {always_neg_pct:.1f}$\%$)'
        ])

        cbar.ax.tick_params(labelsize=8)

        fig.savefig(os.path.join(self.output_dir, "validity_sign_vs_dataset"), bbox_inches="tight")


if __name__ == '__main__':

    # check if the number of command-line arguments is at least 5
    if len(sys.argv) != 2:
        raise Exception('Usage: python toml_conf_file')

    # create an instance of IVMask with the provided arguments
    iv_mask = IVMask.load_conf(sys.argv[1])

    iv_mask.configure()
    iv_mask.sign_estimate()
    iv_mask.plot_sign()
