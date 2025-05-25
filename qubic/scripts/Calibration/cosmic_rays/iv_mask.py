import os
import sys
import glob
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt

from qubicpack.qubicfp import qubicfp
from qubicpack.utilities import NPIXELS

class IVMask:
    """
    A class to process IV (current-voltage) data for TES (Transition Edge Sensor) analysis,
    estimate the TES sign, and generate corresponding mask files and plots.

    It is possible to perform IV curves analysis and create a mask (.txt file) containing two lines:
    - First line: global indices of POSITIVE TESs;
    - Second line: global indices of NEGATIVE TESs;

    If there are no positive (negative) TESs, the row will contain only the value -1.
    The masks are saved in a folder specified by the user or created by the class by default.

    Args
    ---------
    iv_dir             (str): path for input IV datasets
    output_dir         (str, optional): path where analysis outputs will be saved. Default is `iv_analysis`
    make_iv_plots      (bool, optional): if True, generates IV plots for each dataset. Default is `False`

    Attributes
    -----------
    (All of the above)

    datasets          (list): list of the paths to IV datasets
    masks_output      (str): path where mask files will be saved
    logger            (logging.Logger): Logger instance for logging information
    logfname          (str): log file name
    """


    # class attribute to store the sign matrix as a NumPy array
    sign_matrix: np.ndarray = np.array([])

    def __init__(self,
                 iv_dir: str,
                 output_dir: str = "iv_analysis",
                 make_iv_plots: bool = False):

        # attributes
        self.iv_dir = iv_dir
        self.output_dir = output_dir
        self.make_iv_plots = make_iv_plots

        self.datasets: list = []
        self.masks_output: str = ''

        self.logger: logging.Logger = None
        self.logfname: str = ''


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
        self.sign_matrix = np.zeros((len(self.datasets), 2*NPIXELS), dtype=int)


    @staticmethod
    def get_tes_sign(v: np.ndarray, i: np.ndarray, get_points: bool = False, ratio=0.2) -> int | tuple[int, np.ndarray]:
        """
        Determines the TES sign for a given IV curve.

        This static method analyzes the voltage (v) and current (i) arrays to determine if the TES is positive,
        negative, or not valid. It may also return specific data points that highlights the IV curve trend.

        Args
        ---------
        v : np.ndarray
            Array of voltage values
        i : np.ndarray
            Array of current values
        get_points : bool, optional
            Flag to indicate whether to return the points used fto highlight the IV curve trend. Default is False.
        ratio : float, optional
            parameter that defines the start and end intervals of the I-V curve within which the minimum should not be located

        Returns
        -------
        int or tuple[int, np.ndarray]
            Returns 1 if TES is positive, -1 if negative, or 0 if not valid.
            If get_points is True, returns a tuple (TES sign, points array).
        """


        # check if the voltage values are strictly increasing.
        # If not, invert the voltage and current arrays.
        if v[-1] < v[0]:
            v = v[::-1]
            i = i[::-1]

        # assume the TES sign is positive initially
        tes_sign = 'positive'
        not_valid = 0

        # determine the TES sign based on the change in current values; flip if necessary
        if i[i.shape[0] // 2] - i[0] > 0 > i[-1] - i[i.shape[0] // 2]:
            tes_sign = 'negative'
            i = np.fabs(i - i.max())

        # find the index of the minimum current value
        min_idx = np.argmin(i)

        # if the minimum is too close to the edges of the data, consider the TES not valid
        # and return zero (or tuple if get_points is True)
        if 0 <= min_idx <= int(len(i) * ratio) or min_idx >= int(len(i) * (1 - ratio)):
            # TES not valid, return 0 or tuple with an empty array
            return not_valid if not get_points else not_valid, np.array([])

        # define a set of points: 7 equally spaced points before the minimum and 7 equally spaced points after the minimum
        points = np.hstack((np.linspace(0, min_idx, 7, dtype=int),
                            np.linspace(min_idx * (1 + ratio), i.shape[0] - 1, 7, dtype=int)))

        # count the number of points that are before the minimum index
        points_sx_min = np.sum(points < min_idx)

        # check if the segments before and after the minimum have the expected monotonic behavior
        if np.any(np.diff(i[points[:points_sx_min]]) >= 0) or np.any(np.diff(i[points[points_sx_min:]]) <= 0):
            # TES not valid, return 0 or tuple with an empty array
            return not_valid, np.array([])

        # Return the sign of the TES: 1 for positive, -1 for negative; include points if requested
        if not get_points:
            return 1 if tes_sign == 'positive' else -1
        else:
            return (1 if tes_sign == 'positive' else -1), points


    @staticmethod
    def plot_iv_curve(v_tes: np.ndarray,
                      i_tes: np.ndarray,
                      points: np.ndarray,
                      tes: int,
                      asic: int,
                      dt_path: str,
                      tes_sign: int,
                      iv_plots_dir: str):

        """
        Plots an IV curve with marked points and saves the figure.

        This static method generates a plot of the IV curve using the provided voltage and current arrays,
        marks the specified points on the curve, and saves the resulting plot to the given directory.

        Args
        ---------
        v_tes : np.ndarray
            Array of voltage values for the TES
        i_tes : np.ndarray
            Array of current values for the TES
        points : np.ndarray
            Array of indices indicating the points to be highlighted on the curve
        tes : int
            TES index
        asic : int
            Identifier for the ASIC
        dt_path : str
            File path of the dataset being processed
        tes_sign : int
            Numerical sign of the TES (1 for positive, -1 for negative, 0 for invalid)
        iv_plots_dir : str
            Directory where the IV plot image will be saved
        """

        # increment TES and ASIC indices by 1 for display purposes
        tes += 1
        asic += 1

        sign_dict = {1: "Positive", -1: "Negative", 0: "Invalid"}
        title = os.path.basename(dt_path).replace("__", "~").replace("_", "~")

        fig, ax = plt.subplots(figsize=(15, 5), tight_layout=True)
        ax.set(xlabel="V [Volt]", ylabel="I [Ampere]", facecolor='whitesmoke',
               title=rf"$ASIC ~ {asic} ~ - ~ TES ~ {tes * asic}$")

        ax.plot(v_tes, i_tes, label=rf"${sign_dict[tes_sign]}$")
        if points.size:
            ax.scatter(v_tes[points], i_tes[points], color='red')
        fig.suptitle(rf"$I-V ~ curve ~ - ~ {title}$")
        ax.legend(title=rf"$Sign: ~$")

        fig.savefig(os.path.join(iv_plots_dir, f"iv_curve_{asic}_{tes}_{sign_dict[tes_sign]}.png"), dpi=300)
        plt.close(fig)


    def _write_sign_matrix(self, dataset: str, sign_row: np.ndarray):
        """
        Writes the sign matrix for a dataset to a mask file.

        This private method creates a mask file for the given dataset that contains the indices
        of positive and negative TES values. It logs the process and writes the indices to a text file, whose rows
        are divided as follows:
        - First row: global indices of POSITIVE TESs;
        - Second row: global indices of NEGATIVE TESs;
        If there are no positive (negative) TESs, the row will contain only the value -1.

        Args
        ---------
        dataset : str
            The name of the dataset
        sign_row : np.ndarray
            A row from the sign matrix corresponding to the dataset, containing TES sign values
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


    def sign_estimate(self):
        """
        Estimates the TES sign for each dataset in parallel and generates mask files and IV plots.

        This method uses multiprocessing to process each dataset concurrently. For each dataset, it reads the IV data,
        computes the TES sign for each TES, optionally generates IV plots, updates the sign matrix, and writes
        a corresponding mask file.
        """

        # create a multiprocessing pool for parallel processing
        with mp.Pool() as pool:

            # iterate over datasets with progress bar using tqdm for visual feedback
            for dt_row, dt_path in tqdm(enumerate(self.datasets),
                                        ncols=100,
                                        total=len(self.datasets),
                                        file=sys.stdout,
                                        desc='progress',
                                        unit='dataset'):

                # get the basename of the current dataset file
                dt = os.path.basename(dt_path)

                # create an output directory specific to the current dataset
                output_iv = os.path.join(self.output_dir, f"iv_{dt}")
                os.makedirs(output_iv, exist_ok=True)

                # create a directory for IV plots within the dataset-specific directory
                iv_plots_dir = os.path.join(output_iv, f"iv_plots")
                os.makedirs(iv_plots_dir, exist_ok=True)


                self.logger.info(f"Processing {dt}")

                # initialize a qubicfp instance to read and process the dataset
                qubic = qubicfp()
                qubic.verbosity = 0
                qubic.read_qubicstudio_dataset(dt_path)

                # check if the temperautre is in the dataset name
                if 'mk' not in dt.lower():
                    # transform temperature in mK
                    temperature = round(qubic.temperature * 1000)

                    dt += f'-{temperature}mK'

                # Loop through each ASIC in the dataset (filtering out any empty entries)
                for idx_asic, asic in enumerate(filter(lambda x: x, qubic.asic_list)):

                    # Retrieve the voltage and current arrays for the current ASIC
                    v_tes, i_tes = asic.best_iv_curve()

                    # define the workload chunk size based on NPIXELS and available CPU cores
                    chunksize = max(1, NPIXELS // mp.cpu_count())
                    # determine the slice indices for the current ASIC within the overall sign matrix row
                    start, stop = idx_asic * NPIXELS, (idx_asic + 1) * NPIXELS

                    # use pool.starmap to apply get_tes_sign to each TES point in parallel
                    results = pool.starmap(self.get_tes_sign,
                                           zip(v_tes, i_tes, [self.make_iv_plots] * NPIXELS),
                                           chunksize=chunksize)

                    # if IV plots are not requested, assign the results directly to the sign matrix
                    if not self.make_iv_plots:
                        self.sign_matrix[dt_row][start:stop] = results
                    else:
                        # if IV plots are requested, extract the sign and points from the results
                        self.sign_matrix[dt_row][start:stop] = [row[0] for row in results]
                        points = [row[1] for row in results]

                        # plot the IV curve for each TES in parallel using pool.starmap
                        pool.starmap(self.plot_iv_curve,
                                     zip(v_tes,
                                         i_tes,
                                         points,
                                         range(NPIXELS),
                                         [idx_asic] * NPIXELS,
                                         [dt_path] * NPIXELS,
                                         self.sign_matrix[dt_row][start:stop],
                                         [iv_plots_dir] * NPIXELS),
                                     chunksize=chunksize)


                # after processing all ASICs in a dataset, write the corresponding mask file
                self._write_sign_matrix(dt, self.sign_matrix[dt_row])

                # clean up the qubic instance to avoid conflicts with subsequent datasets
                del qubic

        end = "Analysis completed successfully"
        print("\033[32m" + end + "\033[00m", end='\n\n')
        self.logger.info(end)


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

        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets)

        ax.set_title(r'$Validity \ and \ Sign \ of \ a \ tes \ vs. \ IV \ datasets$')
        ax.set_ylabel(r'$IV \ datasets$', fontsize=16)
        ax.set_xlabel(r'$ \# \ TES$', fontsize=16)

        pos = ax.imshow(self.sign_matrix, cmap='copper', aspect='auto')
        cbar = fig.colorbar(pos)
        cbar.set_ticks(ticks=[1, 0, -1], labels=['Positive', 'Not Valid', 'Negative'])
        fig.savefig(os.path.join(self.output_dir, "validity_sign_vs_dataset"), dpi=1000)



if __name__ == '__main__':

    # check if the number of command-line arguments is at least 5; if so, raise an exception with usage instructions
    if len(sys.argv) >= 5:
        raise Exception('Usage: python iv_mask.py <iv_dir> <output_dir> <make_iv_plots: True | False>')

    # retrieve command-line arguments for input directory and output directory
    iv_dir = sys.argv[1]
    output_dir = sys.argv[2]
    # determine if IV plots should be made based on the number of arguments provided
    make_iv_plots = len(sys.argv) == 4

    # create an instance of IVMask with the provided arguments
    iv_mask = IVMask(iv_dir, output_dir, make_iv_plots)

    iv_mask.configure()
    iv_mask.sign_estimate()
    iv_mask.plot_sign()












