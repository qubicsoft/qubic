# coding: utf-8

import os
import sys
import glob
import json
import shutil
import pathlib
import logging
import argparse
import platform
import time as tm
import numpy as np
from tqdm import tqdm
from numba import njit
from typing import Type
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.interpolate import LSQUnivariateSpline

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from qubicpack.qubicfp import qubicfp
import qubicpack.pixel_translation as pt

# dictionary containing the information needed by all processes to read data into shared memory
# (parent and child process communicate only through shared memory (mediator))
SHARED_INFO = {"src_dir": '',
               "dst_dir": '',
               "logfname": '',
               "plots_dir": '',
               "taus_fname": '',
               "time_fname": '',
               "signals_fname": '',
               "signals_clean_fname": '',
               'thermometers': [4, 36, 68, 100],
               "time_shm": {},
               "n_knots": 7,
               "t_knots": Type[np.ndarray],
               "coeff": 5,
               "epsilon": 1e-2,
               "tau_coeff": 50e-3,
               "points_exp_decrease": 6,
               "points_vertical_trend": 3,
               "slope": 1100000
               }

#  the "SharedMemory" module was introduced starting from the 3.9 python version.
#  if the version of the python interpreter running the script is less than 3.9.x, the script will not be executed
version = platform.python_version()
major, minor, micro = tuple(map(int, version.split(".")))

if (major, minor) < (3, 9):
    red_color_code = "\033[91m"
    reset_color_code = "\033[0m"
    error = "ERROR: This script requires at least python 3.9 to be executed"

    print(red_color_code + error + reset_color_code)

    sys.exit(1)


def write_dict_to_json(fname: str, data: dict, indent: str | int | None = 2):
    """
    Writes a dictionary to a .json file

    Parameters
    ---------
    fname: str
        file name in which to save the dictionary

    data: dict
        dictionary containing data to be saved

    indent:
    """

    with open(fname, "w", encoding='utf8') as fout:
        json.dump(data, fout, indent=indent)


def read_json_file(fname: str) -> dict[str, list[float]]:
    """
    Reads time constants to a .json file

    Parameters
    ---------
    fname: str
        file name from which the time constants are read

    Returns
    -------
    dict:
        dictionary whose keys (str) are the number of TESs and whose values (list of float numbers) are
        a list containing the time constants of the related TES
    """

    with open(fname, "r", encoding='utf8') as fin:
        return json.load(fin)


def check_mask(mask_fname: str) -> list[list[int]] | None:
    """
    Verifies that the mask containing the valid TES sign is valid.
    To be valid, it must have 2 rows:\n

    - the first containing all positive TES
    - the second all negative TES
    and be composed of non-negative numbers (indices of the TES).

    If a row consists solely of the negative value -1, it means
    that there is no valid TES for that specific sign.

    Examples of calid mask files:

    Example 1: No positive TES, only negative TES
    -1
    2 4 5 7 13 14 15 18 19 25 26 28 29 30 31 32 33 50
    -------------------------------------------------

    Example 2: Only positive TES, no negative TES
    2 4 5 7 13 14 15 18 19 25 26 28 29 30 31 32 33 50
    -1
    -------------------------------------------------

    Example 3: Both positive and negative TES
    2 4 5 7 13 14 15 18 19 25 26 28 29 30 31 32 33 50
    52 60 71 65 74 79
    -------------------------------------------------

    Parameters
    ----------
    mask_fname : str
        Path of the file containing TES mask

    Returns
    -------
    list[list[int]] | None
        the mask with the indices of valid TES converted to integers;
        otherwise, it returns None.
    """

    # Check the existence of the file
    if not os.path.isfile(mask_fname):
        return None

    # read the file
    with open(mask_fname, 'r') as fin:
        mask = fin.readlines()

    # verify that the mask contains at most 2 rows and that they are composed of digits
    if not len(mask) or len(mask) > 2 or not all(i.isdigit() or int(i) < 0 for r in mask for i in r.strip().split()):
        return None

    mask = [list(map(int, row.strip().split())) for row in mask]

    # check that the digits in the file are non-negative only if the rows consist of more
    # than one element
    if any(i < 0 for r in mask for i in r if len(r) != 1):
        return None

    return mask


def check_mode(mode: str, to_analyze: list[str], tes_keys: list[str]) -> list[str]:
    """
    Verifies the TES analysis mode.
    The possible modes are:\n
    - all: all TES
    - range: TES interval (for example: [10,20])
    - sequence: TES sequence (for example: [1, 5, 80, 100])

    Parameters
    ----------
    mode: str
        TES analysis mode

    to_analyze: list[str]
        - [-1] for all TES
        - an interval for TES range
        - a sequence for TES sequence

    tes_keys: list[str]
        keys related to the signals of the TES present in the .npz files

    Returns
    -------
    list[str]
        list of TES to analyze
    """

    # ASCII code for printing colored strings on the terminal
    red_color_code = "\033[91m"
    reset_color_code = "\033[0m"

    # if the analysis mode is set to "ALL TES," it analyzes all available TES
    if len(to_analyze) == 1 and to_analyze[0] == '-1':
        to_analyze = tes_keys

    # if to_analyze does not exist or is a list containing not valid TES, returns an error
    elif not to_analyze or (isinstance(to_analyze, list) and any(tes not in tes_keys for tes in to_analyze)):
        error = f"Some selected TES are invalid\n{mode = } | {to_analyze = }\nAvailable tes index: {tes_keys}"

        print(red_color_code + error + reset_color_code)
        raise Exception(error)

    # among the available TES, returns the TES present in the requested range
    if mode == 'range':
        iter_ = tes_keys[tes_keys.index(to_analyze[0]): tes_keys.index(to_analyze[1])]

    # among the available TES, returns the TES present in the requested sequence
    elif mode == 'sequence':
        iter_ = to_analyze

    # All valid TES are analyzed
    else:
        iter_ = tes_keys

    logger.info("Mode: %s | Tes: %s", mode, to_analyze)

    return iter_


def get_datasets(src_path: str) -> list:
    """
    Returns a list containing only the paths of datasets
    that contain the folders Hks, Raws, and Sums

    Parameters
    ---------
    src_path: str
        path to the folder containing the dataset

    Returns
    -------
    list:
        a list containing only the paths of datasets
        that contain the folders Hks, Raws, and Sums
    """

    datasets: list[str] = []

    # iterates through all datasets that match any wildcards
    # passed within the src_path parameter
    for folder in glob.glob(src_path):
        for root, dirs, files in os.walk(folder):
            if all(d in dirs for d in ['Hks', 'Raws', 'Sums']):
                datasets.append(root)

    return sorted(datasets, reverse=True)


def make_result_dir(src_path: str, dst_path: str, args: argparse.Namespace = None):
    """
    Create the folder that will contain all the results for a specific dataset and set up its logger

    Parameters
    ---------
    src_path: str
        path to the folder containing the dataset

    dst_path: str
        folder where two sub-folders are created:
        1. input, which contains the .npy (time) and .npz (signals of all TES) files;
        2. output, which contains the .json file of the taus;

    args: argparse.Namespace
        object that contains the arguments passed via command line. Used to retrieve information about:
        1. Slope of linear regression
        2. Number of points of the exponential decreasing trend
        3. Number of points of the linear regression
    """

    # the dictionary is reset with each new dataset to be analyzed
    global SHARED_INFO
    SHARED_INFO = {key: value if value is not str else '' for key, value in SHARED_INFO.items()}

    if args:
        SHARED_INFO['slope'] = args.slope
        SHARED_INFO['points_vertical_trend'] = args.vertical_points
        SHARED_INFO['points_exp_decrease'] = args.exp_points

    # extraction of the name of the folder containing all the data to be analysed
    folder = "crd_" + src_path.split(os.sep)[-1]

    # associate the path where the input and output folders will be created (necessary for data analysis)
    # to the 'observation_dir' key
    SHARED_INFO['observation_dir'] = os.path.join(dst_path, folder if 'link' not in folder else 'no_info')

    # creating folder 'observation_dir'. If it already exists, it does not return an exception
    os.makedirs(SHARED_INFO['observation_dir'], exist_ok=True)

    SHARED_INFO['logfname'] = os.path.join(SHARED_INFO['observation_dir'], f"crd_{src_path.split(os.sep)[-1]}.log")
    # defines general settings for the log file
    logging.basicConfig(filename=SHARED_INFO['logfname'],
                        filemode="w",
                        encoding='utf8',
                        format="%(asctime)s - %(name)s - %(funcName)s - %(message)s",
                        datefmt="%d/%m/%Y | %H:%M:%S",
                        level=logging.INFO,
                        force=True)


def export_data(src_path: str, time_fname: str, signal_raw_fname: str, mask: list[list[int]] = None):
    """
     Reads the data using qubicfp() and exports the time and signals of the TES with their respective signs.
     If the mask is null, all TES are considered valid

     Parameters
     ---------
     src_path: str
         path to the folder containing the dataset

     time_fname: str
         path to the file containing time

     signal_raw_fname: str
         path to the file containing the signals

     mask: list[list[int]], None
         the mask with the indices of valid TES converted to integers;
         if the mask is None: all TES are considered valid and positive
     """

    # if the time file and the file containing the signals exist,
    # it returns the keys associated with the TES that can be analyzed
    if os.path.isfile(time_fname) and os.path.isfile(signal_raw_fname):
        return np.load(signal_raw_fname).files

    print("Exporting data via qubicfp (thermometers will be analyzed but not displayed on the focal plane)...")

    qubic = qubicfp()
    qubic.verbosity = 0
    qubic.read_qubicstudio_dataset(src_path)
    qubic = qubic.tod()

    # saves the shifted time array
    np.save(time_fname, qubic[0] - qubic[0][0])

    # contains the indices saved in the mask, excluding -1 (keys of the .npz file)
    keys = []
    # contains the signals with reversed signs
    masked = []

    mask = mask or [list(range(qubic[1].shape[0]))]

    # the first line of the mask corresponds to positive TES,
    # the second line of the mask corresponds to negative TES
    for row in range(len(mask)):
        # if the row contains only the value "-1" (i.e. no valid TESs),
        # consider the next row
        if len(mask[row]) == 1 and mask[row][0] == -1:
            continue
        for tes in mask[row]:
            # save the TES indices
            keys.append(str(tes))
            # save the TES signal with the corresponding sign.
            # The row of positive TES has index zero, so you have (-1)^0 * TES signal.
            # That of the negative TES has index 1, so you have (-1)^1 * TES signal
            masked.append((-1) ** row * qubic[1][tes])

    # saves the signals in the .npz file
    np.savez(signal_raw_fname, **dict(zip(keys, masked)))


def configure_data(src_path: str, dst_path: str, mask: list[list[int]] | None) -> SharedMemory:
    """
    Configuration for data export and the saving point of the exported files
    (.npy and .npz format)

    Parameters
    ---------
    dst_path: str
        folder where two sub-folders are created:
            1. input, which contains the .npy (time) and .npz (signals of all TES) files;
            2. output, which contains the .json file of the taus;

    src_path: str
        folder containing sky scan data in .fits format

    mask: list[list[]]
        mask with the indices of valid TES converted to integers

    Returns
    ------
    SharedMemory
        the object that manages the shared memory of the time
    """

    logger.info('path where the input and output folders of the preprocessed data are located (npy, npz files + taus, '
                'plots): %s', dst_path)
    logger.info('folder containing sky scan data in .fits format: %s', src_path)

    SHARED_INFO['src_dir'] = os.path.join(SHARED_INFO['observation_dir'], "input")
    SHARED_INFO['dst_dir'] = os.path.join(SHARED_INFO['observation_dir'], "output")

    logger.info('input dir: %s', SHARED_INFO['src_dir'])
    logger.info('output dir: %s', SHARED_INFO['dst_dir'])

    os.makedirs(SHARED_INFO['src_dir'], exist_ok=True)
    os.makedirs(SHARED_INFO['dst_dir'], exist_ok=True)

    logger.info('input dir created')
    logger.info('output dir created')

    SHARED_INFO['time_fname'] = os.path.join(SHARED_INFO['src_dir'], "times_raw.npy")
    SHARED_INFO['signals_fname'] = os.path.join(SHARED_INFO['src_dir'], "signals_raw.npz")
    SHARED_INFO['signals_clean_fname'] = os.path.join(SHARED_INFO['src_dir'], "signals_clean.npz")
    SHARED_INFO['taus_fname'] = os.path.join(SHARED_INFO['dst_dir'], f"crd_taus__{src_path.split(os.sep)[-1]}.json")

    logger.info('time file path: %s', SHARED_INFO['time_fname'])
    logger.info('raw signal file path: %s', SHARED_INFO['signals_fname'])
    logger.info('signal clean file path: %s', SHARED_INFO['signals_clean_fname'])
    logger.info('taus file path: %s', SHARED_INFO['taus_fname'])

    SHARED_INFO['plots_dir'] = os.path.join(SHARED_INFO['dst_dir'], "plots")
    logger.info('plots dir: %s', SHARED_INFO['plots_dir'])

    os.makedirs(SHARED_INFO['plots_dir'], exist_ok=True)
    logger.info('plots dir created')

    # list containing the path to the folder containing the data, the path to which to export the time and signal data
    args = [src_path, SHARED_INFO['time_fname'], SHARED_INFO['signals_fname']]
    logger.info('args for data exporting: %s', args)

    export_data(src_path=src_path, time_fname=SHARED_INFO['time_fname'],
                signal_raw_fname=SHARED_INFO['signals_fname'],
                mask=mask)

    signal_time = np.load(SHARED_INFO['time_fname'])
    SHARED_INFO['t_knots'] = np.linspace(signal_time[1], signal_time[-2], SHARED_INFO['n_knots'])

    logger.info('t_knots: %s', SHARED_INFO['t_knots'])

    # once the time and signal data have been exported, allocate the shared memory for the time data
    time_shm = create_shared_memory(data=signal_time,
                                    name='time_shm',
                                    shm_name='np_time_shared')

    logger.info('time array saved in shared memory')

    return time_shm


def create_shared_memory(data, name: str, shm_name: str) -> SharedMemory:
    """
    Create a block of shared memory in which to save the information contained in "data"

    Parameters
    ---------

    data: np.ndarray
        data to be saved in the shared memory

    name: str
        key, present in SHARED_INFO, associated with the corresponding dictionary

    shm_name: str
        shared memory block identifier

    Returns
    ------

    SharedMemory
        object that manages the shared block of memory

    """

    SHARED_INFO[name]["name"] = shm_name
    SHARED_INFO[name]["size"] = size = data.nbytes
    SHARED_INFO[name]["shape"] = shape = data.shape
    SHARED_INFO[name]["dtype"] = dtype = data.dtype

    # 1. create the shared memory block
    try:
        shm = SharedMemory(create=True, size=size, name=shm_name)
    except FileExistsError:
        logger.warning(f"Shared Memory '%s' already exist. Be aware of side effects", shm_name)
        return SharedMemory(name=shm_name)

    # 2. create a generic vector that can connect to the shared memory block
    dst = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)  # shm.buf: link to the shared block of memory

    # 3. copy the data to dst
    dst[:] = data[:]

    return shm


def interpolated_signal(data: tuple) -> np.ndarray:
    """
    Signal interpolation with a third degree function (atmosphere drift removal).
    Refer to: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQUnivariateSpline.html

    Parameters
    ---------
    data: tuple
        n_tes: int
            TES number

        shared_info: dict
            dictionary containing the information needed by all processes to read data into shared memory

    Returns
    ------
    np.ndarray
        signal cleaned from the atmospheric drift
    """

    n_tes, shared_info = data

    # interior knots of the spline. Must be in ascending order.
    # Knots must satisfy the Schoenberg-Whitney conditions:
    # there must be a subset of data points x[j] such that t[j] < x[j] < t[j+k+1], for j=0, 1,...,n-k-2.
    knots = shared_info["t_knots"]

    # access information relating to the time shm shared memory block
    t_name, t_size, t_shape, t_dtype = shared_info["time_shm"].values()

    # retrieve object that manages shared memory
    time_shm = SharedMemory(name=t_name)

    # create a generic array able to connect to the shared memory block
    times_raw = np.ndarray(shape=t_shape, dtype=t_dtype,
                           buffer=time_shm.buf)  # shm.buf: link to the shared block of memory

    # load the raw signal of the TES to analyze
    signal_raw = np.load(shared_info['signals_fname'])[n_tes]

    # LSQUnivariateSpline object representing a least squares fit
    fit = LSQUnivariateSpline(times_raw, signal_raw, knots)

    return signal_raw - fit(times_raw)


def save_interp_signals() -> list[str]:
    """
    Create a file in .npz format in which saves the signals cleaned by the atmosphere (s_clean)

    Returns
    ------
    list[str]:
        keys of the available TES to be analyzed
    """

    # load the file containing the signal data
    signals = np.load(SHARED_INFO['signals_fname'])
    logger.info("%s signals will be exported in .npz format", len(signals))

    # check if the file containing the leased data exists, otherwise doesn't create it every time
    if os.path.isfile(SHARED_INFO['signals_clean_fname']):
        logger.info("the file '%s' already exist. Skipping", SHARED_INFO['signals_clean_fname'])
        return signals.files

    print('Saving signals cleaned from the atmospheric drift...')

    # mp.Pool is the manager of child processes
    with mp.Pool() as pool:
        # imap() accepts a function and the arguments to pass to the function (respects the input order in output).
        # arguments are packaged as a single argument

        res = pool.imap(interpolated_signal,
                        zip(signals.files, [SHARED_INFO] * len(signals)),
                        chunksize=max(2, len(signals) // mp.cpu_count()))

        # create the .npz file in which I save the fitted signal of each TES
        # numpy sees the .npz file as a list of .npy files [('arr_0', <array_0>), ('arr_1', <array_1>), ...]
        # the asterisk unpacks the generic res iterator containing the fitted signals of each TES
        np.savez(SHARED_INFO['signals_clean_fname'], **dict(zip(signals.files, res)))

    # logger.info("the file '%s' has been created", SHARED_INFO['signals_clean_fname'])
    return signals.files


def exp_decay(t: np.ndarray, a: int, b: int, c: float) -> np.ndarray:
    """
    model representing the exponential decreasing trend.

    Parameters
    ----------
    t: np.ndarray
        time array
    a: int
        steady state value
    b: int
        amplitude
    c: float
        equal to -1/tau, where tau is the time constant
        in the exponential decreasing trend

    Returns
    -------
    np.ndarray
        function with which perform the exponential decreasing fit
    """

    return a + b * np.exp(c * t)


def get_initial_params(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    The function calculates the initial parameters to be passed to scipy.optimize.curve_fit.
    Thanks to:
    https://it.scribd.com/doc/14674814/Regressions-et-equations-integrales
    https://stackoverflow.com/questions/77822770/exponential-fit-is-failing-in-some-cases/77840735#77840735

    Parameters
    ----------
    x: np.ndarray
        time array of the exponential decreasing trend
    y: np.ndarray
        signal array of the exponential decreasing trend

    Returns
    -------
    tuple
        Returns the initial parameters a,b,c which refer to the function `exp_decay`
    """

    s = np.zeros_like(y)
    s[1:] = s[:-1] + 0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])

    c11 = np.sum((x - x[0]) ** 2)
    c12 = np.sum((x - x[0]) * s)
    c21 = c12
    c22 = np.sum(s ** 2)

    cf11 = np.sum((y - y[0]) * (x - x[0]))
    cf21 = np.sum((y - y[0]) * s)

    c_matrix = np.array([[c11, c12],
                         [c21, c22]])

    c_factor = np.array([[cf11],
                         [cf21]])

    c = (np.linalg.inv(c_matrix) @ c_factor)[1]

    ab11 = len(y)
    ab12 = np.sum(np.exp(c * x))
    ab21 = ab12
    ab22 = np.sum(np.exp(2 * c * x))

    abf11 = np.sum(y)
    abf21 = np.sum(y * np.exp(c * x))

    ab_matrix = np.array([[ab11, ab12],
                          [ab21, ab22]])

    ab_factor = np.array([[abf11],
                          [abf21]])

    # if the matrix is singular (det = 0), it cannot be inverted
    # so I replace all null values with a very small value
    ab_matrix = np.where(ab_matrix == 0, 1e-10, ab_matrix)

    a, b = np.linalg.inv(ab_matrix) @ ab_factor

    return a, b, c


def get_fit_candidate(time_raw: np.ndarray, s_clean: np.ndarray, std: np.ndarray) -> np.array:
    """
    It performs the linear fit of a valid candidate

    Parameters
    ---------
    time_raw: np.ndarray
        array of time

    s_clean: np.ndarray
        array of signal values

    std: np.ndarray
        standard deviation of the signal

    Returns
    ------
    np.array
       popt: optimal values for the parameters so that the sum of the
             squared residuals of f(xdata, *popt) - ydata is minimized.
       uncertainty on the time constant
    """

    # index of the maximum value, which corresponds to the first point of exponential decrease
    sig_max_idx = s_clean.argmax()

    # scaling x such that the exponential decrease starts at x = 0
    x = time_raw[sig_max_idx:] - time_raw[0]
    y = s_clean[sig_max_idx:]
    # recovery the initial parameters
    p0 = get_initial_params(x, y)

    # absolute_sigma = True since we know the standard deviation on all the data
    popt, pcov = curve_fit(exp_decay, x, y, p0=p0, sigma=[std] * y.shape[0], absolute_sigma=True)

    return np.array([*popt, np.sqrt(pcov[2][2]) / popt[2] ** 2])


def straight_line(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Straight line passing through two points.

    Parameters
    ---------
    x: np.ndarray
        x coordinates of the points through which the straight line must pass

    y: np.ndarray
        y coordinates of the points through which the straight line must pass

    Returns
    ------
    tuple
        slope[0]: slope of the straight line
        intercept[0]: intercept of the straight line
    """

    # if two coordinates x are equal, returns (n.inf, n.inf)
    if not np.all(np.diff(x)):
        return np.inf, np.inf

    slope = np.diff(y) / np.diff(x)
    intercept = y[0] - slope * x[0]

    return slope[0], intercept[0]


def candidate_filter(tm_candidate: np.ndarray[float],
                     s_candidate: np.ndarray[float],
                     pts_vertical_trend: int,
                     slope: float) -> bool:
    """
    Returns True or False depending on whether the candidate is valid or not.
    Candidate is understood as linear vertical growth followed by exponential decrease.

    Parameters
    ---------
    tm_candidate: np.ndarray
        candidate time values

    s_candidate: np.ndarray
        candidate signal values

    pts_vertical_trend: int
        number of points that constitute the vertical linear growth

    slope: float
        minimum slope of the vertical linear growth

    Returns
    ------
    bool:
        True if the candidate is valid
        False if the candidate is not valid
    """

    if np.any(s_candidate <= 0.):
        return False

    # the index of the maximum in s_candidate indicates the number of points
    # of vertical growth. So, if the index of the maximum is less than the minimum
    # number of points of the linear growth, the candidate is not analyzed
    if (sig_max_idx := s_candidate.argmax()) < pts_vertical_trend:
        return False

    # select and shifted the part about vertical growth
    linear_time = tm_candidate[:sig_max_idx + 1] - tm_candidate[0]
    linear_signal = s_candidate[:sig_max_idx + 1]

    # check whether the slope is less than the minimum slope
    if linregress(linear_time, linear_signal).slope < slope:
        return False

    # select the part about exponential decreasing trend
    tm_exp_decay = tm_candidate[sig_max_idx:]
    s_exp_decay = s_candidate[sig_max_idx:]

    stop = s_exp_decay.shape[0] // 2

    # control to avoid pathological cases in which there is little more than linear decrement
    # (not recognized by the control performed by the for loop).
    # This is the minimum amplitude change required to consider a candidate as a cosmic ray
    if s_exp_decay[1] - s_exp_decay[-1] < 3780:
        return False

    # To verify that a series of decreasing points follows a decreasing exponential trend,
    # connect the first and last points with a straight line and verify that
    # the second and second-to-last points are below that line.
    # The process is iterated until half of the points constituting the decreasing trend are reached
    # (for reasons of symmetry)
    for index in range(stop):

        x = np.array([tm_exp_decay[index], tm_exp_decay[-index - 1], tm_exp_decay[index + 1], tm_exp_decay[-index - 2]])
        y = np.array([s_exp_decay[index], s_exp_decay[-index - 1], s_exp_decay[index + 1], s_exp_decay[-index - 2]])

        # connect two points with a straight line
        slope, intercept = straight_line(x[:2], y[:2])

        # if one of the points is above the line, the trend is not descending exponential
        # and the candidate is discarded
        if np.any(slope * x[2:] + intercept < y[2:]):
            return False

    return True


def multiprocess_filter(pool: mp.Pool,
                        candidates: list[list[int]],
                        function: callable,
                        args: tuple,
                        chunksize: int) -> list:
    """
    Builds a list from those elements of "candidates" for which "function" returns True.
    This is done by parallelizing the filter application to all elements of "candidates".

    Parameters
    ---------

    pool: multiprocessing.Pool
         mp.Pool is the manager of child processes

    candidates: list
            list of all the candidates

    function: callable
            function that acts as a filter

    args: tuple
        additional arguments to pass to the function "function"

    chunksize: int
            degree of load of each processor

    Returns
    ------

    list
        list of candidates
    """

    return [candidate for candidate, keep in zip(candidates,
                                                 pool.starmap(function,
                                                              args,
                                                              chunksize)) if keep]


def get_time_constant(candidate_fit: list[float]) -> list[float] | None:
    """
    The function calculates the time constants for each valid candidate

    Parameters
    ---------
    candidate_fit: list of np.ndarray
                matrix of slope coefficients and intercepts of valid candidates of a TES

    Returns
    -------
    list[float] | None:
        time constant and its uncertainty or None
    """

    if np.isnan(candidate_fit)[2]:
        return None

    return [round(-1 / candidate_fit[2], 6), candidate_fit[-1]]


def tau_filter(tau: float, tau_coeff: float, epsilon: float) -> bool:
    """
    Filters the candidate's time constant based on a given condition

    Parameters
    ----------
    tau: float
        time constant of the valid candidate

    tau_coeff: float
            time constant of the TES with which to compare the time constants of valid candidates of the TES

    epsilon: float
           threshold with which I establish whether the time constant of the TES is comparable with that of the
           valid candidate under examination

    Returns
    ------
    bool:
        True if the time constant is valid
        False if the time constant is not valid
    """

    return np.fabs(tau - tau_coeff) < epsilon


# Caching of compiled functions has several known limitations:
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

@njit(nogil=True, cache=True)
def get_candidates(s_clean: np.ndarray,
                   std_mask: np.ndarray,
                   points_exp_decrease: int,
                   offset: int = 0) -> list[list[int]]:
    """
    Search for potential cosmic rays.
    A candidate consists of vertical growth followed by exponential decrease

    Parameters
    ---------
    s_clean: np.ndarray
        array of signal values cleaned from the atmospheric drift

    std_mask: np.ndarray
        contains the indices of the signal points that are above
        the threshold (currently set at 5 * std(s_clean) + <signal>)

    points_exp_decrease: int
        minimum number of points for the decreasing trend

    offset: int
        add the indices of the generic candidate to locate it
        in the TOD

    Returns
    -------
    list[list[int]]
        list of lists, which contain the start and end index of a candidate
    """

    j = 0
    candidates = list()

    for i in std_mask:
        # skip the std_mask points that are already present in a candidate
        if i < j:
            continue

        points = []

        # iterate over the points above and below the threshold starting
        # from the index point "i"
        for j in range(i, s_clean.shape[0] - 1):

            # add indices to the candidate until the condition is verified:
            # i.e., when the value of the signal at index "j" is less than
            # the value of the signal at index "j+1"
            if s_clean[j] < s_clean[j + 1]:
                # add the last index that satisfies the condition
                points.append(j)

                # once the indices have been acquired for a candidate,
                # add it to the list of candidates only if it has at least
                # "points_per_candidate" points.
                # I save only the initial and final index of the candidate
                # if len(points) > points_per_candidate:
                if len(points) > points_exp_decrease:
                    # find linear vertical growth
                    start = points[0]

                    # find the number of points of linear growth
                    # this while loop stops one index BEFORE the last value
                    # that satisfies the condition, so when I save the candidate
                    # I have to subtract 1 from start
                    while s_clean[max(start - 1, 0)] > s_clean[max(start - 2, 0)]:
                        start -= 1

                    # candidate is vertical growth followed by decreasing exponential.
                    # Contains the initial index of linear growth and the final index
                    # of exponential decreasing trend
                    candidates.append([offset + start - 1, offset + points[-1] + 1])
                break

            # add the consecutive decreasing points to points
            points.append(j)

    return candidates


def get_taus(tes_keys: list[str],
             mode: str,
             to_analyze: list[str]) -> dict[int, dict[str, list[float]]]:
    """
    Return TESs time constants

    Parameters
    ----------
    tes_keys: list[srt]
        all the available TES for analysis

    mode: str
        mode with which to analyze the TESs (ALL, range, sequence)

    to_analyze: list[str]
        TES to analyze

    Returns
    ------
    dict[int, dict[str, list[float]]]:
        dictionary whose keys (str) are the number of TESs and whose values (list of float numbers) are
        a list containing the time constants of the related TES
    """

    # intersection between the available TES and those requested by the user based on the selected mode
    iter_ = check_mode(mode, to_analyze, tes_keys)

    all_taus_found = 0
    taus_per_tes = dict()

    time_raw = np.load(SHARED_INFO["time_fname"])
    signals = np.load(SHARED_INFO["signals_clean_fname"])

    # mp.Pool is the manager of child processes
    with mp.Pool() as pool:

        for n_tes in tqdm(iter_, ncols=100, file=sys.stdout, desc="Progress", unit='tes'):  # colour='WHITE',

            s_clean = signals[n_tes]
            s_std = s_clean.std()
            std_mask = np.where(s_clean > SHARED_INFO['coeff'] * s_std + s_clean.mean())[0]

            if std_mask.size:

                # the parameter offset is necessary because, when I execute the candidate filter,
                # I retrieve the data for each candidate starting from the entire signal and time arrays
                candidates = get_candidates(
                    s_clean=s_clean[std_mask[0]: std_mask[-1] + 3 * SHARED_INFO['points_exp_decrease']],
                    std_mask=std_mask - std_mask[0], points_exp_decrease=SHARED_INFO["points_exp_decrease"],
                    offset=std_mask[0])

                if candidates:

                    chunksize = max(2, len(candidates) // mp.cpu_count())

                    # from here onwards, consider as a candidate the linear vertical growth
                    # followed by exponential decreasing trend
                    tm_candidates = list(map(lambda c: time_raw[c[0]: c[1]], candidates))
                    sig_candidates = list(map(lambda c: s_clean[c[0]: c[1]], candidates))

                    candidates = multiprocess_filter(pool=pool,
                                                     candidates=candidates,
                                                     function=candidate_filter,
                                                     args=zip(tm_candidates,
                                                              sig_candidates,
                                                              [SHARED_INFO['points_vertical_trend']] * len(candidates),
                                                              [SHARED_INFO['slope']] * len(candidates)),
                                                     chunksize=chunksize)

                    chunksize = max(2, len(candidates) // mp.cpu_count())

                    # fit every valid candidate
                    tm_candidates = map(lambda c: time_raw[c[0]:c[1]], candidates)
                    sig_candidates = map(lambda c: s_clean[c[0]:c[1]], candidates)

                    # matrix containing n candidates rows and 4 columns (a,b,c, std(tau))
                    fit_matrix = pool.starmap(get_fit_candidate,
                                              zip(tm_candidates, sig_candidates, [s_std] * len(candidates)),
                                              chunksize)

                    # save the taus of the candidates in the list
                    if taus := pool.map(get_time_constant, fit_matrix, chunksize=chunksize):

                        # saves the time constants and their corresponding start and end indices
                        # of the candidate if the time constant is different from None
                        if data := list(filter(lambda x: x[0], zip(taus, candidates, fit_matrix))):

                            n_tes = int(n_tes)
                            all_taus_found += len(taus)
                            tes, asic = get_tes_asic_from_index(n_tes)

                            taus_per_tes[n_tes] = {'taus': [row[0][0] for row in data]}
                            taus_per_tes[n_tes]['tes'] = [tes, asic]
                            taus_per_tes[n_tes]['sigma'] = [row[0][1] for row in data]
                            taus_per_tes[n_tes]['indexes'] = [row[1] for row in data]
                            taus_per_tes[n_tes]['exp fit params'] = [row[2][:-1].tolist() for row in data]

                            logger.info("#Tes (index): %s | #Time constants: %s", n_tes, len(taus))

                # filter the candidate's time constant based on the time constant of the TES
                # taus_per_tes.extend(multiprocess_filter(pool=pool,
                #                                         candidates=taus,
                #                                         func=tau_filter,
                #                                         args=zip(taus,
                #                                                  [SHARED_INFO['tau_coeff']] * len(taus),
                #                                                  [SHARED_INFO['epsilon']] * len(taus)),
                #                                         chunksize=chunksize))

        logger.info("Number of time constants for all TESs: %s", all_taus_found)

        return taus_per_tes


def release_share_memory():
    """
    It frees shared memory
    """

    for key in SHARED_INFO:
        #  check that the memory freeing operations are carried out only in the presence of shared memory portions
        if 'shm' in key and 'name' in SHARED_INFO[key]:
            # access the shared memory block
            try:
                shm = SharedMemory(name=SHARED_INFO[key]['name'])
                # close access to shared memory block for process calling .close()
                shm.close()
                # release shared memory
                shm.unlink()
                logger.info("released shared memory for '%s'", SHARED_INFO[key]['name'])
                SHARED_INFO[key] = dict()

            except FileNotFoundError:
                logger.exception("Error releasing shared memory for '%s': ", SHARED_INFO[key]['name'])


def get_tes_asic_from_index(tes: int) -> tuple[int, int]:
    """
    Returns the ASIC related to the passed TES

    Parameters
    ----------
    tes: int
        TES index [0, 255]

    Returns
    ------
    tuple[int, int]
        returns the TES index in the range [1, 128] and its corresponding ASIC
    """

    return (tes + 1, 1) if tes + 1 < 129 else (tes - 128 + 1, 2)


def plot_tau_candidates(fig: plt.Figure, taus: dict, shared_info: dict, n_tes: str):
    """
    In the same figure, the function plots the candidates of a TES (one subplot per candidate)
    and a plot containing the time constants of the TES candidates relative to the total acquisition time.

    In the plots related to individual candidates, the following information is displayed:\n
    - Points preceding and following the candidate
    - The candidate itself
    - Threshold above which the starting point of exponential decay is sought (5*sigma + <signal>)
    - Threshold below which the candidate should not fall (sigma + <signal>) due to low signal-to-noise ratio

    Parameters
    ----------
    fig: plt.Figure
        figure in which to make the plots

    taus: dict
        dictionary containing the taus of the TES under analysis

    shared_info: dict
        dictionary containing the information needed by all processes to read data into shared memory

    n_tes: str
        TES index in the range [0, 255]
    """

    # load the entire time and signal of the TES
    time_raw = np.load(shared_info['time_fname'])
    clean_signal = np.load(shared_info['signals_clean_fname'])[n_tes]

    # calculate the mean and standard deviation of the signal
    s_std = clean_signal.std()
    s_mean = clean_signal.mean()

    # performs the plots of the candidates (first row of the figure)
    for index, ax in enumerate(fig.axes[:-1]):

        ax.set_facecolor('whitesmoke')
        ax.set_title(r'$Candidate \ \#{}$'.format(index + 1))
        ax.set_xlabel(r'$time \ [s]$')
        ax.set_ylabel(r'$clean \ signal \ [ADU]$' if not index else '')

        # indices of the start and end of the current candidate
        start, stop = taus['indexes'][index]
        popt = taus['exp fit params'][index]

        # overview points (before and after the candidate)
        ov_t = [*time_raw[max(0, start - 5): start], *time_raw[stop + 1: min(len(time_raw), stop + 5)]]
        ov_s = [*clean_signal[max(0, start - 5): start], *clean_signal[stop + 1: min(len(clean_signal), stop + 5)]]

        # time and signal of the candidate
        t = time_raw[start: stop]
        s = clean_signal[start: stop]

        offset = t[0]
        lin_raise_x = t[:s.argmax() + 1]
        lin_raise_y = s[:s.argmax() + 1]

        lin_reg = linregress(lin_raise_x - offset, lin_raise_y)
        lin_y_est = lin_reg.intercept + lin_reg.slope * (lin_raise_x - offset)

        exp_dec_x = t[s.argmax():]
        exp_dec_y = s[s.argmax():]
        exp_dec_fit_x = np.linspace(exp_dec_x[0], exp_dec_x[-1], 1000)
        exp_dec_y_est = exp_decay(exp_dec_fit_x - offset, *popt)

        ax.scatter(ov_t, ov_s)
        ax.scatter(lin_raise_x, lin_raise_y)
        ax.scatter(exp_dec_x, exp_dec_y)

        ax.plot(exp_dec_fit_x, exp_dec_y_est, 'c--', label=r'$\tau = ' + str(round(-1 / popt[2], 5)) + ' \ s$')
        ax.plot(lin_raise_x, lin_y_est, 'k--', label=r'$slope = ' + str(round(lin_reg.slope, 2)) + ' \ s^{-1}$')

        if not np.all(exp_dec_y > s_mean + shared_info['coeff'] * s_std):
            ax.axhline(s_mean + shared_info['coeff'] * s_std,
                       label=r'$ 5\cdot\sigma + \langle s \rangle $', color='grey',
                       linestyle=(0, (1, 10)))  # loosely dotted

        if np.any(exp_dec_y[-1] < s_mean + s_std):
            ax.axhline(s_mean + s_std, label=r'$ \sigma + \langle s \rangle $', color='grey', linestyle=(0, (1, 10)))

        ax.legend()
        ax.grid(True)

        # plot of all the tau values in the time stream of the data
        all_taus_ax = fig.axes[-1]
        # time constants and their corresponding uncertainties of the TES under examination
        tau = taus['taus'][index]
        sigma = taus['sigma'][index]

        all_taus_ax.errorbar(t[0], tau, yerr=sigma,
                             marker='o', linestyle='dashed',
                             linewidth=1, markersize=5,
                             label=r"${} \pm {}$".format(tau, round(sigma, 5)))

        tes_asic = r'\left[' + ', '.join(map(str, get_tes_asic_from_index(int(n_tes)))) + r'\right]'
        all_taus_ax.legend(title=r'$\tau \pm \sigma$', loc='best')
        all_taus_ax.set_facecolor('whitesmoke')
        all_taus_ax.set_title(r'$\tau \ along \ time \ domain \ | \ TES \ {}$'.format(tes_asic))
        all_taus_ax.set_xlabel(r'$time \ [s]$')
        all_taus_ax.set_ylabel(r'$\tau \ [s]$')
        all_taus_ax.grid(True)


def plot_fp(fig: plt.Figure, taus: dict, shared_info: dict):
    """
    Plots the focal plane where the 2 ASICs are colored differently.
    Highlights the TES that have at least one candidate.
    TES assumed as thermometers are not shown in the plot.

    Parameters
    ----------
    fig : plt.Figure
        figure in which to make the fp plot

    taus: dict
        dictionary containing the taus of the TES under analysis

    shared_info: dict
        dictionary containing the information needed by all processes to read data into shared memory
    """

    fp_ax = fig.axes[1]

    fp_xy_limit = (-1, 17)
    fp_ax.set_xlim(*fp_xy_limit)
    fp_ax.set_ylim(*fp_xy_limit)

    fp_ax.set_title(r'$Focal \ Plane$')
    fp_ax.set_facecolor('whitesmoke')
    fp_ax.tick_params(labelbottom=False, labelleft=False)

    # finds the physical position of the TES in the ASIC
    fp_identity = pt.make_id_focalplane()
    asic_colors = ['cyan', 'orange', 'magenta', 'red', 'black']

    for asic in [1, 2]:

        bbox = dict(boxstyle=mpatches.BoxStyle("Round", pad=0.4),
                    edgecolor=asic_colors[asic - 1],
                    facecolor='white')

        fp_ax.text(**dict(x=0, y=4 - asic, s=f'ASIC {asic}',
                          size=7, rotation='horizontal', bbox=bbox))

        fp_ax.set_axis_off()

        for tes in range(1, 129):

            # in the focal plane, thermometers are not included,
            # but in the plot of all the tau values for all TES,
            # the possible time constants of cosmic rays falling
            # on thermometers are displayed.
            # To exclude the analysis of thermometers upstream,
            # you can use the 'mask' parameter, a file containing
            # two lines with all the TES to be analyzed except
            # for the thermometers themselves
            if tes in shared_info['thermometers']:
                continue

            # recover the position of the TES on the focal plane
            coords = (fp_identity.TES == tes) & (fp_identity.ASIC == asic)
            x, y = fp_identity[coords].col, fp_identity[coords].row

            # highlight a TES if it has a candidate
            alpha = 1 if [tes, asic] in [taus[tes]['tes'] for tes in taus] else 0.4

            bbox = dict(boxstyle=mpatches.BoxStyle("Round", pad=0.4),
                        edgecolor=asic_colors[asic - 1], facecolor='white', alpha=alpha)

            fp_ax.text(**dict(x=x, y=y, s=str(tes), size=4.8, rotation='horizontal', bbox=bbox), alpha=alpha)


def get_non_squared_subplots(n_candidates: int, **fg_kw) -> plt.subplots:
    """
    Returns a figure containing, on the first row, a number of subplots
    equal to the number of candidates of a TES, and on the second row,
    a single plot (containing all the time constants of the TES)

    Parameters
    ---------
    n_candidates: int
        number of candidates of a TES

    fg_kw: dict
        .Figure properties, optional

    Returns
    -------
    fig, ax: plt.subplots
        figure and axes to handle it
    """

    # create a figure
    fig = plt.figure(**fg_kw)

    # create a grid of 2 rows and 1 column associated with the figure
    gs0 = gridspec.GridSpec(2, 1, figure=fig)

    # create a subgrid with 1 row and N candidates to subdivide
    # the first row of the main grid
    gs00 = gs0[0].subgridspec(1, n_candidates)

    # create a subgrid of one row and one column
    gs01 = gs0[1].subgridspec(1, 1)

    # In conclusion, you will have a figure divided into two rows.
    # The first row contains N plots, one for each candidate,
    # and the second row is a single plot with the time constants and their corresponding sigma

    axs = [fig.add_subplot(gs00[0, i]) for i in range(n_candidates)] + [fig.add_subplot(gs01[0])]

    return fig, axs


def plot_taus(shared_info: dict, tes_taus: dict, n_tes: int = -1):
    """
    It performs two types of plots:\n
    - the plots of the time constants of a single TES
    - the plot of all the time constants of all the TES

    To plot the time constants of a single TES, you need to pass the TES number and the list of time constants.
    To plot the time constants of all the TES, you need to pass only the dictionary that has the TES as keys
    and the lists containing the TES time constants as values.

    Parameters
    ----------
    shared_info: dict
        dictionary containing the information needed by all processes to read data into shared memory

    tes_taus: dict
        dict of time constants of a single TES to be plotted

    n_tes: int
        TES number ([0,255]) for plotting the time constants
    """

    dataset = shared_info['observation_dir'].split(os.sep)[-1][3:].replace('_', ' ')

    # if no valid arguments are passed to the function, it does not plot anything
    if not tes_taus:
        return None

    fig_kw = dict(figsize=(16, 10), tight_layout=True)

    # Plot the time constants related to a single TES
    if n_tes >= 0:

        if not (taus := tes_taus['taus']):
            return None

        # if the tes has only one candidate, I plot the center of the first row
        n_axes = len(taus) if len(taus) > 1 else 3
        fig, axes = get_non_squared_subplots(n_axes, **fig_kw)
        fig.suptitle('{} | TES, ASIC : {}'.format(dataset, get_tes_asic_from_index(n_tes)))

        # remove the left and right axes, leaving only the one in the center
        if len(taus) == 1:
            fig.axes[0].remove()
            fig.axes[1].remove()

        plot_tau_candidates(fig, tes_taus, shared_info, str(n_tes))
        fig.savefig(os.path.join(shared_info['plots_dir'], f"{n_tes}_taus_plot"), dpi=1000)

    # plot the time constants of all the TESs
    elif n_tes == -1:

        all_taus = []
        all_taus_sigma = []
        # set n_Candidates = 3 because this allows me to have three equispaced subplots
        # in the first row, of which I will only use the first and remove the other 2
        fig, axes = get_non_squared_subplots(3, **fig_kw)
        fig.suptitle(dataset)

        plot_fp(fig, tes_taus, shared_info)
        ax_title = (f"Taus: all TES | #pts vertical linear trend: ≥ {shared_info['points_vertical_trend']} | #pts exp "
                    f"decay: ≥ {shared_info['points_exp_decrease']}")
        axes[0].remove()
        axes[2].remove()
        td_ax = axes[-1]
        td_ax.set_facecolor('whitesmoke')
        td_ax.set_title(ax_title)
        td_ax.set_xlabel(r"$time \ [s]$")
        td_ax.set_ylabel(r"$\tau \ [s]$")
        td_ax.set_yscale('log')
        td_ax.grid(True)

        time_raw = np.load(shared_info['time_fname'])

        taus = list(map(int, tes_taus.keys()))
        scalar_mappable = plt.cm.ScalarMappable(norm=plt.Normalize(min(taus), max(taus)), cmap='copper')
        colors = scalar_mappable.to_rgba(taus)

        for (tes, taus), color in zip(tes_taus.items(), colors):
            # save all the found time constants
            all_taus.extend(taus['taus'])
            all_taus_sigma.extend(taus['sigma'])

            for index, tau in enumerate(taus['taus']):
                td_ax.errorbar(time_raw[taus['indexes'][index][0]],
                               tau,
                               yerr=taus['sigma'][index],
                               color=color,
                               marker='o', linestyle='dashed',
                               linewidth=1, markersize=5,
                               label=r"${}:{} \pm {}$".format(taus['tes'],
                                                              round(tau, 4),
                                                              round(taus['sigma'][index], 4)))

        taus_mean = np.mean(all_taus)
        taus_std = np.mean(all_taus_sigma) / len(all_taus) ** 0.5

        td_ax.axhline(taus_mean,
                      linestyle='dotted', color='dimgray',
                      label=r'$\langle \tau \rangle: {:.5f} \pm {:.5f} $'.format(taus_mean, taus_std))

        td_ax.legend(title=r'[TES, ASIC] $: \tau \pm \sigma \ [s]$', loc='best')
        fig.savefig(os.path.join(shared_info['plots_dir'], f"taus_for_all_tes"), dpi=1000)


def find_cosmic_rays(src_path: str,
                     dst_path: str,
                     mode: str,
                     to_analyze: list[int] | int,
                     mask_fname: str,
                     remove_files: bool) -> int:
    """
    Searches for cosmic rays starting from the time constants

    Parameters
    ----------
    src_path: str
        folder containing sky scan data in .fits format

    dst_path: str
        folder where two folders are created:
            1. Input, which contains the .npy (time) and .npz (signals of all TES) files;
            2. Output, which contains the .txt file of the taus;

    mode: str
        specifies the type of analysis to be performed:
            - analysis of all tes;
            - analysis of a sequence of tes (e.g., 1 3 38 44);
            - analysis of a range of tes (e.g., from 5 to 120)

    to_analyze: list[int] | int
        TES to analyze

    mask_fname: str
        path to the mask file, containing the information of the TES sign

    remove_files: bool
        remove .npy and .npz after analysis

    Returns
    -------
    int
        number of candidates found for the dataset under analysis
    """

    # start = tm.perf_counter()
    logger.info("configuration of all the necessary variables")

    # configuration of saving point of the exported files (.npy and .npz format)
    _ = configure_data(src_path=src_path, dst_path=dst_path, mask=check_mask(mask_fname=mask_fname))

    logger.info("export clean signals from atmospheric drift")

    # export clean signals in .npz format
    tes_keys = save_interp_signals()
    release_share_memory()

    logger.info("search for the time constants of all TESs")

    # dictionary whose keys are the number of TESs and whose values are
    # a list containing the time constants of the related TES
    start = tm.perf_counter()
    taus_per_tes = get_taus(tes_keys=tes_keys, mode=mode, to_analyze=list(map(str, to_analyze)))

    end = tm.perf_counter()
    execution_time = f"Time to process signals: {end - start:.2f}[s]"
    logger.info(execution_time)

    print("\033[96m" + execution_time + "\033[00m")

    logger.info("write time constants to the file '%s'", SHARED_INFO['taus_fname'])

    write_dict_to_json(SHARED_INFO['taus_fname'], taus_per_tes)

    logger.info("save the time constant plots for single TES and the time constant plot of all TES")
    print("\033[0;33m" + "Saving plots (this may take a while due to fig high resolution)..." + "\033[00m")

    with mp.Pool() as pool:
        pool.starmap(plot_taus,
                     zip([SHARED_INFO] * (len(taus_per_tes) + 1),
                         [taus_per_tes[tes] for tes in taus_per_tes] + [taus_per_tes],
                         list(taus_per_tes.keys()) + [-1]))

    if remove_files:

        for file in glob.glob(SHARED_INFO['src_dir'] + os.sep + "*"):
            os.remove(file)

        os.rmdir(SHARED_INFO['src_dir'])
        logger.info('.npy and .npz files removed')

        if not os.listdir(SHARED_INFO['plots_dir']):
            os.rmdir(SHARED_INFO['plots_dir'])
            os.remove(SHARED_INFO['taus_fname'])
            os.rmdir(SHARED_INFO['dst_dir'])
            logger.info('No time constants found. Plots dir removed')

    end = "Script completed successfully"
    print("\033[92m" + end + "\033[00m", end='\n\n')
    logger.info(end)

    return len([tau for tes in taus_per_tes for tau in taus_per_tes[tes]['taus']])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script for cosmic-rays detection",
                                     fromfile_prefix_chars='@',
                                     formatter_class=argparse.MetavarTypeHelpFormatter)

    parser.add_argument('srcpath',
                        type=pathlib.Path,
                        help='path of the folder containing the data to be analyzed',
                        metavar='source')

    parser.add_argument('-dst', '--dstpath',
                        type=pathlib.Path, default=os.getcwd(),
                        help='path to the folder containing final result',
                        metavar='destination')

    mask_help = """mask that filters the valid TES signals to be saved (with the relative sign). 
        So if a TES is negative, its signal will be exported by changing its sign. 
        The mask is used before exporting files in .npy and .npz format, so it will have no effect on existing files."""

    parser.add_argument('-m', '--mask',
                        type=pathlib.Path,
                        help=mask_help,
                        metavar='mask',
                        default='')

    parser.add_argument('-f', '--from_file',
                        type=pathlib.Path,
                        help='allows you to pass commands through a file (one per line). Ex: -f @crd_args.txt',
                        metavar='from file')

    parser.add_argument('-slp', '--slope',
                        type=float,
                        default=1100000,
                        help='minimum slope of linear regression (vertical growth)')

    parser.add_argument('-vp', '--vertical_points',
                        type=int,
                        default=3,
                        help='minimum number of points (vertical growth) to consider a candidate valid',
                        metavar='vertical points')

    parser.add_argument('-ep', '--exp_points',
                        type=int,
                        default=6,
                        help='minimum number of points (exponential decrease) to consider a candidate valid',
                        metavar='exponential decreasing points')

    parser.add_argument('-r', '--range', type=int, nargs=2, help='tes interval to be analyzed')
    parser.add_argument('-sq', '--sequence', type=int, nargs='+', help='the sequence of tes to be analyzed')

    parser.add_argument('-rf', '--remove_files',
                        action='store_true',
                        help='remove .npy and .npz files after the analysis has been completed')

    args = parser.parse_args()

    src_path = str(args.srcpath)
    dst_path = str(args.dstpath)
    mask = str(args.mask)
    remove_files = args.remove_files

    to_analyze = args.range or args.sequence or [-1]
    mode = 'range' if args.range else ('sequence' if args.sequence else 'all')

    if not (datasets := get_datasets(src_path)):
        print('\033[91m' + f"No datasets found for '{src_path}'" + '\033[0m')
        quit(-1)

    if len(datasets) >= 2 and not remove_files:
        warning = """\033[91mWarning!
                You have selected more than two datasets to analyze, and you haven't requested 
                the removal of .npy and .npz files. Keep in mind that analyzing a single dataset generates .npy and .npz files 
                that will take up approximately 9 GB of space in total! Do you want to MAINTAIN all the .npy and .npz files 
                that will be created (NOT recommended action)? [y/n]\033[0m """

        remove_files = input(warning).lower() != 'y'

    no_crd = list()
    results = dict()
    for src_path in datasets:
        make_result_dir(src_path=src_path, dst_path=dst_path, args=args)

        logger = logging.getLogger(__name__)
        logger.info('observation dir created')
        logger.info('observation dir: %s', SHARED_INFO['observation_dir'])
        logger.info("argparse Namespace: %s", args)

        print('\033[45m' + f'Dataset under analysis: {src_path}' + '\033[00m')

        try:
            n_taus = find_cosmic_rays(src_path=src_path,
                                      dst_path=dst_path,
                                      mode=mode,
                                      to_analyze=to_analyze,
                                      mask_fname=mask,
                                      remove_files=remove_files)

            if n_taus:
                results[os.path.basename(src_path)] = n_taus
            else:
                no_crd.append([SHARED_INFO['logfname'], SHARED_INFO['observation_dir']])

        except Exception:
            logger.exception("Exception while executing find_cosmic_rays")
            release_share_memory()

    write_dict_to_json(os.path.join(dst_path, 'candidates_per_dataset.json'), results, indent=4)

    # close all log files
    logging.shutdown()

    # cleaning of all folders related to datasets that do not have cosmic rays
    for dt in no_crd:
        if remove_files:
            # if a dataset has no cosmic rays, I just keep the log file and
            # put it in the folder that contains all the analyses
            try:
                shutil.move(dt[0], dst_path)
            except Exception:
                logfile = os.path.split(dt[0])[-1]
                os.remove(os.path.join(dst_path, logfile))
                shutil.move(dt[0], dst_path)
            # remove the folder that contained ONLY the log file
            shutil.rmtree(dt[1])

    print('\033[42m' + 'ALL DATASET ANALYZED' + '\033[00m')
