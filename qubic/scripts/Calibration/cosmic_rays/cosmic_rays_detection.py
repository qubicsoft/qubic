# coding: utf-8

import os
import sys
import json
import logging
import platform
import time as tm
import numpy as np
import subprocess as sp
import multiprocessing as mp
import matplotlib.pyplot as plt

from typing import Type, Union
from scipy.interpolate import LSQUnivariateSpline
from multiprocessing.shared_memory import SharedMemory

np.seterr(divide='ignore')  # used to not show a warning (division by zero in log)

# defines general settings for the log file
logging.basicConfig(filename=f"{os.path.basename(__file__).split('.')[0]}.log",
                    filemode="a",
                    # if the file doesn't exist, create it. If the file exists, add to the end of the file
                    encoding='utf8',
                    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
                    datefmt="%d/%m%Y %H:%M:%S",
                    level=logging.INFO)

logger = logging.getLogger(__name__)

# dictionary containing the information needed by all processes to read data into shared memory
# (parent and child process communicate only through shared memory (mediator))
SHARED_INFO = {"input_dir": '',
               "output_dir": '',
               "plots_dir": '',
               "taus_fname": '',
               "time_fname": '',
               "signals_fname": '',
               "signals_clean_fabs_fname": '',
               "time_shm": {},  # in time_shm we save all the information related to the time shared memory block
               # time_shm will be passed in the 'name' parameter in create_shared_memory
               "signals_clean_fabs_shm": {},
               "t_knots": np.array([3000, 3500, 4000, 4500, 5000, 5500, 6000]) + 1.68182e9,
               "coeff": 5,
               "epsilon": 1e-2,
               "tau_coeff": 50e-3,
               "std": Type[np.ndarray],
               "points_per_candidate": 7,
               "points_vertical_trend": 3}

#  the "SharedMemory" module was introduced starting from the 3.9 python version.
#  if the version of the python interpreter running the script is less than 3.9.x, the script will not be executed
version = platform.python_version()
major, minor, micro = tuple(map(int, version.split(".")))

if (major, minor) < (3, 9):
    print("This script requires at least python 3.9 to be executed")
    sys.exit(1)


def configure_data_dir(data_storage_path: str,
                       data_set_path: str,
                       qubic_interpreter: str = sys.executable) -> SharedMemory:
    """
    Configuration of the interpreter for data export and the saving point of the exported files
    (.npy and .npz format)

    Parameters
    ---------

    data_storage_path: str
        folder where two subfolders are created:
            1. input, which contains the .npy (time) and .npz (signals of all TES) files;
            2. output, which contains the .json file of the taus;

    data_set_path: str
        folder containing sky scan data in .fits format

    qubic_interpreter: str
        interpreter for data export

    Returns
    ------
    SharedMemory
        the object that manages the shared memory of the time

    """
    logger.info('path where the input and output folders of the preprocessed data are located (npy, npz files + taus, '
                'plots): %s', data_storage_path)
    logger.info('folder containing sky scan data in .fits format: %s', data_set_path)
    logger.info('interpreter for data export: %s', qubic_interpreter)

    # extraction of the name of the folder containing all the data to be analysed
    folder = "crd_" + data_set_path.split(os.sep)[-1]

    # associate the path where the input and output folders will be created (necessary for data analysis)
    # to the 'observation_dir' key
    SHARED_INFO['observation_dir'] = os.path.join(data_storage_path, folder if 'link' not in folder else 'last_scan')
    logger.info('observation dir: %s', SHARED_INFO['observation_dir'])

    # creating folder 'observation_dir'. If it already exists, it does not return an exception
    os.makedirs(SHARED_INFO['observation_dir'], exist_ok=True)
    logger.info('observation dir created')

    SHARED_INFO['input_dir'] = os.path.join(SHARED_INFO['observation_dir'], "input")
    SHARED_INFO['output_dir'] = os.path.join(SHARED_INFO['observation_dir'], "output")

    logger.info('input dir: %s', SHARED_INFO['input_dir'])
    logger.info('output dir: %s', SHARED_INFO['output_dir'])

    os.makedirs(SHARED_INFO['input_dir'], exist_ok=True)
    os.makedirs(SHARED_INFO['output_dir'], exist_ok=True)

    logger.info('input dir created')
    logger.info('output dir created')

    SHARED_INFO['time_fname'] = os.path.join(SHARED_INFO['input_dir'], "times_raw.npy")
    SHARED_INFO['signals_fname'] = os.path.join(SHARED_INFO['input_dir'], "signals_raw.npz")
    SHARED_INFO['signals_clean_fabs_fname'] = os.path.join(SHARED_INFO['input_dir'], "signals_clean_fabs.npz")
    SHARED_INFO['taus_fname'] = os.path.join(SHARED_INFO['output_dir'], "taus_per_tes.json")

    logger.info('time file path: %s', SHARED_INFO['time_fname'])
    logger.info('raw signal file path: %s', SHARED_INFO['signals_fname'])
    logger.info('signal clean fabs file path: %s', SHARED_INFO['signals_clean_fabs_fname'])
    logger.info('taus file path: %s', SHARED_INFO['taus_fname'])

    SHARED_INFO['plots_dir'] = os.path.join(SHARED_INFO['output_dir'], "plots")
    logger.info('plots dir: %s', SHARED_INFO['plots_dir'])

    os.makedirs(SHARED_INFO['plots_dir'], exist_ok=True)
    logger.info('plots dir created')

    # list containing the path to the folder containing the data, the path to which to export the time and signal data
    args = [data_set_path, SHARED_INFO['time_fname'], SHARED_INFO['signals_fname']]
    logger.info('args for data exporting: %s', args)

    # sp.run creates a subprocess that executes wrapper_qubic.py to which it passes the arguments contained
    # in args to export the time to an .npy file and the signals to a .npz file
    # sp.run returns an exception if the wrapper_qubic.py script failed to export the data
    _ = sp.run([qubic_interpreter, "wrapper_qubic.py"] + args, stdout=sp.DEVNULL, check=True)

    # once the time and signal data have been exported, allocate the shared memory for the time data
    time_shm = create_shared_memory(data=np.load(SHARED_INFO['time_fname']),
                                    name='time_shm',
                                    shm_name='np_time_shared')

    logger.info('time array saved in shared memory')

    return time_shm


def create_shared_memory(data, name: str, shm_name: str):
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
    shm = SharedMemory(create=True, size=size, name=shm_name)

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

    n_tes: int
        TES number

    shared_info: dict
        dictionary containing the information needed by all processes to read data into shared memory

    Returns
    ------

    tuple[n_tes, fit]
        number of tes, CubicSpline object: polynomial coefficients, breakpoints (x passed to fit)
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
    signal_raw = np.load(shared_info['signals_fname'])[str(n_tes)]

    # CubicSpline object: polynomial coefficients, breakpoints (x passed to fit)
    # bc_type: boundary condition -> natural: second derivatives at the extremes of the curve are zero
    # fit = CubicSpline(times_raw, signal_raw, bc_type="natural")
    fit = LSQUnivariateSpline(times_raw, signal_raw, knots)

    res = np.fabs(signal_raw - fit(times_raw))
    
    time_shm.close()

    return res


def save_interp_signals() -> int:
    """
    Create a file in .npz format in which I save the signals in module and cleaned by the atmosphere (s_clean_fabs)

    Returns
    ------
    int
        number of TESs
    """

    # load the file containing the signal data
    signals = np.load(SHARED_INFO['signals_fname'])

    logger.info("%s signals will be exported in .npz format", len(signals.files))

    # check if the file containing the leased data exists, so I don't create it every time
    if os.path.isfile(SHARED_INFO['signals_clean_fabs_fname']):
        logger.info("the file '%s' already exist. Skipping...", SHARED_INFO['signals_clean_fabs_fname'])
        return len(signals)

    # mp.Pool is the manager of child processes
    with mp.Pool() as pool:
        # imap() accepts a function and the arguments to pass to the function (respects the input order in output).
        # arguments are packaged as a single argument

        res = pool.imap(interpolated_signal,
                        [(i, SHARED_INFO) for i in range(len(signals))],
                        chunksize=min(2, len(signals) // mp.cpu_count()))

        # create the .npz file in which I save the fitted signal of each TES
        # numpy sees the .npz file as a list of .npy files [('arr_0', <array_0>), ('arr_1', <array_1>), ...]
        # the asterisk unpacks the generic res iterator containing the fitted signals of each TES
        np.savez(SHARED_INFO['signals_clean_fabs_fname'], *res)

    logger.info("the file '%s' has been created", SHARED_INFO['signals_clean_fabs_fname'])

    return len(signals)


def get_candidates() -> list:
    """
    Search for consecutive points in the signal that follow a decreasing trend

    Returns
    ------

    list[list]
        list of lists, which contain the indices of consecutive decreasing points
    """

    # access information relating to the time shm shared memory block
    sfabs_name, sfabs_size, sfabs_shape, sfabs_dtype = SHARED_INFO["signals_clean_fabs_shm"].values()

    # retrieve object that manages shared memory
    s_clean_fabs_shm = SharedMemory(name=sfabs_name)

    # create a generic array able to connect to the shared memory block
    s_clean_fabs = np.ndarray(shape=sfabs_shape, dtype=sfabs_dtype,
                              buffer=s_clean_fabs_shm.buf)  # shm.buf: link to the shared block of memory

    # associate the standard deviation of the signal to the "std" key of the SHAREDINFO dictionary
    SHARED_INFO["std"] = np.std(s_clean_fabs)

    j = 0
    # minimum number of points per candidate
    points_per_candidate = SHARED_INFO["points_per_candidate"]

    candidates = list()

    # signals indices above shared_info['coeff']*sigma
    index_up_sigma = np.where(s_clean_fabs > SHARED_INFO['coeff'] * SHARED_INFO["std"])[0]

    for i in index_up_sigma:  # for each point in index_up_sigma

        if i < j:  # I skip the index_up_sigma points that are already present in a candidate
            continue
        points = []

        for j in range(i, s_clean_fabs.shape[0] - 1):  # iterate over the points above and below the threshold starting
            # from the index point "i"

            if s_clean_fabs[j] < s_clean_fabs[j + 1]:  # I add indices to the candidate until the condition is verified:
                # i.e. when the value of the signal at index "j" is less than
                # the value of the signal at index "j+1"

                points.append(j)  # I add the last index that satisfies the condition

                if len(points) > points_per_candidate:  # once the indexes have been acquired for a candidate,
                    # I add it to the list of candidates only if it has at least
                    # "points_per_candidate" points
                    # I save only the initial and final index of the candidate
                    candidates.append([points[0], points[-1] + 1])
                break

            points.append(j)  # when I don't enter the first "if", I add the consecutive decreasing points to "points"

    return candidates


def get_fit_candidate(candidate: list, shared_info: dict) -> np.polyfit:
    """
    The function performs the fit of valid candidate

    Parameters
    ---------

    candidate: list
            list of lists, which contain the indices of consecutive decreasing points

    shared_info: dict
            information that all processes need to read data into shared memory

    Returns
    ------

    np.polyfit
        array of coefficients m (angular coeff), q (intercept) of valid candidates

    """

    # reconstruct the candidate starting from its extremal indices
    candidate = list(range(candidate[0], candidate[-1]))

    # access information relating to the time_shm shared memory block
    t_name, t_size, t_shape, t_dtype = shared_info["time_shm"].values()

    # access information relating to the signals_clean_fabs_shm shared memory block
    sfabs_name, sfabs_size, sfabs_shape, sfabs_dtype = shared_info["signals_clean_fabs_shm"].values()

    # retrieve object that manages shared memory
    time_shm = SharedMemory(name=t_name)

    # create a generic array able to connect to the shared memory block
    time = np.ndarray(shape=t_shape, dtype=t_dtype, buffer=time_shm.buf)  # shm.buf: link to the shared block of memory

    # retrieve object that manages shared memory
    s_clean_fabs_shm = SharedMemory(name=sfabs_name)

    # create a generic array able to connect to the shared memory block
    s_clean_fabs = np.ndarray(shape=sfabs_shape, dtype=sfabs_dtype, buffer=s_clean_fabs_shm.buf)

    # retrieve standard deviation of the signal
    std = shared_info["std"]

    # array of coefficients m (angular coeff), q (intercept) of valid candidates
    fit = np.polyfit(x=time[candidate], y=np.log(s_clean_fabs[candidate]), deg=1, w=np.repeat(std, len(candidate)))

    return fit


def straight_line(p1, p2):
    """
    Straight line passing through two points

    Parameters
    ---------

    p1: list
        x,y of the point

    p2: list
        x, y of the point

    Returns
    ------

    lambda function
            straight line equation
    """

    x, y = 0, 1

    m = (p1[y] - p2[y]) / (p1[x] - p2[x])
    q = (p1[x] * p2[y] - p2[x] * p1[y]) / (p1[x] - p2[x])

    return lambda t: m * t + q


def is_vertical(points: np.ndarray):
    
    x = points[:, 0]  # Estrai le coordinate x dei punti
    y = points[:, 1]  # Estrai le coordinate y dei punti
    
    # Calcola la pendenza tra i punti
    slope = np.diff(y) / np.diff(x)
    
    # Verifica se la pendenza è infinita
    is_vertical = np.all(np.isinf(slope))
    
    return is_vertical


# +
def candidate_filter(candidate: list, shared_info: dict) -> bool:
    """
    Returns True or False depending on whether the candidate is valid or not

    Parameters
    ---------

    candidate: list
            a candidate

    shared_info: dict
            information that all processes need to read data into shared memory

    Returns
    ------

    bool:
        True if the candidate is valid
        False if the candidate is not valid
    """

    # reconstruct the candidate starting from its extremal indices
    candidate = list(range(candidate[0], candidate[-1]))

    # access information relating to the time_shm shared memory block
    t_name, t_size, t_shape, t_dtype = shared_info["time_shm"].values()

    # access information relating to the signals_clean_fabs_shm shared memory block
    sfabs_name, sfabs_size, sfabs_shape, sfabs_dtype = shared_info["signals_clean_fabs_shm"].values()

    # retrieve object that manages shared memory
    time_shm = SharedMemory(name=t_name)

    # create a generic array able to connect to the shared memory block
    time = np.ndarray(shape=t_shape, dtype=t_dtype, buffer=time_shm.buf)  # shm.buf: link to the shared block of memory

    # retrieve object that manages shared memory
    s_clean_fabs_shm = SharedMemory(name=sfabs_name)

    # create a generic array able to connect to the shared memory block
    s_clean_fabs = np.ndarray(shape=sfabs_shape, dtype=sfabs_dtype, buffer=s_clean_fabs_shm.buf)

    log_data = np.log(s_clean_fabs[candidate])

    # if the logarithm of the signal takes on the value infinite or nan, the candidate under examination is NOT valid
    if np.isinf(log_data).any() or np.isnan(log_data).any():
        return False
    
    # Given the candidate's first index, I consider the previous
    # "points_vertical_trend" indexes to verify that the signal follows a vertical growth

    # First starting point for verifying vertical growth.
    # It respects the edge conditions of the signal array
    first_vertical_point = max(0, candidate[0] - shared_info["points_vertical_trend"])
    
#     threshold = shared_info['coeff'] * shared_info["std"]
#     first_vertical_point = np.where(s_clean_fabs[first_vertical_point:candidate[0]] > threshold)[0]
    
#     if not first_vertical_point.shape[0]:
        
#         return False
    
#     first_vertical_point = first_vertical_point[0]

#     points = np.empty((shared_info["points_vertical_trend"] + 1, 2))
    
#     points[:, 0] = time[first_vertical_point:candidate[0] + 1]
#     points[:, 1] = s_clean_fabs[first_vertical_point:candidate[0] + 1]
    
#     if not is_vertical(points):
#         return False 

    # check that the "points_vertical_trend" points preceding the candidate are strictly increasing
    for index in range(first_vertical_point, candidate[0] - 1):

        if s_clean_fabs[index] > s_clean_fabs[index + 1]:
            return False

    # the points follow a vertical trend if the straight line connecting the first and last point
    # has an angular coefficient greater than 2 + np.sqrt(3) (theta greater than 75 degrees)

    delta_x = time[candidate[0]] - time[first_vertical_point]
    delta_y = s_clean_fabs[candidate[0]] - s_clean_fabs[first_vertical_point]

    # if the angle is less than 75 degrees, discard the candidate
    if delta_y < delta_x * (2 + np.sqrt(3)): 
        return False

    stop = len(candidate) // 2

    # I establish if a candidate is valid by generating lines connecting
    # point p1 (at index "index") of the candidate under examination with point p2 (at index "-index - 1")
    for index in range(stop):
        p1 = [time[candidate][index], s_clean_fabs[candidate][index]]
        p2 = [time[candidate][-index - 1], s_clean_fabs[candidate][-index - 1]]

        straight = straight_line(p1, p2)

        x1, y1 = time[candidate][index + 1], s_clean_fabs[candidate][index + 1]
        x2, y2 = time[candidate][- index - 2], s_clean_fabs[candidate][- index - 2]

        # For each point (p1) I take its mirror point (p2), connect them with a straight line
        # and check that the point after p1 and the point before p2 are below the straight line
        if straight(x1) - y1 < 0 or straight(x2) - y2 < 0:
            return False

    return True


# -

def multiprocess_filter(pool: mp.Pool, candidates: list, function: callable, args: tuple, chunksize: int) -> list:
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


def get_time_constant(candidate_fit: list) -> Union[float, None]:
    """
    The function calculates the time constants for each valid candidate

    Parameters
    ---------

    candidate_fit: list of np.ndarrays
                matrix of slope coefficients and intercepts of valid candidates of a TES

    Returns
    ------

    float:
        time constant
    """

    # return None if m is Nan
    if np.isnan(candidate_fit)[0]:
        return None

    return round(-1 / candidate_fit[0], 6)


def tau_filter(tau: float, tau_coeff: float, epsilon: float) -> bool:
    """
    The function filters the candidate's time constant based on a given condition

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


def get_taus(tot_tes: int) -> dict[int, list[float]]:
    """
    Return TESs time constants

    Parameters
    ----------

    tot_tes: int
            total number of TESs

    Returns
    ------

    dict:
        dictionary whose keys (str) are the number of TESs and whose values (list of float numbers) are
        a list containing the time constants of the related TES
    """

    all_taus_found = 0
    # n_candidates = dict()
    taus_per_tes = dict()
    # n_valid_candidates = dict()
    

    # mp.Pool is the manager of child processes
    with mp.Pool() as pool:

        # save in shared memory the first signal to be analyzed
        s_clean_fabs_shm = create_shared_memory(data=np.load(SHARED_INFO["signals_clean_fabs_fname"])['arr_0'],
                                                name="signals_clean_fabs_shm",
                                                shm_name="s_clean_fabs_shared")

        # access information relating to the signals_clean_fabs_shm shared memory block
        sfabs_name, sfabs_size, sfabs_shape, sfabs_dtype = SHARED_INFO["signals_clean_fabs_shm"].values()

        # create a generic array able to connect to the shared memory block
        s_clean_fabs = np.ndarray(shape=sfabs_shape, dtype=sfabs_dtype, buffer=s_clean_fabs_shm.buf)

        for n_tes in range(tot_tes):

            # for each TES, I look for candidates
            candidates = get_candidates()
            # if there are candidates
            if candidates:
                
                # n_candidates[f"{n_tes}"] = candidates
                chunksize = max(2, len(candidates) // mp.cpu_count())

                # select only valid candidates
                candidates = multiprocess_filter(pool=pool,
                                                 candidates=candidates,
                                                 function=candidate_filter,
                                                 args=zip(candidates, [SHARED_INFO] * len(candidates)),
                                                 chunksize=chunksize)

                # n_valid_candidates[f"arr_{n_tes}"] = candidates

                # fit every valid candidate
                fit_matrix = pool.starmap(get_fit_candidate,
                                          zip(candidates, [SHARED_INFO] * len(candidates)),
                                          chunksize)
            

                # save the taus of the candidates in the list
                taus = pool.map(get_time_constant, fit_matrix)

                if taus:
                    all_taus_found += len(taus)
                    taus_per_tes[n_tes] = taus
                    logger.info("found %s time constants related to TES n. %s", len(taus), n_tes)

                # # filter the candidate's time constant based on the time constant of the TES
                # taus_per_tes.append(multiprocess_filter(pool=pool,
                #                                          candidates=taus,
                #                                          func=tau_filter,
                #                                          args=zip(taus,
                #                                                   [SHARED_INFO['tau_coeff']] * len(taus),
                #                                                   [SHARED_INFO['epsilon']] * len(taus)),
                #                                          chunksize=chunksize))

            # load the new signal to be analyzed into the shared memory
            if n_tes + 1 < tot_tes:
                s_clean_fabs[:] = np.load(SHARED_INFO["signals_clean_fabs_fname"])[f"arr_{n_tes +  1}"]

            print(f"\rProgress: {(n_tes + 1) / tot_tes * 100:.2f}%", end="")
            
        # write_time_constants('candidates.json', n_candidates)
        # write_time_constants('valid_candidates.json', n_valid_candidates)
        logger.info("number of time constants for all TESs: %s", all_taus_found)

        return taus_per_tes


def released_share_memory():
    """
    It frees shared memory

    """

    for key in SHARED_INFO:

        #  check that the memory freeing operations are carried out only in the presence of shared memory portions
        if 'shm' in key and 'name' in SHARED_INFO[key]:
            # access the shared memory block
            shm = SharedMemory(name=SHARED_INFO[key]['name'])
            # close access to shared memory block for process calling .close()
            shm.close()
            # release shared memory
            shm.unlink()

            logger.info("released shared memory for '%s'", SHARED_INFO[key]['name'])


def plot_taus(n_tes: int = -1, taus: list = None, all_taus: dict = None):
    """
    It performs two types of plots:
    - the plots of the time constants of a single TES
    - the plot of all the time constants of all the TES

    To plot the time constants of a single TES, you need to pass the TES number and the list of time constants.
    To plot the time constants of all the TES, you need to pass only the dictionary that has the TES as keys
    and the lists containing the TES time constants as values.

    Parameters
    ----------

    n_tes: int
        number of the TES to plot its time constants

    taus: list
        list of time constants of a single TES to be plotted

    all_taus: dict
            dictionary containing TES as keys and time constants as values
    """

    # x_ticks = []

    # if no valid arguments are passed to the function, it does not plot anything
    if not n_tes and not taus and not all_taus:
        return

    fig, ax = plt.subplots(figsize=(15, 5))

    # Plot the time constants related to a single TES
    if n_tes >= 0 and taus:
        ax.scatter(list(range(len(taus))), taus)
        ax.set_title(f"Taus for TES {n_tes} - points per candidate >= {SHARED_INFO['points_per_candidate']}")
        ax.set_xlabel("Time constants index")
        ax.set_ylabel("Time constants [s]")
        plt.savefig(os.path.join(SHARED_INFO['plots_dir'], f"{n_tes}_taus_plot"))

    # I plot the time constants of all the TESs
    elif all_taus:

        for key, value in all_taus.items():
            ax.scatter([key] * len(value), value)

        ax.set_title(f"Taus for all TES - points per candidate >= {SHARED_INFO['points_per_candidate']}")
        ax.set_xlabel("n. TES")
        ax.set_ylabel("Time constants [s]")
        ax.set_yscale('log')
        plt.savefig(os.path.join(SHARED_INFO['plots_dir'], f"taus_for_all_tes"))

    # # Code relating to the plot of the filtered time constants
    # for row in taus:
    #     r = np.array(row[1])
    #     mask = r > SHARED_INFO["tau_coeff"]
    #
    #     # if there is at least one tau greater than 0.05 "any" returns True
    #     if any(mask):
    #         x_ticks.append(row[0])  # to x_ticks I append the number of the TES which has the tau > 0.05
    #         ax.scatter([row[0]] * r[mask].shape[0], r[mask])
    #     else:
    #         # plot the tau without printing the TES number as x_tick
    #         ax.scatter([row[0]] * len(r[~mask]), r[~mask], alpha=0.2)

    # ax.plot(range(taus[-1][0] + 1),
    #         [SHARED_INFO["tau_coeff"]] * (taus[-1][0] + 1),
    #         label=f"Threshold = {SHARED_INFO['tau_coeff']}")

    plt.close(fig)


def write_time_constants(fname: str, data: dict):
    """
    Writes time constants to a .json file

    Parameters
    ---------

    fname: str
        file name in which to save the time constants

    data: dict
        dictionary containing TES as keys and time constants as values
    """

    with open(fname, "w", encoding='utf8') as fout:
        json.dump(data, fout, indent=2)


def read_time_constants(fname: str) -> dict[str, list[float]]:
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


def find_cosmic_rays(data_storage_path: str,
                     data_set_path: str,
                     qubic_interpreter: str = sys.executable, ):
    """
    It searches for cosmic rays starting from the time constants

    Parameters
    ----------

    data_storage_path: str
        folder where two subfolders are created:
            1. input, which contains the .npy (time) and .npz (signals of all TES) files;
            2. output, which contains the .txt file of the taus;

    data_set_path: str
        folder containing sky scan data in .fits format

    qubic_interpreter: str
        interpreter for data export

    """

    start = tm.perf_counter()

    logger.info("configuration of all the necessary variables")

    # Configuration of the interpreter for data export and the saving point of the exported files (.npy and .npz format)
    tm_shm = configure_data_dir(data_storage_path=data_storage_path,
                                qubic_interpreter=qubic_interpreter,
                                data_set_path=data_set_path)

    logger.info("export clean signals from atmospheric drift")

    # export clean signals in module in .npz format
    tot_tes = save_interp_signals()

    logger.info("search for the time constants of all TESs")

    # dictionary whose keys are the number of TESs and whose values are
    # a list containing the time constants of the related TES
    taus_per_tes = get_taus(tot_tes=tot_tes)

    end = tm.perf_counter()
    execution_time = f"Time to process signals: {end - start:.2f}[s]"
    released_share_memory()

    logger.info(execution_time)

    print('\n' + execution_time)

    logger.info("write time constants to the file '%s'", SHARED_INFO['taus_fname'])

    write_time_constants(SHARED_INFO['taus_fname'], taus_per_tes)

    logger.info("save the time constant plots for single TES and the time constant plot of all TES")

    for n_tes in taus_per_tes.keys():
        plot_taus(n_tes=n_tes, taus=taus_per_tes[n_tes])

    plot_taus(n_tes=-1, taus=None, all_taus=taus_per_tes)

    logger.info("Script completed successfully")


# +
if __name__ == "__main__":
    
#     data_storage_path = "/Volumes/Data/qubic/data"  
#     qubic_interpreter = "/home/user/anaconda3/envs/qubic/bin/python"  # sys.executable
#     data_set_path = "/Volumes/Data/qubic/data/2022-07-14/2022-07-14_23.54.19__MoonScan_Speed_VE14"  
    
    data_storage_path = "/home/user/sofia_qubic_env/sofia_data"
    qubic_interpreter = "/home/user/anaconda3/envs/qubic/bin/python"  # sys.executable
    data_set_path = "/media/DataQubic/2023-04-18/2023-04-18_12.56.51__skydip1"

    try:
        find_cosmic_rays(data_storage_path=data_storage_path,
                         data_set_path=data_set_path,
                         qubic_interpreter=qubic_interpreter)
    except Exception as e:
        logger.error("exception while executing find_cosmic_rays: %s", e)
        released_share_memory()
