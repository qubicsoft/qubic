import os
import sys
import logging
from pathlib import Path
import multiprocessing as mp
from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager

import numpy as np
import scipy as sp
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.signal import ShortTimeFFT, detrend, medfilt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import matplotlib as mpl
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from qubicpack.qubicfp import qubicfp

from dataclasses import dataclass, field

# Set global Matplotlib rcParams for publication-ready figures
# (LaTeX rendering, serif fonts, consistent font sizes, and high-resolution PDF output)
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.facecolor": "whitesmoke",
    "savefig.dpi": 300,
    "savefig.format": "pdf"})


@dataclass()
class Dataset:
    """
    Container class holding all time-domain data and metadata required
    for the preprocessing and scan-based analysis of QUBIC Time Ordered Data (TOD).

    This dataclass centralizes:
      - the science TOD timeline and detector signals,
      - the housekeeping (HK) timelines and mount pointing information,
      - derived quantities such as scan segmentation, scan direction masks,
        interpolated pointing, and sampling properties.

    The Dataset object is designed to be progressively populated along the
    preprocessing pipeline:
      1. Raw TOD and HK data are loaded.
      2. Scanning frequency and scan sweeps (forth/back) are identified.
      3. Pointing timelines are interpolated onto the TOD sampling.
      4. Scan-dependent masks and indices are constructed and reused by
         all subsequent preprocessing steps.

    By separating data storage from processing logic, this class provides
    a clean interface between raw observations and the preprocessing
    algorithms implemented in the `PreProcess` class.
    """

    # Time axis of the science Time Ordered Data (TOD) [seconds]
    tm: np.ndarray
    # TOD signals array: shape (n_tes, n_samples)
    signals: np.ndarray
    # Time axis of the housekeeping (HK) data [seconds]
    tm_hk: np.ndarray
    # Elevation angle from mount telemetry [degrees]
    elevation: np.ndarray
    # Azimuth angle from mount telemetry [degrees]
    azimuth: np.ndarray
    # Estimated scanning frequency of the telescope [Hz]
    scan_freq: float = 0
    # Estimated global lag between HK timeline and TOD timeline [seconds]
    lag_dt: float = 0.
    # Total number of detected scan sweeps
    n_scans: int = 0
    # Angular velocity of the scan
    omega: np.ndarray = field(init=False)

    # Array encoding scan direction and scan ID:
    # +N -> forth scan number N
    # -N -> back scan number N
    #  0 -> dead time / turn-around region
    scantype: np.ndarray = field(init=False)
    # boolean mask (same length as tm) selecting samples belonging to back scans
    mask_back: np.ndarray = field(init=False)
    # boolean mask (same length as tm) selecting samples belonging to forth scans
    mask_forth: np.ndarray = field(init=False)
    # Elevation interpolated onto the TOD time axis
    elevation_interp: np.ndarray = field(init=False)
    # Azimuth interpolated onto the TOD time axis
    azimuth_interp: np.ndarray = field(init=False)
    # List of scan sweeps.
    # Each tuple contains:
    # (start_index of the scan, end_index of the scan, direction)
    # where direction = +1 (forth) or -1 (back)
    scans: list[tuple[int, int, int]] = field(default_factory=list)

    def __post_init__(self):
        """
        Post-initialization step.
        Allocate boolean masks for forth and back scans with the same length as the TOD
        """
        # Determine the number of time samples in the TOD:
        # if signals is 2D (n_tes, n_samples) take axis=1, otherwise (n_samples,) take axis=0
        n = self.signals.shape[1] if self.signals.ndim == 2 else self.signals.shape[0]
        self.mask_forth = np.zeros(n, dtype=bool)
        self.mask_back = np.zeros(n, dtype=bool)

    @property
    def dt(self):
        """
        Median sampling time of the TOD [seconds]
        """
        return np.median(np.diff(self.tm))

    @property
    def dt_hk(self):
        """
        Median sampling time of the housekeeping timeline [seconds]
        """
        return np.median(np.diff(self.tm_hk))

    @property
    def f_sampling(self):
        """
        Sampling frequency of the TOD [Hz]
        """
        return 1 / self.dt

    @property
    def f_sampling_hk(self):
        """
        Sampling frequency of the housekeeping data [Hz]
        """
        return 1 / self.dt_hk

def mode_histogram(x, bins=100):
    """
    Estimate the mode of a 1D signal using a histogram-based approach.

    The function builds a histogram of the input samples and returns the
    center of the bin with the highest number of entries. In Time Ordered
    Data (TOD) analysis, this provides a robust estimate of the baseline
    level, since noise-dominated values are typically more frequent than
    transient events or outliers.

    :param x: Input 1D signal (e.g. a TOD segment or scan)
    :type: array_like

    :param bins: Number of histogram bins used to estimate the mode.
        The value is cast to an integer.
    :type: int or float, optional

    :return mode_value: Estimated mode of the signal, defined as the center of the most populated histogram bin
    :rtype:: float
    """

    bins = int(bins)

    # Compute the histogram of the signal:
    # counts contains the number of samples in each bin,
    # bin_edges defines the bin boundaries
    counts, bin_edges = np.histogram(x, bins=bins)

    # Compute the center value of each histogram bin
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Identify the bin with the maximum number of entries
    # and return its center as the mode estimate
    return bin_centers[counts.argmax()]


class PreProcess:
    """
    Handles preprocessing of datasets, offering methods for data loading, processing, and analysis.

    The PreProcess class encapsulates functionality for handling datasets, reconstructing information,
    analyzing signals, and managing shared memory for multiprocessing. It offers methods to load data, find
    scans, and perform preprocessing tasks either in parallel (using multiprocessing) or in isolation.

    Attributes:
    -----------
        data: Input data file or path required for the preprocessing task.
        dataset: An optional Dataset object that holds the time, signal, housekeeping, azimuth, and elevation
                 data associated with the preprocessing task.
        output: Directory where preprocessed results, plots, and logs will be saved.
        logger: Logger object responsible for managing log entries throughout the preprocessing tasks.
    """

    @staticmethod
    def extract_chunk_modes(scans, signal, az, el, n_bins=20):
        """
        Compute per-chunk robust mode estimates along each scan (forth/back).

        For each scan (forth/back), the function splits the scan interval
        into `n_bins`. Each bin is defined as a chunk. For every chunk, it estimates the signal
        mode via `mode_histogram` and stores the corresponding median azimuth and
        elevation values.

        This is typically used to build a baseline model (e.g., mode vs az/el)
        that captures slowly-varying systematics correlated with pointing.

        :param scans: List of scan sweeps in the form (start_index, end_index, direction),
            where `direction` is +1 (forth) or -1 (back).
        :type: list[tuple[int, int, int]]

        :param signal: 1D TOD signal for a single TES (length n_samples)
        :type: np.ndarray

        :param az: Azimuth timeline sampled on the TOD axis (length n_samples)
        :type: np.ndarray

        :param el: Elevation timeline sampled on the TOD axis (length n_samples)
        :type: np.ndarray

        :param n_bins: Number of chunks per scan
        :type: int, optional

        :return:
        modes : np.ndarray
            Mode estimate for each chunk (same length as the number of valid chunks).
        az_medians : np.ndarray
            Median azimuth value for each chunk.
        el_medians : np.ndarray
            Median elevation value for each chunk.
        scan_dirs : np.ndarray
            Direction (+1 forth / -1 back) for each chunk.
        scan_ids : np.ndarray
            Scan index (0-based, enumerating the `scans` list) for each chunk.
        """

        modes, az_medians, el_medians, scan_dirs, scan_ids = [], [], [], [], []

        # Loop over scans (each element: start, end, direction)
        for idx, (s, e, dir_) in enumerate(scans):
            # Compute scan length in samples
            scan_len = e - s

            # Skip too-short scans (cannot be reliably split into n_bins chunks)
            if scan_len < n_bins:
                continue

            # Split the scan into n_bins chunks of (approximately) equal size
            for j in range(n_bins):
                # Start index of chunk j
                i0 = s + int(j * scan_len / n_bins)
                # End index of chunk j (exclusive)
                i1 = s + int((j + 1) * scan_len / n_bins)
                # Skip empty/degenerate chunks
                if i0 >= i1:
                    continue

                # Extract chunk data for signal and pointing
                chunk = signal[i0:i1]
                az_chunk = az[i0:i1]
                el_chunk = el[i0:i1]

                # skip the chunk if no samples are present
                if len(chunk) == 0:
                    continue

                # Estimate the mode of the chunk using a histogram peak.
                # Use an adaptive bin count capped at 51 for stability on small chunks.
                m = mode_histogram(chunk, bins=min(len(chunk) / 30, 51))

                # Store results for this chunk
                modes.append(m)
                az_medians.append(np.median(az_chunk))
                el_medians.append(np.median(el_chunk))
                # Direction of the parent scan for each chunk (+1 forth / -1 back)
                scan_dirs.append(dir_)
                # Identifier of the parent scan for each chunk (index in the scans list)
                scan_ids.append(idx)

        # Convert lists into numpy arrays
        return (
            np.array(modes),
            np.array(az_medians),
            np.array(el_medians),
            np.array(scan_dirs),
            np.array(scan_ids))

    @staticmethod
    def _preproc_worker(args):
        """
        Function executed in parallel for preprocessing a single TES signal.

        This method is designed to be called by `multiprocess_preproc` inside a
        multiprocessing pool. It reconstructs shared-memory arrays (time axis,
        housekeeping data, and pointing), builds a minimal Dataset instance for
        a single TES, and applies the full preprocessing chain independently.

        The function returns the cleaned TOD for the given TES index.

        :param args: Tuple containing the following arguments:
            tes_idx, shm_names, shapes, dtypes, scans, scantype, mask_forth, mask_back, signal, output, make_plot
        :type args: tuple
        """

        # Unpack arguments passed by the multiprocessing pool
        tes_idx, shm_names, shapes, dtypes, scans, scantype, mask_forth, mask_back, signal, output, make_plot = args

        # Reconstruct TOD time axis from shared memory
        shm_tm = shared_memory.SharedMemory(name=shm_names["tm"])
        tm = np.ndarray(shapes["tm"], dtype=dtypes["tm"], buffer=shm_tm.buf)

        # Reconstruct housekeeping time axis from shared memory
        shm_hk = shared_memory.SharedMemory(name=shm_names["hk"])
        tm_hk = np.ndarray(shapes["hk"], dtype=dtypes["hk"], buffer=shm_hk.buf)

        # Reconstruct azimuth timeline interpolated on TOD sampling
        shm_az = shared_memory.SharedMemory(name=shm_names["az"])
        azimuth = np.ndarray(shapes["az"], dtype=dtypes["az"], buffer=shm_az.buf)

        # Reconstruct elevation timeline interpolated on TOD sampling
        shm_el = shared_memory.SharedMemory(name=shm_names["el"])
        elevation = np.ndarray(shapes["el"], dtype=dtypes["el"], buffer=shm_el.buf)

        # Create a Dataset object containing only the information
        # needed for preprocessing a single TES
        dataset = Dataset(
            tm=tm,
            signals=signal,  # single TES TOD
            tm_hk=tm_hk,
            elevation=np.array([]),
            azimuth=np.array([]))

        # Attach interpolated pointing to the Dataset
        dataset.azimuth_interp = azimuth
        dataset.elevation_interp = elevation

        # Attach scan information and masks
        dataset.scans = scans
        dataset.scantype = scantype
        dataset.mask_forth = mask_forth
        dataset.mask_back = mask_back

        # Instantiate a PreProcess object using the worker-specific Dataset
        pp = PreProcess(None, dataset, output=output)

        # Remove per-scan median offset
        sig = pp.remove_offset_per_scan(tod=signal, method="median")

        # Subtract azimuth-elevation correlated baseline
        sig = pp.subtract_az_el_baseline(tod=sig, tes_idx=tes_idx, make_plot=True)

        # Remove residual per-scan offset using a robust mode estimate
        sig = pp.remove_offset_per_scan(tod=sig, method="mode")

        # Regularize dead-time regions by linear rescaling
        sig = pp.linear_rescale_chunks(sig, sz=1000)

        if make_plot:
            os.makedirs(os.path.join(output, "interactive"), exist_ok=True)
            # Interactive 3D plot of mode vs azimuth and elevation
            pp.plot_mode_az_el_3d_interactive(
                tod=sig,
                tes_idx=tes_idx,
                output_html=os.path.join(pp.output, "interactive", f"mode_az_el_3d_interactive_{tes_idx}.html"),
            )

            # Static diagnostic plot of quadratic baseline fit
            pp.plot_quadratic_fit(tod=sig, tes_idx=tes_idx)

        # Return TES index and cleaned TOD
        return tes_idx, sig

    def multiprocess_preproc(self,
                             tes_list: list[int],
                             kernel_size: int = 101,
                             make_plot: bool = False,
                             n_bins=20):

        """
        Run the full preprocessing pipeline in parallel over multiple TES channels.

        This method:
          1. Estimates the telescope scanning frequency.
          2. Identifies scan sweeps (forth/back) on the dataset.
          3. Shares common timelines (time axis, HK time, azimuth, elevation)
             across processes using shared memory.
          4. Spawns a multiprocessing pool where each worker preprocesses
             one TES independently.
          5. Collects and returns the cleaned TODs for all requested TES.

        :param tes_list: List of TES indices to preprocess in parallel
        :type: list[int]

        :param kernel_size: Kernel size used for median filtering when identifying scans
        :type: int, optional

        :param make_plot: If True, generate diagnostic plots inside each worker
        :type: bool, optional

        :param n_bins: Number of chunks per scan (used downstream in baseline subtraction).
        :type: int, optional

        :return cleaned: Dictionary mapping TES index (as string) to the cleaned TOD array
        :rtype: dict
        """

        # Estimate the scanning frequency from azimuth data
        self.get_scanning_frequency(method="spectrogram", make_plots=make_plot)

        # Identify scan sweeps (forth/back) on the TOD timeline
        self.find_scans(kernel_size=kernel_size, delta_t=self.dataset.lag_dt, make_plot=make_plot)

        # Extract full TOD array and number of TES
        signals = self.dataset.signals
        n_tes = signals.shape[0]

        # SharedMemoryManager ensures automatic cleanup of shared buffers
        with SharedMemoryManager() as smm:
            shm_handles = {}
            shared_arrays = {}

            # Arrays that are common to all TES
            arrays = {
                "tm": self.dataset.tm,
                "hk": self.dataset.tm_hk,
                "az": self.dataset.azimuth_interp,
                "el": self.dataset.elevation_interp
            }
            shapes = {}
            dtypes = {}

            # Allocate shared memory buffers and copy data into them
            for key, arr in arrays.items():
                shm = smm.SharedMemory(size=arr.nbytes)
                shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
                shared_arr[:] = arr[:]
                shm_handles[key] = shm
                shared_arrays[key] = shared_arr
                shapes[key] = arr.shape
                dtypes[key] = arr.dtype

            # Scan-related information (read-only, no need for shared memory)
            scans = self.dataset.scans
            scantype = self.dataset.scantype
            mask_forth = self.dataset.mask_forth
            mask_back = self.dataset.mask_back

            # Pass only shared memory names to workers
            shm_names = {key: shm_handles[key].name for key in shm_handles}

            # Build argument list for workers
            args_list = []
            for i, tes_, in enumerate(tes_list):
                args = (
                    int(tes_),
                    shm_names,
                    shapes,
                    dtypes,
                    scans,
                    scantype,
                    mask_forth,
                    mask_back,
                    signals[i],
                    self.output,
                    make_plot
                )

                args_list.append(args)

            # Create a multiprocessing pool and dispatch workers
            with mp.Pool() as pool:
                results = pool.map(PreProcess._preproc_worker, args_list)

            # Collect cleaned TODs into a dictionary indexed by TES id
            cleaned = {str(tes_idx): sig for tes_idx, sig in results}
            return cleaned

    def __init__(self, data, dataset: Dataset = None, output="pre_processed_result", logger=None):

        # Store reference to input data (not loaded yet)
        self.data = data
        # Store output directory path
        self.output = output

        os.makedirs(self.output, exist_ok=True)

        # If no external logger is provided, create a default one
        if logger is None:

            # Create a logger associated with this module
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)

            # Create a console handler to log messages to stdout
            console_handler = logging.StreamHandler(stream=sys.stdout)
            console_handler.setLevel(logging.INFO)

            # Define a consistent log message format with timestamps
            formatter = logging.Formatter(
                fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)

            # Attach console handler to the logger
            self.logger.addHandler(console_handler)

            # Create a file handler to save logs inside the output directory
            file_path = os.path.join(self.output, 'preprocessing.log')
            file_handler = logging.FileHandler(file_path, mode='w')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            # Attach file handler to the logger
            self.logger.addHandler(file_handler)

        else:
            # Use the externally provided logger
            self.logger = logger

        # Attach the Dataset object (may be None at initialization)
        self.dataset = dataset

    def load(self):
        """
        Load raw QUBIC data and initialize the Dataset container.

        This method reads a QUBICStudio dataset using `qubicpack`, extracts
        the science Time Ordered Data (TOD), the housekeeping (HK) timeline,
        and the mount pointing information (azimuth and elevation).
        All time axes are shifted to start at zero.

        The loaded data are stored in a `Dataset` object attached to
        `self.dataset`. No preprocessing is performed at this stage.
        """

        # Create a qubicpack interface object
        qubic = qubicfp()
        # Disable verbose output from qubicpack
        qubic.verbosity = 0
        # Read the QUBICStudio dataset from the provided input path
        qubic.read_qubicstudio_dataset(self.data)

        # Log the data source for traceability
        self.logger.info('Loading data from %s', self.data)

        # Load science TOD:
        # tm -> time axis of the TOD
        # signals -> detector signals (shape: n_tes, n_samples)
        tm, signals = qubic.tod()

        # Shift TOD time axis to start at t = 0 s
        tm -= tm[0]
        # Load housekeeping (HK) time axis and shift it to start at 0 s
        tm_hk = qubic.timeaxis(datatype='hk')
        tm_hk -= tm_hk[0]

        # Load mount pointing information from housekeeping telemetry
        azimuth = qubic.azimuth()
        elevation = qubic.elevation()

        # Create the Dataset object holding TOD, HK timeline, and pointing
        self.dataset = Dataset(tm, signals, tm_hk, elevation, azimuth)

        # signals must be a 2D array (n_tes, n_samples)
        assert self.dataset.signals.ndim == 2, "signals must be 2D (ntes, nsamples)"

    @staticmethod
    def spectrogram(sig_time: np.ndarray,
                    signal: np.ndarray,
                    f_sampling=None,
                    nperseg=None,
                    window=None,
                    nfft=None,
                    noverlap=None,
                    eps=1e-12,
                    make_plots: bool = False,
                    plots_dir: str = '.'):

        """
        Compute the spectrogram and power spectral density (PSD) of a signal
        using a Short-Time Fourier Transform (STFT).

        This method is primarily used to estimate the dominant (scanning)
        frequency of the telescope motion from azimuth or similar timelines.

        :param sig_time: Time axis of the signal [seconds]
        :type: np.ndarray

        :param signal: 1D signal array (e.g., azimuth timeline)
        :type: np.ndarray

        :param f_sampling: Sampling frequency [Hz]. If None, it is estimated from sig_time
        :type: float, optional

        :param nperseg: Number of samples per STFT segment
        :type: int, optional

        :param window: Window function used in the STFT
        :type: str or tuple, optional

        :param nfft: Number of FFT points
        :type: int, optional

        :param noverlap: Number of overlapping samples between segments
        :type: int, optional

        :param eps: Small constant added to avoid log(0) when converting to dB
        :type: float, optional

        :param make_plots: If True, generate diagnostic spectrogram and PSD plots
        :type: bool, optional

        :param plots_dir: Directory where plots are saved
        :type: str, optional

        :return:
        f : np.ndarray
            Frequency array [Hz]

        Zxx_db : np.ndarray
            Spectrogram in dB (frequency × time)

        peak_frequency : float
            Dominant frequency estimated from the averaged PSD
        """

        # Estimate sampling frequency if not provided
        if f_sampling is None:
            f_sampling = 1 / (np.median(np.diff(sig_time)))

        # Estimate segment length if not provided
        if nperseg is None:
            nperseg = sp.fft.next_fast_len(len(signal) // 8)

        # ShortTimeFFT (especially in `fft_mode='centered'`) requires enough samples
        # relative to the FFT length. If the dataset is short, we must cap nperseg/nfft.
        n_samples = len(signal)

        # Ensure nperseg is valid and not larger than the signal
        nperseg = int(max(8, nperseg))
        if nperseg > n_samples:
            # Use at most half the available samples (but keep a minimum for stability)
            nperseg = int(max(8, n_samples // 2))
            nperseg = sp.fft.next_fast_len(nperseg)
            # next_fast_len may overshoot; cap again
            nperseg = int(min(nperseg, n_samples))

        # Default window for STFT
        if window is None:
            window = 'nuttall'

        # Estimate FFT length if not provided
        if nfft is None:
            nfft = 2 ** int(np.ceil(np.log2(nperseg)))

        nfft = int(nfft)

        # For centered FFT in ShortTimeFFT, we must have roughly: n_samples >= ceil(nfft/2)
        # i.e. nfft <= 2 * n_samples. Cap nfft accordingly.
        max_nfft = int(2 * n_samples)
        if nfft > max_nfft:
            nfft = sp.fft.next_fast_len(max_nfft)
            nfft = int(min(nfft, max_nfft))

        # Ensure FFT length is not smaller than segment length
        if nfft < nperseg:
            nfft = sp.fft.next_fast_len(nperseg)
            nfft = int(max(nfft, nperseg))

        # Set default overlap to 50% of segment length
        if noverlap is None:
            noverlap = int(0.50 * nperseg)

        # Safety: noverlap must be < nperseg
        noverlap = int(min(max(0, noverlap), nperseg - 1))

        # Create a ShortTimeFFT object defining the STFT configuration
        SFT = ShortTimeFFT.from_window(window,
                                       f_sampling,
                                       nperseg,
                                       fft_mode='centered',
                                       mfft=nfft,
                                       noverlap=noverlap,
                                       scale_to='psd',
                                       phase_shift=None)

        # Compute the short-time Fourier transform
        # Resulting array shape: (frequencies, time segments)
        Sz = SFT.stft(signal)
        # Frequency axis corresponding to the STFT
        f = SFT.f
        # Compute spectrogram power in linear scale
        Sxx = np.abs(Sz) ** 2
        # Convert spectrogram to decibel scale
        Zxx_db = 10 * np.log10(Sxx + eps)

        # Compute the power spectral density (PSD)
        # Average over time in linear scale
        psd_lin = Sxx.mean(axis=1)
        # psd_db = 10 * np.log10(psd_lin + eps)

        # Restrict to positive frequencies and find the dominant peak
        f_positive = f > 0
        peak_frequency = f[f_positive][np.argmax(psd_lin[f_positive])]

        if make_plots:
            # Create figure with spectrogram and PSD
            fig, (ax_sxx, ax_psd) = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), tight_layout=True)
            ax_sxx.set(xlabel=rf"$Time [s] - \Delta t = {SFT.delta_t:.4g} s$".replace(' ', '~'),
                       ylabel=rf"$Frequency [Hz] - \Delta f = {SFT.delta_f:.4g} Hz$".replace(' ', '~'))

            extent = SFT.extent(len(signal), center_bins=True)
            vmin, vmax = np.percentile(Zxx_db, [5, 95])
            kw = dict(origin='lower', aspect='auto', cmap='magma', vmin=vmin, vmax=vmax)
            # Plot spectrogram
            im = ax_sxx.imshow(Zxx_db, extent=extent, **kw)
            fig.colorbar(im, ax=ax_sxx, label=r'$PSD ~ [dB/Hz]$')

            # Plot averaged PSD
            ax_psd.set(xlabel=r'$Frequency ~ [Hz]$', ylabel=r'$Average ~ PSD [dB/Hz]$', facecolor='whitesmoke')
            ax_psd.plot(f, psd_lin, label=rf"$Peak frequency: {peak_frequency:.4g} Hz$".replace(' ', '~'))
            ax_psd.legend()
            ax_psd.grid(True)

            fig.savefig(os.path.join(plots_dir, 'azimuth_spectrogram'))

        return f, Zxx_db, peak_frequency

    def get_scanning_frequency(self, method: str = 'sinusoidal fit', make_plots: bool = False):
        """
        Estimate the telescope scanning frequency from the azimuth timeline.

        The scanning frequency is first roughly estimated in Fourier space from
        the azimuth signal after detrending. This initial guess is then refined
        using one of two methods:
          - 'sinusoidal fit': fit a sine wave to the azimuth timeline,
          - 'spectrogram': identify the dominant frequency from a STFT-based PSD.

        The estimated scanning frequency is stored in `self.dataset.scan_freq`.

        :param method: Method used to refine the scanning frequency estimate.
            Allowed values are 'sinusoidal fit' and 'spectrogram'.
        :type: str, optional

        :param make_plots: If True, generate diagnostic plots related to the selected method.
        :type: bool, optional
        """

        # Log the selected estimation method
        self.logger.info("Method: %s", method)

        # Step 1: detrend azimuth timeline
        # Remove slow drifts from azimuth to suppress low-frequency noise
        azimuth = detrend(self.dataset.azimuth)

        # Step 2: rough frequency estimate from FFT
        # Compute Fourier transform of the detrended azimuth
        fft_azimuth = sp.fft.fft(azimuth)
        # Frequency axis corresponding to the FFT (HK sampling)
        f_azimuth = sp.fft.fftfreq(len(self.dataset.tm_hk), d=self.dataset.dt_hk)
        # Identify the dominant frequency peak (excluding DC component, the one with frequency 0 Hz)
        scanning_frequency_guess = abs(f_azimuth[np.argmax(np.abs(fft_azimuth[1:])) + 1])

        # Log the initial frequency guess
        self.logger.info("Guess Scanning Frequency = %s", scanning_frequency_guess)

        # Step 3: refine estimate according to the selected method
        if method == 'sinusoidal fit':

            # Define sinusoidal model for azimuth motion
            def sin_model(t: np.ndarray, a, omega, phi) -> np.ndarray:
                return a * np.sin(omega * t + phi)

            # Initial parameter guesses for the fit
            amplitude_guess = (azimuth.max() - azimuth.min()) / 2
            omega_guess = 2 * np.pi * scanning_frequency_guess
            phi_guess = 0
            p0 = [amplitude_guess, omega_guess, phi_guess]

            # Perform non-linear least squares fit
            popt, _ = curve_fit(sin_model, self.dataset.tm_hk, azimuth, p0=p0)
            # Extract scanning frequency from the fitted angular frequency
            self.dataset.scan_freq = popt[1] / (2 * np.pi)

            if make_plots:
                # Compute normalized residuals of the fit
                residuals = (sin_model(self.dataset.tm_hk, *popt) - azimuth)
                residuals /= azimuth.std()
                # Fit residuals with a Gaussian distribution
                params = norm.fit(residuals)
                x = np.linspace(residuals.min(), residuals.max(), 1000)
                res_pdf = norm.pdf(x, *params)

                # Diagnostic plots: fit and residual distribution
                fig, (ax_sin, ax_hist) = plt.subplots(nrows=2, ncols=1, figsize=(14, 6), tight_layout=True)
                fig.suptitle(r'$Scanning frequency estimation: sinusoidal fit$'.replace(' ', '~'))

                # Azimuth timeline with fitted frequency
                ax_sin.set(xlabel=r'$time ~ [s]$', ylabel=r'$azimuth ~ [deg]$')
                ax_sin.plot(self.dataset.tm_hk, azimuth,
                            label=rf'$Scanning ~ frequency: {self.dataset.scan_freq:.4g} ~ Hz$')

                # Histogram of residuals with Gaussian fit
                ax_hist.set(xlabel=r'$Residuals$', title=r'$Gaussian Residuals fit$'.replace(' ', '~'))
                ax_hist.hist(residuals, bins='auto', density=True, color="black", histtype="step")
                ax_hist.plot(x, res_pdf, 'r-', label=rf'$ {params[0]:.2g} \pm {params[1]:.2g}$')
                ax_hist.legend()
                ax_hist.grid()

                fig.savefig(os.path.join(self.output, 'azimuth_sin_fit.png'), dpi=600)

        elif method == "spectrogram":

            # Estimate a suitable STFT segment length:
            # ~3 scan periods per segment
            nperseg = round(3 / (self.dataset.dt_hk * scanning_frequency_guess))

            # Cap nperseg for short HK timelines (ShortTimeFFT constraints)
            nperseg = int(max(8, nperseg))
            nperseg = int(min(nperseg, max(8, len(azimuth) // 2)))
            nperseg = int(sp.fft.next_fast_len(nperseg))
            nperseg = int(min(nperseg, len(azimuth)))

            # Refine scanning frequency using the spectrogram-based method
            _, _, self.dataset.scan_freq = self.spectrogram(
                self.dataset.tm_hk,
                azimuth,
                nperseg=nperseg,
                plots_dir=self.output,
                make_plots=make_plots,
            )

        # Log final scanning frequency estimate
        self.logger.info("Scanning Frequency = %s", self.dataset.scan_freq)

    def find_scans(self,
                   min_speed: float | None = 0.1,
                   kernel_size: int | None = 101,
                   delta_t: float = 0.,
                   make_plot: bool = False,
                   n_plot_scans: int = 10):
        """
        Identify telescope scans (forth/back) and build scan masks on the TOD timeline.

        The method uses the mount azimuth housekeeping (HK) timeline to estimate the
        angular velocity, smooth it with a median filter, and classify each HK sample as:
          - +1 : forth scan (positive angular velocity),
          - -1 : back scan (negative angular velocity),
          -  0 : dead-time / turn-around (speed below `min_speed`).

        It then numbers each forth scan by detecting 0 -> +1 transitions and applying
        a cumulative sum. The signed scan number is assigned to the TOD sampling by
        mapping each TOD time to the most recent HK sample (via `searchsorted`),
        optionally applying a global lag correction `delta_t`.

        Finally, the method:
          - builds `self.dataset.scans` as a list of contiguous (start, stop, direction) blocks,
          - builds boolean masks for forth/back samples on the TOD timeline.
          - interpolates azimuth/elevation onto the TOD timeline.


        :param min_speed: Minimum absolute angular velocity (deg/s) is used to classify a sample
            as part of a scan. Values below this are considered turn-around / dead time.
        :type: float, optional

        :param kernel_size: Window size for the median filter applied to the angular velocity.
            Must be an odd integer for `scipy.signal.medfilt`.
        :type: int, optional

        :param delta_t: Global lag between HK and TOD timelines [seconds]. TOD time is shifted by
            `-delta_t` before assigning scan types and interpolating pointing.
        :type: float, optional

        :param make_plot: If True, produce diagnostic plots (optional).
        :type: bool, optional

        :param n_plot_scans: Maximum number of scans to visualize in diagnostic plots (if enabled).
        :type: int, optional
        """

        # Clear any previously detected scans
        self.dataset.scans.clear()

        # Step 1: Prepare HK timeline and pointing (HK sampling)
        # Define an HK time axis starting at 0 s (numerical convenience)
        thk = self.dataset.tm_hk - self.dataset.tm_hk[0]
        # Read raw azimuth and elevation from housekeeping telemetry
        az = self.dataset.azimuth
        el = self.dataset.elevation

        # Step 2: Compute angular velocity and smooth it
        # Estimate an HK sampling time step
        dt_hk = np.median(np.diff(thk))
        # Compute azimuth angular velocity (deg/s) on the HK timeline
        omega = np.gradient(az, dt_hk)
        # Median-filter the velocity to suppress spikes and high-frequency fluctuations
        # kernel_size set to 101 (value taken from jupyter JC:
        # qubic/scripts/users/DiversJC/CalibSalta/Test-SB-Salta.ipynb
        omega_smooth = medfilt(omega, kernel_size)

        # Step 3: Classify each HK sample: dead time / forth / back
        # Dead-time / turn-around: speed too small to classify a direction reliably
        c0 = np.abs(omega_smooth) < min_speed
        # Forth scan: positive angular velocity (excluding dead time)
        cpos = (~c0) & (omega_smooth > 0)
        # Back scan: negative angular velocity (excluding dead time)
        cneg = (~c0) & (omega_smooth < 0)

        # Step 4: Build a sign array on HK sampling: 0, +1, -1
        # Initialize a sign array: 0 = dead time (angular velocity of about 0.1 deg/s), +1 forth, -1 back
        sign0 = np.zeros_like(omega_smooth, dtype=int)
        # Assign +1 where the velocity is positive and above the threshold
        sign0[cpos] = 1
        # Assign -1 where the velocity is negative and below the threshold
        sign0[cneg] = -1

        # Step 5: Number forth scans: detect 0 -> +1 transitions and cumulative-sum them
        # Previous-sample sign (prepend a 0)
        prev = np.concatenate(([0], sign0[:-1]))
        # Start of a forth sweep occurs when the sign goes from 0 to +1
        starts = (sign0 == 1) & (prev == 0)
        # Each start increments the scan counter; constant within a sweep (forth + back).
        # From scan_sums we get the number of sweeps
        scan_nums = np.cumsum(starts)


        # Step 6: Build the signed scan-number array on HK: (+N for forth, -N for back, 0 for dead)
        # Multiply direction sign by scan number to encode direction + scan index
        scantype_hk = sign0 * scan_nums

        # Step 7: Assign scan types to TOD sampling (HK -> TOD mapping)
        # Shift TOD time axis to account for a global HK -> TOD lag
        tm_shifted = self.dataset.tm - delta_t
        # For each TOD sample, find the index of the last HK time <= tm_shifted
        idxs = np.searchsorted(thk, tm_shifted, side='right') - 1
        # Clip indices to valid range (protect edges)
        idxs = np.clip(idxs, 0, scantype_hk.size - 1)
        # Map HK scan classification onto TOD sampling
        scantype = scantype_hk[idxs]

        # Step 8: Interpolate azimuth/elevation onto TOD sampling
        # Interpolate azimuth at TOD times (with lag correction)
        self.dataset.azimuth_interp = np.interp(self.dataset.tm - delta_t, thk, az)
        # Interpolate elevation at TOD times (with lag correction)
        self.dataset.elevation_interp = np.interp(self.dataset.tm - delta_t, thk, el)

        # Step 9: Extract contiguous TOD blocks for each scan index (forth and back)
        # Maximum scan number encountered (absolute value, because back scans are negative)
        max_n = int(np.max(np.abs(scantype)))

        # Loop over scan numbers 1...max_n
        for n in range(1, max_n + 1):

            # Forth sweeps: scantype == +n
            # Indices of TOD samples belonging to forth scan n
            idxs = np.where(scantype == n)[0]

            if idxs.size:
                # Split into contiguous blocks (break where indices are not consecutive)
                splits = np.where(np.diff(idxs) != 1)[0] + 1

                # Each contiguous block is a single scan segment
                for blk in np.split(idxs, splits):
                    # Append (start, stop, direction) with stop being exclusive
                    self.dataset.scans.append((blk[0], blk[-1] + 1, 1))

            # Back sweeps: scantype == -n
            # Indices of TOD samples belonging to back scan n
            idxs = np.where(scantype == -n)[0]

            if idxs.size:
                # Split into contiguous blocks
                splits = np.where(np.diff(idxs) != 1)[0] + 1

                # Each contiguous block is a single scan segment
                for blk in np.split(idxs, splits):
                    self.dataset.scans.append((blk[0], blk[-1] + 1, -1))

        # Step 10: Build boolean masks on TOD sampling for forth/back scans
        # Number of TOD samples
        n = len(self.dataset.tm)

        # Initialize masks
        self.dataset.mask_forth = np.zeros(n, dtype=bool)
        self.dataset.mask_back = np.zeros(n, dtype=bool)

        # Mark each scan interval into the appropriate mask
        for start, stop, direction in self.dataset.scans:
            m = self.dataset.mask_forth if direction == 1 else self.dataset.mask_back
            m[start:stop] = True

        # Store scan-type label per TOD sample
        self.dataset.scantype = scantype
        # Log how many sweep segments were found
        self.logger.info("Found %d scan sweeps", len(self.dataset.scans))

    # TODO: to implement in the new preprocessing version
    # def estimate_mount_lag(self,
    #                        tes_idx: int | None = 32,
    #                        kernel_size: int | None = None,
    #                        inplace: bool = True):
    #     """
    #     Cross–correlate smoothed azimuth and TES signals to estimate
    #     the global lag Deltat between the mount telemetry and the TOD.
    #
    #     Parameters
    #     ----------
    #     tes_idx : int
    #         Index of the TES channel to use (default 0).
    #
    #     Returns
    #     -------
    #     delta_t : float
    #         Lag in seconds (positive ⇒ BACK template is delayed).
    #     """
    #
    #     if not self.dataset.scans:
    #         self.find_scans()
    #
    #     signal = self.dataset.signals[tes_idx] if inplace else self.dataset.signals[tes_idx].copy()
    #
    #     signal[self.dataset.mask_forth] -= np.median(signal[self.dataset.mask_forth])
    #     signal[self.dataset.mask_back] -= np.median(signal[self.dataset.mask_back])
    #
    #     forth_segs, back_segs = [], []
    #     for start, stop, direction in self.dataset.scans:
    #         segment = signal[start:stop]
    #
    #         if direction == 1:
    #             forth_segs.append(segment)
    #         else:
    #             back_segs.append(segment[::-1])  # reverse BACK sweep
    #
    #     if not forth_segs or not back_segs:
    #         self.logger.error("Not enough sweeps collected for lag estimation.")
    #         return 0.0
    #
    #     # Align lengths
    #     L = min(min(map(len, forth_segs)), min(map(len, back_segs)))
    #     forth_stack = np.vstack([s[:L] for s in forth_segs])
    #     back_stack = np.vstack([s[:L] for s in back_segs])
    #
    #     n_fft = 2 * L - 1
    #
    #     forth_fft = sp.fft.fft(forth_stack, n=n_fft, axis=1)
    #     back_fft = sp.fft.fft(back_stack, n=n_fft, axis=1)
    #
    #     corr_fft = forth_fft * back_fft.conj()
    #     correlations = sp.fft.ifft(corr_fft, axis=1).real
    #
    #     # find lags for all segments simultaneously
    #     lag_indices = np.argmax(correlations, axis=1)
    #     lag_samples = lag_indices - (L - 1)
    #
    #     self.dataset.lag_dt = deltaT = np.median(lag_samples * self.dataset.dt)
    #
    #     self.logger.info("Estimated mount lag Delta T = %.4f s (median over %d pairs)", deltaT, len(forth_stack))
    #
    #     return deltaT

    # def get_bins_extrema(self, fwhm: float = 0.68, multiplier: float = 1):
    #
    #     if not self.dataset.scans:
    #         self.find_scans()
    #
    #     forth_len, back_len = [], []
    #     forth_omega, back_omega = [], []
    #     forth_sec, back_sec = [], []
    #
    #     for idx, (start, stop, direction) in enumerate(self.dataset.scans):
    #         len_, omega_, sec_ = (forth_len, forth_omega, forth_sec) if direction == 1 else (back_len, back_omega,
    #                                                                                          back_sec)
    #
    #         len_.append(stop - start)
    #         # TODO: questo non funzionare' perche' omega ha dimensione pari a t_nk
    #         omega_.append(np.median(self.dataset.omega[start:stop]))
    #         sec_.append(1. / np.cos(np.radians(np.median(self.dataset.elevation[start:stop]))))
    #
    #     forth_len = np.array(forth_len)
    #     forth_sec = np.array(forth_sec)
    #     forth_omega = np.array(forth_omega)
    #
    #     back_len = np.array(back_len)
    #     back_sec = np.array(back_sec)
    #     back_omega = np.array(back_omega)
    #
    #     self.logger.info("multiplier: %s | f_sampling: %.2f | forth_omega %.2f | forth_sec %.2f",
    #                      multiplier, self.dataset.f_sampling, np.median(forth_omega), np.median(forth_sec))
    #
    #     Nbeam_forth = multiplier * self.dataset.f_sampling * fwhm / forth_omega * forth_sec
    #     Nbeam_back = multiplier * self.dataset.f_sampling * fwhm / back_omega * back_sec
    #
    #     N = len(self.dataset.signals[0])
    #     n_fft = sp.fft.next_fast_len(N)
    #
    #     signals = self.dataset.signals.copy()
    #     signals -= np.median(signals, axis=1, keepdims=True)
    #     signals_fft = sp.fft.fft(signals, n=n_fft, axis=1)
    #
    #     corr_fft = signals_fft * signals_fft.conj()
    #     auto_corr = sp.fft.ifft(corr_fft, axis=1).real
    #     auto_corr = auto_corr[:, :N]
    #     auto_corr /= auto_corr[:, [0]]
    #
    #     # Trova il primo lag < 1/e per ogni TES
    #     threshold = 1 / np.e
    #     below = (auto_corr < threshold)
    #     Nnoises = np.argmax(below, axis=1)
    #
    #     self.logger.info("Forth: Nsamples: %s | Noises: %s | Nbeams: %s",
    #                      np.median(forth_len), np.median(Nnoises), np.median(Nbeam_forth))
    #
    #     self.logger.info("Back: Nsamples: %s | Noises: %s | Nbeams: %s",
    #                      np.median(back_len), np.median(Nnoises), np.median(Nbeam_back))
    #
    #     return Nbeam_back, Nbeam_forth, Nnoises, forth_len, back_len

    def subtract_az_el_baseline(self,
                                tod: np.ndarray,
                                tes_idx=0,
                                n_bins=20,
                                poly_deg=2,
                                make_plot=False,
                                inplace=True):
        """
        Subtract a 2D azimuth–elevation correlated baseline from the TOD.

        The baseline is estimated by fitting a 2D polynomial (azimuth, elevation)
        to robust per-chunk signal modes, separately for forth and back scans.
        The fitted baseline is then evaluated chunk-by-chunk and subtracted
        from the TOD.

        :param tod: Input Time Ordered Data (TOD) for a single TES
        :type: np.ndarray

        :param tes_idx: TES index used for diagnostics and plotting
        :type: int, optional

        :param n_bins: Number of chunks per scan used to estimate the baseline
        :type: int, optional

        :param poly_deg: Degree of the 2D polynomial used in the azimuth–elevation fit
        :type: int, optional

        :param make_plot: If True, generate diagnostic plots of the fit
        :type: bool, optional

        :param inplace: If True, modify the input TOD array in place.
            If False, operate on a copy.
        :type: bool, optional

        :return signal:  TOD with the azimuth–elevation correlated baseline subtracted
        :rtype: np.ndarray
        """

        # Work in place or on a copy of the input TOD, depending on the flag
        signal = tod if inplace else tod.copy()
        # Retrieve azimuth and elevation timelines interpolated onto the TOD sampling
        az = self.dataset.azimuth_interp
        el = self.dataset.elevation_interp

        # Perform a global 2D polynomial fit of signal mode versus (azimuth, elevation),
        # separately for forth and back scans, using robust per-chunk mode estimates
        poly, reg_forth, reg_back = self.fit_mode_az_el_2d(tes_idx=tes_idx, n_bins=n_bins, poly_deg=poly_deg,
                                                           make_plot=make_plot, tod=tod)

        # Loop over all detected scan sweeps
        for idx, (s, e, dir_) in enumerate(self.dataset.scans):

            # Length of the current scan in samples
            scan_len = e - s

            # Skip scans that are too short to be reliably split into n_bins chunks
            if scan_len < n_bins:
                continue

            # Loop over chunks within the current scan
            for j in range(n_bins):
                # Starting index of chunk j
                i0 = s + int(j * scan_len / n_bins)
                # Ending index (exclusive) of chunk j
                i1 = s + int((j + 1) * scan_len / n_bins)

                # Skip degenerate or empty chunks
                if i0 >= i1:
                    continue

                # Extract azimuth and elevation values for the current chunk
                az_chunk = az[i0:i1]
                el_chunk = el[i0:i1]

                # Build the design matrix for the 2D polynomial model
                # Each row corresponds to one (azimuth, elevation) sample
                X = np.column_stack([az_chunk, el_chunk])

                # Evaluate the fitted baseline for forth scans
                if dir_ == 1 and reg_forth is not None:
                    baseline = reg_forth.predict(poly.transform(X))

                # Evaluate the fitted baseline for back scans
                elif dir_ == -1 and reg_back is not None:
                    baseline = reg_back.predict(poly.transform(X))

                # If no valid fit is available, assume zero baseline
                else:
                    baseline = 0

                # Subtract the estimated baseline from the signal chunk
                signal[i0:i1] -= baseline

        # Return the TOD with the azimuth–elevation correlated baseline removed
        return signal

    def remove_offset_per_scan(self,
                               tod: np.ndarray,
                               method: str = 'median',
                               inplace: bool = True) -> np.ndarray:

        """
        Removes the offset from the time-ordered data (TOD) for each scan in the dataset.

        This method adjusts the data to remove offsets within defined scan intervals using the
        specified reduction method (mean, median, or mode). It allows for in-place modification
        or returns a corrected copy of the input data. Each scan's offset is calculated and
        subtracted independently.

        :param tod: Input time-ordered data array to process
        :type: np.ndarray

        :param method: Specifies the method used to calculate the offset. Available options are:
            'mean': Calculates the mean of the scan segment.
            'median': Calculates the median of the scan segment.
            'mode': Uses a mode-histogram approach for the scan segment.
            Default is 'median'.
        :type: str, optional

        :param inplace: If True, the input data array will be modified in place. If False, a new corrected
            copy of the data will be returned. Default is True.
        :type: bool, optional

        :return: The corrected time-ordered data with the scan-specific offset removed
        :rtype: np.ndarray
        """

        corrected = tod if inplace else tod.copy()

        # choose the offset function
        if method == 'mean':
            offset_func = lambda x: np.mean(x)
        elif method == 'median':
            offset_func = lambda x: np.median(x)
        elif method == 'mode':
            offset_func = lambda x: mode_histogram(x, min(len(x) / 30, 51))
        else:
            raise ValueError(f"Unknown method {method!r}")

        # subtract offset per sweep
        for start, stop, _ in self.dataset.scans:
            segment = corrected[start:stop]
            if segment.size:
                corrected[start:stop] = segment - offset_func(segment)

        return corrected

    def plot_chunk_mode_vs_azimuth(self,
                                   tes_idx=0,
                                   n_bins=20,
                                   make_plot=False,
                                   max_chunks=None,
                                   inplace=True):

        """
        Plots and decorrelates the signal mode per chunk as a function of azimuth and elevation for a time-ordered dataset.
        Each scan (forth or back) is divided into chunks of variable length, and for each chunk, the mode of the signal,
        the median azimuth, and median elevation are computed. The signal mode is plotted against azimuth and elevation,
        and second-order polynomial fits are overlaid for each scan direction.

        :param tes_idx: Index of the TES (Transition Edge Sensor) to process
        :type: int

        :param n_bins:  Number of bins (chunks) per scan. Each chunk will cover a proportionate length of the scan
        :type: int

        :param make_plot: If True, generates plots of mode vs azimuth and saves them. Default is False.
        :type: bool

        :param max_chunks: Limits the processing and plotting to chunks from the first `max_chunks` scans if provided
        :type: Optional[int]

        :param inplace: If True, modifies the signal in-place by subtracting per-scan baseline fits.
            If False, processes a copy of the signal.
        :type: bool
        """

        # Select the TES timeline to process; optionally work on a copy to avoid modifying the dataset in-place
        signal = self.dataset.signals[tes_idx] if inplace else self.dataset.signals[tes_idx].copy()
        # Azimuth and elevation interpolated onto the TOD sampling
        az = self.dataset.azimuth_interp
        el = self.dataset.elevation_interp

        # Compute robust per-chunk mode estimates and the corresponding median pointing values
        # for each chunk, along with scan direction (+1 forth / -1 back) and scan IDs
        modes, az_medians, el_medians, scan_dirs, scan_ids = self.extract_chunk_modes(
            self.dataset.scans, signal, az, el, n_bins=n_bins)

        # Optionally restrict the plot to chunks coming from only the first `max_chunks` scans
        if max_chunks is not None:

            # Get the unique scan identifiers present in the chunk list (sorted)
            unique_ids = np.unique(scan_ids)
            # Keep only the first `max_chunks` scan IDs
            keep_ids = unique_ids[:max_chunks]
            # Build a mask selecting chunks that belong to the desired scan IDs
            mask = np.isin(scan_ids, keep_ids)

            # Apply the mask consistently to all chunk-level arrays
            modes = modes[mask]
            az_medians = az_medians[mask]
            el_medians = el_medians[mask]
            scan_dirs = scan_dirs[mask]
            scan_ids = scan_ids[mask]

            # Decorrelate the TOD for the selected scans only by subtracting
            # a per-scan quadratic baseline as a function of azimuth and then elevation.
            # NOTE: This block modifies `signal` directly for the scan intervals included in `keep_ids`.
            for sid in keep_ids:

                # Select chunks belonging to the current scan
                mask_this_scan = scan_ids == sid

                # Require a minimum number of chunks to fit a quadratic model reliably
                if np.sum(mask_this_scan) < 3:
                    continue

                # Extract chunk-level az/el coordinates and mode values for this scan
                this_scan_az = az_medians[mask_this_scan]
                this_scan_el = el_medians[mask_this_scan]
                this_scan_modes = modes[mask_this_scan]

                # Retrieve the TOD indices (start, end) for this scan segment
                s, e, _ = self.dataset.scans[sid]

                # Baseline fit vs azimuth (quadratic)
                # Fit mode(chunk) as a function of azimuth using a 2nd-degree polynomial
                z_az = np.polyfit(this_scan_az, this_scan_modes, 2)
                # Evaluate the azimuth baseline across the full scan timeline
                baseline_fit_az = np.polyval(z_az, az[s:e])
                # Subtract the azimuth-correlated baseline from the signal in this scan interval
                signal[s:e] = signal[s:e] - baseline_fit_az

                # Baseline fit vs elevation (quadratic)
                # Fit mode(chunk) as a function of elevation using a 2nd-degree polynomial
                z_el = np.polyfit(this_scan_el, this_scan_modes, 2)
                # Evaluate the elevation baseline across the full scan timeline
                baseline_fit_el = np.polyval(z_el, el[s:e])
                # Subtract the elevation-correlated baseline from the signal in this scan interval
                signal[s:e] = signal[s:e] - baseline_fit_el

        # Produce diagnostic plots if requested
        if make_plot:
            cmap = "viridis"
            # Create a 2D scatter plot: chunk mode vs azimuth, colored by elevation
            fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True)
            # Masks separating forth and back chunks
            mask_forth = scan_dirs == 1
            mask_back = scan_dirs == -1

            # Scatter plot for the forth direction (circles)
            sc_forth = ax.scatter(
                az_medians[mask_forth], modes[mask_forth],
                c=el_medians[mask_forth], cmap=cmap, marker='o', label='Forth')

            # Scatter plot for the back direction (triangles)
            sc_back = ax.scatter(
                az_medians[mask_back], modes[mask_back],
                c=el_medians[mask_back], cmap=cmap, marker='^', label='Back')

            # Fit and overlay a quadratic polynomial for each scan separately,
            # plotting the curve only across the chunk azimuth range of that scan.
            unique_ids = np.unique(scan_ids)
            # Flags to add the legend label only once for each direction
            label_forth_added = False
            label_back_added = False

            for sid in unique_ids:

                # Select chunks belonging to the current scan
                mask_this_scan = scan_ids == sid

                # Skip scans with too few chunk points to fit a quadratic curve
                if np.sum(mask_this_scan) < 3:
                    continue

                # Extract per-chunk azimuth and mode for this scan
                azs = az_medians[mask_this_scan]
                ms = modes[mask_this_scan]

                # Direction is constant within a scan; read it from the first chunk
                dir_this = scan_dirs[mask_this_scan][0]
                # Choose a color depending on the scan direction
                col = 'deepskyblue' if dir_this == 1 else 'orange'
                # Add the legend label only for the first scan of each direction
                if dir_this == 1 and not label_forth_added:
                    label = 'Poly fit forth'
                    label_forth_added = True
                elif dir_this == -1 and not label_back_added:
                    label = 'Poly fit back'
                    label_back_added = True
                else:
                    label = None

                # Fit a 2nd-degree polynomial: mode = f(azimuth)
                z = np.polyfit(azs, ms, 2)
                # Evaluate the fitted curve on a dense azimuth grid within this scan range
                az_fit = np.linspace(azs.min(), azs.max(), 100)
                ms_fit = np.polyval(z, az_fit)
                # Overlay the fitted curve
                ax.plot(az_fit, ms_fit, color=col, lw=2, alpha=0.7, label=label)

            ax.set_xlabel("Azimuth [deg]")
            ax.set_ylabel("Signal mode (per chunk)")
            ax.set_title(f"TES {tes_idx} | {n_bins} bin per scan | {max_chunks} chunks total")
            # Add colorbar for elevation encoding
            cbar = fig.colorbar(sc_forth, ax=ax, label="Elevation [deg]")
            ax.legend()
            ax.grid(True)
            # Save the scatter + per-scan fit diagnostic
            fig.savefig(os.path.join(self.output, f"chunk_mode_vs_azimuth_tes{tes_idx}_n_bins{n_bins}_fit.png"))

            fig, axs = plt.subplots(3, 1, figsize=(7, 10), tight_layout=True)

            # Panel 1: mode vs azimuth (after decorrelation), colored by elevation
            sc1 = axs[0].scatter(az_medians, modes, c=el_medians, cmap='viridis', s=20)
            axs[0].axhline(0, color='gray', ls='--')
            axs[0].set_xlabel("Azimuth [deg]")
            axs[0].set_ylabel("Signal mode (decorrelated)")
            axs[0].set_title("Mode vs Azimuth (dopo decorrelazione)")
            plt.colorbar(sc1, ax=axs[0], label="Elevation [deg]")

            # Panel 2: mode vs elevation (after decorrelation), colored by azimuth
            sc2 = axs[1].scatter(el_medians, modes, c=az_medians, cmap='plasma', s=20)
            axs[1].axhline(0, color='gray', ls='--')
            axs[1].set_xlabel("Elevation [deg]")
            axs[1].set_ylabel("Signal mode (decorrelated)")
            axs[1].set_title("Mode vs Elevation (dopo decorrelazione)")
            plt.colorbar(sc2, ax=axs[1], label="Azimuth [deg]")

            # Panel 3: histogram of chunk mode values (after decorrelation)
            axs[2].hist(modes, bins=40, color='navy', alpha=0.7)
            axs[2].set_xlabel("Signal mode (decorrelated)")
            axs[2].set_ylabel("Counts")
            axs[2].set_title("Distribuzione moda per chunk dopo decorrelazione")
            axs[2].grid()

            os.makedirs(os.path.join(self.output, "decorrelation"), exist_ok=True)
            fig.savefig(os.path.join(self.output, "decorrelation", f"decorrelation_diagnostics_tes{tes_idx}_n_bins{n_bins}.png"),
                        dpi=400)

    def fit_mode_az_el_2d(self,
                          tod: np.ndarray,
                          tes_idx=0,
                          n_bins=20,
                          poly_deg=2,
                          make_plot=True,
                          inplace=True):
        """
        Fits 2D polynomial regressions to the signal modes as a function of azimuth and elevation for both
        forth and back scanning directions. This method processes the input time-ordered data (TOD), extracts
        mode data, and fits a polynomial surface while optionally generating a 3D visualization and saving
        the result to a file.


        :param tod: The time-ordered data to process
        :type: np.ndarray

        :param tes_idx: Index of the TES to process. Default is 0
        :type: int

        :param n_bins: Number of bins to use for grouping data per scan. Default is 20
        :type: int

        :param poly_deg: Degree of the polynomial to fit to the data. Default is 2
        :type: int

        :param make_plot: Whether to generate a 3D visualization of the result. Default is True
        :type: bool

        :param inplace: Whether to modify the input TOD in place or work on a copied version. Default
            is True
        :type: bool

        :return: A tuple containing:
            - poly: The polynomial features transformer used for fitting.
            - reg_forth: The linear regression model fitted to the "forth" scan directions, or None if
              insufficient data was available.
            - reg_back: The linear regression model fitted to the "back" scan directions, or None if
              insufficient data was available.
        :rtype: tuple
        """

        # Work in place or on a copy of the input TOD, depending on the flag
        signal = tod if inplace else tod.copy()
        # Retrieve azimuth and elevation timelines interpolated onto the TOD sampling
        az = self.dataset.azimuth_interp
        el = self.dataset.elevation_interp

        # Remove a per-direction offset (median) from the TOD.
        # This centers the forth/back data separately before estimating chunk modes,
        # reducing bias from direction-dependent offsets.
        signal[self.dataset.mask_forth] -= np.median(signal[self.dataset.mask_forth])
        signal[self.dataset.mask_back] -= np.median(signal[self.dataset.mask_back])

        # Extract per-chunk mode estimates along scans, together with
        # the corresponding median azimuth/elevation values and scan direction labels
        modes, az_medians, el_medians, scan_dirs, _ = self.extract_chunk_modes(
            self.dataset.scans, signal, az, el, n_bins=n_bins)

        # Forth scans
        # Select chunks belonging to forth scans
        mask_forth = scan_dirs == 1
        # Design matrix for forth chunks: columns are (azimuth, elevation)
        X_forth = np.column_stack([az_medians[mask_forth], el_medians[mask_forth]])
        # Target vector for forth chunks: robust mode estimate per chunk
        y_forth = modes[mask_forth]

        # Back scans
        mask_back = scan_dirs == -1
        # Select chunks belonging to back scans
        X_back = np.column_stack([az_medians[mask_back], el_medians[mask_back]])
        # Target vector for back chunks: robust mode estimate per chunk
        y_back = modes[mask_back]



        # Fit 2D polynomial for forth
        # Require a minimum number of chunk points to constrain the polynomial surface.
        # (Heuristic: at least a few more points than the number of polynomial terms.)
        poly = PolynomialFeatures(degree=poly_deg)

        if len(y_forth) >= 6:
            # Expand back inputs into polynomial features
            X_forth_poly = poly.fit_transform(X_forth)
            # Fit a linear regression in polynomial-feature space: mode = f(az, el)
            reg_forth = LinearRegression().fit(X_forth_poly, y_forth)
        # Not enough back samples to fit a stable surface
        else:
            reg_forth = None

        # Fit back
        if len(y_back) >= 6:
            X_back_poly = poly.fit_transform(X_back)
            reg_back = LinearRegression().fit(X_back_poly, y_back)
        else:
            reg_back = None

        if make_plot:

            # Create a 3D scatter plot of chunk modes with the fitted polynomial surfaces overlaid
            fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(8, 6), tight_layout=True)
            ax.scatter(az_medians[mask_forth], el_medians[mask_forth], y_forth, c='deepskyblue', s=20, label='Forth')
            ax.scatter(az_medians[mask_back], el_medians[mask_back], y_back, c='orange', s=20, label='Back')

            # Loop over directions and overlay a fitted surface
            for mask, reg, color, label in [(mask_forth, reg_forth, 'deepskyblue', 'Forth fit'),
                                            (mask_back, reg_back, 'orange', 'Back fit')]:
                if reg is not None:
                    #  define an (az,el) gird covering the observed chunk medians for this direction
                    az_range = np.linspace(az_medians[mask].min(), az_medians[mask].max(), 40)
                    el_range = np.linspace(el_medians[mask].min(), el_medians[mask].max(), 40)
                    AZ, EL = np.meshgrid(az_range, el_range)
                    #  Build the grid design matrix and transform into polynomial features
                    X_grid = np.column_stack([AZ.ravel(), EL.ravel()])
                    #  predict the fitted surface (mode) on the grid and reshape to 2D
                    Z = reg.predict(poly.transform(X_grid)).reshape(AZ.shape)
                    ax.plot_surface(AZ, EL, Z, alpha=0.4, color=color, label=label, linewidth=0)

            ax.set(xlabel='Azimuth [deg]',
                   ylabel='Elevation [deg]',
                   zlabel='Signal mode',
                   title=f"Fit 2D Mode = f(Az, El) \nTES {tes_idx} | {n_bins} bin per scan")

            ax.legend()

            os.makedirs(os.path.join(self.output, "mode_fit"), exist_ok=True)
            fig.savefig(os.path.join(self.output, "mode_fit", f"mode_az_el_2d_surface_tes{tes_idx}_n_bins{n_bins}_deg{poly_deg}"))

        return poly, reg_forth, reg_back

    def plot_mode_az_el_3d_interactive(self,
                                       tod: np.ndarray,
                                       tes_idx=-1,
                                       n_bins=20,
                                       poly_deg=2,
                                       output_html=None,
                                       inplace=True):
        """
        Generates a 3D interactive visualization of the signal modes as a function of azimuth and
        elevation. The method uses provided signal data and azimuth/elevation interpolation
        to compute median values and fits polynomial surfaces for "forth" and "back" scan directions.

        :param tod: Time-ordered signal data.
        :type: np.ndarray

        :param tes_idx: The index specifying which channels of the TES dataset to consider. Defaults to -1,
            which typically means using all channels or a predefined setting.
        :type: int, optional

        :param n_bins:  Number of bins used for chunking the signal into azimuth and elevation modes.
            Defaults to 20.
        :type: int, optional

        :param poly_deg: Degree of the polynomial for fitting the signal modes. Defaults to 2.
        :type: int, optional

        :param: output_html: Path to save the generated 3D plot as an HTML file. Defaults to None.
        :type: str, optional

        :param inplace: Determines if the signal data (`tod`) is modified in place or a copy is used.
            Defaults to True.
        :type: bool, optional


        :return: None
            This method visualizes the data in an interactive 3D plot and optionally saves it
            to an output HTML file.
        """

        # Work in place or on a copy of the input TOD, depending on the flag
        signal = tod if inplace else tod.copy()
        # Retrieve azimuth and elevation timelines interpolated onto the TOD sampling
        az = self.dataset.azimuth_interp
        el = self.dataset.elevation_interp

        # Remove a per-direction offset (median) from the TOD to center forth and back data separately
        signal[self.dataset.mask_forth] -= np.median(signal[self.dataset.mask_forth])
        signal[self.dataset.mask_back] -= np.median(signal[self.dataset.mask_back])
        # Extract per-chunk mode estimates along scans, together with
        # the corresponding median azimuth/elevation values and scan direction labels
        modes, az_medians, el_medians, scan_dirs, _ = self.extract_chunk_modes(
            self.dataset.scans, signal, az, el, n_bins=n_bins)

        # Forth scans
        mask_forth = scan_dirs == 1
        # Design matrix for forth chunks: columns are (azimuth, elevation)
        X_forth = np.column_stack([az_medians[mask_forth], el_medians[mask_forth]])
        y_forth = modes[mask_forth]
        # Back scans
        mask_back = scan_dirs == -1
        X_back = np.column_stack([az_medians[mask_back], el_medians[mask_back]])
        y_back = modes[mask_back]

        # Create the polynomial feature transformer for a 2D polynomial surface
        poly = PolynomialFeatures(degree=poly_deg)
        # Initialize the forth regression model (may remain None if too few points)
        reg_forth = None

        # Fit the forth direction only if there are enough chunk points to constrain the surface
        if len(y_forth) >= 6:
            # Expand forth inputs into polynomial features
            X_forth_poly = poly.fit_transform(X_forth)
            # Fit a linear regression in polynomial-feature space: mode = f(az, el)
            reg_forth = LinearRegression().fit(X_forth_poly, y_forth)

        # Initialize the back regression model (may remain None if too few points)
        reg_back = None
        if len(y_back) >= 6:
            X_back_poly = poly.fit_transform(X_back)
            reg_back = LinearRegression().fit(X_back_poly, y_back)

        # Create an interactive Plotly figure
        fig = go.Figure()

        # Add forth chunk points as a 3D scatter trace
        fig.add_trace(go.Scatter3d(
            x=az_medians[mask_forth], y=el_medians[mask_forth], z=y_forth,
            mode='markers', marker=dict(color='deepskyblue', size=3), name='Forth'))

        # Add back chunk points as a 3D scatter trace
        fig.add_trace(go.Scatter3d(
            x=az_medians[mask_back], y=el_medians[mask_back], z=y_back,
            mode='markers', marker=dict(color='orange', size=3), name='Back'))

        # Add the fitted forth surface if a forth regression model is available
        if reg_forth is not None:
            # Define an (az, el) grid covering the observed forth chunk medians
            az_grid = np.linspace(az_medians[mask_forth].min(), az_medians[mask_forth].max(), 40)
            el_grid = np.linspace(el_medians[mask_forth].min(), el_medians[mask_forth].max(), 40)
            AZ, EL = np.meshgrid(az_grid, el_grid)
            # Build the grid design matrix and transform into polynomial features
            X_grid = np.column_stack([AZ.ravel(), EL.ravel()])
            # Predict the fitted surface (mode) on the grid and reshape to 2D
            Z = reg_forth.predict(poly.transform(X_grid)).reshape(AZ.shape)
            # Add the surface to the interactive figure
            fig.add_trace(go.Surface(x=az_grid, y=el_grid, z=Z, opacity=0.5, colorscale='Blues', name='Forth fit',
                                     showscale=False))

        # Add the fitted back surface if a back regression model is available
        if reg_back is not None:
            az_grid = np.linspace(az_medians[mask_back].min(), az_medians[mask_back].max(), 40)
            el_grid = np.linspace(el_medians[mask_back].min(), el_medians[mask_back].max(), 40)
            AZ, EL = np.meshgrid(az_grid, el_grid)
            X_grid = np.column_stack([AZ.ravel(), EL.ravel()])
            Z = reg_back.predict(poly.transform(X_grid)).reshape(AZ.shape)
            fig.add_trace(go.Surface(x=az_grid, y=el_grid, z=Z, opacity=0.5, colorscale='Oranges', name='Back fit',
                                     showscale=False))

        fig.update_layout(
            title=r"3D Fit: Mode = f(Az,El)",
            scene=dict(
                xaxis_title=r"Azimuth [deg]",
                yaxis_title=r"Elevation[deg]",
                zaxis_title=r"Signal mode",
            ),
            width=900,
            height=700,
            showlegend=True
        )

        if output_html is not None:
            fig.write_html(output_html)

    def plot_quadratic_fit(self,
                           tod: np.ndarray,
                           tes_idx=0,
                           n_bins=20,
                           poly_deg=2,
                           inplace=True):
        """
        Plots diagnostic visualizations for a two-dimensional quadratic fit evaluation.

        This method generates plots that assess the quality of the fit by showing:
        - Residuals versus fitted values.
        - Histogram of residuals.

        It utilizes the `fit_mode_az_el_2d` function for performing the fit.

        :param tod: The array of time-ordered data to analyze.
        :type: ndarray

        :param tes_idx: The TES index for labeling output files. Defaults to 0.
        :type: int, optional

        :param n_bins: The number of bins to use per scan for mode extraction. Defaults to 20.
        :type: int, optional

        :param poly_deg: The degree of the polynomial fit. Defaults to 2.
        :type: int, optional

        :param inplace: If True, modifies the `tod` array in place. Otherwise, processes a
            copy of the data. Defaults to True.
        :type: bool, optional

        :return: None
        """

        # Work in place or on a copy of the input TOD, depending on the flag
        signal = tod if inplace else tod.copy()
        # Retrieve azimuth and elevation timelines interpolated onto the TOD sampling
        az = self.dataset.azimuth_interp
        el = self.dataset.elevation_interp

        # Remove a per-direction offset (median) to center forth/back data separately
        signal[self.dataset.mask_forth] -= np.median(signal[self.dataset.mask_forth])
        signal[self.dataset.mask_back] -= np.median(signal[self.dataset.mask_back])
        # Extract per-chunk mode estimates along scans, together with
        # the corresponding median azimuth/elevation values (direction is ignored here)
        modes, az_medians, el_medians, _, _ = self.extract_chunk_modes(
            self.dataset.scans, signal, az, el, n_bins=n_bins)

        # Create the polynomial feature transformer for a 2D polynomial surface
        poly = PolynomialFeatures(degree=poly_deg)
        # Build the design matrix from (azimuth, elevation) chunk medians and expand to polynomial features
        X_poly = poly.fit_transform(np.column_stack([az_medians, el_medians]))
        # Fit a linear regression model in polynomial-feature space: mode = f(az, el)
        reg = LinearRegression().fit(X_poly, modes)
        # Predict fitted mode values at the training points
        fit_vals = reg.predict(X_poly)
        # Compute residuals between observed chunk modes and fitted values
        residuals = modes - fit_vals

        # Create a figure with two panels: residuals vs fitted values, and residual histogram
        fig, axs = plt.subplots(1, 2, figsize=(16, 4), tight_layout=True)

        # 1) Residuals vs fitted values
        axs[0].scatter(fit_vals, residuals, alpha=0.6, s=8)
        axs[0].axhline(0, color='gray', ls='--')
        axs[0].set(xlabel="Fitted mode",
                   ylabel="Residual: mode - fit",
                   title=f"Residual vs fit: mode = f(Az, El) | {n_bins} bin per scan | {poly_deg} deg poly fit")

        axs[0].grid()

        # 2) Histogram of residuals
        axs[1].hist(residuals, bins=30, color='tab:blue', alpha=0.7)
        axs[1].set(xlabel="Residual: mode - fit", ylabel="Counts", title="Residual histogram")
        axs[1].grid()


        os.makedirs(os.path.join(self.output, "quadratic_fit"), exist_ok=True)
        fig.savefig(os.path.join(self.output, "quadratic_fit", f"quadratic_fit2d_tes{tes_idx}_n_bins{n_bins}"))
        plt.close("all")

    def linear_rescale_chunks(self, tod: np.ndarray, sz: int = 1000) -> np.ndarray:
        """
        Adjusts the given data array by linearly rescaling specific chunks based on their characteristics.

        This method processes the input time-ordered data (TOD) array by identifying consecutive
        sections of invalid data (marked by a specific condition) and then adjusts these sections
        by interpolating values based on surrounding valid data. It ensures smooth transitions
        and adjusts the data linearly within identified chunks.

        :param tod: The input time-ordered data array to be adjusted.
        :type: np.ndarray

        :param sz: The number of elements to be considered in the surrounding region when computing
            medians for adjustments, by default 1000.
        :type: int, optional

        :return: The adjusted time-ordered data array with modifications applied to the invalid data sections.
        :rtype: np.ndarray
        """

        # Identify all indices in the TOD corresponding to dead-time / turn-around
        # regions (i.e. samples not belonging to either forth or back scans)
        zero_idxs = np.where(self.dataset.scantype == 0)[0]
        # Split the dead-time indices into contiguous blocks
        if zero_idxs.size:
            # Find boundaries where indices are no longer consecutive
            splits = np.where(np.diff(zero_idxs) != 1)[0] + 1
            # Each block corresponds to a contiguous dead-time segment
            blocks = np.split(zero_idxs, splits)
        else:
            blocks = []

        # Loop over each contiguous dead-time block
        for blk in blocks:
            # First index of the dead block
            start = blk[0]
            # Last index of the dead block
            stop = blk[-1]
            # Length of the dead block in samples
            chunklen = stop - start + 1

            # Case 1: dead block at the beginning of the TOD
            if start == 0:
                # Estimate the reference level from the following valid samples
                right = np.median(tod[stop + 1:stop + sz])
                # Median level inside the dead block
                med = np.median(tod[start:stop + 1])
                # Offset to subtract in order to align the dead block to the right segment
                vals = right + med
                # Apply the correction to the dead block
                tod[start:stop + 1] -= vals

            # Case 2: dead block at the end of the TOD
            elif stop == len(tod) - 1:
                # Estimate the reference level from the preceding valid samples
                left = np.median(tod[start - sz:start])
                # Median level inside the dead block
                med = np.median(tod[start:stop + 1])
                # Offset to subtract in order to align the dead block to the left segment
                vals = left + med
                # Apply the correction to the dead block
                tod[start:stop + 1] -= vals

            # Case 3: dead block in the middle of the TOD
            else:
                # Estimate reference levels from the valid regions before and after the block
                left = np.median(tod[start - 1 - sz:start - 1])
                right = np.median(tod[stop + 1:stop + 1 + sz])
                # Build a linear interpolation ramp between left and right reference levels
                frac = np.linspace(0, 1, chunklen)
                ramp = left + frac * (right - left)
                # Median level inside the dead block
                med = np.median(tod[start:stop + 1])
                # Subtract the difference between the block median and the interpolated ramp,
                # effectively stitching the dead block smoothly between neighboring scans
                tod[start:stop + 1] -= (med - ramp)

        return tod

    def preprocess(self,
                   tes_idx: int = 0,
                   sign_tes: int = 1,
                   *,
                   kernel_size: int | None = 101,
                   make_plots: bool = True):
        """
        One‑shot convenience method that runs the entire pipeline:

            1. Estimate scanning frequency (spectrogram).
            2. Identify scan sweeps (forth / back) on raw timeline.
            3. Estimate FORTH↔BACK lag from a single TES.
            4. Apply global time‑shift.
            5. Re‑identify scans on the corrected timeline.
            6. Decorrelate azimuth × elevation baseline.
            7. Plot mode profiles of an example sweep pair.

        :param: tes_idx: TES channel used to estimate the mount lag and to produce the
            diagnostic plots.
        :type: int

        :param kernel_size: Kernel size forwarded to `find_scans`; if None let the method
            estimate it automatically.
        :type: int | None

        :param make_plots: Produce diagnostic figures (spectrogram is suppressed here
            because it would duplicate the call).
        :type: bool
        """
        self.logger.info("=== PREPROCESS PIPELINE START (TES %d) ===", tes_idx)

        # 1. scanning frequency
        self.get_scanning_frequency(method="spectrogram", make_plots=False)

        # 2. first scan segmentation
        self.find_scans(kernel_size=kernel_size, make_plot=False)

        # 3. lag estimation
        delta_t = 0.191
        self.logger.info("Global lag = %.4f s", delta_t)

        # 5. scan segmentation on corrected timeline
        self.find_scans(kernel_size=kernel_size, delta_t=delta_t, make_plot=make_plots)

        tod = sign_tes * self.dataset.signals[tes_idx].copy()
        sig = self.remove_offset_per_scan(tod=tod, method="median")

        sig = self.subtract_az_el_baseline(tod=sig, make_plot=True)

        # regularize bad regions
        sig = self.remove_offset_per_scan(tod=sig, method="mode")
        sig = self.linear_rescale_chunks(sig, sz=1000)

        self.logger.info("=== PREPROCESS PIPELINE DONE ===")

        # compute and log standard deviation of the cleaned signal
        std_clean = np.std(sig)
        self.logger.info("Standard deviation of cleaned TOD (TES %d): %.6g ADU", tes_idx, std_clean)
        print(f"Standard deviation of cleaned TOD (TES {tes_idx}): {std_clean:.6g} ADU")

        return sig
    
    
    def plot_tod(self,
                 raw_tod: np.ndarray,
                 processed_tod: np.ndarray,
                 plt_title: str,
                 threshold: int = 5,
                 tes_idx: int = 0,
                 time: np.ndarray | None = None,
                 filename: str | None = None):
        """
        Plot raw and processed Time Ordered Data (TOD) as a function of time.

        :param raw_tod: Raw TOD signal.
        :type: np.ndarray

        :param processed_tod: Preprocessed TOD signal.
        :type: np.ndarray

        :param plt_title: Title of the plot.
        :type: str

        :param tes_idx: TES index used only for labeling.
        :type: int

        :param time: Time array in seconds. If None, dataset.tm is used.
        :type: np.ndarray | None

        :param filename: Output filename (without extension). If None, a default name is used.
        :type: str | None
        """

        if time is None:
            time = self.dataset.tm

        fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)

        ax.plot(time, raw_tod, lw=0.8, alpha=0.7, label="Raw TOD")
        ax.plot(time, processed_tod, lw=0.8, alpha=0.9, label="Processed TOD")
        ax.axhline(threshold * processed_tod.std() + processed_tod.mean(), color='gray', ls='--',
                   label=rf"Threshold: {threshold} $\sigma$ + $<TOD>$")

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("ADU")
        ax.set_title(rf"{plt_title} | TES \# {tes_idx + 1}")
        ax.legend()
        ax.grid(True)

        if filename is None:
            filename = f"tod_raw_vs_processed_tes{tes_idx}"

        fig.savefig(os.path.join(self.output, filename))
        plt.close("all")



if __name__ == "__main__":
    # TES and dataset shown in the cosmic ray paper
    tes = 32
    data = '/Volumes/Data/PycharmProjects/qubic/qubic/qubic/scripts/Calibration/cosmic_rays/data/input/2022-07-14/2022-07-14_23.54.19__MoonScan_Speed_VE14'

    pre_process = PreProcess(data, output=f"pre_processing_{tes}")
    pre_process.load()
    tod_corrected = pre_process.preprocess(tes_idx=tes, make_plots=True)

    raw_tod = pre_process.dataset.signals[tes]

    pre_process.plot_tod(raw_tod=raw_tod,
                         plt_title=Path(data).name,
                         processed_tod=tod_corrected,
                         threshold=5,
                         tes_idx=tes)
