# Author: Belén Costanza, Lucas Merlo, Claudia Scóccola

import numpy as np
import os
import bottleneck as bn
from sklearn.cluster import DBSCAN
from sklearn.tree import DecisionTreeRegressor
from typing import Union
import yaml
import pickle
import matplotlib.pyplot as plt


class badIV:
    def __init__(
        self, directory: str, n_times: int, TES_number: int = 256, filename: str = "IVdict2025.yaml"
    ):
        """

        Class for discarding TES with bad IV curves.

        This class identifies TES to be excluded based on the number of times
        they exhibit bad IV curves in a given dataset year (e.g. 2022–2023 or 2025).
        The information is read from a dictionary where each TES index is associated
        with its number of bad IV occurrences. TES not present in the dictionary always have good IV curves.

        Parameters:

        directory: Directory containing the bad IV dictionary.
        n_times: Minimum number of bad IV occurrences required to discard a TES.
        TES_number: Number of TES in the focal plane (default: 256).
        filename: Name of the dictionary file to read
                  (e.g. "IVdict2025.yaml" or "IVdict2023.yaml").
        """

        self.directory = directory
        self.n_times = n_times
        self.TES_number = TES_number
        self.filename = filename

    def load_bad_iv(self):
        """
        Load TES with bad IV curves from a directory.
        """

        path = os.path.join(self.directory, self.filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No bad IV file found at {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return {int(k): int(v) for k, v in data.items()}

    def select_badIV(self):
        """
        Select TES indices whose bad IV occurrence count is >= n_times.

        Parameters
        ----------
        bad_iv_counts : dict[int, int]
            Dictionary {TES_index: number_of_bad_IVs}.
        n_times : int
            Minimum number of bad IV occurrences.

        Returns
        -------
        list[int]
            Sorted list of TES indices satisfying n_fallas >= n_times.
        """
        counts = self.load_bad_iv()

        if not isinstance(self.n_times, int):
            raise TypeError("n_times must be an integer")
        if self.n_times < 0:
            raise ValueError("n_times must be non-negative")

        if counts:
            max_times = max(counts.values())
            if self.n_times > max_times:
                raise ValueError(
                    f"n_times={self.n_times} exceeds maximum possible value ({max_times})"
                )

        bad = {tes for tes, n in counts.items() if n >= self.n_times}

        mask = np.ones(self.TES_number, dtype=bool)
        for tes in bad:
            if 0 <= tes < self.TES_number:
                mask[tes] = False
        return mask


class fluxjumps:
    def __init__(self, thr: float, window_size: int):
        """Class for detection of discontinuities (flux jumps) in TOD.

        This class detects flux jumps by applying a Haar filter to the data and identifying
        discontinuities that exceed specified thresholds. It clusters detected jumps and
        determines their start and end positions.

        Parameters:
            thr: Threshold or list/array of thresholds for the Haar filter. If a flux jump
                 value is higher than the threshold, the time sample is considered a
                 flux jump candidate.
            window_size: Size of the bottleneck moving median window used in the Haar filter.
        """
        self.thr = np.array(thr)
        self.window_size = window_size

        # Constants for jumps_detection method
        self.MAX_JUMPS_BEFORE_RETHRESHOLD = 11
        self.HIGH_THRESHOLD_VALUE = 4e5
        self.THRESHOLD_FRACTION_FOR_EDGES = 0.05

    def haar_function(self, todarray: np.ndarray):
        """Apply Haar filter to detect discontinuities in the TOD array.

        Parameters:
            todarray: 1D array containing the time-ordered data.

        Returns:
            tod_haar: Array of the same size as todarray with Haar filter values.
        """
        if todarray.ndim != 1:
            raise ValueError(f"todarray must be 1D, got shape {todarray.shape}")
        if len(todarray) < self.window_size * 3:
            raise ValueError(
                f"todarray length ({len(todarray)}) must be at least "
                f"3 * window_size ({3 * self.window_size})"
            )

        tod_haar = np.zeros(todarray.size)
        xf = bn.move_median(todarray, self.window_size)[self.window_size :]
        tod_haar[
            self.window_size + self.window_size // 2 : -self.window_size + self.window_size // 2
        ] = xf[: -self.window_size] - xf[self.window_size :]

        return tod_haar

    def find_candidates(self, tod_haar: np.ndarray, thr: np.ndarray):
        """Find flux jump candidates using threshold comparisons.
        Returns boolean mask indicating jump locations.

        Parameters:
            tod_haar: Array of Haar filter values.
            thr: Array of threshold values to test.

        Returns:
            Tuple containing:
                - jumps: Boolean array where True indicates a flux jump candidate.
        """
        if tod_haar.ndim != 1:
            raise ValueError(f"tod_haar must be 1D, got shape {tod_haar.shape}")

        jumps = np.zeros_like(tod_haar, dtype=bool)

        max_haar = np.max(np.abs(tod_haar))

        if max_haar >= thr:
            # Found jumps with this threshold
            jumps = np.abs(tod_haar) >= thr

        return jumps

    def clusters(self, todarray: np.ndarray, jumps: np.ndarray):
        """Cluster detected jump indices to identify distinct flux jump events.

        Uses DBSCAN clustering to group nearby jump indices into clusters,
        where each cluster represents a distinct flux jump event.

        Parameters:
            todarray: 1D array of the TOD.
            jumps: Boolean array indicating jump candidate locations.

        Returns:
            Tuple containing:
                - nc: Number of clusters (flux jump events) found.
                - idx_jumps: Array of indices where jumps occur.
                - clust: DBSCAN cluster object, or None if no clusters found.
        """
        if len(todarray) != len(jumps):
            raise ValueError(
                f"todarray length ({len(todarray)}) must match jumps length ({len(jumps)})"
            )

        idx = np.arange(len(todarray))
        idx_jumps = idx[jumps]

        if idx_jumps.size > 1:
            clust = DBSCAN(eps=self.window_size // 5, min_samples=1).fit(
                np.reshape(idx_jumps, (len(idx_jumps), 1))
            )
            nc = int(np.max(clust.labels_) + 1)
        else:
            nc = 0
            idx_jumps = np.array([], dtype=int)
            clust = None

        return nc, idx_jumps, clust

    def initial_start_end(
        self, nc: int, idx_jumps: np.ndarray, tod_haar: np.ndarray, thr_used: float, clust: DBSCAN
    ):
        """Determine the initial start and end indices of each flux jump cluster.

        For each cluster, finds where the Haar filter values drop below a fraction
        of the threshold to determine the actual start and end of the jump.

        Parameters:
            nc: Number of clusters.
            idx_jumps: Array of indices where jumps occur.
            tod_haar: Array of Haar filter values.
            thr_used: Threshold value used for jump detection.
            clust: DBSCAN cluster object.

        Returns:
            Tuple containing:
                - xc: Array of start indices for each flux jump.
                - xcf: Array of end indices for each flux jump.
        """
        if clust is None or nc == 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        if idx_jumps.size == 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        xc = np.zeros(nc, dtype=int)
        xcf = np.zeros(nc, dtype=int)
        edge_threshold = thr_used * self.THRESHOLD_FRACTION_FOR_EDGES

        for i in range(nc):
            idx_jumps_from_thr = idx_jumps[clust.labels_ == i]

            if len(idx_jumps_from_thr) == 0:
                continue  # jump to next iteration

            first_jump_idx = idx_jumps_from_thr[0]
            last_jump_idx = idx_jumps_from_thr[-1]

            # Find end of jump (where Haar values drop below threshold after last jump)
            end_mask = np.abs(tod_haar[last_jump_idx:]) < edge_threshold
            idx_delta_end_jump = np.where(end_mask)[0][0]

            # Find start of jump (where Haar values drop below threshold before first jump)
            start_mask = np.abs(tod_haar[:first_jump_idx]) < edge_threshold
            last_below_threshold_idx = np.where(start_mask)[0][-1]
            idx_delta_start_jump = first_jump_idx - last_below_threshold_idx

            xc[i] = first_jump_idx - idx_delta_start_jump
            xcf[i] = last_jump_idx + idx_delta_end_jump

        return xc, xcf

    def unique(self, xc: np.ndarray, xcf: np.ndarray):
        """Remove duplicate start and end indices.

        Parameters:
            xc: Array of start indices.
            xcf: Array of end indices.

        Returns:
            Tuple containing:
                - nc_unique: Number of unique jump pairs.
                - xc_unique: Array of unique start indices.
                - xcf_unique: Array of unique end indices.
        """
        xc_unique = np.unique(xc)
        xcf_unique = np.unique(xcf)
        nc_unique = len(xc_unique)

        return nc_unique, xc_unique, xcf_unique

    def change_values(
        self, xc: Union[np.ndarray, list], xcf: Union[np.ndarray, list], max_gap: int = 10
    ):
        """Merge consecutive jumps that are close together.

        Groups jumps that are within max_gap samples of each other into a single
        jump region, which is useful for handling multiple closely-spaced jumps
        as a single discontinuity.

        Parameters:
            xc: Array or list of start indices.
            xcf: Array or list of end indices.
            max_gap: Maximum gap between jumps to consider them consecutive (default: 10).

        Returns:
            Tuple containing:
                - xc2: List of merged start indices.
                - xcf2: List of merged end indices.
        """
        xc2 = []
        xcf2 = []

        i = 0
        while i < len(xc):
            xini = xc[i]
            xfin = xcf[i]
            j = i + 1

            # Group while jumps are within the margin
            while j < len(xc) and xc[j] - xfin <= max_gap:
                xfin = xcf[j]
                j += 1

            xc2.append(xini)
            xcf2.append(xfin)
            i = j

        return xc2, xcf2

    def jumps_detection(self, todarray: np.ndarray, consec: bool = True, nc_cond: bool = False):
        """Main method to detect flux jumps in the TOD array.

        This method performs the complete flux jump detection pipeline:
        1. Applies Haar filter to detect discontinuities
        2. Finds jump candidates using thresholds
        3. Clusters jumps into distinct events
        4. Optionally re-thresholds if too many jumps detected
        5. Determines start and end positions of jumps
        6. Optionally merges consecutive jumps

        Parameters:
            todarray: 1D array containing the TOD.
            consec: If True, merge consecutive jumps that are close together (default: True).
            nc_cond: If True, apply higher threshold if more than MAX_JUMPS_BEFORE_RETHRESHOLD
                    jumps are detected (default: False).

        Returns:
            Tuple containing:
                - nc_unique: Number of unique flux jumps detected.
                - xc_unique: Array of start indices (empty array if no jumps).
                - xcf_unique: Array of end indices (empty array if no jumps).
        """
        if todarray.ndim != 1:
            raise ValueError(f"todarray must be 1D, got shape {todarray.shape}")

        # Step 1: Apply Haar filter
        tod_haar = self.haar_function(todarray)

        # Step 2: Find jump candidates using thresholds
        jumps = self.find_candidates(tod_haar, self.thr)

        # Step 3: Cluster jumps into distinct events
        nc, idx_jumps, clust = self.clusters(todarray, jumps)

        # Optional: Re-threshold if too many jumps detected
        if nc_cond and nc > self.MAX_JUMPS_BEFORE_RETHRESHOLD:
            self.thr = self.HIGH_THRESHOLD_VALUE
            tod_haar = self.haar_function(todarray)
            jumps = self.find_candidates(tod_haar, self.thr)
            nc, idx_jumps, clust = self.clusters(todarray, jumps)

        # Handle case with no jumps
        if nc == 0:
            return 0, np.array([], dtype=int), np.array([], dtype=int)

        # Step 4: Find start and end positions of jumps
        if clust is None:
            return 0, np.array([], dtype=int), np.array([], dtype=int)

        xc, xcf = self.initial_start_end(nc, idx_jumps, tod_haar, self.thr, clust)

        # Step 5: Remove duplicate start/end indices
        nc_unique, xc_unique, xcf_unique = self.unique(xc, xcf)

        # Step 6: Optionally merge consecutive jumps
        if consec:
            xc_unique, xcf_unique = self.change_values(xc_unique, xcf_unique)
            nc_unique = len(xc_unique)

        return nc_unique, np.array(xc_unique), np.array(xcf_unique)


class correction:
    def __init__(self, region_off: int = 5, region_amp: int = 10, change_mode: str = "const"):
        """Class that calculates flux jump amplitudes and applies corrections to TOD.

        This class corrects flux jumps in TOD by:
        1. Calculating the offset amplitude of each jump
        2. Removing the offset from the data after each jump
        3. Replacing the jump region with appropriate signal based on the change_mode

        Parameters:
            region_off: Number of samples to use for calculating offset before/after jumps (default: 5).
            region_amp: Number of samples to use for calculating mean/std before jumps (default: 10).
            change_mode: Method for replacing jump regions. Options:
                        - "init": Use initial mean and std for all jumps
                        - "noise": Use mean and std computed before each jump
                        - "const": Use constrained realization based on power spectrum (default).
        """
        if change_mode not in ["init", "noise", "const"]:
            raise ValueError(
                f"change_mode must be 'init', 'noise', or 'const', got '{change_mode}'"
            )
        if region_off < 1 or region_amp < 1:
            raise ValueError("region_off and region_amp must be positive integers")

        self.region_off = region_off
        self.region_amp = region_amp
        self.change_mode = change_mode

    def calculate_amplitude(self, todarray: np.ndarray, xc: np.ndarray, xcf: np.ndarray, nc: int):
        """Calculate the amplitude offset for each flux jump.

        For each jump, computes the difference between the median value after the jump
        and the median value before the jump. This offset represents the amplitude
        of the discontinuity.

        Parameters:
            todarray: 1D array containing the TOD.
            xc: Array of start indices for each jump.
            xcf: Array of end indices for each jump.
            nc: Number of jumps.

        Returns:
            offset: Array of offset amplitudes for each jump.
        """
        if todarray.ndim != 1:
            raise ValueError(f"todarray must be 1D, got shape {todarray.shape}")
        if len(xc) != nc or len(xcf) != nc:
            raise ValueError(f"xc and xcf must have length {nc}")
        if nc == 0:
            return np.array([], dtype=float)

        offset = np.zeros(nc)
        for i in range(nc):
            # Ensure indices are within bounds
            start_idx = max(0, xc[i] - self.region_off)
            end_idx = min(len(todarray), xcf[i] + self.region_off)

            region_before = todarray[start_idx : xc[i]]
            region_after = todarray[xcf[i] : end_idx]

            if len(region_before) == 0 or len(region_after) == 0:
                offset[i] = 0.0
            else:
                offset[i] = np.median(region_after) - np.median(region_before)

        return offset

    def move_offset(
        self, todarray: np.ndarray, offset: np.ndarray, xc: np.ndarray, xcf: np.ndarray, nc: int
    ):
        """Remove offset from the TOD after each flux jump.

        Applies the calculated offset correction to the data after each jump.
        The offsets are applied in reverse order (from last to first).

        Parameters:
            todarray: 1D array containing the TOD.
            offset: Array of offset amplitudes for each jump.
            xc: Array of start indices for each jump.
            xcf: Array of end indices for each jump.
            nc: Number of jumps.

        Returns:
            tod_new: Copy of todarray with offsets removed after each jump.
        """
        if todarray.ndim != 1:
            raise ValueError(f"todarray must be 1D, got shape {todarray.shape}")
        if len(offset) != nc or len(xc) != nc or len(xcf) != nc:
            raise ValueError(f"offset, xc, and xcf must have length {nc}")
        if nc == 0:
            return todarray.copy()

        tod_new = todarray.copy()
        array_length = len(todarray)

        if nc == 1:
            initial = xcf[0]
            final = array_length
            if initial < array_length:
                tod_new[initial:final] = todarray[initial:final] - offset[0]
        else:
            # Apply offsets in reverse order
            for i in range(len(xcf) - 1, -1, -1):
                initial = xcf[i]
                final = array_length
                if initial < array_length:
                    tod_new[initial:final] = tod_new[initial:final] - offset[i]

        return tod_new

    def changesignal_init(self, tod_new: np.ndarray, xc: np.ndarray, xcf: np.ndarray):
        """Replace jump regions with noise using initial statistics.

        Uses the standard deviation computed from the beginning of the data
        and the mean computed before each jump to generate replacement noise.

        Parameters:
            tod_new: TOD array with offsets removed.
            xc: Array of start indices for each jump.
            xcf: Array of end indices for each jump.

        Returns:
            y_cor: Corrected TOD array with jump regions replaced by noise.
        """
        if tod_new.ndim != 1:
            raise ValueError(f"tod_new must be 1D, got shape {tod_new.shape}")
        if len(xc) == 0:
            return tod_new.copy()

        y_cor = tod_new.copy()
        std = np.std(tod_new[: xc[0]])

        for i in range(len(xc)):
            # Compute mean from region before the jump
            start_idx = max(0, xc[i] - self.region_amp)
            region_before = tod_new[start_idx : xc[i]]
            mean = np.mean(region_before)

            # Generate replacement noise
            jump_length = xcf[i] - xc[i]
            if jump_length > 0:
                ynew = np.random.normal(mean, std, jump_length)
                y_cor[xc[i] : xcf[i]] = ynew

        return y_cor

    def changesignal_noise(self, tod_new: np.ndarray, xc: np.ndarray, xcf: np.ndarray):
        """Replace jump regions with noise using local statistics.

        For each jump, computes the mean and standard deviation from the region
        immediately before the jump, then generates replacement noise using these
        local statistics.

        Parameters:
            tod_new: TOD array with offsets removed.
            xc: Array of start indices for each jump.
            xcf: Array of end indices for each jump.

        Returns:
            y_cor: Corrected TOD array with jump regions replaced by noise.
        """
        if tod_new.ndim != 1:
            raise ValueError(f"tod_new must be 1D, got shape {tod_new.shape}")
        if len(xc) == 0:
            return tod_new.copy()

        y_cor = tod_new.copy()

        for i in range(len(xc)):
            # Compute statistics from region before the jump
            start_idx = max(0, xc[i] - self.region_amp)
            region_before = tod_new[start_idx : xc[i]]

            std = np.std(region_before)
            mean = np.mean(region_before)

            # Generate replacement noise
            jump_length = xcf[i] - xc[i]
            if jump_length > 0:
                ynew = np.random.normal(mean, std, jump_length)
                y_cor[xc[i] : xcf[i]] = ynew

        return y_cor

    def constrained_realization(self, tod_new: np.ndarray, xini: int, xfin: int):
        """Replace jump region with constrained realization based on power spectrum.

        Uses the power spectrum from regions before and after the jump to generate
        a synthetic signal that maintains the statistical properties while smoothly
        connecting the edges.

        Parameters:
            tod_new: TOD array with offsets removed.
            xini: Start index of the jump region.
            xfin: End index of the jump region.

        Returns:
            y: TOD array with jump region replaced by constrained realization.
        """
        if tod_new.ndim != 1:
            raise ValueError(f"tod_new must be 1D, got shape {tod_new.shape}")

        xini = int(xini)
        xfin = int(xfin)

        if xini < 0 or xfin > len(tod_new) or xini >= xfin:
            raise ValueError(
                f"Invalid jump indices: xini={xini}, xfin={xfin}, array_length={len(tod_new)}"
            )

        y = tod_new.copy()
        L = xfin - xini

        # Determine region sizes before and after jump (don't go beyond array boundaries)
        region_pre = min(L // 2, xini)
        region_post = min(L - region_pre, len(y) - xfin)

        total_need = L
        still_need = total_need - (region_pre + region_post)

        # Adjust regions if needed to sum to L
        if still_need > 0:
            extra_pre = min(still_need, region_pre)
            extra_post = still_need - extra_pre
            region_pre += extra_pre
            region_post += extra_post

        # Extract data before and after the jump
        pre = y[xini - region_pre : xini]
        post = y[xfin : xfin + region_post]

        local_data = np.concatenate([pre, post])

        N = len(local_data)

        # Compute power spectrum from data before and after the jump
        fft_data = np.fft.rfft(local_data - np.mean(local_data))
        power_spectrum = np.abs(fft_data) ** 2

        # Generate random phase realization with same power spectrum
        random_phases = np.exp(1j * 2 * np.pi * np.random.rand(len(fft_data)))
        new_fft = np.sqrt(power_spectrum) * random_phases
        sim_signal = np.fft.irfft(new_fft, n=N)

        sim_trunc = sim_signal[:L]

        start_val = y[xini - 1]
        end_val = y[xfin]
        sim_trunc -= np.mean(sim_trunc)  # centrado
        sim_trunc = sim_trunc / np.std(sim_trunc) * np.std(pre)

        # Linear transition to match edges
        window = np.linspace(0, 1, L)
        sim_trunc = (
            sim_trunc
            + (1 - window) * (start_val - sim_trunc[0])
            + window * (end_val - sim_trunc[-1])
        )
        y[xini:xfin] = sim_trunc

        return y

    def correct_TOD(
        self, todarray: np.ndarray, offset: np.ndarray, xc: np.ndarray, xcf: np.ndarray, nc: int
    ):
        """Apply complete correction to the TOD array.

        This is the main correction method that:
        1. Calculates jump amplitudes (if not provided)
        2. Removes offsets from the data
        3. Replaces jump regions based on the selected change_mode

        Parameters:
            todarray: 1D array containing the time-ordered data.
            offset: Array of offset amplitudes for each jump.
            xc: Array of start indices for each jump.
            xcf: Array of end indices for each jump.
            nc: Number of jumps.

        Returns:
            tod_corr: Corrected TOD array with flux jumps removed.
        """
        if todarray.ndim != 1:
            raise ValueError(f"todarray must be 1D, got shape {todarray.shape}")
        if nc == 0:
            return todarray.copy()

        # Step 1: Remove offsets
        tod_new = self.move_offset(todarray, offset, xc, xcf, nc)

        # Step 2: Replace jump regions based on change_mode
        if self.change_mode == "init":
            tod_corr = self.changesignal_init(tod_new, xc, xcf)
        elif self.change_mode == "noise":
            tod_corr = self.changesignal_noise(tod_new, xc, xcf)
        elif self.change_mode == "const":
            tod_corr = tod_new
            for i in range(nc):
                tod_corr = self.constrained_realization(tod_corr, xc[i], xcf[i])
        else:
            raise ValueError(f"Unknown change_mode: {self.change_mode}")

        return tod_corr


class DT:
    def __init__(
        self,
        thr_count: int = 600,
        thr_amp: float = 2e5,
        tol: float = 1e2,
        depth: bool = True,
        depth_number: int = 0,
    ):
        """Class for using decision tree to calculate levels between flux jumps.

        Uses a Decision Tree Regressor to identify discrete levels in the TOD data
        between flux jumps, which helps improve the correction process.

        Parameters:
            thr_count: Threshold for minimum number of repetitions for a level (default: 600).
            thr_amp: Threshold for minimum amplitude difference between levels (default: 2e5).
            tol: Tolerance for assigning signal values to specific DT levels (default: 1e2).
            depth: If True, max_depth is set to the number of flux jumps found.
                   If False, uses depth_number instead (default: True).
            depth_number: Depth to use for DT if depth is False (default: 0).
        """
        if thr_count < 1:
            raise ValueError("thr_count must be positive")
        if thr_amp < 0:
            raise ValueError("thr_amp must be non-negative")
        if tol < 0:
            raise ValueError("tol must be non-negative")
        if not depth and depth_number < 1:
            raise ValueError("depth_number must be positive when depth is False")

        self.thr_count = thr_count
        self.thr_amp = thr_amp
        self.tol = tol
        self.depth = depth
        self.depth_number = depth_number

        # Constants
        self.MIN_DEPTH = 3
        self.TOL_MULTIPLIER = 50

    def define_model(self, tt: np.ndarray, todarray: np.ndarray, num: int):
        """Define and fit a Decision Tree Regressor model.

        Fits a Decision Tree Regressor to the TOD to identify
        discrete levels in the signal.

        Parameters:
            tt: 1D array of time values.
            todarray: 1D array containing the TOD.
            num: Number of flux jumps (used to determine tree depth).

        Returns:
            ypred: Predicted values from the decision tree model.
        """
        if tt.ndim != 1 or todarray.ndim != 1:
            raise ValueError("tt and todarray must be 1D arrays")
        if len(tt) != len(todarray):
            raise ValueError(
                f"tt and todarray must have same length, got {len(tt)} and {len(todarray)}"
            )

        depth = max(self.MIN_DEPTH, num)
        x = tt.reshape(-1, 1)
        regressor = DecisionTreeRegressor(max_depth=depth, random_state=42)
        regressor.fit(x, todarray)
        ypred = regressor.predict(x)

        return ypred

    def uniqueindex(self, ypred: np.ndarray):
        """Find unique predicted values and count their occurrences.

        Parameters:
            ypred: Array of predicted values from the decision tree.

        Returns:
            Tuple containing:
                - predunique: Array of unique predicted values.
                - index: Array of first occurrence indices for each unique value.
                - count: Array of counts for each unique value.
        """
        if ypred.ndim != 1:
            raise ValueError(f"ypred must be 1D, got shape {ypred.shape}")
        if len(ypred) == 0:
            return np.array([]), np.array([], dtype=int), np.array([], dtype=int)

        predunique, index = np.unique(ypred, return_index=True)
        index = np.sort(index)
        predunique = ypred[index]

        count = np.zeros(len(predunique), dtype=int)
        for i in range(len(predunique)):
            count[i] = len(np.where(ypred == predunique[i])[0])

        return predunique, index, count

    def count_filter(self, predunique: np.ndarray, index: np.ndarray, count: np.ndarray):
        """Filter levels based on count threshold.

        Removes levels that don't have enough occurrences (below thr_count).

        Parameters:
            predunique: Array of unique predicted values.
            index: Array of first occurrence indices.
            count: Array of counts for each unique value.

        Returns:
            Tuple containing:
                - filtered_pred: Filtered unique predicted values.
                - filtered_index: Filtered indices.
                - filtered_count: Filtered counts.
        """
        if len(predunique) != len(index) or len(predunique) != len(count):
            raise ValueError("predunique, index, and count must have same length")

        mask = count > self.thr_count
        filtered_pred = predunique[mask]
        filtered_index = index[mask]
        filtered_count = count[mask]

        return filtered_pred, filtered_index, filtered_count

    def amplitude_filter(self, filpred: np.ndarray, filindex: np.ndarray, filcount: np.ndarray):
        """Filter level transitions based on amplitude threshold.

        Identifies transitions between consecutive levels that exceed the
        amplitude threshold (thr_amp).

        Parameters:
            filpred: Filtered unique predicted values.
            filindex: Filtered indices.
            filcount: Filtered counts (unused but kept for consistency).

        Returns:
            Tuple containing:
                - ampnew: List of amplitudes between consecutive levels.
                - valini: List of initial level values.
                - valfin: List of final level values.
                - indexini: List of initial level indices.
                - indexfin: List of final level indices.
        """
        if len(filpred) < 2:
            return [], [], [], [], []

        ampnew = []
        valini = []
        valfin = []
        indexini = []
        indexfin = []

        for i in range(len(filpred) - 1):
            amp = filpred[i + 1] - filpred[i]
            if abs(amp) > self.thr_amp:
                ampnew.append(amp)
                valini.append(filpred[i])
                valfin.append(filpred[i + 1])
                indexini.append(filindex[i])
                indexfin.append(filindex[i + 1])

        return ampnew, valini, valfin, indexini, indexfin

    def calculate_start_end(self, todarray, valini, valfin, indexfin):

        start = np.zeros(len(valini), dtype=int)
        end = np.zeros(len(valini), dtype=int)
        for i in range(len(valini)):
            index1 = np.where(
                (todarray < valini[i] + self.tol) & (todarray > valini[i] - self.tol)
            )[0]
            index2 = np.where(
                (todarray < valfin[i] + self.tol) & (todarray > valfin[i] - self.tol)
            )[0]

            if len(index1) == 0 or len(index2) == 0:
                tol2 = 50 * self.tol
                index1 = np.where((todarray < valini[i] + tol2) & (todarray > valini[i] - tol2))[0]
                index2 = np.where((todarray < valfin[i] + tol2) & (todarray > valfin[i] - tol2))[0]

            end[i] = index2[np.where(index2 > indexfin[i])[0]][0]
            start[i] = np.max(index1[index1 < end[i]])

            if len(valini) > 1:
                if end[i - 1] == start[i]:
                    start[i] += 1

        return start, end

    def calculate_start_end2(
        self, todarray: np.ndarray, valini: list, valfin: list, indexini: list, indexfin: list
    ):
        """Calculate start and end indices for level transitions.

        Determines the start and end positions of transitions between levels
        by finding where the signal matches the initial and final level values
        within the tolerance range.

        Parameters:
            todarray: 1D array containing the time-ordered data.
            valini: List of initial level values.
            valfin: List of final level values.
            indexini: List of initial level indices.
            indexfin: List of final level indices.

        Returns:
            Tuple containing:
                - start: Array of start indices for each transition.
                - end: Array of end indices for each transition.
        """
        if todarray.ndim != 1:
            raise ValueError(f"todarray must be 1D, got shape {todarray.shape}")
        if (
            len(valini) != len(valfin)
            or len(valini) != len(indexini)
            or len(valini) != len(indexfin)
        ):
            raise ValueError("valini, valfin, indexini, and indexfin must have same length")
        if len(valini) == 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        start = np.zeros(len(valini), dtype=int)
        end = np.zeros(len(valini), dtype=int)

        for i in range(len(valini)):
            index1 = np.where(
                (todarray < valini[i] + self.tol) & (todarray > valini[i] - self.tol)
            )[0]
            index2 = np.where(
                (todarray < valfin[i] + self.tol) & (todarray > valfin[i] - self.tol)
            )[0]

            if len(index1) == 0 or len(index2) == 0:
                tol2 = self.TOL_MULTIPLIER * self.tol
                index1 = np.where((todarray < valini[i] + tol2) & (todarray > valini[i] - tol2))[0]
                index2 = np.where((todarray < valfin[i] + tol2) & (todarray > valfin[i] - tol2))[0]

            # Find end index after the final level index
            candidates_end = index2[index2 > indexfin[i]]
            if len(candidates_end) > 0:
                end[i] = candidates_end[0]
            else:
                end[i] = min(indexfin[i] + 1, len(todarray))

            # Find start index between initial and end indices
            candidates_start = index1[(index1 < end[i]) & (index1 > indexini[i])]
            if len(candidates_start) > 0:
                start[i] = np.max(candidates_start)
            else:
                start[i] = min(indexini[i] + 1, len(todarray))

            # Avoid overlap with previous transition
            if len(valini) > 1 and i > 0:
                if end[i - 1] == start[i]:
                    start[i] += 1

        return start, end

    def change_values(
        self, xc: Union[np.ndarray, list], xcf: Union[np.ndarray, list], max_gap: int = 10
    ):
        """Merge consecutive transitions that are close together.

        Groups transitions that are within max_gap samples of each other into
        a single transition region.

        Parameters:
            xc: Array or list of start indices.
            xcf: Array or list of end indices.
            max_gap: Maximum gap between transitions to consider them consecutive (default: 10).

        Returns:
            Tuple containing:
                - xc2: List of merged start indices.
                - xcf2: List of merged end indices.
        """
        xc = np.array(xc)
        xcf = np.array(xcf)

        if len(xc) != len(xcf):
            raise ValueError("xc and xcf must have same length")
        if len(xc) == 0:
            return [], []

        order = np.argsort(xc)
        xc = xc[order]
        xcf = xcf[order]

        xc2 = []
        xcf2 = []

        i = 0
        while i < len(xc):
            xini = xc[i]
            xfin = xcf[i]
            j = i + 1

            # Group while transitions are within the margin
            while j < len(xc) and xc[j] - xfin <= max_gap:
                xfin = xcf[j]
                j += 1

            xc2.append(xini)
            xcf2.append(xfin)
            i = j

        return xc2, xcf2

    def calculate_levels(self, tt: np.ndarray, todarray: np.ndarray, nc: int, consec: bool = True):
        """Calculate level transitions using decision tree analysis.

        This is the main method that performs the complete level calculation pipeline:
        1. Fits a decision tree model to identify discrete levels
        2. Finds unique levels and filters by count
        3. Identifies transitions between levels based on amplitude
        4. Calculates start and end indices for each transition
        5. Optionally merges consecutive transitions

        Parameters:
            tt: 1D array of time values.
            todarray: 1D array containing the time-ordered data.
            nc: Number of flux jumps (used to determine tree depth if depth=True).
            consec: If True, merge consecutive transitions that are close together (default: True).

        Returns:
            Tuple containing:
                - xc_unique: List of start indices for level transitions.
                - xcf_unique: List of end indices for level transitions.
        """
        if tt.ndim != 1 or todarray.ndim != 1:
            raise ValueError("tt and todarray must be 1D arrays")
        if len(tt) != len(todarray):
            raise ValueError(
                f"tt and todarray must have same length, got {len(tt)} and {len(todarray)}"
            )
        if nc < 0:
            raise ValueError("nc must be non-negative")

        # Determine tree depth
        if not self.depth:
            num = self.depth_number
        else:
            num = nc

        # Fit decision tree model
        ypred = self.define_model(tt, todarray, num)

        # Find unique levels and filter
        ypred_unique, index, count = self.uniqueindex(ypred)
        ypred_unique, index, count = self.count_filter(ypred_unique, index, count)

        # Find level transitions based on amplitude
        amplitude, valini, valfin, indexini, indexfin = self.amplitude_filter(
            ypred_unique, index, count
        )

        # Calculate start and end indices for transitions
        xc, xcf = self.calculate_start_end2(todarray, valini, valfin, indexini, indexfin)

        # Optionally merge consecutive transitions
        if consec:
            xc_unique, xcf_unique = self.change_values(xc, xcf)
        else:
            xc_unique, xcf_unique = xc.tolist(), xcf.tolist()

        return xc_unique, xcf_unique


# ================================================================================
# Metrics functions
# ================================================================================


def compute_residual_jumps_with_z(signal, jump_starts, jump_ends, window=100, robust=False):
    """
    Compute residual jump amplitudes and z-scores.

    Parameters
    ----------
    signal : array-like
        Corrected TOD.
    jump_starts : array-like
        Start indices of jumps.
    jump_ends : array-like
        End indices of jumps.
    window : int
        Samples used before and after jump.
    robust : bool
        If True, use median and MAD for robustness.

    Returns
    -------
    residuals : np.ndarray
    z_scores : np.ndarray
    rms_residual : float
    """

    signal = np.asarray(signal)

    residuals = []
    z_scores = []
    sigma = []

    for start, end in zip(jump_starts, jump_ends):
        if start - window < 0:
            continue
        if end + window >= len(signal):
            continue

        before = signal[start - window : start]
        after = signal[end : end + window]

        if robust:
            mu_before = np.median(before)
            mu_after = np.median(after)

            # MAD → robust std estimate
            sigma_before = 1.4826 * np.median(np.abs(before - mu_before))
            sigma_after = 1.4826 * np.median(np.abs(after - mu_after))
        else:
            mu_before = np.mean(before)
            mu_after = np.mean(after)

            sigma_before = np.std(before, ddof=1)
            sigma_after = np.std(after, ddof=1)

        R = mu_after - mu_before

        # standard error of difference of means
        N = window
        # sigma_R = np.sqrt(sigma_before**2 / N + sigma_after**2 / N)
        sigma_local = np.sqrt((sigma_before**2 + sigma_after**2) / 2)

        if sigma_local > 0:
            Z = R / sigma_local
        else:
            Z = np.nan

        residuals.append(R)
        z_scores.append(Z)
        sigma.append(sigma_local)

    residuals = np.array(residuals)
    z_scores = np.array(z_scores)
    sigma = np.array(sigma)

    if len(residuals) > 0:
        rms_residual = np.sqrt(np.mean(residuals**2))
    else:
        rms_residual = np.nan

    return residuals, z_scores, sigma


def compute_residual_metrics_from_results(
    results, todarray, window=100, robust=False, use_dt=True, use_corrected=True
):
    """
    Compute residual jump metrics for all TES using the results dictionary.

    Parameters
    ----------
    results : dict
        Results dictionary produced by the flux jump analysis.
        Expected keys include:
        - 'TES_yes', 'TES_yes_dt'
        - 'jump_data', 'dt_jump_data'
        - 'corrected_data' (dict of corrected TOD per TES index)
    todarray : np.ndarray
        Original TOD array of shape (n_TES, n_samples).
    window : int
        Samples used before and after each jump.
    robust : bool
        If True, use median and MAD for robustness.
    use_dt : bool
        If True, use DT-refined jumps ('dt_jump_data' and 'TES_yes_dt').
        If False, use Haar jumps ('jump_data' and 'TES_yes').
    use_corrected : bool
        If True, use corrected TOD from results['corrected_data'] when available;
        otherwise fall back to the original TOD in todarray.

    Returns
    -------
    metrics_per_tes : dict
        Dictionary keyed by TES index:
        {idx: {'residuals': np.ndarray,
               'z_scores': np.ndarray,
               'sigma': np.ndarray}}
    global_rms_residual : float
        RMS of all residuals concatenated over TES (np.nan if none).
    """

    # Select which jump information to use
    if use_dt and "TES_yes_dt" in results and "dt_jump_data" in results and "corrected_data_dt":
        tes_list = results.get("TES_yes_dt", [])
        jump_dict = results.get("dt_jump_data", {})
        corrected_data = results.get("corrected_data_dt", {})
        start_key = "xcdt"
        end_key = "xcfdt"
    else:
        tes_list = results.get("TES_yes", [])
        jump_dict = results.get("jump_data", {})
        corrected_data = results.get("corrected_data_nodt", {})
        start_key = "xc"
        end_key = "xcf"

    metrics_per_tes = {}
    all_residuals = []

    for idx in tes_list:
        # Choose signal: corrected TOD if available and requested, else original
        if use_corrected and idx in corrected_data:
            signal = np.asarray(corrected_data[idx])
        else:
            signal = np.asarray(todarray[idx])

        if idx not in jump_dict:
            continue

        jump_info = jump_dict[idx]
        starts = np.asarray(jump_info.get(start_key, []), dtype=int)
        ends = np.asarray(jump_info.get(end_key, []), dtype=int)

        if len(starts) == 0 or len(ends) == 0:
            continue

        residuals, z_scores, sigma = compute_residual_jumps_with_z(
            signal,
            starts,
            ends,
            window=window,
            robust=robust,
        )

        metrics_per_tes[idx] = {
            "residuals": residuals,
            "z_scores": z_scores,
            "sigma": sigma,
        }

        if len(residuals) > 0:
            all_residuals.append(residuals)

    if all_residuals:
        all_residuals = np.concatenate(all_residuals)
        global_rms_residual = float(np.sqrt(np.mean(all_residuals**2)))
    else:
        global_rms_residual = np.nan

    return metrics_per_tes, global_rms_residual


# ================================================================================
# Plotting functions
# ================================================================================


def plot_no_jumps(tt, todarray, results):
    # ============================================================================
    # plot TES without jumps
    # ============================================================================

    """
    Parameters:
    -----------
    results : dict
        Results dictionary from analyzed dataset
    todarray : np.ndarray
        Original time-ordered data array
    tt : np.ndarray
        Time axis

    """

    TES_no = results["TES_no"]

    if len(TES_no) > 0:
        n_plot = len(TES_no)
        n_cols = 5
        n_rows = int(np.ceil(n_plot / n_cols))

        fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        if n_rows == 1:
            ax = ax.reshape(1, -1)
        ax = ax.flatten()

        for i, idx in enumerate(TES_no[:n_plot]):
            ax[i].plot(tt / 60, todarray[idx], "b-", linewidth=0.5)
            ax[i].set_title(f"TES {idx + 1} (No jumps)", fontsize=10)
            ax[i].set_xlabel("Time (min)")
            ax[i].set_ylabel("Flux")
            ax[i].grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_plot, len(ax)):
            ax[i].axis("off")

        plt.suptitle(
            f"TES without Flux Jumps (showing {n_plot} of {len(TES_no)})",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    return 0


def plot_jump_detections(tt, todarray, results, DT=True):
    # ============================================================================
    # Plot TES with jumps
    # ============================================================================

    """
    Parameters:
    -----------
    results : dict
        Results dictionary from analyzed dataset
    todarray : np.ndarray
        Original time-ordered data array
    tt : np.ndarray
        Time axis

    """

    TES_yes = results["TES_yes"]
    if DT:
        jump_data = results["dt_jump_data"]
    else:
        jump_data = results["jump_data"]

    if len(TES_yes) > 0:
        n_plot = len(TES_yes)
        n_cols = 4
        n_rows = int(np.ceil(n_plot / n_cols))

        fig, ax = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        if n_rows == 1:
            ax = ax.reshape(1, -1)
        ax = ax.flatten()

        plot_idx = 0
        for idx in TES_yes[:n_plot]:
            if idx in jump_data:
                # Original data with jump markers
                ax[plot_idx].plot(
                    tt / 60, todarray[idx], "b-", linewidth=0.5, label="Original", alpha=0.7
                )
                if DT:
                    xc = jump_data[idx]["xcdt"]
                    xcf = jump_data[idx]["xcfdt"]
                    nc = jump_data[idx]["ncdt"]
                else:
                    xc = jump_data[idx]["xc"]
                    xcf = jump_data[idx]["xcf"]
                    nc = jump_data[idx]["nc"]
                if len(xc) > 0:
                    ax[plot_idx].scatter(
                        tt[xc] / 60,
                        todarray[idx][xc],
                        color="red",
                        marker="o",
                        s=30,
                        label="Jump start",
                        zorder=5,
                    )
                    ax[plot_idx].scatter(
                        tt[xcf] / 60,
                        todarray[idx][xcf],
                        color="green",
                        marker="s",
                        s=30,
                        label="Jump end",
                        zorder=5,
                    )
                ax[plot_idx].set_title(f"TES {idx + 1} ({nc} jumps)", fontsize=10)
                ax[plot_idx].set_xlabel("Time (min)")
                ax[plot_idx].set_ylabel("Flux")
                ax[plot_idx].legend(fontsize=8)
                ax[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, len(ax)):
            ax[i].axis("off")

        plt.suptitle(
            f"TES with Flux Jumps (showing {n_plot} of {len(TES_yes)})",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    return 0


def plot_corrections(tt, todarray, results, DT=True):
    # ============================================================================
    # Plot TES with jumps corrected
    # ============================================================================

    """
    Parameters:
    -----------
    results : dict
        Results dictionary from analyzed dataset
    todarray : np.ndarray
        Original time-ordered data array
    tt : np.ndarray
        Time axis

    """

    TES_yes = results["TES_yes"]
    if DT:
        corrected_data = results["corrected_data_dt"]
    else:
        corrected_data = results["corrected_data_nodt"]

    if len(TES_yes) > 0:
        n_plot = len(TES_yes)
        n_cols = 4
        n_rows = int(np.ceil(n_plot / n_cols))

        fig, ax = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        if n_rows == 1:
            ax = ax.reshape(1, -1)
        ax = ax.flatten()

        plot_idx = 0
        for idx in TES_yes[:n_plot]:
            # Corrected data
            if idx in corrected_data:
                ax[plot_idx].plot(
                    tt / 60, corrected_data[idx], "r-", linewidth=0.8, label="Corrected", alpha=0.8
                )

                ax[plot_idx].set_title(f"TES {idx + 1}", fontsize=10)
                ax[plot_idx].set_xlabel("Time (min)")
                ax[plot_idx].set_ylabel("Flux")
                ax[plot_idx].legend(fontsize=8)
                ax[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, len(ax)):
            ax[i].axis("off")

        plt.suptitle(
            f"TES with Flux Jumps Corrected (showing {n_plot} of {len(TES_yes)})",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    return 0


# ================================================================================
# saving functions
# ================================================================================


def save_results(
    results, output_dir="./results_15_26_08", save_format="pickle", dataset_name="15.26.08"
):
    """
    Save analysis results to disk in pickle format.

    Parameters:
    -----------
    results : dict
        Results dictionary from analyzed dataset
    todarray : np.ndarray
        Original time-ordered data array
    tt : np.ndarray
        Time axis
    output_dir : str
        Directory where to save results (default: "./results_15_26_08")
    save_format : str
        Format to save: pickle"
    dataset_name : str
        Name of the dataset for file naming (default: "15.26.08")

    Returns:
    --------
    saved_files : list
        List of paths to saved files
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    base_name = f"{dataset_name}"
    saved_files = []

    # ============================================================================
    # Save format: Pickle format (.pkl)
    # ============================================================================
    if save_format in ["pickle"]:
        pkl_file = os.path.join(output_dir, f"{base_name}_results.pkl")

        # Prepare complete results dictionary
        save_data = {"results": results, "dataset_name": dataset_name}

        with open(pkl_file, "wb") as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        saved_files.append(pkl_file)
        print(f"Saved pickle file to: {pkl_file}")

    print(f"\n{'=' * 70}")
    print(f"All results saved to: {output_dir}")
    print(f"Total files saved: {len(saved_files)}")
    print(f"{'=' * 70}\n")

    return saved_files


def load_results(load_file, load_format="pickle"):
    """
    Load previously saved analysis results.

    Parameters:
    -----------
    load_file : str
        Path to the saved file
    load_format : str
        Format type: "pickle" (default: "pickle")

    Returns:
    --------
    loaded_data : dict
        Loaded results dictionary
    """

    if load_format == "pickle":
        with open(load_file, "rb") as f:
            loaded_data = pickle.load(f)
        return loaded_data

    else:
        raise ValueError(f"Unsupported load format: {load_format}")
