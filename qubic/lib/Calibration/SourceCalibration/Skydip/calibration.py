import os
import logging

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from numpy.polynomial.polynomial import Polynomial

import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from qubicpack.qubicfp import qubicfp
from qubicpack.tools import read_qubicstudio_dataset
from tqdm import tqdm
from qubic.lib.Calibration.Qfiber import image_asics, qgrid, pix_grid
from qubicpack.pix2tes import pix2tes
from matplotlib.colors import Normalize

plt.rcParams['text.usetex'] = True
plt.rcParams['figure.facecolor'] = '#F5F5F5'

class Calibration:

    def __init__(self,
                 signals_fname: Path,
                 skydips_fname: Path,
                 dest: Path,
                 tau_atm: float,
                 T_atm: float):

        self.signals_to_calibrate_fname = signals_fname # non serve
        self.skydips_fname = skydips_fname
        self.dest = dest
        self.tau_atm = tau_atm
        self.T_atm = T_atm

        self.logger: logging.Logger | None = None
        self.logfname: Path | None = None         # non serve, si usa __name__

        self.input_dir: Path | None = None        # non serve
        self.output_dir: Path | None = None
        self.plots_dir: Path | None = None
        self.observation_dir: Path | None = None

        # ATTRIBUTI DEI SEGNALI DA CALIBRARE
        self.tm_hk = np.ndarray
        self.tm_ = np.ndarray
        self.elevation = np.ndarray
        self.interp_elevation = np.ndarray
        self.signals = np.ndarray
        self.airmass = np.ndarray


        # ATTIRBUTI DEGLI SKYDIP
        self.tm_hk_skydip = np.ndarray
        self.tm_skydip = np.ndarray
        self.tm_hk_or_pt_skydip = np.ndarray
        self.elevation_skydip = np.ndarray
        self.interp_elevation_skydip = np.ndarray
        self.skydip_signals = np.ndarray
        self.airmass_skydip = np.ndarray



        self.resps = np.ndarray
        self.resps_fname = ''  # non serve. metodo di classe "from_file"


    # superflua
    def make_result_dir(self):

        # Build the folder name using a prefix plus the base name of the source path
        folder = f"skydip_calibration_{Path(self.signals_to_calibrate_fname).name}"

        # Combine the destination path and the folder name to get the full results directory path
        self.observation_dir = self.dest / folder

        # Create the results directory (if it exists, no exception is raised)
        os.makedirs(self.observation_dir, exist_ok=True)

        self.input_dir = self.observation_dir / "input"
        self.output_dir = self.observation_dir / "output"

        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self.plots_dir = self.output_dir / "plots"

        os.makedirs(self.plots_dir, exist_ok=True)

        # Construct the full path to the log file
        self.logfname = self.observation_dir / f"{folder}.log"

        # Configure the logging module so that a new log file is created per dataset
        # force = True ensures that one logfile is created for each dataset analysed

        # logger definito dentro __init__
        logging.basicConfig(filename=self.logfname,
                            filemode="w",
                            encoding='utf8',
                            format="%(asctime)s - %(funcName)s - %(message)s",
                            datefmt="%d/%m/%Y | %H:%M:%S",
                            level=logging.INFO,
                            force=True)

        # Create a logger for this module
        self.logger = logging.getLogger(__name__)

        # Kelvin-per-ADU output filename as CSV
        self.kelvin_per_adu_fname = self.output_dir / "kelvin_per_adu.csv"

    # non fa parte della calibrazione
    def read_data(self):

        print("Exporting data via qubicfp...", end="")

        def _axis_is_valid(axis, elev):
            """
            True if axis exists, it's not empty and has the same shape of elev
            """
            return axis is not None and len(axis) != 0 and axis.shape == elev.shape

        # ESPORTO SEGNALI DA CALIBRARE
        qubic_tods = qubicfp()
        # qubic_tods.verbosity = 0
        qubic_tods.read_qubicstudio_dataset(str(self.signals_to_calibrate_fname))
        self.tm_, self.signals = qubic_tods.tod()
        self.tm_ -= self.tm_[0]

        self.logger.info("Shape of signals to calibrate: %s", self.signals.shape)

        self.elevation = qubic_tods.elevation()

        # # np.astype is used to resolve mismatch with Endianess
        # self.elevation_signals = self.elevation_signals.astype(np.float64) if self.elevation_signals is not None else np.zeros(1)

        # Attempt to read housekeeping (HK) and platform time data
        self.tm_hk = qubic_tods.timeaxis(datatype='hk')
        t_plt = qubic_tods.timeaxis(datatype='platform')

        if _axis_is_valid(self.tm_hk, self.elevation):
            t_sel = self.tm_hk
            axis_name = "HK"

        elif _axis_is_valid(t_plt, self.elevation):
            t_sel = t_plt
            axis_name = "platform"

        else:
            msg = ("SIGNALS TO CALIBRATE: Neither HK nor platform time axis is usable: missing/empty or "
                   "shape mismatch with elevation.")
            self.logger.error(msg)
            raise RuntimeError(msg)

        t_sel -= t_sel[0]
        self.interp_elevation = np.interp(self.tm_, t_sel, self.elevation)
        del qubic_tods

        # ESPORTO SKYDIP
        qubic_skydip = qubicfp()
        # qubic_skydip.verbosity = 0

        # Read QubicStudio dataset from the directory specified by self.source
        qubic_skydip.read_qubicstudio_dataset(str(self.skydips_fname))

        # Retrieve the time axis (tm) and the corresponding TES signals
        self.tm_skydip, self.skydip_signals = qubic_skydip.tod()
        self.tm_skydip -= self.tm_skydip[0]

        self.logger.info("Shape of skydip signals: %s", self.skydip_signals.shape)

        self.elevation_skydip = qubic_skydip.elevation()

        # # np.astype is used to resolve mismatch with Endianess
        # self.elevation_skydip = self.elevation_skydip.astype(np.float64) if self.elevation_skydip is not None else np.zeros(1)

        # Attempt to read housekeeping (HK) and platform time data
        self.tm_hk_skydip = qubic_skydip.timeaxis(datatype='hk')
        t_plt_skydip = qubic_skydip.timeaxis(datatype='platform')

        if _axis_is_valid(self.tm_hk_skydip, self.elevation_skydip):
            t_sel = self.tm_hk_skydip
            axis_name = "HK"

        elif _axis_is_valid(t_plt_skydip, self.elevation_skydip):
            t_sel = t_plt_skydip
            axis_name = "platform"

        else:
            msg = ("SKYDIP: Neither HK nor platform time axis is usable: missing/empty or "
                   "shape mismatch with elevation.")
            self.logger.error(msg)
            raise RuntimeError(msg)

        t_sel -= t_sel[0]
        self.tm_hk_or_pt_skydip = t_sel.copy()
        self.interp_elevation_skydip = np.interp(self.tm_skydip, t_sel, self.elevation_skydip)



    # metodo che deve accettare time, tods, elevation dei segnali da calibrare
    def calibrate(self):

        z = np.pi / 2 - np.deg2rad(self.interp_elevation_skydip)
        self.airmass_skydip = 1 / np.cos(z)
        # modello atmosferico dello skydip
        sky_temp = self.T_atm * (1 - np.exp(-self.tau_atm * self.airmass_skydip))

        den = (sky_temp * sky_temp).sum()

        self.logger.info("sky_temp: %s. skydip_signals: %s", sky_temp.shape, self.skydip_signals.shape)

        # stima della responsivity per ogni TES confrontando il suo segnale con il template atmosferico
        self.resps = (self.skydip_signals @ sky_temp) / den
        self.logger.info("responsivities saved in: %s", self.resps_fname)
        np.save(self.resps_fname, self.resps)

        self.logger.info("Signals to calibrate: %s. Resps: %s", self.signals.shape, self.resps.shape)

        # ricalcolo airmass sulle TOD
        z = np.pi / 2 - np.deg2rad(self.interp_elevation)
        self.airmass = 1 / np.cos(z)

        self.logger.info("airmass: %s", self.airmass.shape)

        self.logger.info("resps redimensioned: %s. airmass redimensioned: %s", self.resps[:, None].shape, self.airmass[None, :].shape)

        # Save the detector-level conversion in K / ADU as a CSV file.
        self.kelvin_per_adu = 1.0 / self.resps
        self.logger.info("kelvin per ADU saved in: %s", self.kelvin_per_adu_fname)
        np.savetxt(
            self.kelvin_per_adu_fname,
            self.kelvin_per_adu,
            delimiter=",",
            header="kelvin_per_adu",
            comments=""
        )

        return self.signals / (self.resps[:, None] * np.exp(- self.tau_atm * self.airmass[None, :]))


    def plot_skydip(self):

        skydip_plot_dir = self.plots_dir / "skydip"
        skydip_plot_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(11, 5))

        # for tes_num in range(0):
        s = self.skydip_signals[200]
        s_norm = (s - s.min()) / (s.max() - s.min())

        ax.plot(self.tm_skydip, s_norm)
        ax.set_title(rf"$SkyDip signal$")
        ax.set_xlabel(r"$Time ~ [s]$")
        ax.set_ylabel(r"$Signal ~ [ADU]$")
        ax.set_facecolor('#F5F5F5')

        fig.savefig(skydip_plot_dir / f"skydip.png")

    def plot_skydip_secant(self, n_bins: int = 20):

        skydip_plot_dir = self.plots_dir / "skydip"
        skydip_plot_dir.mkdir(parents=True, exist_ok=True)

        # Check that the skydip airmass and TES signals are available.
        if len(self.airmass_skydip) == 0 or len(self.skydip_signals) == 0:
            msg = "Skydip airmass or skydip signals are empty: cannot produce secant-law plots."
            self.logger.error(msg)
            raise RuntimeError(msg)

        airmass = self.airmass_skydip.astype(np.float64)

        # Build equally spaced bins in airmass once and reuse them for all TES.
        bin_edges = np.linspace(np.min(airmass), np.max(airmass), n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Build the atmospheric model grid once and reuse it for all TES.
        model_airmass = np.linspace(np.min(airmass), np.max(airmass), 300)
        model_template = self.T_atm * (1 - np.exp(-self.tau_atm * model_airmass))

        for tes_num in range(self.skydip_signals.shape[0]):

            # Select the TES signal to analyze.
            tes_signal = self.skydip_signals[tes_num].astype(np.float64)

            binned_airmass = []
            binned_signal = []
            binned_err = []

            # For each airmass bin, compute an estimate of the signal
            # using the median, and estimate the uncertainty from the scatter.
            for left, right, center in zip(bin_edges[:-1], bin_edges[1:], bin_centers):
                mask = (airmass >= left) & (airmass < right)

                if np.count_nonzero(mask) < 5:
                    continue

                signal_chunk = tes_signal[mask]

                binned_airmass.append(center)
                binned_signal.append(np.median(signal_chunk))
                binned_err.append(np.std(signal_chunk) / np.sqrt(np.count_nonzero(mask)))

            binned_airmass = np.asarray(binned_airmass)
            binned_signal = np.asarray(binned_signal)
            binned_err = np.asarray(binned_err)

            if len(binned_airmass) == 0:
                self.logger.warning("TES %s: no valid airmass bins were found for the secant-law plot.", tes_num)
                continue

            # Scale the atmospheric model with the TES responsivity.
            model_signal = self.resps[tes_num] * model_template

            # Add a constant offset so the model can be compared fairly with the binned data.
            model_on_bins = np.interp(binned_airmass, model_airmass, model_signal)
            offset = np.median(binned_signal - model_on_bins)
            model_signal_shifted = model_signal + offset

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.errorbar(binned_airmass, binned_signal, yerr=binned_err, fmt='o', label="Binned skydip")
            ax.plot(model_airmass, model_signal_shifted, label="Atmospheric model + offset")
            ax.set_title(rf"$SkyDip ~ secant ~ test ~ TES {tes_num}$")
            ax.set_xlabel(r"$Airmass$")
            ax.set_ylabel(r"$Signal ~ [ADU]$")
            ax.set_facecolor('#F5F5F5')
            ax.legend()

            fig.tight_layout()
            fig.savefig(skydip_plot_dir / f"skydip_secant_tes_{tes_num}.png")
            plt.close(fig)

    def plot_raw_skydip(self):

        skydip_plot_dir = self.plots_dir / "skydip"
        skydip_plot_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(self.tm_hk_or_pt_skydip, self.elevation_skydip)
        ax.set_title(rf"$SkyDip$")
        ax.set_xlabel(r"$Time ~ [s]$")
        ax.set_ylabel(r"$Elevation ~ [deg]$")
        ax.set_facecolor('#F5F5F5')

        fig.tight_layout()
        fig.savefig(skydip_plot_dir / f"skydip_raw.png")
        plt.close(fig)

    # non riguarda la calibrazione
    def plot_signal(self, tods_calibrated: np.ndarray, tes_num: int = 0):

        fig, (ax_raw, ax_cal) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), tight_layout=True)
        tods_plot_dir = self.plots_dir / "tods"
        tods_plot_dir.mkdir(parents=True, exist_ok=True)

        for tes_num in range(0, 255):

            ax_raw.plot(self.tm_, self.signals[tes_num], label="raw")
            ax_cal.plot(self.tm_, tods_calibrated[tes_num], label="calibrated")
            ax_raw.set_title(rf"$Raw ~ TES ~ {tes_num} ~ signal$")
            ax_raw.set_xlabel(r"$time ~ [s]$")
            ax_raw.set_ylabel(r"$Signal ~ [ADU]$")
            ax_raw.set_facecolor('#F5F5F5')

            ax_cal.set_title(rf"$Calibrated ~ TES ~ {tes_num} ~ signal$")
            ax_cal.set_xlabel(r"$time ~ [s]$")
            ax_cal.set_ylabel(r"$Signal ~ [Kelvin]$")
            ax_cal.set_facecolor('#F5F5F5')

            fig.savefig(os.path.join(tods_plot_dir / f"calibrated_tes_{tes_num}.png"))
            ax_cal.cla()
            ax_raw.cla()


    def plot_focal_plane_tod(self,
                             sample_idx: int,
                             calibrated_tods: np.ndarray | None = None,
                             centered: bool = False,
                             normalized: bool = False,
                             use_calibrated: bool = False,
                             save_name: str | None = None,
                             annotate_values: bool = True,
                             fmt: str = ".2e"):

        fp_plot_dir = self.plots_dir / "focal_plane"
        fp_plot_dir.mkdir(parents=True, exist_ok=True)

        data = calibrated_tods if use_calibrated and calibrated_tods is not None else self.signals

        if not isinstance(data, np.ndarray) or data.ndim != 2:
            msg = "TOD data must be a 2D numpy array with shape (n_tes, n_samples)."
            self.logger.error(msg)
            raise RuntimeError(msg)

        if sample_idx < 0 or sample_idx >= data.shape[1]:
            msg = f"sample_idx {sample_idx} is out of bounds for TODs with {data.shape[1]} samples."
            self.logger.error(msg)
            raise IndexError(msg)

        plot_data = data.astype(np.float64).copy()

        if centered:
            plot_data -= np.median(plot_data, axis=1, keepdims=True)

        if normalized:
            scale = np.max(np.abs(plot_data), axis=1, keepdims=True)
            scale[scale == 0] = 1.0
            plot_data /= scale

        fp_values = plot_data[:, sample_idx]
        fp_image = image_asics(all1=fp_values)

        finite_values = fp_image[np.isfinite(fp_image)]
        if finite_values.size == 0:
            msg = "No finite focal-plane values are available for plotting."
            self.logger.error(msg)
            raise RuntimeError(msg)

        vmin = np.nanmin(finite_values)
        vmax = np.nanmax(finite_values)
        if np.isclose(vmin, vmax):
            vmin -= 1.0
            vmax += 1.0

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(
            fp_image,
            origin='lower',
            interpolation='nearest',
            norm=Normalize(vmin=vmin, vmax=vmax)
        )
        qgrid()
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"$Signal$")

        if annotate_values:
            threshold = 0.5 * (vmin + vmax)
            for row in range(fp_image.shape[0]):
                for col in range(fp_image.shape[1]):
                    value = fp_image[row, col]

                    if not np.isfinite(value):
                        continue

                    text_color = "white" if value < threshold else "black"
                    ax.text(
                        col,
                        row,
                        format(value, fmt),
                        ha="center",
                        va="center",
                        fontsize=6,
                        color=text_color
                    )

        title_label = "Calibrated" if use_calibrated and calibrated_tods is not None else "Raw"
        ax.set_title(rf"${title_label} ~ focal ~ plane ~ at ~ sample ~ {sample_idx}$")
        ax.set_xlabel(r"$Column$")
        ax.set_ylabel(r"$Row$")
        ax.set_facecolor('#F5F5F5')

        fig.tight_layout()

        if save_name is None:
            save_name = f"focal_plane_sample_{sample_idx}.png"

        fig.savefig(fp_plot_dir / save_name, dpi=200)
        plt.close(fig)


    def plot_focal_plane_tods(self,
                              calibrated_tods: np.ndarray | None = None,
                              tes_to_plot: list[int] | None = None,
                              centered: bool = False,
                              normalized: bool = False,
                              use_calibrated: bool = False,
                              alpha: float = 0.8,
                              linewidth: float = 0.4,
                              save_name: str | None = None):

        fp_plot_dir = self.plots_dir / "focal_plane"
        fp_plot_dir.mkdir(parents=True, exist_ok=True)

        data = calibrated_tods if use_calibrated and calibrated_tods is not None else self.signals

        if not isinstance(data, np.ndarray) or data.ndim != 2:
            msg = "TOD data must be a 2D numpy array with shape (n_tes, n_samples)."
            self.logger.error(msg)
            raise RuntimeError(msg)

        plot_data = data.astype(np.float64).copy()

        if centered:
            plot_data -= np.median(plot_data, axis=1, keepdims=True)

        if normalized:
            scale = np.max(np.abs(plot_data), axis=1, keepdims=True)
            scale[scale == 0] = 1.0
            plot_data /= scale

        nrows, ncols = pix_grid.shape
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 18), sharex=True, sharey=True)

        for row in range(nrows):
            for col in range(ncols):
                ax = axes[row, col]
                phys_pix = pix_grid[row, col]
                tes_info = pix2tes(phys_pix)

                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_alpha(0.3)

                if tes_info is None:
                    ax.set_facecolor('#F0F0F0')
                    continue

                tes_num, asic_num = tes_info

                if tes_num is None or asic_num is None:
                    ax.set_facecolor('#F0F0F0')
                    continue

                global_tes_idx = tes_num - 1 + (asic_num - 1) * 128

                if global_tes_idx < 0 or global_tes_idx >= plot_data.shape[0]:
                    ax.set_facecolor('#F0F0F0')
                    continue

                if tes_to_plot is not None and global_tes_idx not in tes_to_plot:
                    ax.set_facecolor('white')
                    continue

                y = plot_data[global_tes_idx]
                if not np.any(np.isfinite(y)):
                    ax.set_facecolor('#F0F0F0')
                    continue

                ax.plot(y, linewidth=linewidth, alpha=alpha)
                ax.set_title(f"TES {global_tes_idx}", fontsize=6, pad=1)

        title_label = "Calibrated" if use_calibrated and calibrated_tods is not None else "Raw"
        fig.suptitle(f"{title_label} focal-plane TODs", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if save_name is None:
            save_name = "focal_plane_tods.png"

        fig.savefig(fp_plot_dir / save_name, dpi=200)
        plt.close(fig)

    def plot_hist(self):

        # Step 1: Calculate number of bins using Square-root Rule
        n = len(self.resps)
        num_bins = int(np.ceil(np.sqrt(n)))

        plt.figure(figsize=(8, 5))
        plt.hist(self.resps, bins=num_bins, color="#8172B3", edgecolor="black", alpha=0.7)
        plt.title(r"$Histogram ~ using ~ Square-root ~ Rule$", fontsize=14, weight='bold')
        plt.xlabel(r"$Responsivities$")
        plt.ylabel(r"$Occurrences$")
        plt.savefig(os.path.join(self.plots_dir, "responsivity_histogram.png"))

if __name__ == "__main__":

    skydip_fname = Path(
        "/qubic/scripts/Calibration/skydip_calibration/data/input/2026-03-11/2026-03-11_16.37.17__SkyDip")
    tods_fname = Path(
        "/qubic/scripts/Calibration/skydip_calibration/data/input/2026-03-11/2026-03-11_15.39.59__Moon_el30")
    dest = Path("/qubic/scripts/Calibration/skydip_calibration/data/output/skydip")

    calib = Calibration(tods_fname, skydip_fname, dest, tau_atm=0.1, T_atm=267)
    # calib.get_datasets()
    calib.make_result_dir()
    calib.read_data()
    calibrated_signals = calib.calibrate()

    # calib.plot_signal(tods_calibrated=calibrated_signals)
    calib.plot_skydip()
    # calib.plot_focal_plane_tod(sample_idx=1000, calibrated_tods=calibrated_signals, use_calibrated=True, annotate_values=True, fmt=".1e")
    calib.plot_focal_plane_tods(calibrated_tods=calibrated_signals, use_calibrated=True, centered=True)
    # calib.plot_skydip_secant(n_bins=20)
    # calib.plot_hist()
    # calib.plot_raw_skydip()