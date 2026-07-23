import sys
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from qubic.lib.Calibration.source_calibration.common import plotting
from qubic.lib.Calibration.source_calibration.common.utils import parse_tes_indices, ColoredFormatter
from qubic.lib.Calibration.source_calibration.common.preprocessing import SkydipIntervals
from qubic.lib.Calibration.source_calibration.skydip.atmosphere.run_atmosphere import Atmosphere
from qubic.lib.Calibration.source_calibration.skydip.atmosphere import run_atmosphere
from qubic.lib.Calibration.source_calibration.skydip.config_calibration import load_skydip_calibration_config
from qubic.lib.Calibration.source_calibration.common.io import prepare_datasets_from_config, iter_saved_datasets
from qubic.lib.Calibration.source_calibration.skydip.calibration import SkydipCalibrationSegment, DatasetConversionFactorSummary
from qubic.lib.Calibration.source_calibration.common.noise import (
    compute_all_skydip_noise_spectra,
    DatasetNoiseSummary,
    DatasetNoiseInFrequencyRange,
)
from scipy.interpolate import RegularGridInterpolator


def external_trend_function(path: str, az0: np.ndarray, el0: np.ndarray) -> np.ndarray:

    trend_map = np.load(path)

    # TODO:
    #  put the last value at 180 for TES 95
    #  put the last value at 360 for TES 127
    az_grid = np.linspace(0.0, 360.0, 360)
    el_grid = np.linspace(30.0, 70.0, 600)

    trend_map = np.asarray(trend_map, dtype=np.float64)

    if trend_map.shape == (el_grid.size, az_grid.size):
        values = trend_map.T
    elif trend_map.shape == (az_grid.size, el_grid.size):
        values = trend_map
    else:
        raise ValueError(
            "External trend map has incompatible shape. "
            f"Got {trend_map.shape}, expected "
            f"{(el_grid.size, az_grid.size)} or {(az_grid.size, el_grid.size)}."
        )

    az0 = np.asarray(az0, dtype=np.float64)
    el0 = np.asarray(el0, dtype=np.float64)

    az0_clipped = np.clip(az0, az_grid[0], az_grid[-1])
    el0_clipped = np.clip(el0, el_grid[0], el_grid[-1])

    interpolator = RegularGridInterpolator(
        points=(az_grid, el_grid),
        values=values,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    points = np.column_stack(
        [
            az0_clipped.ravel(),
            el0_clipped.ravel(),
        ]
    )

    trend = interpolator(points).reshape(az0.shape)

    if not np.all(np.isfinite(trend)):
        trend = np.nan_to_num(
            trend,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    return trend


@dataclass()
class DatasetNoiseVsTauResult:
    dataset_name: str
    tes_indices: tuple[int, ...]
    n_tes: int

    tau_eff: float
    median_conversion_factor_adu_per_k: float

    # From DatasetNoiseInFrequencyRange.from_spectra_frequency_range
    mean_plateau_asd: float
    std_plateau_asd: float

    # From DatasetNoiseSummary.from_spectra
    mean_fit_knee_freq: float
    std_fit_knee_freq: float

    mean_fit_white_noise_asd: float
    std_fit_white_noise_asd: float

    mean_fit_alpha: float
    std_fit_alpha: float

    mean_fit_cutoff_frequency_hz: float
    std_fit_cutoff_frequency_hz: float


def main(config_path: Path):

    # instanzio un logger associato al nome del modulo corrente
    logger = logging.getLogger(__name__)
    # il logger deve processare solo messaggi da livello INFO in su
    logger.setLevel(logging.INFO)
    # rimuovi eventuali handler già attaccati a quel logger
    #  per evitare che, se main() viene chiamata più volte nella
    #  stessa sessione Python, i messaggi vengano stampati duplicati
    logger.handlers.clear()
    # Se propagate=True, cioè il default, il messaggio non viene gestito
    # solo dagli handler che hai aggiunto tu, ma può essere passato anche
    # agli handler dei logger superiori
    logger.propagate = False

    # creo l’handler per stampare nel terminale
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # setto come devono apparire i messaggi nel terminale
    console_handler.setFormatter(ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s",
                                                  datefmt="%d/%m/%Y | %H:%M:%S"))
    # collego il terminale al logger
    logger.addHandler(console_handler)

    # leggo file di configurazione
    config = load_skydip_calibration_config(config_path)

    # salvataggio logfile
    log_file = Path(config.paths.runs[0].output).parents[1] / "run_calibration.log"
    # check cartella che conterra' il file log esista
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # creo l'handler per salvare i messaggi nel file log
    # mode: "w" apre il file in modalità scrittura e
    # lo sovrascrive ad ogni run di run_calibration.py
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    # setto come devono apparire i messaggi nel file log
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                                                datefmt="%d/%m/%Y | %H:%M:%S"))
    # collego il file di log al logger
    logger.addHandler(file_handler)

    # Seguo lo stesso ordine di operazioni di io.py:
    # leggo tutti i dataset QUBICStudio definiti in paths.runs
    # e salvo i corrispondenti file .npy/.npz nei rispettivi output
    prepare_datasets_from_config(config=config,
                                 logger=logger)

    noise_vs_tau_results: list[DatasetNoiseVsTauResult] = []

    # Ora rileggo i file .npy/.npz appena prodotti,
    # un dataset alla volta, usando il generatore
    # dataset e' un oggetto QubicDataset ricostruito dai file .npy e .npz
    # run_paths e' uno degli elementi di config.paths.runs, cioe' ciascun run_paths contiene le tre paths:
    # run_paths.dataset, run_paths.output e run_paths.atmospheric_config presenti nel file di config
    for dataset, run_paths in zip(iter_saved_datasets(config=config, logger=logger), config.paths.runs):

        logger.info("Dataset name: %s", dataset.dataset_name)
        logger.info("Time range UTC: %s -> %s", dataset.start_time_utc, dataset.stop_time_utc)

        # Plot interpolato di azimuth/elevation.
        # Dentro plot_az_el_vs_time, se is_interpolated=True,
        # vengono mostrati automaticamente i constant-azimuth blocks
        plotting.plot_az_el_vs_time(dataset=dataset,
                                    config=config,
                                    is_interpolated=True)

        # Plot del focal plane completo.
        # Ogni subplot corrisponde a un TES nella sua posizione fisica.
        plotting.plot_focal_plane_tods(dataset=dataset,
                                       config=config,
                                       flip_ud=False,
                                       flip_lr=False)

        # Parso e salvo gli indici dei TES contenuti nel config
        tes_indices = parse_tes_indices(tes_indices=config.calibration.tes_indices, n_tes=dataset.signals.shape[0])
        logger.info("Tes to analyze: %s", tes_indices)

        try:
            atmosphere = Atmosphere(config=run_paths.atmospheric_config,
                                    start_obs=dataset.start_time_utc,
                                    end_obs=dataset.stop_time_utc,
                                    dataset_name=dataset.dataset_name)

        except ValueError as exc:
            logger.warning("Skipping dataset %s because atmospheric parameters could not be computed: %s",
                           dataset.dataset_name,
                           exc)
            continue

        dataset_plateau_asds = []
        dataset_plateau_stds = []
        dataset_conversion_factors_adu_per_k = []
        analyzed_tes_indices = []

        dataset_fit_knee_freqs = []
        dataset_fit_white_noise_asds = []
        dataset_fit_alphas = []
        dataset_fit_cutoff_frequencies_hz = []

        for tes_idx in tes_indices:

            logger.info("-" * 15)
            logger.info("Analyzing TES %s", tes_idx)
            logger.info("-" * 15)

            # TODO: DECONMENT TO REMOVE TREND AND ADD "NEW" TO THE
            #  OUTPUT FILES IN THE CONFIG YAML
            # trend = external_trend_function(path="blind_signal_127.npy",
            #                                 az0=dataset.interp_azimuth,
            #                                 el0=dataset.interp_elevation)
            #
            # dataset.signals[tes_idx, :] -= trend

            # Plot semplice dei TOD selezionati nel config
            plotting.plot_tod(dataset=dataset, config=config, tes_idx=tes_idx)

            # Trovo gli intervalli skydip per il dataset corrente
            skydip_intervals = SkydipIntervals.from_dataset(dataset=dataset, config=config)

            if len(skydip_intervals.idx_pairs) == 0:
                logger.warning(
                    "Skipping dataset %s because no valid skydip intervals were found.",
                    dataset.dataset_name,
                )
                continue

            logger.info("Found %d skydips for dataset %s",
                        len(skydip_intervals.idx_pairs),
                        dataset.dataset_name)

            logger.info("Found %d constant-azimuth blocks for dataset %s",
                        len(skydip_intervals.azimuth_blocks),
                        dataset.dataset_name)

            # Plot dell'elevation interpolata e non con overlay dei limiti skydip.
            # Questo serve per verificare visivamente che gli intervalli trovati
            # corrispondano davvero alle salite/discese in elevation.
            plotting.plot_elevation_with_skydip_limits(dataset=dataset,
                                                       config=config,
                                                       skydip_intervals=skydip_intervals)

            # Plot dei TOD selezionati con overlay dei limiti skydip
            plotting.plot_tod(dataset=dataset, config=config, tes_idx=tes_idx, skydip_intervals=skydip_intervals)

            tod_segments = skydip_intervals.extract_tod_segments(
                dataset=dataset,
                tes_idx=tes_idx,
            )

            if len(tod_segments) == 0:
                logger.warning(
                    "Skipping dataset %s, TES %s because no TOD skydip segments were found.",
                    dataset.dataset_name,
                    tes_idx,
                )
                continue

            # TODO: nome fuorviante in quanto non ritorna i segmenti calibrati
            calibrated_segments = SkydipCalibrationSegment.from_tod_segments(
                tod_segments=tod_segments,
                dataset=dataset,
                tau_eff=atmosphere.tau,
                tb_eff_k=atmosphere.T_b,
            )

            conversion_summary = DatasetConversionFactorSummary(segments=calibrated_segments)
            conversion_factors = conversion_summary.save_to_csv(tes_idx=tes_idx, dataset=dataset)
            median_conversion_factor = conversion_summary.median_adu_per_k

            logger.info(
                "Median conversion factor for dataset %s, TES %s: %.6g ADU/K",
                dataset.dataset_name,
                tes_idx,
                median_conversion_factor,
            )


            plotting.plot_skydips_vs_Tatm(segments=calibrated_segments,
                                          config=config,
                                          tes_idx=tes_idx,
                                          output_dir=dataset.calibration_plots_dir)

            plotting.plot_Tatm_vs_airmass(segments=calibrated_segments,
                                          config=config,
                                          tes_idx=tes_idx,
                                          output_dir=dataset.calibration_plots_dir)

            plotting.plot_conversion_factors_histogram(csv_file=conversion_factors,
                                                       config=config,
                                                       tes_idx=tes_idx,
                                                       output_dir=dataset.calibration_plots_dir)

            #TODO: calcolo delle ASD per tutti i segmenti calibrati e non calibrati (skydip_intervals).
            # Quindi la funzione che calcola l'ASD deve prendere semplicemente degli intervalli (skydip_intervals
            # o calibrated_segments).
            # Plot degli spettri per tutti i segments in un unico canvas e, usando la lista di tau e magari un'altra
            # lista per i valori di ASD ad una data frequenza (per tutti i dataset), plottare i valori di ASD
            #  vs tau

            raw_noise_spectra = compute_all_skydip_noise_spectra(segments=tod_segments, config=config)

            raw_noise_fit_summary = DatasetNoiseSummary.from_spectra(
                noise_spectra=raw_noise_spectra,
                tau_eff=atmosphere.tau,
                asd_unit="ADU",
            )

            plotting.plot_skydip_noise_spectra(
                noise_spectra=raw_noise_spectra,
                config=config,
                tes_idx=tes_idx,
                output_dir=dataset.noise_plots_dir,
                is_calibrated=False,
                fit_white_noise_asd=raw_noise_fit_summary.mean_white_noise_asd,
                fit_knee_frequency_hz=raw_noise_fit_summary.mean_knee_frequency_hz,
                fit_alpha=raw_noise_fit_summary.mean_alpha,
                fit_cutoff_frequency_hz=raw_noise_fit_summary.mean_cutoff_frequency_hz,
            )

            # posso passargli tod_segments in quanto la calibrazione la fa al suo interno
            calibrated_noise_spectra = compute_all_skydip_noise_spectra(segments=tod_segments,
                                                                        config=config,
                                                                        conversion_factor_adu_per_k=median_conversion_factor)

            # da noise fit summary voglio ottenere:
            # frequenza di ginocchio media
            # ASD nella regione white noise
            # alpha
            noise_fit_summary = DatasetNoiseSummary.from_spectra(
                noise_spectra=calibrated_noise_spectra,
                tau_eff=atmosphere.tau,
                asd_unit="K",
            )

            plotting.plot_skydip_noise_spectra(
                noise_spectra=calibrated_noise_spectra,
                config=config,
                tes_idx=tes_idx,
                output_dir=dataset.noise_plots_dir,
                is_calibrated=True,
                fit_white_noise_asd=noise_fit_summary.mean_white_noise_asd,
                fit_knee_frequency_hz=noise_fit_summary.mean_knee_frequency_hz,
                fit_alpha=noise_fit_summary.mean_alpha,
                fit_cutoff_frequency_hz=noise_fit_summary.mean_cutoff_frequency_hz)

            # dal noise range summary voglio ottenere l'ASD media nel range di frequenze e la sua std
            noise_range_summary = DatasetNoiseInFrequencyRange.from_spectra_frequency_range(
                noise_spectra=calibrated_noise_spectra,
                config=config,
                tes_idx=tes_idx)

            dataset_plateau_asds.append(noise_range_summary.mean_asd_k_per_sqrt_hz)
            dataset_plateau_stds.append(noise_range_summary.std_asd_k_per_sqrt_hz)

            dataset_fit_knee_freqs.append(noise_fit_summary.mean_knee_frequency_hz)
            dataset_fit_white_noise_asds.append(noise_fit_summary.mean_white_noise_asd)
            dataset_fit_alphas.append(noise_fit_summary.mean_alpha)
            dataset_fit_cutoff_frequencies_hz.append(
                noise_fit_summary.mean_cutoff_frequency_hz
            )

            # Lista dei conversion factor per tutti i TES analizzati
            dataset_conversion_factors_adu_per_k.append(conversion_summary.dataset_conversion_factor_adu_per_k)
            analyzed_tes_indices.append(int(tes_idx))

        if analyzed_tes_indices:
            # ottenuti da noise range summary
            plateau_values = np.asarray(dataset_plateau_asds, dtype=np.float64)
            plateau_stds = np.asarray(dataset_plateau_stds, dtype=np.float64)

            # ottenuti da dataset noise summary
            fit_knee_values = np.asarray(dataset_fit_knee_freqs, dtype=np.float64)
            fit_white_noise_asds = np.asarray(dataset_fit_white_noise_asds, dtype=np.float64)
            fit_alphas = np.asarray(dataset_fit_alphas, dtype=np.float64)
            fit_cutoff_frequencies_hz = np.asarray(
                dataset_fit_cutoff_frequencies_hz,
                dtype=np.float64,
            )

            conversion_factor_values = np.asarray(dataset_conversion_factors_adu_per_k, dtype=np.float64)

            dataset_noise_result = DatasetNoiseVsTauResult(
                dataset_name=dataset.dataset_name,
                tes_indices=tuple(analyzed_tes_indices),
                n_tes=len(analyzed_tes_indices),
                tau_eff=float(atmosphere.tau),
                median_conversion_factor_adu_per_k=float(np.nanmean(conversion_factor_values)),
                # ottenuti da noise range summary
                mean_plateau_asd=float(np.nanmean(plateau_values)),
                std_plateau_asd=(float(np.nanstd(plateau_values, ddof=1))
                                 if plateau_values.size > 1
                                 else 0.0),
                # ottenuti da dataset noise summary
                mean_fit_knee_freq=float(np.nanmean(fit_knee_values)),
                std_fit_knee_freq=(float(np.nanstd(fit_knee_values, ddof=1))
                                           if fit_knee_values.size > 1
                                           else 0.0),
                mean_fit_white_noise_asd=float(np.nanmean(fit_white_noise_asds)),
                std_fit_white_noise_asd=(float(np.nanstd(fit_white_noise_asds, ddof=1))
                                                       if fit_white_noise_asds.size > 1
                                                       else 0.0),
                mean_fit_alpha=float(np.nanmean(fit_alphas)),
                std_fit_alpha=(float(np.nanstd(fit_alphas, ddof=1))
                               if fit_alphas.size > 1
                               else 0.0),
                mean_fit_cutoff_frequency_hz=float(
                    np.nanmean(fit_cutoff_frequencies_hz)
                ),
                std_fit_cutoff_frequency_hz=(
                    float(np.nanstd(fit_cutoff_frequencies_hz, ddof=1))
                    if fit_cutoff_frequencies_hz.size > 1
                    else 0.0
                ))

            noise_vs_tau_results.append(dataset_noise_result)

    noise_vs_tau_output_dir = Path(config.paths.runs[0].output).parents[1]


    plotting.plot_noise_plateau_vs_tau(
        noise_vs_tau_results=noise_vs_tau_results,
        output_path=noise_vs_tau_output_dir / "noise_plateau_vs_tau.html",
        show=config.plots.show)

    plotting.plot_noise_fit_parameter_vs_tau(
        noise_vs_tau_results=noise_vs_tau_results,
        y_attribute="mean_fit_knee_freq",
        y_error_attribute="std_fit_knee_freq",
        y_label="Knee frequency [Hz]",
        title="Fit knee frequency vs tau",
        output_path=noise_vs_tau_output_dir / "fit_knee_frequency_vs_tau.html",
        show=config.plots.show,
        log_y=True,
    )

    plotting.plot_noise_fit_parameter_vs_tau(
        noise_vs_tau_results=noise_vs_tau_results,
        y_attribute="mean_fit_white_noise_asd",
        y_error_attribute="std_fit_white_noise_asd",
        y_label="White noise ASD [K / sqrt(Hz)]",
        title="Fit white noise ASD vs tau",
        output_path=noise_vs_tau_output_dir / "fit_white_noise_asd_vs_tau.html",
        show=config.plots.show,
        log_y=True,
    )

    plotting.plot_noise_fit_parameter_vs_tau(
        noise_vs_tau_results=noise_vs_tau_results,
        y_attribute="mean_fit_alpha",
        y_error_attribute="std_fit_alpha",
        y_label="Alpha",
        title="Fit alpha vs tau",
        output_path=noise_vs_tau_output_dir / "fit_alpha_vs_tau.html",
        show=config.plots.show,
        log_y=False,
    )

    plotting.plot_noise_fit_parameter_vs_tau(
        noise_vs_tau_results=noise_vs_tau_results,
        y_attribute="mean_fit_cutoff_frequency_hz",
        y_error_attribute="std_fit_cutoff_frequency_hz",
        y_label="Cutoff frequency [Hz]",
        title="Fit cutoff frequency vs tau",
        output_path=noise_vs_tau_output_dir / "fit_cutoff_frequency_vs_tau.html",
        show=config.plots.show,
        log_y=True,
    )



if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError(
            "Usage: python run_calibration.py <calibration_config.yaml>"
        )

    main(Path(sys.argv[1]).expanduser().resolve())