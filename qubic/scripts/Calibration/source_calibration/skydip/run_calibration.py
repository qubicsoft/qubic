import sys
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from qubic.lib.Calibration.source_calibration.common import plotting
from qubic.lib.Calibration.source_calibration.common.utils import parse_tes_indices
from qubic.lib.Calibration.source_calibration.common.preprocessing import SkydipIntervals
from qubic.lib.Calibration.source_calibration.skydip.atmosphere.run_atmosphere import Atmosphere
from qubic.lib.Calibration.source_calibration.skydip.atmosphere import run_atmosphere
from qubic.lib.Calibration.source_calibration.skydip.config_calibration import load_skydip_calibration_config
from qubic.lib.Calibration.source_calibration.common.io import prepare_datasets_from_config, iter_saved_datasets
from qubic.lib.Calibration.source_calibration.skydip.calibration import SkydipCalibrationSegment, DatasetConversionFactorSummary
from qubic.lib.Calibration.source_calibration.common.noise import compute_all_skydip_noise_spectra, DatasetNoiseSummary

@dataclass()
class DatasetNoiseVsTauResult:
    dataset_name: str
    tes_indices: tuple[int, ...]
    n_tes: int

    tau_eff: float
    median_conversion_factor_adu_per_k: float

    mean_knee_frequency_hz: float
    std_knee_frequency_hz: float

    mean_plateau_asd_k_per_sqrt_hz: float
    std_plateau_asd_k_per_sqrt_hz: float
    mean_plateau_mad_k_per_sqrt_hz: float


def main(config_path: Path):

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%d/%m/%Y | %H:%M:%S")

    logger = logging.getLogger(__name__)

    # leggo file di configurazione
    config = load_skydip_calibration_config(config_path)

    # Seguo lo stesso ordine di operazioni di io.py:
    # leggo tutti i dataset QUBICStudio definiti in paths.runs
    # e salvo i corrispondenti file .npy/.npz nei rispettivi output.
    prepare_datasets_from_config(config=config,
                                 logger=logger)

    noise_vs_tau_results: list[DatasetNoiseVsTauResult] = []

    # Ora rileggo i file .npy/.npz appena prodotti,
    # un dataset alla volta, usando il generatore
    for dataset, run_paths in zip(
            iter_saved_datasets(config=config, logger=logger),
            config.paths.runs):

        logger.info("Running quicklook for dataset: %s", dataset.dataset_name)

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

        # Plot interpolato di azimuth/elevation.
        # Dentro plot_az_el_vs_time, se is_interpolated=True,
        # vengono mostrati automaticamente i constant-azimuth blocks.
        plotting.plot_az_el_vs_time(dataset=dataset,
                                    config=config,
                                    is_interpolated=True)

        # Plot dell'elevation interpolata e non con overlay dei limiti skydip.
        # Questo serve per verificare visivamente che gli intervalli trovati
        # corrispondano davvero alle salite/discese in elevation.
        plotting.plot_elevation_with_skydip_limits(dataset=dataset,
                                                   config=config,
                                                   skydip_intervals=skydip_intervals)

        # Plot della focal plane completo
        # Ogni subplot corrisponde a un TES nella sua posizione fisica.
        plotting.plot_focal_plane_tods(dataset=dataset,
                                       config=config,
                                       flip_ud=False,
                                       flip_lr=False)

        # parso e salvo gli indici dei TES contenuti nel config
        tes_indices = parse_tes_indices(tes_indices=config.calibration.tes_indices, n_tes=dataset.signals.shape[0])
        logger.info("Tes to analyze: %s", tes_indices)

        atmosphere = Atmosphere(config=run_paths.atmospheric_config,
                                start_obs=dataset.start_time_utc,
                                end_obs=dataset.stop_time_utc,
                                dataset_name=dataset.dataset_name)

        dataset_knee_frequencies_hz = []
        dataset_plateau_asds_k_per_sqrt_hz = []
        dataset_plateau_mads_k_per_sqrt_hz = []
        dataset_conversion_factors_adu_per_k = []
        analyzed_tes_indices = []

        for tes_idx in tes_indices:

            # Plot semplice dei TOD selezionati nel config
            plotting.plot_tod(dataset=dataset, config=config, tes_idx=tes_idx)

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

            calibrated_segments = SkydipCalibrationSegment.from_tod_segments(
                tod_segments=tod_segments,
                dataset=dataset,
                tau_eff=atmosphere.tau,
                tb_eff_k=atmosphere.T_b,
            )

            conversion_summary = DatasetConversionFactorSummary(segments=calibrated_segments)
            conversion_factors = conversion_summary.save_to_csv(tes_idx=tes_idx, dataset=dataset)
            median_conversion_factor = conversion_summary.median_adu_per_k

            print(f"Median Conversion factor for dataset {dataset.dataset_name} and TES {tes_idx}: {median_conversion_factor}")


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

            plotting.plot_skydip_noise_spectra(noise_spectra=raw_noise_spectra,
                                               config=config,
                                               tes_idx=tes_idx,
                                               output_dir=dataset.noise_plots_dir,
                                               is_calibrated=False)


            calibrated_noise_spectra = compute_all_skydip_noise_spectra(
                segments=calibrated_segments,
                config=config,
                conversion_factor_adu_per_k=conversion_summary.dataset_conversion_factor_adu_per_k,
            )

            plotting.plot_skydip_noise_spectra(noise_spectra=calibrated_noise_spectra,
                                               config=config,
                                               tes_idx=tes_idx,
                                               output_dir=dataset.noise_plots_dir,
                                               is_calibrated=True)

            noise_summary = DatasetNoiseSummary.from_spectra(noise_spectra=calibrated_noise_spectra,
                                                             tau_eff=atmosphere.tau)

            dataset_knee_frequencies_hz.append(noise_summary.knee_frequency_hz)
            dataset_plateau_asds_k_per_sqrt_hz.append(noise_summary.plateau_asd_k_per_sqrt_hz)
            dataset_plateau_mads_k_per_sqrt_hz.append(noise_summary.plateau_mad_k_per_sqrt_hz)
            dataset_conversion_factors_adu_per_k.append(
                conversion_summary.dataset_conversion_factor_adu_per_k
            )
            analyzed_tes_indices.append(int(tes_idx))

        if analyzed_tes_indices:
            knee_values = np.asarray(dataset_knee_frequencies_hz, dtype=np.float64)
            plateau_values = np.asarray(dataset_plateau_asds_k_per_sqrt_hz, dtype=np.float64)
            plateau_mad_values = np.asarray(dataset_plateau_mads_k_per_sqrt_hz, dtype=np.float64)
            conversion_factor_values = np.asarray(dataset_conversion_factors_adu_per_k, dtype=np.float64)

            dataset_noise_result = DatasetNoiseVsTauResult(
                dataset_name=dataset.dataset_name,
                tes_indices=tuple(analyzed_tes_indices),
                n_tes=len(analyzed_tes_indices),
                tau_eff=float(atmosphere.tau),
                median_conversion_factor_adu_per_k=float(np.nanmean(conversion_factor_values)),
                mean_knee_frequency_hz=float(np.nanmean(knee_values)),
                std_knee_frequency_hz=(
                    float(np.nanstd(knee_values, ddof=1))
                    if knee_values.size > 1
                    else 0.0
                ),
                mean_plateau_asd_k_per_sqrt_hz=float(np.nanmean(plateau_values)),
                std_plateau_asd_k_per_sqrt_hz=(
                    float(np.nanstd(plateau_values, ddof=1))
                    if plateau_values.size > 1
                    else 0.0
                ),
                mean_plateau_mad_k_per_sqrt_hz=float(np.nanmean(plateau_mad_values)),
            )

            noise_vs_tau_results.append(dataset_noise_result)

    noise_vs_tau_output_dir = config.paths.runs[0].output.parent

    plotting.plot_noise_plateau_vs_tau(
        noise_vs_tau_results=noise_vs_tau_results,
        output_path=noise_vs_tau_output_dir/ "noise_plateau_vs_tau.html",
        show=config.plots.show,
    )











    # spectrum = load_or_run_atmospheric_spectrum(config)
    # tau_eff, tb_eff_k = spectrum.get_effective_atmospheric_parameters(
    #     strategy=config.atmosphere.parameter_strategy,
    # )
    #
    # print("Atmospheric parameters used for calibration")
    # print(f"  strategy: {config.atmosphere.parameter_strategy}")
    # print(f"  tau_eff: {tau_eff:.8f}")
    # print(f"  tb_eff_k: {tb_eff_k:.8f}")
    # print(f"  spectrum_csv: {spectrum.source_path}")
    #
    # skydip_intervals = find_skydip_intervals(
    #     tm=tm,
    #     azimuth=interp_azimuth,
    #     elevation=interp_elevation,
    #     az_velocity_threshold=config.preprocessing.az_velocity_threshold.to_value("deg/s"),
    #     min_block_duration=config.preprocessing.min_block_duration.to_value("s"),
    #     el_smooth_window=config.preprocessing.el_smooth_window,
    #     el_polyorder=config.preprocessing.el_polyorder,
    #     el_velocity_threshold=config.preprocessing.el_velocity_threshold.to_value("deg/s"),
    #     min_run_duration=config.preprocessing.min_run_duration.to_value("s"),
    #     up_left_extension=config.preprocessing.up_left_extension.to_value("s"),
    #     down_right_extension=config.preprocessing.down_right_extension.to_value("s"),
    # )
    #
    # print(f"Found {len(skydip_intervals.idx_pairs)} skydips.")
    # print(f"Found {len(skydip_intervals.azimuth_blocks)} constant-azimuth blocks.")
    #
    # plot_elevation_with_skydip_limits(
    #     tm=tm,
    #     elevation=interp_elevation,
    #     skydip_intervals=skydip_intervals,
    #     output_path=preprocessing_plots_dir / f"skydip_limits_from_elevation_{dataset_name}.html",
    #     title=f"Elevation with detected skydips - {dataset_name}",
    # )
    #
    # min_elevation_deg = 1.0
    #
    # for tes_indices in tes_indices:
    #     print(f"\nProcessing TES {tes_indices}")
    #
    #     tes_output_dir = dataset_output_dir / f"tes_{tes_indices}"
    #     tes_calibration_plots_dir = calibration_plots_dir / f"tes_{tes_indices}"
    #     tes_preprocessing_plots_dir = preprocessing_plots_dir / f"tes_{tes_indices}"
    #     tes_noise_plots_dir = noise_plots_dir / f"tes_{tes_indices}"
    #
    #     tes_output_dir.mkdir(parents=True, exist_ok=True)
    #     tes_calibration_plots_dir.mkdir(parents=True, exist_ok=True)
    #     tes_preprocessing_plots_dir.mkdir(parents=True, exist_ok=True)
    #     tes_noise_plots_dir.mkdir(parents=True, exist_ok=True)
    #
    #     plot_tods_with_skydip_limits(
    #         tm=tm,
    #         y=tods[tes_indices],
    #         skydip_intervals=skydip_intervals,
    #         tes_indices=tes_indices,
    #         normalized=config.tod_processing.normalized,
    #         centered=config.tod_processing.centered,
    #         output_path=tes_preprocessing_plots_dir / f"tes_{tes_indices}_skydip_limits_{dataset_name}.html",
    #         title=f"TOD with skydip limits - {dataset_name} - TES {tes_indices}",
    #         linewidth=config.plots.linewidth,
    #         alpha=config.plots.alpha,
    #     )
    #
    #     segments = plot_all_skydips_signal_vs_t_atm(
    #         tm=tm,
    #         tod=tods[tes_indices],
    #         elevation=interp_elevation,
    #         skydip_intervals=skydip_intervals,
    #         tau_eff=tau_eff,
    #         tb_eff_k=tb_eff_k,
    #         tes_indices=tes_indices,
    #         centered=config.tod_processing.centered,
    #         normalized=config.tod_processing.normalized,
    #         sort_by_airmass=False,
    #         output_dir=tes_calibration_plots_dir / "signal_vs_t_atm_time_order",
    #         min_elevation_deg=min_elevation_deg,
    #         linewidth=config.plots.linewidth,
    #         alpha=config.plots.alpha,
    #         fit_linewidth=config.plots.fit_linewidth,
    #         fit_alpha=config.plots.fit_alpha,
    #     )
    #
    #     sorted_segments = plot_all_skydips_signal_vs_t_atm(
    #         tm=tm,
    #         tod=tods[tes_indices],
    #         elevation=interp_elevation,
    #         skydip_intervals=skydip_intervals,
    #         tau_eff=tau_eff,
    #         tb_eff_k=tb_eff_k,
    #         tes_indices=tes_indices,
    #         centered=config.tod_processing.centered,
    #         normalized=config.tod_processing.normalized,
    #         sort_by_airmass=True,
    #         output_dir=tes_calibration_plots_dir / "signal_vs_t_atm_sorted_by_airmass",
    #         min_elevation_deg=min_elevation_deg,
    #         linewidth=config.plots.linewidth,
    #         alpha=config.plots.alpha,
    #         fit_linewidth=config.plots.fit_linewidth,
    #         fit_alpha=config.plots.fit_alpha,
    #     )
    #
    #     conversion_factor_summary = compute_dataset_conversion_factor_summary(
    #         dataset_name=f"{dataset_name}_tes_{tes_indices}",
    #         results=segments,
    #     )
    #
    #     conversion_factor_summary_csv = tes_output_dir / DATASET_CONVERSION_FACTOR_SUMMARY_FILENAME
    #     save_dataset_conversion_factor_summary(
    #         summary=conversion_factor_summary,
    #         output_csv=conversion_factor_summary_csv,
    #     )
    #
    #     per_skydip_conversion_factors_csv = save_skydip_conversion_factors(
    #         results=segments,
    #         output_csv=tes_output_dir / PER_SKYDIP_CONVERSION_FACTORS_FILENAME,
    #     )
    #
    #     selected_conversion_results = select_conversion_factor_results_for_mode(
    #         results=segments,
    #         summary=conversion_factor_summary,
    #         mode=config.calibration.conversion_factor_mode,
    #     )
    #
    #     selected_sorted_conversion_results = select_conversion_factor_results_for_mode(
    #         results=sorted_segments,
    #         summary=conversion_factor_summary,
    #         mode=config.calibration.conversion_factor_mode,
    #     )
    #
    #     if config.calibration.conversion_factor_mode == "dataset_mean":
    #         conversion_factors_csv = save_skydip_conversion_factors(
    #             results=selected_conversion_results,
    #             output_csv=tes_output_dir / MEAN_CONVERSION_FACTORS_FILENAME,
    #         )
    #     else:
    #         conversion_factors_csv = per_skydip_conversion_factors_csv
    #
    #     segments = selected_conversion_results
    #     sorted_segments = selected_sorted_conversion_results
    #
    #     print(
    #         f"Dataset conversion-factor summary for {dataset_name}, TES {tes_indices}: "
    #         f"mean={conversion_factor_summary.mean_adu_per_k:.6f} ADU/K, "
    #         f"std={conversion_factor_summary.std_adu_per_k:.6f} ADU/K, "
    #         f"median={conversion_factor_summary.median_adu_per_k:.6f} ADU/K, "
    #         f"MAD={conversion_factor_summary.mad_adu_per_k:.6f} ADU/K, "
    #         f"n={conversion_factor_summary.n_valid}"
    #     )
    #
    #     print(f"Saved dataset-level conversion-factor summary to: {conversion_factor_summary_csv}")
    #     print(f"Saved per-skydip conversion factors to: {per_skydip_conversion_factors_csv}")
    #     print(f"Using conversion-factor mode: {config.calibration.conversion_factor_mode}")
    #     print(f"Conversion-factor CSV used for noise conversion: {conversion_factors_csv}")
    #
    #     plot_conversion_factors_histogram(
    #         conversion_factors_csv=per_skydip_conversion_factors_csv,
    #         output_path=tes_calibration_plots_dir / "conversion_factors_per_skydip_histogram.html",
    #         title=f"{dataset_name} - TES {tes_indices} - per-skydip conversion factors",
    #         show=config.plots.show,
    #     )
    #
    #     if config.calibration.conversion_factor_mode == "dataset_mean":
    #         plot_conversion_factors_histogram(
    #             conversion_factors_csv=conversion_factors_csv,
    #             output_path=tes_calibration_plots_dir / "conversion_factors_used_histogram.html",
    #             title=f"{dataset_name} - TES {tes_indices} - conversion factors used",
    #             show=config.plots.show,
    #         )
    #
    #     if config.noise.enabled:
    #         noise_results = compute_all_skydip_noise_spectra(
    #             tm=tm,
    #             tod=tods[tes_indices],
    #             elevation=interp_elevation,
    #             skydip_intervals=skydip_intervals,
    #             tau_eff=tau_eff,
    #             tb_eff_k=tb_eff_k,
    #             conversion_factors_csv=conversion_factors_csv,
    #             min_elevation_deg=min_elevation_deg,
    #             nperseg=config.noise.nperseg,
    #         )
    #
    #         plot_skydip_noise_spectra(
    #             noise_results=noise_results,
    #             output_path=tes_noise_plots_dir / f"tes_{tes_indices}_skydip_noise_spectra.html",
    #             title=f"{dataset_name} - TES {tes_indices} - SkyDip noise spectra",
    #             show=config.plots.show,
    #         )




if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError(
            "Usage: python run_calibration.py <calibration_config.yaml>"
        )

    main(Path(sys.argv[1]).expanduser().resolve())