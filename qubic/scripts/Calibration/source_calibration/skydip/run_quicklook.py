import sys
import logging
from pathlib import Path

from qubic.lib.Calibration.source_calibration.common.io import prepare_datasets_from_config, iter_saved_datasets
from qubic.lib.Calibration.source_calibration.common import plotting
from qubic.lib.Calibration.source_calibration.skydip.config_calibration import load_skydip_calibration_config
from qubic.lib.Calibration.source_calibration.common.preprocessing import find_skydip_intervals

def main(config_path: Path):

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%d/%m/%Y | %H:%M:%S")

    logger = logging.getLogger(__name__)

    # 1. Leggo il file YAML di configurazione
    config = load_skydip_calibration_config(config_path)

    # 2. Seguo lo stesso ordine di operazioni di io.py:
    #    leggo tutti i dataset QUBICStudio definiti in paths.runs
    #    e salvo i corrispondenti file .npy/.npz nei rispettivi output.
    prepare_datasets_from_config(config=config,
                                 logger=logger)

    # 3. Ora rileggo i file .npy/.npz appena prodotti,
    #    un dataset alla volta, usando il generatore
    for dataset in iter_saved_datasets(config=config, logger=logger):

        logger.info("Running quicklook for dataset: %s", dataset.dataset_name)

        # 4. Plot semplice dei TOD selezionati nel config.
        plotting.plot_tod(
            dataset=dataset,
            config=config,
        )

        # 5. Plot interpolato di azimuth/elevation.
        #    Dentro plot_az_el_vs_time, se is_interpolated=True,
        #    vengono mostrati automaticamente i constant-azimuth blocks.
        plotting.plot_az_el_vs_time(
            dataset=dataset,
            config=config,
            is_interpolated=True,
        )

        # 6. Trovo gli intervalli skydip per il dataset corrente.
        #    Questa funzione usa:
        #        dataset.time
        #        dataset.interp_azimuth
        #        dataset.interp_elevation
        #        config.preprocessing.*
        skydip_intervals = find_skydip_intervals(
            dataset=dataset,
            config=config,
        )

        logger.info(
            "Found %d skydips for dataset %s",
            len(skydip_intervals.idx_pairs),
            dataset.dataset_name,
        )

        logger.info(
            "Found %d constant-azimuth blocks for dataset %s",
            len(skydip_intervals.azimuth_blocks),
            dataset.dataset_name,
        )

        # 7. Plot dei TOD selezionati con overlay dei limiti skydip.
        #    Come plot_tods, questa funzione cicla su tutti i TES selezionati
        #    in config.calibration.tes_indices.
        plotting.plot_tod_with_skydip_limits(
            dataset=dataset,
            config=config,
            skydip_intervals=skydip_intervals,
        )

        # 8. Plot dell'elevation interpolata con overlay dei limiti skydip.
        #    Questo serve per verificare visivamente che gli intervalli trovati
        #    corrispondano davvero alle salite/discese in elevation.
        plotting.plot_elevation_with_skydip_limits(
            dataset=dataset,
            config=config,
            skydip_intervals=skydip_intervals,
        )

        # 9. Plot della focal plane completa.
        #    Ogni subplot corrisponde a un TES nella sua posizione fisica.
        #    Questo plot usa dataset.signals, quindi rispetta già centered/normalized
        #    applicati in QubicDataset.from_config().
        plotting.plot_focal_plane_tods(
            dataset=dataset,
            config=config,
            flip_ud=False,
            flip_lr=False,
        )



if __name__ == "__main__":
    main(config_path=Path(sys.argv[1]).expanduser().resolve())

