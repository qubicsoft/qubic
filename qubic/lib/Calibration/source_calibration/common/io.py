import sys
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from qubicpack.qubicfp import qubicfp
from qubic.lib.Calibration.source_calibration.skydip.config_calibration import SkydipCalibrationConfig, load_skydip_calibration_config
from qubic.lib.Calibration.source_calibration.common.utils import parse_utc_datetime, parse_tes_indices, capture_external_output


@dataclass(slots=True)
class QubicDataset:
    """
    Ogni oggetto QubicDataset sa da quale cartella e' stato leetto e
    in quale cartella salvare i dati
    """
    dataset: Path
    output_dir: Path

    time: np.ndarray
    signals: np.ndarray

    elevation: np.ndarray
    azimuth: np.ndarray | None

    interp_azimuth: np.ndarray | None
    interp_elevation: np.ndarray

    azimuth_time: np.ndarray | None
    elevation_time: np.ndarray | None

    input_dir: Path = field(init=False)
    housekeeping_dir: Path = field(init=False)
    calibration_plots_dir: Path = field(init=False)
    preprocessing_plots_dir: Path = field(init=False)
    noise_plots_dir: Path = field(init=False)

    def __post_init__(self):
        """
        Con __post_init__ il path viene calcolato
        una volta sola alla creazione dell’oggetto
        """

        self.output_dir = self.output_dir.expanduser().resolve()
        self.dataset = self.dataset.expanduser().resolve()

        self.input_dir = self.output_dir / "input"
        self.housekeeping_dir = self.output_dir / "plots" / "housekeeping"
        self.calibration_plots_dir = self.output_dir / "plots" / "calibration"
        self.preprocessing_plots_dir = self.output_dir / "plots" / "preprocessing"
        self.noise_plots_dir = self.output_dir / "plots" / "noise"

    @property
    def dataset_name(self) -> str:
        return self.dataset.name

    @property
    def start_time_utc(self) -> datetime:
        return parse_utc_datetime(self.dataset_name)

    @property
    def duration_s(self) -> float:
        return np.max(self.time) - np.min(self.time)

    @property
    def stop_time_utc(self) -> datetime:
        return self.start_time_utc + timedelta(seconds=self.duration_s)


    @classmethod
    def from_config(cls,
                    config: SkydipCalibrationConfig,
                    run_index: int,
                    logger: logging.Logger | None = None) -> "QubicDataset":
        """
        Legge un dataset QUBICStudio definito nello YAML e costruisce
        un oggetto QubicDataset completo
        """

        run = config.paths.runs[run_index]

        dataset_path = run.dataset
        output_dir = run.output

        if logger is not None:
            logger.info("Reading QUBICStudio dataset: %s", dataset_path.name)

        qubic = qubicfp()
        qubic.verbosity = 0

        with capture_external_output() as (stdout_buffer, stderr_buffer):
            qubic.read_qubicstudio_dataset(str(dataset_path))
            time, signals = qubic.tod()

        # time, signals = qubic.tod()
        time -= time[0]

        if config.tod_processing.centered:
            signals -= np.nanmedian(signals, axis=1, keepdims=True)

        if config.tod_processing.normalized:
            scale = np.nanmax(np.abs(signals), axis=1, keepdims=True)
            scale[~np.isfinite(scale) | (scale == 0)] = 1.0
            signals /= scale


        elevation = qubic.elevation()
        elevation_time = qubic.timeaxis(datatype="el")

        if elevation is None or elevation_time is None:
            message = f"Elevation or time_elevation not found in dataset: {dataset_path}"
            if logger is not None:
                logger.error(message)
            raise ValueError(message)

        azimuth = qubic.azimuth()
        azimuth_time = qubic.timeaxis(datatype="az")

        if azimuth is None or azimuth_time is None:
            message = f"Azimuth or time_azimuth not found in dataset: {dataset_path}"
            if logger is not None:
                logger.error(message)
            raise ValueError(message)

        azimuth = np.asarray(azimuth, dtype=np.float64)
        elevation = np.asarray(elevation, dtype=np.float64)
        azimuth_time = np.asarray(azimuth_time, dtype=np.float64)
        elevation_time = np.asarray(elevation_time, dtype=np.float64)

        azimuth_time -= azimuth_time[0]
        elevation_time -= elevation_time[0]

        interp_elevation = np.interp(time, elevation_time, elevation)
        interp_azimuth = np.interp(time, azimuth_time, azimuth)

        if logger is not None:
            logger.info("%s path: %s", dataset_path.name, dataset_path)
            logger.info("%s output: %s", dataset_path.name, output_dir)
            logger.info("%s TOD shape: %s", dataset_path.name, signals.shape)
            logger.info("%s el shape: %s", dataset_path.name, elevation.shape)
            logger.info("%s el time axis: %s", dataset_path.name, elevation_time.shape)
            logger.info("%s az shape: %s", dataset_path.name, azimuth.shape)
            logger.info("%s az time axis: %s", dataset_path.name, azimuth_time.shape)
            logger.info("%s interpolated el: %s", dataset_path.name, interp_elevation.shape)
            logger.info("%s interpolated az: %s", dataset_path.name, interp_azimuth.shape)

        return cls(
            dataset=dataset_path,
            output_dir=output_dir,
            time=time,
            signals=signals,
            elevation=elevation,
            azimuth=azimuth,
            interp_azimuth=interp_azimuth,
            interp_elevation=interp_elevation,
            azimuth_time=azimuth_time,
            elevation_time=elevation_time)

    def create_output_tree(self, tes_indices: list[int]) -> None:
        """
        Crea su disco la struttura standard di output per questo dataset.
        """

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.housekeeping_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_plots_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessing_plots_dir.mkdir(parents=True, exist_ok=True)
        self.noise_plots_dir.mkdir(parents=True, exist_ok=True)

        for tes_index in tes_indices:
            (self.calibration_plots_dir / f"tes_{tes_index}").mkdir(
                parents=True,
                exist_ok=True,
            )
            (self.preprocessing_plots_dir / f"tes_{tes_index}").mkdir(
                parents=True,
                exist_ok=True,
            )
            (self.noise_plots_dir / f"tes_{tes_index}").mkdir(
                parents=True,
                exist_ok=True,
            )

    def save_arrays(self,
                    tes_indices: list[int],
                    logger: logging.Logger | None = None) -> None:

        """
        Crea le cartelle di output e salva gli array principali come .npy/.npz.
        """

        self.create_output_tree(tes_indices=tes_indices)

        np.save(self.input_dir / f"interp_elevation_{self.dataset_name}.npy", self.interp_elevation)
        np.save(self.input_dir / f"interp_azimuth_{self.dataset_name}.npy", self.interp_azimuth)

        np.save(self.input_dir / f"time_elevation_{self.dataset_name}.npy", self.elevation_time)
        np.save(self.input_dir / f"time_azimuth_{self.dataset_name}.npy", self.azimuth_time)
        np.save(self.input_dir / f"elevation_{self.dataset_name}.npy", self.elevation)
        np.save(self.input_dir / f"azimuth_{self.dataset_name}.npy", self.azimuth)

        np.save(self.input_dir / f"time_{self.dataset_name}.npy", self.time)
        np.savez_compressed(self.input_dir / f"signals_{self.dataset_name}.npz", tods=self.signals)

        if logger is not None:
            logger.info("Saved arrays in %s", self.input_dir)

    @classmethod
    def from_saved_arrays(
            cls,
            dataset: Path,
            output_dir: Path,
            logger: logging.Logger | None = None,
    ) -> "QubicDataset":
        """
        Legge i file .npy e .npz prodotti da save_arrays()
        e ricostruisce un oggetto QubicDataset.
        """

        dataset = dataset.expanduser().resolve()
        output_dir = output_dir.expanduser().resolve()

        input_dir = output_dir / "input"
        dataset_name = dataset.name

        signals_path = input_dir / f"signals_{dataset_name}.npz"

        with np.load(signals_path) as data:
            if "tods" not in data:
                raise KeyError(f"Key 'tods' not found in {signals_path}.")
            signals = data["tods"]

        if signals.ndim != 2:
            raise ValueError(
                f"TOD data in {signals_path} must be a 2D array with shape "
                "(n_tes, n_samples)."
            )

        loaded_dataset = cls(
            dataset=dataset,
            output_dir=output_dir,
            time=np.load(input_dir / f"time_{dataset_name}.npy"),
            signals=signals,
            elevation=np.load(input_dir / f"elevation_{dataset_name}.npy"),
            azimuth=np.load(input_dir / f"azimuth_{dataset_name}.npy"),
            interp_azimuth=np.load(input_dir / f"interp_azimuth_{dataset_name}.npy"),
            interp_elevation=np.load(input_dir / f"interp_elevation_{dataset_name}.npy"),
            azimuth_time=np.load(input_dir / f"time_azimuth_{dataset_name}.npy"),
            elevation_time=np.load(input_dir / f"time_elevation_{dataset_name}.npy"),
        )

        if logger is not None:
            logger.info("Loaded saved arrays from %s", input_dir)

        return loaded_dataset

def prepare_datasets_from_config(config: SkydipCalibrationConfig,
                                 logger: logging.Logger | None = None) -> None:
    """
    Legge tutti i dataset QUBICStudio definiti in config.paths.runs
    e salva gli array .npy/.npz nei rispettivi output.
    """

    for run_index, run in enumerate(config.paths.runs):

        if logger is not None:
            logger.info("-" * 60)
            logger.info(
                "Preparing dataset %d/%d: %s",
                run_index + 1,
                len(config.paths.runs),
                run.dataset,
            )

        dataset = QubicDataset.from_config(
            config=config,
            run_index=run_index,
            logger=logger,
        )

        tes_indices = parse_tes_indices(tes_indices=config.calibration.tes_indices,
                                        n_tes=dataset.signals.shape[0])

        dataset.save_arrays(
            tes_indices=tes_indices,
            logger=logger,
        )

def iter_saved_datasets(config: SkydipCalibrationConfig,
                        logger: logging.Logger | None = None):
    """
    Itera sui dataset salvati in formato .npy/.npz.

    Carica un solo dataset alla volta, quindi e' adatto a TOD grandi.
    """

    for run_index, run in enumerate(config.paths.runs):

        if logger is not None:
            logger.info("=" * 80)
            logger.info(
                "Loading saved dataset %d/%d: %s",
                run_index + 1,
                len(config.paths.runs),
                run.output,
            )

        yield QubicDataset.from_saved_arrays(
            dataset=run.dataset,
            output_dir=run.output,
            logger=logger,
        )


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d/%m/%Y | %H:%M:%S",
    )

    logger = logging.getLogger(__name__)

    config = load_skydip_calibration_config(
        Path(sys.argv[1]).expanduser().resolve()
    )

    prepare_datasets_from_config(
        config=config,
        logger=logger,
    )
