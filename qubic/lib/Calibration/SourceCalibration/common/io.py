import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qubicpack.qubicfp import qubicfp


@dataclass(slots=True)
class QubicDataset:
    dataset_path: Path

    time: np.ndarray
    signals: np.ndarray

    elevation: np.ndarray
    azimuth: np.ndarray | None

    interp_azimuth: np.ndarray | None
    interp_elevation: np.ndarray

    azimuth_time: np.ndarray | None
    elevation_time: np.ndarray | None



def read_qubic_dataset(dataset_path: str | Path,
                       logger: logging.Logger | None = None,
                       log_label: str = "DATASET") -> QubicDataset:
    """
    Read a QUBICStudio dataset and return the TODs together with the
    elevation interpolated on the TOD time axis.

    Parameters
    ----------
    dataset_path
        Path to the QUBICStudio dataset directory.
    logger
        Optional logger used to report shapes and chosen time axis.
    log_label
        Prefix used in log messages to distinguish different datasets.
    """
    dataset_path = Path(dataset_path)

    qubic_data = qubicfp()
    qubic_data.read_qubicstudio_dataset(str(dataset_path))

    time, signals = qubic_data.tod()
    time = np.asarray(time, dtype=np.float64)
    time -= time[0]

    # reading the elevation and elevation_time
    elevation = qubic_data.elevation()
    elevation = None if elevation is None else np.asarray(elevation, dtype=np.float64)

    elevation_time = qubic_data.timeaxis(datatype="platform")
    elevation_time = None if elevation_time is None else np.asarray(elevation_time, dtype=np.float64)
    elevation_time -= elevation_time[0]

    # reading the azimuth and azimuth_time
    azimuth = qubic_data.azimuth()
    azimuth = None if azimuth is None else np.asarray(azimuth, dtype=np.float64)

    azimuth_time = qubic_data.timeaxis(datatype="azimuth")
    azimuth_time = None if azimuth_time is None else np.asarray(azimuth_time, dtype=np.float64)
    azimuth_time -= azimuth_time[0]

    interp_elevation = np.interp(time, elevation_time, elevation)
    interp_azimuth = np.interp(time, azimuth_time, azimuth)

    if logger is not None:
        logger.info("%s path: %s", dataset_path.name, dataset_path)
        logger.info("%s TOD shape: %s", dataset_path.name, signals.shape)
        logger.info("%s elevation shape: %s", dataset_path.name, elevation.shape)
        logger.info("%s azimuth shape: %s", dataset_path.name, azimuth.shape)
        logger.info("%s azimuth time axis: %s", dataset_path.name, interp_azimuth.shape)
        logger.info("%s elevation time axis: %s", dataset_path.name, interp_elevation.shape)
        logger.info("%s interpolated elevation: %s", dataset_path.name, interp_elevation.shape)
        logger.info("%s interpolated azimuth: %s", dataset_path.name, interp_azimuth.shape)


    return QubicDataset(
        dataset_path=dataset_path,
        time=time,
        signals=signals,
        elevation=elevation,
        azimuth=azimuth,
        interp_azimuth=interp_azimuth,
        interp_elevation=interp_elevation,
        azimuth_time=azimuth_time,
        elevation_time=elevation_time)


def save_dataset_arrays(dataset: QubicDataset,
                        output_dir: str | Path,
                        logger: logging.Logger | None = None) -> None:
    """
    Save the main time/pointing arrays as .npy files and the TODs as a .npz file.

    Saved files
    -----------
    - <prefix>_time.npy
    - <prefix>_elevation.npy
    - <prefix>_interp_elevation.npy
    - <prefix>_selected_time.npy
    - <prefix>_azimuth.npy (if available)
    - <prefix>_interp_azimuth.npy (if available)
    - <prefix>_tods.npz
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"interp_elevation_{dataset.dataset_path.name}.npy", dataset.interp_elevation)
    np.save(output_dir / f"interp_azimuth_{dataset.dataset_path.name}.npy", dataset.interp_azimuth)

    np.save(output_dir / f"time_elevation_{dataset.dataset_path.name}.npy", dataset.elevation_time)
    np.save(output_dir / f"time_azimuth_{dataset.dataset_path.name}.npy", dataset.azimuth_time)

    np.save(output_dir / f"elevation_{dataset.dataset_path.name}.npy", dataset.elevation)
    np.save(output_dir / f"azimuth_{dataset.dataset_path.name}.npy", dataset.azimuth)

    np.save(output_dir / f"time_{dataset.dataset_path.name}.npy", dataset.time)
    np.savez_compressed(output_dir / f"signals_{dataset.dataset_path.name}.npz", tods=dataset.signals)

    if logger is not None:
        logger.info("Saved arrays for in %s",output_dir)



if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d/%m/%Y | %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    dataset_path = Path("/Volumes/Data/PycharmProjects/calibration/qubic/scripts/Calibration/skydip_calibration/data/input/2026-03-11/2026-03-11_16.48.15__SkyDip")
    skydip_path = Path("/path/to/skydip")
    out_dir = Path("/Volumes/Data/PycharmProjects/calibration/qubic/scripts/Calibration/skydip_calibration/data/output") / dataset_path.name / "input"

    dataset = read_qubic_dataset(dataset_path=dataset_path, logger=logger)


    save_dataset_arrays(dataset=dataset,
                        output_dir=out_dir,
                        logger=logger)
