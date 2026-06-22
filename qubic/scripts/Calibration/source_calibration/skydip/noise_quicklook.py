from pathlib import Path

import numpy as np

from qubic.lib.Calibration.source_calibration.common.io import QubicDataset, read_qubic_dataset
from qubic.lib.Calibration.source_calibration.common.plotting import (
    plot_single_tod,
    plot_az_el_vs_time,
    plot_scan,
    plot_interp_pointing,
    plot_subscan,
    plot_masked_tod,
    plot_cleaning_steps,
    plot_tod_with_az_segments,
)
from qubic.lib.Calibration.source_calibration.common.legacy_preprocessing import (
    extract_scan_turning_points,
    interpolate_pointing_to_tod_time,
    build_scan_mask,
    split_scan,
    extract_azimuth_segments,
    remove_dc_offset,
    smooth_tods,
)



def load_dataset(dataset_path: str | Path) -> QubicDataset:
    """
    Read a QUBIC dataset and return the TOD container.
    """
    return read_qubic_dataset(dataset_path)



def run_single_tod_quicklook(dataset: QubicDataset,
                             tes_index: int,
                             output_dir: str | Path) -> None:
    """
    Run all available quicklook plots for a single TES TOD.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    single_tod_dir = output_dir / "single_tod"
    preprocessing_dir = output_dir / "preprocessing"

    plot_single_tod(
        tods=dataset.signals,
        tes_index=tes_index,
        output_path=single_tod_dir,
        centered=False,
        normalized=False,
    )

    if dataset.azimuth is None or dataset.elevation is None:
        return

    plot_az_el_vs_time(az=dataset.azimuth, el=dataset.elevation, output_path=output_dir / "az_el.pdf")

    scan_config = extract_scan_turning_points(dataset.azimuth)

    if dataset.azimuth_time is None:
        return

    plot_scan(
        time_=dataset.azimuth_time,
        az=dataset.azimuth,
        scan_config=scan_config,
        output_path=output_dir,
        title="Scan turning points",
    )

    top_peakset_hk = np.array(
        [[peak["start"], peak["stop"]] for peak in scan_config.tops_peakset],
        dtype=int,
    ) if scan_config.tops_peakset else np.empty((0, 2), dtype=int)

    bottom_peakset_hk = np.array(
        [[peak["start"], peak["stop"]] for peak in scan_config.bottoms_peakset],
        dtype=int,
    ) if scan_config.bottoms_peakset else np.empty((0, 2), dtype=int)

    interp_data = interpolate_pointing_to_tod_time(
        tm_hk=dataset.azimuth_time,
        tm_tod=dataset.time,
        azimuth=dataset.azimuth,
        elevation=dataset.elevation,
        top_peakset=top_peakset_hk,
        bottom_peakset=bottom_peakset_hk,
    )

    plot_interp_pointing(
        time_=dataset.time,
        interp_data=interp_data,
        output_path=preprocessing_dir / "interp_pointing.pdf",
        title="Interpolated pointing",
    )

    scan_mask = build_scan_mask(
        top_arr_interp=interp_data.top_peakset,
        bottom_arr_interp=interp_data.bottom_peakset,
        n_samples=dataset.signals.shape[1],
    )
    scan_intervals = split_scan(scan_mask.mask_scan)
    azimuth_segments = extract_azimuth_segments(interp_data, scan_intervals)

    plot_subscan(
        time_=dataset.time,
        interp_data=interp_data,
        scan_intervals=scan_intervals,
        output_path=preprocessing_dir / "subscans.pdf",
        n_sweeps_to_plot=3,
        title="Subscans and sweeps",
    )

    plot_masked_tod(
        time_=dataset.time,
        tod=dataset.signals[tes_index],
        scan_mask=scan_mask,
        output_path=preprocessing_dir / f"tes_{tes_index}_masked.pdf",
        title=f"TES {tes_index} masked TOD",
    )

    cleaned_tods = remove_dc_offset(dataset.signals)
    cleaned_tods = smooth_tods(cleaned_tods)

    plot_cleaning_steps(
        time_=dataset.time,
        raw_tod=dataset.signals[tes_index],
        scan_mask=scan_mask,
        cleaned_tod=cleaned_tods[tes_index],
        output_path=preprocessing_dir / f"tes_{tes_index}_cleaning.pdf",
        title=f"TES {tes_index} cleaning steps",
    )

    plot_tod_with_az_segments(
        time_=dataset.time,
        tod=dataset.signals[tes_index],
        azimuth_segments=azimuth_segments,
        output_path=preprocessing_dir / f"tes_{tes_index}_azimuth_segments.pdf",
        title=f"TES {tes_index} with azimuth segments",
    )


if __name__ == "__main__":
    dataset_path = Path(
        "/qubic/scripts/Calibration/skydip/data/dataset/2026-04-01/2026-04-01_13.31.20__dome_closed_azimuth_scan_full_range")
    output_dir = Path("noise_quicklook_output")
    tes_index = 95

    dataset = load_dataset(dataset_path)
    run_single_tod_quicklook(
        dataset=dataset,
        tes_index=tes_index,
        output_dir=output_dir,
    )