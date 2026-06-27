import numpy as np
from pathlib import Path
from astropy import units as u
from dataclasses import dataclass
from scipy.signal import savgol_filter

from qubic.lib.Calibration.source_calibration.common.io import QubicDataset
from qubic.lib.Calibration.source_calibration.common.utils import _to_value
from qubic.lib.Calibration.source_calibration.skydip.config_calibration import SkydipCalibrationConfig


def _prepare_savgol_window(n_samples: int,
                           window_length: int,
                           polyorder: int) -> int:

    if n_samples < 3:
        raise ValueError("Not enough samples for Savitzky-Golay smoothing.")

    win = min(window_length, n_samples)

    if win % 2 == 0:
        win -= 1

    min_valid = polyorder + 2
    if min_valid % 2 == 0:
        min_valid += 1

    if win < min_valid:
        win = min_valid

    if win >= n_samples:
        win = n_samples - 1
        if win % 2 == 0:
            win -= 1

    if win <= polyorder or win < 3:
        raise ValueError("Could not determine a valid smoothing window.")

    return win

def _find_constant_azimuth_blocks(tm: np.ndarray,
                                  azimuth: np.ndarray,
                                  az_velocity_threshold: u.Quantity,
                                  min_block_duration: u.Quantity) -> list[np.ndarray]:

    az_velocity_threshold = _to_value(az_velocity_threshold, u.deg / u.s)
    min_block_duration = _to_value(min_block_duration, u.s)

    daz_dt = np.gradient(azimuth, tm)
    # True: qui l’azimuth è sostanzialmente costante
    # False: qui sta cambiando in modo significativo
    is_flat = np.abs(daz_dt) <= az_velocity_threshold

    blocks = []
    # start serve a ricordare dove inizia un blocco flat.
    # start = None : non sono dentro nessun blocco costante
    start = None

    for i, ok in enumerate(is_flat):

        # qui inizia un nuovo blocco di azimuth costante
        if ok and start is None:
            start = i

        # Poi continua il ciclo. Finché ok resta True,
        # non succede nulla: siamo già dentro il blocco.
        # se entro nell'elif vuol dire:
        # ero dentro un blocco flat, ma ora il blocco è finito
        elif not ok and start is not None:

            # se il primo False è all’indice i, l’ultimo punto
            # valido del blocco era quello prima
            stop = i - 1

            # controllo se il blocco dura abbastanza
            if tm[stop] - tm[start] >= min_block_duration:

                # salvo gli indici del blocco
                # nota:  uso np.arange(start, i), non np.arange(start, stop),
                # perché np.arange esclude l’estremo finale
                blocks.append(np.arange(start, i, dtype=int))

            # resetto start: sono uscito dal blocco, ora non sono più
            # dentro un blocco costante
            start = None

    # gestisce il caso in cui il blocco flat arrivi fino alla fine dell’array
    if start is not None:
        stop = azimuth.size - 1
        if tm[stop] - tm[start] >= min_block_duration:
            blocks.append(np.arange(start, azimuth.size, dtype=int))

    return blocks

@dataclass
class SkydipIntervals:
    # [start_idx, stop_idx], for each skydip
    idx_pairs: np.ndarray
    # up or down, for each skydip
    directions: np.ndarray
    # azimuth mean for each skydip
    azimuth_mean: np.ndarray
    # block ID at constant azimuth at which the skydip belongs to
    block_ids: np.ndarray
    # smoothed elevation data
    elevation_smooth: np.ndarray
    # direction of elevation change for each block (1 for up, -1 for down, 0 for flat)
    elevation_direction: np.ndarray
    # list of azimuth blocks (constant azimuth)
    azimuth_blocks: list[np.ndarray]

    @classmethod
    def from_dataset(cls,
                     dataset: QubicDataset,
                     config: SkydipCalibrationConfig) -> "SkydipIntervals":

        az_velocity_threshold = _to_value(config.preprocessing.az_velocity_threshold, u.deg / u.s)
        el_velocity_threshold = _to_value(config.preprocessing.el_velocity_threshold, u.deg / u.s)

        min_block_duration = _to_value(config.preprocessing.min_block_duration, u.s)

        up_left_extension = _to_value(config.preprocessing.up_left_extension, u.s)
        down_right_extension = _to_value(config.preprocessing.down_right_extension, u.s)

        el_smooth_window = config.preprocessing.el_smooth_window
        el_polyorder = config.preprocessing.el_polyorder

        win = _prepare_savgol_window(n_samples=dataset.time.size,
                                     window_length=el_smooth_window,
                                     polyorder=el_polyorder)

        elevation_smooth = savgol_filter(dataset.interp_elevation,
                                         window_length=win,
                                         polyorder=el_polyorder)

        # Derivata discreta dell'elevation smoothing
        delv_dt = np.gradient(elevation_smooth, dataset.time)

        elevation_direction = np.zeros_like(delv_dt, dtype=int)
        elevation_direction[delv_dt > el_velocity_threshold] = 1
        elevation_direction[delv_dt < -el_velocity_threshold] = -1

        azimuth_blocks = _find_constant_azimuth_blocks(tm=dataset.time,
                                                       azimuth=dataset.interp_azimuth,
                                                       az_velocity_threshold=az_velocity_threshold,
                                                       min_block_duration=min_block_duration)

        idx_pairs = []
        directions = []
        azimuth_mean = []
        block_ids = []

        # Lavoro un blocco ad azimuth costante per volta.
        for block_id, block in enumerate(azimuth_blocks):

            block_dir = elevation_direction[block]
            block_az = dataset.interp_azimuth[block]

            # Tiene traccia di un tratto monotono in corso.
            # run_start: indice locale dentro block_dir
            run_start = None
            run_sign = 0

            # scorre la direzione dell’elevation punto per punto,
            # ma dentro un solo blocco di azimuth costante
            for local_idx, sign in enumerate(block_dir):

                if sign == 0:

                    # Chiude una run già aperta, ma prima la espande
                    # su eventuali plateau adiacenti.
                    if run_start is not None:
                        run_stop = local_idx - 1

                        expanded_start, expanded_stop = _expand_run_over_zero_velocity_edges(
                            block=block,
                            block_dir=block_dir,
                            tm=dataset.time,
                            run_start=run_start,
                            run_stop=run_stop,
                            run_sign=run_sign,
                            up_left_extension=up_left_extension,
                            down_right_extension=down_right_extension,
                        )

                        if dataset.time[block[expanded_stop]] - dataset.time[block[expanded_start]] >= min_block_duration:
                            start_idx = int(block[expanded_start])
                            stop_idx = int(block[expanded_stop])

                            idx_pairs.append([start_idx, stop_idx])
                            directions.append("up" if run_sign > 0 else "down")
                            azimuth_mean.append(float(np.nanmean(block_az)))
                            block_ids.append(block_id)

                        run_start = None
                        run_sign = 0

                    continue

                # apre una nuova run quando sign diverso da zero
                if run_start is None:
                    run_start = local_idx
                    run_sign = int(sign)
                    continue

                if sign != run_sign:
                    run_stop = local_idx - 1

                    expanded_start, expanded_stop = _expand_run_over_zero_velocity_edges(
                        block=block,
                        block_dir=block_dir,
                        tm=dataset.time,
                        run_start=run_start,
                        run_stop=run_stop,
                        run_sign=run_sign,
                        up_left_extension=up_left_extension,
                        down_right_extension=down_right_extension,
                    )

                    if dataset.time[block[expanded_stop]] - dataset.time[block[expanded_start]] >= min_block_duration:
                        start_idx = int(block[expanded_start])
                        stop_idx = int(block[expanded_stop])

                        idx_pairs.append([start_idx, stop_idx])
                        directions.append("up" if run_sign > 0 else "down")
                        azimuth_mean.append(float(np.nanmean(block_az)))
                        block_ids.append(block_id)

                    run_start = local_idx
                    run_sign = int(sign)

            if run_start is not None:
                run_stop = len(block_dir) - 1

                expanded_start, expanded_stop = _expand_run_over_zero_velocity_edges(
                    block=block,
                    block_dir=block_dir,
                    tm=dataset.time,
                    run_start=run_start,
                    run_stop=run_stop,
                    run_sign=run_sign,
                    up_left_extension=up_left_extension,
                    down_right_extension=down_right_extension,
                )

                if dataset.time[block[expanded_stop]] - dataset.time[block[expanded_start]] >= min_block_duration:
                    start_idx = int(block[expanded_start])
                    stop_idx = int(block[expanded_stop])

                    idx_pairs.append([start_idx, stop_idx])
                    directions.append("up" if run_sign > 0 else "down")
                    azimuth_mean.append(float(np.nanmean(block_az)))
                    block_ids.append(block_id)

        if len(idx_pairs) == 0:
            return cls(
                idx_pairs=np.empty((0, 2), dtype=int),
                directions=np.array([], dtype=object),
                azimuth_mean=np.array([], dtype=np.float64),
                block_ids=np.array([], dtype=int),
                elevation_smooth=elevation_smooth,
                elevation_direction=elevation_direction,
                azimuth_blocks=azimuth_blocks,
            )

        return cls(
            idx_pairs=np.asarray(idx_pairs, dtype=int),
            directions=np.asarray(directions, dtype=object),
            azimuth_mean=np.asarray(azimuth_mean, dtype=np.float64),
            block_ids=np.asarray(block_ids, dtype=int),
            elevation_smooth=elevation_smooth,
            elevation_direction=elevation_direction,
            azimuth_blocks=azimuth_blocks,
        )

    def extract_tod_segments(self,
                             dataset: QubicDataset,
                             tes_idx: int) -> list[dict]:
        """
        Extract one TOD segment for each skydip interval for a given TES.

        SkydipIntervals stores only the skydip boundaries. This method uses those
        boundaries to cut the dataset time stream and the selected TES signal.
        """

        segments: list[dict] = []

        for skydip_id, ((start_idx, stop_idx), direction, az_mean) in enumerate(
                zip(
                    self.idx_pairs,
                    self.directions,
                    self.azimuth_mean,
                ),
                start=1,
        ):
            start_idx = int(start_idx)
            stop_idx = int(stop_idx)

            if start_idx < 0 or stop_idx >= dataset.time.size or stop_idx < start_idx:
                raise ValueError(
                    f"Invalid skydip interval for skydip_id={skydip_id}: "
                    f"start={start_idx}, stop={stop_idx}."
                )

            segments.append({
                "skydip_id": skydip_id,
                "start_idx": start_idx,
                "stop_idx": stop_idx,
                "direction": str(direction),
                "azimuth_mean": float(az_mean),
                "time": dataset.time[start_idx:stop_idx + 1],
                "signal": dataset.signals[tes_idx, start_idx:stop_idx + 1],
            })

        return segments


def _expand_run_over_zero_velocity_edges(block: np.ndarray,
                                         block_dir: np.ndarray,
                                         tm: np.ndarray,
                                         run_start: int,
                                         run_stop: int,
                                         run_sign: int,
                                         up_left_extension: float = 3.0,
                                         down_right_extension: float = 1.0) -> tuple[int, int]:
    """
    Expand a monotonic run over adjacent zero-velocity samples with an asymmetric
    rule:
    - for an upward run, extend on the low-elevation side (left edge);
    - for a downward run, extend on the low-elevation side (right edge).
    """
    if run_sign == 0:
        return run_start, run_stop

    left = run_start
    right = run_stop

    if run_sign > 0:
        while left > 0 and block_dir[left - 1] == 0:
            trial_left = left - 1
            if tm[block[run_start]] - tm[block[trial_left]] > up_left_extension:
                break
            left = trial_left
    else:
        while right < len(block_dir) - 1 and block_dir[right + 1] == 0:
            trial_right = right + 1
            if tm[block[trial_right]] - tm[block[run_stop]] > down_right_extension:
                break
            right = trial_right

    return left, right

# def find_skydip_intervals(dataset: "QubicDataset",
#                           config: SkydipCalibrationConfig) -> SkydipIntervals:
#         """
#         Find skydip intervals from the interpolated pointing stored in a QubicDataset.
#
#         This function uses:
#             dataset.time
#             dataset.interp_azimuth
#             dataset.interp_elevation
#
#         and the preprocessing parameters stored in:
#             config.preprocessing
#         """
#
#         az_velocity_threshold = _to_value(config.preprocessing.az_velocity_threshold, u.deg / u.s)
#         el_velocity_threshold = _to_value(config.preprocessing.el_velocity_threshold, u.deg / u.s)
#
#         min_block_duration = _to_value(config.preprocessing.min_block_duration, u.s)
#         min_run_duration = _to_value(config.preprocessing.min_run_duration, u.s)
#
#         el_smooth_window = config.preprocessing.el_smooth_window
#         el_polyorder = config.preprocessing.el_polyorder
#
#
#         up_left_extension = _to_value(config.preprocessing.up_left_extension, u.s)
#         down_right_extension = _to_value(config.preprocessing.down_right_extension, u.s)
#
#         win = _prepare_savgol_window(n_samples=dataset.time.size,
#                                      window_length=el_smooth_window,
#                                      polyorder=el_polyorder)
#
#         elevation_smooth = savgol_filter(dataset.interp_elevation,
#                                          window_length=win,
#                                          polyorder=el_polyorder)
#
#         # Derivata discreta dell'elevation smoothing
#         delv_dt = np.gradient(elevation_smooth, dataset.time)
#
#         elevation_direction = np.zeros_like(delv_dt, dtype=int)
#         elevation_direction[delv_dt > el_velocity_threshold] = 1
#         elevation_direction[delv_dt < -el_velocity_threshold] = -1
#
#         azimuth_blocks = _find_constant_azimuth_blocks(tm=dataset.time,
#                                                        azimuth=dataset.interp_azimuth,
#                                                        az_velocity_threshold=az_velocity_threshold,
#                                                        min_block_duration=min_block_duration)
#
#         idx_pairs = []
#         directions = []
#         azimuth_mean = []
#         block_ids = []
#
#         # Lavoro un blocco ad azimuth costante per volta.
#         for block_id, block in enumerate(azimuth_blocks):
#
#             block_dir = elevation_direction[block]
#             block_az = dataset.interp_azimuth[block]
#
#             # Tiene traccia di un tratto monotono in corso.
#             run_start = None
#             run_sign = 0
#
#             for local_idx, sign in enumerate(block_dir):
#
#                 # Plateau o zona quasi piatta.
#                 if sign == 0:
#
#                     if run_start is not None:
#                         run_stop = local_idx - 1
#
#                         expanded_start, expanded_stop = _expand_run_over_zero_velocity_edges(
#                             block=block,
#                             block_dir=block_dir,
#                             tm=dataset.time,
#                             run_start=run_start,
#                             run_stop=run_stop,
#                             run_sign=run_sign,
#                             up_left_extension=up_left_extension,
#                             down_right_extension=down_right_extension,
#                         )
#
#                         if dataset.time[block[expanded_stop]] - dataset.time[block[expanded_start]] >= min_run_duration:
#                             start_idx = int(block[expanded_start])
#                             stop_idx = int(block[expanded_stop])
#
#                             idx_pairs.append([start_idx, stop_idx])
#                             directions.append("up" if run_sign > 0 else "down")
#                             azimuth_mean.append(float(np.nanmean(block_az)))
#                             block_ids.append(block_id)
#
#                         run_start = None
#                         run_sign = 0
#
#                     continue
#
#                 if run_start is None:
#                     run_start = local_idx
#                     run_sign = int(sign)
#                     continue
#
#                 # Se stavi salendo e ora scendi, oppure viceversa,
#                 # la run precedente è finita.
#                 if sign != run_sign:
#                     run_stop = local_idx - 1
#
#                     expanded_start, expanded_stop = _expand_run_over_zero_velocity_edges(
#                         block=block,
#                         block_dir=block_dir,
#                         tm=dataset.time,
#                         run_start=run_start,
#                         run_stop=run_stop,
#                         run_sign=run_sign,
#                         up_left_extension=up_left_extension,
#                         down_right_extension=down_right_extension,
#                     )
#
#                     if dataset.time[block[expanded_stop]] - dataset.time[block[expanded_start]] >= min_run_duration:
#                         start_idx = int(block[expanded_start])
#                         stop_idx = int(block[expanded_stop])
#
#                         idx_pairs.append([start_idx, stop_idx])
#                         directions.append("up" if run_sign > 0 else "down")
#                         azimuth_mean.append(float(np.nanmean(block_az)))
#                         block_ids.append(block_id)
#
#                     run_start = local_idx
#                     run_sign = int(sign)
#
#             if run_start is not None:
#                 run_stop = len(block_dir) - 1
#
#                 expanded_start, expanded_stop = _expand_run_over_zero_velocity_edges(
#                     block=block,
#                     block_dir=block_dir,
#                     tm=dataset.time,
#                     run_start=run_start,
#                     run_stop=run_stop,
#                     run_sign=run_sign,
#                     up_left_extension=up_left_extension,
#                     down_right_extension=down_right_extension,
#                 )
#
#                 if dataset.time[block[expanded_stop]] - dataset.time[block[expanded_start]] >= min_run_duration:
#                     start_idx = int(block[expanded_start])
#                     stop_idx = int(block[expanded_stop])
#
#                     idx_pairs.append([start_idx, stop_idx])
#                     directions.append("up" if run_sign > 0 else "down")
#                     azimuth_mean.append(float(np.nanmean(block_az)))
#                     block_ids.append(block_id)
#
#         if len(idx_pairs) == 0:
#             return SkydipIntervals(
#                 idx_pairs=np.empty((0, 2), dtype=int),
#                 directions=np.array([], dtype=object),
#                 azimuth_mean=np.array([], dtype=np.float64),
#                 block_ids=np.array([], dtype=int),
#                 elevation_smooth=elevation_smooth,
#                 elevation_direction=elevation_direction,
#                 azimuth_blocks=azimuth_blocks,
#             )
#
#         return SkydipIntervals(
#             idx_pairs=np.asarray(idx_pairs, dtype=int),
#             directions=np.asarray(directions, dtype=object),
#             azimuth_mean=np.asarray(azimuth_mean, dtype=np.float64),
#             block_ids=np.asarray(block_ids, dtype=int),
#             elevation_smooth=elevation_smooth,
#             elevation_direction=elevation_direction,
#             azimuth_blocks=azimuth_blocks,
#         )

if __name__ == "__main__":

    import sys
    import logging

    from qubic.lib.Calibration.source_calibration.common.io import iter_saved_datasets
    from qubic.lib.Calibration.source_calibration.skydip.config_calibration import (
        load_skydip_calibration_config,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d/%m/%Y | %H:%M:%S",
    )

    logger = logging.getLogger(__name__)

    config = load_skydip_calibration_config(
        Path(sys.argv[1]).expanduser().resolve()
    )

    for dataset in iter_saved_datasets(
        config=config,
        logger=logger,
    ):

        logger.info(
            "Finding skydip intervals for dataset: %s",
            dataset.dataset_name,
        )

        skydip_intervals = SkydipIntervals.from_dataset(dataset=dataset, config=config)

        print()
        print(f"Dataset: {dataset.dataset_name}")
        print(f"Found {len(skydip_intervals.idx_pairs)} skydips.")
        print(f"Found {len(skydip_intervals.azimuth_blocks)} constant-azimuth blocks.")

        for i, ((start_idx, stop_idx), direction, az_mean, block_id) in enumerate(
            zip(
                skydip_intervals.idx_pairs,
                skydip_intervals.directions,
                skydip_intervals.azimuth_mean,
                skydip_intervals.block_ids,
            ),
            start=1,
        ):
            print(
                f"Skydip {i}: "
                f"block={block_id}, "
                f"direction={direction}, "
                f"start_idx={start_idx}, "
                f"stop_idx={stop_idx}, "
                f"start_time={dataset.time[start_idx]:.2f} s, "
                f"stop_time={dataset.time[stop_idx]:.2f} s, "
                f"az_mean={az_mean:.2f} deg"
            )
