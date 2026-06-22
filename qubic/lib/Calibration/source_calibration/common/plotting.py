import csv

import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qubic.lib.Calibration.Qfiber import pix2tes, pix_grid

from qubic.lib.Calibration.source_calibration.common.io import QubicDataset
from qubic.lib.Calibration.source_calibration.common.preprocessing import (
    SkydipIntervals,
    _find_constant_azimuth_blocks,
)
from qubic.lib.Calibration.source_calibration.skydip.calibration import SkydipCalibrationSegment
from qubic.lib.Calibration.source_calibration.skydip.config_calibration import (
    SkydipCalibrationConfig)

def _auto_histogram_nbins(values: np.ndarray) -> int:
    """
    Choose a reasonable number of histogram bins automatically.

    Uses the Freedman-Diaconis rule when possible, with simple fallbacks for
    small or nearly constant samples.
    """

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    n = values.size

    if n <= 1:
        return 1

    vmin = float(np.min(values))
    vmax = float(np.max(values))

    if np.isclose(vmin, vmax):
        return 1

    q25, q75 = np.percentile(values, [25, 75])
    iqr = float(q75 - q25)

    if iqr <= 0 or not np.isfinite(iqr):
        return int(np.ceil(np.sqrt(n)))

    bin_width = 2.0 * iqr / np.cbrt(n)

    if bin_width <= 0 or not np.isfinite(bin_width):
        return int(np.ceil(np.sqrt(n)))

    nbins = int(np.ceil((vmax - vmin) / bin_width))

    return max(1, nbins)


def plot_tod(dataset: QubicDataset,
             config: SkydipCalibrationConfig,
             tes_idx: int,
             skydip_intervals: SkydipIntervals | None = None):
    """
    Plot the TOD of a single TES.
    """

    if config.tod_processing.centered:
        processing = "centered"
    elif config.tod_processing.normalized:
        processing = "normalized"
    else:
        processing = "raw"

    tod = dataset.signals[tes_idx]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataset.time, y=tod, mode="lines"))
    fig.update_traces(line=dict(width=config.plots.linewidth),
                      opacity=config.plots.alpha)

    if skydip_intervals is not None:

        # Overlay degli intervalli skydip
        for i, ((start_idx, stop_idx), direction, az_mean) in enumerate(
                zip(skydip_intervals.idx_pairs,
                    skydip_intervals.directions,
                    skydip_intervals.azimuth_mean), start=1):

            start_idx = int(start_idx)
            stop_idx = int(stop_idx)

            shade_color = ("rgba(0, 128, 0, 0.08)"
                           if direction == "up" else "rgba(255, 0, 0, 0.08)")

            line_color = ("rgba(0, 128, 0, 0.85)"
                          if direction == "up" else "rgba(255, 0, 0, 0.85)")

            fig.add_vrect(x0=float(dataset.time[start_idx]),
                          x1=float(dataset.time[stop_idx]),
                          fillcolor=shade_color,
                          line_width=0,
                          layer="below")

            fig.add_vline(x=float(dataset.time[start_idx]),
                          line_dash="dash",
                          line_width=1.2,
                          line_color=line_color)

            fig.add_vline(x=float(dataset.time[stop_idx]),
                          line_dash="dot",
                          line_width=1.2,
                          line_color=line_color)

            fig.add_annotation(
                x=0.5 * (float(dataset.time[start_idx]) + float(dataset.time[stop_idx])),
                y=1.03,
                xref="x",
                yref="paper",
                text=f"{i} ({direction}, az={az_mean:.0f})",
                showarrow=False,
                font=dict(size=10))

        fig.update_layout(title=f"{dataset.dataset_name} - TOD TES {tes_idx} with skydip limits - {processing}",
                          xaxis_title="Time [s]",
                          yaxis_title="Signal [ADU]")

        output_file = (dataset.preprocessing_plots_dir / f"tes_{tes_idx}" / f"tes_{tes_idx}_{processing}_with_skydip_limits.html")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_file)

    else:

        fig.update_layout(title=dict(text=f"{dataset.dataset_name} - TOD TES {tes_idx} - {processing}"),
                          xaxis_title="Time [s]",
                          yaxis_title="Signal [ADU]")

        output_file = dataset.preprocessing_plots_dir / f"tes_{tes_idx}" / f"tes_{tes_idx}_{processing}.html"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_file)

    if config.plots.show:
        fig.show(renderer="browser")


def plot_az_el_vs_time(dataset: QubicDataset,
                       config: SkydipCalibrationConfig,
                       is_interpolated: bool = True):
    """
    Plot azimuth and elevation versus time for one QubicDataset.
    If is_interpolated=False, uses the original housekeeping time axes:
        dataset.elevation_time, dataset.elevation
        dataset.azimuth_time, dataset.azimuth
    If is_interpolated=True, uses the TOD time axis:
        dataset.time, dataset.interp_elevation
        dataset.time, dataset.interp_azimuth

    In the interpolated case, constant-azimuth blocks are always detected
    and overlaid on both the elevation and azimuth panels.

    The output path is taken from dataset.housekeeping_dir.
    """

    azimuth_blocks = None

    if is_interpolated:
        tm_az = tm_el = dataset.time
        az = dataset.interp_azimuth
        el = dataset.interp_elevation

        output_file = dataset.housekeeping_dir / "interp_az_el_vs_time.html"
        title = f"{dataset.dataset_name} - Interpolated azimuth/elevation"

        # Nel caso interpolato calcolo SEMPRE i blocchi ad azimuth costante
        azimuth_blocks = _find_constant_azimuth_blocks(tm=tm_az,
                                                       azimuth=az,
                                                       az_velocity_threshold=config.preprocessing.az_velocity_threshold,
                                                       min_block_duration=config.preprocessing.min_block_duration)
    else:
        tm_el = dataset.elevation_time
        tm_az = dataset.azimuth_time
        az = dataset.azimuth
        el = dataset.elevation

        output_file = dataset.housekeeping_dir / "az_el_vs_time.html"
        title = f"{dataset.dataset_name} - azimuth/elevation"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tm_el, y=el, mode='lines', name='Elevation', xaxis='x', yaxis='y'))
    fig.add_trace(go.Scatter(x=tm_az, y=az, mode='lines', name='Azimuth', xaxis='x2', yaxis='y2'))
    fig.update_traces(line=dict(width=config.plots.linewidth), opacity=config.plots.alpha)

    if azimuth_blocks:

        for block_id, block in enumerate(azimuth_blocks, start=1):

            # se il blocco e' vuoto, passa a quello successivo
            if block.size == 0:
                continue

            start_idx = int(block[0])
            stop_idx = int(block[-1])

            x0 = float(tm_az[start_idx])
            x1 = float(tm_az[stop_idx])
            az_mean = float(np.nanmean(az[block]))

            # Disegno lo stesso intervallo sia su elevation sia su azimuth.
            for row in (1, 2):

                fig.add_vrect(
                    x0=x0,
                    x1=x1,
                    fillcolor="rgba(255, 0, 0, 0.06)",
                    line_width=0,
                    layer="below",
                    row=row,
                    col=1,
                )

                fig.add_vline(
                    x=x0,
                    line_dash="dash",
                    line_width=1.1,
                    line_color="rgba(255, 0, 0, 0.75)",
                    row=row,
                    col=1,
                )

                fig.add_vline(
                    x=x1,
                    line_dash="dot",
                    line_width=1.1,
                    line_color="rgba(255, 0, 0, 0.75)",
                    row=row,
                    col=1,
                )

            fig.add_annotation(
                x=0.5 * (x0 + x1),
                y=1.03,
                xref="x",
                yref="paper",
                text=f"block {block_id}<br>az={az_mean:.0f} deg",
                showarrow=False,
                font=dict(size=10),
            )

    fig.update_layout(title=title,
                      grid=dict(rows=2, columns=1, pattern='independent'),
                      xaxis=dict(title="Time [s]"),
                      yaxis=dict(title="Elevation [deg]"),
                      xaxis2=dict(title="Time [s]"),
                      yaxis2=dict(title="Azimuth [deg]"),
                      showlegend=False)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_file)
    if config.plots.show:
        fig.show(renderer="browser")



def plot_elevation_with_skydip_limits(dataset: QubicDataset,
                                      config: SkydipCalibrationConfig,
                                      skydip_intervals: SkydipIntervals) -> None:
    """
    Plot interpolated elevation and highlight detected skydip intervals
    """

    tm = dataset.time
    elevation = dataset.interp_elevation
    elevation_smooth = skydip_intervals.elevation_smooth
    fig = go.Figure()

    # Interpolated elevation
    fig.add_trace(
        go.Scatter(
            x=tm,
            y=elevation,
            mode="lines",
            name="Elevation",
            line=dict(width=config.plots.linewidth),
            opacity=config.plots.alpha,
        )
    )

    # Smoothed elevation used for skydip detection
    fig.add_trace(
        go.Scatter(
            x=tm,
            y=elevation_smooth,
            mode="lines",
            name="Smoothed elevation",
            line=dict(width=config.plots.linewidth + 0.3),
            opacity=config.plots.alpha,
        )
    )

    # Overlay degli intervalli skydip
    for i, ((start_idx, stop_idx), direction, az_mean) in enumerate(
        zip(
            skydip_intervals.idx_pairs,
            skydip_intervals.directions,
            skydip_intervals.azimuth_mean,
        ),
        start=1,
    ):
        start_idx = int(start_idx)
        stop_idx = int(stop_idx)

        shade_color = (
            "rgba(0, 128, 0, 0.10)"
            if direction == "up"
            else "rgba(255, 0, 0, 0.10)"
        )

        line_color = (
            "rgba(0, 128, 0, 0.85)"
            if direction == "up"
            else "rgba(255, 0, 0, 0.85)"
        )

        fig.add_vrect(
            x0=float(tm[start_idx]),
            x1=float(tm[stop_idx]),
            fillcolor=shade_color,
            line_width=0,
            layer="below",
        )

        fig.add_vline(
            x=float(tm[start_idx]),
            line_dash="dash",
            line_width=1.2,
            line_color=line_color,
        )

        fig.add_vline(
            x=float(tm[stop_idx]),
            line_dash="dot",
            line_width=1.2,
            line_color=line_color,
        )

        fig.add_annotation(
            x=0.5 * (float(tm[start_idx]) + float(tm[stop_idx])),
            y=1.03,
            xref="x",
            yref="paper",
            text=f"{i} ({direction}, az={az_mean:.0f})",
            showarrow=False,
            font=dict(size=10),
        )

    fig.update_layout(
        title=f"{dataset.dataset_name} - Elevation with detected skydips",
        xaxis_title="Time [s]",
        yaxis_title="Elevation [deg]",
    )

    output_file = dataset.housekeeping_dir / "elevation_with_skydip_limits.html"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_file)

    if config.plots.show:
        fig.show(renderer="browser")


def plot_focal_plane_tods(dataset: QubicDataset,
                          config: SkydipCalibrationConfig,
                          flip_ud: bool = False,
                          flip_lr: bool = False) -> None:
    """
    Plot the TOD of each TES in its focal-plane position.
    """

    plot_data = dataset.signals

    if config.tod_processing.centered:
        processing = "centered"
    elif config.tod_processing.normalized:
        processing = "normalized"
    else:
        processing = "raw"

    nrows, ncols = pix_grid.shape

    subplot_titles_grid = [["" for _ in range(ncols)] for _ in range(nrows)]
    trace_specs: list[tuple[int, int, int, np.ndarray]] = []

    for row in range(nrows):
        for col in range(ncols):

            display_row = nrows - 1 - row if flip_ud else row
            display_col = ncols - 1 - col if flip_lr else col

            phys_pix = pix_grid[row, col]
            tes_info = pix2tes(phys_pix)

            if tes_info is None:
                continue

            tes_num, asic_num = tes_info

            if tes_num is None or asic_num is None:
                continue

            global_tes_idx = tes_num - 1 + (asic_num - 1) * 128

            if global_tes_idx < 0 or global_tes_idx >= plot_data.shape[0]:
                continue

            y = plot_data[global_tes_idx]

            if not np.any(np.isfinite(y)):
                continue

            subplot_titles_grid[display_row][display_col] = f"TES {global_tes_idx}"
            trace_specs.append(
                (
                    display_row + 1,
                    display_col + 1,
                    global_tes_idx,
                    y,
                )
            )

    subplot_titles = [
        subplot_titles_grid[row][col]
        for row in range(nrows)
        for col in range(ncols)
    ]

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.004,
        vertical_spacing=0.012,
    )

    x = dataset.time

    asic_colors = {
        1: "#1f77b4",
        2: "#ff7f0e",
    }

    for display_row, display_col, global_tes_idx, y in trace_specs:

        asic_id = 1 if global_tes_idx < 128 else 2

        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="lines",
                name=f"TES {global_tes_idx}",
                line=dict(
                    width=config.plots.linewidth,
                    color=asic_colors[asic_id],
                ),
                opacity=config.plots.alpha,
                showlegend=False,
                hovertemplate=(
                    f"TES {global_tes_idx}<br>"
                    f"ASIC {asic_id}<br>"
                    "time=%{x:.2f} s<br>"
                    "signal=%{y:.3g}<extra></extra>"
                ),
            ),
            row=display_row,
            col=display_col,
        )

    fig.update_layout(
        title=f"{dataset.dataset_name} - Focal-plane TODs - {processing}",
        width=2100,
        height=2200,
        margin=dict(l=40, r=40, t=170, b=60),
        showlegend=True,
    )

    # Dummy traces solo per avere una legenda ASIC 1 / ASIC 2.
    fig.add_trace(
        go.Scattergl(
            x=[None],
            y=[None],
            mode="lines",
            name="ASIC 1",
            line=dict(width=3, color=asic_colors[1]),
            showlegend=True,
        )
    )

    fig.add_trace(
        go.Scattergl(
            x=[None],
            y=[None],
            mode="lines",
            name="ASIC 2",
            line=dict(width=3, color=asic_colors[2]),
            showlegend=True,
        )
    )

    for annotation in fig.layout.annotations:
        annotation.font.size = 8
        annotation.yshift = 6

    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
    )

    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
    )

    output_file = (
        dataset.preprocessing_plots_dir
        / f"focal_plane_tods_{processing}.html"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_file)

    if config.plots.show:
        fig.show(renderer="browser")


def plot_skydips_vs_Tatm(segments: list[SkydipCalibrationSegment],
                         config: SkydipCalibrationConfig,
                         tes_idx: int,
                         output_dir: Path):

    for segment in segments:

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x = segment.t_atm,
                y = segment.signal,
                mode = "lines",
                line = dict(width=config.plots.linewidth),
                opacity = config.plots.alpha,
                name = (f"Skydip {segment.skydip_id} "
                        f"({segment.direction}, az={segment.azimuth_mean:.0f})")
            )
        )

        signal_fit = segment.slope_adu_per_k * segment.t_atm + segment.intercept_adu

        fig.add_trace(
            go.Scatter(
                x = segment.t_atm,
                y = signal_fit,
                mode = "lines",
                line = dict(width=config.plots.fit_linewidth,
                        dash="dash"),
                opacity = config.plots.alpha,
                name = f"Fit: {segment.slope_adu_per_k:.3f} ADU/K"
            )
        )

        fig.update_layout(
            title = f"Skydips vs Tatm",
            xaxis_title = "Tatm [K]",
            yaxis_title = "Skydip signal [ADU]",
            showlegend = True,
        )

        if output_dir is not None:
            output_file = (Path(output_dir) / f"tes_{tes_idx}" / (f"skydip_{segment.skydip_id:02d}_"
                                               f"{segment.direction}_"
                                               f"az_{segment.azimuth_mean:.0f}_"
                                               f"signal_vs_Tatm.html"))

            output_file.parent.mkdir(parents=True, exist_ok=True)

            fig.write_html(output_file)

        if config.plots.show:
            fig.show(renderer="browser")


def plot_Tatm_vs_airmass(segments: list[SkydipCalibrationSegment],
                         config: SkydipCalibrationConfig,
                         tes_idx: int,
                         output_dir: Path):

    for segment in segments:

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x = segment.airmass,
                y = segment.t_atm,
                mode = "lines",
                line = dict(width=config.plots.linewidth),
                opacity = config.plots.alpha,
                name = (f"Skydip {segment.skydip_id} "
                        f"({segment.direction}, az={segment.azimuth_mean:.0f})")
            )
        )

        fig.update_layout(
            title = (
                f"Skydip {segment.skydip_id} - T_Atm vs airmass "
                f"({segment.direction}, az={segment.azimuth_mean:.0f})"
            ),
            xaxis_title="Airmass",
            yaxis_title="T_Atm [K]",
            showlegend=True,
        )

        output_file = (
                Path(output_dir)
                /  f"tes_{tes_idx}" /(f"skydip_{segment.skydip_id:02d}_{segment.direction}_"
                    f"az_{segment.azimuth_mean:.0f}_"
                    f"Tatm_vs_airmass.html"
                )
        )

        output_file.parent.mkdir(parents=True, exist_ok=True)

        fig.write_html(output_file)

        if config.plots.show:
            fig.show(renderer="browser")


def plot_conversion_factors_histogram(csv_file: Path,
                                      config: SkydipCalibrationConfig,
                                      tes_idx: int,
                                      output_dir: Path):

    slopes = []

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            slopes.append(float(row.get("slope_adu_per_k", "")))

        slopes = np.asarray(slopes, dtype=np.float64)
        median = float(np.median(slopes))
        mad = float(np.median(np.abs(slopes - median)))

        n_bins = _auto_histogram_nbins(slopes)

        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=slopes,
                name="Conversion Factors [ADU/K]",
                opacity=config.plots.alpha,
                nbinsx=n_bins
            )
        )

        fig.add_vline(x=median,
                      line_width=2.0,
                      line_dash="dash",
                      annotation_text=f"Median: {median:.3f} ADU/K",
                      annotation_position="top right")

        fig.add_annotation(xref="paper",
                           yref="paper",
                           x=0.98,
                           y=0.98,
                           xanchor="right",
                           yanchor="top",
                           showarrow=False,
                           align="right",
                           text=(f"N: {slopes.size}<br>"
                                 f"Bins: {n_bins}<br>"
                                 f"Median: {median:.3g} ADU/K<br>"
                                 f"MAD: {mad:.3g} ADU/K<br>"),
                           bgcolor="rgba(255,255,255,0.75)",
                           bordercolor="rgba(0,0,0,0.25)",
                           borderwidth=1)

        fig.update_layout(title="Histogram of skydip slopes",
                          xaxis_title="Slope [ADU/K]",
                          yaxis_title="Count",
                          showlegend=False)

        output_file = (
                Path(output_dir)
                / f"tes_{tes_idx}" / "conversion_factors_histogram.html")

        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_file)

        if config.plots.show:
            fig.show(renderer="browser")

def plot_noise_plateau_vs_tau(noise_vs_tau_results: list,
                              output_path: str | Path | None = None,
                              title: str | None = None,
                              show: bool = True,
                              renderer: str = "browser") -> None:
    """
    Plot the dataset-averaged skydip noise plateau ASD versus atmospheric tau.

    Each item in noise_vs_tau_results is expected to expose the attributes
    produced by DatasetNoiseVsTauResult in run_calibration.py:
        - dataset_name
        - tau_eff
        - mean_plateau_asd_k_per_sqrt_hz
        - std_plateau_asd_k_per_sqrt_hz
        - mean_knee_frequency_hz
        - n_tes
    """

    if not noise_vs_tau_results:
        raise ValueError("noise_vs_tau_results is empty.")

    tau_values = np.asarray(
        [result.tau_eff for result in noise_vs_tau_results],
        dtype=np.float64,
    )

    plateau_values = np.asarray(
        [result.mean_plateau_asd_k_per_sqrt_hz for result in noise_vs_tau_results],
        dtype=np.float64,
    )

    plateau_errors = np.asarray(
        [result.std_plateau_asd_k_per_sqrt_hz for result in noise_vs_tau_results],
        dtype=np.float64,
    )

    knee_values = np.asarray(
        [result.mean_knee_frequency_hz for result in noise_vs_tau_results],
        dtype=np.float64,
    )

    dataset_names = [
        result.dataset_name for result in noise_vs_tau_results
    ]

    n_tes_values = [
        result.n_tes for result in noise_vs_tau_results
    ]

    valid = (
        np.isfinite(tau_values)
        & np.isfinite(plateau_values)
        & np.isfinite(plateau_errors)
        & np.isfinite(knee_values)
    )

    if not np.any(valid):
        raise ValueError("No finite noise-vs-tau points were found.")

    tau_values = tau_values[valid]
    plateau_values = plateau_values[valid]
    plateau_errors = plateau_errors[valid]
    knee_values = knee_values[valid]
    dataset_names = [
        name for name, keep in zip(dataset_names, valid) if keep
    ]
    n_tes_values = [
        n_tes for n_tes, keep in zip(n_tes_values, valid) if keep
    ]

    order = np.argsort(tau_values)

    tau_values = tau_values[order]
    plateau_values = plateau_values[order]
    plateau_errors = plateau_errors[order]
    knee_values = knee_values[order]
    dataset_names = [
        dataset_names[i] for i in order
    ]
    n_tes_values = [
        n_tes_values[i] for i in order
    ]

    customdata = np.column_stack(
        [
            knee_values,
            n_tes_values,
        ]
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=tau_values,
            y=plateau_values,
            mode="markers+lines",
            name="Dataset mean plateau ASD",
            text=dataset_names,
            customdata=customdata,
            error_y=dict(
                type="data",
                array=plateau_errors,
                visible=True,
            ),
            hovertemplate=(
                "Dataset: %{text}<br>"
                "tau: %{x:.4g}<br>"
                "plateau ASD: %{y:.4g} K/√Hz<br>"
                "mean knee frequency: %{customdata[0]:.4g} Hz<br>"
                "n TES: %{customdata[1]:.0f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title or "Dataset-averaged skydip noise plateau versus atmospheric tau",
        xaxis_title="Tau",
        yaxis_title="Mean plateau ASD [K/√Hz]",
        showlegend=True,
    )
    fig.update_yaxes(type="log")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)

    if show:
        fig.show(renderer=renderer)

def plot_skydip_noise_spectra(noise_spectra: list,
                              config: SkydipCalibrationConfig,
                              tes_idx: int,
                              output_dir: Path,
                              is_calibrated: bool = False):
    """
    Plot the skydip noise spectra.

    The input is expected to be a list of SkydipNoiseSpectrum objects produced by
    compute_all_skydip_noise_spectra.

    Supported y_key values are:
        - "asd_adu_per_sqrt_hz" for raw spectra in ADU/sqrt(Hz);
        - "asd_k_per_sqrt_hz" for calibrated spectra in K/sqrt(Hz).
    """
    if is_calibrated:
        y_axis_title = "ASD [ADU/√Hz]"
        title = "Calibrated skydip noise spectra"
        output_file = (Path(output_dir) / f"tes_{tes_idx}" / "calibrated_skydip_noise_spectra.html")

    else:
        y_axis_title = "ASD [K/√Hz]"
        title = "Raw skydip noise spectra"
        output_file = (Path(output_dir) / f"tes_{tes_idx}" / "raw_skydip_noise_spectra.html")

    fig = go.Figure()
    n_plotted = 0

    for spectrum in noise_spectra:

        if is_calibrated:
            asd = spectrum.asd_k_per_sqrt_hz

        else:
            asd = spectrum.asd_adu_per_sqrt_hz

        if asd is None:
            continue

        order = np.argsort(spectrum.frequency_hz)
        frequency_hz = spectrum.frequency_hz[order]
        asd = asd[order]

        fig.add_trace(
            go.Scatter(
                x=frequency_hz,
                y=asd,
                mode="lines",
                name=(
                    f"Skydip {spectrum.skydip_id} "
                    f"({spectrum.direction}, az={spectrum.azimuth_mean:.0f})"
                ),
                line=dict(width=config.plots.linewidth),
                opacity=config.plots.alpha,
                hovertemplate=(
                    f"Skydip {spectrum.skydip_id}<br>"
                    f"direction: {spectrum.direction}<br>"
                    f"azimuth: {spectrum.azimuth_mean:.2f} deg<br>"
                    "frequency: %{x:.4g} Hz<br>"
                    "ASD: %{y:.4g}<extra></extra>"
                ),
            )
        )

        n_plotted += 1


    fig.update_layout(
        title=title,
        xaxis_title="Frequency [Hz]",
        yaxis_title=y_axis_title,
        showlegend=True,
    )

    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")

    if output_dir is not None:
        output_file = (Path(output_dir) / f"tes_{tes_idx}" / "raw_skydip_noise_spectra.html")

        output_file.parent.mkdir(parents=True, exist_ok=True)

        fig.write_html(output_file)

    if config.plots.show:
        fig.show(renderer="browser")