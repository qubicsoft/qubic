from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from qubic.lib.Calibration.Qfiber import pix2tes, pix_grid



def load_tods_from_npz(npz_path: str | Path, key: str = "tods") -> np.ndarray:
    """
    Load TODs from a .npz file saved by save_dataset_arrays().

    Parameters
    ----------
    npz_path
        Path to the .npz file containing the TOD matrix.
    key
        Name of the array stored in the .npz file. Default is ``tods``.

    Returns
    -------
    np.ndarray
        TOD matrix with shape (n_tes, n_samples).
    """
    npz_path = Path(npz_path)

    with np.load(npz_path) as data:
        if key not in data:
            raise KeyError(f"Key '{key}' not found in {npz_path}.")
        tods = np.asarray(data[key], dtype=np.float64)

    if tods.ndim != 2:
        raise ValueError(
            f"TOD data in {npz_path} must be a 2D array with shape (n_tes, n_samples)."
        )

    return tods



def plot_focal_plane_tods(tods: np.ndarray,
                          output_path: str | Path,
                          tes_to_plot: list[int] | None = None,
                          centered: bool = False,
                          normalized: bool = False,
                          alpha: float = 0.8,
                          linewidth: float = 0.4,
                          title: str | None = None,
                          flip_ud: bool = False,
                          flip_lr: bool = False) -> None:
    """
    Plot the full TOD of each TES in its focal-plane position.
    """
    if not isinstance(tods, np.ndarray) or tods.ndim != 2:
        raise ValueError("TOD data must be a 2D numpy array with shape (n_tes, n_samples).")

    plot_data = tods.astype(np.float64).copy()

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
            display_row = nrows - 1 - row if flip_ud else row
            display_col = ncols - 1 - col if flip_lr else col

            ax = axes[display_row, display_col]
            phys_pix = pix_grid[row, col]
            tes_info = pix2tes(phys_pix)

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_alpha(0.3)

            if tes_info is None:
                ax.set_facecolor("#F0F0F0")
                continue

            tes_num, asic_num = tes_info

            if tes_num is None or asic_num is None:
                ax.set_facecolor("#F0F0F0")
                continue

            global_tes_idx = tes_num - 1 + (asic_num - 1) * 128

            if global_tes_idx < 0 or global_tes_idx >= plot_data.shape[0]:
                ax.set_facecolor("#F0F0F0")
                continue

            if tes_to_plot is not None and global_tes_idx not in tes_to_plot:
                ax.set_facecolor("white")
                continue

            y = plot_data[global_tes_idx]
            if not np.any(np.isfinite(y)):
                ax.set_facecolor("#F0F0F0")
                continue

            ax.plot(y, linewidth=linewidth, alpha=alpha)
            ax.set_title(f"TES {global_tes_idx}", fontsize=6, pad=1)

    fig.suptitle(title or "Focal-plane TODs", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":

    dataset_name = "2026-03-11_16.48.15__SkyDip"
    input_dir = Path("/Volumes/Data/PycharmProjects/calibration/qubic/scripts/Calibration/skydip_calibration/data/output") / dataset_name / "input"
    output_dir = Path("/Volumes/Data/PycharmProjects/calibration/qubic/scripts/Calibration/skydip_calibration/data/output") / dataset_name / "plots"

    tods_path = input_dir / f"signals_{dataset_name}.npz"
    tods = load_tods_from_npz(tods_path)

    plot_focal_plane_tods(tods=tods,
                          output_path=output_dir / "focal_plane_tods.png",
                          centered=False,
                          normalized=True,
                          title=f"Focal-plane TODs - {dataset_name}",
                          flip_ud=True,
                          flip_lr=True)