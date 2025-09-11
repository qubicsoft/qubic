import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import torch
from pygsp import graphs
from scipy import sparse

from qubic.lib.AnalyticalSolution.utils import get_high_coverage_indexes


def healpix_weightmatrix(nside=16, nest=True, indexes=None, dtype=np.float32):
    """Sparse adjacency for a HEALPix graph (8-neighbour connectivity)."""

    if not nest:
        raise NotImplementedError("Only NEST ordering is supported.")

    if indexes is None:
        indexes = range(12 * nside**2)

    npix = len(indexes)
    if npix >= (max(indexes) + 1):
        usefast = True
        indexes = range(npix)
    else:
        usefast = False
        indexes = list(indexes)

    x, y, z = hp.pix2vec(nside, indexes, nest=nest)
    coords = np.vstack([x, y, z]).T.astype(dtype)

    neighbors = hp.get_all_neighbours(nside, indexes, nest=nest)
    col_index = neighbors.T.reshape(npix * 8)
    row_index = np.repeat(indexes, 8)

    if usefast:
        keep = (col_index < npix) & (col_index >= 0)
        col_index = col_index[keep]
        row_index = row_index[keep]
    else:
        col_set = set(indexes)
        keep = [c in col_set for c in col_index]
        inv_map = [np.nan] * (12 * nside**2)
        for i, idx in enumerate(indexes):
            inv_map[idx] = i
        col_index = [inv_map[el] for el, k in zip(col_index, keep) if k]
        row_index = [inv_map[el] for el, k in zip(row_index, keep) if k]

    coords = np.asarray(coords, dtype=dtype)
    distances = np.sum((coords[row_index] - coords[col_index]) ** 2, axis=1)
    kernel_width = float(np.mean(distances))
    weights = np.exp(-distances / (2 * kernel_width)).astype(dtype)

    return sparse.csr_matrix((weights, (row_index, col_index)), shape=(npix, npix), dtype=dtype)


def healpix_graph(nside=16, nest=True, lap_type="normalized", indexes=None, dtype=np.float32):
    """Pygsp Graph on HEALPix sphere with xyz coordinates and zero signal."""

    if indexes is None:
        indexes = range(12 * nside**2)

    all_pix = range(hp.nside2npix(nside))
    x, y, z = hp.pix2vec(nside, all_pix, nest=nest)
    coords = np.vstack([x, y, z]).T[indexes]

    W = healpix_weightmatrix(nside=nside, nest=nest, indexes=indexes, dtype=dtype)
    G = graphs.Graph(W, lap_type=lap_type, coords=coords)
    G.signal = np.zeros(len(indexes), dtype=dtype)
    return G


def get_nside_from_graph(G):
    """Infer nside from graph size (assumes full-sky)."""

    return int(np.sqrt(G.N / 12))


def get_high_coverage_indexes(cov_map, threshold=0.15):
    """Ring-ordered pixel indices with coverage ≥ threshold · max(coverage)."""

    cov = np.asarray(cov_map)
    unseen = cov == hp.UNSEEN
    vmax = np.nanmax(cov[~unseen]) if (~unseen).any() else 0.0
    if vmax <= 0:
        return np.array([], dtype=int)
    mask = (~unseen) & (cov >= threshold * vmax)
    return np.flatnonzero(mask).astype(int)


def get_G_masked_by_cov(G, nside, cov_map, threshold=0.15):
    """Subgraph of G for pixels above a coverage threshold (coverage is RING)."""

    seen_ring = get_high_coverage_indexes(cov_map, threshold=threshold)
    seen_nest = hp.ring2nest(nside, seen_ring)
    Gp = G.subgraph(seen_nest)
    Gp.signal = G.signal[seen_nest]
    Gp.coords = G.coords[seen_nest]
    return Gp


def subgraph_from_seen(G, seen_indexes_nest):
    """Subgraph from an array of NEST indices."""

    idx = np.asarray(seen_indexes_nest, dtype=int)
    Gp = G.subgraph(idx)
    Gp.signal = G.signal[idx]
    Gp.coords = G.coords[idx]
    return Gp


def plot_sky_3d(graph_map, elev=10, azim=0, edges=False, vmin=None, vmax=None):
    """3D plot of a graph signal on the HEALPix sphere."""

    sig = graph_map.signal
    if (torch is not None) and isinstance(sig, torch.Tensor):
        sig = sig.detach().cpu().numpy()
    sig = sig.copy()
    if vmin is None:
        good = sig != hp.UNSEEN
        vmin = sig[good].min() if good.any() else np.min(sig)
    if vmax is None:
        vmax = sig.max()

    graph_map.plotting.update(vertex_size=0.6)
    fig = plt.figure(figsize=[8, 6])
    ax = fig.add_subplot(111, projection="3d")
    graph_map.plot_signal(sig, show_edges=edges, ax=ax, limits=[vmin, vmax])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=elev, azim=azim)
    plt.show()


# ---------- Multi-feature variants (I,Q,U,Coverage) ----------


def healpix_weightmatrix_multifeature(nside=16, nest=True, indexes=None, dtype=np.float32):
    """Adjacency for multi-feature graphs (same as scalar case)."""

    return healpix_weightmatrix(nside=nside, nest=nest, indexes=indexes, dtype=dtype)


def healpix_graph_multifeature(nside=16, nest=True, lap_type="normalized", indexes=None, dtype=np.float32):
    """Pygsp Graph with signal shape (N,4): [I,Q,U,Coverage]."""

    if indexes is None:
        indexes = range(12 * nside**2)

    all_pix = range(hp.nside2npix(nside))
    x, y, z = hp.pix2vec(nside, all_pix, nest=nest)
    coords = np.vstack([x, y, z]).T[indexes]

    W = healpix_weightmatrix_multifeature(nside=nside, nest=nest, indexes=indexes, dtype=dtype)
    G = graphs.Graph(W, lap_type=lap_type, coords=coords)
    G.signal = np.zeros((len(indexes), 4), dtype=dtype)
    return G


def get_G_masked_by_cov_multifeature(G, nside, cov_map, threshold=0.15):
    """Subgraph for multi-feature graph using coverage threshold (coverage is RING)."""

    seen_ring = get_high_coverage_indexes(cov_map, threshold=threshold)
    seen_nest = hp.ring2nest(nside, seen_ring)
    Gp = G.subgraph(seen_nest)
    Gp.signal = G.signal[seen_nest]
    Gp.coords = G.coords[seen_nest]
    return Gp


def plot_sky_3d_multifeature(graph_map, feature_index=0, elev=10, azim=0, edges=False, vmin=None, vmax=None):
    """3D plot of a selected feature from a multi-feature graph."""

    sig = graph_map.signal[:, feature_index]
    if (torch is not None) and isinstance(sig, torch.Tensor):
        sig = sig.detach().cpu().numpy()
    sig = sig.copy()
    if vmin is None:
        good = sig != hp.UNSEEN
        vmin = sig[good].min() if good.any() else np.min(sig)
    if vmax is None:
        vmax = sig.max()

    graph_map.plotting.update(vertex_size=0.6)
    fig = plt.figure(figsize=[8, 6])
    ax = fig.add_subplot(111, projection="3d")
    graph_map.plot_signal(sig, show_edges=edges, ax=ax, limits=[vmin, vmax])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=elev, azim=azim)
    plt.show()
