import numpy as np
import healpy as hp

def get_high_coverage_indexes(cov_map, threshold=0.15):
    """Ring-ordered pixel indices with coverage ≥ threshold x max(coverage)."""

    cov = np.asarray(cov_map)
    unseen = (cov == hp.UNSEEN)
    vmax = np.nanmax(cov[~unseen]) if (~unseen).any() else 0.0
    if vmax <= 0:
        return np.array([], dtype=int)
    mask = (~unseen) & (cov >= threshold * vmax)
    return np.flatnonzero(mask).astype(int)