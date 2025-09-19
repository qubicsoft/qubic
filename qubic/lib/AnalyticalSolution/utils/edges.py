import numpy as np
import torch
from itertools import combinations

def build_coobs_edges(P, device='cpu', max_pairs_per_row=None):
    """
    Build the list of Projection coobserved edges.
    The function returns a list of edges where each pair corresponds to pixels being seen simultaneously
    by multiple peaks of the projection operator P.
    """

    idx = P.matrix.data.index
    src, dst = [], []
    rng = np.random.default_rng()
    for row in idx:
        row = row[row >= 0]
        if row.size < 2:
            continue
        if max_pairs_per_row is not None and row.size*(row.size-1)//2 > max_pairs_per_row:
            all_pairs = list(combinations(row, 2))
            choose = rng.choice(len(all_pairs), size=max_pairs_per_row, replace=False)
            a, b = zip(*[all_pairs[k] for k in choose])
        else:
            a, b = zip(*combinations(row, 2))
        a = np.asarray(a); b = np.asarray(b)
        m = a > b
        a[m], b[m] = b[m], a[m]
        src.append(a); dst.append(b)
    if not src:
        return torch.empty(2, 0, dtype=torch.long, device=device)
    src = np.concatenate(src); dst = np.concatenate(dst)
    order = np.lexsort((dst, src))
    src, dst = src[order], dst[order]
    keep = np.ones_like(src, dtype=bool)
    keep[1:] = (src[1:] != src[:-1]) | (dst[1:] != dst[:-1])
    edge = np.stack([src[keep], dst[keep]])
    return torch.as_tensor(edge, dtype=torch.long, device=device)
