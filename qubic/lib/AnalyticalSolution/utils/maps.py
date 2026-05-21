import numpy as np
import healpy as hp
import torch

def mask_map_unseen(sky_map, unseen_ring):
    """ Mask a full map with hp.UNSEEN in unseen pixels. By default, hp works with ring ordering. """
    sky_map[unseen_ring] = hp.UNSEEN
    return sky_map

def expand_ring_masked_map(sky_map_reconstructed, seen_indexes_ring, nside, value = hp.UNSEEN):
    """ Expand a masked map in ring ordering to full map with hp.UNSEEN in unseen pixels. """
    m = np.full(12*nside**2, value, dtype=float)
    m[seen_indexes_ring] = sky_map_reconstructed
    return m

def expand_nest_masked_map(sky_map_reconstructed, seen_indexes_nest, nside, value = hp.UNSEEN):
    """ Expand a masked map in nest ordering to full map with hp.UNSEEN in unseen pixels. """
    m = np.full(12*nside**2, value, dtype=float)
    m[seen_indexes_nest] = sky_map_reconstructed
    return m

def nest2ring_masked_map(sky_map_reconstructed, seen_indexes_nest, nside, value = hp.UNSEEN):
    """ n2r reordering of a masked map, with unseen pixels filled with zeros. """
    m_nest = np.full(12*nside**2, value, dtype=float)
    m_nest[seen_indexes_nest] = sky_map_reconstructed
    return hp.reorder(m_nest, n2r=True)

def ring2nest_masked_map(sky_map_reconstructed, seen_indexes_ring, nside, value = hp.UNSEEN):
    """ r2n reordering of a masked map, with unseen pixels filled with zeros. """
    m_ring = np.full(12*nside**2, value, dtype=float)
    m_ring[seen_indexes_ring] = sky_map_reconstructed
    return hp.reorder(m_ring, r2n=True)

def full_nest_to_ring(local, npix, seen_nest, seen_ring):
    """ Convert a local nest-ordered masked map to a full ring-ordered map from a torch tensor. """
    full = torch.zeros((npix, 3), dtype=local.dtype, device=local.device)
    full.index_copy_(0, seen_nest, local)
    full.index_copy_(0, seen_ring, local)
    return full

def local_to_full_ring(local, npix, seen_indexes_ring):
    """ Convert a local nest-ordered masked map to a full ring-ordered map from a torch tensor. """
    full_ring = torch.zeros((npix, 3),dtype=local.dtype,device=local.device)
    seen_ring = torch.as_tensor(seen_indexes_ring, dtype=torch.long, device=local.device)
    full_ring.index_copy_(0, seen_ring, local)
    return full_ring

def scale_map(map, S_I = 1e17, S_QU = 1e18):
    return np.stack([map[:,0]/S_I, map[:,1]/S_QU, map[:,2]/S_QU ], axis=1)

def get_covcut_area_info(cov_map, thresholds=(0.15, 0.02)):
    cov = np.asarray(cov_map)
    unseen = cov == hp.UNSEEN
    valid = ~unseen

    vmax = np.nanmax(cov[valid]) if valid.any() else 0.0
    if vmax <= 0:
        return {}

    npix = len(cov)
    nside = hp.npix2nside(npix)
    pix_area_sr = hp.nside2pixarea(nside)
    pix_area_deg2 = hp.nside2pixarea(nside, degrees=True)

    out = {}

    for thr in thresholds:
        mask = valid & (cov >= thr * vmax)
        pix = np.flatnonzero(mask)

        area_sr = len(pix) * pix_area_sr
        area_deg2 = len(pix) * pix_area_deg2
        sky_fraction = len(pix) / npix

        out[thr] = {
            "npix": len(pix),
            "area_sr": area_sr,
            "area_deg2": area_deg2,
            "sky_fraction": sky_fraction,
            "indexes": pix,
        }

    # enlargement from first threshold to second
    t1, t2 = thresholds
    out["area_ratio"] = out[t2]["area_deg2"] / out[t1]["area_deg2"]
    out["extra_pixels"] = np.setdiff1d(out[t2]["indexes"], out[t1]["indexes"])

    return out
