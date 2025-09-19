import numpy as np
import healpy as hp
import torch

def mask_map_unseen(sky_map, unseen):
    """ Mask a full map with hp.UNSEEN in unseen pixels. By default, hp works with ring ordering. """
    sky_map[unseen] = hp.UNSEEN
    return sky_map

def expand_ring_masked_map(sky_map_reconstructed, seen_indexes_ring, nside):
    """ Expand a masked map in ring ordering to full map with hp.UNSEEN in unseen pixels. """
    m = np.full(12*nside**2, hp.UNSEEN, dtype=float)
    m[seen_indexes_ring] = sky_map_reconstructed
    return m

def expand_nest_masked_map(sky_map_reconstructed, seen_indexes_nest, nside):
    """ Expand a masked map in nest ordering to full map with hp.UNSEEN in unseen pixels. """
    m = np.full(12*nside**2, hp.UNSEEN, dtype=float)
    m[seen_indexes_nest] = sky_map_reconstructed
    return m

def nest2ring_masked_map(sky_map_reconstructed, seen_indexes_nest, nside):
    """ n2r reordering of a masked map, with unseen pixels filled with zeros. """
    m_nest = np.zeros(12*nside**2, dtype=float)
    m_nest[seen_indexes_nest] = sky_map_reconstructed
    return hp.reorder(m_nest, n2r=True)

def ring2nest_masked_map(sky_map_reconstructed, seen_indexes_ring, nside):
    """ r2n reordering of a masked map, with unseen pixels filled with zeros. """
    m_ring = np.zeros(12*nside**2, dtype=float)
    m_ring[seen_indexes_ring] = sky_map_reconstructed
    return hp.reorder(m_ring, r2n=True)

def full_nest_to_ring(local, npix, seen_nest, seen_ring):
    """ Convert a local nest-ordered masked map to a full ring-ordered map from a torch tensor. """
    full = torch.zeros((npix, 3), dtype=local.dtype, device=local.device)
    full.index_copy_(0, seen_nest, local)
    full.index_copy_(0, seen_ring, local)
    return full
