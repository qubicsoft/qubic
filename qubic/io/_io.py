from __future__ import division
from astropy.io import fits
from collections import Container
from pyoperators.utils import ndarraywrap
import healpy as hp
import numpy as np

__all__ = ['read_map', 'write_map']

_default_extnames = {
    1: ('I_STOKES',),
    3: ('I_STOKES', 'Q_STOKES', 'U_STOKES'),
    6: ('II', 'IQ', 'IU', 'QQ', 'QU', 'UU')}


def read_map(filename, field=None, dtype=float, nest=False, partial=False):
    """
    Read one or more compressed Healpix maps from a FITS file.

    Parameters
    ----------
    filename : str
        The FITS file name.
    field : int, str or tuple of int or str, optional
        The fields to be read. Default: all. By convention 0 is temperature,
        1 is Q, 2 is U. It can be a tuple to read multiple columns (0, 1, 2).
        Extension names can also be used: 'I_STOKES'
    dtype : data type, optional
        Force the conversion to some data type. Default: np.float64
    nest : bool, optional
        If True, return the map in NEST ordering, otherwise in RING ordering;
        use fits keyword ORDERING to decide whether conversion is needed or not
        If None, no conversion is performed.
    partial : bool, optional
        If True, return the partial map and the mask of valid pixels.

    Returns
    -------
    maps [, mask[, header]] : array of shape (N, M)
        The Healpix maps, where M is the number of maps and N the number
        of pixels.
    mask : bool array
        The selection mask (True means valid) if the partial keyword is
        set to True.

    """
    if isinstance(filename, file):
        modes = {'+ab': 'append'}
        mode = modes.get(''.join(sorted(filename.mode)), 'readonly')
    else:
        mode = 'readonly'
    hdus = fits.open(filename, mode=mode)
    header = hdus[0].header
    if 'format' not in header or header['format'] != 'HPX_QB':
        # fall back to healpy's read_map
        if len(hdus) != 2 or 'pixtype' not in hdus[1].header or \
           hdus[1].header['pixtype'] != 'HEALPIX':
            raise TypeError('This file cannot be read as a Healpix map.')
        header = hdus[1].header
        if field is None:
            field = range(header['TFIELDS'])
        out = hp.read_map(filename, dtype=dtype, nest=nest, field=field,
                          h=True, verbose=False)
        out, header = out[:-1], fits.header.Header(out[-1])
        out = np.column_stack(out)
        out = out.view(ndarraywrap)
        out.header = header
        if partial:
            return out, None
        return out

    nside = header['nside']
    npix = 12 * nside**2
    ordering = header['ordering']
    if field is None:
        field = range(header['nmaps'])
    elif not isinstance(field, Container) or isinstance(field, str):
        field = (field,)

    npix_ = npix
    if header['hasmask'] and header['hasmask']:
        mask = hdus[1].data.view(bool)
        mask_ = mask
        ifirst = 2
        outfunc = lambda shape, **keywords: np.full(shape, np.nan, **keywords)
        if partial:
            npix_ = np.sum(mask)
            mask_ = Ellipsis
    else:
        mask = None
        mask_ = Ellipsis
        ifirst = 1
        outfunc = np.empty
    if dtype is None:
        bitpix_table = {8: np.uint8, 16: np.int16, 32: np.int32, 64: np.int64,
                        -32: np.float32, -64: np.float64}
        dtype = bitpix_table[hdus[ifirst].header['bitpix']]
    out = outfunc((len(field), npix_), dtype=dtype, order='f')

    if partial and (nest and ordering == 'RING' or
                    not nest and ordering == 'NESTED'):
        raise NotImplementedError('The partial keyword cannot be set while cha'
                                  'nging the ordering.')
    reorder = False
    if nest and ordering == 'RING':
        indexing = hp.pixelfunc.nest2ring(nside, np.arange(npix))
        reorder = True
    elif not nest and ordering == 'NESTED':
        indexing = hp.pixelfunc.ring2nest(nside, np.arange(npix))
        reorder = True

    for i, field_ in enumerate(field):
        ihdu = field_ if isinstance(field_, str) else field_ + ifirst
        out_ = out[i]
        out_[mask_] = hdus[ihdu].data
        if reorder:
            np.take(out_, indexing, out=out_)
        try:
            out_[hp.pixelfunc.mask_bad(out_)] = hp.UNSEEN
        except OverflowError:
            pass

    if len(field) == 1:
        out = out[0]
    else:
        out = out.T
    out = out.view(ndarraywrap)
    out.header = header
    if partial:
        out = out, mask
    return out


def write_map(filename, map, mask=None, nest=False, dtype=np.float32,
              coord=None, extnames=None, compress=True):
    """
    Write one or more compressed (complete or partial) Healpix maps
    as FITS file.

    Parameters
    ----------
    filename : str
        The FITS file name.
    map : array of shape (N) or (N, M)
        The partial or complete input Healpix maps.
    mask : bool array, optional
        The mask controlling partial maps, such as complete[mask] == partial
        (True means valid).
    nest : bool, optional
        If True, ordering scheme is assumed to be NESTED, otherwise, RING.
        Default: RING. The map ordering is not modified by this function,
        the input map array should already be in the desired ordering.
    coord : str
        The coordinate system, typically 'E' for Ecliptic, 'G' for Galactic or
        'C' for Celestial (equatorial).
    extnames : str or list
        The FITS extension names, by default, we use:
            - I_STOKES for 1 component,
            - I/Q/U_STOKES for 3 components,
            - II, IQ, IU, QQ, QU, UU for 6 components,
            - DATA_0, DATA_1... otherwise
    """
    map = np.asanyarray(map, order='f') #XXX astropy issue #2150
    if map.ndim not in (1, 2):
        raise ValueError('Invalid dimensions of the healpix map(s).')
    if map.ndim == 1:
        map = map.reshape(-1, 1)
    nmaps = map.shape[1]
    if mask is not None:
        mask = np.asarray(mask, np.uint8)
        if mask.ndim != 1:
            raise ValueError('Invalid dimensions of healpix the mask.')
        npix = mask.size
    else:
        npix = map.shape[0]
    try:
        coord = map.header['coordsys'].upper()
    except (AttributeError, KeyError):
        pass
    try:
        ordering = map.header['ordering'].upper()
        if ordering not in ('NESTED', 'RING'):
            raise ValueError("Invalid ordering scheme '{}'.".format(ordering))
    except (AttributeError, KeyError):
        ordering = 'NESTED' if nest else 'RING'
    nside = hp.npix2nside(npix)

    if compress and map.dtype != int: #XXX avoid crash: astropy issue #2153
        _imageHDU = fits.CompImageHDU
    else:
        _imageHDU = fits.ImageHDU

    primary = fits.PrimaryHDU()
    primary.header['nside'] = nside, 'Resolution parameter of HEALPIX'
    primary.header['ordering'] = (ordering, 'Pixel ordering scheme, '
                                  'either RING or NESTED')
    if coord:
        primary.header['coordsys'] = (coord, 'Ecliptic, Galactic or Celestial '
                                      '(equatorial)')
    primary.header['format'] = 'HPX_QB'
    primary.header['nmaps'] = nmaps
    primary.header['hasmask'] = mask is not None
    if hasattr(map, 'header'):
        for key in sorted(set(map.header.keys()) -
                          set(('nside', 'format', 'nmaps', 'hasmask'))):
            primary.header[key] = map.header[key]
    hdus = [primary]

    if mask is not None:
        hdu = fits.CompImageHDU(mask)
        hdu.header.set('VALIDMASK')
        hdus.append(hdu)

    if extnames is None:
        extnames = _default_extnames.get(
            nmaps, ('DATA_{}'.format(i + 1) for i in range(nmaps)))

    for m, extname in zip(map.T, extnames):
        hdu = _imageHDU(np.array(m, dtype=dtype, copy=False))
        hdu.header.set(extname)
        hdus.append(hdu)

    fits.HDUList(hdus).writeto(filename, clobber=True)
