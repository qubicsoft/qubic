from __future__ import division

import healpy as hp
import numpy as np
import tempfile
from numpy.testing import assert_equal, assert_raises
from pyoperators.utils.testing import assert_is_none, assert_same
from qubic.io import read_map, write_map

dtypes = (np.uint8, np.int16, np.int32, np.int64, np.float32, np.float64)
nside = 16
npix = 12 * nside**2


def test_write_read():
    complete = np.arange(npix, dtype=float)
    complete[npix // 10:] = np.nan
    mask = np.isfinite(complete)
    partial = complete[mask]

    def func(map, msk, partial, compress):
        with tempfile.TemporaryFile('a+b') as tmpfile:
            write_map(tmpfile, map, msk, compress=compress)
            tmpfile.seek(0)
            actual = read_map(tmpfile, partial=partial)
        if partial:
            out, mask_ = actual
            if msk is None:
                assert_is_none(mask_)
                actual = out
            else:
                actual = np.full(mask_.shape, np.nan)
                actual[mask_] = out
        assert_equal(actual.dtype.byteorder, '=')
        assert_same(actual, complete)
    for map, msk in (complete, None), (partial, mask):
        for partial in False, True:
            for compress in False, True:
                yield func, map, msk, partial, compress


def test_write_read_field():
    complete = np.arange(3*npix, dtype=float).reshape(npix, 3)
    mask = np.zeros(npix, bool)
    mask[:npix // 10] = True
    complete[~mask] = np.nan
    partial = complete[mask]
    fields = 0, (2,), (0, 1), (1, 2), (0, 1, 2), (0, 1, 0)

    def func(dtype, field, map, msk):
        with tempfile.TemporaryFile('a+b') as tmpfile:
            write_map(tmpfile, map.astype(dtype), msk, dtype=None)
            tmpfile.seek(0)
            actual = read_map(tmpfile, field=field, dtype=None)
        assert actual.dtype == dtype
        if np.isscalar(field):
            assert_equal(actual, complete[..., field].astype(dtype))
        elif len(field) == 1:
            assert_equal(actual, complete[..., field[0]].astype(dtype))
        else:
            for i, f in enumerate(field):
                assert_equal(actual[..., i], complete[..., f].astype(dtype))
    for map, msk in (complete, None), (partial, mask):
        for dtype in dtypes:
            for field in fields:
                yield func, dtype, field, map, msk


def test_read_header():
    with tempfile.TemporaryFile('a+b') as tmpfile:
        write_map(tmpfile, np.arange(12), coord='E')
        tmpfile.seek(0)
        actual, header = read_map(tmpfile, h=True)
    assert_equal(header['coordsys'], 'E')
    assert_equal(header['nmaps'], 1)
    assert not header['hasmask']


def test_read_healpy():
    def func(h, partial):
        with tempfile.NamedTemporaryFile('a+b') as tmpfile:
            hp.write_map(tmpfile.name, (np.arange(12), np.arange(12) + 1),
                         coord='E')
            actual = read_map(tmpfile.name, h=h, partial=partial)
            expected = hp.read_map(tmpfile.name, h=h, field=(0, 1))
        if partial:
            assert_is_none(actual[1])
            if h:
                actual = actual[0], actual[2]
            else:
                actual = actual[0]
        if h:
            assert actual[1].items() == expected[-1]
            assert_equal(actual[0], np.column_stack(expected[:-1]))
        else:
            assert_equal(actual, np.column_stack(expected))
    for h in False, True:
        for partial in False, True:
            yield func, h, partial


def test_read_nest():
    def func(nwrite, nread):
        with tempfile.TemporaryFile('a+b') as tmpfile:
            write_map(tmpfile, np.arange(12*16), nest=nwrite)
            tmpfile.seek(0)
            actual = read_map(tmpfile, nest=nread)
        with tempfile.NamedTemporaryFile('a+b') as tmpfile:
            hp.write_map(tmpfile.name, np.arange(12*16), nest=nwrite)
            expected = hp.read_map(tmpfile.name, nest=nread)
        assert_equal(expected, actual)
    for nwrite in False, True:
        for nread in False, True:
            yield func, nwrite, nread


def test_write_errors():
    assert_raises(ValueError, write_map, 'test.fits', np.array(3))
    assert_raises(ValueError, write_map, 'test.fits', np.zeros((1, 1, 1)))
    assert_raises(ValueError, write_map, 'test.fits', np.zeros(12), 3)
    assert_raises(ValueError, write_map, 'test.fits', np.zeros(13))
