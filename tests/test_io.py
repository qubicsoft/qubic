import tempfile

import pytest
import healpy as hp
import numpy as np
from numpy.testing import assert_equal
from qubic.io import read_map, write_map

DTYPES = (np.uint8, np.int16, np.int32, np.int64, np.float32, np.float64)
NSIDE = 16
NPIX = 12 * NSIDE**2

TEST_WRITE_READ_MAP = np.arange(NPIX, dtype=float)
TEST_WRITE_READ_MAP[NPIX // 10:] = np.nan
TEST_WRITE_READ_MASK = np.isfinite(TEST_WRITE_READ_MAP)
TEST_WRITE_READ_PARTIAL = TEST_WRITE_READ_MAP[TEST_WRITE_READ_MASK]


@pytest.mark.parametrize(
    'map, mask',
    [
        (TEST_WRITE_READ_MAP, None),
        (TEST_WRITE_READ_PARTIAL, TEST_WRITE_READ_MASK),
    ]
)
@pytest.mark.parametrize('partial', [False, True])
@pytest.mark.parametrize('compress', [False, True])
def test_write_read(map, mask, partial, compress):
    with tempfile.TemporaryFile('a+b') as tmpfile:
        write_map(tmpfile, map, mask, compress=compress)
        tmpfile.seek(0)
        actual = read_map(tmpfile, partial=partial)
    if partial:
        out, mask_ = actual
        if mask is None:
            assert mask_ is None
            actual = out
        else:
            actual = np.full(mask_.shape, np.nan)
            actual[mask_] = out
    assert actual.dtype.byteorder == '='
    assert_equal(actual, TEST_WRITE_READ_MAP)


TEST_WRITE_READ_FIELD_MAP = np.arange(3*NPIX, dtype=float).reshape(NPIX, 3)
TEST_WRITE_READ_FIELD_MASK = np.zeros(NPIX, bool)
TEST_WRITE_READ_FIELD_MASK[:NPIX // 10] = True
TEST_WRITE_READ_FIELD_MAP[~TEST_WRITE_READ_FIELD_MASK] = np.nan
TEST_WRITE_READ_FIELD_PARTIAL = TEST_WRITE_READ_FIELD_MAP[TEST_WRITE_READ_FIELD_MASK]


@pytest.mark.parametrize(
    'map, mask',
    [
        (TEST_WRITE_READ_FIELD_MAP, None),
        (TEST_WRITE_READ_FIELD_PARTIAL, TEST_WRITE_READ_FIELD_MASK),
    ]
)
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('field', [0, (2,), (0, 1), (1, 2), (0, 1, 2), (0, 1, 0)])
def test_write_read_field(map, mask, dtype, field):
    def astype(map, dtype):
        if np.dtype(dtype).kind != 'f':
            map = np.nan_to_num(map)
        return map.astype(dtype)

    with tempfile.TemporaryFile('a+b') as tmpfile:
        write_map(tmpfile, astype(map, dtype), mask, dtype=None)
        tmpfile.seek(0)
        actual = read_map(tmpfile, field=field, dtype=None)

    assert actual.dtype == dtype
    expected = astype(TEST_WRITE_READ_FIELD_MAP, dtype)
    if np.isscalar(field):
        assert_equal(actual, expected[..., field].astype(dtype))
    elif len(field) == 1:
        assert_equal(actual, expected[..., field[0]].astype(dtype))
    else:
        for i, f in enumerate(field):
            assert_equal(actual[..., i], expected[..., f].astype(dtype))


def test_read_header():
    with tempfile.TemporaryFile('a+b') as tmpfile:
        write_map(tmpfile, np.arange(12), coord='E')
        tmpfile.seek(0)
        actual = read_map(tmpfile)
    header = actual.header
    assert header['coordsys'] == 'E'
    assert header['nmaps'] == 1
    assert not header['hasmask']


@pytest.mark.parametrize('h', [False, True])
@pytest.mark.parametrize('partial', [False, True])
def test_read_healpy(h, partial):
    with tempfile.NamedTemporaryFile('a+b') as tmpfile:
        hp.write_map(
            tmpfile.name,
            (np.ones(12), np.arange(12) + 1),
            coord='E',
        )
        actual = read_map(tmpfile.name, partial=partial)
        expected = hp.read_map(tmpfile.name, h=h, field=(0, 1))
    if partial:
        actual, mask = actual
        assert mask is None
    if h:
        expected, header = expected
        assert list(actual.header.items()) == header
    assert_equal(actual.T, expected)


@pytest.mark.parametrize('nwrite', [False, True])
@pytest.mark.parametrize('nread', [False, True])
def test_read_nest(nwrite, nread):
    with tempfile.TemporaryFile('a+b') as tmpfile:
        write_map(tmpfile, np.arange(12*16), nest=nwrite)
        tmpfile.seek(0)
        actual = read_map(tmpfile, nest=nread)
    with tempfile.NamedTemporaryFile('a+b') as tmpfile:
        hp.write_map(tmpfile.name, np.arange(12*16), nest=nwrite)
        expected = hp.read_map(tmpfile.name, nest=nread)
    assert_equal(expected, actual)


@pytest.mark.parametrize(
    'args',
    [
        (np.array(3),),
        (np.zeros((1, 1, 1)),),
        (np.zeros(12), 3),
        (np.zeros(13),)
    ]
)
def test_write_map_error(args):
    with pytest.raises(ValueError):
        write_map('test.fits', *args)
