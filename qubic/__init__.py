from importlib.metadata import version as _version
from . import TES
from . import calfiles
from . import data
from . import dicts
from . import doc
from . import io
from . import level1
from . import lib


def full(shape, fill_value, dtype=None, order='C'):
    import numpy as np
    out = np.empty(shape, dtype=dtype, order=order)
    out[...] = fill_value
    return out


def full_like(a, fill_value, dtype=None, order='K', subok=True):
    import numpy as np
    out = np.empty_like(a, dtype=dtype, order=order, subok=subok)
    out[...] = fill_value
    return out

import numpy
if numpy.__version__ < '1.8':
    numpy.full = full
    numpy.full_like = full_like
del full, full_like, numpy

__version__ = "2.0.0"
