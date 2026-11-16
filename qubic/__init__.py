from importlib.metadata import version as _version


import pandas as pd

def pandas_test(stage):
    try:
        pd.Index(["a", "b"])
        print(stage, "PANDAS OK")
    except Exception as e:
        print(stage, "PANDAS FAILED:", repr(e))

pandas_test("Before QUBIC imports")

from . import TES
pandas_test("After TES")

from . import calfiles
pandas_test("After calfiles")

from . import data
pandas_test("After data")

from . import dicts
pandas_test("After dicts")

from . import doc
pandas_test("After doc")

from . import io
pandas_test("After io")

from . import level1
pandas_test("After level1")

from . import lib
pandas_test("After lib")

# from . import TES
# from . import calfiles
# from . import data
# from . import dicts
# from . import doc
# from . import io
# from . import level1
# from . import lib


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
from packaging.version import Version
if Version(numpy.__version__) < Version("1.8"):
# if numpy.__version__ < '1.8': # bad comparison, as "1.26.4" < "1.8" is True
    numpy.full = full
    numpy.full_like = full_like
del full, full_like, numpy

__version__ = _version('qubic')
