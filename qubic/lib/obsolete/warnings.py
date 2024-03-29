from __future__ import absolute_import, division, print_function
import warnings
from warnings import warn


class QubicWarning(UserWarning):
    pass


class QubicDeprecationWarning(DeprecationWarning):
    pass

warnings.simplefilter('always', category=QubicWarning)
warnings.simplefilter('module', category=QubicDeprecationWarning)
del warnings
