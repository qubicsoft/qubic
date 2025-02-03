# FGBuster
# Copyright (C) 2019 Davide Poletti, Josquin Errard and the FGBuster developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

""" Parametric spectral energy distribution (SED)

Unified API for evaluating SEDs, see :class:`Component`.

This module also provides a handy way of generating a :class:`Component` from
analytic expressions, see the :class:`AnalyticComponent`. For components
frequently used (e.g. power law, gray body, CMB) these are already
prepared.
"""

import numpy as np
import sympy
from astropy.cosmology import Planck15
from scipy import constants
from sympy import DiracDelta, Function, Piecewise, symbols, sympify
from sympy.parsing.sympy_parser import parse_expr

from fgbuster.component_model import *

class Monochromatic(AnalyticComponent):
    """Cosmic microwave background

    Parameters
    ----------
    units:
        Output units (K_CMB and K_RJ available)
    """

    active = False

    def __init__(self, nu0, units="K_CMB"):
        # Prepare the analytic expression
        # self.nu = nu
        analytic_expr = f"0.0000001 / (0.0000001 + (nu - {nu0})**2)"

        if units == "K_CMB":
            pass
        elif units == "K_RJ":
            analytic_expr += " / " + K_RJ2K_CMB
        else:
            raise ValueError("Unsupported units: %s" % units)

        kwargs = {}  #'active': active}

        super(Monochromatic, self).__init__(analytic_expr, **kwargs)

        # self._set_default_of_free_symbols()

