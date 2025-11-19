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

"""Parametric spectral energy distribution (SED)

Unified API for evaluating SEDs, see :class:`Component`.

This module also provides a handy way of generating a :class:`Component` from
analytic expressions, see the :class:`AnalyticComponent`. For components
frequently used (e.g. power law, gray body, CMB) these are already
prepared.
"""

from fgbuster.component_model import (
    H_OVER_K,
    K_RJ2K_CMB,
    K_RJ2K_CMB_NU0,
    AnalyticComponent,
)


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


class invModifiedBlackBody(AnalyticComponent):
    """Inverse Modified Black body

    Parameters
    ----------
    nu0: float
        Reference frequency
    temp: float
        Black body temperature
    beta_d: float
        Spectral index
    units:
        Output units (K_CMB and K_RJ available)
    """

    _REF_BETA = 1.54
    _REF_TEMP = 20.0

    def __init__(self, nu0, temp=None, beta_d=None, units="K_CMB"):
        # Prepare the analytic expression

        analytic_expr = "(exp(nu / temp * h_over_k) -1) / (exp(nu0 / temp * h_over_k) - 1) * (nu0 / nu)**(1 + beta_d)"
        if "K_CMB" in units:
            analytic_expr += " / " + K_RJ2K_CMB_NU0
        elif "K_RJ" in units:
            pass
        else:
            raise ValueError("Unsupported units: %s" % units)

        # Parameters in the analytic expression are
        # - Fixed parameters -> into kwargs
        # - Free parameters -> renamed according to the param_* convention
        kwargs = {"nu0": nu0, "beta_d": beta_d, "temp": temp, "h_over_k": H_OVER_K}

        super().__init__(analytic_expr, **kwargs)

        self._set_default_of_free_symbols(beta_d=self._REF_BETA, temp=self._REF_TEMP)


class invPowerLaw(AnalyticComponent):
    """Inverse Power law

    Parameters
    ----------
    nu0: float
        Reference frequency
    beta_pl: float
        Spectral index
    nu_pivot: float
        Pivot frequency for the running
    running: float
        Curvature of the power law
    units:
        Output units (K_CMB and K_RJ available)
    """

    _REF_BETA = -3
    _REF_RUN = 0.0
    _REF_NU_PIVOT = 70.0

    def __init__(self, nu0, beta_pl=None, nu_pivot=None, running=0.0, units="K_CMB"):
        if nu_pivot == running is None:
            print("Warning: are you sure you want both nu_pivot and the runningto be free parameters?")

        # Prepare the analytic expression
        analytic_expr = "(nu0 / nu)**(beta_pl + running * log(nu / nu_pivot))"
        if "K_CMB" in units:
            analytic_expr += " / " + K_RJ2K_CMB_NU0
        elif "K_RJ" in units:
            pass
        else:
            raise ValueError("Unsupported units: %s" % units)

        kwargs = {"nu0": nu0, "nu_pivot": nu_pivot, "beta_pl": beta_pl, "running": running}

        super().__init__(analytic_expr, **kwargs)

        self._set_default_of_free_symbols(beta_pl=self._REF_BETA, running=self._REF_RUN, nu_pivot=self._REF_NU_PIVOT)
