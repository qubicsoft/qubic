""" Parametric spectral energy distribution (SED)

code inspired by FGBuster
written by Mathias Regnier

Unified API for evaluating SEDs, see :class:`Component`.

This module also provides a handy way of generating a :class:`Component` from
analytic expressions, see the :class:`AnalyticComponent`. For components
frequently used (e.g. power law, gray body, CMB) these are already
prepared.
"""

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

        
