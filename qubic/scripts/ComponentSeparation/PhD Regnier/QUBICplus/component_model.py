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

import os.path as op
import numpy as np
import sympy
from sympy.parsing.sympy_parser import parse_expr
import scipy
from scipy import constants
from sympy import *
from astropy.cosmology import Planck15
import sympy

__all__ = [
    'Component',
    'AnalyticComponent',
    'CMB',
    'ThermalSZ',
    'Dust',
    'Synchrotron',
    'ModifiedBlackBody',
    'PowerLaw',
    'FreeFree',
]


lambdify = lambda x, y: sympy.lambdify(x, y, 'numpy')

H_OVER_K = constants.h * 1e9 / constants.k

# Conversion factor at frequency nu
K_RJ2K_CMB = ('(expm1(h_over_k * nu / Tcmb)**2'
              '/ (exp(h_over_k * nu / Tcmb) * (h_over_k * nu / Tcmb)**2))')
K_RJ2K_CMB = K_RJ2K_CMB.replace('Tcmb', str(Planck15.Tcmb(0).value))
K_RJ2K_CMB = K_RJ2K_CMB.replace('h_over_k', str(H_OVER_K))

# Conversion factor at frequency nu divided by the one at frequency nu0
K_RJ2K_CMB_NU0 = K_RJ2K_CMB + ' / ' + K_RJ2K_CMB.replace('nu', 'nu0')


def bandpass_integration(f):
    ''' Decorator for bandpass integration

    Parameters
    ----------
    f: callable
        Function to evaluate an SED. Its first argument must be a frequency
        array. The other positional or keyword arguments are arbitrary.

    Returns
    -------
    f: callable
        The function now accepts as the first argument

        * array with the frequencies, as before (delta bandpasses)
        * the list or tuple with the bandpasses. Each entry is a pair of arrays
          (frequencies, transmittance). The SED is evaluated at these frequencies
          multiplied by the transmittance and integrated with the trapezoid rule.

        Note that the routine does not perform anything more that this. In
        particular it does NOT:

        * normalize the transmittance to 1 or any other value
        * perform any unit conversion before integrating the SED

        Make sure you normalize and "convert the units" of the
        transmittance in such a way that you get the correct result.
    '''
    def integrated_f(nu, *params, **kwargs):
        # It is user's responsibility to provide weights in the same units
        # as the components
        if isinstance(nu, (list, tuple)):
            out_shape = f(np.array(100.), *params, **kwargs).shape[:-1]
            res = np.empty(out_shape + (len(nu),))
            for i, (band_nu, band_w) in enumerate(nu):
                res[..., i] = np.trapz(
                    f(band_nu, *params, **kwargs) * band_w,
                    band_nu * 1e9)
            return res
        return f(nu, *params)

    return integrated_f


class Component(object):
    """ Abstract class for SED evaluation

    It defines the API.
    """

    def _add_last_dimension_if_ndarray(self, param):
        try:
            # Lambdified expressions always output an ndarray with shape
            # (param_dim_1, ..., param_dim_n, n_freq). However, parameters and
            # frequencies are both symbols with (no special meaning). In order
            # to impose the shape of the output, we append a dimension to the
            # parameters and let the broadcasting inside the lambdified
            # expressions do the rest
            return param[..., np.newaxis]
        except TypeError:
            # param is an scalar value, no special treatment is required
            return param

    def eval(self, nu, *params):
        """ Evaluate the SED

        Parameters
        ----------
        nu: array, tuple or list
            Frequencies or banpasses for the SED evaluation
            See the result of :func:`bandpass_integration`.

        *params: float or ndarray
            Value of each of the free parameters. They can be arrays and, in
            this case, they should be broadcastable to a common shape.

        Returns
        -------
        result: ndarray
            SED. The shape is ``np.broadcast(*params).shape + nu.shape``
            (or broadcastable to it).
            In particular, if the parameters are all floats, the shape is
            `nu.shape`.

        """
        assert len(params) == self.n_param
        if params and np.broadcast(*params).ndim == 0:
            # Parameters are all scalars.
            # This case is frequent and easy, thus leave early
            return self._lambda(nu, *params)

        # Make sure that broadcasting rules will apply correctly when passing
        # the parameters to the lambdified functions:
        # last axis has to be nu, but that axis is missing in the parameters
        new_params = [self._add_last_dimension_if_ndarray(p) for p in params]
        return self._lambda(nu, *new_params)

    def diff(self, nu, *params):
        """ Evaluate the derivative of the SED

        Parameters
        ----------
        nu: array
            Frequencies or banpasses for the SED evaluation
            See the result of :func:`bandpass_integration`.
        *params: float or ndarray
            Value of the free parameters. They can be arrays and, in this case,
            they should be broadcastable to a common shape.

        Returns
        -------
        result: list
            It contains the derivative with respect to each parameter. See
            :meth:`eval` for more details about the format of the
            evaluated derivative
        """
        assert len(params) == self.n_param
        if not params:
            return []
        elif np.broadcast(*params).ndim == 0:
            # Parameters are all scalars.
            # This case is frequent and easy, thus leave early
            return [self._lambda_diff[i_p](nu, *params)
                    for i_p in range(self.n_param)]

        # Make sure that broadcasting rules will apply correctly when passing
        # the parameters to the lambdified functions:
        # last axis has to be nu, but that axis is missing in the parameters
        new_params = [self._add_last_dimension_if_ndarray(p) for p in params]

        res = []
        for i_p in range(self.n_param):
            res.append(self._lambda_diff[i_p](nu, *new_params))
        return res

    def diff_diff(self, nu, *params):
        assert len(params) == self.n_param
        if not params:
            return [[]]
        elif np.broadcast(*params).ndim == 0:
            # Parameters are all scalars.
            # This case is frequent and easy, thus leave early
            return [[self._lambda_diff_diff[i_p][j_p](nu, *params)
                     for i_p in range(self.n_param)]
                    for j_p in range(self.n_param)]

        # Make sure that broadcasting rules will apply correctly when passing
        # the parameters to the lambdified functions:
        # last axis has to be nu, but that axis is missing in the parameters
        new_params = [self._add_last_dimension_if_ndarray(p) for p in params]

        res = []
        for i_p in range(self.n_param):
            res.append([self._lambda_diff[i_p][j_p](nu, *new_params)
                        for j_p in range(self.n_param)])
        return res

    @property
    def params(self):
        """ Name of the free parameters
        """
        return self._params

    @property
    def n_param(self):
        """ Number of free parameters
        """
        return len(self._params)

    def _set_default_of_free_symbols(self, **kwargs):
        # Note that
        # - kwargs can contain also keys that are not free symbols
        # - only values of the free symbols are considered
        # - these values are stored in the right order
        self.defaults = [kwargs[symbol] for symbol in self.params]


    @property
    def defaults(self):
        """ Default values of the free parameters
        """
        try:
            assert len(self._defaults) == self.n_param
        except (AttributeError, AssertionError):
            print("Component: unexpected number of or uninitialized defaults, "
                  "returning ones")
            return [1.] * self.n_param
        return self._defaults

    @defaults.setter
    def defaults(self, new_defaults):
        assert len(new_defaults) == self.n_param, ("The length of the defaults"
                                                   "should be %i"%self.n_param)
        self._defaults = new_defaults

    def __getattr__(self, attr):
        # Helpful messages when virtual attribute are not defined
        message = ("Attempt to either use a bare 'Component' object or to"
                   "use an incomplete child class.")
        if attr == '_lambda':
            message += (" Child classes should store in '_lambda'"
                        "the bare SED evaluator or, alternatively, override"
                        "'Component.eval'")
        elif attr == '_lambda_diff':
            message += (" Child classes should store in '_lambda_diff'"
                        "the list of bare evaluators of the derivative of"
                        "the SED for each parameter or, alternatively,"
                        "override 'Component.diff'")
        elif attr == '_lambda_diff_diff':
            message += (" Child classes should store in '_lambda_diff_diff'"
                        "the list of lists of the bare evaluators of the "
                        "second derivatives of the the SED for each"
                        "combination of parameters or, alternatively,"
                        "override 'Component.diff_diff'")
        elif attr == '_params':
            message += (" Child classes should store in '_params'"
                        "the list of the free parameters")
        else:
            raise AttributeError("'%s' object has no attribute '%s'"
                                 % (type(self).__name__, attr))
        raise NotImplementedError(message)


class AnalyticComponent(Component):
    """ Component defined analytically

    Class that allows analytic definition and automatic (symbolic)
    differentiation of it using `sympy`_.


    Parameters
    ----------
    analytic_expr: str
        Analytic expression for the SED. The variable representing the
        frequency is assumed to be ``nu``. You can not use names that produce
        clashes with `sympy`_ definitions (e.g, `functions`_).
        Notable forbidden names are *beta*, *gamma*.
    **fixed_params: float
        Fix the value of the desired variables. If a variable is not specified
        or is set equal to ``None``, it will be a free parameters.

    Note
    ----
    Difference with respect to a `sympy.Expression`

    * Efficient evaluators of the SED and its derivatives are prepared at
      construction time
    * Following the API specified in :class:`Component`, ``nu`` has a special
      meaning and has a dedicated dimension (the last one) when evaluations are
      performed
    * ``diff`` (and ``diff_diff``) return the evaluation of the derivatives with
      respect to all the free parameters, not the expression of the
      derivatives with respect to a specific parameter

    Note also that

    * You can trade a longer construction time for faster evaluation time by
      setting ``component_model.lambdify`` to
      ``sympy.utilities.autowrap.ufuncify``.
      After constructing the anlytic component you can revert back the change by
      setting ``component_model.lambdify`` back to ``sympy.lambdify``.
      The gain can negligible or considerable depending on the analytic
      expression.

    .. _functions: https://docs.sympy.org/latest/modules/functions/index.html
    .. _sympy: https://docs.sympy.org/latest/modules/functions/index.html
    """

    def __init__(self, analytic_expr, **fixed_params):
        self._fixed_params = fixed_params
        self._expr = parse_expr(analytic_expr).subs(fixed_params)
        self._params = sorted([str(s) for s in self._expr.free_symbols])
        self._defaults = []

        # NOTE: nu is in symbols (at index 0) but it is not in self._params
        if 'nu' in self._params:
            self._params.pop(self._params.index('nu'))
        self._params.insert(0, 'nu')
        symbols = sympy.symbols(self._params)
        self._params.pop(0)

        # Create lambda functions

        _lambdify = lambda *args, **kwargs: bandpass_integration(
            lambdify(*args, **kwargs))
        self._lambda = _lambdify(symbols, self._expr)
        lambdify_diff_param = lambda param: _lambdify(
            symbols, self._expr.diff(param))
        self._lambda_diff = [lambdify_diff_param(p) for p in self._params]
        lambdify_diff_diff_params = lambda param1, param2: _lambdify(
            symbols, self._expr.diff(param1, param2))
        self._lambda_diff_diff = []
        for p1 in self._params:
            self._lambda_diff_diff.append(
                [lambdify_diff_diff_params(p1, p2) for p2 in self._params])

    def __repr__(self):
        return repr(self._expr)


class ModifiedBlackBody(AnalyticComponent):
    """ Modified Black body

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
    _REF_TEMP = 20.

    def __init__(self, nu0, temp=None, beta_d=None, units='K_CMB'):
        # Prepare the analytic expression
        # Note: beta_d (not beta) avoids collision with sympy beta functions
        #TODO: Use expm1 and get Sympy processing it as a symbol
        analytic_expr = ('(exp(nu0 / temp * h_over_k) -1)'
                         '/ (exp(nu / temp * h_over_k) - 1)'
                         '* (nu / nu0)**(1 + beta_d)')
        if 'K_CMB' in units:
            analytic_expr += ' * ' + K_RJ2K_CMB_NU0
        elif 'K_RJ' in units:
            pass
        else:
            raise ValueError("Unsupported units: %s"%units)

        # Parameters in the analytic expression are
        # - Fixed parameters -> into kwargs
        # - Free parameters -> renamed according to the param_* convention
        kwargs = {
            'nu0': nu0, 'beta_d': beta_d, 'temp': temp, 'h_over_k': H_OVER_K
        }

        super(ModifiedBlackBody, self).__init__(analytic_expr, **kwargs)

        self._set_default_of_free_symbols(
            beta_d=self._REF_BETA, temp=self._REF_TEMP)


class PowerLaw(AnalyticComponent):
    """ Power law

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
    _REF_RUN = 0.
    _REF_NU_PIVOT = 70.

    def __init__(self, nu0, beta_pl=None, nu_pivot=None, running=0.,
                 units='K_CMB'):
        if nu_pivot == running == None:
            print('Warning: are you sure you want both nu_pivot and the running'
                  'to be free parameters?')

        # Prepare the analytic expression
        analytic_expr = '(nu / nu0)**(beta_pl + running * log(nu / nu_pivot))'
        if 'K_CMB' in units:
            analytic_expr += ' * ' + K_RJ2K_CMB_NU0
        elif 'K_RJ' in units:
            pass
        else:
            raise ValueError("Unsupported units: %s"%units)

        kwargs = {'nu0': nu0, 'nu_pivot': nu_pivot,
                  'beta_pl': beta_pl, 'running': running}

        super(PowerLaw, self).__init__(analytic_expr, **kwargs)

        self._set_default_of_free_symbols(
            beta_pl=self._REF_BETA, running=self._REF_RUN, nu_pivot=self._REF_NU_PIVOT)


class CMB(AnalyticComponent):
    """ Cosmic microwave background

    Parameters
    ----------
    units:
        Output units (K_CMB and K_RJ available)
    """

    def __init__(self, units='K_CMB'):
        # Prepare the analytic expression
        analytic_expr = ('1')
        if units == 'K_CMB':
            pass
        elif units == 'K_RJ':
            analytic_expr += ' / ' + K_RJ2K_CMB
        else:
            raise ValueError("Unsupported units: %s"%units)

        super(CMB, self).__init__(analytic_expr)

        if 'K_CMB' in units:
            self.eval = bandpass_integration(lambda nu: np.ones_like(nu))


class ThermalSZ(AnalyticComponent):
    """ Thermal Sunyaev-Zeldovich

    Parameters
    ----------
    units:
        Output units (K_CMB and K_RJ available)
    """

    def __init__(self, units='uK_CMB'):
        # Prepare the analytic expression
        x_nu = '(nu * h_over_k / Tcmb)'
        analytic_expr = 'Tcmb * (x_nu * (exp(x_nu) + 1) / expm1(x_nu) - 4)'
        analytic_expr = analytic_expr.replace('x_nu', x_nu)
        if 'K_CMB' in units:
            pass
        elif 'K_RJ' in units:
            analytic_expr += ' / ' + K_RJ2K_CMB
        else:
            raise ValueError("Unsupported units: %s"%units)
        if units[0] == 'u':
            analytic_expr = '1e6 * ' + analytic_expr
        elif units[0] == 'm':
            analytic_expr = '1e3 * ' + analytic_expr


        kwargs = dict(Tcmb=Planck15.Tcmb(0).value, h_over_k=H_OVER_K)
        super(ThermalSZ, self).__init__(analytic_expr, **kwargs)


class FreeFree(AnalyticComponent):
    """ Free-free

    Anlytic model for bremsstrahlung emission (Draine, 2011)
    Above 1GHz it is essentially equivalent to a power law.

    Parameters
    ----------
    logEM:
        Logarithm (base ten) of the integrated squared electron density along a
        line of sight in cm^-3 pc
    Te:
        Electron temperature
    units:
        Output units (K_CMB and K_RJ available)
    """

    _REF_LOGEM = 0.
    _REF_TE = 7000.

    def __init__(self, logEM=None, Te=None, units='K_CMB'):
        # Prepare the analytic expression. Planck15 X, Table 4
        # NOTE: PySM uses power a power law instead
        T4 = 'Te * 1e-4'
        gff = 'log(exp(5.960 - (sqrt(3) / pi) * log(nu * (T4)**(-3 / 2))) + exp(1))'
        tau = '0.05468 * Te**(- 3 / 2) / nu**2 * 10**(EM) * (gff)'
        analytic_expr = '1e6 * Te * (1 - exp(-(tau)))'
        analytic_expr = analytic_expr.replace('tau', tau)
        analytic_expr = analytic_expr.replace('gff', gff)
        analytic_expr = analytic_expr.replace('T4', T4)
        if 'K_CMB' in units:
            analytic_expr += ' * ' + K_RJ2K_CMB
        elif 'K_RJ' in units:
            pass
        else:
            raise ValueError("Unsupported units: %s"%units)

        kwargs = dict(logEM=logEM, Te=Te, tau=tau, gff=gff, T4=T4)

        super(FreeFree, self).__init__(analytic_expr, **kwargs)

        self._set_default_of_free_symbols(
            logEM=self._REF_LOGEM, Te=self._REF_TE)


class Dust(ModifiedBlackBody):
    """ Alias of :class:`ModifiedBlackBody`
    """
    pass


class Synchrotron(PowerLaw):
    """ Alias of :class:`PowerLaw`
    """
    pass

class ModifiedBlackBody_2b(AnalyticComponent):
    """ Modified Black body

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
    _REF_BETA_0 = 1.54
    _REF_BETA_1 = 1.54
    _REF_TEMP = 20.
    _REF_NUBREAK = 200.

    def __init__(self, nu0, nubreak=None, temp=None, beta_d0=None, beta_d1=None, units='K_CMB'):
        # Prepare the analytic expression
        # Note: beta_d (not beta) avoids collision with sympy beta functions
        #TODO: Use expm1 and get Sympy processing it as a symbol
        def double_beta_dust_FGB_Model():

            H_OVER_K = constants.h * 1e9 / constants.k
            # Conversion factor at frequency nu
            K_RJ2K_CMB = ('(expm1(h_over_k * nu / Tcmb)**2'
                          '/ (exp(h_over_k * nu / Tcmb) * (h_over_k * nu / Tcmb)**2))')
            K_RJ2K_CMB = K_RJ2K_CMB.replace('Tcmb', str(Planck15.Tcmb(0).value))
            K_RJ2K_CMB = K_RJ2K_CMB.replace('h_over_k', str(H_OVER_K))
            K_RJ2K_CMB_NU0 = K_RJ2K_CMB + ' / ' + K_RJ2K_CMB.replace('nu', 'nu0')

            analytic_expr1 = ('(exp(nu0 / temp * h_over_k) - 1)'
                             '/ (exp(nu / temp * h_over_k) - 1)'
                             '* (nu / nu0)**(1 + beta_d0) * (nu0 / nubreak)**(beta_d0-beta_d1) * (1-(0.5 + 0.5*tanh((nu-nubreak)*500)))')#' * (1-heaviside(nu-nubreak ,0.5))')

            analytic_expr2 = ('(exp(nu0 / temp * h_over_k) - 1)'
                             '/ (exp(nu / temp * h_over_k) - 1)'
                             '* (nu / nu0)**(1 + beta_d1) * (0.5 + 0.5*tanh((nu-nubreak)*500))')#' * heaviside(nu-nubreak,0.5)')
            analytic_expr = analytic_expr1 + ' + ' + analytic_expr2

            return analytic_expr
        analytic_expr = (double_beta_dust_FGB_Model())
        if 'K_CMB' in units:
            pass
            #analytic_expr += ' * ' + K_RJ2K_CMB_NU0
        elif 'K_RJ' in units:
            pass
        else:
            raise ValueError("Unsupported units: %s"%units)

        # Parameters in the analytic expression are
        # - Fixed parameters -> into kwargs
        # - Free parameters -> renamed according to the param_* convention
        kwargs = {
            'nu0': nu0, 'nubreak': nubreak, 'beta_d0':beta_d0, 'beta_d1':beta_d1, 'temp': temp, 'h_over_k': H_OVER_K
        }
        super(ModifiedBlackBody_2b, self).__init__(analytic_expr, **kwargs)

        self._set_default_of_free_symbols(
            nubreak=self._REF_NUBREAK, beta_d0=self._REF_BETA_0, beta_d1=self._REF_BETA_1, temp=self._REF_TEMP)



class Dust_2b(ModifiedBlackBody_2b):
    """ Alias of :class:`ModifiedBlackBody_2b`
    """
    pass
