'''
code inspired by FGBuster
written by Mathias Regnier
'''

from fgbuster.component_model import AnalyticComponent

class ModifiedBlackBodyDecorrelated(AnalyticComponent):
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

    def __init__(self, nu0, lcorr, beta_d=None, temp=20, units='K_CMB'):
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

class COLine(AnalyticComponent):
    """ Cosmic microwave background

    Parameters
    ----------
    units:
        Output units (K_CMB and K_RJ available)
    """

    active = False

    def __init__(self, nu, active, units='K_CMB'):
        # Prepare the analytic expression
        self.nu = nu
        if active :
            analytic_expr = ('1')
        else:
            analytic_expr = ('0')
        if units == 'K_CMB':
            pass
        elif units == 'K_RJ':
            analytic_expr += ' / ' + K_RJ2K_CMB
        else:
            raise ValueError("Unsupported units: %s"%units)
        
        kwargs = {'active': active}

        super(COLine, self).__init__(analytic_expr, **kwargs)

        #self._set_default_of_free_symbols()
