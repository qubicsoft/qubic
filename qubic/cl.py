from __future__ import division

import numpy as np
try:
    import matplotlib.pyplot as mp
except:
    pass
import os

__all__ = ['read_spectra', 'plot_spectra']


def read_spectra():
    """
    Read a file containing Cl spectra for r=0.1.

    Returns
        C_TT, C_EE, C_BB, C_TE : 4-tuple of ndarrays
        The Cl spectra for the T-T, E-E, T-E and T-E correlations.

    """
    filename = os.path.join(os.path.dirname(__file__),
                            'data', 'cl_r=0.1bis2.txt')
    # in this file, the spectra start from l=2
    ell, b, c, d, e, f, g, h, i, j, k = np.loadtxt(filename, unpack=True)
    ctt = np.r_[0, 0, b * 2e12 * np.pi / (ell * (ell + 1))]
    cee = np.r_[0, 0, c * 2e12 * np.pi / (ell * (ell + 1))]
    cte = np.r_[0, 0, e * 2e12 * np.pi / (ell * (ell + 1))]
    cbb = np.r_[0, 0, h * 2e12 * np.pi / (ell * (ell + 1))]
    return ctt, cee, cbb, cte


def plot_spectra(spectra, label_bb='$C_\ell^{BB}$'):
    """
    Plot C_TT, C_EE, C_BB and C_TE spectra.

    Parameters
    ----------
    spectra : 4-tuple
        The C_TT, C_EE, C_BB and C_TE spectra (starting from l=0).

    """
    ctt, cee, cbb, cte = spectra
    ell = np.arange(ctt.size)
    norm = (ell * (ell + 1)) / (2 * np.pi)
    mp.semilogy(ell, np.sqrt(norm * ctt), 'k', label='$C_\ell^{TT}$')
    mp.plot(ell, np.sqrt(norm * cte), 'g', label='$C_\ell^{TE}$')
    mp.plot(ell, np.sqrt(-norm * cte), 'g--',)
    mp.plot(ell, np.sqrt(norm * cee), 'b', label='$C_\ell^{EE}$')
    mp.plot(ell, np.sqrt(norm * cbb), color='brown', label=label_bb)
    mp.xlim(0, 600)
    mp.ylim(0.01, 100)
    mp.xlabel('$\ell$')
    mp.ylabel('$\sqrt{\ell(\ell+1)|C_\ell|/2\pi}$    $[\mu K]$')
    mp.legend(loc='best', frameon=False)
