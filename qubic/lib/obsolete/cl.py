from __future__ import division
from pyoperators.utils import settingerr
from .data import PATH
import numpy as np

__all__ = ['camb_spectra', 'read_spectra', 'plot_spectra', 'semilogy_spectra']


def camb_spectra(lmax, **params):
    """
    Wrapper around `pycamb.camb`. Return Cl_TT, Cl_EE, Cl_BB, Cl_EB
    non-weighted by l(l+1)/2pi, starting from l=0.

    """
    import pycamb
    T, E, B, X = pycamb.camb(lmax + 1, **params)
    ell_camb = np.arange(0, lmax + 1)
    fact_camb = 2 * np.pi / (ell_camb * (ell_camb + 1))
    T = np.r_[0, T * fact_camb[1:]]
    E = np.r_[0, E * fact_camb[1:]]
    B = np.r_[0, B * fact_camb[1:]]
    X = np.r_[0, X * fact_camb[1:]]
    return [T, E, B, X]


def read_spectra(r):
    """
    Read a camb-generated Cl file using the parameters from
    Planck 2013 results XV. CMB Power spectra, table 8, Planck+WP,
    Archiv 1303.5075:
        H0=67.04
        omegab=0.022032 / (H0 / 100)**2
        omegac=0.12038 / (H0 / 100)**2
        omegak=0
        scalar_index=0.9624
        reion__use_optical_depth=True
        reion__optical_depth=0.0925
        WantTensors=True
        scalar_amp=np.exp(3.098) * 1e-10
        DoLensing=False

    Parameters
    ----------
    r : float
        The tensor-to-scalar ratio (available values are: 0, 0.001, 0.002,
        0.005, 0.01, 0.02, 0.05, 0.1, 0.2).

    Returns
    -------
    C_TT, C_EE, C_BB, C_TE : 4-tuple of ndarrays
        The power spectra for the T-T, E-E, T-E and T-E correlations,
        not weighted by l(l+1)/2pi, starting from l=0.

    """
    filename = PATH + 'planck1303.5075_r{}.txt'.format(r)
    ell, ctt, cee, cbb, cte = np.loadtxt(filename, unpack=True)
    return ctt, cee, cbb, cte


def plot_spectra(ell, spectra=None, lmax=None, Dl=False, yerr=None, xerr=None,
                 loc='best', **keywords):
    """
    Plot Cl power spectra as l(l+1)Cl/2pi Cl.

    plot_spectra([ell,] spectra)

    Parameters
    ----------
    ell : integer array, optional
        The l values of the spectra. If not provided, it is assumed that
        the Cls start from 0 and are not binned.
    spectra : array-like, 4-tuple or 6-tuple
        The Cl spectrum or the TT, EE, BB and TE or TT, EE, BB, TE, EB, TB
        Cl spectra.
    Dl : boolean, optional
        Use `D_l` as label instead of `l(l+1)Cl/2pi`.
    lmax : integer
        Plot upper x limit.
    xerr, yerr : array-like
        If set, the function `errorbar` is called instead of `plot`.
    loc : string
        keyword passed to `legend` if there is more than one input spectrum.

    """
    import matplotlib.pyplot as mp
    if spectra is None:
        spectra = ell
        ell = None
    if lmax is None:
        lmax = np.max(ell)
    if not isinstance(spectra, (list, np.ndarray, tuple)):
        raise TypeError('Invalid type for the power spectra.')
    if not isinstance(spectra[0], (list, np.ndarray, tuple)):
        nspectra = 1
        spectra = (spectra,)
    else:
        nspectra = len(spectra)
    if nspectra not in (1, 4, 6):
        raise ValueError('Invalid number of spectra.')
    if Dl:
        label = '$\mathcal{{D}}_\ell^{{{}}}$'
    else:
        label = '$\ell(\ell+1)C_\ell^{{{}}}/2\pi$'

    if ell is None:
        ell = np.arange(spectra[0].size)
    fact = ell * (ell + 1) / (2 * np.pi)

    if nspectra == 1:
        if xerr is None and yerr is None:
            mp.plot(ell, fact * spectra[0], **keywords)
        else:
            mp.errorbar(ell, fact * spectra[0], yerr=fact * yerr, xerr=xerr,
                        **keywords)
    else:
        if nspectra == 6:
            ctt, cee, cbb, cte, ceb, ctb = spectra
            mp.plot(ell, fact * ceb, color='brown', label=label.format('EB'))
            mp.plot(ell, fact * ctb, color='cyan', label=label.format('TB'))
        else:
            ctt, cee, cbb, cte = spectra
        mp.plot(ell, fact * ctt, 'k', label=label.format('TT'))
        mp.plot(ell, fact * cte, 'g', label=label.format('TE'))
        mp.plot(ell, fact * cee, 'b', label=label.format('EE'))
        if 'color' not in keywords:
            keywords['color'] = 'r'
        if 'label' not in keywords:
            keywords['label'] = label.format('BB')
        if xerr is None and yerr is None:
            mp.plot(ell, fact * cbb, **keywords)
        else:
            mp.errorbar(ell, fact * cbb, yerr=fact*yerr, xerr=xerr, **keywords)

    mp.xlim(0, lmax)
    mp.xlabel('$\ell$')
    mp.ylabel(label.format('') + ' [$\mu$ K$^2$]')
    if nspectra > 1:
        mp.legend(loc=loc, frameon=False)


def semilogy_spectra(ell, spectra=None, lmax=None, Dl=False, yerr=None,
                     xerr=None, loc='best', **keywords):
    """
    Plot Cl power spectra as sqrt(l(l+1)Cl/2pi) Cl with log scaling
    on the y axis.

    semilogy_spectra([ell,] spectra)

    Parameters
    ----------
    ell : integer array, optional
        The l values of the spectra. If not provided, it is assumed that
        the Cls start from 0 and are not binned.
    spectra : array-like, 4-tuple or 6-tuple
        The Cl spectrum or the TT, EE, BB and TE or TT, EE, BB, TE, EB, TB
        Cl spectra.
    Dl : boolean, optional
        Use `D_l` as label instead of `l(l+1)Cl/2pi`.
    lmax : integer
        Plot upper x limit.
    loc : string
        keyword passed to `legend` if there is more than one input spectrum.

    """
    import matplotlib.pyplot as mp
    if spectra is None:
        spectra = ell
        ell = None
    if lmax is None:
        lmax = np.max(ell)
    if not isinstance(spectra, (list, np.ndarray, tuple)):
        raise TypeError('Invalid type for the power spectra.')
    if not isinstance(spectra[0], (list, np.ndarray, tuple)):
        nspectra = 1
        spectra = (spectra,)
    else:
        nspectra = len(spectra)
    if nspectra not in (1, 4, 6):
        raise ValueError('Invalid number of spectra.')
    if Dl:
        label = '$\sqrt{{\mathcal{{D}}_\ell^{{{}}}}}$'
    else:
        label = '$\sqrt{{\ell(\ell+1)C_\ell^{{{}}}/2\pi}}$'

    if ell is None:
        ell = np.arange(spectra[0].size)
    fact = ell * (ell + 1) / (2 * np.pi)

    with settingerr(invalid='ignore'):
        if nspectra == 1:
            c = fact * spectra[0]
            sc = np.sqrt(c)
            if xerr is None and yerr is None:
                p = mp.plot(ell, np.sqrt(c), **keywords)
            else:
                if yerr is not None:
                    yerr = [sc - np.sqrt(c - fact * yerr),
                            np.sqrt(c + fact * yerr) - sc]
                p = mp.errorbar(ell, sc, yerr=yerr, xerr=xerr, **keywords)
            mp.plot(ell, np.sqrt(-c), linestyle='--', color=p[0].get_color())
        else:
            if nspectra == 6:
                ctt, cee, cbb, cte, ceb, ctb = spectra
                mp.plot(ell, np.sqrt(fact * ceb), color='brown',
                        label=label.format('EB'))
                mp.plot(ell, np.sqrt(-fact * ceb), color='brown',
                        linestyle='--')
                mp.plot(ell, np.sqrt(fact * ctb), color='cyan',
                        label=label.format('TB'))
                mp.plot(ell, np.sqrt(-fact * ctb), color='cyan',
                        linestyle='--')
            else:
                ctt, cee, cbb, cte = spectra
            mp.plot(ell, np.sqrt(fact * ctt), 'k', label=label.format('TT'))
            mp.plot(ell, np.sqrt(fact * cte), 'g', label=label.format('TE'))
            mp.plot(ell, np.sqrt(-fact * cte), 'g', linestyle='--')
            mp.plot(ell, np.sqrt(fact * cee), 'b', label=label.format('EE'))
            c = fact * cbb
            sc = np.sqrt(c)
            if 'color' not in keywords:
                keywords['color'] = 'r'
            if 'label' not in keywords:
                keywords['label'] = label.format('BB')
            if xerr is None and yerr is None:
                mp.plot(ell, sc, **keywords)
            else:
                if yerr is not None:
                    yerr = [sc - np.sqrt(c - fact * yerr),
                            np.sqrt(c + fact * yerr) - sc]
                mp.errorbar(ell, sc, yerr=yerr, xerr=xerr, **keywords)

    mp.yscale('log')
    mp.xlim(0, lmax)
    mp.ylim(0.01, 100)
    mp.xlabel('$\ell$')
    mp.ylabel(label.format('') + ' [$\mu$ K]')
    if nspectra > 1:
        mp.legend(loc=loc, frameon=False)
