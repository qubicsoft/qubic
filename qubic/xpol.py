from __future__ import division

import healpy as hp
import numpy as np
import qubic._flib as flib

__all__ = ['Xpol']


class Xpol(object):
    """
    (Cross-) power spectra estimation using the Xpol method.
    Hinshaw et al. 2003, Tristram 2005.

    Example
    -------
    xpol = Xpol(mask, lmin, lmax, delta_ell)
    ell_binned = xpol.ell_binned
    biased, unbiased = xpol.get_spectra(map)
    biased, unbiased = xpol.get_spectra(map1, map2)

    """
    def __init__(self, mask, lmin, lmax, delta_ell):
        """
        Parameters
        ----------
        mask : boolean Healpix map
            Mask defining the region of interest (of value True)
        lmin : int
            Lower bound of the first l bin.
        lmax : int
            Highest l value to be considered. The inclusive upper bound of
            the last l bin is lesser or equal to this value.
        delta_ell :
            The l bin width.

        """
        mask = np.asarray(mask)
        lmin = int(lmin)
        lmax = int(lmax)
        if lmin < 1:
            raise ValueError('Input lmin is less than 1.')
        if lmax < lmin:
            raise ValueError('Input lmax is less than lmin.')
        delta_ell = int(delta_ell)
        wl = hp.anafast(mask)[:lmax+1]
        self.mask = mask
        self.lmin = lmin
        self.lmax = lmax
        self.delta_ell = delta_ell
        self.wl = wl
        self.ell_binned, self._p, self._q = self._bin_ell()
        mll_binned = self._get_Mll()
        self.mll_binned_inv = np.linalg.inv(mll_binned)

    def bin_spectra(self, spectra):
        """
        Average spectra in bins specified by lmin, lmax and delta_ell,
        weighted by `l(l+1)`.

        """
        spectra = np.asarray(spectra)
        lmax = spectra.shape[-1] - 1
        if lmax < self.lmax:
            raise ValueError('The input spectra do not have enough l.')
        fact_binned = 2 * np.pi / (self.ell_binned * (self.ell_binned + 1))
        return np.dot(spectra[..., :self.lmax+1], self._p.T) * fact_binned

    def get_spectra(self, map1, map2=None):
        """
        Return biased and Xpol-debiased estimations of the power spectra of
        a Healpix map or of the cross-power spectra if *map2* is provided.

        xpol = Xpol(mask, lmin, lmax, delta_ell)
        biased, unbiased = xpol.get_spectra(map1, [map2])

        The unbiased Cls are binned. The number of bins is given by
        (lmax - lmin) // delta_ell, using the values specified in the Xpol's
        object initialisation. As a consequence, the upper bound of the highest
        l bin may be less than lmax. The central value of the bins can be
        obtained through the attribute `xpol.ell_binned`.

        Parameters
        ----------
        map1 : Nx3 or 3xN array
            The I, Q, U Healpix maps.
        map2 : Nx3 or 3xN array, optional
            The I, Q, U Healpix maps.

        Returns
        -------
        biased : float array of shape (6, lmax+1)
            The anafast's pseudo (cross-) power spectra for TT, EE, BB, TE, EB,
            TB. The corresponding l values are given by `np.arange(lmax + 1)`.

        unbiased : float array of shape (6, nbins)
            The Xpol's (cross-) power spectra for TT, EE, BB, TE, EB, TB.
            The corresponding l values are given by `xpol.ell_binned`.

        """
        map1 = np.asarray(map1)
        if map1.shape[-1] == 3:
            map1 = map1.T
        if map2 is None:
            biased = hp.anafast(map1 * self.mask, pol=True)
        else:
            map2 = np.asarray(map2)
            if map2.shape[-1] == 3:
                map2 = map2.T
            biased = hp.anafast(map1 * self.mask, map2 * self.mask, pol=True)
        biased = np.array([cl[:self.lmax+1] for cl in biased])
        binned = self.bin_spectra(biased)
        fact_binned = self.ell_binned * (self.ell_binned + 1) / (2 * np.pi)
        binned *= fact_binned
        unbiased = np.dot(self.mll_binned_inv, binned.ravel()).reshape(6, -1)
        unbiased /= fact_binned
        return biased, unbiased

    def _bin_ell(self):
        nbins = (self.lmax - self.lmin + 1) // self.delta_ell
        start = self.lmin + np.arange(nbins) * self.delta_ell
        stop = start + self.delta_ell
        ell_binned = (start + stop - 1) / 2

        ell2 = np.arange(self.lmax + 1)
        ell2 = ell2 * (ell2 + 1) / (2 * np.pi)
        p = np.zeros((nbins, self.lmax + 1))
        q = np.zeros((self.lmax + 1, nbins))

        for b, (a, z) in enumerate(zip(start, stop)):
            p[b, a:z] = ell2[a:z] / (z - a)
            q[a:z, b] = 1 / ell2[a:z]

        return ell_binned, p, q

    def _get_Mll_blocks(self):
        TT_TT, EE_EE, EE_BB, TE_TE, EB_EB, ier = flib.xpol.mll_blocks_pol(
            self.lmax, self.wl)
        if ier > 0:
            msg = ['Either L2 < ABS(M2) or L3 < ABS(M3).',
                   'Either L2+ABS(M2) or L3+ABS(M3) non-integer.'
                   'L1MAX-L1MIN not an integer.',
                   'L1MAX less than L1MIN.',
                   'NDIM less than L1MAX-L1MIN+1.'][ier-1]
            raise RuntimeError(msg)
        return TT_TT, EE_EE, EE_BB, TE_TE, EB_EB

    def _get_Mll(self, binning=True):
        TT_TT, EE_EE, EE_BB, TE_TE, EB_EB = self._get_Mll_blocks()
        if binning:
            def func(x):
                return np.dot(np.dot(self._p, x), self._q)
            n = len(self.ell_binned)
            TT_TT = func(TT_TT)
            EE_EE = func(EE_EE)
            EE_BB = func(EE_BB)
            TE_TE = func(TE_TE)
            EB_EB = func(EB_EB)
        else:
            n = self.lmax + 1
        out = np.zeros((6*n, 6*n))
        out[  0:  n,   0:  n] = TT_TT
        out[  n:2*n,   n:2*n] = EE_EE
        out[2*n:3*n, 2*n:3*n] = EE_EE
        out[  n:2*n, 2*n:3*n] = EE_BB
        out[2*n:3*n,   n:2*n] = EE_BB
        out[3*n:4*n, 3*n:4*n] = TE_TE
        out[4*n:5*n, 4*n:5*n] = TE_TE
        out[5*n:6*n, 5*n:6*n] = EB_EB
        return out
