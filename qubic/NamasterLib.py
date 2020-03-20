import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt

__all__ = ['Namaster']


class Namaster(object):

    def __init__(self, mask, lmin, lmax, delta_ell):

        mask = np.asarray(mask)
        lmin = int(lmin)
        lmax = int(lmax)
        delta_ell = int(delta_ell)
        if lmin < 1:
            raise ValueError('Input lmin is less than 1.')
        if lmax < lmin:
            raise ValueError('Input lmax is less than lmin.')
        wl = hp.anafast(mask)[:lmax + 1]
        self.mask = mask
        self.lmin = lmin
        self.lmax = lmax
        self.delta_ell = delta_ell
        self.wl = wl
        self.ell_binned, self._p, self._q = self._bin_ell()

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
        return np.dot(spectra[..., :self.lmax + 1], self._p.T) * fact_binned

    def get_map(self, d, spectra):
        mp_i, mp_q, mp_u = hp.synfast(spectra,
                                      nside=d['nside'],
                                      fwhm=np.deg2rad(d['synthbeam_peak150_fwhm']),
                                      pixwin=True,
                                      new=True,
                                      verbose=False)
        return np.array((mp_i, mp_q, mp_u))

    def get_fields(self, map, d):
        mp_i, mp_q, mp_u = map
        mask_apo = nmt.mask_apodization(self.mask, 1, apotype='C1')
        beam = hp.gauss_beam(np.deg2rad(d['synthbeam_peak150_fwhm']),
                             self.lmax)
        f0 = nmt.NmtField(mask_apo, [mp_i], beam=beam)
        # This creates a spin-2 field with both pure E and B.
        # Note that generally it's not a good idea to purify both,
        # since you'll lose sensitivity on E
        f2 = nmt.NmtField(mask_apo, [mp_q, mp_u], beam=beam, purify_e=False, purify_b=True)

        return f0, f2

    def compute_master(self, field_a, field_b, workspace):
        cl_coupled = nmt.compute_coupled_cell(field_a, field_b)
        cl_decoupled = workspace.decouple_cell(cl_coupled)
        return cl_decoupled

    def get_spectra(self, map, d):

        # Select a binning scheme
        b = nmt.NmtBin(d['nside'], nlb=20, is_Dell=True)
        leff = b.get_effective_ells()

        f0, f2 = self.get_fields(map, d)
        w_tt = nmt.NmtWorkspace()
        w_tt.compute_coupling_matrix(f0, f0, b)

        w_te = nmt.NmtWorkspace()
        w_te.compute_coupling_matrix(f0, f2, b)

        w_bbee = nmt.NmtWorkspace()
        w_bbee.compute_coupling_matrix(f2, f2, b)

        cl_tt = self.compute_master(f0, f0, w_tt)
        cl_te = self.compute_master(f0, f2, w_te)
        cl_bbee = self.compute_master(f2, f2, w_bbee)

        return leff, cl_tt, cl_te, cl_bbee

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
