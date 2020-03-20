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

    def get_fields(self, map, d, purify_e=False, purify_b=True):
        mp_i, mp_q, mp_u = map
        mask_apo = nmt.mask_apodization(self.mask, 10.0, apotype='C1')
        #beam = hp.gauss_beam(np.deg2rad(d['synthbeam_peak150_fwhm']), self.lmax)

        f0 = nmt.NmtField(mask_apo, [mp_i, mp_i])#, beam=beam)
        # This creates a spin-2 field with both pure E and B.
        # Note that generally it's not a good idea to purify both,
        # since you'll lose sensitivity on E
        f2 = nmt.NmtField(mask_apo, [mp_q, mp_u], purify_e=purify_e, purify_b=purify_b)
        return f0, f2

    def compute_master(self, field_a, field_b, workspace):
        cl_coupled = nmt.compute_coupled_cell(field_a, field_b)
        cl_decoupled = workspace.decouple_cell(cl_coupled)
        return cl_decoupled

    def get_spectra(self, map, d, workspace=None, purify_e=False, purify_b=True):

        # Select a binning scheme
        b = nmt.NmtBin(d['nside'], nlb=16, is_Dell=True)
        leff = b.get_effective_ells()

        f0, f2 = self.get_fields(map, d, purify_e=purify_e, purify_b=purify_b)
        if workspace is None:
            w00= nmt.NmtWorkspace()
            w00.compute_coupling_matrix(f0, f0, b)

            w22 = nmt.NmtWorkspace()
            w22.compute_coupling_matrix(f2, f2, b)
            w = [w00, w22]
        else:
            w = workspace

        c00 = self.compute_master(f0, f0, w[0])
        c22 = self.compute_master(f2, f2, w[1])

        spectra = np.array([c00[0], c22[0], c22[3]])

        return leff, spectra, w

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
