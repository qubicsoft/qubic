import numpy as np
import healpy as hp
import pymaster as nmt

__all__ = ['Namaster']


class Namaster(object):

    def __init__(self, seenpix, lmin, lmax, delta_ell):

        seenpix = np.asarray(seenpix)
        lmin = int(lmin)
        lmax = int(lmax)
        delta_ell = int(delta_ell)
        if lmin < 1:
            raise ValueError('Input lmin is less than 1.')
        if lmax < lmin:
            raise ValueError('Input lmax is less than lmin.')
        self.seenpix = seenpix
        self.lmin = lmin
        self.lmax = lmax
        self.delta_ell = delta_ell
        self.ell_binned, self._p, self._q = self._bin_ell()


    def bin_spectra(self, spectra):
        """
        Average spectra in bins specified by lmin, lmax and delta_ell,
        weighted by `l(l+1)`.
        spectra: 1D array
            Input spectrum starting from l=0.

        """
        spectra = np.asarray(spectra)
        lmax = spectra.shape[-1] - 1
        if lmax < self.lmax:
            raise ValueError('The input spectra do not have enough l.')

        fact_binned = 2 * np.pi / (self.ell_binned * (self.ell_binned + 1))
        return np.dot(spectra[..., :self.lmax + 1], self._p.T) * fact_binned

    def get_apodized_mask(self, aposize=10.0, apotype='C1'):
        """
        Make an apodized mask. The pure-B formalism requires the mask to be
        differentiable along the edges. The 'C1' and 'C2' apodization types
        supported by mask_apodization achieve this.
        Parameters
        ----------
        aposize: apodization scale in degrees.
            10.0 by default.
        apotype: apodization type.
            Three methods implemented: C1, C2 and Smooth.
            'C1' by default.

        """
        msk = np.zeros_like(self.seenpix)
        msk[self.seenpix] = 1.

        mask_apo = nmt.mask_apodization(msk, aposize=aposize, apotype=apotype)
        return mask_apo

    def get_fields(self, map, d, mask_apo, purify_e=False, purify_b=True):
        """

        Parameters
        ----------
        map: array
            IQU maps, shape (3, #pixels)
        d: Qubic dictionary
        mask_apo: array
            Apodized mask.
        purify_e: bool
            False by default.
        purify_b: bool
            True by default.
            Note that generally it's not a good idea to purify both,
            since you'll lose sensitivity on E
        Returns
        -------
        f0, f2: spin-0 and spin-2 Namaster fields.

        """
        mp_t, mp_q, mp_u = map
        beam = hp.gauss_beam(np.deg2rad(d['synthbeam_peak150_fwhm']), self.lmax)

        f0 = nmt.NmtField(mask_apo, [mp_t])#, beam=beam)

        f2 = nmt.NmtField(mask_apo, [mp_q, mp_u], purify_e=purify_e, purify_b=purify_b)
        return f0, f2

    def compute_master(self, field_a, field_b, workspace):
        """
        Parameters
        ----------
        field_a: NmtField
        field_b: NmtField
        workspace: NmtWorkspace
            Contains the coupling matrix

        Returns
        -------
            decoupled Cls.

        """
        cl_coupled = nmt.compute_coupled_cell(field_a, field_b)
        cl_decoupled = workspace.decouple_cell(cl_coupled)
        return cl_decoupled

    def get_spectra(self, map, d, mask_apo, nlb=16, purify_e=False, purify_b=True):
        """
        Get spectra from IQU maps.
        Parameters
        ----------
        map: array
            IQU maps, shape (3, #pixels)
        d: Qubic dictionary
        mask_apo: array
            Apodized mask.
        nlb: int
            Constant bandpower width. By default it is 16.
        purify_e: bool
            False by default.
        purify_b: bool
            True by default.
            Note that generally it's not a good idea to purify both,
            since you'll lose sensitivity on E

        Returns
        -------
        leff: effective l
        spectra: TT, EE, BB, TE spectra
        w: List containing the NmtWorkspaces [w00, w22, w02]

        """

        # Select a binning scheme
        b = nmt.NmtBin(d['nside'], nlb=nlb, is_Dell=True, lmax=self.lmax)
        leff = b.get_effective_ells()

        # Get fields
        f0, f2 = self.get_fields(map, d, mask_apo, purify_e=purify_e, purify_b=purify_b)

        # Make workspaces
        w00 = nmt.NmtWorkspace()
        w00.compute_coupling_matrix(f0, f0, b)

        w22 = nmt.NmtWorkspace()
        w22.compute_coupling_matrix(f2, f2, b)

        w02 = nmt.NmtWorkspace()
        w02.compute_coupling_matrix(f0, f2, b)
        w = [w00, w22, w02]

        # Get Cls
        c00 = self.compute_master(f0, f0, w[0])
        c22 = self.compute_master(f2, f2, w[1])
        c02 = self.compute_master(f0, f2, w[2])

        # Put the 4 spectra in one array
        spectra = np.array([c00[0], c22[0], c22[3], c02[0]]).T
        print('Getting TT, EE, BB, TE spectra in that order.')

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
