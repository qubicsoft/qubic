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
        self.ells, self.weights, self.bpws = self._binning()

    def get_binning(self, d):
        """
        Binning informations.
        Parameters
        ----------
        d: Qubic dictionary

        Returns
        -------
        The binned monopoles and the NmtBin object
        """
        b = nmt.NmtBin(d['nside'], bpws=self.bpws, ells=self.ells, weights=self.weights, is_Dell=True)
        ell_binned = b.get_effective_ells()

        return ell_binned, b

    def bin_spectra(self, input_cls, d):
        """
        Bin a spectrum.
        Parameters
        ----------
        input_cls: 1D array
            input Cls spectrum
        d: Qubic dictionary

        Returns
        -------
        The binned power spectrum Dl = l(l+1)*Cl.

        """
        ell_binned, b = self.get_binning(d)
        input_cls_reshape = np.reshape(input_cls[:self.lmax + 1], (1, self.lmax + 1))
        cls_binned = b.bin_cell(input_cls_reshape)

        fact = 2 * np.pi / (ell_binned * (ell_binned + 1))

        return fact * cls_binned

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

    def get_fields(self, map, d, mask_apo, purify_e=False, purify_b=True, beam_correction=False):
        """

        Parameters
        ----------
        map: array
            IQU maps, shape (3, #pixels)
        d: Qubic dictionary
        mask_apo: array
            Apodized mask.
        purify_e: bool, optional
            False by default.
        purify_b: bool, optional
            True by default.
            Note that generally it's not a good idea to purify both,
            since you'll lose sensitivity on E
        beam_correction: bool, optional
            If True, a correction by the Qubic beam is applied
        Returns
        -------
        f0, f2: spin-0 and spin-2 Namaster fields.

        """
        mp_t, mp_q, mp_u = map
        if beam_correction:
            beam = hp.gauss_beam(np.deg2rad(d['synthbeam_peak150_fwhm']), self.lmax)
        else:
            beam = None

        f0 = nmt.NmtField(mask_apo,
                          [mp_t],
                          beam=beam)

        f2 = nmt.NmtField(mask_apo,
                          [mp_q, mp_u],
                          purify_e=purify_e,
                          purify_b=purify_b,
                          beam=beam)
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

    def get_spectra(self, map, d, mask_apo, purify_e=False, purify_b=True, w=None,
                    beam_correction=False, pixwin_correction=False):
        """
        Get spectra from IQU maps.
        Parameters
        ----------
        map: array
            IQU maps, shape (3, #pixels)
        d: Qubic dictionary
        mask_apo: array
            Apodized mask.
        purify_e: bool, optional
            False by default.
        purify_b: bool, optional
            True by default.
            Note that generally it's not a good idea to purify both,
            since you'll lose sensitivity on E
        w: list with Namaster workspace [w00, w22, w02]
            If None the workspaces will be created.
        beam_correction: bool, optional
            If True, a correction by the Qubic beam is applied.
        pixwin_correction: bool, optional
            If True, a correction for the pixel integration function is applied.

        Returns
        -------
        ell_binned
        spectra: TT, EE, BB, TE spectra, Dl = ell * (ell + 1) / 2 * PI * Cl
        w: List containing the NmtWorkspaces [w00, w22, w02]

        """

        ell_binned, b = self.get_binning(d)

        # Get fields
        f0, f2 = self.get_fields(map, d, mask_apo,
                                 purify_e=purify_e,
                                 purify_b=purify_b,
                                 beam_correction=beam_correction)

        # Make workspaces
        if w is None:
            w00 = nmt.NmtWorkspace()
            w00.compute_coupling_matrix(f0, f0, b)

            w22 = nmt.NmtWorkspace()
            w22.compute_coupling_matrix(f2, f2, b)

            w02 = nmt.NmtWorkspace()
            w02.compute_coupling_matrix(f0, f2, b)
            w = [w00, w22, w02]
        else:
            w00 = w[0]
            w22 = w[1]
            w02 = w[2]

        # Get Cls
        c00 = self.compute_master(f0, f0, w00)
        c22 = self.compute_master(f2, f2, w22)
        c02 = self.compute_master(f0, f2, w02)

        # Put the 4 spectra in one array
        spectra = np.array([c00[0], c22[0], c22[3], c02[0]]).T
        print('Getting TT, EE, BB, TE spectra in that order.')

        if pixwin_correction:
            pwb = self.get_pixwin_correction(d)
            for i in range(4):
                spectra[:, i] /= (pwb[1] ** 2)

        return ell_binned, spectra, w

    def get_pixwin_correction(self, d):
        nside = d['nside']
        pw = hp.pixwin(nside, pol=True)
        pw = [pw[0][: self.lmax + 1], pw[1][: self.lmax + 1]]

        ell_binned, b = self.get_binning(d)
        pwb = 2 * np.pi / (ell_binned * (ell_binned + 1)) * b.bin_cell(np.array(pw))

        return pwb

    def _binning(self):
        ells = np.arange(self.lmin, self.lmax, dtype='int32')  # Array of multipoles
        weights = 1. / self.delta_ell * np.ones_like(ells)  # Array of weights
        bpws = -1 + np.zeros_like(ells)  # Array of bandpower indices
        i = 0
        print(self.lmax - self.delta_ell)
        while self.delta_ell * (i + 1) < (self.lmax - self.delta_ell):
            bpws[self.delta_ell * i: self.delta_ell * (i + 1)] = i
            i += 1

        return ells, weights, bpws
