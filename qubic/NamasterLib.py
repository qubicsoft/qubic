import numpy as np
import healpy as hp
import pymaster as nmt

__all__ = ['Namaster']


class Namaster(object):

    def __init__(self, weight_mask, lmin, lmax, delta_ell, aposize=10.0, apotype='C1'):
        """

        Parameters
        ----------
        weight_mask: ndarray
        lmin: int
            Minimal l for power spectrum.
        lmax: int
            Maximal l for power spectrum.
        delta_ell: int

        aposize: apodization scale in degrees.
            10.0 by default.
        apotype: apodization type.
            Three methods implemented: C1, C2 and Smooth.
            'C1' by default.
        """

        lmin = int(lmin)
        lmax = int(lmax)
        delta_ell = int(delta_ell)
        if lmin < 1:
            raise ValueError('Input lmin is less than 1.')
        if lmax < lmin:
            raise ValueError('Input lmax is less than lmin.')
        self.lmin = lmin
        self.lmax = lmax
        self.delta_ell = delta_ell
        self.ells, self.weights, self.bpws = self._binning()
        self.aposize = aposize
        self.apotype = apotype
        self.ell_binned = None
        self.fsky = None

        # ## Mask
        if weight_mask is not None:
            self.weight_mask = np.asarray(weight_mask)
            self.mask_apo = self.get_apodized_mask()
            self.fsky = np.sum(self.mask_apo / np.max(self.mask_apo)) / len(self.mask_apo)

        self.f0 = None
        self.f2 = None
        self.f0bis = None
        self.f2bis = None
        self.w = None
        self.cw = None

    def get_binning(self, nside):
        """
        Binning informations.
        Parameters
        ----------
        nside: the nside of the maps

        Returns
        -------
        The binned monopoles and the NmtBin object
        """
        b = nmt.NmtBin(nside,
                       bpws=self.bpws,
                       ells=self.ells,
                       weights=self.weights,
                       lmax=self.lmax,
                       is_Dell=True)
        ell_binned = b.get_effective_ells()
        self.ell_binned = ell_binned

        return ell_binned, b

    def bin_spectra(self, input_cls, nside):
        """
        Bin a spectrum.
        Parameters
        ----------
        input_cls: 1D array
            input Cls spectrum
        nside: the nside of the maps

        Returns
        -------
        The binned power spectrum Dl = l(l+1)*Cl.

        """
        ell_binned, b = self.get_binning(nside)
        input_cls_reshape = np.reshape(input_cls[:self.lmax + 1], (1, self.lmax + 1))
        cls_binned = b.bin_cell(input_cls_reshape)

        fact = 2 * np.pi / (ell_binned * (ell_binned + 1))

        return fact * cls_binned

    def get_apodized_mask(self):
        """
        Make an apodized mask. The pure-B formalism requires the mask to be
        differentiable along the edges. The 'C1' and 'C2' apodization types
        supported by mask_apodization achieve this.
        """
        mask_apo = nmt.mask_apodization(self.weight_mask,
                                        aposize=self.aposize,
                                        apotype=self.apotype)
        return mask_apo

    def get_fields(self, map, mask_apo=None, purify_e=False, purify_b=True, beam_correction=None):
        """

        Parameters
        ----------
        map: array
            IQU maps, shape (3, #pixels)
        mask_apo: array, optionnal (if not given the one used at the object's instanciation is used)
            Apodized mask.
        purify_e: bool, optional
            False by default.
        purify_b: bool, optional
            True by default.
            Note that generally it's not a good idea to purify both,
            since you'll lose sensitivity on E
        beam_correction: bool, optional
            None by default.
            If True, a correction by the Qubic beam at 150GHz is applied.
            You can also give the beam FWHM you want to correct for.
        Returns
        -------
        f0, f2: spin-0 and spin-2 Namaster fields.

        """

        # The maps may contain hp.UNSEEN - They must be replaced with zeros
        undefpix = map == hp.UNSEEN
        map[undefpix] = 0
        mp_t, mp_q, mp_u = map
        nside = hp.npix2nside(len(mp_t))

        if mask_apo is None:
            mask_apo = self.mask_apo

        if beam_correction is not None:
            if beam_correction is True:
                # Default value for QUBIC at 150 GHz
                beam = hp.gauss_beam(np.deg2rad(0.39268176),
                                     lmax=3 * nside - 1)
            else:
                beam = hp.gauss_beam(np.deg2rad(beam_correction),
                                     lmax=3 * nside - 1)
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

        self.f0 = f0
        self.f2 = f2

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

    def get_spectra(self, map, mask_apo=None, map2=None, purify_e=False, purify_b=True, w=None,
                    beam_correction=None, pixwin_correction=None, verbose=True):
        """
        Get spectra from IQU maps.
        Parameters
        ----------
        map: array
            IQU maps, shape (3, #pixels)
        mask_apo: array, optional (if not given then the maks used at the object's instanciation is used)
            Apodized mask.
        map2: array
            IQU maps, shape (3, #pixels) for Cross-Spectra
        purify_e: bool, optional
            False by default.
        purify_b: bool, optional
            True by default.
            Note that generally it's not a good idea to purify both,
            since you'll lose sensitivity on E
        w: list with Namaster workspace [w00, w22, w02]
            If None the workspaces will be created.
        beam_correction: bool, optional
            None by default.
            If True, a correction by the Qubic beam at 150GHz is applied.
            You can also give the beam FWHM you want to correct for.
        pixwin_correction: bool, optional
            If not None, a correction for the pixel integration function is applied.
        verbose: bool, optional
            True by default.
        Returns
        -------
        ell_binned
        spectra: TT, EE, BB, TE spectra, Dl = ell * (ell + 1) / 2 * PI * Cl
        w: List containing the NmtWorkspaces [w00, w22, w02]

        """
        nside = hp.npix2nside(len(map[0]))
        self.ell_binned, b = self.get_binning(nside)

        if mask_apo is None:
            mask_apo = self.mask_apo

        # Get fields
        f0, f2 = self.get_fields(map, mask_apo=mask_apo,
                                 purify_e=purify_e,
                                 purify_b=purify_b,
                                 beam_correction=beam_correction)

        # Cross-Spectra case
        if map2 is not None:
            f0bis, f2bis = self.get_fields(map2, mask_apo=mask_apo,
                                           purify_e=purify_e,
                                           purify_b=purify_b,
                                           beam_correction=beam_correction)
        else:
            f0bis = f0
            f2bis = f2

        self.f0 = f0
        self.f0bis = f0bis
        self.f2 = f2
        self.f2bis = f2bis

        # Make workspaces
        if w is None:
            w00 = nmt.NmtWorkspace()
            w00.compute_coupling_matrix(f0, f0bis, b)

            w22 = nmt.NmtWorkspace()
            w22.compute_coupling_matrix(f2, f2bis, b)

            w02 = nmt.NmtWorkspace()
            w02.compute_coupling_matrix(f0, f2bis, b)
            w = [w00, w22, w02]
            self.w = w
        else:
            w00 = w[0]
            w22 = w[1]
            w02 = w[2]

        # Get Cls
        c00 = self.compute_master(f0, f0bis, w00)
        c22 = self.compute_master(f2, f2bis, w22)
        c02 = self.compute_master(f0, f2bis, w02)

        # Put the 4 spectra in one array
        spectra = np.array([c00[0], c22[0], c22[3], c02[0]]).T
        if verbose:
            print('Getting TT, EE, BB, TE spectra in that order.')

        if pixwin_correction is not None:
            pwb = self.get_pixwin_correction(nside)
            for i in range(4):
                spectra[:, i] /= (pwb[1] ** 2)

        return self.ell_binned, spectra, w

    def get_pixwin_correction(self, nside):
        pw = hp.pixwin(nside, pol=True, lmax=self.lmax)
        pw = [pw[0][: self.lmax + 1], pw[1][: self.lmax + 1]]

        ell_binned, b = self.get_binning(nside)
        pwb = 2 * np.pi / (ell_binned * (ell_binned + 1)) * b.bin_cell(np.array(pw))

        return pwb

    def _binning(self):
        ells = np.arange(self.lmin, self.lmax, dtype='int32')  # Array of multipoles
        weights = 1. / self.delta_ell * np.ones_like(ells)  # Array of weights
        bpws = -1 + np.zeros_like(ells)  # Array of bandpower indices
        i = 0
        # print(self.lmax - self.delta_ell)
        while self.delta_ell * (i + 1) < (self.lmax - self.delta_ell):
            bpws[self.delta_ell * i: self.delta_ell * (i + 1)] = i
            i += 1

        return ells, weights, bpws

    def knox_errors(self, clth):
        dcl = np.sqrt(2. / (2 * self.ell_binned + 1) / self.fsky / self.delta_ell) * clth
        return dcl

    def knox_covariance(self, clth):
        dcl = self.knox_errors(clth)
        return np.diag(dcl ** 2)

    def get_covariance_coeff(self):
        if self.cw is None:
            cw = nmt.NmtCovarianceWorkspace()
            cw.compute_coupling_coefficients(self.f0, self.f0, self.f0, self.f0, lmax=self.lmax)
            self.cw = cw

    def get_covariance_TT_TT(self, cl_tt):
        self.get_covariance_coeff()
        w00 = self.w[0]
        covar_00_00 = nmt.gaussian_covariance(self.cw,
                                              0, 0, 0, 0,  # Spins of the 4 fields
                                              [cl_tt * 0],  # TT
                                              [cl_tt * 0],  # TT
                                              [cl_tt * 0],  # TT
                                              [cl_tt * 0],  # TT
                                              w00, wb=w00).reshape([len(self.ell_binned), 1,
                                                                    len(self.ell_binned), 1])
        covar_TT_TT = covar_00_00[:, 0, :, 0]
        return covar_TT_TT

    def get_covariance_EE_EE(self, cl_ee):
        self.get_covariance_coeff()
        w22 = self.w[1]
        n_ell = len(cl_ee)
        cl_eb = np.zeros(n_ell)
        cl_bb = np.zeros(n_ell)
        covar_22_22 = nmt.gaussian_covariance(self.cw, 2, 2, 2, 2,  # Spins of the 4 fields
                                              [cl_ee, cl_eb,
                                               cl_eb, cl_bb],  # EE, EB, BE, BB
                                              [cl_ee, cl_eb,
                                               cl_eb, cl_bb],  # EE, EB, BE, BB
                                              [cl_ee, cl_eb,
                                               cl_eb, cl_bb],  # EE, EB, BE, BB
                                              [cl_ee, cl_eb,
                                               cl_eb, cl_bb],  # EE, EB, BE, BB
                                              w22, wb=w22).reshape([len(self.ell_binned), 4,
                                                                    len(self.ell_binned), 4])

        covar_EE_EE = covar_22_22[:, 0, :, 0]
        # covar_EE_EB = covar_22_22[:, 0, :, 1]
        # covar_EE_BE = covar_22_22[:, 0, :, 2]
        # covar_EE_BB = covar_22_22[:, 0, :, 3]
        # covar_EB_EE = covar_22_22[:, 1, :, 0]
        # covar_EB_EB = covar_22_22[:, 1, :, 1]
        # covar_EB_BE = covar_22_22[:, 1, :, 2]
        # covar_EB_BB = covar_22_22[:, 1, :, 3]
        # covar_BE_EE = covar_22_22[:, 2, :, 0]
        # covar_BE_EB = covar_22_22[:, 2, :, 1]
        # covar_BE_BE = covar_22_22[:, 2, :, 2]
        # covar_BE_BB = covar_22_22[:, 2, :, 3]
        # covar_BB_EE = covar_22_22[:, 3, :, 0]
        # covar_BB_EB = covar_22_22[:, 3, :, 1]
        # covar_BB_BE = covar_22_22[:, 3, :, 2]
        # covar_BB_BB = covar_22_22[:, 3, :, 3]

        return covar_EE_EE

    def get_covariance_BB_BB(self, cl_bb):
        w22 = self.w[1]
        n_ell = len(cl_bb)
        cl_eb = np.zeros(n_ell)
        cl_ee = np.zeros(n_ell)
        covar_22_22 = nmt.gaussian_covariance(self.cw, 2, 2, 2, 2,  # Spins of the 4 fields
                                              [cl_ee, cl_eb,
                                               cl_eb, cl_bb],  # EE, EB, BE, BB
                                              [cl_ee, cl_eb,
                                               cl_eb, cl_bb],  # EE, EB, BE, BB
                                              [cl_ee, cl_eb,
                                               cl_eb, cl_bb],  # EE, EB, BE, BB
                                              [cl_ee, cl_eb,
                                               cl_eb, cl_bb],  # EE, EB, BE, BB
                                              w22, wb=w22)

        covar_22_22 = np.reshape(covar_22_22, (len(self.ell_binned), 4, len(self.ell_binned), 4))

        # covar_EE_EE = covar_22_22[:, 0, :, 0]
        # covar_EE_EB = covar_22_22[:, 0, :, 1]
        # covar_EE_BE = covar_22_22[:, 0, :, 2]
        # covar_EE_BB = covar_22_22[:, 0, :, 3]
        # covar_EB_EE = covar_22_22[:, 1, :, 0]
        # covar_EB_EB = covar_22_22[:, 1, :, 1]
        # covar_EB_BE = covar_22_22[:, 1, :, 2]
        # covar_EB_BB = covar_22_22[:, 1, :, 3]
        # covar_BE_EE = covar_22_22[:, 2, :, 0]
        # covar_BE_EB = covar_22_22[:, 2, :, 1]
        # covar_BE_BE = covar_22_22[:, 2, :, 2]
        # covar_BE_BB = covar_22_22[:, 2, :, 3]
        # covar_BB_EE = covar_22_22[:, 3, :, 0]
        # covar_BB_EB = covar_22_22[:, 3, :, 1]
        # covar_BB_BE = covar_22_22[:, 3, :, 2]
        covar_BB_BB = covar_22_22[:, 3, :, 3]

        return covar_BB_BB

    def get_covariance_TE_TE(self, cl_te):
        self.get_covariance_coeff()
        w02 = self.w[2]
        n_ell = len(cl_te)
        cl_tt = np.zeros(n_ell)
        cl_bb = np.zeros(n_ell)
        cl_ee = np.zeros(n_ell)
        cl_eb = np.zeros(n_ell)
        cl_tb = np.zeros(n_ell)
        covar_02_02 = nmt.gaussian_covariance(self.cw, 0, 2, 0, 2,  # Spins of the 4 fields
                                              [cl_tt],  # TT
                                              [cl_te, cl_tb],  # TE, TB
                                              [cl_te, cl_tb],  # ET, BT
                                              [cl_ee, cl_eb,
                                               cl_eb, cl_bb],  # EE, EB, BE, BB
                                              w02, wb=w02)
        covar_02_02 = np.reshape(covar_02_02, (len(self.ell_binned), 2, len(self.ell_binned), 2))
        covar_TE_TE = covar_02_02[:, 0, :, 0]
        # covar_TE_TB = covar_02_02[:, 0, :, 1]
        # covar_TB_TE = covar_02_02[:, 1, :, 0]
        # covar_TB_TB = covar_02_02[:, 1, :, 1]

        return covar_TE_TE
