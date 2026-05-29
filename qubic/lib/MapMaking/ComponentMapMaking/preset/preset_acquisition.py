import numpy as np
from fgbuster import component_model as c
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

from qubic.lib.Instrument.Qnoise import QubicTotNoise


class PresetAcquisition:
    r"""Preset Acquisition.

    Instance to initialize the Components Map-Making. It defines the data acquisition variables and methods.

    Parameters
    ----------
    seed_noise : int
        Seed for random noise generation.
    preset_tools : object
        Class containing tools and simulation parameters.
    preset_external : object
        Class containing external variables and methods.
    preset_qubic : object
        Class containing qubic operator and variables and methods.
    preset_sky : object
        Class containing sky varaibles and methods.
    preset_comp : object
        Class containing component variables and methods.
    preset_mixing_matrix : object
        Class containing mixing-matrix variables and methods.
    preset_gain : object
        Class containing detector gain variables and methods.

    Attributes
    ----------
    seed_noise: int
        Seed for random noise realization.
    params_foregrounds: dict
        Dictionary containing the paramters associated with foregrounds.
    ites_to_converge: int
        Max iteration number for PCG algorithm.
    invN: BlockDiagonalOperator
        Inverse noise covariance matrix with input shape :math:`(N_{det}*N_{samples} + N_{pix}*N_{planck}*N_{stokes})` and output shape :math:`(N_{det}*N_{samples} + N_{pix}*N_{planck}*N_{stokes})`.
    M: DiagonalOperator
        Preconditioner for PCG algorithm.
    fwhm_qubic_rec: array_like
        Resolution of the reconstructed maps.
    H: BlockColumnOperator
        Pointing matrix.
    TOD_qubic: array_like
        QUBIC simulated TOD.
    TOD_external: array_like
        Planck simulated TOD.
    TOD_obs: array_like
        Simulated observed TOD.
    beta_iter: array_like
        Spectral indices, if d1 :math:`(12 \times Nside_{\beta}^2, N_{comp}-1)`, if not :math:`(N_{comp}-1)`.
    allbeta: array_like
        Spectral indeices, if d1 :math:`(iter, 12 \times Nside_{\beta}^2, N_{comp}-1)`, if not :math:`(iter, N_{comp}-1)`.
    Amm_iter: array_like
        Mixing matrix.

    """

    # Notes
    # -----
    # invN :
    #     .operands[0] = QUBIC, .operands[1] = Planck\
    #     .operands[0].operands[0] = ReshapeOperator (Ndet, Nsamples) ---> (Ndet*Nsamples), .operands[0].operands[1] = ReshapeOperator (Ndet*Nsamples) ---> (Ndet, Nsamples)\
    #     .operands[0].operands[0/1].operands[0] = 150 GHz focal plane , .operands[0].operands[0/1].operands[1] = 220 GHz focal plane\
    # H :
    #     .operands[0] = QUBIC / .operands[1] = Planck
    #     .operands[0].operands = Frequency Sub-Bands
    #     .operands[0].operands[0].operands[0] = 150 GHz / .operands[0].operands[0].operands[1] = 220 GHz\

    def __init__(
        self,
        seed_noise_qubic,
        seed_noise_planck,
        seed_start_pcg,
        preset_tools,
        preset_external,
        preset_qubic,
        preset_sky,
        preset_comp,
        preset_mixing_matrix,
        preset_gain,
    ):
        """
        Initialization.

        """

        ### Import preset Gain, Mixing Matrix, Foregrounds, Sky, QUBIC & tools
        self.preset_tools = preset_tools
        self.preset_external = preset_external
        self.preset_qubic = preset_qubic
        self.preset_sky = preset_sky
        self.preset_comp = preset_comp
        self.preset_mixingmatrix = preset_mixing_matrix
        self.preset_gain = preset_gain

        ### Set noise seeds + PCG start
        self.seed_noise_qubic = seed_noise_qubic
        self.seed_noise_planck = seed_noise_planck
        self.seed_start_pcg = seed_start_pcg

        ### Define tolerance of the rms variations
        self.ites_to_converge = self.preset_tools.params["PCG"]["ites_to_converge"]
        self.rms_plot = np.zeros((1, 2))
        self.convergence = []

        ### Inverse noise-covariance matrix
        self.preset_tools.mpi._print_message("    => Building inverse noise covariance matrix")
        self.invN = self.preset_qubic.joint_out.get_invntt_operator(
            self.preset_tools.params["QUBIC"]["NOISE"]["ndet"],
            self.preset_tools.params["QUBIC"]["NOISE"]["npho150"],
            self.preset_tools.params["QUBIC"]["NOISE"]["npho220"],
            self.preset_tools.params["PLANCK"]["level_noise_planck"],
            self.preset_sky.seenpix,
        )

        ### Get convolution
        self.preset_tools.mpi._print_message("    => Getting convolution")
        self.fwhm_qubic_tod, self.fwhm_qubic_mapmaking, self.fwhm_qubic_rec = self.get_convolution()
        n_planck_subs = (
            len(self.preset_external.external_nus)
            * self.preset_external.params_external["nsub_planck"]
        )
        conv_in = self.preset_qubic.params_qubic["convolution_in"]
        conv_out = self.preset_qubic.params_qubic["convolution_out"]

        if conv_in and conv_out:
            # Degrade Planck to QUBIC's finest beam; maps are reconstructed at min(allfwhm),
            # so the Planck reconstruction operator needs no additional convolution (fwhm=0).
            self.fwhm_planck_tod = np.full(n_planck_subs, self.fwhm_qubic_tod.min())
        elif conv_in and not conv_out:
            # Planck TOD is generated from components_in_convolved (GLS prediction), so no extra
            # beam is needed here — the reconstruction beam is already baked into components_in_convolved.
            self.fwhm_planck_tod = np.zeros(n_planck_subs)
        elif not conv_in and conv_out:
            # No beam applied in TOD generation; Planck gets no convolution either.
            self.fwhm_planck_tod = np.zeros(n_planck_subs)
        else:  # not conv_in and not conv_out
            self.fwhm_planck_tod = np.zeros(n_planck_subs)

        self.fwhm_planck_mapmaking = np.full(n_planck_subs, self.fwhm_qubic_mapmaking.min())

        print("fwhm_qubic_tod : ", self.fwhm_qubic_tod.shape)
        print("fwhm_planck_tod : ", self.fwhm_planck_tod.shape)

        self.fwhm_tod = np.concatenate((self.fwhm_qubic_tod, self.fwhm_planck_tod))
        self.fwhm_mapmaking = np.concatenate((self.fwhm_qubic_mapmaking, self.fwhm_planck_mapmaking))
        self.fwhm_rec = self.fwhm_qubic_rec

        ### Get observed data
        self.preset_tools.mpi._print_message("    => Getting observational data")
        self.get_tod()

        ### Compute initial guess for PCG
        self.preset_tools.mpi._print_message("    => Initializing starting point")
        self.get_x0()

    def _get_subband_weights(self):
        ndet = self.preset_qubic.params_qubic["NOISE"]["ndet"]
        npho150 = self.preset_qubic.params_qubic["NOISE"]["npho150"]
        npho220 = self.preset_qubic.params_qubic["NOISE"]["npho220"]
        is_uwb = self.preset_qubic.params_qubic["instrument"] == "UWB"
        allfwhm = self.preset_qubic.joint_in.qubic.allfwhm
        nsub_per_band = len(allfwhm) // 2

        if is_uwb:
            # UWB: all sub-bands share one focal plane with combined noise
            # sigma_combined² = sigma_det² + sigma_pho_150² + sigma_pho_220².
            # Uniform weights give standard (unweighted) LS component separation,
            # consistent with QubicInstrumentType.get_invntt_operator for UWB.
            sigma_combined_sq = max(ndet**2 + npho150**2 + npho220**2, 1e-30)
            return np.ones(len(allfwhm)) / sigma_combined_sq
        else:
            # DB: independent focal planes with band-specific noise.
            sigma_150_sq = max(ndet**2 + npho150**2, 1e-30)
            sigma_220_sq = max(ndet**2 + npho220**2, 1e-30)
            weights = np.zeros(len(allfwhm))
            weights[:nsub_per_band] = 1.0 / sigma_150_sq
            weights[nsub_per_band:] = 1.0 / sigma_220_sq
            return weights

    def _build_mixing_matrix(self, allnus):
        comp_names = self.preset_comp.components_name_out
        nsub = len(allnus)
        ncomp = len(comp_names)
        A = np.zeros((nsub, ncomp))
        for cidx, comp_name in enumerate(comp_names):
            if comp_name == "Dust":
                f = c.Dust(
                    nu0=self.preset_comp.params_foregrounds["Dust"]["nu0"],
                    beta_d=self.preset_comp.params_foregrounds["Dust"]["beta_init"][0],
                    temp=20,
                )
                A[:, cidx] = f.eval(allnus)
            elif comp_name == "Synchrotron":
                f = c.Synchrotron(
                    nu0=self.preset_comp.params_foregrounds["Synchrotron"]["nu0"],
                    beta_pl=self.preset_comp.params_foregrounds["Synchrotron"]["beta_init"][0],
                )
                A[:, cidx] = f.eval(allnus)
            else:
                A[:, cidx] = 1.0
        return A

    def _compute_gls_weights(self):
        """
        Returns W (ncomp, nsub): GLS component separation weights.
        W = (A^T N^{-1} A)^{-1} A^T N^{-1}
        W can have negative entries for sub-bands dominated by another component.
        """
        allnus = self.preset_qubic.joint_out.qubic.allnus
        noise_inv = self._get_subband_weights()
        A = self._build_mixing_matrix(allnus)
        N_inv_A = noise_inv[:, None] * A
        F = A.T @ N_inv_A
        return np.linalg.inv(F) @ N_inv_A.T

    def get_convolution(self):
        """Convolutions.

        Method to define all angular resolutions of the instrument at each frequency.

        This method sets the Full Width at Half Maximum (FWHM) for Time-Ordered Data (TOD) and map-making processes.
        `self.fwhm_qubic_tod` represents the real angular resolution, and `self.fwhm_qubic_mapmaking` represents the beams used for reconstruction.

        The method checks the `convolution_in` and `convolution_out` parameters to determine the appropriate FWHM values.
        It also calculates the reconstructed FWHM based on these parameters.

        Returns
        -------
        fwhm_qubic_tod: array_like
            Resolutions to create input TOD.
        fwhm_qubic_mapmaking: array_like
            Resolutions for map-making.
        fwhm_qubic_rec: array_like
            Resolutions of reconstructed maps.

        """

        nside = self.preset_tools.params["SKY"]["nside"]
        conv_in = self.preset_qubic.params_qubic["convolution_in"]
        conv_out = self.preset_qubic.params_qubic["convolution_out"]
        allfwhm = self.preset_qubic.joint_in.qubic.allfwhm

        fwhm_qubic_tod = allfwhm if conv_in else allfwhm * 0
        fwhm_qubic_mapmaking = (
            np.sqrt(allfwhm**2 - np.min(allfwhm) ** 2) if conv_out else allfwhm * 0
        )

        self.components_in_convolved = np.zeros(np.shape(self.preset_comp.components_out))

        if conv_in and conv_out:
            fwhm_qubic_rec = np.full(len(self.preset_comp.components_model_out), np.min(allfwhm))
            C = HealpixConvolutionGaussianOperator(np.min(fwhm_qubic_tod))
            for icomp, _ in enumerate(self.preset_comp.components_name_out):
                self.components_in_convolved[icomp] = C(self.preset_comp.components_in[icomp])

        elif conv_in and not conv_out:
            fwhm_rec_override = self.preset_qubic.params_qubic.get("fwhm_rec", None)
            fwhm_qubic_rec = np.zeros(len(self.preset_comp.components_model_out))
            if fwhm_rec_override is not None:
                fwhm_rec_list = fwhm_rec_override if isinstance(fwhm_rec_override, list) else [fwhm_rec_override] * len(self.preset_comp.components_name_out)
                for icomp in range(len(self.preset_comp.components_name_out)):
                    fwhm_comp = fwhm_rec_list[icomp]
                    fwhm_qubic_rec[icomp] = fwhm_comp
                    C = HealpixConvolutionGaussianOperator(fwhm=fwhm_comp, lmax=3 * nside - 1)
                    self.components_in_convolved[icomp] = C(self.preset_comp.components_in[icomp])
            else:
                # Full GLS prediction: components_in_convolved[i] = Σⱼ W[i,j] * Bⱼ(Σₖ A[j,k]*m_in[k])
                # Accounts for cross-component leakage (e.g. CMB bleeding into Dust I through
                # negative-weight 150 GHz sub-bands), which a single-component Gaussian cannot capture.
                allnus = self.preset_qubic.joint_out.qubic.allnus
                W = self._compute_gls_weights()  # (ncomp, nsub)
                A = self._build_mixing_matrix(allnus)  # (nsub, ncomp)
                ncomp = len(self.preset_comp.components_name_out)
                for jsub in range(len(allnus)):
                    sky_j = sum(
                        A[jsub, k] * self.preset_comp.components_in[k] for k in range(ncomp)
                    )
                    B_j = HealpixConvolutionGaussianOperator(fwhm=fwhm_qubic_tod[jsub], lmax=3 * nside - 1)
                    blurred_j = B_j(sky_j)
                    for icomp in range(ncomp):
                        self.components_in_convolved[icomp] += W[icomp, jsub] * blurred_j
                # Dominant positive-band FWHM per component — used for fwhm_planck_tod and logging only
                for icomp in range(ncomp):
                    pos = W[icomp] > 0
                    fwhm_qubic_rec[icomp] = np.sum(W[icomp, pos] * fwhm_qubic_tod[pos]) / np.sum(W[icomp, pos])

        elif not conv_in and conv_out:
            fwhm_qubic_rec = np.zeros(len(self.preset_comp.components_model_out))
            self.components_in_convolved = self.preset_comp.components_in.copy()

        else:
            fwhm_qubic_rec = np.zeros(len(self.preset_comp.components_model_out))
            self.components_in_convolved = self.preset_comp.components_in.copy()

        # Print the FWHM values
        self.preset_tools.mpi._print_message(f"FWHM for TOD making : {fwhm_qubic_tod}")
        self.preset_tools.mpi._print_message(f"FWHM for reconstruction : {fwhm_qubic_mapmaking}")
        self.preset_tools.mpi._print_message(f"Reconstructed FWHM : {fwhm_qubic_rec}")

        return fwhm_qubic_tod, fwhm_qubic_mapmaking, fwhm_qubic_rec

    def get_noise(self):
        """Noise.

        Method to define QUBIC noise, can generate Dual band or Wide Band noise by following:

        - Dual Band: n = [Ndet + Npho_150, Ndet + Npho_220]
        - Wide Band: n = [Ndet + Npho_150 + Npho_220]

        Depending on the instrument type specified in the preset_qubic parameters, this method will
        instantiate either QubicWideBandNoise or QubicDualBandNoise and return the total noise.

        Returns
        -------
        noise: array_like
            The total noise array, flattened.

        """

        noise = QubicTotNoise(
            self.preset_qubic.dict,
            self.preset_qubic.joint_out.qubic.sampling,
            self.preset_qubic.joint_out.qubic.scene,
        )

        return noise.total_noise(
            self.preset_qubic.params_qubic["NOISE"]["ndet"],
            self.preset_qubic.params_qubic["NOISE"]["npho150"],
            self.preset_qubic.params_qubic["NOISE"]["npho220"],
            seed_noise=self.seed_noise_qubic,
        ).ravel()

    def get_tod(self):
        r"""TOD.

        Generate fake observational data from QUBIC and external experiments.

        This method simulates observational data, including astrophysical foregrounds contamination using `self.beta` and systematics using `self.g`.
        The data generation follows the formula: :math:`\vec{d} = H.A.\vec{c} + \vec{n}`.

        Attributes
        ----------
        H: Operator
            Pointing matrix operator.
        TOD_qubic: array_like
            The simulated observational data for QUBIC.
        TOD_external: array_like
            The simulated observational data for external experiments.
        TOD_obs: array_like
            The combined observational data from QUBIC and external experiments.
        nsampling_x_ndetectors: int
            The number of samples in `self.TOD_qubic`.

        """

        ### Build joint acquisition operator
        print("Amm_in : ", self.preset_mixingmatrix.Amm_in.shape)
        self.H = self.preset_qubic.joint_in.get_operator(
            A=self.preset_mixingmatrix.Amm_in,
            gain=self.preset_gain.gain_in,
            fwhm=self.fwhm_tod,
        )

        ### Build noise variables
        noise_external = self.preset_qubic.joint_in.external.get_noise(
            planck_ntot=self.preset_tools.params["PLANCK"]["level_noise_planck"],
            weight_planck=self.preset_tools.params["PLANCK"]["weight_planck"],
            seenpix=self.preset_sky.seenpix,
            seed=self.seed_noise_planck,
        )

        noise_qubic = self.get_noise()

        ### Create QUBIC TOD — store signal part separately for the reference PCG
        self.TOD_qubic_signal = (self.H.operands[0])(self.preset_comp.components_in)
        self.TOD_qubic = self.TOD_qubic_signal + noise_qubic
        self.nsampling_x_ndetectors = self.TOD_qubic.shape[0]

        ### Create external TOD
        # For conv_in=True, conv_out=False: Planck TOD is generated from components_in_convolved
        # (the GLS prediction, beam already baked in) with fwhm_planck_tod=0. This makes inside-patch
        # and outside-patch Planck constraints point to the same reference, eliminating the resolution
        # discontinuity at the patch boundary. For other convolution modes, generate from raw m_in.
        conv_in = self.preset_qubic.params_qubic["convolution_in"]
        conv_out = self.preset_qubic.params_qubic["convolution_out"]
        planck_source = self.components_in_convolved if (conv_in and not conv_out) else self.preset_comp.components_in
        self.TOD_external = self.H.operands[1](planck_source) + noise_external.ravel()
        planck_source_zeroed = planck_source.copy()
        planck_source_zeroed[:, ~self.preset_sky.seenpix] = 0
        self.TOD_planck_signal_zeroed = self.H.operands[1](planck_source_zeroed)
        self.TOD_external_zero_outside_patch = self.TOD_planck_signal_zeroed + noise_external.ravel()

        #! Tom : Here, we are computing TOD from maps, then reshape to refound the maps, convolve the maps, and then reshape again to have the TOD... It is really dumb
        # _r = ReshapeOperator(self.TOD_external.shape, (len(self.preset_external.external_nus), 12 * self.preset_sky.params_sky["nside"] ** 2, 3))
        # maps_external = _r(self.TOD_external)

        # ### Reconvolve Planck data toward QUBIC angular resolution
        # #! Tom : correct this part, we  don't want to reconvolve here
        # if self.preset_qubic.params_qubic["convolution_in"] or self.preset_qubic.params_qubic["convolution_out"]:
        #     C = HealpixConvolutionGaussianOperator(
        #         fwhm=self.preset_qubic.joint_in.qubic.allfwhm[-1] * 0,
        #         lmax=3 * self.preset_sky.params_sky["nside"],
        #     )
        #     for i in range(maps_external.shape[0]):
        #         maps_external[i] = C(maps_external[i])

        # # if self.preset_tools.params["PCG"]["fix_pixels_outside_patch"]:
        # #     maps_external[:, ~self.preset_sky.seenpix_qubic, :] = 0
        # #     self.TOD_external = _r.T(maps_external)

        # #     self.seenpix_external = np.tile(self.preset_sky.seenpix_qubic, (maps_external.shape[0], 3, 1)).reshape(maps_external.shape)

        # ## Planck dataset with 0 outside QUBIC patch (Planck is assumed on the full sky)
        # maps_external[:, ~self.preset_sky.seenpix_qubic, :] = 0
        # self.TOD_external_zero_outside_patch = _r.T(maps_external)

        ### Observed TOD (Planck is assumed on the full sky)
        self.TOD_obs = np.r_[self.TOD_qubic, self.TOD_external]
        self.TOD_obs_zero_outside = np.r_[
            self.TOD_qubic, self.TOD_external_zero_outside_patch.ravel()
        ]
        # Signal-only concatenated TOD used by reference PCG to compute components_in_convolved
        self.TOD_obs_signal = np.r_[self.TOD_qubic_signal, self.TOD_planck_signal_zeroed]

    def get_x0(self):
        """PCG starting point.

        Define starting point of the convergence.

        The argument 'set_comp_to_0' multiplies the pixel values by a given factor. You can decide
        to convolve the map by a beam with an FWHM in radians.

        This method initializes the beta_iter and Amm_iter attributes based on the foreground model parameters.
        It also applies convolution and noise to the components based on the preset parameters.

        Raises
        ------
        TypeError
            If an unrecognized component name is encountered.

        """

        ### Fix random state
        np.random.seed(self.seed_start_pcg)

        self.beta_iter, self.Amm_iter = self.preset_mixingmatrix._get_beta_iter()

        # Build beta map for spatially varying spectral index
        self.allbeta = np.array([self.beta_iter])
        # C1 = [HealpixConvolutionGaussianOperator(fwhm=self.fwhm_qubic_rec[i], lmax=3 * self.preset_tools.params["SKY"]["nside"]) for i in range(len(self.preset_comp.components_model_out))]
        C2 = HealpixConvolutionGaussianOperator(
            fwhm=self.preset_tools.params["INITIAL"]["fwhm0"],
            lmax=3 * self.preset_tools.params["SKY"]["nside"] - 1,
        )
        # Constant spectral index -> maps have shape (Ncomp, Npix, Nstk)
        istk = 0
        mypix = self.preset_sky.seenpix

        for i, comp_name in enumerate(self.preset_comp.components_name_out):
            # self.preset_comp.components_iter[i] = C2(
            #     C1[i](self.preset_comp.components_iter[i]) # Two convolutions in a row?
            # )
            # self.preset_comp.components_iter[i] = C2(
            #     self.preset_comp.components_convolved_out[i]
            # )
            # self.preset_comp.components_iter[i] = C2(
            #     self.components_convolved_recon[i]
            # )
            self.preset_comp.components_iter[i] = C2(self.components_in_convolved[i])
            for istk in range(3):
                if istk == 0:
                    key = "I"
                else:
                    key = "P"

                initial_factor = (
                    self.preset_tools.params["INITIAL"][
                        "qubic_patch_{}_{}".format(key, comp_name[: min(4, len(comp_name))].lower())
                    ]
                    * self.preset_tools.params["INITIAL"]["global_{}".format(key)]
                )
                self.preset_comp.components_iter[i, mypix, istk] *= initial_factor
                # To make it more uniform, either name the components "cmb", "dust", "sync", "co" or the files "qubic_patch_I_CMB", "qubic_patch_I_Dust", "qubic_patch_I_Synchrotron", "qubic_patch_I_CO"
                self.preset_comp.components_iter[i, mypix, istk] += np.random.normal(
                    0,
                    self.preset_tools.params["INITIAL"]["sig_map_noise"],
                    self.preset_comp.components_iter[i, mypix, istk].shape,
                )

        # else:
        #     self.allbeta = np.array([self.beta_iter])
        #     C1 = HealpixConvolutionGaussianOperator(fwhm=self.fwhm_qubic_rec, lmax=3*self.preset_tools.params['SKY']['nside'])
        #     C2 = HealpixConvolutionGaussianOperator(fwhm=self.preset_tools.params['INITIAL']['fwhm0'], lmax=3*self.preset_tools.params['SKY']['nside'])
        #     ### Varying spectral indices -> maps have shape (Nstk, Npix, Ncomp)
        #     for i in range(len(self.preset_comp.components_model_out)):
        #         if self.preset_comp.components_name_out[i] == 'CMB':
        #             #print(self.preset_comp.components_iter.shape)
        #             self.preset_comp.components_iter[:, :, i] = C2(C1(self.preset_comp.components_iter[:, :, i].T)).T
        #             self.preset_comp.components_iter[1:, self.preset_sky.seenpix, i] *= self.preset_tools.params['INITIAL']['qubic_patch_cmb']

        #         elif self.preset_comp.components_name_out[i] == 'Dust':
        #             self.preset_comp.components_iter[:, :, i] = C2(C1(self.preset_comp.components_iter[:, :, i].T)).T + np.random.normal(0, self.preset_tools.params['INITIAL']['sig_map_noise'], self.preset_comp.components_iter[:, :, i].T.shape).T
        #             self.preset_comp.components_iter[1:, self.preset_sky.seenpix, i] *= self.preset_tools.params['INITIAL']['qubic_patch_dust']

        #         elif self.preset_comp.components_name_out[i] == 'Synchrotron':
        #             self.preset_comp.components_iter[:, :, i] = C2(C1(self.preset_comp.components_iter[:, :, i].T)).T + np.random.normal(0, self.preset_tools.params['INITIAL']['sig_map_noise'], self.preset_comp.components_iter[:, :, i].T.shape).T
        #             self.preset_comp.components_iter[1:, self.preset_sky.seenpix, i] *= self.preset_tools.params['INITIAL']['qubic_patch_sync']

        #         elif self.preset_comp.components_name_out[i] == 'CO':
        #             self.preset_comp.components_iter[:, :, i] = C2(C1(self.preset_comp.components_iter[:, :, i].T)).T + np.random.normal(0, self.preset_tools.params['INITIAL']['sig_map_noise'], self.preset_comp.components_iter[:, :, i].T.shape).T
        #             self.preset_comp.components_iter[1:, self.preset_sky.seenpix, i] *= self.preset_tools.params['INITIAL']['qubic_patch_co']
        #         else:
        #             raise TypeError(f'{self.preset_comp.components_name_out[i]} not recognize')
