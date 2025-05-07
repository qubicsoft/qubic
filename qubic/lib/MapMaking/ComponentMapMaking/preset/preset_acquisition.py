import numpy as np
from pyoperators import DiagonalOperator

from ....Instrument.Qacquisition import *
from ....Instrument.Qnoise import QubicTotNoise
from fgbuster import component_model as c

#from qubic.lib.Instrument.Qacquisition import *
#from qubic.lib.Instrument.Qnoise import *
#import qubic.lib.MapMaking.ComponentMapMaking.Qcomponent_model as c


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
    rms_tolerance: float
        Tolerance for PCG algorithm.
    ites_to_converge: int
        Max iteration number for PCG algorithm.
    invN: BlockDiagonalOperator
        Inverse noise covariance matrix with input shape :math:`(N_{det}*N_{samples} + N_{pix}*N_{planck}*N_{stokes})` and output shape :math:`(N_{det}*N_{samples} + N_{pix}*N_{planck}*N_{stokes})`.
    M: DiagonalOperator
        Preconditioner for PCG algorithm.
    fwhm_rec: array_like
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
        self.rms_tolerance = self.preset_tools.params["PCG"]["tol_rms"]
        self.ites_to_converge = self.preset_tools.params["PCG"]["ites_to_converge"]
        self.rms_plot = np.zeros((1, 2))

        ### Inverse noise-covariance matrix
        self.preset_tools.mpi._print_message(
            "    => Building inverse noise covariance matrix"
        )
        self.invN = self.preset_qubic.joint_out.get_invntt_operator(
            mask=self.preset_sky.mask
        )

        ### Preconditioner
        #self.preset_tools._print_message("    => Creating preconditioner")
        #self.M = self.get_preconditioner(
        #    A_qubic=self.preset_mixingmatrix.Amm_in[
        #        :6#self.preset_qubic.params_qubic["nsub_out"]
        #    ],
        #    A_ext=self.preset_mixingmatrix.Amm_in[
        #        7:#self.preset_qubic.params_qubic["nsub_out"]:
        #    ],
        #    precond=self.preset_qubic.params_qubic["preconditioner"],
        #    thr=self.preset_tools.params["PLANCK"]["thr_planck"],
        #)

        ### Get convolution
        self.preset_tools.mpi._print_message("    => Getting convolution")
        self.fwhm_tod, self.fwhm_mapmaking, self.fwhm_rec = self.get_convolution()

        ### Get observed data
        self.preset_tools.mpi._print_message("    => Getting observational data")
        self.get_tod()

        ### Compute initial guess for PCG
        self.preset_tools.mpi._print_message("    => Initializing starting point")
        self.get_x0()

    def get_approx_hth(self):
        r"""Approx HTH.

        Calculates the approximation of :math:`H^T.H`.

        Returns
        -------
        approx_hth: array_like
            The approximation of :math:`H^T.H` with shape :math:`(N_{sub}^{in}, N_{pixel}, 3)`.
        approx_hth_ext: array_like
            The approximation of :math:`H^T.H` for Planck.

        """

        # Approximation of H.T H
        approx_hth = np.empty(
            (len(self.preset_qubic.joint_out.qubic.H),)
            + self.preset_qubic.joint_out.qubic.H[0].shapein
        )
        vector = np.ones(self.preset_qubic.joint_out.qubic.H[0].shapein)
        for index in range(self.preset_qubic.params_qubic["nsub_out"]):
            approx_hth[index] = (
                self.preset_qubic.joint_out.qubic.H[index].T
                * self.preset_qubic.joint_out.qubic.invn150 # why invn150?
                * self.preset_qubic.joint_out.qubic.H[index](vector)
            )

        invN_ext = self.preset_qubic.joint_out.external.get_invntt_operator(
            mask=self.preset_sky.mask
        )

        _r = ReshapeOperator(
            (
                len(self.preset_qubic.joint_out.external.nus),
                approx_hth.shape[1],
                approx_hth.shape[2],
            ),
            invN_ext.shapein,
        )
        approx_hth_ext = invN_ext(np.ones(invN_ext.shapein))

        return approx_hth, _r.T(approx_hth_ext)

    def get_preconditioner(self, seenpix, A_qubic, A_ext, precond=True, thr=0):
        """Preconditioner for PCG algorithm.

        Calculates and returns the preconditioner matrix for the optimization process.

        Parameters
        ----------
        A_qubic : array_like
            QUBIC mixing matrix.
        A_ext : array_like
            Planck mixing matrix.
        precond : bool, optional
            Tells if you want a precontioner or not, by default True
        thr : int, optional
            Threshold to define the pixels seen by QUBIC, by default 0

        Returns
        -------
        M: DiagonalOperator
            Preconditioner for PCG algorithm.

        """

        if precond:

            # Calculate the approximate H^T * H matrix
            approx_hth, approx_hth_ext = self.get_approx_hth()

            # Create a preconditioner matrix with dimensions (number of components, number of pixels, 3)
            preconditioner = np.ones(
                (
                    len(self.preset_comp.components_model_out),
                    approx_hth.shape[1],
                    approx_hth.shape[2],
                )
            )

            for icomp in range(len(self.preset_comp.components_model_out)):
                for istk in range(3):
                    precond_ext = 1 / (approx_hth_ext[:, :, 0].T @ A_ext[..., icomp] ** 2)
                    precond_ext[precond_ext == np.inf] = 0
                    preconditioner[icomp, :, istk] = precond_ext
                    preconditioner[icomp, seenpix, istk] *= self.preset_tools.params["PLANCK"]["weight_planck"]

                    # We sum over the frequencies, take the inverse, and only keep the information on the patch.
                    precond_qubic = 1 / (approx_hth[:, :, 0].T @ (A_qubic[..., icomp]) ** 2)
                    preconditioner[icomp, self.preset_sky.seenpix_qubic, istk] += precond_qubic[self.preset_sky.seenpix_qubic] 

            M = DiagonalOperator(preconditioner[:, seenpix, :])
            return M
        else:
            return None

    def _get_scalar_acquisition_operator(self):
        """
        Function that will compute "scalar acquisition operatord" by applying the acquisition operators to a vector full of ones.
        These scalar operators will be used to compute the resolutions in the case where we do not add convolutions during reconstruction.
        """
        ### Import the acquisition operators
        acquisition_operators = self.preset_qubic.joint_out.qubic.H

        ### Create the vector full of ones which will be used to compute the scalar operators
        vector_ones = np.ones(acquisition_operators[0].shapein)

        ### Apply each sub_operator on the vector
        scalar_acquisition_operators = np.empty(
            len(self.preset_qubic.joint_out.qubic.allnus)
        )
        for freq in range(len(self.preset_qubic.joint_out.qubic.allnus)):
            scalar_acquisition_operators[freq] = np.mean(
                acquisition_operators[freq](vector_ones)
            )
        return scalar_acquisition_operators

    def get_convolution(self):
        """Convolutions.

        Method to define all angular resolutions of the instrument at each frequency.

        This method sets the Full Width at Half Maximum (FWHM) for Time-Ordered Data (TOD) and map-making processes.
        `self.fwhm_tod` represents the real angular resolution, and `self.fwhm_mapmaking` represents the beams used for reconstruction.

        The method checks the `convolution_in` and `convolution_out` parameters to determine the appropriate FWHM values.
        It also calculates the reconstructed FWHM based on these parameters.

        Returns
        -------
        fwhm_tod: array_like
            Resolutions to create input TOD.
        fwhm_mapmaking: array_like
            Resolutions for map-making.
        fwhm_rec: array_like
            Resolutions of reconstructed maps.

        """

        # Initialize FWHM arrays to 0
        fwhm_tod = self.preset_qubic.joint_in.qubic.allfwhm * 0
        fwhm_mapmaking = self.preset_qubic.joint_in.qubic.allfwhm * 0

        # Check if convolution_in is True
        if self.preset_qubic.params_qubic["convolution_in"]:
            fwhm_tod = self.preset_qubic.joint_in.qubic.allfwhm

        # Check if convolution_out is True
        if self.preset_qubic.params_qubic["convolution_out"]:
            fwhm_mapmaking = np.sqrt(
                self.preset_qubic.joint_in.qubic.allfwhm**2
                - np.min(self.preset_qubic.joint_in.qubic.allfwhm) ** 2
            )

        # Calculate the reconstructed FWHM based on convolution parameters
        if (
            self.preset_qubic.params_qubic["convolution_in"]
            and self.preset_qubic.params_qubic["convolution_out"]
        ):
            fwhm_rec = np.min(self.preset_qubic.joint_in.qubic.allfwhm) # min of allfwhm?
        elif (
            not self.preset_qubic.params_qubic["convolution_in"]
            and not self.preset_qubic.params_qubic["convolution_out"]
        ):
            fwhm_rec = np.mean(self.preset_qubic.joint_in.qubic.allfwhm) * 0
        elif (
            self.preset_qubic.params_qubic["convolution_in"]
            and not self.preset_qubic.params_qubic["convolution_out"]
        ):
            scalar_acquisition_operators = self._get_scalar_acquisition_operator()
            fwhm_rec = np.zeros(len(self.preset_comp.components_model_out))
            for comp, comp_name in enumerate(self.preset_comp.components_name_out):
                if comp_name == "CMB":
                    factor = scalar_acquisition_operators
                elif comp_name == "Dust":
                    f_dust = c.Dust(
                        nu0=self.preset_comp.params_foregrounds["Dust"]["nu0"],
                        beta_d=self.preset_comp.params_foregrounds["Dust"]["beta_init"][0],
                        temp=20
                    )
                    factor = (scalar_acquisition_operators
                        * f_dust.eval(self.preset_qubic.joint_out.qubic.allnus))
                elif self.preset_comp.components_name_out[comp] == "Synchrotron":
                    f_sync = c.Synchrotron(
                        nu0=self.preset_comp.params_foregrounds["Synchrotron"]["nu0"],
                        beta_pl=self.preset_comp.params_foregrounds["Synchrotron"]["beta_init"][0],
                    )
                    factor = (scalar_acquisition_operators
                        * f_sync.eval(self.preset_qubic.joint_out.qubic.allnus))
                fwhm_rec[comp] = np.sum(factor * fwhm_tod) / (np.sum(factor))

        elif (
            not self.preset_qubic.params_qubic["convolution_in"]
            and not self.preset_qubic.params_qubic["convolution_out"]
        ):
            fwhm_rec = np.zeros(len(self.preset_comp.components_model_out))

        # Print the FWHM values
        self.preset_tools.mpi._print_message(f"FWHM for TOD making : {fwhm_tod}")
        self.preset_tools.mpi._print_message(f"FWHM for reconstruction : {fwhm_mapmaking}")
        self.preset_tools.mpi._print_message(f"Reconstructed FWHM : {fwhm_rec}")

        return fwhm_tod, fwhm_mapmaking, fwhm_rec
    
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
                duration=[
                        self.preset_qubic.params_qubic["NOISE"]["duration_150"],
                        self.preset_qubic.params_qubic["NOISE"]["duration_220"],
                    ],
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
        self.H = self.preset_qubic.joint_in.get_operator(
            A=self.preset_mixingmatrix.Amm_in,
            gain=self.preset_gain.gain_in,
            fwhm=self.fwhm_tod,
        )

        ### Build noise variables
        noise_external = (
            self.preset_qubic.joint_in.external.get_noise(seed=self.seed_noise_planck)
            * self.preset_tools.params["PLANCK"]["level_noise_planck"]
        )
        noise_qubic = self.get_noise()

        ### Create QUBIC TOD
        self.TOD_qubic = (self.H.operands[0])(
            self.preset_comp.components_in
        ) + noise_qubic
        self.nsampling_x_ndetectors = self.TOD_qubic.shape[0]

        ### Create external TOD
        self.TOD_external = (self.H.operands[1])(
            self.preset_comp.components_in[:, :, :]
        ) + noise_external

        _r = ReshapeOperator(
            self.TOD_external.shape,
            (
                len(self.preset_external.external_nus),
                12 * self.preset_sky.params_sky["nside"] ** 2,
                3,
            ),
        )
        maps_external = _r(self.TOD_external)

        ### Reconvolve Planck data toward QUBIC angular resolution
        if (
            self.preset_qubic.params_qubic["convolution_in"]
            or self.preset_qubic.params_qubic["convolution_out"]
        ):

            C = HealpixConvolutionGaussianOperator(
                fwhm=self.preset_qubic.joint_in.qubic.allfwhm[-1], # or put the computed fwhm?
                lmax=3*self.preset_sky.params_sky["nside"] - 1, #Max useful is 3*nside - 1
            )
            for i in range(maps_external.shape[0]):
                maps_external[i] = C(maps_external[i])

        # if self.preset_tools.params['PCG']['fix_pixels_outside_patch']:
        # maps_external[:, ~self.preset_sky.seenpix_qubic, :] = 0
        self.TOD_external = _r.T(maps_external)

        # self.seenpix_external = np.tile(self.preset_sky.seenpix_qubic, (maps_external.shape[0], 3, 1)).reshape(maps_external.shape)

        ### Planck dataset with 0 outside QUBIC patch (Planck is assumed on the full sky)
        maps_external[:, ~self.preset_sky.seenpix_qubic, :] = 0
        self.TOD_external_zero_outside_patch = _r.T(maps_external)

        ### Observed TOD (Planck is assumed on the full sky)
        self.TOD_obs = np.r_[self.TOD_qubic, self.TOD_external]
        self.TOD_obs_zero_outside = np.r_[self.TOD_qubic, self.TOD_external_zero_outside_patch]

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
        C1 = [HealpixConvolutionGaussianOperator(
            fwhm=self.fwhm_rec[i],
            lmax=3*self.preset_tools.params["SKY"]["nside"] - 1) for i in range(len(self.preset_comp.components_model_out))]
        C2 = HealpixConvolutionGaussianOperator(
            fwhm=self.preset_tools.params["INITIAL"]["fwhm0"],
            lmax=3*self.preset_tools.params["SKY"]["nside"] - 1,
        )
        # Constant spectral index -> maps have shape (Ncomp, Npix, Nstk)
        istk = 0
        mypix = self.preset_sky.seenpix
        for i, comp_name in enumerate(self.preset_comp.components_name_out):
            self.preset_comp.components_iter[i] = C2(
                C1[i](self.preset_comp.components_iter[i]) # Two convolutions in a row?
            )
            for istk in range(3):
                if istk == 0:
                    key = "I"
                else:
                    key = "P"
                self.preset_comp.components_iter[
                    i, mypix, istk
                ] *= self.preset_tools.params["INITIAL"]["qubic_patch_{}_{}".format(key, comp_name[:min(4, len(comp_name))].lower())]
                # To make it more uniform, either name the components "cmb", "dust", "sync", "co" or the files "qubic_patch_I_CMB", "qubic_patch_I_Dust", "qubic_patch_I_Synchrotron", "qubic_patch_I_CO"
                self.preset_comp.components_iter[
                    i, mypix, istk
                ] += np.random.normal(
                    0,
                    self.preset_tools.params["INITIAL"]["sig_map_noise"],
                    self.preset_comp.components_iter[i, mypix, istk].shape,
                )
            # raise TypeError(
            #     f"{self.preset_comp.components_name_out[i]} not recognized"
            # )