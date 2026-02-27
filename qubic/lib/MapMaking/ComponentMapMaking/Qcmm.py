import os

import fgbuster.mixingmatrix as mm
import healpy as hp
import numpy as np
import yaml
from pyoperators import BlockDiagonalOperator, DiagonalOperator, PackOperator, ReshapeOperator
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

from qubic.lib.MapMaking.ComponentMapMaking.mixing_matrix.blindMM import BlindMM
from qubic.lib.MapMaking.ComponentMapMaking.mixing_matrix.mixedMM import MixedMM
from qubic.lib.MapMaking.ComponentMapMaking.mixing_matrix.parametricMM import ParametricMM
from qubic.lib.MapMaking.ComponentMapMaking.preset.preset import PresetInitialisation
from qubic.lib.MapMaking.Qcg import pcg
from qubic.lib.MapMaking.Qmap_plotter import PlotsCMM, plot_cross_spectrum
from qubic.lib.Qfoldertools import create_folder_if_not_exists, do_gif
from qubic.lib.Qhdf5 import HDF5Dict
from qubic.lib.Qspectra import Spectra

class PipelineComponentMapMaking:
    """
    Instance to reconstruct component maps using QUBIC abilities.

    Parameters
    ----------
    comm : MPI communicator
        MPI common communicator define by MPI.COMM_WORLD.
    seed : int
        Seed for random CMB realization.
    seed_noise : int, optional
        Seed for random noise realization, by default None.

    """

    def __init__(self, comm, parameters_file):
        """
        Initialize Pipeline instance.

        """

        # ### Creating noise seed
        # mpitools = MpiTools(comm)
        # seed_noise = mpitools.get_random_value(init_seed=None)

        ### Initialization
        self.preset = PresetInitialisation(comm).initialize(parameters_file)
        self.plots = PlotsCMM(self.preset, dogif=True)

        self.fsub = int(self.preset.qubic.joint_out.qubic.nsub / self.preset.comp.params_foregrounds["bin_mixing_matrix"])
        self.allAmm_iter = None

        ### Create variables for stopping condition
        self._rms_noise_qubic_patch_per_ite = np.empty((self.preset.tools.params["PCG"]["ites_to_converge"], len(self.preset.comp.components_name_out)))
        self._rms_noise_qubic_patch_per_ite[:] = np.nan

    def get_preconditioner(self):
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

        if not self.preset.tools.params["QUBIC"]["preconditioner"]:
            return None

        ncomp = len(self.preset.comp.components_model_out)
        nside = self.preset.sky.params_sky["nside"]
        npix = 12 * nside**2
        # nsub = self.preset_qubic.params_qubic["nsub_out"]
        # no_det = len(self.preset_qubic.joint_out.qubic.multiinstrument[0].detector)

        H_i = self.preset.qubic.joint_out.get_operator(A=self.preset.acquisition.Amm_iter, gain=self.preset.gain.gain_iter, fwhm=self.preset.acquisition.fwhm_mapmaking, nu_co=self.preset.comp.nu_co)

        # we only need the first element because for CMM the H.T H is almost flat!
        sky_shape = (ncomp, npix, 3)
        diagonal = np.zeros(sky_shape)
        for comp in range(sky_shape[0]):
            for pixel in range(1):
                for stokes in range(1):
                    basis_vector = np.zeros(sky_shape)
                    basis_vector[comp, pixel, stokes] = 1
                    Hv = H_i.T(H_i)(basis_vector)
                    diagonal[comp, pixel, stokes] = Hv[comp, pixel, stokes]

        stacked_matrix = np.array([np.full((npix, 3), diagonal[comp, 0, 0]) for comp in range(ncomp)])

        stacked_matrix_inv = 1.0 / stacked_matrix

        preconditioner_simpleinv = BlockDiagonalOperator([DiagonalOperator(comp, broadcast="rightward") for comp in stacked_matrix_inv], new_axisin=0)

        return preconditioner_simpleinv

    def call_pcg(self, max_iterations, seenpix):
        """Precontioned Conjugate Gradiant algorithm.

        Method to call the PCG from PyOperators package.

        Parameters
        ----------
        max_iterations : int
            Maximum number of iterations for the PCG algorithm.
        seenpix : array_like
            Boolean array that define the pixels observed by QUBIC.

        """

        if self._steps == 0:
            self.plots._display_allcomponents(
                input_maps=self.preset.acquisition.components_in_convolved, reconstructed_maps=self.preset.comp.components_iter, ki=-1, reso=self.preset.tools.params["PCG"]["reso_plot"]
            )

        ### Initialize PCG starting point
        w = self.preset.tools.params["PLANCK"]["weight_planck"]
        initial_maps = np.zeros_like(self.preset.comp.components_iter[:, seenpix, :])  # we should always start from zero, if you wish to start from Planck simply do weight_planck = 1.

        ### Update the preconditioner M
        self.preset.acquisition.M = self.get_preconditioner()

        ### Run PCG
        if self._steps == 0:
            num_iter = self.preset.tools.params["PCG"]["n_init_iter_pcg"]
            maxiter = self.preset.tools.params["PCG"]["n_init_iter_pcg"]
        else:
            num_iter = self.preset.tools.params["PCG"]["n_iter_pcg"]
            maxiter = max_iterations

        if self.preset.tools.params["PCG"]["do_gif"]:
            self.gif_folder = "CMM/" + self.preset.tools.params["foldername"] + "/Plots/maps_iter/"
        else:
            self.gif_folder = None

        ### PCG
        results = pcg(
            A=self.preset.A,
            b=self.preset.b,
            comm=self.preset.comm,
            x0=initial_maps,
            M=self.preset.acquisition.M,
            tol=self.preset.tools.params["PCG"]["tol_pcg"],
            disp=True,
            maxiter=maxiter,
            gif_folder=self.gif_folder,
            job_id=self.preset.job_id,
            seenpix=seenpix,
            seenpix_plot=self.preset.sky.seenpix,
            center=self.preset.sky.center,
            reso=self.preset.tools.params["PCG"]["reso_plot"],
            fwhm_plot=self.preset.tools.params["PCG"]["fwhm_plot"],
            input=self.preset.acquisition.components_in_convolved,
            iter_init=self._steps * num_iter,
            is_planck=True,
        )["x"]

        ### Update components
        self.preset.comp.components_iter[:, seenpix, :] = results["x"].copy() + w * self.preset.comp.components_out[:, seenpix, :].copy()

        self.preset.acquisition.convergence.append(results["convergence"].copy())
        ### Plot if asked
        if self.preset.tools.rank == 0:
            if self.preset.tools.params["PCG"]["do_gif"]:
                do_gif(
                    self.gif_folder,
                )
            self.plots.display_maps(input_maps=self.preset.acquisition.components_in_convolved, reconstructed_maps=self.preset.comp.components_iter, seenpix=seenpix, ki=self._steps)
            self.plots._display_allcomponents(
                input_maps=self.preset.acquisition.components_in_convolved,
                reconstructed_maps=self.preset.comp.components_iter,
                ki=self._steps,
                gif=self.preset.tools.params["PCG"]["do_gif"],
                reso=self.preset.tools.params["PCG"]["reso_plot"],
            )
            self.plots.plot_rms_iteration(self.preset.acquisition.rms_plot, ki=self._steps)

    def update_components(self, seenpix):
        r"""
        Method that solves the map-making equation :math:`(H^T . N^{-1} . H) . x = H^T . N^{-1} . d`, using OpenMP / MPI solver.

        This method updates the components of the map by solving the map-making equation using an OpenMP / MPI solver.
        The equation is of the form :math:`(H^T . N^{-1} . H) . \vec{c} = H^T . N^{-1} . \vec{TOD}`, where H_i is the operator obtained from the preset,
        U is a reshaped operator, and x_planck and xI are intermediate variables used in the calculations.


        Parameters
        ----------
        seenpix : array_like
            Boolean array that define the pixels observed by QUBIC.

        """
        H_i = self.preset.qubic.joint_out.get_operator(
            A=self.preset.acquisition.Amm_iter,
            gain=self.preset.gain.gain_iter,
            fwhm=self.preset.acquisition.fwhm_mapmaking,
            nu_co=self.preset.comp.nu_co,
        )

        U = (
            ReshapeOperator(
                (len(self.preset.comp.components_name_out) * sum(seenpix) * 3),
                (len(self.preset.comp.components_name_out), sum(seenpix), 3),
            )
            * PackOperator(
                np.broadcast_to(
                    seenpix[None, :, None],
                    (len(self.preset.comp.components_name_out), seenpix.size, 3),
                ).copy()
            )
        ).T

        ### Update components when pixels outside the patch are fixed (assumed to be 0)
        self.preset.A = U.T * H_i.T * self.preset.acquisition.invN * H_i * U

        w = self.preset.tools.params["PLANCK"]["weight_planck"]
        weight_mask = np.where(seenpix[None, :, None], w, 1.0)  # the 1.0 adds planck outside the patch, the weight_planck adds planck inside the patch
        x_planck_full = self.preset.comp.components_out * weight_mask

        self.preset.b = U.T(H_i.T * self.preset.acquisition.invN * (self.preset.acquisition.TOD_obs - H_i(x_planck_full)))

        ### Run PCG
        self.call_pcg(self.preset.tools.params["PCG"]["n_iter_pcg"], seenpix=seenpix)

    def get_tod_comp(self):
        """Component TOD.

        Method that produces Time-Ordered Data (TOD) using the component maps computed at the current iteration.

        This method initializes a zero-filled numpy array `tod_comp` with dimensions based on the number of components,
        the number of sub-components (multiplied by 2), and the product of the number of detectors and samples.
        It then iterates over each component and sub-component to compute the TOD by applying a convolution operator
        (if specified) and a mapping operator to the component maps.

        Returns
        -------
        tod_comp: array_like
            A numpy array containing the computed TOD for each component and sub-component.

        """

        tod_comp = np.zeros(
            (
                len(self.preset.comp.components_name_out),
                self.preset.qubic.joint_out.qubic.nsub,
                self.preset.qubic.joint_out.qubic.ndets * self.preset.qubic.joint_out.qubic.nsamples,
            )
        )

        #! raise ValueError("Tom : is it correct to use this H here ?")
        #! H should include FWHM mapmaking
        for i in range(len(self.preset.comp.components_name_out)):
            for j in range(self.preset.qubic.joint_out.qubic.nsub):
                tod_comp[i, j] = self.preset.qubic.joint_out.qubic.H[j](self.preset.comp.components_iter[i]).ravel()

        return tod_comp

    def get_constrains(self):
        """Constraints for scipy.minimize.

        Generate constraints readable by `scipy.optimize.minimize`.

        Returns
        -------
        constraints: list
            A list of constraint dictionaries for optimize.minimize.

        """

        constraints = []
        n = (self.preset.comp.params_foregrounds["bin_mixing_matrix"] - 1) * (len(self.preset.comp.components_name_out) - 1)

        ### Dust only : constraint ==> SED is increasing
        if self.preset.comp.params_foregrounds["Dust"]["Dust_out"] and not self.preset.comp.params_foregrounds["Synchrotron"]["Synchrotron_out"]:
            for i in range(n):
                constraints.append({"type": "ineq", "fun": lambda x, i=i: x[i + 1] - x[i]})

        ### Synchrotron only : constraint ==> SED is decreasing
        elif not self.preset.comp.params_foregrounds["Dust"]["Dust_out"] and self.preset.comp.params_foregrounds["Synchrotron"]["Synchrotron_out"]:
            for i in range(n):
                constraints.append({"type": "ineq", "fun": lambda x, i=i: x[i] - x[i + 1]})

        ### No component : constraint ==> None
        elif not self.preset.comp.params_foregrounds["Dust"]["Dust_out"] and not self.preset.comp.params_foregrounds["Synchrotron"]["Synchrotron_out"]:
            return None

        ### Dust & Synchrotron : constraint ==> SED is increasing for one component and decrasing for the other one
        elif self.preset.comp.params_foregrounds["Dust"]["Dust_out"] and self.preset.comp.params_foregrounds["Synchrotron"]["Synchrotron_out"]:
            for i in range(n):
                # Dust
                if i % 2 == 0:
                    constraints.append({"type": "ineq", "fun": lambda x, i=i: x[i + 2] - x[i]})
                # Sync
                else:
                    constraints.append({"type": "ineq", "fun": lambda x, i=i: x[i] - x[i + 2]})

        return constraints

    def get_tod_comp_superpixel(self, index):
        if self.preset.tools.rank == 0:
            print("Computing contribution of each super-pixel")
        _index = np.zeros(12 * self.preset.comp.params_foregrounds["Dust"]["nside_beta_out"] ** 2)
        _index[index] = index.copy()
        _index_nside = hp.ud_grade(_index, self.preset.qubic.joint_out.external.nside)
        tod_comp = np.zeros(
            (
                len(self.preset.comp.components_name_out),
                self.preset.qubic.joint_out.qubic.nsub,
                len(index),
                self.preset.qubic.joint_out.qubic.ndets * self.preset.qubic.joint_out.qubic.nsamples,
            )
        )

        maps_conv = self.preset.comp.components_iter.copy()

        for j in range(self.preset.qubic.params_qubic["nsub_out"]):
            for icomp in range(len(self.preset.comp.components_name_out)):
                C = HealpixConvolutionGaussianOperator(fwhm=self.preset.acquisition.fwhm_mapmaking[j], lmax=3 * self.preset.sky.params_sky["nside"] - 1)

                maps_conv[icomp] = C(self.preset.comp.components_iter[icomp, :, :]).copy()
                for ii, i in enumerate(index):
                    maps_conv_i = maps_conv.copy()
                    _i = _index_nside == i
                    for stk in range(3):
                        maps_conv_i[:, :, stk] *= _i
                    tod_comp[icomp, j, ii] = self.preset.qubic.joint_out.qubic.H[j](maps_conv_i[icomp]).ravel()
        return tod_comp

    def update_mixing_matrix(self, beta, previous_mixingmatrix, icomp):
        """Update Mixing Matrix.

        Method to update the mixing matrix using the current fitted value of the beta parameter and the parametric model associated.
        Only used when hybrid parametric-blind fit is selected !

        Parameters
        ----------
        beta : int
            Spectral index.
        previous_mixingmatrix : array_like
            Mixing Matrix at the previous iteration :math:`(N_{sub} + N_{integr} . N_{Planck}, N_{comp})`.
        icomp : int
            Component index.

        Returns
        -------
        updated_mixingmatrix: array_like
            The updated Mixing Matrix.

        """

        ### Build mixing matrix according to the choosen model and the beta parameter
        model_mixingmatrix = mm.MixingMatrix(*self.preset.comp.components_out).eval(self.preset.qubic.joint_out.qubic.allnus, *beta)

        ### Update the mixing matrix according to the one computed using the beta parameter
        updated_mixingmatrix = previous_mixingmatrix

        for ii in range(self.preset.comp.params_foregrounds["bin_mixing_matrix"]):
            updated_mixingmatrix[ii * self.fsub : (ii + 1) * self.fsub, icomp] = model_mixingmatrix[ii * self.fsub : (ii + 1) * self.fsub, icomp]

        return updated_mixingmatrix

    def get_mixing_matrix_method(self):
        """Decide whether to run `parametric`, `blind`, or `parametric_blind`."""
        comps = self.preset.comp.components_name_out

        # default: method of first foreground component (index 1 in components_name_out)
        method_0 = self.preset.comp.params_foregrounds[comps[1]]["type"]
        method = method_0

        # if any subsequent non-CO component has a different method -> parametric_blind
        if len(comps) > 1:
            for component in comps[2:]:
                if component != "CO" and self.preset.comp.params_foregrounds[component]["type"] != method_0:
                    return "parametric_blind"
        return method

    def fit_mixing_matrix(self):
        method = self.get_mixing_matrix_method()
        self.nfev = 0

        # d1 model
        if self.preset.comp.params_foregrounds["Dust"]["nside_beta_out"] != 0:
            index_num = hp.ud_grade(self.preset.sky.seenpix, self.preset.comp.params_foregrounds["Dust"]["nside_beta_out"])
            self.preset.mixingmatrix._index_seenpix_beta = np.where(index_num)[0]

            ### Simulated TOD for each components, nsub, npix with shape (npix, nsub, ncomp, nsnd)
            tod_comp = self.get_tod_comp_superpixel(self.preset.mixingmatrix._index_seenpix_beta)

            ### Store fixed beta (those denoted with hp.UNSEEN are variable)
            beta_map = self.preset.acquisition.beta_iter.copy()
            beta_map[:, self.preset.mixingmatrix._index_seenpix_beta] = hp.UNSEEN
        # d0 or d6 model
        else:
            tod_comp = self.get_tod_comp()
            self.preset.mixingmatrix._index_seenpix_beta = 0

        if method == "parametric":
            updater = ParametricMM(self)
        elif method == "blind":
            updater = BlindMM(self)
        elif method == "parametric_blind":
            updater = MixedMM(self)
        else:
            raise TypeError(f"Unknown method {method}")

        if self.preset.comp.params_foregrounds["Dust"]["model"] == "d1":
            updater.update(tod_comp, beta_map)
        else:
            updater.update(tod_comp)

    def give_intercal(self, D, d, _invn):
        r"""Detectors intercalibration.

        Semi-analytical method for gains estimation. (cf CMM paper)

        Parameters
        ----------
        D : array_like
            Simlulated data, given by the formula : :math:`\vec{D} = H.A.\vec{c}`,
            where H is the pointing matrix, A the mixing matrix and c the component vector.

        d : array_like
            Observed data, given by the formula : :math:`\vec{d} = G.\vec{D} + \vec{n}`,
            where G is the detectors' intercalibration matrix and n the noise vector.
        _invn : Diagonal Operator
            Inverse noise covariance matrix.

        Returns
        -------
        g: array_like
            Intercalibration vector.

        """

        _r = ReshapeOperator(
            self.preset.qubic.joint_out.qubic.ndets * self.preset.qubic.joint_out.qubic.nsamples,
            (
                self.preset.qubic.joint_out.qubic.ndets,
                self.preset.qubic.joint_out.qubic.nsamples,
            ),
        )

        return (1 / np.sum(_r(D) * _invn(_r(D)), axis=1)) * np.sum(_r(D) * _invn(_r(d)), axis=1)

    def update_gain(self):
        r"""Update detectors gain.

        Method that compute and print gains of each detectors using semi-analytical method decribed in "give_intercal" function,
        using the formula : :math:`g^i = \frac{TOD_{obs}^i}{TOD_{sim}^i}`.

        """

        raise ValueError("The method Pipeline.update_gain is broken.")

        # self.H_i = self.preset.qubic.joint_out.get_operator(  # Amm is now called A
        #     self.preset.acquisition.beta_iter,
        #     Amm=self.preset.acquisition.Amm_iter,
        #     gain=np.ones(self.preset.gain.gain_iter.shape),
        #     fwhm=self.preset.acquisition.fwhm_mapmaking,
        #     nu_co=self.preset.comp.nu_co,
        # )
        # self.nsampling = self.preset.qubic.joint_out.qubic.nsamples
        # self.ndets = self.preset.qubic.joint_out.qubic.ndets

        # # When (if) rewritten, the code will not have conditions on self.preset.qubic.params_qubic["instrument"] value
        # # Also, the shapes of inv, H_i and TOD were modified

        # if self.preset.qubic.params_qubic["instrument"] == "UWB":  # rewrite this?
        #     _r = ReshapeOperator(
        #         self.preset.qubic.joint_out.qubic.ndets * self.preset.qubic.joint_out.qubic.nsamples,
        #         (self.preset.qubic.joint_out.qubic.ndets, self.preset.qubic.joint_out.qubic.nsamples),
        #     )

        #     TODi_Q = self.preset.acquisition.invN.operands[0](self.H_i.operands[0](self.preset.comp.components_iter)[: self.ndets * self.nsampling])
        #     print("invN", self.preset.acquisition.invN.operands[0].operands[1])
        #     print("reshape", _r.shapein, _r.shapeout)
        #     print("invN shape", self.preset.acquisition.invN.operands[0].operands[1].shapein, self.preset.acquisition.invN.operands[0].operands[1].shapeout)
        #     print("TODi_Q", TODi_Q.shape)
        #     self.preset.gain.gain_iter = self.give_intercal(TODi_Q, _r(self.preset.acquisition.TOD_qubic), self.preset.acquisition.invN.operands[0].operands[1])
        #     self.preset.gain.gain_iter /= self.preset.gain.gain_iter[0]
        #     self.preset.gain.all_gain = np.concatenate((self.preset.gain.all_gain, np.array([self.preset.gain.gain_iter])), axis=0)

        # elif self.preset.qubic.params_qubic["instrument"] == "DB":
        #     TODi_Q_150 = self.H_i.operands[0](self.preset.comp.components_iter)[: self.ndets * self.nsampling]
        #     TODi_Q_220 = self.H_i.operands[0](self.preset.comp.components_iter)[self.ndets * self.nsampling : 2 * self.ndets * self.nsampling]

        #     g150 = self.give_intercal(
        #         TODi_Q_150,
        #         self.preset.acquisition.TOD_qubic[: self.ndets * self.nsampling],
        #         self.preset.acquisition.invN.operands[0].operands[1].operands[0],
        #     )
        #     g220 = self.give_intercal(
        #         TODi_Q_220,
        #         self.preset.acquisition.TOD_qubic[self.ndets * self.nsampling : 2 * self.ndets * self.nsampling],
        #         self.preset.acquisition.invN.operands[0].operands[1].operands[1],
        #     )

        #     self.preset.gain.gain_iter = np.array([g150, g220]).T
        #     self.preset.Gi = join_data(self.preset.tools.comm, self.preset.gain.gain_iter)
        #     print("gain_iter", self.preset.gain.gain_iter.shape, self.preset.Gi.shape)
        #     print("all_gain", self.preset.gain.all_gain.shape)
        #     print("all_gain_in", self.preset.gain.all_gain_in.shape)
        #     self.preset.gain.all_gain = np.concatenate((self.preset.gain.all_gain, np.array(self.preset.gain.gain_iter)), axis=0)

        #     print("all_gain", self.preset.gain.all_gain.shape)
        #     if self.preset.tools.rank == 0:
        #         print(np.mean(self.preset.gain.gain_iter - self.preset.gain.gain_in, axis=0))
        #         print(np.std(self.preset.gain.gain_iter - self.preset.gain.gain_in, axis=0))

        # self.plots.plot_gain_iteration(
        #     self.preset.gain.all_gain - self.preset.gain.all_gain_in, alpha=0.03, ki=self._steps
        # )

    def save_data(self, step):
        """Save data.

        Method that save data for each iterations.
        It saves components, gains, spectral index, coverage, seen pixels.

        Parameters
        ----------
        step : int
            Step number.

        """

        if self.preset.tools.rank == 0:
            if self.preset.tools.params["save_iter"] != 0:
                if (step + 1) % self.preset.tools.params["save_iter"] == 0:
                    if self.preset.tools.params["lastite"]:
                        if step != 0:
                            os.remove("CMM/" + self.preset.tools.params["foldername"] + "/Dict/" + self.preset.tools.params["filename"] + f"_{str(self.preset.job_id)}.h5")
                        dictionary = {
                            "maps_in": self.preset.comp.components_in,
                            "maps_in_convolved": self.preset.acquisition.components_in_convolved,
                            "maps": self.preset.comp.components_iter,
                            "maps_noise": self.preset.acquisition.components_in_convolved - self.preset.comp.components_iter,
                            "comps_name": self.preset.comp.components_name_out,
                            "beta": self.preset.acquisition.allbeta,
                            "beta_true": self.preset.mixingmatrix.beta_in,
                            "index_beta": self.preset.mixingmatrix._index_seenpix_beta,
                            "g": self.preset.gain.all_gain_in,
                            "gi": self.preset.gain.all_gain,
                            "all_gain": self.preset.gain.all_gain_iter,
                            "A": self.preset.acquisition.Amm_iter,
                            "Atrue": self.preset.mixingmatrix.Amm_in,
                            "G": self.preset.gain.all_gain_in,
                            "nus_in": self.preset.mixingmatrix.nus_eff_in,
                            "nus_out": self.preset.mixingmatrix.nus_eff_out,
                            "center": self.preset.sky.center,
                            "coverage": self.preset.sky.coverage,
                            "seenpix": self.preset.sky.seenpix,
                            "fsky": self.preset.sky.fsky,
                            "fwhm_in": self.preset.acquisition.fwhm_tod,
                            "fwhm_out": self.preset.acquisition.fwhm_mapmaking,
                            "fwhm_rec": self.preset.acquisition.fwhm_rec,
                            "parameters": self.preset.tools.params,
                            "convergence": self.preset.acquisition.convergence,
                            "TOD_qubic": self.preset.acquisition.TOD_qubic,
                            "TOD_external": self.preset.acquisition.TOD_external,
                            "qubic_dict": {k: v for k, v in self.preset.qubic.dict.items() if k != "comm"},  # Need to remove the MPI communictor, which is not suppurted by pickle
                        }
                        HDF5Dict().save_dict("CMM/" + self.preset.tools.params["foldername"] + "/Dict/" + self.preset.tools.params["filename"] + f"_{str(self.preset.job_id)}.h5", dictionary)

    def _stop_condition(self):
        """
        Method that stop the convergence if there are more than k steps.

        """

        if self._steps >= self.preset.tools.params["PCG"]["n_iter_loop"] - 1:
            self._info = False

        self._steps += 1

    def run(self):
        """Pipeline.

        Method to run the pipeline by following :

            1) Initialize simulation using `PresetSims` instance reading `params.yml`.

            2) Solve map-making equation knowing spectral index and gains.

            3) Fit spectral index knowing components and gains.

            4) Fit gains knowing components and sepctral index.

            5) Repeat 2), 3) and 4) until convergence.

        """

        self._info = True
        self._steps = 0
        while self._info:
            ### Display iteration
            self.preset.tools._display_iter(self._steps)

            ### Update self.fg.components_iter^{k} -> self.fg.components_iter^{k+1}
            self.update_components(seenpix=self.preset.sky.seenpix)

            ### Update self.preset.acquisition.beta_iter^{k} -> self.preset.acquisition.beta_iter^{k+1}
            if self.preset.comp.params_foregrounds["fit_mixing_matrix"]:
                self.fit_mixing_matrix()

            ### Update self.gain.gain_iter^{k} -> self.gain.gain_iter^{k+1}
            if self.preset.qubic.params_qubic["GAIN"]["fit_gain"]:
                self.update_gain()

            ### Wait for all processes and save data inside pickle file
            self.preset.tools.comm.Barrier()
            self.save_data(self._steps)

            ### Compute the rms of the noise per iteration to later analyze its convergence in _stop_condition
            # self._compute_maxrms_array()

            ### Stop the loop when self._steps > k
            self._stop_condition()


class PipelineEnd2End:
    """
    CMM Pipeline.

    Wrapper for End-to-End pipeline.
    """

    def __init__(self, comm, parameters_path):
        self.comm = comm
        self.parameters_path = parameters_path
        with open(self.parameters_path, "r") as tf:
            self.params = yaml.safe_load(tf)

        self.job_id = os.environ.get("SLURM_JOB_ID")

        self.folder = (
            "CMM/" + f"{self.params['Foregrounds']['Dust']['type']}_{self.params['Foregrounds']['Dust']['model']}_{self.params['QUBIC']['instrument']}_" + self.params["foldername"] + "/Dict/"
        )
        self.file = self.folder + self.params["filename"] + f"_{self.job_id}.h5"
        self.file_spectrum = (
            "CMM/"
            + f"{self.params['Foregrounds']['Dust']['type']}_{self.params['Foregrounds']['Dust']['model']}_{self.params['QUBIC']['instrument']}_"
            + self.params["foldername"]
            + "/Spectrum/"
            + "spectrum_"
            + self.params["filename"]
            + f"_{self.job_id}.h5"
        )

        self.mapmaking = None

    def main(self, specific_file=None):
        if self.params["Pipeline"]["mapmaking"]:
            self.mapmaking = PipelineComponentMapMaking(self.comm, self.parameters_path)

            self.mapmaking.run()

        if self.params["Pipeline"]["spectrum"]:
            if self.comm.Get_rank() == 0:
                create_folder_if_not_exists(
                    self.comm,
                    "CMM/"
                    + f"{self.params['Foregrounds']['Dust']['type']}_{self.params['Foregrounds']['Dust']['model']}_{self.params['QUBIC']['instrument']}_"
                    + self.params["foldername"]
                    + "/Spectrum/",
                )

            if self.mapmaking is not None:
                self.spectrum = Spectra(self.file)
            else:
                self.spectrum = Spectra(specific_file)

            ### Signal
            print("\n===============================================")
            print("========= Cross-spectra with Sky =============")
            print("===============================================\n")
            DlBB_maps = self.spectrum.run(maps=self.spectrum.maps)

            ### Noise
            print("\n===============================================")
            print("========= Cross-spectra with Residual =========")
            print("===============================================\n")
            DlBB_noise = self.spectrum.run(maps=self.spectrum.dictionary["maps_noise"])

            dict_solution = {
                "comp": self.spectrum.dictionary["comps_name"],
                "ell": self.spectrum.ell,
                "Dls": DlBB_maps,
                "Nls": DlBB_noise,
                "parameters": self.params,
                "delta_ell": self.params["Spectrum"]["dl"],
                "fsky": self.spectrum.dictionary["fsky"],
            }

            if self.params["Spectrum"]["plot_spectrum"]:
                create_folder_if_not_exists(
                    self.comm,
                    "CMM/"
                    + f"{self.params['Foregrounds']['Dust']['type']}_{self.params['Foregrounds']['Dust']['model']}_{self.params['QUBIC']['instrument']}_"
                    + self.params["foldername"]
                    + "/Spectrum/Plots/",
                )
                ### QUBIC only plots
                N = len(self.spectrum.dictionary["comps_name"])
                plot_cross_spectrum(
                    nus=self.spectrum.dictionary["comps_name"],
                    ell=self.spectrum.ell,
                    Dl=DlBB_maps,
                    Dl_err=DlBB_noise,
                    ymodel=None,
                    nrec=N,
                    figsize=(30, 30),
                    name="CMM/"
                    + f"{self.params['Foregrounds']['Dust']['type']}_{self.params['Foregrounds']['Dust']['model']}_{self.params['QUBIC']['instrument']}_"
                    + self.params["foldername"]
                    + "/Spectrum/Plots/"
                    + f"QUBIC_{self.job_id}.svg",
                )
            HDF5Dict().save_dict(self.file_spectrum, dict_solution)
