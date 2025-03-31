### General packages
import gc
import os
import sys
import pickle
from functools import partial

import healpy as hp
import numpy as np
from pyoperators import MPI
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
from scipy.optimize import fmin_l_bfgs_b, minimize

import fgbuster.mixingmatrix as mm

### Local packages
from ...Instrument.Qacquisition import *
from ...MapMaking.Qcg import pcg
from pyoperators import pcg as pcg_op
from ...Qfoldertools import *
from ...MapMaking.Qmap_plotter import *
from ...Qmpi_tools import MpiTools

# from simtools.mpi_tools import *
from ...Instrument.Qnoise import *

from .Qcostfunc import (
    Chi2Blind,
    Chi2DualBand,
    Chi2Parametric_alt,
    Chi2UltraWideBand,
)

# from simtools.analysis import *
from .preset.preset import *


class Pipeline:
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

        ### Creating noise seed
        mpitools = MpiTools(comm)
        seed_noise = mpitools.get_random_value(init_seed=None)
        
        ### Initialization
        self.preset = PresetInitialisation(comm, seed_noise).initialize(parameters_file)
        self.plots = PlotsCMM(self.preset, dogif=True)
        if (
            self.preset.comp.params_foregrounds["Dust"]["type"] == "blind"
            or self.preset.comp.params_foregrounds["Synchrotron"]["type"] == "blind"
        ):
            self.chi2 = Chi2Blind(self.preset)
        else:
            pass

        self.fsub = int(
            self.preset.qubic.joint_out.qubic.nsub
            / self.preset.comp.params_foregrounds["bin_mixing_matrix"]
        )

        ### Create variables for stopping condition
        self._rms_noise_qubic_patch_per_ite = np.empty(
            (
                self.preset.tools.params["PCG"]["ites_to_converge"],
                len(self.preset.comp.components_name_out),
            )
        )
        self._rms_noise_qubic_patch_per_ite[:] = np.nan

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
            self.plots._display_allcomponents(seenpix, ki=-1)

        ### Initialize PCG starting point
        initial_maps = self.preset.comp.components_iter[:, seenpix, :].copy()
        print(self.preset.comp.components_iter.shape)
        strop

        ### Update the precondtionner M
        #if self._steps == 0:
        self.preset.acquisition.M = self.preset.acquisition.get_preconditioner(seenpix=seenpix,
            A_qubic=self.preset.acquisition.Amm_iter[:self.preset.qubic.params_qubic["nsub_out"]],
        #self.preset.acquisition.M = self.preset.acquisition.get_preconditioner(A_qubic=self.preset.mixingmatrix.Amm_in[:self.preset.qubic.params_qubic["nsub_out"]],
            A_ext=self.preset.mixingmatrix.Amm_in[self.preset.qubic.params_qubic["nsub_out"]:],
            precond=self.preset.qubic.params_qubic["preconditioner"], 
            thr=self.preset.tools.params["PLANCK"]["thr_planck"]
        )

        ### Run PCG
        if self._steps == 0:
            num_iter = self.preset.tools.params["PCG"]["n_init_iter_pcg"]
            maxiter = self.preset.tools.params["PCG"]["n_init_iter_pcg"]
        else:
            num_iter = self.preset.tools.params["PCG"]["n_iter_pcg"]
            maxiter = max_iterations

        if self.preset.tools.params["PCG"]["do_gif"]:
            gif_folder = f"CMM/jobs/{self.preset.job_id}/iter/"
        else:
            gif_folder = None

        ### PCG
        result = pcg(
            A=self.preset.A,
            b=self.preset.b,
            comm=self.preset.comm,
            x0=initial_maps,
            M=self.preset.acquisition.M,
            tol=self.preset.tools.params["PCG"]["tol_pcg"],
            disp=True,
            maxiter=maxiter,
            gif_folder=gif_folder,
            job_id=self.preset.job_id,
            seenpix=seenpix,
            seenpix_plot=self.preset.sky.seenpix,
            center=self.preset.sky.center,
            reso=self.preset.tools.params["PCG"]["reso_plot"],
            fwhm_plot=self.preset.tools.params["PCG"]["fwhm_plot"],
            input=self.preset.comp.components_out,
            iter_init=self._steps * num_iter,
            is_planck=True,
        )["x"]["x"]

        ### Update components
        self.preset.comp.components_iter[:, seenpix, :] = result.copy()

        ### Plot if asked
        if self.preset.tools.rank == 0:
            if self.preset.tools.params["PCG"]["do_gif"]:
                do_gif(
                    f"CMM/jobs/{self.preset.job_id}/iter/",
                    "iter_",
                    output="animation.gif",
                )
            self.plots.display_maps(seenpix, ki=self._steps)
            self.plots._display_allcomponents(
                seenpix,
                ki=self._steps,
                gif=self.preset.tools.params["PCG"]["do_gif"],
                reso=self.preset.tools.params["PCG"]["reso_plot"],
            )
            # self.plots._display_allresiduals(self.preset.comp.components_iter[:, self.preset.sky.seenpix, :], self.preset.sky.seenpix, ki=self._steps)
            self.plots.plot_rms_iteration(
                self.preset.acquisition.rms_plot, ki=self._steps
            )

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

        if self.preset.qubic.params_qubic["convolution_out"]:
            x_planck = self.preset.comp.components_convolved_out * (
                1 - seenpix[None, :, None]
            )
        else:
            x_planck = self.preset.comp.components_out * (1 - seenpix[None, :, None])
        self.preset.b = U.T(
            H_i.T
            * self.preset.acquisition.invN
            * (self.preset.acquisition.TOD_obs - H_i(x_planck))
        )


        # TO BE REMOVE
        ### Update components when intensity maps are fixed
        # elif self.preset.tools.params['PCG']['fixI']:
        #    mask = np.ones((len(self.preset.comp.components_name_out), 12*self.preset.sky.params_sky['nside']**2, 3))
        #    mask[:, :, 0] = 0
        #    P = (
        #        ReshapeOperator(PackOperator(mask).shapeout, (len(self.preset.comp.components_name_out), 12*self.preset.sky.params_sky['nside']**2, 2)) *
        #        PackOperator(mask)
        #        ).T

        #    xI = self.preset.comp.components_convolved_out * (1 - mask)
        #    self.preset.A = P.T * H_i.T * self.preset.acquisition.invN * H_i * P
        #    self.preset.b = P.T (H_i.T * self.preset.acquisition.invN * (self.preset.acquisition.TOD_obs - H_i(xI)))

        ### Update components
        # else:
        # self.preset.A = H_i.T * self.preset.acquisition.invN * H_i
        # self.preset.b = H_i.T * self.preset.acquisition.invN * self.preset.acquisition.TOD_obs

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
                self.preset.qubic.joint_out.qubic.ndets
                * self.preset.qubic.joint_out.qubic.nsamples,
            )
        )

        for i in range(len(self.preset.comp.components_name_out)):
            for j in range(self.preset.qubic.joint_out.qubic.nsub):
                if self.preset.qubic.params_qubic["convolution_out"]:
                    C = HealpixConvolutionGaussianOperator(
                        fwhm=self.preset.acquisition.fwhm_mapmaking[j],
                        lmax=3 * self.preset.sky.params_sky["nside"],
                    )
                else:
                    C = HealpixConvolutionGaussianOperator(
                        fwhm=0, lmax=3 * self.preset.sky.params_sky["nside"]
                    )
                tod_comp[i, j] = self.preset.qubic.joint_out.qubic.H[j](
                    C(self.preset.comp.components_iter[i])
                ).ravel()

        return tod_comp

    def callback(self, x):
        """Callback for scipy.minimize.

        Method to make callback function readable by `scipy.optimize.minimize`.

        This method is intended to be used as a callback function during the optimization
        process. It is called by the optimizer at each iteration.

        The method performs the following actions:
        1. Synchronizes all processes using a barrier to ensure that all processes reach this point before proceeding.
        2. If the current process is the root process (rank 0), it performs the following:
            a. Every 5 iterations (when `self.nfev` is a multiple of 5), it prints the current iteration number and the parameter values rounded to 5 decimal places.
        3. Increments the iteration counter `self.nfev` by 1.

        Parameters
        ----------
        x : array_like
            The current parameter values at the current iteration of the optimization.

        """

        self.preset.tools.comm.Barrier()
        if self.preset.tools.rank == 0:
            if (self.nfev % 1) == 0:
                print(
                    f"Iter = {self.nfev:4d}   x = {[np.round(x[i], 5) for i in range(len(x))]}   qubic log(L) = {np.log(np.round(self.chi2.Lqubic, 5))}  planck log(L) = {np.log(np.round(self.chi2.Lplanck, 5))}"
                )
            self.nfev += 1

    def get_constrains(self):
        """Constraints for scipy.minimize.

        Generate constraints readable by `scipy.optimize.minimize`.

        Returns
        -------
        constraints: list
            A list of constraint dictionaries for optimize.minimize.

        """

        constraints = []
        n = (self.preset.comp.params_foregrounds["bin_mixing_matrix"] - 1) * (
            len(self.preset.comp.components_name_out) - 1
        )

        ### Dust only : constraint ==> SED is increasing
        if (
            self.preset.comp.params_foregrounds["Dust"]["Dust_out"]
            and not self.preset.comp.params_foregrounds["Synchrotron"][
                "Synchrotron_out"
            ]
        ):
            for i in range(n):
                constraints.append(
                    {"type": "ineq", "fun": lambda x, i=i: x[i + 1] - x[i]}
                )

        ### Synchrotron only : constraint ==> SED is decreasing
        elif (
            not self.preset.comp.params_foregrounds["Dust"]["Dust_out"]
            and self.preset.comp.params_foregrounds["Synchrotron"]["Synchrotron_out"]
        ):
            for i in range(n):
                constraints.append(
                    {"type": "ineq", "fun": lambda x, i=i: x[i] - x[i + 1]}
                )

        ### No component : constraint ==> None
        elif (
            not self.preset.comp.params_foregrounds["Dust"]["Dust_out"]
            and not self.preset.comp.params_foregrounds["Synchrotron"][
                "Synchrotron_out"
            ]
        ):
            return None

        ### Dust & Synchrotron : constraint ==> SED is increasing for one component and decrasing for the other one
        elif (
            self.preset.comp.params_foregrounds["Dust"]["Dust_out"]
            and self.preset.comp.params_foregrounds["Synchrotron"]["Synchrotron_out"]
        ):
            for i in range(n):
                # Dust
                if i % 2 == 0:
                    constraints.append(
                        {"type": "ineq", "fun": lambda x, i=i: x[i + 2] - x[i]}
                    )
                # Sync
                else:
                    constraints.append(
                        {"type": "ineq", "fun": lambda x, i=i: x[i] - x[i + 2]}
                    )

        return constraints

    def get_tod_comp_superpixel(self, index):
        if self.preset.tools.rank == 0:
            print("Computing contribution of each super-pixel")
        _index = np.zeros(
            12 * self.preset.comp.params_foregrounds["Dust"]["nside_beta_out"] ** 2
        )
        _index[index] = index.copy()
        _index_nside = hp.ud_grade(_index, self.preset.qubic.joint_out.external.nside)
        tod_comp = np.zeros(
            (
                len(index),
                self.preset.qubic.joint_out.qubic.nsub,
                len(self.preset.comp.components_name_out),
                self.preset.qubic.joint_out.qubic.ndets
                * self.preset.qubic.joint_out.qubic.nsamples,
            )
        )

        maps_conv = self.preset.comp.components_iter.copy()

        for j in range(self.preset.qubic.params_qubic["nsub_out"]):
            for icomp in range(len(self.preset.comp.components_name_out)):
                if self.preset.qubic.params_qubic["convolution_out"]:
                    C = HealpixConvolutionGaussianOperator(
                        fwhm=self.preset.acquisition.fwhm_mapmaking[j],
                        lmax=3 * self.preset.sky.params_sky["nside"],
                    )
                else:
                    C = HealpixConvolutionGaussianOperator(
                        fwhm=0, lmax=3 * self.preset.sky.params_sky["nside"]
                    )
                maps_conv[icomp] = C(
                    self.preset.comp.components_iter[icomp, :, :]
                ).copy()
                for ii, i in enumerate(index):
                    maps_conv_i = maps_conv.copy()
                    _i = _index_nside == i
                    for stk in range(3):
                        maps_conv_i[:, :, stk] *= _i
                    tod_comp[ii, j, icomp] = self.preset.qubic.joint_out.qubic.H[j](
                        maps_conv_i[icomp]
                    ).ravel()
        return tod_comp

    def update_mixing_matrix(self, beta, previous_mixingmatrix, icomp):
        """Update Mixing Matrix.

        Method to update the mixing matrix using the current fitted value of the beta parameter and the parametric model associated.
        Only use when hybrid parametric-blind fit is selected !

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
        mixingmatrix = mm.MixingMatrix(*self.preset.comp.components_out)
        model_mixingmatrix = mixingmatrix.eval(
            self.preset.qubic.joint_out.qubic.allnus, *beta
        )

        ### Update the mixing matrix according to the one computed using the beta parameter
        updated_mixingmatrix = previous_mixingmatrix
        for ii in range(self.preset.comp.params_foregrounds["bin_mixing_matrix"]):
            updated_mixingmatrix[ii * self.fsub : (ii + 1) * self.fsub, icomp] = (
                model_mixingmatrix[ii * self.fsub : (ii + 1) * self.fsub, icomp]
            )

        return updated_mixingmatrix

    def update_spectral_index(self):
        """Update spectral index.

        Method that perform step 3) of the pipeline for 2 possible designs : Two Bands and Ultra Wide Band

        """
        method_0 = self.preset.comp.params_foregrounds[self.preset.comp.components_name_out[1]]["type"]
        if len(self.preset.comp.components_name_out) > 1:
            cpt = 2
            while cpt < len(self.preset.comp.components_name_out):
                if (self.preset.comp.components_name_out[cpt] != "CO"
                    and self.preset.comp.params_foregrounds[self.preset.comp.components_name_out[cpt]]["type"] != method_0):
                    method = "parametric_blind"
                cpt += 1
        try:
            method == "parametric_blind"
        except:
            method = method_0

        tod_comp = self.get_tod_comp()
        self.nfev = 0
        self.preset.mixingmatrix._index_seenpix_beta = 0

        if method == "parametric":
            if self.preset.comp.params_foregrounds["Dust"]["nside_beta_out"] == 0:

                previous_beta = self.preset.acquisition.beta_iter.copy()

                if self.preset.qubic.params_qubic["instrument"] == "DB":
                    self.chi2 = Chi2DualBand(self.preset, tod_comp, parametric=True)
                elif self.preset.qubic.params_qubic["instrument"] == "UWB":
                    self.chi2 = Chi2UltraWideBand(
                        self.preset, tod_comp, parametric=True
                    )

                self.preset.acquisition.beta_iter = minimize(
                    self.chi2,
                    x0=self.preset.acquisition.beta_iter,
                    method="BFGS",
                    callback=self.callback,
                    tol=1e-10,
                ).x

                self.preset.acquisition.Amm_iter = self.chi2._get_mixingmatrix(
                    nus=self.preset.qubic.joint_out.allnus,
                    x=self.preset.acquisition.beta_iter,
                )
                # print(Ai.shape, Ai)
                # for inu in range(self.preset.qubic.joint_out.qubic.nsub):
                #    for icomp in range(1, len(self.preset.comp.components_name_out)):
                #        self.preset.acquisition.Amm_iter[inu, icomp] = Ai[inu, icomp]

                del tod_comp
                gc.collect()

                if self.preset.tools.rank == 0:

                    print(f"Iteration k     : {previous_beta}")
                    print(
                        f"Iteration k + 1 : {self.preset.acquisition.beta_iter.copy()}"
                    )
                    print(
                        f"Truth           : {self.preset.mixingmatrix.beta_in.copy()}"
                    )
                    print(
                        f"Residuals       : {self.preset.mixingmatrix.beta_in - self.preset.acquisition.beta_iter}"
                    )

                    # if len(self.preset.comp.components_name_out) > 2:
                    #    self.plots.plot_beta_iteration(self.preset.acquisition.allbeta[:, 0], truth=self.preset.mixingmatrix.beta_in[0], ki=self._steps)
                    # else:
                    #    self.plots.plot_beta_iteration(self.preset.acquisition.allbeta, truth=self.preset.mixingmatrix.beta_in, ki=self._steps)

                self.preset.tools.comm.Barrier()

                self.preset.acquisition.allbeta = np.concatenate(
                    (
                        self.preset.acquisition.allbeta,
                        np.array([self.preset.acquisition.beta_iter]),
                    ),
                    axis=0,
                )

            else:

                index_num = hp.ud_grade(
                    self.preset.sky.seenpix_qubic,
                    self.preset.comp.params_foregrounds["Dust"]["nside_beta_out"],
                )  #
                self.preset.mixingmatrix._index_seenpix_beta = np.where(
                    index_num == True
                )[0]

                ### Simulated TOD for each components, nsub, npix with shape (npix, nsub, ncomp, nsnd)
                tod_comp = self.get_tod_comp_superpixel(
                    self.preset.mixingmatrix._index_seenpix_beta
                )

                ### Store fixed beta (those denoted with hp.UNSEEN are variable)
                beta_fixed = self.preset.acquisition.beta_iter.copy()
                beta_fixed[:, self.preset.mixingmatrix._index_seenpix_beta] = hp.UNSEEN
                chi2 = Chi2DualBand(
                    self.preset, tod_comp, parametric=True, full_beta_map=beta_fixed
                )
                # chi2 = Chi2Parametric(self.preset, tod_comp, self.preset.acquisition.beta_iter, seenpix_wrap=None)

                previous_beta = self.preset.acquisition.beta_iter[
                    :, self.preset.mixingmatrix._index_seenpix_beta
                ].copy()
                self.nfev = 0

                beta_i = fmin_l_bfgs_b(
                    chi2,
                    x0=previous_beta,
                    callback=self.callback,
                    approx_grad=True,
                    epsilon=1e-6,
                    maxls=20,
                    maxiter=20,
                )[0]

                self.preset.acquisition.beta_iter[chi2.seenpix_beta] = beta_i

                del tod_comp
                gc.collect()

                self.preset.acquisition.allbeta = np.concatenate(
                    (
                        self.preset.acquisition.allbeta,
                        np.array([self.preset.acquisition.beta_iter]),
                    ),
                    axis=0,
                )

                if self.preset.tools.rank == 0:

                    print(f"Iteration k     : {previous_beta}")
                    print(
                        f"Iteration k + 1 : {self.preset.acquisition.beta_iter[:, self.preset.mixingmatrix._index_seenpix_beta].copy()}"
                    )
                    print(
                        f"Truth           : {self.preset.mixingmatrix.beta_in[:, self.preset.mixingmatrix._index_seenpix_beta].copy()}"
                    )
                    print(
                        f"Residuals       : {self.preset.mixingmatrix.beta_in[:, self.preset.mixingmatrix._index_seenpix_beta] - self.preset.acquisition.beta_iter[:, self.preset.mixingmatrix._index_seenpix_beta]}"
                    )
                    self.plots.plot_beta_iteration(
                        self.preset.acquisition.allbeta[
                            :, :, self.preset.mixingmatrix._index_seenpix_beta
                        ],
                        truth=self.preset.mixingmatrix.beta_in[
                            :, self.preset.mixingmatrix._index_seenpix_beta
                        ],
                        ki=self._steps,
                    )
        elif method == "blind":
            previous_step = self.preset.acquisition.Amm_iter[: self.preset.qubic.joint_out.qubic.nsub, 1:].copy()
            if self._steps == 0:
                self.allAmm_iter = np.array([self.preset.acquisition.Amm_iter])

            if self.preset.comp.params_foregrounds["blind_method"] == "minimize":

                if self.preset.qubic.params_qubic["instrument"] == "DB":
                    self.chi2 = Chi2DualBand(self.preset, tod_comp, parametric=False)
                elif self.preset.qubic.params_qubic["instrument"] == "UWB":
                    self.chi2 = Chi2UltraWideBand(
                        self.preset, tod_comp, parametric=False
                    )

                x0 = []
                bnds = []
                for inu in range(
                    self.preset.comp.params_foregrounds["bin_mixing_matrix"]
                ):
                    for icomp in range(1, len(self.preset.comp.components_name_out)):
                        x0 += [
                            np.mean(
                                self.preset.acquisition.Amm_iter[
                                    inu * self.fsub : (inu + 1) * self.fsub, icomp
                                ]
                            )
                        ]
                        bnds += [(0, None)]

                Ai = minimize(
                    self.chi2,
                    x0=x0,
                    # bounds=bnds,
                    method="L-BFGS-B",
                    # constraints=self.get_constrains(),
                    callback=self.callback,
                    tol=1e-10,
                ).x
                Ai = self.chi2._fill_A(
                    Ai
                )  # Ai.reshape((self.preset.qubic.joint_out.qubic.nsub, len(self.preset.comp.components_name_out)-1))

                for inu in range(self.preset.qubic.joint_out.qubic.nsub):
                    for icomp in range(1, len(self.preset.comp.components_name_out)):
                        self.preset.acquisition.Amm_iter[inu, icomp] = Ai[inu, icomp]
            elif self.preset.comp.params_foregrounds["blind_method"] == "PCG":
                
                tod_comp_binned = np.zeros((tod_comp.shape[0], self.preset.comp.params_foregrounds["bin_mixing_matrix"], tod_comp.shape[-1],))
                
                for k in range(len(self.preset.comp.components_name_out)):
                    for i in range(self.preset.comp.params_foregrounds["bin_mixing_matrix"]):
                        tod_comp_binned[k, i] = np.sum(tod_comp[k, i * self.fsub : (i + 1) * self.fsub], axis=0)

                tod_cmb150 = self.preset.tools.comm.allreduce(
                    np.sum(tod_comp[0, :int(tod_comp.shape[1]/2)], axis=0),
                    op=MPI.SUM,
                )
                tod_cmb220 = self.preset.tools.comm.allreduce(
                    np.sum(tod_comp[0, int(tod_comp.shape[1]/2):int(tod_comp.shape[1])], axis=0), op=MPI.SUM)

                tod_in_150 = self.preset.tools.comm.allreduce(
                    self.preset.acquisition.TOD_qubic[: int(self.preset.acquisition.TOD_qubic.shape[0] / 2)], op=MPI.SUM
                )
                tod_in_220 = self.preset.tools.comm.allreduce(
                    self.preset.acquisition.TOD_qubic[
                        int(self.preset.acquisition.TOD_qubic.shape[0] / 2) : int(
                            self.preset.acquisition.TOD_qubic.shape[0]
                        )
                    ],
                    op=MPI.SUM,
                )

                tod_without_cmb = np.r_[
                    tod_in_150 - tod_cmb150, tod_in_220 - tod_cmb220
                ]
                
                tod_without_cmb_reshaped = np.sum(
                    tod_without_cmb.reshape((2, int(self.preset.acquisition.TOD_qubic.shape[0] / 2))), axis=0
                )

                dnu = self.preset.tools.comm.allreduce(tod_comp_binned[1:], op=MPI.SUM)
                dnu = dnu.reshape((dnu.shape[0] * dnu.shape[1], dnu.shape[2]))

                A = 1e20 * (dnu @ dnu.T)
                b = 1e20 * (dnu @ tod_without_cmb_reshaped)

                s = pcg_op(A, b, disp=False, tol=1e-40, maxiter=10000)

                k = 0
                for i in range(1, len(self.preset.comp.components_name_out)):
                    for ii in range(
                        self.preset.comp.params_foregrounds["bin_mixing_matrix"]):
                        self.preset.acquisition.Amm_iter[ii * self.fsub : (ii + 1) * self.fsub, i] = s["x"][k]  # Ai[k]
                        k += 1
            elif self.preset.comp.params_foregrounds["blind_method"] == "alternate":
                for i in range(len(self.preset.comp.components_name_out)):
                    if self.preset.comp.components_name_out[i] != "CMB":
                        print("I am fitting ", self.preset.comp.components_name_out[i])
                        fun = partial(
                            self.chi2._qu_alt,
                            tod_comp=tod_comp,
                            A=self.preset.acquisition.Amm_iter,
                            icomp=i,
                        )

                        ### Starting point
                        x0 = []
                        bnds = []
                        for ii in range(
                            self.preset.comp.params_foregrounds["bin_mixing_matrix"]
                        ):
                            x0 += [
                                np.mean(
                                    self.preset.acquisition.Amm_iter[
                                        ii * self.fsub : (ii + 1) * self.fsub, i
                                    ]
                                )
                            ]
                            bnds += [(0, None)]
                        if self._steps == 0:
                            x0 = np.array(x0) * 1 + 0

                        Ai = minimize(
                            fun,
                            x0=x0,
                            callback=self.callback,
                            bounds=bnds,
                            method="SLSQP",
                            tol=1e-10,
                        ).x

                        for ii in range(
                            self.preset.comp.params_foregrounds["bin_mixing_matrix"]
                        ):
                            self.preset.acquisition.Amm_iter[
                                ii * self.fsub : (ii + 1) * self.fsub, i
                            ] = Ai[ii]
            else:
                raise TypeError(
                    f"{self.preset.comp.params_foregrounds['blind_method']} is not yet implemented.."
                )

            self.allAmm_iter = np.concatenate(
                (self.allAmm_iter, np.array([self.preset.acquisition.Amm_iter])), axis=0
            )

            if self.preset.tools.rank == 0:
                print(f'Iteration k     : {previous_step.ravel()}')
                print(f'Iteration k + 1 : {self.preset.acquisition.Amm_iter[:self.preset.qubic.joint_out.qubic.nsub, 1:].ravel()}')
                print(f'Truth           : {self.preset.mixingmatrix.Amm_in[:self.preset.qubic.joint_out.qubic.nsub, 1:].ravel()}')
                print(f'Residuals       : {self.preset.mixingmatrix.Amm_in[:self.preset.qubic.joint_out.qubic.nsub, 1:].ravel() - self.preset.acquisition.Amm_iter[:self.preset.qubic.joint_out.qubic.nsub, 1:].ravel()}')
                self.plots.plot_sed(
                    self.preset.qubic.joint_in.qubic.allnus,
                    self.preset.mixingmatrix.Amm_in[:self.preset.qubic.joint_in.qubic.nsub, 1:],
                    self.preset.qubic.joint_out.qubic.allnus,
                    self.preset.acquisition.Amm_iter[:self.preset.qubic.joint_out.qubic.nsub, 1:],
                    ki=self._steps,
                    gif=self.preset.tools.params["PCG"]["do_gif"],
                )

                if self.preset.tools.params["PCG"]["do_gif"]:
                    do_gif(
                        f"CMM/jobs/{self.preset.job_id}/A_iter/",
                        "A_iter",
                        output="animation_A_iter.gif",
                    )
                    # do_gif(f'jobs/{self.preset.job_id}/allcomps/', 'iter_', output='animation.gif')
            del tod_comp
            gc.collect()
            
            
        elif method == "parametric_blind":
            previous_step = self.preset.acquisition.Amm_iter[
                : self.preset.qubic.joint_out.qubic.nsub * 2, 1:
            ].copy()
            if self._steps == 0:
                self.allAmm_iter = np.array([self.preset.acquisition.Amm_iter])
            for i in range(len(self.preset.comp.components_name_out)):
                if self.preset.comp.components_name_out[i] != "CMB":
                    if (
                        self.preset.comp.params_foregrounds[
                            self.preset.comp.components_name_out[i]
                        ]["type"]
                        == "parametric"
                    ):
                        print(
                            "I am fitting ", self.preset.comp.components_name_out[i], i
                        )

                        # if self._steps==0:
                        #    self.preset.acquisition.beta_iter = self.preset.acquisition.beta_iter
                        previous_beta = self.preset.acquisition.beta_iter.copy()

                        chi2 = Chi2Parametric_alt(
                            self.preset,
                            tod_comp,
                            self.preset.acquisition.Amm_iter,
                            i,
                            seenpix_wrap=None,
                        )

                        self.preset.acquisition.beta_iter[i - 1] = np.array(
                            [
                                fmin_l_bfgs_b(
                                    chi2,
                                    x0=self.preset.acquisition.beta_iter[i - 1],
                                    callback=self.callback,
                                    approx_grad=True,
                                    epsilon=1e-6,
                                )[0]
                            ]
                        )

                        self.preset.acquisition.Amm_iter = self.update_mixing_matrix(
                            self.preset.acquisition.beta_iter,
                            self.preset.acquisition.Amm_iter,
                            i,
                        )

                    else:
                        print(
                            "I am fitting ", self.preset.comp.components_name_out[i], i
                        )

                        fun = partial(
                            self.chi2._qu_alt,
                            tod_comp=tod_comp,
                            A=self.preset.acquisition.Amm_iter,
                            icomp=i,
                        )

                        ### Starting point
                        x0 = []
                        bnds = []
                        for ii in range(
                            self.preset.comp.params_foregrounds["bin_mixing_matrix"]
                        ):
                            for j in range(
                                1, len(self.preset.comp.components_name_out)
                            ):
                                x0 += [
                                    np.mean(
                                        self.preset.acquisition.Amm_iter[
                                            ii * self.fsub : (ii + 1) * self.fsub, j
                                        ]
                                    )
                                ]
                                bnds += [(0, None)]

                        Ai = minimize(
                            fun,
                            x0=x0,
                            callback=self.callback,
                            bounds=bnds,
                            method="SLSQP",
                            tol=1e-10,
                        ).x

                        for ii in range(
                            self.preset.comp.params_foregrounds["bin_mixing_matrix"]
                        ):
                            self.preset.acquisition.Amm_iter[
                                ii * self.fsub : (ii + 1) * self.fsub, i
                            ] = Ai[ii]

            self.allAmm_iter = np.concatenate(
                (self.allAmm_iter, np.array([self.preset.acquisition.Amm_iter])), axis=0
            )

            if self.preset.tools.rank == 0:
                print(f"Iteration k     : {previous_step.ravel()}")
                print(
                    f"Iteration k + 1 : {self.preset.acquisition.Amm_iter[:self.preset.qubic.joint_out.qubic.nsub*2, 1:].ravel()}"
                )
                print(
                    f"Truth           : {self.preset.mixingmatrix.Amm_in[:self.preset.qubic.joint_out.qubic.nsub*2, 1:].ravel()}"
                )
                print(
                    f"Residuals       : {self.preset.mixingmatrix.Amm_in[:self.preset.qubic.joint_out.qubic.nsub*2, 1:].ravel() - self.preset.acquisition.Amm_iter[:self.preset.qubic.joint_out.qubic.nsub*2, 1:].ravel()}"
                )
                self.plots.plot_sed(
                    self.preset.qubic.joint_out.qubic.allnus,
                    self.allAmm_iter[
                        :, : self.preset.qubic.joint_out.qubic.nsub * 2, 1:
                    ],
                    ki=self._steps,
                    truth=self.preset.mixingmatrix.Amm_in[
                        : self.preset.qubic.joint_out.qubic.nsub * 2, 1:
                    ],
                    gif=self.preset.tools.params["PCG"]["do_gif"],
                )

            del tod_comp
            gc.collect()

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
            self.preset.qubic.joint_out.qubic.ndets
            * self.preset.qubic.joint_out.qubic.nsamples,
            (
                self.preset.qubic.joint_out.qubic.ndets,
                self.preset.qubic.joint_out.qubic.nsamples,
            ),
        )

        return (1 / np.sum(_r(D) * _invn(_r(D)), axis=1)) * np.sum(
            _r(D) * _invn(_r(d)), axis=1
        )

    def update_gain(self):
        r"""Update detectors gain.

        Method that compute and print gains of each detectors using semi-analytical method decribed in "give_intercal" function,
        using the formula : :math:`g^i = \frac{TOD_{obs}^i}{TOD_{sim}^i}`.

        """

        self.H_i = self.preset.qubic.joint_out.get_operator(
            #self.preset.acquisition.beta_iter,
            A=self.preset.acquisition.Amm_iter,
            gain=np.ones(self.preset.gain.gain_iter.shape),
            fwhm=self.preset.acquisition.fwhm_mapmaking,
            nu_co=self.preset.comp.nu_co,
        )
        self.nsampling = self.preset.qubic.joint_out.qubic.nsamples
        self.ndets = self.preset.qubic.joint_out.qubic.ndets

        if self.preset.qubic.params_qubic["instrument"] == "UWB":
            _r = ReshapeOperator(
                self.preset.qubic.joint_out.qubic.ndets
                * self.preset.qubic.joint_out.qubic.nsamples,
                (self.preset.qubic.joint_out.qubic.ndets, self.preset.qubic.joint_out.qubic.nsamples),
            )

            TODi_Q = self.preset.acquisition.invN.operands[0](
                self.H_i.operands[0](self.preset.comp.components_iter)[
                    : self.ndets * self.nsampling
                ]
            )
            print("invN", self.preset.acquisition.invN.operands[0].operands[1])
            print("reshape", _r.shapein, _r.shapeout)
            print("invN shape", self.preset.acquisition.invN.operands[0].operands[1].shapein, self.preset.acquisition.invN.operands[0].operands[1].shapeout)
            print("TODi_Q", TODi_Q.shape)
            self.preset.gain.gain_iter = self.give_intercal(
                TODi_Q, _r(self.preset.acquisition.TOD_qubic), self.preset.acquisition.invN.operands[0].operands[1]
            )
            self.preset.gain.gain_iter /= self.preset.gain.gain_iter[0]
            self.preset.gain.all_gain = np.concatenate(
                (self.preset.gain.all_gain, np.array([self.preset.gain.gain_iter])), axis=0
            )

        elif self.preset.qubic.params_qubic["instrument"] == "DB":

            TODi_Q_150 = self.H_i.operands[0](self.preset.comp.components_iter)[
                : self.ndets * self.nsampling
            ]
            TODi_Q_220 = self.H_i.operands[0](self.preset.comp.components_iter)[
                self.ndets * self.nsampling : 2 * self.ndets * self.nsampling
            ]

            g150 = self.give_intercal(
                TODi_Q_150,
                self.preset.acquisition.TOD_qubic[: self.ndets * self.nsampling],
                self.preset.acquisition.invN.operands[0].operands[1].operands[0],
            )
            g220 = self.give_intercal(
                TODi_Q_220,
                self.preset.acquisition.TOD_qubic[
                    self.ndets * self.nsampling : 2 * self.ndets * self.nsampling
                ],
                self.preset.acquisition.invN.operands[0].operands[1].operands[1],
            )

            self.preset.gain.gain_iter = np.array([g150, g220]).T
            self.preset.Gi = join_data(
                self.preset.tools.comm, self.preset.gain.gain_iter
            )
            print("gain_iter", self.preset.gain.gain_iter.shape, self.preset.Gi.shape)
            print("all_gain", self.preset.gain.all_gain.shape)
            print("all_gain_in", self.preset.gain.all_gain_in.shape)
            self.preset.gain.all_gain = np.concatenate(
                (self.preset.gain.all_gain, np.array(self.preset.gain.gain_iter)), axis=0)
            
            print("all_gain", self.preset.gain.all_gain.shape)
            if self.preset.tools.rank == 0:
                print(np.mean(self.preset.gain.gain_iter - self.preset.gain.gain_in, axis=0))
                print(np.std(self.preset.gain.gain_iter - self.preset.gain.gain_in, axis=0))

        # self.plots.plot_gain_iteration(
        #     self.preset.gain.all_gain - self.preset.gain.all_gain_in, alpha=0.03, ki=self._steps
        # )

    def save_data(self, step):
        f"""Save data.
        
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
                            os.remove(
                                "CMM/"
                                + self.preset.tools.params["foldername"]
                                + "/maps/"
                                + self.preset.tools.params["filename"]
                                + f"_seed{str(self.preset.tools.params['CMB']['seed'])}_{str(self.preset.job_id)}_k{step-1}.pkl"
                            )

                    with open(
                        "CMM/"
                        + self.preset.tools.params["foldername"]
                        + "/maps/"
                        + self.preset.tools.params["filename"]
                        + f"_seed{str(self.preset.tools.params['CMB']['seed'])}_{str(self.preset.job_id)}_k{step}.pkl",
                        "wb",
                    ) as handle:
                        pickle.dump(
                            {
                                "components": self.preset.comp.components_in,
                                "components_i": self.preset.comp.components_iter,
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
                                "fwhm_in": self.preset.acquisition.fwhm_tod,
                                "fwhm_out": self.preset.acquisition.fwhm_mapmaking,
                                "fwhm_rec": self.preset.acquisition.fwhm_rec,
                                "qubic_dict": {k:v for k,v in self.preset.qubic.dict.items() if k != 'comm'}, # Need to remove the MPI communictor, which is not suppurted by pickle
                                #'fwhm':self.preset.acquisition.fwhm_tod,
                                #'acquisition.fwhm_rec':self.preset.acquisition.fwhm_mapmaking
                            },
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )

    def _compute_map_noise_qubic_patch(self):
        """

        Compute the rms of the noise within the qubic patch.

        """
        nbins = 1  # average over the entire qubic patch

        # if self.preset.comp.params_foregrounds['Dust']['nside_beta_out'] == 0:
        if self.preset.qubic.params_qubic["convolution_out"]:
            residual = (
                self.preset.comp.components_iter
                - self.preset.comp.components_convolved_out
            )
        else:
            residual = (
                self.preset.comp.components_iter - self.preset.comp.components_out
            )
        # else:
        #     if self.preset.qubic.params_qubic['convolution_out']:
        #         residual = self.preset.comp.components_iter.T - self.preset.comp.components_convolved_out
        #     else:
        #         residual = self.preset.comp.components_iter.T - self.preset.comp.components_out.T
        rms_maxpercomp = np.zeros(len(self.preset.comp.components_name_out))

        for i in range(len(self.preset.comp.components_name_out)):
            angs, I, Q, U, dI, dQ, dU = get_angular_profile(
                residual[i],
                thmax=self.preset.angmax,
                nbins=nbins,
                doplot=False,
                allstokes=True,
                separate=True,
                integrated=True,
                center=self.preset.sky.center,
            )

            ### Set dI to 0 to only keep polarization fluctuations
            dI = 0
            rms_maxpercomp[i] = np.max([dI, dQ, dU])
        return rms_maxpercomp

    def _compute_maxrms_array(self):

        if self._steps <= self.preset.tools.params["PCG"]["ites_to_converge"] - 1:
            self._rms_noise_qubic_patch_per_ite[self._steps, :] = (
                self._compute_map_noise_qubic_patch()
            )
        elif self._steps > self.preset.tools.params["PCG"]["ites_to_converge"] - 1:
            self._rms_noise_qubic_patch_per_ite[:-1, :] = (
                self._rms_noise_qubic_patch_per_ite[1:, :]
            )
            self._rms_noise_qubic_patch_per_ite[-1, :] = (
                self._compute_map_noise_qubic_patch()
            )

    def _stop_condition(self):
        """
        Method that stop the convergence if there are more than k steps.

        """

        if self._steps >= self.preset.tools.params["PCG"]["ites_to_converge"] - 1:

            deltarms_max_percomp = np.zeros(len(self.preset.comp.components_name_out))

            for i in range(len(self.preset.comp.components_name_out)):
                deltarms_max_percomp[i] = np.max(
                    np.abs(
                        (
                            self._rms_noise_qubic_patch_per_ite[:, i]
                            - self._rms_noise_qubic_patch_per_ite[-1, i]
                        )
                        / self._rms_noise_qubic_patch_per_ite[-1, i]
                    )
                )

            deltarms_max = np.max(deltarms_max_percomp)
            # if self.preset.tools.rank == 0:
            #    print(f'Maximum RMS variation for the last {self.preset.acquisition.ites_rms_tolerance} iterations: {deltarms_max}')

            if deltarms_max < self.preset.tools.params["PCG"]["tol_rms"]:
                print(
                    f"RMS variations lower than {self.preset.acquisition.rms_tolerance} for the last {self.preset.acquisition.ites_rms_tolerance} iterations."
                )

                ### Update components last time with converged parameters
                # self.update_components(maxiter=100)
                self._info = False

        if self._steps >= self.preset.tools.params["PCG"]["n_iter_loop"] - 1:

            ### Update components last time with converged parameters
            # self.update_components(maxiter=100)

            ### Wait for all processes and save data inside pickle file
            # self.preset.tools.comm.Barrier()
            # self.save_data()

            self._info = False

        self._steps += 1

    def main(self):
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
        # print(self.preset.acquisition.beta_iter)
        while self._info:
            ### Display iteration
            self.preset.tools._display_iter(self._steps)

            ### Update self.fg.components_iter^{k} -> self.fg.components_iter^{k+1}
            self.update_components(seenpix=self.preset.sky.seenpix)

            ### Update self.preset.acquisition.beta_iter^{k} -> self.preset.acquisition.beta_iter^{k+1}
            if self.preset.comp.params_foregrounds["fit_spectral_index"]:
                self.update_spectral_index()

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
