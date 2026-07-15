import gc

import numpy as np
from scipy.optimize import minimize

from qubic.lib.MapMaking.ComponentMapMaking.mixing_matrix.fittingMM import FittingMM
from qubic.lib.MapMaking.ComponentMapMaking.Qchi2MM import Chi2, finite_diff_hessian
from qubic.lib.Qfoldertools import do_gif


class BlindMM(FittingMM):
    def _bin_by_freq(self, arr, n_bins, fsub_out):
        """Average `arr` over each block of `fsub_out` consecutive output frequencies.

        The blind fit only has `n_bins` (bin_mixing_matrix) independent values, each shared
        across `fsub_out` frequencies (see compute_mixing_matrix_blind); plotting at the full
        per-frequency resolution would show every fitted value duplicated fsub_out times.
        """
        return np.array([arr[i * fsub_out : (i + 1) * fsub_out].mean(axis=0) for i in range(n_bins)])

    def _bin_bandwidth(self, bw, n_bins, fsub_out):
        """Total bandwidth of each block of `fsub_out` consecutive, contiguous output
        frequencies (bandwidths add, unlike the frequencies/values averaged in _bin_by_freq)."""
        return np.array([bw[i * fsub_out : (i + 1) * fsub_out].sum() for i in range(n_bins)])

    def update(self, tod_comp):
        previous_step = self.preset.acquisition.Amm_iter[: self.preset.qubic.joint_out.qubic.nsub, 1:].copy()

        n_bins = self.preset.comp.params_foregrounds["bin_mixing_matrix"]
        # NB: joint_out.qubic.fsub is nsub/nrec (the dual-band split, nrec=2), unrelated to
        # bin_mixing_matrix. The blind fit bins nsub_out frequencies into n_bins groups directly.
        fsub_out = self.preset.qubic.joint_out.qubic.nsub // n_bins

        if self.selfCMM.allAmm_iter is None:
            self.selfCMM.allAmm_iter = np.array([self.preset.acquisition.Amm_iter])
            self.plots.plot_sed(
                self.preset.qubic.joint_in.qubic.allnus,
                self.preset.mixingmatrix.Amm_in[: self.preset.qubic.joint_in.qubic.nsub, 1:],
                self._bin_by_freq(self.preset.qubic.joint_out.qubic.allnus, n_bins, fsub_out),
                self._bin_by_freq(
                    self.preset.acquisition.Amm_iter[: self.preset.qubic.joint_out.qubic.nsub, 1:], n_bins, fsub_out
                ),
                nus_out_bw=self._bin_bandwidth(self.preset.qubic.joint_out.qubic.allnus_bw, n_bins, fsub_out),
                nus_in_bw=self.preset.qubic.joint_in.qubic.allnus_bw[: self.preset.qubic.joint_in.qubic.nsub],
                ki=self._steps - 1,
                gif=self.preset.tools.params["PCG"]["do_gif"],
            )

        self.chi2 = Chi2(
            self.preset,
            tod_comp,
            parametric=False,
        )
        x0 = []
        bnds = []
        for inu in range(n_bins):
            for icomp in range(1, len(self.preset.comp.components_name_out)):
                x0 += [np.mean(self.preset.acquisition.Amm_iter[inu * fsub_out : (inu + 1) * fsub_out, icomp])]
                bnds += [(0, None)]

        res = minimize(
            self.chi2,
            x0=x0,
            # bounds=bnds,
            method="L-BFGS-B",
            # constraints=self.get_constrains(),
            callback=self.callback,
            tol=1e-10,
        )
        Ai = self.chi2.compute_mixing_matrix_blind(res.x)

        # 1-sigma uncertainty. L-BFGS-B's hess_inv is a rank-limited quasi-Newton
        # approximation built only from the gradient history of *this* minimize() call:
        # it can be unreliable if the fit converges in few iterations or if bins are
        # correlated. Cross-check it against a Hessian computed directly from the
        # objective (finite differences) and use that one for the reported error bars.
        sigma_x_lbfgs = np.sqrt(np.diag(np.asarray(res.hess_inv.todense())))
        H_fd = finite_diff_hessian(self.chi2, res.x)
        sigma_x = np.sqrt(np.diag(np.linalg.inv(H_fd)))
        if self.preset.tools.rank == 0:
            print(f"nit = {res.nit}, success = {res.success}")
            print(f"sigma (L-BFGS-B hess_inv)  : {sigma_x_lbfgs}")
            print(f"sigma (finite-diff Hessian): {sigma_x}")
        Ai_err = self.chi2.compute_mixing_matrix_blind(sigma_x)
        Ai_err[:, 0] = 0.0  # CMB column is fixed to 1, not part of the blind fit

        for inu in range(self.preset.qubic.joint_out.qubic.nsub):
            for icomp in range(1, len(self.preset.comp.components_name_out)):
                self.preset.acquisition.Amm_iter[inu, icomp] = Ai[inu, icomp]
                self.preset.acquisition.Amm_iter_err[inu, icomp] = Ai_err[inu, icomp]

        self._log(previous_step)
        self._finalize()
        del tod_comp
        gc.collect()

    def _log(self, previous_step):
        if self.preset.tools.rank != 0:
            return
        print(f"Iteration k     : {previous_step.ravel()}")
        print(f"Iteration k + 1 : {self.preset.acquisition.Amm_iter[: self.preset.qubic.joint_out.qubic.nsub, 1:].ravel()}")
        print(f"Truth           : {self.preset.mixingmatrix.Amm_in[: self.preset.qubic.joint_out.qubic.nsub, 1:].ravel()}")
        print(
            f"Residuals       : {self.preset.mixingmatrix.Amm_in[: self.preset.qubic.joint_out.qubic.nsub, 1:].ravel() - self.preset.acquisition.Amm_iter[: self.preset.qubic.joint_out.qubic.nsub, 1:].ravel()}"
        )

    def _finalize(self):
        self.preset.tools.comm.Barrier()
        n_bins = self.preset.comp.params_foregrounds["bin_mixing_matrix"]
        fsub_out = self.preset.qubic.joint_out.qubic.nsub // n_bins
        self.selfCMM.allAmm_iter = np.concatenate((self.selfCMM.allAmm_iter, np.array([self.preset.acquisition.Amm_iter])), axis=0)
        self.plots.plot_sed(
            self.preset.qubic.joint_in.qubic.allnus,
            self.preset.mixingmatrix.Amm_in[: self.preset.qubic.joint_in.qubic.nsub, 1:],
            self._bin_by_freq(self.preset.qubic.joint_out.qubic.allnus, n_bins, fsub_out),
            self._bin_by_freq(
                self.preset.acquisition.Amm_iter[: self.preset.qubic.joint_out.qubic.nsub, 1:], n_bins, fsub_out
            ),
            A_out_err=self._bin_by_freq(
                self.preset.acquisition.Amm_iter_err[: self.preset.qubic.joint_out.qubic.nsub, 1:], n_bins, fsub_out
            ),
            nus_out_bw=self._bin_bandwidth(self.preset.qubic.joint_out.qubic.allnus_bw, n_bins, fsub_out),
            nus_in_bw=self.preset.qubic.joint_in.qubic.allnus_bw[: self.preset.qubic.joint_in.qubic.nsub],
            ki=self._steps,
            gif=self.preset.tools.params["PCG"]["do_gif"],
        )

        # Convergence of each (binned) mixing-matrix element across outer iterations,
        # one plot per foreground component (skip column 0, the fixed CMB scaling).
        nsub_out = self.preset.qubic.joint_out.qubic.nsub
        nus_binned = self._bin_by_freq(self.preset.qubic.joint_out.qubic.allnus, n_bins, fsub_out)
        for icomp in range(1, len(self.preset.comp.components_name_out)):
            A_history = np.array(
                [self._bin_by_freq(snap[:nsub_out, icomp], n_bins, fsub_out) for snap in self.selfCMM.allAmm_iter]
            )
            truth = self._bin_by_freq(self.preset.mixingmatrix.Amm_in[:nsub_out, icomp], n_bins, fsub_out)
            self.plots.plot_mixing_matrix_iteration(
                A_history,
                truth=truth,
                nus=nus_binned,
                name=self.preset.comp.components_name_out[icomp],
                ki=self._steps,
            )

        if self.preset.tools.params["PCG"]["do_gif"]:
            do_gif(
                "CMM/" + self.preset.tools.params["foldername"] + "/Plots/A_iter/",
                output="animation_A_iter.gif",
                fps=1,
            )
