import gc

import numpy as np
from scipy.optimize import minimize

from qubic.lib.MapMaking.ComponentMapMaking.mixing_matrix.fittingMM import FittingMM
from qubic.lib.MapMaking.ComponentMapMaking.Qchi2MM import Chi2
from qubic.lib.Qfoldertools import do_gif


class BlindMM(FittingMM):
    def update(self, tod_comp):
        previous_step = self.preset.acquisition.Amm_iter[: self.preset.qubic.joint_out.qubic.nsub, 1:].copy()

        if self.selfCMM.allAmm_iter is None:
            self.selfCMM.allAmm_iter = np.array([self.preset.acquisition.Amm_iter])
            self.plots.plot_sed(
                self.preset.qubic.joint_in.qubic.allnus,
                self.preset.mixingmatrix.Amm_in[: self.preset.qubic.joint_in.qubic.nsub, 1:],
                self.preset.qubic.joint_out.qubic.allnus,
                self.preset.acquisition.Amm_iter[: self.preset.qubic.joint_out.qubic.nsub, 1:],
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
        for inu in range(self.preset.comp.params_foregrounds["bin_mixing_matrix"]):
            for icomp in range(1, len(self.preset.comp.components_name_out)):
                x0 += [np.mean(self.preset.acquisition.Amm_iter[inu * self.preset.qubic.joint_out.qubic.fsub : (inu + 1) * self.preset.qubic.joint_out.qubic.fsub, icomp])]
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
        Ai = self.chi2.compute_mixing_matrix_blind(Ai)

        for inu in range(self.preset.qubic.joint_out.qubic.nsub):
            for icomp in range(1, len(self.preset.comp.components_name_out)):
                self.preset.acquisition.Amm_iter[inu, icomp] = Ai[inu, icomp]

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
        self.selfCMM.allAmm_iter = np.concatenate((self.selfCMM.allAmm_iter, np.array([self.preset.acquisition.Amm_iter])), axis=0)
        self.plots.plot_sed(
            self.preset.qubic.joint_in.qubic.allnus,
            self.preset.mixingmatrix.Amm_in[: self.preset.qubic.joint_in.qubic.nsub, 1:],
            self.preset.qubic.joint_out.qubic.allnus,
            self.preset.acquisition.Amm_iter[: self.preset.qubic.joint_out.qubic.nsub, 1:],
            ki=self._steps,
            gif=self.preset.tools.params["PCG"]["do_gif"],
        )

        if self.preset.tools.params["PCG"]["do_gif"]:
            do_gif(
                "CMM/" + self.preset.tools.params["foldername"] + "/Plots/A_iter/",
                output="animation_A_iter.gif",
                fps=1,
            )
