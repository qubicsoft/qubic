import gc

import numpy as np
from scipy.optimize import minimize

from qubic.lib.MapMaking.ComponentMapMaking.mixing_matrix.fittingMM import FittingMM
from qubic.lib.MapMaking.ComponentMapMaking.Qchi2MM import MixedChi2, ParamLayout
from qubic.lib.Qfoldertools import do_gif

class MixedMM(FittingMM):
    def update(self, tod_comp):
        if self.selfCMM.allAmm_iter is None:
            self.selfCMM.allAmm_iter = np.array([self.preset.acquisition.Amm_iter])
            
        previous_beta = self.preset.acquisition.beta_iter.copy()
        previous_amm = self.preset.acquisition.Amm_iter.copy()

        x0 = []
        beta_indices = []
        blind_indices = []

        cursor = 0

        for i, comp in enumerate(self.preset.comp.components_name_out):
            if comp == "CMB":
                continue

            params = self.preset.comp.params_foregrounds[comp]

            # parametric
            if params["type"] == "parametric":
                x0.append(self.preset.acquisition.beta_iter[i-1])
                beta_indices.append((i, cursor))
                cursor += 1

            # blind
            else:
                Amm0 = self.preset.acquisition.Amm_iter[:, i]
                x0.extend(Amm0)
                blind_indices.append((i, cursor, len(Amm0)))
                cursor += len(Amm0)

        x0 = np.asarray(x0, dtype=float)

        layout = ParamLayout(
            beta_indices=beta_indices,
            blind_indices=blind_indices,
            ndim=len(x0),
        )

        self.chi2 = MixedChi2(self.preset, tod_comp, layout)

        res = minimize(
            self.chi2,
            x0,
            method="L-BFGS-B",
            callback=self.callback,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        beta, Amm = self.chi2.unpack(res.x)

        for comp, b in beta.items():
            self.preset.acquisition.beta_iter[comp-1] = b

        for comp, v in Amm.items():
            self.preset.acquisition.Amm_iter[:, comp] = v

        self._log(previous_beta, previous_amm)
        self._finalize()

    def _log(self, previous_beta, previous_amm):
        if self.preset.tools.rank !=0:
            return
        print("------------------- Beta -------------------")
        print(f"Iteration k     : {previous_beta}")
        print(f"Iteration k + 1 : {self.preset.acquisition.beta_iter}")
        print(f"Truth           : {self.preset.mixingmatrix.beta_in}")
        print(f"Residuals       : {self.preset.mixingmatrix.beta_in - self.preset.acquisition.beta_iter}")
        print("--------------------------------------------")
        print("--------------- MixingMatrix ---------------")
        print(f"Iteration k     : {previous_amm.ravel()}")
        print(f"Iteration k + 1 : {self.preset.acquisition.Amm_iter[: self.preset.qubic.joint_out.qubic.nsub, 1:].ravel()}")
        print(f"Truth           : {self.preset.mixingmatrix.Amm_in[: self.preset.qubic.joint_out.qubic.nsub, 1:].ravel()}")
        print(
            f"Residuals       : {self.preset.mixingmatrix.Amm_in[: self.preset.qubic.joint_out.qubic.nsub, 1:].ravel() - self.preset.acquisition.Amm_iter[: self.preset.qubic.joint_out.qubic.nsub, 1:].ravel()}"
        )

    def _finalize(self):
        self.preset.tools.comm.Barrier()
        
        # Beta
        self.preset.acquisition.allbeta = np.concatenate(
            (self.preset.acquisition.allbeta, np.array([self.preset.acquisition.beta_iter])),
            axis=0,
        )
        self.plots.plot_beta_iteration(
            self.preset.acquisition.allbeta,
            truth=self.preset.mixingmatrix.beta_in,
            ki=self._steps,
        )
        
        # Mixing Matrix
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
