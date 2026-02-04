import gc

import healpy as hp
import numpy as np
from scipy.optimize import minimize

from qubic.lib.MapMaking.ComponentMapMaking.mixing_matrix.fittingMM import FittingMM
from qubic.lib.MapMaking.ComponentMapMaking.Qchi2MM import Chi2


class ParametricMM(FittingMM):
    def update(self, tod_comp, beta_map=None):
        _, self.seenpix_beta = np.where(beta_map == hp.UNSEEN)

        previous_beta = self.preset.acquisition.beta_iter.copy()[:, self.seenpix_beta]

        self.chi2 = Chi2(self.preset, tod_comp, parametric=True, beta_map=beta_map)

        res = minimize(
            self.chi2,
            x0=self.preset.acquisition.beta_iter[:, self.seenpix_beta].ravel(),
            method="L-BFGS-B",
            callback=self.callback,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        self.preset.acquisition.beta_iter[:, self.seenpix_beta] = res.x

        self.preset.acquisition.Amm_iter = self.chi2.compute_mixing_matrix_parametric(
            nus=self.preset.qubic.joint_out.allnus,
            x=self.preset.acquisition.beta_iter,
        ).transpose((1, 0, 2))

        self._log(previous_beta)
        self._finalize()
        del tod_comp
        gc.collect()

    def _log(self, previous_beta):
        if self.preset.tools.rank != 0:
            return
        print(f"Iteration k     : {previous_beta}")
        print(f"Iteration k + 1 : {self.preset.acquisition.beta_iter[:, self.seenpix_beta]}")
        print(f"Truth           : {self.preset.mixingmatrix.beta_in[:, self.seenpix_beta]}")
        print(f"Residuals       : {self.preset.mixingmatrix.beta_in[:, self.seenpix_beta] - self.preset.acquisition.beta_iter[:, self.seenpix_beta]}")

    def _finalize(self):
        self.preset.tools.comm.Barrier()
        self.preset.acquisition.allbeta = np.concatenate(
            (self.preset.acquisition.allbeta, np.array([self.preset.acquisition.beta_iter])),
            axis=0,
        )

        self.plots.plot_beta_iteration(
            self.preset.acquisition.allbeta[..., self.seenpix_beta],
            truth=self.preset.mixingmatrix.beta_in[:, self.seenpix_beta],
            ki=self._steps,
        )
