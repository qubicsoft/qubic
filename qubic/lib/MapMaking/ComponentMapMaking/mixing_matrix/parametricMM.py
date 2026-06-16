import gc

import numpy as np
from scipy.optimize import minimize

from qubic.lib.MapMaking.ComponentMapMaking.mixing_matrix.fittingMM import FittingMM
from qubic.lib.MapMaking.ComponentMapMaking.Qchi2MM import Chi2


class ParametricMM(FittingMM):
    def update(self, tod_comp, beta_map=None):
        if beta_map is not None:
            self.seenpix_beta = self.preset.mixingmatrix._index_seenpix_beta
        else:
            self.seenpix_beta = None
        previous_beta = self.preset.acquisition.beta_iter.copy()[:, self.seenpix_beta]

        if beta_map is not None:
            margin = self.preset.comp.params_foregrounds["Dust"].get("beta_bounds_margin", 0.5)
            beta_true = self.preset.mixingmatrix.beta_in[:, self.seenpix_beta].ravel()
            bounds = [(b - margin, b + margin) for b in beta_true]
        else:
            bounds = None

        self.chi2 = Chi2(self.preset, tod_comp, parametric=True, beta_map=beta_map)

        beta_prior_sigma = float(
            self.preset.comp.params_foregrounds["Dust"].get("beta_prior_sigma", 0)
        )
        beta_prior_mean = float(self.preset.comp.params_foregrounds["Dust"]["beta_init"][0])

        if beta_prior_sigma > 0:
            prior_weight = 1.0 / beta_prior_sigma**2
            def _obj(x):
                return self.chi2(x) + 0.5 * prior_weight * np.sum((x - beta_prior_mean) ** 2)
        else:
            _obj = self.chi2

        res = minimize(
            _obj,
            x0=self.preset.acquisition.beta_iter[:, self.seenpix_beta].ravel(),
            method="L-BFGS-B",
            jac=None,
            callback=self.callback,
            bounds=bounds,
            options={"eps": 1e-6, "maxls": 20, "maxiter": 20},
        )

        self.preset.acquisition.beta_iter[:, self.seenpix_beta] = res.x

        A = self.chi2.compute_mixing_matrix_parametric(
            nus=self.preset.qubic.joint_out.allnus,
            x=self.preset.acquisition.beta_iter,
        )
        self.preset.acquisition.Amm_iter = A.transpose((1, 0, 2)) if A.ndim == 3 else A

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
        print(
            f"Residuals       : {self.preset.mixingmatrix.beta_in[:, self.seenpix_beta] - self.preset.acquisition.beta_iter[:, self.seenpix_beta]}"
        )

    def _finalize(self):
        self.preset.tools.comm.Barrier()
        self.preset.acquisition.allbeta = np.concatenate(
            (self.preset.acquisition.allbeta, np.array([self.preset.acquisition.beta_iter])),
            axis=0,
        )

        if self.preset.tools.rank == 0:
            self.plots.plot_beta_iteration(
                self.preset.acquisition.allbeta[..., self.seenpix_beta],
                truth=self.preset.mixingmatrix.beta_in[:, self.seenpix_beta],
                ki=self._steps,
            )
