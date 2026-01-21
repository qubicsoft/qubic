import gc

import numpy as np
from scipy.optimize import minimize

from qubic.lib.MapMaking.ComponentMapMaking.mixing_matrix.fittingMM import FittingMM
from qubic.lib.MapMaking.ComponentMapMaking.Qchi2MM import Chi2, ComponentChi2


class MixedMM(FittingMM):
    def update(self, tod_comp):
        previous_step = self.preset.acquisition.Amm_iter[: self.preset.qubic.joint_out.qubic.nsub, 1:].copy()

        if self.selfCMM.allAmm_iter is None:
            self.selfCMM.allAmm_iter = np.array([self.preset.acquisition.Amm_iter])

        for i, comp in enumerate(self.preset.comp.components_name_out):
            if comp == "CMB":
                continue

            params = self.preset.comp.params_foregrounds[comp]
            self.chi2 = Chi2(self.preset, tod_comp, parametric=False)

            # ==========================
            # PARAMETRIC
            # ==========================
            if params["type"] == "parametric":
                if self.preset.tools.rank == 0:
                    print(f"Fitting {comp} with parametric method (component index {i})")

                obj = ComponentChi2(
                    self.chi2,
                    A_ref=self.preset.acquisition.beta_iter,
                    icomp=i,
                    mode="parametric",
                )

                x0 = np.array([float(self.preset.acquisition.beta_iter[i])])

                bounds = None
                if "beta_bounds" in params:
                    bounds = [tuple(params["beta_bounds"])]

                res = minimize(
                    obj,
                    x0=x0,
                    method="L-BFGS-B",
                    bounds=bounds,
                    callback=self.callback,
                    options={"ftol": 1e-9},
                )

                if not res.success:
                    print(f"Warning: parametric fit for {comp} did not converge: {res.message}")

                beta_fitted = float(res.x.item())
                self.preset.acquisition.beta_iter[i] = beta_fitted
                self.preset.acquisition.Amm_iter[:, i] = obj._parametric(beta_fitted)

            # ==========================
            # BLIND
            # ==========================
            else:
                if self.preset.tools.rank == 0:
                    print(f"Fitting {comp} with blind method (component index {i})")

                obj = ComponentChi2(
                    self.chi2,
                    A_ref=self.preset.acquisition.Amm_iter,
                    icomp=i,
                    mode="blind",
                )

                x0 = []
                for ibin in range(self.preset.comp.params_foregrounds["bin_mixing_matrix"]):
                    mean_val = np.mean(self.preset.acquisition.Amm_iter[ibin * self.fsub : (ibin + 1) * self.fsub, i])
                    x0.append(mean_val)
                x0 = np.asarray(x0, dtype=float)

                bounds = None
                if params.get("blind_nonnegative", False):
                    bounds = [(0.0, None) for _ in range(len(x0))]

                res = minimize(
                    obj,
                    x0=x0,
                    method="SLSQP",
                    bounds=bounds,
                    callback=self.callback,
                    tol=1e-10,
                )

                if not res.success:
                    print(f"Warning: blind fit for {comp} did not converge: {res.message}")

                new_column = obj._blind_column(res.x)
                assert new_column.shape[0] == self.preset.acquisition.Amm_iter.shape[0], "Blind fit returned column with wrong length"
                self.preset.acquisition.Amm_iter[:, i] = new_column

        self._log(previous_step)
        self._finalize()

        del tod_comp
        gc.collect()
