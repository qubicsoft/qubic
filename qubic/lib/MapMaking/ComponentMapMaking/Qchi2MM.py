from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import fgbuster.mixingmatrix as mm
import healpy as hp
import numpy as np
from pyoperators import MPI


@dataclass
class ParamLayout:
    beta_indices: list  # [(comp_index, position_in_x)]
    blind_indices: list  # [(comp_index, start, length)]
    ndim: int


def _dot(x, y, comm):
    d = np.array(np.dot(x.ravel(), y.ravel()))

    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, d)
    return d


class AbstractChi2(ABC):
    """
    Common base class for chi2-like objects.
    Subclasses must implement compute_mixing_matrix(x) and, if needed,
    compute_chi_square(x). A helper compute_qubic_chi(A) is provided
    because both original classes shared that logic.
    """

    def __init__(self, preset, TOD_sim, layout: Optional[ParamLayout] = None, beta_map: Optional[np.ndarray] = None):
        self.preset = preset
        self.TOD_sim = TOD_sim
        self.layout = layout
        self.beta_map = beta_map

        # Deduce shapes for 3D or 4D TOD_sim
        if self.TOD_sim.ndim == 3:
            # (ncomp, nfreq, nsampling_ndet)
            self.ncomp, self.nfreq, self.nsampling_ndet = self.TOD_sim.shape
            self.npix = 1
        elif self.TOD_sim.ndim == 4:
            # (ncomp, nfreq, npix, nsampling_ndet)
            self.ncomp, self.nfreq, self.npix, self.nsampling_ndet = self.TOD_sim.shape
            self.seenpix_beta = self.beta_map == hp.UNSEEN

        else:
            raise TypeError("TOD_sim should have 3 or 4 dimensions.")

        self.nus = preset.qubic.joint_out.allnus
        self.nFP = preset.qubic.joint_out.qubic.nFocalPlanes
        self.nsub = self.nfreq // self.nFP
        self.fsub = self.nfreq // self.preset.comp.params_foregrounds["bin_mixing_matrix"]

        # Pre-definition of Planck Likelihood
        self.Lplanck = 0

        # Build per-focal-plane TOD_sim
        self.TOD_sim_fp = []
        for i in range(self.nFP):
            block = self.TOD_sim[:, self.nsub * i : self.nsub * (i + 1)]
            if block.ndim == 3:  # 3D block: (ncomp, nsub, nsampling_ndet)
                resh = block.reshape((self.ncomp * self.nsub * 1, self.nsampling_ndet))
            else:  # 4D block: (ncomp, nsub, npix, nsampling_ndet)
                resh = block.reshape((self.ncomp * self.nsub * self.npix, self.nsampling_ndet))
            self.TOD_sim_fp.append(resh)
        self.TOD_sim_fp = np.asarray(self.TOD_sim_fp)

    @abstractmethod
    def compute_mixing_matrix(self, x) -> np.ndarray:
        """Return mixing matrix A with shape (nfreq, ncomp)."""
        raise NotImplementedError

    def compute_qubic_chi(self, A):
        """
        Shared computation of the QUBIC time-domain chi^2 given mixing
        matrix A. Returns Lqubic and stores it on the instance.
        """
        
        # Select seen super_pixels
        A_seen = A[self.seenpix_beta[0]]  # (npix, nsub*nFP, ncomp)
        A_seen = A_seen.reshape(
            A_seen.shape[0], self.nFP, self.nsub, self.ncomp
        ).transpose(1, 2, 0, 3)  # (nFP, nsub, npix, ncomp)
        A_flat = A_seen.reshape(self.nFP, -1)  # (nFP, nsub*npix*ncomp)

        ysim = np.concatenate([
            A_flat[i] @ self.TOD_sim_fp[i]
            for i in range(self.nFP)
        ])

        residuals = ysim - self.preset.acquisition.TOD_qubic

        Lqubic = 0.5 * _dot(
            residuals.T,
            self.preset.acquisition.invN.operands[0](residuals),
            self.preset.comm,
        )

        return Lqubic

    def compute_qubic_chi_varying_beta(self, A):
        ysim_parts = []
        for i in range(self.nFP):
            a_slice = A[self.seenpix_beta[0], self.nsub * i : self.nsub * (i + 1)]  # (npix, nsub, ncomp)
            vec = a_slice.T.reshape(self.ncomp * self.nsub * a_slice.shape[0]) @ self.TOD_sim_fp[i]
            ysim_parts.append(vec)

        ysim = np.concatenate(ysim_parts, axis=0)
        residuals = ysim - self.preset.acquisition.TOD_qubic

        Lqubic = 0.5 * _dot(
            residuals.T,
            self.preset.acquisition.invN.operands[0](residuals),
            self.preset.comm,
        )

        return Lqubic

    # default call: subclasses may override if they need to alter behavior
    def __call__(self, x):
        Lqubic = self.compute_qubic_chi(self.compute_mixing_matrix(x))
        # allow subclasses to add Lplanck if needed (they should set self.Lplanck)
        return Lqubic + getattr(self, "Lplanck", 0)


class MixedChi2(AbstractChi2):
    def __init__(self, preset, TOD_sim, layout: ParamLayout):
        super().__init__(preset, TOD_sim, layout=layout)
        self.layout = layout

    # packing
    def unpack(self, x):
        beta = {}
        Amm = {}

        for comp, idx in self.layout.beta_indices:
            beta[comp] = x[idx]

        for comp, start, n in self.layout.blind_indices:
            Amm[comp] = x[start : start + n]

        return beta, Amm

    def compute_mixing_matrix(self, x):
        beta, Amm = self.unpack(x)
        A = self.preset.acquisition.Amm_iter.copy()

        # parametric
        for comp, b in beta.items():
            model = mm.MixingMatrix(self.preset.comp.components_model_out[comp - 1])
            A[:, comp] = model.eval(self.nus, b)[:, 0]

        # blind
        for comp, v in Amm.items():
            A[:, comp] = v

        return A

    def __call__(self, x):
        A = self.compute_mixing_matrix(x)
        return self.compute_qubic_chi(A)


class Chi2(AbstractChi2):
    def __init__(self, preset, TOD_sim, parametric=True, beta_map=None):
        # original class raised on 4D TOD; keep that behaviour explicit
        super().__init__(preset, TOD_sim, beta_map=beta_map)
        self.parametric = parametric
        self.beta_map = beta_map

    def compute_mixing_matrix_parametric(self, nus, x):
        """
        Parametric case
        """

        mixingmatrix = mm.MixingMatrix(*self.preset.comp.components_model_out)
        return mixingmatrix.eval(nus, *x)

    def compute_mixing_matrix_blind(self, x):
        """
        Blind case
        """
        A = np.ones((self.nfreq, self.ncomp))
        k = 0
        for i in range(self.preset.comp.params_foregrounds["bin_mixing_matrix"]):
            for j in range(1, self.ncomp):
                A[i * self.fsub : (i + 1) * self.fsub, j] = np.array([x[k]] * self.fsub)
                k += 1
        return A

    def compute_mixing_matrix(self, x):
        if self.parametric:
            return self.compute_mixing_matrix_parametric(self.nus, x)
        else:
            return self.compute_mixing_matrix_blind(x)

    def compute_chi_square_fix_beta(self, x):
        A = self.compute_mixing_matrix(x)
        self.Lqubic = self.compute_qubic_chi(A)

        self.Lplanck = 0
        if self.parametric:
            Aext = A[self.nfreq :]

            # TODO : should we convolve here ? Tom : I think we should if convolutin_out==True, but fwhm_mapmaking is not adapted for that, we need to compute specific fwhm for planck
            H_planck = self.preset.qubic.joint_out.external.get_operator(A=Aext)

            comp = self.preset.comp.components_iter.copy()
            comp[:, ~self.preset.sky.seenpix] = 0

            ysim_pl = H_planck(comp)
            _residuals_pl = np.r_[ysim_pl] - self.preset.acquisition.TOD_external_zero_outside_patch

            self.Lplanck = 0.5 * _dot(
                _residuals_pl.T,
                self.preset.acquisition.invN.operands[1](_residuals_pl),
                self.preset.comm,
            )

        return self.Lqubic, self.Lplanck

    def compute_chi_square_varying_beta(self, x):
        ### Fill the full sky map of beta with the unknowns
        beta_map = self.beta_map.copy()
        x = x.reshape(((self.ncomp - 1) * self.npix))
        beta_map[self.seenpix_beta] = x

        ### Compute the mixing matrix for the full sky
        A = self.compute_mixing_matrix_parametric(self.nus, beta_map)
        Aext = A[:, self.nfreq :]

        ### Qubic chi2
        self.Lqubic = self.compute_qubic_chi_varying_beta(A)

        ### Planck chi2
        H_planck = self.preset.qubic.joint_out.external.get_operator(A=Aext.transpose(1, 0, 2))
        ysim_pl = H_planck(self.preset.comp.components_iter.copy())
        residuals_pl = np.r_[ysim_pl] - self.preset.acquisition.TOD_external
        self.Lplanck = 0.5 * _dot(residuals_pl.T, self.preset.acquisition.invN.operands[1](residuals_pl), self.preset.comm)

        return self.Lqubic, self.Lplanck

    def compute_chi_square(self, x):
        if self.TOD_sim.ndim == 3:
            return self.compute_chi_square_fix_beta(x)
        elif self.TOD_sim.ndim == 4:
            return self.compute_chi_square_varying_beta(x)
        else:
            raise TypeError("TOD_sim should have 3 or 4 dimensions.")

    def __call__(self, x):
        Lqubic, Lplanck = self.compute_chi_square(x)
        L = Lqubic
        if self.parametric:
            L += Lplanck
        return L
