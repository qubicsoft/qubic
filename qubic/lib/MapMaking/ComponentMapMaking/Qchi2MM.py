import fgbuster.mixingmatrix as mm
import healpy as hp
import numpy as np
from pyoperators import MPI


def _dot(x, y, comm):
    d = np.array(np.dot(x.ravel(), y.ravel()))

    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, d)
    return d


class ComponentChi2:
    def __init__(self, chi2, A_ref, icomp, mode):
        self.chi2 = chi2
        self.A_ref = A_ref
        self.icomp = icomp
        self.mode = mode
        self.A = np.empty_like(self.chi2.preset.acquisition.Amm_iter)

    def _parametric(self, x):
        """Return the parametric mixing matrix for this component only."""
        mixingmatrix = mm.MixingMatrix(*[self.chi2.preset.comp.components_model_out[self.icomp]])
        return mixingmatrix.eval(self.chi2.nus, *x)

    def _blind(self, x):
        """Return the blind mixing matrix column for this component."""
        nsub = self.chi2.fsub
        nbin = self.chi2.preset.comp.params_foregrounds["bin_mixing_matrix"]
        A_col = np.ones(self.chi2.nfreq)
        for ibin in range(nbin):
            start = ibin * nsub
            end = (ibin + 1) * nsub
            A_col[start:end] = x[ibin]
        return A_col

    def __call__(self, x):
        """Compute chi² for this component given parameters x."""
        if self.mode == "parametric":
            self.A[:, self.icomp] = self._parametric(x)
        else:
            self.A[:, self.icomp] = self._blind(x)

        return self.chi2.compute_chi_square_with_A(self.A, self.icomp)


class Chi2:
    def __init__(self, preset, TOD_sim, parametric=True, full_beta_map=None):
        self.preset = preset
        self.TOD_sim = TOD_sim
        self.parametric = parametric
        self.nus = self.preset.qubic.joint_out.allnus
        self.full_beta_map = full_beta_map

        ### If constant spectral index
        if self.TOD_sim.ndim == 3:  # Can be concatenated with the case self.TOD_sim.ndim == 4 by adding npix = 1 or npix = self.npix
            npix = 1
        ### If varying spectral indices
        elif self.TOD_sim.ndim == 4:
            self.seenpix_beta = np.where(self.full_beta_map == hp.UNSEEN)
            npix = self.npix
        else:
            raise TypeError("TOD_sim should have 3 or 4 dimensions.")

        self.nFP = self.preset.qubic.joint_out.qubic.nFocalPlanes
        self.ncomp, self.nfreq, self.nsampling_ndet = self.TOD_sim.shape
        self.nsub = self.nfreq // self.nFP
        self.fsub = self.nfreq // self.preset.comp.params_foregrounds["bin_mixing_matrix"]
        self.Lplanck = 0

        self.TOD_sim_fp = []
        for i in range(self.nFP):
            self.TOD_sim_fp.append(self.TOD_sim[:, self.nsub * i : self.nsub * (i + 1)].reshape((self.ncomp * self.nsub * npix, self.nsampling_ndet)))
        self.TOD_sim_fp = np.array(self.TOD_sim_fp)

    def compute_chi_square_with_A(self, A, icomp):
        ysim = []
        for i in range(self.nFP):
            print("A", A.shape)
            print(self.TOD_sim_fp.shape)
            ysim.append(A[icomp, self.nsub * i : self.nsub * (i + 1)].T @ self.TOD_sim_fp[i])

        ysim = np.concatenate(ysim, axis=0)
        residuals = ysim - self.preset.acquisition.TOD_qubic

        self.Lqubic = 0.5 * _dot(
            residuals.T,
            self.preset.acquisition.invN.operands[0](residuals),
            self.preset.comm,
        )

        return self.Lqubic

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

    def compute_chi_square(self, x):
        ### If constant spectral index
        if self.TOD_sim.ndim == 3:
            # If parametric -> we compute the mixing matrix element according to the spectral index
            if self.parametric:
                print("x", x)
                A = self.compute_mixing_matrix_parametric(self.nus, x)

            # If blind -> we treat the mixing matrix element as free parameters
            else:
                A = self.compute_mixing_matrix_blind(x)

            ### Separe the mixing matrix element for 150 and 220 GHz if needed
            ysim = []
            for i in range(self.nFP):
                ysim.append(A[self.nsub * i : self.nsub * (i + 1)].T.reshape((self.ncomp * self.nsub)) @ self.TOD_sim_fp[i])

            ### Create simulated TOD
            ysim = np.concatenate(ysim, axis=0)

            ### Compute residuals in time domain
            _residuals = ysim - self.preset.acquisition.TOD_qubic

            self.Lqubic = 0.5 * _dot(_residuals.T, self.preset.acquisition.invN.operands[0](_residuals), self.preset.comm)

            if self.parametric:
                # Note: We can use Planck in the Mixing Matrix only for the parametric case !
                ### Separe QUBIC and Planck
                Aext = A[self.nfreq :]

                # TODO : should we convolve here ? Tom : I think we should if convolutin_out==True, but fwhm_mapmaking is not adapted for that, we need to compute specific fwhm for planck
                H_planck = self.preset.qubic.joint_out.external.get_operator(A=Aext)

                ### Compute Planck part of the chi^2
                mycomp = self.preset.comp.components_iter.copy()
                mycomp[:, ~self.preset.sky.seenpix] = 0

                ysim_pl = H_planck(mycomp)

                _residuals_pl = np.r_[ysim_pl] - self.preset.acquisition.TOD_external_zero_outside_patch

                self.Lplanck = 0.5 * _dot(_residuals_pl.T, self.preset.acquisition.invN.operands[1](_residuals_pl), self.preset.comm)

            return self.Lqubic, self.Lplanck

        elif self.TOD_sim.ndim == 4:
            raise ValueError("d1 model is not implemented.")

        else:
            raise TypeError("dsim should have 3 or 4 dimensions.")

    def __call__(self, x):
        Lqubic, Lplanck = self.compute_chi_square(x)

        L = Lqubic
        if self.parametric:
            L += Lplanck

        return L
