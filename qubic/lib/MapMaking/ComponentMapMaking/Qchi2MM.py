import healpy as hp
import numpy as np
from pyoperators import MPI


def _dot(x, y, comm):
    d = np.array(np.dot(x.ravel(), y.ravel()))

    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, d)
    return d


class Chi2:
    def __init__(self, preset, TOD_sim, full_beta_map=None):
        self.preset = preset
        self.TOD_sim = TOD_sim
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

        self.TOD_sim_fp = []
        for i in range(self.nFP):
            self.TOD_sim_fp.append(self.TOD_sim[:, self.nsub * i : self.nsub * (i + 1)].reshape((self.ncomp * self.nsub * npix, self.nsampling_ndet)))
        self.TOD_sim_fp = np.array(self.TOD_sim_fp)

    def compute_mixing_matrix(self, x):
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

    def __call__(self, x):
        ### If constant spectral index
        if self.TOD_sim.ndim == 3:
            A = self.compute_mixing_matrix(x)

            ### Separe the mixing matrix element for 150 and 220 GHz if needed
            ysim = []
            for i in range(self.nFP):
                ysim.append(A[self.nsub * i : self.nsub * (i + 1)].T.reshape((self.ncomp * self.nsub)) @ self.TOD_sim_fp[i])

            ### Create simulated TOD
            ysim = np.concatenate(ysim, axis=0)

            ### Separe QUBIC and Planck
            Aext = A[self.nfreq :]

            # TODO : should we convolve here ? Tom : I think we should if convolutin_out==True, but fwhm_mapmaking is not adapted for that, we need to compute specific fwhm for planck
            H_planck = self.preset.qubic.joint_out.external.get_operator(A=Aext)

            ### Compute Planck part of the chi^2
            mycomp = self.preset.comp.components_iter.copy()
            mycomp[:, ~self.preset.sky.seenpix] = 0

            ysim_pl = H_planck(mycomp)

            ### Compute residuals in time domain
            _residuals = ysim - self.preset.acquisition.TOD_qubic
            self.Lqubic = 0.5 * _dot(_residuals.T, self.preset.acquisition.invN.operands[0](_residuals), self.preset.comm)

            _residuals_pl = np.r_[ysim_pl] - self.preset.acquisition.TOD_external_zero_outside_patch

            self.Lplanck = 0.5 * _dot(_residuals_pl.T, self.preset.acquisition.invN.operands[1](_residuals_pl), self.preset.comm)

            return self.Lqubic + self.Lplanck

        elif self.TOD_sim.ndim == 4:  # this implementation is exactly the same as for the DB, which feels wrong?
            # It is broken anyway

            x = x.reshape((self.ncomp - 1, self.npix))
            A = self._get_mixingmatrix(self.nus, x)

            ### Separe the mixing matrix element for 150 and 220 GHz
            Aq150 = A[:, : self.nsub, :].reshape((self.ncomp * self.nsub * self.npix))
            Aq220 = A[:, self.nsub : 2 * self.nsub, :].reshape((self.ncomp * self.nsub * self.npix))

            ### Create simulated TOD
            ysim = np.concatenate((Aq150 @ self.TOD_sim150, Aq220 @ self.TOD_sim220), axis=0)

            if self.parametric:
                ### Fill the full sky map of beta with the unknowns
                full_map_beta = self.full_beta_map.copy()
                x = x.reshape(((self.ncomp - 1) * self.npix))
                full_map_beta[self.seenpix_beta] = x

                ### Compute the mixing matrix for the full sky
                A = self._get_mixingmatrix(self.nus, full_map_beta)

                ### Separe QUBIC and Planck and switch axes
                Aext = np.transpose(A[:, 2 * self.nsub :, :], (1, 0, 2))

                H_planck = self.preset.qubic.joint_out.external.get_operator(A=Aext, convolution=False)

                ### Compute Planck part of the chi^2
                ysim_pl = H_planck(self.preset.comp.components_iter)

                ### Compute residuals in time domain
                _residuals = np.r_[ysim, ysim_pl] - self.TOD_obs

                return 0.5 * _dot(_residuals.T, self.preset.acquisition.invN(_residuals), self.preset.comm)

            else:
                raise TypeError("Varying mixing matrix along the LOS is not yet implemented")
        else:
            raise TypeError("dsim should have 3 or 4 dimensions.")
