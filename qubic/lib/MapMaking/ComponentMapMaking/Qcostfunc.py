from pyoperators import *
import fgbuster.mixingmatrix as mm
from ...Qfoldertools import *


def _dot(x, y, comm):
    d = np.array(np.dot(x.ravel(), y.ravel()))

    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, d)
    return d

class Chi2InstrumentType:

    def __init__(self, preset, dsim, instr_type, parametric=True, full_beta_map=None):

        self.preset = preset
        self.dsim = dsim
        self.nus = self.preset.qubic.joint_out.allnus
        self.parametric = parametric
        self.full_beta_map = full_beta_map

        if instr_type == "DB": # this will later be implemented at a dictionary level!
            self.nFocalPlanes = 2
        elif instr_type == "UWB":
            self.nFocalPlanes = 1
        elif instr_type == "MB":
            self.nFocalPlanes = 1
        else:
            raise ValueError("Instrument type {} is not implemented.".format(instr_type))
        
        ### If parametric, we use the QUBIC + Planck data
        if self.parametric:
            if self.dsim.ndim == 3:
                self.dobs = self.preset.acquisition.TOD_obs_zero_outside.copy()
            elif self.dsim.ndim == 4:
                self.dobs = self.preset.acquisition.TOD_obs.copy()
            # self.dobs = self.preset.acquisition.TOD_qubic.copy()
        ### If blind, we use the QUBIC data only
        else:
            self.dobs = self.preset.acquisition.TOD_qubic.copy()

        ### If constant spectral index # diff ici
        if self.dsim.ndim == 3: # Can be concatenated with the case self.dsim.ndim == 4 by adding npix = 1 or npix = self.npix
            npix = 1
        ### If varying spectral indices
        elif self.dsim.ndim == 4:
            self.seenpix_beta = np.where(self.full_beta_map == hp.UNSEEN)
            npix = self.npix
        else:
            raise TypeError("dsim should have 3 or 4 dimensions.")

        self.nc, self.nf, self.nsnd = self.dsim.shape
        self.nsub = int(self.nf / self.nFocalPlanes)

        self.dsim_fp = []
        for i in range(self.nFocalPlanes):
            # print("\nWhere tuple?")
            # print(self.nsub)
            # print(self.nsub*i)
            self.dsim_fp.append(self.dsim[:, self.nsub*i : self.nsub*(i+1)].reshape((self.nc * self.nsub * npix, self.nsnd)))
        # Missing self.dsim150 and self.dsim220 with new definition, use self.dsim_fp[0] and self.dsim_fp[1] instead

    def _fill_A(self, x): # idem (garder la version UWB, fonctionne mieux)

        fsub = int(self.nf / self.preset.comp.params_foregrounds["bin_mixing_matrix"])
        A = np.ones((self.nf, self.nc))
        k = 0
        for i in range(self.preset.comp.params_foregrounds["bin_mixing_matrix"]):
            for j in range(1, self.nc):
                A[i * fsub : (i + 1) * fsub, j] = np.array([x[k]] * fsub)
                k += 1
        return A

    def _get_mixingmatrix(self, nus, x): # idem

        ### Compute mixing matrix
        mixingmatrix = mm.MixingMatrix(*self.preset.comp.components_model_out)
        return mixingmatrix.eval(nus, *x)

    def __call__(self, x):

        ### If constant spectral index
        if self.dsim.ndim == 3:

            ### If parametric -> we compute the mixing matrix element according to the spectral index
            if self.parametric:
                A = self._get_mixingmatrix(self.nus, x)
            ### If blind -> we treat the mixing matrix element as free parameters
            else:
                ### Fill mixing matrix for the FG components
                A = self._fill_A(x)
            # print(A.shape)
            # stop
            ### Separe the mixing matrix element for 150 and 220 GHz if needed

            ysim = []
            for i in range(self.nFocalPlanes):
                ysim.append(A[self.nsub*i : self.nsub*(i + 1)].T.reshape((self.nc * self.nsub)) @ self.dsim_fp[i])

            ### Create simulated TOD
            ysim = np.concatenate(ysim, axis=0)

            if self.parametric:
                ### Separe QUBIC and Planck
                Aext = A[self.nf :].copy()

                H_planck = self.preset.qubic.joint_out.external.get_operator(
                    A=Aext, convolution=False
                )

                ### Compute Planck part of the chi^2
                mycomp = self.preset.comp.components_iter.copy()
                mycomp[:, ~self.preset.sky.seenpix_qubic, :] = 0
                
                ysim_pl = H_planck(mycomp)
                
                ### Compute residuals in time domain
                _residuals = np.r_[ysim] - self.preset.acquisition.TOD_qubic
                self.Lqubic = _dot(
                    _residuals.T,
                    self.preset.acquisition.invN.operands[0](_residuals),
                    self.preset.comm,
                )

                _residuals_pl = (
                    np.r_[ysim_pl]
                    - self.preset.acquisition.TOD_external_zero_outside_patch
                )
                
                self.Lplanck = _dot(
                    _residuals_pl.T,
                    self.preset.acquisition.invN.operands[1](_residuals_pl),
                    self.preset.comm,
                )
                return self.Lqubic + self.Lplanck
            else:
                ### Compute residuals in time domain
                _residuals = ysim - self.dobs

                return _dot(
                    _residuals.T,
                    self.preset.acquisition.invN.operands[0](_residuals),
                    self.preset.comm,
                )
                # Why so muchdifferent for DB?
                _residuals = ysim - self.preset.acquisition.TOD_qubic
                self.Lplanck = 0
                self.Lqubic = _dot( _residuals.T, self.preset.acquisition.invN.operands[0](_residuals), self.preset.comm)
                return self.Lqubic
        elif self.dsim.ndim == 4: # this implementation is exactly the same as for the DB, which feels wrong?
            # It is broken anyway

            # print(x, x.shape, self.nc-1, self.npix)
            x = x.reshape((self.nc - 1, self.npix))
            A = self._get_mixingmatrix(self.nus, x)

            ### Separe the mixing matrix element for 150 and 220 GHz
            Aq150 = A[:, : self.nsub, :].reshape((self.nc * self.nsub * self.npix))
            Aq220 = A[:, self.nsub : 2 * self.nsub, :].reshape(
                (self.nc * self.nsub * self.npix)
            )

            ### Create simulated TOD
            ysim = np.concatenate((Aq150 @ self.dsim150, Aq220 @ self.dsim220), axis=0)

            if self.parametric:

                ### Fill the full sky map of beta with the unknowns
                full_map_beta = self.full_beta_map.copy()
                x = x.reshape(((self.nc - 1) * self.npix))
                full_map_beta[self.seenpix_beta] = x

                ### Compute the mixing matrix for the full sky
                A = self._get_mixingmatrix(self.nus, full_map_beta)

                ### Separe QUBIC and Planck and switch axes
                Aext = np.transpose(A[:, 2 * self.nsub :, :], (1, 0, 2))

                H_planck = self.preset.qubic.joint_out.external.get_operator(
                    A=Aext, convolution=False
                )

                ### Compute Planck part of the chi^2
                # mycomp = self.preset.comp.components_iter.copy()
                # seenpix_comp = np.tile(self.preset.sky.seenpix_qubic, (mycomp.shape[0], 3, 1)).reshape(mycomp.shape)
                ysim_pl = H_planck(self.preset.comp.components_iter.copy())

                ### Compute residuals in time domain
                _residuals = np.r_[ysim, ysim_pl] - self.dobs

                return _dot(
                    _residuals.T,
                    self.preset.acquisition.invN(_residuals),
                    self.preset.comm,
                )
            else:
                raise TypeError(
                    "Varying mixing matrix along the LOS is not yet implemented"
                )
        else:
            raise TypeError("dsim should have 3 or 4 dimensions.")

class Chi2Parametric:

    def __init__(self, preset, d, betamap, seenpix_wrap=None):

        self.preset = preset
        self.d = d  # shape -> (ncomp, Nsub, NsNd)
        self.betamap = betamap

        if np.ndim(self.d) == 3:
            self.nc, self.nf, self.nsnd = self.d.shape
            self.constant = True
        else:

            if self.preset.qubic.params_qubic["instrument"] == "UWB":
                pass
            else:
                self.nf = self.d.shape[1]
                self.d150 = self.d[:, : int(self.nf / 2)].copy()
                self.d220 = self.d[:, int(self.nf / 2) : int(self.nf)].copy()
                _sh = self.d150.shape
                _rsh = ReshapeOperator(
                    self.d150.shape, (_sh[0] * _sh[1], _sh[2], _sh[3])
                )
                self.d150 = _rsh(self.d150)
                self.d220 = _rsh(self.d220)
                self.dcmb150 = np.sum(self.d150[:, 0, :], axis=0).copy()
                self.dfg150 = self.d150[:, 1, :].copy()
                self.dcmb220 = np.sum(self.d220[:, 0, :], axis=0).copy()
                self.dfg220 = self.d220[:, 1, :].copy()
                self.npixnf, self.nc, self.nsnd = self.d150.shape

            index_num = hp.ud_grade(
                self.preset.sky.seenpix_qubic,
                self.preset.comp.params_foregrounds["Dust"]["nside_beta_out"],
            )  #
            index = np.where(index_num == True)[0]
            self._index = index
            self.seenpix_wrap = seenpix_wrap
            self.constant = False

    def _get_mixingmatrix(self, nus, x):
        mixingmatrix = mm.MixingMatrix(*self.preset.comp.components_model_out)

        # if self.constant:
        return mixingmatrix.eval(nus, *x)
        # else:
        #    return mixingmatrix.eval(nus, x)

    def __call__(self, x):
        if self.constant:
            A = self._get_mixingmatrix(self.preset.qubic.joint_out.qubic.allnus, x)
            self.betamap = x.copy()

            if self.preset.qubic.params_qubic["instrument"] == "UWB":
                ysim = np.zeros(self.nsnd)
                for ic in range(self.nc):
                    ysim += A[:, ic] @ self.d[ic]
            else:
                ysim = np.zeros(int(self.nsnd * 2))
                # print(A.shape)
                # stop
                for ic in range(self.nc):
                    ysim[: int(self.nsnd)] += (
                        A[: int(self.nf / 2), ic] @ self.d[ic, : int(self.nf / 2)]
                    )
                    ysim[int(self.nsnd) : int(self.nsnd * 2)] += (
                        A[int(self.nf / 2) : int(self.nf), ic]
                        @ self.d[ic, int(self.nf / 2) : int(self.nf)]
                    )
        else:
            if self.seenpix_wrap is None:
                self.betamap[self._index, 0] = x.copy()
            else:
                self.betamap[self.seenpix_wrap, 0] = x.copy()

            if self.preset.qubic.params_qubic["instrument"] == "UWB":
                ysim = np.zeros(self.nsnd)
                for ic in range(self.nc):
                    for ip, p in enumerate(self._index):
                        ysim += A[ip, :, ic] @ self.d[ip, :, ic]
            else:
                ysim = np.zeros(int(self.nsnd * 2))
                Atot = self._get_mixingmatrix(self.betamap[self._index])
                A150 = Atot[:, 0, : int(self.nf / 2), 1].ravel()
                A220 = Atot[:, 0, int(self.nf / 2) : int(self.nf), 1].ravel()

                ysim[: int(self.nsnd)] = (A150 @ self.dfg150) + self.dcmb150
                ysim[int(self.nsnd) : int(self.nsnd * 2)] = (
                    A220 @ self.dfg220
                ) + self.dcmb220

        _r = ysim - self.preset.acquisition.TOD_qubic
        H_planck = self.preset.qubic.joint_out.get_operator(
            self.betamap,
            gain=self.preset.gain.gain_iter,
            fwhm=self.preset.acquisition.fwhm_mapmaking,
            nu_co=self.preset.comp.nu_co,
        ).operands[1]
        tod_pl_s = H_planck(self.preset.comp.components_iter)

        _r_pl = self.preset.acquisition.TOD_external - tod_pl_s

        LLH = _dot(
            _r.T, self.preset.acquisition.invN.operands[0](_r), self.preset.comm
        ) + _r_pl.T @ self.preset.acquisition.invN.operands[1](_r_pl)
        # LLH = _r.T @ self.preset.acquisition.invN.operands[0](_r)

        # return _dot(_r.T, self.preset.acquisition.invN.operands[0](_r), self.preset.comm) + _r_pl.T @ self.preset.acquisition.invN.operands[1](_r_pl)
        return LLH


class Chi2Parametric_alt:

    def __init__(self, preset, d, A_blind, icomp, seenpix_wrap=None):

        self.preset = preset
        self.d = d
        # self.betamap = betamap
        self.A_blind = A_blind
        self.icomp = icomp
        self.nsub = self.preset.qubic.joint_out.qubic.Nsub
        self.fsub = int(
            self.nsub * 2 / self.preset.comp.params_foregrounds["bin_mixing_matrix"]
        )
        self.nc = len(self.preset.comp.components_out)

        self.constant = True

        # if np.ndim(self.d) == 3:
        #     self.nc, self.nf, self.nsnd = self.d.shape
        #     self.constant = True
        # else:

        # if self.preset.qubic.params_qubic['instrument'] == 'UWB':
        #     pass
        # else:
        #     self.nf = self.d.shape[1]
        #     self.d150 = self.d[:, :int(self.nf/2)].copy()
        #     self.d220 = self.d[:, int(self.nf/2):int(self.nf)].copy()
        #     _sh = self.d150.shape
        #     _rsh = ReshapeOperator(self.d150.shape, (_sh[0]*_sh[1], _sh[2], _sh[3]))
        #     self.d150 = _rsh(self.d150)
        #     self.d220 = _rsh(self.d220)
        #     self.dcmb150 = np.sum(self.d150[:, 0, :], axis=0).copy()
        #     self.dfg150 = self.d150[:, 1, :].copy()
        #     self.dcmb220 = np.sum(self.d220[:, 0, :], axis=0).copy()
        #     self.dfg220 = self.d220[:, 1, :].copy()
        #     self.npixnf, self.nc, self.nsnd = self.d150.shape

        #     index_num = hp.ud_grade(self.preset.sky.seenpix_qubic, self.preset.comp.params_foregrounds['Dust']['nside_fit'])    #
        #     index = np.where(index_num == True)[0]
        #     self._index = index
        #     self.seenpix_wrap = seenpix_wrap
        #     self.constant = False

    def _get_mixingmatrix(self, x):
        mixingmatrix = mm.MixingMatrix(self.preset.comp.components_out[self.icomp])
        if self.constant:
            return mixingmatrix.eval(self.preset.qubic.joint_out.qubic.allnus, *x)
        else:
            return mixingmatrix.eval(self.preset.qubic.joint_out.qubic.allnus, x)

    def get_mixingmatrix_comp(self, x):
        A_comp = self._get_mixingmatrix(x)
        A_blind = self.A_blind
        print("test", A_comp.shape, A_blind.shape)
        for ii in range(self.preset.comp.params_foregrounds["bin_mixing_matrix"]):
            A_blind[ii * self.fsub : (ii + 1) * self.fsub, self.icomp] = A_comp[
                ii * self.fsub : (ii + 1) * self.fsub
            ]
        return A_blind

    def __call__(self, x):

        if self.constant:
            if self.preset.qubic.params_qubic["instrument"] == "DB":
                ### CMB contribution
                tod_cmb_150 = np.sum(self.d[0, : self.nsub, :], axis=0)
                tod_cmb_220 = np.sum(self.d[0, self.nsub : 2 * self.nsub, :], axis=0)

                ###Â FG contributions
                tod_comp_150 = self.d[1:, : self.nsub, :].copy()
                tod_comp_220 = self.d[1:, self.nsub : 2 * self.nsub, :].copy()

                ### Describe the data as d = d_cmb + A . d_fg
                d_150 = tod_cmb_150.copy()
                d_220 = tod_cmb_220.copy()
            A = self.get_mixingmatrix_comp(x)

            for i in range(self.nc - 1):
                for j in range(self.nsub):
                    d_150 += A[: self.nsub, (i + 1)][j] * tod_comp_150[i, j]
                    d_220 += (
                        A[self.nsub : self.nsub * 2, (i + 1)][j] * tod_comp_220[i, j]
                    )

            ### Residuals
            _r = np.r_[d_150, d_220] - self.preset.acquisition.TOD_qubic

            ### Chi^2
            self.chi2 = _dot(
                _r.T, self.preset.acquisition.invN.operands[0](_r), self.preset.comm
            )

        # else:
        #     if self.seenpix_wrap is None:
        #         self.betamap[self._index, 0] = x.copy()
        #     else:
        #         self.betamap[self.seenpix_wrap, 0] = x.copy()

        #     if self.preset.qubic.params_qubic['instrument'] == 'UWB':
        #         ysim = np.zeros(self.nsnd)
        #         for ic in range(self.nc):
        #             for ip, p in enumerate(self._index):
        #                 ysim += A[ip, :, ic] @ self.d[ip, :, ic]
        #     else:
        #         ysim = np.zeros(int(self.nsnd*2))
        #         Atot = self._get_mixingmatrix(self.betamap[self._index])
        #         A150 = Atot[:, 0, :int(self.nf/2), 1].ravel()
        #         A220 = Atot[:, 0, int(self.nf/2):int(self.nf), 1].ravel()

        #         ysim[:int(self.nsnd)] = (A150 @ self.dfg150) + self.dcmb150
        #         ysim[int(self.nsnd):int(self.nsnd*2)] = (A220 @ self.dfg220) + self.dcmb220

        # _r = ysim - self.preset.acquisition.TOD_qubic
        # H_planck = self.preset.qubic.joint_out.get_operator(self.betamap,
        #                                             gain=self.preset.gain.gain_iter,
        #                                             fwhm=self.preset.acquisition.fwhm_mapmaking,
        #                                             nu_co=self.preset.comp.nu_co).operands[1]
        # tod_pl_s = H_planck(self.preset.comp.components_iter)

        # _r_pl = self.preset.acquisition.TOD_external - tod_pl_s
        # LLH = _dot(_r.T, self.preset.acquisition.invN.operands[0](_r), self.preset.comm) + _r_pl.T @ self.preset.acquisition.invN.operands[1](_r_pl)

        return self.chi2


class Chi2Blind:

    def __init__(self, preset):

        self.preset = preset
        self.nc = len(self.preset.comp.components_out)
        self.nf = self.preset.qubic.joint_out.qubic.nsub
        self.nsnd = (
            self.preset.qubic.joint_out.qubic.ndets
            * self.preset.qubic.joint_out.qubic.nsamples
        )
        self.nsub = self.preset.qubic.joint_out.qubic.nsub

    def _reshape_A(self, x):
        nf, nc = x.shape
        x_reshape = np.array([])
        for i in range(nc):
            x_reshape = np.append(x_reshape, x[:, i].ravel())
        return x_reshape

    def _fill_A(self, x):

        fsub = int(
            self.nsub * 2 / self.preset.comp.params_foregrounds["bin_mixing_matrix"]
        )
        A = np.ones((self.nsub * 2, self.nc - 1))
        k = 0
        for i in range(self.preset.comp.params_foregrounds["bin_mixing_matrix"]):
            for j in range(self.nc - 1):
                A[i * fsub : (i + 1) * fsub, j] = np.array([x[k]] * fsub)
                k += 1
        return A.ravel()

    def _reshape_A_transpose(self, x):

        fsub = int(
            self.nsub * 2 / self.preset.comp.params_foregrounds["bin_mixing_matrix"]
        )
        x_reshape = np.ones(self.nsub * 2)

        for i in range(self.preset.comp.params_foregrounds["bin_mixing_matrix"]):
            x_reshape[i * fsub : (i + 1) * fsub] = np.array([x[i]] * fsub)
        return x_reshape

    def _qu(self, x, tod_comp):
        ### Fill mixing matrix if fsub different to 1
        x = self._fill_A(x)

        ### CMB contribution
        tod_cmb_150 = np.sum(tod_comp[0, : self.nsub, :], axis=0)
        tod_cmb_220 = np.sum(tod_comp[0, self.nsub : 2 * self.nsub, :], axis=0)

        if self.preset.qubic.params_qubic["instrument"] == "DB":

            ### Mixing matrix element for each nus
            A150 = x[: self.nsub * (self.nc - 1)].copy()
            A220 = x[self.nsub * (self.nc - 1) : self.nsub * 2 * (self.nc - 1)].copy()

            ###Â FG contributions
            tod_comp_150 = tod_comp[1:, : self.nsub, :].copy()
            tod_comp_220 = tod_comp[1:, self.nsub : 2 * self.nsub, :].copy()

            ### Describe the data as d = d_cmb + A . d_fg
            d_150 = tod_cmb_150.copy()  # + A150 @ tod_comp_150
            d_220 = tod_cmb_220.copy()  # + A220 @ tod_comp_220
            k = 0

            ### Recombine data with MM amplitude
            for i in range(self.nsub):
                for j in range(self.nc - 1):
                    d_150 += A150[k] * tod_comp_150[j, i]
                    d_220 += A220[k] * tod_comp_220[j, i]
                    k += 1

            ### Residuals
            _r = np.r_[d_150, d_220] - self.preset.acquisition.TOD_qubic

            ### Chi^2
            self.chi2 = _dot(
                _r.T, self.preset.acquisition.invN.operands[0](_r), self.preset.comm
            )

        return self.chi2

    def _qu_alt(self, x, tod_comp, A, icomp):

        x = self._reshape_A_transpose(x)

        if self.preset.qubic.params_qubic["instrument"] == "DB":
            ### CMB contribution
            tod_cmb_150 = np.sum(tod_comp[0, : self.nsub, :], axis=0)
            tod_cmb_220 = np.sum(tod_comp[0, self.nsub : 2 * self.nsub, :], axis=0)

            ###Â FG contributions
            tod_comp_150 = tod_comp[1:, : self.nsub, :].copy()
            tod_comp_220 = tod_comp[1:, self.nsub : 2 * self.nsub, :].copy()

            ### Describe the data as d = d_cmb + A . d_fg
            d_150 = tod_cmb_150.copy()
            d_220 = tod_cmb_220.copy()

            for i in range(self.nc - 1):
                for j in range(self.nsub):
                    if i + 1 == icomp:
                        d_150 += x[: self.nsub][j] * tod_comp_150[i, j]
                        d_220 += x[self.nsub : self.nsub * 2][j] * tod_comp_220[i, j]
                    else:
                        d_150 += A[: self.nsub, (i + 1)][j] * tod_comp_150[i, j]
                        d_220 += (
                            A[self.nsub : self.nsub * 2, (i + 1)][j]
                            * tod_comp_220[i, j]
                        )

        ### Residuals
        _r = np.r_[d_150, d_220] - self.preset.acquisition.TOD_qubic

        ### Chi^2
        self.chi2 = _dot(
            _r.T, self.preset.acquisition.invN.operands[0](_r), self.preset.comm
        )

        return self.chi2
