def update_spectral_index(self):  # this function is too complex and has code duplication
    """Update spectral index.

    Method that perform step 3) of the pipeline for 2 possible designs : Two Bands and Ultra Wide Band

    """

    ### Fitting method for the first component which is not CMB (always index 0)
    method = method_0 = self.preset.comp.params_foregrounds[self.preset.comp.components_name_out[1]]["type"]

    ### Loop over the mixing matrix fitting method for the different component
    ### If they are different, we assume that we want to run an alternate parametric/blind estimation
    if len(self.preset.comp.components_name_out) > 1:
        for component in self.preset.comp.components_name_out[2:]:
            if component != "CO" and self.preset.comp.params_foregrounds[component]["type"] != method_0:
                method = "parametric_blind"
                break

    tod_comp = self.get_tod_comp()
    self.nfev = 0
    self.preset.mixingmatrix._index_seenpix_beta = 0

    if method == "parametric":
        ### Model without spatial variation of spectral index
        if self.preset.comp.params_foregrounds["Dust"]["nside_beta_out"] == 0:
            previous_beta = self.preset.acquisition.beta_iter.copy()
            self.chi2 = Chi2(self.preset, tod_comp, parametric=True)

            ### Fit using scipy.optimize.minimize
            res = minimize(self.chi2, x0=self.preset.acquisition.beta_iter, method="L-BFGS-B", callback=self.callback, options={"maxiter": 1000, "ftol": 1e-9})
            self.preset.acquisition.beta_iter = res.x

            self.preset.acquisition.Amm_iter = self.chi2.compute_mixing_matrix_parametric(nus=self.preset.qubic.joint_out.allnus, x=self.preset.acquisition.beta_iter)

            del tod_comp
            gc.collect()

            if self.preset.tools.rank == 0:
                print(f"Iteration k     : {previous_beta}")
                print(f"Iteration k + 1 : {self.preset.acquisition.beta_iter}")
                print(f"Truth           : {self.preset.mixingmatrix.beta_in}")
                print(f"Residuals       : {self.preset.mixingmatrix.beta_in - self.preset.acquisition.beta_iter}")

            self.preset.tools.comm.Barrier()
            self.preset.acquisition.allbeta = np.concatenate((self.preset.acquisition.allbeta, np.array([self.preset.acquisition.beta_iter])), axis=0)

            self.plots.plot_beta_iteration(self.preset.acquisition.allbeta, truth=self.preset.mixingmatrix.beta_in, ki=self._steps)

        ### Model with spatial variation of spectral index
        else:
            raise ValueError("d1 model not implemented yet.")

    elif method == "blind":
        previous_step = self.preset.acquisition.Amm_iter[: self.preset.qubic.joint_out.qubic.nsub, 1:].copy()

        if self._steps == 0:
            self.allAmm_iter = np.array([self.preset.acquisition.Amm_iter])
            self.plots.plot_sed(
                self.preset.qubic.joint_in.qubic.allnus,
                self.preset.mixingmatrix.Amm_in[: self.preset.qubic.joint_in.qubic.nsub, 1:],
                self.preset.qubic.joint_out.qubic.allnus,
                self.preset.acquisition.Amm_iter[: self.preset.qubic.joint_out.qubic.nsub, 1:],
                ki=self._steps - 1,
                gif=self.preset.tools.params["PCG"]["do_gif"],
            )

        ### Blind using scipy.optimize.minimize
        if self.preset.comp.params_foregrounds["blind_method"] == "minimize":
            # Neveer used? It won't work as it is for now
            self.chi2 = Chi2(
                self.preset,
                tod_comp,
                parametric=False,
            )
            x0 = []
            bnds = []
            for inu in range(self.preset.comp.params_foregrounds["bin_mixing_matrix"]):
                for icomp in range(1, len(self.preset.comp.components_name_out)):
                    x0 += [np.mean(self.preset.acquisition.Amm_iter[inu * self.fsub : (inu + 1) * self.fsub, icomp])]
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

        ### Blind using PCG
        elif self.preset.comp.params_foregrounds["blind_method"] == "PCG":
            raise ValueError("Blind PCG is not implemtented yet.")

        ### Blind using scipy.optimize.minimize in an alternate manner, with a loop over components
        elif self.preset.comp.params_foregrounds["blind_method"] == "alternate":
            raise ValueError("Blind alternate is not implemented yet.")
        else:
            raise TypeError(f"{self.preset.comp.params_foregrounds['blind_method']} is not yet implemented..")

        self.allAmm_iter = np.concatenate((self.allAmm_iter, np.array([self.preset.acquisition.Amm_iter])), axis=0)

        if self.preset.tools.rank == 0:
            print(f"Iteration k     : {previous_step.ravel()}")
            print(f"Iteration k + 1 : {self.preset.acquisition.Amm_iter[: self.preset.qubic.joint_out.qubic.nsub, 1:].ravel()}")
            print(f"Truth           : {self.preset.mixingmatrix.Amm_in[: self.preset.qubic.joint_out.qubic.nsub, 1:].ravel()}")
            print(
                f"Residuals       : {self.preset.mixingmatrix.Amm_in[: self.preset.qubic.joint_out.qubic.nsub, 1:].ravel() - self.preset.acquisition.Amm_iter[: self.preset.qubic.joint_out.qubic.nsub, 1:].ravel()}"
            )
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

        del tod_comp
        gc.collect()

    elif method == "parametric_blind":
        previous_step = self.preset.acquisition.Amm_iter[: self.preset.qubic.joint_out.qubic.nsub * 2, 1:].copy()

        if self._steps == 0:
            self.allAmm_iter = np.array([self.preset.acquisition.Amm_iter])

        for i in range(len(self.preset.comp.components_name_out)):
            if self.preset.comp.components_name_out[i] != "CMB":
                if self.preset.comp.params_foregrounds[self.preset.comp.components_name_out[i]]["type"] == "parametric":
                    print("Fitting ", self.preset.comp.components_name_out[i], i, " with parametric method.")

                    previous_beta = self.preset.acquisition.beta_iter.copy()

                    chi2 = Chi2(self.preset, tod_comp, self.preset.acquisition.Amm_iter, i)

                    self.preset.acquisition.beta_iter[i - 1] = np.array([fmin_l_bfgs_b(chi2, x0=self.preset.acquisition.beta_iter[i - 1], callback=self.callback, approx_grad=True, epsilon=1e-6)[0]])

                    self.preset.acquisition.Amm_iter = self.update_mixing_matrix(self.preset.acquisition.beta_iter, self.preset.acquisition.Amm_iter, i)

                else:
                    print("I am fitting ", self.preset.comp.components_name_out[i], i, " with blind")

                    fun = partial(
                        self.chi2._qu_alt,
                        tod_comp=tod_comp,
                        A=self.preset.acquisition.Amm_iter,
                        icomp=i,
                    )

                    ### Starting point
                    x0 = []
                    bnds = []
                    for ii in range(self.preset.comp.params_foregrounds["bin_mixing_matrix"]):
                        for j in range(1, len(self.preset.comp.components_name_out)):
                            x0 += [np.mean(self.preset.acquisition.Amm_iter[ii * self.fsub : (ii + 1) * self.fsub, j])]
                            bnds += [(0, None)]

                    Ai = minimize(
                        fun,
                        x0=x0,
                        callback=self.callback,
                        bounds=bnds,
                        method="SLSQP",
                        tol=1e-10,
                    ).x

                    for ii in range(self.preset.comp.params_foregrounds["bin_mixing_matrix"]):
                        self.preset.acquisition.Amm_iter[ii * self.fsub : (ii + 1) * self.fsub, i] = Ai[ii]

            if self.preset.tools.rank == 0:
                print(f"Iteration k     : {previous_step.ravel()}")
                print(f"Iteration k + 1 : {self.preset.acquisition.Amm_iter[: self.preset.qubic.joint_out.qubic.nsub, 1:].ravel()}")
                print(f"Truth           : {self.preset.mixingmatrix.Amm_in[: self.preset.qubic.joint_out.qubic.nsub, 1:].ravel()}")
                print(
                    f"Residuals       : {self.preset.mixingmatrix.Amm_in[: self.preset.qubic.joint_out.qubic.nsub, 1:].ravel() - self.preset.acquisition.Amm_iter[: self.preset.qubic.joint_out.qubic.nsub, 1:].ravel()}"
                )
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

            del tod_comp
            gc.collect()
