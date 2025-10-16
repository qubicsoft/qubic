import os

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import yaml
from getdist import MCSamples, plots
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator


def _plot_reconstructed_maps(
    maps,
    m_in,
    seenpix,
    name_file,
    center,
    max,
    min,
    reso=15,
    iter=0,
):
    """

    Save a PNG with the actual maps at iteration i. It assumes that maps is 3-dimensional

    """
    stk = ["I", "Q", "U"]

    Nmaps, _, Nstk = maps.shape
    res = maps - m_in

    k = 0
    plt.figure(figsize=(12, 3 * Nmaps))
    for imaps in range(Nmaps):
        maps[imaps, ~seenpix, :] = hp.UNSEEN
        res[imaps, ~seenpix, :] = hp.UNSEEN
        for istk in range(Nstk):
            hp.gnomview(
                maps[imaps, :, istk],
                rot=center,
                reso=reso,
                cmap="jet",
                sub=(Nmaps, Nstk * 2, k + 1),
                notext=True,
                max=max[imaps, istk],
                min=min[imaps, istk],
                title=f"Output - {stk[istk]}",
            )
            hp.gnomview(
                res[imaps, :, istk],
                rot=center,
                reso=reso,
                cmap="jet",
                sub=(Nmaps, Nstk * 2, k + 2),
                notext=True,
                max=max[imaps, istk],
                min=min[imaps, istk],
                title=f"Residuals - {stk[istk]}",
            )
            k += 2
    plt.suptitle(f"Iteration #{iter}", fontsize=15)
    plt.tight_layout()
    plt.savefig(name_file)
    plt.close()


class Plots:
    """

    Instance for plotting results of Monte-Carlo Markov Chain (i.e emcee).

    """

    def __init__(self):
        pass

    def _make_samples(self, chain, names, labels):
        self.sample = MCSamples(samples=chain, names=names, labels=labels)

    def make_list_free_parameter(self):
        """

        Make few list :
            - fp       : list of value of free parameters
            - fp_name  : list of name for each values
            - fp_latex : list of name in LateX for each values

        """

        fp = []
        fp_name = []
        fp_latex = []
        k = 0

        for iname, name in enumerate(self.params["Sky"].keys()):
            try:
                # print('yes')
                for jname, n in enumerate(self.params["Sky"][name]):
                    if type(self.params["Sky"][name][n]) is list:
                        # print(self.params['Sky'][name][n], n)
                        if self.params["Sky"][name][n][1] == "f":
                            fp += [self.params["Sky"][name][n][0]]
                            fp_latex += [self.params["Sky"][name][n][2]]
                            fp_name += [list(self.params["Sky"][name].keys())[k]]
                    k += 1
                k = 0
            except Exception:
                pass

        return fp, fp_name, fp_latex

    def _set_marker(self, values):
        """

        Define the markers to see the input values in GetDist plot.

        """

        dict = {}
        for ii, i in enumerate(values):
            # print(self.names[ii], values[ii])
            dict[self.names[ii]] = values[ii]

        if self.params["Sampler"]["markers"] is False:
            dict = None
        return dict

    def get_convergence(self, chain, job_id):
        """

        chain assumed to be not flat with shape (nsamples, nwalkers, nparams)

        """

        with open("params.yml", "r") as stream:
            self.params = yaml.safe_load(stream)

        self.values, self.names, self.labels = self.make_list_free_parameter()

        plt.figure(figsize=(4, 4))

        for i in range(chain.shape[2]):
            plt.subplot(chain.shape[2], 1, i + 1)
            plt.plot(chain[:, :, i], "-b", alpha=0.2)
            plt.plot(np.mean(chain[:, :, i], axis=1), "-r", alpha=1)
            plt.axhline(self.values[i], ls="--", color="black")
            plt.ylabel(self.names[i], fontsize=12)

        plt.xlabel("Iterations", fontsize=12)
        plt.savefig(f"allplots_{job_id}/Convergence_chain.png")
        plt.close()

    def get_triangle(self, chain, names, labels, job_id):
        """

        Make triangle plot of each estimated parameters

        """

        with open("params.yml", "r") as stream:
            self.params = yaml.safe_load(stream)

        self.values, self.names, self.labels = self.make_list_free_parameter()
        self.marker = self._set_marker(self.values)
        print(self.marker)
        self._make_samples(chain, names, labels)

        plt.figure(figsize=(8, 8))
        # Triangle plot
        g = plots.get_subplot_plotter()
        g.triangle_plot(
            [self.sample],
            filled=True,
            markers=self.marker,
            title_limit=self.params["Sampler"]["title_limit"],
        )
        # title_limit=1)
        plt.savefig(f"allplots_{job_id}/triangle_plot.png")
        plt.close()

    def get_Dl_plot(self, ell, Dl, Dl_err, nus, job_id, figsize=(10, 10), model=None):
        plt.figure(figsize=figsize)

        k = 0
        for i in range(len(nus)):
            for j in range(len(nus)):
                plt.subplot(len(nus), len(nus), k + 1)
                plt.errorbar(ell, Dl[k], yerr=Dl_err[k], fmt="or")
                if model is not None:
                    plt.errorbar(ell, model[k], fmt="-k")
                plt.title(f"{nus[i]:.0f}x{nus[j]:.0f}")
                # plt.yscale('log')
                k += 1

        plt.tight_layout()
        plt.savefig(f"allplots_{job_id}/Dl_plot.png")
        plt.close()


class PlotsFMM:
    def __init__(self, seenpix):
        self.stk = ["I", "Q", "U"]
        self.seenpix = seenpix

    def plot_frequency_maps(self, m_in, m_out, center, nus, reso=15, nsig=3, filename=None):
        Nf, _, Nstk = m_in.shape
        res = m_out - m_in

        plt.figure(figsize=(15, 3.3 * Nf))
        k = 1
        for inu in range(Nf):
            for istk in range(Nstk):
                max_out = np.max(np.abs(m_out[inu, self.seenpix, istk]))
                max_res = np.max(np.abs(res[inu, self.seenpix, istk]))
                hp.gnomview(
                    m_out[inu, :, istk],
                    rot=center,
                    reso=reso,
                    cmap="jet",
                    min=-max_out,
                    max=max_out,
                    sub=(Nf, 6, k),
                    title=f"{nus[inu]:.1f} GHz - Output {self.stk[istk]}",
                    notext=True,
                )

                hp.gnomview(
                    res[inu, :, istk],
                    rot=center,
                    reso=reso,
                    cmap="jet",
                    min=-max_res,
                    max=max_res,
                    sub=(Nf, 6, k + 1),
                    title=f"{nus[inu]:.1f} GHz - Residuals {self.stk[istk]}",
                    notext=True,
                )
                k += 2

        if filename is not None:
            plt.savefig(filename)
        plt.close()

    def plot_FMM_old(
        self,
        m_in,
        m_out,
        center,
        seenpix,
        nus,
        job_id,
        figsize=(10, 8),
        istk=1,
        nsig=3,
        name="signal",
    ):
        m_in[:, ~seenpix, :] = hp.UNSEEN
        m_out[:, ~seenpix, :] = hp.UNSEEN

        plt.figure(figsize=figsize)

        k = 1
        for i in range(self.params["QUBIC"]["nrec"]):
            hp.gnomview(
                m_in[i, :, istk],
                rot=center,
                reso=15,
                cmap="jet",
                min=-nsig * np.std(m_out[0, seenpix, istk]),
                max=nsig * np.std(m_out[0, seenpix, istk]),
                sub=(self.params["QUBIC"]["nrec"], 3, k),
                title=r"Input - $\nu$ = " + f"{nus[i]:.0f} GHz",
            )
            hp.gnomview(
                m_out[i, :, istk],
                rot=center,
                reso=15,
                cmap="jet",
                min=-nsig * np.std(m_out[0, seenpix, istk]),
                max=nsig * np.std(m_out[0, seenpix, istk]),
                sub=(self.params["QUBIC"]["nrec"], 3, k + 1),
                title=r"Output - $\nu$ = " + f"{nus[i]:.0f} GHz",
            )

            res = m_in[i, :, istk] - m_out[i, :, istk]
            res[~seenpix] = hp.UNSEEN

            hp.gnomview(
                res,
                rot=center,
                reso=15,
                cmap="jet",
                min=-nsig * np.std(m_out[0, seenpix, istk]),
                max=nsig * np.std(m_out[0, seenpix, istk]),
                sub=(self.params["QUBIC"]["nrec"], 3, k + 2),
            )

            k += 3
        plt.savefig(f"FMM/allplots_{job_id}/frequency_maps_{self.stk[istk]}_{name}.png")
        plt.close()

    def plot_FMM_mollview(self, m_in, m_out, nus, job_id, figsize=(10, 8), istk=1, nsig=3, fwhm=0):
        C = HealpixConvolutionGaussianOperator(fwhm=fwhm)
        plt.figure(figsize=figsize)

        k = 1
        for i in range(self.params["QUBIC"]["nrec"]):
            hp.mollview(
                C(m_in[i, :, istk]),
                cmap="jet",
                min=-nsig * np.std(m_out[0, :, istk]),
                max=nsig * np.std(m_out[0, :, istk]),
                sub=(self.params["QUBIC"]["nrec"], 3, k),
                title=r"Input - $\nu$ = " + f"{nus[i]:.0f} GHz",
            )

            hp.mollview(
                C(m_out[i, :, istk]),
                cmap="jet",
                min=-nsig * np.std(m_out[0, :, istk]),
                max=nsig * np.std(m_out[0, :, istk]),
                sub=(self.params["QUBIC"]["nrec"], 3, k + 1),
                title=r"Output - $\nu$ = " + f"{nus[i]:.0f} GHz",
            )

            hp.mollview(
                C(m_in[i, :, istk]) - C(m_out[i, :, istk]),
                cmap="jet",
                min=-nsig * np.std(m_out[0, :, istk]),
                max=nsig * np.std(m_out[0, :, istk]),
                sub=(self.params["QUBIC"]["nrec"], 3, k + 2),
            )

            k += 3
        plt.savefig(f"allplots_{job_id}/frequency_maps_{self.stk[istk]}_moll.png")
        plt.close()


class PlotsCMM:
    """

    Instance to produce plots on the convergence.

    Arguments :
    ===========
        - jobid : Int number for saving figures.
        - dogif : Bool to produce GIF.

    """

    def __init__(self, preset, dogif=True):
        self.preset = preset
        self.job_id = self.preset.job_id
        self.dogif = dogif
        self.params = self.preset.tools.params

    def plot_sed(self, nus_in, A_in, nus_out, A_out, figsize=(8, 6), ki=0, gif=False):
        """
        Plots the Spectral Energy Distribution (SED) and saves the plot as a PNG file.

        Parameters:
        nus (array-like): Array of frequency values.
        A (array-like): Array of amplitude values.
        figsize (tuple, optional): Size of the figure. Defaults to (8, 6).
        truth (array-like, optional): Array of true values for comparison. Defaults to None.
        ki (int, optional): Iteration index for file naming. Defaults to 0.

        Returns:
        None
        """

        if self.params["Plots"]["conv_beta"]:
            nf_in, nc_in = A_in.shape
            nf_out, nc_out = A_out.shape
            fsub = int(nf_in / nf_out)
            plt.figure(figsize=figsize)

            for ic in range(nc_in):
                plt.plot(nus_in, A_in[:, ic], "-k")

            for inu in range(nf_out):
                plt.errorbar(nus_out[inu], np.mean(A_in[inu * fsub : (inu + 1) * fsub]), fmt="og")

            for ic in range(nc_out):
                plt.errorbar(nus_out, A_out[:, ic], fmt="xb")

            plt.xlim(120, 260)
            eps = 0.4
            eps_max = A_in.max() * (1 + eps)
            eps_min = A_in.min() * (1 - eps)
            plt.ylim(eps_min, eps_max)
            plt.yscale("log")

            plt.savefig(f"CMM/jobs/{self.job_id}/A_iter/A_iter{ki + 1}.png")

            if self.preset.tools.rank == 0:
                if ki > 0 and gif is False:
                    os.remove(f"CMM/jobs/{self.job_id}/A_iter/A_iter{ki}.png")

            plt.close()

    def plot_beta_iteration(self, beta, figsize=(8, 6), truth=None, ki=0):
        """
        Method to plot beta as a function of iteration. Beta can have shape (niter) or (niter, nbeta).

        Parameters:
        beta : numpy.ndarray
            Array containing beta values for each iteration. Can be 1D or 2D.
        figsize : tuple, optional
            Size of the figure to be plotted. Default is (8, 6).
        truth : numpy.ndarray or float, optional
            True value(s) of beta to be plotted as a reference line. Default is None.
        ki : int, optional
            Iteration index for saving the plot. Default is 0.

        Returns:
        None
        """

        if self.params["Plots"]["conv_beta"]:
            niter = beta.shape[0]
            alliter = np.arange(0, niter, 1)

            plt.figure(figsize=figsize)
            plt.subplot(2, 1, 1)

            ### Constant beta on the sky
            if np.ndim(beta) == 2:
                plt.plot(alliter[1:] - 1, beta[1:])
                if truth is not None:
                    plt.axhline(truth, ls="--", color="red")

            ### Varying beta on the sky
            else:
                for i in range(beta.shape[1]):
                    plt.plot(alliter, beta[:, i], "-k", alpha=0.3)
                    if truth is not None:
                        for j in range(truth.shape[1]):
                            plt.axhline(truth[i, j], ls="--", color="red")

            plt.subplot(2, 1, 2)
            if np.ndim(beta) == 1:
                plt.plot(alliter[1:] - 1, abs(truth - beta[1:]))
            else:
                for i in range(beta.shape[1]):
                    plt.plot(alliter, abs(truth[i] - beta[:, i]), "-k", alpha=0.3)
            plt.yscale("log")
            plt.savefig(f"CMM/jobs/{self.job_id}/A_iter/beta_iter{ki + 1}.png")

            if ki > 0:
                os.remove(f"CMM/jobs/{self.job_id}/A_iter/beta_iter{ki}.png")
            plt.close()

    def _display_allresiduals(self, map_i, seenpix, figsize=(14, 10), ki=0):
        """
        Display all components of the Healpix map with Gaussian convolution.

        Parameters:
        seenpix (array-like): Boolean array indicating the pixels that are seen.
        figsize (tuple): Size of the figure to be plotted. Default is (14, 10).
        ki (int): Iteration index for saving the figure. Default is 0.

        This function generates and saves a figure showing the output maps and residuals
        for each component and Stokes parameter (I, Q, U). The maps are convolved using
        a Gaussian operator and displayed using Healpix's gnomview function.
        """
        stk = ["I", "Q", "U"]
        if self.params["Plots"]["maps"]:
            plt.figure(figsize=figsize)
            k = 0
            r = self.preset.A(map_i) - self.preset.b
            map_res = np.ones(self.preset.comp.components_iter.shape) * hp.UNSEEN
            map_res[:, seenpix, :] = r

            for istk in range(3):
                for icomp in range(len(self.preset.comp.components_name_out)):
                    _reso = 15
                    nsig = 3

                    hp.gnomview(
                        map_res[icomp, :, istk],
                        rot=self.preset.sky.center,
                        reso=_reso,
                        notext=True,
                        title=f"{self.preset.comp.components_name_out[icomp]} - {stk[istk]} - r = A x - b",
                        cmap="jet",
                        sub=(3, len(self.preset.comp.components_out), k + 1),
                        min=-nsig * np.std(r[icomp, :, istk]),
                        max=nsig * np.std(r[icomp, :, istk]),
                    )
                    k += 1

            plt.tight_layout()
            plt.savefig(f"CMM/jobs/{self.job_id}/allcomps/allres_iter{ki + 1}.png")

            # if self.preset.tools.rank == 0:
            #    if ki > 0:
            #        os.remove(f'jobs/{self.job_id}/allcomps/allres_iter{ki}.png')
            plt.close()

    def _display_allcomponents(self, seenpix, figsize=(14, 10), ki=0, gif=True, reso=15):
        """
        Display all components of the Healpix map with Gaussian convolution.

        Parameters:
        seenpix (array-like): Boolean array indicating the pixels that are seen.
        figsize (tuple): Size of the figure to be plotted. Default is (14, 10).
        ki (int): Iteration index for saving the figure. Default is 0.

        This function generates and saves a figure showing the output maps and residuals
        for each component and Stokes parameter (I, Q, U). The maps are convolved using
        a Gaussian operator and displayed using Healpix's gnomview function.
        """
        # C = [HealpixConvolutionGaussianOperator(
        #     fwhm=self.preset.acquisition.fwhm_rec[i],
        #     lmax=3 * self.params["SKY"]["nside"]) for i in range(len(self.preset.comp.components_name_out))]
        stk = ["I", "Q", "U"]
        if self.params["Plots"]["maps"]:
            plt.figure(figsize=figsize)
            k = 0
            for istk in range(3):
                for icomp in range(len(self.preset.comp.components_name_out)):
                    # if self.preset.comp.params_foregrounds['Dust']['nside_beta_out'] == 0:

                    # map_in = C[icomp](self.preset.comp.components_out[icomp, :, istk]).copy() # why?
                    map_in = self.preset.acquisition.components_in_convolved[icomp, :, istk].copy()
                    map_out = self.preset.comp.components_iter[icomp, :, istk].copy()

                    # sig = np.std(self.preset.comp.components_out[icomp, seenpix, istk])
                    # map_in[~seenpix] = hp.UNSEEN
                    # map_out[~seenpix] = hp.UNSEEN

                    # else:
                    #     if self.preset.qubic.params_qubic['convolution_in']:
                    #         map_in = self.preset.comp.components_convolved_out[icomp, :, istk].copy()
                    #         map_out = self.preset.comp.components_iter[istk, :, icomp].copy()
                    #         sig = np.std(self.preset.comp.components_convolved_out[icomp, seenpix, istk])
                    #     else:
                    #         map_in = self.preset.comp.components_out[istk, :, icomp].copy()
                    #         map_out = self.preset.comp.components_iter[istk, :, icomp].copy()
                    #         sig = np.std(self.preset.comp.components_out[istk, seenpix, icomp])
                    #     map_in[~seenpix] = hp.UNSEEN
                    #     map_out[~seenpix] = hp.UNSEEN

                    r = map_in - map_out
                    # nsig = 2
                    hp.gnomview(
                        map_out,
                        rot=self.preset.sky.center,
                        reso=reso,
                        notext=True,
                        title=f"{self.preset.comp.components_name_out[icomp]} - {stk[istk]} - Output",
                        cmap="jet",
                        sub=(3, len(self.preset.comp.components_out) * 2, k + 1),
                        # min=-nsig * sig,
                        # max=nsig * sig,
                    )
                    k += 1
                    hp.gnomview(
                        r,
                        rot=self.preset.sky.center,
                        reso=reso,
                        notext=True,
                        title=f"{self.preset.comp.components_name_out[icomp]} - {stk[istk]} - Residual",
                        cmap="jet",
                        sub=(3, len(self.preset.comp.components_out) * 2, k + 1),
                        # min=-nsig * np.std(r[seenpix]),
                        # max=nsig * np.std(r[seenpix]),
                    )
                    k += 1

            plt.tight_layout()
            plt.savefig(f"CMM/jobs/{self.job_id}/allcomps/allcomps_iter{ki + 1}.png")

            if self.preset.tools.rank == 0:
                if ki > 0 and gif is False:
                    os.remove(f"CMM/jobs/{self.job_id}/allcomps/allcomps_iter{ki}.png")
            plt.close()

    def display_maps(self, seenpix, figsize=(14, 8), nsig=6, ki=0, view="gnomview"):
        """

        Method to display maps at given iteration.

        Arguments:
        ----------
            - seenpix : array containing the id of seen pixels.
            - ngif    : Int number to create GIF with ngif PNG image.
            - figsize : Tuple to control size of plots.
            - nsig    : Int number to compute errorbars.

        """
        if self.params["Plots"]["maps"]:
            stk = ["I", "Q", "U"]
            rms_i = np.zeros((1, 2))

            for istk, s in enumerate(stk):
                plt.figure(figsize=figsize)

                k = 0

                for icomp in range(len(self.preset.comp.components_name_out)):
                    # if self.preset.comp.params_foregrounds['Dust']['nside_beta_out'] == 0:
                    if self.preset.qubic.params_qubic["convolution_in"]:
                        map_in = self.preset.acquisition.components_in_convolved[icomp, :, istk].copy()
                        map_out = self.preset.comp.components_iter[icomp, :, istk].copy()
                    else:
                        map_in = self.preset.comp.components_out[icomp, :, istk].copy()
                        map_out = self.preset.comp.components_iter[icomp, :, istk].copy()

                    # else:
                    #     if self.preset.qubic.params_qubic['convolution_in']:
                    #         map_in = self.preset.comp.components_convolved_out[icomp, :, istk].copy()
                    #         map_out = self.preset.comp.components_iter[istk, :, icomp].copy()
                    #     else:
                    #         map_in = self.preset.comp.components_out[istk, :, icomp].copy()
                    #         map_out = self.preset.comp.components_iter[istk, :, icomp].copy()

                    sig = np.std(map_in[seenpix])
                    map_in[~seenpix] = hp.UNSEEN
                    map_out[~seenpix] = hp.UNSEEN
                    r = map_in - map_out
                    r[~seenpix] = hp.UNSEEN
                    if icomp == 0:
                        if istk > 0:
                            rms_i[0, istk - 1] = np.std(r[seenpix])

                    _reso = 15
                    nsig = 3
                    if view == "gnomview":
                        hp.gnomview(
                            map_in,
                            rot=self.preset.sky.center,
                            reso=_reso,
                            notext=True,
                            title="",
                            cmap="jet",
                            sub=(len(self.preset.comp.components_out), 3, k + 1),
                            min=-nsig * sig,
                            max=nsig * sig,
                        )
                        hp.gnomview(
                            map_out,
                            rot=self.preset.sky.center,
                            reso=_reso,
                            notext=True,
                            title="",
                            cmap="jet",
                            sub=(len(self.preset.comp.components_out), 3, k + 2),
                            min=-nsig * sig,
                            max=nsig * sig,
                        )
                        hp.gnomview(
                            r,
                            rot=self.preset.sky.center,
                            reso=_reso,
                            notext=True,
                            title=f"{np.std(r[seenpix]):.3e}",
                            cmap="jet",
                            sub=(len(self.preset.comp.components_out), 3, k + 3),
                            min=-nsig * sig,
                            max=nsig * sig,
                        )
                    elif view == "mollview":
                        hp.mollview(
                            map_in,
                            notext=True,
                            title="",
                            cmap="jet",
                            sub=(len(self.preset.comp.components_out), 3, k + 1),
                            min=-nsig * sig,
                            max=nsig * sig,
                        )
                        hp.mollview(
                            map_out,
                            notext=True,
                            title="",
                            cmap="jet",
                            sub=(len(self.preset.comp.components_out), 3, k + 2),
                            min=-nsig * sig,
                            max=nsig * sig,
                        )
                        hp.mollview(
                            r,
                            notext=True,
                            title=f"{np.std(r[seenpix]):.3e}",
                            cmap="jet",
                            sub=(len(self.preset.comp.components_out), 3, k + 3),
                            min=-nsig * sig,
                            max=nsig * sig,
                        )
                    k += 3

                plt.tight_layout()
                plt.savefig(f"CMM/jobs/{self.job_id}/{s}/maps_iter{ki + 1}.png")

                if self.preset.tools.rank == 0:
                    if ki > 0:
                        os.remove(f"CMM/jobs/{self.job_id}/{s}/maps_iter{ki}.png")

                plt.close()
            self.preset.acquisition.rms_plot = np.concatenate((self.preset.acquisition.rms_plot, rms_i), axis=0)

    def plot_gain_iteration(self, gain, figsize=(8, 6), ki=0):
        """

        Method to plot convergence of reconstructed gains.

        Arguments :
        -----------
            - gain    : Array containing gain number (1 per detectors). It has the shape (Niteration, Ndet, 2) for Two Bands design and (Niteration, Ndet) for Wide Band design
            - alpha   : Transparency for curves.
            - figsize : Tuple to control size of plots.

        """

        if self.params["Plots"]["conv_gain"]:
            plt.figure(figsize=figsize)

            # plt.hist(gain[:, i, j])
            if self.preset.qubic.params_qubic["type"] == "two":
                color = ["red", "blue"]
                for j in range(2):
                    plt.hist(gain[-1, :, j], bins=20, color=color[j])
            #        plt.plot(alliter-1, np.mean(gain, axis=1)[:, j], color[j], alpha=1)
            #        for i in range(ndet):
            #            plt.plot(alliter-1, gain[:, i, j], color[j], alpha=alpha)

            # elif self.preset.qubic.params_qubic['type'] == 'wide':
            #    color = ['--g']
            #    plt.plot(alliter-1, np.mean(gain, axis=1), color[0], alpha=1)
            #    for i in range(ndet):
            #        plt.plot(alliter-1, gain[:, i], color[0], alpha=alpha)

            # plt.yscale('log')
            # plt.ylabel(r'|$g_{reconstructed} - g_{input}$|', fontsize=12)
            # plt.xlabel('Iterations', fontsize=12)
            plt.xlim(-0.1, 0.1)
            plt.ylim(0, 100)
            plt.axvline(0, ls="--", color="black")
            plt.savefig(f"CMM/jobs/{self.job_id}/gain_iter{ki + 1}.png")

            if self.preset.tools.rank == 0:
                if ki > 0:
                    os.remove(f"CMM/jobs/{self.job_id}/gain_iter{ki}.png")

            plt.close()

    def plot_rms_iteration(self, rms, figsize=(8, 6), ki=0):
        if self.params["Plots"]["conv_rms"]:
            plt.figure(figsize=figsize)

            plt.plot(rms[1:, 0], "-b", label="Q")
            plt.plot(rms[1:, 1], "-r", label="U")

            plt.yscale("log")

            plt.tight_layout()
            plt.savefig(f"CMM/jobs/{self.job_id}/rms_iter{ki + 1}.png")

            if self.preset.tools.rank == 0:
                if ki > 0:
                    os.remove(f"CMM/jobs/{self.job_id}/rms_iter{ki}.png")

            plt.close()
