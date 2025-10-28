import os

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import yaml
from getdist import MCSamples, plots
from matplotlib.gridspec import GridSpec
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


def Dl2Cl(ell, Dl):
    """
    Convert angular power spectrum from D_ell to C_ell.

    Formula
    -------
    C_ell = D_ell * 2*pi / (ell * (ell + 1))

    Parameters
    ----------
    ell : array_like
        Multipole moments (1D). Values must be > 0 (division by zero otherwise).
    Dl : array_like
        D_ell values. Can be scalar, 1D (same length as `ell`) or broadcastable to `ell`.
        Typical shape in this module: (n_nus, n_nus, n_bins).

    Returns
    -------
    ndarray
        C_ell values with the same shape as the broadcast result of `Dl` and `ell`.

    Notes
    -----
    - Uses elementwise broadcasting; ensure `ell` and `Dl` shapes are compatible.
    - No guard is performed for ell == 0; caller must avoid or mask such entries.
    """
    return Dl * 2 * np.pi / (ell * (ell + 1))


def Cl2BK(ell, Cl):
    """
    Convert C_ell to the bandpower-like quantity 100 * ell * C_ell / (2*pi).

    Formula
    -------
    result = 100 * ell * C_ell / (2*pi)

    Parameters
    ----------
    ell : array_like
        Multipole moments (1D). Should match or broadcast with `Cl`.
    Cl : array_like
        C_ell values. Can be scalar, 1D or broadcastable to `ell`.
        Typical shape in this module: (n_nus, n_nus, n_bins).

    Returns
    -------
    ndarray
        Transformed bandpower values with shape equal to the broadcasted inputs.
    """
    return 100 * ell * Cl / (2 * np.pi)


def plot_cross_spectrum(nus, ell, Dl, Dl_err, ymodel, label_model="CMB + Dust", nbins=None, nrec=2, mode="Dl", figsize=None, title=None, name=None, dpi=300):
    """
    Plot the upper-triangle matrix of cross-angular power spectra D_ell (and optional model).

    The function arranges a len(nus) x len(nus) grid and fills only the upper triangle
    (including diagonal) with small subplots labelled by the frequency pair `nus[i] x nus[j]`.
    It draws data errorbars, an optional second series (Dl - noise if `Dl_err` provided),
    and a model line (from `ymodel`) either in D_ell units or transformed to the
    "100 * ell * C_ell / (2*pi)" units depending on `mode`.

    Parameters
    ----------
    nus : array_like
        1D array of frequency identifiers (used for subplot annotations). Length = n_nus.
    ell : array_like
        1D array of multipole moments. Length >= nbins (if nbins provided).
        Must be > 0 to avoid division-by-zero in conversions.
    Dl : ndarray
        Data D_ell values, expected shape (n_nus, n_nus, n_ell) or broadcastable to that.
    Dl_err : ndarray or None
        Errors on Dl with same shape as `Dl` (or broadcastable). If provided, an additional
        series `Dl - Dl_err` will be plotted where applicable. Errors are absolute-valued
        (the function applies np.abs).
    ymodel : ndarray or None
        Model values for plotting. Expected shape (n_nus, n_nus, n_ell) (or broadcastable).
        If None, no model line is drawn.
    label_model : str, optional
        Legend label for the model line (default: "CMB + Dust").
    nbins : int or None, optional
        Number of ell bins to plot. If None (default) uses len(ell).
    nrec : int, optional
        Number of "recon" channels used to choose subplot background color and styling.
        Default is 2.
    mode : {"Dl", ...}, optional
        If "Dl" the data are plotted in D_ell units. Otherwise the model/data are
        transformed via `_Dl2Cl` and `_Cl2BK` before plotting (matching original behaviour).
    ft_nus : int, optional
        Font size for the subplot frequency annotations (default: 10).
    figsize : tuple, optional
        Matplotlib figure size (default: (10, 8)).
    title : str or None, optional
        Suptitle appended to the fixed prefix "Angular Cross-Power Spectra".
        If None, only the prefix is used.
    name : str or None, optional
        If provided, the figure is saved to this filename as a PDF.

    Side effects
    ------------
    - Creates a matplotlib figure, shows it with plt.show() and optionally saves it.
    - Does not return the figure (returns None). If you need the figure object, modify the
    function to `return fig` after creation.

    Notes
    -----
    - The function preserves exact plotting order, labels and colours of the original code.
    - The caller must ensure shapes of `nus`, `ell`, `Dl`, `Dl_err`, and `ymodel` are compatible.
    """

    n = len(nus)
    if figsize is None:
        figsize = (2.2 * n, 2.2 * n)

    # Dynamic font scaling based on figure height
    count_factor = 1 / np.sqrt(n)
    scale_factor = (figsize[1] / 8.0) * count_factor

    ft_axis = max(int(13 * scale_factor), 7)
    ft_nus = max(int(15 * scale_factor), 8)
    ft_title = max(int(32 * np.sqrt(scale_factor)), 14)

    # defaults & preproc (preserve original behavior)
    if nbins is None:
        nbins = len(ell)

    # Dl2 := Dl - Dl_err (only if Dl_err provided)
    Dl2 = Dl - Dl_err if Dl_err is not None else None

    # keep absolute-valued errors as in original
    Dl_err = np.abs(Dl_err) if Dl_err is not None else None
    Dl2_err = np.abs(Dl2) if Dl2 is not None else None

    ell_sel = ell[:nbins]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(len(nus), len(nus), figure=fig)

    kp = 0  # subplot counter (used to control first-plot labelling)

    def _model_plot(ax, i, j, kp):
        """Plot model lines respecting the original oddity:
        - if kp == 0: plot *only* the mode-specific model (with label)
        - else: if ymodel exists, plot both Dl model and transformed Cl model (no label)
        """
        if ymodel is None:
            return

        y = ymodel[i, j, :nbins]
        if kp == 0:
            if mode == "Dl":
                ax.plot(ell_sel, y, "--r", label=label_model)
            else:
                ax.plot(ell_sel, Cl2BK(ell_sel, Dl2Cl(ell_sel, y)), "--r", label=label_model)
        else:
            # replicate original behavior: plot Dl model and (always) transformed model if ymodel present
            ax.plot(ell_sel, y, "--r")
            ax.plot(ell_sel, Cl2BK(ell_sel, Dl2Cl(ell_sel, y)), "--r")

    def _plot_errorbars(ax, i, j, color_main, color_second=None, label_main=None, label_second=None):
        """Add errorbars in either 'Dl' or transformed mode."""
        main = Dl[i, j, :nbins]
        main_err = Dl_err[i, j, :nbins] if Dl_err is not None else None

        # plot main series
        if mode == "Dl":
            ax.errorbar(ell_sel, main, yerr=main_err, capsize=5, color=color_main, fmt="-o", label=label_main)
        else:
            y_main = Cl2BK(ell_sel, Dl2Cl(ell_sel, main))
            yerr_main = Cl2BK(ell_sel, Dl2Cl(ell_sel, main_err)) if main_err is not None else None
            ax.errorbar(ell_sel, y_main, yerr=yerr_main, capsize=5, color=color_main, fmt="-o", label=label_main)

        # optional second series (Dl - noise)
        if Dl2 is not None:
            sec = Dl2[i, j, :nbins]
            sec_err = Dl2_err[i, j, :nbins] if Dl2_err is not None else None
            if mode == "Dl":
                ax.errorbar(ell_sel, sec, yerr=sec_err, capsize=5, color=color_second or "orange", fmt="-o", label=label_second)
            else:
                y_sec = Cl2BK(ell_sel, Dl2Cl(ell_sel, sec))
                yerr_sec = Cl2BK(ell_sel, Dl2Cl(ell_sel, sec_err)) if sec_err is not None else None
                ax.errorbar(ell_sel, y_sec, yerr=yerr_sec, capsize=5, color=color_second or "orange", fmt="-o", label=label_second)

    # iterate over upper triangle (including diagonal)
    for i in range(len(nus)):
        for j in range(i, len(nus)):
            ax = fig.add_subplot(gs[i, j])

            # labels (preserve font sizes)
            if i == j:
                ax.set_xlabel(r"$\ell$", fontsize=2 * ft_axis)
                if mode == "Dl":
                    ax.set_ylabel(r"$\mathcal{D}_{\ell}$", fontsize=2 * ft_axis)
                else:
                    ax.set_ylabel(r"100 $ \frac{\ell \mathcal{C}_{\ell}}{2 \pi}$", fontsize=2 * ft_axis)
            else:
                ax.tick_params(axis="x", labelrotation=30)
                ax.tick_params(axis="y", labelrotation=-45)

            # model plotting (keeps original behavior/oddity)
            _model_plot(ax, i, j, kp)

            ax.patch.set_alpha(0.2)
            ax.annotate(f"{nus[i]:.0f}x{nus[j]:.0f}", xy=(0.1, 0.9), fontsize=ft_nus, xycoords="axes fraction", color="black", weight="bold")

            # facecolor + plotting choices exactly as original
            if i < nrec and j < nrec:
                ax.set_facecolor("blue")
                if kp == 0:
                    _plot_errorbars(
                        ax,
                        i,
                        j,
                        color_main="darkblue",
                        color_second="orange",
                        label_main=r"$\mathcal{D}_{\ell}^{\nu_1 \times \nu_2}$",
                        label_second=r"$\mathcal{D}_{\ell}^{\nu_1 \times \nu_2} - \mathcal{N}_{\ell}^{\nu_1 \times \nu_2}$",
                    )
                else:
                    _plot_errorbars(
                        ax,
                        i,
                        j,
                        color_main="darkblue",
                        color_second="orange",
                    )
            elif i < nrec and j >= nrec:
                ax.set_facecolor("skyblue")
                _plot_errorbars(ax, i, j, color_main="blue", color_second="orange")
            else:
                ax.set_facecolor("green")
                if kp == 0:
                    _plot_errorbars(
                        ax,
                        i,
                        j,
                        color_main="blue",
                        color_second="orange",
                        label_main=r"$\mathcal{D}_{\ell}^{\nu_1 \times \nu_2}$",
                        label_second=r"$\mathcal{D}_{\ell}^{\nu_1 \times \nu_2} - \mathcal{N}_{\ell}^{\nu_1 \times \nu_2}$",
                    )
                else:
                    _plot_errorbars(ax, i, j, color_main="blue", color_second="orange")

            kp += 1
            plt.xticks(fontsize=ft_axis)
            plt.yticks(fontsize=ft_axis)

    # title / legend / save / show
    if title is not None:
        title = "Angular Cross-Power Spectra" + title
    else:
        title = "Angular Cross-Power Spectra"
    fig.suptitle(title, fontsize=ft_title, fontweight="bold", y=0.96)
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.9, wspace=0.25, hspace=0.30)
    fig.legend(fontsize=ft_title, bbox_to_anchor=(0.4, 0.4))
    if name is not None:
        plt.savefig(name)


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

            plt.savefig("CMM/" + self.preset.tools.params["foldername"] + "/Plots/A_iter/A_iter{ki + 1}.png")

            if self.preset.tools.rank == 0:
                if ki > 0 and gif is False:
                    os.remove("CMM/" + self.preset.tools.params["foldername"] + "/Plots/A_iter/A_iter{ki}.png")

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
            plt.savefig("CMM/" + self.preset.tools.params["foldername"] + "/Plots/A_iter/beta_iter{ki + 1}.png")

            if ki > 0:
                os.remove("CMM/" + self.preset.tools.params["foldername"] + "/Plots/A_iter/beta_iter{ki}.png")
            plt.close()

    def _display_allresiduals(self, map_i, seenpix, ki=0):
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
        Nmaps, _, Nstk = map_i.shape

        stk = ["I", "Q", "U"]
        if self.params["Plots"]["maps"]:
            plt.figure(figsize=(10, 5 * Nmaps))
            k = 0
            r = self.preset.A(map_i) - self.preset.b
            map_res = np.ones(self.preset.comp.components_iter.shape) * hp.UNSEEN
            map_res[:, seenpix, :] = r

            for istk in range(Nstk):
                for icomp in range(len(self.preset.comp.components_name_out)):
                    reso = 15
                    nsig = 3

                    hp.gnomview(
                        map_res[icomp, :, istk],
                        rot=self.preset.sky.center,
                        reso=reso,
                        notext=True,
                        title=f"{self.preset.comp.components_name_out[icomp]} - {stk[istk]} - r = A x - b",
                        cmap="jet",
                        sub=(Nstk, len(self.preset.comp.components_out), k + 1),
                        min=-nsig * np.std(r[icomp, :, istk]),
                        max=nsig * np.std(r[icomp, :, istk]),
                    )
                    k += 1

            plt.tight_layout()
            plt.savefig("CMM/" + self.preset.tools.params["foldername"] + f"/Plots/allcomps/allres_iter{ki + 1}.png")

            plt.close()

    def _display_allcomponents(self, ki=0, gif=True, reso=15):
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
            maps_in = self.preset.acquisition.components_in_convolved
            maps_rec = self.preset.comp.components_iter
            maps_res = maps_rec - maps_in

            Nmaps, _, Nstk = maps_res.shape
            k = 0

            plt.figure(figsize=(5 * Nmaps, 10))
            for istk in range(Nstk):
                for icomp in range(len(self.preset.comp.components_name_out)):
                    hp.gnomview(
                        maps_rec[icomp, :, istk],
                        rot=self.preset.sky.center,
                        reso=reso,
                        notext=True,
                        title=f"{self.preset.comp.components_name_out[icomp]} - {stk[istk]} - Output",
                        cmap="jet",
                        sub=(Nstk, len(self.preset.comp.components_out) * 2, k + 1),
                    )
                    k += 1
                    hp.gnomview(
                        maps_res[icomp, :, istk],
                        rot=self.preset.sky.center,
                        reso=reso,
                        notext=True,
                        title=f"{self.preset.comp.components_name_out[icomp]} - {stk[istk]} - Residual",
                        cmap="jet",
                        sub=(Nstk, len(self.preset.comp.components_out) * 2, k + 1),
                    )
                    k += 1

            plt.tight_layout()
            plt.savefig("CMM/" + self.preset.tools.params["foldername"] + f"/Plots/allcomps/allcomps_iter{ki + 1}.png")

            if self.preset.tools.rank == 0:
                if ki > 0 and gif is False:
                    os.remove("CMM/" + self.preset.tools.params["foldername"] + f"/Plots/allcomps/allcomps_iter{ki}.png")
            plt.close()

    def display_maps(self, seenpix, ki=0, reso=15, view="gnomview"):
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

            maps_in = self.preset.acquisition.components_in_convolved
            maps_rec = self.preset.comp.components_iter
            maps_res = maps_rec - maps_in

            maps_in[:, ~seenpix, :] = hp.UNSEEN
            maps_rec[:, ~seenpix, :] = hp.UNSEEN
            maps_res[:, ~seenpix, :] = hp.UNSEEN

            Nmaps, _, _ = maps_res.shape
            k = 0

            for istk, s in enumerate(stk):
                plt.figure(figsize=(3.5 * Nmaps, 12))
                k = 0

                for icomp in range(len(self.preset.comp.components_name_out)):
                    if icomp == 0:
                        if istk > 0:
                            rms_i[0, istk - 1] = np.std(maps_res[icomp, seenpix, istk])

                    if view == "gnomview":
                        hp.gnomview(
                            maps_in[icomp, :, istk],
                            rot=self.preset.sky.center,
                            reso=reso,
                            notext=True,
                            title=f"Input {stk[istk]}",
                            cmap="jet",
                            sub=(len(self.preset.comp.components_out), 3, k + 1),
                        )
                        hp.gnomview(
                            maps_rec[icomp, :, istk],
                            rot=self.preset.sky.center,
                            reso=reso,
                            notext=True,
                            title=f"Output {stk[istk]}",
                            cmap="jet",
                            sub=(len(self.preset.comp.components_out), 3, k + 2),
                        )
                        hp.gnomview(
                            maps_res[icomp, :, istk],
                            rot=self.preset.sky.center,
                            reso=reso,
                            notext=True,
                            title=f"Residual {stk[istk]} - Std = {np.std(maps_res[icomp, seenpix, istk]):.3e}",
                            cmap="jet",
                            sub=(len(self.preset.comp.components_out), 3, k + 3),
                        )
                    elif view == "mollview":
                        hp.mollview(
                            maps_in,
                            notext=True,
                            title=f"Input {stk[istk]}",
                            cmap="jet",
                            sub=(len(self.preset.comp.components_out), 3, k + 1),
                        )
                        hp.mollview(
                            maps_rec,
                            notext=True,
                            title=f"Output {stk[istk]}",
                            cmap="jet",
                            sub=(len(self.preset.comp.components_out), 3, k + 2),
                        )
                        hp.mollview(
                            maps_rec,
                            notext=True,
                            title=f"Residual {stk[istk]} - Std = {np.std(maps_rec[icomp, seenpix, istk]):.3e}",
                            cmap="jet",
                            sub=(len(self.preset.comp.components_out), 3, k + 3),
                        )
                    k += 3

                plt.tight_layout()
                plt.savefig("CMM/" + self.preset.tools.params["foldername"] + f"/Plots/{s}/maps_iter{ki + 1}.png")

                if self.preset.tools.rank == 0:
                    if ki > 0:
                        os.remove("CMM/" + self.preset.tools.params["foldername"] + f"/Plots/{s}/maps_iter{ki}.png")

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
            plt.savefig("CMM/" + self.preset.tools.params["foldername"] + f"/Plots/A_iter/gain_iter{ki + 1}.png")

            if self.preset.tools.rank == 0:
                if ki > 0:
                    os.remove("CMM/" + self.preset.tools.params["foldername"] + f"/Plots/A_iter/gain_iter{ki}.png")

            plt.close()

    def plot_rms_iteration(self, rms, figsize=(8, 6), ki=0):
        if self.params["Plots"]["conv_rms"]:
            plt.figure(figsize=figsize)

            plt.plot(rms[1:, 0], "-b", label="Q")
            plt.plot(rms[1:, 1], "-r", label="U")

            plt.yscale("log")

            plt.tight_layout()
            plt.savefig("CMM/" + self.preset.tools.params["foldername"] + f"/Plots/A_iter/rms_iter{ki + 1}.png")

            if self.preset.tools.rank == 0:
                if ki > 0:
                    os.remove("CMM/" + self.preset.tools.params["foldername"] + f"/Plots/A_iter/rms_iter{ki}.png")

            plt.close()
