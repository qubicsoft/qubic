import os
import pickle as pkl
import sys
from copy import deepcopy

import corner
import matplotlib.pyplot as plt
import numpy as np
import yaml
from getdist import MCSamples, plots
from pyoperators import MPI
from qubic.lib.MapMaking.FrequencyMapMaking.Qspectra_component import SkySpectra
from qubic.lib.Qfit import FitEllSpace
from qubic.lib.Qfoldertools import MergeAllFiles, create_folder_if_not_exists
from qubic.lib.Qmpi_tools import MpiTools

comm = MPI.COMM_WORLD
mpi = MpiTools(comm)

########################################################
###################### Parameters ######################
########################################################

### Parameters file
fit_parameters_path = str(sys.argv[1])
with open(fit_parameters_path, "r") as f:
    fit_params = yaml.safe_load(f)

### Folder where spectrum should be stored
folder = fit_params["simulation_path"]
folder_spectrum = folder + "/Spectrum/"
folder_save = folder + "/Fit/" + fit_params["name"] + "/"
create_folder_if_not_exists(comm, folder_save)

### Import Spectrum parameters
# TODO: should we really modify these values
nbins = fit_params["Spectrum"]["nbins"]
sample_variance = fit_params["Spectrum"]["sample_variance"]
diagonal = fit_params["Spectrum"]["diagonal"]
# TODO: compute fsky from coverage
fsky = fit_params["Spectrum"]["fsky"]
dl = fit_params["Spectrum"]["dl"]

### Import MCMC parameters
# Frequency used for fitting
nus_qubic = fit_params["MCMC"]["nus_qubic"]
nus_planck = fit_params["MCMC"]["nus_planck"]
nus_index = np.array(nus_qubic + nus_planck)

discard = fit_params["MCMC"]["discard"]
nwalkers = fit_params["MCMC"]["nwalkers"]
nsteps = fit_params["MCMC"]["nsteps"]
verbose = fit_params["MCMC"]["verbose"]

########################################################
###################### Fit #############################
########################################################

### Concatenate all realizations
files = MergeAllFiles(folder_spectrum)
mpi._print_message(f"Number of realizations : {files.number_of_realizations}")

### Check if all files have the same parameters
mpi._print_message("    => Checking that all spectrum are from simulations with same parameters")
parameters_all_files = files._reads_all_files("parameters", verbose=verbose)

# Remove the seed before comparison
def remove_seed(p):
    p = deepcopy(p)  # avoid modifying original dict
    try:
        del p["QUBIC"]["NOISE"]["seed_noise"]
        del p["PLANCK"]["seed_noise"]
    except KeyError:
        pass
    return p

parameters_all_clean = [remove_seed(p) for p in parameters_all_files]
parameters = parameters_all_clean[0]

test_all_same = all(p == parameters for p in parameters_all_clean)

if test_all_same:
    mpi._print_message("    => All Parameters are the same !")
else:
    raise ValueError("All Parameters aren't the same ! Check your simulations !!!")

### Check if frequencies that you want to fit have the correct size
mpi._print_message("    => Checking frequencies used for fitting")
if nus_qubic.__len__() != parameters["QUBIC"]["nrec"]:
    raise ValueError("QUBIC frequencies are wrongly defined, it must match Nrec value !")
if nus_planck.__len__() != 7:
    raise ValueError("Planck frequencies are wrongly defined, it must be only 7 frequencies !")

### Multipoles
mpi._print_message("    => Reading multipoles")
ell = files._reads_one_file(0, "ell")[:nbins]

### Frequencies
mpi._print_message("    => Reading frequencies")
nus = files._reads_one_file(0, "nus")[nus_index]
mpi._print_message(f"nus : {nus}")

### Compute mean of signal and noise
mpi._print_message("    => Averaging signal and noise power spectra")
BBsignal = np.mean(files._reads_all_files("Dls", verbose=verbose), axis=0)[nus_index, :, :nbins][:, nus_index, :nbins]
BBnoise = files._reads_all_files("Nls")[:, :, nus_index, :nbins][:, nus_index, :, :nbins]

### Remove noise bias
mpi._print_message("    => Removing noise bias")
BBsignal -= np.mean(BBnoise, axis=0)

### Define sky model in ell space
sky = SkySpectra(ell, nus)

### Fit of cosmological parameters
mpi._print_message("    => Fitting parameters")
fit = FitEllSpace(ell, BBsignal, BBnoise, model=sky.model, parameters_file=fit_params, sample_variance=sample_variance, fsky=fsky, dl=dl, diagonal=diagonal)
samples, samples_flat = fit.run(nsteps, nwalkers, discard=discard, comm=comm)

fit_parameters_names = fit.get_fitting_parameters_names()

### Plot MCMC
n_params = samples.shape[-1]

# Handle case: 1 parameter → axes is a single object
fig, axes = plt.subplots(n_params, 1, figsize=(12, 3 * n_params))

if n_params == 1:
    axes = [axes]  # wrap in list for uniform iteration

fig.suptitle("MCMC Chain Evolution", fontsize=14)

for iparam, ax in enumerate(axes):
    # Plot all walkers
    for walker in range(nwalkers):
        ax.plot(samples[:, walker, iparam], alpha=0.5, color="k")

    # Compute stats
    mean = np.mean(samples[..., iparam])
    std = np.std(samples[..., iparam])

    # Mean and ±1σ lines
    ax.axhline(mean, color="r", linestyle="--", label=f"Mean: {mean:.3f}")
    ax.axhline(mean + std, color="b", linestyle=":", label=f"±1σ: {std:.3f}")
    ax.axhline(mean - std, color="b", linestyle=":")

    # Burn-in line
    if discard > 0:
        ax.axvline(discard, color="g", linestyle="--", alpha=0.5, label="Burn-in")

    # Labels & formatting
    ax.set_ylabel(fit_parameters_names[iparam])
    ax.set_xlim(0, nsteps)
    ax.grid(True, alpha=0.3)

    # Put legend outside
    ax.legend(loc="right", bbox_to_anchor=(1.25, 0.5))

    if iparam == n_params - 1:
        ax.set_xlabel("Step number")

plt.tight_layout()
plt.savefig(folder_save + "mcmc_samples.svg", bbox_inches="tight", dpi=300)
plt.close()

### Triangle Plots
means = np.mean(samples_flat, axis=0)
stds = np.std(samples_flat, axis=0)
n_params = len(fit_parameters_names)

##############################
# --------- CORNER --------- #
##############################

if n_params == 1:
    # Corner cannot do a triangle with one param → do 1D posterior
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    ax.hist(samples_flat[:, 0], bins=40, density=True, alpha=0.6)
    ax.axvline(means[0], linestyle="--", lw=1.5)
    ax.axvline(means[0] - stds[0], linestyle=":", lw=1)
    ax.axvline(means[0] + stds[0], linestyle=":", lw=1)
    ax.set_xlabel(fit_parameters_names[0])
    ax.set_ylabel("Density")
    ax.text(0.05, 0.95, rf"$\mu={means[0]:.3f}$" + "\n" + rf"$\sigma={stds[0]:.3f}$", transform=ax.transAxes, ha="left", va="top", bbox=dict(facecolor="white", alpha=0.8))
    fig.suptitle("Posterior — Corner (1D)", fontsize=16, fontweight="bold", y=1.01)
else:
    corner_kwargs = dict(
        labels=[rf"${p}$" for p in fit_parameters_names],
        show_titles=True,
        title_fmt=".3f",
        quantiles=[0.16, 0.5, 0.84],
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 14},
        color="black",
        hist_kwargs={"density": True, "lw": 1.5},
        smooth=1.0,
    )

    fig = corner.corner(samples_flat, **corner_kwargs)
    fig.suptitle("Posterior Triangle — Corner", fontsize=16, fontweight="bold")

    axes = np.array(fig.axes).reshape(n_params, n_params)
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax = axes[i, i]
        txt = rf"$\mu = {mean:.3f}$" + "\n" + rf"$\sigma = {std:.3f}$"
        ax.text(0.05, 0.95, txt, transform=ax.transAxes, ha="left", va="top", fontsize=9, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

out_svg = os.path.join(folder_save, "triangle_corner.svg")
fig.savefig(out_svg, bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"[OK] Corner plot saved: {out_svg}")


###############################
# --------- GETDIST --------- #
###############################

gds_samples = MCSamples(
    samples=samples_flat,
    names=fit_parameters_names,
    labels=[p.replace("_", r"\_") for p in fit_parameters_names],
    settings={"smooth_scale_1D": 0.5, "smooth_scale_2D": 0.6, "mult_bias_correction_order": 0},
)

g = plots.get_single_plotter() if n_params == 1 else plots.get_subplot_plotter()

if n_params == 1:
    g.plot_1d(gds_samples, fit_parameters_names[0], filled=True)
    ax = g.fig.axes[0]
    ax.axvline(means[0], linestyle="--", lw=1.5)
    ax.axvline(means[0] - stds[0], linestyle=":", lw=1)
    ax.axvline(means[0] + stds[0], linestyle=":", lw=1)
    ax.text(0.5, 1.0, fit_parameters_names[0] + rf" = ${means[0]:.3f} \pm {stds[0]:.3f}$", transform=ax.transAxes, ha="center", va="bottom", fontsize=12)
    plt.suptitle("Posterior — GetDist", fontsize=16, fontweight="bold", y=1.05)

else:
    g.settings.num_plot_contours = 2
    g.settings.alpha_filled_add = 0.6
    g.settings.linewidth = 2
    g.settings.figure_legend_frame = False
    g.settings.axes_fontsize = 12
    g.settings.axes_labelsize = 14
    g.settings.legend_fontsize = 12
    g.settings.title_limit_fontsize = 12

    g.triangle_plot(gds_samples, params=fit_parameters_names, filled=True)

    axes = g.subplots
    for i in range(n_params):
        for j in range(i + 1):
            ax = axes[i, j]
            if i == j:
                ax.axvline(means[j], linestyle="--", lw=1.5)
                ax.axvline(means[j] - stds[j], linestyle=":", lw=1)
                ax.axvline(means[j] + stds[j], linestyle=":", lw=1)
                ax.text(0.5, 1.02, fit_parameters_names[i] + rf" $ = {means[j]:.3f} \pm {stds[j]:.3f}$", transform=ax.transAxes, ha="center", va="bottom", fontsize=10, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
            else:
                ax.axhline(means[i], linestyle="--", lw=1.5)
                ax.axhline(means[i] - stds[i], linestyle=":", lw=1)
                ax.axhline(means[i] + stds[i], linestyle=":", lw=1)
                ax.axvline(means[j], linestyle="--", lw=1.5)
                ax.axvline(means[j] - stds[j], linestyle=":", lw=1)
                ax.axvline(means[j] + stds[j], linestyle=":", lw=1)

    plt.suptitle("Posterior Triangle — GetDist", fontsize=16, fontweight="bold", y=1.05)

out_svg = os.path.join(folder_save, "triangle_getdist.svg")
plt.savefig(out_svg, bbox_inches="tight", dpi=300)
plt.close()
del g
print(f"[OK] GetDist plot saved: {out_svg}")

### Save Pickle
dict = {"samples": samples, "samples_flat": samples_flat, "parameters": files._reads_one_file(0, "parameters"), "fit_parameters": fit_params}
with open(folder_save + "mcmc.pkl", "wb") as handle:
    pkl.dump(dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
