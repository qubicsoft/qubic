import os
import pickle as pkl
import sys

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
folder_save = folder + "/Fit/"
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
parameters = parameters_all_files[0]
test_all_same = np.all(parameters_all_files == parameters)
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
fig, axes = plt.subplots(samples.shape[-1], 1, figsize=(12, 3 * samples.shape[-1]))
fig.suptitle("MCMC Chain Evolution", fontsize=14)

for iparam, ax in enumerate(axes):
    # Plot all walkers
    for walker in range(nwalkers):
        ax.plot(samples[:, walker, iparam], alpha=0.5, color="k")

    # Plot mean and std
    mean = np.mean(samples[..., iparam])
    std = np.std(samples[..., iparam])
    ax.axhline(mean, color="r", linestyle="--", label=f"Mean: {mean:.3f}")
    ax.axhline(mean + std, color="b", linestyle=":", label=f"±1σ: {std:.3f}")
    ax.axhline(mean - std, color="b", linestyle=":")

    # Add vertical line for burn-in
    if discard > 0:
        ax.axvline(discard, color="g", linestyle="--", alpha=0.5, label="Burn-in")

    # Customize each subplot
    ax.set_ylabel(fit_parameters_names[iparam])
    ax.set_xlim(0, nsteps)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="right", bbox_to_anchor=(1.2, 0.5))

    # Only show x-label for the bottom plot
    if iparam == samples.shape[-1] - 1:
        ax.set_xlabel("Step number")

# Adjust layout
plt.tight_layout()
plt.savefig(folder_save + "mcmc_samples.svg", bbox_inches="tight", dpi=300)
plt.close()

### Triangle Plots
means = np.mean(samples_flat, axis=0)
stds = np.std(samples_flat, axis=0)

### Triangle Plot - Corner
corner_kwargs = dict(
    labels=[rf"${p}$" for p in fit_parameters_names],  # LaTeX labels
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
fig.suptitle("Posterior Triangle — Corner", fontsize=16, fontweight="bold", y=1.01)

axes = np.array(fig.axes).reshape(len(fit_parameters_names), len(fit_parameters_names))
for i, (mean, std) in enumerate(zip(means, stds)):
    ax = axes[i, i]
    txt = rf"$\mu = {mean:.3f}$" + "\n" + rf"$\sigma = {std:.3f}$"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, ha="left", va="top", fontsize=9, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

out_svg = os.path.join(folder_save, "triangle_corner.svg")
fig.savefig(out_svg, bbox_inches="tight", dpi=300)
plt.close(fig)

print(f"[OK] Corner plot saved: {out_svg}")

### Triangle Plot - GetDist
gds_samples = MCSamples(
    samples=samples_flat,
    names=fit_parameters_names,
    labels=[p.replace("_", r"\_") for p in fit_parameters_names],
    settings={
        "smooth_scale_1D": 0.5,
        "smooth_scale_2D": 0.6,
        "mult_bias_correction_order": 0,
    },
)

g = plots.get_subplot_plotter()
g.settings.num_plot_contours = 2
g.settings.alpha_filled_add = 0.6
g.settings.linewidth = 2
g.settings.figure_legend_frame = False
g.settings.axes_fontsize = 12
g.settings.axes_labelsize = 14
g.settings.legend_fontsize = 12
g.settings.title_limit_fontsize = 12

# Create the triangle plot
g.triangle_plot(
    gds_samples,
    params=fit_parameters_names,
    filled=True,
)

# Get the figure and its axes
fig = g.fig
axes = g.subplots

# Add lines and text to each subplot
n = len(fit_parameters_names)
for i in range(n):
    for j in range(i + 1):  # Only lower triangle
        ax = axes[i, j]
        if ax is not None:  # Check if the axis exists
            # Add vertical lines
            ax.axvline(means[j], color="red", linestyle="--", lw=1.5)
            ax.axvline(means[j] - stds[j], color="blue", linestyle=":", lw=1)
            ax.axvline(means[j] + stds[j], color="blue", linestyle=":", lw=1)

            # Add horizontal lines (for non-diagonal plots)
            if i != j:
                ax.axhline(means[i], color="red", linestyle="--", lw=1.5)
                ax.axhline(means[i] - stds[i], color="blue", linestyle=":", lw=1)
                ax.axhline(means[i] + stds[i], color="blue", linestyle=":", lw=1)

            # Add text only on diagonal plots
            if i == j:
                ax.text(
                    0.5,
                    1.08,
                    rf"{fit_parameters_names[i]}: ${means[i]:.3f} \pm {stds[i]:.3f}$",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=10,
                )

plt.suptitle("Posterior Triangle — GetDist", fontsize=16, fontweight="bold", y=1.03)

out_svg = os.path.join(folder_save, "triangle_getdist.svg")
plt.savefig(out_svg, bbox_inches="tight", dpi=300)
plt.close()
del g  # prevent destructor warning

print(f"[OK] GetDist plot saved: {out_svg}")

### Save Pickle
dict = {"samples": samples, "samples_flat": samples_flat, "parameters": files._reads_one_file(0, "parameters"), "fit_parameters": fit_params}
with open(folder_save + "mcmc.pkl", "wb") as handle:
    pkl.dump(dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
