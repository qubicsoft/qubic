import pickle as pkl
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml
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
parameters_all_files = files._reads_all_files("parameters", verbose=True)
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

### Save samples
dict = {"samples": samples, "samples_flat": samples_flat, "parameters": files._reads_one_file(0, "parameters")}
with open(folder_save + "mcmc.pkl", "wb") as handle:
    pkl.dump(dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

### Plot
plt.figure()
for iparam in range(samples.shape[-1]):
    plt.subplot(samples.shape[-1], 1, iparam + 1)
    plt.plot(samples[..., iparam], "-k", alpha=0.1)
plt.savefig(folder_save + "mcmc_samples.svg")

### Print
print()
print(f"Average : {np.mean(samples_flat, axis=0)}")
print(f"Error : {np.std(samples_flat, axis=0)}")
print()
