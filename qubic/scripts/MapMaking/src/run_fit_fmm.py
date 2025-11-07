import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
from pyoperators import MPI
from qubic.lib.MapMaking.FrequencyMapMaking.Qspectra_component import SkySpectra
from qubic.lib.Qfit import FitEllSpace
from qubic.lib.Qfoldertools import MergeAllFiles, create_folder_if_not_exists
from qubic.lib.Qmpi_tools import MpiTools

comm = MPI.COMM_WORLD
mpi = MpiTools(comm)

############################
######## Parameters ########
############################

### Folder where spectrum should be stored
folder = "FMM/test/"
folder_spectrum = folder + "Spectrum/"
folder_save = folder + "Fit/"
create_folder_if_not_exists(comm, folder_save)

### Parameters file
parameters_file = "FMM/fit_params.yml"

### Number of multipole (minimum is fixed at what you defined during the map-making)
NBINS = 7
SAMPLE_VARIANCE = True
DIAGONAL = False
FSKY = 0.015
DL = 30

### Which frequency you want ot choose
nus_index = np.array([True, True, False, False, False, False, False, False, False])

DISCARD = 100
NWALKERS = 10
NSTEPS = 200
VERBOSE = False

############################

### Concatenate all realizations
files = MergeAllFiles(folder_spectrum)
mpi._print_message(f"Number of realizations : {files.number_of_realizations}")

### Check if all files have the same parameters
mpi._print_message("    => Checking that all spectrum are from simulations with same parameters")
parameters_all_files = files._reads_all_files("parameters", verbose=True)
test_all_same = np.all(parameters_all_files == parameters_all_files[0])
if test_all_same:
    mpi._print_message("    => All Parameters are the same !")
else:
    raise ValueError("    => All Parameters aren't the same ! Check your simulations !!!")

### Multipoles
mpi._print_message("    => Reading multipoles")
ell = files._reads_one_file(0, "ell")[:NBINS]

### Frequencies
mpi._print_message("    => Reading frequencies")
nus = files._reads_one_file(0, "nus")[nus_index]
mpi._print_message(f"nus : {nus}")


BBsignal = np.mean(files._reads_all_files("Dls", verbose=VERBOSE), axis=0)[nus_index, :, :NBINS][:, nus_index, :NBINS]
BBnoise = files._reads_all_files("Nls")[:, :, nus_index, :NBINS][:, nus_index, :, :NBINS]

### Remove noise bias
mpi._print_message("    => Removing noise bias")
BBsignal -= np.mean(BBnoise, axis=0)

### Define sky model in ell space
sky = SkySpectra(ell, nus)

### Fit of cosmological parameters
mpi._print_message("    => Fitting parameters")
fit = FitEllSpace(ell, BBsignal, BBnoise, model=sky.model, parameters_file=parameters_file, sample_variance=SAMPLE_VARIANCE, fsky=FSKY, dl=DL, diagonal=DIAGONAL)
samples, samples_flat = fit.run(NSTEPS, NWALKERS, discard=DISCARD, comm=comm)

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
