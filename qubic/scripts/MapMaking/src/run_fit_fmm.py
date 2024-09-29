import matplotlib.pyplot as plt
import numpy as np
from pyoperators import MPI

from qubic.lib.Qfit import FitEllSpace
from qubic.lib.Qfoldertools import MergeAllFiles
from qubic.lib.MapMaking.FrequencyMapMaking.Qspectra_component import SkySpectra
from qubic.lib.Qmpi_tools import MpiTools

############################
######## Parameters ########
############################

### Folder where spectrum should be stored
folder_spectrum = 'path/to/your/data/folder'

### Parameters file
parameters_file = 'path/to/your/parameter_file.txt'

### Number of multipole (minimum is fixed at what you defined during the map-making)
NBINS = 10
SAMPLE_VARIANCE = True
DIAGONAL = False
FSKY = 0.01
DL = 30

### Which frequency you want ot choose ?
nus_index = np.array([True, True, True, True,
                      False, False, False, False, False, True, True])

DISCARD = 100
NWALKERS = 10
NSTEPS = 200
VERBOSE = False

############################

comm = MPI.COMM_WORLD
mpi = MpiTools(comm)

### Concatenate all realizations
files = MergeAllFiles(folder_spectrum)
mpi._print_message(f'Number of realizations : {files.number_of_realizations}')
### Multipoles
mpi._print_message('    => Reading multipoles')
ell = files._reads_one_file(0, "ell")[:NBINS]

### Frequencies
mpi._print_message('    => Reading frequencies')
nus = files._reads_one_file(0, "nus")[nus_index]
mpi._print_message(f'nus : {nus}')


BBsignal = np.mean(files._reads_all_files("Dls", verbose=VERBOSE), axis=0)[nus_index, :, :NBINS][:, nus_index, :NBINS]
BBnoise = files._reads_all_files("Nl")[:, :, nus_index, :NBINS][:, nus_index, :, :NBINS]

### Remove noise bias
mpi._print_message('    => Removing noise bias')
BBsignal -= np.mean(BBnoise, axis=0)

### Define sky model in ell space
sky = SkySpectra(ell, nus)

### Fit of cosmological parameters
mpi._print_message('    => Fitting parameters')
fit = FitEllSpace(ell, BBsignal, BBnoise, model=sky.model, parameters_file=parameters_file, sample_variance=SAMPLE_VARIANCE, fsky=FSKY, dl=DL, diagonal=DIAGONAL)
samples, samples_flat = fit.run(NSTEPS, NWALKERS, discard=DISCARD, comm=comm)

plt.figure()

plt.subplot(3, 1, 1)
plt.plot(samples[..., 0], "-k", alpha=0.1)
plt.axhline(0)
plt.subplot(3, 1, 2)
plt.plot(samples[..., 1], "-k", alpha=0.1)
plt.subplot(3, 1, 3)
plt.plot(samples[..., 2], "-k", alpha=0.1)

plt.show()

print()
print(f"Average : {np.mean(samples_flat, axis=0)}")
print(f"Error : {np.std(samples_flat, axis=0)}")
print()
