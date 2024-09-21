import os

import matplotlib.pyplot as plt
import numpy as np
from multiprocess import Pool
from pyoperators import MPI
from schwimmbad import MPIPool

from lib.Qfit import FitEllSpace
from lib.Qfoldertools import MergeAllFiles
from qubic.lib.MapMaking.FrequencyMapMaking.Qspectra_component import SkySpectra

comm = MPI.COMM_WORLD

### Concatenate all realizations
files = MergeAllFiles("/Users/mregnier/Desktop/git/Pipeline/src/FMM/CMBDUST_nrec2_new_code/spectrum/")

nus_index = np.array([True, True, False, False, False, False, False, False, True])
NBINS = 16

ell = files._reads_one_file(0, "ell")[:NBINS]
nus = files._reads_one_file(0, "nus")[nus_index]

BBsignal = np.mean(files._reads_all_files("Dls"), axis=0)[:, nus_index, :NBINS][
    nus_index, :, :NBINS
]
BBnoise = files._reads_all_files("Nl")[:, :, nus_index, :NBINS][:, nus_index, :, :NBINS]
BBsignal -= np.mean(BBnoise, axis=0)

sky = SkySpectra(ell, nus)
fit = FitEllSpace(ell, BBsignal, BBnoise, model=sky.model)

samples, samples_flat = fit.run(300, 10, discard=200, comm=comm)

plt.figure()
plt.plot(samples[..., 0], "-k", alpha=0.1)
plt.axhline(0)
plt.show()

print()
print(f"Average : {np.mean(samples_flat, axis=0)}")
print(f"Error : {np.std(samples_flat, axis=0)}")
print()
