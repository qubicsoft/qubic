from selfcal_lib import *

basedir = '../'
dictfilename = basedir + 'global_source.dict'

d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)

q = qubic.QubicMultibandInstrument(d)

s = qubic.QubicScene(d)

# Source parameters : positions and spectral irradiance
# We move the source instead of the instrument
phi = np.arange(0., 0.8, 0.2)
theta = np.arange(0., 0.8, 0.2)
irradiance = 1.

# Horns (indeces on the instrument)
baseline_manip = [19, 1]

# Get all combinations in the focal plane
S_tot, Cminus_i, Cminus_j, Sminus_ij, Ci, Cj, Sij = selfcal_data(q[0], theta, phi,
                                                                 irradiance, baseline_manip, reso=34,
                                                                 xmin=-0.06, xmax=0.06,
                                                                 dead_switch=None, doplot=True)

# Reduce to the quarter of the focal plane
S_tot_quarter = full2quarter(S_tot)

# Plot example for the first pointing
figure()
imshow((S_tot - Cminus_i)[:, :, 0], interpolation='nearest')
title('$S_{tot} - C_{-i}$ ')
colorbar()
