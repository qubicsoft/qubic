"""
Noiseless simulation and map-making using the gaussian approximation
for the synthetic beam.

"""
from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
import os
import qubic
from qubic import (
    create_random_pointings, gal2equ, map2tod, QubicAcquisition,
    SimpleAcquisition, tod2map_all)

# read the input map
DATAPATH = os.path.join(os.path.dirname(qubic.__file__), 'data',
                        'syn256_pol.fits')
input_map = qubic.io.read_map(DATAPATH, field='I_STOKES')

# let's take the galactic north pole as the center of the observation field
center_gal = 0, 90
center = gal2equ(center_gal[0], center_gal[1])

# sampling model
np.random.seed(0)
sampling = create_random_pointings(center, 1000, 10)

# acquisition model
acq = QubicAcquisition(150, sampling, kind='I')
acq_simple = SimpleAcquisition(150, sampling, kind='I')
hit = acq.get_hitmap()
C = acq.get_convolution_peak_operator()

tod = map2tod(acq, input_map)
output_map, coverage = tod2map_all(acq, tod)
output_map[coverage == 0] = np.nan
coverage[coverage == 0] = np.nan

tod_simple = map2tod(acq_simple, input_map)
output_map_simple, coverage_simple = tod2map_all(acq_simple, tod_simple)
output_map_simple[coverage_simple == 0] = np.nan
coverage_simple[coverage_simple == 0] = np.nan


# some display
def display(x, title, min=-200, max=200):
    hp.gnomview(x, rot=center_gal, reso=5, xsize=600, min=min, max=max,
                title=title)

display(input_map, 'Original map')
display(C(input_map), 'Convolved original map')
display(output_map, 'Reconstructed map (gaussian approximation)')
display(output_map_simple, 'Reconstructed map (simple model)')
maxval = coverage_simple.max()
display(coverage, 'Coverage', min=0, max=maxval)
display(coverage_simple, 'Coverage (simple model)', min=0, max=maxval)
mp.show()
