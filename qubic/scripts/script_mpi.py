"""
Noiseless simulation and map-making using the gaussian approximation
for the synthetic beam.

"""
from __future__ import division
import numpy as np
import os
import qubic
from qubic import (
    QubicAcquisition, create_random_pointings, gal2equ, map2tod, tod2map_all)

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
hit = acq.get_hitmap()

# Produce the Time-Ordered data
tod = map2tod(acq, input_map)
output_map, coverage = tod2map_all(acq, tod)

print acq.comm.rank, output_map[coverage > 0][:5]
print acq.comm.rank, hit[hit > 0][:10]
