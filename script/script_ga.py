"""
Noiseless simulation and map-making using the gaussian approximation
for the synthetic beam.

"""
from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
import os

from pyoperators import pcg, DiagonalOperator, UnpackOperator
from pysimulators import ProjectionInMemoryOperator
from qubic import QubicConfiguration, QubicInstrument, create_random_pointings

path = os.path.dirname(__file__)

kmax = 2
qubic = QubicInstrument('monochromatic,nopol')
pointings = create_random_pointings(1000, 10)
input_map = hp.read_map(os.path.join(path, '..', 'test', 'data', 'syn256.fits'))

# configure observation
obs = QubicConfiguration(qubic, pointings)
C = obs.get_convolution_peak_operator()
P = obs.get_projection_peak_operator(kmax=kmax)
H = P * C

# Produce the Time-Ordered data
tod = H(input_map)
obs.save_simulation('mysimul', input_map, tod, 'my info')

# map-making
coverage = P.T(np.ones_like(tod))
mask = coverage < 10
P.matrix.pack(mask)
P_packed = ProjectionInMemoryOperator(P.matrix)
unpack = UnpackOperator(mask)
solution = pcg(P_packed.T * P_packed, P_packed.T(tod), M=DiagonalOperator(1/coverage[~mask]), disp=True)
output_map = unpack(solution['x'])

# some display
orig = input_map.copy()
orig[mask] = np.nan
hp.gnomview(orig, rot=[0,90], reso=5, xsize=600, min=-200, max=200, title='Original map')
cmap = C(input_map)
cmap[mask] = np.nan
hp.gnomview(cmap, rot=[0,90], reso=5, xsize=600, min=-200, max=200, title='Convolved original map')
output_map[mask] = np.nan
hp.gnomview(output_map, rot=[0,90], reso=5, xsize=600, min=-200, max=200, title='Reconstructed map (simulpeak)')

mp.show()
