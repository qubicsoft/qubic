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
from pyoperators import pcg, DiagonalOperator, UnpackOperator
from qubic import (
    QubicAcquisition, QubicInstrument, create_random_pointings, gal2equ)

# read the input map
DATAPATH = os.path.join(os.path.dirname(qubic.__file__), 'data',
                        'syn256_pol.fits')
input_map = qubic.io.read_map(DATAPATH, field='I_STOKES')

# let's take the galactic north pole as the center of the observation field
center = gal2equ(0, 90)

# instrument model
instrument = QubicInstrument('monochromatic,nopol')

# observation model
np.random.seed(0)
sampling = create_random_pointings(center, 1000, 10)

# acquisition model
acq = QubicAcquisition(instrument, sampling)
hit = acq.get_hitmap()

C = acq.get_convolution_peak_operator()
P = acq.get_projection_peak_operator()
H = P * C

# Produce the Time-Ordered data
tod = H(input_map)
#acq.save_simulation('mysimul', input_map, tod, 'my info')

# map-making
coverage = P.pT1()
mask = coverage > 10
P.restrict(mask)
unpack = UnpackOperator(mask)
solution = pcg(P.T * P, P.T(tod),
               M=DiagonalOperator(1/coverage[mask]), disp=True)
output_map = unpack(solution['x'])

# some display
def display(x, title):
    x = x.copy()
    x[~mask] = np.nan
    hp.gnomview(x, rot=center_gal, reso=5, xsize=600, min=-200, max=200,
                title=title)

display(input_map, 'Original map')
display(C(input_map), 'Convolved original map')
display(output_map, 'Reconstructed map (gaussian approximation)')
mp.show()
