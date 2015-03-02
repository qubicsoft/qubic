"""
Noiseless simulation and map-making using the gaussian approximation
for the synthetic beam.

"""
from __future__ import division
from pyoperators import pcg, DiagonalOperator, UnpackOperator
from qubic import QubicAcquisition, create_random_pointings, gal2equ
import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
import qubic

# read the input map
input_map = qubic.io.read_map(qubic.data.PATH + 'syn256_pol.fits',
                              field='I_STOKES')

# let's take the galactic north pole as the center of the observation field
center_gal = 0, 90
center = gal2equ(center_gal[0], center_gal[1])

# sampling model
np.random.seed(0)
sampling = create_random_pointings(center, 1000, 10)

# acquisition model
acq = QubicAcquisition(150, sampling, kind='I')
hit = acq.get_hitmap()

C = acq.get_convolution_peak_operator()
P = acq.get_projection_operator()
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
