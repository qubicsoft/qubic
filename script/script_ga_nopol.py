"""
Noiseless simulation and map-making using the gaussian approximation
for the synthetic beam.

"""
from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np

from pyoperators import pcg, DiagonalOperator, UnpackOperator
from pysimulators import SphericalGalactic2EquatorialOperator
from qubic import QubicAcquisition, QubicInstrument, create_random_pointings

# Let's take the galactic north pole as the center of the observation field
center_gal = [0, 90]
center = SphericalGalactic2EquatorialOperator(degrees=True)(center_gal)
kmax = 2

np.random.seed(0)
qubic = QubicInstrument('monochromatic,nopol', kmax=kmax)
pointings = create_random_pointings(center, 1000, 10)
input_map = hp.read_map('test/data/syn256.fits')

# configure observation
obs = QubicAcquisition(qubic, pointings)
hit = obs.get_hitmap()

C = obs.get_convolution_peak_operator()
P = obs.get_projection_peak_operator()
H = P * C

# Produce the Time-Ordered data
tod = H(input_map)
#obs.save_simulation('mysimul', input_map, tod, 'my info')

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
