from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
from pyoperators import DiagonalOperator, PackOperator, pcg, rules_inplace
from pysimulators import SphericalEquatorial2GalacticOperator
from qubic import QubicAcquisition, QubicInstrument, create_sweeping_pointings

# acquisition parameters
nside = 256
racenter = 0.0
deccenter = -57.0
angspeed = 1    # deg/sec
delta_az = 15.
angspeed_psi = 0.1
maxpsi = 45.
nsweeps_el = 300
duration = 24   # hours
ts = 20         # seconds

# get the observation model
pointing = create_sweeping_pointings(
    [racenter, deccenter], duration, ts, angspeed, delta_az, nsweeps_el,
    angspeed_psi, maxpsi)
pointing.angle_hwp = np.random.random_integers(0, 7, pointing.size) * 22.5

# get the instrument model with only one detector
idetector = 0
instrument = QubicInstrument('monochromatic')
mask_packed = np.ones(len(instrument.detector.packed), bool)  # remove all
mask_packed[idetector] = False                                # except this one
mask_unpacked = instrument.unpack(mask_packed)
instrument = QubicInstrument('monochromatic', removed=mask_unpacked)

# get the acquisition model from the instrument and observation models
obs = QubicAcquisition(instrument, pointing)

# get it as operators
convolution = obs.get_convolution_peak_operator()
projection = obs.get_projection_peak_operator()
hwp = obs.get_hwp_operator()
polarizer = obs.get_polarizer_operator()

# restrict the projection to the observed sky pixels
coverage = projection.pT1()
mask = coverage > 0
projection.restrict(mask)
pack = PackOperator(mask, broadcast='rightward')

with rules_inplace():
    H = polarizer * (hwp * projection)

# get noiseless timeline
x0 = np.array(hp.read_map('test/data/syn256_pol.fits', field=(0, 1, 2))).T
x0_convolved = convolution(x0)
y = H(pack(x0_convolved))

# inversion through Preconditioned Conjugate Gradient
preconditioner = DiagonalOperator(1/coverage[mask], broadcast='rightward')
solution = pcg(H.T * H, H.T(y), M=preconditioner, disp=True, tol=1e-3)
output_map = pack.T(solution['x'])


# some display
def display(input, msg):
    for i, (kind, lim) in enumerate(zip('IQU', [200, 10, 10])):
        map = input[..., i].copy()
        map[~mask] = np.nan
        hp.gnomview(map, rot=center, reso=5, xsize=400, min=-lim, max=lim,
                    title=msg + ' ' + kind, sub=(1, 3, i+1))

e2g = SphericalEquatorial2GalacticOperator(degrees=True)
center = e2g([racenter, deccenter])

mp.figure(1)
display(x0, 'Original map')
mp.figure(2)
display(output_map, 'Reconstructed map')
mp.figure(3)
display(output_map-x0, 'Difference map')
mp.show()
