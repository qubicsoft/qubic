from __future__ import division

import matplotlib.pyplot as mp
import numexpr as ne
import numpy as np
from scipy.constants import c, pi

from pyoperators import MaskOperator
from pysimulators import create_fitsheader, DiscreteSurface
from qubic import QubicInstrument
from qubic.beams import GaussianBeam

qubic = QubicInstrument('monochromatic,nopol')

NU = 150e9                   # [Hz]
LAMBDA = c / NU              # [m]
FOCAL_LENGTH = 0.3           # [m]

SOURCE_POWER = 1             # [W]
SOURCE_THETA = np.radians(0) # [rad]
SOURCE_PHI = np.radians(45)  # [rad]

PRIMARY_BEAM_FWHM = 14       # [degrees]
SECONDARY_BEAM_FWHM = 14     # [degrees]

HORN_OPEN = np.zeros(qubic.horn.shape, dtype=bool)  # all closed
HORN_OPEN[10,10] = True
HORN_OPEN[11,11] = True
HORN_OPEN[0,1] = True
HORN_OPEN[17,18] = True
HORN_OPEN = ~qubic.horn.removed  # all open

DETECTOR_OFFSET = [0, 0]  # [m]

NPOINT_DETECTOR_PLANE = 512**2 # number of detector plane sampling points
DETECTOR_PLANE_LIMITS = (np.nanmin(qubic.detector.vertex[..., 0]),
                         np.nanmax(qubic.detector.vertex[..., 0]))  # [m]

# to check energy conservation (unrealistic detector plane):
#DETECTOR_PLANE_LIMITS = (-4, 4) # [m]

def ang2vec(theta_rad, phi_rad):
    sintheta = np.sin(theta_rad)
    return np.array([sintheta * np.cos(phi_rad),
                     sintheta * np.sin(phi_rad),
                     np.cos(theta_rad)])


########
# BEAMS
########
primary_beam = GaussianBeam(PRIMARY_BEAM_FWHM)
secondary_beam = GaussianBeam(SECONDARY_BEAM_FWHM, backward=True)


########
# HORNS
########
nhorn_open = np.sum(HORN_OPEN)
horn_vec = np.column_stack([qubic.horn.center[HORN_OPEN],
                            np.zeros(nhorn_open)])


#################
# DETECTOR PLANE
#################
def norm(x):
    return x / np.sqrt(np.sum(x**2, axis=-1))[..., None]
ndet_x = int(np.sqrt(NPOINT_DETECTOR_PLANE))
a = np.r_[DETECTOR_PLANE_LIMITS[0]:DETECTOR_PLANE_LIMITS[1]:ndet_x*1j]
det_x, det_y = np.meshgrid(a, a)
det_spacing = (DETECTOR_PLANE_LIMITS[1] - DETECTOR_PLANE_LIMITS[0]) / ndet_x
det_vec = np.dstack([det_x, det_y, np.zeros_like(det_x) - FOCAL_LENGTH])
det_uvec = norm(det_vec)
det_theta = np.arccos(det_uvec[..., 2])

# solid angle of a detector plane pixel (gnomonic projection)
central_pixel_sr = np.arctan(det_spacing / FOCAL_LENGTH)**2
detector_plane_pixel_sr = -central_pixel_sr * np.cos(det_theta)**3


############
# DETECTORS
############
qubic.detector.vertex += DETECTOR_OFFSET
header = create_fitsheader((ndet_x, ndet_x), cdelt=det_spacing, crval=(0, 0),
                           ctype=['X---CAR', 'Y---CAR'], cunit=['m', 'm'])
detector_plane = DiscreteSurface.fromfits(header)
integ = MaskOperator(qubic.detector.removed) * \
        detector_plane.get_integration_operator(qubic.detector.vertex)


########
# MODEL
########
def get_model_A():
    """ Phase and transmission from the switches to the focal plane. """
    transmission = np.sqrt(secondary_beam(det_theta) /
                           secondary_beam.sr *
                           detector_plane_pixel_sr)[..., None]
    const = 2j * pi / LAMBDA
    product = np.dot(det_uvec, horn_vec.T)
    return ne.evaluate('transmission * exp(const * product)')


def get_model_B():
    """ Phase and transmission from the source to the switches. """
    source_uvec = ang2vec(SOURCE_THETA, SOURCE_PHI)
    source_E = np.sqrt(SOURCE_POWER) * np.sqrt(primary_beam(SOURCE_THETA))
    const = 2j * pi / LAMBDA
    product = np.dot(horn_vec, source_uvec)
    return ne.evaluate('source_E * exp(const * product)')


def get_model():
    return np.sum(get_model_A() * get_model_B(), axis=-1)


###############
# COMPUTATIONS
###############
E = get_model()
I = np.abs(E)**2
D = integ(I)


##########
# DISPLAY
##########
mp.figure()
mp.imshow(np.log10(I), interpolation='nearest', origin='lower')
mp.autoscale(False)
qubic.detector.plot(transform=detector_plane.topixel)
mp.figure()
mp.imshow(D, interpolation='nearest')
mp.gca().format_coord = lambda x,y: 'x={} y={} z={}'.format(x, y, D[x,y])
mp.show()
print('Given {} horns, we get {} W in the detector plane and {} W in the detec'
      'tors.'.format(nhorn_open, np.sum(I), np.sum(D)))
