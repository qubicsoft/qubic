"""
We compute the output power collected on the focal plane, given 1 W entering
the instrument through the open horns.

"""
from __future__ import division

import matplotlib.pyplot as mp
import numpy as np
from pyoperators import MaskOperator
from pysimulators import create_fitsheader, SceneGrid
from qubic import QubicInstrument

NU = 150e9                   # [Hz]

SOURCE_THETA = np.radians(0) # [rad]
SOURCE_PHI = np.radians(0)   # [rad]

NPOINT_FOCAL_PLANE = 512**2  # number of detector plane sampling points

qubic = QubicInstrument(filter_nu=NU)

# example for a baseline selection:
#qubic.horn.open[:] = False
#qubic.horn.open[0] = True
#qubic.horn.open[14] = True

FOCAL_PLANE_LIMITS = (np.nanmin(qubic.detector.vertex[..., 0]),
                      np.nanmax(qubic.detector.vertex[..., 0]))  # [m]

# to check energy conservation (unrealistic detector plane):
FOCAL_PLANE_LIMITS = (-0.2, 0.2) # [m]


#################
# FOCAL PLANE
#################
nfp_x = int(np.sqrt(NPOINT_FOCAL_PLANE))
a = np.r_[FOCAL_PLANE_LIMITS[0]:FOCAL_PLANE_LIMITS[1]:nfp_x*1j]
fp_x, fp_y = np.meshgrid(a, a)
fp_spacing = (FOCAL_PLANE_LIMITS[1] - FOCAL_PLANE_LIMITS[0]) / nfp_x


############
# DETECTORS
############
header = create_fitsheader((nfp_x, nfp_x), cdelt=fp_spacing, crval=(0, 0),
                           ctype=['X---CAR', 'Y---CAR'], cunit=['m', 'm'])
focal_plane = SceneGrid.fromfits(header)
integ = MaskOperator(qubic.detector.all.removed) * \
        focal_plane.get_integration_operator(
            focal_plane.topixel(qubic.detector.all.vertex[..., :2]))


###############
# COMPUTATIONS
###############

# we make sure that exacty 1 W goes through the open horns
SOURCE_POWER = 1 / (np.sum(qubic.horn.open) * np.pi * qubic.horn.radius**2)
E = qubic.get_response(SOURCE_THETA, SOURCE_PHI, SOURCE_POWER,
                       x=fp_x, y=fp_y, area=fp_spacing**2)
I = np.abs(E)**2
D = integ(I)


##########
# DISPLAY
##########
mp.figure()
mp.imshow(np.log10(I), interpolation='nearest', origin='lower')
mp.autoscale(False)
qubic.detector.plot(transform=focal_plane.topixel)
mp.figure()
mp.imshow(np.log(D), interpolation='nearest')
mp.gca().format_coord = lambda x, y: 'x={} y={} z={}'.format(x, y, D[x, y])
mp.show()
print('From an input of 1 W, we get {} W on the detector plane and {} W in the'
      ' detectors.'.format(np.sum(I), np.sum(D)))
