from __future__ import division
from pyoperators import pcg
from pysimulators import profile
from qubic import (
    create_random_pointings, equ2gal, QubicAcquisition, PlanckAcquisition,
    QubicPlanckAcquisition, QubicScene)
from qubic.data import PATH
from qubic.io import read_map
import healpy as hp
import matplotlib.pyplot as mp
import numpy as np

racenter = 0.0      # deg
deccenter = -57.0   # deg
maxiter = 1000
tol = 5e-6

sky = read_map(PATH + 'syn256_pol.fits')
nside = 256
sampling = create_random_pointings([racenter, deccenter], 1000, 10)
scene = QubicScene(nside)

acq_qubic = QubicAcquisition(150, sampling, scene, effective_duration=1)
convolved_sky = acq_qubic.instrument.get_convolution_peak_operator()(sky)
acq_planck = PlanckAcquisition(150, acq_qubic.scene, true_sky=convolved_sky)
acq_fusion = QubicPlanckAcquisition(acq_qubic, acq_planck)

H = acq_fusion.get_operator()
invntt = acq_fusion.get_invntt_operator()
y = acq_fusion.get_observation()

A = H.T * invntt * H
b = H.T * invntt * y

solution_fusion = pcg(A, b, disp=True, maxiter=maxiter, tol=tol)

acq_qubic = QubicAcquisition(150, sampling, scene, effective_duration=1)
H = acq_qubic.get_operator()
invntt = acq_qubic.get_invntt_operator()
y, sky_convolved = acq_qubic.get_observation(sky, convolution=True)

A = H.T * invntt * H
b = H.T * invntt * y

solution_qubic = pcg(A, b, disp=True, maxiter=maxiter, tol=tol)


# some display
def display(input, msg, iplot=1):
    out = []
    for i, (kind, lim) in enumerate(zip('IQU', [50, 5, 5])):
        map = input[..., i]
        out += [hp.gnomview(map, rot=center, reso=5, xsize=800, min=-lim,
                            max=lim, title=msg + ' ' + kind,
                            sub=(3, 3, iplot + i), return_projected_map=True)]
    return out

center = equ2gal(racenter, deccenter)
mp.figure(1)
mp.clf()
solution_qubic['x'][solution_qubic['x'] == 0] = np.nan
display(convolved_sky, 'Original map', iplot=1)
display(solution_qubic['x'], 'Reconstructed map', iplot=4)
res_qubic = display(solution_qubic['x'] - convolved_sky,
                    'Difference map', iplot=7)

mp.figure(2)
mp.clf()
display(convolved_sky, 'Original map', iplot=1)
display(solution_fusion['x'], 'Reconstructed map', iplot=4)
res_fusion = display(solution_fusion['x'] - convolved_sky,
                     'Difference map', iplot=7)

mp.figure(3)
for res, color in zip((res_qubic, res_fusion), ('blue', 'green')):
    for i, kind in enumerate('IQU'):
        axis = mp.subplot(3, 1, i+1)
        x, y = profile(res[i]**2)
        x *= 5 / 60
        y = np.sqrt(y)
        y *= np.degrees(np.sqrt(4 * np.pi / acq_qubic.scene.npixel))
        mp.plot(x, y, color=color)
        mp.title(kind)
        mp.ylabel('Sensitivity [$\mu$K deg]')

mp.xlabel('Angular distance [degrees]')

# BICEP-2 / Planck
# sigmas = 1.2 * np.array([1 / np.sqrt(2), 1, 1])
sigmas = np.std(acq_planck.sigma, 0)
for i, sigma in enumerate(sigmas):
    axis = mp.subplot(3, 1, i+1)
    mp.axhline(sigma, color='red')

hp.mollzoom(solution_fusion['x'][:, 0] - convolved_sky[:, 0])

mp.show()
