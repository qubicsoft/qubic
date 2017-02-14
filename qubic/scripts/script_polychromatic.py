from __future__ import division
import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
from qubic import (create_random_pointings, gal2equ,
                  read_spectra,
                  compute_freq,
                  QubicScene,
                  QubicMultibandInstrument,
                  QubicPolyAcquisition,
                  PlanckAcquisition,
                  QubicPolyPlanckAcquisition)

nside = 256
# Observe a patch at the galactic north pole
center_gal = 0, 90
center = gal2equ(center_gal[0], center_gal[1])

band = 150
tol = 1e-4

# Compute frequencies at which we sample the TOD.
# As input we set 150 [GHz] - the central frequency of the band and
# the relative bandwidth -- 0.25.
# The bandwidth is assumed to be uniform.
# The number of frequencies is set by an optional parameter Nfreq, 
# which is, by default, 15 for 150 GHz band
# and 20 for 220 GHz band.
Nbfreq, nus_edge, nus, deltas, Delta, Nbbands = compute_freq(band, 0.25)

# Use random pointings for the test
p = create_random_pointings(center, 1000, 10)

# Polychromatic instrument model
q = QubicMultibandInstrument(filter_nus=nus * 1e9,
                             filter_relative_bandwidths=nus / deltas,
                             ripples=True) # The peaks of the synthesized beam are modeled with "ripples"

s = QubicScene(nside=nside, kind='IQU')

# Polychromatic acquisition model
a = QubicPolyAcquisition(q, p, s, effective_duration=2)

sp = read_spectra(0)
x0 = np.array(hp.synfast(sp, nside, new=True, pixwin=True)).T

TOD, maps_convolved = a.get_observation(x0)
maps_recon = a.tod2map(TOD, tol=tol)

cov = a.get_coverage().sum(axis=0)

mp.figure(1)
_max = [300, 5, 5]
for i, (inp, rec, iqu) in enumerate(zip(maps_convolved.T, maps_recon.T, 'IQU')):
    inp[cov < cov.max() * 0.01] = hp.UNSEEN
    rec[cov < cov.max() * 0.01] = hp.UNSEEN
    diff = inp - rec
    diff[cov < cov.max() * 0.01] = hp.UNSEEN
    hp.gnomview(inp, rot=center_gal, reso=5, xsize=700, fig=1,
        sub=(3, 3, i + 1), min=-_max[i], max=_max[i], title='Input, {}'.format(iqu))
    hp.gnomview(rec, rot=center_gal, reso=5, xsize=700, fig=1,
        sub=(3, 3, i + 4), min=-_max[i], max=_max[i], title='Recon, {}'.format(iqu))
    hp.gnomview(diff, rot=center_gal, reso=5, xsize=700, fig=1,
        sub=(3, 3, i+7), min=-_max[i], max=_max[i], title='Diff, {}'.format(iqu))

# Now let's play with the fusion acquisition
planck = PlanckAcquisition(band, a.scene, true_sky=maps_convolved[0])
acq_fusion = QubicPlanckAcquisition(a, acq_planck)

TOD, maps_convolved = a.get_observation(x0)
maps_recon_fusion = a.tod2map(TOD, tol=tol)

mp.figure(2)
for i, (inp, rec, iqu) in enumerate(zip(maps_convolved.T, maps_recon_fusion.T, 'IQU')):
    inp[cov < cov.max() * 0.01] = hp.UNSEEN
    rec[cov < cov.max() * 0.01] = hp.UNSEEN
    diff = inp - rec
    diff[cov < cov.max() * 0.01] = hp.UNSEEN
    hp.gnomview(inp, rot=center_gal, reso=5, xsize=700, fig=1,
        sub=(3, 3, i + 1), min=-_max[i], max=_max[i], title='Input, {}'.format(iqu))
    hp.gnomview(rec, rot=center_gal, reso=5, xsize=700, fig=1,
        sub=(3, 3, i + 4), min=-_max[i], max=_max[i], title='Recon, {}'.format(iqu))
    hp.gnomview(diff, rot=center_gal, reso=5, xsize=700, fig=1,
        sub=(3, 3, i+7), min=-_max[i], max=_max[i], title='Diff, {}'.format(iqu))
