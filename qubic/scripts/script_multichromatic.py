from __future__ import division
import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
from qubic import (create_random_pointings, gal2equ,
                  read_spectra,
                  compute_freq,
                  QubicScene,
                  QubicMultibandInstrument,
                  QubicMultibandAcquisition,
                  PlanckAcquisition,
                  QubicMultibandPlanckAcquisition)

# These two functions should be rewritten and added to qubic package

def scaling_dust(freq1, freq2, sp_index=1.59): 
    '''
    Calculate scaling factor for dust contamination
    Frequencies are in GHz
    '''
    freq1 = float(freq1)
    freq2 = float(freq2)
    x1 = freq1 / 56.78
    x2 = freq2 / 56.78
    S1 = x1**2. * np.exp(x1) / (np.exp(x1) - 1)**2.
    S2 = x2**2. * np.exp(x2) / (np.exp(x2) - 1)**2.
    vd = 375.06 / 18. * 19.6
    scaling_factor_dust = (np.exp(freq1 / vd) - 1) / \
                          (np.exp(freq2 / vd) - 1) * \
                          (freq2 / freq1)**(sp_index + 1)
    scaling_factor_termo = S1 / S2 * scaling_factor_dust
    return scaling_factor_termo

def cmb_plus_dust(cmb, dust, Nbsubbands, sub_nus):
    '''
    Sum up clean CMB map with dust using proper scaling coefficients
    '''
    Nbpixels = cmb.shape[0]
    x0 = np.zeros((Nbsubbands, Nbpixels, 3))
    for i in range(Nbsubbands):
        x0[i, :, 0] = cmb.T[0] + dust.T[0] * scaling_dust(150, sub_nus[i], 1.59)
        x0[i, :, 1] = cmb.T[1] + dust.T[1] * scaling_dust(150, sub_nus[i], 1.59)
        x0[i, :, 2] = cmb.T[2] + dust.T[2] * scaling_dust(150, sub_nus[i], 1.59)
    return x0

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
# Here we use only Nf=5 frequencies
Nf = 5
relative_bandwidth = 0.25
Nbfreq, nus_edge, nus, deltas, Delta, Nbbands = compute_freq(band, relative_bandwidth, 5)

# Use random pointings
p = create_random_pointings(center, 1000, 10)

# Polychromatic instrument model
q = QubicMultibandInstrument(filter_nus=nus * 1e9,
                             filter_relative_bandwidths=nus / deltas,
                             ripples=True) # The peaks of the synthesized beam are modeled with "ripples"

s = QubicScene(nside=nside, kind='IQU')

# Multi-band acquisition model
# Nsb=2 sub-bands to reconstruct the CMB
Nsb = 2
Nbfreq, nus_edge, nus_, deltas_, Delta_, Nbbands_ = compute_freq(band, relative_bandwidth, Nsb)
a = QubicMultibandAcquisition(q, p, s, nus_edge, effective_duration=2)

# Generate the input CMB map
sp = read_spectra(0)
cmb = np.array(hp.synfast(sp, nside, new=True, pixwin=True)).T

# Generate the dust map
coef = 1.39e-2
ell = np.arange(1, 1000)
fact = (ell * (ell + 1)) / (2 * np.pi)
spectra_dust = [np.zeros(len(ell)), 
                coef * (ell / 80.)**(-0.42) / (fact * 0.52), 
                coef * (ell / 80.)**(-0.42) / fact, 
                np.zeros(len(ell))]
dust = np.array(hp.synfast(spectra_dust, nside, new=True, pixwin=True)).T

# Combine CMB and dust. As output we have N 3-component maps of sky.
x0 = cmb_plus_dust(cmb, dust, Nbbands, nus)

# Simulate the TOD. Here we use Nf frequencies over the 150 GHz band
TOD, maps_convolved = a.get_observation(x0)

# Reconstruct CMB. Use Nsb sub-bands
maps_recon = a.tod2map(TOD, tol=tol)

# Get coverage maps. This returns Nf coverage maps
cov = a.get_coverage()

# We need coverages for Nsb sub-bands
cov = np.array([cov[(nus > nus_edge[i]) * (nus < nus_edge[i+1])].mean(axis=0) for i in xrange(Nsb)])

_max = [300, 5, 5]
for iband, (inp, rec, c) in enumerate(zip(maps_convolved, maps_recon, cov)):
    mp.figure(iband + 1)
    for i, (inp_, rec_, iqu) in enumerate(zip(inp.T, rec.T, 'IQU')):
        inp_[c < c.max() * 0.01] = hp.UNSEEN
        rec_[c < c.max() * 0.01] = hp.UNSEEN
        diff = inp_ - rec_
        diff[c < c.max() * 0.01] = hp.UNSEEN
        hp.gnomview(inp_, rot=center_gal, reso=5, xsize=700, fig=1,
            sub=(3, 3, i + 1), min=-_max[i], max=_max[i], title='Input convolved, {}, {:.0f} GHz'.format(iqu, nus[iband]))
        hp.gnomview(rec_, rot=center_gal, reso=5, xsize=700, fig=1,
            sub=(3, 3, i + 4), min=-_max[i], max=_max[i], title='Recon, {}, {:.0f} GHz'.format(iqu, nus[iband]))
        hp.gnomview(diff, rot=center_gal, reso=5, xsize=700, fig=1,
            sub=(3, 3, i+7), min=-_max[i], max=_max[i], title='Diff, {}, {:.0f} GHz'.format(iqu, nus[iband]))

mp.show()