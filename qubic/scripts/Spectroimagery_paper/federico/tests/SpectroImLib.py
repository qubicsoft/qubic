from __future__ import division
import healpy as hp
import numpy as np

import qubic


# ========================================
# These 3 functions have been put in input_sky.py (will bee remove soon)
# def scaling_dust(freq1, freq2, sp_index=1.59):
#     '''
#     Calculate scaling factor for dust contamination
#     Frequencies are in GHz
#     '''
#     freq1 = float(freq1)
#     freq2 = float(freq2)
#     x1 = freq1 / 56.78
#     x2 = freq2 / 56.78
#     S1 = x1 ** 2. * np.exp(x1) / (np.exp(x1) - 1) ** 2.
#     S2 = x2 ** 2. * np.exp(x2) / (np.exp(x2) - 1) ** 2.
#     vd = 375.06 / 18. * 19.6
#     scaling_factor_dust = (np.exp(freq1 / vd) - 1) / \
#                           (np.exp(freq2 / vd) - 1) * \
#                           (freq2 / freq1) ** (sp_index + 1)
#     scaling_factor_termo = S1 / S2 * scaling_factor_dust
#     return scaling_factor_termo
#
#
# def cmb_plus_dust(cmb, dust, Nbsubbands, sub_nus, kind='IQU'):
#     '''
#     Sum up clean CMB map with dust using proper scaling coefficients
#     '''
#     Nbpixels = cmb.shape[0]
#     nstokes = len(kind)  # Number of stokes parameters used in the simu
#     x0 = np.zeros((Nbsubbands, Nbpixels, 3))
#     # Let's fill the maps:
#     for i in range(Nbsubbands):
#         for istokes in xrange(nstokes):
#             if kind == 'QU':  # This condition keeps the order IQU in the healpix map
#                 x0[i, :, istokes + 1] = cmb.T[istokes + 1] + dust.T[istokes + 1] * scaling_dust(150, sub_nus[i], 1.59)
#             else:
#                 x0[i, :, istokes] = cmb.T[istokes] + dust.T[istokes] * scaling_dust(150, sub_nus[i], 1.59)
#     return x0
#
#
# def create_input_sky(d, skypars):
#     Nf = int(d['nf_sub'])
#     band = d['filter_nu'] / 1e9
#     filter_relative_bandwidth = d['filter_relative_bandwidth']
#     _, _, nus_in, _, _, Nbbands_in = qubic.compute_freq(band, filter_relative_bandwidth, Nf)
#     # seed
#     if d['seed']:
#         np.random.seed(d['seed'])
#         # Generate the input CMB map
#         sp = qubic.read_spectra(skypars['r'])
#         cmb = np.array(hp.synfast(sp, d['nside'], new=True, pixwin=True, verbose=False)).T
#         # Generate the dust map
#         coef = skypars['dust_coeff']
#         ell = np.arange(1, 3 * d['nside'])
#         fact = (ell * (ell + 1)) / (2 * np.pi)
#         spectra_dust = [np.zeros(len(ell)),
#                         coef * (ell / 80.) ** (-0.42) / (fact * 0.52),
#                         coef * (ell / 80.) ** (-0.42) / fact,
#                         np.zeros(len(ell))]
#         dust = np.array(hp.synfast(spectra_dust, d['nside'], new=True, pixwin=True, verbose=False)).T
#
#         # Combine CMB and dust. As output we have N 3-component maps of sky.
#         x0 = cmb_plus_dust(cmb, dust, Nbbands_in, nus_in, d['kind'])
#         return x0
# ===================================

def input_sky_pysm(sky_config, d):
    """
    Create as many skies as the number of input frequencies.
    
    The parameter `sky_config` must be a `pysm` configuration dictionary while 
    `d` must be the qubic configuration dictionary. For more details see the 
    `pysm` documentation at the floowing link: 
    https://pysm-public.readthedocs.io/en/latest/index.html

    Return a vector of shape (number_of_input_subfrequencies, npix, 3)
    """
    sky = pysm.Sky(sky_config)
    sky_signal = sky.signal()
    Nf = int(d['nf_sub'])
    band = d['filter_nu']/1e9
    filter_relative_bandwidth = d['filter_relative_bandwidth']
    _, _, nus_in, _, _, Nbbands_in = qubic.compute_freq(
        band, filter_relative_bandwidth, Nf)
    return np.rollaxis(sky_signal(nu=nus_in), 2, 1)


def create_acquisition_operator_TOD(pointing, d):
    # scene
    s = qubic.QubicScene(d)
    if d['nf_sub']==1:
        q = qubic.QubicInstrument(d)
        return qubic.QubicAcquisition(q, pointing, s, d)
    else:
        # Polychromatic instrument model
        q = qubic.QubicMultibandInstrument(d)
        # number of sub frequencies to build the TOD
        _, nus_edge_in, _, _, _, _ = qubic.compute_freq(d['filter_nu'] / 1e9,
                                                    d['filter_relative_bandwidth'],
                                                    d['nf_sub'])  # Multiband instrument model
        # Multi-band acquisition model for TOD fabrication
        return qubic.QubicMultibandAcquisition(q, pointing, s, d, nus_edge_in)


def create_TOD(d, pointing, x0):
    atod = create_acquisition_operator_TOD(pointing, d)
    TOD, _ = atod.get_observation(x0, noiseless=d['noiseless'])
    return TOD


def create_acquisition_operator_REC(pointing, d, nf_sub_rec):
    # Polychromatic instrument model
    q = qubic.QubicMultibandInstrument(d)
    # scene
    s = qubic.QubicScene(d)
    # number of sub frequencies for reconstruction
    _, nus_edge, _, _, _, _ = qubic.compute_freq(d['filter_nu'] / 1e9,
                                                 d['filter_relative_bandwidth'], nf_sub_rec)
    # Operator for Maps Reconstruction
    arec = qubic.QubicMultibandAcquisition(q, pointing, s, d, nus_edge)
    return arec


def get_hitmap(instrument, scene, pointings, threshold=0.01):
    beams = instrument.get_synthbeam(scene)
    ratio = beams / np.max(beams, axis=1)[..., None]
    beams[ratio >= threshold] = 1
    beams[beams != 1] = 0
    beam = np.sum(beams, axis=0)
    t, p = hp.pix2ang(scene.nside, np.arange(hp.nside2npix(scene.nside)))
    rot_beams = np.zeros((len(pointings), len(beam)))
    for i, (theta, phi) in enumerate(zip(pointings.galactic[:, 1],
                                         pointings.galactic[:, 0])):
        r = hp.Rotator(deg=False, rot=[np.deg2rad(phi),
                                       np.pi / 2 - np.deg2rad(theta)])
        trot, prot = r(t, p)
        rot_beams[i] = hp.get_interp_val(beam, trot, prot)
    return rot_beams


def reconstruct_maps(TOD, d, pointing, nf_sub_rec, x0=None):
    _, nus_edge, nus, _, _, _ = qubic.compute_freq(d['filter_nu'] / 1e9,
                                                   d['filter_relative_bandwidth'], nf_sub_rec)
    arec = create_acquisition_operator_REC(pointing, d, nf_sub_rec)
    cov = arec.get_coverage()
    maps_recon = arec.tod2map(TOD, cov=cov, tol=d['tol'], maxiter=1500)
    if x0 is None:
        return maps_recon, cov, nus, nus_edge
    else:
        _, maps_convolved = arec.get_observation(x0)
        maps_convolved = np.array(maps_convolved)
        return maps_recon, cov, nus, nus_edge, maps_convolved
