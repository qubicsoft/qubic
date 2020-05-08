from __future__ import division

import healpy as hp
import numpy as np

import qubic


def create_acquisition_operator_TOD(pointing, d, verbose=False):
    # scene
    s = qubic.QubicScene(d)
    if d['nf_sub'] == 1:
        if verbose :
            print('Making a QubicInstrument.')
        q = qubic.QubicInstrument(d)
        return qubic.QubicAcquisition(q, pointing, s, d)
    else:
        # Polychromatic instrument model
        if verbose :
            print('Making a QubicMultibandInstrument.')
        q = qubic.QubicMultibandInstrument(d)
        # number of sub frequencies to build the TOD
        _, nus_edge_in, _, _, _, _ = qubic.compute_freq(d['filter_nu'] / 1e9, d['nf_sub'],  # Multiband instrument model
                                                        d['filter_relative_bandwidth'])
        # Multi-band acquisition model for TOD fabrication
        return qubic.QubicMultibandAcquisition(q, pointing, s, d, nus_edge_in)


def create_TOD(d, pointing, x0, verbose=False):
    atod = create_acquisition_operator_TOD(pointing, d, verbose=verbose)

    if d['nf_sub'] == 1:
        TOD, maps_convolved = atod.get_observation(x0[0], noiseless=d['noiseless'])
    else:
        TOD, maps_convolved = atod.get_observation(x0, noiseless=d['noiseless'])
    return TOD, maps_convolved


def create_acquisition_operator_REC(pointing, d, nf_sub_rec):
    # scene
    s = qubic.QubicScene(d)
    if d['nf_sub'] == 1:
        q = qubic.QubicInstrument(d)
        # one subfreq for recons
        _, nus_edge, _, _, _, _ = qubic.compute_freq(d['filter_nu'] / 1e9, nf_sub_rec, d['filter_relative_bandwidth'])
        arec = qubic.QubicAcquisition(q, pointing, s, d)

        return arec
    else:
        q = qubic.QubicMultibandInstrument(d)
        # number of sub frequencies for reconstruction
        _, nus_edge, _, _, _, _ = qubic.compute_freq(d['filter_nu'] / 1e9, nf_sub_rec, d['filter_relative_bandwidth'])
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
    for i, (theta, phi) in enumerate(zip(pointings.galactic[:, 1], pointings.galactic[:, 0])):
        r = hp.Rotator(deg=False, rot=[np.deg2rad(phi), np.pi / 2 - np.deg2rad(theta)])
        trot, prot = r(t, p)
        rot_beams[i] = hp.get_interp_val(beam, trot, prot)
    return rot_beams


def reconstruct_maps(TOD, d, pointing, nf_sub_rec, x0=None):
    _, nus_edge, nus, _, _, _ = qubic.compute_freq(d['filter_nu'] / 1e9, nf_sub_rec, d['filter_relative_bandwidth'])
    arec = create_acquisition_operator_REC(pointing, d, nf_sub_rec)
    cov = arec.get_coverage()
    maps_recon, nit, error = arec.tod2map(TOD, d, cov=cov)
    if not d['verbose']:
        print('niterations = {}, error = {}'.format(nit, error))
    if x0 is None:
        return maps_recon, cov, nus, nus_edge
    else:
        if d['nf_sub'] == 1:
            _, maps_convolved = arec.get_observation(x0[0], noiseless=d['noiseless'])
        else:
            _, maps_convolved = arec.get_observation(x0, noiseless=d['noiseless'])
        maps_convolved = np.array(maps_convolved)
        return maps_recon, cov, nus, nus_edge, maps_convolved
