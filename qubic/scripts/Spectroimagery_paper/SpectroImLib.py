from __future__ import division
import healpy as hp
import numpy as np

import qubic


def create_acquisition_operator_TOD(pointing, d):
    # scene
    s = qubic.QubicScene(d)
    if d['nf_sub']==1:
        # Monochromatic instrument model
        q = qubic.QubicInstrument(d)
        # Acquisition model for TOD fabrication
        return qubic.QubicAcquisition(q, pointing, s, d)
    else:
        # Multiband instrument model
        q = qubic.QubicMultibandInstrument(d)
        # Number of sub frequencies to build the TOD
        _, nus_edge_in, _, _, _, _ = qubic.compute_freq(d['filter_nu'] / 1e9,
                                                    d['filter_relative_bandwidth'],
                                                    d['nf_sub'])
        # Multi-band acquisition model for TOD fabrication
        return qubic.QubicMultibandAcquisition(q, pointing, s, d, nus_edge_in)


def create_TOD(d, pointing, x0):
    atod = create_acquisition_operator_TOD(pointing, d)
    if d['nf_sub']==1:
        x0 = np.reshape(x0, (x0.shape[1], x0.shape[2]))
    tod = atod.get_observation(x0, convolution=False, noiseless=d['noiseless'])
    return tod


def create_acquisition_operator_REC(pointing, d, nf_sub_rec):
    # scene
    s = qubic.QubicScene(d)
    if nf_sub_rec==1:
        # Monochromatic instrument model
        q = qubic.QubicInstrument(d)
        # Operator for Maps Reconstruction
        return qubic.QubicAcquisition(q, pointing, s, d)
    else:
        # Multiband instrument model
        q = qubic.QubicMultibandInstrument(d)

        # number of sub frequencies for reconstruction
        _, nus_edge, _, _, _, _ = qubic.compute_freq(d['filter_nu'] / 1e9,
                                                     d['filter_relative_bandwidth'],
                                                     nf_sub_rec)
        # Operator for Maps Reconstruction
        return qubic.QubicMultibandAcquisition(q, pointing, s, d, nus_edge)


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


def reconstruct_maps(tod, d, pointing, nf_sub_rec, x0=None):
    _, nus_edge, nus, _, _, _ = qubic.compute_freq(d['filter_nu'] / 1e9,
                                                   d['filter_relative_bandwidth'], nf_sub_rec)
    arec = create_acquisition_operator_REC(pointing, d, nf_sub_rec)
    cov = arec.get_coverage()
    maps_recon = arec.tod2map(tod, d, cov=cov)
    if x0 is None:
        return maps_recon, cov, nus, nus_edge
    else:
        if nf_sub_rec==1:
            _, maps_convolved = arec.get_observation(np.mean(x0, axis=0))
        else:
            _, maps_convolved = arec.get_observation(x0)
        maps_convolved = np.array(maps_convolved)
        return maps_recon, cov, nus, nus_edge, maps_convolved
