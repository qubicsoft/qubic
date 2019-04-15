from __future__ import division

import healpy as hp
import numpy as np

import os
import pysm
import qubic
import warnings


__all__ = ['sky', 'Planck_sky', 'Qubic_sky']


class sky(object):
    """
    Define a sky object as seen by an instrument.
    """
    def __init__(self, skyconfig, d, instrument):
        """
        Parameters:
        skyconfig  : a skyconfig dictionary to pass to (as expected by) `PySM`
        d          : input dictionary from which the parameters are read
        instrument : a `PySM` instrument describing the instrument

        For more details about `PySM` see its documentation at the floowing link:
        https://pysm-public.readthedocs.io/en/latest/index.html
        """
        self.skyconfig = skyconfig
        self.dictionary = d
        self.instrument = instrument
        self.sky = pysm.Sky(skyconfig)

        
    def get_simple_sky_map(self):
        """
        Create as many skies as the number of the qubic sub-frequencies. 
        Instrumental effects are not considered. For this purpose use the 
        `get_sky_map` method.
        Return a vector of shape (number_of_input_subfrequencies, npix, 3)
        """
        sky_signal = self.sky.signal()
        Nf = self.dictionary['nf_sub']
        band = self.dictionary['filter_nu']/1e9
        filter_relative_bandwidth = self.dictionary['filter_relative_bandwidth']
        _, _, central_nus, _, _, _ = qubic.compute_freq(
            band, filter_relative_bandwidth, Nf)
        return np.rollaxis(sky_signal(nu=central_nus), 2, 1)

    
    def get_sky_map(self, noise_map=False):
        """
        Returns the maps saved in the `output_directory` containing the 
        `output_prefix`. If
        there are no maps in the `ouput_directory` they will be created.
        """
        output, noise = self.instrument.observe(self.sky, write_outputs=False)
        if output.shape != noise.shape:
            warnings.warn("signal and noise maps have different shapes!")
        if noise_map:
            return np.rollaxis(output+noise, 2, 1), np.rollaxis(noise, 2, 1)
        else:
            return np.rollaxis(output+noise, 2, 1)

    
class Planck_sky(sky):
    """
    Define a sky object as seen by Planck.
    """
    def __init__(self, skyconfig, d, band=143, channel_length=100):
        self.band = band
        self.planck_central_nus = np.array(
            [30, 44, 70, 100, 143, 217, 353, 545, 857])
        self.planck_relative_bandwidths = np.array(
            [0.2, 0.2, 0.2, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33])
        self.planck_beams = np.array([33, 24, 14, 10, 7.1, 5, 5, 5, 5])
        self.planck_Isensitivities_pixel = np.array(
            [2, 2.7, 4.7, 2.5, 2.2, 4.8, 14.7, 147, 6700])
        self.planck_Psensitivities_pixel = np.array(
            [2.8, 3.9, 6.7, 4.0, 4.2, 9.8, 29.8, np.NaN, np.NaN])
        self.planck_channels = self.create_planck_bandwidth(channel_length)
        self.planck_channels_names = [
            '33_GHz', '44_GHz','70_GHz','100_GHz', '143_GHz', '217_GHz',
            '353_GHz', '545_GHz', '857_GHz']

        if band is not None:
            idx = np.argwhere(self.planck_central_nus == band)[0][0]
            instrument = pysm.Instrument({
                'nside': d['nside'],
                'frequencies' : self.planck_central_nus[idx:idx+1], # GHz
                'use_smoothing' : True,
                'beams' : self.planck_beams[idx:idx+1], # arcmin 
                'add_noise' : True, 
                'noise_seed' : 0,  
                'sens_I': self.get_planck_sensitivity("I")[idx:idx+1],
                'sens_P': self.get_planck_sensitivity("P")[idx:idx+1],
                'use_bandpass' : True,  
                'channel_names' : self.planck_channels_names[idx:idx+1],
                'channels' : self.planck_channels[idx:idx+1],
                'output_units' : 'uK_RJ',
                'output_directory' : "./",
                'output_prefix' : "planck_one",
                'pixel_indices' : None})
        else:
            instrument = pysm.Instrument({
                'nside': d['nside'],
                'frequencies' : self.planck_central_nus, # GHz
                'use_smoothing' : True,
                'beams' : self.planck_beams, # arcmin 
                'add_noise' : True,  
                'noise_seed' : 0,  
                'sens_I': self.get_planck_sensitivity("I"),
                'sens_P': self.get_planck_sensitivity("P"),
                'use_bandpass' : True, 
                'channel_names' : self.planck_channels_names,
                'channels' : self.planck_channels,
                'output_units' : 'uK_RJ',
                'output_directory' : "./",
                'output_prefix' : "planck_all",
                'pixel_indices' : None})
            
        sky.__init__(self, skyconfig, d, instrument)

        
    def create_planck_bandwidth(self, length):
        """
        Returns a list of bandwidths and respectively weights correponding to the
        ideal Planck bandwidths. `planck_central_nus` must be an array containing
        the central frequency of the channel while the 
        `planck_relative_bandwidth` parameter must be an array containig the 
        relative bandwidths for each Planck channel. `length` is the length of 
        the output array; default is 100.
        """
        halfband = self.planck_relative_bandwidths * self.planck_central_nus / 2
        bandwidths = np.zeros((len(self.planck_relative_bandwidths), length))
        v = []
        for i, hb in enumerate(halfband):
            bandwidths[i] = np.linspace(self.planck_central_nus[i] - hb,
                                        self.planck_central_nus[i] + hb,
                                        num=length)
            v.append((bandwidths[i], np.ones_like(bandwidths[i])))
        return v

    
    def get_planck_sensitivity(self, kind):
        """
        Convert the sensitiviy per pixel to sensitivity per arcmin. Units are 
        'uK_CMB arcmin'. The sensitivity per pixel is given by:
        sigma_pix = sigma_arcmin / sqrt(FHWM_beam**2) 
        since it contains FHWM_beam**2 arcminute-beams that sum as a sum with the
        propagation of errors law.
        """
        C = pysm.pysm.convert_units("uK_RJ", "uK_CMB", self.planck_central_nus)
        if kind == "I":
            return self.planck_Isensitivities_pixel * self.planck_beams * C
        elif kind == "P":
            return self.planck_Psensitivities_pixel * self.planck_beams * C
        else:
            raise ValueError("kind must be `I` or `P` ")


class Qubic_sky(sky):
    """
    Define a sky object as seen by Qubic
    """
    def __init__(self, skyconfig, d):

        Nf = d['nf_sub']
        band = d['filter_nu']/1e9
        filter_relative_bandwidth = d['filter_relative_bandwidth']
        _, _, central_nus, _, _, _ = qubic.compute_freq(
            band, filter_relative_bandwidth, Nf)
        names = [np.str(np.round(cn, 2)) for cn in central_nus]
        names = [n.replace('.', 'p') for n in names]
        instrument = pysm.Instrument({
            'nside': d['nside'],
            'frequencies' : central_nus, # GHz
            'use_smoothing' : False,
            'beams': np.ones_like(central_nus), # arcmin 
            'add_noise': False,  
            'noise_seed' : 0,  
            'sens_I': np.ones_like(central_nus),
            'sens_P': np.ones_like(central_nus),
            'use_bandpass': False,  
            'channel_names': names,
            'channels': np.ones_like(central_nus),
            'output_units': 'uK_RJ',
            'output_directory': "./",
            'output_prefix': "qubic",
            'pixel_indices': None})
        
        sky.__init__(self, skyconfig, d, instrument)
        

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
        _, nus_edge_in, _, _, _, _ = qubic.compute_freq(
            d['filter_nu'] / 1e9, d['filter_relative_bandwidth'], d['nf_sub'])
        
        return qubic.QubicMultibandAcquisition(q, pointing, s, d, nus_edge_in)


def create_TOD(d, pointing, x0):
    atod = create_acquisition_operator_TOD(pointing, d)
    if d['nf_sub']==1:
        TOD = atod.get_observation(x0[0], noiseless=d['noiseless'])
    else:
        TOD, _ = atod.get_observation(x0, noiseless=d['noiseless'])
    return TOD


def create_acquisition_operator_REC(pointing, d, nf_sub_rec):
    # Polychromatic instrument model
    q = qubic.QubicMultibandInstrument(d)
    # scene
    s = qubic.QubicScene(d)
    # number of sub frequencies for reconstruction
    _, nus_edge, _, _, _, _ = qubic.compute_freq(d['filter_nu'] / 1e9,
                                                 d['filter_relative_bandwidth'],
                                                 nf_sub_rec)
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
    _, nus_edge, nus, _, _, _ = qubic.compute_freq(
        d['filter_nu'] / 1e9, d['filter_relative_bandwidth'], nf_sub_rec)
    arec = create_acquisition_operator_REC(pointing, d, nf_sub_rec)
    cov = arec.get_coverage()
    maps_recon = arec.tod2map(TOD, cov=cov, tol=d['tol'], maxiter=1500)
    if x0 is None:
        return maps_recon, cov, nus, nus_edge
    else:
        _, maps_convolved = arec.get_observation(x0)
        maps_convolved = np.array(maps_convolved)
        return maps_recon, cov, nus, nus_edge, maps_convolved
