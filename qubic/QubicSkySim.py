from __future__ import division

import healpy as hp
import numpy as np
import random
import string
import os
import pysm
import pysm.units as u
import camb

import qubic

__all__ = ['sky', 'Qubic_sky']


class sky(object):
    """
    Define a sky object as seen by an instrument.
    """

    def __init__(self, skyconfig, d, instrument, out_dir, out_prefix):
        """
        Parameters:
        skyconfig  : a skyconfig dictionary to pass to (as expected by) `PySM`
        d          : input dictionary, from which the following Parameters are read
        instrument : a `PySM` instrument describing the instrument
        out_dir    : default path where the sky maps will be saved
        out_prefix : default word for the output files

        For more details about `PySM` see the `PySM` documentation at the floowing link: 
        https://pysm-public.readthedocs.io/en/latest/index.html
        """
        self.skyconfig = skyconfig
        self.nside = d['nside']
        self.dictionary = d
        self.instrument = instrument
        self.output_directory = out_dir
        self.output_prefix = out_prefix
        self.input_cmb_maps = None
        self.input_cmb_spectra = None
        iscmb = False
        preset_strings = []
        for k in skyconfig.keys():
            if k == 'cmb':
                iscmb = True
                keyword = skyconfig[k]
                if isinstance(keyword, dict):
                    # the CMB part is defined via a dictionary
                    # This can be either a set of maps, a set of CAMB spectra, or whatever
                    # In the second case it might also contain the seed (None means rerun it each time)
                    # In the third case we recompute some CAMB spectra and generate the maps
                    keys = keyword.keys()
                    if 'IQUMaps' in keys:
                        # this is the case where we have IQU maps
                        mymaps = keyword['IQUMaps']
                        self.input_cmb_maps = mymaps
                        self.input_cmb_spectra = None
                    elif 'CAMBSpectra' in keys:
                        # this is the case where we have CAMB Spectra
                        # Note that they are in l(l+1) CL/2pi so we have to change that for synfast
                        totDL = keyword['CAMBSpectra']
                        ell = keyword['ell']
                        mycls = Dl2Cl_without_monopole(ell, totDL)
                        # set the seed if needed
                        if 'seed' in keys:
                            np.random.seed(keyword['seed'])
                        mymaps = hp.synfast(mycls.T, self.nside, verbose=False, new=True)
                        self.input_cmb_maps = mymaps
                        self.input_cmb_spectra = totDL
                    else:
                        raise ValueError('Bad Dictionary given for PySM in the CMB part - see QubicSkySim.py for details')
                else:
                    # the CMB part is not defined via a dictionary but only by the seed for synfast
                    # No map nor CAMB spectra was given, so we recompute them
                    # The assumed cosmology is the default one given in the get_CAMB_Dl() function below
                    if keyword is not None: np.random.seed(keyword)
                    ell, totDL, unlensedCL = get_camb_Dl(lmax=3 * self.nside)
                    mycls = Dl2Cl_without_monopole(ell, totDL)
                    mymaps = hp.synfast(mycls.T, self.nside, verbose=False, new=True)
                    self.input_cmb_maps = mymaps
                    self.input_cmb_spectra = totDL

                # Write a tenporary file with the maps so the PySM can read them
                rndstr = random_string(10)
                hp.write_map('/tmp/' + rndstr, mymaps)
                cmbmap = pysm.CMBMap(self.nside, map_IQU='/tmp/' + rndstr)
                os.remove('/tmp/' + rndstr)
            else:
                # we add the other predefined components
                preset_strings.append(skyconfig[k])
        self.sky = pysm.Sky(nside=self.nside, preset_strings=preset_strings)
        if iscmb: self.sky.add_component(cmbmap)

    def get_simple_sky_map(self):
        """
        Create as many skies as the number of input frequencies. Instrumental
        effects are not considered. For this use the `get_sky_map` method.
        Return a vector of shape (number_of_input_subfrequencies, npix, 3)
        """
        npix = 12 * self.nside ** 2
        Nf = int(self.dictionary['nf_sub'])
        band = self.dictionary['filter_nu'] / 1e9
        filter_relative_bandwidth = self.dictionary['filter_relative_bandwidth']
        _, nus_edge, nus_in, _, _, Nbbands_in = qubic.compute_freq(band, Nf, filter_relative_bandwidth)

        sky = np.zeros((Nf, npix, 3))
        for i in range(Nf):
            themaps_iqu = self.sky.get_emission([nus_edge[i], nus_edge[i + 1]] * u.GHz)
            #print('Integrating from: {} to {} and converting to muKCMB at {}'.format(nus_edge[i], nus_edge[i + 1],nus_in[i] ))
            sky[i, :, :] = np.array(themaps_iqu.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(nus_in[i] * u.GHz))).T
        return sky

    def read_sky_map(self):
        """
        Returns the maps saved in the `output_directory` containing the `output_prefix`.
        """
        map_list = [s for s in os.listdir(self.output_directory) if self.output_prefix in s]
        map_list = [m for m in map_list if 'total' in m]
        if len(map_list) > len(self.instrument.Frequencies):
            map_list = np.array(
                [[m for m in map_list if x in m] for x in self.instrument.Channel_Names]).ravel().tolist()
        maps = np.zeros((len(map_list), hp.nside2npix(self.nside), 3))
        for i, title in enumerate(map_list):
            maps[i] = hp.read_map(title, field=(0, 1, 2)).T
        return map_list, maps

    def get_sky_map(self):
        """
        Returns the maps saved in the `output_directory` containing the `output_prefix`. If
        there are no maps in the `ouput_directory` they will be created.
        """
        sky_map_list, sky_map = self.read_sky_map()
        if len(sky_map_list) < len(self.instrument.Frequencies):
            self.instrument.observe(self.sky)
            sky_map_list, sky_map = self.read_sky_map()
        return sky_map


#### This part has been commented as it is not yet compatible with PySM3
#### it was written by F. Incardona using PySM2
# class Planck_sky(sky):
#     """
#     Define a sky object as seen by Planck.
#     """

#     def __init__(self, skyconfig, d, output_directory="./", output_prefix="planck_sky", band=143):
#         self.band = band
#         self.planck_central_nus = np.array([30, 44, 70, 100, 143, 217, 353, 545, 857])
#         self.planck_relative_bandwidths = np.array([0.2, 0.2, 0.2, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33])
#         self.planck_beams = np.array([33, 24, 14, 9.2, 7.1, 5.5, 5, 5, 5])
#         self.planck_Isensitivities_pixel = np.array([2, 2.7, 4.7, 2, 2.2, 4.8, 14.7, 147, 6700])
#         self.planck_Psensitivities_pixel = np.array([2.8, 3.9, 6.7, np.NaN, 4.2, 9.8, 29.8, np.NaN, np.NaN])
#         self.planck_channels = self.create_planck_bandwidth()
#         self.planck_channels_names = ['33_GHz', '44_GHz', '70_GHz', '100_GHz', '143_GHz', '217_GHz', '353_GHz',
#                                       '545_GHz', '857_GHz']

#         if band is not None:
#             idx = np.argwhere(self.planck_central_nus == band)[0][0]
#             instrument = pysm.Instrument(
#                 {'nside': d['nside'], 'frequencies': self.planck_central_nus[idx:idx + 1],  # GHz
#                  'use_smoothing': True, 'beams': self.planck_beams[idx:idx + 1],  # arcmin
#                  'add_noise': True,  # If True `sens_I` and `sens_Q` are required
#                  'noise_seed': 0,  # Not used if `add_noise` is False
#                  'sens_I': self.get_planck_sensitivity("I")[idx:idx + 1],  # Not used if `add_noise` is False
#                  'sens_P': self.get_planck_sensitivity("P")[idx:idx + 1],  # Not used if `add_noise` is False
#                  'use_bandpass': True,  # If True pass banpasses  with the key `channels`
#                  'channel_names': self.planck_channels_names[idx:idx + 1],
#                  'channels': self.planck_channels[idx:idx + 1], 'output_units': 'uK_RJ',
#                  'output_directory': output_directory, 'output_prefix': output_prefix, 'pixel_indices': None})
#         else:
#             instrument = {'nside': d['nside'], 'frequencies': self.planck_central_nus,  # GHz
#                                           'use_smoothing': True, 'beams': self.planck_beams,  # arcmin
#                                           'add_noise': True,  # If True `sens_I` and `sens_Q` are required
#                                           'noise_seed': 0,  # Not used if `add_noise` is False
#                                           'sens_I': self.get_planck_sensitivity("I"),
#                                           # Not used if `add_noise` is False
#                                           'sens_P': self.get_planck_sensitivity("P"),
#                                           # Not used if `add_noise` is False
#                                           'use_bandpass': True,  # If True pass banpasses  with the key `channels`
#                                           'channel_names': self.planck_channels_names, 'channels': self.planck_channels,
#                                           'output_units': 'uK_RJ', 'output_directory': output_directory,
#                                           'output_prefix': output_prefix, 'pixel_indices': None}

#         sky.__init__(self, skyconfig, d, instrument, output_directory, output_prefix)

#     def create_planck_bandwidth(self, length=100):
#         """
#         Returns a list of bandwidths and respectively weights correponding to the ideal Planck bandwidths.
#         `planck_central_nus` must be an array containing the central frequency of the channel while the
#         `planck_relative_bandwidth` parameter must be an array containig the relative bandwidths for 
#         each Planck channel. `length` is the length of the output array; default is 100.
#         """
#         halfband = self.planck_relative_bandwidths * self.planck_central_nus / 2
#         bandwidths = np.zeros((len(self.planck_relative_bandwidths), length))
#         v = []
#         for i, hb in enumerate(halfband):
#             bandwidths[i] = np.linspace(self.planck_central_nus[i] - hb, self.planck_central_nus[i] + hb, num=length)
#             v.append((bandwidths[i], np.ones_like(bandwidths[i])))
#         return v

#     def get_planck_sensitivity(self, kind):
#         """
#         Convert the sensitiviy per pixel to sensitivity per arcmin.
#         """
#         if kind == "I":
#             return self.planck_Isensitivities_pixel * self.planck_beams ** 2
#         return self.planck_Psensitivities_pixel * self.planck_beams ** 2


class Qubic_sky(sky):
    """
    Define a sky object as seen by Qubic
    """

    def __init__(self, skyconfig, d, output_directory="./", output_prefix="qubic_sky"):
        _, nus_edge_in, central_nus, deltas, _, _ = qubic.compute_freq(d['filter_nu'] / 1e9, d['nf_sub'],
                                                                       # Multiband instrument model
                                                                       d['filter_relative_bandwidth'])
        self.qubic_central_nus = central_nus
        self.qubic_resolution_nus = 61.347409 / self.qubic_central_nus
        self.qubic_channels_names = ["{:.3s}".format(str(i)) + "_GHz" for i in self.qubic_central_nus]

        instrument = {'nside': d['nside'], 'frequencies': central_nus,  # GHz
                      'use_smoothing': False, 'beams': np.ones_like(central_nus),  # arcmin
                      'add_noise': False,  # If True `sens_I` and `sens_Q` are required
                      'noise_seed': 0.,  # Not used if `add_noise` is False
                      'sens_I': np.ones_like(central_nus),  # Not used if `add_noise` is False
                      'sens_P': np.ones_like(central_nus),  # Not used if `add_noise` is False
                      'use_bandpass': False,  # If True pass banpasses  with the key `channels`
                      'channel_names': self.qubic_channels_names,  # np.ones_like(central_nus),
                      'channels': np.ones_like(central_nus), 'output_units': 'uK_RJ',
                      'output_directory': output_directory, 'output_prefix': output_prefix,
                      'pixel_indices': None}

        sky.__init__(self, skyconfig, d, instrument, output_directory, output_prefix)


def get_camb_Dl(lmax=2500, H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06, As=2e-9, ns=0.965, r=0.):
    #### Inspired from: https://camb.readthedocs.io/en/latest/CAMBdemo.html
    # NB: this returns Dl = l(l+1)Cl/2pi
    # Python CL arrays are all zero based (starting at L=0), Note L=0,1 entries will be zero by default.
    # The different DL are always in the order TT, EE, BB, TE (with BB=0 for unlensed scalar results).
    ####
    # Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()
    # This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.WantTensors = True
    pars.InitPower.set_params(As=As, ns=ns, r=r)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    # calculate results for these parameters
    results = camb.get_results(pars)
    # get dictionary of CAMB power spectra
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    totDL = powers['total']
    unlensedDL = powers['unlensed_total']
    # Python CL arrays are all zero based (starting at L=0), Note L=0,1 entries will be zero by default.
    # The different CL are always in the order TT, EE, BB, TE (with BB=0 for unlensed scalar results).
    ls = np.arange(totDL.shape[0])
    return ls, totDL, unlensedDL


def Dl2Cl_without_monopole(ls, totDL):
    cls = np.zeros_like(totDL)
    for i in range(4):
        cls[2:, i] = 2 * np.pi * totDL[2:, i] / (ls[2:] * (ls[2:] + 1))
    return cls


def random_string(nchars):
    lst = [random.choice(string.ascii_letters + string.digits) for n in range(nchars)]
    str = "".join(lst)
    return (str)
