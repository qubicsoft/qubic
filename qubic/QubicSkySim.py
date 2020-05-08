from __future__ import division

import healpy as hp
import numpy as np
import random as rd
import string
import os
import pysm
import pysm.units as u
from pysm import utils
from pylab import *
from scipy.optimize import curve_fit

import qubic
from qubic import camb_interface as qc
from qubic import fibtools as ft


__all__ = ['sky', 'Qubic_sky']


class sky(object):
    """
    Define a sky object as seen by an instrument.
    """

    def __init__(self, skyconfig, d, instrument, out_dir, out_prefix, lmax=None):
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
        if lmax is None:
            self.lmax = 3*d['nside']
        else:
            self.lmax = lmax

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
                        raise ValueError(
                            'Bad Dictionary given for PySM in the CMB part - see QubicSkySim.py for details')
                else:
                    # The CMB part is not defined via a dictionary but only by the seed for synfast
                    # No map nor CAMB spectra was given, so we recompute them.
                    # The assumed cosmology is the default one given in the get_CAMB_Dl() function below.
                    if keyword is not None: np.random.seed(keyword)
                    ell, totDL, unlensedCL = get_camb_Dl(lmax=self.lmax)
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
        if iscmb:
            self.sky.add_component(cmbmap)

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
        #ww = (np.ones(Nf+1) * u.uK_CMB).to_value(u.uK_RJ,equivalencies=u.cmb_equivalencies(nus_edge*u.GHz))
        for i in range(Nf):

            ###### Marche pas mal mais je ne comprends pas pkoi
            nfreqinteg = 5
            nus = np.linspace(nus_edge[i], nus_edge[i + 1], nfreqinteg)
            #print('Nus: ',nus)
            ### Filter in one in uK_CMB
            filter_uK_CMB = np.ones(len(nus), dtype=np.double)
            #print('Filter CMB:',filter_uK_CMB)
            filter_uK_CMB_normalized = utils.normalize_weights(nus * u.GHz, filter_uK_CMB)
            #print('Filter CMB Normalized:',filter_uK_CMB_normalized)
            #print('Inverse Filter CMB Normalized:',1./filter_uK_CMB_normalized)
            weights = 1./filter_uK_CMB_normalized
 


            # #### THIS IS WHAT WOULD MAKE SENSE BUT DOES NOT WORK ~ 5% on maps w.r.t. input
            # nfreqinteg = 5
            # nus = np.linspace(nus_edge[i], nus_edge[i + 1], nfreqinteg)
            # freqs = utils.check_freq_input(nus)
            # convert_to_uK_RJ = (np.ones(len(freqs), dtype=np.double) * u.uK_CMB).to_value(
            # u.uK_RJ, equivalencies=u.cmb_equivalencies(freqs))
            # #print('Convert_to_uK_RJ :',convert_to_uK_RJ)
            # weights = np.ones(nfreqinteg) * convert_to_uK_RJ

            ### Integrate through band using filter shape defined in weights
            themaps_iqu = self.sky.get_emission(nus * u.GHz, weights=weights)
            sky[i, :, :] = np.array(themaps_iqu.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(nus_in[i] * u.GHz))).T
            # ratio = np.mean(self.input_cmb_maps[0,:]/sky[i,:,0])
            # print('Ratio to initial: ',ratio)


        return sky

    # ### This is not supported yet....
    # def read_sky_map(self):
    #     """
    #     Returns the maps saved in the `output_directory` containing the `output_prefix`.
    #     """
    #     map_list = [s for s in os.listdir(self.output_directory) if self.output_prefix in s]
    #     map_list = [m for m in map_list if 'total' in m]
    #     if len(map_list) > len(self.instrument.Frequencies):
    #         map_list = np.array(
    #             [[m for m in map_list if x in m] for x in self.instrument.Channel_Names]).ravel().tolist()
    #     maps = np.zeros((len(map_list), hp.nside2npix(self.nside), 3))
    #     for i, title in enumerate(map_list):
    #         maps[i] = hp.read_map(title, field=(0, 1, 2)).T
    #     return map_list, maps

    # def get_sky_map(self):
    #     """
    #     Returns the maps saved in the `output_directory` containing the `output_prefix`. If
    #     there are no maps in the `ouput_directory` they will be created.
    #     """
    #     sky_map_list, sky_map = self.read_sky_map()
    #     if len(sky_map_list) < len(self.instrument.Frequencies):
    #         self.instrument.observe(self.sky)
    #         sky_map_list, sky_map = self.read_sky_map()
    #     return sky_map


# ### This part has been commented as it is not yet compatible with PySM3
# ### it was written by F. Incardona using PySM2
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

    def get_fullsky_convolved_maps(self, FWHMdeg=None, verbose=None):
        """
        This returns full sky maps at each subfrequency convolved by the beam of  the  isntrument at
        each frequency or with another beam if FWHMdeg is provided.
        when FWHMdeg is 0, the maps are not convolved.

        Parameters
        ----------
        FWHMdeg
        verbose

        Returns
        -------

        """

        # First get the full sky maps
        fullmaps = self.get_simple_sky_map()
        Nf = self.dictionary['nf_sub']

        # Now convolve them to the FWHM at each sub-frequency or to a common beam if forcebeam is None
        fwhms = np.zeros(Nf)
        if FWHMdeg is not None:
            fwhms += FWHMdeg
        else:
            fwhms = self.dictionary['synthbeam_peak150_fwhm'] * 150. / self.qubic_central_nus
        for i in range(Nf):
            if fwhms[i] != 0:
                fullmaps[i, :, :] = hp.sphtfunc.smoothing(fullmaps[i, :, :].T, fwhm=np.deg2rad(fwhms[i]),
                                                          verbose=verbose).T
            self.instrument['beams'][i] = fwhms[i]

        return fullmaps

    def get_partial_sky_maps_withnoise(self, coverage, sigma_sec, 
                                        Nyears=3, verbose=False, FWHMdeg=None, seed=None,
                                        effective_variance_invcov=None):
        """
        This returns maps in the same way as with get_simple_sky_map but cut according to the coverage
        and with noise added according to this coverage and the RMS in muK.sqrt(sec) given by sigma_sec
        The default integration time is three years but can be modified with optional variable Nyears
        Note that the maps are convolved with the instrument beam by default, or with FWHMdeg (can be an array)
        if provided.
        If seed is provided, it will be used for the noise realization. If not it will be a new realization at
        each call.
        The optional effective_variance_invcov keyword is a modification law to be applied to the coverage in order to obtain
        more realistic noise profile. It is a law for effective RMS as a function of inverse coverage and is 2D array
        with the first one being (nx samples) inverse coverage and the second being the corresponding effective variance to be
        used through interpolation when generating the noise.
        Parameters
        ----------
        coverage
        sigma_sec
        Nyears
        verbose
        FWHMdeg
        seed
        effective_variance_invcov

        Returns
        -------

        """

        # First get the convolved maps
        maps = self.get_fullsky_convolved_maps(FWHMdeg=FWHMdeg, verbose=verbose)

        # Restrict maps to observed pixels
        seenpix = coverage > 0
        maps[:, ~seenpix, :] = 0

        # Now pure noise maps
        self.noisemaps = np.zeros_like(maps)
        for i in range(self.dictionary['nf_sub']):
            self.noisemaps[i, :, :] += self.create_noise_maps(sigma_sec, coverage,
                                                              Nyears=Nyears, verbose=verbose, seed=seed,
                                                              effective_variance_invcov=effective_variance_invcov)

        return maps + self.noisemaps

    def create_noise_maps(self, sigma_sec, coverage, covcut =0.1,
                            Nyears=3, verbose=False, seed=None, effective_variance_invcov=None):
        """
        This returns a realization of noise maps for I, Q and U with no correlation between them, according to a
        noise RMS map built according to the coverage specified as an attribute to the class
        The optional effective_variance_invcov keyword is a modification law to be applied to the coverage in order to obtain
        more realistic noise profile. It is a law for effective RMS as a function of inverse coverage and is 2D array
        with the first one being (nx samples) inverse coverage and the second being the corresponding effective variance to be
        used through interpolation when generating the noise.
        Parameters
        ----------
        sigma_sec
        coverage
        Nyears
        verbose
        seed
        effective_variance_invcov

        Returns
        -------

        """
        # Seen pixels
        seenpix = (coverage/np.max(coverage)) > covcut
        npix = seenpix.sum()
        # The theoretical noise in I for the coverage
        ideal_noise = self.theoretical_noise_maps(sigma_sec, coverage, Nyears=Nyears, verbose=verbose)
        if effective_variance_invcov is None:
            thnoise = ideal_noise
        else:
            thnoise = np.zeros_like(ideal_noise)
            correction = np.interp(np.max(coverage[seenpix])/coverage[seenpix], 
                effective_variance_invcov[0,:], effective_variance_invcov[1,:])
            thnoise[seenpix] = ideal_noise[seenpix] * np.sqrt(correction)


        noise_maps = np.zeros((len(coverage), 3))
        if seed is not None:
            np.random.seed(seed)
        noise_maps[seenpix, 0] = np.random.randn(npix) * thnoise[seenpix]
        noise_maps[seenpix, 1] = np.random.randn(npix) * thnoise[seenpix] * np.sqrt(2)
        noise_maps[seenpix, 2] = np.random.randn(npix) * thnoise[seenpix] * np.sqrt(2)
        return noise_maps

    def theoretical_noise_maps(self, sigma_sec, coverage, Nyears=3, verbose=False):
        """
        This returns a map of the RMS noise (not an actual realization, just the expected RMS - No covariance)

        Parameters
        ----------
        sigma_sec
        coverage
        Nyears
        verbose

        Returns
        -------

        """
        # ###### Noise normalization
        # We assume we have integrated for a time Ttot in seconds with a sigma per root sec sigma_sec
        Ttot = Nyears * 365 * 24 * 3600  # in seconds
        if verbose:
            print('Total time is {} seconds'.format(Ttot))
        # Oberved pixels
        thepix = coverage > 0
        # Normalized coverage (sum=1)
        covnorm = coverage / np.sum(coverage)
        if verbose:
            print('Normalized coverage sum: {}'.format(np.sum(covnorm)))

        # Time per pixel
        Tpix = np.zeros_like(covnorm)
        Tpix[thepix] = Ttot * covnorm[thepix]
        if verbose:
            print('Sum Tpix: {} s  ; Ttot = {} s'.format(np.sum(Tpix), Ttot))

        # RMS per pixel
        Sigpix = np.zeros_like(covnorm)
        Sigpix[thepix] = sigma_sec / np.sqrt(Tpix[thepix])
        if verbose:
            print('Total noise (with no averages in pixels): {}'.format(np.sum((Sigpix * Tpix) ** 2)))
        return Sigpix


def get_camb_Dl(lmax=2500, H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06, As=2e-9, ns=0.965, r=0.):
    return qc.get_camb_Dl(lmax=2500, H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06, As=2e-9, ns=0.965, r=0.)


def Dl2Cl_without_monopole(ls, totDL):
    return qc.Dl2Cl_without_monopole(ls, totDL)


def random_string(nchars):
    lst = [rd.choice(string.ascii_letters + string.digits) for n in range(nchars)]
    str = "".join(lst)
    return (str)

def get_noise_invcov_profile(maps, cov, covcut=0.1, nbins=100, fit=True, label='', norm=False, allstokes=False):
    seenpix = cov > (covcut*np.max(cov))
    covnorm = cov / np.max(cov)
   
    xx, yyI, dx, dyI, _ = ft.profile(np.sqrt(1./covnorm[seenpix]), maps[seenpix,0], nbins=nbins, plot=False)
    xx, yyQ, dx, dyQ, _ = ft.profile(np.sqrt(1./covnorm[seenpix]), maps[seenpix,1], nbins=nbins, plot=False)
    xx, yyU, dx, dyU, _ = ft.profile(np.sqrt(1./covnorm[seenpix]), maps[seenpix,2], nbins=nbins, plot=False)
    avg = np.sqrt((dyI**2+dyQ**2/2+dyU**2/2)/3)
    if norm:
        fact = xx[0]/avg[0]
    else:
        fact = 1.
    myY = (avg/xx) * fact

    p=plot(xx**2,myY, 'o', label=label)
    if allstokes:
        plot(xx**2, dyI/xx*fact, label=label+' I', alpha=0.3)
        plot(xx**2, dyQ/xx*fact/np.sqrt(2), label=label+' Q/sqrt(2)', alpha=0.3)
        plot(xx**2, dyU/xx*fact/np.sqrt(2), label=label+' U/sqrt(2)', alpha=0.3)

    if fit:
        mymodel = lambda x, a, b, c, d, e: (a + b*x + c*np.exp(-d*(x-e)))#/(a+b+c*np.exp(-d*(1-e)))
        ok = isfinite(myY)
        myfit = curve_fit(mymodel, xx[ok]**2, myY[ok], p0=[0.5,0.16, 0.3,2,1],maxfev=100000, ftol=1e-7)
        plot(xx**2, mymodel(xx**2, *myfit[0]),  label=label+' Fit', color=p[0].get_color())
        invcov_samples = np.linspace(1, 15, 1000)
        eff_v = mymodel(invcov_samples, *myfit[0])**2
        effective_variance_invcov = np.array([invcov_samples, eff_v])


    xlabel('1./cov normed')
    if norm:
        add_yl = ' (Normalized to 1 at 1)'
    else:
        add_yl=''
    ylabel('RMS Ratio w.r.t linear scaling'+add_yl)

    if fit:
        return xx, myY, effective_variance_invcov
    else:
        return xx, myY, None


def get_angular_profile(maps, thmax=25, nbins=20, label='', center=np.array([316.44761929,-58.75808063]), allstokes=False, fontsize=None):
    vec0 = hp.ang2vec(center[0], center[1], lonlat=True)
    vecpix = hp.pix2vec(256, np.arange(12*256**2))
    angs = np.degrees(np.arccos(np.dot(vec0,vecpix)))
    rng = np.array([0, thmax])
    xx, yyI, dx, dyI, _ = ft.profile(angs, maps[:,0], nbins=nbins, plot=False, rng=rng)
    xx, yyQ, dx, dyQ, _ = ft.profile(angs, maps[:,1], nbins=nbins, plot=False, rng=rng)
    xx, yyU, dx, dyU, _ = ft.profile(angs, maps[:,2], nbins=nbins, plot=False, rng=rng)
    avg = np.sqrt((dyI**2+dyQ**2/2+dyU**2/2)/3)
    plot(xx, avg, 'o', label=label)
    if allstokes:
        plot(xx, dyI, label=label+' I', alpha=0.3)
        plot(xx, dyQ/np.sqrt(2), label=label+' Q/sqrt(2)', alpha=0.3)
        plot(xx, dyU/np.sqrt(2), label=label+' U/sqrt(2)', alpha=0.3)
    xlabel('Angle [deg.]')
    ylabel('RMS')
    legend(fontsize=fontsize)
    return xx, avg






