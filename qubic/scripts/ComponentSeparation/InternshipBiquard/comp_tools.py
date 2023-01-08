# -*- coding: utf-8 -*-
"""A set of tools for component separation."""

import os
import pickle
import numpy as np
import numpy.ma as ma
import healpy as hp
import numpy.random
import qubic
from qubic import QubicSkySim as Qss
from astropy.io import fits


# credits to J. Errard for these two functions
def intersect_mask(maps):
    if hp.pixelfunc.is_ma(maps):
        mask = maps.mask
    else:
        mask = maps == hp.UNSEEN

    # Mask entire pixel if any of the frequencies in the pixel is masked
    return np.any(mask, axis=tuple(range(maps.ndim - 1)))


def format_alms(alms, lmin=0, nulling_option=True):
    lmax = hp.Alm.getlmax(alms.shape[-1])
    alms = np.asarray(alms, order='C')
    alms = alms.view(np.float64)
    em = hp.Alm.getlm(lmax)[1]
    em = np.stack((em, em), axis=-1).reshape(-1)
    mask_em = [m != 0 for m in em]
    alms[..., mask_em] *= np.sqrt(2)
    if nulling_option:
        alms[..., np.arange(1, lmax + 1, 2)] = hp.UNSEEN  # mask imaginary m = 0
        mask_alms = intersect_mask(alms)
        alms[..., mask_alms] = 0  # thus no contribution to the spectral likelihood
    alms = np.swapaxes(alms, 0, -1)
    if lmin != 0:
        ell = hp.Alm.getlm(lmax)[0]
        ell = np.stack((ell, ell), axis=-1).reshape(-1)
        mask_lmin = [ll < lmin for ll in ell]
        if nulling_option:
            alms[mask_lmin, ...] = hp.UNSEEN
    return alms


def add_errors_alm(alm_in, relative_error=1e-3, inplace=True):
    errors = np.random.random(len(alm_in))
    if inplace:
        alm_in += errors * relative_error
        return alm_in
    else:
        return alm_in * (1 + errors * relative_error)


def append_to_npz(folder_path, file_name, dico) -> None:
    """
    Save new data to a .npz archive, concatenate new values with already existing ones.

    :param folder_path: the path to the npz file
    :param file_name: the name of the .npz (assumed to be in output/...)
    :param dico: the dictionary containing the new data to save
    """
    try:
        new_dico = {}
        with np.load(folder_path + file_name) as old_npz:
            for k, v in dico.items():
                if k in old_npz.files:
                    new_dico[k] = np.concatenate((old_npz[k], v), axis=0)
                else:
                    new_dico[k] = v
        np.savez(folder_path + file_name, **new_dico)
    except FileNotFoundError:
        np.savez(folder_path + file_name, **dico)


def find_mantissa_exponent(x, e):
    n = 0
    if x < 1:
        while x * e ** n < 1:
            n += 1
    elif x > e:
        while x * e ** n > e:
            n -= 1
    return x * e ** n, n


def generate_cmb_dust_maps(dico_fast_simulator, coverage, n_years, noise_profile, nunu, sc,
                           seed=None, save_maps=False, return_maps=True, dust_only=False, fwhm_gen=None,
                           iib=True, noise_covcut=None):
    """
    Save CMB+Dust maps to FITS image format for later use, and/or return them immediately.

    :param dico_fast_simulator: dictionary for FastSimulator at the desired frequency (150 or 220)
    :param coverage: the sky coverage
    :param int n_years: number of integration years
    :param bool noise_profile: include noise profile (inhomogeneity)
    :param bool nunu: include noise frequency correlations
    :param bool sc: include noise spatial correlations
    :param seed: seed for the map generation (if None, a random seed is taken)
    :param bool|None save_maps: save maps in the FITS format (warning: check code first!)
    :param bool|None return_maps: whether the function has to return the generated maps
    :param bool|None dust_only: generate sky maps containing only dust (no cmb)
    :param float|None fwhm_gen: smooth maps to this fwhm during generation
    :param bool iib: integrate simulated maps into output bands
    :param float|None noise_covcut: coverage cut when generating noise maps

    :return: cmb+dust maps with noise, cmb+dust noiseless, noise only maps
    """
    if seed is None:
        seed = np.random.randint(1000000)
    if dust_only:
        sky_config = {'dust': 'd0'}  # see d0 in https://pysm3.readthedocs.io/en/latest/models.html
    else:
        sky_config = {'dust': 'd0', 'cmb': seed}
    qubic_sky = Qss.Qubic_sky(sky_config, dico_fast_simulator)

    if noise_covcut is None and coverage is not None:
        x, n = find_mantissa_exponent(np.min(coverage[coverage > 0]), 10)
        noise_covcut = np.floor(x * 10) / 10 ** (n + 1)
    if coverage is None:  # maps are full-sky
        coverage = np.ones(hp.nside2npix(dico_fast_simulator['nside']))
        noise_covcut = 0.1  # arbitrary but None would raise an error in the FastSimulator

    cmb_dust, cmb_dust_noiseless, cmb_dust_noise_only, coverage_eff = \
        qubic_sky.get_partial_sky_maps_withnoise(coverage=coverage,
                                                 Nyears=n_years,
                                                 noise_profile=noise_profile,  # noise inhomogeneity
                                                 nunu_correlation=nunu,  # noise frequency correlations
                                                 spatial_noise=sc,  # noise spatial correlations
                                                 verbose=False,
                                                 seed=None,
                                                 FWHMdeg=fwhm_gen,
                                                 integrate_into_band=iib,
                                                 noise_covcut=noise_covcut,
                                                 )
    if save_maps:
        save_dir = "/media/simon/CLE32/qubic/maps/"
        if noise_profile:
            save_dir += "with_noise_profile/"
        else:
            save_dir += "no_noise_profile/"
        save_dir += "{:d}ghz/".format(int(dico_fast_simulator['filter_nu'] / 1e9))
        save_dir += "{}/".format(seed)
        try:
            os.mkdir(save_dir)
        except FileExistsError:
            pass

        common_fmt = "{}bands_{}y"  # .format(band, nsub, nyears, seed)
        common = common_fmt.format(dico_fast_simulator['nf_recon'], n_years)

        hdu_cmb_dust = fits.PrimaryHDU(cmb_dust)
        hdu_cmb_dust.writeto(save_dir + common + "_cmbdust.fits", overwrite=True)

        hdu_cmb_dust_noiseless = fits.PrimaryHDU(cmb_dust_noiseless)
        hdu_cmb_dust_noiseless.writeto(save_dir + common + "_cmbdust_noiseless.fits", overwrite=True)

        hdu_cmb_dust_noise_only = fits.PrimaryHDU(cmb_dust_noise_only)
        hdu_cmb_dust_noise_only.writeto(save_dir + common + "_noise.fits", overwrite=True)

    if return_maps:
        return cmb_dust, cmb_dust_noiseless, cmb_dust_noise_only, coverage_eff
    else:
        return


def get_coverage_from_file(folder_path, file_name=None, dtheta=40, pointing=6000):
    """
    Get coverage map from saved file.

    :param folder_path: the path to the file (optional)
    :param file_name: the name of the file (optional)
    :param dtheta: angle of coverage (default: 40°)
    :param pointing: number of sky points (default: 6000)

    :return: the array containing the coverage map
    """

    if file_name is None:
        file_name = 'coverage_dtheta_{}_pointing_{}'.format(dtheta, pointing)
    t = pickle.load(open(folder_path + file_name + '.pkl', 'rb'))
    return t['coverage'], file_name


def get_depths(noise_maps, pix_size, mask=None, pixel_weights=None):
    """Compute depth_i and depth_p (sensitivities) from noise maps.

    :param noise_maps: the noise maps
    :param pix_size: the pixel size in arcmin
    :param mask: the mask to apply
    :param pixel_weights: weighting of pixels (coverage)

    :return: depth_i and depth_p of the map
    """
    # apply pixel weights
    weighted_maps = np.empty_like(noise_maps)
    weighted_maps[...] = noise_maps[...] * pixel_weights

    # apply mask
    noise_ma = ma.array(weighted_maps, mask=mask)

    # noise estimation (in I component) using the noise maps
    depth_i = ma.getdata(ma.std(noise_ma[:, 0, :], axis=1))
    depth_i *= pix_size

    # noise estimation (in Q & U components)
    depth_p = ma.getdata(ma.std(noise_ma[:, 1:, :], axis=(1, 2)))
    depth_p *= pix_size

    return depth_i, depth_p


def get_kernel_fwhms_for_smoothing(fwhms, target=None):
    """
    Get fwhm values of kernels that have to be used in order to smooth maps to the same resolution.

    :param fwhms: the list of original fwhms
    :param float target: the target resolution. If None, it is considered to be the biggest fwhm of the maps.
    :return: the list of fwhms for the kernels, and the (updated) target
    """
    if target is None:
        target = np.max(fwhms)

    diff = target ** 2 - np.square(fwhms)
    diff[diff < 0] = 0

    return np.sqrt(diff), target


def get_sub_freqs_and_resolutions(dico_fast_simulator):
    """
    Give the frequency sub-bands and corresponding angular resolutions around f = 150 or 220 GHz.

    :param dico_fast_simulator: instrument dictionary containing frequency band and nbr of sub-bands wanted
    :return: Tuple (freqs, fwhms) containing the list of the central frequencies
        and the list of resolutions (in degrees).
    """
    band = dico_fast_simulator['filter_nu'] / 1e9
    n = int(dico_fast_simulator['nf_recon'])
    filter_relative_bandwidth = dico_fast_simulator['filter_relative_bandwidth']
    _, _, nus_in, _, _, _ = qubic.compute_freq(band,
                                               Nfreq=n,
                                               relative_bandwidth=filter_relative_bandwidth)
    # nus_in are in GHz so we use inverse scaling of resolution with frequency
    # we know the fwhm at 150 GHz so the factor is 150 / (target frequency)
    return nus_in, dico_fast_simulator['synthbeam_peak150_fwhm'] * 150 / nus_in


def print_list(list_, fmt=''):
    list_fmt = "[" + ", ".join(["{:" + fmt + "}"] * len(list_)) + "]"
    print(list_fmt.format(*list_))


def same_resolution(maps_in, map_fwhms_deg, fwhm_target=None, verbose=False):
    """
    Return copies of input maps smoothed to a common resolution.

    :param maps_in: array containing the original maps (size of 1st dimension = # of sub-bands)
    :param list[float] map_fwhms_deg: list of fwhms of the maps (in degrees)
    :param float fwhm_target: the common resolution to which the function smooths the maps (if specified).
        If not specified, the target resolution is the lowest of all input maps.
    :param bool verbose: make the function verbose
    :return: (maps_out, fwhm_out) with maps_out an array containing the maps smoothed down to a common resolution
        and fwhm_out the common (new) fwhm
    """
    # define common output resolution
    kernel_fwhms, fwhm_out = get_kernel_fwhms_for_smoothing(map_fwhms_deg, fwhm_target)

    # create array to contain output maps
    maps_out = np.zeros_like(maps_in)

    # loop over input maps (ie. over sub-bands)
    for i, kernel_fwhm in enumerate(kernel_fwhms):
        if kernel_fwhm > 1e-6:
            if verbose:
                print('  -> smooth sub-band {:d} (fwhm={:.3f}°) with kernel of {:.3f}°'.format(i, map_fwhms_deg[i],
                                                                                               kernel_fwhm))
            maps_out[i, :, :] = hp.sphtfunc.smoothing(maps_in[i, :, :],
                                                      fwhm=np.radians(kernel_fwhm),
                                                      verbose=False)
        else:
            if verbose:
                print('  -> no convolution needed, sub-map {:d} already at required resolution.'.format(i))
            maps_out[i, :, :] = maps_in[i, :, :]

    return maps_out, fwhm_out
