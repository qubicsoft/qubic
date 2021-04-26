"""Functions using the FgBuster library for QUBIC component separation."""

import fgbuster as fgb
import healpy as hp
import numpy as np


def same_resol(maps_in, map_fwhms_deg, fwhm_target=None, verbose=False):
    """
    Put all maps at the same resolution.

    :param maps_in: array containing the maps to be smoothed (size of 1st dimension = # of sub-bands)
    :param list[float] map_fwhms_deg: list of fwhms of the maps (in degrees)
    :param float fwhm_target: the common resolution to which the function smooths the maps (if specified).
        If not specified, the target resolution is the lowest of all input maps.
    :param bool verbose: make the function verbose

    :return: (maps_out, fwhm_out, delta_fwhm) with
        - maps_out: array containing the maps smoothed down to a common resolution
        - fwhm_out: common (new) fwhm
    """
    # define common output resolution
    sh = np.shape(maps_in)
    nb_bands = sh[0]
    if fwhm_target is None:
        fwhm_out = np.max(map_fwhms_deg)
        if verbose:
            print("input maps will be smoothed down to minimal resolution (fwhm={:.6f}°)".format(fwhm_out))
    else:
        fwhm_out = fwhm_target
        if verbose:
            print("input maps will be smoothed down to specified resolution (fwhm={:.6f}°)".format(fwhm_out))

    # create array to contain output maps
    maps_out = np.zeros_like(maps_in)

    # loop over input maps (ie. over sub-bands)
    for i in range(nb_bands):
        fwhm_in = map_fwhms_deg[i]
        kernel_fwhm = np.sqrt(fwhm_out ** 2 - fwhm_in ** 2)
        if verbose:
            print('Sub-band {:d}: going from fwhm={:6f}° to fwhm={:6f}°'.format(i, fwhm_in, fwhm_out))

        if kernel_fwhm > 1e-6:
            if verbose:
                print('    -> convolution with {:6f}° fwhm kernel.'.format(kernel_fwhm))
            maps_out[i, :, :] = hp.sphtfunc.smoothing(maps_in[i, :, :],
                                                      fwhm=np.radians(kernel_fwhm),
                                                      verbose=False)
        else:
            if verbose:
                print('    -> no convolution needed, map already at required resolution.')
            maps_out[i, :, :] = maps_in[i, :, :]

    return maps_out, fwhm_out


class CompSep(object):
    """
    Class that brings together different methods of component separations. Currently, there is only 'fg_buster'
    definition which work with 2 components (CMB and Dust).
    """

    def __init__(self, d):

        self.nside = d['nside']
        self.npix = 12 * self.nside ** 2
        self.nb_bands = int(d['nf_recon'])
        self.l_min = 20
        self.l_max = 2 * self.nside - 1
        self.delta_ell = 16

    def fg_buster(self, maps_in=None, components=None, map_freqs=None, map_fwhms_deg=None, target=None, ok_pix=None,
                  stokes='IQU', verbose=False):
        """
        Perform FgBuster algorithm.

        :param maps_in: array of maps to be separated -> shape=(nb_bands, nb_stokes, npix)
        :param components: list storing the components of the mixing matrix (ie. that we want to separate).
        Dust must have nu0 in input and we can fix the temperature.
        :param map_freqs: list storing the frequencies of the maps
        :param map_fwhms_deg: list type of full width at half maximum (in degrees). It can be different values.
        :param target: if target is not None, "same_resol" definition is applied and put all the maps at the same
        resolution. If target is None, make sure that all the resolution are the same.
        :param ok_pix: boolean array type which exclude the edges of the map.
        :param stokes: Stokes parameters concerned by the separation
        :param bool verbose: print progress

        :return: Dictionary which contains the amplitude of each components, the estimated parameter beta_d and dust
        temperature.
        """

        qubic_instrument = fgb.get_instrument('Qubic' + str(self.nb_bands) + 'bands')
        # qubic_instrument = fgb.get_instrument('QUBIC')

        # specify correct frequency and FWHM
        qubic_instrument.frequency = map_freqs
        qubic_instrument.fwhm = map_fwhms_deg

        # Change resolution of each map if it's necessary
        maps_in, _ = same_resol(maps_in, map_fwhms_deg, fwhm_target=target, verbose=verbose)

        # Apply FG Buster
        if stokes == 'IQU':
            res = fgb.basic_comp_sep(components, qubic_instrument, maps_in[:, :, ok_pix])

        elif stokes == 'QU':
            res = fgb.basic_comp_sep(components, qubic_instrument, maps_in[:, 1:, ok_pix])

        elif stokes == 'I':
            res = fgb.basic_comp_sep(components, qubic_instrument, maps_in[:, 0, ok_pix])

        else:
            raise TypeError("incorrect specification of Stokes parameters (must be either 'IQU', 'QU' or 'I')")

        return res

    def internal_linear_combination(self, maps_in=None, components=None, map_freqs=None, map_fwhms_deg=None,
                                    target=None):
        """
        Perform Internal Linear Combination (ILC) algorithm.

        :param maps_in: maps from which to estimate CMB signal
        :param components: list storing the components of the mixing matrix
        :param map_freqs: list storing the frequencies of the maps
        :param map_fwhms_deg: list storing the fwhms of the maps (in degrees). It may contain different values.
        :param target:

        :return: Dictionary for each Stokes parameter (I, Q, U) in a list.
            To have the amplitude, we can write r[ind_stk].s[0].
        """

        qubic_instrument = fgb.get_instrument('Qubic' + str(self.nb_bands) + 'bands')

        # specify correct frequency and FWHM
        qubic_instrument.frequency = map_freqs
        qubic_instrument.fwhm = map_fwhms_deg

        r = []

        # change resolutions of the maps if necessary
        maps_in, _ = same_resol(maps_in, map_fwhms_deg, fwhm_target=target, verbose=True)

        # Apply ILC for each stokes parameter
        for i in range(3):
            r.append(fgb.ilc(components, qubic_instrument, maps_in[:, i, :]))

        return r

    def ilc_2_tab(self, x, seen_pix):

        tab_cmb = np.zeros((self.nb_bands, 3, self.npix))

        for i in range(3):
            tab_cmb[0, i, seen_pix] = x[0].s[0]

        return tab_cmb
