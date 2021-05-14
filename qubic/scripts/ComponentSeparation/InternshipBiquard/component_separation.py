"""
Functions using the FgBuster library for QUBIC component separation.

For FgBuster dust spectral index estimation, command line arguments should be:
- local (=0) or CC (=1) execution
- frequency band (150 or 220 GHz)
- nb of simulations to perform
- nb of sub-bands
- nb of years of observation
- QubicSkySim parameters spatial correlations, frequency correlations (11 = both True)
"""

# General imports
import os
import pickle
import sys
import numpy as np
import numpy.ma as ma
from astropy.io import fits
import healpy as hp

# Qubic modules
import qubic
from qubic import QubicSkySim as Qss

# Imports needed for component separation
import fgbuster as fgb

# Define data / output directories
QUBIC_DATADIR = os.environ['QUBIC_DATADIR']  # TODO: determine if useless since QSS upgrade by Louise

# for LOCAL EXECUTION
OUTDIR_LOCAL = "/home/simon/PycharmProjects/qubic_comp_sep/output/"
DATADIR_LOCAL = "/home/simon/PycharmProjects/qubic_comp_sep/data/"

# for execution on CC-IN2P3
OUTDIR_CC = "/sps/qubic/Users/sbiquard/qubic/qubic_comp_sep/output/"
DATADIR_CC = "/sps/qubic/Users/sbiquard/qubic/qubic_comp_sep/data/"

if len(sys.argv) > 1:
    LOC = int(sys.argv[1])  # 0 for local (falaise), 1 for CC
else:
    LOC = 0

if LOC == 0:
    OUTDIR = OUTDIR_LOCAL
    DATADIR = DATADIR_LOCAL
elif LOC == 1:
    OUTDIR = OUTDIR_CC
    DATADIR = DATADIR_CC
else:
    raise ValueError("Specify where the execution takes place (0 = local, 1 = CC)")


def append_to_npz(file_name, dico) -> None:
    """
    Save new data to a .npz archive, concatenate new values with already existing ones.

    :param file_name: the name of the .npz (assumed to be in output/...)
    :param dico: the dictionary containing the new data to save
    """
    try:
        new_dico = {}
        with np.load(OUTDIR + file_name) as old_npz:
            for k, v in dico.items():
                if k in old_npz.files:
                    new_dico[k] = np.concatenate((old_npz[k], v), axis=0)
                else:
                    new_dico[k] = v
        np.savez(OUTDIR + file_name, **new_dico)
    except FileNotFoundError:
        np.savez(OUTDIR + file_name, **dico)


def generate_cmb_dust_maps(dico, coverage, n_years, noise_profile, nunu, sc,
                           seed=None, save_maps=False, return_maps=True):
    """
    Save CMB+Dust maps to FITS image format for later use, and/or return them immediately.

    :param dico: dictionary for FastSimulator at the desired frequency (150 or 220)
    :param coverage: the sky coverage
    :param int n_years: number of integration years
    :param bool noise_profile: include noise profile (inhomogeneity)
    :param bool nunu: include noise frequency correlations
    :param bool sc: include noise spatial correlations
    :param seed: seed for the map generation (if None, a random seed is taken)
    :param bool save_maps: save maps in the FITS format (warning: check code first! default: False)
    :param bool return_maps: whether the function has to return the generated maps (default: True)

    :return: cmb+dust maps with noise, cmb+dust noiseless, and noise only maps
    """
    if seed is None:
        seed = np.random.randint(1000000)
    sky_config = {'dust': 'd0', 'cmb': seed}  # d0 is described in https://pysm3.readthedocs.io/en/latest/models.html
    qubic_sky = Qss.Qubic_sky(sky_config, dico)
    cmb_dust, cmb_dust_noiseless, cmb_dust_noise_only, _ = \
        qubic_sky.get_partial_sky_maps_withnoise(coverage=coverage,
                                                 Nyears=n_years,
                                                 noise_profile=noise_profile,  # noise inhomogeneity
                                                 nunu_correlation=nunu,  # noise frequency correlations
                                                 spatial_noise=sc,  # noise spatial correlations
                                                 verbose=False,
                                                 seed=seed,
                                                 )
    if save_maps:
        save_dir = "/media/simon/CLE32/qubic/maps/"
        if noise_profile:
            save_dir += "with_noise_profile/"
        else:
            save_dir += "no_noise_profile/"
        save_dir += "{:d}ghz/".format(int(dico['filter_nu'] / 1e9))
        save_dir += "{}/".format(seed)
        try:
            os.mkdir(save_dir)
        except FileExistsError:
            pass

        common_fmt = "{}bands_{}y"  # .format(band, nsub, nyears, seed)
        common = common_fmt.format(dico['nf_recon'], n_years)
        # TODO: add more possibilities

        hdu_cmb_dust = fits.PrimaryHDU(cmb_dust)
        hdu_cmb_dust.writeto(save_dir + common + "_cmbdust.fits", overwrite=True)

        hdu_cmb_dust_noiseless = fits.PrimaryHDU(cmb_dust_noiseless)
        hdu_cmb_dust_noiseless.writeto(save_dir + common + "_cmbdust_noiseless.fits", overwrite=True)

        hdu_cmb_dust_noise_only = fits.PrimaryHDU(cmb_dust_noise_only)
        hdu_cmb_dust_noise_only.writeto(save_dir + common + "_noise.fits", overwrite=True)

    if return_maps:
        return cmb_dust, cmb_dust_noiseless, cmb_dust_noise_only
    else:
        return


def get_coverage_from_file(file_name=None):
    """
    Get coverage map from saved file.

    :param file_name: the name of the file

    :return: the array containing the coverage map
    """
    if file_name is None:
        t = pickle.load(open(DATADIR + 'coverage_dtheta_40_pointing_6000.pkl', 'rb'))
    else:
        t = pickle.load(open(DATADIR + file_name, 'rb'))
    return t['coverage']


def get_sub_freqs_and_resolutions(dico):
    """
    Give the frequency sub-bands and corresponding angular resolutions around f = 150 or 220 GHz.

    :param dico: instrument dictionary containing frequency band and nbr of sub-bands wanted

    :return: Tuple (freqs, fwhms) containing the list of the central frequencies
        and the list of resolutions (in degrees).
    """
    band = dico['filter_nu'] / 1e9
    n = int(dico['nf_recon'])
    filter_relative_bandwidth = dico['filter_relative_bandwidth']
    _, _, nus_in, _, _, _ = qubic.compute_freq(band,
                                               Nfreq=n,
                                               relative_bandwidth=filter_relative_bandwidth)
    # nus_in are in GHz so we use inverse scaling of resolution with frequency
    # we know the fwhm at 150 GHz so the factor is 150 / (target frequency)
    return nus_in, dico['synthbeam_peak150_fwhm'] * 150 / nus_in


def read_arguments():
    """Read parameters / command-line arguments for local (personal computer) execution.

    :return: frequency band (GHz), nb of simulations, nb of sub-bands, nb of years and sky_sim parameters.
    """
    nargs = len(sys.argv) - 1
    if nargs:
        frequency_band = int(sys.argv[2])
        nb_simu = int(sys.argv[3])
        nb_bands = int(sys.argv[4])
        nb_years = int(sys.argv[5])
        sky_sim_param = sys.argv[6]
    else:
        frequency_band = int(input("Main frequency band (150 or 220 GHz) = "))
        nb_simu = int(input("Number of simulations to run = "))
        nb_bands = int(input("Number of sub-bands for {} GHz = ".format(frequency_band)))
        nb_years = int(input("Number of observation years = "))
        sky_sim_param = input("spatial correlations, frequency correlations (11 = both True) = ")

    print()
    spatial_correlations = bool(int(sky_sim_param[0]))
    nunu_correlations = bool(int(sky_sim_param[1]))

    return frequency_band, nb_simu, nb_bands, nb_years, spatial_correlations, nunu_correlations


def run_beta_dust_estimation_combine_bands(n_sim: int, n_sub: int, n_years: int, noise_profile=False, nunu=False,
                                           sc=False, verbose=True, combine_bands=True, single_band_freq=None,
                                           stokes='IQU', estimate_dust_temperature=False):
    """
    Run sky simulations (using `QubicSkySim`) and determine spectral parameters with FgBuster, using `n_sub` * `n_sub`
    at 150 & 220 GHz simultaneously.

    Possible keyword arguments are: *verbose* option (boolean, default: True), *stokes* (can be 'IQU', 'I' or 'QU',
    default: 'IQU') which determines the set of Stokes parameters used for the separation, *estimate_dust_temperature*
    (boolean, default: False), *dust_temp* (float, default: 20.) which is the default dust temperature for the dust
    component model, *combine_bands* (boolean, default: True) which allows to choose whether both 150 and 220 GHz
    bands are combined for the separation, and *single_band_frequency* (int, default: 150) which is the chosen
    frequency band in case of non-combined separation.

    :param int n_sub: number of sub-bands
    :param int n_sim: number of simulations
    :param int n_years: number of years of observation for simulated maps
    :param noise_profile: QubicSkySim "noise_profile" (ie. noise inhomogeneity)
    :param bool nunu: QubicSkySim "nunu_correlation" (ie. noise frequency correlations)
    :param bool sc: QubicSkySim "spatial_noise" (ie. spatial noise correlations)
    :param bool verbose: print progress information (default: True)
    :param bool combine_bands: combine 150 and 220 GHz bands (each with *n_sub* sub-bands)
    :param int|None single_band_freq: frequency band to use if not combined (must be 150 or 220)
    :param str stokes: Stokes parameters for which to perform the separation (must be 'IQU', 'I' or 'QU')
    :param bool estimate_dust_temperature: estimate dust temperature in addition to spectral index (default: False)
    """

    # define useful variables, determine what bands to use, get FastSimulator dictionaries, etc.
    dico150 = qubic_dicts[150]
    dico220 = qubic_dicts[220]
    dico150['nf_recon'] = dico220['nf_recon'] = n_sub  # nbr of reconstructed bands (output)
    dico150['nf_sub'] = dico220['nf_sub'] = 4 * n_sub  # nbr of simulated bands (input)

    # create dictionary for the component separation
    dico = {'nside': 256,
            'npix': hp.nside2npix(256),
            'nyears': n_years,
            'verbose': verbose}
    if estimate_dust_temperature:
        dico['components'] = [fgb.CMB(), fgb.Dust(150.)]
    else:
        dico['components'] = [fgb.CMB(), fgb.Dust(150., temp=20.)]

    # get sub-frequencies and fwhms
    if combine_bands:
        freqs150, fwhms150 = get_sub_freqs_and_resolutions(dico150)
        freqs220, fwhms220 = get_sub_freqs_and_resolutions(dico220)
        dico.update({'nbands': 2 * n_sub,
                     'freqs': np.concatenate((freqs150, freqs220), axis=0),
                     'fwhms': np.concatenate((fwhms150, fwhms220), axis=0)})
    else:
        single_dico = qubic_dicts[single_band_freq]
        freqs, fwhms = get_sub_freqs_and_resolutions(single_dico)
        dico.update(single_dico)
        dico.update({'nbands': n_sub,
                     'freqs': freqs,
                     'fwhms': fwhms})

    # get coverage map, define pixels to use & arguments for map generation, etc.
    coverage = get_coverage_from_file()
    # okpix_inside = (coverage > (0.5 * np.max(coverage)))
    t_mask = np.empty((dico['nbands'], 3, dico['npix']), dtype=bool)
    t_mask[...] = (coverage < (0.5 * np.max(coverage)))
    dico['mask'] = t_mask

    # arguments for map generation
    map_args = coverage, n_years, noise_profile, nunu, sc

    # declare variable to store the results
    beta_values_cmbdust = []
    beta2 = []
    temp_values_cmbdust = []

    # simulate n_sim realisations of sky maps using QubicSkySim
    for i_sim in range(n_sim):

        if verbose:
            print("entering iteration number {:d} of n_sim={:d}".format(i_sim + 1, n_sim))
            print("generate sky maps...")

        # generate sky map with random seed
        # if bands are combined, need to generate the maps with a common seed
        if combine_bands:
            common_seed = np.random.randint(1000000)
            cmbdust150, _, noise150 = generate_cmb_dust_maps(dico150, *map_args, seed=common_seed)
            cmbdust220, _, noise220 = generate_cmb_dust_maps(dico220, *map_args, seed=common_seed)
            cmbdust = np.concatenate((cmbdust150, cmbdust220), axis=0)
            noise = np.concatenate((noise150, noise220), axis=0)
        else:
            cmbdust, _, noise = generate_cmb_dust_maps(dico, *map_args)

        # modify shape of maps for FgBuster, add them to dictionary
        cmbdust = np.transpose(cmbdust, (0, 2, 1))
        noise = np.transpose(noise, (0, 2, 1))
        dico['maps_in'] = cmbdust
        dico['noise_in'] = noise
        if verbose:
            print("sky maps generated\n")

        # create BasicCompSep instance
        comp_separation = BasicCompSep(dico)

        # perform component separation using FgBuster
        fg_res = comp_separation.fg_buster(stokes=stokes)
        fg_res_2 = comp_separation.fg_buster(stokes=stokes, target=fwhm_150)
        beta_cmbdust = fg_res.x[0]
        beta_values_cmbdust.append(beta_cmbdust)
        beta2.append(fg_res_2.x[0])
        if estimate_dust_temperature:
            temp_cmbdust = fg_res.x[1]
            temp_values_cmbdust.append(temp_cmbdust)
        if verbose:
            print("fgbuster separation terminated (message: {})\n\n".format(fg_res.message))

        del cmbdust
        del noise
        del comp_separation

    if estimate_dust_temperature:
        res = np.array(beta_values_cmbdust), np.array(temp_values_cmbdust)
    else:
        res = np.array(beta_values_cmbdust), np.array(beta2)

    return res


def same_resol(maps_in, map_fwhms_deg, fwhm_target=None, verbose=False):
    """
    Put all maps at the same resolution.

    :param maps_in: array containing the maps to be smoothed (size of 1st dimension = # of sub-bands)
    :param list[float] map_fwhms_deg: list of fwhms of the maps (in degrees)
    :param float fwhm_target: the common resolution to which the function smooths the maps (if specified).
        If not specified, the target resolution is the lowest of all input maps.
    :param bool verbose: make the function verbose

    :return: (maps_out, fwhm_out) with maps_out an array containing the maps smoothed down to a common resolution
    and fwhm_out the common (new) fwhm
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


class BasicCompSep(object):
    """
    Class that provides tools for basic component separation with FgBuster.
    Documentation:
    https://fgbuster.github.io/fgbuster/api/fgbuster.separation_recipes.html#fgbuster.separation_recipes.basic_comp_sep
    """

    def __init__(self, d):
        """
        Create instance of BasicCompSep.

        :param d: dictionary containing all required parameters
        """

        self.verbose = d['verbose']

        self.nside = d['nside']
        self.npix = d['npix']
        self.nbands = d['nbands']  # number of sub-bands
        self.nyears = d['nyears']  # number of years of integration
        instrument_name = 'Qubic' + str(self.nbands) + 'bands'
        self.instrument = fgb.get_instrument(instrument_name)  # get instrument from experiments.yaml in cmbdb package
        self.maps = d['maps_in']  # maps to be separated
        self.noise = d['noise_in']  # noise maps (can be None)
        self.freqs = d['freqs']  # frequencies of the maps
        self.instrument.frequency = self.freqs
        self.fwhms = d['fwhms']  # fwhms (in degrees) of the maps
        self.mask = d['mask']  # mask that indicates pixels to consider for separation (True=bad pixel)
        self.components = d['components']  # sky components (CMB, Dust, ...)

        # solver options
        self.solver_method = 'BFGS'
        self.solver_tol = 1
        self.solver_options = {'disp': False}  # disp=True raises KeyError in FgBuster modules...

        if self.verbose:
            print("Create BasicCompSep instance...")
            print("  ->  nside = {}".format(self.nside))
            print("  ->   npix = {}".format(self.npix))
            print("  -> nbands = {}".format(self.nbands))
            print("  -> nyears = {}".format(self.nyears))
            print("  -> instrument = {}".format(instrument_name))
            print("  -> shape of maps = {}".format(np.shape(self.maps)))
            list_fmt = "[" + ", ".join(["{:.1f}"] * self.nbands) + "]"
            print("  -> map frequencies (GHz) = " + list_fmt.format(*self.freqs))
            list_fmt = "[" + ", ".join(["{:.3f}"] * self.nbands) + "]"
            print("  ->   map fwhms (degrees) = " + list_fmt.format(*self.fwhms))
            print("  -> mask defines {} good pixels".format(np.sum(np.logical_not(self.mask))))
            print("  -> {} components specified".format(len(self.components)))
            print("  ->    solver method = {}".format(self.solver_method))
            print("  -> solver tolerance = {}".format(self.solver_tol))
            print("  ->   solver options = {}".format(self.solver_options))

        # OLD CODE
        # self.l_min = 20
        # self.l_max = 2 * self.nside - 1
        # self.delta_ell = 16

    def fg_buster(self, stokes='IQU', target=None):
        """
        Perform FgBuster algorithm.

        :param str stokes: Stokes parameters for which to perform the separation.
        :param float target: target resolution for the maps. If None, maps are smoothed to
        the lowest resolution of them all.

        :return: Dictionary which contains the amplitude of each components, the estimated parameter beta_d and dust
        temperature.
        """

        # change resolution of each map if necessary
        if self.verbose:
            print("smooth input maps to common resolution...")
        maps, fwhm = same_resol(self.maps, self.fwhms, fwhm_target=target, verbose=False)
        if self.verbose:
            print("maps smoothed to {:.3f} degrees".format(fwhm))

        # build MaskedArray maps
        maps_m = ma.array(maps, mask=self.mask)

        # compute noise levels
        if self.noise is None:  # if noise maps are not given
            if self.verbose:
                print("estimate roughly noise level (noise maps not provided)...")
            depth_i = 2. * np.sqrt(2 * self.nbands / (3 * self.nyears)) / np.ones(self.nbands)
            depth_p = depth_i * np.sqrt(2)
        else:
            if self.verbose:
                print("compute noise level with noise maps provided...")
            noise, fwhm = same_resol(self.noise, self.fwhms, fwhm_target=target, verbose=False)
            noise_m = ma.array(noise, mask=self.mask)
            # noise estimation (in I component) using the noise maps
            depth_i = ma.getdata(ma.std(noise_m[:, 0, :], axis=1))
            depth_i *= np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60
            # noise estimation (in Q & U components)
            depth_p = ma.getdata(ma.std(noise_m[:, 1:, :], axis=(1, 2)))
            depth_p *= np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60

        # update parameters of instrument with sensibilities
        self.instrument.depth_i = depth_i
        self.instrument.depth_p = depth_p
        if self.verbose:
            list_fmt = "[" + ", ".join(["{:.2f}"] * self.nbands) + "]"
            print("  -> depth_i = " + list_fmt.format(*depth_i))
            print("  -> depth_p = " + list_fmt.format(*depth_p))

        # trying to compute noise after smoothing analytically
        # noise_in_m = ma.array(noise_in, mask=t_mask)
        # depth_1 = ma.getdata(ma.std(noise_in_m[:, 0, :], axis=1))
        # depth_2 = depth_1.copy()
        # depth_3 = depth_1.copy()
        #
        # depth_1 *= fwhms / fwhm
        # depth_2 *= (fwhms / fwhm) ** 2
        # depth_3[1:] /= np.sqrt(np.pi * (fwhm ** 2 - np.square(fwhms[1:]))) * 2 * 2.35
        # depth_1 *= np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60
        # depth_2 *= np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60
        # depth_3 *= np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60

        # apply FG Buster
        if stokes == 'IQU':
            res = fgb.basic_comp_sep(self.components,
                                     self.instrument,
                                     maps_m,
                                     method=self.solver_method,
                                     tol=self.solver_tol,
                                     options=self.solver_options
                                     )

        elif stokes == 'QU':
            res = fgb.basic_comp_sep(self.components,
                                     self.instrument,
                                     maps_m[:, 1:, :],
                                     method=self.solver_method,
                                     tol=self.solver_tol,
                                     options=self.solver_options
                                     )

        elif stokes == 'I':
            res = fgb.basic_comp_sep(self.components,
                                     self.instrument,
                                     maps_m[:, 0, :],
                                     method=self.solver_method,
                                     tol=self.solver_tol,
                                     options=self.solver_options
                                     )

        else:
            raise TypeError("incorrect specification of Stokes parameters (must be either 'IQU', 'QU' or 'I')")

        return res


# TODO: 1) compute noise analytically rather than using smoothed maps --> complicated
# TODO: 2) find a way to perform separation in harmonic space
# TODO: 3) use weighted_comp_sep to account for noise profile (circular dependence)
# TODO: 4) check results with 220 GHz maps smoothed to 150 GHz resolution (to see if bias in combined separation is
#          due to smoothing
# TODO: 5) rewrite functions as methods of the BasicCompSep class


if __name__ == "__main__":
    # Qubic dictionaries for 150 GHz and 220 Ghz
    config_150, config_220 = 'FI-150', 'FI-220'
    d150_name = QUBIC_DATADIR + '/doc/FastSimulator/FastSimDemo_{}.dict'.format(config_150)
    d220_name = QUBIC_DATADIR + '/doc/FastSimulator/FastSimDemo_{}.dict'.format(config_220)
    d150, d220 = qubic.qubicdict.qubicDict(), qubic.qubicdict.qubicDict()
    d150.read_from_file(d150_name)
    d220.read_from_file(d220_name)
    qubic_dicts = {150: d150, 220: d220}

    fwhm_150 = 0.43

    # arguments = read_arguments()
    r = run_beta_dust_estimation_combine_bands(5, 3, 3, combine_bands=True, single_band_freq=None)

    print("  average =", np.mean(r))
    print("deviation =", np.std(r))

    sys.exit(0)

# OLD CODE
# def internal_linear_combination(self, maps_in=None, components=None, freqs=None, map_fwhms_deg=None,
#                                 target=None):
#     """
#     Perform Internal Linear Combination (ILC) algorithm.
#
#     :param maps_in: maps from which to estimate CMB signal
#     :param components: list storing the components of the mixing matrix
#     :param freqs: list storing the frequencies of the maps
#     :param map_fwhms_deg: list storing the fwhms of the maps (in degrees). It may contain different values.
#     :param target:
#
#     :return: Dictionary for each Stokes parameter (I, Q, U) in a list.
#         To have the amplitude, we can write r[ind_stk].s[0].
#     """
#
#     qubic_instrument = fgb.get_instrument('Qubic' + str(self.nbands) + 'bands')
#
#     # specify correct frequency and FWHM
#     qubic_instrument.frequency = freqs
#     qubic_instrument.fwhm = map_fwhms_deg
#
#     r = []
#
#     # change resolutions of the maps if necessary
#     maps_in, _ = same_resol(maps_in, map_fwhms_deg, fwhm_target=target, verbose=True)
#
#     # Apply ILC for each stokes parameter
#     for i in range(3):
#         r.append(fgb.ilc(components, qubic_instrument, maps_in[:, i, :]))
#
#     return r
#
# def ilc_2_tab(self, x, seen_pix):
#
#     tab_cmb = np.zeros((self.nbands, 3, self.npix))
#
#     for i in range(3):
#         tab_cmb[0, i, seen_pix] = x[0].s[0]
#
#     return tab_cmb
