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

import comp_tools as tools

# General imports
import os
import sys
import numpy as np
import numpy.ma as ma
import healpy as hp

# Qubic modules
import qubic

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

# Qubic dictionaries for 150 GHz and 220 Ghz
config_150, config_220 = 'FI-150', 'FI-220'
d150_name = QUBIC_DATADIR + '/doc/FastSimulator/FastSimDemo_{}.dict'.format(config_150)
d220_name = QUBIC_DATADIR + '/doc/FastSimulator/FastSimDemo_{}.dict'.format(config_220)
d150, d220 = qubic.qubicdict.qubicDict(), qubic.qubicdict.qubicDict()
d150.read_from_file(d150_name)
d220.read_from_file(d220_name)
qubic_dicts = {150: d150, 220: d220}

fwhm_150 = 0.43


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


def run_beta_dust_estimation_combine_bands(n_sim: int, n_sub: int, n_years: int, noise_profile=False, nu_corr=False,
                                           spatial_corr=False, verbose=True, combine_bands=True, band=None,
                                           stokes='IQU', estimate_dust_temperature=False, use_noise_maps=False):
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
    :param bool nu_corr: QubicSkySim "nunu_correlation" (ie. noise frequency correlations)
    :param bool spatial_corr: QubicSkySim "spatial_noise" (ie. spatial noise correlations)
    :param bool verbose: print progress information (default: True)
    :param bool combine_bands: combine 150 and 220 GHz bands (each with *n_sub* sub-bands)
    :param int|None band: frequency band to use if not combined (must be 150 or 220)
    :param str stokes: Stokes parameters for which to perform the separation (must be 'IQU', 'I' or 'QU')
    :param bool estimate_dust_temperature: estimate dust temperature in addition to spectral index (default: False)
    :param bool use_noise_maps: use noise maps to compute sensitivities
    """

    # indicate what components the maps contain
    if estimate_dust_temperature:
        components = [fgb.CMB(), fgb.Dust(150.)]
    else:
        components = [fgb.CMB(), fgb.Dust(150., temp=20.)]

    # create BasicCompSep instance with minimal parameters
    comp_separation = BasicCompSep(256, n_sub, combine_bands, n_years, components,
                                   verbose=verbose, band=band, use_noise_maps=use_noise_maps)

    # create mask indicating what pixels to use
    comp_separation.update_mask()

    # declare variable to store the results
    beta_values_cmbdust = []
    temp_values_cmbdust = []

    # simulate n_sim realisations of sky maps using QubicSkySim
    for i_sim in range(n_sim):

        if verbose:
            print("entering iteration number {:d} of n_sim={:d}".format(i_sim + 1, n_sim))
            print("generate sky maps...")

        # generate new sky maps
        comp_separation.generate_new_maps(noise_profile, nu_corr, spatial_corr)
        if verbose:
            print("sky maps generated")

        # put maps at the same resolution before separation
        comp_separation.put_maps_at_same_resolution()

        # perform component separation using FgBuster
        fg_res = comp_separation.perform_separation(stokes=stokes)
        beta_cmbdust = fg_res.x[0]
        beta_values_cmbdust.append(beta_cmbdust)
        if estimate_dust_temperature:
            temp_cmbdust = fg_res.x[1]
            temp_values_cmbdust.append(temp_cmbdust)
        if verbose:
            print("fgbuster separation terminated (message: {})\n\n".format(fg_res.message))

    if estimate_dust_temperature:
        res = np.array(beta_values_cmbdust), np.array(temp_values_cmbdust)
    else:
        res = np.array(beta_values_cmbdust)

    return res


class BasicCompSep(object):
    """
    Class that provides tools for basic component separation with FgBuster.
    Documentation:
    https://fgbuster.github.io/fgbuster/api/fgbuster.separation_recipes.html#fgbuster.separation_recipes.basic_comp_sep
    """

    def __init__(self, nside: int, nsub: int, combine_bands: bool, nyears: int, components, **kwargs):
        """
        Create instance of BasicCompSep.

        :param kwargs: additional information for initialization. If parameter combine_bands is set to False,
            kwargs must include an attribute 'band' which indicates which frequency band to use (150 or 220).
        """
        # map pixelation
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.pix_size = hp.nside2resol(nside, arcmin=True)

        # determine total number of sub-bands, corresponding frequencies and fwhms
        self.combine = combine_bands
        d150['nf_recon'] = d220['nf_recon'] = nsub  # nbr of reconstructed bands (output)
        d150['nf_sub'] = d220['nf_sub'] = 4 * nsub  # nbr of simulated bands (input)
        if combine_bands:
            self.band = None
            freqs150, fwhms150 = tools.get_sub_freqs_and_resolutions(d150)
            freqs220, fwhms220 = tools.get_sub_freqs_and_resolutions(d220)
            self.nbands = 2 * nsub
            self.freqs = np.concatenate((freqs150, freqs220), axis=0)
            self.fwhms = np.concatenate((fwhms150, fwhms220), axis=0)
        else:
            try:
                band = kwargs['band']
                freqs, fwhms = tools.get_sub_freqs_and_resolutions(qubic_dicts[band])
                self.nbands = nsub
                self.freqs = freqs
                self.fwhms = fwhms
                self.band = band
            except KeyError:
                print("please specify the frequency to use, or set combine_bands=True")

        # save original map fwhms
        self.original_fwhms = self.fwhms
        self.new_fwhm = self.fwhms[0]

        # integration time and coverage
        self.nyears = nyears
        if 'coverage_file' in kwargs:
            self.coverage = tools.get_coverage_from_file(DATADIR, file_name=kwargs['coverage_file'])
        else:
            self.coverage = tools.get_coverage_from_file(DATADIR)

        # sky components (CMB, Dust, ...)
        self.components = components

        # get instrument from experiments.yaml in cmbdb package
        instrument_name = 'Qubic' + str(self.nbands) + 'bands'
        self.instrument = fgb.get_instrument(instrument_name)
        self.instrument.frequency = self.freqs
        self.original_depth_i = None
        self.original_depth_p = None

        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']
        else:
            self.verbose = True
        if 'use_noise_maps' in kwargs:
            self.use_noise_maps = kwargs['use_noise_maps']
        else:
            self.use_noise_maps = False

        # mask (created with method update_mask)
        self.mask = None

        # maps (created with method generate_new_maps)
        self.maps = None
        self.noise = None

        # OLD CODE
        # self.l_min = 20
        # self.l_max = 2 * self.nside - 1
        # self.delta_ell = 16

    def perform_separation(self, stokes='IQU', method='BFGS', tol=1.0, solver_options=None):
        """
        Perform FgBuster algorithm.

        :param str stokes: Stokes parameters for which to perform the separation.
        :param str method: type of solver (default: 'BFGS')
            (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html for possibilities)
        :param float tol: tolerance for termination (default: 1.0)
        :param solver_options: dictionary containing solver options
            (default: {'disp': False} because True raises errors during FgBuster execution)
        :return: Dictionary which contains the amplitude of each components, the estimated parameter beta_d and dust
            temperature.
        """

        # build MaskedArray maps
        maps_m = ma.array(self.maps, mask=self.mask)

        # compute noise levels
        if self.use_noise_maps:
            assert self.noise is not None
            if self.verbose:
                print("compute noise level with noise maps provided...")
            noise_same_resol, _ = tools.same_resolution(self.noise, self.original_fwhms, self.new_fwhm)
            new_depth_i, new_depth_p = tools.get_depths(noise_same_resol, self.pix_size, self.mask)
        else:
            if self.verbose:
                print("estimate noise level analytically (noise maps not provided)...")
            ker_fwhms = tools.get_kernel_fwhms_for_smoothing(self.original_fwhms, self.new_fwhm)
            pix_size_deg = self.pix_size / 60  # in degrees instead of arcmin
            ker_sigmas_eff = np.sqrt(np.square(ker_fwhms / 2.355) + np.square(pix_size_deg) / 12)
            # the factor 1/12 is because RMS of uniform distribution on [0, 1] is 1/sqrt(12)
            correction = ker_sigmas_eff * np.sqrt(4 * np.pi) / pix_size_deg
            new_depth_i = self.original_depth_i / correction
            new_depth_p = self.original_depth_p / correction

            # approximate estimation
            # depth_i = 2. * np.sqrt(2 * self.nbands / (3 * self.nyears)) / np.ones(self.nbands)
            # depth_p = depth_i * np.sqrt(2)

        # update parameters of instrument with sensibilities
        self.instrument.depth_i = new_depth_i
        self.instrument.depth_p = new_depth_p
        if self.verbose:
            list_fmt = "[" + ", ".join(["{:.2f}"] * self.nbands) + "]"
            print("  -> depth_i = " + list_fmt.format(*new_depth_i))
            print("  -> depth_p = " + list_fmt.format(*new_depth_p))

        # apply FG Buster on desired Stokes parameters
        if stokes == 'IQU':
            map_slice = maps_m
        elif stokes == 'I':
            map_slice = maps_m[:, 0, :]
        elif stokes == 'QU':
            map_slice = maps_m[:, 1:, :]
        else:
            raise TypeError("incorrect specification of Stokes parameters (must be either 'IQU', 'QU' or 'I')")

        if solver_options is None:
            solver_options = {'disp': False}

        res = fgb.basic_comp_sep(self.components,
                                 self.instrument,
                                 map_slice,
                                 method=method,
                                 tol=tol,
                                 options=solver_options
                                 )
        return res

    def generate_new_maps(self, noise_profile=False, nu_corr=False, spatial_corr=False):
        """
        Generate new maps with given parameters.

        :param noise_profile: add noise profile (inhomogeneity)
        :param nu_corr: add frequency correlations
        :param spatial_corr: add spatial correlations
        """

        map_args = self.coverage, self.nyears, noise_profile, nu_corr, spatial_corr
        if self.combine:
            common_seed = np.random.randint(1000000)
            cmbdust150, _, noise150 = tools.generate_cmb_dust_maps(d150, *map_args, seed=common_seed)
            cmbdust220, _, noise220 = tools.generate_cmb_dust_maps(d220, *map_args, seed=common_seed)
            cmbdust = np.concatenate((cmbdust150, cmbdust220), axis=0)
            noise = np.concatenate((noise150, noise220), axis=0)
        else:
            cmbdust, _, noise = tools.generate_cmb_dust_maps(qubic_dicts[self.band], *map_args)

        # modify shape of maps for FgBuster
        cmbdust = np.transpose(cmbdust, (0, 2, 1))
        noise = np.transpose(noise, (0, 2, 1))

        self.maps = cmbdust
        self.noise = noise

        # reset fwhms
        self.fwhms = self.original_fwhms
        self.original_depth_i, self.original_depth_p = tools.get_depths(self.noise, self.pix_size, self.mask)

    def put_maps_at_same_resolution(self, target=None):
        """
        Put the maps at the same resolution.

        :param float target: optional. Specify the common new resolution of the maps.
        """
        # don't smooth noise maps, because in reality they are not provided
        # if self.noise is not None:
        #     self.noise, _ = tools.same_resolution(self.noise, self.fwhms, fwhm_target=target)
        self.maps, self.new_fwhm = tools.same_resolution(self.maps, self.fwhms, fwhm_target=target)
        self.fwhms = self.new_fwhm + np.zeros(self.nbands)
        if self.verbose:
            print("maps smoothed to fwhm={:.3f} degrees".format(self.new_fwhm))

    def update_mask(self, threshold=0.5):
        """
        Update the mask.

        :param threshold: value of coverage above which pixels are selected.
        """
        mask = np.empty((self.nbands, 3, self.npix), dtype=bool)
        mask[...] = (self.coverage < (threshold * np.max(self.coverage)))
        self.mask = mask


# TODO: 1) compute noise analytically rather than using smoothed maps
# 1) -> done but there is a small difference (of approximately 1 uK.arcmin), no idea why
# TODO: 2) find a way to perform separation in harmonic space
# TODO: 3) use weighted_comp_sep to account for noise profile (circular dependence)
# TODO: 4) check results with 220 GHz maps smoothed to 150 GHz resolution (to see if bias in combined separation is
#          due to smoothing
# 4) -> it seems that the bias could be due to this smoothing (220 GHz results after smoothing to 0.43° fwhm are
#       compatible with the hypothesis)


if __name__ == "__main__":
    # arguments = read_arguments()
    r = run_beta_dust_estimation_combine_bands(15, 3, 3, combine_bands=True, band=None, use_noise_maps=True)
    q = run_beta_dust_estimation_combine_bands(15, 3, 3, combine_bands=True, band=None, use_noise_maps=False)

    print("  average =", np.mean(r))
    print("deviation =", np.std(r))

    print("  average =", np.mean(q))
    print("deviation =", np.std(q))

    sys.exit(0)
