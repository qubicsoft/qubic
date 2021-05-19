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

# general imports
import sys
import numpy as np
import numpy.ma as ma
import healpy as hp

# qubic modules
import qubic

# imports needed for component separation
import fgbuster as fgb

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

# qubic dictionaries for 150 GHz and 220 Ghz
config_150, config_220 = 'FI-150', 'FI-220'
d150_name = '/home/simon/Documents/qubic/qubicsoft/qubic/doc/FastSimulator/FastSimDemo_{}.dict'.format(config_150)
d220_name = '/home/simon/Documents/qubic/qubicsoft/qubic/doc/FastSimulator/FastSimDemo_{}.dict'.format(config_220)
d150, d220 = qubic.qubicdict.qubicDict(), qubic.qubicdict.qubicDict()
d150.read_from_file(d150_name)
d220.read_from_file(d220_name)
qubic_dicts = {150: d150, 220: d220}


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


def run_beta_dust_estimation(n_sim: int, n_sub: int, n_years: int, noise_profile=False, nu_corr=False,
                             spatial_corr=False, verbose=True, combine_bands=True, band=None, noisy=True,
                             stokes='IQU', estimate_dust_temperature=False, noise_computation_use_map=False,
                             dust_only=False, smooth_at_gen=True, fwhm_gen=None, nside=256, iib=True):
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
    :param bool|None noise_profile: QubicSkySim "noise_profile" (ie. noise inhomogeneity)
    :param bool|None nu_corr: QubicSkySim "nunu_correlation" (ie. noise frequency correlations)
    :param bool|None spatial_corr: QubicSkySim "spatial_noise" (ie. spatial noise correlations)
    :param bool|None verbose: print progress information (default: True)
    :param bool|None combine_bands: combine 150 and 220 GHz bands (each with n_sub sub-bands)
    :param int|None band: frequency band to use if not combined (must be 150 or 220)
    :param bool|None noisy: include noise in the maps to separate
    :param str|None stokes: Stokes parameters for which to perform the separation (must be 'IQU', 'I' or 'QU')
    :param bool|None estimate_dust_temperature: estimate dust temperature in addition to spectral index
    :param bool|None noise_computation_use_map: if True, involves smoothing noise maps to compute directly the RMS of
        noise pixels, otherwise uses the original level (before smoothing) and derives the noise level after smoothing
        with an analytical formula.
    :param bool|None dust_only: generate only dust maps (no CMB)
    :param bool|None smooth_at_gen: smooth maps at generation by QubicSkySim
    :param float|None fwhm_gen: specify a unique fwhm (=resolution) when generating maps (needs smooth_gen=True)
    :param int nside: specify the nside when generating maps (default: 256)
    :param bool iib: integrate simulated maps into bands (if False, simulate only n_sub maps)
    """

    # indicate what components the maps contain
    cmb_component = fgb.CMB()
    if estimate_dust_temperature:
        dust_component = fgb.Dust(150.)
    else:
        dust_component = fgb.Dust(150., temp=20.)
    if dust_only:
        components = [dust_component]
    else:
        components = [cmb_component, dust_component]

    # create BasicCompSep instance with minimal parameters
    comp_separation = BasicCompSep(nside, n_sub, combine_bands, n_years, components,
                                   verbose=verbose, band=band, noisy=noisy,
                                   noise_computation_use_map=noise_computation_use_map,
                                   smooth_at_gen=smooth_at_gen, fwhm_gen=fwhm_gen,
                                   dust_only=dust_only, iib=iib)

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
        d150['nside'] = nside
        d220['nside'] = nside
        self.npix = hp.nside2npix(nside)
        self.pix_size = hp.nside2resol(nside, arcmin=True)

        # determine total number of sub-bands, corresponding frequencies and fwhms
        self.combine = combine_bands
        d150['nf_sub'] = d220['nf_sub'] = 4 * nsub  # nbr of simulated bands (input)
        d150['nf_recon'] = d220['nf_recon'] = nsub  # nbr of reconstructed bands (output)

        if 'iib' in kwargs:
            self.iib = kwargs['iib']
        else:
            self.iib = True

        if combine_bands:
            self.band = None
            freqs150, fwhms150 = tools.get_sub_freqs_and_resolutions(d150)
            freqs220, fwhms220 = tools.get_sub_freqs_and_resolutions(d220)
            self.nbands = 2 * nsub
            self.freqs = np.concatenate((freqs150, freqs220), axis=0)
            self.fwhms = np.concatenate((fwhms150, fwhms220), axis=0)
        else:
            try:
                self.band = kwargs['band']
                self.freqs, self.fwhms = tools.get_sub_freqs_and_resolutions(qubic_dicts[self.band])
                self.nbands = nsub
            except KeyError:
                print("please specify the frequency to use, or set combine_bands=True")

        # determine when and if to smooth maps
        if 'smooth_at_gen' in kwargs:
            self.smooth_at_gen = kwargs['smooth_at_gen']
        else:
            self.smooth_at_gen = True

        if self.smooth_at_gen:
            if 'fwhm_gen' in kwargs and kwargs['fwhm_gen'] is not None:
                self.fwhm_gen = kwargs['fwhm_gen']
            else:
                self.fwhm_gen = self.fwhms[0]
            self.fwhms = np.ones(self.nbands) * self.fwhm_gen
        else:
            self.fwhm_gen = None

        # save original map fwhms
        self.original_fwhms = self.fwhms
        self.new_fwhm = None

        # integration time and coverage
        self.nyears = nyears
        if 'coverage_file' in kwargs:
            cov = tools.get_coverage_from_file(DATADIR, file_name=kwargs['coverage_file'])
        else:
            cov = tools.get_coverage_from_file(DATADIR)
        self.coverage = hp.ud_grade(cov, nside_out=self.nside)

        # sky components (CMB, Dust, ...)
        self.components = components

        # get instrument from experiments.yaml in cmbdb package
        # instrument_name = 'Qubic' + str(self.nbands) + 'bands'
        # self.instrument = fgb.get_instrument(instrument_name)
        self.instrument = fgb.get_instrument('Qubic')
        self.instrument.frequency = self.freqs
        self.instrument.fwhm = self.original_fwhms
        self.original_depth_i = None
        self.original_depth_p = None

        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']
        else:
            self.verbose = True

        if 'noise_computation_use_map' in kwargs:
            self.noise_computation_use_map = kwargs['noise_computation_use_map']
        else:
            self.noise_computation_use_map = False

        if 'noisy' in kwargs:
            self.noisy = kwargs['noisy']
        else:
            self.noisy = True

        if 'dust_only' in kwargs:
            self.dust_only = kwargs['dust_only']
        else:
            self.dust_only = False

        # mask (created with method update_mask)
        self.mask = None

        # maps (created with method generate_new_maps)
        self.sky_maps = None
        self.noise_maps = None

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
        maps_m = ma.array(self.sky_maps, mask=self.mask)

        # compute noise levels
        if self.noisy:
            if self.noise_computation_use_map:
                assert self.noise_maps is not None
                if self.verbose:
                    print("compute noise level with noise maps provided...")
                noise_same_resol, _ = tools.same_resolution(self.noise_maps, self.original_fwhms, self.new_fwhm)
                new_depth_i, new_depth_p = tools.get_depths(noise_same_resol, self.pix_size, self.mask)
            else:
                if self.verbose:
                    print("estimate noise level analytically (noise maps not provided)...")
                ker_fwhms, _ = tools.get_kernel_fwhms_for_smoothing(self.original_fwhms, self.new_fwhm)
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

        :param bool noise_profile: add noise profile (inhomogeneity)
        :param bool nu_corr: add frequency correlations
        :param bool spatial_corr: add spatial correlations
        """

        map_args = self.coverage, self.nyears, noise_profile, nu_corr, spatial_corr
        if self.combine:
            common_seed = np.random.randint(1000000)
            kw = {'seed': common_seed, 'dust_only': self.dust_only, 'fwhm_gen': self.fwhm_gen, 'iib': self.iib}
            a, b, c = tools.generate_cmb_dust_maps(d150, *map_args, **kw)
            d, e, f = tools.generate_cmb_dust_maps(d220, *map_args, **kw)
            m1 = np.concatenate((a, d), axis=0)
            m2 = np.concatenate((b, e), axis=0)
            m3 = np.concatenate((c, f), axis=0)
        else:
            kw = {'dust_only': self.dust_only, 'fwhm_gen': self.fwhm_gen, 'iib': self.iib}
            m1, m2, m3 = tools.generate_cmb_dust_maps(qubic_dicts[self.band], *map_args, **kw)

        # take noisy maps or not
        if self.noisy:
            sky_maps = m1
        else:
            sky_maps = m2
        noise_maps = m3

        # modify shape of maps for FgBuster
        self.sky_maps = np.transpose(sky_maps, (0, 2, 1))
        self.noise_maps = np.transpose(noise_maps, (0, 2, 1))

        # reset fwhms
        self.fwhms = self.original_fwhms
        self.original_depth_i, self.original_depth_p = tools.get_depths(self.noise_maps, self.pix_size, self.mask)

    def put_maps_at_same_resolution(self, target=None):
        """
        Put the maps at the same resolution.

        :param float target: optional. Specify the common new resolution of the maps.
        """
        if not self.smooth_at_gen:
            self.sky_maps, self.new_fwhm = tools.same_resolution(self.sky_maps, self.fwhms, fwhm_target=target)
            self.fwhms = self.new_fwhm + np.zeros(self.nbands)
            if self.verbose:
                print("maps smoothed to fwhm={:.5f} degrees".format(self.new_fwhm))
        else:
            if self.verbose:
                print("maps were smoothed at generation")

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


if __name__ == "__main__":
    # parameters
    nsim = 1
    ns = 3
    ny = 3
    noise = False

    p = run_beta_dust_estimation(nsim, ns, ny, combine_bands=False, band=150, noisy=noise, iib=False)
    q = run_beta_dust_estimation(nsim, ns, ny, combine_bands=False, band=220, noisy=noise, iib=False)

    print(p)
    print(q)

    sys.exit(0)
