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
from astropy.io import fits

# qubic modules
import qubic

# imports needed for component separation
import fgbuster as fgb

import warnings

warnings.filterwarnings("ignore")

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
                             spatial_corr=False, alm_space=False, verbose=True, combine_bands=True, band=None,
                             noisy=True, stokes='IQU', noise_computation_use_map=False, dust_only=False,
                             smooth_at_gen=False, fwhm_gen=None, nside=256, iib=True, full_sky=False,
                             noise_covcut=None, add_alm_error=False, pixwin_correction=True, test_all_stokes=False,
                             ):
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
    :param bool alm_space: perform separation in harmonic (alm) space
    :param bool|None verbose: print progress information (default: True)
    :param bool|None combine_bands: combine 150 and 220 GHz bands (each with n_sub sub-bands)
    :param int|None band: frequency band to use if not combined (must be 150 or 220)
    :param bool|None noisy: include noise in the maps to separate
    :param str|None stokes: Stokes parameters for which to perform the separation (must be 'IQU', 'I' or 'QU')
    :param bool|None noise_computation_use_map: if True, involves smoothing noise maps to compute directly the RMS of
        noise pixels, otherwise uses the original level (before smoothing) and derives the noise level after smoothing
        with an analytical formula.
    :param bool|None dust_only: generate only dust maps (no CMB)
    :param bool|None smooth_at_gen: smooth maps at generation by QubicSkySim
    :param float|None fwhm_gen: specify a unique fwhm (=resolution) when generating maps (needs smooth_gen=True)
    :param int nside: specify the nside when generating maps (default: 256)
    :param bool iib: integrate simulated maps into bands (if False, simulate only n_sub maps)
    :param bool full_sky: use a uniform coverage on the full sky
    :param float|None noise_covcut: coverage cut when generating noise maps
    """

    # indicate what components the maps contain
    cmb_component = fgb.CMB()

    dust_nu0 = 150.
    dust_component = fgb.Dust(dust_nu0, temp=20.)

    components = [dust_component]
    if not dust_only:
        components.append(cmb_component)

    # create BasicCompSep instance with minimal parameters
    comp_separation = BasicCompSep(nside, n_sub, combine_bands, n_years, components,
                                   verbose=verbose, band=band, noisy=noisy,
                                   noise_computation_use_map=noise_computation_use_map,
                                   smooth_at_gen=smooth_at_gen, fwhm_gen=fwhm_gen,
                                   dust_only=dust_only, iib=iib, full_sky=full_sky,
                                   noise_covcut=noise_covcut)

    # declare variable to store the results
    beta_values_cmbdust = []

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
        if not alm_space:
            comp_separation.put_maps_at_same_resolution()

        # perform component separation using FgBuster
        stk_list = [stokes] if not test_all_stokes else ['IQU', 'I', 'QU']
        for stk in stk_list:
            fg_res = comp_separation.perform_separation(alm_space, stokes=stk, lmax=None,
                                                        add_alm_error=add_alm_error,
                                                        pixwin_correction=pixwin_correction)
            if verbose:
                print(stk + " fgbuster separation terminated (message: {})".format(fg_res.message))
            beta_cmbdust = fg_res.x[0]
            beta_values_cmbdust.append(beta_cmbdust)

        if verbose:
            print("\n\n")

    return np.array(beta_values_cmbdust)


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

        if 'noise_covcut' in kwargs:
            self.noise_covcut = kwargs['noise_covcut']
        else:
            self.noise_covcut = None

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
            self.smooth_at_gen = False

        if self.smooth_at_gen:
            if 'fwhm_gen' in kwargs and kwargs['fwhm_gen'] is not None:
                self.fwhm_gen = kwargs['fwhm_gen']
            else:
                self.fwhm_gen = self.fwhms[0]
            self.fwhms = np.ones(self.nbands) * self.fwhm_gen
        else:
            self.fwhm_gen = None

        # save original map fwhms and stuff
        self.beams = self.fwhms
        self.new_fwhm = None

        # integration time and coverage
        self.nyears = nyears
        if 'full_sky' in kwargs:
            self.full_sky = kwargs['full_sky']
        else:
            self.full_sky = False

        # only one coverage is implemented right now: Qubic coverage with dtheta=40
        if not self.full_sky:
            cover, file_name = tools.get_coverage_from_file(DATADIR)
            self.coverage = hp.ud_grade(cover, nside_out=self.nside)
        else:
            self.coverage = None

        # if 'coverage_file' in kwargs:
        #     cover = tools.get_coverage_from_file(DATADIR, file_name=kwargs['coverage_file'])
        # else:
        #     cover = tools.get_coverage_from_file(DATADIR)
        self.coverage_eff = None

        # sky components (CMB, Dust, ...)
        self.components = components

        # get instrument from experiments.yaml in cmbdb package
        # instrument_name = 'Qubic' + str(self.nbands) + 'bands'
        # self.instrument = fgb.get_instrument(instrument_name)
        self.instrument = fgb.get_instrument('Qubic')
        self.instrument.frequency = self.freqs
        self.instrument.fwhm = self.beams
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

        # harmonic space
        self.lmin = 0
        self.lmax = 2 * self.nside
        # self.delta_ell = 16

    def get_alm_maps(self, apply_correction, true_resol=False, use_pixel_weights=True,
                     ref_fwhm_arcmin=None, lmax=None, add_error=False, pixwin_correction=True):
        """
        Compute alm maps from pixel maps and format them for FgBuster.

        :param bool apply_correction: apply correction for maps at different resolutions
        :param bool true_resol: correct the alm series to a fwhm=0 resolution
        :param bool use_pixel_weights: use individual pixel weights for alm computation (should be set to True, except
            for testing)
        :param float ref_fwhm_arcmin: reference resolution for alm correction in arcminutes
        :return: alm maps ready for FgBuster separation
        """
        l_max = self.lmax if lmax is None else lmax

        ell = np.arange(start=0, stop=l_max + 1)

        if ref_fwhm_arcmin is None:
            ref_fwhm_arcmin = 60 * d150['synthbeam_peak150_fwhm'] * 150. / self.freqs[0]
        ref_sigma_rad = np.deg2rad(ref_fwhm_arcmin / 60.) / 2.355
        ref_fl = np.exp(- 0.5 * np.square(ref_sigma_rad * ell))

        beam_sigmas_rad = np.deg2rad(self.instrument.fwhm) / 2.355

        pixwin = hp.pixwin(self.nside, lmax=l_max) if pixwin_correction else np.ones(l_max + 1)

        # compute maps
        alm_maps = None
        for f in range(self.nbands):
            if use_pixel_weights:
                # update FITS file with pixel weights using effective sky coverage
                weights_hdu = fits.PrimaryHDU(self.coverage_eff)
                weights_hdu.writeto("data/full_weights/healpix_full_weights_nside_{:04d}.fits".format(self.nside),
                                    overwrite=True)
                # now compute the alms with Healpy
                alms = hp.map2alm(self.sky_maps[f], lmax=l_max, pol=True,
                                  use_pixel_weights=True, datapath="data/")
            else:
                alms = hp.map2alm(self.sky_maps[f], lmax=l_max, pol=True)
            correction = None
            if f == 0:
                sh = np.shape(alms)
                alm_maps = np.empty((self.nbands, sh[0], 2 * sh[1]))
            if apply_correction:
                # use pixel window function or not ??
                gauss_fl = np.exp(- 0.5 * np.square(beam_sigmas_rad[f] * ell))
                if not true_resol:
                    correction = ref_fl / gauss_fl / pixwin
                else:
                    correction = 1 / gauss_fl / pixwin
            for i, t in enumerate(alms):
                alm_maps[f, i] = tools.format_alms(hp.almxfl(t, correction) if apply_correction else t)
                if add_error:
                    alm_maps[f, i] = tools.add_errors_alm(alm_maps[f, i])
        return alm_maps

    def perform_separation(self, alm_space,
                           ref_fwhm=None,
                           lmax=None,
                           add_alm_error=False,
                           pixwin_correction=True,
                           stokes='IQU', method='BFGS', tol=1e-12, solver_options=None):
        """
        Perform FgBuster algorithm.

        :param alm_space: perform separation in harmonic (alm) space
        :param str stokes: Stokes parameters for which to perform the separation.
        :param str method: type of solver (default: 'BFGS')
            (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html for possibilities)
        :param float tol: tolerance for termination
        :param solver_options: dictionary containing solver options
            (default: {'disp': False} because True raises errors during FgBuster execution)
        :return: Dictionary containing all results.
        """

        # check if separation happens in harmonic space, compute alms
        if alm_space:
            apply_correction = not self.smooth_at_gen
            maps_to_use = self.get_alm_maps(apply_correction, ref_fwhm_arcmin=ref_fwhm, lmax=lmax,
                                            add_error=add_alm_error, pixwin_correction=pixwin_correction)
        # otherwise, use masked pixel maps
        else:
            maps_to_use = ma.array(self.sky_maps, mask=self.mask, fill_value=hp.UNSEEN)

        # compute noise levels
        if self.noisy:
            if not alm_space:
                if self.noise_computation_use_map:
                    assert self.noise_maps is not None
                    if self.verbose:
                        print("compute noise level with noise maps provided...")
                    noise_same_resol, _ = tools.same_resolution(self.noise_maps, self.beams, self.new_fwhm)
                    new_depth_i, new_depth_p = tools.get_depths(noise_same_resol, self.pix_size, self.mask)
                else:
                    if self.verbose:
                        print("estimate noise level analytically (noise maps not provided)...")
                    ker_fwhms, _ = tools.get_kernel_fwhms_for_smoothing(self.beams, self.new_fwhm)
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

            else:
                self.instrument.depth_i = self.original_depth_i
                self.instrument.depth_p = self.original_depth_p

            if self.verbose:
                list_fmt = "[" + ", ".join(["{:.2f}"] * self.nbands) + "]"
                print("  -> depth_i = " + list_fmt.format(*self.instrument.depth_i))
                print("  -> depth_p = " + list_fmt.format(*self.instrument.depth_p))

        # apply FG Buster on desired Stokes parameters
        if stokes == 'IQU':
            map_slice = maps_to_use
        elif stokes == 'I':
            map_slice = maps_to_use[:, 0, :]
        elif stokes == 'QU':
            map_slice = maps_to_use[:, 1:, :]
        else:
            raise TypeError("incorrect specification of Stokes parameters (must be either 'IQU', 'QU' or 'I')")

        if solver_options is None:
            solver_options = {}
        solver_options['disp'] = True
        fg_args = self.components, self.instrument, map_slice
        fg_kwargs = {'method': method, 'tol': tol, 'options': solver_options}
        try:
            res = fgb.basic_comp_sep(*fg_args, **fg_kwargs)
        except KeyError:
            fg_kwargs['options']['disp'] = False
            res = fgb.basic_comp_sep(*fg_args, **fg_kwargs)

        return res

    def generate_new_maps(self, noise_profile=False, nu_corr=False, spatial_corr=False):
        """
        Generate new maps with given parameters.

        :param bool noise_profile: add noise profile (inhomogeneity)
        :param bool nu_corr: add frequency correlations
        :param bool spatial_corr: add spatial correlations
        """

        map_args = None if self.full_sky else self.coverage.copy(), self.nyears, noise_profile, nu_corr, spatial_corr
        kw = {'dust_only': self.dust_only, 'fwhm_gen': self.fwhm_gen, 'iib': self.iib,
              'noise_covcut': self.noise_covcut}
        if self.combine:
            common_seed = np.random.randint(1000000)
            kw.update({'seed': common_seed})
            a, b, c, g = tools.generate_cmb_dust_maps(d150, *map_args, **kw)
            d, e, f, _ = tools.generate_cmb_dust_maps(d220, *map_args, **kw)
            m1 = np.concatenate((a, d), axis=0)
            m2 = np.concatenate((b, e), axis=0)
            m3 = np.concatenate((c, f), axis=0)
            self.coverage_eff = g  # coverage_eff for 150 and 220 are the same
        else:
            m1, m2, m3, c = tools.generate_cmb_dust_maps(qubic_dicts[self.band], *map_args, **kw)
            self.coverage_eff = c

        # take noisy maps or not
        sky_maps = m1 if self.noisy else m2
        noise_maps = m3

        # modify shape of maps for FgBuster
        self.sky_maps = np.transpose(sky_maps, (0, 2, 1))
        self.noise_maps = np.transpose(noise_maps, (0, 2, 1))

        # update the mask with unseen pixels
        # threshold can be adjusted indirectly by using noise_covcut at map generation
        self.update_mask(threshold=0.0)

        # set/reset fwhms, set instrument beams
        self.fwhms = self.beams
        self.original_depth_i, self.original_depth_p = tools.get_depths(self.noise_maps, self.pix_size,
                                                                        self.mask, self.coverage_eff)

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

    def update_mask(self, threshold=0.):
        """
        Update the mask. The mask is primarily used to exclude unseen pixels during noise computation.

        :param threshold: value of coverage above which pixels are selected.
        """
        mask = np.empty((self.nbands, 3, self.npix), dtype=bool)
        mask[...] = (self.coverage_eff <= (threshold * np.max(self.coverage_eff)))
        self.mask = mask

# Tasks to do:
# TODO: 1) fix analytical computation of noise depths (especially @ 150 GHz ?)
# TODO: 2) fix noisy separation in alm space in case of partial sky maps (depth computation)
# TODO: 3) use weighted_comp_sep to account for noise profile (circular dependence)

# General questions:
# - why does the estimation of beta work better when separating on QU rather than IQU or I ?

# - why is the correction of alm series so imprecise (errors up to 1e-2) and how is it connected to the bias
# induced in the estimations of beta ? in particular, why is this bias always towards lower values of beta ?

# - how does a sharp coverage cut perturb the estimations ?
# why is this effect more visible at 150 GHz than at 220 GHz ?

# - is it relevant to correct alm series to a "infinite" resolution or is it impossible because of sky
# pixelation ?
# - why are the estimations so much better when suppressing multipoles >=~ 200 ?
# (possible answer: because alm correction is not good enough above this scale ?)

# - ...


if __name__ == "__main__":
    # parameters
    nsim = 1
    ns = 3
    ny = 3

    dd = {'noisy': False,
          'alm_space': True,
          'dust_only': False,
          'iib': False,
          'noise_computation_use_map': True,
          'smooth_at_gen': False,
          'full_sky': False,
          'noise_covcut': None,
          'add_alm_error': False,
          'pixwin_correction': False,  # since maps are directly generated at desired nside...
          'test_all_stokes': True,
          }

    p = run_beta_dust_estimation(nsim, ns, ny, combine_bands=False, band=150, **dd)
    q = run_beta_dust_estimation(nsim, ns, ny, combine_bands=False, band=220, **dd)
    r = run_beta_dust_estimation(nsim, ns, ny, combine_bands=True, **dd)

    tools.print_list(p - 1.54, '.7f')
    tools.print_list(q - 1.54, '.7f')
    tools.print_list(r - 1.54, '.7f')

    sys.exit(0)
