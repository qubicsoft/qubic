"""
Perform FgBuster separation on simulated Qubic sky maps with many frequency sub-bands.
Command line arguments should be:
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
from astropy.io import fits

# Qubic modules
import qubic
from qubic import QubicSkySim as qss

# Imports needed for component separation
import fgbuster as fgb

# Import useful functions from other script
import component_separation

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


def generate_cmb_dust_maps(dico, coverage, n_years, sc, nunu, save_maps=False, return_maps=True):
    """Save CMB+Dust maps to FITS image format for later use, and/or return them immediately."""

    seed = np.random.randint(1000000)
    sky_config = {'dust': 'd0', 'cmb': seed}
    qubic_sky = qss.Qubic_sky(sky_config, dico)
    cmb_dust, cmb_dust_noiseless, cmb_dust_noise_only, _ = \
        qubic_sky.get_partial_sky_maps_withnoise(coverage=coverage,
                                                 Nyears=n_years,
                                                 spatial_noise=sc,  # noise spatial correlations
                                                 nunu_correlation=nunu,  # noise frequency correlations
                                                 verbose=False,
                                                 seed=seed,
                                                 )
    if save_maps:
        common_fmt = "band{}_sub{}_years{}_spatial{}_nunu{}_seed{}.fits"
        common = common_fmt.format(dico['band'], dico['nf_recon'], n_years, sc, nunu, seed)

        hdu_cmb_dust = fits.PrimaryHDU(cmb_dust)
        hdu_cmb_dust.writeto("data/cmb_dust_maps/cmb_dust" + common)

        hdu_cmb_dust_noiseless = fits.PrimaryHDU(cmb_dust_noiseless)
        hdu_cmb_dust_noiseless.writeto("data/cmb_dust_maps/cmb_dust_noiseless" + common)

        hdu_cmb_dust_noise_only = fits.PrimaryHDU(cmb_dust_noise_only)
        hdu_cmb_dust_noise_only.writeto("data/cmb_dust_maps/cmb_dust_noise_only" + common)

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


def run_simu_single_band(band: int, n_sim: int, n_sub: int, n_years: int, sc: bool, nunu: bool, **kwargs):
    """
    Run sky simulations (using `QubicSkySim`) and determine spectral parameters.

    Possible keyword arguments are: *verbose* option (boolean), *run_on_noise* (boolean) which determines if the
    noise maps should be fed to the FgBuster algorithm, and *stokes* (can be 'IQU', 'I' or 'QU') which determines
    the set of Stokes parameters used for the separation.

    :param int band: main band frequency (150 or 220 GHz)
    :param int n_sub: number of sub-bands
    :param int n_sim: number of simulations
    :param int n_years: number of years of observation for simulated maps
    :param bool sc: QubicSkySim "spatial_noise" (ie. spatial noise correlations)
    :param bool nunu: QubicSkySim "nunu_correlation" (ie. noise frequency correlations)
    """
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']
    else:
        verbose = False

    print("run single band simulation with parameters:")
    print("     band = {}".format(band))
    print("    n_sub = {}".format(n_sub))
    print("    n_sim = {}".format(n_sim))
    print("  n_years = {}".format(n_years))
    print("       sc = {}".format(sc))
    print("     nunu = {}".format(nunu))

    # get dictionary for given frequency
    dico = qubic_dicts[band]

    # adjust number of sub-bands for sky simulation and reconstruction
    dico['nf_recon'] = n_sub
    dico['nf_sub'] = 4 * n_sub  # should be integer multiple of 'nf_recon', like 4 or 5

    # get coverage map and define pixels to use
    coverage = get_coverage_from_file()
    okpix_inside = (coverage > (0.5 * np.max(coverage)))
    if verbose:
        print("coverage from file ok")

    # get sub-bands characteristics
    freqs, fwhms = get_sub_freqs_and_resolutions(dico)
    if verbose:
        print("sub-bands frequencies and resolutions ok")
        print("   --> {} sub-bands".format(n_sub))
        list_fmt = "[" + ", ".join(["{:.6f}"] * len(freqs)) + "]"
        print("   --> centered around frequencies " + list_fmt.format(*freqs))

    # define common resolution for input maps (lowest one)
    # fwhm_common = np.max(fwhms) + 1e8

    # variable to store the results of parameter estimation
    if 'run_on_noise' in kwargs:
        run_on_noise = kwargs['run_on_noise']
    else:
        run_on_noise = False
    beta_values_cmbdust = []
    beta_values_noiseless = []
    beta_values_noise_only = []

    if 'stokes' in kwargs:
        stokes = kwargs['stokes']
    else:
        stokes = 'IQU'
    dic_stk = {'I': 0, 'Q': 1, 'U': 2}
    sig_to_noise = np.zeros((n_sim, len(stokes)))  # n_sim values for each Stokes parameter

    for i_sim in range(n_sim):

        if verbose:
            print("entering iteration number {:d} of n_sim={:d} of loop".format(i_sim + 1, n_sim))

        # generate sky map using QubicSkySim, with random seed
        if verbose:
            print("generate sky map...")

        cmb_dust, cmb_dust_noiseless, cmb_dust_noise_only = \
            generate_cmb_dust_maps(dico, coverage, n_years, sc, nunu)

        for stk in stokes:
            i_stk = dic_stk[stk]
            for i_band in range(n_sub):
                sig_to_noise[i_sim, i_stk] += np.std(cmb_dust[i_band, :, i_stk]) \
                                              / np.std(cmb_dust_noise_only[i_band, :, i_stk])
            sig_to_noise[i_sim, i_stk] /= n_sub

        # modify shape to match FgBuster standard
        cmb_dust = np.transpose(cmb_dust, (0, 2, 1))
        if verbose:
            print("sky map generated")

        # perform component separation using FgBuster
        if verbose:
            print("perform fgbuster separation...")
        comp_separation = component_separation.CompSep(dico)

        fg_res_cmbdust = comp_separation.fg_buster(maps_in=cmb_dust,
                                                   components=[fgb.CMB(), fgb.Dust(band, temp=20.)],
                                                   map_freqs=freqs,
                                                   map_fwhms_deg=fwhms,
                                                   # target=fwhm_common,
                                                   ok_pix=okpix_inside,
                                                   stokes=stokes)
        beta_cmbdust = fg_res_cmbdust.x[0]
        beta_values_cmbdust.append(beta_cmbdust)

        if run_on_noise:
            cmb_dust_noiseless = np.transpose(cmb_dust_noiseless, (0, 2, 1))
            cmb_dust_noise_only = np.transpose(cmb_dust_noise_only, (0, 2, 1))
            fg_res_noiseless = comp_separation.fg_buster(maps_in=cmb_dust_noiseless,
                                                         components=[fgb.CMB(), fgb.Dust(band, temp=20.)],
                                                         map_freqs=freqs,
                                                         map_fwhms_deg=fwhms,
                                                         # target=fwhm_common,
                                                         ok_pix=okpix_inside,
                                                         stokes=stokes)
            fg_res_noise_only = comp_separation.fg_buster(maps_in=cmb_dust_noise_only,
                                                          components=[fgb.CMB(), fgb.Dust(band, temp=20.)],
                                                          map_freqs=freqs,
                                                          map_fwhms_deg=fwhms,
                                                          # target=fwhm_common,
                                                          ok_pix=okpix_inside,
                                                          stokes=stokes)
            beta_noiseless = fg_res_noiseless.x[0]
            beta_noise_only = fg_res_noise_only.x[0]
            beta_values_noiseless.append(beta_noiseless)
            beta_values_noise_only.append(beta_noise_only)

        if verbose:
            print("fgbuster separation performed (beta = {:.3f})".format(beta_cmbdust))

        del cmb_dust
        del cmb_dust_noiseless
        del cmb_dust_noise_only

    if verbose:
        print("execution terminated, writing / printing results...\n")

    if run_on_noise:
        return np.array(beta_values_cmbdust), np.array(beta_values_noiseless), np.array(beta_values_noise_only), \
               sig_to_noise
    else:
        return np.array(beta_values_cmbdust), sig_to_noise


# TODO: add function run_simu_double_band which uses both 150 and 220 GHz bands for separation


if __name__ == "__main__":
    # Qubic dictionaries for 150 GHz and 220 Ghz
    config_150, config_220 = 'FI-150', 'FI-220'
    d150_name = QUBIC_DATADIR + '/doc/FastSimulator/FastSimDemo_{}.dict'.format(config_150)
    d220_name = QUBIC_DATADIR + '/doc/FastSimulator/FastSimDemo_{}.dict'.format(config_220)
    d150, d220 = qubic.qubicdict.qubicDict(), qubic.qubicdict.qubicDict()
    d150.read_from_file(d150_name)
    d220.read_from_file(d220_name)
    qubic_dicts = {150: d150, 220: d220}

    # arguments = read_arguments()
    
    nsimu = 5
    results_dic_IQU = {}
    results_dic_I = {}
    for f in [150, 220]:
        for nsb in [3, 4, 5]:
            for nu in [False, True]:
                for s in [False, True]:
                    # IQU separation
                    beta_IQU, signoise_IQU = run_simu_single_band(f, nsimu, nsb, 400, s, nu, stokes='IQU')
                    results_dic_IQU.update({'beta_{}_{}_{}{}'.format(f, nsb, str(int(nu)), str(int(s))): beta_IQU,
                                            'signoiseI_{}_{}_{}{}'.format(f, nsb, str(int(nu)),
                                                                          str(int(s))): np.ravel(signoise_IQU[:, 0]),
                                            'signoiseQ_{}_{}_{}{}'.format(f, nsb, str(int(nu)),
                                                                          str(int(s))): np.ravel(signoise_IQU[:, 1]),
                                            'signoiseU_{}_{}_{}{}'.format(f, nsb, str(int(nu)),
                                                                          str(int(s))): np.ravel(signoise_IQU[:, 2])
                                            })
                    # I separation
                    beta_I, signoise_I = run_simu_single_band(f, nsimu, nsb, 400, s, nu, stokes='I')
                    results_dic_I.update({'beta_{}_{}_{}{}'.format(f, nsb, str(int(nu)), str(int(s))): beta_I,
                                          'signoise_{}_{}_{}{}'.format(f, nsb, str(int(nu)),
                                                                       str(int(s))): np.ravel(signoise_I)
                                          })

    np.savez(OUTDIR + "SignalToNoise_stokesIQU_5sim_400years.npz", **results_dic_IQU)
    np.savez(OUTDIR + "SignalToNoise_stokesI_5sim_400years.npz", **results_dic_I)

    sys.exit(0)
