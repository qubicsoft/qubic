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

# Qubic modules
import qubic
from qubic import QubicSkySim as qss

# Imports needed for component separation
import fgbuster as fgb

# Import useful functions from other script
import component_separation

# Define data / output directories
QUBIC_DATADIR = os.environ['QUBIC_DATADIR']

# for LOCAL EXECUTION
OUTDIR_LOCAL = "/home/simon/PycharmProjects/qubic_comp_sep/output"
DATADIR_LOCAL = "/home/simon/PycharmProjects/qubic_comp_sep/data"

# for execution on CC-IN2P3
OUTDIR_CC = "/sps/qubic/Users/sbiquard/qubic/qubic_comp_sep/output"
DATADIR_CC = "/sps/qubic/Users/sbiquard/qubic/qubic_comp_sep/data"

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


def append_to_npy(fname, data) -> None:
    """Append 1D data to a .npy binary file."""

    try:  # try to load existing data
        content = np.load(fname, allow_pickle=True)
        new_content = np.concatenate((content, data), axis=None)
        np.save(fname, new_content)
    except IOError:  # if file does not yet exist, save data directly to file
        np.save(fname, data)


def get_coverage_from_file(file_name=None):
    """
    Get coverage map from saved file.

    :param file_name: the name of the file

    :return: the array containing the coverage map
    """
    if file_name is None:
        t = pickle.load(open(DATADIR + '/coverage_dtheta_40_pointing_6000.pkl', 'rb'))
    else:
        t = pickle.load(open(DATADIR + '/{}.pkl'.format(file_name), 'rb'))
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


def print_results(band, beta_res) -> None:
    print("{:d} GHz results".format(band))
    print("---------------")
    list_fmt = "[" + ", ".join(["{:.5f}"] * len(beta_res)) + "]"
    print("results = " + list_fmt.format(*beta_res))
    print("average beta  = {:.5f}".format(np.mean(beta_res)))
    print("std deviation = {:.5f}".format(np.std(beta_res)))
    print()


def read_arguments():
    """Read parameters / command-line arguments for local (personal computer) execution.

    :return: frequency band (GHz), nb of simulations, nb of sub-bands, nb of years and sky_sim parameters.
    """
    if len(sys.argv) - 1:
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
    spatial_correlations = bool(sky_sim_param[0])
    nunu_correlations = bool(sky_sim_param[1])

    return frequency_band, nb_simu, nb_bands, nb_years, spatial_correlations, nunu_correlations


def run_simu_single_band(band: int,
                         n_sim: int,
                         n_sub: int,
                         n_years: int,
                         sc: bool,
                         nunu: bool,
                         verbose=True):
    """
    Run sky simulations (using `QubicSkySim`) and determine spectral parameters.

    :param int band: main band frequency (150 or 220 GHz)
    :param int n_sub: number of sub-bands
    :param int n_sim: number of simulations
    :param int n_years: number of years of observation for simulated maps
    :param bool sc: QubicSkySim "spatial_noise" (ie. spatial noise correlations)
    :param bool nunu: QubicSkySim "nunu_correlation" (ie. noise frequency correlations)
    :param bool verbose: print progress indications during process
    """
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
    fwhm_common = np.max(fwhms)

    # variable to store the results of parameter estimation
    beta_values = []

    for i_sim in range(n_sim):

        if verbose:
            print("entering iteration number {:d} of n_sim={:d} of loop".format(i_sim + 1, n_sim))

        # generate sky map using QubicSkySim, with random seed
        if verbose:
            print("generate sky map...")
        seed = np.random.randint(1000000)
        sky_config = {'dust': 'd0', 'cmb': seed}
        qubic_sky = qss.Qubic_sky(sky_config, dico)
        cmb_dust, cmb_dust_noiseless, cmb_dust_noise, _ = \
            qubic_sky.get_partial_sky_maps_withnoise(coverage=coverage,
                                                     Nyears=n_years,
                                                     verbose=False,
                                                     FWHMdeg=fwhm_common,
                                                     seed=seed,
                                                     spatial_noise=sc,  # noise spatial correlations
                                                     nunu_correlation=nunu,  # noise frequency correlations
                                                     )

        # modify shape to match FgBuster standard
        cmb_dust = np.transpose(cmb_dust, (0, 2, 1))
        # cmb_dust_noiseless = np.transpose(cmb_dust_noiseless, (0, 2, 1))
        # cmb_dust_noise = np.transpose(cmb_dust_noise, (0, 2, 1))
        if verbose:
            print("sky map generated with seed {}".format(seed))

        # perform component separation using FgBuster
        if verbose:
            print("perform fgbuster separation...")
        comp_separation = component_separation.CompSep(dico)

        fg_res = comp_separation.fg_buster(maps_in=cmb_dust,
                                           components=[fgb.CMB(), fgb.Dust(band, temp=20.)],
                                           map_freqs=freqs,
                                           map_fwhms_deg=fwhms,
                                           ok_pix=okpix_inside,
                                           stokes='IQU')
        beta_dust_estimate = fg_res.x[0]
        beta_values.append(beta_dust_estimate)
        if verbose:
            print("fgbuster separation performed (beta = {:.3f})".format(beta_dust_estimate))

        del cmb_dust
        del cmb_dust_noiseless
        del cmb_dust_noise

    if verbose:
        print("execution terminated, writing / printing results...\n")

    res = np.array(beta_values)

    return res


# To do:
# - add function run_simu_double_band which uses both 150 and 220 GHz bands for separation
# - ...


if __name__ == "__main__":
    # Qubic dictionaries for 150 GHz and 220 Ghz
    config_150, config_220 = 'FI-150', 'FI-220'
    d150_name = QUBIC_DATADIR + '/doc/FastSimulator/FastSimDemo_{}.dict'.format(config_150)
    d220_name = QUBIC_DATADIR + '/doc/FastSimulator/FastSimDemo_{}.dict'.format(config_220)
    d150, d220 = qubic.qubicdict.qubicDict(), qubic.qubicdict.qubicDict()
    d150.read_from_file(d150_name)
    d220.read_from_file(d220_name)
    qubic_dicts = {150: d150, 220: d220}

    # read arguments and run simulation
    arguments = read_arguments()
    f_band, _, nb_sub, _, _, _ = arguments
    beta_results = run_simu_single_band(*arguments)
    if LOC == 0:
        print_results(f_band, beta_results)

    # write the results to file
    out_fmt = OUTDIR + "/FgBuster_SingleBand{}_Nsub{}.npy"
    append_to_npy(out_fmt.format(f_band, nb_sub), beta_results)

    sys.exit(0)
