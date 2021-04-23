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
OUTPUT_DIR = "/home/simon/Documents/qubic/component_separation/output"
DATADIR = "/home/simon/Documents/qubic/component_separation/data"


def get_sub_freqs(band: int):
    """
    Give the frequency sub-bands and corresponding angular resolutions around f = 150 or 220 GHz.
    :param band: main band frequency (150 or 220 GHz)
    :return: Tuple (freqs, fwhms) containing the list of the central frequencies and the resolutions (in degrees).
    """
    dico = qubic_dicts[band]
    n = int(dico['nf_sub'])
    _, _, frequencies, _, _, _ = qubic.compute_freq(band, n)
    if n == 1:
        resolutions = []
    elif n == 2:
        resolutions = []
    elif n == 3:
        resolutions = [0.42999269, 0.39543908, 0.36366215]
    elif n == 4:
        resolutions = [0.43468571, 0.40821527, 0.38335676, 0.36001202]
    elif n == 5:
        resolutions = [0.43750306, 0.41605639, 0.39566106, 0.37626551, 0.35782075]
    else:
        raise ValueError("Number of sub-bands not supported")

    return frequencies, resolutions


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


def run_simu_beta_dust(band: int, n_sub: int, n_sim: int, verbose=False):
    """
    Run sky simulations (using QubicSkySim) and determine spectral parameters.
    :param int band: main band frequency (150 or 220 GHz)
    :param int n_sub: number of sub-bands to simulate
    :param int n_sim: number of simulations (a different seed is given to QubicSkySim each time)
    :param bool verbose: print progress indications during process
    """
    # get dictionary for given frequency
    dico = qubic_dicts[band]

    # adjust number of sub-bands for sky simulation and reconstruction
    dico['nf_sub'] = n_sub
    dico['nf_recon'] = n_sub

    # center = qubic.equ2gal(dico['RA_center'], dico['DEC_center'])  # center in galactic spherical coordinates
    # npix = 12 * dico['nside'] ** 2

    # get coverage map and define pixels to use
    coverage = get_coverage_from_file()
    okpix_inside = (coverage > (0.5 * np.max(coverage)))
    # coverage_qubic = np.loadtxt(DATADIR + '/mask_fgbuster')
    # okpix_qubic = coverage_qubic != 0
    if verbose:
        print("coverage from file ok")

    # get sub-bands characteristics
    freqs, fwhms = get_sub_freqs(band)
    if verbose:
        print("sub-bands frequencies and resolutions ok")
        print("   --> {} sub-bands".format(n_sub))
        list_fmt = "[" + ", ".join(["{:.6f}"] * len(freqs)) + "]"
        print("   --> centered around frequencies " + list_fmt.format(*freqs))

    # define common resolution for input maps (lowest one)
    fwhm_common = np.max(fwhms)

    # QubicSkySim options
    integration_into_band = True
    nunu_correlation = True
    spatial_noise = True
    n_years = 400

    # variable to store the results of parameter estimation
    beta_values = []

    for i_sim in range(n_sim):

        betas = []
        if verbose:
            print("entering iteration number {:d} / n_sim={:d} of loop".format(i_sim + 1, n_sim))

        # generate sky map using QubicSkySim, with random seed
        if verbose:
            print("generate sky map...")
        seed = round(np.random.rand() * 100000)
        sky_config = {'dust': 'd0', 'cmb': seed}
        qubic_sky = qss.Qubic_sky(sky_config, dico)
        cmb_dust, cmb_dust_noiseless, cmb_dust_noise, _ = \
            qubic_sky.get_partial_sky_maps_withnoise(coverage=coverage,
                                                     Nyears=n_years,
                                                     verbose=False,
                                                     FWHMdeg=fwhm_common,
                                                     seed=seed,
                                                     spatial_noise=spatial_noise,
                                                     nunu_correlation=nunu_correlation,
                                                     integrate_into_band=integration_into_band)

        # modify shape to match FgBuster standard
        cmb_dust = np.transpose(cmb_dust, (0, 2, 1))
        # cmb_dust_noiseless = np.transpose(cmb_dust_noiseless, (0, 2, 1))
        # cmb_dust_noise = np.transpose(cmb_dust_noise, (0, 2, 1))
        if verbose:
            print("sky map generated with seed {}".format(seed))

        # perform component separation using FgBuster
        print("perform fgbuster separation...")
        comp_separation = component_separation.CompSep(dico)
        for freq in freqs:
            fg_res = comp_separation.fg_buster(maps_in=cmb_dust,
                                               components=[fgb.CMB(), fgb.Dust(freq, temp=20.)],
                                               map_freqs=freqs,
                                               map_fwhms_deg=fwhms,
                                               # target=fwhm_common,
                                               ok_pix=okpix_inside,
                                               stokes='IQU')
            beta_dust_estimated = fg_res.x[0]
            betas.append(beta_dust_estimated)
            if verbose:
                print("At sub-frequency {:.3f} GHz dust index estimate is {:.3f}".format(freq, beta_dust_estimated))

            # array_est_fg_3bands[m, 0, f, :, okpix_inside] = fg_res.s[0, :, :].T
            # array_est_fg_3bands[m, 1, f, :, okpix_inside] = fg_res.s[1, :, :].T

        beta_values.append(np.mean(betas))

        del cmb_dust
        del cmb_dust_noiseless
        del cmb_dust_noise

    return beta_values


if __name__ == "__main__":
    # Qubic dictionaries for 150 GHz and 220 Ghz
    config_150, config_220 = 'FI-150', 'FI-220'
    d150_name = QUBIC_DATADIR + '/doc/FastSimulator/FastSimDemo_{}.dict'.format(config_150)
    d220_name = QUBIC_DATADIR + '/doc/FastSimulator/FastSimDemo_{}.dict'.format(config_220)
    d150, d220 = qubic.qubicdict.qubicDict(), qubic.qubicdict.qubicDict()
    d150.read_from_file(d150_name)
    d220.read_from_file(d220_name)
    qubic_dicts = {150: d150, 220: d220}

    # read command-line arguments
    nb_simu = int(sys.argv[1])
    nb_bands_150 = int(sys.argv[2])
    nb_bands_220 = int(sys.argv[3])

    beta150 = run_simu_beta_dust(150, nb_bands_150, nb_simu, verbose=False)
    # beta220 = run_simu_beta_dust(220, nb_bands_220, nb_simu)

    print(beta150)
    # print(beta220)

    sys.exit(0)
