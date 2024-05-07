from __future__ import division
import os
import healpy as hp
import random as rd
import string
import pysm3 as pysm
import pysm3.units as u
from pysm3 import utils
from pylab import *
from scipy.optimize import curve_fit
import pickle

import qubic
from qubic import camb_interface as qc
from qubic import fibtools as ft
from qubic.utils import progress_bar
from qubic import analytical_forecast_lib as ana

__all__ = ['sky', 'Qubic_sky']


# Return the correlation matrix from the covariance matrix.
def cov2corr(mat):
    sh = np.shape(mat)
    if sh[0] == 1:
        return mat
    outmat = np.zeros_like(mat)
    for i in range(sh[0]):
        for j in range(sh[1]):
            outmat[i, j] = mat[i, j] / np.sqrt(mat[i, i] * mat[j, j])
    return outmat


def corr2cov(mat, diagvals):
    sh = np.shape(mat)
    if sh[0] == 1:
        return mat
    outmat = np.zeros_like(mat)
    for i in range(sh[0]):
        for j in range(sh[1]):
            outmat[i, j] = mat[i, j] * np.sqrt(diagvals[i] * diagvals[j])
    return outmat


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
        self.npix = 12 * self.nside ** 2
        self.dictionary = d
        self.Nfin = int(self.dictionary['nf_sub'])
        self.Nfout = int(self.dictionary['nf_recon'])
        self.filter_nu = int(self.dictionary['filter_nu'] / 1e9) # frequency of the detector, 150GHz or 220 GHz
        self.filter_relative_bandwidth = self.dictionary['filter_relative_bandwidth']
        self.instrument = instrument
        self.output_directory = out_dir
        self.output_prefix = out_prefix
        self.input_cmb_maps = None
        self.input_cmb_spectra = None
        if lmax is None:
            self.lmax = 3 * d['nside']
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
                        mycls = qc.Dl2Cl_without_monopole(ell, totDL)
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
                    # The assumed cosmology is the default one given in the get_CAMB_Dl() function
                    # from camb_interface library.
                    if keyword is not None:
                        np.random.seed(keyword)
                    ell, totDL, unlensedCL = qc.get_camb_Dl(lmax=self.lmax)
                    mycls = qc.Dl2Cl_without_monopole(ell, totDL)
                    mymaps = hp.synfast(mycls.T, self.nside, verbose=False, new=True)
                    self.input_cmb_maps = mymaps
                    self.input_cmb_spectra = totDL

                # Write a temporary file with the maps so the PySM can read them
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
        Create as many skies as the number of input frequencies.
        Instrumental effects are not considered.
        Return a vector of shape (number_of_input_subfrequencies, npix, 3), 3 for I, Q and U.
        """
        # qubic.compute_freq uniformely ditributes in logarithmic scale the sub-bands.
        # nus_edge is the list of edges between sub-bands (see polyacquisition.py for more details).
        _, nus_edge, _, _, _, _ = qubic.compute_freq(self.filter_nu, self.Nfin, self.filter_relative_bandwidth)

        sky = np.zeros((self.Nfin, self.npix, 3))

        for i in range(self.Nfin):
            nfreqinteg = 5
            freqs = np.linspace(nus_edge[i], nus_edge[i + 1], nfreqinteg)
            weights = np.ones(nfreqinteg)
            sky[i, :, :] = (
                    self.sky.get_emission(freqs * u.GHz, weights) * utils.bandpass_unit_conversion(freqs * u.GHz, weights, u.uK_CMB)).T

        return sky

def random_string(nchars):
    lst = [rd.choice(string.ascii_letters + string.digits) for n in range(nchars)]
    str = "".join(lst)
    return (str)



class Qubic_sky(sky):
    """
    Define a sky object as seen by Qubic
    """

    def __init__(self, skyconfig, d, output_directory="./", output_prefix="qubic_sky"):
        self.Nfin = int(d['nf_sub'])
        self.Nfout = int(d['nf_recon'])
        self.filter_relative_bandwidth = d['filter_relative_bandwidth']
        self.filter_nu = int(d['filter_nu'] / 1e9) # 150GHz or 220 GHz for QUBIC
        # qubic.compute_freq uniformely ditributes in logarithmic scale the sub-bands.
        # nus_edge_in is the list of edges between sub-bands,
        # central_nus is the list of the center frequency of each sub-band (see polyacquisition.py for more details).
        _, nus_edge_in, central_nus, _, _, _ = qubic.compute_freq(self.filter_nu,
                                                                       self.Nfin,
                                                                       self.filter_relative_bandwidth)
        self.qubic_central_nus = central_nus
        # THESE LINES HAVE TO BE CONFIRMED/IMPROVED in future since fwhm = lambda / (P Delta_x)
        # is an approximation for the resolution
        if d['config'] == 'FI':
            self.fi2td = 1
        elif d['config'] == 'TD':
            P_FI = 22  # horns in the largest baseline in the FI
            P_TD = 8  # horns in the largest baseline in the TD
            self.fi2td = (P_FI - 1) / (P_TD - 1)
        #
        self.qubic_resolution_nus = d['synthbeam_peak150_fwhm'] * 150 / self.qubic_central_nus * self.fi2td
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
        This returns full sky maps at each subfrequency convolved by the beam of the instrument at
        each frequency or with another beam if FWHMdeg is provided.
        when FWHMdeg is 0, the maps are not convolved.

        Parameters
        ----------
        FWHMdeg: float
        verbose: bool

        Returns
        -------
        Vector of shape (number_of_input_subfrequencies, npix, 3), 3 for I, Q and U.
        
        """

        # First get the full sky maps
        fullmaps = self.get_simple_sky_map()

        # Convolve the maps
        fwhms, fullmaps = self.smoothing(fullmaps, FWHMdeg, self.Nfin, self.qubic_central_nus, verbose=verbose)
        self.instrument['beams'] = fwhms

        return fullmaps

    def smoothing(self, maps, FWHMdeg, Nf, central_nus, verbose=False):
        """Convolve the maps to the FWHM at each sub-frequency or to a common beam if FWHMdeg is given."""
        fwhms = np.zeros(Nf)
        if FWHMdeg is not None:
            fwhms += FWHMdeg
        else:
            fwhms = self.dictionary['synthbeam_peak150_fwhm'] * 150. / central_nus * self.fi2td
        for i in range(Nf):
            if fwhms[i] != 0:
                maps[i, :, :] = hp.sphtfunc.smoothing(maps[i, :, :].T, fwhm=np.deg2rad(fwhms[i]),
                                                      verbose=verbose).T
        return fwhms, maps

    
    def get_partial_sky_maps_withnoise(self, coverage=None, version_FastSim='02', sigma_sec=None,
                                       Nyears=3., FWHMdeg=None, seed=None,
                                       noise_profile=True, spatial_noise=False, nunu_correlation=True,
                                       noise_only=False, integrate_into_band=True,
                                       verbose=False, noise_covcut=0.1):
        """
        This returns maps in the same way as with get_simple_sky_map but cut according to the coverage
        and with noise added according to this coverage and the RMS in muK.sqrt(sec) given by sigma_sec
        The default integration time is 3 years but can be modified with optional variable Nyears
        Note that the maps are convolved with the instrument beam by default, or with FWHMdeg (can be an array)
        if provided.
        If seed is provided, it will be used for the noise realization. If not, it will be a new realization at
        each call.
        The optional effective_variance_invcov keyword is a modification law to be applied to the coverage in order to obtain
        more realistic noise profile. It is a law for effective RMS as a function of inverse coverage and is 2D array
        with the first one being (nx samples) inverse coverage and the second being the corresponding effective variance to be
        used through interpolation when generating the noise.
        
        Parameters
        ----------
        coverage: array
            Coverage map of the sky.
            By default, we load a coverage centered on the galactic center with 10000 pointings.
        version_FastSim: str
            Version of the FastSimulator files: 01, 02, 03... For now, only 01 and 02 exists.
        sigma_sec: float
        Nyears: float
            Integration time for observation to scale the noise, by default it is 3. Should be 4 for the FastSimulator version 01.
        FWHMdeg:
        seed:
        noise_profile: bool
            If True, a modification law is applied to the coverage in order to obtain more realistic noise profile.
            It is a law for effective RMS as a function of inverse coverage
        spatial_noise: bool
            If True, spatial noise correlations are added. False by default. Only exist for version of FastSimulator 01.
        nunu_correlation: bool
            If True, correlations between frequency sub-bands are added. True by default.
        noise_only: bool
            If True, only returns the noise maps and the coverage (without the sky signal).
        integrate_into_band: bool
            If True, averaging input sub-band maps into reconstruction sub-bands. True by default.
        verbose: bool

        Returns
        -------
        maps + noisemaps, maps, noisemaps, coverage

        """

        ### Check version_FastSim is 01 or 02
        if version_FastSim != '02' and version_FastSim != '01':
            raise NameError("Only version_FastSim 01 and 02 exist")

        ### Input bands. qubic.compute_freq uniformely ditributes in logarithmic scale the sub-bands (see polyacquisition.py for more details).
        Nfreq_edges, nus_edge, nus, deltas, Delta, Nbbands = qubic.compute_freq(self.filter_nu,
                                                                                self.Nfin,
                                                                                self.filter_relative_bandwidth)
        ### Output bands
        # Check Nfout is between 1 and 8.
        if self.Nfout < 1 or self.Nfout > 8:
            raise NameError("Nfout should be contained between 1 and 8 for FastSimulation.")
        if self.dictionary['config'] == 'TD' and self.Nfout > 5:
            raise NameError("Nfout should be contained between 1 and 5 for configuration TD.")
        Nfreq_edges_out, nus_edge_out, nus_out, deltas_out, Delta_out, Nbbands_out = qubic.compute_freq(self.filter_nu,
                                                                                                        self.Nfout,
                                                                                                        self.filter_relative_bandwidth)

        # First get the convolved maps
        if noise_only is False:
            maps = np.zeros((self.Nfout, self.npix, 3))
            if integrate_into_band:
                if verbose:
                    print('Convolving each input frequency map')
                maps_all = self.get_fullsky_convolved_maps(FWHMdeg=FWHMdeg, verbose=verbose)

                # Now averaging maps into reconstruction sub-bands maps
                if verbose:
                    print('Averaging input maps from input sub-bands into reconstruction sub-bands:')
                for i in range(self.Nfout):
                    print('doing band {} {} {}'.format(i, nus_edge_out[i], nus_edge_out[i + 1]))
                    inband = (nus > nus_edge_out[i]) & (nus < nus_edge_out[i + 1])
                    maps[i, :, :] = np.mean(maps_all[inband, :, :], axis=0)
            else:
                for i in range(self.Nfout):
                    freq = nus_out[i]
                    maps[i, :, :] = (self.sky.get_emission(freq * u.GHz)
                                     * utils.bandpass_unit_conversion(freq * u.GHz,
                                                                      weights=None,
                                                                      output_unit=u.uK_CMB)).T
                _, maps = self.smoothing(maps, FWHMdeg, self.Nfout, nus_out, verbose=verbose)

        ##############################################################################################################
        # Restore data for FastSimulation ############################################################################
        ##############################################################################################################
        #### Directory for fast simulations
        dir_fast = os.path.join(os.path.dirname(__file__), 'data', f'FastSimulator_version{version_FastSim}')

        with open(dir_fast + os.sep + 'DataFastSimulator_{}{}_nfsub_{}.pkl'.format(self.dictionary['config'],
                                                                          str(self.filter_nu),
                                                                          self.Nfout),
                  "rb") as file:
            DataFastSim = pickle.load(file)
            print(file)
        
        # Read Coverage map
        if coverage is None:
            DataFastSimCoverage = pickle.load(open(dir_fast + os.sep + 'DataFastSimulator_{}{}_coverage.pkl'.format(
                                                       self.dictionary['config'],
                                                       str(self.filter_nu)), "rb"))
            coverage = DataFastSimCoverage['coverage']

        # Read noise normalization
        if sigma_sec is None:
            if version_FastSim == '01':
                #### Integration time assumed in FastSim files
                fastsimfile_effective_duration = 2.
                #### Beware ! Initial End-To-End simulations that produced the first FastSimulator were done with
                #### Effective_duration = 4 years and this is the meaning of signoise
                #### New files were done with 2 years and as result the signoise needs to be multiplied by sqrt(effective_duration/4)
                sigma_sec = DataFastSim['signoise'] * np.sqrt(fastsimfile_effective_duration / 4.)
            else:
                # Integration time of 3 years, this is the meaning of signoise
                sigma_sec = DataFastSim['signoise']

        # Read Noise Profile
        if noise_profile is True:
            effective_variance_invcov = DataFastSim['effective_variance_invcov']
        else:
            effective_variance_invcov = None

        # Read Spatial noise correlation if version of FastSimulator is 01
        if version_FastSim == '01':
            if spatial_noise is True:
                clnoise = DataFastSim['clnoise']
            else:
                clnoise = None
        elif spatial_noise is True:
            raise NameError("No spatial correlation for FastSimulator version 02")
        else:
            clnoise = None

        # Read Noise Profile
        if nunu_correlation is True:
            covI = DataFastSim['CovI']
            covQ = DataFastSim['CovQ']
            covU = DataFastSim['CovU']
            sub_bands_cov = [covI, covQ, covU]
        else:
            sub_bands_cov = None
        
        ##############################################################################################################
        # Now pure noise maps
        if verbose:
            print('Making noise realizations')
        noisemaps = self.create_noise_maps(sigma_sec, coverage, nsub=self.Nfout,
                                           Nyears=Nyears, verbose=verbose, seed=seed,
                                           effective_variance_invcov=effective_variance_invcov,
                                           clnoise=clnoise,
                                           sub_bands_cov=sub_bands_cov,
                                           covcut=noise_covcut)
        if self.Nfout == 1:
            noisemaps = np.reshape(noisemaps, (1, len(coverage), 3))
        seenpix = noisemaps[0, :, 0] != 0
        coverage[~seenpix] = 0

        if noise_only:
            return noisemaps, coverage
        else:
            maps[:, ~seenpix, :] = 0
            return maps + noisemaps, maps, noisemaps, coverage

    def create_noise_maps(self, sigma_sec, coverage, covcut=0.1, nsub=1,
                          Nyears=3, verbose=False, seed=None,
                          effective_variance_invcov=None,
                          clnoise=None,
                          sub_bands_cov=None):

        """
        This returns a realization of noise maps for I, Q and U with no correlation between them, according to a
        noise RMS map built according to the coverage specified as an attribute to the class
        The optional effective_variance_invcov keyword is a modification law to be applied to the coverage in order to obtain
        more realistic noise profile. It is a law for effective RMS as a function of inverse coverage and is 2D array
        with the first one being (nx samples) inverse coverage and the second being the corresponding effective variance to be
        used through interpolation when generating the noise.
        The clnoise option only exists for FatSimulator version 01. The clnoise option is used to apply a convolution to the noise 
        to obtain spatially correlated noise. This cl should be calculated from the c(theta) of the noise that can be measured using 
        the function ctheta_parts() below. The transformation, of this C(theta) into Cl has to be done using wrappers on camb 
        function found in camb_interface.py of the QUBIC software: the functions to back and forth from ctheta to cl are: cl_2_ctheta 
        and ctheta_2_cell. The simulation of the noise itself calls a function of camb_interface called simulate_correlated_map().
        The sub_bands_cov option is the covariance of I, Q and U to have correlations between frequency.
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
        seenpix = (coverage / np.max(coverage)) > covcut
        npix = seenpix.sum()

        # Sigma_sec for each Stokes: by default they are the same unless there is non trivial covariance
        if sub_bands_cov is None:
            fact_I = np.ones(nsub)
            fact_Q = np.ones(nsub)
            fact_U = np.ones(nsub)
        else:
            fact_I = 1. / np.sqrt(np.diag(sub_bands_cov[0] / sub_bands_cov[0][0, 0]))
            fact_Q = 1. / np.sqrt(np.diag(sub_bands_cov[1] / sub_bands_cov[1][0, 0]))
            fact_U = 1. / np.sqrt(np.diag(sub_bands_cov[2] / sub_bands_cov[2][0, 0]))

        all_sigma_sec_I = fact_I * sigma_sec
        all_sigma_sec_Q = fact_Q * sigma_sec
        all_sigma_sec_U = fact_U * sigma_sec

        thnoiseI = np.zeros((nsub, len(seenpix)))
        thnoiseQ = np.zeros((nsub, len(seenpix)))
        thnoiseU = np.zeros((nsub, len(seenpix)))
        for isub in range(nsub):
            # The theoretical noise in I for the coverage
            ideal_noise_I = self.theoretical_noise_maps(all_sigma_sec_I[isub], coverage, Nyears=Nyears, verbose=verbose)
            ideal_noise_Q = self.theoretical_noise_maps(all_sigma_sec_Q[isub], coverage, Nyears=Nyears, verbose=verbose)
            ideal_noise_U = self.theoretical_noise_maps(all_sigma_sec_U[isub], coverage, Nyears=Nyears, verbose=verbose)
            sh = np.shape(ideal_noise_I)
            if effective_variance_invcov is None:
                thnoiseI[isub, :] = ideal_noise_I
                thnoiseQ[isub, :] = ideal_noise_Q
                thnoiseU[isub, :] = ideal_noise_U
            else:
                my_effective_variance_invcov = effective_variance_invcov[isub].copy()
                sh = np.shape(my_effective_variance_invcov)
                if sh[0] == 2:
                    ### We have the same correction for I, Q and U
                    correction = np.interp(np.max(coverage[seenpix]) / coverage[seenpix],
                                           my_effective_variance_invcov[0, :], my_effective_variance_invcov[1, :])
                    thnoiseI[isub, seenpix] = ideal_noise_I[seenpix] * np.sqrt(correction)
                    thnoiseQ[isub, seenpix] = ideal_noise_Q[seenpix] * np.sqrt(correction)
                    thnoiseU[isub, seenpix] = ideal_noise_U[seenpix] * np.sqrt(correction)
                else:
                    ### We have distinct correction for I and QU
                    correctionI = np.interp(np.max(coverage[seenpix]) / coverage[seenpix],
                                            my_effective_variance_invcov[0, :], my_effective_variance_invcov[1, :])
                    correctionQU = np.interp(np.max(coverage[seenpix]) / coverage[seenpix],
                                             my_effective_variance_invcov[0, :], my_effective_variance_invcov[2, :])
                    thnoiseI[isub, seenpix] = ideal_noise_I[seenpix] * np.sqrt(correctionI)
                    thnoiseQ[isub, seenpix] = ideal_noise_Q[seenpix] * np.sqrt(correctionQU)
                    thnoiseU[isub, seenpix] = ideal_noise_U[seenpix] * np.sqrt(correctionQU)

        noise_maps = np.zeros((nsub, len(coverage), 3))
        if seed is not None:
            np.random.seed(seed)

        ### Simulate variance 1 maps for each sub-band independently
        for isub in range(nsub):
            if clnoise is None:
                ### With no spatial correlation
                if verbose:
                    if isub == 0:
                        print('Simulating noise maps with no spatial correlation')
                IrndFull = np.random.randn(self.npix)
                QrndFull = np.random.randn(self.npix) * np.sqrt(2)
                UrndFull = np.random.randn(self.npix) * np.sqrt(2)
            else:
                ### With spatial correlations given by cl which is the Legendre transform of the targetted C(theta)
                ### NB: here one should not expect the variance of the obtained maps to make complete sense because
                ### of ell space truncation. They have however the correct Cl spectrum in the relevant ell range 
                ### (up to lmax = 2*nside)
                if verbose:
                    if isub == 0:
                        print('Simulating noise maps with spatial correlation')
                IrndFull = qc.simulate_correlated_map(self.nside, 1., clin=clnoise, verbose=False)
                QrndFull = qc.simulate_correlated_map(self.nside, 1., clin=clnoise, verbose=False) * np.sqrt(2)
                UrndFull = qc.simulate_correlated_map(self.nside, 1., clin=clnoise, verbose=False) * np.sqrt(2)
            ### put them into the whole sub-bandss array
            noise_maps[isub, seenpix, 0] = IrndFull[seenpix]
            noise_maps[isub, seenpix, 1] = QrndFull[seenpix]
            noise_maps[isub, seenpix, 2] = UrndFull[seenpix]

        ### If there is non-diagonal noise covariance between sub-bands (spectro-imaging case)
        if nsub > 1:
            if sub_bands_cov is not None:
                if verbose:
                    print('Simulating noise maps sub-bands covariance')
                ### We get the eigenvalues and eigenvectors of the sub-band covariance matrix divided by its 0,0 element
                ### The reason for this is that the overall noise is given by the input parameter sigma_sec which we do not
                ### want to override

                wI, vI = np.linalg.eig(sub_bands_cov[0] / sub_bands_cov[0][0, 0])
                wQ, vQ = np.linalg.eig(sub_bands_cov[1] / sub_bands_cov[1][0, 0])
                wU, vU = np.linalg.eig(sub_bands_cov[2] / sub_bands_cov[2][0, 0])

                ### Multiply the maps by the sqrt(eigenvalues)
                for isub in range(nsub):
                    noise_maps[isub, seenpix, 0] *= np.sqrt(wI[isub])
                    noise_maps[isub, seenpix, 1] *= np.sqrt(wQ[isub])
                    noise_maps[isub, seenpix, 2] *= np.sqrt(wU[isub])
                ### Apply the rotation to each Stokes Parameter separately

                noise_maps[:, seenpix, 0] = np.dot(vI, noise_maps[:, seenpix, 0])
                noise_maps[:, seenpix, 1] = np.dot(vQ, noise_maps[:, seenpix, 1])
                noise_maps[:, seenpix, 2] = np.dot(vU, noise_maps[:, seenpix, 2])

        # Now normalize the maps with the coverage behaviour and the sqrt(2) for Q and U
        noise_maps[:, seenpix, 0] *= thnoiseI[:, seenpix]
        noise_maps[:, seenpix, 1] *= thnoiseQ[:, seenpix]
        noise_maps[:, seenpix, 2] *= thnoiseU[:, seenpix]

        if nsub == 1:
            return noise_maps[0, :, :]
        else:
            return noise_maps

    def theoretical_noise_maps(self, sigma_sec, coverage, Nyears=3, verbose=False):
        """
        This returns a map of the RMS noise (not an actual realization, just the expected RMS - No covariance)

        Parameters
        ----------
        sigma_sec: float
            Noise level.
        coverage: array
            The coverage map.
        Nyears: int
        verbose: bool

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



class FastNoise(ana.AnalyticalForecast):
    '''
    Class to compute noise maps for the QUBIC instrument using the formula from analytical_forecst_lib.py.

    Arguments : - nus : array(float), frequency bands. For Qubic: np.array([150, 220])
                - nside : float
                - NEPdet : array(float), as same lenght than nus. Noise Equivalent Power for detectors.
                - NEPpho : array(float), as same lenght than nus. Noise Equivalent Power for photons.
                - mixing_matrix : array(float), define the mixing matrix between the different components. 
                    CMB : np.array([[1],[1]]) / CMB + Dust : np.array([[1, 1],[1, 2.92]])
                - correlation : Bool, corelation between frquency bands (To be implemented)
                - fwhm : array(float), as same lenght than nus (To be checked)
    '''
    
    def __init__(self, nus, nside, NEPdet, NEPpho, mixing_matrix = np.array([[1, 1],[1, 2.92]]), correlation = False, fwhm=np.array([0, 0]), Nyrs=3, Nh=400, fsky=0.0182, instr='DB'):
        
        # Check that NEPdet and NEPpho have the same length as nus.
        if len(nus) != len(NEPdet) or len(nus) != len(NEPpho):
            raise NameError("NEPdet and NEPpho should have the same length as nus")

        ana.AnalyticalForecast.__init__(self, nus, NEPdet, NEPpho, fwhm=fwhm, Nyrs=Nyrs, Nh=Nh, fsky=fsky, nside=nside, instr=instr)

        self.nside = nside
        
        # We compute NET from NEP using analytical_forecast_lib.py
        self.NETs = np.zeros(len(self.nus))
        for i in range(len(self.nus)):
            self.NETs[i] = ana.NoiseEquivalentTemperature(self.NEPs[i], self.nus[i]).NETs

        # We compute depths for frequency maps from NET using analytical_forecast_lib.py
        self.depths_FMM = self._get_effective_depths(self.NETs)
        
        # We define the mixing matrix
        self.A = mixing_matrix
        self.ncomp = mixing_matrix.shape[0]
        
    def get_noise_from_depths(self, depths, unit='uK_CMB'):
        '''
        Function to generate sky maps from depths values

        return : - array(len(depths), nstokes, npix)
        '''

        # Number of wanted maps
        n = np.shape(depths)[0]

        # Number of sky pixels
        n_pix = hp.nside2npix(self.nside)

        # Normal distribution
        res = np.random.normal(size=(n_pix, 3, n))

        # We take into account the fact that I maps have 2 times more photon than Q and U
        res[:, 0, :] /= np.sqrt(2)

        # Apply the noise level computed to the maps
        depths *= u.arcmin * u.uK_CMB
        depths = depths.to(getattr(u, unit) * u.arcmin,
            equivalencies=u.cmb_equivalencies(self.nus * u.GHz))
        res *= depths.value / hp.nside2resol(self.nside, True) # depths / pixel size in radian
        return res.T   
    
    def get_noise_realisation_FMM(self, unit='uK_CMB'):
        '''
        Function to generate noise frequency maps
        '''
    
        return self.get_noise_from_depths(self.depths_FMM)
    
    def get_noise_realisation_CMM(self, unit='uK_CMB'):
        '''
        Function to generate noise components maps
        '''
        
        # Compute the noise power spectra
        bl = np.array([hp.gauss_beam(b, lmax=2*self.nside) for b in self.fwhm])
        nl = (bl / (self.depths_FMM)[:, np.newaxis])**2

        # Noise mixing into component maps
        AtNA = np.einsum('fi, fl, fj -> lij', self.A, nl, self.A)

        # Compute noise for the maps
        depths_CMM = np.zeros(self.ncomp)
        for i in range(self.ncomp):
            depths_CMM[i] = np.sqrt(np.linalg.pinv(AtNA))[0][i, i] * hp.nside2resol(self.nside, True)

        return self.get_noise_from_depths(depths_CMM)



def get_noise_invcov_profile(maps, coverage, covcut=0.1, nbins=100, fit=True, label='',
                             norm=False, allstokes=False, fitlim=None, doplot=False, QUsep=True):
    '''
    maps: input maps
    nbins: int. Number of bins for the profile.
    fit: bool. Whether to perform fitting.
    norm: bool. Whether to normalize the data. Normalized to 1 at 1.
    allstockes: bool. If True, process all Stokes parameters.
    fitlim: Limits for fitting.
    doplot: bool. Whether to plot the result. Plots the RMS ratio w.r.t. linear scaling for average IQU or for I and QU (depending on QUsep). 
        If fit is True, it also plots the fitted models.
    QUsep: bool. If True, consider Q and U separately from I.

    The fitting model is:  a+bx+ce^(-d(x-e))+fx^2+gx^3.

    Returns xx, myY, effective_variance_invcov
    xx: xx**2 is the inverse coverage
    myY: RMS ratio w.r.t linear scaling for averaged IQU.
    effective_variance_invcov:  fitted models for I and QU (if QUsep is True)
    '''

    seenpix = coverage > (covcut * np.max(coverage))
    covnorm = coverage / np.max(coverage)

    xx, yyI, dx, dyI, _ = ft.profile(np.sqrt(1. / covnorm[seenpix]), maps[seenpix, 0], nbins=nbins, plot=False)
    xx, yyQ, dx, dyQ, _ = ft.profile(np.sqrt(1. / covnorm[seenpix]), maps[seenpix, 1], nbins=nbins, plot=False)
    xx, yyU, dx, dyU, _ = ft.profile(np.sqrt(1. / covnorm[seenpix]), maps[seenpix, 2], nbins=nbins, plot=False)
    avg = np.sqrt((dyI ** 2 + dyQ ** 2 / 2 + dyU ** 2 / 2) / 3)
    avgQU = np.sqrt((dyQ ** 2 / 2 + dyU ** 2 / 2) / 2)
    if norm:
        fact = xx[0] / avg[0]
    else:
        fact = 1.
    myY = (avg / xx) * fact
    myYI = (dyI / xx) * fact
    myYQU = (avgQU / xx) * fact

    if doplot:
        if QUsep is False:
            p = plot(xx ** 2, myY, 'o', label=label + ' IQU')
            if allstokes:
                plot(xx ** 2, myYI, label=label + ' I', alpha=0.3)
                plot(xx ** 2, myYQU, label=label + ' Average Q, U /sqrt(2)', alpha=0.3)
        else:
            pi = plot(xx ** 2, myYI, 'o', label=label + ' I')
            pqu = plot(xx ** 2, myYQU, 'o', label=label + ' QU / sqrt(2)')

    if fit:
        ok = isfinite(myY)
        if fitlim is not None:
            print('Clipping fit from {} to {}'.format(fitlim[0], fitlim[1]))
            ok = ok & (xx >= fitlim[0]) & (xx <= fitlim[1])
        if QUsep is False:
            mymodel = lambda x, a, b, c, d, e: (a + b * x + c * np.exp(-d * (x - e)))  # /(a+b+c*np.exp(-d*(1-e)))
            myfit = curve_fit(mymodel, xx[ok] ** 2, myY[ok], p0=[np.min(myY[ok]), 0.4, 0, 2, 1.5], maxfev=100000,
                              ftol=1e-7)
        else:
            mymodel = lambda x, a, b, c, d, e, f, g: (
                    a + b * x + f * x ** 2 + g * x ** 3 + c * np.exp(-d * (x - e)))  # /(a+b+c*np.exp(-d*(1-e)))
            myfitI = curve_fit(mymodel, xx[ok] ** 2, myYI[ok], p0=[np.min(myY[ok]), 0.4, 0, 2, 1.5, 0., 0.],
                               maxfev=100000, ftol=1e-7)
            myfitQU = curve_fit(mymodel, xx[ok] ** 2, myYQU[ok], p0=[np.min(myY[ok]), 0.4, 0, 2, 1.5, 0., 0.],
                                maxfev=100000, ftol=1e-7)
        if doplot:
            if QUsep is False:
                plot(xx ** 2, mymodel(xx ** 2, *myfit[0]), label=label + ' Fit', color=p[0].get_color())
            else:
                plot(xx ** 2, mymodel(xx ** 2, *myfitI[0]), label=label + ' Fit I', color=pi[0].get_color())
                plot(xx ** 2, mymodel(xx ** 2, *myfitQU[0]), label=label + ' Fit QU / sqrt(2)',
                     color=pqu[0].get_color())

            # print(myfit[0])
        # Interpolation of the fit from invcov = 1 to 15
        invcov_samples = np.linspace(1, 15, 1000)
        if QUsep is False:
            eff_v = mymodel(invcov_samples, *myfit[0]) ** 2
            # Avoid extrapolation problem for pixels before the first bin or after the last one.
            eff_v[invcov_samples < xx[0] ** 2] = mymodel(xx[0] ** 2, *myfit[0]) ** 2
            eff_v[invcov_samples > xx[-1] ** 2] = mymodel(xx[-1] ** 2, *myfit[0]) ** 2

            effective_variance_invcov = np.array([invcov_samples, eff_v])
        else:
            eff_vI = mymodel(invcov_samples, *myfitI[0]) ** 2
            eff_vQU = mymodel(invcov_samples, *myfitQU[0]) ** 2
            # Avoid extrapolation problem for pixels before the first bin or after the last one.
            eff_vI[invcov_samples < xx[0] ** 2] = mymodel(xx[0] ** 2, *myfitI[0]) ** 2
            eff_vQU[invcov_samples > xx[-1] ** 2] = mymodel(xx[-1] ** 2, *myfitQU[0]) ** 2

            effective_variance_invcov = np.array([invcov_samples, eff_vI, eff_vQU])

    if doplot:
        xlabel('1./cov normed')
        if norm:
            add_yl = ' (Normalized to 1 at 1)'
        else:
            add_yl = ''
        ylabel('RMS Ratio w.r.t linear scaling' + add_yl)

    if fit:
        return xx, myY, effective_variance_invcov
    else:
        return xx, myY, None


def get_angular_profile(maps, thmax=25, nbins=20, label='', center=np.array([316.44761929, -58.75808063]),
                        allstokes=False, fontsize=None, doplot=False, separate=False):
    '''
    maps: input maps
    thmax: int. Maximum angle (in degrees) for the angular profile.
    nbins: int. Number of bins for the profile.
    center: Center point for computing angular distances.
    allstockes: bool. If True, plot individual Stokes parameters.
    doplot: bool. If True, plot the results.
    separate: bool. If True, return separate RMS values for I, Q, and U.

    If doplot is True, plots the RMS of the average IQU (and of I, Q and U if allstockes is True) as a function of the angular distance.

    Returns angular distance, RMS I, RMS Q, RMS U if separate is True.
    Returns angular distance, RMS IQU otherwise.
    '''

    vec0 = hp.ang2vec(center[0], center[1], lonlat=True)
    sh = np.shape(maps)
    ns = hp.npix2nside(sh[0])
    vecpix = hp.pix2vec(ns, np.arange(12 * ns ** 2))
    angs = np.degrees(np.arccos(np.dot(vec0, vecpix)))
    rng = np.array([0, thmax])
    xx, yyI, dx, dyI, _ = ft.profile(angs, maps[:, 0], nbins=nbins, plot=False, rng=rng)
    xx, yyQ, dx, dyQ, _ = ft.profile(angs, maps[:, 1], nbins=nbins, plot=False, rng=rng)
    xx, yyU, dx, dyU, _ = ft.profile(angs, maps[:, 2], nbins=nbins, plot=False, rng=rng)
    avg = np.sqrt((dyI ** 2 + dyQ ** 2 / 2 + dyU ** 2 / 2) / 3)
    if doplot:
        plot(xx, avg, 'o', label=label)
        if allstokes:
            plot(xx, dyI, label=label + ' I', alpha=0.3)
            plot(xx, dyQ / np.sqrt(2), label=label + ' Q/sqrt(2)', alpha=0.3)
            plot(xx, dyU / np.sqrt(2), label=label + ' U/sqrt(2)', alpha=0.3)
        xlabel('Angle [deg.]')
        ylabel('RMS')
        legend(fontsize=fontsize)
    if separate:
        return xx, dyI, dyQ, dyU
    else:
        return xx, avg


def correct_maps_rms(maps, cov, effective_variance_invcov):
    '''
    Maps are corrected by a multiplying factor: sqrt(coverage/max(coverage))/sqrt(effective_variance_invcov).
    Returns corrected maps without angular dependance of the noise.
    '''

    okpix = cov > 0
    newmaps = maps * 0
    sh = np.shape(effective_variance_invcov)
    if sh[0] == 2:
        # Correction for I, Q, and U combined
        correction = np.interp(np.max(cov) / cov[okpix], effective_variance_invcov[0, :],
                               effective_variance_invcov[1, :])
        for s in range(3):
            newmaps[okpix, s] = maps[okpix, s] / np.sqrt(correction) * np.sqrt(cov[okpix] / np.max(cov))
    else:
        # Correction for separate I and QU
        correctionI = np.interp(np.max(cov) / cov[okpix], effective_variance_invcov[0, :],
                                effective_variance_invcov[1, :])
        correctionQU = np.interp(np.max(cov) / cov[okpix], effective_variance_invcov[0, :],
                                 effective_variance_invcov[2, :])
        newmaps[okpix, 0] = maps[okpix, 0] / np.sqrt(correctionI) * np.sqrt(cov[okpix] / np.max(cov))
        newmaps[okpix, 1] = maps[okpix, 1] / np.sqrt(correctionQU) * np.sqrt(cov[okpix] / np.max(cov))
        newmaps[okpix, 2] = maps[okpix, 2] / np.sqrt(correctionQU) * np.sqrt(cov[okpix] / np.max(cov))

    return newmaps


def flatten_noise(maps, coverage, nbins=20, doplot=False, normalize_all=False, QUsep=True):
    '''
    Combines get_noise_invcov_profile and correct_maps_rms.

    normalize_all: bool. If True, normalizes all sub-bands independently.

    Returns corrected maps, list of effective_variance_invcov, list of normalized noise level.
    '''

    sh = np.shape(maps)
    if len(sh) == 2:
        maps = np.reshape(maps, (1, sh[0], sh[1]))

    out_maps = np.zeros_like(maps)
    newsh = np.shape(maps)
    all_fitcov = []
    all_norm_noise = []
    if doplot:
        figure()
    for isub in range(newsh[0]):
        xx, yy, fitcov = get_noise_invcov_profile(maps[isub, :, :], coverage, nbins=nbins, norm=False,
                                                  label='sub-band: {}'.format(isub), fit=True,
                                                  doplot=doplot, allstokes=True, QUsep=QUsep)
        all_norm_noise.append(yy[0])
        if doplot:
            legend(fontsize=10)
        all_fitcov.append(fitcov)
        if normalize_all:
            out_maps[isub, :, :] = correct_maps_rms(maps[isub, :, :], coverage, fitcov)
        else:
            out_maps[isub, :, :] = correct_maps_rms(maps[isub, :, :], coverage, all_fitcov[0])

    if len(sh) == 2:
        return out_maps[0, :, :], all_fitcov
    else:
        return out_maps, all_fitcov, all_norm_noise


def map_corr_neighbtheta(themap_in, ipok_in, thetamin, thetamax, nbins, degrade=None, verbose=True):
    '''
    themap_in: The input map.
    ipok_in: List of pixels to consider.
    thetamin: Minimum angular distance in degrees.
    thetamax: Maximum angular distance in degrees.
    nbins: int. Number of bins for angular distances.
    degrade: Parameter for map degradation (optional). If provided, it degrades the input map to the 
        specified resolution and updates the list of pixels accordingly.

    Returns mythetas, corrfct, errs.
    mythetas: list of angular distances.
    corrfct: correlation function calculated for each angular distance bin, C(theta).
    errs: corresponding errors.
    '''

    if degrade is None:
        themap = themap_in.copy()
        ipok = ipok_in.copy()
    else:
        themap = hp.ud_grade(themap_in, degrade)
        mapbool = themap_in < -1e30
        mapbool[ipok_in] = True
        mapbool = hp.ud_grade(mapbool, degrade)
        ip = np.arange(12 * degrade ** 2)
        ipok = ip[mapbool]
    rthmin = np.radians(thetamin)
    rthmax = np.radians(thetamax)
    thvals = np.linspace(rthmin, rthmax, nbins + 1)
    ns = hp.npix2nside(len(themap))
    thesum = np.zeros(nbins)
    thesum2 = np.zeros(nbins)
    thecount = np.zeros(nbins)
    if verbose: bar = progress_bar(len(ipok), 'Pixels')
    for i in range(len(ipok)):
        if verbose: bar.update()
        valthis = themap[ipok[i]]
        v = hp.pix2vec(ns, ipok[i])
        # ipneighb_inner = []
        ipneighb_inner = list(hp.query_disc(ns, v, np.radians(thetamin)))
        for k in range(nbins):
            thmin = thvals[k]
            thmax = thvals[k + 1]
            ipneighb_outer = list(hp.query_disc(ns, v, thmax))
            ipneighb = ipneighb_outer.copy()
            for l in ipneighb_inner: ipneighb.remove(l)
            valneighb = themap[ipneighb]
            thesum[k] += np.sum(valthis * valneighb)
            thesum2[k] += np.sum((valthis * valneighb) ** 2)
            thecount[k] += len(valneighb)
            ipneighb_inner = ipneighb_outer.copy()

    mm = thesum / thecount
    mm2 = thesum2 / thecount
    errs = np.sqrt(mm2 - mm ** 2) / np.sqrt(np.sqrt(thecount))
    corrfct = thesum / thecount
    mythetas = np.degrees(thvals[:-1] + thvals[1:]) / 2
    return mythetas, corrfct, errs


def get_angles(ip0, ips, ns):
    v = np.array(hp.pix2vec(ns, ip0))
    vecs = np.array(hp.pix2vec(ns, ips))
    th = np.degrees(np.arccos(np.dot(v.T, vecs)))
    return th


def ctheta_parts(themap, ipok, thetamin, thetamax, nbinstot, nsplit=4, degrade_init=None,
                 verbose=True):
    '''
    ctheta_parts does the same computation as map_corr_neighbtheta, but more efficiently by splitting the computation into multiple parts.

    nsplit: int. Number of parts to split the computation into.
    '''

    allthetalims = np.linspace(thetamin, thetamax, nbinstot + 1)
    thmin = allthetalims[:-1]
    thmax = allthetalims[1:]
    idx = np.arange(nbinstot) // (nbinstot // nsplit)
    if degrade_init is None:
        nside_init = hp.npix2nside(len(themap))
    else:
        nside_init = degrade_init
    nside_part = nside_init // (2 ** idx)
    thall = np.zeros(nbinstot)
    cthall = np.zeros(nbinstot)
    errcthall = np.zeros(nbinstot)
    for k in range(nsplit):
        thispart = idx == k
        mythmin = np.min(thmin[thispart])
        mythmax = np.max(thmax[thispart])
        mynbins = nbinstot // nsplit
        mynside = nside_init // (2 ** k)
        if verbose: print(
            'Doing {0:3.0f} bins between {1:5.2f} and {2:5.2f} deg at nside={3:4.0f}'.format(mynbins, mythmin, mythmax,
                                                                                             mynside))
        myth, mycth, errs = map_corr_neighbtheta(themap, ipok, mythmin, mythmax, mynbins, degrade=mynside,
                                                 verbose=verbose)
        cthall[thispart] = mycth
        errcthall[thispart] = errs
        thall[thispart] = myth

        ### One could also calculate the average of the distribution of pixels within the ring instead of the simplistic thetas
    dtheta = allthetalims[1] - allthetalims[0]
    thall = 2. / 3 * ((thmin + dtheta) ** 3 - thmin ** 3) / ((thmin + dtheta) ** 2 - thmin ** 2)
    ### But it actually changes very little
    # print('coucou')
    return thall, cthall, errcthall


def get_cov_nunu(maps, coverage, nbins=20, QUsep=True, return_flat_maps=False):
    # This function returns the sub-frequency, sub_frequency covariance matrix for each stoke parameter
    # it does not attend to check for covariance between Stokes parameters (this should be incorporated later)
    # it returns the three covariance matrices as well as the fitted function of coverage that was used to
    # flatten the noise RMS in the maps before covariance calculation (this is for subsequent possible use)
    # NB: because this is done with  maps that are flattened, the RMS is put to 1 for I (and should be sqrt(2) for Q and U
    # so this covariance absorbs the overall maps variances

    ### First normalize by coverage
    new_sub_maps, all_fitcov, all_norm_noise = flatten_noise(maps, coverage, nbins=nbins, doplot=False, QUsep=QUsep)

    ### Now calculate the covariance matrix for each sub map
    sh = np.shape(maps)

    if len(sh) == 2:
        okpix = new_sub_maps[:, 0] != 0
        cov_I = np.array([[np.cov(new_sub_maps[okpix, 0])]])
        cov_Q = np.array([[np.cov(new_sub_maps[okpix, 1])]])
        cov_U = np.array([[np.cov(new_sub_maps[okpix, 2])]])
    else:
        okpix = new_sub_maps[0, :, 0] != 0
        cov_I = np.cov(new_sub_maps[:, okpix, 0])
        cov_Q = np.cov(new_sub_maps[:, okpix, 1])
        cov_U = np.cov(new_sub_maps[:, okpix, 2])
        if sh[0] == 1:
            cov_I = np.array([[cov_I]])
            cov_Q = np.array([[cov_Q]])
            cov_U = np.array([[cov_U]])

    if return_flat_maps:
        return cov_I, cov_Q, cov_U, all_fitcov, all_norm_noise, new_sub_maps
    else:
        return cov_I, cov_Q, cov_U, all_fitcov, all_norm_noise
