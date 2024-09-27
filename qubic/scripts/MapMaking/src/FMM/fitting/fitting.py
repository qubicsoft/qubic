#### General packages
import os
import os.path as op
import pickle
import sys
from multiprocessing import Pool

import emcee
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import qubic
import scipy
import yaml
from matplotlib.gridspec import *
from pyoperators import *
from schwimmbad import MPIPool

from qubic.lib.MapMaking.FrequencyMapMaking.Qspectra_component import CMB, Foreground

# sys.path.append('/pbs/home/t/tlaclave/sps/Pipeline')

#### QUBIC packages


comm = MPI.COMM_WORLD


def _Dl2Cl(ell, Dl):
    _f = ell * (ell + 1) / (2 * np.pi)
    return Dl / _f


def _Cl2BK(ell, Cl):
    return 100 * ell * Cl / (2 * np.pi)


__path__ = os.path.dirname(os.path.abspath(__file__))


class data:
    """
    Class to extract of the power spectra computed with spectrum.py and to compute useful things
    """

    def __init__(self):

        with open(__path__ + "/fitting_config.yaml", "r") as stream:
            self.params = yaml.safe_load(stream)

        self.path_repository = self.params["data"]["path"]
        self.path_spectra = self.path_repository + self.params["data"]["foldername"]
        self.path_fit = self.path_repository + "/fit/"

        ### Create fit folder
        if comm.Get_rank() == 0:
            if not os.path.isdir(self.path_fit):
                os.makedirs(self.path_fit)

        ### Read datasets
        (
            self.power_spectra_sky,
            self.power_spectra_noise,
            self.simu_parameters,
            self.coverage,
            self.nus,
            self.ell,
        ) = self.import_power_spectra(self.path_spectra)
        self._f = self.ell * (self.ell + 1) / (2 * np.pi)
        self.nrec = self.simu_parameters["QUBIC"]["nrec"]
        try:
            self.nsub = self.simu_parameters["QUBIC"]["nsub"]
            self.fsub = int(self.nsub / self.nrec)
        except:
            self.fsub = self.simu_parameters["QUBIC"]["fsub"]
            self.nsub = self.fsub * self.nrec

        self.nreal = len(self.power_spectra_sky)
        if comm.Get_rank() == 0:
            print(f"Number of realizations : {self.nreal}")

        ######## Test
        # self.dict, self.dict_mono = self.get_dict()
        # self.Q = acq.QubicFullBandSystematic(self.dict, self.simu_parameters['QUBIC']['nsub'], self.simu_parameters['QUBIC']['nrec'], kind='UWB')
        # #joint = acq.JointAcquisitionFrequencyMapMaking(self.dict, 'UWB', self.simu_parameters['QUBIC']['nrec'], self.simu_parameters['QUBIC']['nsub'])
        # list_h = self.Q.H
        # h_list = np.empty(len(self.Q.allnus))
        # vec_ones = np.ones(list_h[0].shapein)
        # for freq in range(len(self.Q.allnus)):
        #     h_list[freq] = np.mean(list_h[freq](vec_ones))
        # print('Q.allnus', self.Q.allnus)
        # print('nus', self.nus)
        # def f_beta(nu, nu_0, beta):
        #     return (nu/nu_0)**beta * (np.exp(scipy.constants.h*nu*1e9/(scipy.constants.k * 20)) - 1) / (np.exp(scipy.constants.h*nu_0*1e9/(scipy.constants.k * 20)) - 1)

        # corrected_allnus = []
        # for i in range(self.simu_parameters['QUBIC']['nrec']):
        #     fraction = (np.sum(h_list[i*self.fsub:(i+1)*self.fsub] * f_beta(self.Q.allnus[i*self.fsub:(i+1)*self.fsub], 353, 1.53)) / np.sum(h_list[i*self.fsub:(i+1)*self.fsub]))
        #     fun = lambda nu: np.abs(fraction - f_beta(nu, 353, 1.53))
        #     x0 = self.nus[i]
        #     corrected_nu = scipy.optimize.minimize(fun, x0)
        #     corrected_allnus.append(corrected_nu['x'])

        # print('corrected allnus', corrected_allnus)
        # print('nrec', self.nrec)
        # print('nus', self.nus)
        # self.nus[:self.nrec] = corrected_allnus
        # print('new nus', self.nus)
        # stop

        ### Select bandpowers for fitting
        bp_to_rm = self.select_bandpower()
        self.nfreq = len(self.nus)
        self.nspecs = (self.nfreq * (self.nfreq + 1)) // 2
        self.nspecs_qubic = (self.nrec * (self.nrec + 1)) // 2

        ### Remove bandpowers not selected
        self.power_spectra_sky = np.delete(self.power_spectra_sky, bp_to_rm, 1)
        self.power_spectra_sky = np.delete(self.power_spectra_sky, bp_to_rm, 2)
        self.power_spectra_noise = np.delete(self.power_spectra_noise, bp_to_rm, 1)
        self.power_spectra_noise = np.delete(self.power_spectra_noise, bp_to_rm, 2)

        ### Average and standard deviation from realizations
        self.mean_ps_sky, self.error_ps_sky = self.compute_mean_std(
            self.power_spectra_sky
        )
        self.mean_ps_noise, self.error_ps_noise = self.compute_mean_std(
            self.power_spectra_noise
        )

        if self.params["simu"]["noise"] is True:
            self.mean_ps_sky = self.spectra_noise_correction(
                self.mean_ps_sky, self.mean_ps_noise
            )

        self.ps_sky_reshape = np.zeros((self.nreal, self.nspecs, len(self.ell)))
        self.ps_noise_reshape = np.zeros((self.nreal, self.nspecs, len(self.ell)))
        self.mean_ps_sky_reshape = np.zeros((self.nspecs, len(self.ell)))
        self.mean_ps_noise_reshape = np.zeros((self.nspecs, len(self.ell)))

        for ireal in range(self.nreal):
            k = 0
            for i in range(self.nfreq):
                for j in range(i, self.nfreq):
                    self.ps_sky_reshape[ireal, k] = self.power_spectra_sky[ireal][
                        i, j
                    ].copy()
                    self.ps_noise_reshape[ireal, k] = self.power_spectra_noise[ireal][
                        i, j
                    ].copy()
                    k += 1
        k = 0
        for i in range(self.nfreq):
            for j in range(i, self.nfreq):
                self.mean_ps_sky_reshape[k] = self.mean_ps_sky[i, j].copy()
                self.mean_ps_noise_reshape[k] = self.mean_ps_noise[i, j].copy()
                k += 1

        # print(np.std(self.power_spectra_noise, axis=0)[..., 0])
        # stop

    def get_ultrawideband_config(self):
        """

        Method that pre-compute UWB configuration.

        """

        nu_up = 247.5
        nu_down = 131.25
        nu_ave = np.mean(np.array([nu_up, nu_down]))
        delta = nu_up - nu_ave

        return nu_ave, 2 * delta / nu_ave

    def get_dict(self):
        """

        Method to modify the qubic dictionary.

        """

        nu_ave, delta_nu_over_nu = self.get_ultrawideband_config()

        args = {
            "npointings": self.simu_parameters["QUBIC"]["npointings"],
            "nf_recon": self.simu_parameters["QUBIC"]["nrec"],
            "nf_sub": self.simu_parameters["QUBIC"]["nsub"],
            "nside": self.simu_parameters["SKY"]["nside"],
            "MultiBand": True,
            "period": 1,
            "RA_center": self.simu_parameters["SKY"]["RA_center"],
            "DEC_center": self.simu_parameters["SKY"]["DEC_center"],
            "filter_nu": nu_ave * 1e9,
            "noiseless": False,
            "comm": comm,
            "dtheta": self.simu_parameters["QUBIC"]["dtheta"],
            "nprocs_sampling": 1,
            "nprocs_instrument": comm.Get_size(),
            "photon_noise": True,
            "nhwp_angles": 3,
            "effective_duration": 3,
            "filter_relative_bandwidth": delta_nu_over_nu,
            "type_instrument": "wide",
            "TemperatureAtmosphere150": None,
            "TemperatureAtmosphere220": None,
            "EmissivityAtmosphere150": None,
            "EmissivityAtmosphere220": None,
            "detector_nep": float(
                self.simu_parameters["QUBIC"]["NOISE"]["detector_nep"]
            ),
            "synthbeam_kmax": self.simu_parameters["QUBIC"]["SYNTHBEAM"][
                "synthbeam_kmax"
            ],
        }

        args_mono = args.copy()
        args_mono["nf_recon"] = 1
        args_mono["nf_sub"] = 1

        ### Get the default dictionary
        dictfilename = "dicts/pipeline_demo.dict"
        d = qubic.qubicdict.qubicDict()
        d.read_from_file(dictfilename)
        dmono = d.copy()
        for i in args.keys():

            d[str(i)] = args[i]
            dmono[str(i)] = args_mono[i]

        return d, dmono

    def select_bandpower(self):
        """
        Function to remove some bamdpowers if they are not selected.

        Return :
            - list containing the indices for removed bandpowers.
        """
        k = 0
        bp_to_rm = []
        for ii, i in enumerate(self.nus):
            if ii < self.params["simu"]["nrec"]:
                if self.params["NUS"]["qubic"]:
                    pass
                    # k += [self.params['NUS']['qubic'][1]
                else:
                    bp_to_rm += [ii]
                    k += 1

            else:
                if self.params["NUS"][f"{i:.0f}GHz"] is False:
                    bp_to_rm += [ii]
        # print(bp_to_rm)
        self.nus = np.delete(self.nus, bp_to_rm, 0)
        return bp_to_rm

    def import_power_spectra(self, path):
        """
        Function to import all the power spectra computed with spectrum.py and store in pickle files

        Argument :
            - path (str) : path to indicate where the pkl files are

        Return :
            - sky power spectra (array) [nreal, nrec/ncomp, nrec/ncomp, len(ell)]
            - noise power spectra (array) [nreal, nrec/ncomp, nrec/ncomp, len(ell)]
            - simulations parameters (dict)
            - simulations coverage (array)
            - bands frequencies for FMM (array) [nrec]
        """

        power_spectra_sky, power_spectra_noise = [], []
        names = os.listdir(path)
        if self.params["data"]["n_real"] == -1:
            nreals = len(names)
        else:
            nreals = self.params["data"]["n_real"]
        for i in range(nreals):
            # if comm.Get_rank() == 0:
            # print(f"======== Importing power spectrum {i+1} / {self.params['data']['n_real']} ==========")
            ps = pickle.load(open(path + "/" + names[i], "rb"))
            power_spectra_sky.append(ps["Dls"][:, :, : self.params["nbins"]])
            power_spectra_noise.append(ps["Nl"][:, :, : self.params["nbins"]])

        return (
            power_spectra_sky,
            power_spectra_noise,
            ps["coverage"],
            ps["nus"],
            ps["ell"][: self.params["nbins"]],
        )

    def compute_mean_std(self, ps):
        """
        Function to compute the mean ans the std on our power spectra realisations

        Argument :
            - power spectra array (array) [nreal, nrec/ncomp, nrec/ncomp, len(ell)]

        Return :
            - mean (array) [nrec/ncomp, nrec/ncomp, len(ell)]
            - std (array) [nrec/ncomp, nrec/ncomp, len(ell)]
        """
        return np.mean(ps, axis=0), np.std(ps, axis=0)

    def spectra_noise_correction(self, mean_data, mean_noise):
        """
        Function to remove the mean of the noise realisations to the spectra computed

        Arguments :
            - mean sky power spectra (array) [nrec/ncomp, nrec/ncomp, len(ell)] : array that will contain the mean of all the auto and cross spectra of the sky realisations
            - mean noise power spectra (array) [nrec/ncomp, nrec/ncomp, len(ell)] : array that will contain the mean of all the auto and cross spectra of the noise realisation

        Return :
            - corrected mean sky power spectra (array) [nrec/ncomp, nrec/ncomp, len(ell)]
        """

        for i in range(np.shape(mean_data)[0]):
            for j in range(np.shape(mean_data)[1]):
                mean_data[i, j, :] -= mean_noise[i, j, :]
        return mean_data


class Fitting(data):
    """
    Class to perform MCMC on the chosen sky parameters
    """

    def __init__(self):

        if comm.Get_rank() == 0:
            print("\n=========== Fitting ===========\n")

        data.__init__(self)
        self.sky_parameters = self.params["SKY_PARAMETERS"]
        self.ndim, self.sky_parameters_fitted_names, self.sky_parameters_all_names = (
            self.ndim_and_parameters_names()
        )

        ### Compute noise covariance/correlation matrix
        self.noise_cov_matrix = np.zeros(
            (self.nspecs, len(self.ell), len(self.ell), self.nspecs)
        )
        samples = self.ps_noise_reshape.reshape(
            (self.ps_noise_reshape.shape[0], self.nspecs * len(self.ell))
        )
        self.noise_cov_matrix = np.cov(samples, rowvar=False)
        self.noise_correlation_matrix = np.corrcoef(samples, rowvar=False)
        self.covariance = self.noise_cov_matrix.copy()
        self.covariance_to_save = self.noise_cov_matrix.copy()
        self.correlation_to_save = self.noise_correlation_matrix.copy()

        self.index_only_qubic = []
        self.index_both = []
        self.index_only_planck = []
        k = 0

        for i in range(self.nspecs):
            for j in range(self.nspecs):
                if (i < self.nspecs_qubic) and (j < self.nspecs_qubic):
                    self.index_only_qubic += [(i, j)]
                elif (i < self.nspecs_qubic) or (j < self.nspecs_qubic):
                    self.index_both += [(i, j)]
                elif (i > self.nspecs_qubic) and (j < self.nspecs_qubic):
                    self.index_both += [(i, j)]
                elif (i < self.nspecs_qubic) and (j > self.nspecs_qubic):
                    self.index_both += [(i, j)]
                else:
                    self.index_only_planck += [(i, j)]
                k += 1

        ### If wanted, add the sample variance
        if self.params["sample_variance"]:
            self.sample_cov_matrix = self._fill_sample_variance(
                self.mean_ps_sky
            ).reshape((self.nspecs * len(self.ell), self.nspecs * len(self.ell)))
            self.covariance += self.sample_cov_matrix.copy()

        self.inv_cov = np.linalg.pinv(self.covariance)

        ### Remove off-diagonal elements in covariance matrix for QUBIC data
        if self.params["diagonal_qubic"]:
            for ii in self.index_only_qubic:
                self.inv_cov[
                    ii[0] * len(self.ell) : (ii[0] + 1) * len(self.ell),
                    ii[1] * len(self.ell) : (ii[1] + 1) * len(self.ell),
                ] *= np.eye(len(self.ell))

        ### Remove off-diagonal elements in covariance matrix for Planck data
        if self.params["diagonal_planck"]:
            for ii in self.index_only_planck:
                self.inv_cov[
                    ii[0] * len(self.ell) : (ii[0] + 1) * len(self.ell),
                    ii[1] * len(self.ell) : (ii[1] + 1) * len(self.ell),
                ] *= np.eye(len(self.ell))

        # for ii in self.index_both:
        #    self.inv_cov[ii[0]*len(self.ell):(ii[0]+1)*len(self.ell), ii[1]*len(self.ell):(ii[1]+1)*len(self.ell)] *= np.eye(len(self.ell))

        ### Initiate models
        self.cmb = CMB(self.ell, self.nus)
        self.foregrounds = Foreground(self.ell, self.nus)
        model = self.cmb.model_cmb(0, 1) + self.foregrounds.model_dust(
            0, -0.17, 1.54, 1, 353
        )

        ### Produce plot of all the spectra
        self._get_Dl_plot(
            self.nus,
            self.ell,
            self.mean_ps_sky,
            self.error_ps_noise,
            model,
            nbins=self.params["nbins"],
            nrec=self.nrec,
        )

    def _get_Dl_plot(self, nus, ell, Dl, Dl_err, ymodel, nbins=8, nrec=2):
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(len(nus), len(nus), figure=fig)

        k1 = 0
        kp = 0
        for i in range(len(nus)):
            for j in range(i, len(nus)):
                ax = fig.add_subplot(gs[i, j])

                if k1 == 0:
                    ax.plot(
                        ell[:nbins],
                        _Cl2BK(ell[:nbins], _Dl2Cl(ell[:nbins], ymodel[i, j, :nbins])),
                        "--r",
                        label=f"r + Dust + noise",
                    )
                else:
                    ax.plot(
                        ell[:nbins],
                        _Cl2BK(ell[:nbins], _Dl2Cl(ell[:nbins], ymodel[i, j, :nbins])),
                        "--r",
                    )

                ax.patch.set_alpha(0.3)

                ax.annotate(
                    f"{nus[i]:.0f}x{nus[j]:.0f}",
                    xy=(0.1, 0.9),
                    xycoords="axes fraction",
                    color="black",
                    weight="bold",
                )
                if i < nrec and j < nrec:
                    ax.set_facecolor("blue")
                    if k1 == 0:
                        ax.errorbar(
                            ell[:nbins],
                            _Cl2BK(
                                ell[:nbins], _Dl2Cl(ell[:nbins], Dl[i, j, :nbins])
                            ),  # Dls_mean[kp] - Nl_mean[kp],
                            yerr=_Cl2BK(
                                ell[:nbins], _Dl2Cl(ell[:nbins], Dl_err[i, j, :nbins])
                            ),
                            capsize=5,
                            color="darkblue",
                            fmt="o",
                            label=r"$\mathcal{D}_{\ell}^{\nu_1 \times \nu_2}$",
                        )

                    else:
                        ax.errorbar(
                            ell[:nbins],
                            _Cl2BK(
                                ell[:nbins], _Dl2Cl(ell[:nbins], Dl[i, j, :nbins])
                            ),  # Dls_mean[kp] - Nl_mean[kp],
                            yerr=_Cl2BK(
                                ell[:nbins], _Dl2Cl(ell[:nbins], Dl_err[i, j, :nbins])
                            ),
                            capsize=5,
                            color="darkblue",
                            fmt="o",
                        )
                elif i < nrec and j >= nrec:
                    ax.set_facecolor("skyblue")
                    ax.errorbar(
                        ell[:nbins],
                        _Cl2BK(
                            ell[:nbins], _Dl2Cl(ell[:nbins], Dl[i, j, :nbins])
                        ),  # Dls_mean[kp] - Nl_mean[kp],
                        yerr=_Cl2BK(
                            ell[:nbins], _Dl2Cl(ell[:nbins], Dl_err[i, j, :nbins])
                        ),
                        capsize=5,
                        color="blue",
                        fmt="o",
                    )
                else:
                    ax.set_facecolor("green")
                    ax.errorbar(
                        ell[:nbins],
                        _Cl2BK(
                            ell[:nbins], _Dl2Cl(ell[:nbins], Dl[i, j, :nbins])
                        ),  # Dls_mean[kp] - Nl_mean[kp],
                        yerr=_Cl2BK(
                            ell[:nbins], _Dl2Cl(ell[:nbins], Dl_err[i, j, :nbins])
                        ),
                        capsize=5,
                        color="darkgreen",
                        fmt="o",
                    )
                # ax[i, j].set_title(f'{data["nus"][i]:.0f}x{data["nus"][j]:.0f}')
                kp += 1
            else:
                pass  # ax.axis('off')
            k1 += 1

        plt.tight_layout()
        plt.savefig("Dls.png")
        plt.close()

    def ndim_and_parameters_names(self):
        """
        Function to create the name list of the parameter(s) that you want to find with the MCMC and to compute the number of these parameters

        Return :
            - ndim (int) : number of parameters you want to fit
            - sky_parameters_fitted_names (array) [ndim] : list that contains the names of the fitted parameters
            - sky_parameters_all_names (array) : list that contains the names of all the sky parameters
        """

        ndim = 0
        sky_parameters_fitted_names = []
        sky_parameters_all_names = []

        for parameter in self.sky_parameters:
            sky_parameters_all_names.append(parameter)
            if self.sky_parameters[parameter][0] is True:
                ndim += 1
                sky_parameters_fitted_names.append(parameter)

        return ndim, sky_parameters_fitted_names, sky_parameters_all_names

    def dl_to_cl(self, dl):
        cl = np.zeros(self.ell.shape[0])
        for i in range(self.ell.shape[0]):
            cl[i] = dl[i] * (2 * np.pi) / (self.ell[i] * (self.ell[i] + 1))
        return cl

    def knox_errors(self, dlth):
        dcl = (
            np.sqrt(
                2 / ((2 * self.ell + 1) * 0.01 * self.simu_parameters["Spectrum"]["dl"])
            )
            * dlth
        )
        return dcl

    def knox_covariance(self, dlth):
        dcl = self.knox_errors(dlth)
        return np.eye(len(self.ell)) * dcl**2

    def prior(self, x):
        """
        Function to define priors to help the MCMC convergence

        Argument :
            - x (array) [ndim] : array that contains the numbers randomly generated by the mcmc

        Return :
            - (float) : inf if the prior is not respected, 0 otherwise
        """

        for isky_param, sky_param in enumerate(x):
            name = self.sky_parameters_fitted_names[isky_param]

            if (
                sky_param < self.sky_parameters[name][3]
                or sky_param > self.sky_parameters[name][4]
            ):
                return -np.inf

        return 0

    def initial_conditions(self):
        """
        Function to computes the MCMC initial conditions

        Return :
            - p0 (array) [nwalkers, ndim] : array that contains all the initial conditions for the mcmc
        """

        nwalkers = self.params["MCMC"]["nwalkers"]

        p0 = np.zeros((nwalkers, self.ndim))
        for i in range(nwalkers):
            for j in range(self.ndim):
                name = self.sky_parameters_fitted_names[j]
                p0[i, j] = (
                    np.random.random() * self.params["SKY_PARAMETERS"][name][2]
                    + self.params["SKY_PARAMETERS"][name][1]
                )

        return p0

    def ptform_uniform(self, u):
        """
        Function to perform an uniform prior transform for the Nested fitting

        Argument :
            - x (array) [ndim] : array that contains the numbers randomly generated by the mcmc

        Return :
            - ptform (array) [ndim]
        """

        ptform = []
        cpt = 0
        for iname in self.sky_parameters_all_names:
            if self.params["SKY_PARAMETERS"][iname][0] is True:
                ptform.append(
                    u[cpt] * self.params["SKY_PARAMETERS"][iname][2]
                    - self.params["SKY_PARAMETERS"][iname][1]
                )
                cpt += 1
        return ptform

    def _fill_sample_variance(self, bandpower):

        indices_tr = np.triu_indices(len(self.nus))
        matrix = np.zeros((self.nspecs, len(self.ell), self.nspecs, len(self.ell)))
        factor_modecount = 1 / (
            (2 * self.ell + 1) * 0.015 * self.simu_parameters["Spectrum"]["dl"]
        )
        for ii, (i1, i2) in enumerate(zip(indices_tr[0], indices_tr[1])):
            for jj, (j1, j2) in enumerate(zip(indices_tr[0], indices_tr[1])):
                covar = (
                    bandpower[i1, j1, :] * bandpower[i2, j2, :]
                    + bandpower[i1, j2, :] * bandpower[i2, j1, :]
                ) * factor_modecount
                matrix[ii, :, jj, :] = np.diag(covar)
        return matrix

    def _reshape_spectra(self, bandpower):

        bandpower_reshaped = np.zeros((self.nspecs, len(self.ell)))
        k = 0
        for i in range(self.nfreq):
            for j in range(i, self.nfreq):
                bandpower_reshaped[k] = bandpower[i, j].copy()
                k += 1
        return bandpower_reshaped

    def loglikelihood(self, tab):
        """
        loglikelihood function

        Argument :
            - x (array) [ndim] : array that contains the numbers randomly generated by the mcmc

        Return :
            - (float) : loglikelihood function
        """
        tab_parameters = np.zeros(len(self.params["SKY_PARAMETERS"]))
        cpt = 0

        for i, iname in enumerate(self.params["SKY_PARAMETERS"]):
            if self.params["SKY_PARAMETERS"][iname][0] is not True:
                tab_parameters[i] = self.params["SKY_PARAMETERS"][iname][0]
            else:
                tab_parameters[i] = tab[cpt]
                cpt += 1
        r, Alens, nu0_d, Ad, alphad, betad, deltad, nu0_s, As, alphas, betas, eps = (
            tab_parameters
        )

        # Define the sky model & the sample variance associated
        model = self.cmb.model_cmb(r, Alens)
        model += self.foregrounds.model_dust(Ad, alphad, betad, deltad, nu0_d)

        model = self._reshape_spectra(model)
        _r = model - (self.mean_ps_sky_reshape)
        _r = _r.reshape((self.nspecs * len(self.ell)))

        loglike = self.prior(tab) - 0.5 * (_r.T @ self.inv_cov @ _r)

        return loglike

    def save_data(self, name, d):
        """

        Method to save data using pickle convention.

        """

        with open(name, "wb") as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def run(self):
        """
        Funtion to perform the MCMC and save the results
        """

        # Define the MCMC parameters, initial conditions and ell list
        nwalkers = self.params["MCMC"]["nwalkers"]
        mcmc_steps = self.params["MCMC"]["mcmc_steps"]
        p0 = self.initial_conditions()

        if comm.Get_size() != 1:
            # Start the MCMC
            with MPIPool() as pool:
                sampler = emcee.EnsembleSampler(
                    nwalkers, self.ndim, log_prob_fn=self.loglikelihood, pool=pool
                )
                sampler.run_mcmc(p0, mcmc_steps, progress=True)
        else:
            # Start the MCMC
            with Pool() as pool:
                sampler = emcee.EnsembleSampler(
                    nwalkers, self.ndim, log_prob_fn=self.loglikelihood, pool=pool
                )
                sampler.run_mcmc(p0, mcmc_steps, progress=True)

        self.samples_flat = sampler.get_chain(
            flat=True,
            discard=self.params["MCMC"]["discard"],
            thin=self.params["MCMC"]["thin"],
        )
        self.samples = sampler.get_chain()

        name = []
        for inu, i in enumerate(self.params["NUS"]):
            if inu == 0 and self.params["NUS"][str(i)]:
                name += ["qubic"]
            else:
                if self.params["NUS"][str(i)]:
                    name += [str(i)]

        name = "_".join(name)
        self.save_data(
            self.path_fit + f"{self.params['data']['filename']}" + ".pkl",
            {
                "nus": self.nus,
                "covariance_matrix": self.covariance_to_save,
                "correlation_matrix": self.correlation_to_save,
                "ell": self.ell,
                "samples": self.samples,
                "samples_flat": self.samples_flat,
                "fitted_parameters_names": self.sky_parameters_fitted_names,
                "parameters": self.params,
                "Dls": self.power_spectra_sky,
                "Nls": self.power_spectra_noise,
                "simulation_parameters": self.simu_parameters,
            },
        )
        print("Fitting done !!!")


fit = Fitting()
fit.run()

if comm.Get_rank() == 0:

    print()
    print(f"Average : {np.mean(fit.samples_flat, axis=0)}")
    print(f"Error : {np.std(fit.samples_flat, axis=0)}")
    print()

    plt.figure()
    print(fit.samples.shape)
    for i in range(fit.samples.shape[2]):
        plt.subplot(fit.samples.shape[2], 1, i + 1)
        if i == 0:
            plt.axhline(0, color="black", ls="--")
            plt.ylim(-0.07, 0.07)
        plt.plot(fit.samples[:, :, i], "-k", alpha=0.1)
        plt.plot(np.mean(fit.samples[:, :, i], axis=1), "-b", alpha=0.5)
        plt.plot(
            np.mean(fit.samples[:, :, i], axis=1)
            + np.std(fit.samples[:, :, i], axis=1),
            "-r",
            alpha=0.5,
        )
        plt.xlim(0, fit.samples.shape[0])
    plt.savefig("chains.png")
    plt.close()
