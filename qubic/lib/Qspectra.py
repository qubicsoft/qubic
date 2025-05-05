import pickle

import numpy as np
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

from qubic.lib.Fitting import Qnamaster as nam


class Spectra:
    def __init__(self, filename, covcut=0.2):
        self.filename = filename

        ### Get the job ID
        self.jobid = filename.split("/")[-1].split("_")[-1][:-4]

        self.dictionary = self.open_file()
        self.maps = self.dictionary["maps"].copy()

        self.seenpix = self.dictionary["coverage"] / self.dictionary["coverage"].max() > covcut
        self.fwhm = self.dictionary["fwhm_rec"]

        ### Initiate namaster from qubic soft
        self._init_namaster()

    def open_file(self):
        with open(self.filename, "rb") as f:
            dictionary = pickle.load(f)

        self.spectrum = dictionary["parameters"]["Spectrum"]
        self.sky = dictionary["parameters"]["SKY"]

        return dictionary

    def _init_namaster(self):
        self.namaster = nam.Namaster(
            weight_mask=self.seenpix,
            lmin=self.spectrum["lmin"],
            lmax=self.spectrum["lmax"],
            delta_ell=self.spectrum["dl"],
        )

        self.ell = self.namaster.get_binning(self.sky["nside"])[0]
        self.nbins = len(self.ell)

    def _get_BB_spectra(self, map, map2=None, fwhm=0):
        BB = self.namaster.get_spectra(map=map, map2=map2, verbose=False, beam_correction=np.rad2deg(fwhm))[1][:, 2]
        return BB

    def run(self, maps):
        nmaps = self.maps.shape[0]

        BBspectra = np.zeros((nmaps, nmaps, self.nbins))

        ### Ensure that non seen pixels is 0
        maps[:, ~self.seenpix, :] = 0

        for irec in range(nmaps):
            for jrec in range(irec, nmaps):
                print(f"======== Cross-spectra with maps {irec} x {jrec} ========")

                if self.fwhm[irec] > self.fwhm[jrec]:
                    fwhm = np.sqrt(self.fwhm[irec] ** 2 - self.fwhm[jrec] ** 2)
                    C = HealpixConvolutionGaussianOperator(fwhm=fwhm)
                    map1 = maps[irec]
                    map2 = C(maps[jrec])
                else:
                    fwhm = np.sqrt(self.fwhm[jrec] ** 2 - self.fwhm[irec] ** 2)
                    C = HealpixConvolutionGaussianOperator(fwhm=fwhm)
                    map1 = C(maps[irec])
                    map2 = maps[jrec]
                BBspectra[irec, jrec, :] = self._get_BB_spectra(map=map1.T, map2=map2.T, fwhm=fwhm)

                if irec != jrec:
                    BBspectra[jrec, irec] = BBspectra[irec, jrec]

        return BBspectra


# class Spectrumold:
#     """
#     Class to compute the different spectra for our realisations
#     """

#     def __init__(self, file, verbose=True):
#         self.spectra_time_0 = time.time()

#         print("\n=========== Power Spectra ===========\n")

#         filename = os.path.split(file)
#         self.jobid = filename[1].split("_")[1].split(".")[0]
#         print("Job id found : ", self.jobid)

#         self.path_spectrum = os.path.join(os.path.dirname(os.path.dirname(file)), "spectrum")
#         if not os.path.isdir(self.path_spectrum):
#             os.makedirs(self.path_spectrum)

#         with open("params.yml", "r") as stream:
#             self.params = yaml.safe_load(stream)

#         with open(file, "rb") as f:
#             self.dict_file = pickle.load(f)

#         self.verbose = verbose
#         self.sky_maps = self.dict_file["maps"].copy()
#         self.noise_maps = self.dict_file["maps_noise"].copy()

#         self.nus = self.dict_file["nus"]
#         self.nfreq = len(self.nus)
#         self.nrec = self.params["QUBIC"]["nrec"]
#         self.nsub = self.params["QUBIC"]["nsub"]
#         self.fsub = int(self.nsub / self.nrec)
#         self.nside = self.params["SKY"]["nside"]
#         self.nsub = int(self.fsub * self.nrec)

#         _, nus150, _, _, _, _ = compute_freq(150, Nfreq=self.fsub - 1)
#         _, nus220, _, _, _, _ = compute_freq(220, Nfreq=self.fsub - 1)

#         self.fwhm150 = self._get_fwhm_during_MM(nus150)
#         self.fwhm220 = self._get_fwhm_during_MM(nus220)

#         self.allfwhm = self._get_allfwhm()

#         ############ Test
#         if self.params["QUBIC"]["convolution_in"] and not self.params["QUBIC"]["convolution_out"]:
#             self.dict, self.dict_mono = self.get_dict()
#             self.Q = acq.QubicFullBandSystematic(
#                 self.dict,
#                 self.params["QUBIC"]["nsub"],
#                 self.params["QUBIC"]["nrec"],
#                 kind="UWB",
#             )
#             # joint = acq.JointAcquisitionFrequencyMapMaking(self.dict, 'UWB', self.params['QUBIC']['nrec'], self.params['QUBIC']['nsub'])
#             list_h = self.Q.H
#             h_list = np.empty(len(self.Q.allnus))
#             vec_ones = np.ones(list_h[0].shapein)
#             for freq in range(len(self.Q.allnus)):
#                 h_list[freq] = np.mean(list_h[freq](vec_ones))

#             fwhm = [self._get_fwhm_during_MM(i) for i in self.Q.allnus]

#             def f_beta(nu, nu_0, beta):
#                 return (nu / nu_0) ** beta * (np.exp(c.h * nu * 1e9 / (c.k * 20)) - 1) / (np.exp(c.h * nu_0 * 1e9 / (c.k * 20)) - 1)

#             corrected_allfwhm = []
#             corrected_allnus = []
#             for i in range(self.nrec):
#                 corrected_fwhm = np.sum(
#                     h_list[i * self.fsub : (i + 1) * self.fsub] * f_beta(self.Q.allnus[i * self.fsub : (i + 1) * self.fsub], 353, 1.53) * fwhm[i * self.fsub : (i + 1) * self.fsub]
#                 ) / (
#                     np.sum(
#                         h_list[i * self.fsub : (i + 1) * self.fsub]
#                         * f_beta(
#                             self.Q.allnus[i * self.fsub : (i + 1) * self.fsub],
#                             353,
#                             1.53,
#                         )
#                     )
#                 )
#                 corrected_allfwhm.append(corrected_fwhm)
#                 fraction = np.sum(h_list[i * self.fsub : (i + 1) * self.fsub] * f_beta(self.Q.allnus[i * self.fsub : (i + 1) * self.fsub], 353, 1.53)) / np.sum(
#                     h_list[i * self.fsub : (i + 1) * self.fsub]
#                 )

#                 def objective_function(nu):
#                     return np.abs(fraction - f_beta(nu, 353, 1.53))

#                 x0 = self.nus[i]
#                 corrected_nu = minimize(objective_function, x0)
#                 corrected_allnus.append(corrected_nu["x"])

#             print("nus", self.nus)
#             print("fwhm", self.allfwhm)
#             self.nus[: self.nrec] = corrected_allnus
#             self.allfwhm[: self.nrec] = corrected_allfwhm
#             print("corrected nus", self.nus)
#             print("corrected fwhm", self.allfwhm)

#         self.kernels150 = np.sqrt(self.fwhm150[0] ** 2 - self.fwhm150[-1] ** 2)
#         self.kernels220 = np.sqrt(self.fwhm220[0] ** 2 - self.fwhm220[-1] ** 2)
#         self.kernels = np.array([self.kernels150, self.kernels220])

#         # Define Namaster class
#         self.coverage = self.dict_file["coverage"]
#         self.seenpix = self.coverage / np.max(self.coverage) > 0.2

#         self.namaster = nam.Namaster(
#             weight_mask=list(np.array(self.seenpix)),
#             lmin=self.params["Spectrum"]["lmin"],
#             lmax=self.params["Spectrum"]["lmax"],
#             delta_ell=self.params["Spectrum"]["dl"],
#         )

#         self.ell = self.namaster.get_binning(self.params["SKY"]["nside"])[0]
#     def get_ultrawideband_config(self):
#         """

#         Method that pre-compute UWB configuration.

#         """

#         nu_up = 247.5
#         nu_down = 131.25
#         nu_ave = np.mean(np.array([nu_up, nu_down]))
#         delta = nu_up - nu_ave

#         return nu_ave, 2 * delta / nu_ave

#     def get_dict(self):
#         """

#         Method to modify the qubic dictionary.

#         """

#         nu_ave, delta_nu_over_nu = self.get_ultrawideband_config()

#         args = {
#             "npointings": self.params["QUBIC"]["npointings"],
#             "nf_recon": self.params["QUBIC"]["nrec"],
#             "nf_sub": self.params["QUBIC"]["nsub"],
#             "nside": self.params["SKY"]["nside"],
#             "MultiBand": True,
#             "period": 1,
#             "RA_center": self.params["SKY"]["RA_center"],
#             "DEC_center": self.params["SKY"]["DEC_center"],
#             "filter_nu": nu_ave * 1e9,
#             "noiseless": False,
#             "comm": comm,
#             "dtheta": self.params["QUBIC"]["dtheta"],
#             "nprocs_sampling": 1,
#             "nprocs_instrument": comm.Get_size(),
#             "photon_noise": True,
#             "nhwp_angles": 3,
#             "effective_duration": 3,
#             "filter_relative_bandwidth": delta_nu_over_nu,
#             "type_instrument": "wide",
#             "TemperatureAtmosphere150": None,
#             "TemperatureAtmosphere220": None,
#             "EmissivityAtmosphere150": None,
#             "EmissivityAtmosphere220": None,
#             "detector_nep": float(self.params["QUBIC"]["NOISE"]["detector_nep"]),
#             "synthbeam_kmax": self.params["QUBIC"]["SYNTHBEAM"]["synthbeam_kmax"],
#         }

#         args_mono = args.copy()
#         args_mono["nf_recon"] = 1
#         args_mono["nf_sub"] = 1

#         ### Get the default dictionary
#         dictfilename = "dicts/pipeline_demo.dict"
#         d = qubic.qubicdict.qubicDict()
#         d.read_from_file(dictfilename)
#         dmono = d.copy()
#         for i in args.keys():
#             d[str(i)] = args[i]
#             dmono[str(i)] = args_mono[i]

#         return d, dmono

#     def _get_fwhm_during_MM(self, nu):
#         return np.deg2rad(0.39268176 * 150 / nu)

#     def _get_allfwhm(self):
#         """
#         Function to compute the fwhm for all sub bands.

#         Return :
#             - allfwhm (list [nfreq])
#         """
#         allfwhm = np.zeros(self.nfreq)
#         for i in range(self.nfreq):
#             if self.params["QUBIC"]["convolution_in"] is False:  # and self.params['QUBIC']['reconvolution_after_MM'] is False:
#                 allfwhm[i] = 0
#             elif self.params["QUBIC"]["convolution_in"] is True:  # and self.params['QUBIC']['reconvolution_after_MM'] is False:
#                 allfwhm[i] = self.dict_file["fwhm_rec"][i]

#         return allfwhm

#     def compute_auto_spectrum(self, map, fwhm):
#         """
#         Function to compute the auto-spectrum of a given map

#         Argument :
#             - map(array) [nrec/ncomp, npix, nstokes] : map to compute the auto-spectrum
#             - allfwhm(float) : in radian
#         Return :
#             - (list) [len(ell)] : BB auto-spectrum
#         """

#         DlBB = self.namaster.get_spectra(map=map.T, map2=None, verbose=False, beam_correction=np.rad2deg(fwhm))[1][:, 2]
#         return DlBB

#     def compute_cross_spectrum(self, map1, fwhm1, map2, fwhm2):
#         """
#         Function to compute cross-spectrum, taking into account the different resolution of each sub-bands

#         Arguments :
#             - map1 & map2 (array [nrec/ncomp, npix, nstokes]) : the two maps needed to compute the cross spectrum
#             - fwhm1 & fwhm2 (float) : the respective fwhm for map1 & map2 in radian

#         Return :
#             - (list) [len(ell)] : BB cross-spectrum
#         """

#         # Put the map with the highest resolution at the worst one before doing the cross spectrum
#         # Important because the two maps had to be at the same resolution and you can't increase the resolution
#         if fwhm1 < fwhm2:
#             C = HealpixConvolutionGaussianOperator(fwhm=np.sqrt(fwhm2**2 - fwhm1**2))
#             convoluted_map = C * map1
#             return self.namaster.get_spectra(
#                 map=convoluted_map.T,
#                 map2=map2.T,
#                 verbose=False,
#                 beam_correction=np.rad2deg(fwhm2),
#             )[1][:, 2]
#         else:
#             C = HealpixConvolutionGaussianOperator(fwhm=np.sqrt(fwhm1**2 - fwhm2**2))
#             convoluted_map = C * map2
#             return self.namaster.get_spectra(
#                 map=map1.T,
#                 map2=convoluted_map.T,
#                 verbose=False,
#                 beam_correction=np.rad2deg(fwhm1),
#             )[1][:, 2]

#     def compute_array_power_spectra(self, maps):
#         """
#         Function to fill an array with all the power spectra computed

#         Argument :
#             - maps (array [nreal, nrec/ncomp, npix, nstokes]) : all your realisation maps

#         Return :
#             - power_spectra_array (array [nrec/ncomp, nrec/ncomp]) : element [i, i] is the auto-spectrum for the reconstructed sub-bands i
#                                                                      element [i, j] is the cross-spectrum between the reconstructed sub-band i & j
#         """

#         power_spectra_array = np.zeros((self.nfreq, self.nfreq, len(self.ell)))

#         for i in range(self.nfreq):
#             for j in range(i, self.nfreq):
#                 print(f"====== {self.nus[i]:.0f}x{self.nus[j]:.0f} ======")
#                 if i == j:
#                     # Compute the auto-spectrum
#                     power_spectra_array[i, j] = self.compute_auto_spectrum(maps[i], self.allfwhm[i])
#                     print(power_spectra_array[i, j, :3])
#                     # stop
#                 else:
#                     # Compute the cross-spectrum
#                     power_spectra_array[i, j] = self.compute_cross_spectrum(maps[i], self.allfwhm[i], maps[j], self.allfwhm[j])
#         return power_spectra_array

#     def compute_power_spectra(self):
#         """
#         Function to compute the power spectra array for the sky and for the noise realisations

#         Return :
#             - sky power spectra array (array [nrec/ncomp, nrec/ncomp])
#             - noise power spectra array (array [nrec/ncomp, nrec/ncomp])
#         """

#         sky_power_spectra = self.compute_array_power_spectra(self.sky_maps)
#         noise_power_spectra = self.compute_array_power_spectra(self.noise_maps)
#         return sky_power_spectra, noise_power_spectra

#     def save_data(self, name, d):
#         """

#         Method to save data using pickle convention.

#         """

#         with open(name, "wb") as handle:
#             pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

#     def run(self):
#         self.Dl, self.Nl = self.compute_power_spectra()

#         print("Power spectra computed !!!")

#         spectra_time = time.time() - self.spectra_time_0
#         if comm is None:
#             print(f"Power Spectra computation done in {spectra_time:.3f} s")
#         else:
#             if comm.Get_rank() == 0:
#                 print(f"Power Spectra computation done in {spectra_time:.3f} s")

#         self.save_data(
#             self.path_spectrum + "/" + f"spectrum_{self.jobid}.pkl",
#             {
#                 "nus": self.nus,
#                 "ell": self.ell,
#                 "Dls": self.Dl,
#                 "Nl": self.Nl,
#                 "coverage": self.coverage,
#                 "parameters": self.params,
#                 "duration": spectra_time,
#             },
#         )
