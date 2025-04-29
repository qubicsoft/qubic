# import numpy as np
# import qubic

# from .Qacquisition import *
# from ..Qdictionary import qubicDict

# class QubicNoise:

#     def __init__(
#         self,
#         band,
#         npointings,
#         comm=None,
#         size=1,
#         detector_nep=4.7e-17,
#         seed_noise=None,
#         duration=3,
#     ):

#         if band != 150 and band != 220:
#             raise TypeError("Please choose the QubicWideBandNoise method.")

#         '''
#         NOTE:  the following code should be revisited!
#         Why do we read the dictionary here?  It should be an input parameter.
#         This is forcing this object to *always* use the pipeline_demo.dict
#         '''
#         dictfilename = "dicts/pipeline_demo.dict"
#         d = qubicDict()
#         d.read_from_file(dictfilename)
#         self.seed_noise = seed_noise

#         d["TemperatureAtmosphere150"] = None
#         d["TemperatureAtmosphere220"] = None
#         d["EmissivityAtmosphere150"] = None
#         d["EmissivityAtmosphere220"] = None
#         d["detector_nep"] = detector_nep
#         self.npointings = npointings
#         d["npointings"] = npointings
#         d["comm"] = comm
#         d["nprocs_instrument"] = size
#         d["nprocs_sampling"] = 1
#         d["interp_projection"] = False

#         self.dict = d.copy()
#         self.dict["filter_nu"] = int(band)
#         self.dict["nf_sub"] = 1
#         self.dict["nf_recon"] = 1
#         self.dict["period"] = 1
#         self.dict["type_instrument"] = ""
#         self.dict["effective_duration"] = duration

#         '''
#         NOTE: the following code must be modified
#         the QubicIntegrated is to be replaced by QubicAcquisition, but with the correct arguments
#         self.acq = QubicAcquisition(instrument, sampling, scene, self.dict)
#         '''
#         self.acq = QubicIntegrated(self.dict, Nsub=1, Nrec=1)
#         print(f"Duration at {band} GHz is {duration} yrs")

#     def get_noise(self, det_noise, pho_noise):
#         n = self.detector_noise() * 0

#         if det_noise:
#             n += self.detector_noise()
#         if pho_noise:
#             n += self.photon_noise()
#         return n

#     def photon_noise(self):
#         return self.acq.get_noise(
#             det_noise=False, photon_noise=True, seed=self.seed_noise
#         )

#     def detector_noise(self):
#         return self.acq.get_noise(
#             det_noise=True, photon_noise=False, seed=self.seed_noise
#         )

#     def total_noise(self, wdet, wpho):
#         ndet = wdet * self.detector_noise()
#         npho = wpho * self.photon_noise()
#         return ndet + npho


# class QubicWideBandNoise:

#     def __init__(self, d, npointings, detector_nep=4.7e-17, duration=3):

#         self.d = d
#         self.npointings = npointings
#         self.detector_nep = detector_nep
#         self.duration = duration

#     def total_noise(self, wdet, wpho150, wpho220, seed_noise=None):

#         Qubic150 = QubicNoise(
#             150,
#             self.npointings,
#             comm=self.d["comm"],
#             size=self.d["nprocs_instrument"],
#             detector_nep=self.detector_nep,
#             seed_noise=seed_noise,
#             duration=self.duration,
#         )
#         Qubic220 = QubicNoise(
#             220,
#             self.npointings,
#             comm=self.d["comm"],
#             size=self.d["nprocs_instrument"],
#             detector_nep=self.detector_nep,
#             seed_noise=seed_noise + 1,
#             duration=self.duration,
#         )

#         ndet = wdet * Qubic150.detector_noise()
#         npho150 = wpho150 * Qubic150.photon_noise()
#         npho220 = wpho220 * Qubic220.photon_noise()

#         return ndet + npho150 + npho220


# class QubicDualBandNoise:

#     def __init__(self, d, npointings, detector_nep=4.7e-17, duration=[3, 3]):

#         self.d = d
#         self.npointings = npointings
#         self.detector_nep = detector_nep
#         self.duration = duration

#     def total_noise(self, wdet, wpho150, wpho220, seed_noise=None):

#         Qubic150 = QubicNoise(
#             150,
#             self.npointings,
#             comm=self.d["comm"],
#             size=self.d["nprocs_instrument"],
#             detector_nep=self.detector_nep,
#             seed_noise=seed_noise,
#             duration=self.duration[0],
#         )
#         Qubic220 = QubicNoise(
#             220,
#             self.npointings,
#             comm=self.d["comm"],
#             size=self.d["nprocs_instrument"],
#             detector_nep=self.detector_nep,
#             seed_noise=seed_noise + 1,
#             duration=self.duration[1],
#         )

#         ndet150 = wdet * Qubic150.detector_noise().ravel()
#         ndet220 = wdet * Qubic220.detector_noise().ravel()
#         npho150 = wpho150 * Qubic150.photon_noise().ravel()
#         npho220 = wpho220 * Qubic220.photon_noise().ravel()

#         return np.r_[ndet150 + npho150, ndet220 + npho220]


import numpy as np
import qubic

from .Qacquisition import *
from ..Qdictionary import qubicDict

class QubicNoise:
    # gives Qubic noise for one band

    def __init__(
        self,
        d,
        band,
        sampling,
        scene,
        rng_noise,
        duration,
        comm=None,
        size=1,
    ):

        if band != 150 and band != 220:
            raise TypeError("Unknown band '{}'.".format(band))

        self.rng_noise = rng_noise

        self.seed_photon = rng_noise.integers(10000000)
        self.seed_detector = rng_noise.integers(10000000)

        self.dict = d.copy()

        self.dict["TemperatureAtmosphere150"] = None
        self.dict["TemperatureAtmosphere220"] = None
        self.dict["EmissivityAtmosphere150"] = None
        self.dict["EmissivityAtmosphere220"] = None
        self.dict["comm"] = comm
        self.dict["nprocs_instrument"] = size
        self.dict["nprocs_sampling"] = 1

        self.dict["band"] = int(band)
        self.dict["filter_nu"] = int(band) * 1e9
        self.dict["nf_sub"] = 1
        self.dict["nf_recon"] = 1
        self.dict["period"] = 1
        self.dict["type_instrument"] = ""
        self.dict["effective_duration"] = duration
        # get_pointing(self.dict) and QubicScene(self.dict) don't depend on the params modified here, they can be computed outside the class
        self.acq = QubicAcquisition(QubicInstrument(self.dict), sampling, scene, self.dict)


        print(f"Duration at {band} GHz is {duration} yrs")

    def get_noise(self, det_noise, pho_noise):
        n = self.detector_noise() * 0

        if det_noise:
            n += self.detector_noise()
        if pho_noise:
            n += self.photon_noise()
        return n

    def photon_noise(self):
        return self.acq.get_noise(
            det_noise=False, photon_noise=True, seed=self.seed_photon
        )

    def detector_noise(self):
        return self.acq.get_noise(
            det_noise=True, photon_noise=False, seed=self.seed_detector
        )


class QubicTotNoise:
    ### Gives Qubic noise for all bands: for UWB, they are coadded; for DB it returns two values
    ### For MB, it returns only the 150 band noise

    def __init__(self, d, sampling, scene, duration=3):
        # we ask for the sampling and scene, as they are already determined when the noise is called
        self.type = d["instrument_type"]
        self.d = d
        self.sampling = sampling
        self.scene = scene
        self.detector_nep = d["detector_nep"]
        if self.type == "DB": # this will later be implemented at a dictionary level!
            self.band_used = [150, 220]
        elif self.type == "UWB":
            self.band_used = [150, 220]
        elif self.type == "MB":
            self.band_used = [150]
        else:
            raise ValueError("Instrument type {} is not implemented.".format(self.type))
        
        # if only one duration is given, then it means that both focal planes (if they are two) observed for the same time
        if isinstance(duration, (int, float)):
            self.duration = [duration] * len(self.band_used)
        else:
            if self.type == "UWB" and (duration[0] != duration[1]):
                raise TypeError("The duration for bands 150 and 220 has to be the same for the UWB instrument.")
            self.duration = duration

    def total_noise(self, wdet, wpho150, wpho220, seed_noise=None):
        rng_noise = np.random.default_rng(seed=seed_noise) # The way the randomness is treated is NOT GOOD, if doing more than one run (in parallel for example)
        wpho = np.array([wpho150, wpho220])
        npho = []
        ndet = []
        
        for i, band in enumerate(self.band_used):
            QnoiseBand = QubicNoise(
            self.d,
            band,
            self.sampling,
            self.scene,
            rng_noise=rng_noise,
            duration=self.duration[i],
            comm=self.d["comm"],
            size=self.d["nprocs_instrument"],
            )
            npho.append(wpho[i] * QnoiseBand.photon_noise())
            ndet.append(wdet * QnoiseBand.detector_noise())
            # if self.type == "UWB":
            #     print("UWB")
            #     npho.append(wpho[i] * QnoiseBand.photon_noise())
            #     # print(np.shape(npho[i]))
            #     # sys.exit()
            #     ndet.append(wdet * QnoiseBand.detector_noise())
            # elif self.type == "DB":
            #     print("DB")
            #     npho.append(wpho[i] * QnoiseBand.photon_noise().ravel())#?
            #     ndet.append(wdet * QnoiseBand.detector_noise().ravel())#?
            #     # print(np.shape(npho[i]))
            #     # sys.exit()
        if self.type == "UWB":
            return ndet[0] + npho[0] + npho[1]
        elif self.type == "DB":
            return np.r_[ndet[0] + npho[0], ndet[1] + npho[1]]
        elif self.type == "MB":
            return ndet[0] + npho[0]
        else:
            raise TypeError("Can't build noise for instrument type = {}.".format(self.type))