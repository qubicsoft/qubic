import numpy as np
import qubic

from .Qacquisition import *
from ..Qdictionary import qubicDict
from ..Qsamplings import get_pointing

class QubicNoise:
    # gives Qubic noise for one band

    def __init__(
        self,
        band,
        sampling,
        scene,
        rng_noise,
        comm=None,
        size=1,
        detector_nep=4.7e-17,
        duration=3
    ):

        if band != 150 and band != 220:
            # raise TypeError("Please choose the QubicWideBandNoise method.") # ? Doesn't match the previous implementation
            raise TypeError("Unknown band '{}'.".format(band))

        '''
        NOTE:  the following code should be revisited!
        Why do we read the dictionary here?  It should be an input parameter.
        This is forcing this object to *always* use the pipeline_demo.dict
        '''
        dictfilename = "dicts/pipeline_demo.dict"
        d = qubicDict()
        d.read_from_file(dictfilename)
        self.rng_noise = rng_noise

        d["TemperatureAtmosphere150"] = None
        d["TemperatureAtmosphere220"] = None
        d["EmissivityAtmosphere150"] = None
        d["EmissivityAtmosphere220"] = None
        d["detector_nep"] = detector_nep
        self.npointings = sampling.index.size
        d["npointings"] = self.npointings
        d["comm"] = comm
        d["nprocs_instrument"] = size
        d["nprocs_sampling"] = 1

        self.dict = d.copy()
        self.dict["band"] = int(band)
        self.dict["filter_nu"] = int(band) * 1e9
        self.dict["nf_sub"] = 1
        self.dict["nf_recon"] = 1
        self.dict["period"] = 1
        self.dict["type_instrument"] = ""
        self.dict["effective_duration"] = duration
        # self.acq = QubicIntegrated(self.dict, Nsub=1, Nrec=1)
        self.acq = QubicAcquisition(QubicInstrument(self.dict), sampling, scene, self.dict) # get_pointing(self.dict) et QubicScene(self.dict) ne dépendent pas des params du dictionnaire modifiés ici, donc on peut les calculer à l'extérieur


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
            det_noise=False, photon_noise=True, rng_noise=self.rng_noise
        )

    def detector_noise(self):
        return self.acq.get_noise(
            det_noise=True, photon_noise=False, rng_noise=self.rng_noise
        )

    # unused?
    # def total_noise(self, wdet, wpho):
    #     ndet = wdet * self.detector_noise()
    #     npho = wpho * self.photon_noise()
    #     return ndet + npho


class QubicTotNoise:
    # gives Qubic noise for all bands: for UWB, they are coadded; for DB it returns two values

    def __init__(self, type, d, sampling, scene, detector_nep=4.7e-17, duration=[3]):
        # we ask for the sampling and scene, as they are already determined when the noise is called
        # is type in d?
        self.type = type
        self.d = d
        self.sampling = sampling
        self.scene = scene
        self.detector_nep = detector_nep
        # if only one duration is given, then it means that both bands observed for the same time
        if len(duration) == 1:
            self.duration = [duration[0], duration[0]]
        else:
            if self.type == "UWB" and (duration[0] != duration[1]):
                raise TypeError("The duration for bands 150 and 220 has to be the same for the UWB instruement.")
            self.duration = duration

    def total_noise(self, wdet, wpho150, wpho220, seed_noise=None):
        rng_noise = np.random.default_rng(seed=seed_noise)
        wpho = np.array([wpho150, wpho220])
        npho = []
        ndet = []
        
        for i, band in enumerate([150, 220]):
            QnoiseBand = QubicNoise(
            band,
            self.sampling,
            self.scene,
            rng_noise=rng_noise,
            comm=self.d["comm"],
            size=self.d["nprocs_instrument"],
            detector_nep=self.detector_nep,
            duration=self.duration[i]
            )
            if self.type == "UWB":
                print("UWB")
                npho.append(wpho[i] * QnoiseBand.photon_noise())
                # print(np.shape(npho[i]))
                # sys.exit()
                ndet.append(wdet * QnoiseBand.detector_noise())
            elif self.type == "DB":
                print("DB")
                npho.append(wpho[i] * QnoiseBand.photon_noise().ravel())#?
                ndet.append(wdet * QnoiseBand.detector_noise().ravel())#?
                # print(np.shape(npho[i]))
                # sys.exit()
        if self.type == "UWB":
            return ndet[0] + npho[0] + npho[1]
        elif self.type == "DB":
            return np.r_[ndet[0] + npho[0], ndet[1] + npho[1]]
        else:
            raise TypeError("Can't build noise for instrument type = {}.".format(self.type))