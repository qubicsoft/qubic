import numpy as np

from qubic.lib.Instrument.Qacquisition import QubicAcquisition
from qubic.lib.Instrument.Qinstrument import QubicInstrument


class QubicNoise:
    # gives Qubic noise for one band

    def __init__(self, d, band, sampling, scene, rng_noise, duration, comm=None, size=1):
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

        print(f"Duration at at {band} GHz is {duration} yrs")

    def get_noise(self, det_noise, pho_noise):
        if det_noise:
            n = self.detector_noise()
        else:
            n = np.zeros((len(self.acq.instrument), len(self.acq.sampling)))
        if pho_noise:
            n += self.photon_noise()
        return n

    def photon_noise(self, wpho=1):
        if wpho == 0:
            return np.zeros((len(self.acq.instrument), len(self.acq.sampling)))
        else:
            return wpho * self.acq.get_noise(det_noise=False, photon_noise=True, seed=self.seed_photon)

    def detector_noise(self, wdet=1):
        if wdet == 0:
            return np.zeros((len(self.acq.instrument), len(self.acq.sampling)))
        else:
            return wdet * self.acq.get_noise(det_noise=True, photon_noise=False, seed=self.seed_detector)


class QubicTotNoise:
    ### Gives Qubic noise for all bands: for UWB, they are coadded; for DB it returns two values
    ### For MB, it returns only the 150 band noise

    def __init__(self, d, sampling, scene):
        # we ask for the sampling and scene, as they are already determined when the noise is called
        self.type = d["instrument_type"]
        self.d = d
        self.sampling = sampling
        self.scene = scene
        self.detector_nep = d["detector_nep"]
        if self.type == "DB":  # this will later be implemented at a dictionary level!
            self.band_used = [150, 220]
            self.duration = [d["effective_duration150"], d["effective_duration220"]]
        elif self.type == "UWB":
            self.band_used = [150, 220]
            self.duration = [d["effective_duration150"], d["effective_duration220"]]
        elif self.type == "MB":
            self.band_used = [150]
            self.duration = [d["effective_duration150"]]
            print('---------------------------', self.duration, '--------------------------')
        else:
            raise ValueError("Instrument type {} is not implemented.".format(self.type))

        # if only one duration is given, then it means that both focal planes (if they are two) observed for the same time
        #if isinstance(duration, (int, float)):
        #    self.duration = [duration] * len(self.band_used)
        #else:
        #    if self.type == "UWB" and (duration[0] != duration[1]):
        #        raise TypeError("The duration for bands 150 and 220 has to be the same for the UWB instrument.")
        #    self.duration = duration

    def total_noise(self, wdet, wpho150, wpho220, seed_noise=None):
        rng_noise = np.random.default_rng(seed=seed_noise)  # The way the randomness is treated is NOT GOOD, if doing more than one run (in parallel for example)
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
            npho.append(QnoiseBand.photon_noise(wpho[i]))
            ndet.append(QnoiseBand.detector_noise(wdet))

        if self.type == "UWB":
            return ndet[0] + npho[0] + npho[1]
        elif self.type == "DB":
            return np.r_[ndet[0] + npho[0], ndet[1] + npho[1]]
        elif self.type == "MB":
            return ndet[0] + npho[0]
        else:
            raise TypeError("Can't build noise for instrument type = {}.".format(self.type))
