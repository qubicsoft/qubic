import numpy as np
import qubic

from .Qacquisition import *
from ..Qdictionary import qubicDict, rewrite_dict

class QubicNoise:

    def __init__(
        self,
        band,
        npointings,
        comm=None,
        size=1,
        detector_nep=4.7e-17,
        seed_noise=None,
        duration=3,
    ):

        if band != 150 and band != 220:
            raise TypeError("Please choose the QubicWideBandNoise method.")

        '''
        NOTE:  the following code should be revisited!
        Why do we read the dictionary here?  It should be an input parameter.
        This is forcing this object to *always* use the pipeline_demo.dict
        '''
        
        args = {"detector_nep": detector_nep,
                "npointings": npointings,
                "comm": comm,
                "nprocs_instrument": size,
                "filter_nu": int(band),
                "nf_sub": 1,
                "nf_recon": 1,
                "type_instrument": "",
                "effective_duration": duration}
        
        self.dict = rewrite_dict(args=args, dict_filename="dicts/pipeline_fmm.dict")
        
        self.seed_noise = seed_noise
        self.npointings = npointings

        '''
        NOTE: the following code must be modified
        the QubicIntegrated is to be replaced by QubicAcquisition, but with the correct arguments
        self.acq = QubicAcquisition(instrument, sampling, scene, self.dict)
        '''
        self.acq = QubicIntegrated(self.dict, Nsub=1, Nrec=1)
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
            det_noise=False, photon_noise=True, seed=self.seed_noise
        )

    def detector_noise(self):
        return self.acq.get_noise(
            det_noise=True, photon_noise=False, seed=self.seed_noise
        )

    def total_noise(self, wdet, wpho):
        ndet = wdet * self.detector_noise()
        npho = wpho * self.photon_noise()
        return ndet + npho


class QubicWideBandNoise:

    def __init__(self, d, npointings, detector_nep=4.7e-17, duration=3):

        self.d = d
        self.npointings = npointings
        self.detector_nep = detector_nep
        self.duration = duration

    def total_noise(self, wdet, wpho150, wpho220, seed_noise=None):

        Qubic150 = QubicNoise(
            150,
            self.npointings,
            comm=self.d["comm"],
            size=self.d["nprocs_instrument"],
            detector_nep=self.detector_nep,
            seed_noise=seed_noise,
            duration=self.duration,
        )
        Qubic220 = QubicNoise(
            220,
            self.npointings,
            comm=self.d["comm"],
            size=self.d["nprocs_instrument"],
            detector_nep=self.detector_nep,
            seed_noise=seed_noise + 1,
            duration=self.duration,
        )

        ndet = wdet * Qubic150.detector_noise()
        npho150 = wpho150 * Qubic150.photon_noise()
        npho220 = wpho220 * Qubic220.photon_noise()

        return ndet + npho150 + npho220


class QubicDualBandNoise:

    def __init__(self, d, npointings, detector_nep=4.7e-17, duration=[3, 3]):

        self.d = d
        self.npointings = npointings
        self.detector_nep = detector_nep
        self.duration = duration

    def total_noise(self, wdet, wpho150, wpho220, seed_noise=None):

        Qubic150 = QubicNoise(
            150,
            self.npointings,
            comm=self.d["comm"],
            size=self.d["nprocs_instrument"],
            detector_nep=self.detector_nep,
            seed_noise=seed_noise,
            duration=self.duration[0],
        )
        Qubic220 = QubicNoise(
            220,
            self.npointings,
            comm=self.d["comm"],
            size=self.d["nprocs_instrument"],
            detector_nep=self.detector_nep,
            seed_noise=seed_noise + 1,
            duration=self.duration[1],
        )

        ndet150 = wdet * Qubic150.detector_noise().ravel()
        ndet220 = wdet * Qubic220.detector_noise().ravel()
        npho150 = wpho150 * Qubic150.photon_noise().ravel()
        npho220 = wpho220 * Qubic220.photon_noise().ravel()

        return np.r_[ndet150 + npho150, ndet220 + npho220]
