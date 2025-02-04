import numpy as np
import qubic

from .Qacquisition import *
from ..Qdictionary import qubicDict

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
