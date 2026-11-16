import numpy as np
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

from qubic.lib.Fitting import Qnamaster as nam
from qubic.lib.Qhdf5 import HDF5Dict


class Spectra:
    def __init__(self, filename):
        self.filename = filename

        ### Get the job ID
        self.jobid = filename.split("/")[-1].split("_")[-1][:-4]

        self.dictionary = self.open_file()
        self.maps = self.dictionary["maps"].copy()

        self.seenpix = self.dictionary["seenpix"]
        self.fwhm = self.dictionary["fwhm_rec"]

        ### Initiate namaster from qubic soft
        self._init_namaster()

    def open_file(self):
        dictionary = HDF5Dict().load_dict(self.filename)
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
        self.namaster.get_binning(self.sky["nside"])
        self.ell = self.namaster.ell_binned
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
