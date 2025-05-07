import healpy as hp
import numpy as np
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

from ....Fitting import Qnamaster as nam
from ....Qsamplings import equ2gal


class PresetSky:
    r"""Preset Sky.

    Instance to initialize the Components Map-Making. It defines the observed sky variables and methods.

    Parameters
    ----------
    preset_tools : object
        Class containing tools and simulation parameters.
    preset_qubic : object
        Class containing qubic operator and variables and methods.

    Attributes
    ----------
    center: tuple
        Defines the center of the QUBIC observation patch in (RA, DEC).
    coverage: array_like
        Defines the coverage of the QUBIC scanning strategy.
    max_coverage: float
        Defines the maximum of the coverage.
    seenpix_qubic: array_like
        Boolean array defining the pixels seen by QUBIC instrument.
    seenpix: array_like
        Boolean array defining the pixel with coverage > at a threshold chosen in params.yml.
    namaster: object
        NaMaster class instance.
    ell: array_like
        Multipole array, defining through NaMaster.
    cl2dl: array_like
        Array containing conversion factor to compute :math:`D_{\ell}` from :math:`C_{\ell}`.

    """

    def __init__(self, preset_tools, preset_qubic):
        """
        Initialize.

        """
        ### Import preset tools
        self.preset_tools = preset_tools

        ### Define variable for SKY parameters
        self.params_sky = self.preset_tools.params["SKY"]

        ### Center of the QUBIC patch
        self.preset_tools.mpi._print_message("    => Getting center of the QUBIC patch")
        self.center = equ2gal(
            self.params_sky["RA_center"], self.params_sky["DEC_center"]
        )

        ### Compute coverage map
        self.preset_tools.mpi._print_message("    => Computing coverage")
        self.coverage = preset_qubic.joint_out.qubic.coverage
        self.max_coverage = np.max(self.coverage)

        ### Compute seen pixels
        self.preset_tools.mpi._print_message("    => Computing cut between Planck & QUBIC")

        # All the pixels seen by QUBIC
        self.seenpix_qubic = self.coverage / self.max_coverage > 0
        self.seenpix_015 = self.coverage / self.max_coverage > 0.15

        # Pixels seen enough by QUBIC, according to the threshold defined in params.yml. The others will be replaced by Planck
        self.seenpix = (
            self.coverage / self.max_coverage
            > self.preset_tools.params["PLANCK"]["thr_planck"]
        )

        ### Define the map of betas across the patch if 'nside_beta_out' != 0
        if self.preset_tools.params["Foregrounds"]["Dust"]["nside_beta_out"] != 0:
            # Build super-pixels seen by QUBIC from maps pixels
            self.seenpix_beta = hp.ud_grade(
                self.seenpix,
                self.preset_tools.params["Foregrounds"]["Dust"]["nside_beta_out"],
            )
            # Compute their coverage
            self.coverage_beta = self.get_coverage()
        else:
            self.coverage_beta = None

        ### Build mask for weighted Planck data
        self.preset_tools.mpi._print_message("    => Creating mask")
        self.mask = np.ones(12 * self.params_sky["nside"] ** 2)
        self.mask[self.seenpix] = self.preset_tools.params["PLANCK"]["weight_planck"]
        C = HealpixConvolutionGaussianOperator(
            fwhm=self.preset_tools.params["PLANCK"]["fwhm_weight_planck"],
            lmax=3*self.params_sky["nside"] - 1,
        )
        self.mask = C(self.mask)

        ### Initialize namaster
        self.preset_tools.mpi._print_message("    => Initializing Namaster")
        self.get_spectra_namaster_informations()

    def get_coverage(self):
        """Coverage.

        Calculate the coverage mask for the QUBIC patch, according with the number of beta that you want to reconstruct.

        This function computes the angular distance between a center point and all
        pixels on a sphere, sorts these distances, and creates a mask that marks
        the closest pixels as covered.

        Returns
        -------
        mask: array_like
            A mask array where covered pixels are marked with 1 and uncovered pixels are marked with 0.

        """

        vec_center = np.array(hp.ang2vec(self.center[0], self.center[1], lonlat=True))
        vec_pix = np.array(
            hp.pix2vec(
                self.preset_tools.params["Foregrounds"]["Dust"]["nside_beta_out"],
                np.arange(
                    12
                    * self.preset_tools.params["Foregrounds"]["Dust"]["nside_beta_out"]
                    ** 2
                ),
            )
        )

        angle = np.arccos(np.dot(vec_center, vec_pix))
        indices = np.argsort(angle)

        mask = np.zeros(
            12 * self.preset_tools.params["Foregrounds"]["Dust"]["nside_beta_out"] ** 2
        )
        pix_inside_patch = indices[
            : self.preset_tools.params["Foregrounds"]["Dust"]["nside_beta_in"]
        ]
        mask[pix_inside_patch] = 1

        return mask

    def get_spectra_namaster_informations(self):
        """NaMaster.

        Initializes the Namaster object and computes the ell and cl2dl arrays.

        This method sets up the Namaster object using parameters from the preset_tools
        and computes the ell array and cl2dl conversion factor for power spectrum analysis.

        """

        self.namaster = nam.Namaster(
            self.seenpix,
            lmin=self.preset_tools.params["SPECTRUM"]["lmin"],
            lmax=3 * self.preset_tools.params["SKY"]["nside"],
            delta_ell=self.preset_tools.params["SPECTRUM"]["dl"],
            aposize=self.preset_tools.params["SPECTRUM"]["aposize"],
        )
        self.ell, _ = self.namaster.get_binning(
            self.preset_tools.params["SKY"]["nside"]
        )
        self.cl2dl = self.ell * (self.ell + 1) / (2 * np.pi)
