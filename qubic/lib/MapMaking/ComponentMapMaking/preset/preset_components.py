import os

import healpy as hp
import numpy as np
import pysm3
import pysm3.units as u
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
from pysm3 import utils

from qubic.data import PATH


class PresetComponents:
    """Preset Components.

    Instance to initialize the Components Map-Making. It defines the Foregrounds variables and methods.

    Parameters
    ----------
    seed_noise : int
        Seed for random CMB noise generation.
    preset_tools : object
        Class containing tools and simulation parameters.
    preset_qubic : object
        Class containing qubic operator and variables and methods.

    Attributes
    ----------
    params_cmb: dict
        Dictionary containing the parameters associated with CMB.
    params_foregrounds: dict
        Dictionary containing the parameters associated with foregrounds.
    seed: int
        Seed for random CMB noise generation.
    skyconfig: dict
        Dictionary containing the wanted sky configuration for PySM.
    components_model: list
        List containing the FGBuster instance relative to the wanted components.
    components_name: list
        List containing the name of the components.
    components: array_like
        Components maps according to chosen model.
    components_convolved: array_like
        Convolved components maps.
    components_iter: array_like
        Initilize array on which we will iterate to reconstruct components maps.
    nu_co: int
        Frequency of the monochromatic emission, None if not chosen.

    """

    def __init__(self, preset_tools, preset_qubic):
        """
        Initialize.

        """
        ### Import preset QUBIC & tools
        self.preset_tools = preset_tools
        self.preset_qubic = preset_qubic

        ### Define variable for Foregrounds parameters
        self.params_cmb = self.preset_tools.params["CMB"]
        self.params_foregrounds = self.preset_tools.params["Foregrounds"]

        ### Define seed for CMB generation and noise
        #self.seed = seed

        ### Skyconfig
        self.preset_tools._print_message("    => Creating sky configuration")
        self.skyconfig_in = self.get_sky_config(key="in")
        self.skyconfig_out = self.get_sky_config(key="out")

        ### Define model for reconstruction
        self.preset_tools._print_message("    => Creating model")
        self.components_model_in, self.components_name_in = (
            self.preset_qubic.get_components_fgb(key="in")
        )
        self.components_model_out, self.components_name_out = (
            self.preset_qubic.get_components_fgb(key="out")
        )

        ### Compute true components
        self.preset_tools._print_message("    => Creating components")
        self.components_in, self.components_convolved_in, _ = self.get_components(
            self.skyconfig_in
        )
        self.components_out, self.components_convolved_out, self.components_iter = (
            self.get_components(self.skyconfig_out)
        )

        ### Monochromatic emission
        if self.preset_tools.params["Foregrounds"]["CO"]["CO_in"]:
            self.nu_co = self.preset_tools.params["Foregrounds"]["CO"]["nu0_co"]
        else:
            self.nu_co = None

    def get_sky_config(self, key):
        """Sky configuration.

        Method to define the sky model used by PySM3 to generate a fake sky.

        Parameters
        ----------
        key : str
            The key to access the specific parameters in the prest configuration.
            Can be either "in" or "out".

        Returns
        -------
        skyconfig: dict
            Dictionary containing the sky model configuration.

        Example
        -------
        sky = {'cmb': 42, 'Dust': 'd0'}

        """

        sky = {}
        if self.params_cmb["cmb"]:
            sky["CMB"] = self.preset_tools.params['CMB']['seed']

        if self.preset_tools.params["Foregrounds"]["Dust"][f"Dust_{key}"]:
            sky["Dust"] = self.preset_tools.params["Foregrounds"]["Dust"]["model_d"]

        if self.preset_tools.params["Foregrounds"]["Synchrotron"][f"Synchrotron_{key}"]:
            sky["Synchrotron"] = self.preset_tools.params["Foregrounds"]["Synchrotron"][
                "model_s"
            ]

        if self.preset_tools.params["Foregrounds"]["CO"][f"CO_{key}"]:
            sky["coline"] = "co2"

        return sky

    def give_cl_cmb(self, r=0, Alens=1.0):
        r""":math:`C_{\ell}^{BB}` CMB.

        Generates the CMB BB power spectrum with optional lensing and tensor contributions.

        Parameters
        ----------
        r : int, optional
            Tensor-to-scalar ratio, by default 0
        Alens : float, optional
            Lensing amplitude, by default 1.0

        Returns
        -------
        power_spectrum: array_like
            CMB power spectrum according to r and Alens.

        """

        # Read the lensed scalar power spectrum from the FITS file
        power_spectrum = hp.read_cl(PATH + "Cls_Planck2018_lensed_scalar.fits")[
            :, :4000
        ]

        # Adjust the lensing amplitude if Alens is not the default value
        if Alens != 1.0:
            power_spectrum[2] *= Alens

        # Add tensor contributions if r is not zero
        if r:
            power_spectrum += (
                r
                * hp.read_cl(
                    PATH + "Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits"
                )[:, :4000]
            )

        return power_spectrum

    def polarized_I(self, m, nside, polarization_fraction=0):
        """Polarized intensity map.

        Calculates the polarized intensity map.

        Parameters
        ----------
        m : array_like
            Input map to be polarised
        nside : int
            Nside parameter of the HEALPix map.
        polarization_fraction : float, optional
            Fraction of polarization, by default 0

        Returns
        -------
        p_map: array_like
            Array containing the polarized intensity map with cosine and sine components.

        """

        # Read and downgrade the polarization angle map to the desired nside resolution
        polangle = hp.ud_grade(
            hp.read_map(PATH + "psimap_dust90_512.fits"), nside
        )

        # Read and downgrade the depolarization map to the desired nside resolution
        depolmap = hp.ud_grade(
            hp.read_map(PATH + "gmap_dust90_512.fits"), nside
        )

        # Calculate the cosine of twice the polarization angle
        cospolangle = np.cos(2.0 * polangle)

        # Calculate the sine of twice the polarization angle
        sinpolangle = np.sin(2.0 * polangle)

        # Calculate the polarized intensity map by scaling the input map with the depolarization map and polarization fraction
        p_map = polarization_fraction * depolmap * hp.ud_grade(m, nside)

        # Return the polarized intensity map with cosine and sine components
        return p_map * np.array([cospolangle, sinpolangle])

    def get_components(self, skyconfig):
        """Components maps.

        Read configuration dictionary which contains every compoenent and their associated model.
        The CMB is randomly generated from a specific seed.
        Astrophysical foregrounds come from PySM 3.

        Parameters
        ----------
        skyconfig : dict
            Dictionary containing the configuration for each component.

        Returns
        -------
        components: array_like
            Components maps according to chosen model.
        components_convolved: array_like
            Convolved components maps.
        components_iter: array_like
            Initilize array on which we will iterate to reconstruct components maps.

        Raises
        ------
        TypeError
            Raises if the chosen model does not exist.

        """

        ### Initialization
        components = np.zeros(
            (len(skyconfig), 12 * self.preset_tools.params["SKY"]["nside"] ** 2, 3)
        )
        components_convolved = np.zeros(
            (len(skyconfig), 12 * self.preset_tools.params["SKY"]["nside"] ** 2, 3)
        )

        ### Compute convolution operator if needed
        if (
            self.preset_qubic.params_qubic["convolution_in"]
            or self.preset_qubic.params_qubic["convolution_out"]
        ):
            C = HealpixConvolutionGaussianOperator(
                fwhm=self.preset_qubic.joint_in.qubic.allfwhm[-1],
                lmax=3 * self.preset_tools.params["SKY"]["nside"],
            )
        else:
            C = HealpixConvolutionGaussianOperator(fwhm=0)

        ### Compute CMB power spectrum according Planck data
        mycls = self.give_cl_cmb(r=self.params_cmb["r"], Alens=self.params_cmb["Alens"])

        ### Build components list
        for icomp, comp_name in enumerate(skyconfig.keys()):
            # CMB case
            if comp_name == "CMB":
                np.random.seed(skyconfig[comp_name])
                cmb = hp.synfast(
                    mycls,
                    self.preset_tools.params["SKY"]["nside"],
                    verbose=False,
                    new=True,
                ).T
                components[icomp] = cmb.copy()
                components_convolved[icomp] = C(cmb).copy()

            # Dust case
            elif comp_name == "Dust":
                sky_dust = pysm3.Sky(
                    nside=self.preset_tools.params["SKY"]["nside"],
                    preset_strings=[
                        self.preset_tools.params["Foregrounds"]["Dust"]["model_d"]
                    ],
                    output_unit="uK_CMB",
                )

                sky_dust.components[0].mbb_temperature = (
                    20 * sky_dust.components[0].mbb_temperature.unit
                )
                map_Dust = (
                    np.array(
                        sky_dust.get_emission(
                            self.preset_tools.params["Foregrounds"]["Dust"]["nu0_d"]
                            * u.GHz,
                            None,
                        ).T
                        * utils.bandpass_unit_conversion(
                            self.preset_tools.params["Foregrounds"]["Dust"]["nu0_d"]
                            * u.GHz,
                            None,
                            u.uK_CMB,
                        )
                    )
                    * self.preset_tools.params["Foregrounds"]["Dust"]["amplification_d"]
                )
                components[icomp] = map_Dust.copy()
                components_convolved[icomp] = C(map_Dust).copy()

            # Synchrotron case
            elif comp_name == "Synchrotron":
                sky_sync = pysm3.Sky(
                    nside=self.preset_tools.params["SKY"]["nside"],
                    preset_strings=[
                        self.preset_tools.params["Foregrounds"]["Synchrotron"][
                            "model_s"
                        ]
                    ],
                    output_unit="uK_CMB",
                )

                map_sync = (
                    np.array(
                        sky_sync.get_emission(
                            self.preset_tools.params["Foregrounds"]["Synchrotron"][
                                "nu0_s"
                            ]
                            * u.GHz,
                            None,
                        ).T
                        * utils.bandpass_unit_conversion(
                            self.preset_tools.params["Foregrounds"]["Synchrotron"][
                                "nu0_s"
                            ]
                            * u.GHz,
                            None,
                            u.uK_CMB,
                        )
                    )
                    * self.preset_tools.params["Foregrounds"]["Synchrotron"][
                        "amplification_s"
                    ]
                )
                components[icomp] = map_sync.copy()
                components_convolved[icomp] = C(map_sync).copy()

            # CO emission case
            elif comp_name == "coline":
                map_co = hp.ud_grade(
                    hp.read_map(PATH + "CO_line.fits") * 10,
                    self.preset_tools.params["SKY"]["nside"],
                )
                map_co_polarised = self.polarized_I(
                    map_co,
                    self.preset_tools.params["SKY"]["nside"],
                    polarization_fraction=self.preset_tools.params["Foregrounds"]["CO"][
                        "polarization_fraction"
                    ],
                )
                sky_co = np.zeros(
                    (12 * self.preset_tools.params["SKY"]["nside"] ** 2, 3)
                )
                sky_co[:, 0] = map_co.copy()
                sky_co[:, 1:] = map_co_polarised.T.copy()
                components[icomp] = sky_co.copy()
                components_convolved[icomp] = C(sky_co).copy()

            else:
                raise TypeError("Choose right foreground model (d0, s0, ...)")

        # if self.preset_tools.params['Foregrounds']['Dust']['nside_beta_out'] != 0:
        #     components = components.T.copy()
        components_iter = components.copy()

        return components, components_convolved, components_iter
