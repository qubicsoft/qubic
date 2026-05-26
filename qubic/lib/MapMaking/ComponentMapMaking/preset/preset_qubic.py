from fgbuster import component_model as c

from qubic.lib.Instrument.Qacquisition import JointAcquisitionComponentsMapMaking
from qubic.lib.MapMaking.ComponentMapMaking import Qcomponent_model as model_co
from qubic.lib.Qdictionary import qubicDict

# import to create qubic_patch
import numpy as np
import healpy as hp
from pysimulators import CartesianEquatorial2GalacticOperator

class PresetQubic:
    """Preset QUBIC.

    Instance to initialize the Components Map-Making. It defines QUBIC variables and methods.

    Parameters
    ----------
    preset_tools : object
        Class containing tools and simulation parameters.
    preset_external : object

    Attributes
    ----------
    dict: dict
        Dictionary defining QUBIC caracteristics.
    joint_in: object
        Class defining the QUBIC intrument to build the intial TOD.
    joint_out: object
        Class defining the QUBIC intrument to reconstruct the sky map.

    """

    def __init__(self, preset_tools, preset_external):
        """Initialize."""
        ### Import preset tools
        self.preset_tools = preset_tools

        ### Define QUBIC parameters variable
        self.params_qubic = self.preset_tools.params["QUBIC"]

        ### MPI common arguments
        self.comm = self.preset_tools.comm
        self.size = self.comm.Get_size()

        ### QUBIC dictionary
        self.preset_tools.mpi._print_message("    => Reading QUBIC dictionary")
        self.dict = self.get_dict()

        ### Define model for reconstruction
        components_fgb_in, _ = self.get_components_fgb(key="in")
        components_fgb_out, _ = self.get_components_fgb(key="out")

        if self.preset_tools.params["Foregrounds"]["CO"]["CO_in"]:
            nu_co = self.preset_tools.params["Foregrounds"]["CO"]["nu0"]
        else:
            nu_co = None

        params_sky = self.preset_tools.params["SKY"]
        ### We want to keep only the useful pixels in memory, in order to be able to go at higher NSIDE
        # it has to be done before creating H
        dist_peak = np.sqrt(2)*np.radians(10)*self.preset_tools.params["QUBIC"]["SYNTHBEAM"]["synthbeam_kmax"] #+ np.radians(10) # the furthest peak from telescope l.o.s.
        radius_patch_rad = np.radians(self.preset_tools.params["QUBIC"]["dtheta"]) + dist_peak  # to make it larger than self.seenpix_qubic
        vec_centre_patch = hp.ang2vec(params_sky["RA_center"], params_sky["DEC_center"], lonlat=True) # equatorial coord
        c2g = CartesianEquatorial2GalacticOperator()
        vec_centre_patch = c2g(vec_centre_patch) # galactic (to match Qsamplings)
        inside_patch = hp.query_disc(params_sky["nside"], vec_centre_patch, radius_patch_rad)
        self.qubic_patch = hp.query_disc(params_sky["nside"], vec_centre_patch, radius_patch_rad * (1 + self.preset_tools.params["QUBIC"]["apod"]))
        # returns unique elements in cell_ids_large not in cell_ids, sorted
        self.apod_patch = np.setdiff1d(self.qubic_patch, inside_patch)
        # del self.qubic_patch
        del inside_patch
        dir_centre_patch = hp.rotator.vec2dir(vec_centre_patch)
        vec_pix_apod = hp.pix2vec(params_sky["nside"], self.apod_patch)
        dir_pix_apod = hp.rotator.vec2dir(vec_pix_apod)
        # angular distance from the centre of the patch
        dist_pix_apod = hp.rotator.angdist(dir_centre_patch, dir_pix_apod)
        # the factor we need to multiply the pixels from the apodisation ring with
        self.cos_apod = 1/2 * (1 + np.cos(np.pi*(dist_pix_apod - radius_patch_rad)/(self.preset_tools.params["QUBIC"]["apod"] * radius_patch_rad)))
        del dist_pix_apod

        ### To test without the patch
        # self.qubic_patch = None

        ### Joint acquisition for QUBIC operator
        self.preset_tools.mpi._print_message("    => Building QUBIC operator")
        self.joint_in = JointAcquisitionComponentsMapMaking(
            self.dict,
            components_fgb_in,
            self.params_qubic["nsub_in"],
            preset_external.external_nus,
            preset_external.params_external["nsub_planck"],
            nu_co=nu_co,
            weight_planck=preset_external.params_external["weight_planck"],
            qubic_patch=self.qubic_patch,
        )

        if self.params_qubic["nsub_in"] == self.params_qubic["nsub_out"]:
            H_tojoint = self.joint_in.qubic.H
        else:
            H_tojoint = None

        self.joint_out = JointAcquisitionComponentsMapMaking(
            self.dict,
            components_fgb_out,
            self.params_qubic["nsub_out"],
            preset_external.external_nus,
            preset_external.params_external["nsub_planck"],
            nu_co=nu_co,
            H=H_tojoint,
            weight_planck=preset_external.params_external["weight_planck"],
            qubic_patch=self.qubic_patch,
        )

    def get_dict(self):
        """QUBIC dictionary.

        Method to modify the qubic dictionary.

        Parameters
        ----------
        key : str, optional
            Can be "in" or "out".
            It is used to build respectively the instances to generate the TODs or to reconstruct the sky maps,
            by default "in".

        Returns
        -------
        dict_qubic: dict
            Modified QUBIC dictionary.

        """

        ### Construct the arguments dictionary with required parameters
        args = {
            "npointings": self.params_qubic["npointings"],
            "nf_recon": 1,
            "nf_sub": self.params_qubic["nsub_in"],
            "nside": self.preset_tools.params["SKY"]["nside"],
            "MultiBand": True,
            "period": 1,
            "RA_center": self.preset_tools.params["SKY"]["RA_center"],
            "DEC_center": self.preset_tools.params["SKY"]["DEC_center"],
            "filter_nu": 150 * 1e9,
            "noiseless": False,
            "comm": self.comm,
            "kind": "IQU",
            "verbose": False,
            "dtheta": self.params_qubic["dtheta"],
            "nprocs_sampling": 1,
            "nprocs_instrument": self.size,
            "photon_noise": True,
            "nhwp_angles": 3,
            #'effective_duration': 3,
            "effective_duration150": self.params_qubic["NOISE"]["duration_150"],
            "effective_duration220": self.params_qubic["NOISE"]["duration_220"],
            "filter_relative_bandwidth": 0.25,  # difference_frequency_nu_over_nu,
            #'type_instrument': 'wide',
            "TemperatureAtmosphere150": None,
            "TemperatureAtmosphere220": None,
            "EmissivityAtmosphere150": None,
            "EmissivityAtmosphere220": None,
            "detector_nep": float(self.params_qubic["NOISE"]["detector_nep"]),
            "synthbeam_kmax": self.params_qubic["SYNTHBEAM"]["synthbeam_kmax"],
            "synthbeam_fraction": self.params_qubic["SYNTHBEAM"]["synthbeam_fraction"],
            "interp_projection": False,
            "instrument_type": self.params_qubic["instrument"],
            "config": self.params_qubic["configuration"],
        }

        ### Get the default dictionary
        dictfilename = "dicts/pipeline_demo.dict"
        d = qubicDict()
        d.read_from_file(dictfilename)

        ### Update the default dictionary with the constructed parameters
        for i in args.keys():
            d[str(i)] = args[i]

        return d

    def get_components_fgb(self, key):
        """Components FGbuster

        Method to build a dictionary containing all the wanted components to generate sky maps.
        Based on FGBuster.

        Returns
        -------
        dict_comps: dict
            Dictionary containing the component instances.

        """

        components = []
        components_name = []

        if self.preset_tools.params["CMB"]["cmb"]:
            components += [c.CMB()]
            components_name += ["CMB"]

        if self.preset_tools.params["Foregrounds"]["Dust"][f"Dust_{key}"]:
            components += [
                c.Dust(
                    nu0=self.preset_tools.params["Foregrounds"]["Dust"]["nu0"],
                    temp=20,
                )
            ]
            components_name += ["Dust"]

        if self.preset_tools.params["Foregrounds"]["Synchrotron"][f"Synchrotron_{key}"]:
            components += [c.Synchrotron(nu0=self.preset_tools.params["Foregrounds"]["Synchrotron"]["nu0"])]
            components_name += ["Synchrotron"]

        if self.preset_tools.params["Foregrounds"]["CO"][f"CO_{key}"]:
            components += [
                # c.COLine(
                #    nu=self.preset_tools.params["Foregrounds"]["CO"]["nu0"],
                #    active=False,
                # )
                model_co.Monochromatic(
                    nu0=self.preset_tools.params["Foregrounds"]["CO"]["nu0"],
                )
            ]
            components_name += ["CO"]

        return components, components_name
