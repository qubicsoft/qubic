from fgbuster.component_model import Dust, CMB
from .. import QcomponentModel.Monochromatic as Monochromatic
from ....Instrument.Qacquisition import JointAcquisitionComponentsMapMaking
from ....Qdictionary import qubicDict


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
            nu_co = self.preset_tools.params["Foregrounds"]["CO"]["nu0_co"]
        else:
            nu_co = None

        ### Joint acquisition for QUBIC operator
        self.preset_tools.mpi._print_message("    => Building QUBIC operator")
        self.joint_in = JointAcquisitionComponentsMapMaking(
            self.dict,
            self.params_qubic["instrument"],
            components_fgb_in,
            self.params_qubic["nsub_in"],
            preset_external.external_nus,
            preset_external.params_external["nintegr_planck"],
            nu_co=nu_co,
        )

        if self.params_qubic["nsub_in"] == self.params_qubic["nsub_out"]:
            self.joint_out = JointAcquisitionComponentsMapMaking(
                self.dict,
                self.params_qubic["instrument"],
                components_fgb_out,
                self.params_qubic["nsub_in"],
                preset_external.external_nus,
                preset_external.params_external["nintegr_planck"],
                nu_co=nu_co,
                H=self.joint_in.qubic.H,
            )
        else:
            self.joint_out = JointAcquisitionComponentsMapMaking(
                self.dict,
                self.params_qubic["instrument"],
                components_fgb_out,
                self.params_qubic["nsub_out"],
                preset_external.external_nus,
                preset_external.params_external["nintegr_planck"],
                nu_co=nu_co,
                H=None,
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
            "config": "FI",
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
            components += [CMB()]
            components_name += ["CMB"]

        if self.preset_tools.params["Foregrounds"]["Dust"][f"Dust_{key}"]:
            components += [
                Dust(
                    nu0=self.preset_tools.params["Foregrounds"]["Dust"]["nu0_d"],
                    temp=20,
                )
            ]
            components_name += ["Dust"]

        if self.preset_tools.params["Foregrounds"]["Synchrotron"][f"Synchrotron_{key}"]:
            components += [
                c.Synchrotron(
                    nu0=self.preset_tools.params["Foregrounds"]["Synchrotron"]["nu0_s"]
                )
            ]
            components_name += ["Synchrotron"]

        if self.preset_tools.params["Foregrounds"]["CO"][f"CO_{key}"]:
            components += [
                #c.COLine(
                #    nu=self.preset_tools.params["Foregrounds"]["CO"]["nu0_co"],
                #    active=False,
                #)
                Monochromatic(
                    nu0=self.preset_tools.params["Foregrounds"]["CO"]["nu0_co"],
                )
            ]
            components_name += ["CO"]

        return components, components_name
