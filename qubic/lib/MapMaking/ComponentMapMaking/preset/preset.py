from .preset_acquisition import *
from .preset_external_data import *
from .preset_components import *
from .preset_gain import *
from .preset_mixingmatrix import *
from .preset_qubic import *
from .preset_sky import *
from .preset_tools import *
from ....Qfoldertools import create_folder_if_not_exists


class PresetInitialisation:
    """Preset initialization.

    Instance to initialize the Components Map-Making reading all the preset files.

    Arguments
    ---------
    comm: MPI communicator
        MPI common communicator (define by MPI.COMM_WORLD).
    seed: int
        Seed for random CMB realization.
    seed_noise: int
        Seed for random noise realization.

    """

    def __init__(self, comm, seed_noise):
        """
        Initialize the class with MPI communication.

        """

        ### MPI common arguments
        self.comm = comm
        self.seed_noise = seed_noise
        
        self.tools = None
        self.qubic = None
        self.fg = None
        self.gain = None
        self.acquisition = None
        self.external = None
        self.sky = None
        self.mixingmatrix = None

    def initialize(self, parameters_file):
        """Initialization.

        Method that initialize and return all the preset instances that will be used in Component Map-Making Pipeline.

        """
        self.tools = PresetTools(self.comm, parameters_file)
        self.tools.mpi._print_message("========= Initialization =========")
        self.tools.mpi._print_message("    => Checking simulation parameters")
        self.tools.check_for_errors()
        self.tools.mpi._print_message("    => No error detected !")
        self.tools.mpi._print_message("    => Getting job ID")
        self.job_id = os.environ.get("SLURM_JOB_ID")
        self.tools.mpi._print_message("    => Creating folders")
        if self.tools.rank == 0:
            if self.tools.params["save_iter"] != 0:
                self.tools.params["foldername"] = (
                    f"{self.tools.params['Foregrounds']['Dust']['type']}_{self.tools.params['Foregrounds']['Dust']['model_d']}_{self.tools.params['QUBIC']['instrument']}_"
                    + self.tools.params["foldername"]
                )
                create_folder_if_not_exists(self.comm, "CMM/" + self.tools.params["foldername"] + "/maps/")
            if (
                self.tools.params["Plots"]["maps"] == True
                or self.tools.params["Plots"]["conv_beta"] == True
            ):
                create_folder_if_not_exists(self.comm, f"CMM/jobs/{self.job_id}/I")
                create_folder_if_not_exists(self.comm, f"CMM/jobs/{self.job_id}/Q")
                create_folder_if_not_exists(self.comm, f"CMM/jobs/{self.job_id}/U")
                create_folder_if_not_exists(self.comm, f"CMM/jobs/{self.job_id}/allcomps")
                create_folder_if_not_exists(self.comm, f"CMM/jobs/{self.job_id}/A_iter")

        self.tools.mpi._print_message("========= External Data =========")
        self.external = PresetExternal(self.tools)

        self.tools.mpi._print_message("========= QUBIC =========")
        self.qubic = PresetQubic(self.tools, self.external)

        self.tools.mpi._print_message("========= Components =========")
        self.comp = PresetComponents(self.tools, self.qubic)

        self.tools.mpi._print_message("========= Sky =========")
        self.sky = PresetSky(self.tools, self.qubic)

        self.tools.mpi._print_message("========= GAIN =========")
        self.gain = PresetGain(self.tools, self.qubic)

        self.tools.mpi._print_message("========= Mixing Matrix =========")
        self.mixingmatrix = PresetMixingMatrix(self.tools, self.qubic, self.comp)

        self.tools.mpi._print_message("========= Acquisition =========")
        self.acquisition = PresetAcquisition(
            self.seed_noise,
            self.tools,
            self.external,
            self.qubic,
            self.sky,
            self.comp,
            self.mixingmatrix,
            self.gain,
        )

        self.tools.display_simulation_configuration()

        return self
