class PresetExternal:
    """Preset External Data.

    Instance to initialize the Components Map-Making. It defines the external data variables and methods.

    Parameters
    ----------
    preset_tools: object
        Class containing tools and simulation parameters.

    Attributes
    ----------
    external_nus : list
        List containing Planck maps frequency.

    """

    def __init__(self, preset_tools):
        """
        Initialization.

        """

        ### Define Planck parameters variable
        self.params_external = preset_tools.params["PLANCK"]

        ### External frequencies
        preset_tools.mpi._print_message("    => Computing Planck frequency bands")
        self.external_nus = self.get_external_nus()

    def get_external_nus(self):
        """External frequencies.

        Method to create a Python list of external frequencies by reading the `params.yml` file.

        This method reads the `params.yml` file and checks for the presence of specific frequency
        values under the 'PLANCK' section. If a frequency is present, it is added to the external
        frequencies list.

        Returns
        -------
        external_nus : list
            List containing Planck maps frequency.

        """

        ### List of all Planck's frequency bands
        allnus = [30, 44, 70, 100, 143, 217, 353]

        ### Build variable with selected frequency bands
        external = []
        for nu in allnus:
            if self.params_external[f"{nu:.0f}GHz"]:
                external += [nu]

        return external
