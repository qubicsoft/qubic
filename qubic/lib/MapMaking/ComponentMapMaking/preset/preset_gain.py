import numpy as np

from qubic.lib.Qmpi_tools import join_data


class PresetGain:
    """Preset Detectors Gain.

    Instance to initialize the Components Map-Making. It defines the input detectors gain variables and methodes.

    Parameters
    ----------
    preset_tools : object
        Class containing tools and simulation parameters.
    preset_qubic : object
        Class containing qubic operator and variables and methods.

    """

    def __init__(self, preset_tools, preset_qubic):
        """
        Initialize.

        """
        ### Import preset QUBIC & tools
        self.preset_qubic = preset_qubic
        self.preset_tools = preset_tools

        ### Get input detectors gain
        self.preset_tools.mpi._print_message("    => Getting detectors gain")
        self.get_input_gain()

    def get_input_gain(self):
        """Input gains.

        Generates and processes input gain values for the instrument based on preset parameters.

        This method sets the `gain_in`, `all_gain_in`, `gain_iter`, `all_gain`, and `all_gain_iter`
        attributes of the instance. The gain values are generated using a normal distribution and may be
        adjusted based on the instrument type and preset parameters.

        """

        np.random.seed(None) # rewrite randomness!
        if self.preset_qubic.params_qubic["instrument"] == "UWB":
            self.gain_in = np.random.normal(
                1,
                self.preset_qubic.params_qubic["GAIN"]["sig_gain"],
                self.preset_qubic.joint_in.qubic.ndets,
            )
        else:
            self.gain_in = np.random.normal(
                1,
                self.preset_qubic.params_qubic["GAIN"]["sig_gain"],
                (self.preset_qubic.joint_in.qubic.ndets, 2),
            )

        self.all_gain_in = join_data(self.preset_tools.comm, self.gain_in)

        if self.preset_qubic.params_qubic["GAIN"]["fit_gain"]:
            gain_err = 0.2
            self.gain_iter = np.random.uniform(
                self.gain_in - gain_err / 2,
                self.gain_in + gain_err / 2,
                self.gain_in.shape,
            )
        else:
            self.gain_iter = np.ones(self.gain_in.shape)

        self.all_gain = join_data(self.preset_tools.comm, self.gain_iter)
        self.all_gain_iter = np.array([self.gain_iter])
