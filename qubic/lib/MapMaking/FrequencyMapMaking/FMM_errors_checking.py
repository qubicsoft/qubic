class ErrorChecking:
    """
    Class to check for errors in the Frequency Map Making process.
    """

    def __init__(self, params):
        """
        Initialize the ErrorChecking class with the given parameters file.

        Parameters:
        - params: parameters file object containing parameters for error checking.
        """
        self.params = params

    def check_errors(self):
        """Errors check.

        Checks for various parameter errors in the 'params.yml' file.

        Raises:
            TypeError: If any of the parameter checks fail.
        """

        # Check if the instrument is either 'DB', 'UWB' or 'MB'
        if self.params["QUBIC"]["instrument"] not in ["DB", "UWB", "MB"]:
            raise TypeError("You must choose DB, UWB or MB instrument")

        if self.params["QUBIC"]["nrec"] == 1 and self.params["PLANCK"]["external_data"]:
            raise TypeError("Adding Planck for Nrec=1 case is not yet implemented")

        if self.params["QUBIC"]["instrument"] == "DB" and self.params["QUBIC"]["nrec"] == 1:
            raise TypeError("You can't reconstruct one map in Dual Band configuration")

        if self.params["QUBIC"]["instrument"] != "MB":  # We might want to build odd nsub with "MB"
            # Check if nsub_in is even
            if self.params["QUBIC"]["nsub_in"] % 2 != 0:
                raise TypeError("The argument nsub_in should be even")
            if self.params["QUBIC"]["nrec"] % 2 != 0 and self.params["QUBIC"]["nrec"] > 2:
                raise TypeError("The argument nrec should be even")

            # Check if nsub_out is even
            if self.params["QUBIC"]["nsub_out"] % 2 != 0:
                raise TypeError("The argument nsub_out should be even")

        # Check if nsub_out is a multiple of nrec
        if self.params["QUBIC"]["nsub_out"] % self.params["QUBIC"]["nrec"] != 0:
            raise TypeError("nrec should be a multiple of nsub_out")
