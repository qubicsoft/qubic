import sys

from pyoperators import MPI

from qubic.lib.MapMaking.FrequencyMapMaking.Qfmm import PipelineEnd2End

### Common MPI arguments
comm = MPI.COMM_WORLD

parameters_file = str(sys.argv[1])

try:
    file_spectrum = str(sys.argv[2])
except Exception:
    file_spectrum = None


if __name__ == "__main__":
    ### Initialization
    pipeline = PipelineEnd2End(comm, parameters_path=parameters_file)

    ### Execution
    pipeline.main(specific_file=file_spectrum)
