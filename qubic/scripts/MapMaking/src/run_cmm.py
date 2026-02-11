import sys
from pyoperators import MPI

from qubic.lib.MapMaking.ComponentMapMaking.Qcmm import Pipeline


### Common MPI arguments
comm = MPI.COMM_WORLD

parameters_file = str(sys.argv[1])

if __name__ == "__main__":

    ### Initialization
    pipeline = Pipeline(comm, parameters_file=parameters_file)
        
    ### Execution
    pipeline.main()

    