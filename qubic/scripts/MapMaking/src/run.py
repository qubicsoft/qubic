import sys
from pyoperators import MPI
import os
from qubic.lib.MapMaking.FrequencyMapMaking.Qfmm import PipelineEnd2End

from qubic.lib.MapMaking.ComponentMapMaking.Qcmm import Pipeline


### Common MPI arguments
comm = MPI.COMM_WORLD

simu = 'CMM'

path = os.path.dirname(os.path.realpath(__file__))
paramters_file = path + '/FMM/configuration_files/test_params.txt'

if __name__ == "__main__":

    if simu == 'FMM':
        
        try:
            file = str(sys.argv[1])
        except IndexError:
            file = None

        ### Initialization
        pipeline = PipelineEnd2End(comm, parameters_path=paramters_file)
        
        ### Execution
        pipeline.main(specific_file=file)

    elif simu == 'CMM':

        seed_noise = int(sys.argv[1])
        
        ### Initialization
        pipeline = Pipeline(comm, 1, seed_noise)
        
        ### Execution
        pipeline.main()

    