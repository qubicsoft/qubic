import sys
from pyoperators import MPI
import os
from qubic.lib.MapMaking.FrequencyMapMaking.Qfmm import PipelineEnd2End

from CMM.pipeline import Pipeline


### Common MPI arguments
comm = MPI.COMM_WORLD

simu = 'FMM'

path = os.path.dirname(os.path.realpath(__file__))
paramters_file = path + '/test_params.txt'

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

    