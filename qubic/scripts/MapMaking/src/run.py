import sys
from pyoperators import MPI

from qubic.lib.MapMaking.FrequencyMapMaking.Qfmm import PipelineEnd2End
from qubic.lib.MapMaking.ComponentMapMaking.Qcmm import Pipeline


### Common MPI arguments
comm = MPI.COMM_WORLD

simu = 'FMM'

parameters_file = 'FMM/configuration_files/params.txt'
file_spectrum = None

if __name__ == "__main__":

    if simu == 'FMM':

        ### Initialization
        pipeline = PipelineEnd2End(comm, parameters_path=parameters_file)
        
        ### Execution
        pipeline.main(specific_file=file_spectrum)

    elif simu == 'CMM':
        
        #raise  NotImplementedError('Not implemented yet')
        
        ### Initialization
        pipeline = Pipeline(comm, parameters_file=parameters_file)
        
        ### Execution
        pipeline.main()

    