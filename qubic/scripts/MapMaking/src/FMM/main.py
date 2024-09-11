import Qmixing_matrix

print(Qmixing_matrix)
sto


import sys

from pipeline import *
from pyoperators import *

try:
    file = str(sys.argv[1])
except IndexError:
    file = 1

if __name__ == "__main__":

    ### Common MPI arguments
    comm = MPI.COMM_WORLD

    ### Initialization
    pipeline = PipelineEnd2End(comm)

    ### Execution
    pipeline.main(specific_file=file)
