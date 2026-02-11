import numpy as np
from pipeline import Pipeline
from pyoperators import *
import sys

seed_noise = int(sys.argv[1])

### MPI common arguments
comm = MPI.COMM_WORLD

if __name__ == "__main__":
    
    pipeline = Pipeline(comm, 1, seed_noise)
    
    pipeline.main()