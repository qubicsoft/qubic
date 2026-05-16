import os
import sys
import multiprocessing as mp
from cosmic_rays.crd import Crd

def main():
    mp.set_start_method("spawn")
    # Check if the configuration file (passed as the first command-line argument) exists
    if not os.path.isfile(conf := sys.argv[1]):
        # Raise an error if the file is not found
        raise ValueError(f'File `{conf}` not found')

    # Create a Crd object by reading the configuration from the specified file
    crd = Crd.read_config(conf)

    # Run the cosmic ray detection analysis
    crd.find_cosmic_rays()
