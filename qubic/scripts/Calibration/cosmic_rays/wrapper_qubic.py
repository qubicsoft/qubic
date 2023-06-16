import os
import sys
import numpy as np
from qubicpack.qubicfp import qubicfp


def export_data(data_path: str, time_fname: str, signal_raw_fname: str):
    
    if os.path.isfile(time_fname) and os.path.isfile(signal_raw_fname):
        return 
    
    qubic = qubicfp()
    qubic.read_qubicstudio_dataset(data_path)
    qubic = qubic.tod()

    with open(time_fname, "wb") as fout:
        np.save(fout, qubic[0])

    # dict(zip(arg1, arg2)) creates a dictionary whose 
    # keys are represented by arg1 and whose values are represented by arg2
    np.savez(signal_raw_fname, **dict(zip(map(str, range(len(qubic[1]))), qubic[1]))) 


if __name__ == "__main__":
    
    # I start from index 1 because the first argument is always the name of the script.
    # wrapper_qubic.py is executed as follows:
    # python wrapper_qubic.py <.fits files path> <time file path (.npy format)> <signal file path (.npz format)>.
    # It unpacks the passed arguments and assigns them in order to data_path, time_fname, signal_raw_fname
    export_data(*sys.argv[1:]) 
