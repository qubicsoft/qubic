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

    np.savez(signal_raw_fname, **dict(zip(map(str, range(len(qubic[1]))), qubic[1])))


if __name__ == "__main__":
    export_data(*sys.argv[1:])
