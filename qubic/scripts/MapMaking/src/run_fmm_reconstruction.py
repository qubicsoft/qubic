"""
Run FMM reconstruction-only on a pre-computed TOD file.

Usage
-----
    mpirun -n <N> python run_fmm_reconstruction.py params.yaml \\
        --tod_file FMM/test/Dict/tod_real_0000_0003.h5
"""

import argparse
import os
import tempfile

from pyoperators import MPI
from ruamel.yaml import YAML

from qubic.lib.MapMaking.FrequencyMapMaking.Qfmm import PipelineEnd2End

comm = MPI.COMM_WORLD

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run FMM map-making on a pre-computed TOD file."
    )
    parser.add_argument("parameters_file", type=str, help="Path to parameters YAML file.")
    parser.add_argument(
        "--tod_file",
        type=str,
        required=True,
        help="Path to the pre-computed TOD HDF5 file (produced by run_fmm_noise_real.py).",
    )
    args = parser.parse_args()

    yaml = YAML()
    yaml.preserve_quotes = True

    with open(args.parameters_file) as f:
        params = yaml.load(f)

    params["path_tod"] = args.tod_file
    params["simulate_tod"] = False
    params["Pipeline"]["mapmaking"] = True
    params["Pipeline"]["spectrum"] = False

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(params, tmp)
        tmp_path = tmp.name

    try:
        pipeline = PipelineEnd2End(comm, parameters_path=tmp_path)
        pipeline.main()
    finally:
        os.unlink(tmp_path)
