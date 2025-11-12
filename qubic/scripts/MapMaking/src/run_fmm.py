import argparse

import yaml
from pyoperators import MPI
from qubic.lib.MapMaking.FrequencyMapMaking.Qfmm import PipelineEnd2End

comm = MPI.COMM_WORLD

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QUBIC FMM pipeline with optional seed.")
    parser.add_argument("parameters_file", type=str, help="Path to parameters file.")
    parser.add_argument("file_spectrum", nargs="?", default=None, help="Optional spectrum file.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for this job.")
    args = parser.parse_args()

    # Set seed if provided
    if args.seed is not None:
        with open(args.parameters_file, "r") as f:
            params = yaml.safe_load(f)

        params["QUBIC"]["NOISE"]["seed_noise"] = args.seed

        with open(args.parameters_file, "w") as f:
            yaml.safe_dump(params, f, sort_keys=False)

    # Initialize and run the pipeline
    pipeline = PipelineEnd2End(comm, parameters_path=args.parameters_file)
    pipeline.main(specific_file=args.file_spectrum)
