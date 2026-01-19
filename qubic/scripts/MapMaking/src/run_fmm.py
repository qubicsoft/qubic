import argparse

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
        from ruamel.yaml import YAML

        yaml = YAML()
        yaml.preserve_quotes = True

        with open(args.parameters_file) as f:
            params = yaml.load(f)

        params["QUBIC"]["NOISE"]["seed_noise"] = args.seed
        params["PLANCK"]["seed_noise"] = args.seed

        with open(args.parameters_file, "w") as f:
            yaml.dump(params, f)
            
        print(f"INFO Using SEED={args.seed}")

    # Initialize and run the pipeline
    pipeline = PipelineEnd2End(comm, parameters_path=args.parameters_file)
    pipeline.main(specific_file=args.file_spectrum)
