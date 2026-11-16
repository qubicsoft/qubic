import argparse
import os
import tempfile

from pyoperators import MPI
from qubic.lib.MapMaking.FrequencyMapMaking.Qfmm import PipelineEnd2End

comm = MPI.COMM_WORLD

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QUBIC FMM pipeline with optional seed.")
    parser.add_argument("parameters_file", type=str, help="Path to parameters file.")
    parser.add_argument("file_spectrum", nargs="?", default=None, help="Optional spectrum file.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for this job.")
    args = parser.parse_args()

    params_path = args.parameters_file
    tmp_path = None

    if args.seed is not None:
        from ruamel.yaml import YAML

        yaml = YAML()
        yaml.preserve_quotes = True

        with open(args.parameters_file) as f:
            params = yaml.load(f)

        params["QUBIC"]["NOISE"]["seed_noise"] = args.seed
        params["PLANCK"]["seed_noise"] = args.seed

        # Write to a per-job private temp file in /tmp so concurrent array jobs
        # never touch the shared params file. tempfile guarantees a unique path
        # and the finally block removes it even if the pipeline crashes.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(params, tmp)
            tmp_path = tmp.name

        params_path = tmp_path
        print(f"INFO Using SEED={args.seed}")

    try:
        pipeline = PipelineEnd2End(comm, parameters_path=params_path)
        pipeline.main(specific_file=args.file_spectrum)
    finally:
        if tmp_path is not None:
            os.unlink(tmp_path)
