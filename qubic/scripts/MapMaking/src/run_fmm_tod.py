"""
Run multiple FMM TOD-only simulations sequentially, freeing RAM between each,
then combine the resulting QUBIC TOD chunks into a single HDF5 file.

Usage
-----
    mpirun -n <N> python run_fmm_tod.py params.yaml --n_sims 4 --base_seed 100
    mpirun -n <N> python run_fmm_tod.py params.yaml --n_sims 4 --output my_tod.h5

The script forces simulate_tod=True regardless of the params file.
The combined output file stores:
    "tod"             : np.concatenate(QUBIC chunks) + PLANCK part (from first sim)
    "qubic_tod_size"  : length of the QUBIC-only part in the combined TOD
    "n_sims"          : number of simulations combined
    "npointings_per_sim" : npointings used for each simulation

Goal: simulate a very large TOD (N * npointings samples) without ever holding
all of it in RAM at once.  Reconstruction is then run separately with path_tod
pointing to the combined file (possibly with lower nsub_out / nrec).
"""

import argparse
import gc
import os
import tempfile

import numpy as np
from pyoperators import MPI
from ruamel.yaml import YAML

from qubic.lib.MapMaking.FrequencyMapMaking.Qfmm import PipelineEnd2End
from qubic.lib.Qfoldertools import create_folder_if_not_exists
from qubic.lib.Qhdf5 import HDF5Dict

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def _run_sim(comm, params):
    """Write a temp params file, run one TOD-only simulation, then clean up."""
    # Always enforce TOD-only mode regardless of what the params dict contains
    params["simulate_tod"] = True
    params["Pipeline"]["mapmaking"] = True
    params["Pipeline"]["spectrum"] = False

    yaml = YAML()
    yaml.preserve_quotes = True

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(params, tmp)
        tmp_path = tmp.name

    try:
        pipeline = PipelineEnd2End(comm, parameters_path=tmp_path)
        pipeline.main()
        del pipeline
        gc.collect()
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run N FMM TOD simulations sequentially and combine them into one HDF5 file."
    )
    parser.add_argument("parameters_file", type=str, help="Path to parameters YAML file.")
    parser.add_argument("--n_sims", type=int, default=2, help="Number of simulations to run (default: 2).")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=None,
        help="Base seed for noise. Simulation i uses base_seed + i. Random if not set.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HDF5 file for combined TOD. Default: FMM/<foldername>/Dict/tod_combined_<job_id>.h5",
    )
    parser.add_argument(
        "--job_id",
        type=int,
        default=0,
        help="SLURM array task ID (default: 0). Used to namespace output files and offset seeds.",
    )
    args = parser.parse_args()

    yaml = YAML()
    yaml.preserve_quotes = True

    with open(args.parameters_file) as f:
        params = yaml.load(f)

    foldername = params["foldername"]
    dict_folder = f"FMM/{foldername}/Dict"
    create_folder_if_not_exists(comm, dict_folder)

    output_file = args.output or f"{dict_folder}/tod_combined_{args.job_id:04d}.h5"

    # Determine the base seed: explicit arg, or fall back to whatever is in the params file.
    # Offset by job_id * n_sims so concurrent array jobs never share a seed.
    base_seed = args.base_seed if args.base_seed is not None else params["QUBIC"]["NOISE"]["seed_noise"]
    base_seed += args.job_id * args.n_sims

    tod_chunk_files = []

    # ── Run simulations ──────────────────────────────────────────────────────
    for i in range(args.n_sims):
        seed = base_seed + i

        if rank == 0:
            print(f"\n{'=' * 50}")
            print(f"  Simulation {i + 1}/{args.n_sims} | seed={seed}")
            print(f"{'=' * 50}\n")

        params["QUBIC"]["NOISE"]["seed_noise"] = seed
        params["PLANCK"]["seed_noise"] = seed

        _run_sim(comm, params)

        # Rank 0 renames tod.h5 so the next iteration does not overwrite it
        chunk_file = f"{dict_folder}/tod_chunk_{args.job_id:04d}_{i:04d}.h5"
        if rank == 0:
            os.rename(f"{dict_folder}/tod.h5", chunk_file)
            print(f"  → chunk saved: {chunk_file}")
        tod_chunk_files.append(chunk_file)

        comm.Barrier()

    # ── Combine chunks ───────────────────────────────────────────────────────
    if rank == 0:
        print(f"\n{'=' * 50}")
        print(f"  Combining {args.n_sims} TOD chunks")
        print(f"{'=' * 50}\n")

        tod_qubic_chunks = []
        tod_noiseless_qubic_chunks = []
        tod_planck = None
        tod_noiseless_planck = None

        for chunk_file in tod_chunk_files:
            data = HDF5Dict().load_dict(chunk_file)
            qubic_size = int(data["qubic_tod_size"])
            tod_full = data["tod"]

            tod_qubic_chunks.append(tod_full[:qubic_size])
            print(f"  Loaded {chunk_file} | QUBIC shape: {tod_full[:qubic_size].shape}")

            if "tod_noiseless_qubic" in data:
                tod_noiseless_qubic_chunks.append(data["tod_noiseless_qubic"])

            # Keep the PLANCK part from the first chunk (map-based, same for all sims)
            if tod_planck is None and qubic_size < len(tod_full):
                tod_planck = tod_full[qubic_size:]
                print(f"  PLANCK part shape: {tod_planck.shape} (taken from first chunk)")

            if tod_noiseless_planck is None and "tod_noiseless_planck" in data:
                tod_noiseless_planck = data["tod_noiseless_planck"]

        tod_qubic_combined = np.concatenate(tod_qubic_chunks, axis=0)
        if tod_planck is not None:
            tod_combined = np.r_[tod_qubic_combined, tod_planck]
        else:
            tod_combined = tod_qubic_combined

        print(f"\n  Combined TOD shape : {tod_combined.shape}")
        print(f"  QUBIC part size    : {len(tod_qubic_combined)}")

        save_dict = {
            "tod": tod_combined,
            "qubic_tod_size": len(tod_qubic_combined),
            "n_sims": args.n_sims,
            "npointings_per_sim": params["QUBIC"]["npointings"],
        }
        if tod_noiseless_qubic_chunks:
            save_dict["tod_noiseless_qubic"] = np.concatenate(tod_noiseless_qubic_chunks, axis=0)
            print(f"  Noiseless QUBIC TOD saved (shape: {save_dict['tod_noiseless_qubic'].shape})")
        if tod_noiseless_planck is not None:
            save_dict["tod_noiseless_planck"] = tod_noiseless_planck

        HDF5Dict().save_dict(output_file, save_dict)
        print(f"\n  Combined TOD saved to {output_file}")

        # Clean up individual chunk files
        for chunk_file in tod_chunk_files:
            os.remove(chunk_file)
        print(f"  Removed {args.n_sims} temporary chunk files.")

    comm.Barrier()
