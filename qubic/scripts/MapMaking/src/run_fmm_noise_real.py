"""
Generate N independent noise realizations from a single TOD simulation.

Instead of running the full pipeline N times, this script:
  1. Runs PipelineEnd2End once  →  computes H*s and maps_input_convolved (expensive),
     saves tod.h5 with noiseless components.
  2. Keeps the pipeline object in memory and calls regenerate_noise(seed) N times
     (cheap: pure RNG) to build and save one TOD file per realization.

Usage
-----
    mpirun -n <N> python run_fmm_noise_real.py params.yaml \\
        --n_real 10 --base_seed 100
"""

import argparse
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


def _build_planck_tod(fmm):
    """Assemble the Planck TOD from the (fixed) signal and the current noise realization."""
    nrec = fmm.params["QUBIC"]["nrec"]
    nside = fmm.params["SKY"]["nside"]
    tod_planck = np.zeros((max(nrec, 2), 12 * nside**2, 3))
    for irec in range(nrec):
        noise = fmm.noise_planck[0] if irec < nrec / 2 else fmm.noise_planck[1]
        tod_planck[irec] = fmm.maps_input_convolved[irec] + noise
    return tod_planck.ravel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate N noise realizations from a single TOD observation."
    )
    parser.add_argument("parameters_file", type=str, help="Path to parameters YAML file.")
    parser.add_argument("--n_real", type=int, default=10, help="Number of noise realizations (default: 10).")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=None,
        help="Base seed. Realization i uses base_seed + i. Falls back to params file if not set.",
    )
    parser.add_argument(
        "--job_id",
        type=int,
        default=0,
        help="SLURM array task ID (default: 0). Offsets seeds so concurrent jobs never overlap.",
    )
    args = parser.parse_args()

    yaml = YAML()
    yaml.preserve_quotes = True

    with open(args.parameters_file) as f:
        params = yaml.load(f)

    # Force TOD-only mode
    params["simulate_tod"] = True
    params["Pipeline"]["mapmaking"] = True
    params["Pipeline"]["spectrum"] = False

    foldername = params["foldername"]
    dict_folder = f"FMM/{foldername}/Dict"
    create_folder_if_not_exists(comm, dict_folder)

    base_seed = args.base_seed if args.base_seed is not None else params["QUBIC"]["NOISE"]["seed_noise"]
    base_seed += args.job_id * args.n_real

    # ── Phase 1: single full pipeline run ────────────────────────────────────
    if rank == 0:
        print(f"\n{'=' * 50}")
        print(f"  Phase 1: computing noiseless TOD  |  seed={base_seed}")
        print(f"{'=' * 50}\n")

    params["QUBIC"]["NOISE"]["seed_noise"] = base_seed
    params["PLANCK"]["seed_noise"] = base_seed

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(params, tmp)
        tmp_path = tmp.name

    try:
        pipeline = PipelineEnd2End(comm, parameters_path=tmp_path)
        pipeline.main()
    finally:
        os.unlink(tmp_path)

    # pipeline.mapmaking is the live PipelineFrequencyMapMaking object
    fmm = pipeline.mapmaking

    # ── Phase 2: N noise realizations ────────────────────────────────────────
    for i in range(args.n_real):
        seed = base_seed + i

        if rank == 0:
            print(f"\n{'=' * 50}")
            print(f"  Realization {i + 1}/{args.n_real}  |  seed={seed}")
            print(f"{'=' * 50}\n")

        fmm.regenerate_noise(seed)

        tod_qubic = fmm._tod_signal_qubic + fmm.noiseq
        if fmm.params["PLANCK"]["external_data"]:
            tod_planck = _build_planck_tod(fmm)
            tod = np.r_[tod_qubic, tod_planck]
        else:
            tod = tod_qubic

        out_file = f"{dict_folder}/tod_real_{args.job_id:04d}_{i:04d}.h5"
        if rank == 0:
            HDF5Dict().save_dict(
                out_file,
                {"tod": tod, "qubic_tod_size": len(fmm.noiseq)},
            )
            print(f"  → saved: {out_file}")

        comm.Barrier()

    if rank == 0:
        print(f"\n  Done. {args.n_real} realizations saved to {dict_folder}/")
