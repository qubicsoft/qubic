"""
Run one FMM reconstruction from a pre-computed noiseless TOD.

Loads the noiseless TOD (produced by run_fmm_tod.py), generates a single
independent noise realization for the given real_id, and runs the PCG
map-making reconstruction.

Usage
-----
    mpirun -n <P> python run_fmm_reconstruction.py params.yaml \\
        --noiseless_tod FMM/test/Dict/tod_combined_0000.h5 \\
        --real_id 0 --base_seed 100
"""

import argparse
import os
import tempfile

import numpy as np
from pyoperators import MPI
from ruamel.yaml import YAML

from qubic.lib.MapMaking.FrequencyMapMaking.Qfmm import PipelineFrequencyMapMaking
from qubic.lib.Qfoldertools import create_folder_if_not_exists
from qubic.lib.Qhdf5 import HDF5Dict

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstruct one noise realization from a pre-computed noiseless TOD."
    )
    parser.add_argument("parameters_file", type=str, help="Path to parameters YAML file.")
    parser.add_argument(
        "--noiseless_tod",
        type=str,
        required=True,
        help="Path to the combined noiseless TOD HDF5 (produced by run_fmm_tod.py).",
    )
    parser.add_argument(
        "--real_id",
        type=int,
        default=0,
        help="Realization index (= SLURM_ARRAY_TASK_ID). Used to offset the seed.",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=None,
        help="Base seed. This realization uses base_seed + real_id.",
    )
    args = parser.parse_args()

    yaml = YAML()
    yaml.preserve_quotes = True

    with open(args.parameters_file) as f:
        params = yaml.load(f)

    foldername = params["foldername"]
    filename   = params["filename"]
    dict_folder = f"FMM/{foldername}/Dict"
    create_folder_if_not_exists(comm, dict_folder)

    base_seed = args.base_seed if args.base_seed is not None else params["QUBIC"]["NOISE"]["seed_noise"]
    seed = base_seed + args.real_id

    if rank == 0:
        print(f"\n{'=' * 50}")
        print(f"  Realization {args.real_id}  |  seed={seed}")
        print(f"  Noiseless TOD : {args.noiseless_tod}")
        print(f"{'=' * 50}\n")

    # ── Init pipeline ─────────────────────────────────────────────────────────
    # path_tod makes the pipeline:
    #   • read n_sims / npointings_per_sim from the file and adjust npointings
    #   • skip building joint_tod (not needed for reconstruction)
    #   • NOT generate noise in __init__ (we do it below with the right seed)
    params["path_tod"]     = args.noiseless_tod
    params["simulate_tod"] = False
    params["Pipeline"]["mapmaking"] = True
    params["Pipeline"]["spectrum"]  = False

    slurm_job_id = os.environ.get("SLURM_JOB_ID", f"real{args.real_id:04d}")
    output_file = f"{dict_folder}/{filename}_real_{args.real_id:04d}_{slurm_job_id}.h5"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(params, tmp)
        tmp_path = tmp.name

    try:
        fmm = PipelineFrequencyMapMaking(comm, output_file, params)

        # ── Generate noise for this realization ──────────────────────────────
        fmm.regenerate_noise(seed)

        # ── Load noiseless TOD from disk and add noise ────────────────────────
        if rank == 0:
            print("  Loading noiseless TOD from disk …")
        data = HDF5Dict().load_dict(args.noiseless_tod)
        tod_noiseless_qubic = data["tod_noiseless_qubic"]

        tod_qubic = tod_noiseless_qubic + fmm.noiseq

        if fmm.params["PLANCK"]["external_data"]:
            nrec  = fmm.params["QUBIC"]["nrec"]
            nside = fmm.params["SKY"]["nside"]
            tod_planck = np.zeros((max(nrec, 2), 12 * nside**2, 3))
            for irec in range(nrec):
                noise = fmm.noise_planck[0] if irec < nrec / 2 else fmm.noise_planck[1]
                tod_planck[irec] = fmm.maps_input_convolved[irec] + noise
            fmm.TOD = np.r_[tod_qubic, tod_planck.ravel()]
        else:
            fmm.TOD = tod_qubic

        # ── Reconstruct ───────────────────────────────────────────────────────
        fmm._run_reconstruction(output_file)

        if rank == 0:
            print(f"\n  Saved → {output_file}")

    finally:
        os.unlink(tmp_path)
