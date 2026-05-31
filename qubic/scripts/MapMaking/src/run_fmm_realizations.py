"""
Run N independent noise realizations with full reconstruction in a single pipeline init.

The noiseless TOD (H*s + maps_input_convolved) is computed once. Noise is then
regenerated N times with distinct seeds, and the PCG reconstruction is run on each
assembled TOD. This avoids recomputing the expensive H*s operator for every realization.

Usage
-----
    mpirun -n <P> python run_fmm_realizations.py params.yaml \\
        --n_real 10 --base_seed 100 [--job_id 0]

Output files
------------
    FMM/<foldername>/Dict/<filename>_real_<job_id>_<i>.h5   (one per realization)
"""

import argparse
import os
import tempfile

import numpy as np
from pyoperators import MPI
from ruamel.yaml import YAML

from qubic.lib.MapMaking.FrequencyMapMaking.Qfmm import PipelineFrequencyMapMaking
from qubic.lib.Qfoldertools import create_folder_if_not_exists

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run N FMM reconstructions from a single H*s computation."
    )
    parser.add_argument("parameters_file", type=str, help="Path to parameters YAML file.")
    parser.add_argument("--n_real", type=int, default=10,
                        help="Number of noise realizations (default: 10).")
    parser.add_argument("--base_seed", type=int, default=None,
                        help="Base seed. Realization i uses base_seed + job_id*n_real + i.")
    parser.add_argument("--job_id", type=int, default=0,
                        help="SLURM array task ID. Offsets seeds so concurrent jobs never overlap.")
    args = parser.parse_args()

    yaml = YAML()
    yaml.preserve_quotes = True

    with open(args.parameters_file) as f:
        params = yaml.load(f)

    # Force reconstruction mode: no TOD-only stop, no external TOD loading
    params["simulate_tod"] = False
    params["path_tod"] = None
    params["Pipeline"]["mapmaking"] = True
    params["Pipeline"]["spectrum"] = False

    foldername = params["foldername"]
    filename   = params["filename"]
    dict_folder = f"FMM/{foldername}/Dict"
    create_folder_if_not_exists(comm, dict_folder)

    base_seed = args.base_seed if args.base_seed is not None else params["QUBIC"]["NOISE"]["seed_noise"]
    # Offset so each array task uses a non-overlapping seed range
    base_seed += args.job_id * args.n_real

    # Dummy output file for init (overridden per realization below)
    init_output = f"{dict_folder}/{filename}_real_{args.job_id:04d}_init.h5"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(params, tmp)
        tmp_path = tmp.name

    try:
        # ── Phase 1: single pipeline init ────────────────────────────────────
        # Builds joint, joint_tod, H operator, input maps, maps_input_convolved.
        # Also generates initial noise (will be replaced in the loop below).
        if rank == 0:
            print(f"\n{'=' * 50}")
            print(f"  Initialising pipeline  |  job_id={args.job_id}")
            print(f"{'=' * 50}\n")

        fmm = PipelineFrequencyMapMaking(comm, init_output, params)

        # ── Phase 2: compute noiseless TOD once ──────────────────────────────
        # get_tod() applies H to the sky maps and stores the result in
        # fmm._tod_signal_qubic.  The noise term (fmm.noiseq) added here is
        # discarded immediately — we regenerate it per realization below.
        if rank == 0:
            print(f"\n{'=' * 50}")
            print(f"  Computing noiseless TOD  (H*s)")
            print(f"{'=' * 50}\n")

        fmm.TOD = fmm.get_tod()   # sets fmm._tod_signal_qubic as a side-effect

        # ── Phase 3: N independent realizations ──────────────────────────────
        for i in range(args.n_real):
            seed = base_seed + i

            if rank == 0:
                print(f"\n{'=' * 50}")
                print(f"  Realization {i + 1}/{args.n_real}  |  seed={seed}")
                print(f"{'=' * 50}\n")

            # Replace noise with a fresh realization seeded by `seed`
            fmm.regenerate_noise(seed)

            # Assemble noisy TOD: noiseless signal + new noise
            tod_qubic = fmm._tod_signal_qubic + fmm.noiseq

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

            out_file = f"{dict_folder}/{filename}_real_{args.job_id:04d}_{i:04d}.h5"
            fmm._run_reconstruction(out_file)

            comm.Barrier()

        if rank == 0:
            print(f"\n  Done. {args.n_real} reconstructions saved to {dict_folder}/")

    finally:
        os.unlink(tmp_path)
