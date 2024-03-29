# Components Map-Making

Repository that contain End-2-End pipeline for components reconstruction directly from Time-Ordered Data (TOD) for QUBIC experiments.

# Description

We perform alternate estimation to reconstruct sky pixels, astrophysical description done by spectral indices and systematics (gain detectors only for now). The code is based on parametric description of astrophysical foregrounds but this will evolve in the near futur. 

The code is organized as follow :

* `acquisition` folder contain tools to perform QUBIC reconstruction
* `cosmo` folder contain tools to constrains tensor-to-scalar ratio
* `fgb` folder contain tools based on FG-Buster (<https://github.com/fgbuster/fgbuster>) to describe astrophysical emissions and compute mixing matrix
* `simtools` folder contains useful library to perform simulations
* `solver` folder contains conjugate-gradient used in the reconstruction
* `main.py` python file allow to run the whole code
* `pipeline.py` python file contain the whole description of the pipeline called by `main.py`

# Run 

To use the code, you can clone the repository using :

```
git clone https://github.com/mathias77515/CMM-Pipeline
```

Be careful that every qubicsoft dependencies are correctly installed. The code can be run locally but much more efficient in Computing Cluster using SLURM system. To send jobs on computing clusters with SLURM system, use the command :

```
sbatch main.sh {SEED_CMB} {SEED_NOISE}
```

To modify memory requirements, please modify `main.sh` file, especially lines :

* `#SBATCH --partition=your_partition` to run on different sub-systems.
* `#SBATCH --nodes=number_of_nodes` to split data on different nodes.
* `#SBATCH --ntasks-per-node=Number_of_taks` to run several MPI tasks.
* `#SBATCH --cpus-per-task=number_of_CPU` to ask for several CPU for OpenMP system.
* `#SBATCH --mem=number_of_Giga` to ask for memory (in GB, let the letter G at the end i.e 6G)
* `#SBATCH --time=day-hours:minutes:seconds` to ask for more execution time.
