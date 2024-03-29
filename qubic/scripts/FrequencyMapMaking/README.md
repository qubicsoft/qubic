# Pipeline

Repository that contain End-2-End pipeline to perform QUBIC frequency map-making, power spectrum estimation and cosmological analysis. 

# Run 

Be careful that every qubicsoft dependencies are correctly installed. The code can be run locally but more efficient in Computing Cluster using SLURM system. To send jobs on computing clusters with SLURM system, use the command :

```
sbatch main.sh
```

To modify memory requirements, please modify `main.sh` file, especially lines :

* `#SBATCH --partition=your_partition` to run on different sub-systems.
* `#SBATCH --nodes=number_of_nodes` to split data on different nodes.
* `#SBATCH --ntasks-per-node=Number_of_taks` to run several MPI tasks.
* `#SBATCH --cpus-per-task=number_of_CPU` to ask for several CPU for OpenMP system.
* `#SBATCH --mem=number_of_Giga` to ask for memory (in GB, let the letter G at the end i.e 5G)
* `#SBATCH --time=day-hours:minutes:seconds` to ask for more execution time.
