# Component Map-Making

This folder contains all the files needed to run the component map-making for the QUBIC experiments. This method uses highly parallelized library such as `PyOperators` or `PySimulators`, then the simulations have to be performed on a computing cluster.

```ruby
sbatch --mem=your_memory_requirement --nodes=1 --partition=your_partition --ntasks=number_of_CPUs CMM_constant_spectral_index.sh
```
