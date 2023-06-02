#!/bin/bash -l

export NUMBA_NUM_THREADS=5
export MKL_NUM_THREADS=5
export NUMEXPR_NUM_THREADS=5
export OMP_NUM_THREADS=5
export OPENBLAS_NUM_THREADS=5
export VECLIB_MAXIMUM_THREADS=5
export PYOPERATORS_NO_MPI=5

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/pbs/home/n/nmirongr/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/pbs/home/n/nmirongr/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/pbs/home/n/nmirongr/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/pbs/home/n/nmirongr/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate myqubic

python CMM_constant_spectral_index.py
