#!/bin/bash

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

python time_constants_computation_alldatasets.py 0 2
