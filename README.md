# QUBIC

Simulation and map-making tools for the QUBIC experiment


## pysimulators and pyoperators

The qubic Python package uses pysimulators and pyoperators to build the instrument model. They can be found at github:

https://github.com/pchanial/pyoperators

https://github.com/pchanial/pysimulators


## Requirements

The QUBIC software requires Python 3.8, 3.9 or 3.10.


## Installation

First, the repository needs to be cloned:

```bash
git clone git@github.com:qubicsoft/qubic
cd qubic
```

It is recommended (even mandatory for MacOS) to install the qubic software using the Anaconda platform, with the correct architecture (arm64 for M1 and x86_64 otherwise).

```bash
conda config --add channels conda-forge
conda create --yes --name venv-qubic python==3.10
conda activate venv-qubic
conda install --yes gfortran pyfftw healpy namaster
SETUPTOOLS_ENABLE_FEATURES="legacy-editable" pip install -e .
```

On Linux, after installing gfortran, installation can be performed with the vanilla Python interpreter, except that the package NaMaster needs to be [installed](https://namaster.readthedocs.io/en/latest/installation.html) independently.

```bash
python3 -mvenv venv-qubic
source venv-qubic/bin/activate
SETUPTOOLS_ENABLE_FEATURES="legacy-editable" pip install -e .
```

For more information, please look at the QUBIC project wiki:
http://qubic.in2p3.fr/wiki/pmwiki.php/DataAnalysis/HowTo
