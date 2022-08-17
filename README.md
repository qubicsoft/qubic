# QUBIC

Simulation and map-making tools for the QUBIC experiment


## pysimulators and pyoperators

The qubic Python package uses pysimulators and pyoperators to build the instrument model. They can be found at github:

https://github.com/pchanial/pyoperators

https://github.com/pchanial/pysimulators


## Installation from source

First, the repository needs to be cloned:

```bash
git clone git@github.com:qubicsoft/qubic
cd qubic
```

On linux, the installation should be straightforward after installing gfortran:

```bash
python3.9 -mvenv venv-qubic
source venv-qubic/bin/activate
pip install -e .
```

On MacOSX, it is preferable to use the Conda platform, selecting the correct architecture (arm64 for M1 and x86_64 otherwise), then:

```bash
conda config --add channels conda-forge
conda create --yes --name venv-qubic
conda activate venv-qubic
conda install --yes gfortran pyfftw healpy
pip install -e .
```

For more information, please look on the QUBIC project wiki:
http://qubic.in2p3.fr/wiki/pmwiki.php/DataAnalysis/HowTo



