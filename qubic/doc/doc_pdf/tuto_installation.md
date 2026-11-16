Quick tutorial to install Qubicsoft

```bash
git clone git@github.com:qubicsoft/qubic.git
conda create -n venv-qubic python=3.10.0
conda activate venv-qubic
conda install -c conda-forge namaster=1.6
pip install "pip<24"
pip install "setuptools<71"
pip install "wheel<0.45"
SETUPTOOLS_ENABLE_FEATURES="legacy-editable" pip install --no-binary pyoperators,pysimulators -e . # pip install .
```