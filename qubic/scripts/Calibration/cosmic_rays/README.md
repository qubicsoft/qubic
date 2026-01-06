# Cosmic Ray Detection Analysis

### Data Preprocessing and Instrumental Effects

The cosmic ray detection algorithm operates exclusively on preprocessed signals, making use of the tools provided by 
the `preprocessing.py` module.
These preprocessing steps decorrelate TOD from azimuth and elevation effects conditioning and baseline removal

In addition, the algorithm accounts for the winding orientation of the SQUID coils by analyzing the IV curves and 
extracting the corresponding TES sign using the `iv_mask.py` module. This step is crucial to ensure consistent signal polarity across detectors.


### Initialize a poetry environment
In the cosmic rays project folder, the poetry environment can be initialized by running:
```poetry install```
This command creates a virtual environment in Poetry’s default directory and installs all necessary dependencies.
To create the virtual environment inside the project directory instead, run the following command:
```
poetry config virtualenvs.in-project true --local
poetry install
```

The cosmic ray detection analysis is executed by running:
```
# in the root directory: qubic/qubic/scripts/Calibration/cosmic_rays
poetry run crd-analysis configs/all_scanning_strategy.toml
```
This script allows you to run the analysis on one or multiple datasets, depending on the configuration file selected 
from the config directory.
An example of a valid toml configuration is:

```
# mandatory parameters
[path]
# This type of path is interpreted by the algorithm to search for all datasets containing the keywords entered in 
# parentheses. In this case, it will analyze all datasets containing the words: 
# moonscan, skyscan, scanfast, skydip, dome. 
source = "/media/DataQubic/**/*(MoonScan, SkyScan, ScanFast, SkyDip, dome)*"

[optional]
exclude = ["IV", "2023"]
destination = "cr_results"
# folder containing the txt files of the masks obtained from the four curves
mask = "masks"                  
vertical_points = 3
exponential_points = 6
std_coeff = 2
# analysis of all TESs in the focal plane
analysis_type = "all"
# TES to be analyzed [-1]: all TESs 
tes = [-1]
# removes intermediate files, leaving only files with preprocessed TODs
remove_files = true
dpi = 300
```

### Parameter Exploration

The cosmic ray detection analysis can also be executed using the `parameters_script.py` script:
```
# in the root directory: qubic/qubic/scripts/Calibration/cosmic_rays
poetry run parameter-analysis configs/all_scanning_strategy.toml
```
The parameters_script.py script performs a systematic exploration of the main parameters involved in the cosmic ray 
detection process:
- Detection threshold: the standard-deviation threshold above which the search for cosmic ray candidates is triggered.
- Rising phase length: the minimum number of data points required to define the rising part of a cosmic ray candidate.
- Exponential decay length: the minimum number of data points required to reliably fit the exponential decay following 
  the peak.

Each parameter combination is evaluated independently, and the results are stored in separate output directories.


