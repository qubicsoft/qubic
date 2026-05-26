import yaml
import subprocess
import numpy as np
import os

script_path = "../../MapMaking/src/run_fmm.py"

print("CWD:", os.getcwd())
print("Trying path:", script_path)
print("Absolute path:", os.path.abspath(script_path))
print("Exists:", os.path.exists(script_path))

npointings_list = np.linspace(100, 10000, 6, dtype=int)

for npointings in npointings_list:
    with open("params.yaml", "r") as f:
        dict_qubic = yaml.safe_load(f)

    dict_qubic["QUBIC"]["npointings"] = int(npointings)
    print(dict_qubic["QUBIC"]["npointings"])
    dict_qubic["filename"] = f"test_{npointings}"

    param_file = f"params_{npointings}.yaml"

    with open(param_file, "w") as f:
        yaml.safe_dump(dict_qubic, f)

    subprocess.run(
        ["python", "../../../MapMaking/src/run_fmm.py", param_file], check=True
    )
