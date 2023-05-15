import os
from pathlib import Path

PATH = os.path.dirname(__file__) + os.path.sep
DATA_PATH = Path(__file__).parent

del os
del Path