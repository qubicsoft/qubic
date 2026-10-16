from dataclasses import dataclass
from pydantic import BaseModel, Field, BeforeValidator, DirectoryPath, ConfigDict, field_validator
from pathlib import Path
from typing import Annotated
import astropy.units as u
from astropy.coordinates import SkyCoord
from qubicpack.qubicfp import qubicfp
from qubicpack.utilities import NPIXELS
import numpy as np
from qubic.lib.GalacticPlane.preprocessing import get_tes_mask

ROOT_DIR = Path(__file__).parent
INSTRUMENTS_CONFIG_PATH = ROOT_DIR / "common" / "instruments.yaml"
DATA_DIR = ROOT_DIR.parents[1] / "scripts" / "GalacticPlane" / "data"

DatatDirPath = Annotated[DirectoryPath, BeforeValidator(lambda p: DATA_DIR / "input" / p)]
Unit = Annotated[u.UnitBase, BeforeValidator(u.Unit)]
TesType = list[int]

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid",
                              frozen=False,
                              validate_default=True,
                              str_strip_whitespace=True,
                              arbitrary_types_allowed=True)

class Region(StrictModel):

    name: str
    center: SkyCoord

    @field_validator("center", mode='before')
    @classmethod
    def parse_center(cls, v):
        if isinstance(v, str):
            return SkyCoord(v, unit="deg", frame="galactic")
        raise ValueError(f"Unable to parse {v}")

@dataclass
class qubic:

    datapath: DatatDirPath
    tes: TesType

    time_0: float = Field(init=False, default=0.0)
    time: np.ndarray = Field(init=False, default=np.array([]))
    signals: np.ndarray = Field(init=False, default=np.array([]))

    # gia' interpolati al tempo dell'elettronica
    elevation: np.ndarray = Field(init=False, default=np.array([]))
    azimuth: np.ndarray = Field(init=False, default=np.array([]))

    bounds: tuple[int, int] = Field(init=False, default=(0, 0))
    mask: int = 0

    def read(self, fraction=0.1):

        qubic = qubicfp()
        qubic.verbosity = 0
        qubic.read_qubicstudio_dataset(str(self.datapath))

        tm, signals = qubic.tod()
        t0 = tm[0]
        tm -= t0

        elevation = qubic.elevation()
        elevation_time = qubic.timeaxis(datatype="el")

        azimuth = qubic.azimuth()
        azimuth_time = qubic.timeaxis(datatype="az")

        # alcuni dataset hanno un endianess diverso
        azimuth = np.asarray(azimuth, dtype=np.float64)
        elevation = np.asarray(elevation, dtype=np.float64)
        azimuth_time = np.asarray(azimuth_time, dtype=np.float64)
        elevation_time = np.asarray(elevation_time, dtype=np.float64)

        interp_elevation = np.interp(tm, elevation_time, elevation)
        interp_azimuth = np.interp(tm, azimuth_time, azimuth)

        tes_mask = Path(self.datapath / "metadata" / "valid_tes.csv")
        # TODO: da implementare
        # tod_bounds = Path(self.path / "metadata" / "tod_bounds.csv")

        if not tes_mask.exists():
            raise RuntimeError("Missing valid_tes.csv file")

        tes_asic = np.loadtxt(tes_mask, dtype=int, delimiter=",")
        mask = get_tes_mask(tes_asic)

        bounds = int(fraction * len(tm)), int((1 - fraction) * len(tm))

        # if tod_bounds.exists():
        #     tod_bounds = np.loadtxt(tod_bounds)
        #     valid_tes = get_valid_tes(mask, n_tes=len(signals))
        #     bounds = get_common_range(tod_bounds[valid_tes])

        self.time_0 = t0
        self.time = tm
        self.signals = signals
        self.elevation = interp_elevation
        self.azimuth = interp_azimuth
        self.mask = mask
        self.bounds = bounds


    @field_validator("tes", mode='before')
    @classmethod
    def parse_tes(cls, v):
        if isinstance(v, str) and v.lower() == "all":
            return list(range((2 * NPIXELS)))

        elif isinstance(v, int):
            return list(v)

        elif isinstance(v, list):
            return v

        raise ValueError(f"Unable to parse TES {v}")


class Planck(BaseModel):
    path: Path
    frequency: float



class Config(BaseModel):
    output: Path

    def __post_init__(self):
        self.output.mkdir(parents=True, exist_ok=True)
