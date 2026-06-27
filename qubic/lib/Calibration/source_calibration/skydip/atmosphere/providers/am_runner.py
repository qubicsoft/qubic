import subprocess
import astropy.units as u
from pathlib import Path
from dataclasses import dataclass

from qubic.lib.Calibration.source_calibration.skydip.atmosphere.providers.am_template import AmTemplate
from qubic.lib.Calibration.source_calibration.common.utils import _to_value

AM_CSV_HEADER = "frequency_GHz,tau,tx,Trj_K,Tb_K"


@dataclass(frozen=True)
class AMRunConfig:
    """
    Configuration for one AM run.
    """

    am_executable: Path
    cookbook_dir: Path
    template: AmTemplate

    f_min_ghz: u.Quantity
    f_max_ghz: u.Quantity
    step_mhz: u.Quantity
    zenith_angle_deg: u.Quantity

    tground_k: float
    water_vapor_scale: float

    @property
    def amc_path(self) -> Path:
        return self.template.path(self.cookbook_dir)

    def command(self) -> list[str]:
        return [
            str(self.am_executable),
            str(self.amc_path),
            f"{_to_value(self.f_min_ghz, u.GHz):.12g}", "GHz",
            f"{_to_value(self.f_max_ghz, u.GHz):.12g}", "GHz",
            f"{_to_value(self.step_mhz, u.MHz):.12g}", "MHz",
            f"{_to_value(self.zenith_angle_deg, u.deg):.12g}", "deg",
            f"{float(self.tground_k):.12g}", "K",
            f"{float(self.water_vapor_scale):.12g}",
        ]

    def output_stem(self) -> str:
        return (
            f"{self.template.site}_{self.template.season}_{self.template.h2o_percentile}_"
            f"{_to_value(self.f_min_ghz, u.GHz):.0f}_{_to_value(self.f_max_ghz, u.GHz):.0f}GHz_"
            f"z{_to_value(self.zenith_angle_deg, u.deg):.0f}deg_"
            f"T{self.tground_k:.2f}K_"
            f"scale{self.water_vapor_scale:.3f}"
        )


def write_am_output_to_csv(am_stdout: str, output_csv: str | Path) -> Path:
    """
    runner prende stdout di AM e lo converte in CSV leggibile da AtmosphericSpectrum.
    """

    output_csv = output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []

    for line in am_stdout.splitlines():
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        parts = stripped.split()

        if len(parts) < 5:
            continue

        try:
            values = [float(parts[i]) for i in range(5)]
        except ValueError:
            continue

        rows.append(",".join(f"{value:.12g}" for value in values))

    if len(rows) == 0:
        raise ValueError("AM output did not contain any valid spectral rows.")

    with open(output_csv, "w", newline="") as f:
        f.write(AM_CSV_HEADER + "\n")
        f.write("\n".join(rows))
        f.write("\n")

    return output_csv


def write_am_spectrum_to_csv(config: AMRunConfig,
                             output_csv:  Path,
                             cwd: str | Path | None = None) -> Path:


    command = config.command()

    result = subprocess.run(
        command,
        cwd=None if cwd is None else Path(cwd),
        check=False,
        text=True,
        capture_output=True,
    )

    try:
        output_csv = Path(output_csv).expanduser().resolve()

        if output_csv.suffix.lower() != ".csv":
            output_csv = output_csv / f"{config.output_stem()}.csv"

        output_csv.parent.mkdir(parents=True, exist_ok=True)

        written_csv = write_am_output_to_csv(
            am_stdout=result.stdout,
            output_csv=output_csv,
        )
    except ValueError as exc:
        command_text = " ".join(command)
        stdout = result.stdout.strip() if result.stdout else ""
        stderr = result.stderr.strip() if result.stderr else ""

        message = [
            "AM did not produce a usable atmospheric spectrum.",
            f"Return code: {result.returncode}",
            f"Command: {command_text}",
        ]

        if stdout:
            message.extend(["", "AM stdout:", stdout])
        if stderr:
            message.extend(["", "AM stderr:", stderr])

        raise RuntimeError("\n".join(message)) from exc

    if result.returncode != 0:
        print(
            "Warning: AM returned a non-zero exit code "
            f"({result.returncode}), but a valid spectrum was written to {written_csv}. "
            "AM diagnostic output was suppressed."
        )

    return written_csv

