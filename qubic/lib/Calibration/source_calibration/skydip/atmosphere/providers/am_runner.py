import sys
import subprocess
import numpy as np
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


if __name__ == "__main__":

    from qubic.lib.Calibration.source_calibration.common import io
    from qubic.lib.Calibration.source_calibration.skydip.atmosphere.config_atmosphere import load_atmosphere_config
    from qubic.lib.Calibration.source_calibration.skydip.atmosphere.providers.am_template import AmTemplateLibrary
    from qubic.lib.Calibration.source_calibration.skydip.atmosphere.providers.apex_pwv import ApexPWVTimeSeries
    from qubic.lib.Calibration.source_calibration.skydip.atmosphere.providers.weather_station import WeatherStationTimeSeries

    config_path = Path(sys.argv[1]).expanduser().resolve()

    atmosphere_config = load_atmosphere_config(config_path)

    dataset = io.read_qubic_dataset(dataset_path=atmosphere_config.dataset_path)

    skydip_start_time_utc = dataset.start_time_utc
    skydip_stop_time_utc = dataset.stop_time_utc
    skydip_duration_s = dataset.duration_s

    library = AmTemplateLibrary.default()
    template = library.get(site=atmosphere_config.am.site,
                           season=atmosphere_config.am.season,
                           h2o_percentile=atmosphere_config.am.h2o_percentile)

    apex = ApexPWVTimeSeries.from_csv(atmosphere_config.apex_pwv_csv)
    pwv_apex_mm = apex.median_pwv_between(start_time_utc=skydip_start_time_utc,
                                          stop_time_utc=skydip_stop_time_utc)

    water_vapor_scale = template.water_vapor_scale_from_pwv(pwv_apex_mm)

    weather = WeatherStationTimeSeries.from_csv(csv_path=atmosphere_config.weather_station_csv,
                                                datetime_column_index=atmosphere_config.weather_station.datetime_column_index,
                                                temperature_column_index=atmosphere_config.weather_station.temperature_column_index,
                                                delimiter=atmosphere_config.weather_station.delimiter)

    tground_k = weather.mean_temperature_K_between(start_time_utc=skydip_start_time_utc,
                                                   stop_time_utc=skydip_stop_time_utc)

    am_config = AMRunConfig(am_executable=atmosphere_config.am.executable,
                            cookbook_dir=atmosphere_config.am.cookbook_dir,
                            template=template,
                            f_min_ghz=atmosphere_config.am.freq_min_ghz,
                            f_max_ghz=atmosphere_config.am.freq_max_ghz,
                            step_mhz=atmosphere_config.am.step_mhz,
                            zenith_angle_deg=atmosphere_config.am.zenith_angle_deg,
                            tground_k=tground_k,
                            water_vapor_scale=water_vapor_scale)

    output_csv = atmosphere_config.weather_dir / f"{am_config.output_stem()}.csv"

    print("AM runner config")
    print(f"  config_path: {config_path}")
    print(f"  dataset_name: {atmosphere_config.dataset_path.name}")
    print(f"  dataset_duration_s: {skydip_duration_s:.3f}")
    print(f"  skydip_start_time_utc: {skydip_start_time_utc.isoformat()}")
    print(f"  skydip_stop_time_utc: {skydip_stop_time_utc.isoformat()}")
    print(f"  pwv_apex_mm: {pwv_apex_mm:.4f}")
    print(f"  template_pwv_scale1_mm: {template.pwv_scale1_mm:.4f}")
    print(f"  water_vapor_scale: {water_vapor_scale:.4f}")
    print(f"  tground_k: {tground_k:.4f}")
    print(f"  amc_path: {am_config.amc_path}")
    print(f"  amc_path_exists: {am_config.amc_path.exists()}")
    print(f"  output_csv: {output_csv}")
    print("  command:")
    print("    " + " ".join(am_config.command()))

    try:

        generated_csv = write_am_spectrum_to_csv(config=am_config,
                                                 output_csv=output_csv)

        print(f"  generated_csv: {generated_csv}")

    except RuntimeError as exc:

        print("\nAM execution failed. Full diagnostic follows:\n")
        print(exc)

        raise