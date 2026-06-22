import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class AmTemplate:
    """
    Metadata for one ALMA AM cookbook atmospheric template.

    pwv_scale1_mm is the PWV represented by the template when the AM H2O
    scale factor is equal to 1.0.
    """

    site: str
    season: str
    h2o_percentile: int
    # e' il PWV di riferimento del template
    # ex: se water_vapor_scale = 1.0,
    # il template MAM_50 rappresenta un'atmosfera con PWV = 1.22496 mm
    pwv_scale1_mm: float
    uses_tground: bool = True

    @property
    def name(self) -> str:
        suffix = "_Tground" if self.uses_tground else ""
        return f"{self.site}_{self.season}_{self.h2o_percentile}{suffix}"

    @property
    def filename(self) -> str:
        """
        nome del file .amc
        """
        return f"{self.name}.amc"

    def path(self, cookbook_dir: str | Path) -> Path:
        return Path(cookbook_dir) / "sites" / self.site / self.filename

    def water_vapor_scale_from_pwv(self, pwv_mm: float) -> float:
        """
        pwv_mm: PWV inferred from APEX data in the time interval of our skydip
        """
        if not np.isfinite(pwv_mm) or pwv_mm <= 0:
            raise ValueError("pwv_mm must be finite and strictly positive.")

        return pwv_mm / self.pwv_scale1_mm


@dataclass(frozen=True)
class ScaledAmTemplate:
    """
    AM template selected for one dataset, together with the water-vapor
    scale factor inferred from the APEX PWV.
    """

    template: AmTemplate
    pwv_apex_mm: float
    water_vapor_scale: float

    @property
    def name(self) -> str:
        return self.template.name

    @property
    def filename(self) -> str:
        return self.template.filename

    def path(self, cookbook_dir: str | Path) -> Path:
        return self.template.path(cookbook_dir)


@dataclass(frozen=True)
class AmTemplateLibrary:
    """
    Collection of AM template PWV reference values.
    """

    templates: tuple[AmTemplate, ...]

    @classmethod
    def default(cls) -> "AmTemplateLibrary":
        return cls(templates=(AmTemplate(site="ALMA", season="DJF", h2o_percentile=50, pwv_scale1_mm=2.49424),
                              AmTemplate(site="ALMA", season="JFM", h2o_percentile=50, pwv_scale1_mm=2.70543),
                              AmTemplate(site="ALMA", season="MAM", h2o_percentile=50, pwv_scale1_mm=1.22496),
                              AmTemplate(site="ALMA", season="JJA", h2o_percentile=50, pwv_scale1_mm=0.65579),
                              AmTemplate(site="ALMA", season="SON", h2o_percentile=50, pwv_scale1_mm=0.77899),
                              AmTemplate(site="ALMA", season="annual", h2o_percentile=50, pwv_scale1_mm=1.03217)))

    def get(self, season: str, h2o_percentile: int = 50, site: str = "ALMA") -> AmTemplate:

        for template in self.templates:
            if template.site == site and template.season == season and template.h2o_percentile == h2o_percentile:
                return template

        available = ", ".join(f"{t.site}_{t.season}_{t.h2o_percentile}" for t in self.templates)

        raise ValueError(
            f"No ALMA template available for site={site!r}, season={season!r}, "
            f"h2o_percentile={h2o_percentile}. Available templates: {available}."
        )

    def get_scaled(
        self,
        pwv_apex_mm: float,
        season: str,
        h2o_percentile: int = 50,
        site: str = "ALMA",
    ) -> ScaledAmTemplate:
        """
        Select an AM template and compute the water-vapor scale factor
        from the APEX PWV measured during one dataset.
        """

        template = self.get(
            site=site,
            season=season,
            h2o_percentile=h2o_percentile,
        )

        water_vapor_scale = template.water_vapor_scale_from_pwv(
            pwv_mm=pwv_apex_mm,
        )

        return ScaledAmTemplate(
            template=template,
            pwv_apex_mm=pwv_apex_mm,
            water_vapor_scale=water_vapor_scale,
        )

if __name__ == "__main__":
    from qubic.lib.Calibration.source_calibration.common import io
    from qubic.lib.Calibration.source_calibration.skydip.atmosphere.providers.apex_pwv import ApexPWVTimeSeries
    from qubic.lib.Calibration.source_calibration.skydip.atmosphere.config_atmosphere import load_atmosphere_config

    config_path = Path(sys.argv[1]).expanduser().resolve()
    config = load_atmosphere_config(config_path)

    # test con PWV derivato da APEX
    dataset = io.read_qubic_dataset(dataset_path=config.dataset_path)
    apex = ApexPWVTimeSeries.from_csv(config.apex_pwv_csv)
    pwv_apex_mm = apex.median_pwv_between(start_time_utc=dataset.start_time_utc,
                                          stop_time_utc=dataset.stop_time_utc)

    library = AmTemplateLibrary.default()

    template = library.get(site=config.am.site,
                           season=config.am.season,
                           h2o_percentile=config.am.h2o_percentile)

    print("AM template library")
    print(f"  config_path: {config_path}")
    print(f"  executable: {config.am.executable}")
    print(f"  cookbook_dir: {config.am.cookbook_dir}")
    print(f"  site: {config.am.site}")
    print(f"  season: {config.am.season}")
    print(f"  h2o_percentile: {config.am.h2o_percentile}")
    print(f"  template_name: {template.name}")
    print(f"  template_filename: {template.filename}")
    print(f"  template_path: {template.path(config.am.cookbook_dir)}")
    print(f"  template_path_exists: {template.path(config.am.cookbook_dir).exists()}")
    print(f"  pwv_scale1_mm: {template.pwv_scale1_mm:.5f}")
    print(f"  pwv_apex_mm: {pwv_apex_mm:.5f}")
    print(f"  water_vapor_scale: {template.water_vapor_scale_from_pwv(pwv_apex_mm):.5f}")