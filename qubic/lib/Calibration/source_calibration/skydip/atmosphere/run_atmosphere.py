from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

from qubic.lib.Calibration.source_calibration.skydip.atmosphere.config_atmosphere import load_atmosphere_config
from qubic.lib.Calibration.source_calibration.skydip.atmosphere.providers.apex_pwv import ApexPWVTimeSeries
from qubic.lib.Calibration.source_calibration.skydip.atmosphere.providers.weather_station import \
    WeatherStationTimeSeries
from qubic.lib.Calibration.source_calibration.skydip.atmosphere.providers.am_runner import AMRunConfig, write_am_spectrum_to_csv
from qubic.lib.Calibration.source_calibration.skydip.atmosphere.providers.am_template import (
    ScaledAmTemplate,
    AmTemplateLibrary,
)
from qubic.lib.Calibration.source_calibration.skydip.atmosphere.spectrum import AtmosphericSpectrum


@dataclass
class Atmosphere:

    config: Path
    start_obs: str | datetime
    end_obs: str | datetime
    dataset_name: str | None = None

    # attributi che NON passo dall'esterno,
    # ma che vengono creati automaticamente
    # pwv mediano calcolato sull'intervallo temporale del dataset
    pwv: float = field(init=False)
    t_ground_k: float = field(init=False)
    scaled_am_template: ScaledAmTemplate = field(init=False)
    am_config: AMRunConfig = field(init=False)
    tau: float = field(init=False)
    T_b: float = field(init=False)


    def __post_init__(self):

        config = load_atmosphere_config(self.config)

        # retrieve PWV from Apex
        # apex_object contiene:
        # apex_object.time_utc: array NumPy che contiene i tempi associati (di tipo datetime) ) ai campioni PWV,
        # apex_object.source_path: path risolto completo del file APEX CSV,
        # apex_object.pwv_mm: array NumPy che contiene i PWV APEX in mm letti dal CSV
        apex_object= ApexPWVTimeSeries.from_csv(csv_path=config.apex_pwv_csv)
        self.pwv = apex_object.median_pwv_between(start_time_utc=self.start_obs,
                                                  stop_time_utc=self.end_obs)

        # retrieve ground temperature from weather station
        # weather_station contiene:
        # weather_station.time_utc: array NumPy che contiene i tempi associati (di tipo datetime) ai campioni di temperatura,
        # weather_station.source_path: path risolto completo del file CSV,
        # weather_station.temperature_C: array NumPy che contiene i campioni di temperatura in Celsius letti dal CSV
        weather_station = WeatherStationTimeSeries.from_csv(csv_path=config.weather_station_csv,
                                                            datetime_column_index=config.weather_station.datetime_column_index,
                                                            temperature_column_index=config.weather_station.temperature_column_index,
                                                            delimiter=config.weather_station.delimiter)

        self.t_ground_k = weather_station.mean_temperature_K_between(start_time_utc=self.start_obs,
                                                                     stop_time_utc=self.end_obs)

        # Costruisco una libreria di template AM disponibili
        template_library = AmTemplateLibrary.default()

        # Sceglie il template giusto in base a site, season, h2o_percentile.
        # Calcola il fattore di scala del vapore acqueo usando il PWV APEX del dataset.
        # Restituisce un oggetto ScaledAmTemplat che contiene:
        # self.scaled_am_template.template: oggetto AMTemplate in uso
        # self.scaled_am_template.pwv_apex_mm: PWV APEX del dataset
        # self.scaled_am_template.water_vapor_scale: fattore di scala del vapore acqueo
        self.scaled_am_template = template_library.get_scaled(pwv_apex_mm=self.pwv,
                                                              site=config.am.site,
                                                              season=config.am.season,
                                                              h2o_percentile=config.am.h2o_percentile)
        # run am usando lo scaled_am_template
        self.am_config = AMRunConfig(
            am_executable=config.am_executable,
            cookbook_dir=config.am_cookbook_dir,
            template=self.scaled_am_template.template,
            f_min_ghz=config.am.frequency_min,
            f_max_ghz=config.am.frequency_max,
            step_mhz=config.am.step,
            zenith_angle_deg=config.am.zenith_angle,
            tground_k=self.t_ground_k,
            water_vapor_scale=self.scaled_am_template.water_vapor_scale,
        )

        if self.dataset_name is None:
            spectrum_csv_path = config.input_dir / f"{self.am_config.output_stem()}.csv"
        else:
            dataset_label = str(self.dataset_name).replace("/", "_").replace(" ", "_")
            spectrum_csv_path = config.input_dir / f"{dataset_label}__{self.am_config.output_stem()}.csv"

        spectrum_csv = write_am_spectrum_to_csv(
            config=self.am_config,
            output_csv=spectrum_csv_path,
        )

        spectrum = AtmosphericSpectrum.from_csv(spectrum_csv)

        self.tau, self.T_b = spectrum.get_effective_atmospheric_parameters(strategy=config.parameters_estimation.parameter_strategy)

