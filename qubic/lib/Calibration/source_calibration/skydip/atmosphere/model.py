import sys
import logging
from dataclasses import dataclass
from pathlib import Path

from qubic.lib.Calibration.source_calibration.common.io import prepare_datasets_from_config, iter_saved_datasets
from qubic.lib.Calibration.source_calibration.skydip.atmosphere.config_atmosphere import load_atmosphere_config
from qubic.lib.Calibration.source_calibration.skydip.atmosphere.providers.apex_pwv import ApexPWVTimeSeries
from qubic.lib.Calibration.source_calibration.skydip.atmosphere.providers.weather_station import \
    WeatherStationTimeSeries
from qubic.lib.Calibration.source_calibration.skydip.config_calibration import load_skydip_calibration_config

from qubic.lib.Calibration.source_calibration.skydip.atmosphere.providers.am_template import (
    ScaledAmTemplate,
    AmTemplateLibrary,
)

from qubic.lib.Calibration.source_calibration.skydip.atmosphere.providers.am_runner import (
    AMRunConfig,
    write_am_spectrum_to_csv)

from qubic.lib.Calibration.source_calibration.skydip.atmosphere.spectrum import AtmosphericSpectrum


@dataclass
class AtmosphericModel:
    """
    Atmospheric model associated with one skydip dataset.

    It stores the atmospheric quantities inferred from external data
    and, after running AM, the corresponding atmospheric spectrum.
    """

    pwv_apex_mm: float
    tground_k: float

    scaled_am_template: ScaledAmTemplate

    atmospheric_spectrum: AtmosphericSpectrum | None = None

    @classmethod
    def from_config(cls,
                    dataset,
                    atm_config) -> "AtmosphericModel":
        """
        Build the atmospheric model for one dataset using its atmosphere config.
        """

        apex_pwv = ApexPWVTimeSeries.from_csv(csv_path=atm_config.apex_pwv_csv)

        pwv_apex_mm = apex_pwv.median_pwv_between(start_time_utc=dataset.start_time_utc,
                                                  stop_time_utc=dataset.stop_time_utc)

        weather_station = WeatherStationTimeSeries.from_csv(
            csv_path=atm_config.weather_station_csv,
            datetime_column_index=atm_config.weather_station.datetime_column_index,
            temperature_column_index=atm_config.weather_station.temperature_column_index,
            delimiter=atm_config.weather_station.delimiter,
        )

        tground_k = weather_station.mean_temperature_K_between(start_time_utc=dataset.start_time_utc,
                                                               stop_time_utc=dataset.stop_time_utc)

        template_library = AmTemplateLibrary.default()

        scaled_am_template = template_library.get_scaled(
            pwv_apex_mm=pwv_apex_mm,
            site=atm_config.am.site,
            season=atm_config.am.season,
            h2o_percentile=atm_config.am.h2o_percentile,
        )

        return cls(
            dataset_name=dataset.dataset_name,
            pwv_apex_mm=pwv_apex_mm,
            tground_k=tground_k,
            scaled_am_template=scaled_am_template,
        )

    def to_am_run_config(self,
                         atm_config) -> AMRunConfig:
        """
        Build the AMRunConfig needed to execute AM for this dataset.
        """

        am_output_dir = atm_config.input_dir
        am_output_csv = am_output_dir / f"{am_run_config.output_stem()}.csv"

        return AMRunConfig(
            am_executable=atm_config.am_executable,
            cookbook_dir=atm_config.am_cookbook_dir,
            template=self.scaled_am_template.template,
            f_min_ghz=atm_config.am.freq_min_ghz,
            f_max_ghz=atm_config.am.freq_max_ghz,
            step_mhz=atm_config.am.step_mhz,
            zenith_angle_deg=atm_config.am.zenith_angle_deg,
            tground_k=self.tground_k,
            water_vapor_scale=self.scaled_am_template.water_vapor_scale,
        )

    def get_atmospheric_model(cls,
                              atm_config) -> "AtmosphericModel":
        """
        Run AM for this atmospheric model, write the AM output to CSV,
        read it back as an AtmosphericSpectrum, and store it in the model.
        """

        am_run_config = self.to_am_run_config(
            atm_config=atm_config,
        )

        generated_csv = write_am_spectrum_to_csv(
            config=am_run_config,
            output_csv=output_csv,
        )

        atmospheric_spectrum = AtmosphericSpectrum.from_csv(
            csv_path=generated_csv,
        )

        self.atmospheric_spectrum = atmospheric_spectrum

        return atmospheric_spectrum


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d/%m/%Y | %H:%M:%S",
    )

    logger = logging.getLogger(__name__)

    # 1. Leggo il file YAML di configurazione
    config = load_skydip_calibration_config(Path(sys.argv[1]).expanduser().resolve())

    # 2. Seguo lo stesso ordine di operazioni di io.py:
    #    leggo tutti i dataset QUBICStudio definiti in paths.runs
    #    e salvo i corrispondenti file .npy/.npz nei rispettivi output.
    prepare_datasets_from_config(config=config,
                                 logger=logger)

    # 3. Ora rileggo i file .npy/.npz appena prodotti,
    #    un dataset alla volta, usando il generatore
    for run, dataset in zip(config.paths.runs, iter_saved_datasets(config=config, logger=logger)):

        logger.info("Running quicklook for dataset: %s", dataset.dataset_name)
        # atm_config contiene l'atm config per un dataset
        atm_config = load_atmosphere_config(run.atmospheric_config)

        atmospheric_model = AtmosphericModel.from_config(dataset=dataset,
                                                         atm_config=atm_config)

        am_run_config = atmospheric_model.to_am_run_config(
            atm_config=atm_config,
        )



        logger.info(
            "Atmospheric model for %s: PWV_APEX=%.5f mm, Tground=%.3f K, template=%s, water_vapor_scale=%.5f",
            atmospheric_model.dataset_name,
            atmospheric_model.pwv_apex_mm,
            atmospheric_model.tground_k,
            atmospheric_model.scaled_am_template.name,
            atmospheric_model.scaled_am_template.water_vapor_scale,
        )

        logger.info(
            "Running AM. Output CSV: %s",
            am_output_csv,
        )

        atmospheric_spectrum = atmospheric_model.get_atmospheric_model(
            atm_config=atm_config,
            output_csv=am_output_csv,
        )

        logger.info(
            "Generated AM spectrum: %s",
            atmospheric_spectrum.source_path,
        )

        tau_eff, tb_eff_k = atmospheric_spectrum.get_effective_atmospheric_parameters(
            strategy="band_average",
        )

        logger.info(
            "Effective atmospheric parameters for %s: tau_eff=%.6f, Tb_eff=%.6f K",
            atmospheric_model.dataset_name,
            tau_eff,
            tb_eff_k,
        )

        # passare al template di AM pwv_apex_mm in modo che dentro si calcoli il water_vapor scale factor
        # instazio l'oggeto AM_config che dentro ha tutto e salvo su csv AM
        # calcolo lo spettro dal csv AM