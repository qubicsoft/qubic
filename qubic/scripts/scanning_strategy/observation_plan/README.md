# Observation Plan


### Initialize a poetry environment
In the observation_plan project folder, the poetry environment can be initialized by running:
```poetry install```
This command creates a virtual environment in Poetry’s default directory and installs all necessary dependencies.
To create the virtual environment inside the project directory instead, run the following command:
```
poetry config virtualenvs.in-project true --local
poetry install
```

The observation plan is obtained by running:
```
# in the root directory: qubic/scripts/scanning_strategy/observation_plan
poetry run observation_plan configs/conf.toml
```

### Parameters

The configuration file is located in the `configs` folder. 

```
[general]
output_dir = "data/output"
time_resolution = "30 min"
# configuration file for matplotlib plots
mpl_style = "styles/modern_style.mplstyle"

[observatory]
name = "Qubic"
lon = -66.8714
lat = -24.1844
height = 4820.0
pressure = 0.5533
relative_humidity = 0.20
timezone = "America/Argentina/Salta"
description = "Qubic telescope on Alto Chorrillos, Salta"

# constraints for the observation plan
[constraints]
point = [
    { type = "AltitudeConstraint", min = 15.0, max = 85.0 },
    { type = "AirmassConstraint", max = 3.0 },
    { type = "SunSeparationConstraint", min = 50.0 }
]

extended = [
    { type = "AltitudeConstraint", min = 15.0, max = 85.0 },
    { type = "AirmassConstraint", max = 3.0 },
    { type = "SunSeparationConstraint", min = 50.0 },
    { type = "MoonSeparationConstraint", min = 30.0 }
]

# files containing the sources coordinates to observed 
[sources]
regions_file = "data/sources/sky_regions.ecsv"
point_sources_file = "data/sources/point_sources.txt"

# observation parameters
[run]
year = 2026
month = 1
# Radius set for the extended sources (in degrees) 
radius = 14.0
# informations about the sampling of the extended sources
radial_samples = 50
polar_angle_samples = 100
# time resolution of the trajectories in the polar plots
trajectory_time_resolution = "20 min"
monthly_visibility_start = "2026-01-01T00:00:00"
monthly_visibility_end = "2026-05-31T23:59:59"
monthly_heatmap_resolution = "30 min"

[tasks]
load_point_sources = true
load_extended_sources = true
process_moon = true
run_constraints = true
run_trajectory_plots = true
write_trajectories = true
run_monthly_visibility = true
run_monthly_heatmaps = true
plot_sidereal = true
```




