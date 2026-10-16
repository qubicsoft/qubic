import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from astropy.coordinates import SkyCoord, AltAz, EarthLocation
    from qubicpack.qubicfp import qubicfp
    from pathlib import Path
    import astropy.units as u 

    return Path, np, qubicfp


@app.cell
def _(Path):
    path = Path("input/qubic/2026-09-29_00.28.36__right-center_el31.7/")
    return (path,)


@app.cell
def _(path, qubicfp):
    qubic = qubicfp()
    qubic.read_qubicstudio_dataset(str(path))
    return (qubic,)


@app.cell
def _(np, qubic):
    tm, signals = qubic.tod()

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
    return


@app.cell
def _():
    tes_idx = 95
    ra = 244.9994772674375
    dec = -61.447033097594954

    lon = -66.8714
    lat = -24.1844
    height = 4820.0
    return


app._unparsable_cell(
    r"""
    # QUBIC location
    location = EarthLocation(
        lat=lat * u.deg,
        lon=lon * u.deg,
        height = heigh * 
    )
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
