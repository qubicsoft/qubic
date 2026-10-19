import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell
def _():
    from dataclasses import dataclass, replace
    from pathlib import Path

    import astropy.units as u
    import healpy as hp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go

    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time
    from datetime import datetime, timezone

    from plotly.subplots import make_subplots
    from qubicpack.qubicfp import qubicfp
    from scipy.stats import binned_statistic_2d

    return (
        AltAz,
        EarthLocation,
        Path,
        SkyCoord,
        Time,
        binned_statistic_2d,
        dataclass,
        datetime,
        go,
        hp,
        make_subplots,
        mo,
        np,
        px,
        qubicfp,
        replace,
        timezone,
        u,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #TODO:
    - testare con maschera (scegliere una maschera che sia coerente con segnali multi-dataset)
    - testare con maschera che taglia 10% ad inizio e fine
    - introdurre salvataggio html
    - correggere per la rotazione terrestre le scansioni ad elev costante
    """)
    return


@app.cell
def _(EarthLocation, Path, u):
    root = Path("input/qubic")
    site = EarthLocation(lat=-24.18696100 * u.deg, lon=-66.47827748 * u.deg, height=4868.590 * u.m)
    return root, site


@app.cell
def _(
    AltAz,
    Path,
    SkyCoord,
    Time,
    binned_statistic_2d,
    dataclass,
    go,
    hp,
    make_subplots,
    np,
    px,
    qubicfp,
    replace,
    u,
):
    @dataclass
    class Observation:

        time: np.ndarray
        signals: np.ndarray          
        azimuth: np.ndarray
        elevation: np.ndarray
        ra: np.ndarray
        dec: np.ndarray

        @classmethod
        def load(cls, paths, site):
            if isinstance(paths, (str, Path)):
                paths = [paths]

            chunks = []
            for path in paths:

                qubic = qubicfp()
                qubic.read_qubicstudio_dataset(str(path))

                time, signals = qubic.tod()

                # alcuni dataset hanno un endianess diverso. np.asarray uniforma
                azimuth = np.interp(
                    time,
                    np.asarray(qubic.timeaxis(datatype="az"), dtype=float),
                    np.asarray(qubic.azimuth(), dtype=float))
    
                elevation = np.interp(
                    time,
                    np.asarray(qubic.timeaxis(datatype="el"), dtype=float),
                    np.asarray(qubic.elevation(), dtype=float))

                chunks.append((time, signals, azimuth, elevation))

                del qubic

            # ordino i datasets in base al primo istante temporale di acquisizione
            # cosi da non dipendere dal nome dei dataset ma dai dati che contengono.
            # 
            # FIXME: 16/0TT/2026 - 21:24
            # potrebbe essere un sorting sbagliato perche' si segue la sorgente
            # tramite un moto ad elevazione crescente ed uno ad elevazione decrescente.
            # Quindi devo ordinare in base alla tupla (tempo, elevazione)
            chunks.sort(key=lambda chunk: chunk[0][0])

            time = np.concatenate([c[0] for c in chunks])
            signals = np.concatenate([c[1] for c in chunks], axis=1)
            azimuth = np.concatenate([c[2] for c in chunks])
            elevation = np.concatenate([c[3] for c in chunks])

            sky = SkyCoord(
                az=azimuth * u.deg,
                alt=elevation * u.deg,
                frame=AltAz(obstime=Time(time, format="unix", scale="utc"), location=site)).icrs

            return cls(time, signals, azimuth, elevation, sky.ra.deg, sky.dec.deg)

        def normalize(self):
            signals = self.signals - np.median(self.signals, axis=1, keepdims=True)
            signals /= signals.std(axis=1, keepdims=True)
            return replace(self, signals=signals)

        def select(self, mask):
            return replace(
                self,
                time=self.time[mask].copy(),
                signals=self.signals[:, mask].copy(),
                azimuth=self.azimuth[mask].copy(),
                elevation=self.elevation[mask].copy(),
                ra=self.ra[mask].copy(),
                dec=self.dec[mask].copy())


        @classmethod
        def combine(cls, observations):

            observations = sorted(observations, key=lambda o: o.time[0])

            return cls(
                time=np.concatenate([o.time for o in observations]),
                signals=np.concatenate([o.signals for o in observations], axis=1),
                azimuth=np.concatenate([o.azimuth for o in observations]),
                elevation=np.concatenate([o.elevation for o in observations]),
                ra=np.concatenate([o.ra for o in observations]),
                dec=np.concatenate([o.dec for o in observations]))

        def make_healpix(self, tes, nside=256):

            pixels = hp.ang2pix(nside, self.ra, self.dec, lonlat=True)
            count = hp.nside2npix(nside)
            hits = np.bincount(pixels, minlength=count)
            total = np.bincount(pixels, weights=self.signals[tes], minlength=count)
            sky = np.full(count, hp.UNSEEN)
            observed = hits > 0
            sky[observed] = total[observed] / hits[observed]
   
            return sky, hits

        def plot_tod(self, tes):
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_scatter(x=self.time,
                            y=self.signals[tes],
                            name="Signal",
                            secondary_y=False)

            fig.add_scatter(x=self.time,
                            y=self.azimuth,
                            name="Azimuth",
                            secondary_y=True)

            fig.update_layout(title=f"TOD - TES {tes}",
                              xaxis_title="Unix time [s]")

            fig.update_yaxes(title_text="Signal",
                             secondary_y=False)

            fig.update_yaxes(title_text="Azimuth [deg]",
                             secondary_y=True)

            return fig

        def plot_intensity_scatter(self, tes):

            fig = go.Figure(
                go.Scattergl(x=self.ra,
                             y=self.dec,
                             mode="markers",
                             marker=dict(size=2,
                                         color=self.signals[tes],
                                         colorscale="RdBu_r",
                                         showscale=True)))

            fig.update_layout(
                title=f"Campioni - TES {tes}",
                xaxis_title="RA [deg]",
                yaxis_title="Dec [deg]")

            fig.update_xaxes(autorange="reversed")

            return fig

        def plot_binned(self, tes, bins=100):

            image, ra_edges, dec_edges, _ = binned_statistic_2d(self.ra,
                                                                self.dec,
                                                                self.signals[tes],
                                                                statistic="median",
                                                                bins=bins)

            fig = px.imshow(
                image.T,
                x=(ra_edges[:-1] + ra_edges[1:]) / 2,
                y=(dec_edges[:-1] + dec_edges[1:]) / 2,
                origin="lower",
                aspect="equal",
                color_continuous_scale="RdBu_r",
                title=f"Mappa binned - TES {tes}",
                labels={
                    "x": "RA [deg]",
                    "y": "Dec [deg]",
                    "color": "Signal"})

            fig.update_xaxes(autorange="reversed")

            return fig

        @staticmethod
        def _project(sky, projector, coord=None):
            nside = hp.get_nside(sky)

            image = projector.projmap(sky,
                                      lambda x, y, z: hp.vec2pix(nside, x, y, z),
                                      coord=coord)

            image[hp.mask_bad(image) | ~np.isfinite(image)] = np.nan

            return image

        def plot_gnomonic(self, tes, nside=256, reso=6.0, xsize=500):
            sky, _ = self.make_healpix(tes, nside)

            projector = hp.projector.GnomonicProj(
                rot=(np.median(self.ra), np.median(self.dec)),
                reso=reso,
                xsize=xsize)

            projector.set_flip("astro")

            image = self._project(sky, projector)

            x, y = projector.ij2xy()
            ra, dec = projector.xy2ang(x.ravel(),
                                       y.ravel(),
                                       lonlat=True)

            ra = ra.reshape(x.shape)
            dec = dec.reshape(y.shape)

            fig = go.Figure(
                go.Heatmap(
                    z=image,
                    x=np.rad2deg(x[0]) * 60,
                    y=np.rad2deg(y[:, 0]) * 60,
                    colorscale="RdBu_r",
                    colorbar=dict(title="Signal"),
                    customdata=np.stack([ra, dec], axis=-1),
                    hovertemplate=(
                        "RA: %{customdata[0]:.4f}°<br>"
                        "Dec: %{customdata[1]:.4f}°<br>"
                        "Signal: %{z:.4f}"
                        "<extra></extra>")))

            fig.update_layout(
                title=f"Gnomonic - TES {tes}",
                height=700,
                xaxis_title="X [arcmin]",
                yaxis_title="Y [arcmin]")

            fig.update_yaxes(scaleanchor="x", scaleratio=1)

            return fig

        def plot_mollweide(self, sky, title="Mappa globale", xsize=1000):

            projector = hp.projector.MollweideProj(coord="G", xsize=xsize)

            image = self._project(sky, projector, coord="C")

            left, right, bottom, top = projector.get_extent()

            x = np.linspace(left, right, image.shape[1])
            y = np.linspace(bottom, top, image.shape[0])

            xx, yy = np.meshgrid(x, y)

            lon, lat = projector.xy2ang(
                xx,
                yy,
                lonlat=True,
                direct=True)

            fig = go.Figure(
                go.Heatmap(
                    z=image,
                    x=x,
                    y=y,
                    colorscale="RdBu_r",
                    colorbar=dict(title="Signal"),
                    customdata=np.stack([lon % 360, lat], axis=-1),
                    hovertemplate=(
                        "l: %{customdata[0]:.2f}°<br>"
                        "b: %{customdata[1]:.2f}°<br>"
                        "Signal: %{z:.4f}"
                        "<extra></extra>")))

            fig.update_layout(title=title, height=600)

            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)

            return fig



    return (Observation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Dataset

    Ogni elemento di `groups` è una singola osservazione. Per un'osservazione
    spezzata su più directory, sostituisci il suo elemento con una lista di
    percorsi: `groups = [[path1, path2], path3, ...]`.
    """)
    return


@app.cell
def _(root):
    groups = sorted(
        (p for p in root.rglob("*right-center_el*") if p.is_dir()),
        key=lambda p: float(p.name.rsplit("_el", 1)[-1]))

    groups
    return (groups,)


@app.cell
def _(groups, mo):
    mo.md(f"""
    **Osservazioni trovate:** {len(groups)}
    """)
    return


@app.cell
def _(Observation, groups, site):
    observations = [Observation.load(paths, site) for paths in groups]
    return (observations,)


@app.cell
def _():
    tes = 95
    nside = 256
    preview_index = 0
    normalize_each = False
    preview_plot = "gnomonic"  # "tod", "scatter", "binned", "gnomonic"
    return normalize_each, nside, preview_index, preview_plot, tes


@app.cell
def _(normalize_each, observations):
    processed = [o.normalize() for o in observations] if normalize_each else observations
    return (processed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Singola osservazione

    Seleziona il tipo di figura modificando `preview_plot` nella cella dei parametri.
    Per isolare un intervallo: `selected = observations[0].select(mask)`.
    """)
    return


@app.cell
def _(nside, preview_index, preview_plot, processed, tes):
    preview = processed[preview_index]

    figures = {
        "tod": lambda: preview.plot_tod(tes),
        "scatter": lambda: preview.plot_intensity_scatter(tes),
        "binned": lambda: preview.plot_binned(tes),
        "gnomonic": lambda: preview.plot_gnomonic(tes, nside)
    }

    fig_gnom = figures[preview_plot]()
    fig_gnom.write_html("gnomic.html")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Mappa globale - coordinate galattiche
    """)
    return


@app.cell
def _(Observation, processed):
    global_observation = Observation.combine(processed)
    return (global_observation,)


@app.cell
def _(global_observation, nside, tes):
    skymap, hits = global_observation.make_healpix(tes, nside)
    return hits, skymap


@app.cell
def _(global_observation, skymap, tes):
    global_observation.plot_mollweide(skymap, title=f"Mappa globale - TES {tes}")
    return


@app.cell
def _(global_observation):
    global_observation.plot_binned(tes=95, bins=200)
    return


@app.cell
def _(hits, mo, np):
    mo.md(f"""
    **Pixel osservati:** {np.count_nonzero(hits):,} · **Campioni:** {hits.sum():,}
    """)
    return


@app.cell
def _(datetime, go, np, observations, timezone):
    fig = go.Figure()

    for obs in sorted(observations, key=lambda o: o.time[0]):

        time_ = datetime.fromtimestamp(obs.time[0], tz=timezone.utc).strftime("%H:%M")

        elevation = np.median(obs.elevation)

        fig.add_trace(go.Scattergl(
            x=obs.ra,
            y=obs.dec,
            mode="markers",
            marker=dict(size=2),
            name=f"{time_} - El {elevation:.1f}°"))

    fig.update_layout(
        title="Copertura delle singole osservazioni",
        xaxis_title="RA [deg]",
        yaxis_title="Dec [deg]",
        height=800)

    fig.update_xaxes(autorange="reversed")

    fig.write_html("copertura_sing_obs.html")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
