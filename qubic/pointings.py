# coding: utf-8
from __future__ import division

import numpy as np
from astropy.time import Time, TimeDelta
from numpy.random import random_sample as randomu
from pyoperators import (
    Cartesian2SphericalOperator, Rotation3dOperator,
    Spherical2CartesianOperator)
from pyoperators.utils import deprecated, isscalarlike
from pysimulators import (
    PointingHorizontal, CartesianEquatorial2GalacticOperator,
    CartesianEquatorial2HorizontalOperator,
    CartesianHorizontal2EquatorialOperator,
    SphericalEquatorial2GalacticOperator,
    SphericalGalactic2EquatorialOperator,
    SphericalEquatorial2HorizontalOperator,
    SphericalHorizontal2EquatorialOperator)
from pysimulators.interfaces.healpy import Cartesian2HealpixOperator

__all__ = ['QubicSampling',
           'create_random_pointings',
           'create_sweeping_pointings',
           'equ2gal',
           'equ2hor',
           'gal2equ',
           'gal2hor',
           'hor2equ',
           'hor2gal']

DOMECLAT = -(75 + 6 / 60)
DOMECLON = 123 + 20 / 60


class QubicSampling(PointingHorizontal):
    """
    Attributes
    ----------
    azimuth : array-like
        The pointing azimuth [degrees].
    elevation : array-like
        The pointing elevation [degrees].
    pitch : array-like
        The instrument pitch angle [degrees].
    angle_hwp : array-like
        The half-wave plate angle [degrees].
    time : array-like
        Elapsed time for each pointing since the observation start date,
        in seconds.
    date_obs : string or astropy.time.Time
        The observation start date.
    sampling_period : float
        The sampling period [s].
    latitude : float
        Telescope latitude [degrees].
    longitude : float
        Telescope longitude [degrees].

    Examples
    --------
    pointing = QubicSampling(azimuth=azimuth, elevation=elevation,
                             angle_hwp=angle_hwp, sampling_period=...)
    pointing = QubicSampling(azimuth, elevation, pitch, angle_hwp,
                             sampling_period=...)

    """
    DEFAULT_DATE_OBS = '2016-01-01 00:00:00'
    DEFAULT_SAMPLING_PERIOD = 1
    DEFAULT_LATITUDE = DOMECLAT
    DEFAULT_LONGITUDE = DOMECLON

    def __init__(self, *args, **keywords):
        angle_hwp = keywords.get('angle_hwp', None)
        if len(args) == 4:
            args = list(args)
            angle_hwp = args.pop()
        PointingHorizontal.__init__(self, angle_hwp=angle_hwp, *args,
                                    **keywords)

    @property
    @deprecated('use len() instead.')
    def size(self):
        return len(self)

    def tohealpix(self, nside):
        time = self.date_obs + TimeDelta(self.time, format='sec')
        c2h = Cartesian2HealpixOperator(nside)
        e2g = CartesianEquatorial2GalacticOperator()
        h2e = CartesianHorizontal2EquatorialOperator(
            'NE', time, self.latitude, self.longitude)
        s2c = Spherical2CartesianOperator('azimuth,elevation', degrees=True)
        rotation = c2h(e2g(h2e(s2c)))
        return rotation(np.array([self.azimuth, self.elevation]).T)

QubicPointing = QubicSampling


def create_random_pointings(center, npointings, dtheta, date_obs=None,
                            sampling_period=None, latitude=None,
                            longitude=None):
    """
    Return pointings randomly and uniformly distributed in a spherical cap.

    Parameters
    ----------
    center : 2-tuple
        The R.A. and declination of the center of the FOV, in degrees.
    npointings : int
        The number of requested pointings
    dtheta : float
        The maximum angular distance to the center.
    date_obs : str or astropy.time.Time, optional
        The starting date of the observation (UTC).
    sampling_period : float, optional
        The sampling period of the pointings, in seconds.
    latitude : float, optional
        The observer's latitude [degrees]. Default is DOMEC's.
    longitude : float, optional
        The observer's longitude [degrees]. Default is DOMEC's.

    """
    cosdtheta = np.cos(np.radians(dtheta))
    theta = np.degrees(np.arccos(cosdtheta +
                                 (1 - cosdtheta) * randomu(npointings)))
    phi = randomu(npointings) * 360
    pitch = randomu(npointings) * 360
    p = QubicSampling(
        npointings, date_obs=date_obs, sampling_period=sampling_period,
        latitude=latitude, longitude=longitude)
    time = p.date_obs + TimeDelta(p.time, format='sec')
    c2s = Cartesian2SphericalOperator('azimuth,elevation', degrees=True)
    e2h = CartesianEquatorial2HorizontalOperator(
        'NE', time, p.latitude, p.longitude)
    rot = Rotation3dOperator("ZY'", center[0], 90-center[1], degrees=True)
    s2c = Spherical2CartesianOperator('zenith,azimuth', degrees=True)
    rotation = c2s(e2h(rot(s2c)))
    coords = rotation(np.asarray([theta.T, phi.T]).T)
    p.azimuth = coords[..., 0]
    p.elevation = coords[..., 1]
    p.pitch = pitch
    return p


def create_sweeping_pointings(
        center, duration, sampling_period, angspeed, delta_az,
        nsweeps_per_elevation, angspeed_psi, maxpsi, date_obs=None,
        latitude=None, longitude=None):
    """
    Return pointings according to the sweeping strategy:
    Sweep around the tracked FOV center azimuth at a fixed elevation, and
    update elevation towards the FOV center at discrete times.

    Parameters
    ----------
    center : array-like of size 2
        The R.A. and Declination of the center of the FOV.
    duration : float
        The duration of the observation, in hours.
    sampling_period : float
        The sampling period of the pointings, in seconds.
    angspeed : float
        The pointing angular speed, in deg / s.
    delta_az : float
        The sweeping extent in degrees.
    nsweeps_per_elevation : int
        The number of sweeps during a phase of constant elevation.
    angspeed_psi : float
        The pitch angular speed, in deg / s.
    maxpsi : float
        The maximum pitch angle, in degrees.
    latitude : float, optional
        The observer's latitude [degrees]. Default is DOMEC's.
    longitude : float, optional
        The observer's longitude [degrees]. Default is DOMEC's.
    date_obs : str or astropy.time.Time, optional
        The starting date of the observation (UTC).
    return_hor : bool, optional
        Obsolete keyword.

    Returns
    -------
    pointings : QubicPointing
        Structured array containing the azimuth, elevation and pitch angles,
        in degrees.

    """
    nsamples = int(np.ceil(duration * 3600 / sampling_period))
    out = QubicPointing(
        nsamples, date_obs=date_obs, sampling_period=sampling_period,
        latitude=latitude, longitude=longitude)
    racenter = center[0]
    deccenter = center[1]
    backforthdt = delta_az / angspeed * 2

    # compute the sweep number
    isweeps = np.floor(out.time / backforthdt).astype(int)

    # azimuth/elevation of the center of the field as a function of time
    azcenter, elcenter = equ2hor(racenter, deccenter, out.time,
                                 date_obs=out.date_obs, latitude=out.latitude,
                                 longitude=out.longitude)

    # compute azimuth offset for all time samples
    daz = out.time * angspeed
    daz = daz % (delta_az * 2)
    mask = daz > delta_az
    daz[mask] = -daz[mask] + 2 * delta_az
    daz -= delta_az / 2

    # elevation is kept constant during nsweeps_per_elevation
    elcst = np.zeros(nsamples)
    ielevations = isweeps // nsweeps_per_elevation
    nelevations = ielevations[-1] + 1
    for i in xrange(nelevations):
        mask = ielevations == i
        elcst[mask] = np.mean(elcenter[mask])

    # azimuth and elevations to use for pointing
    azptg = azcenter + daz
    elptg = elcst

    ### scan psi as well
    pitch = out.time * angspeed_psi
    pitch = pitch % (4 * maxpsi)
    mask = pitch > (2 * maxpsi)
    pitch[mask] = -pitch[mask] + 4 * maxpsi
    pitch -= maxpsi

    out.azimuth = azptg
    out.elevation = elptg
    out.pitch = pitch
    return out


def _format_sphconv(a, b, date_obs=None, time=None):
    incoords = np.empty(np.broadcast(a, b).shape + (2,))
    incoords[..., 0] = a
    incoords[..., 1] = b
    if date_obs is None:
        return incoords
    time = Time(date_obs if isscalarlike(time)
                         else [date_obs], scale='utc') + \
           TimeDelta(time, format='sec')
    return incoords, time


def equ2gal(ra, dec):
    """
    equ2gal(ra, dec) -> l, b
    Equatorial to galactic spherical conversion. Angles are in degrees.

    """
    incoords = _format_sphconv(ra, dec)
    outcoords = SphericalEquatorial2GalacticOperator(degrees=True)(incoords)
    return outcoords[..., 0], outcoords[..., 1]


def gal2equ(l, b):
    """
    gal2equ(l, b) -> ra, dec
    Galactic to equatorial spherical conversion. Angles are in degrees.

    """
    incoords = _format_sphconv(l, b)
    outcoords = SphericalGalactic2EquatorialOperator(degrees=True)(incoords)
    return outcoords[..., 0], outcoords[..., 1]


def equ2hor(ra, dec, time, date_obs=QubicPointing.DEFAULT_DATE_OBS,
            latitude=DOMECLAT, longitude=DOMECLON):
    """
    equ2hor(ra, dec, time, [date_obs, [latitude, [longitude]]]) -> az, el
    Equatorial to horizontal spherical conversion. Angles are in degrees.

    Parameters
    ----------
    time : array-like
        Elapsed time in seconds since date_obs.
    date_obs : string
        The starting date, UTC.
    latitude : float
        The observer's latitude geolocation. Default is Dome C.
    longitude : float
        The observer's longitude geolocation. Default is Dome C.

    Example
    -------
    >>> equ2hor(0, 0, 0, date_obs='2000-01-01 00:00:00')
    (array(135.71997181016644), array(-10.785386358099927))

    """
    incoords, time = _format_sphconv(ra, dec, date_obs, time)
    outcoords = SphericalEquatorial2HorizontalOperator(
        'NE', time, latitude, longitude, degrees=True)(incoords)
    return outcoords[..., 0], outcoords[..., 1]


def hor2equ(azimuth, elevation, time, date_obs=QubicPointing.DEFAULT_DATE_OBS,
            latitude=DOMECLAT, longitude=DOMECLON):
    """
    hor2equ(az, el, time, [date_obs, [latitude, [longitude]]]) -> ra, dec
    Horizontal to equatorial spherical conversion. Angles are in degrees.


    Parameters
    ----------
    time : array-like
        Elapsed time in seconds since date_obs.
    date_obs : string
        The starting date, UTC.
    latitude : float
        The observer's latitude geolocation. Default is Dome C.
    longitude : float
        The observer's longitude geolocation. Default is Dome C.

    Example
    -------
    >>> hor2equ(135.71997181016644, -10.785386358099927, 0,
    ...         date_obs='2000-01-01 00:00:00')
    (array(1.1927080055488187e-14), array(-1.2722218725854067e-14))

    """
    incoords, time = _format_sphconv(azimuth, elevation, date_obs, time)
    outcoords = SphericalHorizontal2EquatorialOperator(
        'NE', time, latitude, longitude, degrees=True)(incoords)
    return outcoords[..., 0], outcoords[..., 1]


def gal2hor(l, b, time, date_obs=QubicPointing.DEFAULT_DATE_OBS,
            latitude=DOMECLAT, longitude=DOMECLON):
    """
    gal2hor(l, b, time, [date_obs, [latitude, [longitude]]]) -> az, el
    Galactic to horizontal spherical conversion. Angles are in degrees.


    Parameters
    ----------
    time : array-like
        Elapsed time in seconds since date_obs.
    date_obs : string
        The starting date, UTC.
    latitude : float
        The observer's latitude geolocation. Default is Dome C.
    longitude : float
        The observer's longitude geolocation. Default is Dome C.

    Example
    -------
    >>> gal2hor(0, 0, 0)
    (array(50.35837815921487), array(39.212362279976155))

    """
    incoords, time = _format_sphconv(l, b, date_obs, time)
    g2e = SphericalGalactic2EquatorialOperator(degrees=True)
    e2h = SphericalEquatorial2HorizontalOperator(
        'NE', time, latitude, longitude, degrees=True)
    outcoords = e2h(g2e(incoords))
    return outcoords[..., 0], outcoords[..., 1]


def hor2gal(azimuth, elevation, time, date_obs=QubicPointing.DEFAULT_DATE_OBS,
            latitude=DOMECLAT, longitude=DOMECLON):
    """
    hor2gal(az, el, time, [date_obs, [latitude, [longitude]]]) -> l, b
    Horizontal to galactic spherical conversion. Angles are in degrees.

    Parameters
    ----------
    time : array-like
        Elapsed time in seconds since date_obs.
    date_obs : string
        The starting date, UTC.
    latitude : float
        The observer's latitude geolocation. Default is Dome C.
    longitude : float
        The observer's longitude geolocation. Default is Dome C.

    Example
    -------
    >>> hor2gal(50.35837815921487, 39.212362279976155, 0)
    (array(4.452776554048925e-14), array(-7.63333123551244e-14))

    """
    incoords, time = _format_sphconv(azimuth, elevation, date_obs, time)
    h2e = SphericalHorizontal2EquatorialOperator(
        'NE', time, latitude, longitude, degrees=True)
    e2g = SphericalEquatorial2GalacticOperator(degrees=True)
    outcoords = e2g(h2e(incoords))
    return outcoords[..., 0], outcoords[..., 1]
