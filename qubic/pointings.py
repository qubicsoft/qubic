# coding: utf-8
from __future__ import division

import numpy as np
from astropy.time import TimeDelta
from numpy.random import random_sample as randomu
from pyoperators import (
    Cartesian2SphericalOperator, Rotation3dOperator,
    Spherical2CartesianOperator)
from pysimulators import (
    PointingHorizontal, CartesianEquatorial2GalacticOperator,
    CartesianEquatorial2HorizontalOperator,
    CartesianHorizontal2EquatorialOperator)
from pysimulators.interfaces.healpy import Cartesian2HealpixOperator

__all__ = ['QubicPointing',
           'create_random_pointings',
           'create_sweeping_pointings']

DOMECLAT = -(75 + 6 / 60)
DOMECLON = 123 + 20 / 60


class QubicPointing(PointingHorizontal):
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
    pointing = QubicPointing((azimuth, elevation, pitch))
    pointing = QubicPointing(azimuth=azimuth, elevation=elevation,
                             angle_hwp=angle_hwp)

    """
    MANDATORY_NAMES = 'azimuth', 'elevation'
    DEFAULT_DATE_OBS = '2016-01-01 00:00:00'
    DEFAULT_DTYPE = [('azimuth', float), ('elevation', float),
                     ('pitch', float), ('angle_hwp', float), ('time', float)]
    DEFAULT_SAMPLING_PERIOD = 1
    DEFAULT_LATITUDE = DOMECLAT
    DEFAULT_LONGITUDE = DOMECLON

    def __new__(cls, x=None, azimuth=None, elevation=None, pitch=None,
                angle_hwp=None, time=None, date_obs=None, sampling_period=None,
                latitude=None, longitude=None, dtype=None, copy=True):
        return PointingHorizontal.__new__(
            cls, x=x, azimuth=azimuth, elevation=elevation, pitch=pitch,
            angle_hwp=angle_hwp, time=time, date_obs=date_obs,
            sampling_period=sampling_period, latitude=latitude,
            longitude=longitude, dtype=dtype, copy=copy)

    def tohealpix(self, nside):
        time = self.date_obs + TimeDelta(self.time, format='sec')
        r = (Cartesian2HealpixOperator(nside) *
             CartesianEquatorial2GalacticOperator() *
             CartesianHorizontal2EquatorialOperator('NE', time, self.latitude,
                                                    self.longitude) *
             Spherical2CartesianOperator('azimuth,elevation', degrees=True))
        return r(np.array([self.azimuth, self.elevation]).T)


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
    p = QubicPointing.zeros(
        npointings, date_obs=date_obs, sampling_period=sampling_period,
        latitude=latitude, longitude=longitude)
    time = p.date_obs + TimeDelta(p.time, format='sec')
    rotation = (
        Cartesian2SphericalOperator('azimuth,elevation', degrees=True) *
        CartesianEquatorial2HorizontalOperator(
            'NE', time, p.latitude, p.longitude) *
        Rotation3dOperator("ZY'", center[0], 90 - center[1], degrees=True) *
        Spherical2CartesianOperator('zenith,azimuth', degrees=True))
    coords = rotation(np.asarray([theta.T, phi.T]).T)
    p.azimuth = coords[..., 0]
    p.elevation = coords[..., 1]
    p.pitch = pitch
    return p


def create_sweeping_pointings(
        center, duration, sampling_period, angspeed, delta_az,
        nsweeps_per_elevation, angspeed_psi, maxpsi, date_obs=None,
        latitude=None, longitude=None, return_hor=True):
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
    out = QubicPointing.zeros(
        nsamples, date_obs=date_obs, sampling_period=sampling_period,
        latitude=latitude, longitude=longitude)
    racenter = center[0]
    deccenter = center[1]
    backforthdt = delta_az / angspeed * 2

    jd = (out.date_obs + TimeDelta(out.time, format='sec')).jd
    tsamples = _gst2lst(_jd2gst(jd), out.longitude)

    # compute the sweep number
    isweeps = np.floor(out.time / backforthdt).astype(int)

    # azimuth/elevation of the center of the field as a function of time
    azcenter, elcenter = _equ2hor(racenter, deccenter, out.latitude, tsamples)

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

    if not return_hor:
        # convert them to RA, Dec
        raptg, decptg = _hor2equ(azptg, elptg, out.latitude, tsamples)
        theta = 90 - decptg
        phi = raptg
        pointings = np.array([theta, phi, pitch]).T
        return pointings

    out.azimuth = azptg
    out.elevation = elptg
    out.pitch = pitch
    return out


def _jd2gst(jd):
    """
    Convert julian dates into Greenwich Sidereal Time.

    From Practical Astronomy With Your Calculator.
    """
    jd0 = np.floor(jd - 0.5) + 0.5
    T = (jd0 - 2451545.0) / 36525
    T0 = 6.697374558 + 2400.051336 * T + 0.000025862 * T**2
    T0 %= 24
    ut = (jd - jd0) * 24
    T0 += ut * 1.002737909
    T0 %= 24
    return T0


def _gst2lst(gst, geolon):
    """
    Convert Greenwich Sidereal Time into Local Sidereal Time.

    """
    # geolon: Geographic longitude EAST in degrees.
    return (gst + geolon / 15.) % 24


def _equ2hor(ra, dec, geolat, lst):
    """
    Convert from ra/dec to az/el (by Ken Genga).

    """
    # Imports
    from numpy import arccos, arcsin, cos, pi, sin, where

    d2r = pi/180.0
    r2d = 180.0/pi
    sin_dec = sin(dec*d2r)
    phi_rad = geolat*d2r
    sin_phi = sin(phi_rad)
    cos_phi = cos(phi_rad)
    ha = 15.0*_ra2ha(ra, lst)
    sin_el = sin_dec*sin_phi + cos(dec*d2r)*cos_phi*cos(ha*d2r)
    el = arcsin(sin_el)*r2d

    az = arccos((sin_dec - sin_phi * sin_el) / (cos_phi * cos(el * d2r)))
    az = where(sin(ha*d2r) > 0.0, 2.0*pi-az, az)*r2d

    # Later
    return az, el


def _hor2equ(az, el, geolat, lst):
    """
    Convert from az/el to ra/dec (by Ken Genga).

    """
    # Imports
    from numpy import arccos, arcsin, cos, pi, sin, where

    d2r = pi/180.0
    r2d = 180.0/pi
    az_r = az*d2r
    el_r = el*d2r
    geolat_r = geolat*d2r

    # Convert to equatorial coordinates
    cos_el = cos(el_r)
    sin_el = sin(el_r)
    cos_phi = cos(geolat_r)
    sin_phi = sin(geolat_r)
    cos_az = cos(az_r)
    sin_dec = sin_el*sin_phi + cos_el*cos_phi*cos_az
    dec = arcsin(sin_dec) * r2d
    cos_ha = (sin_el - sin_phi * sin_dec) / (cos_phi * cos(dec * d2r))
    cos_ha = where(cos_ha <= -1.0, -0.99999, cos_ha)
    cos_ha = where(cos_ha >= 1.0,  0.99999, cos_ha)
    ha = arccos(cos_ha)
    ha = where(sin(az_r) > 0.0, 2.0 * pi - ha, ha) * r2d

    ra = lst*15.0-ha
    ra = where(ra >= 360.0, ra - 360.0, ra)
    ra = where(ra < 0.0, ra + 360.0, ra)

    # Later
    return ra, dec


def _ra2ha(ra, lst):
    """
    Converts a right ascension to an hour angle.

    """
    return (lst - ra / 15.0) % 24
