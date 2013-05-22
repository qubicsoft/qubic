# coding: utf-8
from __future__ import division

import astropy
import numpy as np
from astropy.time import Time

from numpy.random import random_sample as randomu

__all__ = ['create_random_pointings',
           'create_sweeping_pointings']


DOMECLON = 123 + 20 / 60
DOMECLAT = -(75 + 6 / 60)

def create_random_pointings(npointings, dtheta):
    """
    Return the Euler angles (φ,θ,ψ) of the ZY'Z'' intrinsic rotation
    as (θ,φ,ψ) triplets.

    """
    dtheta=np.radians(dtheta)
    theta = np.degrees(np.arccos(np.cos(dtheta) + (1 - np.cos(dtheta)) *
                       randomu(npointings)))
    phi = randomu(npointings) * 360
    pitch = randomu(npointings) * 360
    pointings = np.array([theta, phi, pitch]).T
    return pointings

def create_sweeping_pointings(center, duration, sampling_period, angspeed,
                              delta_az, nsweeps_el, angspeed_psi, maxpsi,
                              lon=DOMECLON, lat=DOMECLAT,
                              date_obs='2014-01-01 00:00:00', return_hor=False):
    """
    Return pointing according to the sweeping strategy:
    Sweep around the tracked FOV center azimuth at a fixed elevation, and
    update elevation towards the FOV center at discrete times.

    Parameters
    ----------
    center : array-like of size 2
        The R.A. and Declination of the center of the FOV.
    duration : float
        The duration of the observation, in hours.
    sampling_period : float
        The sampling period of the pointings.
    angspeed : float
        The pointing angular speed, in deg / s.
    delta_az : float
        The sweeping extent in degrees.
    nsweeps_el : int
        The number of sweeps during a phase of constant elevation.
    angspeed_psi : float
        The pitch angular speed, in deg / s.
    maxpsi : float
        The maximum pitch angle, in degrees.
    lon : float, optional
        The observer's longitude. Default is DOMEC's.
    lat : float, optional
        The observer's latitude. Default is DOMEC's.
    date_obs : str, optional
        The starting date of the observation (UTC).
    return_hor : bool
        If True, return the azimuth and elevation of the pointings.

    Returns
    -------
    pointings : ndarray of shape (N,3)
        The Euler angles (φ,θ,ψ) of the ZY'Z'' intrinsic rotation as (θ,φ,ψ)
        triplets for the sweeping strategy.

    """
    t0 = Time(date_obs, scale='utc')
    racenter = center[0]
    deccenter = center[1]
    backforthdt = delta_az / angspeed * 2
    
    # time samples
    nsamples = duration * 3600 / sampling_period
    tunix = t0.unix + np.arange(nsamples) * sampling_period
    #XXX Astropy's time module doesn't handle leap seconds correctly
    if astropy.__version__ < '0.4':
        tunix -= 27
    tsamples = _gst2lst(_jd2gst(_unix2jd(tunix)), lon)

    # count of the number of sweeps
    nsweeps = duration * 3600 / backforthdt
    swnum = np.floor(np.linspace(0, nsweeps, nsamples)).astype(int)

    # azimuth/elevation of the center of the field as a function of time
    azcenter, elcenter = _equ2hor(racenter, deccenter, lat, tsamples)

    # compute azimuth offset for all time samples
    daz = np.arange(nsamples) * sampling_period * angspeed
    daz = daz % (delta_az * 2)
    mask = daz > delta_az
    daz[mask] = -daz[mask] + 2 * delta_az
    daz = daz - delta_az / 2

    # elevation is kept constant during nsweeps_el sweeps
    elcst = np.zeros(nsamples)
    nelevation = max(swnum // nsweeps_el) + 1
    for i in xrange(nelevation):
        mask = swnum // nsweeps_el == i
        elcst[mask] = np.mean(elcenter[mask])

    # azimuth and elevations to use for pointing
    azptg = azcenter + daz
    elptg = elcst

    if return_hor:
        return azptg, elptg

    # convert them to RA, Dec
    raptg, decptg = _hor2equ(azptg, elptg, lat, tsamples)
    theta = 90 - decptg
    phi = raptg

    ### scan psi as well
    pitch = (np.arange(nsamples) * sampling_period * angspeed_psi)
    pitch = pitch % (4 * maxpsi)
    mask = pitch > (2 * maxpsi)
    pitch[mask] = -pitch[mask] + 4 * maxpsi
    pitch = pitch - maxpsi
    
    pointings = np.array([theta, phi, pitch]).T
    return pointings


def _unix2jd(unixtime):
    """
    Convert unix time into julian day
    """
    # Jan, 1st 1970, 00:00:00 is jd = 2440587.5
    return unixtime / 86400 + 2440587.5


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
   sin_el  = sin_dec*sin_phi + cos(dec*d2r)*cos_phi*cos(ha*d2r)
   el = arcsin(sin_el)*r2d

   az = arccos( (sin_dec-sin_phi*sin_el)/(cos_phi*cos(el*d2r)))
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
   az_r     = az*d2r
   el_r     = el*d2r
   geolat_r = geolat*d2r

   # Convert to equatorial coordinates
   cos_el  = cos(el_r)
   sin_el  = sin(el_r)
   cos_phi = cos(geolat_r)
   sin_phi = sin(geolat_r)
   cos_az  = cos(az_r)
   sin_dec = sin_el*sin_phi + cos_el*cos_phi*cos_az
   dec     = arcsin(sin_dec)*r2d
   cos_ha  = (sin_el-sin_phi*sin_dec)/(cos_phi*cos(dec*d2r))
   cos_ha  = where(cos_ha <= -1.0, -0.99999, cos_ha)
   cos_ha  = where(cos_ha >=  1.0,  0.99999, cos_ha)
   ha      = arccos(cos_ha)
   ha      = where( sin(az_r) > 0.0 , 2.0*pi-ha, ha)*r2d

   ra      = lst*15.0-ha
   ra = where(ra >= 360.0, ra-360.0, ra)
   ra = where(ra <    0.0, ra+360.0, ra)

   # Later
   return ra, dec


def _ra2ha(ra, lst):
   """
   Converts a right ascension to an hour angle.
   """
   return (lst - ra / 15.0) % 24

