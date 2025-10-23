import numpy as np
from astroplan import Constraint
from astropy.coordinates import get_body, AltAz


class SunSeparationConstraint(Constraint):
    """
    Constrain the distance between the Sun and some targets.
    """

    def __init__(self, min=None, max=None):
        """
        Parameters
        ----------
        min : `~astropy.units.Quantity` or `None` (optional)
            Minimum acceptable separation between Sun and target (inclusive).
            `None` indicates no limit.
        max : `~astropy.units.Quantity` or `None` (optional)
            Maximum acceptable separation between Sun and target (inclusive).
            `None` indicates no limit.
        """
        self.min = min
        self.max = max

    def compute_constraint(self, times, observer, targets):
        # use get_body rather than get sun here, since
        # it returns the Sun's coordinates in an observer
        # centred frame, so the separation is as-seen
        # by the observer.
        # 'get_sun' returns ICRS coords.
        altaz_frame = AltAz(obstime=times,
                            location=observer.location)

        targets = np.atleast_1d(targets)

        disk_altaz = targets[:, np.newaxis].transform_to(altaz_frame)

        sun = get_body("sun", time=times, location=observer.location).transform_to(altaz_frame)

        solar_separation = disk_altaz.separation(sun)

        if self.min is None and self.max is not None:
            mask = self.max >= solar_separation
        elif self.max is None and self.min is not None:
            mask = self.min <= solar_separation
        elif self.min is not None and self.max is not None:
            mask = ((self.min <= solar_separation) &
                    (solar_separation <= self.max))
        else:
            raise ValueError("No max and/or min specified in "
                             "SunSeparationConstraint.")
        return np.all(mask, axis=0)


class MoonSeparationConstraint(Constraint):
    """
    Constrain the distance between the Earth's moon and some targets.
    """

    def __init__(self, min=None, max=None, ephemeris=None):
        """
        Parameters
        ----------
        min : `~astropy.units.Quantity` or `None` (optional)
            Minimum acceptable separation between moon and target (inclusive).
            `None` indicates no limit.
        max : `~astropy.units.Quantity` or `None` (optional)
            Maximum acceptable separation between moon and target (inclusive).
            `None` indicates no limit.
        ephemeris : str, optional
            Ephemeris to use.  If not given, use the one set with
            ``astropy.coordinates.solar_system_ephemeris.set`` (which is
            set to 'builtin' by default).
        """
        self.min = min
        self.max = max
        self.ephemeris = ephemeris

    def compute_constraint(self, times, observer, targets):

        altaz_frame = AltAz(obstime=times,
                            location=observer.location)

        targets = np.atleast_1d(targets)

        moon = get_body("moon", times, location=observer.location, ephemeris=self.ephemeris).transform_to(altaz_frame)
        # note to future editors - the order matters here
        # moon.separation(targets) is NOT the same as targets.separation(moon)
        # the former calculates the separation in the frame of the moon coord
        # which is GCRS, and that is what we want.

        disk_altaz = targets[:, np.newaxis].transform_to(altaz_frame)

        moon_separation = disk_altaz.separation(moon)

        if self.min is None and self.max is not None:
            mask = self.max >= moon_separation
        elif self.max is None and self.min is not None:
            mask = self.min <= moon_separation
        elif self.min is not None and self.max is not None:
            mask = ((self.min <= moon_separation) &
                    (moon_separation <= self.max))
        else:
            raise ValueError("No max and/or min specified in "
                             "MoonSeparationConstraint.")
        return np.all(mask, axis=0)
