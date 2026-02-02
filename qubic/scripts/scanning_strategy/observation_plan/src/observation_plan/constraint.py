import numpy as np
from astroplan import Constraint
from astropy.coordinates import get_body, AltAz


class SunSeparationConstraint(Constraint):
    """
    Constraint ensuring that the separation between the Sun and the target
    satisfies the specified minimum and/or maximum bounds. It is used in
    astronomical observations to establish constraints on target visibility
    concerning the Sun's position.

    :param min: Minimum acceptable separation between the Sun and the target (inclusive).
        A value of `None` indicates no minimum limit.
    :type: `astropy.units.Quantity` or None

    :param max: Maximum acceptable separation between the Sun and the target (inclusive).
        A value of `None` indicates no maximum limit.
    :type: `astropy.units.Quantity` or None
    """

    def __init__(self, min=None, max=None):

        self.min = min
        self.max = max

    def compute_constraint(self, times, observer, targets):
        """
        Compute whether the targets meet the solar separation constraint as observed
        from a specific location and time.

        This method determines the angular separation between the Sun and the specified
        target(s) at the given times and location. It evaluates whether the separation
        is within the bounds specified by `self.min` and `self.max`.

        :param times: The time or times at which to compute the constraint.
        :type: astropy.time.Time

        :param observer: The location of the observer on Earth.
        :type: astropy.coordinates.EarthLocation

        :param targets:  The celestial object(s) for which to compute the solar separation
            constraint.
        :type: astropy.coordinates.SkyCoord or array-like

        :return: A boolean array indicating whether each target satisfies the solar
            separation constraint at the given times. Each element corresponds to a
            target-time combination, with `True` indicating the constraint is met.
        :rtype: numpy.ndarray
        """

        # Build the local AltAz reference frame for the observer at the requested times
        altaz_frame = AltAz(obstime=times,
                            location=observer.location)

        # Ensure `targets` is at least 1D so we can index it uniformly (even for a single target)
        targets = np.atleast_1d(targets)

        # Transform targets into the observer's AltAz frame.
        # The `[:, np.newaxis]` adds a dummy axis so the output has shape (n_targets, n_times),
        # matching the time grid and enabling broadcasting for separations.
        disk_altaz = targets[:, np.newaxis].transform_to(altaz_frame)

        # use `get_body` rather than `get_sun` here, since
        # it returns the Sun's coordinates in an observer
        # centred frame, so the separation is as-seen
        # by the observer.
        # 'get_sun' returns ICRS coords.
        sun = get_body("sun", time=times, location=observer.location).transform_to(altaz_frame)

        # Compute the angular separation between each target and the Sun
        # Result shape is (n_targets, n_times).
        solar_separation = disk_altaz.separation(sun)

        # Case 1: only an upper bound is provided -> separation must be <= max
        if self.min is None and self.max is not None:
            mask = self.max >= solar_separation

        # Case 2: only a lower bound is provided -> separation must be >= min
        elif self.max is None and self.min is not None:
            mask = self.min <= solar_separation

        # Case 3: both bounds are provided -> min <= separation <= max
        elif self.min is not None and self.max is not None:
            mask = ((self.min <= solar_separation) &
                    (solar_separation <= self.max))

        # Case 4: neither bound is provided -> this is a configuration error
        else:
            raise ValueError("No max and/or min specified in "
                             "SunSeparationConstraint.")

        # Reduce over targets: require that ALL targets satisfy the constraint at each time.
        # The output is a boolean array with one value per time.
        return np.all(mask, axis=0)


class MoonSeparationConstraint(Constraint):
    """
    Represents a constraint on the separation between the Moon and the target.

    The MoonSeparationConstraint evaluates whether targets satisfy the specified
    minimum and/or maximum angular separation from the Moon at given times and
    locations. This is useful for observational planning to avoid targets being
    too close to or too far from the Moon during observations. The constraint
    uses a specified or default solar system ephemeris for its calculations.


    :param min: Minimum acceptable separation between the Moon and the target
        (inclusive). None indicates no lower limit.
    :type: `astropy.units.Quantity` or None

    :param max:  Maximum acceptable separation between the Moon and the target
        (inclusive). None indicates no upper limit.
    :type: `astropy.units.Quantity` or None

    :param ephemeris: Ephemeris to use for Moon position calculations. If None, the default
        ephemeris set by `astropy.coordinates.solar_system_ephemeris` will be
        used.
    :type: str or None
    """

    def __init__(self, min=None, max=None, ephemeris=None):
        """
        :param min: Minimum acceptable separation between moon and target (inclusive).
            `None` indicates no limit.
        :type: `astropy.units.Quantity` or `None` (optional)

        :param max: Maximum acceptable separation between moon and target (inclusive).
            `None` indicates no limit.
        :type: `astropy.units.Quantity` or `None` (optional)

        :param ephemeris: Ephemeris to use. If not given, use the one set with
            `astropy.coordinates.solar_system_ephemeris.set` (which is
            set to 'builtin' by default).
        :type: str, optional
        """

        self.min = min
        self.max = max
        self.ephemeris = ephemeris

    def compute_constraint(self, times, observer, targets):
        """
        Computes a constraint based on the separation between celestial targets and the Moon, considering specific
        minimum and maximum separation thresholds. The function calculates the angular separation for each target
        in the local AltAz reference frame of the observer and determines whether the targets meet the separation
        criteria.

        :param times: The times at which to compute the separation.
        :type: `astropy.time.Time`

        :param observer: The location of the observer on Earth.
        :type: `astropy.coordinates.EarthLocation`

        :param targets:  The celestial targets for which to compute the separation constraint.
        :type: `astropy.coordinates.SkyCoord` or array-like

        :return: A boolean array indicating whether the targets satisfy the separation constraint for all provided times.
        :rtype: numpy.ndarray
        """

        # Build the local AltAz reference frame for the observer at the requested times
        altaz_frame = AltAz(obstime=times,
                            location=observer.location)

        # Ensure `targets` is at least 1D so we can index it uniformly (even for a single target)
        targets = np.atleast_1d(targets)

        # use `get_body`, since it returns the Moon's coordinates in an observer
        # centred frame, so the separation is as-seen by the observer.
        # 'get_moon' returns ICRS coords.
        # note to future editors - the order matters here
        # moon.separation(targets) is NOT the same as targets.separation(moon)
        # the former calculates the separation in the frame of the moon coord
        # which is GCRS, and that is what we want.
        moon = get_body("moon", times, location=observer.location, ephemeris=self.ephemeris).transform_to(altaz_frame)

        # Transform targets into the observer's AltAz frame.
        # The `[:, np.newaxis]` adds a dummy axis so the output has shape (n_targets, n_times),
        # matching the time grid and enabling broadcasting for separations.
        disk_altaz = targets[:, np.newaxis].transform_to(altaz_frame)

        # Compute the angular separation between each target and the Moon
        # Result shape is (n_targets, n_times).
        moon_separation = disk_altaz.separation(moon)

        # Case 1: only an upper bound is provided -> separation must be <= max
        if self.min is None and self.max is not None:
            mask = self.max >= moon_separation

        # Case 2: only a lower bound is provided -> separation must be >= min
        elif self.max is None and self.min is not None:
            mask = self.min <= moon_separation

        # Case 3: both bounds are provided -> min <= separation <= max
        elif self.min is not None and self.max is not None:
            mask = ((self.min <= moon_separation) &
                    (moon_separation <= self.max))

        # Case 4: neither bound is provided -> this is a configuration error
        else:
            raise ValueError("No max and/or min specified in "
                             "MoonSeparationConstraint.")

        # Reduce over targets: require that ALL targets satisfy the constraint at each time.
        # The output is a boolean array with one value per time.
        return np.all(mask, axis=0)