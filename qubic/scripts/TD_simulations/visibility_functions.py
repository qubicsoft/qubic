def daily_observing_time(time_s, date_obs, max_gap_s=10.0):
    """
    Compute total observing time per day from a stream of pointing timestamps.

    Parameters
    ----------
    time_s : array-like of float
        Pointing times in seconds relative to `date_obs`.
    date_obs : astropy.time.Time
        Reference absolute time corresponding to zero in `time_s`.
    max_gap_s : float, optional
        Maximum allowed gap (in seconds) between two consecutive pointings to
        treat them as continuous observing. Larger gaps are treated as non-observing
        periods and are excluded. Default is 10.0.

    Returns
    -------
    days : astropy.time.Time
        One timestamp per observed day, at 00:00:00 of that day.
    sec_day : numpy.ndarray
        Total observing time per day in seconds 
    """
    u = date_obs.unix + time_s

    # Keep only consecutive intervals that represent continuous observing
    dt = np.diff(u)
    valid = (dt > 0.0) & (dt <= float(max_gap_s))
    if not np.any(valid):
        return Time([], format="unix", scale=t.scale), np.array([], dtype=float)

    # Start/end times of valid intervals
    starts = u[:-1][valid]
    ends = u[1:][valid]

    # Day index for each interval start/end.
    d0 = (starts // 86400.0).astype(np.int64)
    d1 = (ends // 86400.0).astype(np.int64)

    dmin = d0.min()
    dmax = d1.max()
    nday = int(dmax - dmin + 1)

    sec_day = np.zeros(nday, dtype=float)

    # Intervals fully inside one day: add full duration to that day.
    same = (d0 == d1)
    if np.any(same):
        idx = d0[same] - dmin
        sec_day += np.bincount(idx, weights=ends[same] - starts[same], minlength=nday)

    # Intervals crossing midnight: split contribution across two days.
    cross = ~same 
    if np.any(cross):
        midnight = (d0[cross] + 1) * 86400.0
        sec_day += np.bincount(d0[cross] - dmin, weights=midnight - starts[cross], minlength=nday)
        sec_day += np.bincount(d1[cross] - dmin, weights=ends[cross] - midnight, minlength=nday)

    # Keep only days with non-zero observing time.
    keep = sec_day > 0.0
    day_idx = np.arange(dmin, dmax+1, dtype=np.int64)[keep]
    days = Time(day_idx * 86400.0, format='unix', scale=date_obs.scale)
    days.format = 'iso'

    return days, sec_day[keep]
