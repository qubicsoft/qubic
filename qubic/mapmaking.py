from __future__ import division

import numpy as np
from itertools import izip
from pyoperators import (
    DiagonalOperator, IdentityOperator, PackOperator, pcg, rule_manager)
from .acquisition import QubicAcquisition
from .utils import progress_bar

__all__ = ['map2tod', 'tod2map_all', 'tod2map_each']


def map2tod(acquisition, map, convolution=False):
    """
    tod = map2tod(acquisition, map)
    tod, convolved_map = map2tod(acquisition, map, convolution=True)

    Parameters
    ----------
    acquisition : QubicAcquisition
        The QUBIC acquisition.
    map : I, QU or IQU maps
        Temperature, QU or IQU maps of shapes npix, (npix, 2), (npix, 3)
        with npix = 12 * nside**2
    convolution : boolean, optional
        Set to True to convolve the input map by a gaussian and return it.

    Returns
    -------
    tod : array
        The Time-Ordered-Data of shape (ndetectors, ntimes).
    convolved_map : array
        The convolved map, if the convolution keyword is set.

    """
    if convolution:
        convolution = acquisition.get_convolution_peak_operator()
        map = convolution(map)
    projection = acquisition.get_projection_peak_operator()
    hwp = acquisition.get_hwp_operator()
    polarizer = acquisition.get_polarizer_operator()
    response = acquisition.get_detector_response_operator()

    with rule_manager(inplace=True):
        H = response * polarizer * (hwp * projection)
    if acquisition.instrument.sky == 'QU':
        H = acquisition.get_subtract_grid_operator() * H

    tod = H(map)

    if convolution:
        return tod, map

    return tod


def _tod2map(acq, tod, coverage_threshold, disp, tol):
    projection = acq.get_projection_peak_operator(verbose=disp)
    coverage = projection.pT1()
    mask = coverage > coverage_threshold
    projection.restrict(mask)
    pack = PackOperator(mask, broadcast='rightward')
    hwp = acq.get_hwp_operator()
    polarizer = acq.get_polarizer_operator()
    response = acq.get_detector_response_operator()
    with rule_manager(inplace=True):
        H = response * polarizer * (hwp * projection)
    if acq.instrument.sky == 'QU':
        H = acq.get_subtract_grid_operator() * H

    invNtt = acq.get_invntt_operator()
    preconditioner = DiagonalOperator(1/coverage[mask], broadcast='rightward')
    solution = pcg(H.T * invNtt * H, H.T(invNtt(tod)), M=preconditioner,
                   disp=disp, tol=tol)
    output_map = pack.T(solution['x'])
    return output_map, coverage


def tod2map_all(acquisition, tod, coverage_threshold=0, disp=True, tol=1e-4):
    """
    Compute map using all detectors.

    map, coverage = tod2map_all(acquisition, tod, [coverage_threshold,
                                coverage_threshold, disp, tol])

    Parameters
    ----------
    acquisition : QubicAcquisition
        The QUBIC acquisition.
    tod : array-like
        The Time-Ordered-Data of shape (ndetectors, ntimes).
    coverage_threshold : float, optional
        The coverage threshold used to reject map pixels from
        the reconstruction.
    disp : boolean, optional
        Display of solver's iterations.
    tol : float, optional
        Solver tolerance.

    Returns
    -------
    map : I, QU or IQU maps
        Temperature, QU or IQU maps of shapes npix, (npix, 2), (npix, 3)
        with npix = 12 * nside**2
    """
    return _tod2map(acquisition, tod, coverage_threshold, disp, tol)


def tod2map_each(acquisition, tod, coverage_threshold=0, disp=True, tol=1e-4):
    """
    Compute average map from each detector.

    map, coverage = tod2map_each(acquisition, tod, [coverage_threshold,
                                 coverage_threshold, disp, tol])

    Parameters
    ----------
    acquisition : QubicAcquisition
        The QUBIC acquisition.
    tod : array-like
        The Time-Ordered-Data of shape (ndetectors, ntimes).
    coverage_threshold : float, optional
        The coverage threshold used to reject map pixels from
        the reconstruction.
    disp : boolean, optional
        Display of solver's iterations.
    tol : float, optional
        Solver tolerance.

    Returns
    -------
    map : I, QU or IQU maps
        Temperature, QU or IQU maps of shapes npix, (npix, 2), (npix, 3)
        with npix = 12 * nside**2

    """
    instrument = acquisition.instrument
    x = np.zeros(instrument.sky.shape)
    n = np.zeros(instrument.sky.size)
    if disp:
        bar = progress_bar(len(instrument), 'TOD2MAP_EACH')
    for i, t in izip(instrument, tod):
        acq = QubicAcquisition(i, acquisition.sampling)
        x_, n_ = _tod2map(acq, t[None, :], coverage_threshold, False, tol)
        x += x_
        n += n_
        if disp:
            bar.update()

    return np.nan_to_num(x / n[:, None]), n
