# coding: utf-8
from __future__ import division

import numpy as np
import os
import re
from astropy.io import fits
from glob import glob
from os.path import join

__all__ = ['QubicCalibration']

PATH = join(os.path.dirname(__file__), 'calfiles')

FWHM_DEG = 14
FOCAL_LENGTH = 0.3
NHORNS = 400
HORN_KAPPA = 1.344
HORN_THICKNESS = 0.001
FILE_DETARRAY = 'CalQubic_DetArray_v*.fits'

class QubicCalibration(object):
    """
    Class representing the QUBIC calibration tree. It stores the calibration
    file names and "hardcoded" values and provides access to them.

    If the path name of a calibration file is relative, it is first searched
    relatively to the working directory and if not found, in the calibration
    path.

    """
    def __init__(self, path=PATH, fwhm_deg=FWHM_DEG, focal_length=FOCAL_LENGTH,
                 nhorns=NHORNS, horn_kappa=HORN_KAPPA,
                 horn_thickness=HORN_THICKNESS, detarray=FILE_DETARRAY):
        """
        Parameters
        ----------
        path : str, optional
            The directory path of the calibration tree. The default one is
            the one that is contained in the qubic package.
        fwhm_deg : float, optional
            The primary beam FWHM, in degrees.
        focal_length : float, optional
            The instrument focal length.
        nhorns : int, optional
            The number of back-to-back horns.
        horn_kappa : float, optional
            The horn kappa value.
        horn_thickness : float, optional
            Half the distance between two adjacent horn collecting surfaces,
            in meters.
        detarray : str, optional
            The detector array calibration file name.

        """
        self.path = os.path.abspath(path)
        self.fwhm_deg = fwhm_deg
        self.focal_length = focal_length
        self.nhorns = nhorns
        self.horn_kappa = horn_kappa
        self.horn_thickness = horn_thickness
        self.detarray = self._newest(detarray)

    def __str__(self):
        state = [('path', self.path),
                 ('fwhm_deg', self.fwhm_deg),
                 ('focal_length', self.focal_length),
                 ('nhorns', self.nhorns),
                 ('horn_kappa', self.horn_kappa),
                 ('horn_thickness', self.horn_thickness),
                 ('detarray', self.detarray),
                ]
        return '\n'.join([a + ': ' + repr(v) for a,v in state])

    __repr__ = __str__

    def get(self, name, *args):
        """
        Access calibration files.

        Parameters
        ----------
        name : str
            One of the following
                - 'fwhm'
                - 'focal length'
                - 'horn'
                - 'detarray'.

        """
        if name == 'fwhm':
            return self.fwhm_deg
        elif name == 'focal length':
            return self.focal_length
        elif name == 'horn':
            return self.nhorns, self.horn_kappa, self.horn_thickness
        if name == 'detarray':
            hdus = fits.open(self.detarray)
            version = hdus[0].header['format version']
            center, corner = hdus[1].data, hdus[2].data
            shape = center.shape[:-1]
            n = shape[0] * shape[1]
            if version == '1.0':
                removed = np.zeros(shape, bool)
                index = np.arange(n, dtype=np.int16).reshape(shape)
                quadrant = np.zeros(shape, np.int8)
            else:
                removed = hdus[3].data.view(bool)
                index = hdus[4].data
                quadrant = hdus[5].data
            return shape, center, corner, removed, index, quadrant
        raise ValueError("Invalid calibration item: '{}'".format(name))

    def _newest(self, filename):
        if '*' not in filename:
            if not os.path.exists(filename):
                filename = join(self.path, filename)
            if not os.path.exists(filename):
                raise ValueError("No calibration file '{}'.".format(filename))
            return os.path.abspath(filename)

        filenames = glob(filename)
        if len(filenames) == 0:
            filename = join(self.path, filename)
            filenames = glob(filename)
            if len(filenames) == 0:
                raise ValueError("No calibration files '{}'.".format(filename))
        regex = re.compile(filename.replace('*', '(.*)'))
        version = sorted(regex.search(f).group(1) for f in filenames)[-1]
        return os.path.abspath(filename.replace('*', version))
