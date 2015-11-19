# coding: utf-8
from __future__ import division
from astropy.io import fits
from ConfigParser import ConfigParser
from glob import glob
from os.path import join
from pysimulators import Layout, LayoutGrid
from .calfiles import PATH
from .horns import HornLayout
import numpy as np
import os
import re

__all__ = ['QubicCalibration']

FILE_DETARRAY = 'CalQubic_DetArray_v*.fits'
FILE_HORNARRAY = 'CalQubic_HornArray_v*.fits'
FILE_OPTICS = 'CalQubic_Optics_v*'
FILE_PRIMBEAM = 'CalQubic_PrimBeam_v*.fits'


class QubicCalibration(object):
    """
    Class representing the QUBIC calibration tree. It stores the calibration
    file names and "hardcoded" values and provides access to them.

    If the path name of a calibration file is relative, it is first searched
    relatively to the working directory and if not found, in the calibration
    path.

    """
    def __init__(self, path=PATH, detarray=FILE_DETARRAY,
                 hornarray=FILE_HORNARRAY, optics=FILE_OPTICS,
                 primbeam=FILE_PRIMBEAM):
        """
        Parameters
        ----------
        path : str, optional
            The directory path of the calibration tree. The default one is
            the one that is contained in the qubic package.
        detarray : str, optional
            The detector array layout calibration file name.
        hornarray : str, optional
            The horn array layout calibration file name.
        optics : str, optional
            The optics parameters calibration file name.
        primbeam : str, optional
            The primary beam parameter calibration file name.

        """
        self.path = os.path.abspath(path)
        self.detarray = self._newest(detarray)
        self.hornarray = self._newest(hornarray)
        self.optics = self._newest(optics)
        self.primbeam = self._newest(primbeam)

    def __str__(self):
        state = [('path', self.path),
                 ('detarray', self.detarray),
                 ('hornarray', self.hornarray),
                 ('optics', self.optics),
                 ('primbeam', self.primbeam)]
        return '\n'.join([a + ': ' + repr(v) for a, v in state])

    __repr__ = __str__

    def get(self, name, *args):
        """
        Access calibration files.

        Parameters
        ----------
        name : str
            One of the following:
                - 'detarray'
                - 'hornarray'
                - 'optics'
                - 'primbeam'

        """
        if name == 'detarray':
            hdus = fits.open(self.detarray)
            version = hdus[0].header['format version']
            corner = hdus[2].data
            shape = corner.shape[:-2]
            n = shape[0] * shape[1]
            if version == '1.0':
                removed = np.zeros(shape, bool)
                index = np.arange(n, dtype=np.int32).reshape(shape)
                quadrant = np.zeros(shape, np.int8)
                efficiency = np.ones(shape)
            else:
                removed = hdus[3].data.view(bool)
                index = hdus[4].data
                quadrant = hdus[5].data
                if version > '2.0':
                    efficiency = hdus[6].data
                else:
                    efficiency = np.ones(shape)
            return shape, corner, removed, index, quadrant, efficiency

        elif name == 'hornarray':
            hdus = fits.open(self.hornarray)
            version = hdus[0].header['format version']
            if version == '1.0':
                h = hdus[0].header
                spacing = h['spacing']
                center = hdus[1].data
                shape = center.shape[:-1]
                layout = Layout(shape, center=center, radius=h['innerrad'],
                                open=None)
                layout.spacing = spacing
            elif version == '2.0':
                h = hdus[0].header
                spacing = h['spacing']
                xreflection = h['xreflection']
                yreflection = h['yreflection']
                radius = h['radius']
                selection = ~hdus[1].data.view(bool)
                layout = LayoutGrid(
                    removed.shape, spacing, selection=selection, radius=radius,
                    xreflection=xreflection, yreflection=yreflection,
                    open=None)
            else:
                h = hdus[1].header
                spacing = h['spacing']
                xreflection = h['xreflection']
                yreflection = h['yreflection']
                angle = h['angle']
                radius = h['radius']
                selection = ~hdus[2].data.view(bool)
                shape = selection.shape
                layout = HornLayout(
                    shape, spacing, selection=selection, radius=radius,
                    xreflection=xreflection, yreflection=yreflection,
                    angle=angle, startswith1=True, id=None, open=None)
                layout.id = np.arange(len(layout))
            layout.center = np.concatenate(
                [layout.center, np.full_like(layout.center[..., :1], 0)], -1)
            layout.open = np.ones(len(layout), bool)
            return layout

        elif name == 'optics':
            dtype = [('name', 'S16'), ('temperature', float),
                     ('transmission', float), ('emissivity', float),
                     ('nstates_pol', int)]
            if self.optics.endswith('fits'):
                header = fits.open(self.optics)[0].header
                return {'focal length': header['flength'],
                        'detector efficiency': 1.,
                        'components': np.empty(0, dtype=dtype)}
            parser = ConfigParser()
            parser.read(self.optics)
            keys = 'focal length',
            out = dict((key, parser.getfloat('general', key)) for key in keys)
            raw = parser.items('components')
            components = np.empty(len(raw), dtype=dtype)
            for i, r in enumerate(raw):
                component = (r[0],) + tuple(float(_) for _ in r[1].split(', '))
                components[i] = component
            out['components'] = components
            return out

        elif name == 'primbeam':
            header = fits.open(self.primbeam)[0].header
            fwhm0_deg = header['fwhm']
            return fwhm0_deg
            # nu0 = hdu['freq']
            # return lambda nu: fwhm0_deg * (nu0 / nu)

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
