# coding: utf-8
from __future__ import division

import astropy.io.fits as pyfits
import healpy as hp
import numpy as np
import os
import time
import types
import yaml
from glob import glob
from pyoperators.utils import strplural
from .instrument import QubicInstrument
from .operators import HealpixConvolutionGaussianOperator

__all__ = ['QubicConfiguration']

class QubicConfiguration(object):
    """
    The QubicConfiguration class, which represents the instrument and
    pointing setups.

    """
    def __init__(self, pointing, instrument=None, block_id=None,
                 selection=None):
        """
        Parameters
        ----------
        pointing : array-like of shape (n,3) or sequence of
            The triplets (θ,φ,ψ), where (φ,θ,ψ) are the Euler angles
            of the intrinsic ZY'Z'' rotations. Note the ordering of the angles.
            θ : co-latitude
            φ : longitude
            ψ : minus the position angle
        instrument : QubicInstrument, optional
           The Qubic instrumental setup.
        block_id : string or sequence of, optional
           The pointing block identifier.
        selection : integer or sequence of, optional
           The indices of the pointing sequence to be selected to construct
           the pointing configuration.

        """
        if not isinstance(instrument, (QubicInstrument, types.NoneType)):
            raise TypeError("Invalid type for the instrument ('{}' instead of '"
                            "QubicInstrument').".format(type(
                            instrument).__name__))
        if instrument is None:
            instrument = QubicInstrument()
        if not isinstance(pointing, (list, tuple)):
            pointing = (pointing,)
        elif isinstance(pointing, types.GeneratorType):
            pointing = tuple(pointing)
        pointing = [np.asarray(p) for p in pointing]
        if len(pointing) == 3 and all(p.ndim == 0 for p in pointing):
            pointing = [np.hstack(pointing)]
        if any(p.ndim not in (1, 2) or p.shape[-1] != 3 for p in pointing):
            raise ValueError('Invalid pointing dimensions.')
        if len(pointing) > 1 and all(p.ndim == 1 for p in pointing):
            pointing = [np.vstack(pointing)]
        pointing = [np.atleast_2d(p) for p in pointing]
        if selection is None:
            selection = tuple(range(len(pointing)))
        pointing = [pointing[i] for i in selection]
        if block_id is not None:
            block_id = [block_id[i] for i in selection]
        if not isinstance(block_id, (list, tuple, types.NoneType)):
            block_id = (block_id,)
            if any(not isinstance(i, str) for i in block_id):
                raise TypeError('The block id is not a string.')

        self.instrument = instrument
        self.pointing = np.concatenate(pointing)
        self.block = self._get_block(pointing, block_id)

    def __str__(self):
        return 'Pointings:\n    {} in {}\n'.format(self.get_nsamples(),
            strplural('block', len(self.block))) + 'Instrument:\n' + \
            ('\n'.join(('    ') + i for i in str(self.instrument).splitlines()))

    def get_nsamples(self):
        """ Return the number of valid pointings. """
        return sum(self.block.n)

    def get_ndetectors(self):
        """ Return the number of valid detectors. """
        return self.instrument.detector.size

    def get_pointing_hitmap(self, nside=None):
        """
        Return a healpy map whose values are the number of times a pointing
        hits the pixel.

        """
        if nside is None:
            nside = self.instrument.sky.nside
        hit = np.zeros(12 * nside**2)
        theta, phi = self.pointings[...,0], self.pointings[...,1]
        ipixel = hp.ang2pix(nside, np.radians(theta), np.radians(phi))
        for i in ipixel:
            hit[i] += 1
        return hit

    def get_convolution_peak_operator(self, fwhm=np.radians(0.64883707),
                                      **keywords):
        return HealpixConvolutionGaussianOperator(self.instrument.sky.nside,
                                                  fwhm=fwhm, **keywords)

    def get_projection_peak_operator(self, kmax=2):
        return self.instrument.get_projection_peak_operator(self.pointing,
                                                            kmax=kmax)

    @classmethod
    def load(cls, filename, instrument=None, selection=None):
        """
        Load a QUBIC configuration, and info.

        obs, info = QubicConfiguration.load(filename, [instrument=None,
                                            selection=None])

        Parameters
        ----------
        filename : string
           The QUBIC configuration file name.
        instrument : QubicInstrument, optional
           The Qubic instrumental setup.
        selection : integer or sequence of
           The indices of the pointing blocks to be selected to construct
           the pointing configuration.

        Returns
        -------
        obs : QubicConfiguration
           The QUBIC configuration instance as read from the file.
        info : string
           The info file stored alongside the configuration.

        """
        if not isinstance(filename, str):
            raise TypeError("The input filename is not a string.")
        if instrument is None:
            instrument = cls._get_instrument_from_file(filename)
        with open(os.path.join(filename, 'info.txt')) as f:
            info = f.read()
        ptg, ptg_id = cls._get_files_from_selection(filename, 'ptg', selection)
        return QubicConfiguration(ptg, instrument=instrument, selection= \
                                  selection, block_id=ptg_id), info

    @classmethod
    def load_observation(cls, filename, instrument=None, selection=None):
        """
        Load a QUBIC configuration, info and TOD.

        obs, tod, info = QubicConfiguration.load_observation(filename,
                             [instrument=None, selection=None])

        Parameters
        ----------
        filename : string
           The QUBIC configuration file name.
        instrument : QubicInstrument, optional
           The Qubic instrumental setup.
        selection : integer or sequence of
           The indices of the pointing blocks to be selected to construct
           the pointing configuration.

        Returns
        -------
        obs : QubicConfiguration
           The QUBIC configuration instance as read from the file.
        tod : ndarray
           The stored time-ordered data.
        info : string
           The info file stored alongside the configuration.

        """
        obs, info = cls.load(filename, instrument=instrument,
                             selection=selection)
        tod, tod_id = obs._get_files_from_selection(filename, 'tod', selection)
        if len(tod) != len(obs.block):
            raise ValueError('The number of pointing and tod files is not the s'
                             'ame.')
        if any(p != t for p, t in zip(obs.block.identifier, tod_id)):
            raise ValueError('Incompatible pointing and tod files.')
        tod = np.hstack(tod)
        return obs, tod, info

    @classmethod
    def load_simulation(cls, filename, instrument=None, selection=None):
        """
        Load a simulation, including the QUBIC configuration, info, TOD and
        input map.

        obs, input_map, tod, info = QubicConfiguration.load_simulation(filename,
                                        [instrument=None, selection=None])

        Parameters
        ----------
        filename : string
           The QUBIC configuration file name.
        instrument : QubicInstrument, optional
           The Qubic instrumental setup.
        selection : integer or sequence of
           The indices of the pointing blocks to be selected to construct
           the pointing configuration.

        Returns
        -------
        obs : QubicConfiguration
           The QUBIC configuration instance as read from the file.
        input_map : Healpy map
           The simulation input map.
        tod : Tod
           The stored time-ordered data.
        info : string
           The info file of the simulation.

        """
        obs, tod, info = cls.load_observation(filename, instrument=instrument,
                                              selection=selection)
        input_map = hp.read_map(os.path.join(filename, 'input_map.fits'))
        return obs, input_map, tod, info

    def save(self, filename, info):
        """
        Write a Qubic configuration to disk.
    
        Parameters
        ----------
        filename : string
            The output path of the directory in which the configuration will be
            saved.
        info : string
            All information deemed necessary to describe the configuration.
    
        """
        self._save_configuration(filename, info)
        self._save_ptg(filename)

    def save_observation(self, filename, tod, info):
        """
        Write a QUBIC configuration to disk with a TOD.
    
        Parameters
        ----------
        filename : string
            The output path of the directory in which the simulation will be
            saved.
        tod : array-like
            The simulated time ordered data, of shape (ndetectors, npointings).
        info : string
            All information deemed necessary to describe the simulation.
    
        """
        self._save_configuration(filename, info)
        self._save_ptg_tod(filename, tod)

    def save_simulation(self, filename, input_map, tod, info):
        """
        Write a QUBIC configuration to disk with a TOD and an input image.
    
        Parameters
        ----------
        filename : string
            The output path of the directory in which the simulation will be
            saved.
        input_map : ndarray, optional
            For simulations, the input Healpix map.
        tod : array-like
            The simulated time ordered data, of shape (ndetectors, npointings).
        info : string
            All information deemed necessary to describe the simulation.
    
        """
        self.save_observation(filename, tod, info)
        hp.write_map(os.path.join(filename, 'input_map.fits'), input_map)

    def _save_configuration(self, filename, info):
        # create directory
        try:
            os.mkdir(filename)
        except OSError:
            raise OSError("The path '{}' already exists.".format(filename))

        # instrument state
        with open(os.path.join(filename, 'instrument.txt'), 'w') as f:
            f.write(str(self.instrument))

        # info file
        with open(os.path.join(filename, 'info.txt'), 'w') as f:
            f.write(info)

    def _save_ptg(self, filename):
        for b in self.block:
            postfix = self._get_time_id() + '.fits'
            ptg = self.pointing[b.start:b.stop]
            file_ptg = os.path.join(filename, 'ptg_' + postfix)
            hdu_ptg = pyfits.PrimaryHDU(ptg)
            pyfits.HDUList([hdu_ptg]).writeto(file_ptg)

    def _save_ptg_tod(self, filename, tod):
        for b in self.block:
            postfix = self._get_time_id() + '.fits'
            p = self.pointing[b.start:b.stop]
            t = tod[:,b.start:b.stop]
            file_ptg = os.path.join(filename, 'ptg_' + postfix)
            file_tod = os.path.join(filename, 'tod_' + postfix)
            hdu_ptg = pyfits.PrimaryHDU(p)
            hdu_tod = pyfits.PrimaryHDU(t)
            pyfits.HDUList([hdu_ptg]).writeto(file_ptg)
            pyfits.HDUList([hdu_tod]).writeto(file_tod)

    @staticmethod
    def _get_block(pointing, block_id):
        npointings = [p.shape[0] for p in pointing]
        start = np.concatenate([[0], np.cumsum(npointings)[:-1]])
        stop = np.cumsum(npointings)
        block = np.recarray(len(pointing), dtype=[('start', int), ('stop', int),
                                                  ('n', int), ('id', 'S29')])
        block.n = npointings
        block.start = start
        block.stop = stop
        block.identifier = block_id if block_id is not None else ''
        return block

    @staticmethod
    def _get_instrument_from_file(filename):
        with open(os.path.join(filename, 'instrument.txt')) as f:
            keywords = yaml.load(f.read())
        return QubicInstrument(**keywords)

    @staticmethod
    def _get_files_from_selection(filename, filetype, selection):
        """ Read files from selection, without reading them twice. """
        files = sorted(glob(os.path.join(filename, filetype + '*.fits')))
        if selection is None:
            return [pyfits.open(f)[0].data for f in files], \
                   [f[-13:-5] for f in files]
        if not isinstance(selection, (list, tuple)):
            selection = (selection,)
        iuniq, inv = np.unique(selection, return_inverse=True)
        uniq_data = [pyfits.open(files[i])[0].data for i in iuniq]
        uniq_id = [files[i][-13:-5] for i in iuniq]
        return [uniq_data[i] for i in inv], [uniq_id[i] for i in inv]

    @staticmethod
    def _get_time_id():
        t = time.time()
        return time.strftime('%Y:%m:%d_%H:%M:%S', time.localtime(t)) + \
               '{:.9f}'.format(t-int(t))[1:]
