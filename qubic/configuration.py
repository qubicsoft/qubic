# coding: utf-8
from __future__ import division

import astropy.io.fits as pyfits
import healpy as hp
import numpy as np
import os
import time
import yaml
from glob import glob
from pyoperators.utils import ifirst
from pysimulators import Configuration
from .calibration import QubicCalibration
from .instrument import QubicInstrument
from .operators import HealpixConvolutionGaussianOperator

__all__ = ['QubicConfiguration']

class QubicConfiguration(Configuration):
    """
    The QubicConfiguration class, which represents the instrument and
    pointing setups.

    """
    def __init__(self, instrument, pointing, block_id=None, selection=None):
        """
        Parameters
        ----------
        instrument : str or QubicInstrument
            Module name (only 'monochromatic' for now), or a QubicInstrument
            instance.
        pointing : array-like of shape (n,3) or sequence of
            The triplets (θ,φ,ψ), where (φ,θ,ψ) are the Euler angles
            of the intrinsic ZY'Z'' rotations. Note the ordering of the angles.
            θ : co-latitude
            φ : longitude
            ψ : minus the position angle
        block_id : string or sequence of, optional
           The pointing block identifier.
        selection : integer or sequence of, optional
           The indices of the pointing sequence to be selected to construct
           the pointing configuration.

        """
        if not isinstance(instrument, (QubicInstrument, str)):
            raise TypeError("Invalid type for the instrument ('{}' instead of "
                            "'QubicInstrument' or 'str').".format(type(
                            instrument).__name__))
        if isinstance(instrument, str):
            raise NotImplementedError('Module names not fixed yet.')
        Configuration.__init__(self, instrument, pointing, block_id=block_id,
                               selection=selection)

    def get_pointing_hitmap(self, nside=None):
        """
        Return a healpy map whose values are the number of times a pointing
        hits the pixel.

        """
        if nside is None:
            nside = self.instrument.sky.nside
        hit = np.zeros(12 * nside**2)
        theta, phi = self.pointing[..., 0], self.pointing[..., 1]
        ipixel = hp.ang2pix(nside, np.radians(theta), np.radians(phi))
        for i in ipixel:
            hit[i] += 1
        return hit

    def get_convolution_peak_operator(self, fwhm=np.radians(0.64883707),
                                      **keywords):
        """
        Return an operator that convolves the Healpix sky by the gaussian
        kernel that, if used in conjonction with the peak sampling operator,
        best approximates the synthetic beam.

        Parameters
        ----------
        fwhm : float, optional
            The Full Width Half Maximum of the gaussian.

        """
        return HealpixConvolutionGaussianOperator(self.instrument.sky.nside,
                                                  fwhm=fwhm, **keywords)

    def get_projection_peak_operator(self, kmax=2):
        """
        Return the projection operator for the peak sampling.

        Parameters
        ----------
        kmax : int, optional
            The diffraction order above which the peaks are ignored.
            For a value of 0, only the central peak is sampled.

        """
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
        return QubicConfiguration(instrument, ptg, selection=selection,
                                  block_id=ptg_id), info

    @classmethod
    def _load_observation(cls, filename, instrument=None, selection=None):
        """
        Load a QUBIC configuration, info and TOD.

        obs, tod, info = QubicConfiguration._load_observation(filename,
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
        tod, tod_id = cls._get_files_from_selection(filename, 'tod', selection)
        if len(tod) != len(obs.block):
            raise ValueError('The number of pointing and tod files is not the '
                             'same.')
        if any(p != t for p, t in zip(obs.block.identifier, tod_id)):
            raise ValueError('Incompatible pointing and tod files.')
        tod = np.hstack(tod)
        return obs, tod, info

    @classmethod
    def load_simulation(cls, filename, instrument=None, selection=None):
        """
        Load a simulation, including the QUBIC configuration, info, TOD and
        input map.

        obs, input_map, tod, info = QubicConfiguration.load_simulation(
            filename, [instrument=None, selection=None])

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
        obs, tod, info = cls._load_observation(filename, instrument=instrument,
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

    def _save_observation(self, filename, tod, info):
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
        self._save_observation(filename, tod, info)
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
            t = tod[:, b.start:b.stop]
            file_ptg = os.path.join(filename, 'ptg_' + postfix)
            file_tod = os.path.join(filename, 'tod_' + postfix)
            hdu_ptg = pyfits.PrimaryHDU(p)
            hdu_tod = pyfits.PrimaryHDU(t)
            pyfits.HDUList([hdu_ptg]).writeto(file_ptg)
            pyfits.HDUList([hdu_tod]).writeto(file_tod)

    @staticmethod
    def _get_instrument_from_file(filename):
        with open(os.path.join(filename, 'instrument.txt')) as f:
            lines = f.readlines()[1:]
        sep = ifirst(lines, '\n')
        keywords_instrument = yaml.load(''.join(lines[:sep]))
        name = keywords_instrument.pop('name')
        keywords_calibration = yaml.load(''.join(lines[sep+2:]))
        calibration = QubicCalibration(**keywords_calibration)
        return QubicInstrument(name, calibration, **keywords_instrument)

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
        return time.strftime('%Y-%m-%d_%H:%M:%S',
                             time.localtime(t)) + '{:.9f}'.format(t-int(t))[1:]
