# coding: utf-8
from __future__ import division

import astropy.io.fits as pyfits
import healpy as hp
import numpy as np
import os
import time
import yaml
from astropy.time import TimeDelta
from glob import glob
from pyoperators import (
    DenseBlockDiagonalOperator, HomothetyOperator, IdentityOperator,
    ReshapeOperator, Rotation2dOperator, Rotation3dOperator)
from pyoperators.utils import ifirst
from pysimulators import (
    Acquisition, CartesianEquatorial2HorizontalOperator,
    CartesianGalactic2EquatorialOperator,
    CartesianHorizontal2EquatorialOperator,
    CartesianEquatorial2GalacticOperator,
    ConvolutionTruncatedExponentialOperator)
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
from .calibration import QubicCalibration
from .instrument import QubicInstrument

__all__ = ['QubicAcquisition']


class QubicAcquisition(Acquisition):
    """
    The QubicAcquisition class, which represents the instrument and
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
        Acquisition.__init__(self, instrument, pointing, block_id=block_id,
                             selection=selection)
        # XXX HACK
        from .pointings import QubicPointing
        if isinstance(pointing, QubicPointing):
            self.pointing = pointing

    def get_hitmap(self, nside=None):
        """
        Return a healpy map whose values are the number of times a pointing
        hits the pixel.

        """
        if nside is None:
            nside = self.instrument.sky.nside
        ipixel = self.pointing.tohealpix(nside)
        npixel = 12 * nside**2
        return np.histogram(ipixel, bins=npixel, range=(0, npixel))[0]

    def get_rotation(self):
        """
        Return the instrument-to-galactic rotation matrix.

        """
        p = self.pointing
        time = p.date_obs + TimeDelta(p.time, format='sec')
        r = CartesianEquatorial2GalacticOperator() * \
            CartesianHorizontal2EquatorialOperator(
                'NE', time, p.latitude, p.longitude) * \
            Rotation3dOperator("ZY'Z''", p.azimuth, 90 - p.elevation, p.pitch,
                               degrees=True)
        return r

    def get_rotation_g2i(self):
        """
        Return the galactic-to-instrument rotation matrix.

        """
        p = self.pointing
        time = p.date_obs + TimeDelta(p.time, format='sec')
        r = Rotation3dOperator("ZY'Z''", p.azimuth, 90 - p.elevation, p.pitch,
                               degrees=True).T * \
            CartesianEquatorial2HorizontalOperator(
                'NE', time, p.latitude, p.longitude) * \
            CartesianGalactic2EquatorialOperator()
        return r

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
        return HealpixConvolutionGaussianOperator(fwhm=fwhm, **keywords)

    def get_detector_response_operator(self, tau=None):
        """
        Return the operator for the bolometer responses.

        """
        if tau is None:
            tau = self.instrument.detector.packed.tau
        sampling_period = self.pointing.get_sampling_period()
        shapein = (self.get_ndetectors(), self.pointing.size)
        return ConvolutionTruncatedExponentialOperator(tau / sampling_period,
                                                       shapein=shapein)

    def get_hwp_operator(self):
        """
        Return the rotation matrix for the half-wave plate.

        """
        if self.instrument.sky.kind == 'I':
            return IdentityOperator()
        if self.instrument.sky.kind == 'QU':
            return Rotation2dOperator(-4 * self.pointing.angle_hwp,
                                      degrees=True)
        return Rotation3dOperator('X', -4 * self.pointing.angle_hwp,
                                  degrees=True)

    def get_invntt_operator(self):
        """
        Return the inverse time-time noise correlation matrix as an Operator.

        """

        l = self.instrument.detector.packed
        fs = 1 / self.pointing.get_sampling_period()
        return Acquisition.get_invntt_operator(
            self, sigma=l.sigma, fknee=l.fknee, fslope=l.fslope,
            sampling_frequency=fs, ncorr=self.instrument.detector.ncorr)

    def get_polarizer_operator(self):
        """
        Return operator for the polarizer grid.

        """
        if self.instrument.sky.kind == 'I':
            return HomothetyOperator(1 / self.instrument.detector.ngrids)

        if self.instrument.detector.ngrids == 1:
            raise ValueError(
                'Polarized input not handled by a single detector grid.')

        nd = len(self.instrument)
        nt = len(self.pointing)
        grid = self.instrument.detector.packed.quadrant // 4
        z = np.zeros(self.get_ndetectors())
        if self.instrument.sky.kind == 'QU':
            data = np.array([0.5 - grid, z]).T[:, None, None, :]
        else:
            data = np.array([z + 0.5, 0.5 - grid, z]).T[:, None, None, :]
        return ReshapeOperator((nd, nt, 1), (nd, nt)) * \
            DenseBlockDiagonalOperator(data)

    def get_projection_peak_operator(self, rotation=None, dtype=None,
                                     synthbeam_fraction=None):
        """
        Return the projection operator for the peak sampling.

        Parameters
        ----------
        rotation : array (npointings, 3, 3)
            The Instrument-to-Reference rotation matrix.
        dtype : dtype
            The datatype of the elements in the projection matrix.
        synthbeam_fraction : float, optional
            Override the instrument synthetic beam fraction.

        """
        if rotation is None:
            rotation = self.get_rotation_g2i()
        return self.instrument.get_projection_peak_operator(
            rotation, synthbeam_fraction=synthbeam_fraction, dtype=dtype)

    @classmethod
    def load(cls, filename, instrument=None, selection=None):
        """
        Load a QUBIC acquisition, and info.

        obs, info = QubicAcquisition.load(filename, [instrument=None,
                                          selection=None])

        Parameters
        ----------
        filename : string
           The QUBIC acquisition file name.
        instrument : QubicInstrument, optional
           The Qubic instrumental setup.
        selection : integer or sequence of
           The indices of the pointing blocks to be selected to construct
           the pointing configuration.

        Returns
        -------
        obs : QubicAcquisition
            The QUBIC acquisition instance as read from the file.
        info : string
            The info file stored alongside the acquisition.

        """
        if not isinstance(filename, str):
            raise TypeError("The input filename is not a string.")
        if instrument is None:
            instrument = cls._get_instrument_from_file(filename)
        with open(os.path.join(filename, 'info.txt')) as f:
            info = f.read()
        ptg, ptg_id = cls._get_files_from_selection(filename, 'ptg', selection)
        return QubicAcquisition(instrument, ptg, selection=selection,
                                block_id=ptg_id), info

    @classmethod
    def _load_observation(cls, filename, instrument=None, selection=None):
        """
        Load a QUBIC acquisition, info and TOD.

        obs, tod, info = QubicAcquisition._load_observation(filename,
                             [instrument=None, selection=None])

        Parameters
        ----------
        filename : string
           The QUBIC acquisition file name.
        instrument : QubicInstrument, optional
           The Qubic instrumental setup.
        selection : integer or sequence of
           The indices of the pointing blocks to be selected to construct
           the pointing configuration.

        Returns
        -------
        obs : QubicAcquisition
           The QUBIC acquisition instance as read from the file.
        tod : ndarray
           The stored time-ordered data.
        info : string
           The info file stored alongside the acquisition.

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
        Load a simulation, including the QUBIC acquisition, info, TOD and
        input map.

        obs, input_map, tod, info = QubicAcquisition.load_simulation(
            filename, [instrument=None, selection=None])

        Parameters
        ----------
        filename : string
           The QUBIC acquisition file name.
        instrument : QubicInstrument, optional
           The Qubic instrumental setup.
        selection : integer or sequence of
           The indices of the pointing blocks to be selected to construct
           the pointing configuration.

        Returns
        -------
        obs : QubicAcquisition
           The QUBIC acquisition instance as read from the file.
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
        Write a Qubic acquisition to disk.

        Parameters
        ----------
        filename : string
            The output path of the directory in which the acquisition will be
            saved.
        info : string
            All information deemed necessary to describe the acquisition.

        """
        self._save_acquisition(filename, info)
        self._save_ptg(filename)

    def _save_observation(self, filename, tod, info):
        """
        Write a QUBIC acquisition to disk with a TOD.

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
        self._save_acquisition(filename, info)
        self._save_ptg_tod(filename, tod)

    def save_simulation(self, filename, input_map, tod, info):
        """
        Write a QUBIC acquisition to disk with a TOD and an input image.

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

    def _save_acquisition(self, filename, info):
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
