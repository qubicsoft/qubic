# coding: utf-8
from __future__ import division

import astropy.io.fits as pyfits
import healpy as hp
import numpy as np
import operator
import os
import time
import yaml
from glob import glob
from pyoperators import (
    BlockColumnOperator, BlockDiagonalOperator, BlockRowOperator,
    CompositionOperator, I, MPIDistributionIdentityOperator, MPI, proxy_group,
    rule_manager)
from pyoperators.utils import ifirst
from pyoperators.utils.mpi import as_mpi
from pysimulators import Acquisition, ProjectionOperator
from .calibration import QubicCalibration
from .instrument import QubicInstrument, SimpleInstrument
from .scene import QubicScene

__all__ = ['QubicAcquisition',
           'SimpleAcquisition']


class SimpleAcquisition(Acquisition):
    """
    The Acquisition class for the single-peak simple instrument.

    """
    def __init__(self, instrument, sampling, scene=None, block=None,
                 calibration=None, detector_fknee=0, detector_fslope=1,
                 detector_ncorr=10, detector_ngrids=2, detector_nep=6e-17,
                 detector_tau=0.01, filter_relative_bandwidth=0.25,
                 polarizer=True, synthbeam_dtype=np.float32, absolute=False,
                 kind='IQU', nside=256, temperature=True, max_nbytes=None,
                 nprocs_instrument=None, nprocs_sampling=None, comm=None):
        """
        acq = SimpleAcquisition(band, sampling,
                                [scene=|absolute=, kind=, nside=, temperature=],
                                nprocs_instrument=, nprocs_sampling=, comm=)
        acq = SimpleAcquisition(instrument, sampling,
                                [scene=|absolute=, kind=, nside=, temperature=],
                                nprocs_instrument=, nprocs_sampling=, comm=)

        Parameters
        ----------
        band : int
            The module nominal frequency, in GHz.
        scene : QubicScene, optional
            The discretized observed scene (the sky).
        block : tuple of slices, optional
            Partition of the samplings.
        absolute : boolean, optional
            If true, the scene pixel values include the CMB background and the
            fluctuations in units of Kelvin, otherwise it only represents the
            fluctuations, in microKelvin.
        kind : 'I', 'QU' or 'IQU', optional
            The sky kind: 'I' for intensity-only, 'QU' for Q and U maps,
            and 'IQU' for intensity plus QU maps.
        nside : int, optional
            The Healpix scene's nside.
        instrument : SimpleInstrument, optional
            The SimpleInstrument instance.
        calibration : QubicCalibration, optional
            The calibration tree.
        detector_fknee : array-like, optional
            The detector 1/f knee frequency in Hertz.
        detector_fslope : array-like, optional
            The detector 1/f slope index.
        detector_ncorr : int, optional
            The detector 1/f correlation length.
        detector_nep : array-like, optional
            The detector NEP [W/sqrt(Hz)].
        detector_ngrids : 1 or 2, optional
            Number of detector grids.
        detector_tau : array-like, optional
            The detector time constants in seconds.
        filter_relative_bandwidth : float, optional
            The filter relative bandwidth Δν/ν.
        polarizer : boolean, optional
            If true, the polarizer grid is present in the optics setup.
        synthbeam_dtype : dtype, optional
            The data type for the synthetic beams (default: float32).
            It is the dtype used to store the values of the pointing matrix.
        max_nbytes : int or None, optional
            Maximum number of bytes to be allocated for the acquisition's
            operator.
        nprocs_instrument : int, optional
            For a given sampling slice, number of procs dedicated to
            the instrument.
        nprocs_sampling : int, optional
            For a given detector slice, number of procs dedicated to
            the sampling.
        comm : mpi4py.MPI.Comm, optional
            The acquisition's MPI communicator. Note that it is transformed
            into a 2d cartesian communicator before being stored as the 'comm'
            attribute. The following relationship must hold:
                comm.size = nprocs_instrument * nprocs_sampling

        """
        if not isinstance(instrument, SimpleInstrument):
            filter_nu = band * 1e9
            instrument = SimpleInstrument(
                calibration=calibration, detector_fknee=detector_fknee,
                detector_fslope=detector_fslope, detector_ncorr=detector_ncorr,
                detector_ngrids=detector_ngrids, detector_nep=detector_nep,
                detector_tau=detector_tau, filter_nu=filter_nu,
                filter_relative_bandwidth=filter_relative_bandwidth,
                polarizer=polarizer, synthbeam_dtype=synthbeam_dtype)
        if scene is None:
            scene = QubicScene(absolute=absolute, kind=kind, nside=nside,
                               temperature=temperature)
        Acquisition.__init__(
            self, instrument, sampling, scene, block,
            max_nbytes=max_nbytes, nprocs_instrument=nprocs_instrument,
            nprocs_sampling=nprocs_sampling, comm=comm)

    def get_coverage(self, out=None):
        """
        Return the acquisition scene coverage given by P.T(1).

        """
        P = self.get_projection_operator()
        if isinstance(P, ProjectionOperator):
            out = P.pT1(out=out)
        else:
            out = P.operands[0].pT1(out=out)
            for op in P.operands[1:]:
                op.pT1(out, operation=operator.iadd)
        self.get_distribution_operator().T(out, out=out)
        return out

    def get_hitmap(self, nside=None):
        """
        Return a healpy map whose values are the number of times a pointing
        hits the pixel.

        """
        if nside is None:
            nside = self.scene.nside
        ipixel = self.sampling.healpix(nside)
        npixel = 12 * nside**2
        hit = np.histogram(ipixel, bins=npixel, range=(0, npixel))[0]
        self.sampling.comm.Allreduce(MPI.IN_PLACE, as_mpi(hit), op=MPI.SUM)
        return hit

    def get_aperture_integration_operator(self):
        """
        Integrate flux density in the telescope aperture.
        Convert signal from W / m^2 / Hz into W / Hz.

        """
        return self.instrument.get_aperture_integration_operator()

    def get_convolution_peak_operator(self, fwhm=None, **keywords):
        """
        Return an operator that convolves the Healpix sky by the gaussian
        kernel that, if used in conjonction with the peak sampling operator,
        best approximates the synthetic beam.

        Parameters
        ----------
        fwhm : float, optional
            The Full Width Half Maximum of the gaussian, in radians.

        """
        return self.instrument.get_convolution_peak_operator(fwhm=fwhm,
                                                             **keywords)

    def get_detector_integration_operator(self):
        """
        Integrate flux density in detector solid angles.
        Convert W / sr into W.

        """
        return self.instrument.get_detector_integration_operator()

    def get_detector_response_operator(self):
        """
        Return the operator for the bolometer responses.

        """
        return BlockDiagonalOperator(
            [self.instrument.get_detector_response_operator(self.sampling[b])
             for b in self.block], axisin=1)

    def get_distribution_operator(self):
        """
        Return the MPI distribution operator.

        """
        return MPIDistributionIdentityOperator(self.comm)

    def get_filter_operator(self):
        """
        Return the filter operator.
        Convert units from W/Hz to W.

        """
        return self.instrument.get_filter_operator()

    def get_hwp_operator(self):
        """
        Return the operator for the bolometer responses.

        """
        return BlockDiagonalOperator(
            [self.instrument.get_hwp_operator(self.sampling[b], self.scene)
             for b in self.block], axisin=1)

    def get_unit_conversion_operator(self):
        """
        Convert sky temperature into W / m^2 / Hz.

        If the scene has been initialised with the 'absolute' keyword, the
        scene is assumed to include the CMB background and the fluctuations
        (in Kelvin) and the operator follows the non-linear Planck law.
        Otherwise, the scene only includes the fluctuations (in microKelvin)
        and the operator is linear (i.e. the output also corresponds to power
        fluctuations).

        """
        nu = self.instrument.filter.nu
        return self.scene.get_unit_conversion_operator(nu)

    def get_operator(self):
        """
        Return the operator of the acquisition. Note that the operator is only
        linear if the scene temperature is differential.

        """
        distribution = self.get_distribution_operator()
        temp = self.get_unit_conversion_operator()
        aperture = self.get_aperture_integration_operator()
        filter = self.get_filter_operator()
        projection = self.get_projection_operator()
        hwp = self.get_hwp_operator()
        polarizer = self.get_polarizer_operator()
        integ = self.get_detector_integration_operator()
        response = self.get_detector_response_operator()

        with rule_manager(inplace=True):
            H = CompositionOperator([
                response, integ, polarizer, hwp * projection, filter, aperture,
                temp, distribution])
        if self.scene == 'QU':
            H = self.get_subtract_grid_operator()(H)
        return H

    def get_operator_nbytes(self):
        # return self.get_invntt_nbytes() + self.get_projection_nbytes()
        return self.get_projection_nbytes()

    def get_polarizer_operator(self):
        """
        Return operator for the polarizer grid.

        """
        return BlockDiagonalOperator(
            [self.instrument.get_polarizer_operator(
                self.sampling[b], self.scene) for b in self.block], axisin=1)

    def get_projection_operator(self, verbose=True):
        """
        Return the projection operator for the peak sampling.
        Convert units from W to W/sr.

        Parameters
        ----------
        verbose : bool, optional
            If true, display information about the memory allocation.

        """
        f = self.instrument.get_projection_operator
        if len(self.block) == 1:
            return BlockColumnOperator(
                [f(self.sampling[b], self.scene, verbose=verbose)
                 for b in self.block], axisout=1)
        #XXX HACK
        def callback(i):
            p = f(self.sampling[self.block[i]], self.scene, verbose=False)
            return p
        shapeouts = [(len(self.instrument), s.stop-s.start) +
                      self.scene.shape[1:] for s in self.block]
        proxies = proxy_group(len(self.block), callback, shapeouts=shapeouts)
        return BlockColumnOperator(proxies, axisout=1)

    def get_add_grids_operator(self):
        """ Return operator to add signal from detector pairs. """
        nd = len(self.instrument)
        if nd % 2 != 0:
            raise ValueError('Odd number of detectors.')
        partitionin = 2 * (len(self.instrument) // 2,)
        return BlockRowOperator([I, I], axisin=0, partitionin=partitionin)

    def get_subtract_grids_operator(self):
        """ Return operator to subtract signal from detector pairs. """
        nd = len(self.instrument)
        if nd % 2 != 0:
            raise ValueError('Odd number of detectors.')
        partitionin = 2 * (len(self.instrument) // 2,)
        return BlockRowOperator([I, -I], axisin=0, partitionin=partitionin)

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


class QubicAcquisition(SimpleAcquisition):
    """
    The QubicAcquisition class, which combines the instrument, sampling and
    scene models.

    """
    def __init__(self, instrument, sampling, scene=None, block=None,
                 calibration=None, detector_nep=4.7e-17, detector_fknee=0,
                 detector_fslope=1, detector_ncorr=10, detector_ngrids=2,
                 detector_tau=0.01, filter_relative_bandwidth=0.25,
                 polarizer=True,  primary_beam=None, secondary_beam=None,
                 synthbeam_dtype=np.float32, synthbeam_fraction=0.99,
                 absolute=False, kind='IQU', nside=256, temperature=True,
                 max_nbytes=None, nprocs_instrument=None, nprocs_sampling=None,
                 comm=None):
        """
        acq = QubicAcquisition(band, sampling,
                               [scene=|absolute=, kind=, nside=, temperature=],
                               nprocs_instrument=, nprocs_sampling=, comm=)
        acq = QubicAcquisition(instrument, sampling,
                               [scene=|absolute=, kind=, nside=, temperature=],
                               nprocs_instrument=, nprocs_sampling=, comm=)

        Parameters
        ----------
        band : int
            The module nominal frequency, in GHz.
        scene : QubicScene, optional
            The discretized observed scene (the sky).
        block : tuple of slices, optional
            Partition of the samplings.
        absolute : boolean, optional
            If true, the scene pixel values include the CMB background and the
            fluctuations in units of Kelvin, otherwise it only represents the
            fluctuations, in microKelvin.
        kind : 'I', 'QU' or 'IQU', optional
            The sky kind: 'I' for intensity-only, 'QU' for Q and U maps,
            and 'IQU' for intensity plus QU maps.
        nside : int, optional
            The Healpix scene's nside.
        instrument : QubicInstrument, optional
            The QubicInstrument instance.
        calibration : QubicCalibration, optional
            The calibration tree.
        detector_fknee : array-like, optional
            The detector 1/f knee frequency in Hertz.
        detector_fslope : array-like, optional
            The detector 1/f slope index.
        detector_ncorr : int, optional
            The detector 1/f correlation length.
        detector_nep : array-like, optional
            The detector NEP [W/sqrt(Hz)].
        detector_ngrids : 1 or 2, optional
            Number of detector grids.
        detector_tau : array-like, optional
            The detector time constants in seconds.
        filter_relative_bandwidth : float, optional
            The filter relative bandwidth Δν/ν.
        polarizer : boolean, optional
            If true, the polarizer grid is present in the optics setup.
        primary_beam : function f(theta [rad], phi [rad]), optional
            The primary beam transmission function.
        secondary_beam : function f(theta [rad], phi [rad]), optional
            The secondary beam transmission function.
        synthbeam_dtype : dtype, optional
            The data type for the synthetic beams (default: float32).
            It is the dtype used to store the values of the pointing matrix.
        synthbeam_fraction: float, optional
            The fraction of significant peaks retained for the computation
            of the synthetic beam.
        max_nbytes : int or None, optional
            Maximum number of bytes to be allocated for the acquisition's
            operator.
        nprocs_instrument : int, optional
            For a given sampling slice, number of procs dedicated to
            the instrument.
        nprocs_sampling : int, optional
            For a given detector slice, number of procs dedicated to
            the sampling.
        comm : mpi4py.MPI.Comm, optional
            The acquisition's MPI communicator. Note that it is transformed
            into a 2d cartesian communicator before being stored as the 'comm'
            attribute. The following relationship must hold:
                comm.size = nprocs_instrument * nprocs_sampling

        """
        if not isinstance(instrument, QubicInstrument):
            filter_nu = instrument * 1e9
            instrument = QubicInstrument(
                calibration=calibration, detector_fknee=detector_fknee,
                detector_fslope=detector_fslope, detector_ncorr=detector_ncorr,
                detector_nep=detector_nep, detector_ngrids=detector_ngrids,
                detector_tau=detector_tau, filter_nu=filter_nu,
                filter_relative_bandwidth=filter_relative_bandwidth,
                polarizer=polarizer, primary_beam=primary_beam,
                secondary_beam=secondary_beam, synthbeam_dtype=synthbeam_dtype,
                synthbeam_fraction=synthbeam_fraction)
        if scene is None:
            scene = QubicScene(absolute=absolute, kind=kind, nside=nside,
                               temperature=temperature)
        SimpleAcquisition.__init__(
            self, instrument, sampling, scene, block=block,
            max_nbytes=max_nbytes, nprocs_instrument=nprocs_instrument,
            nprocs_sampling=nprocs_sampling, comm=comm)
