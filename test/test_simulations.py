from __future__ import division
from pysimulators import FitsArray, BeamGaussian
from qubic import (
    create_random_pointings, QubicCalibration, QubicAcquisition,
    QubicInstrument, QubicSampling, QubicScene)
import numpy as np
import os
import qubic

NPTG = 5
ANGLE = 10
DATAPATH = os.path.join(os.path.dirname(__file__), 'data')
MAPIQU = qubic.io.read_map(qubic.data.PATH + 'syn256_pol.fits')
FILEPTG = os.path.join(DATAPATH, 'ptg_np{}_{}deg.fits'.format(NPTG, ANGLE))
FILETODI = os.path.join(DATAPATH,
                        'todI_np{}_{}deg_v1.fits'.format(NPTG, ANGLE))
FILETODIQU = os.path.join(DATAPATH,
                          'todIQU_np{}_{}deg_v1.fits'.format(NPTG, ANGLE))
INSTRUMENT = QubicInstrument()


def write_reference():
    np.random.seed(0)
    p = create_random_pointings([0, 90], NPTG, ANGLE)
    FitsArray([p.azimuth, p.elevation, p.pitch, p.angle_hwp]).save(FILEPTG)
    for kind, map, filename in zip(['I', 'IQU'], [MAPIQU[..., 0], MAPIQU],
                                   [FILETODI, FILETODIQU]):
        acq = QubicAcquisition(INSTRUMENT, p, nside=256, kind=kind)
        tod = acq.get_observation(map, noiseless=True)
        FitsArray(tod).save(filename)


def test():
    def func(kind, map, filename):
        acq = QubicAcquisition(INSTRUMENT, p, nside=256, kind=kind)
        tod = acq.get_observation(map, noiseless=True)
        ref = FitsArray(filename)
        np.testing.assert_almost_equal(tod, ref)

    a = FitsArray(FILEPTG)
    p = QubicSampling(azimuth=a[0], elevation=a[1], pitch=a[2], angle_hwp=a[3])
    for kind, map, filename in zip(['I', 'IQU'], [MAPIQU[..., 0], MAPIQU],
                                   [FILETODI, FILETODIQU]):
        yield func, kind, map, filename
