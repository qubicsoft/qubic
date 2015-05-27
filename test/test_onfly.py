from __future__ import division
import numpy as np
from pyoperators.utils.testing import assert_same
from qubic import (
    QubicAcquisition, QubicInstrument, QubicScene, create_random_pointings)
from qubic.mapmaking import tod2map_all, tod2map_each


def test():
    instrument = QubicInstrument()[:10]
    sampling = create_random_pointings([0, 90], 30, 5)
    nside = 64
    scenes = QubicScene(nside, kind='I'), QubicScene(nside)

    def func(scene, max_nbytes, ref1, ref2, ref3, ref4, ref5, ref6):
        acq = QubicAcquisition(instrument, sampling, scene,
                               max_nbytes=max_nbytes)
        sky = np.ones(scene.shape)
        H = acq.get_operator()
        actual1 = H(sky)
        assert_same(actual1, ref1, atol=10)
        actual2 = H.T(actual1)
        assert_same(actual2, ref2, atol=10)
        actual2 = (H.T * H)(sky)
        assert_same(actual2, ref2, atol=10)
        actual3, actual4 = tod2map_all(acq, ref1, disp=False)
        assert_same(actual3, ref3, atol=10000)
        assert_same(actual4, ref4)
        actual5, actual6 = tod2map_each(acq, ref1, disp=False)
        assert_same(actual5, ref5, atol=1000)
        assert_same(actual6, ref6)

    for scene in scenes:
        acq = QubicAcquisition(instrument, sampling, scene)
        sky = np.ones(scene.shape)
        nbytes_per_sampling = acq.get_operator_nbytes() // len(acq.sampling)
        H = acq.get_operator()
        ref1 = H(sky)
        ref2 = H.T(ref1)
        ref3, ref4 = tod2map_all(acq, ref1, disp=False)
        ref5, ref6 = tod2map_each(acq, ref1, disp=False)
        for max_sampling in 10, 29, 30:
            max_nbytes = None if max_sampling is None \
                              else max_sampling * nbytes_per_sampling
            yield (func, scene, max_nbytes, ref1, ref2, ref3, ref4, ref5, ref6)
