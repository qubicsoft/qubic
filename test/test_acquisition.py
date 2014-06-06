from __future__ import division

import numpy as np
import os
import qubic
import shutil
from numpy.testing import assert_equal
from pyoperators.utils.testing import assert_same, skiptest
from qubic import (
    QubicCalibration, QubicAcquisition, QubicInstrument, QubicSampling,
    create_random_pointings, gal2equ, map2tod)
from qubic.io import read_map
from uuid import uuid1

outpath = ''


def setup():
    global outpath
    outpath = 'test-' + str(uuid1())[:8]
    os.mkdir(outpath)


def teardown():
    shutil.rmtree(outpath)

DATAPATH = os.path.join(os.path.dirname(qubic.__file__), 'data')
input_map = read_map(os.path.join(DATAPATH, 'syn256_pol.fits'))
np.random.seed(0)
center = gal2equ(0, 90)
sampling = create_random_pointings(center, 100, 10)

ptg = [1., 0, 180]
ptg2 = [1., 1, 180]
pta = np.asarray(ptg)
pta2 = np.asarray(ptg2)
ptgs = ptg, [ptg], [ptg, ptg], [[ptg, ptg], [ptg2, ptg2, ptg2]], \
       pta, [pta], [pta, pta], [[pta, pta], [pta2, pta2, pta2]], \
       np.asarray([pta]), np.asarray([pta, pta]), [np.asarray([pta, pta])], \
       [np.asarray([pta, pta]), np.asarray([pta2, pta2, pta2])]
block_n = [[1], [1], [2], [2, 3],
           [1], [1], [2], [2, 3],
           [1], [2], [2], [2, 3], [2, 3]]
caltree = QubicCalibration(detarray='CalQubic_DetArray_v1.fits')
removed = np.ones((32, 32), dtype=bool)
removed[30:, 30:] = False
qubic = QubicInstrument(calibration=caltree, nu=160e9,
                        removed=removed, synthbeam_fraction=0.99)


@skiptest
def test_pointing():
    def func(p, n):
        p = np.asarray(p)
        smp = QubicSampling(azimuth=p[..., 0], elevation=p[..., 1],
                            pitch=p[..., 2])
        acq = QubicAcquisition(qubic, smp)
        assert_equal(acq.block.n, n)
        assert_equal(len(acq.pointing), sum(n))
    for p, n in zip(ptgs, block_n):
        yield func, p, n


@skiptest
def test_load_save():
    info = 'test\nconfig'

    def func_acquisition(obs, info):
        filename_ = os.path.join(outpath, 'config-' + str(uuid1()))
        obs.save(filename_, info)
        obs2, info2 = QubicAcquisition.load(filename_)
        assert_equal(str(obs), str(obs2))
        assert_equal(info, info2)

    def func_observation(obs, tod, info):
        filename_ = os.path.join(outpath, 'obs-' + str(uuid1()))
        obs._save_observation(filename_, tod, info)
        obs2, tod2, info2 = QubicAcquisition._load_observation(filename_)
        assert_equal(str(obs), str(obs2))
        assert_equal(tod, tod2)
        assert_equal(info, info2)

    def func_simulation(obs, input_map, tod, info):
        filename_ = os.path.join(outpath, 'simul-' + str(uuid1()))
        obs.save_simulation(filename_, input_map, tod, info)
        obs2, input_map2, tod2, info2 = QubicAcquisition.load_simulation(
            filename_)
        assert_equal(str(obs), str(obs2))
        assert_equal(input_map, input_map2)
        assert_equal(tod, tod2)
        assert_equal(info, info2)

    for p in ptgs:
        p = np.asarray(p)
        smp = QubicSampling(azimuth=p[..., 0], elevation=p[..., 1],
                            pitch=p[..., 2])
        acq = QubicAcquisition(qubic, smp)
        P = acq.get_projection_peak_operator()
        tod = P(input_map)
        yield func_acquisition, acq, info
        yield func_observation, acq, tod, info
        yield func_simulation, acq, input_map, tod, info


def test_add_subtract_grid_operator():
    acq = QubicAcquisition(150, sampling)
    tod = map2tod(acq, input_map, convolution=False)
    add = acq.get_add_grids_operator()
    sub = acq.get_subtract_grids_operator()
    assert_same(add(tod), tod[:992] + tod[992:])
    assert_same(sub(tod), tod[:992] - tod[992:])
