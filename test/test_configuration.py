from __future__ import division

import healpy as hp
import numpy as np
import os
import shutil
from qubic import QubicCalibration, QubicConfiguration, QubicInstrument
from numpy.testing import assert_equal
from uuid import uuid1

inpath = os.path.join(os.path.dirname(__file__), 'data')
input_map = hp.read_map(os.path.join(inpath, 'syn256.fits'))
outpath = 'test-' + str(uuid1())[:8]
os.mkdir(outpath)

ptg = [1., 0, 180]
ptg2 = [1., 1, 180]
pta = np.asarray(ptg)
pta2 = np.asarray(ptg2)
ptgs = ptg, [ptg], [ptg,ptg], [[ptg,ptg], [ptg2,ptg2,ptg2]], \
       pta, [pta], [pta,pta], [[pta,pta], [pta2,pta2,pta2]], \
       np.asarray([pta]), np.asarray([pta,pta]), [np.asarray([pta,pta])], \
       [np.asarray([pta,pta]), np.asarray([pta2,pta2,pta2])]
block_n = [[1], [1], [2], [2,3],
           [1], [1], [2], [2,3],
           [1], [2], [2], [2,3], [2,3]]
caltree = QubicCalibration(fwhm_deg=15, focal_length=0.2,
                           detarray='CalQubic_DetArray_v1.fits')
qubic = QubicInstrument('monochromatic,nopol', caltree, nu=160e9)
qubic.detector.removed = True
qubic.detector.removed[30:,30:] = False

def test_qubicconfiguration_pointing():
    def func(p, n):
        obs = QubicConfiguration(qubic, p)
        assert_equal(obs.block.n, n)
        assert_equal(len(obs.pointing), sum(n))
    for p, n in zip(ptgs, block_n):
        yield func, p, n

def test_qubicconfiguration_load_save():
    info = 'test\nconfig'
    def func_configuration(obs, info):
        filename_ = os.path.join(outpath, 'config-' + str(uuid1()))
        obs.save(filename_, info)
        obs2, info2 = QubicConfiguration.load(filename_)
        assert_equal(str(obs), str(obs2))
        assert_equal(info, info2)
    def func_observation(obs, tod, info):
        filename_ = os.path.join(outpath, 'obs-' + str(uuid1()))
        obs._save_observation(filename_, tod, info)
        obs2, tod2, info2 = QubicConfiguration._load_observation(filename_)
        assert_equal(str(obs), str(obs2))
        assert_equal(tod, tod2)
        assert_equal(info, info2)
    def func_simulation(obs, input_map, tod, info):
        filename_ = os.path.join(outpath, 'simul-' + str(uuid1()))
        obs.save_simulation(filename_, input_map, tod, info)
        obs2, input_map2, tod2, info2 = QubicConfiguration.load_simulation(
            filename_)
        assert_equal(str(obs), str(obs2))
        assert_equal(input_map, input_map2)
        assert_equal(tod, tod2)
        assert_equal(info, info2)
        
    for p in ptgs:
        obs = QubicConfiguration(qubic, p)
        P = obs.get_projection_peak_operator(kmax=2)
        tod = P(input_map)
        yield func_configuration, obs, info
        yield func_observation, obs, tod, info
        yield func_simulation, obs, input_map, tod, info

def teardown():
    shutil.rmtree(outpath)
