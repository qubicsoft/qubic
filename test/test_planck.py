from __future__ import division
from pyoperators.utils.testing import assert_same
from pysimulators.interfaces.healpy import SceneHealpixCMB
from qubic import (
    create_random_pointings, PlanckAcquisition, QubicAcquisition,
    QubicPlanckAcquisition)
import numpy as np
import qubic

SKY = qubic.io.read_map(qubic.data.PATH + 'syn256_pol.fits')


def test():
    np.random.seed(0)
    scene = SceneHealpixCMB(256, kind='IQU')
    acq = PlanckAcquisition(150, scene, true_sky=SKY)
    obs = acq.get_observation()
    invNtt = acq.get_invntt_operator()
    chi2_red = np.sum((obs - SKY) * invNtt(obs - SKY) / SKY.size)
    assert abs(chi2_red - 1) <= 0.001


def test_noiseless():
    sampling = create_random_pointings([0, 90], 100, 10)
    acq_qubic = QubicAcquisition(150, sampling)
    acq_planck = PlanckAcquisition(150, acq_qubic.scene, true_sky=SKY)
    acq_fusion = QubicPlanckAcquisition(acq_qubic, acq_planck)
    np.random.seed(0)
    y1 = acq_fusion.get_observation()
    np.random.seed(0)
    y2 = acq_fusion.get_observation(noiseless=True) + acq_fusion.get_noise()
    assert_same(y1, y2)
