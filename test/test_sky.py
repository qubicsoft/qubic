from __future__ import division
from numpy.testing import assert_allclose, assert_equal
from pyoperators import pcg, DiagonalOperator, UnpackOperator
from pyoperators.utils.testing import assert_same, skiptest
from qubic import QubicAcquisition, create_sweeping_pointings
import numpy as np
import qubic

# read the input map
I, Q, U = qubic.io.read_map(qubic.data.PATH + 'syn256_pol.fits').T

# observation model
np.random.seed(0)
racenter = 0.0
deccenter = -57.0
angspeed = 1  # deg/sec
delta_az = 15.
angspeed_psi = 0.1
maxpsi = 45.
nsweeps_el = 300
duration = 1   # hours
ts = 20        # seconds
pointings = create_sweeping_pointings(
    [racenter, deccenter], duration, ts, angspeed, delta_az, nsweeps_el,
    angspeed_psi, maxpsi)

tods = {}
pTxs = {}
pT1s = {}
pTx_pT1s = {}
cbiks = {}
outputs = {}
kinds = ['I', 'QU', 'IQU']
input_maps = [I,
              np.array([Q, U]).T,
              np.array([I, Q, U]).T]
for kind, input_map in zip(kinds, input_maps):
    acq = QubicAcquisition(150, pointings, kind=kind)
    P = acq.get_projection_operator()
    W = acq.get_hwp_operator()
    H = W * P
    coverage = P.pT1()
    tod = H(input_map)
    tods[kind] = tod
    pTxs[kind] = H.T(tod)[coverage > 0]
    if kind != 'QU':
        pTx_pT1 = P.pTx_pT1(tod)
        pTx_pT1s[kind] = pTx_pT1[0][coverage > 0], pTx_pT1[1][coverage > 0]
    cbik = P.canonical_basis_in_kernel()
    mask = coverage > 10
    P = P.restrict(mask, inplace=True)
    unpack = UnpackOperator(mask, broadcast='rightward')
    x0 = unpack.T(input_map)

    M = DiagonalOperator(1 / coverage[mask], broadcast='rightward')

    H = W * P
    solution = pcg(H.T * H, H.T(tod), M=M, disp=True, tol=1e-5)
    pT1s[kind] = coverage
    cbiks[kind] = cbik
    outputs[kind] = solution['x']


def test_sky():
    assert_same(tods['I'], tods['IQU'][..., 0].astype(np.float32))
    assert_same(tods['QU'], tods['IQU'][..., 1:].astype(np.float32))

    assert_same(pTxs['I'], pTxs['IQU'][..., 0].astype(np.float32))
    assert_same(pTxs['QU'], pTxs['IQU'][..., 1:].astype(np.float32))

    assert_same(pTx_pT1s['I'][0], pTx_pT1s['IQU'][0].astype(np.float32))
    assert_same(pTx_pT1s['I'][1], pTx_pT1s['IQU'][1].astype(np.float32))


@skiptest
def test_sky2():
    assert_allclose(outputs['I'], outputs['IQU'][..., 0], atol=2e-2)
    assert_allclose(outputs['QU'], outputs['IQU'][..., 1:], atol=2e-2)

    for k in ('QU', 'IQU'):
        assert_equal(cbiks[k], cbiks['I'])
        assert_same(pT1s[k], pT1s['I'].astype(np.float32), rtol=15)
