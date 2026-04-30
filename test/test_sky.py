import numpy as np
from numpy.testing import assert_allclose, assert_equal
from pyoperators import DiagonalOperator, UnpackOperator, pcg
from pyoperators.utils.testing import assert_same, skiptest

from qubic.data import PATH as data_path
from qubic.dicts import PATH as dicts_path
from qubic.io._io import read_map
from qubic.lib.Instrument.Qacquisition import QubicAcquisition
from qubic.lib.Instrument.Qinstrument import QubicInstrument
from qubic.lib.Qdictionary import qubicDict
from qubic.lib.Qsamplings import get_pointing
from qubic.lib.Qscene import QubicScene

# read the input map
I, Q, U = read_map(data_path + "syn256_pol.fits").T

# observation model
np.random.seed(0)
qubic_dict = qubicDict()
qubic_dict.read_from_file(dicts_path + "pipeline_demo.dict")
qubic_dict["duration"] = 1
qubic_dict["random_pointing"] = False
qubic_dict["sweeping_pointing"] = True
pointings = get_pointing(qubic_dict)
print(pointings)
instrument = QubicInstrument(qubic_dict)
scene = QubicScene(qubic_dict)

tods = {}
pTxs = {}
pT1s = {}
pTx_pT1s = {}
cbiks = {}
outputs = {}
kinds = ["I", "QU", "IQU"]
input_maps = [I, np.array([Q, U]).T, np.array([I, Q, U]).T]
for kind, input_map in zip(kinds, input_maps):
    qubic_dict["kind"] = kind
    acq = QubicAcquisition(instrument, pointings, scene, qubic_dict)
    P = acq.get_projection_operator()
    W = acq.get_hwp_operator()
    H = W * P
    coverage = P.pT1()
    tod = H(input_map)
    tods[kind] = tod
    pTxs[kind] = H.T(tod)[coverage > 0]
    if kind != "QU":
        pTx_pT1 = P.pTx_pT1(tod)
        pTx_pT1s[kind] = pTx_pT1[0][coverage > 0], pTx_pT1[1][coverage > 0]
    cbik = P.canonical_basis_in_kernel()
    mask = coverage > 10
    P = P.restrict(mask, inplace=True)
    unpack = UnpackOperator(mask, broadcast="rightward")
    x0 = unpack.T(input_map)

    M = DiagonalOperator(1 / coverage[mask], broadcast="rightward")

    H = W * P
    solution = pcg(H.T * H, H.T(tod), M=M, disp=True, tol=1e-5)
    pT1s[kind] = coverage
    cbiks[kind] = cbik
    outputs[kind] = solution["x"]


def test_sky():
    assert_same(tods["I"], tods["IQU"][..., 0].astype(np.float32))
    assert_same(tods["QU"], tods["IQU"][..., 1:].astype(np.float32))

    assert_same(pTxs["I"], pTxs["IQU"][..., 0].astype(np.float32))
    assert_same(pTxs["QU"], pTxs["IQU"][..., 1:].astype(np.float32))

    assert_same(pTx_pT1s["I"][0], pTx_pT1s["IQU"][0].astype(np.float32))
    assert_same(pTx_pT1s["I"][1], pTx_pT1s["IQU"][1].astype(np.float32))


@skiptest
def test_sky2():
    assert_allclose(outputs["I"], outputs["IQU"][..., 0], atol=2e-2)
    assert_allclose(outputs["QU"], outputs["IQU"][..., 1:], atol=2e-2)

    for k in ("QU", "IQU"):
        assert_equal(cbiks[k], cbiks["I"])
        assert_same(pT1s[k], pT1s["I"].astype(np.float32), rtol=15)


test_sky()
