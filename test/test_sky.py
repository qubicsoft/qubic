import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from pyoperators import DiagonalOperator, UnpackOperator, pcg
from pyoperators.utils.testing import assert_same

from qubic.data import PATH as data_path
from qubic.dicts import PATH as dicts_path
from qubic.io._io import read_map
from qubic.lib.Instrument.Qacquisition import QubicAcquisition
from qubic.lib.Instrument.Qinstrument import QubicInstrument
from qubic.lib.Qdictionary import qubicDict
from qubic.lib.Qsamplings import get_pointing
from qubic.lib.Qscene import QubicScene


# Using a reduced number of samples for debugging
NDEBUG_POINTINGS = 1000

# Forcing the input sky map to float32 because the projection matrices are built in float32
sky_map = read_map(data_path + "syn256_pol.fits").astype(np.float32)
I, Q, U = sky_map.T

# Observation model
qubic_dict = qubicDict()
qubic_dict.read_from_file(dicts_path + "pipeline_demo.dict")
qubic_dict["random_pointing"] = False
qubic_dict["sweeping_pointing"] = True

tods = {}
pTxs = {}
pT1s = {}
pTx_pT1s = {}
cbiks = {}
outputs = {}

# Sky configurations
kinds = ["I", "QU", "IQU"]

# Building input maps with float32 type to avoid float64/float32 mismatches
input_maps = [
    I.astype(np.float32),                     # I-only map, shape (Npix,)
    np.array([Q, U], dtype=np.float32).T,     # QU map, shape (Npix, 2)
    np.array([I, Q, U], dtype=np.float32).T,  # IQU map, shape (Npix, 3)
]

for kind, input_map in zip(kinds, input_maps):

    # Setting the current sky configuration before building scene
    qubic_dict["kind"] = kind

    # With smaller number of pointing to debug
    pointings = get_pointing(qubic_dict)[:NDEBUG_POINTINGS]

    # Recreating instrument and scene after setting qubic_dict["kind"]
    instrument = QubicInstrument(qubic_dict)
    scene = QubicScene(qubic_dict)

    acq = QubicAcquisition(instrument, pointings, scene, qubic_dict)

    print("\n==============================")
    print("kind:", kind)
    print("scene.kind:", scene.kind)
    print("input_map shape:", input_map.shape)
    print("input_map dtype:", input_map.dtype)
    print("pointings used:", len(pointings))

    P = acq.get_projection_operator()
    W = acq.get_hwp_operator()
    H = W * P

    # Forcing TOD to float32 
    tod = H(input_map).astype(np.float32)
    tods[kind] = tod

    coverage = P.pT1()

    proj = H.T(tod).astype(np.float32)
    pTxs[kind] = proj[coverage > 0]

    if kind != "QU":
        pTx_pT1 = P.pTx_pT1(tod)
        pTx_pT1s[kind] = ( pTx_pT1[0][coverage > 0], pTx_pT1[1][coverage > 0])

    cbik = P.canonical_basis_in_kernel()
    mask = coverage > 10

    P = P.restrict(mask, inplace=True)

    unpack = UnpackOperator(mask, broadcast="rightward")

    x0 = unpack.T(input_map).astype(np.float32)

    M = DiagonalOperator((1 / coverage[mask]).astype(np.float32),broadcast="rightward")

    H = W * P

    # Forcing the right side of the PCG to float32
    pcg_right_side = H.T(tod).astype(np.float32)

    solution = pcg(H.T * H, pcg_right_side, M=M, disp=True, tol=1e-5)

    pT1s[kind] = coverage
    cbiks[kind] = cbik

    outputs[kind] = solution["x"].astype(np.float32)

# Function to adapt tolerance to float32
def assert_close_float32(actual, desired, rtol=1e-5, eps_factor=100):
    desired = desired.astype(np.float32)
    actual = actual.astype(np.float32)

    scale = np.nanmax(np.abs(desired))
    #atol = factor × precision float32 × amplitude 
    atol = eps_factor * np.finfo(np.float32).eps * scale

    assert_allclose(actual, desired, rtol=rtol, atol=atol)

def test_sky():
    assert_close_float32(tods["I"], tods["IQU"][..., 0])
    assert_close_float32(tods["QU"], tods["IQU"][..., 1:])

    assert_close_float32(pTxs["I"], pTxs["IQU"][..., 0], rtol=1e-3, eps_factor=1000)
    assert_close_float32(pTxs["QU"], pTxs["IQU"][..., 1:], rtol=1e-3, eps_factor=1000)

    assert_close_float32(pTx_pT1s["I"][0], pTx_pT1s["IQU"][0], rtol=1e-3, eps_factor=1000)
    assert_close_float32(pTx_pT1s["I"][1], pTx_pT1s["IQU"][1], rtol=1e-3, eps_factor=1000)


@pytest.mark.skip(reason="test skipped")
def test_sky2():
    assert_allclose(outputs["I"], outputs["IQU"][..., 0], atol=2e-2)
    assert_allclose(outputs["QU"], outputs["IQU"][..., 1:], atol=2e-2)

    for k in ("QU", "IQU"):
        assert_equal(cbiks[k], cbiks["I"])
        assert_same(pT1s[k], pT1s["I"].astype(np.float32), rtol=15)