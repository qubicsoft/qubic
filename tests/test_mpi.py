import numpy as np
import pytest

from pyoperators import MPI
from pyoperators.utils.testing import assert_same
from qubic.lib.Instrument.Qacquisition import QubicAcquisition, QubicInstrument, QubicScene, get_pointing
from qubic.lib.Qdictionary import qubicDict
#from qubic import QubicAcquisition, QubicInstrument, QubicScene, get_pointing
from qubic.mapmaking import tod2map_all, tod2map_each
#from qubic.qubicdict import qubicDict

pytestmark = pytest.mark.mpi

RANK = MPI.COMM_WORLD.rank
SIZE = MPI.COMM_WORLD.size


@pytest.mark.parametrize('kind', ['I', 'IQU'])
def test(kind):
    np.random.seed(0)
    config = qubicDict()
    config.read_from_file('pipeline_demo.dict')
    config['kind'] = kind
    config['npointings'] = 30
    config['synthbeam_dtype'] = float
    config['comm'] = MPI.COMM_SELF

    instrument = QubicInstrument(config)[:8]
    pointings = get_pointing(config)
    scene = QubicScene(config)
    sky = scene.ones()

    serial_acq = QubicAcquisition(instrument, pointings, scene, config)
    assert serial_acq.comm.size == 1
    serial_H = serial_acq.get_operator()
    serial_invntt = serial_acq.get_invntt_operator()
    ref_tod = serial_H(sky)
    ref_backproj = serial_H.T(serial_invntt(ref_tod))
    ref1, ref2 = tod2map_all(serial_acq, ref_tod, disp=False, maxiter=2)
    ref3, ref4 = tod2map_each(serial_acq, ref_tod, disp=False, maxiter=2)

    config = qubicDict()
    config.read_from_file('pipeline_demo.dict')
    config['kind'] = kind
    config['synthbeam_dtype'] = float
    config['comm'] = MPI.COMM_WORLD
    config['nprocs_instrument'] = nprocs_instrument = SIZE
    parallel_acq = QubicAcquisition(instrument, pointings, scene, config)
    assert parallel_acq.comm.size == SIZE
    assert parallel_acq.instrument.comm.size == nprocs_instrument
    assert parallel_acq.sampling.comm.size == SIZE / nprocs_instrument
    parallel_H = parallel_acq.get_operator()
    parallel_invntt = parallel_acq.get_invntt_operator()
    parallel_tod = parallel_H(sky)
    actual_tod = np.vstack(MPI.COMM_WORLD.allgather(parallel_tod))
    assert_same(actual_tod, ref_tod, atol=20)

    actual_backproj = parallel_H.T(parallel_invntt(parallel_tod))
    assert_same(actual_backproj, ref_backproj, atol=20)

    actual1, actual2 = tod2map_all(parallel_acq, parallel_tod, disp=False, maxiter=2)
    assert_same(actual1, ref1, atol=100)
    assert_same(actual2, ref2, atol=100)

    actual3, actual4 = tod2map_each(parallel_acq, parallel_tod, disp=False, maxiter=2)
    assert_same(actual3, ref3, atol=100)
    assert_same(actual4, ref4, atol=100)

