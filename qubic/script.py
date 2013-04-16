from __future__ import division

import healpy as hp
import numpy as np
import qubic_v1
import time

from instrument import QubicInstrument, peak_pointing_matrix
from pyoperators import pcg, DiagonalOperator, UnpackOperator
from pysimulators import ProjectionInMemoryOperator

kmax = 2
nside_map = 128
simulpath = '/home/chanial/qubic_data/sim_1024_1000_dtheta10_random'

info, map_orig, ptg, tod = qubic_v1.read_simulation(simulpath, syb=False)
tod = tod.T

q = QubicInstrument(ndetector=info['ndetector'], nside=nside_map)

t0 = time.time()
m = peak_pointing_matrix(q, kmax, ptg)
H = ProjectionInMemoryOperator(m)
#hp.gnomview(coverage, rot=[0,90], reso=5)

coverage = H.T(np.ones_like(tod))
mask = coverage < 10
m.pack(mask)
H = ProjectionInMemoryOperator(m)

unpack = UnpackOperator(mask)
solution = pcg(H.T * H, H.T(tod)), disp=True, M=DiagonalOperator(1/coverage[~mask]))
map_new = unpack(solution['x'])

print time.time() - t0

