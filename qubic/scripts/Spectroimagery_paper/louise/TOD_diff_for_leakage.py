from __future__ import division
import sys
import os
import time
import pysm
import qubic
import glob

import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
import SpectroImLib as si

from pysimulators import FitsArray
from pysm.nominal import models

mp.rc('text', usetex = False)

### Instrument ###
d = qubic.qubicdict.qubicDict()
dp = qubic.qubicdict.qubicDict()
d.read_from_file("test_spectroim.dict")
d['MultiBand'] = True # spectro imager
d['nf_sub'] = 16
dp.read_from_file("test_spectroim.dict")
dp['MultiBand'] = False
dp['nf_sub'] = 1


### Sky ###
sky_config = {'dust': models('d1', d['nside']),
    'synchrotron': models('s1', d['nside']),
    'freefree': models('f1', d['nside']), #not polarized
    'cmb': models('c1', d['nside']),
    'ame': models('a1', d['nside'])} #not polarized

planck_sky = si.Planck_sky(sky_config, d)
x0_planck = planck_sky.get_sky_map()
#x0_planck[1:,:,:] = x0_planck[0,:,:]

qubic_sky = si.Qubic_sky(sky_config, d)
x0_qubic = qubic_sky.get_sky_map()
