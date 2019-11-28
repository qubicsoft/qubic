from __future__ import division
import os

from pysm.nominal import models

import qubic
from qubic import SpectroImLib as si
from qubicpack.utilities import Qubic_DataDir

from pysimulators import FitsArray

# Get a dictionary
dictfilename = os.environ['QUBIC_DICT']+'spectroimaging.dict'

d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)

nf_sub = [15,]#[2, 4, 5, 10, 12, 14, 15, 16, 18, 20, 22, 24]
dirc = './'

os.makedirs(dirc, exist_ok=True)

for nf in nf_sub:
    print(nf)
    d['nf_sub'] = nf
    sky_config = {'dust': models('d1', d['nside']), 'cmb': models('c1', d['nside'])}
    # sky_config = {'cmb': models('c1', d['nside'])}

    Qubic_sky = si.Qubic_sky(sky_config, d)
    x0 = Qubic_sky.get_simple_sky_map()
    dirc2 = dirc
    try:
        os.makedirs(dirc2)
    except:
        pass
    FitsArray(x0).save(dirc2 + 'CMB_nside{}_nfsub{}.fits'.format(d['nside'], nf))
