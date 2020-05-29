from __future__ import division
import os
import sys

import pysm
import qubic
from qubic import QubicSkySim as qss

from qubicpack.utilities import Qubic_DataDir
from pysimulators import FitsArray

# This script creates input sky maps. You have to choose the sky_config and nf_sub
# The arguments are the output directory and a word for the map names:

# Get a dictionary
if 'QUBIC_DATADIR' in os.environ:
    pass
else:
    raise NameError('You should define an environment variable QUBIC_DATADIR')

global_dir = Qubic_DataDir(datafile='instrument.py', datadir=os.environ['QUBIC_DATADIR'])
dictfilename = global_dir + '/dicts/spectroimaging_article.dict'

# dictfilename = os.environ['QUBIC_DICT']+'spectroimaging_article.dict'

d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)

# out_dir = sys.argv[1]
# if out_dir[-1] != '/':
#     out_dir = out_dir + '/'
# os.makedirs(out_dir, exist_ok=True)

# name = sys.argv[2]

nf_sub = [3]#[2, 4, 5, 10, 12, 14, 15, 16, 18, 20, 22, 24]

# ============= Sky config =====================
# CMB map
# ell, totDL, unlensedDL = qss.get_camb_Dl(r=0., lmax=3*d['nside']-1)
# cmb_dict = {'CAMBSpectra':totDL, 'ell':ell, 'seed':None}
# sky_config = {'cmb': cmb_dict}

# Dust map
sky_config = {'dust': 'd1'}

# Synchrotron map
# sky_config = {'synchrotron': 's1'}

# ============== Make the sky =======================
for nf in nf_sub:
    print(nf)
    d['nf_sub'] = nf

    Qubic_sky = qss.Qubic_sky(sky_config, d)
    x0 = Qubic_sky.get_simple_sky_map()

    # FitsArray(x0).save(out_dir + name + '_nside{}_nfsub{}.fits'.format(d['nside'], nf))
