import os
import sys
import numpy as np
from itertools import combinations_with_replacement

import qubic
from qubicpack.utilities import Qubic_DataDir
from qubic import QubicSkySim as qss
from qubic import NamasterLib as nam

# To run the script: $ python Fastsim_spectroimMC_onlymaps.py rep_save nbands config

# Repository for dictionary and input maps
if 'QUBIC_DATADIR' in os.environ:
    pass
else:
    raise NameError('You should define an environment variable QUBIC_DATADIR')

global_dir = Qubic_DataDir(datafile='instrument.py', datadir=os.environ['QUBIC_DATADIR'])
print('global directory:', global_dir)

# Repository where maps and spectra will be saved
rep_save = sys.argv[1]

# Get a dictionary
dictionary = global_dir + '/dicts/spectroimaging_article.dict'
d = qubic.qubicdict.qubicDict()
d.read_from_file(dictionary)

# Number of bands
nbands = int(sys.argv[2])
d['nf_recon'] = nbands
d['nf_sub'] = nbands

all_nfrecon = [1, 2, 3, 4, 5, 8]
if nbands not in all_nfrecon:
    raise ValueError('Wrong number of subbands !')
print('\n nbands:', nbands)

# Config and frequency
config = sys.argv[3]
if config not in ['FI150', 'FI220']:
    raise ValueError('The config should be FI150 or FI220')

d['filter_nu'] = int(config[-3:]) * 1e9

# Input sky
sky_config = {'dust': 'd1'}
Qubic_sky = qss.Qubic_sky(sky_config, d)

rnd_name = qss.random_string(10)

# ================== Make maps =============================
# Getting noise realisations with FastSimulator
nreals = 20
npix = 12 * d['nside'] ** 2
noisemaps = np.zeros((nreals, nbands, npix, 3))

# Make maps with no spatial correlations
for r in range(nreals):
    noisemaps[r, ...], coverage = Qubic_sky.get_partial_sky_maps_withnoise(coverage=None,
                                                                           noise_only=True,
                                                                           spatial_noise=False)

# Reduce it to a patch
seenmap = coverage > np.max(coverage) * 0.1
noisepatch = noisemaps[:, :, seenmap, :]

# Save the noisy patch
np.save(rep_save + f'/noisepatch_NOspatialcorr_nbands{nbands}_' + config + '_' + rnd_name + '.npy',
        noisepatch)
