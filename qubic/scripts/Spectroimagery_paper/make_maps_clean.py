from __future__ import division
import os
import sys
import time
import datetime
import shutil

import healpy as hp
import numpy as np

from pysimulators import FitsArray
import qubic

import ReadMC
import SpectroImLib as si

dictfilename = './spectroimaging.dict'
dictmaps = 'maps/'
out_dir = './'
try:
    os.makedirs(out_dir)
except:
    pass

today = datetime.datetime.now().strftime('%Y%m%d')

name = today + '_' + sys.argv[1]

# Number of noise realisations
nreals = int(sys.argv[2])

d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)

''' Parameters to be change for simulations:

[0] Sky creation:	d['nf_sub'] = 12 to 24 ? 4/5 diferent values?

[1] Pointing: 		d['random_pointing'] = True, 
					d['npointings'] = [1000,1500,2000]

[2] TOD creation: 	d['noiseless'] = [True, False]
					if False: change d['detector_nep'] only? 

[3] Reconstruction:	d['nf_sub_rec'] = [1,2,3,4,5,6,7,8] ? 
					tol = [5e-4, 1e-4, 5e-5, 1e-5, 5e-6] :o
					
'''

# Save the dictionary
shutil.copyfile(dictfilename, out_dir + name + '.dict')

# ==========================================

t0 = time.time()

# ===== Sky Creation or Reading =====

x0 = FitsArray(dictmaps + 'nf_sub={}/nf_sub={}.fits'.format(d['nf_sub'], d['nf_sub']))
print('Input Map with shape:', np.shape(x0))

# ==== Pointing strategy ====

p = qubic.get_pointing(d)

# ==== TOD making ====
for j in range(nreals):

    TOD, maps_convolved = si.create_TOD(d, p, x0)
    print('TOD with shape:', np.shape(TOD))

    # ==== Reconstruction ====
    for i, nf_sub_rec in enumerate(d['nf_recon']):
        print('-------------------------- Map-Making on {} sub-map(s)'.format(nf_sub_rec))
        maps_recon, cov, nus, nus_edge, maps_convolved = si.reconstruct_maps(TOD, d, p, nf_sub_rec, x0=x0)
        if nf_sub_rec == 1:
            maps_recon = np.reshape(maps_recon, np.shape(maps_convolved))
        # Look at the coverage of the sky
        cov = np.sum(cov, axis=0)
        maxcov = np.max(cov)
        unseen = cov < maxcov * 0.1
        maps_convolved[:, unseen, :] = hp.UNSEEN
        maps_recon[:, unseen, :] = hp.UNSEEN
        print('************************** Map-Making on {} sub-map(s)Done'.format(nf_sub_rec))

        name_map = '_nfsub{0}_nfrecon{1}_noiseless{2}_nptg{3}_tol{4}_{5}.fits'.format(d['nf_sub'],
                                                                                      d['nf_recon'][i],
                                                                                      d['noiseless'],
                                                                                      d['npointings'],
                                                                                      d['tol'],
                                                                                      str(j).zfill(2))
        ReadMC.save_simu_fits(maps_recon, cov, nus, nus_edge, maps_convolved,
                              out_dir, name + name_map)

        t1 = time.time()
        print('************************** All Done in {} minutes'.format((t1 - t0) / 60))
