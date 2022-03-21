import numpy as np
import pysm3
import pysm3.units as u
from pysm3 import utils
import healpy as hp
import fgbuster
import matplotlib.pyplot as plt
import os
import qubic
from qubic import mcmc
center = qubic.equ2gal(-30, -30)
import warnings
import qubicplus
import pickle
import sys
from qubic import NamasterLib as nam
import definitions
print(fgbuster.__path__)
warnings.filterwarnings("ignore")

freqs = np.array([85., 95., 145., 155., 220., 270.])
bandwidth = np.array([20.4, 22.8, 31.9, 34.1, 48.4, 59.4])
dnu_nu = bandwidth/freqs
beam_fwhm = np.array([25.5, 25.5, 22.7, 22.7, 13., 13.])
mukarcmin_TT = np.array([2.02, 1.78, 3.89, 4.16, 10.15, 17.4])
mukarcmin_EE = np.array([1.34, 1.18, 1.8, 1.93, 4.71, 8.08])
mukarcmin_BB = np.array([1.27, 1.12, 1.76, 1.89, 4.6, 7.89])
ell_min = np.array([30, 30, 30, 30, 30, 30])
nside = np.array([512, 512, 512, 512, 512, 512])
edges_min = freqs * (1. - dnu_nu/2)
edges_max = freqs * (1. + dnu_nu/2)
edges = [[edges_min[i], edges_max[i]] for i in range(len(freqs))]
s4_config = {
    'nbands': len(freqs),
    'frequency': freqs,
    'depth_p': 0.5*(mukarcmin_EE + mukarcmin_BB),
    'depth_i': mukarcmin_TT,
    'depth_e': mukarcmin_EE,
    'depth_b': mukarcmin_BB,
    'fwhm': beam_fwhm,
    'bandwidth': bandwidth,
    'dnu_nu': dnu_nu,
    'ell_min': ell_min,
    'nside': nside,
    'fsky': 0.03,
    'ntubes': 12,
    'nyears': 7.,
    'edges': edges,
    'effective_fraction': np.zeros(len(freqs))+1.
            }

# ~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~ #

qp_nsub = np.array([5, 5, 1, 1, 5, 5])
qp_effective_fraction = np.array([1, 1, 1, 1, 1, 1])
qp_config = definitions.qubicify(s4_config, qp_nsub, qp_effective_fraction)

#Corrected depths
#qp_config['depth_p'][:3] = s4_config['depth_p'][:3]
#qp_config['depth_i'][:3] = s4_config['depth_i'][:3]

#qp_config['depth_p'][-2:] = s4_config['depth_p'][-2:]
#qp_config['depth_i'][-2:] = s4_config['depth_i'][-2:]

N=int(sys.argv[1])
ite=int(sys.argv[2])
nubreak=int(sys.argv[3])
r=float(sys.argv[4])

# ~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~ #

dl=35
lmin=21

print('\n//////// Creating Namaster class')
maskpix = np.zeros(12*256**2)
covmap = definitions.get_coverage(0.03, 256)
pixok = covmap > 0
maskpix[pixok] = 1
Namaster = nam.Namaster(maskpix, lmin=lmin, lmax=355, delta_ell=dl)
ell_binned, _=Namaster.get_binning(256)
print('//////// Done')

db=np.linspace(-0.1, 0.1, 11)
prop=[0, 0.1, 0.2, 0.3, 1]

cl=np.zeros((((len(prop), len(db), N, 1, len(ell_binned), 4))))
param=np.zeros((((len(prop), len(db), 2*N, 2))))

for idb, jdb in enumerate(db):

    print('\\\\\\\\\\\\ db = {:.2f} ////////////'.format(jdb))
    config=[s4_config, qp_config]

    for jprop, jpro in enumerate(prop):



        if jpro == 0:
            maps_db=definitions._get_maps_without_noise(s4_config, db=jdb, nubreak=nubreak, prop=prop[jprop], r=r)
            leff, cl_db, param[jprop, idb]=definitions._get_param(s4_config, maps_db, N=N,
                                                                Namaster=Namaster, prop=prop[jprop])
        elif jpro == 1:
            maps_db=definitions._get_maps_without_noise(qp_config, db=jdb, nubreak=nubreak, prop=prop[jprop], r=r)
            leff, cl_db, param[jprop, idb]=definitions._get_param(qp_config, maps_db, N=N,
                                                                Namaster=Namaster, prop=prop[jprop])

        else:
            maps_db=definitions._get_maps_without_noise(config, db=jdb, nubreak=nubreak, prop=prop[jprop], r=r)
            leff, cl_db, param[jprop, idb]=definitions._get_param(config, maps_db, N=N,
                                                                    Namaster=Namaster, prop=prop[jprop])


        cl[jprop, idb]=cl_db.copy()


#print(clBB)
print()
print(np.mean(param, axis=2)[:, :, 0])

pickle.dump([leff, cl, param, db, sys.argv], open('/pbs/home/m/mregnier/sps1/QUBIC+/d0/cls/results/cls_nolensing_fitd0_2b_r{:.3f}_iib10_QU_fixtempfixsync_truenub{}_{}reals_{}.pkl'.format(r, nubreak, N, ite), "wb"))
