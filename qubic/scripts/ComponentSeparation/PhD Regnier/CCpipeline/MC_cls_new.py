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

import sys

freqs = np.array([20., 30., 40., 85., 95., 145., 155., 220., 270.])
bandwidth = np.array([5., 9., 12., 20.4, 22.8, 31.9, 34.1, 48.4, 59.4])
dnu_nu = bandwidth/freqs
beam_fwhm = np.array([11., 72.8, 72.8, 25.5, 25.5, 22.7, 22.7, 13., 13.])
mukarcmin_TT = np.array([16.5, 9.36, 11.85, 2.02, 1.78, 3.89, 4.16, 10.15, 17.4])
mukarcmin_EE = np.array([10.87, 6.2, 7.85, 1.34, 1.18, 1.8, 1.93, 4.71, 8.08])
mukarcmin_BB = np.array([10.23, 5.85, 7.4, 1.27, 1.12, 1.76, 1.89, 4.6, 7.89])
ell_min = np.array([30, 30, 30, 30, 30, 30, 30, 30, 30])
nside = np.array([512, 512, 512, 512, 512, 512, 512, 512, 512])
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

N_bands=int(sys.argv[7])
qp_nsub = np.array([N_bands, N_bands, N_bands, N_bands, N_bands, N_bands, N_bands, N_bands, N_bands])
name_split=definitions._give_name_splitbands(qp_nsub)
qp_effective_fraction = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
qp_config = definitions.qubicify(s4_config, qp_nsub, qp_effective_fraction)

#Corrected depths
qp_config['depth_p'][:3] = s4_config['depth_p'][:3]
qp_config['depth_i'][:3] = s4_config['depth_i'][:3]

N=int(sys.argv[1])
ite=int(sys.argv[2])
nubreak=int(sys.argv[3])
r=float(sys.argv[4])
iib=int(sys.argv[5])
fixsync=int(sys.argv[6])
fit=int(sys.argv[8])
model_input=str(sys.argv[9])
print(model_input)

# ~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~ #

'''
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
'''

dl=35
lmin=21

print('\n//////// Creating Namaster class')
maskpix = np.zeros(12*256**2)
covmap = definitions.get_coverage(0.03, 256)
pixok = covmap > 0
maskpix[pixok] = 1
Namaster = nam.Namaster(maskpix, lmin=lmin, lmax=355, delta_ell=dl)
ell_binned, _=Namaster.get_binning(256)

db=[0]#np.linspace(-0.1, 0.1, 7)
prop=[0, 1]
nparam=1
name_s=''
if fit == 0:
    nparam=1
    if fixsync == 0:
        nparam+=1
    else:
        name_s='_fixsync'
elif fit == 1:
    nparam=3
    if fixsync == 0:
        nparam+=1
    else:
        name_s='_fixsync'
maskpix = np.zeros(12*256**2)
covmap = definitions.get_coverage(0.03, 256)
pixok = covmap > 0
maskpix[pixok] = 1
Namaster = nam.Namaster(maskpix, lmin=lmin, lmax=355, delta_ell=dl)
cl=np.zeros((((len(prop), len(db), N, len(ell_binned), 4))))
if model_input == 'd0':
    param=np.zeros((((len(prop), len(db), 2*N, nparam))))
    NSIDE_est=0
    nside_out=256
else:
    NSIDE_est=16
    nside_out=256
    param=np.zeros((((len(prop), len(db), 2*N, nparam, 12*NSIDE_est**2))))
print(param.shape)
if fit == 0:
    if fixsync == 0:
        bnds=[(0.5, 3), (-5, -1)]
        comp=[fgbuster.component_model.CMB(),fgbuster.component_model.Dust(nu0=100, temp=20),fgbuster.component_model.Synchrotron(nu0=100)]
        #comp[1].defaults=[1.54]
    else:
        bnds=[(0.5, 3)]
        comp=[fgbuster.component_model.CMB(),fgbuster.component_model.Dust(nu0=100, temp=20)]
        #comp[1].defaults=[1.54]
if fit == 1:
    if fixsync == 0:
        bnds=[(0.5, 3), (-5, -1)]
        comp=[fgbuster.component_model.CMB(),fgbuster.component_model.Dust_2b(nu0=100, temp=20, break_width=0.3),fgbuster.component_model.Synchrotron(nu0=100)]
        #comp[1].defaults=[1.54, 1.54, nubreak]
    else:
        bnds=[(0.5, 3)]
        comp=[fgbuster.component_model.CMB(),fgbuster.component_model.Dust_2b(nu0=100, temp=20, break_width=0.3)]
        #comp[1].defaults=[1.54, 1.54, nubreak]

for idb, jdb in enumerate(db):

    print('\\\\\\\\\\\\ db = {:.2f} ////////////'.format(jdb))
    config=[s4_config, qp_config]

    for jprop, jpro in enumerate(prop):
        covmap = definitions.get_coverage(0.03, 256)

        if jpro == 0:
            if jdb != 0:
                model='d02b'
            else:
                model=model_input

            print('input model : ', model_input)
            #maps_db=definitions._get_maps_without_noise(s4_config, db=jdb, nubreak=nubreak, prop=prop[jprop], r=r, iib=iib)
            map, instrs4 = definitions.give_me_maps_instr(s4_config, r, covmap, jdb, nubreak, jpro, iib*N_bands, model=model, nside_out=nside_out, nside_index=NSIDE_est, fixsync=fixsync)
            print('Shape of input maps -> ', map.shape)

            k=0
            l=1
            for i in range(N):
                print(i)

                noise=definitions._get_noise(s4_config, jpro, nside_out=nside_out)
                inputs1=map[:, :, :]+noise[:, :, :].copy()




                print()
                print('     Components Separations')
                print()
                if NSIDE_est==0:
                    r1=fgbuster.separation_recipes.basic_comp_sep(comp, instrs4, inputs1[:, 1:, pixok], tol=1e-18, method='TNC', options={'maxiter':10000000})
                else:
                    covmap = definitions.get_coverage(0.03, nside_out)
                    pixok=covmap>0
                    inputs1[:, :, ~pixok]=hp.UNSEEN
                    r1=fgbuster.separation_recipes.basic_comp_sep(comp, instrs4, inputs1[:, 1:, :], tol=1e-18, method='TNC', options={'maxiter':10000000}, nside=NSIDE_est, bounds=bnds)
                print("Done")

                noise=definitions._get_noise(s4_config, jpro, nside_out=nside_out)
                inputs2=map[:, :, :]+noise[:, :, :].copy()


                if NSIDE_est==0:
                    r2=fgbuster.separation_recipes.basic_comp_sep(comp, instrs4, inputs2[:, 1:, pixok], tol=1e-18, method='TNC', options={'maxiter':10000000})
                else:
                    covmap = definitions.get_coverage(0.03, nside_out)
                    pixok=covmap>0
                    inputs2[:, :, ~pixok]=hp.UNSEEN
                    r2=fgbuster.separation_recipes.basic_comp_sep(comp, instrs4, inputs2[:, 1:, :], tol=1e-18, method='TNC', options={'maxiter':10000000}, nside=NSIDE_est, bounds=bnds)

                print("Done")


                print()
                print(r1.x, r2.x)
                print()



                param[jprop, idb, k]=r1.x
                param[jprop, idb, l]=r2.x
                l+=2
                k+=2

                if model_input != 'd0':
                    r1.x=hp.pixelfunc.ud_grade(r1.x, nside_out)
                    r2.x=hp.pixelfunc.ud_grade(r2.x, nside_out)

                #covmap = definitions.get_coverage(0.025, 256)

                leff, cls=definitions.get_cls(r1.x, r2.x, comp, instrs4, inputs1, inputs2, covmap, s4_config['frequency'], Namaster)

                cl[jprop, idb, i]=cls.copy()
                print(cl[jprop, idb, i])


        if jpro == 1:
            if jdb != 0:
                model='d02b'
            else:
                model=model_input

            print('input model : ', model_input)
            #maps_db=definitions._get_maps_without_noise(s4_config, db=jdb, nubreak=nubreak, prop=prop[jprop], r=r, iib=iib)
            map, instrqp = definitions.give_me_maps_instr(qp_config, r, covmap, jdb, nubreak, jpro, iib, model=model, nside_out=nside_out, nside_index=NSIDE_est, fixsync=fixsync)
            print('Shape of input maps -> ', map.shape)

            k=0
            l=1
            for i in range(N):
                print(i)

                noise=definitions._get_noise(qp_config, jpro, nside_out=nside_out)
                inputs1=map[:, :, :]+noise[:, :, :].copy()

                print()
                print('     Components Separations')
                print()
                if NSIDE_est==0:
                    r1=fgbuster.separation_recipes.basic_comp_sep(comp, instrqp, inputs1[:, 1:, pixok], tol=1e-18, method='TNC', options={'maxiter':10000000})
                else:
                    covmap = definitions.get_coverage(0.03, nside_out)
                    pixok=covmap>0
                    inputs1[:, :, ~pixok]=hp.UNSEEN
                    r1=fgbuster.separation_recipes.basic_comp_sep(comp, instrqp, inputs1[:, 1:, :], tol=1e-18, method='TNC', options={'maxiter':10000000}, nside=NSIDE_est, bounds=bnds)
                print("Done")

                noise=definitions._get_noise(qp_config, jpro, nside_out=nside_out)
                inputs2=map[:, :, :]+noise[:, :, :].copy()


                if NSIDE_est==0:
                    r2=fgbuster.separation_recipes.basic_comp_sep(comp, instrqp, inputs2[:, 1:, pixok], tol=1e-18, method='TNC', options={'maxiter':10000000})
                else:
                    covmap = definitions.get_coverage(0.03, nside_out)
                    pixok=covmap>0
                    inputs2[:, :, ~pixok]=hp.UNSEEN
                    r2=fgbuster.separation_recipes.basic_comp_sep(comp, instrqp, inputs2[:, 1:, :], tol=1e-18, method='TNC', options={'maxiter':10000000}, nside=NSIDE_est, bounds=bnds)

                print("Done")


                print()
                print(r1.x, r2.x)
                print()



                param[jprop, idb, k]=r1.x
                param[jprop, idb, l]=r2.x
                l+=2
                k+=2

                if model_input != 'd0':
                    r1.x=hp.pixelfunc.ud_grade(r1.x, nside_out)
                    r2.x=hp.pixelfunc.ud_grade(r2.x, nside_out)

                #covmap = definitions.get_coverage(0.025, 256)

                leff, cls=definitions.get_cls(r1.x, r2.x, comp, instrqp, inputs1, inputs2, covmap, qp_config['frequency'], Namaster)

                cl[jprop, idb, i]=cls.copy()
                print(cl[jprop, idb, i])






#print(clBB)
#print()
print(np.mean(param, axis=2))
print(np.std(param, axis=2))
print()
print(np.mean(cl, axis=2))

pickle.dump([leff, cl, param, db, sys.argv], open('/pbs/home/m/mregnier/sps1/QUBIC+/d0/cls/results/cls_{}fitwithd1_split{}_nolensing_r{:.3f}_iib{:.0f}_QU{}_truenub{}_{}reals_{}.pkl'.format(model_input, name_split, r, iib, name_s, nubreak, N, ite), "wb"))
