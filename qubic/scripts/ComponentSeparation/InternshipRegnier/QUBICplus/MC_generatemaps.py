from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import sys
import pysm3.units as u
from pysm3 import utils
import os

import s4bi
from scipy import constants
import healpy as hp
import numpy as np
import qubicplus
from qubic import NamasterLib as nam
import scipy
import pysm3
import qubic
import pickle

### CMB-S4 config

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

def fct_subopt(nus):
    subnus = [150., 220]
    subval = [1.4, 1.2]
    fct_subopt = np.poly1d(np.polyfit(subnus, subval, 1))
    return fct_subopt(nus)

subnus = [150., 220]
subval = [1.4, 1.2]

def get_coverage(fsky, nside, center_radec=[0., -57.]):
    center = qubic.equ2gal(center_radec[0], center_radec[1])
    uvcenter = np.array(hp.ang2vec(center[0], center[1], lonlat=True))
    uvpix = np.array(hp.pix2vec(nside, np.arange(12*nside**2)))
    ang = np.arccos(np.dot(uvcenter, uvpix))
    indices = np.argsort(ang)
    okpix = ang < -1
    okpix[indices[0:int(fsky * 12*nside**2)]] = True
    mask = np.zeros(12*nside**2)
    mask[okpix] = 1
    return mask

def qubicify(config, qp_nsub, qp_effective_fraction):
    nbands = np.sum(qp_nsubs)
    qp_config = config.copy()
    for k in qp_config.keys():
        qp_config[k]=[]
    qp_config['nbands'] = nbands
    qp_config['fsky'] = config['fsky']
    qp_config['ntubes'] = config['ntubes']
    qp_config['nyears'] = config['nyears']
    qp_config['initial_band'] = []

    for i in range(len(config['frequency'])):
        #print(config['edges'][i][0], config['edges'][i][-1])
        newedges = np.linspace(config['edges'][i][0], config['edges'][i][-1], qp_nsubs[i]+1)
        #print(newedges)
        newfreqs = (newedges[0:-1]+newedges[1:])/2
        newbandwidth = newedges[1:] - newedges[0:-1]
        newdnu_nu = newbandwidth / newfreqs
        newfwhm = config['fwhm'][i] * config['frequency'][i]/newfreqs
        scalefactor_noise = np.sqrt(qp_nsubs[i]) * fct_subopt(config['frequency'][i]) / qp_effective_fraction[i]
        newdepth_p = config['depth_p'][i] * np.ones(qp_nsub[i]) * scalefactor_noise
        newdepth_i = config['depth_i'][i] * np.ones(qp_nsub[i]) * scalefactor_noise
        newdepth_e = config['depth_e'][i] * np.ones(qp_nsub[i]) * scalefactor_noise
        newdepth_b = config['depth_b'][i] * np.ones(qp_nsub[i]) * scalefactor_noise
        newell_min = np.ones(qp_nsub[i]) * config['ell_min'][i]
        newnside = np.ones(qp_nsub[i]) * config['nside'][i]
        neweffective_fraction = np.ones(qp_nsub[i]) * qp_effective_fraction[i]
        initial_band = np.ones(qp_nsub[i]) * config['frequency'][i]

        for k in range(qp_nsubs[i]):
            if qp_effective_fraction[i] != 0:
                qp_config['frequency'].append(newfreqs[k])
                if i >= 3:
                    qp_config['depth_p'].append(newdepth_p[k])
                    qp_config['depth_i'].append(newdepth_i[k])
                    qp_config['depth_e'].append(newdepth_e[k])
                    qp_config['depth_b'].append(newdepth_b[k])
                else:
                    qp_config['depth_p'].append(s4_config['depth_p'][i])
                    qp_config['depth_i'].append(s4_config['depth_i'][i])
                    qp_config['depth_e'].append(s4_config['depth_e'][i])
                    qp_config['depth_b'].append(s4_config['depth_b'][i])
                qp_config['fwhm'].append(newfwhm[k])
                qp_config['bandwidth'].append(newbandwidth[k])
                qp_config['dnu_nu'].append(newdnu_nu[k])
                qp_config['ell_min'].append(newell_min[k])
                qp_config['nside'].append(newnside[k])

                qp_config['effective_fraction'].append(neweffective_fraction[k])
                qp_config['initial_band'].append(initial_band[k])
        for k in range(qp_nsubs[i]+1):
            if qp_effective_fraction[i] != 0:
                qp_config['edges'].append(newedges[k])

        #qp_config['depth_p'][:3] = s4_config['depth_p'][:3]
        #qp_config['depth_i'][:3] = s4_config['depth_i'][:3]

    fields = ['frequency', 'depth_p', 'depth_i', 'depth_e', 'depth_b', 'fwhm', 'bandwidth',
              'dnu_nu', 'ell_min', 'nside', 'edges', 'effective_fraction', 'initial_band']
    for j in range(len(fields)):
        qp_config[fields[j]] = np.array(qp_config[fields[j]])

    return qp_config


covmap = get_coverage(0.03, nside=256)
thr = 0
mymask = (covmap > (np.max(covmap)*thr)).astype(int)
pixok = mymask > 0


qp_nsubs = np.array([1, 1, 1, 5, 5, 5, 5, 5, 5])
qp_effective_fraction = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
qp_config=qubicify(s4_config, qp_nsubs, qp_effective_fraction)

ref_fwhm=float(sys.argv[1])
ite=int(sys.argv[2])
ins=int(sys.argv[3])
normal=int(sys.argv[4])
beta0=float(sys.argv[5])
beta1=float(sys.argv[6])
nubreak=int(sys.argv[7])
nu0=145

print("###################")
print('ref_fwhm is {}'.format(ref_fwhm))
print('beta0 is {}'.format(beta0))
print('beta1 is {}'.format(beta1))
print('nubreak is {}'.format(nubreak))
print('ite is {}'.format(ite))
print('ins is {}'.format(ins))
print('normal is {}'.format(normal))
print("###################")


def run_MC_generation(config, skyconfig, ref_fwhm, covmap, beta_in):

    thr = 0
    mymask = (covmap > (np.max(covmap)*thr)).astype(int)
    pixok = mymask > 0

    beta_out=beta_in
    print(beta_out)
    skyconfig['cmb']=np.random.randint(10000000)

    print('Creation of first map..')
    map1, _, _ = qubicplus.BImaps(skyconfig, config).getskymaps(same_resol=ref_fwhm,
                                                                iib=False,
                                                                verbose=False,
                                                                coverage=covmap,
                                                                noise=True,
                                                                signoise=1.,
                                                                beta=beta_out)

    print('First map created !')
    print('Creation of second map..')
    map2, _, _ = qubicplus.BImaps(skyconfig, config).getskymaps(same_resol=ref_fwhm,
                                                                iib=False,
                                                                verbose=False,
                                                                coverage=covmap,
                                                                noise=True,
                                                                signoise=1.,
                                                                beta=beta_out)
    print('Second map created !')

    return map1, map2

print('Simulation started')

if normal == 1 :
    tab_beta=None
    typedust='d0'
    name='truebeta'
    print(ref_fwhm)
elif normal == 0 :
    tab_beta=[beta0, beta1, nubreak, nu0]
    typedust='d02b'
    name='2beta'
    print(ref_fwhm)
else:
    raise TypeError('choose 0 for one spectral index or 1 for modified BB !')

if ins == 0:
    instr='S4'
    print(ref_fwhm)
    map1, map2 = run_MC_generation(s4_config, {'dust':typedust, 'cmb':42}, ref_fwhm, covmap, tab_beta)
elif ins == 1:
    instr='BI'
    print(ref_fwhm)
    map1, map2 = run_MC_generation(qp_config, {'dust':typedust, 'cmb':42}, ref_fwhm, covmap, tab_beta)
else:
    raise TypeError('choose 0 for CMB-S4 or 1 for BI !')

print("done !")

#print(clqp[0, 0, :, 2])
#print(clqp[0, 0, :, 2])

mydict = {'map1':map1,
          'map2':map2,
          'sysargv':sys.argv,
          's4_config':s4_config,
          'qp_config':qp_config}

CC=1
if CC == 1:
    path='/pbs/home/m/mregnier/sps1/QUBIC+'
else:
    path=''
output = open(path+'/results/onereals_maps_fwhm{}_instrument{}_{}.pkl'.format(ref_fwhm, instr, ite), 'wb')
pickle.dump(mydict, output)
output.close()
