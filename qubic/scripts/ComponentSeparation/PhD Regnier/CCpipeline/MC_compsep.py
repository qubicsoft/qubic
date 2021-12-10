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
from fgbuster.component_model import (CMB, Dust, Dust_2b, Synchrotron, AnalyticComponent)
from fgbuster import basic_comp_sep, get_instrument

def fct_subopt(nus):
    subnus = [150., 220]
    subval = [1.4, 1.2]
    fct_subopt = np.poly1d(np.polyfit(subnus, subval, 1))
    return fct_subopt(nus)

subnus = [150., 220]
subval = [1.4, 1.2]


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

import fgbuster as fgb
from fgbuster import basic_comp_sep, get_instrument, Dust, CMB


def separate(comp, instr, maps_to_separate, tol=1e-5, print_option=False):
    solver_options = {}
    solver_options['disp'] = False
    fg_args = comp, instr, maps_to_separate
    fg_kwargs = {'method': 'Nelder-Mead', 'tol': tol, 'options': solver_options}
    try:
        res = fgb.basic_comp_sep(*fg_args, **fg_kwargs)
    except KeyError:
        fg_kwargs['options']['disp'] = False
        res = fgb.basic_comp_sep(*fg_args, **fg_kwargs)
    if print_option:
        print()
        print("message:", res.message)
        print("success:", res.success)
        print("result:", res.x)

    return res

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

qp_nsubs = np.array([1, 1, 1, 5, 5, 5, 5, 5, 5])
qp_effective_fraction = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
qp_config=qubicify(s4_config, qp_nsubs, qp_effective_fraction)

covmap = get_coverage(0.03, nside=256)

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
print('ite is {}'.format(ite))
print('ins is {}'.format(ins))
print('normal is {}'.format(normal))
print("###################")

def open_pkl(path, name, name_conf):

    with open(path+name, 'rb') as f:
        data = pickle.load(f)

    map1=data['map1']
    map2=data['map2']
    config=data[name_conf]

    return map1, map2, config

def get_instr(name_conf, ref_fwhm, config):

    if name_conf=='qp_config':
        instr = get_instrument('Qubic+')
        ind_nu=15

    elif name_conf=='s4_config':
        instr = get_instrument('CMBS4')
        ind_nu=5
    else:
        raise TypeError('Choose QP or S4')

    instr.fwhm = np.ones(len(config['frequency']))*ref_fwhm*60
    instr.depth_i = config['depth_i']
    instr.depth_p = config['depth_p']

    return instr, ind_nu

def get_comp(sky, beta):
    comp=[]
    for indi, i in enumerate(sky.keys()):
        if i =='dust':
            comp.append(fgb.component_model.Dust_2b(nu0=beta[3]))
        elif i =='cmb':
            comp.append(fgb.component_model.CMB())

        elif i =='synchrotron':
            comp.append(fgb.component_model.Synchrotron(nu0=beta[3]))
        else:
            pass

    return comp

def run_MC_separation(name_conf, skyconfig, ref_fwhm, covmap, name_instr, ite, normal, beta_out):

    map1, map2, config = open_pkl('/pbs/home/m/mregnier/sps1/QUBIC+/results/', 'onereals_maps_fwhm{}_instrument{}_{}.pkl'.format(ref_fwhm, name_instr, ite), name_conf)

    # Define instrument
    instr, ind_nu = get_instr(name_conf, ref_fwhm, config)
    if normal == 1:
        arg_est=np.zeros(2)
    else:
        arg_est=np.zeros(4)

    # Define component
    if beta_out is None:
        comp=[fgb.component_model.Dust(nu0=nu0), fgb.component_model.CMB()]
    else:
        comp = get_comp(skyconfig, [1.44, 1.64, nubreak])

    thr = 0
    mymask = (covmap > (np.max(covmap)*thr)).astype(int)
    pixok = mymask > 0

    res1=separate(comp, instr, map1[:, :, pixok], tol=1e-6)
    res2=separate(comp, instr, map2[:, :, pixok], tol=1e-6)

    if normal == 1:
        arg_est[0] = res1.x[0]
        arg_est[1] = res1.x[1]
    else:
        arg_est[0] = res1.x[0]
        arg_est[1] = res1.x[1]
        arg_est[2] = res1.x[2]
        arg_est[3] = res1.x[3]
    #print(arg_est)
    print(res1.x)
    print(res2.x)

    compmaps1=np.zeros(((res1.s.shape[0], 3, 12*256**2)))
    compmaps2=np.zeros(((res2.s.shape[0], 3, 12*256**2)))

    compmaps1[:, :, pixok] = res1.s
    compmaps2[:, :, pixok] = res2.s



    return [map1, compmaps1], [map2, compmaps2]

print('Simulation started')

if normal == 1 :
    typedust='d0'
    name='truebeta'
    beta_out=None
elif normal == 0 :
    typedust='d02b'
    name='2beta'
    beta_out=[beta0, beta1, nubreak, nu0]
else:
    raise TypeError('choose 0 for one spectral index or 1 for modified BB !')

if ins == 0:
    name_instr='S4'
    name_conf='s4_config'
    allcomp1, allcomp2 = run_MC_separation(name_conf, {'dust':typedust, 'cmb':42}, ref_fwhm, covmap, name_instr, ite, normal, beta_out)
elif ins == 1:
    name_instr='BI'
    name_conf='qp_config'
    allcomp1, allcomp2 = run_MC_separation(name_conf, {'dust':typedust, 'cmb':42}, ref_fwhm, covmap, name_instr, ite, normal, beta_out)
else:
    raise TypeError('choose 0 for CMB-S4 or 1 for BI !')

print("done !")

#print(clqp[0, 0, :, 2])
#print(clqp[0, 0, :, 2])

mydict = {'allcomp1':allcomp1,
          'allcomp2':allcomp2,
          'sysargv':sys.argv}

CC=1
if CC == 1:
    path='/pbs/home/m/mregnier/sps1/QUBIC+/results'
else:
    path=''
output = open(path+'/compsep_maps_fwhm{}_instrument{}_{}.pkl'.format(ref_fwhm, name_instr, ite), 'wb')
pickle.dump(mydict, output)
output.close()
