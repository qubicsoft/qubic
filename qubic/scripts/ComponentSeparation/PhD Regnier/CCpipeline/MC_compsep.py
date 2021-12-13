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
import fgbuster

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


def separate(comp, instr, maps_to_separate, tol=1e-5, print_option=False):
    solver_options = {}
    solver_options['disp'] = False
    fg_args = comp, instr, maps_to_separate
    fg_kwargs = {'method': 'TNC', 'tol': tol, 'options': solver_options}
    try:
        res = fgbuster.separation_recipes.basic_comp_sep(*fg_args, **fg_kwargs)
    except KeyError:
        fg_kwargs['options']['disp'] = False
        res = fgbuster.separation_recipes.basic_comp_sep(*fg_args, **fg_kwargs)
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
ins=str(sys.argv[3])
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
print('beta0 is {}'.format(beta0))
print('beta1 is {}'.format(beta1))
print('nubreak is {}'.format(nubreak))
print("###################")

def open_pkl(path, name):

    with open(path+name, 'rb') as f:
        data = pickle.load(f)

    map1=data['map1']
    map2=data['map2']

    return map1, map2

def get_instr(ins, ref_fwhm, config):

    if ins=='BI':
        instr = fgbuster.observation_helpers.get_instrument('Qubic+')
        ind_nu=15
        n_nu=33
    elif ins=='S4':
        instr = fgbuster.observation_helpers.get_instrument('CMBS4')
        n_nu=9
        ind_nu=5
    else:
        raise TypeError('Choose QP or S4')

    instr.fwhm = np.ones(n_nu)*ref_fwhm*60

    return instr

def get_comp(sky):
    comp=[]
    for indi, i in enumerate(sky.keys()):
        if i =='dust':
            print("-> Add dust in comp")
            comp.append(fgbuster.component_model.Dust_2b(nu0=145.))
        elif i =='cmb':
            print("-> Add cmb in comp")
            comp.append(fgbuster.component_model.CMB())

        elif i =='synchrotron':
            print("-> Add sync in comp")
            comp.append(fgbuster.component_model.Synchrotron(nu0=145.))
        else:
            pass

    return comp


def run_MC_separation(config, skyconfig, ref_fwhm, covmap, ite, beta_out, ins):

    print('Loading maps')
    map1, map2 = open_pkl('/pbs/home/m/mregnier/sps1/QUBIC+/results/', 'onereals_maps_fwhm{}_instrument{}_{}.pkl'.format(ref_fwhm, ins, ite))
    print(map1.shape)
    print('Done!')
    # Define instrument
    print('begin instr')
    instr = get_instr(ins, ref_fwhm, config)
    instr.depth_p=config['depth_p']
    instr.depth_i=config['depth_i']
    print(instr)
    print('end instr')
    # Define component
    if beta_out is None:
        comp=[fgbuster.component_model.Dust(nu0=145.), fgbuster.component_model.CMB(), fgbuster.component_model.Synchrotron(nu0=145.)]
        print('1', comp)
    else:
        comp = comp=[fgbuster.component_model.Dust_2b(nu0=145.), fgbuster.component_model.CMB(), fgbuster.component_model.Synchrotron(nu0=145.)]
        print('2', comp)

    #print('end comp')

    thr = 0
    mymask = (covmap > (np.max(covmap)*thr)).astype(int)
    pixok = mymask > 0
    # Just to be sure
    map1[:, :, ~pixok]=hp.UNSEEN
    map2[:, :, ~pixok]=hp.UNSEEN
    print('Before separation')
    res1=separate(comp, instr, map1[:, :, pixok])
    res2=separate(comp, instr, map2[:, :, pixok])
    print('After separation')
    print(res1.x)
    print(res2.x)

    #new_dust_maps1=qubicplus.get_scaled_dust_dbmmb_map(nu0, [nu0], res1.x[0], res1.x[1], res1.x[2], 256, 0.03,
    #                                                    radec_center=[0., -57.], temp=20)

    #new_dust_maps2=qubicplus.get_scaled_dust_dbmmb_map(nu0, [nu0], res2.x[0], res2.x[1], res2.x[2], 256, 0.03,
    #                                                    radec_center=[0., -57.], temp=20)

    return [res1.x, res2.x]

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

if ins == 'S4':
    name_instr='S4'
    name_conf='s4_config'
    param_est = run_MC_separation(s4_config, {'cmb':42, 'dust':typedust, 'synchrotron':'s0'}, ref_fwhm, covmap, ite, beta_out, ins)
elif ins == 'BI':
    name_instr='BI'
    name_conf='qp_config'
    param_est = run_MC_separation(qp_config, {'cmb':42, 'dust':typedust, 'synchrotron':'s0'}, ref_fwhm, covmap, ite, beta_out, ins)
else:
    raise TypeError('choose 0 for CMB-S4 or 1 for BI !')

print("done !")

#print(clqp[0, 0, :, 2])
#print(clqp[0, 0, :, 2])

mydict = {'param_est':param_est,
          'sysargv':sys.argv}

CC=1
if CC == 1:
    path='/pbs/home/m/mregnier/sps1/QUBIC+/results'
else:
    path=''
output = open(path+'/paramest_2beta_maps_fwhm{}_instrument{}_{}.pkl'.format(ref_fwhm, name_instr, ite), 'wb')
pickle.dump(mydict, output)
output.close()
