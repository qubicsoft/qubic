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
#import fgbuster
import fgbuster
center = qubic.equ2gal(0, -57)
from fgbuster.component_model import (CMB, Dust, Dust_2b, Synchrotron, AnalyticComponent)
from fgbuster import basic_comp_sep, get_instrument
# If there is not this command, the kernel shut down every time..
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pickle

# If there is not this command, the kernel shut down every time..
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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

N=int(sys.argv[1])
ref_fwhm=float(sys.argv[2])
beta0=float(sys.argv[3])
beta1=float(sys.argv[4])
nubreak=int(sys.argv[5])
nu0=int(sys.argv[6])
lmin=int(sys.argv[7])
dl=30
ite=int(sys.argv[8])
ins=int(sys.argv[9])

print('Ins is {}'.format(ins))


def get_instr(config, ref_fwhm):

    if config==qp_config:
        instr = get_instrument('Qubic+')
        ind_nu=15

    elif config==s4_config:
        instr = get_instrument('CMBS4')
        ind_nu=5
    else:
        raise TypeError('Choose QP or S4')

    instr.fwhm = np.ones(len(config['frequency']))*ref_fwhm*60

    return instr, ind_nu


def get_comp(sky, beta):
    comp=[]
    for indi, i in enumerate(sky.keys()):
        if i =='dust':
            comp.append(fgb.component_model.Dust_2b(nu0=beta[3], nubreak=beta[2]))
        elif i =='cmb':
            comp.append(fgb.component_model.CMB())

        elif i =='synchrotron':
            comp.append(fgb.component_model.Synchrotron(nu0=30))
        else:
            pass

    return comp

def compute_cl(lmin, delta_ell, nside, pixok, noise1, noise2, beam):
    maskpix = np.zeros(12*nside**2)
    maskpix[pixok] = 1
    Namaster = nam.Namaster(maskpix, lmin=lmin, lmax=2*nside-1, delta_ell=delta_ell)

    # Compute cross-spectra
    w=None
    leff, cl_noise, w = Namaster.get_spectra(noise1*np.sqrt(2),
                                             map2 = noise2*np.sqrt(2),
                                             purify_e=False,
                                             purify_b=True,
                                             w=w,
                                             verbose=False,
                                             beam_correction=beam,
                                             pixwin_correction=True)

    print(cl_noise.shape)

    return leff, cl_noise


def run_MC_separation(N, config, skyconfig, ref_fwhm, covmap, beta_in, lmin, dl):

    thr = 0
    mymask = (covmap > (np.max(covmap)*thr)).astype(int)
    pixok = mymask > 0

    clnoise=np.zeros((((N, 1, 16, 4))))

    for mc in range(N):

        beta_out=beta_in
        print(beta_out)
        skyconfig['cmb']=np.random.randint(10000000)

        map1_2b, _, _ = qubicplus.BImaps(skyconfig, config).getskymaps(
                                                                    same_resol=ref_fwhm,
                                                                    iib=False,
                                                                    verbose=True,
                                                                    coverage=covmap,
                                                                    noise=True,
                                                                    signoise=1.,
                                                                    beta=beta_out)

        map2_2b, _, _ = qubicplus.BImaps(skyconfig, config).getskymaps(
                                                                    same_resol=ref_fwhm,
                                                                    iib=False,
                                                                    verbose=True,
                                                                    coverage=covmap,
                                                                    noise=True,
                                                                    signoise=1.,
                                                                    beta=beta_out)


        # Define instrument
        instr, ind_nu = get_instr(config, ref_fwhm)

        # Define component
        if beta_out is None:
            comp=[fgb.component_model.Dust(nu0=145), fgb.component_model.CMB()]
        else:
            comp = get_comp(skyconfig, beta_out)

        res1=separate(comp, instr, map1_2b[:, :, pixok], tol=1e-6)
        res2=separate(comp, instr, map2_2b[:, :, pixok], tol=1e-6)

        print(res1.x)
        print(res2.x)

        # Isolate noise
        newmap1_isolate=map1_2b[ind_nu].copy()
        newmap2_isolate=map1_2b[ind_nu].copy()
        for indj, j in enumerate(skyconfig.keys()):
            newmap1_isolate[:, pixok] -= res1.s[indj]
            newmap2_isolate[:, pixok] -= res2.s[indj]

        print(newmap1_isolate.shape)

        # To make sure that coverage is respected
        newmap1_isolate[:, ~pixok]=0
        newmap2_isolate[:, ~pixok]=0


        leff, clnoise[mc, 0] = compute_cl(lmin, dl, 256, pixok, newmap1_isolate, newmap2_isolate, beam=ref_fwhm)


    return leff, clnoise

if ins == 0:
    leff, cl = run_MC_separation(N, s4_config, {'dust':'d02b', 'cmb':42, 'synchrotron':'s0'}, ref_fwhm, covmap, [beta0, beta1, nubreak, nu0], lmin, dl)
elif ins == 1:
    leff, cl = run_MC_separation(N, qp_config, {'dust':'d02b', 'cmb':42, 'synchrotron':'s0'}, ref_fwhm, covmap, [beta0, beta1, nubreak, nu0], lmin, dl)
else:
    raise TypeError('choose 0 for CMB-S4 or 1 for BI !')

print("done !")

#print(clqp[0, 0, :, 2])
#print(clqp[0, 0, :, 2])

mydict = {'leff':leff,
          'cl':cl,
          'sysargv':sys.argv}

path='/pbs/home/m/mregnier/sps1/QUBIC+/fraction'
output = open(path+'/results/cl_bis_ins{}_lmin{}_dl{}_{}reals_{}.pkl'.format(ins, lmin, dl, N, ite), 'wb')
pickle.dump(mydict, output)
output.close()
