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
print(fgbuster.__path__)

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

#Corrected depths
qp_config['depth_p'][:3] = s4_config['depth_p'][:3]
qp_config['depth_i'][:3] = s4_config['depth_i'][:3]

covmap = get_coverage(0.03, nside=256)
thr = 0
mymask = (covmap > (np.max(covmap)*thr)).astype(int)
pixok = mymask > 0

N=int(sys.argv[1])

import fgbuster

def paramCompSep(config, name, noisy=True, skyconfig={'cmb':42, 'dust':'d02b', 'synchrotron':'s0'}):

    print(skyconfig['cmb'])

    nside=256
    covmap = get_coverage(0.03, nside)
    thr = 0
    mymask = (covmap > (np.max(covmap)*thr)).astype(int)
    pixok = mymask > 0
    beta=[1.44, 1.64, 265]
    nu0=145

    print('######################')
    print('nu0 = {:.0f} GHz'.format(nu0))
    print('beta0 = {:.3f}'.format(beta[0]))
    print('beta1 = {:.3f}'.format(beta[1]))
    print('nubreak = {:.3f} GHz'.format(beta[2]))
    print('sky fraction = {:.2f} %'.format(0.03*100))
    print('###################### \n \n')

    ### Generate maps

    print("Maps generation")

    if noisy:
        maps_to_separate, _, _ = qubicplus.BImaps(skyconfig, config).getskymaps(same_resol=0,
                                      verbose=False,
                                      coverage=covmap,
                                      noise=noisy,
                                      signoise=1.,
                                      beta=beta)
    else:
        maps_to_separate = qubicplus.BImaps(skyconfig, config).getskymaps(same_resol=0,
                                      verbose=False,
                                      coverage=covmap,
                                      noise=noisy,
                                      signoise=1.,
                                      beta=beta)

    print("Initialize instrument")
    instr=fgbuster.observation_helpers.get_instrument(name)
    instr.frequency = config['frequency']
    instr.fwhm = config['fwhm']
    instr.depth_i = config['depth_i']
    instr.depth_p = config['depth_p']

    # Define components
    print("Define components")
    comp=[fgbuster.component_model.Dust_2b(nu0=nu0, temp=20),
          fgbuster.component_model.CMB(),
          fgbuster.component_model.Synchrotron(nu0=nu0)]


    options={'disp':False, 'gtol': 1e-12, 'eps': 1e-12, 'maxiter': 100, 'ftol': 1e-12 }
    tol=1e-18
    method='TNC'

    fg_args = comp, instr, maps_to_separate[:, :, pixok]
    fg_kwargs = {'method':method, 'tol':tol, 'options':options}
    print('Separation')
    res = fgbuster.basic_comp_sep(*fg_args, **fg_kwargs)

    #print(res.x)

    print('\nFit of spectral indices -> ', res.x)
    print('Estimated error bar on spectral indices -> ', np.diag(res.Sigma))

    print('Estimation of Mixing Matrix')
    # Estimation of mixing matrix
    A = fgbuster.mixingmatrix.MixingMatrix(*comp)
    A_ev = A.evaluator(instr.frequency)
    # Mixing matrix evaluation at max L
    A_maxL = A_ev(np.round(res.x, 3))


    # pixel seen
    ind=np.where(pixok != 0)[0]
    mysolution=np.ones(((3, 3, 12*nside**2)))*hp.UNSEEN
    if noisy:

        invN = np.diag(hp.nside2resol(256, arcmin=True) / (instr.depth_p))**2
        inv_AtNA = np.linalg.inv(A_maxL.T.dot(invN).dot(A_maxL))

        # Loop over pixels
        for i in ind:
            inv_AtNA_dot_At_dot_invN=inv_AtNA.dot(A_maxL.T).dot(invN)
            # Loop over stokes parameters
            for s in range(3):
                mysolution[:, s, i] = inv_AtNA_dot_At_dot_invN.dot(maps_to_separate[:, s, i])
        #print('Shape of inv_AtNA_dot_At_dot_invN -> ', inv_AtNA_dot_At_dot_invN.shape)
    else:
        print('\n          >>> building s = Wd in pixel space \n')
        mysol = fgbuster.algebra.Wd(A_maxL, maps_to_separate[:, :, pixok].T).T
        mysolution[:, :, pixok]=mysol.copy()

    # Normalization
    ind_nu=np.where(config['frequency']==nu0)[0][0]

    for c in range(len(comp)):
        mysolution[c, :, :]*=A_maxL[ind_nu, c]

    return mysolution, res.x, maps_to_separate

print('Simulation started')

param_s4 = np.zeros((N, 4))
param_bi = np.zeros((N, 4))

maps_s4_1 = np.zeros((((N, 3, 3, np.sum(pixok)))))
maps_bi_1 = np.zeros((((N, 3, 3, np.sum(pixok)))))
maps_s4_2 = np.zeros((((N, 3, 3, np.sum(pixok)))))
maps_bi_2 = np.zeros((((N, 3, 3, np.sum(pixok)))))

allmaps_to_separate_s4_1 = np.zeros((((N, 9, 3, np.sum(pixok)))))
allmaps_to_separate_bi_1 = np.zeros((((N, len(qp_config['frequency']), 3, np.sum(pixok)))))

allmaps_to_separate_s4_2 = np.zeros((((N, 9, 3, np.sum(pixok)))))
allmaps_to_separate_bi_2 = np.zeros((((N, len(qp_config['frequency']), 3, np.sum(pixok)))))

cls4 = np.zeros(((N, 16, 4)))
clbi = np.zeros(((N, 16, 4)))

maskpix = np.zeros(12*256**2)
maskpix[pixok] = 1
Namaster = nam.Namaster(maskpix, lmin=40, lmax=2*256-1, delta_ell=30)

for i in range(N):
    seed=np.random.randint(10000000)
    recons_oneseed_s4_1=np.zeros((((1, 3, 3, np.sum(pixok)))))
    recons_oneseed_bi_1=np.zeros((((1, 3, 3, np.sum(pixok)))))

    recons_oneseed_s4_2=np.zeros((((1, 3, 3, np.sum(pixok)))))
    recons_oneseed_bi_2=np.zeros((((1, 3, 3, np.sum(pixok)))))

    for mc in range(1):
        sols4_1, _, maps_to_separate_s4_1=paramCompSep(s4_config, 'CMBS4', noisy=True, skyconfig={'cmb':seed, 'dust':'d02b', 'synchrotron':'s0'})
        sols4_2, _, maps_to_separate_s4_2=paramCompSep(s4_config, 'CMBS4', noisy=True, skyconfig={'cmb':seed, 'dust':'d02b', 'synchrotron':'s0'})

        solbi_1, _, maps_to_separate_bi_1=paramCompSep(qp_config, 'Qubic+', noisy=True, skyconfig={'cmb':seed, 'dust':'d02b', 'synchrotron':'s0'})
        solbi_2, _, maps_to_separate_bi_2=paramCompSep(qp_config, 'Qubic+', noisy=True, skyconfig={'cmb':seed, 'dust':'d02b', 'synchrotron':'s0'})

        recons_oneseed_s4_1[mc]=sols4_1[:, :, pixok]
        recons_oneseed_bi_1[mc]=solbi_1[:, :, pixok]

        recons_oneseed_s4_2[mc]=sols4_2[:, :, pixok]
        recons_oneseed_bi_2[mc]=solbi_2[:, :, pixok]
    print(1)

    m_recons_oneseed_s4_1 = np.mean(recons_oneseed_s4_1, axis=0)
    m_recons_oneseed_bi_1 = np.mean(recons_oneseed_bi_1, axis=0)
    m_recons_oneseed_s4_2 = np.mean(recons_oneseed_s4_2, axis=0)
    m_recons_oneseed_bi_2 = np.mean(recons_oneseed_bi_2, axis=0)
    print(2)
    maps_s4_1[i]=m_recons_oneseed_s4_1
    maps_bi_1[i]=m_recons_oneseed_bi_1
    maps_s4_2[i]=m_recons_oneseed_s4_2
    maps_bi_2[i]=m_recons_oneseed_bi_2
    print(3)
    allmaps_to_separate_s4_1[i]=maps_to_separate_s4_1[:, :, pixok].copy()
    allmaps_to_separate_bi_1[i]=maps_to_separate_bi_1[:, :, pixok].copy()
    allmaps_to_separate_s4_2[i]=maps_to_separate_s4_2[:, :, pixok].copy()
    allmaps_to_separate_bi_2[i]=maps_to_separate_bi_2[:, :, pixok].copy()

truebeta=[1.44, 1.64, 265]

mydict = {'maps_s4_1':maps_s4_1,
          'maps_bi_1':maps_bi_1,
          'maps_s4_2':maps_s4_2,
          'maps_bi_2':maps_bi_2,
          'param_s4':param_s4,
          'param_bi':param_bi,
          'allmaps_to_separate_s4_1':allmaps_to_separate_s4_1,
          'allmaps_to_separate_bi_1':allmaps_to_separate_bi_1,
          'allmaps_to_separate_s4_2':allmaps_to_separate_s4_2,
          'allmaps_to_separate_bi_2':allmaps_to_separate_bi_2,
          'N':N,
          'nu0':145,
          'truebeta':truebeta}

CC=1
if CC == 1:
    path='/pbs/home/m/mregnier/sps1/QUBIC+/results'
else:
    path=''
output = open(path+'/compmaps_aftercompsep_{}reals.pkl'.format(N), 'wb')
pickle.dump(mydict, output)
output.close()
