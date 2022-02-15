import numpy as np
import pysm3
import pysm3.units as u
from pysm3 import utils
import healpy as hp
import fgbuster
import matplotlib.pyplot as plt
import os
import qubic
center = qubic.equ2gal(-30, -30)
import warnings
import qubicplus
import pickle
import sys
from qubic import NamasterLib as nam
warnings.filterwarnings("ignore")
print(fgbuster.__path__)


def get_coverage(fsky, nside, center_radec=[-30, -30]):
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


### QUBIC Sub-optimality : values from Louise Mousset's PhD thesis
def fct_subopt(nus):
    subnus = [150., 220]
    subval = [1.4, 1.2]
    fct_subopt = np.poly1d(np.polyfit(subnus, subval, 1))
    return fct_subopt(nus)

subnus = [150., 220]
subval = [1.4, 1.2]


def qubicify(config, qp_nsubs, qp_effective_fraction):
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
        scalefactor_noise = np.sqrt(qp_nsubs[i]) * fct_subopt(config['frequency'][i])# / qp_effective_fraction[i]
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
                qp_config['depth_p'].append(newdepth_p[k])
                qp_config['depth_i'].append(newdepth_i[k])
                qp_config['depth_e'].append(newdepth_e[k])
                qp_config['depth_b'].append(newdepth_b[k])
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
    fields = ['frequency', 'depth_p', 'depth_i', 'depth_e', 'depth_b', 'fwhm', 'bandwidth',
              'dnu_nu', 'ell_min', 'nside', 'edges', 'effective_fraction', 'initial_band']
    for j in range(len(fields)):
        qp_config[fields[j]] = np.array(qp_config[fields[j]])

    return qp_config



qp_nsub = np.array([1, 1, 1, 5, 5, 5, 5, 5, 5])
qp_effective_fraction = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
qp_config = qubicify(s4_config, qp_nsub, qp_effective_fraction)

#Corrected depths
qp_config['depth_p'][:3] = s4_config['depth_p'][:3]
qp_config['depth_i'][:3] = s4_config['depth_i'][:3]
def get_maps_for_namaster_QU(comp, nside):

    '''

    This function take maps with shape (QU x npix) and return maps with shape (IQU x npix) where
    I component is zero. It take place when you apply comp sep over QU only.

    '''

    new_comp=np.zeros((3, 12*nside**2))
    new_comp[1:]=comp[0].copy()
    return new_comp
def get_comp_for_fgb(nu0, model, fix_temp, bw=1., x0=[], fixsync=True):
    comp=[fgbuster.component_model.CMB()]
    if model == '1b':
        if fix_temp is not None :
            comp.append(fgbuster.component_model.Dust(nu0=nu0, temp=fix_temp))
            comp[1].defaults=x0
        else:
            comp.append(fgbuster.component_model.Dust(nu0=nu0))
            comp[1].defaults=x0
    elif model == '2b':
        if fix_temp is not None :
            comp.append(fgbuster.component_model.Dust_2b(nu0=nu0, temp=fix_temp, break_width=bw))
            comp[1].defaults=x0
        else:
            comp.append(fgbuster.component_model.Dust_2b(nu0=nu0, break_width=bw))
            comp[1].defaults=x0
    else:
        raise TypeError('Not the good model')

    if fixsync:
        comp.append(fgbuster.component_model.Synchrotron(nu0=145, beta_pl=-3))
    else:
        comp.append(fgbuster.component_model.Synchrotron(nu0=145))
        comp[2].defaults=[-3]

    return comp
def get_comp_from_MixingMatrix(r, comp, instr, data, delta_beta, covmap, model, noise, nside):

    """

    This function estimate components from MixingMatrix of fgbuster with estimated parameters

    """

    pixok=covmap>0

    # Define Mixing Matrix from FGB
    A=fgbuster.mixingmatrix.MixingMatrix(*comp)
    A_ev=A.evaluator(np.array(instr.frequency))
    A_maxL=A_ev(np.array(r.x))

    if noise:
        invN = np.diag(hp.nside2resol(nside, arcmin=True) / (instr.depth_p))**2
        maps_separe=fgbuster.algebra.Wd(A_maxL, data.T, invN=invN).T
    else:
        maps_separe=fgbuster.algebra.Wd(A_maxL, data.T).T

    maps_separe[:, :, ~pixok]=hp.UNSEEN

    return maps_separe
def get_good_config(config, prop):
    config1=config[0]
    config2=config[1]
    nus=np.array(list(config1['frequency'])+list(config2['frequency']))
    depth1_i=config1['depth_i']/(np.sqrt(prop[0]))
    depth1_p=config1['depth_p']/(np.sqrt(prop[0]))
    depth2_i=config2['depth_i']/(np.sqrt(prop[1]))
    depth2_p=config2['depth_p']/(np.sqrt(prop[1]))

    depth_i=np.array(list(depth1_i)+list(depth2_i))
    depth_p=np.array(list(depth1_p)+list(depth2_p))
    fwhm=np.zeros(42)

    if prop[0] == 1 :
        depth_i=config1['depth_i']
        depth_p=config1['depth_p']
        nus=config1['frequency']
        fwhm=np.zeros(9)
    elif prop[1] == 1:
        depth_i=config2['depth_i']
        depth_p=config2['depth_p']
        nus=config2['frequency']
        fwhm=np.zeros(33)
    else:
        pass



    return nus, depth_i, depth_p, fwhm
def get_cov_for_weighted(n_freq, depths_i, depths_p, coverage, nside=256):
    npix=12*nside**2
    ind=coverage > 0

    noise_cov = np.ones(((n_freq, 3, npix)))

    for i in range(n_freq):
        noise_cov[i, 0] = np.ones(npix)*depths_i[i]**2
        noise_cov[i, 1] = np.ones(npix)*depths_p[i]**2
        noise_cov[i, 2] = np.ones(npix)*depths_p[i]**2

    noise_cov[:, :, ~ind]=hp.UNSEEN

    return noise_cov
def ParamCompSep(config, prop, nside, skyconfig, model, noise=True, delta_beta=0.05, fix_temp=None, nubreak=260, x0=[], break_width=1., fixsync=True):

    covmap = get_coverage(0.03, nside)
    pixok = covmap>0

    if prop[0] == 1 :
        conf=config[0]
        inputs, inputs_noiseless, _ = qubicplus.BImaps(skyconfig, conf, nside=nside).getskymaps(
                                      same_resol=0,
                                      verbose=False,
                                      coverage=covmap,
                                      noise=True,
                                      beta=[1.54-delta_beta, 1.54+delta_beta, nubreak, break_width],
                                      fix_temp=fix_temp)

    elif prop[1] == 1:
        conf=config[1]
        inputs, inputs_noiseless, _ = qubicplus.BImaps(skyconfig, conf, nside=nside).getskymaps(
                                      same_resol=0,
                                      verbose=False,
                                      coverage=covmap,
                                      noise=True,
                                      beta=[1.54-delta_beta, 1.54+delta_beta, nubreak, break_width],
                                      fix_temp=fix_temp)

    else:
        inputs, inputs_noiseless, _ = qubicplus.combinedmaps(skyconfig, config, nside=nside, prop=prop).getskymaps(
                                      same_resol=0,
                                      verbose=False,
                                      coverage=covmap,
                                      noise=True,
                                      beta=[1.54-delta_beta, 1.54+delta_beta, nubreak, break_width],
                                      fix_temp=fix_temp)




    ###### Preparation for COMPSEP ######
    nus, depth_i, depth_p, fwhm = get_good_config(config, prop)

    if len(nus) == 9:
        name='CMBS4'
        n_freq=9
    elif len(nus)==33:
        name='Qubic+'
        n_freq=33
    else:
        name='CMBS4BI'
        n_freq=42

    print('Define instrument')
    print()
    instr = fgbuster.get_instrument(name)
    instr.frequency=nus
    instr.depth_i=depth_i
    instr.depth_p=depth_p
    instr.fwhm=fwhm

    print('Define components')
    print()
    comp=get_comp_for_fgb(nu0=145, model=model, fix_temp=fix_temp, x0=x0, bw=break_width, fixsync=fixsync)

    print('##### COMP SEP #####')
    print()

    options={'maxiter': 10000}
    tol=1e-18

    r=fgbuster.basic_comp_sep(comp, instr, inputs[:, 1:, pixok], tol=tol, options=options)#,
                                 #bounds=bnds)

    print('Done! \n')
    print(r.x)

    components=get_comp_from_MixingMatrix(r, comp, instr, inputs[:, 1:, :], delta_beta, covmap, model, noise, nside)
    components_noiseless=get_comp_from_MixingMatrix(r, comp, instr, inputs_noiseless[:, 1:, :], delta_beta, covmap, model, False, nside)
    return components, components_noiseless, r

nside=256
nb_param=1
if int(sys.argv[4]) == 1:
    T=20
    name_T='_fixtemp'
else:
    T=None
    name_T=''
    nb_param+=1

if int(sys.argv[5]) == 1:
    fix_sync=True
    name_s='_fixsync'
else:
    fix_sync=False
    name_s=''
    nb_param+=1
print('T = {}'.format(T))
print('Fix sync -> {}'.format(fix_sync))
print('# of params -> {}'.format(nb_param))
fix_temp=T
fix_sync=fix_sync
noise=True


N=int(sys.argv[1])           # Number of iterations
ite=int(sys.argv[2])         # To save
nubreak=int(sys.argv[3])     # True value of nubreak
#nuinf=int(sys.argv[4])       # Lower bounds for nubreak initialization
#nusup=int(sys.argv[5])       # Upper bounds for nubreak initialization


model='1b'
dust_config='d02b'
bw=0.3

maskpix = np.zeros(12*nside**2)
covmap = get_coverage(0.03, nside)
pixok = covmap > 0
maskpix[pixok] = 1
Namaster = nam.Namaster(maskpix, lmin=40, lmax=2*nside-1, delta_ell=30)
props=np.linspace(0, 1, 2)
db=np.linspace(0.0, 0.05, 6)


param_comb = np.zeros(((((2, N, len(props), len(db), nb_param)))))
rms_est_cmb = np.zeros((((N, len(props), len(db), 1))))
rms_est_dust = np.zeros((((N, len(props), len(db), 1))))
cl=np.zeros(((((N, len(props), len(db), 16, 4)))))
tabseed=np.zeros(N)

for j in range(N):
    seed=np.random.randint(1000000)
    tabseed[j]=seed
    print("seed is {}".format(seed))
    #nubreak_init=np.random.randint(nuinf, nusup)
    for i in range(len(props)):
        for k in range(len(db)):
            if fix_temp is not None:
                x0=[1.54]
            else:
                x0=[1.54, 20]
            print('###################')
            print(' BI fration : {}%'.format(props[i]*100))
            print(' Delta beta : {}'.format(db[k]))
            print(' Init : ', x0)
            print('###################')
            BIprop=props[i]
            S4prop=1-BIprop
            frac=[S4prop, BIprop]

            CMB, CMB_noiseless, _ = qubicplus.BImaps({'cmb':seed}, s4_config, nside=nside).getskymaps(
                                          same_resol=0,
                                          verbose=False,
                                          coverage=covmap,
                                          noise=True,
                                          beta=[1.54-db[k], 1.54+db[k], nubreak, bw],
                                          fix_temp=fix_temp)

            DUST, DUST_noiseless, _ = qubicplus.BImaps({'dust':'d02b'}, s4_config, nside=nside).getskymaps(
                                          same_resol=0,
                                          verbose=False,
                                          coverage=covmap,
                                          noise=True,
                                          beta=[1.54-db[k], 1.54+db[k], nubreak, bw],
                                          fix_temp=fix_temp)

            comp1, comp1_noiseless, r_comb_1 = ParamCompSep([s4_config, qp_config],
                                    prop=frac,
                                    skyconfig={'cmb':seed, 'dust':'d02b', 'synchrotron':'s0'},
                                    model=model,
                                    noise=noise,
                                    delta_beta=db[k],
                                    fix_temp=fix_temp,
                                    nside=nside,
                                    nubreak=nubreak,
                                    x0=x0,
                                    break_width=bw,
                                    fixsync=fix_sync)

            comp2, comp2_noiseless, r_comb_2 = ParamCompSep([s4_config, qp_config],
                                    prop=frac,
                                    skyconfig={'cmb':seed, 'dust':'d02b', 'synchrotron':'s0'},
                                    model=model,
                                    noise=noise,
                                    delta_beta=db[k],
                                    fix_temp=fix_temp,
                                    nside=nside,
                                    nubreak=nubreak,
                                    x0=x0,
                                    break_width=bw,
                                    fixsync=fix_sync)

            param_comb[0, j, i, k]=r_comb_1.x
            param_comb[1, j, i, k]=r_comb_2.x

            # Store the RMS
            rms_est_cmb[j, i, k]=np.std(comp2_noiseless[0, 0, pixok]-CMB_noiseless[5, 1, pixok])
            rms_est_dust[j, i, k]=np.std(comp2_noiseless[1, 0, pixok]-DUST_noiseless[5, 1, pixok])
            print('')
            print('RMS of CMB residuals -> {:.5f}'.format(rms_est_cmb[j, i, k, 0]))
            print('RMS of DUST residuals -> {:.5f}'.format(rms_est_dust[j, i, k, 0]))
            print('')

            # Place QU stokes parameters and 0 for I
            maps1=get_maps_for_namaster_QU(comp1, nside=256)
            maps2=get_maps_for_namaster_QU(comp2, nside=256)

            # To be sure that not seen pixels are 0 and not hp.UNSEEN
            maps1[:, ~pixok]=0
            maps2[:, ~pixok]=0

            w=None
            leff, cl[j, i, k], _ = Namaster.get_spectra(maps1, map2=maps2,
                                         purify_e=False,
                                         purify_b=True,
                                         w=w,
                                         verbose=False,
                                         beam_correction=None,
                                         pixwin_correction=False)

            print(cl[j, i, k, :, 2])


# Save cmb and dust maps and cls in pkl file
pickle.dump([leff, cl, param_comb, rms_est_cmb, rms_est_dust, tabseed, db, props, bw, sys.argv], open('/pbs/home/m/mregnier/sps1/QUBIC+/d0/cls/results/cls_QU{}{}_truenub{:.0f}_{}reals_{}.pkl'.format(name_T, name_s, nubreak, N, ite), "wb"))
#pickle.dump([cls4, leff, r_2b_s4_1.x, r_2b_s4_2.x, sys.argv], open('/pbs/home/m/mregnier/sps1/QUBIC+/results/cls_NSIDE_PATCH{}_db{}_{}reals_{}.pkl'.format(NSIDE_PATCH, db, N, ite), "wb"))
