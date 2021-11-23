import numpy as np
import healpy as hp 
import fgbuster
from fgbuster import CMB, Dust, Synchrotron, AnalyticComponent, ModifiedBlackBody
import pysm3
import pysm3.units as u
import healpy as hp
import numpy as np
import warnings
from fgbuster import AnalyticComponent
from sympy import Heaviside
from fgbuster import AnalyticComponent
from sympy import Heaviside
from scipy import constants
from astropy.cosmology import Planck15

import qubic

warnings.filterwarnings("ignore")






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

def qubicify(config, qp_nsubs, qp_effective_fraction, suboptimality=None):
    nbands = np.sum(qp_nsubs)
    if suboptimality is None:
        suboptimality = np.ones(len(qp_nsubs)).astype(bool)
    qp_config = config.copy()
    for k in qp_config.keys():
        qp_config[k]=[]
    qp_config['nbands'] = nbands
    qp_config['fsky'] = config['fsky']
    qp_config['ntubes'] = config['ntubes']
    qp_config['nyears'] = config['nyears']
    qp_config['initial_band'] = []
    
    for i in range(len(config['frequency'])):
        newedges = np.linspace(config['edges'][i][0], config['edges'][i][1], qp_nsubs[i]+1)
        newfreqs = (newedges[0:-1]+newedges[1:])/2
        newbandwidth = newedges[1:] - newedges[0:-1]
        newdnu_nu = newbandwidth / newfreqs
        newfwhm = config['fwhm'][i] * config['frequency'][i]/newfreqs
        scalefactor_noise = np.sqrt(qp_nsubs[i]) / qp_effective_fraction[i]
        print(suboptimality[i])
        if suboptimality[i]:
            scalefactor_noise *= fct_subopt(config['frequency'][i])
        newdepth_p = config['depth_p'][i] * np.ones(qp_nsubs[i]) * scalefactor_noise
        newdepth_i = config['depth_i'][i] * np.ones(qp_nsubs[i]) * scalefactor_noise
        newdepth_e = config['depth_e'][i] * np.ones(qp_nsubs[i]) * scalefactor_noise
        newdepth_b = config['depth_b'][i] * np.ones(qp_nsubs[i]) * scalefactor_noise
        newell_min = np.ones(qp_nsubs[i]) * config['ell_min'][i]
        newnside = np.ones(qp_nsubs[i]) * config['nside'][i]
        neweffective_fraction = np.ones(qp_nsubs[i]) * qp_effective_fraction[i]
        initial_band = np.ones(qp_nsubs[i]) * config['frequency'][i]
        
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
                qp_config['edges'].append(newedges[k])
                qp_config['effective_fraction'].append(neweffective_fraction[k])
                qp_config['initial_band'].append(initial_band[k])
    
    fields = ['frequency', 'depth_p', 'depth_i', 'depth_e', 'depth_b', 'fwhm', 'bandwidth', 
              'dnu_nu', 'ell_min', 'nside', 'edges', 'effective_fraction', 'initial_band']
    for j in range(len(fields)):
        qp_config[fields[j]] = np.array(qp_config[fields[j]])
        
    return qp_config


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

def get_component_maps(components, ref_freqs, nside, fsky, center_radec=[0., -57.]):
    maps = []
    mask = get_coverage(fsky, nside, center_radec=center_radec)
    okpix = mask == 1
    for c,f in zip(components, ref_freqs):
        print('Doing: '+c)
        thesky = pysm3.Sky(nside=nside, preset_strings=[c], output_unit="uK_CMB")
        themaps = np.zeros((4, 12*nside**2))     # four are I, Q, U and P
        themaps[0:3,:] = thesky.get_emission(f * u.GHz)                  #.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(f*u.GHz))
        themaps[3,:] = np.sqrt(themaps[1,:]**2 + themaps[2,:]**2)
        themaps[:, ~okpix] = hp.UNSEEN
        maps.append(themaps)
    return maps


def double_beta_dust_FGB_Model():

    H_OVER_K = constants.h * 1e9 / constants.k
    # Conversion factor at frequency nu
    K_RJ2K_CMB = ('(expm1(h_over_k * nu / Tcmb)**2'
                  '/ (exp(h_over_k * nu / Tcmb) * (h_over_k * nu / Tcmb)**2))')
    K_RJ2K_CMB = K_RJ2K_CMB.replace('Tcmb', str(Planck15.Tcmb(0).value))
    K_RJ2K_CMB = K_RJ2K_CMB.replace('h_over_k', str(H_OVER_K))
    K_RJ2K_CMB_NU0 = K_RJ2K_CMB + ' / ' + K_RJ2K_CMB.replace('nu', 'nu0')

    analytic_expr1 = ('(exp(nu0 / temp * h_over_k) -1)'
                     '/ (exp(nu / temp * h_over_k) - 1)'
                     '* (nu / nu0)**(1 + beta_d0)   * (nu0 / nubreak)**(beta_d0-beta_d1) * '+K_RJ2K_CMB_NU0 + '* (1-heaviside(nu-nubreak,0.5))')

    analytic_expr2 = ('(exp(nu0 / temp * h_over_k) -1)'
                     '/ (exp(nu / temp * h_over_k) - 1)'
                     '* (nu / nu0)**(1 + beta_d1) * '+K_RJ2K_CMB_NU0 + '* heaviside(nu-nubreak,0.5)')
    analytic_expr = analytic_expr1 + ' + ' + analytic_expr2

    return analytic_expr





