# +
from qubic import camb_interface as qc
import healpy as hp
import numpy as np
import qubicplus
import pysm3
import pysm3.units as u
from pysm3 import utils
from qubic import camb_interface as qc
from pylab import *
from qubic import NamasterLib as nam
import os
import random as rd
import string
import qubic
from importlib import reload
from fgbuster.component_model import (CMB, Dust, Dust_2b, Synchrotron, AnalyticComponent)
from scipy import constants
import fgbuster
import warnings
warnings.filterwarnings("ignore")


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


def get_edges(nus, bandwidth):
    edges=np.zeros((len(nus), 2))
    dnu_nu=bandwidth/nus
    edges_max=nus * (1. + dnu_nu/2)
    edges_min=nus * (1. - dnu_nu/2)
    for i in range(len(nus)):
        edges[i, 0]=edges_min[i]
        edges[i, 1]=edges_max[i]
    return edges


#take into account BI sup-opt (noise freq. correlations) at any given frequency
def fct_subopt(nus):
    subnus = [150., 220]
    subval = [1.4, 1.2]
    fct_subopt = np.poly1d(np.polyfit(subnus, subval, 1))
    return fct_subopt(nus)


#create a BI version of another experiment, eg. CMB-s4 (config=s4_config)
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

                qp_config['effective_fraction'].append(neweffective_fraction[k])
                qp_config['initial_band'].append(initial_band[k])
        edges=get_edges(np.array(qp_config['frequency']), np.array(qp_config['bandwidth']))
        qp_config['edges']=edges.copy()
        #for k in range(qp_nsubs[i]+1):
        #    if qp_effective_fraction[i] != 0:
        #        qp_config['edges'].append(newedges[k])
    fields = ['frequency', 'depth_p', 'depth_i', 'depth_e', 'depth_b', 'fwhm', 'bandwidth',
              'dnu_nu', 'ell_min', 'nside', 'edges', 'effective_fraction', 'initial_band']
    for j in range(len(fields)):
        qp_config[fields[j]] = np.array(qp_config[fields[j]])

    return qp_config


def plot_s4_bi_config_sensitivity(s4_config, qp_config):
    #check sensitivities and freq. bands
    s4_freq = s4_config['frequency']
    bi_freq = qp_config['frequency']
    
    figure(figsize=(16, 5))
    subplot(1, 2, 1)
    title('Depth I')
    errorbar(s4_freq, s4_config['depth_i'], xerr = s4_config['bandwidth']/2, fmt='ro', label='S4')
    errorbar(bi_freq, qp_config['depth_i'], xerr = qp_config['bandwidth']/2, fmt='bo', label='BI')
    xlabel('Freq. [GHz]')
    ylabel(r'Depth [$\mu K \times$arcmin]')
    legend(loc='best')

    subplot(1, 2, 2)
    title('Depth P')
    errorbar(s4_freq, s4_config['depth_p'] ,xerr = s4_config['bandwidth']/2, fmt='ro', label='S4')
    errorbar(bi_freq, qp_config['depth_p'], xerr = qp_config['bandwidth']/2, fmt='bo', label='BI')
    xlabel('Freq. [GHz]')
    ylabel(r'Depth [$\mu K \times$arcmin]')
    legend(loc='best')
    return

def check_s4_bi_noise_proportion(s4_config, qp_config):
    #check proportion btw S4 and BI sensitivity level

    #for each band take 1. depth of S4 central freq, 2. depth of BI at the same freq 3.sub-opt factor at that freq
    #4. check that BI sens is sqrt(5)*sub_opt wrt S4 sens

    #take only freq above 50 GHz
    freq_set = s4_config['frequency'][3:]
    print('In this sub-set of frequencies: ', freq_set)
    #sub-opt factor at s4 freq
    sub_opt_values = [fct_subopt(freq_val) for freq_val in freq_set]

    #s4_depts i and p
    s4_dep_i = s4_config['depth_i'][3:]
    s4_dep_p = s4_config['depth_p'][3:]

    #select subset of BI sens, by taking only sens at freq=S4_freqs

    ind_vec = []
    for freq_val in freq_set:
        indx = np.where(np.round(qp_config['frequency'][3:],3)==np.round(freq_val,3))[0][0]
        ind_vec.append(indx)
    
    
    bi_dep_i = [ np.round(qp_config['depth_i'][3:][i],3) for i in ind_vec ]
    bi_dep_p = [ np.round(qp_config['depth_p'][3:][i],3) for i in ind_vec ]

    expected_bi_dep_i = np.round(s4_dep_i*np.sqrt(5)*sub_opt_values, 3)
    expected_bi_dep_p = np.round(s4_dep_p*np.sqrt(5)*sub_opt_values, 3)

    #print if the BI sensitivities are correctly rescaled or not
    if ((bi_dep_i==expected_bi_dep_i).all()==True and (bi_dep_p==expected_bi_dep_p).all()==True):
        print('The BI sensitivities are correctly rescaled in I and P')
    else:
        print('ERROR: The BI sensitivities are uncorrectly rescaled in I and/or P. Check you code!')
        return
    
# COMPSEP RELATED FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_maps_for_namaster_QU(comp, nside):
    '''
    This function take maps with shape (N_comp, QU, npix) and return cmb maps with shape (IQU x npix) where
    I component is zero. To be used when you apply comp sep over QU only.
    '''
    new_comp=np.zeros((3, 12*nside**2))
    new_comp[1:]=comp[0].copy() #here it takes only the CMB!!!
    return new_comp


def get_comp_for_fgb(nu0, model, fix_temp, bw=1., x0=[], fixsync=True):
    '''
    This function returns a list 'comp' containing the fgb components
    to be passed as input of the component separation function
    INPUT: ref. freq, dust_model, fix_temp=None if T_d is fixed,
    break_steepness for the 2-beta dust model, initial guess x0, fixsync=True if the synchrotron has no parameter
    '''
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
        comp.append(fgbuster.component_model.Synchrotron(nu0=nu0, beta_pl=-3))
    else:
        comp.append(fgbuster.component_model.Synchrotron(nu0=nu0))
        comp[2].defaults=[-3]

    return comp


def get_comp_from_MixingMatrix(r, comp, instr, data, covmap, noise, nside):
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
    '''
    This function defines and returns the: frequency, depths, fwhm
    of the instrument used in the FGB component separation.
    The instrument can be: a full S4 configuration, full S4-BI configuration or an hybrid configuration
    '''
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
    '''
    This function is used in case of "weighted_comp_sep" with FGB
    '''
    npix=12*nside**2
    ind=coverage > 0

    noise_cov = np.ones(((n_freq, 3, npix)))

    for i in range(n_freq):
        noise_cov[i, 0] = np.ones(npix)*depths_i[i]**2
        noise_cov[i, 1] = np.ones(npix)*depths_p[i]**2
        noise_cov[i, 2] = np.ones(npix)*depths_p[i]**2

    noise_cov[:, :, ~ind]=hp.UNSEEN

    return noise_cov
