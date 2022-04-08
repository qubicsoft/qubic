import numpy as np
import pickle
import sys
import qubicplus
import pysm3
import pysm3.units as u
from pysm3 import utils
import numpy as np
from qubic import camb_interface as qc
import healpy as hp
import matplotlib.pyplot as plt
from qubic import NamasterLib as nam
import os
import scipy
import random as rd
from getdist import plots, MCSamples
import getdist
import string
from qubic import mcmc
import qubic
from importlib import reload
import pickle
from scipy import constants
import fgbuster
import warnings
from scipy.optimize import curve_fit
from getdist import densities

def _rj2cmb(freqs):
    return (np.ones_like(freqs) * u.K_RJ).to(
        u.K_CMB, equivalencies=u.cmb_equivalencies(freqs * u.GHz)).value


def _cmb2rj(freqs):
    return (np.ones_like(freqs) * u.K_CMB).to(
        u.K_RJ, equivalencies=u.cmb_equivalencies(freqs * u.GHz)).value

def _rj2jysr(freqs):
    return (np.ones_like(freqs) * u.K_RJ).to(
        u.Jy / u.sr, equivalencies=u.cmb_equivalencies(freqs * u.GHz)).value


def _jysr2rj(freqs):
    return (np.ones_like(freqs) * u.Jy / u.sr).to(
        u.K_RJ, equivalencies=u.cmb_equivalencies(freqs * u.GHz)).value


def _cmb2jysr(freqs):
    return (np.ones_like(freqs) * u.K_CMB).to(
        u.Jy / u.sr, equivalencies=u.cmb_equivalencies(freqs * u.GHz)).value


def _jysr2cmb(freqs):
    return (np.ones_like(freqs) * u.Jy / u.sr).to(
        u.K_CMB, equivalencies=u.cmb_equivalencies(freqs * u.GHz)).value

print(fgbuster.__path__)
warnings.filterwarnings("ignore")

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
def get_edges(nus, bandwidth):
    edges=np.zeros((len(nus), 2))
    dnu_nu=bandwidth/nus
    edges_max=nus * (1. + dnu_nu/2)
    edges_min=nus * (1. - dnu_nu/2)
    for i in range(len(nus)):
        edges[i, 0]=edges_min[i]
        edges[i, 1]=edges_max[i]
    return edges
def get_cov_for_weighted(n_freq, depths_i, depths_p, coverage, nside=256):
    npix=12*nside**2
    ind=coverage > 0

    noise_cov = np.ones(((n_freq, 3, npix)))

    for i in range(n_freq):
        noise_cov[i, 0] = np.ones(npix)*depths_i[i]**2
        noise_cov[i, 1] = np.ones(npix)*depths_p[i]**2
        noise_cov[i, 2] = np.ones(npix)*depths_p[i]**2

    return noise_cov
def fct_subopt(nus):
    subnus = [150., 220]
    subval = [1.4, 1.2]
    fct_subopt = np.poly1d(np.polyfit(subnus, subval, 1))
    return fct_subopt(nus)
def qubicify(config, qp_nsub, qp_effective_fraction):
    nbands = np.sum(qp_nsub)
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
        newedges = np.linspace(config['edges'][i][0], config['edges'][i][-1], qp_nsub[i]+1)
        #print(newedges)
        newfreqs = (newedges[0:-1]+newedges[1:])/2
        newbandwidth = newedges[1:] - newedges[0:-1]
        newdnu_nu = newbandwidth / newfreqs
        newfwhm = config['fwhm'][i] * config['frequency'][i]/newfreqs
        scalefactor_noise = np.sqrt(qp_nsub[i])# * fct_subopt(config['frequency'][i])# / qp_effective_fraction[i]
        newdepth_p = config['depth_p'][i] * np.ones(qp_nsub[i]) * scalefactor_noise
        newdepth_i = config['depth_i'][i] * np.ones(qp_nsub[i]) * scalefactor_noise
        newdepth_e = config['depth_e'][i] * np.ones(qp_nsub[i]) * scalefactor_noise
        newdepth_b = config['depth_b'][i] * np.ones(qp_nsub[i]) * scalefactor_noise
        newell_min = np.ones(qp_nsub[i]) * config['ell_min'][i]
        newnside = np.ones(qp_nsub[i]) * config['nside'][i]
        neweffective_fraction = np.ones(qp_nsub[i]) * qp_effective_fraction[i]
        initial_band = np.ones(qp_nsub[i]) * config['frequency'][i]

        for k in range(qp_nsub[i]):
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
def get_maps_for_namaster_QU(comp, nside):

    '''

    This function take maps with shape (QU x npix) and return maps with shape (IQU x npix) where
    I component is zero. It take place when you apply comp sep over QU only.

    '''

    new_comp=np.zeros((3, 12*nside**2))
    new_comp[1:]=comp[0].copy()
    return new_comp
def get_comp_for_fgb(nu0, model, fix_temp, bw=0.3, x0=[], fixsync=True):
    comp=[fgbuster.component_model.CMB()]
    if model == 'd0':
        if fix_temp is not None :
            comp.append(fgbuster.component_model.Dust(nu0=nu0, temp=fix_temp))
            comp[1].defaults=x0
        else:
            comp.append(fgbuster.component_model.Dust(nu0=nu0))
            comp[1].defaults=x0
    elif model == 'd02b':
        if fix_temp is not None :
            comp.append(fgbuster.component_model.Dust_2b(nu0=nu0, temp=fix_temp, break_width=bw))
            comp[1].defaults=x0
        else:
            comp.append(fgbuster.component_model.Dust_2b(nu0=nu0, break_width=bw))
            comp[1].defaults=x0
    elif model == 'running':
        if fix_temp is not None :
            comp.append(fgbuster.component_model.Dust_running(nu0=nu0, temp=fix_temp))
            comp[1].defaults=x0
        else:
            comp.append(fgbuster.component_model.Dust_running(nu0=nu0))
            comp[1].defaults=x0
    else:
        raise TypeError('Not the good model')

    if fixsync:
        comp.append(fgbuster.component_model.Synchrotron(nu0=145, beta_pl=-3))
    else:
        comp.append(fgbuster.component_model.Synchrotron(nu0=145))
        comp[2].defaults=[-3]

    return comp
def get_comp_from_MixingMatrix(r, comp, instr, data, covmap, noise, nside, conf):

    """

    This function estimate components from MixingMatrix of fgbuster with estimated parameters

    """

    pixok=covmap>0

    instr.frequency=conf['frequency']
    # Define Mixing Matrix from FGB
    A=fgbuster.mixingmatrix.MixingMatrix(*comp)
    A_ev=A.evaluator(np.array(instr.frequency))
    A_maxL=A_ev(np.array(r))

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
def _get_model(Number):
    if Number==0:
        model='d0'
    elif Number==1:
        model='d02b'
    elif Number==2:
        model='running'
    else:
        raise TypeErrror('Please, choose 0, 1 or 2 (for d0, d02b or running model)')
    return model
def _get_nb_params(model, fixsync):
    if model =='d0' and fixsync == 0: nb_param=2
    elif model =='d0' and fixsync == 1: nb_param=1
    elif model =='d02b' and fixsync == 0: nb_param=4
    elif model =='d02b' and fixsync == 1: nb_param=3
    elif model =='running' and fixsync == 0: nb_param=3
    elif model =='running' and fixsync == 1: nb_param=2
    else: raise TypeError('Choose the good model...')

    return nb_param
def _get_clBB(map1, map2, Namaster):
    w=None
    leff, cl, _ = Namaster.get_spectra(map1, map2=map2,
                                 purify_e=False,
                                 purify_b=True,
                                 w=w,
                                 verbose=False,
                                 beam_correction=None,
                                 pixwin_correction=False)

    # Apply a correction -> remove the first and last bins
    clBB=cl[:, 2]

    return leff, clBB
def _get_x0(model):

    if model=='d0':
        x0=[1.54]
    elif model=='running':
        x0=[0, 1.54]
    elif model=='d02b':
        x0=[1.54, 1.54, 200]
    else:
        raise TypeError('Please choose the good model...')
    return x0
def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
def gauss_fit(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma])
    return popt
def ana_likelihood(rv, leff, fakedata, errors, model, prior,
                   mylikelihood=mcmc.LogLikelihood, covariance_model_funct=None, otherp=None):
    ll = mylikelihood(xvals=leff, yvals=fakedata, errors=errors,
                            model = model, flatprior=prior, covariance_model_funct=covariance_model_funct)

    like = np.zeros_like(rv)
    for i in range(len(rv)):
        like[i] = np.exp(ll([rv[i]]))
        #print(rv[i],ll([rv[i]]),like[i])
    maxL = rv[like == np.max(like)]
    cumint = scipy.integrate.cumtrapz(like, x=rv)
    cumint = cumint / np.max(cumint)
    onesigma = np.interp(0.68, cumint, rv[1:])
    if otherp:
        other = np.interp(otherp, cumint, rv[1:])
        return like, cumint, onesigma, other, maxL
    else:
        return like, cumint, onesigma, maxL


def explore_like(leff, cl, errors, lmin, dl, cc, rv, otherp=None,
                 cov=None, plotlike=False, plotcls=False,
                 verbose=False, sample_variance=True, mytitle='', color=None, mylabel='',my_ylim=None):

#     print(lmin, dl, cc)
#     print(leff)
#     print(scl_noise[:,2])
    ### Create Namaster Object
    # Unfortunately we need to recalculate fsky for calculating sample variance
    nside = 256
    lmax = 355
    if cov is None:
        Namaster = nam.Namaster(None, lmin=lmin, lmax=lmax, delta_ell=dl)
        Namaster.fsky = 0.03
    else:
        okpix = cov > (np.max(cov) * float(cc))
        maskpix = np.zeros(12*nside**2)
        maskpix[okpix] = 1
        Namaster = nam.Namaster(maskpix, lmin=lmin, lmax=lmax, delta_ell=dl)
        #Namaster.fsky = 0.03

#     print('Fsky: {}'.format(Namaster.fsky))
    lbinned, b = Namaster.get_binning(nside)

    ### Bibnning CambLib
#     binned_camblib = qc.bin_camblib(Namaster, '../../scripts/QubicGeneralPaper2020/camblib.pickle',
#                                     nside, verbose=False)
    global_dir='/pbs/home/m/mregnier/sps1/QUBIC+/d0/cls'#os.getcwd()
    binned_camblib = qc.bin_camblib(Namaster, global_dir+'/camblib.pkl',
                                    nside, verbose=False)


    ### Redefine the function for getting binned Cls
    def myclth(ell,r):
        clth = qc.get_Dl_fromlib(ell, r, lib=binned_camblib, unlensed=True)[1]
        #print(clth)
        #ell, clth, unlensedCL = qc.get_camb_Dl(lmax=3*256, r=0)
        return clth[1:]
    allfakedata = myclth(leff, 0.)
    #lll, totDL, unlensedCL = qc.get_camb_Dl(lmax=3*256, r=0)
    ### And we need a fast one for BB only as well
    def myBBth(ell, r):
        clBB = qc.get_Dl_fromlib(ell, r, lib=binned_camblib, unlensed=True, specindex=2)[1]

        return clBB[1:]

    ### Fake data
    fakedata = cl.copy()#myBBth(leff, 0.)


    if sample_variance:
        covariance_model_funct = Namaster.knox_covariance
    else:
        covariance_model_funct = None
    if otherp is None:
        like, cumint, allrlim, maxL = ana_likelihood(rv, leff, fakedata,
                                            errors,
                                            myBBth, [[0,1]],
                                           covariance_model_funct=covariance_model_funct)
    else:
        like, cumint, allrlim, other, maxL = ana_likelihood(rv, leff, fakedata,
                                            errors,
                                            myBBth, [[0,1]],
                                           covariance_model_funct=covariance_model_funct, otherp=otherp)

    if otherp is None:
        return like, cumint, allrlim, maxL
    else:
        return like, cumint, allrlim, other, maxL
def confidence_interval(x,px):
    # return: list of (min, max, has_min, has_top) values
    #.        where has_min and has_top are True or False depending on whether lower and upper limit exists
    vals = [0.68, 0.95]
    d = densities.Density1D(x, px)
    lims = []
    szlim = []
    for i in range(len(vals)):
        limits = d.getLimits(vals[i])
        lims.append(limits)
        szlim.append((limits[1]-limits[0])/2)

    ### Change sigma to 2 sigma if this is an uppeer/lower limit
    if lims[1][2] or lims[1][3]:
        szlim[0] = szlim[1]
        szlim[1] = szlim[1]*2
    return lims, szlim
def get_like_onereals(ell, cl, tab, coverage, covbin):

    delta_ell=35
    covcut=0
    nr=20000
    rv=np.linspace(-0.01,0.01,nr)

    lmin=21

    #like=np.zeros((cl.shape[0], 10000))

    #rlim68_sup=np.zeros(cl.shape[0])
    #rlim95_sup=np.zeros(cl.shape[0])
    rlim68=np.zeros(cl.shape[0])
    rlim95=np.zeros(cl.shape[0])

    maxL=np.zeros(cl.shape[0])
    like=np.zeros(((cl.shape[0], nr)))

    mycl=cl[:, :, 2, 0]
    print('shape cls -> ', cl.shape)
    if covbin is not None:
        myerrcl=covbin[:, :, 2].copy()
    else:
        myerrcl=np.std(cl, axis=0)[:, 2, 0]

    for j in range(mycl.shape[0]):
        print(j)

        like[j], _, r68, r95, new_ml = explore_like(ell, mycl[j],
                   myerrcl, lmin, delta_ell, covcut, rv, cov=coverage, plotlike=False, plotcls=False,
                         verbose=False, sample_variance=False, otherp=0.95)

        #print(new_ml)
        #conf, sig = confidence_interval(rv,like[j])
        #rlim68_inf[j]=#conf[0][0]
        #rlim68[j]=r68#conf[0][1]
        #rlim95[j]=#conf[1][0]
        #rlim95_sup[j]=#conf[1][1]
        popt=gauss_fit(rv, like[j])
        print(popt)
        #argmaxL=np.argmax(like[j])
        maxL[j]=popt[2]
        rlim68[j]=r68
        rlim95[j]=r95#popt[3]

        #maxL[j,i]=ml

        print('maxL -> {:.10f}'.format(maxL[j]))
        #print('sigma(r) -> {:.10f}'.format(rlim68_sup[j]))
        print('sigma(r) -> {:.10f}'.format(rlim68[j]))#rlim68[j]-maxL[j]))
        print('95% Upper limits -> {:.10f}'.format(rlim95[j]))



    return maxL, rlim68, rlim95
def _get_fgb_tools(config, prop, iib, fixsync, fit):

    if fit == 0:

        if fixsync == 0:
            comp=[fgbuster.component_model.CMB(),
            fgbuster.component_model.Dust(nu0=100, temp=20),
            fgbuster.component_model.Synchrotron(nu0=100)]
            comp[1].defaults=[1.54]
            comp[2].defaults=[-3]
        else:
            comp=[fgbuster.component_model.CMB(),
            fgbuster.component_model.Dust(nu0=100, temp=20),
            fgbuster.component_model.Synchrotron(nu0=100, beta_pl=-3)]
            comp[1].defaults=[1.54]
    elif fit == 1 :
        if fixsync == 0:
            comp=[fgbuster.component_model.CMB(),
            fgbuster.component_model.Dust_2b(nu0=100, temp=20, break_width=0.3),
            fgbuster.component_model.Synchrotron(nu0=100)]
            comp[1].defaults=[1.54, 1.54, 150]
        else:
            comp=[fgbuster.component_model.CMB(),
            fgbuster.component_model.Dust_2b(nu0=100, temp=20, break_width=0.3),
            fgbuster.component_model.Synchrotron(nu0=100, beta_pl=-3)]
            comp[1].defaults=[1.54, 1.54, 150]



    #instr=fgbuster.get_instrument('INSTRUMENT')
    #if prop == 0 or prop == 1 :
    #    instr.frequency = config['frequency']
    #    instr.depth_i=config['depth_i']
    #    instr.depth_p=config['depth_p']
    #else:
    #    nus=np.array(list(config[0]['frequency'])+list(config[1]['frequency']))
    #    depth_i=np.array(list(config[0]['depth_i'])+list(config[1]['depth_i']))
    #    depth_p=np.array(list(config[0]['depth_p'])+list(config[1]['depth_p']))
    #    instr.frequency=nus
    #    instr.depth_i=depth_i
    #    instr.depth_p=depth_p

    #if iib > 1 :
    instr=get_instr(config, N_SAMPLE_BAND=100, prop=prop)


    return comp, instr
def get_comp_from_MixingMatrix(r, comp, instr, data, covmap, nside, nus):

    """

    This function estimate components from MixingMatrix of fgbuster with estimated parameters

    """

    pixok=covmap>0
    ind=np.where(pixok != 0)[0]
    print(ind)

    #instr.frequency=nus
    # Define Mixing Matrix from FGB
    A=fgbuster.mixingmatrix.MixingMatrix(*comp)
    A_ev=A.evaluator(nus)


    print(len(r))
    if len(r[0])==2:
        print('d0model for reconstruction')
        A_maxL=A_ev(np.array(r))
        invN = np.diag(hp.nside2resol(nside, arcmin=True) / (instr.depth_p))**2
        maps_separe=fgbuster.algebra.Wd(A_maxL, data.T, invN=invN).T

    else:
        maps_separe=np.zeros((len(comp), 2, 12*nside**2))
        invN = np.diag(hp.nside2resol(nside, arcmin=True) / (instr.depth_p))**2
        for i in range(len(ind)):
            A_maxL=A_ev(np.array(r)[:, ind[i]])
            maps_separe[:, :, ind[i]]=fgbuster.algebra.Wd(A_maxL, data[:, :, ind[i]].T, invN=invN).T

    maps_separe[:, :, ~pixok]=hp.UNSEEN

    return maps_separe
def get_maps_for_namaster_QU(comp, nside):

    '''

    This function take maps with shape (QU x npix) and return maps with shape (IQU x npix) where
    I component is zero. It take place when you apply comp sep over QU only.

    '''

    new_comp=np.zeros((3, 12*nside**2))
    new_comp[1:]=comp[0].copy()
    return new_comp
def _give_name_splitbands(A):
    name=''
    for i in range(len(A)):
        name+=str(A[i])
    return name
def _get_param(config, maps, N, Namaster, prop, iib, fixsync, fit):

    nside=256
    comp, instr=_get_fgb_tools(config, prop=prop, iib=iib, fixsync=fixsync, fit=fit)
    print(comp)
    covmap = get_coverage(0.03, nside)
    pixok = covmap>0
    ell_binned, _=Namaster.get_binning(256)
    cl = np.zeros((((N, 1, len(ell_binned), 4))))
    if fit == 0:
        nparam=1
        if fixsync == 0:
            nparam+=1
    elif fit == 1:
        nparam=3
        if fixsync == 0:
            nparam+=1
    param=np.zeros((2*N, nparam))
    j=0
    k=1

    for i in range(N):
        print(i)

        noise1=_get_noise(config, prop=prop)
        maps1_noisy=maps[:, :, :]+noise1[:, :, :].copy()
        noise2=_get_noise(config, prop=prop)
        maps2_noisy=maps[:, :, :]+noise2[:, :, :].copy()

        print('    ///// Components Separation')
        r1=fgbuster.separation_recipes.basic_comp_sep(comp, instr, maps1_noisy[:, 1:, pixok])
        r2=fgbuster.separation_recipes.basic_comp_sep(comp, instr, maps2_noisy[:, 1:, pixok])
        print('        -> ', r1.x)
        print('        -> ', r2.x)

        param[j]=r1.x
        param[k]=r2.x
        j+=2
        k+=2

        if prop==0 or prop ==1:
            nus=config['frequency']
        else:
            nus=np.array(list(config[0]['frequency'])+list(config[1]['frequency']))

        print('    ///// Reconstructed maps')
        components1=get_comp_from_MixingMatrix(r1.x, comp, instr, maps1_noisy[:, 1:, :], covmap, True,
        nside, nus=nus)
        components2=get_comp_from_MixingMatrix(r2.x, comp, instr, maps2_noisy[:, 1:, :], covmap, True,
        nside, nus=nus)

        new_comp1=get_maps_for_namaster_QU(components1, nside)
        new_comp2=get_maps_for_namaster_QU(components2, nside)

        print('    ///// Get Cls')
        print()
        w=None
        leff, cls, _ = Namaster.get_spectra(new_comp1, map2=new_comp2,
                                 purify_e=False,
                                 purify_b=True,
                                 w=w,
                                 verbose=False,
                                 beam_correction=None,
                                 pixwin_correction=False)

        cl[i, 0]=cls.copy()

    return leff, cl, param
def create_noisemaps(signoise, nus, nside, depth_i, depth_p, npix):
    #np.random.seed(None)
    N = np.zeros(((len(nus), 3, npix)))
    for ind_nu, nu in enumerate(nus):

        sig_i=signoise*depth_i[ind_nu]/(np.sqrt(hp.nside2pixarea(nside, degrees=True)) * 60)
        N[ind_nu, 0] = np.random.normal(0, sig_i, 12*nside**2)

        sig_p=signoise*depth_p[ind_nu]/(np.sqrt(hp.nside2pixarea(nside, degrees=True)) * 60)
        N[ind_nu, 1] = np.random.normal(0, sig_p, 12*nside**2)*np.sqrt(2)
        N[ind_nu, 2] = np.random.normal(0, sig_p, 12*nside**2)*np.sqrt(2)

    return N
def _get_maps_without_noise(config, db, nubreak, prop, r, iib):

    nside=256
    covmap = get_coverage(0.03, nside)
    pixok = covmap>0
    if db == 0:
        dust_config='d0'
    else:
        dust_config='d02b'

    skyconfig={'cmb':42, 'dust':dust_config, 'synchrotron':'s0'}

    print('Skyconfig is ', skyconfig)

    if prop == 0 :
        iib_instr=iib*5
        _, maps, _=qubicplus.BImaps(skyconfig, config, r=r, nside=nside).getskymaps(same_resol=0,
                                                                  verbose=False,
                                                                  coverage=covmap,
                                                                  iib=iib_instr,
                                                                  noise=True,
                                                                  signoise=1.,
                                                                  beta=[1.54-db, 1.54, nubreak, 0.3],
                                                                  fix_temp=20)
    elif prop == 1 :
        iib_instr=iib
        _, maps, _=qubicplus.BImaps(skyconfig, config, r=r, nside=nside).getskymaps(same_resol=0,
                                                                  verbose=False,
                                                                  coverage=covmap,
                                                                  iib=iib_instr,
                                                                  noise=True,
                                                                  signoise=1.,
                                                                  beta=[1.54-db, 1.54, nubreak, 0.3],
                                                                  fix_temp=20)
    else:
        iib_instr=iib
        frac=[1-prop, prop]
        _, maps, _=qubicplus.combinedmaps(skyconfig, config, nside=nside, r=r, prop=frac).getskymaps(
                                          same_resol=0,
                                          verbose=True,
                                          coverage=covmap,
                                          noise=True,
                                          beta=[1.54-db, 1.54, nubreak, 0.3],
                                          fix_temp=20,
                                          iib=iib_instr)

    return maps
def _get_noise(config, prop, nside_out):

    nside=256
    npix=12*nside**2

    np.random.seed(None)
    covmap = get_coverage(0.03, nside)
    pixok = covmap>0

    if prop == 0 or prop == 1:
        nus=config['frequency']
        N = np.zeros(((len(nus), 3, npix)))
        depth_i = config['depth_i']
        depth_p = config['depth_p']

    else:
        frac=[1-prop, prop]
        config1=config[0]
        config2=config[1]
        print(frac)
        nus=list(config1['frequency'])+list(config2['frequency'])
        N = np.zeros(((len(nus), 3, npix)))
        depth1_i=config1['depth_i']/np.sqrt(frac[0])
        depth1_p=config1['depth_p']/np.sqrt(frac[0])
        depth2_i=config2['depth_i']/np.sqrt(frac[1])
        depth2_p=config2['depth_p']/np.sqrt(frac[1])

        depth_i=np.array(list(depth1_i)+list(depth2_i))
        depth_p=np.array(list(depth1_p)+list(depth2_p))

    for ind_nu, nu in enumerate(nus):

        sig_i=depth_i[ind_nu]/(np.sqrt(hp.nside2pixarea(nside, degrees=True)) * 60)
        N[ind_nu, 0] = np.random.normal(0, sig_i, 12*nside**2)

        sig_p=depth_p[ind_nu]/(np.sqrt(hp.nside2pixarea(nside, degrees=True)) * 60)
        N[ind_nu, 1] = np.random.normal(0, sig_p, 12*nside**2)*np.sqrt(2)
        N[ind_nu, 2] = np.random.normal(0, sig_p, 12*nside**2)*np.sqrt(2)

    newN=N.copy()
    if nside_out != 256:
        newN=np.zeros((N.shape[0], 3, 12*nside_out**2))
        for i in range(N.shape[0]):
            newN[i]=hp.pixelfunc.ud_grade(N[i], nside_out)

    return newN
def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
def gauss_fit(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma])
    return popt
def get_instr(config, N_SAMPLE_BAND, prop):

    if prop == 0 or prop == 1:
        freq_maps=config['frequency']
        bandpasses = config['bandwidth']
        depth_i=config['depth_i']
        depth_p=config['depth_p']

    else:
        config1=config[0]
        config2=config[1]
        frac=[1-prop, prop]
        freq_maps=list(config1['frequency'])+list(config2['frequency'])
        depth1_i=config1['depth_i']/np.sqrt(frac[0])
        depth1_p=config1['depth_p']/np.sqrt(frac[0])
        depth2_i=config2['depth_i']/np.sqrt(frac[1])
        depth2_p=config2['depth_p']/np.sqrt(frac[1])

        depth_i=np.array(list(depth1_i)+list(depth2_i))
        depth_p=np.array(list(depth1_p)+list(depth2_p))
        bandpasses = list(config1['bandwidth'])+list(config2['bandwidth'])
    instrument=fgbuster.get_instrument('INSTRUMENT')
    freq_maps_bp_integrated = np.zeros_like(freq_maps)
    new_list_of_freqs_flat = []
    new_list_of_freqs = []
    #freqs_init = instrument.frequency*1.0
    for f in range(freq_maps_bp_integrated.shape[0]):

        fmin = freq_maps[f]-bandpasses[f]/2
        fmax = freq_maps[f]+bandpasses[f]/2
        #### bandpass_frequencies = np.linspace(fmin, fmax, fsteps) * u.GHz
        freqs = np.linspace(fmin, fmax, N_SAMPLE_BAND)
        weights_flat = np.ones(N_SAMPLE_BAND)

        weights = weights_flat.copy() / _jysr2rj(freqs)
        weights /= _rj2cmb(freqs)
        weights /= np.trapz(weights, freqs * 1e9)

        new_list_of_freqs.append((freqs, weights))

    instrument.frequency = new_list_of_freqs
    instrument.depth_i=depth_i
    instrument.depth_p=depth_p
    instrument.fwhm=np.zeros(len(freq_maps))

    return instrument


def get_cls(r1, r2, comp, instr, map1, map2, covmap, nus, Namaster):

    nside=256
    components1=get_comp_from_MixingMatrix(r1, comp, instr, map1[:, 1:, :], covmap,
    nside, nus)
    components2=get_comp_from_MixingMatrix(r2, comp, instr, map2[:, 1:, :], covmap,
    nside, nus)

    new_comp1=get_maps_for_namaster_QU(components1, nside)
    new_comp2=get_maps_for_namaster_QU(components2, nside)

    print('    ///// Get Cls')
    print()
    w=None
    leff, cls, _ = Namaster.get_spectra(new_comp1, map2=new_comp2,
                             purify_e=False,
                             purify_b=True,
                             w=w,
                             verbose=False,
                             beam_correction=None,
                             pixwin_correction=True)

    return leff, cls


def give_me_maps_instr(config, r, covmap, db, nubreak, prop, iib, model, nside_out, nside_index, fixsync):
    if db == 0:
        dust=model
        if model == 'd0':
            sync='s0'
        else:
            sync='s1'
    else:
        dust='d02b'

    if fixsync:
        skyconfig={'cmb':42, 'dust':dust}
    else:
        skyconfig={'cmb':42, 'dust':dust, 'synchrotron':sync}
    print(skyconfig)
    _, map, _=qubicplus.BImaps(skyconfig, config, r=r, nside=256).getskymaps(same_resol=0,
                                                          verbose=False,
                                                          coverage=covmap,
                                                          iib=iib,
                                                          noise=True,
                                                          signoise=1.,
                                                          beta=[1.54-db, 1.54, nubreak, 0.3],
                                                          fix_temp=20,
                                                          nside_index=nside_index)


    newmap=map.copy()
    if nside_out != 256:
        newmap=np.zeros((map.shape[0], 3, 12*nside_out**2))
        for i in range(map.shape[0]):
            newmap[i]=hp.pixelfunc.ud_grade(map[i], nside_out)

    instr=get_instr(config, N_SAMPLE_BAND=iib, prop=prop)
    if iib == 1 :
        instr=fgbuster.get_instrument('INSTRUMENT')
        instr.frequency=config['frequency']
        instr.depth_i=config['depth_i']
        instr.depth_p=config['depth_p']


    return newmap, instr
