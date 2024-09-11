import numpy as np
import healpy as hp
import qubic
import os
import matplotlib.pyplot as plt
from qubic import progress_bar
import fgbuster
import pickle
import warnings
import pysm3.units as u
from pysm3 import utils
import pysm3
from qubic import NamasterLib as nam
from qubic import QubicSkySim as qss
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")
from qubic import NamasterLib as nam
import os.path as op
CMB_CL_FILE = op.join('/home/regnier/work/regnier/forecast_deco/Cls_Planck2018_%s.fits')

###################################
#            Components           #
###################################

def foregrounds(dust_model, nus, nside, NSIDE_PATCH, extra_args=None, nus_edge=None):

    if dust_model == 'd0':
        sync_model = 's0'
    else:
        sync_model='s1'
    settings=[dust_model, sync_model]
    #print('settings : ', settings)

    if extra_args is not None:
        for i in extra_args:
            #print(i, extra_args[i])
            pysm3.sky.PRESET_MODELS[dust_model][str(i)]=extra_args[i]
    sky = pysm3.Sky(nside = nside, preset_strings=settings)

    if dust_model == 'd1' or dust_model == 'd2' or dust_model == 'd3' or dust_model == 'd6' or dust_model == 'd10':
        list_index=[sky.components[0].mbb_index, sky.components[1].pl_index]
        sky.components[0].mbb_temperature=20*sky.components[0].mbb_temperature.unit
        betad=hp.ud_grade(np.array(sky.components[0].mbb_index), NSIDE_PATCH)
        betas=hp.ud_grade(np.array(sky.components[1].pl_index), NSIDE_PATCH)
    #    for spectral_param in list_index:
    #            spectral_param[:] = hp.ud_grade(hp.ud_grade(spectral_param.value, NSIDE_PATCH),
    #                                        nside) * spectral_param.unit
    elif dust_model == 'd4' or dust_model == 'd5' or dust_model == 'd7' or dust_model == 'd8' or dust_model == 'd9':
        betad=np.zeros(12*NSIDE_PATCH**2)
        betas=np.zeros(12*NSIDE_PATCH**2)
    else:
        betad=1.54
        betas=-3
        
    
    fg=np.zeros((len(nus), 3, 12*nside**2))
    
    index=np.array([betad, betas])
    for j in range(len(nus)):
        if nus_edge is not None:
            nfreqinteg = 100
            freqs = np.linspace(nus_edge[j, 0], nus_edge[j, 1], nfreqinteg)
            weights = np.ones(nfreqinteg)
            fg[j]=np.array(sky.get_emission(freqs*u.GHz, weights)*utils.bandpass_unit_conversion(freqs*u.GHz, weights, u.uK_CMB))
        else:
            fg[j]=np.array(sky.get_emission(nus[j]*u.GHz)*utils.bandpass_unit_conversion(nus[j]*u.GHz, None, u.uK_CMB))
    fg_100GHz=np.array(sky.get_emission(100*u.GHz)*utils.bandpass_unit_conversion(100*u.GHz, None, u.uK_CMB))
    return fg, index, fg_100GHz




def myBBth(ell, r, Alens):
    clBB = cl2dl(ell, _get_Cl_cmb(Alens=Alens, r=r)[2, ell.astype(int)-1])
    return clBB
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
def get_clBB(Namaster, map1, map2):

    leff, cls, _ = Namaster.get_spectra(map1, map2=map2,
                                 purify_e=False,
                                 purify_b=True,
                                 w=None,
                                 verbose=False,
                                 beam_correction=None,
                                 pixwin_correction=False)

    return leff, cls[:, 2]
def cl2dl(ell, cl):

    dl=np.zeros(ell.shape[0])
    for i in range(ell.shape[0]):
        dl[i]=(ell[i]*(ell[i]+1)*cl[i])/(2*np.pi)
    return dl
def dl2cl(ell, dl):

    cl=np.zeros(ell.shape[0])
    for i in range(ell.shape[0]):
        cl[i]=(dl[i]*(2*np.pi))/(ell[i]*(ell[i]+1))
    return cl
def combine_config(list_of_config):
    mynewnus=[]
    mynewdepth_i=[]
    mynewdepth_p=[]
    mynewbandwidth=[]
    mynewedges=[]
    myfsky=list_of_config[0]['fsky']
    
    for conf in list_of_config:
        mynus=conf['frequency']
        mydepth=conf['depth_p']
        band=conf['bandwidth']
        #mynus_edge=conf['edges']
        mynewedges.append(conf['edges'])
        for j in range(len(mynus)):
            mynewnus.append(mynus[j])
            mynewdepth_i.append(1e3)
            mynewbandwidth.append(band[j])
            mynewdepth_p.append(mydepth[j])
            
    #print(mynewedges)
    dict={}
    dict['frequency']=np.array(mynewnus)
    dict['fsky']=myfsky
    dict['depth_i']=np.array(mynewdepth_i)
    dict['depth_p']=np.array(mynewdepth_p)
    dict['bandwidth']=mynewbandwidth
    if len(list_of_config) == 2:
        dict['edges']=np.concatenate((mynewedges[0], mynewedges[1]), axis=0)
    else:
        dict['edges']=mynewedges[0]
    return dict
def _get_noise(nus, nside, config):

    npix=12*nside**2
    nfreqs=len(nus)
    np.random.seed(None)

    N = np.zeros(((nfreqs, 3, npix)))
    depth_i = config['depth_i']
    depth_p = config['depth_p']

    for ind_nu, nu in enumerate(nus):

        sig_i=depth_i[ind_nu]/(np.sqrt(hp.nside2pixarea(nside, degrees=True)) * 60)
        N[ind_nu, 0] = np.random.normal(0, sig_i, 12*nside**2)

        sig_p=depth_p[ind_nu]/(np.sqrt(hp.nside2pixarea(nside, degrees=True)) * 60)
        N[ind_nu, 1] = np.random.normal(0, sig_p, 12*nside**2)#*np.sqrt(2)
        N[ind_nu, 2] = np.random.normal(0, sig_p, 12*nside**2)#*np.sqrt(2)

    return N
def fct_subopt(nus, nsubs):
    '''
    This function evaluates QUBIC sub-optimality on r at a given input frequency using
    L.Mousset's work (QUBIC paper II)
    '''
    subnus = [150., 220]
    nsubs_vec = [2,3,4,5,6,7,8]
    subval = []
    subval.append([1.25,1.1]) #2
    subval.append([1.35,1.15]) #3
    subval.append([1.35, 1.1]) #4
    subval.append([1.4, 1.2]) #5
    subval.append([1.52, 1.32]) #6
    subval.append([1.5, 1.4]) #7
    subval.append([1.65,1.45]) #8
    fct_subopt = np.poly1d(np.polyfit(subnus, subval[nsubs_vec.index(nsubs)], 1))
    return fct_subopt(nus)
def get_edges(nus, bandwidth):
    edges=np.zeros((len(nus), 2))
    dnu_nu=bandwidth/nus
    edges_max=nus * (1. + dnu_nu/2)
    edges_min=nus * (1. - dnu_nu/2)
    for i in range(len(nus)):
        edges[i, 0]=edges_min[i]
        edges[i, 1]=edges_max[i]
    return edges
def qubicify(config, qp_nsubs, qp_effective_fraction):
    nbands = np.sum(qp_nsubs)
    qp_config = config.copy()
    for k in qp_config.keys():
        qp_config[k]=[]
    qp_config['nbands'] = nbands
    qp_config['fsky'] = config['fsky']

    for i in range(len(config['frequency'])):
        newedges = np.linspace(config['edges'][i][0], config['edges'][i][-1], qp_nsubs[i]+1)
        #print(newedges)
        newfreqs = (newedges[0:-1]+newedges[1:])/2
        newbandwidth = newedges[1:] - newedges[0:-1]
        newdnu_nu = newbandwidth / newfreqs
        #print('qp : ', qp_nsubs[0])
        scalefactor_noise = np.sqrt(qp_nsubs[i]) * fct_subopt(config['frequency'][i], qp_nsubs[0])#fct_subopt(config['frequency'][i])
        print('scaling : ', scalefactor_noise)
        print(config['depth_p'])
        newdepth_p = config['depth_p'][i] * np.ones(qp_nsubs[i]) * scalefactor_noise
        newdepth_i = config['depth_i'][i] * np.ones(qp_nsubs[i]) * scalefactor_noise
        initial_band = np.ones(qp_nsubs[i]) * config['frequency'][i]
        for k in range(qp_nsubs[i]):
            if qp_effective_fraction[i] != 0:
                qp_config['frequency'].append(newfreqs[k])
                qp_config['depth_p'].append(newdepth_p[k])
                qp_config['depth_i'].append(newdepth_i[k])
                qp_config['bandwidth'].append(newbandwidth[k])
        edges=get_edges(np.array(qp_config['frequency']), np.array(qp_config['bandwidth']))
        qp_config['edges']=edges.copy()
        
    fields = ['frequency', 'depth_p', 'depth_i', 'bandwidth', 'edges']
    for j in range(len(fields)):
        qp_config[fields[j]] = np.array(qp_config[fields[j]])

    return qp_config
def get_list_config(name, nsub):
    myconf=[]
    tab=np.arange(0, len(name), 1)
    for i in range(len(name)):
        if i % 2 == 0:
            with open('/home/regnier/work/regnier/forecast_deco/dict/{}_config.pkl'.format(name[i:i+2]), 'rb') as f:
                config = pickle.load(f)
                if name[i:i+2] == 'PL':
                    myconf.append(config)
                elif nsub!=1 and i == 0:
                    qp_effective_fraction=np.ones(config['nbands'])
                    qp_config=qubicify(config, np.ones(config['nbands']).astype(int)*nsub, qp_effective_fraction)
                    myconf.append(qp_config)
                elif nsub==1 or i != 0:
                    myconf.append(config)
    return myconf
def _get_Cl_cmb(Alens, r):
    power_spectrum = hp.read_cl(CMB_CL_FILE%'lensed_scalar')[:,:4000]
    if Alens != 1.:
        power_spectrum[2] *= Alens
    if r:
        power_spectrum += r * hp.read_cl(CMB_CL_FILE%'unlensed_scalar_and_tensor_r1')[:,:4000]
    return power_spectrum
def give_me_cmb(nus, seed, r, nside, Alens):

    npix=12*nside**2
    ell=np.arange(2*nside-1)
    mycls = _get_Cl_cmb(Alens=Alens, r=r)
    mycls[1]=np.zeros(4000)
    mycls[3]=np.zeros(4000)

    np.random.seed(seed)
    maps = hp.synfast(mycls, nside, verbose=False, new=True)
    mymaps=np.zeros((nus.shape[0], 3, npix))
    for i in range(nus.shape[0]):
        mymaps[i]=maps.copy()
    return mymaps
def get_clBB(Namaster, map1, map2):

    leff, cls, _ = Namaster.get_spectra(map1, map2=map2,
                                 purify_e=False,
                                 purify_b=True,
                                 w=None,
                                 verbose=False,
                                 beam_correction=None,
                                 pixwin_correction=False)

    return leff, cls[:, 2]
def give_me_freqs_fwhm(dic, Nb) :
    band = dic['filter_nu'] / 1e9
    filter_relative_bandwidth = dic['filter_relative_bandwidth']
    a, nus_edge, nus_in, df, e, Nbbands_in = qubic.compute_freq(band, Nb, filter_relative_bandwidth)
    return nus_in, dic['synthbeam_peak150_fwhm'] * 150 / nus_in
def _get_noise(nus, instr, nside):
    
    nfreqs=len(nus)
    npix=12*nside**2
    np.random.seed(None)

    N = np.zeros(((nfreqs, 3, npix)))
    depth_i = instr.depth_i
    depth_p = instr.depth_p

    for ind_nu, nu in enumerate(nus):

        sig_i=depth_i[ind_nu]/(np.sqrt(hp.nside2pixarea(nside, degrees=True)) * 60)
        N[ind_nu, 0] = np.random.normal(0, sig_i, 12*nside**2)

        sig_p=depth_p[ind_nu]/(np.sqrt(hp.nside2pixarea(nside, degrees=True)) * 60)
        N[ind_nu, 1] = np.random.normal(0, sig_p, 12*nside**2)#*np.sqrt(2)
        N[ind_nu, 2] = np.random.normal(0, sig_p, 12*nside**2)#*np.sqrt(2)

    return N
def run_mc_qubic(N, Namaster, dust_model, instr, nside, extra, Alens, r, pixok, nus_edge, nus, noiseless, noisy, nside_fgb):
    
    comp=[fgbuster.CMB(), fgbuster.Dust(nu0=100, temp=20), fgbuster.Synchrotron(nu0=100)]
    ncomp=len(comp)

    nbin=len(Namaster.ell_binned)
    mycl=np.zeros((N, nbin))
        

    if nside_fgb==0:
        mybeta=np.zeros((N, 2))
    else:
        npix_fgb=12*nside_fgb**2
        mybeta=np.zeros((N, 2))#, npix_fgb))

    ###### Display some informations
    print('\n==========================================================================')
    print('General informations : ')
    print('    N         : {}'.format(N))
    print('    Nside     : {}'.format(nside))
    print('    leff      : {}'.format(Namaster.ell_binned))
    print('    nus       : {}'.format(nus))
    print('    Depth     : {}'.format(instr.depth_p))
    if noiseless is False: typ='Full pipeline'
    else: typ='Propagation of noise'
    print('    Type      : {}'.format(typ))
    print('    Noisy     : {}'.format(noisy))
          
    print('CMB component : ')
    print('    r         : {:.3f}'.format(r))
    print('    Alens     : {:.3f}'.format(Alens))
          
    print('Dust component : ')
    print('    Model     : {}'.format(dust_model))
    print('    Nside_fgb : {}'.format(nside_fgb))
    print('==========================================================================\n')

    RMS=np.zeros(N)

    mycomp1_est=np.zeros((N, ncomp, 2, 12*nside**2))
    total_residuals1=np.zeros((N, 2, 12*nside**2))
    total_residuals1_fore=np.zeros((N, 2, 12*nside**2))
    mycomp2_est=np.zeros((N, ncomp, 2, 12*nside**2))
    total_residuals2=np.zeros((N, 2, 12*nside**2))
    index_true=np.zeros((N, 2, 12*nside_fgb**2))
    allfore=np.zeros((N, len(nus), 2, 12*nside**2))
    for i in range(N):
        print(i)

        ####################################
        #  Setup CMB + Foregrounds + Noise #
        ####################################

        seed=np.random.randint(10000000)
        mycmb=give_me_cmb(nus, seed, r=r, nside=nside, Alens=Alens)
        if dust_model == 'd0':
            fore, _, fore_150 = foregrounds(dust_model, nus, nside=nside, NSIDE_PATCH=nside_fgb, extra_args=extra, nus_edge=nus_edge)
        else:
            fore, index_true[i], fore_150 = foregrounds(dust_model, nus, nside=nside, NSIDE_PATCH=nside_fgb, extra_args=extra, nus_edge=nus_edge)
            
        allfore[i, :, 0, :]=fore[:, 1].copy()
        allfore[i, :, 1, :]=fore[:, 2].copy()
        #fore_150[:, ~pixok]=0
        myinputs=fore.copy()
        myinputs+=mycmb.copy()
        myinputs[:, :, ~pixok]=hp.UNSEEN

        if noisy:
            noise1=_get_noise(nus, instr, nside)
            noise2=_get_noise(nus, instr, nside)
        else:
            noise1=np.zeros((len(nus), 3, 12*nside**2))
            noise2=np.zeros((len(nus), 3, 12*nside**2))
            
        myinputs1_noisy=myinputs.copy()
        myinputs2_noisy=myinputs.copy()
        myinputs1_noisy+=noise1.copy()
        myinputs2_noisy+=noise2.copy()
            
        if nside_fgb == 0:
            to_fgb1=myinputs1_noisy[:, 1:, pixok].copy()
            to_fgb2=myinputs2_noisy[:, 1:, pixok].copy()
        else:
            to_fgb1=myinputs1_noisy[:, 1:, :].copy()
            to_fgb2=myinputs2_noisy[:, 1:, :].copy()
            
            
        ####################################
        #####  Components Separation   #####
        ####################################
            
        d1=fgbuster.basic_comp_sep(comp, instr, to_fgb1, method='TNC', tol=1e-18, nside=nside_fgb)
        d2=fgbuster.basic_comp_sep(comp, instr, to_fgb2, method='TNC', tol=1e-18, nside=nside_fgb)
        mybeta[i]=d1.x.copy()
            
        if nside_fgb != 0:
            index_seen=np.where(d1.x[0] != hp.UNSEEN)[0]
            print('Average estimated beta_d : {:.8f}'.format(np.mean(mybeta[:, :, index_seen], axis=2)[i, 0]))
            print('Average estimated beta_s : {:.8f}'.format(np.mean(mybeta[:, :, index_seen], axis=2)[i, 1]))
            print()
            print('Average residuals beta_d : {:.8f}'.format(np.mean(mybeta[:, :, index_seen][i, 0]-index_true[i, 0][index_seen])))
            print('Average residuals beta_s : {:.8f}'.format(np.mean(mybeta[:, :, index_seen][i, 1]-index_true[i, 1][index_seen])))
        else:
            print('Average estimated beta_d : {:.8f}'.format(mybeta[i, 0]))
            print('Average estimated beta_s : {:.8f}'.format(mybeta[i, 1]))
            
        if nside_fgb == 0:
            print(d1.x, d2.x)
            for icomp in range(ncomp):
                mycomp1_est[i, icomp, :, pixok]=d1.s[icomp].T.copy()
                mycomp2_est[i, icomp, :, pixok]=d2.s[icomp].T.copy()
        else:
            for icomp in range(ncomp):
                mycomp1_est[i, icomp, :, :]=d1.s[icomp].copy()
                mycomp2_est[i, icomp, :, :]=d2.s[icomp].copy()

        ### Compute residuals maps
        total_residuals1[i]=mycomp1_est[i, 0, :]-mycmb[0, 1:]
        #total_residuals2[i]=mycomp2_est[i, 0, :]-mycmb[0, 1:]
        total_residuals1_fore[i]=(mycomp1_est[i, 1, :]+mycomp1_est[i, 2, :]) - fore_150[1:].copy()
            
        ####################################
        #########    BB spectrum   #########
        ####################################
            
        RMS[i]=np.std(total_residuals1[i, 0, pixok])
        print('CMB residuals : {:.6f}'.format(RMS[i]))
        mycmb_est_to_namaster1=np.zeros((3, 12*nside**2))
        mycmb_est_to_namaster1[1:]=mycomp1_est[i, 0, :, :].copy()
        mycmb_est_to_namaster1[:, ~pixok]=0                                      # Just to be sure
            
        mycmb_est_to_namaster2=np.zeros((3, 12*nside**2))
        mycmb_est_to_namaster2[1:]=mycomp2_est[i, 0, :, :].copy()
        mycmb_est_to_namaster2[:, ~pixok]=0                                      # Just to be sure
            
        le, mycl[i] = get_clBB(Namaster, map1=mycmb_est_to_namaster1, map2=mycmb_est_to_namaster2)
            
    return le, mycl, total_residuals1, total_residuals1_fore, mybeta, index_true, allfore
def get_instr_simple(config):

    freq_maps=config['frequency']
    bandpasses = config['bandwidth']
    depth_i=config['depth_i']
    depth_p=config['depth_p']
    instrument=fgbuster.get_instrument('INSTRUMENT')
    instrument.frequency = freq_maps
    instrument.depth_i=depth_i
    instrument.depth_p=depth_p
    return instrument
def get_instr(mynus, config, N_SAMPLE_BAND):

    freq_maps=mynus
    bandpasses = config['bandwidth']
    depth_i=config['depth_i']
    depth_p=config['depth_p']

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
    #instrument.fwhm=np.zeros(len(freq_maps))

    return instrument

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




def closest_value(input_list, input_value):
    arr = np.asarray(input_list)
    i = (np.abs(arr - input_value)).argmin()
    return arr[i]
def where_closest_value(input_list, input_value):
    i=closest_value(input_list, input_value)
    argi = np.where(input_list == i)[0][0]
    return argi
def give_me_fg(allnus, nus_exp, allfg, nside):
    maps_exp = np.zeros((len(nus_exp), 3, 12*nside**2))
    index=[]
    for ii, i in enumerate(nus_exp):
        arg = where_closest_value(allnus, i)
        index.append(arg)
        maps_exp[ii]=allfg[arg].copy()
    return maps_exp, allnus[index]

def generate_fg(sky, seed, nus, real_nus, nside, nus_edge):
    nfreqs=len(real_nus)
    nus_eff=np.zeros(nfreqs)
    seed_eff = np.zeros(nfreqs)
    
    for i in range(nfreqs):
        myargs = where_closest_value(nus, real_nus[i])
        nus_eff[i] = nus[myargs].copy()
        seed_eff[i] = seed[myargs].copy()
    
    myobs=np.zeros((nfreqs, 3, 12*nside**2))
    for i in range(nfreqs):
        np.random.seed(int(seed_eff[i]))
        freqs = np.linspace(nus_edge[j, 0], nus_edge[j, 1], nfreqinteg)
        weights = np.ones(nfreqinteg)
        myobs[i]=np.array(sky.get_emission(nus_eff[i] * u.GHz, 
                                             weights)*utils.bandpass_unit_conversion(freqs*u.GHz, 
                                                                                     weights, 
                                                                                     u.uK_CMB))

    return myobs

class Forecast:
    
    def __init__(self, nus, Namaster, dust, instr, nside, extra, Alens, r, pixok, sky, config):

        self.nus=nus
        self.Namaster=Namaster
        self.dust=dust
        self.instr=instr
        self.r=r
        self.extra=extra
        self.Alens=Alens
        self.nside=nside
        self.pixok=pixok
        self.npix=12*self.nside**2
        self.sky=sky
        self.config=config
        self.nus150=[150]
        
    def generate_fg(self, allnus, seed, nus, nus_edge, bandpass=True):
        nfreqinteg=100
        nfreqs=len(nus)
        nus_eff=np.zeros(nfreqs)
        seed_eff=np.zeros(nfreqs)
    
        for i in range(nfreqs):
            
            myargs = where_closest_value(allnus, nus[i])
            nus_eff[i] = np.round(allnus[myargs], 3).copy()
            seed_eff[i] = seed[myargs].copy()

    
        myobs=np.zeros((nfreqs, 3, 12*self.nside**2))

        for i in range(nfreqs):
            # Bandpass
            if bandpass:
                freqs = np.linspace(nus_edge[i, 0], nus_edge[i, 1], nfreqinteg)
                weights = np.ones(nfreqinteg)
            
                # Fix seed
                np.random.seed(int(seed_eff[i]))
                myobs[i]=np.array(self.sky.get_emission(freqs * u.GHz, weights)*utils.bandpass_unit_conversion(freqs*u.GHz, 
                                                                                     weights, 
                                                                                     u.uK_CMB))
            else:
                # Fix seed
                np.random.seed(int(seed_eff[i]))
                myobs[i]=np.array(self.sky.get_emission(nus_eff[i] * u.GHz, 
                                             None)*utils.bandpass_unit_conversion(nus_eff[i]*u.GHz, 
                                                                                     None, 
                                                                                     u.uK_CMB))

        return myobs, nus_eff
        
    def give_cl_cmb(self):
        power_spectrum = hp.read_cl(CMB_CL_FILE%'lensed_scalar')[:,:4000]
        if self.Alens != 1.:
            power_spectrum[2] *= self.Alens
        if self.r:
            power_spectrum += self.r * hp.read_cl(CMB_CL_FILE%'unlensed_scalar_and_tensor_r1')[:,:4000]
        return power_spectrum
    def give_me_cmb(self, seed):

        
        ell=np.arange(2*self.nside-1)
        mycls = self.give_cl_cmb()#_get_Cl_cmb(Alens=self.Alens, r=self.r)
        mycls[1]=np.zeros(4000)
        mycls[3]=np.zeros(4000)

        np.random.seed(seed)
        maps = hp.synfast(mycls, self.nside, verbose=False, new=True)
        mymaps=np.zeros((self.nus.shape[0], 3, self.npix))
        for i in range(self.nus.shape[0]):
            mymaps[i]=maps.copy()
        return mymaps
    
    
    def _get_noise(self):
    
        self.nfreqs=len(self.nus)
        np.random.seed(None)

        N = np.zeros(((self.nfreqs, 3, self.npix)))
        #depth_i = self.instr.depth_i
        #depth_p = self.instr.depth_p

        for ind_nu, nu in enumerate(self.nus):

            sig_i=self.instr.depth_i[ind_nu]/(np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60)
            N[ind_nu, 0] = np.random.normal(0, sig_i, 12*self.nside**2)

            sig_p=self.instr.depth_p[ind_nu]/(np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60)
            N[ind_nu, 1] = np.random.normal(0, sig_p, 12*self.nside**2)#*np.sqrt(2)
            N[ind_nu, 2] = np.random.normal(0, sig_p, 12*self.nside**2)#*np.sqrt(2)

        return N
    
    
    def get_clBB(self, map1, map2):
        self.Namaster.aposize = 4

        leff, cls, _ = self.Namaster.get_spectra(map1, map2=map2,
                                 purify_e=False,
                                 purify_b=True,
                                 w=None,
                                 verbose=False,
                                 beam_correction=None,
                                 pixwin_correction=False)

        return leff, cls[:, 2]
    
    
    def do_FGB(self, maps, nside_for_FGB, nu0):
        
        comp=[fgbuster.CMB(), fgbuster.Dust(nu0=nu0, temp=20), fgbuster.Synchrotron(nu0=nu0)]
        
        if nside_for_FGB == 0:
            to_fgb=maps[:, 1:, self.pixok].copy()
        else:
            to_fgb=maps[:, 1:, :].copy()
            #to_fgb=maps[:, 1:, :].copy()

        r=fgbuster.basic_comp_sep(comp, self.instr, to_fgb, method='L-BFGS-B', tol=1e-18, nside=nside_for_FGB)
        
        return r
        
    def prepare_to_nam(self, maps):
        
        to_namaster=np.zeros((3, 12*self.nside**2))
        to_namaster[1:]=maps[0, :, :].copy()
        to_namaster[:, ~self.pixok]=0                                      # Just to be sure
        
        return to_namaster
        
    def RUN_MC(self, allnus, allseed, NSIDE_PATCH):
        
        mycomp1_est=np.zeros((3, 2, 12*self.nside**2))
        mycomp2_est=np.zeros((3, 2, 12*self.nside**2))
        
        print('CMB GENERATION WITH R = {:.3f} & ALENS = {:.1f}'.format(self.r, self.Alens))
        seed=np.random.randint(10000000)
        mycmb=self.give_me_cmb(seed)
        
        bp=True
        if self.dust == 'd5' or self.dust == 'd7' or self.dust == 'd8':
            bp=False
        real_fore, nus_eff = self.generate_fg(allnus, allseed, self.nus, self.config['edges'], bandpass=bp)
        real_fore_150, nus_eff_150 = self.generate_fg(allnus, allseed, self.nus150, self.config['edges'], bandpass=False)

        
        noise1=self._get_noise()
        noise2=self._get_noise()
        
        inputs1 = mycmb.copy()
        inputs1 += real_fore.copy()
        inputs1 += noise1.copy()
        inputs2 = mycmb.copy()
        inputs2 += real_fore.copy()
        inputs2 += noise2.copy()
        
        print(1)
        inputs1[:, :, ~self.pixok] = hp.UNSEEN
        inputs2[:, :, ~self.pixok] = hp.UNSEEN
        
        d1=self.do_FGB(inputs1, NSIDE_PATCH, nu0=150)#nus_eff_150[0])
        print(2)
        d2=self.do_FGB(inputs2, NSIDE_PATCH, nu0=150)#nus_eff_150[0])
        
        ncomp=3
        if NSIDE_PATCH == 0:
            print(d1.x, d2.x)
            for icomp in range(ncomp):
                mycomp1_est[icomp, :, self.pixok]=d1.s[icomp].T.copy()
                mycomp2_est[icomp, :, self.pixok]=d2.s[icomp].T.copy()
        else:
            for icomp in range(ncomp):
                mycomp1_est[icomp, :, :]=d1.s[icomp].copy()
                mycomp2_est[icomp, :, :]=d2.s[icomp].copy()

        ### Compute residuals maps
        total_residuals=mycomp1_est[0, :, :]-mycmb[0, 1:, :].copy()
        #total_residuals_fore=(mycomp1_est[1, :, :]+mycomp1_est[2, :, :])# - real_fore_150[0, 1:].copy()
        RMSc=np.std(total_residuals[0, self.pixok])
        #RMSd=np.std(total_residuals_fore[0, self.pixok])
        #print('CMB residuals : {:.10f} [muK^2]'.format(RMSc))
        #print('Dust residuals : {:.10f} [muK^2]'.format(RMSd))
        
        maps_to_nam1 = self.prepare_to_nam(mycomp1_est)
        maps_to_nam2 = self.prepare_to_nam(mycomp2_est)
        
        leff, clBB = self.get_clBB(maps_to_nam1, maps_to_nam2)
        
        print('leff : ', leff[:5])
        
        print('BB spectra : ', clBB[:5])
        
        
        return leff, clBB, total_residuals, d1.x, mycmb
        
        
    
#le, mycl, total_residuals1, total_residuals1_fore, mybeta, index_true, allfore
        
        
        
        
        
        
        
        
def foregrounds_all_freqs(freqs, nside, dust, NSIDE_PATCH, extra=None, bandpass=False):
        
        npix=12*nside**2
        fg=np.zeros((len(freqs), 3, npix))
        
        if dust == 'd0':
            sync_model = 's0'
        else:
            sync_model='s1'
        settings=[dust, sync_model]
        #print(settings)
        if extra is not None:
            for i in extra:
                print(i, extra[i])
                pysm3.sky.PRESET_MODELS[dust][str(i)]=extra[i]
                
        sky = pysm3.Sky(nside = nside, preset_strings=settings)
        
        if dust == 'd1' or dust == 'd2' or dust == 'd3' or dust == 'd6' or dust == 'd10':
            list_index=[sky.components[0].mbb_index, sky.components[1].pl_index]
            sky.components[0].mbb_temperature=20*sky.components[0].mbb_temperature.unit
            betad=hp.ud_grade(np.array(sky.components[0].mbb_index), NSIDE_PATCH)
            betas=hp.ud_grade(np.array(sky.components[1].pl_index), NSIDE_PATCH)
            #    for spectral_param in list_index:
            #            spectral_param[:] = hp.ud_grade(hp.ud_grade(spectral_param.value, NSIDE_PATCH),
            #                                        nside) * spectral_param.unit
        elif dust == 'd4' or dust == 'd5' or dust == 'd7' or dust == 'd8' or dust == 'd9':
            betad=np.zeros(12*NSIDE_PATCH**2)
            betas=np.zeros(12*NSIDE_PATCH**2)
        else:
            betad=1.54
            betas=-3
            
        index=np.array([betad, betas])
        
        if bandpass:pass
        else:
            for inu, nu in enumerate(freqs):
                #print(inu, nu)
                fg[inu]=np.array(sky.get_emission(nu*u.GHz)*utils.bandpass_unit_conversion(nu*u.GHz, None, u.uK_CMB))
        return fg, index