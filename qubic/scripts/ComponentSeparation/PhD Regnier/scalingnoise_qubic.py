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
import warnings
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
def get_clBB(Namaster, map1, map2):

    leff, cls, _ = Namaster.get_spectra(map1, map2=map2,
                                 purify_e=False,
                                 purify_b=True,
                                 w=None,
                                 verbose=False,
                                 beam_correction=None,
                                 pixwin_correction=False)

    return leff, cls[:, 2]

from qubic import NamasterLib as nam
from qubic import QubicSkySim as qss
import sys
import pickle

def cl2dl(ell, cl):

    dl=np.zeros(ell.shape[0])
    for i in range(ell.shape[0]):
        dl[i]=(ell[i]*(ell[i]+1)*cl[i])/(2*np.pi)
    return dl
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
def dl2cl(ell, dl):

    cl=np.zeros(ell.shape[0])
    for i in range(ell.shape[0]):
        cl[i]=(dl[i]*(2*np.pi))/(ell[i]*(ell[i]+1))
    return cl
def combine_config(list_of_config):
    mynewnus=[]
    mynewdepth_i=[]
    mynewdepth_p=[]
    myfsky=list_of_config[0]['fsky']
    for conf in list_of_config:
        mynus=conf['frequency']
        mydepth=conf['depth_p']
        for j in range(len(mynus)):
            mynewnus.append(mynus[j])
            mynewdepth_i.append(1e3)
            mynewdepth_p.append(mydepth[j])
    dict={}
    dict['frequency']=np.array(mynewnus)
    dict['fsky']=myfsky
    dict['depth_i']=np.array(mynewdepth_i)
    dict['depth_p']=np.array(mynewdepth_p)
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
        N[ind_nu, 1] = np.random.normal(0, sig_p, 12*nside**2)*np.sqrt(2)
        N[ind_nu, 2] = np.random.normal(0, sig_p, 12*nside**2)*np.sqrt(2)

    return N
def get_list_config(name, nsub):
    myconf=[]
    tab=np.arange(0, len(name), 1)
    for i in range(len(name)):
        if i % 2 == 0:
            #print(i)
            with open('/pbs/home/m/mregnier/sps1/QUBIC+/forecast_decorrelation/{}_config.pkl'.format(name[i:i+2]), 'rb') as f:
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
        power_spectrum += r * hp.read_cl(CMB_CL_FILE
                                         %'unlensed_scalar_and_tensor_r1')[:,:4000]
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
from qubic import QubicSkySim as qss
def give_me_freqs_fwhm(dic, Nb) :
    band = dic['filter_nu'] / 1e9
    filter_relative_bandwidth = dic['filter_relative_bandwidth']
    a, nus_edge, nus_in, df, e, Nbbands_in = qubic.compute_freq(band, Nb, filter_relative_bandwidth)
    return nus_in, dic['synthbeam_peak150_fwhm'] * 150 / nus_in
import os.path as op
CMB_CL_FILE = op.join('/pbs/home/m/mregnier/sps1/QUBIC+/forecast_decorrelation/Cls_Planck2018_%s.fits')
def foregrounds(dust_model, nus, nside, extra_args=None):
    settings=[dust_model, 's1']
    #if sync:
        #print('add sync')
    #    settings.append('s1')

    if extra_args is not None:
        for i in extra_args:
            print(i, extra_args[i])
            pysm3.sky.PRESET_MODELS[dust_model][str(i)]=extra_args[i]
    sky = pysm3.Sky(nside = nside, preset_strings=settings)

    list_index=[sky.components[0].mbb_index, sky.components[1].pl_index]
    #if sync:
    #    list_index.append(sky.components[1].pl_index)


    fg=np.zeros((len(nus), 3, 12*nside**2))
    if dust_model != 'd0':
        sky.components[0].mbb_temperature=20*sky.components[0].mbb_temperature.unit
        for spectral_param in list_index:
            spectral_param[:] = hp.ud_grade(hp.ud_grade(spectral_param.value, 8),
                                        nside) * spectral_param.unit

    for j in range(len(nus)):
        fg[j]=np.array(sky.get_emission(nus[j]*u.GHz)*utils.bandpass_unit_conversion(nus[j]*u.GHz,
                                                                            None, u.uK_CMB))
    return fg


from qubic import NamasterLib as nam
nside=256
lmin=40
lmax=2*nside-1
dl=30

N=int(sys.argv[1])
exp=str(sys.argv[2])
nbands=int(sys.argv[3])

fsky=0.015
comp=[fgbuster.CMB(), fgbuster.Dust(nu0=148.95, temp=20), fgbuster.Synchrotron(nu0=148.95), fgbuster.Synchrotron(nu0=148.95)]
instr=fgbuster.get_instrument('INSTRUMENT')

if exp == 'BKPL':
    print('Bicep/Keck')
    myconf=get_list_config(name=exp, nsub=1)
    config=combine_config(myconf)
    covmap=get_coverage(config['fsky'], nside)
    pixok=covmap>0
    instr.frequency=config['frequency']

    Alens=1
elif exp == 'SO':
    print('Simons Observatory')
    myconf=get_list_config(name=exp, nsub=1)
    config=combine_config(myconf)
    covmap=get_coverage(config['fsky'], nside)
    pixok=covmap>0
    instr.frequency=config['frequency']

    Alens=0.5
elif exp == 'S4':
    print('CMB-S4')
    myconf=get_list_config(name=exp, nsub=1)
    config=combine_config(myconf)
    covmap=get_coverage(config['fsky'], nside)
    pixok=covmap>0
    instr.frequency=config['frequency']
    Alens=0.1
elif exp =='QUBIC':
    print('QUBIC')

    # Read dictionary

    filename='/pbs/home/m/mregnier/Libs/qubic/qubic/doc/FastSimulator/FastSimDemo_FI-150.dict'
    d150 = qubic.qubicdict.qubicDict()
    d150.read_from_file(filename)
    d150['nside'] = nside
    center = qubic.equ2gal(d150['RA_center'], d150['DEC_center'])
    d150['nf_recon'] = nbands
    d150['nf_sub'] = nbands    ### this is OK as we use noise-only simulations

    filename='/pbs/home/m/mregnier/Libs/qubic/qubic/doc/FastSimulator/FastSimDemo_FI-220.dict'
    d220 = qubic.qubicdict.qubicDict()
    d220.read_from_file(filename)
    d220['nside'] = nside
    center = qubic.equ2gal(d220['RA_center'], d220['DEC_center'])
    d220['nf_recon'] = nbands
    d220['nf_sub'] = nbands

    mynus150, _=give_me_freqs_fwhm(d150, nbands)
    mynus220, _=give_me_freqs_fwhm(d220, nbands)
    instr.frequency=list(mynus150)+list(mynus220)+list(np.array([30, 44.1, 70.4, 353.001]))
    covmap=get_coverage(0.015, nside)
    pixok=covmap>0
    Alens=1
else:
    raise TypeError('Choose Bicep/Keck or QUBIC')

print(instr.frequency, '\n')
print(Alens, '\n')

maskpix = np.zeros(12*nside**2)
pixok = covmap > 0
maskpix[pixok] = 1
Namaster = nam.Namaster(maskpix, lmin=lmin, lmax=lmax, delta_ell=dl)
Namaster.fsky = fsky
Namaster.ell_binned, _=Namaster.get_binning(nside)


def run_mc_qubic(N, Namaster, dust_model, instr, nside, extra, Alens):

    mycl=np.zeros((N, 16))
    nside_fgb=8
    npix_fgb=12*nside_fgb**2
    mybeta=np.zeros((N, npix_fgb))
    RMS=np.zeros(N)

    for i in range(N):
        seed=np.random.randint(10000000)
        mycmb=give_me_cmb(np.array(instr.frequency), seed, r=0, nside=nside, Alens=Alens)
        myinputs=foregrounds(dust_model, instr.frequency, nside=nside, extra_args=extra)+mycmb.copy()

        myinputs[:, :, ~pixok]=hp.UNSEEN

        #CompSep
        r=fgbuster.basic_comp_sep(comp, instr, myinputs[:, 1:, :], nside=8)
        print(r.x)
        res=r.s[0, 0]-mycmb[0, 1]
        RMS[i]=np.std(res[pixok])
        print(i)
        mybeta[i]=r.x[0]
        maps_for_nam=np.zeros((3, 12*nside**2))
        maps_for_nam[1:, :]=r.s[0].copy()
        maps_for_nam[:, ~pixok]=0

        le, mycl[i] = get_clBB(Namaster, map1=maps_for_nam, map2=None)

    print('\n Average RMS : {:.3e}'.format(np.mean(RMS)))
    return le, mycl, mybeta

dust='d6'
extra={'correlation_length':15}
leff, cl, beta = run_mc_qubic(N, Namaster=Namaster, dust_model=dust, instr=instr, nside=nside, extra=extra, Alens=Alens)
print()
cls=_get_Cl_cmb(Alens=1, r=0)
print(np.mean(cl, axis=0))
print(cl2dl(leff, cls[2, leff.astype(int)]))
print()

# write python dict to a file
mydict = {'mycl': cl, 'leff': leff, 'mybeta':beta}
output = open('/pbs/home/m/mregnier/sps1/scalingnoise/results/cl_{}_{}_nbands{}_{}reals.pkl'.format(exp, dust, nbands, N), 'wb')
pickle.dump(mydict, output)
output.close()
