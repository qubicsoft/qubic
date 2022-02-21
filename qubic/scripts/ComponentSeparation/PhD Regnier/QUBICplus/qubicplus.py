from qubic import camb_interface as qc
import healpy as hp
import numpy as np
import qubicplus
import pysm3
import pysm3.units as u
from pysm3 import utils
from qubic import camb_interface as qc
import matplotlib.pyplot as plt
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
def create_dust_with_model2beta(nside, nus, betad0, betad1, nubreak, temp, break_width):

    # Create 353 GHz dust maps
    sky = pysm3.Sky(nside=nside, preset_strings=['d0'])
    maps_353GHz = sky.get_emission(353*u.GHz, None)*utils.bandpass_unit_conversion(353*u.GHz,None, u.uK_CMB)

    comp2b=[fgbuster.component_model.Dust_2b(nu0=353, break_width=break_width)]

    A2b = fgbuster.MixingMatrix(*comp2b)
    A2b_ev = A2b.evaluator(nus)
    A2b_maxL = A2b_ev([betad0, betad1, nubreak, temp])

    new_dust_map=np.zeros(((len(nus), 3, 12*nside**2)))
    for i in range(len(nus)):
        new_dust_map[i]=A2b_maxL[i, 0]*maps_353GHz

    return new_dust_map
def create_sync(nside, nus):

    # Create 353 GHz dust maps
    sky = pysm3.Sky(nside=nside, preset_strings=['s0'])
    maps_70GHz = sky.get_emission(70*u.GHz, None)*utils.bandpass_unit_conversion(70*u.GHz,None, u.uK_CMB)

    comp=[fgbuster.component_model.Synchrotron(nu0=70)]

    A = fgbuster.MixingMatrix(*comp)
    A_ev = A.evaluator(nus)
    A_maxL = A_ev(np.array([-3]))

    new_sync_map=np.zeros(((len(nus), 3, 12*nside**2)))
    for i in range(len(nus)):
        new_sync_map[i]=A_maxL[i, 0]*maps_70GHz

    return new_sync_map
def create_dustd0(nside, nus):

    # Create 353 GHz dust maps
    sky = pysm3.Sky(nside=nside, preset_strings=['d0'])


    #comp=[fgbuster.component_model.Dust(nu0=353, beta_d=1.54, temp=20)]

    #A = fgbuster.MixingMatrix(*comp)
    #A_ev = A.evaluator(nus)
    #A_maxL = A_ev()

    new_dust_map=np.zeros(((len(nus), 3, 12*nside**2)))
    for i in range(len(nus)):
        maps_XXXGHz = sky.get_emission(nus[i]*u.GHz, None)*utils.bandpass_unit_conversion(nus[i]*u.GHz,None, u.uK_CMB)
        new_dust_map[i]=maps_XXXGHz.copy()

    return new_dust_map
def get_freqs_inter(edges, N):

    '''

    This function returns intermediate frequency for integration into bands from edges.

    '''
    if N == 1:
        return np.array([np.mean(edges)])
    freqs_inter=np.linspace(edges[0], edges[1], N)
    return freqs_inter
def create_dustd1(nside, nus):

    maps_dust = np.zeros(((len(nus), 3, 12*nside**2)))
    sky = pysm3.Sky(nside=nside, preset_strings=['d1'])
    for i, j in enumerate(nus):
        maps_dust[i] = sky.get_emission(j*u.GHz).to(
            getattr(u, 'uK_CMB'),
            equivalencies=u.cmb_equivalencies(j * u.GHz))

    return maps_dust
def eval_scaled_dust_dbmmb_map(nu_ref, nu_test, beta0, beta1, nubreak, nside, fsky, radec_center, temp):
    #def double-beta dust model
    analytic_expr = s4bi.double_beta_dust_FGB_Model(units='K_CMB')
    dbdust = AnalyticComponent(analytic_expr, nu0=nu_ref, h_over_k=constants.h * 1e9 / constants.k, temp=temp)
    scaling_factor = dbdust.eval(nu_test, beta0, beta1, nubreak)

    sky=pysm3.Sky(nside=nside, preset_strings=['d0'])
    dust_map_ref = np.zeros((3, 12*nside**2)) #this way the output is w/0 units!!
    dust_map_ref[0:3,:]=sky.get_emission(nu_ref*u.GHz, None)*utils.bandpass_unit_conversion(nu_ref*u.GHz,None, u.uK_CMB)


    map_test=dust_map_ref*scaling_factor

    #mask = s4bi.get_coverage(fsky, nside, center_radec=radec_center)

    return map_test
def get_scaled_dust_dbmmb_map(nu_ref, nu_vec, beta0, beta1, nubreak, nside, fsky, radec_center, temp):
    #eval at each freq. In this way it can be called both in the single-freq and the multi-freq case
    n_nu=len(nu_vec)
    dust_map= np.zeros((n_nu, 3, 12*nside**2))
    for i in range(n_nu):
        map_eval=eval_scaled_dust_dbmmb_map(nu_ref, nu_vec[i], beta0, beta1, nubreak, nside, fsky, radec_center, temp)
        #hp.mollview(map_eval[1])
        dust_map[i,:,:]=map_eval[:,:]
    return dust_map
def give_me_nus(nu, largeur, Nf):
    largeurq=largeur/Nf
    min=nu-largeur
    max=nu+largeur
    arr = np.linspace(min, max, Nf+1)
    mean_nu = np.zeros(Nf)

    for i in range(len(arr)-1):
        mean_nu[i]=np.mean(np.array([arr[i], arr[i+1]]))

    return mean_nu
def smoothing(maps, FWHMdeg, Nf, central_nus, verbose=True):
        """Convolve the maps to the FWHM at each sub-frequency or to a common beam if FWHMdeg is given."""
        fwhms = np.zeros(Nf)
        if FWHMdeg is not None:
            fwhms += FWHMdeg
        for i in range(Nf):
            if fwhms[i] != 0:
                maps[i, :, :] = hp.sphtfunc.smoothing(maps[i, :, :].T, fwhm=np.deg2rad(fwhms[i]),
                                                      verbose=verbose).T
        return fwhms, maps
def random_string(nchars):
    lst = [rd.choice(string.ascii_letters + string.digits) for n in range(nchars)]
    str = "".join(lst)
    return (str)
def get_multiple_nus(nu, bw, nf):
    nus=np.zeros(nf)
    edge=np.linspace(nu-(bw/2), nu+(bw/2), nf+1)
    for i in range(len(edge)-1):
        #print(i, i+1)
        nus[i]=np.mean([edge[i], edge[i+1]])
    return nus
def give_me_nu_fwhm_S4_2_qubic(nu, largeur, Nf, fwhmS4):

    def give_me_fwhm(nu, nuS4, fwhmS4):
        return fwhmS4*nuS4/nu

    largeurq=largeur/Nf
    min=nu*(1-largeur/2)
    max=nu*(1+largeur/2)
    arr = np.linspace(min, max, Nf+1)
    mean_nu = get_multiple_nus(nu, largeur, Nf)

    fwhm = give_me_fwhm(mean_nu, nu, fwhmS4)

    return mean_nu, fwhm
def get_fg_notconvolved(model, nu, nside=256):

    sky = pysm3.Sky(nside=nside, preset_strings=[model])
    maps = np.zeros(((len(nu), 3, 12*nside**2)))
    for indi, i in enumerate(nu) :
        maps[indi] = sky.get_emission(i*u.GHz, None)*utils.bandpass_unit_conversion(i*u.GHz,None, u.uK_CMB)

    return maps
def scaling_factor(maps, nus, analytic_expr, beta0, beta1, nubreak):
    nb_nus = maps.shape[0]
    newmaps = np.zeros(maps.shape)
    #print(sed1b)
    for i in range(nb_nus):
        _, sed1b = sed(analytic_expr, nus[i], beta1, beta1, nu0=nus[i], nubreak=nubreak)
        _, sed2b = sed(analytic_expr, nus[i], beta0, beta1, nu0=nus[i], nubreak=nubreak)
        print('nu is {} & Scaling factor is {:.8f}'.format(nus[i], sed2b))
        newmaps[i] = maps[i] * sed2b
    return newmaps, sed1b, sed2b
def sed(analytic_expr, nus, beta0, beta1, temp=20, hok=constants.h * 1e9 / constants.k, nubreak=200, nu0=200):
    sed_expr = AnalyticComponent(analytic_expr,
                             nu0=nu0,
                             beta_d0=beta0,
                             beta_d1=beta1,
                             temp=temp,
                             nubreak=nubreak,
                             h_over_k = hok)
    return nus, sed_expr.eval(nus)
def create_noisemaps(signoise, nus, nside, depth_i, depth_p, npix):
    np.random.seed(None)
    N = np.zeros(((len(nus), 3, npix)))
    for ind_nu, nu in enumerate(nus):

        sig_i=signoise*depth_i[ind_nu]/(np.sqrt(hp.nside2pixarea(nside, degrees=True)) * 60)
        N[ind_nu, 0] = np.random.normal(0, sig_i, 12*nside**2)

        sig_p=signoise*depth_p[ind_nu]/(np.sqrt(hp.nside2pixarea(nside, degrees=True)) * 60)
        N[ind_nu, 1] = np.random.normal(0, sig_p, 12*nside**2)*np.sqrt(2)
        N[ind_nu, 2] = np.random.normal(0, sig_p, 12*nside**2)*np.sqrt(2)

    return N
def give_me_maps_d1_modified(nus, nubreak, covmap, delta_b, nside, fix_temp=None, nside_index=256):

    maps_dust = np.ones(((len(nus), 3, 12*nside**2)))*hp.UNSEEN
    ind=np.where(covmap > 0)[0]
    sky = pysm3.Sky(nside=nside, preset_strings=['d1'])

    maps_dust = sky.get_emission(353*u.GHz, None)*utils.bandpass_unit_conversion(353*u.GHz,None, u.uK_CMB)
    map_index=np.array(sky.components[0].mbb_index)
    if fix_temp is not None:
        sky.components[0].mbb_temperature=fix_temp
        map_temperature=np.array(np.ones(12*nside**2)*sky.components[0].mbb_temperature)
    else:
        map_temperature=np.array(sky.components[0].mbb_temperature)

    if nside_index != 256 :
        map_temperature=hp.pixelfunc.ud_grade(map_temperature, nside_index)
        map_index=hp.pixelfunc.ud_grade(map_index, nside_index)
        map_temperature=hp.pixelfunc.ud_grade(map_temperature, 256)
        map_index=hp.pixelfunc.ud_grade(map_index, 256)

    #hp.mollview(map_temperature, sub=(1, 2, 1))
    #hp.mollview(map_index, sub=(1, 2, 2))
    #print(map_index.shape)

    # Evaluation of Mixing Matrix for 2 beta model
    comp2b=[fgbuster.component_model.Dust_2b(nu0=353)]
    A2b = fgbuster.MixingMatrix(*comp2b)
    A2b_ev = A2b.evaluator(nus)

    new_dust_map=np.ones(((len(nus), 3, 12*nside**2)))*hp.UNSEEN
    for i in ind :

        A2b_maxL = A2b_ev([np.array(map_index)[i]-delta_b, np.array(map_index)[i]+delta_b, nubreak, np.array(map_temperature)[i]])

        for j in range(len(nus)):
            new_dust_map[j, :, i]=A2b_maxL[j, 0]*maps_dust[:, i]

    return new_dust_map, [map_index, map_temperature]

class BImaps(object):

    def __init__(self, skyconfig, dict, r=0, nside=256):
        self.dict = dict
        self.skyconfig = skyconfig
        self.nus = self.dict['frequency']
        self.bw = self.dict['bandwidth']
        self.fwhm = self.dict['fwhm']
        self.fwhmdeg = self.fwhm/60
        self.depth_i = self.dict['depth_i']
        self.depth_p = self.dict['depth_p']
        self.edges=self.dict['edges']

        self.nside=nside
        self.npix = 12*self.nside**2
        self.lmax= 3 * self.nside
        self.r=r

        for k in skyconfig.keys():
            if k == 'cmb':
                self.seed = self.skyconfig['cmb']

    def get_cmb(self, coverage):

        okpix = coverage > (np.max(coverage) * float(0))
        maskpix = np.zeros(12*self.nside**2)
        maskpix[okpix] = 1
        Namaster = nam.Namaster(maskpix, lmin=40, lmax=2*self.nside-1, delta_ell=30)
        ell=np.arange(2*self.nside-1)

        global_dir=os.getcwd()
        binned_camblib = qc.bin_camblib(Namaster, global_dir + '/camblib.pkl', self.nside, verbose=False)

        cls = qc.get_Dl_fromlib(ell, self.r, lib=binned_camblib, unlensed=False)[0]
        mycls = qc.Dl2Cl_without_monopole(ell, cls)


        np.random.seed(self.seed)
        maps = hp.synfast(mycls.T, self.nside, verbose=False, new=True)
        return maps

    def get_sky(self, coverage):
        setting = []
        iscmb=False
        for k in self.skyconfig:
            if k == 'cmb' :
                iscmb=True
                maps = self.get_cmb(coverage)

                rndstr = random_string(10)
                hp.write_map('/tmp/' + rndstr, maps)
                cmbmap = pysm3.CMBMap(self.nside, map_IQU='/tmp/' + rndstr)
                os.remove('/tmp/' + rndstr)
                #setting.append(skyconfig[k])
            elif k=='dust':
                pass
            else:
                setting.append(self.skyconfig[k])

        sky = pysm3.Sky(nside=self.nside, preset_strings=setting)
        if iscmb:
            sky.add_component(cmbmap)

        return sky

    def getskymaps(self, same_resol=None, verbose=False, coverage=None, iib=1, noise=False, signoise=1., beta=[], fix_temp=None, nside_index=256):

        """

        """

        sky=self.get_sky(coverage)
        allmaps = np.zeros(((len(self.nus), 3, self.npix)))
        if same_resol is not None:
            self.fwhmdeg = np.ones(len(self.nus))*np.max(same_resol)

        for i in self.skyconfig.keys():
            if i == 'cmb':
                cmbmap = self.get_cmb(coverage)
                for j in range(len(self.nus)):
                    allmaps[j]+=cmbmap
                map_index=[]
            elif i == 'dust':
                dustmaps=np.zeros(((len(self.nus), 3, self.npix)))
                if self.skyconfig[i] == 'd0':
                    if verbose:
                        print('Model : {}'.format(self.skyconfig[i]))

                    for i in range(len(self.nus)):
                        print(self.edges[i])
                        nus_inter=get_freqs_inter(self.edges[i], iib)
                        print(nus_inter)
                        dustmaps_inter=create_dustd0(self.nside, nus_inter)#get_fg_notconvolved('d0', self.nus, nside=self.nside)
                        mean_dust_maps=np.mean(dustmaps_inter, axis=0)
                        dustmaps[i]=mean_dust_maps.copy()
                    allmaps+=dustmaps
                    map_index=[]


                elif self.skyconfig[i] == 'd02b':
                    if verbose:
                        print('Model : d02b -> Twos spectral index beta ({:.2f} and {:.2f}) with nu_break = {:.2f}'.format(beta[0], beta[1], beta[2]))

                    for i in range(len(self.nus)):
                        print(i)
                        #print(self.edges[i])
                        nus_inter=get_freqs_inter(self.edges[i], iib)
                        #print(nus_inter)
                        #add Elenia's definition
                        dustmaps_inter=create_dust_with_model2beta(self.nside, nus_inter, beta[0], beta[1], beta[2], temp=20, break_width=beta[3])
                        #print(dustmaps_inter.shape)
                        mean_dust_maps=np.mean(dustmaps_inter, axis=0)
                        dustmaps[i]=mean_dust_maps.copy()
                    allmaps+=dustmaps
                    map_index=[]
                else:
                    print('No dust')

            elif i == 'synchrotron':
                syncmaps=np.zeros(((len(self.nus), 3, self.npix)))
                print('Model : {}'.format(self.skyconfig[i]))

                for i in range(len(self.nus)):
                    nus_inter=get_freqs_inter(self.edges[i], iib)
                    print(nus_inter)
                    sync_maps_inter=create_sync(self.nside, nus_inter)
                    mean_sync_maps=np.mean(sync_maps_inter, axis=0)
                    syncmaps[i]=mean_sync_maps.copy()
                allmaps+=syncmaps
                map_index=[]

            else:
                print('No more components')
                #pass

        #hp.mollview(allmaps[0, 1])

        if same_resol != 0:
            for j in range(len(self.fwhmdeg)):
                if verbose:
                    print('Convolution to {:.2f} deg'.format(self.fwhmdeg[j]))
                allmaps[j] = hp.sphtfunc.smoothing(allmaps[j, :, :], fwhm=np.deg2rad(self.fwhmdeg[j]),verbose=False)


        if noise:
            noisemaps = create_noisemaps(signoise, self.nus, self.nside, self.depth_i, self.depth_p, self.npix)
            maps_noisy = allmaps+noisemaps

            if coverage is not None:
                pixok = coverage > 0
                maps_noisy[:, :, ~pixok] = hp.UNSEEN
                noisemaps[:, :, ~pixok] = hp.UNSEEN
                allmaps[:, :, ~pixok] = hp.UNSEEN

            return maps_noisy, allmaps, noisemaps
        return allmaps

class combinedmaps(object):

    def __init__(self, skyconfig, dict, r=0, nside=256, prop=[]):

        self.dict1=dict[0]
        self.dict2=dict[1]
        self.skyconfig = skyconfig
        self.nus=np.array(list(self.dict1['frequency'])+list(self.dict2['frequency']))
        self.n_freq=len(self.nus)

        self.depth1_i=self.dict1['depth_i']/np.sqrt(prop[0])
        self.depth1_p=self.dict1['depth_p']/np.sqrt(prop[0])
        self.depth2_i=self.dict2['depth_i']/np.sqrt(prop[1])
        self.depth2_p=self.dict2['depth_p']/np.sqrt(prop[1])
        self.edges1=self.dict1['edges']
        self.edges2=self.dict2['edges']
        self.edges=list(self.edges1)+list(self.edges2)

        self.depth_i=np.array(list(self.depth1_i)+list(self.depth2_i))
        self.depth_p=np.array(list(self.depth1_p)+list(self.depth2_p))

        if prop[0] == 1 :
            print('\n-------------------------------------')
            print("You're using a full S4 configuration")
            print('-------------------------------------\n')
            self.depth_i=self.dict1['depth_i']
            self.depth_p=self.dict1['depth_p']
            self.nus=self.dict1['frequency']
            self.n_freq=len(self.nus)
        elif prop[1] == 1:
            print('\n-------------------------------------')
            print("You're using a full BI configuration")
            print('-------------------------------------\n')
            self.depth_i=self.dict2['depth_i']
            self.depth_p=self.dict2['depth_p']
            self.nus=self.dict2['frequency']
            self.n_freq=len(self.nus)
        else:
            print("\nYou're using a combination -> {:.2f} % of CMB-S4 and {:.2f} % of BI \n".format(prop[0]*100, prop[1]*100))



        self.nside=nside
        self.npix = 12*self.nside**2
        self.lmax= 3 * self.nside
        self.r=r

        for k in skyconfig.keys():
            if k == 'cmb':
                self.seed = self.skyconfig['cmb']


    def get_cmb(self, coverage):

        okpix = coverage > (np.max(coverage) * float(0))
        maskpix = np.zeros(12*self.nside**2)
        maskpix[okpix] = 1
        Namaster = nam.Namaster(maskpix, lmin=40, lmax=2*self.nside-1, delta_ell=30)
        ell=np.arange(2*self.nside-1)

        global_dir=os.getcwd()
        binned_camblib = qc.bin_camblib(Namaster, global_dir + '/camblib.pkl', self.nside, verbose=False)

        cls = qc.get_Dl_fromlib(ell, self.r, lib=binned_camblib, unlensed=False)[0]
        mycls = qc.Dl2Cl_without_monopole(ell, cls)


        np.random.seed(self.seed)
        maps = hp.synfast(mycls.T, self.nside, verbose=False, new=True)
        return maps

    def get_sky(self, coverage):
        setting = []
        iscmb=False
        for k in self.skyconfig:
            if k == 'cmb' :
                iscmb=True
                maps = self.get_cmb(coverage)

                rndstr = random_string(10)
                hp.write_map('/tmp/' + rndstr, maps)
                cmbmap = pysm3.CMBMap(self.nside, map_IQU='/tmp/' + rndstr)
                os.remove('/tmp/' + rndstr)
                #setting.append(skyconfig[k])
            elif k=='dust':
                pass
            else:
                setting.append(self.skyconfig[k])

        sky = pysm3.Sky(nside=self.nside, preset_strings=setting)
        if iscmb:
            sky.add_component(cmbmap)

        return sky

    def getskymaps(self, same_resol=0, verbose=False, coverage=None, noise=False, beta=[], fix_temp=None, iib=1):

        signoise=1.
        """

        """

        #print(nside_index)

        sky=self.get_sky(coverage)
        allmaps = np.zeros(((len(self.nus), 3, self.npix)))
        allmaps=np.zeros(((len(self.nus), 3, self.npix)))

        for i in self.skyconfig.keys():
            if i == 'cmb':
                cmbmap = self.get_cmb(coverage)
                for j in range(len(self.nus)):
                    allmaps[j]+=cmbmap
                map_index=[]
            elif i == 'dust':
                dustmaps=np.zeros(((len(self.nus), 3, self.npix)))
                if self.skyconfig[i] == 'd0':
                    if verbose:
                        print('Model : {}'.format(self.skyconfig[i]))

                    for i in range(len(self.nus)):
                        print(self.edges[i])
                        nus_inter=get_freqs_inter(self.edges[i], iib)
                        print(nus_inter)
                        dustmaps_inter=create_dustd0(self.nside, nus_inter)#get_fg_notconvolved('d0', self.nus, nside=self.nside)
                        mean_dust_maps=np.mean(dustmaps_inter, axis=0)
                        dustmaps[i]=mean_dust_maps.copy()
                    allmaps+=dustmaps
                    map_index=[]

                elif self.skyconfig[i] == 'd02b':
                    if verbose:
                        print('Model : d02b -> Twos spectral index beta ({:.2f} and {:.2f}) with nu_break = {:.2f}'.format(beta[0], beta[1], beta[2]))

                    for i in range(len(self.nus)):
                        print(self.edges[i])
                        nus_inter=get_freqs_inter(self.edges[i], iib)
                        print(nus_inter)
                        #add Elenia's definition
                        dustmaps_inter=create_dust_with_model2beta(self.nside, nus_inter, beta[0], beta[1], beta[2], temp=20, break_width=beta[3])
                        #print(dustmaps_inter.shape)
                        mean_dust_maps=np.mean(dustmaps_inter, axis=0)
                        dustmaps[i]=mean_dust_maps.copy()
                    allmaps+=dustmaps
                    map_index=[]
                else:
                    print('No dust')

            elif i == 'synchrotron':
                syncmaps=np.zeros(((len(self.nus), 3, self.npix)))
                if verbose:
                    print('Model : {}'.format(self.skyconfig[i]))

                for i in range(len(self.nus)):
                    nus_inter=get_freqs_inter(self.edges[i], iib)
                    sync_maps_inter=create_sync(self.nside, nus_inter)
                    mean_sync_maps=np.mean(sync_maps_inter, axis=0)
                    syncmaps[i]=mean_sync_maps.copy()
                allmaps+=syncmaps
                map_index=[]

            else:
                print('No more components')


        if same_resol != 0:
            self.fwhmdeg = [same_resol]*self.n_freq
            for j in range(len(self.fwhmdeg)):
                if verbose:
                    print('Convolution to {:.2f} deg'.format(self.fwhmdeg[j]))
                allmaps[j] = hp.sphtfunc.smoothing(allmaps[j, :, :], fwhm=np.deg2rad(self.fwhmdeg[j]),verbose=False)


        if noise:
            noisemaps = create_noisemaps(signoise, self.nus, self.nside, self.depth_i, self.depth_p, self.npix)
            maps_noisy = allmaps+noisemaps

            if coverage is not None:
                pixok = coverage > 0
                maps_noisy[:, :, ~pixok] = hp.UNSEEN
                noisemaps[:, :, ~pixok] = hp.UNSEEN
                allmaps[:, :, ~pixok] = hp.UNSEEN

            return maps_noisy, allmaps, noisemaps
        return allmaps
