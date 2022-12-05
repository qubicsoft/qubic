import os.path as op
import numpy as np
import pylab as pl
import healpy as hp
import scipy as sp
from fgbuster.algebra import comp_sep, W_dBdB, W_dB, W, _mmm, _utmv, _mmv
from fgbuster.mixingmatrix import MixingMatrix
from fgbuster.observation_helpers import standardize_instrument, get_instrument
import qubicplus
import fgbuster
import pysm3
import pysm3.units as u
from pysm3 import utils
from qubic import NamasterLib as nam
from qubic import mcmc
import os
from qubic import camb_interface as qc
import scipy

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

class Forecast(object):

    def __init__(self, instrument, components, d_fgs, lmin, lmax):

        self.instrument = standardize_instrument(instrument)
        self.nside = hp.npix2nside(d_fgs.shape[-1])
        self.n_stokes = d_fgs.shape[1]
        self.n_freqs = d_fgs.shape[0]
        self.invN = np.diag(hp.nside2resol(self.nside, arcmin=True) / (instrument.depth_p))**2
        self.mask = d_fgs[0, 0, :] != 0.
        self.fsky = self.mask.astype(float).sum() / self.mask.size
        self.ell = np.arange(lmin, lmax+1)
        self.lmin=lmin
        self.lmax=lmax
        self.d_fgs=d_fgs
        #print('fsky = ', self.fsky)


        print('======= ESTIMATION OF SPECTRAL PARAMETERS =======')
        self.A = MixingMatrix(*components)
        self.A_ev = self.A.evaluator(instrument.frequency)
        self.A_dB_ev = self.A.diff_evaluator(instrument.frequency)

        x0 = np.array([x for c in components for x in c.defaults])
        if self.n_stokes == 3:  # if T and P were provided, extract P
            d_comp_sep = d_fgs[:, 1:, :]
        else:
            d_comp_sep = d_fgs

        self.res = comp_sep(self.A_ev, d_comp_sep.T, self.invN, self.A_dB_ev, self.A.comp_of_dB, x0)
        self.res.params = self.A.params
        #res.s = res.s.T
        self.A_maxL = self.A_ev(self.res.x)
        self.A_dB_maxL = self.A_dB_ev(self.res.x)
        self.A_dBdB_maxL = self.A.diff_diff_evaluator(self.instrument.frequency)(self.res.x)

        print('res.x = ', self.res.x)

    def _get_Cl_noise(self):
        i_cmb = self.A.components.index('CMB')
        try:
            bl = np.array([hp.gauss_beam(np.radians(b/60.), lmax=self.lmax)
                       for b in self.instrument.fwhm])
        except AttributeError:
            bl = np.ones((len(self.instrument.frequency), self.lmax+1))

        nl = (bl / np.radians(self.instrument.depth_p/60.)[:, np.newaxis])**2
        AtNA = np.einsum('fi, fl, fj -> lij', self.A_maxL, nl, self.A_maxL)
        inv_AtNA = np.linalg.inv(AtNA)
        return inv_AtNA.swapaxes(-3, -1)[i_cmb, i_cmb, self.lmin:]

    def _get_cls_fg(self):

         print ('======= COMPUTATION OF CL_FGS =======')
         if self.n_stokes == 3:
             d_spectra = self.d_fgs
         else:  # Only P is provided, add T for map2alm
             d_spectra = np.zeros((self.n_freqs, 3, self.d_fgs.shape[2]), dtype=self.d_fgs.dtype)
             d_spectra[:, 1:] = self.d_fgs

         # Compute cross-spectra
         almBs = [hp.map2alm(freq_map, lmax=self.lmax, iter=10)[2] for freq_map in d_spectra]
         Cl_fgs = np.zeros((self.n_freqs, self.n_freqs, self.lmax+1), dtype=self.d_fgs.dtype)
         for f1 in range(self.n_freqs):
             for f2 in range(self.n_freqs):
                 if f1 > f2:
                     Cl_fgs[f1, f2] = Cl_fgs[f2, f1]
                 else:
                     Cl_fgs[f1, f2] = hp.alm2cl(almBs[f1], almBs[f2], lmax=self.lmax)

         Cl_fgs = Cl_fgs[..., self.lmin:] / self.fsky
         return Cl_fgs

    def _get_sys_stat_residuals(self):
        Cl_fgs=self._get_cls_fg()
        i_cmb = self.A.components.index('CMB')
        print('======= ESTIMATION OF STAT AND SYS RESIDUALS =======')

        W_maxL = W(self.A_maxL, invN=self.invN)[i_cmb, :]
        W_dB_maxL = W_dB(self.A_maxL, self.A_dB_maxL, self.A.comp_of_dB, invN=self.invN)[:, i_cmb]
        W_dBdB_maxL = W_dBdB(self.A_maxL, self.A_dB_maxL, self.A_dBdB_maxL,
                             self.A.comp_of_dB, invN=self.invN)[:, :, i_cmb]
        V_maxL = np.einsum('ij,ij...->...', self.res.Sigma, W_dBdB_maxL)

        # Check dimentions
        assert ((self.n_freqs,) == W_maxL.shape == W_dB_maxL.shape[1:]
                           == W_dBdB_maxL.shape[2:] == V_maxL.shape)
        assert (len(self.res.params) == W_dB_maxL.shape[0]
                                == W_dBdB_maxL.shape[0] == W_dBdB_maxL.shape[1])

        # elementary quantities defined in Stompor, Errard, Poletti (2016)
        Cl_xF = {}
        Cl_xF['yy'] = _utmv(W_maxL, Cl_fgs.T, W_maxL)  # (ell,)
        Cl_xF['YY'] = _mmm(W_dB_maxL, Cl_fgs.T, W_dB_maxL.T)  # (ell, param, param)
        Cl_xF['yz'] = _utmv(W_maxL, Cl_fgs.T, V_maxL )  # (ell,)
        Cl_xF['Yy'] = _mmv(W_dB_maxL, Cl_fgs.T, W_maxL)  # (ell, param)
        Cl_xF['Yz'] = _mmv(W_dB_maxL, Cl_fgs.T, V_maxL)  # (ell, param)

        # bias and statistical foregrounds residuals
        #self.res.noise = Cl_noise
        self.res.bias = Cl_xF['yy'] + 2 * Cl_xF['yz']  # S16, Eq 23
        self.res.stat = np.einsum('ij, lij -> l', self.res.Sigma, Cl_xF['YY'])  # E11, Eq. 12
        self.res.var = self.res.stat**2 + 2 * np.einsum('li, ij, lj -> l', # S16, Eq. 28
                                              Cl_xF['Yy'], self.res.Sigma, Cl_xF['Yy'])

        return self.res.bias, self.res.stat, self.res.var
class ForecastMC(object):

    def __init__(self, config, skyconfig, r, covmap, beta, nside):

        self.config=config
        self.skyconfig=skyconfig
        self.r=r
        self.covmap=covmap
        self.beta=beta
        self.nside=nside
        _, self.fg, _=qubicplus.BImaps(self.skyconfig, self.config, r=self.r, nside=256).getskymaps(same_resol=0,
                                                          verbose=False,
                                                          coverage=self.covmap,
                                                          iib=1,
                                                          noise=True,
                                                          signoise=1.,
                                                          beta=self.beta,
                                                          fix_temp=20,
                                                          nside_index=16)

        if self.nside!=256:
            self.newfg=np.zeros((self.fg.shape[0], self.fg.shape[0], 12*self.nside**2))
            for i in range(self.fg.shape[0]):
                self.newfg[i]=hp.pixelfunc.ud_grade(self.fg[i], self.nside)
        else:
            self.newfg=self.fg.copy()
        print(self.newfg.shape)


    def _get_instr_nsamples(self, N_SAMPLE_BAND):

        if len(self.config['frequency'])==9:
            prop=0
        else:
            prop=1

        if prop == 0 or prop == 1:
            freq_maps=self.config['frequency']
            bandpasses = self.config['bandwidth']
            depth_i=self.config['depth_i']
            depth_p=self.config['depth_p']

        else:
            pass
            '''
            config1=config[0]
            config2=config[1]
            frac=[1-prop, prop]
            freq_maps=list(config1['frequency'][ind_deco[0]:ind_deco[1]])+list(config2['frequency'][ind_deco[0]:ind_deco[1]])
            depth1_i=config1['depth_i']/np.sqrt(frac[0])
            depth1_p=config1['depth_p']/np.sqrt(frac[0])
            depth2_i=config2['depth_i']/np.sqrt(frac[1])
            depth2_p=config2['depth_p']/np.sqrt(frac[1])

            depth_i=np.array(list(depth1_i)+list(depth2_i))
            depth_p=np.array(list(depth1_p)+list(depth2_p))
            bandpasses = list(config1['bandwidth'])+list(config2['bandwidth'])
            '''
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

        if N_SAMPLE_BAND == 1:
            instrument.frequency = freq_maps
            instrument.depth_i=depth_i
            instrument.depth_p=depth_p
            instrument.fwhm=np.zeros(len(freq_maps))
        if N_SAMPLE_BAND > 1:
            instrument.frequency = new_list_of_freqs
            instrument.depth_i=depth_i
            instrument.depth_p=depth_p
            instrument.fwhm=np.zeros(len(freq_maps))

        return instrument
    def _get_noise(self, nside_out):

        if len(self.config['frequency'])==9:
            prop=0
        else:
            prop=1

        nside=256
        npix=12*nside**2

        np.random.seed(None)

        if prop == 0 or prop == 1:
            nus=self.config['frequency']
            N = np.zeros(((len(nus), 3, npix)))
            depth_i = self.config['depth_i']
            depth_p = self.config['depth_p']

        else:
            pass
            '''
            frac=[1-prop, prop]
            config1=self.config[0]
            config2=self.config[1]
            print(frac)
            nus=list(config1['frequency'])+list(config2['frequency'])
            N = np.zeros(((len(nus), 3, npix)))
            depth1_i=config1['depth_i']/np.sqrt(frac[0])
            depth1_p=config1['depth_p']/np.sqrt(frac[0])
            depth2_i=config2['depth_i']/np.sqrt(frac[1])
            depth2_p=config2['depth_p']/np.sqrt(frac[1])

            depth_i=np.array(list(depth1_i)+list(depth2_i))
            depth_p=np.array(list(depth1_p)+list(depth2_p))
            '''

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
    def _get_inputs(self, covmap, beta, nside_index):
        _, map, _=qubicplus.BImaps(self.skyconfig, self.config, r=self.r, nside=256).getskymaps(same_resol=0,
                                                          verbose=False,
                                                          coverage=covmap,
                                                          iib=1,
                                                          noise=True,
                                                          signoise=1.,
                                                          beta=beta,
                                                          fix_temp=20,
                                                          nside_index=nside_index)

        return map
    def _get_cmb(self, covmap, seed):
        _, map, _=qubicplus.BImaps({'cmb':seed}, self.config, r=self.r, nside=256).getskymaps(same_resol=0,
                                                          verbose=False,
                                                          coverage=covmap,
                                                          iib=1,
                                                          noise=True,
                                                          signoise=1.,
                                                          beta=[],
                                                          fix_temp=20,
                                                          nside_index=0)

        return map
    def _define_defaults_comp(self, comp, x0):
        comp[1].defaults=x0
    def _get_comp_instr(self, nu0, fit, x0):

        comp=[]
        comp.append(fgbuster.component_model.CMB())
        for i in self.skyconfig.keys():
            if i == 'dust':
                if fit=='d0' or fit == 'd1':
                    comp.append(fgbuster.component_model.Dust(nu0=nu0, temp=20))
                else:
                    comp.append(fgbuster.component_model.Dust_2b(nu0=nu0, temp=20, break_width=0.3))
            elif i == 'synchrotron':
                comp.append(fgbuster.component_model.Synchrotron(nu0=nu0))

        self._define_defaults_comp(comp, x0)
        if len(self.config['frequency']) == 9:
            instr = get_instrument('CMBS4')
        else:
            instr = get_instrument('CMBS4BI')

        return comp, instr
    def _get_components_est(self, comp, instr, data, beta, pixok):
        print(beta, beta.shape)
        """

        This function estimate components from MixingMatrix of fgbuster with estimated parameters

        """
        nside=256
        A=fgbuster.mixingmatrix.MixingMatrix(*comp)
        A_ev=A.evaluator(self.config['frequency'])
        invN = np.diag(hp.nside2resol(nside, arcmin=True) / (instr.depth_p))**2
        newmaps=np.zeros((len(comp), 3, 12*nside**2))

        if beta.shape[-1]>=12:
            ind=np.where(pixok != 0)[0]
            maps_separe=np.zeros((len(comp), 2, 12*nside**2))
            invN = np.diag(hp.nside2resol(nside, arcmin=True) / (instr.depth_p))**2
            for i in range(len(ind)):
                A_maxL=A_ev(np.array(beta)[:, ind[i]])
                maps_separe[:, :, ind[i]]=fgbuster.algebra.Wd(A_maxL, data[:, :, ind[i]].T, invN=invN).T
        else:
            A_maxL=A_ev(beta)
            maps_separe=fgbuster.algebra.Wd(A_maxL, data.T, invN=invN).T
        newmaps[:, 1:, :]=maps_separe.copy()

        return newmaps
    def _get_clBB(self, map1, map2, Namaster):
        w=None
        leff, cl, _ = Namaster.get_spectra(map1, map2=map2,
                                     purify_e=False,
                                     purify_b=True,
                                     w=w,
                                     verbose=False,
                                     beam_correction=None,
                                     pixwin_correction=True)

        clBB=cl[:, 2]

        return leff, clBB
    def ana_likelihood(self, rv, leff, fakedata, errors, model, prior,mylikelihood=mcmc.LogLikelihood, covariance_model_funct=None, otherp=None):
        ll = mylikelihood(xvals=leff, yvals=fakedata, errors=errors,model = model, flatprior=prior, covariance_model_funct=covariance_model_funct)

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
    def explore_like(self, leff, cl, errors, lmin, dl, cc, rv, otherp=None, cov=None, sample_variance=False):

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
            Namaster.fsky = 0.03

        #     print('Fsky: {}'.format(Namaster.fsky))
        lbinned, b = Namaster.get_binning(nside)

        ### Bibnning CambLib
        #     binned_camblib = qc.bin_camblib(Namaster, '../../scripts/QubicGeneralPaper2020/camblib.pickle',
        #                                     nside, verbose=False)
        global_dir='/pbs/home/m/mregnier/sps1/QUBIC+/d0/cls'
        binned_camblib = qc.bin_camblib(Namaster, global_dir+'/camblib.pkl', nside, verbose=False)


        ### Redefine the function for getting binned Cls
        def myclth(ell,r):
            clth = qc.get_Dl_fromlib(ell, r, lib=binned_camblib, unlensed=True)[1]
            return clth
        allfakedata = myclth(leff, 0.)
        #lll, totDL, unlensedCL = qc.get_camb_Dl(lmax=3*256, r=0)
        ### And we need a fast one for BB only as well
        def myBBth(ell, r):
            clBB = qc.get_Dl_fromlib(ell, r, lib=binned_camblib, unlensed=True, specindex=2)[1]
            return clBB

        ### Fake data
        fakedata = cl.copy()#myBBth(leff, 0.)


        if sample_variance:
            covariance_model_funct = Namaster.knox_covariance
        else:
            covariance_model_funct = None

        if otherp is None:
            like, cumint, allrlim, maxL = self.ana_likelihood(rv, leff, fakedata,errors,myBBth, [[0,1]],covariance_model_funct=covariance_model_funct)
        else:
            like, cumint, allrlim, other, maxL = self.ana_likelihood(rv, leff, fakedata, errors, myBBth, [[0,1]],covariance_model_funct=covariance_model_funct, otherp=otherp)

        if otherp is None:
            return like, cumint, allrlim, maxL
        else:
            return like, cumint, allrlim, other, maxL

    def _RUN_MC(self, N, fit, x0, covmap, beta=[], nside_param=0, fixcmb=True, noiseless=False):

        maskpix = np.zeros(12*256**2)
        pixok = covmap > 0
        maskpix[pixok] = 1
        Namaster = nam.Namaster(maskpix, lmin=21, lmax=355, delta_ell=35)

        comp, _ = self._get_comp_instr(nu0=145, fit=fit, x0=x0)
        instr=self._get_instr_nsamples(N_SAMPLE_BAND=10)
        inputs=self.newfg.copy()
        clBB=np.zeros((N, 9))
        clres=np.zeros((N, 9))
        components=np.zeros((N, 3, 3, np.sum(pixok)))
        if nside_param != 0:
            beta=np.zeros((N, fgbuster.MixingMatrix(*comp).n_param, 12*256**2))
        else:
            beta=np.zeros((N, fgbuster.MixingMatrix(*comp).n_param))

        print('********** Separation ************')

        for i in range(N):

            if fixcmb:
                seed=42
            else:
                seed=np.random.randint(1000000)

            print(seed)
            cmb=self._get_cmb(self.covmap, seed)
            if self.nside!=256:
                newcmb=np.zeros((self.newfg.shape[0], self.newfg.shape[0], 12*self.nside**2))
                for i in range(self.fg.shape[0]):
                    newcmb[i]=hp.pixelfunc.ud_grade(cmb[i], self.nside)
            else:
                newcmb=cmb.copy()

            data=inputs+newcmb.copy()
            print(data.shape)

            r1, components1 = self._separation(comp, instr, data, self.covmap, beta, nside_param, fit, x0, noiseless)
            r2, components2 = self._separation(comp, instr, data, self.covmap, beta, nside_param, fit, x0, noiseless)

            beta[i]=r1.copy()

            components1[:, :, ~pixok]=0
            components2[:, :, ~pixok]=0

            components[i]=components1[:, :, pixok].copy()

            leff, clBB[i]=self._get_clBB(components1[0], components2[0], Namaster)


            #res1=cmb[0]-components1[0]
            #res2=cmb[0]-components2[0]
            #res1[:, ~pixok]=0
            #res2[:, ~pixok]=0
            #leff, clres[i]=self._get_clBB(res1, res2, Namaster)
            #print(cmb[0, 1, pixok]-components1[0, 1, pixok])

        return leff, clBB, beta, components


    def _separation(self, comp, instr, inputs, covmap, beta, nside_param, fit, x0, noiseless):


        pixok=covmap>0
        nside=256
        noise=self._get_noise(nside_out=256)
        if noiseless:
            data_noisy=inputs.copy()#+noise.copy()
        else:
            data_noisy=inputs+noise.copy()
        if nside_param != 0 :
            newdata=data_noisy[:, 1:, :].copy()
            newdata[:, :, ~pixok]=hp.UNSEEN
        else:
            newdata=data_noisy[:, 1:, pixok].copy()

        r=fgbuster.separation_recipes.basic_comp_sep(comp, instr, newdata, tol=1e-18, nside=nside_param)
        if nside_param != 0:
            newr=hp.pixelfunc.ud_grade(r.x, 256)
        else:
            newr=r.x
        components=self._get_components_est(comp, instr, data_noisy[:, 1:, :], newr, pixok)
        components[:, :, ~pixok]=0
        #print(components.shape)

        return newr, components
