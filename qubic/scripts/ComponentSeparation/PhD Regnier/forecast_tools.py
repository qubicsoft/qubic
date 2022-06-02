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
import qubic
import os.path as op
CMB_CL_FILE = op.join('/pbs/home/m/mregnier/sps1/QUBIC+/forecast/Cls_Planck2018_%s.fits')

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
def _get_Cl_cmb(Alens, r):
    power_spectrum = hp.read_cl(CMB_CL_FILE%'lensed_scalar')[:,:4000]
    if Alens != 1.:
        power_spectrum[2] *= Alens
    if r:
        power_spectrum += r * hp.read_cl(CMB_CL_FILE
                                         %'unlensed_scalar_and_tensor_r1')[:,:4000]
    return power_spectrum
def get_fg_res(dfg, lmax=2*256-1):

    almBs = hp.map2alm(dfg, lmax=lmax, iter=10)[2]
    cl=hp.sphtfunc.alm2cl(almBs, lmax=lmax)/0.03    #Divide by seen sky fraction

    return cl
def confidence_interval(x,px):
    from getdist import densities
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

    """

    This object is a toolbox to make the forecast for a given instrument. You have to create your instrument by a dictionary and specify only frequencies and depth.

    ##########
    Parameters:
        - config : Dictionnary of your instrument
        - r : float, value of the Tensor-to-scalar ratio r that your are considering
        - Alens : float, value of the lensing
        - covmap : array, Coverage map of your instrument. If you're using a combination of instrument, that have the same coverage map.

    """

    def __init__(self, config, r, Alens, radec, fsky):

        self.config=config
        self.nus=self.config['frequency']
        self.depth_p=self.config['depth_p']
        self.nside=128
        self.npix=12*self.nside**2
        self.nfreqs=len(self.nus)
        self.w=None
        self.lmin=20#2
        self.lmax=220#180
        #self.lmax=355
        self.dl=20#15
        self.r=r
        self.Alens=Alens
        self.cc=0
        self.fsky=fsky
        self.radec=radec
        self.covmap=get_coverage(self.fsky, self.nside, center_radec=self.radec)


        maskpix = np.zeros(12*self.nside**2)
        self.pixok = self.covmap > 0
        maskpix[self.pixok] = 1
        self.Namaster = nam.Namaster(maskpix, lmin=self.lmin, lmax=self.lmax, delta_ell=self.dl)
        self.Namaster.fsky = self.fsky
        self.Namaster.ell_binned, _=self.Namaster.get_binning(self.nside)
        #self.Namaster.ell_binned=self.Namaster.ell_binned[:-1]
        #print(self.Namaster.ell_binned.shape)


        print('*********************************')
        print('********** Parameters ***********')
        print('*********************************')
        print()
        print('    Frequency [GHz] : {}'.format(self.nus))
        print('    Nside : {}'.format(self.nside))
        print('    Fsky : {}'.format(self.fsky))
        print('    Patch sky : {}'.format(self.radec))
        print('    lmin : {}'.format(self.lmin))
        print('    lmax : {}'.format(self.lmax))
        print('    dl : {}'.format(self.dl))
        print('    r : {}'.format(self.r))
        print('    Alens : {}'.format(self.Alens))
        print()
        print('*********************************')
        print('*********************************')
        print('*********************************\n')
    def residuals_fg_cl(self, cmb, cmb_est):
        self.pixok = self.covmap > 0
        res=cmb-cmb_est
        res[:, ~self.pixok]=0
        cl=get_fg_res(res, lmax=self.lmax)

        return np.arange(1, 513, 1), cl
    def instrument(self):

        instr=fgbuster.get_instrument('INSTRUMENT')
        instr.frequency=self.nus
        instr.depth_p=self.depth_p
        comp=[fgbuster.CMB(), fgbuster.Dust(nu0=145, temp=20)]#, fgbuster.Synchrotron(nu0=145)]

        return instr, comp
    def compute_cmb(self, seed):


        ell=np.arange(2*self.nside-1)
        mycls = _get_Cl_cmb(Alens=self.Alens, r=self.r)
        mycls[1, :]=np.zeros(4000)
        mycls[3, :]=np.zeros(4000)

        np.random.seed(seed)
        maps = hp.synfast(mycls, self.nside, verbose=False, new=True)
        mymaps=np.zeros((self.nus.shape[0], 3, self.npix))
        for i in range(self.nus.shape[0]):
            mymaps[i]=maps.copy()
        return mymaps
    def preset_fg(self, dustmodel, dict_dust):

        #print('********** Define foregrounds settings **********')
        #print()
        k=0
        for i in dict_dust.keys():
            if len(dict_dust)==k:
                break
            else:
                print('Preset of dust model : {} with {}'.format(i, dict_dust[i]))
                pysm3.sky.PRESET_MODELS[dustmodel][str(i)]=dict_dust[i]
            k+=1
        print()
    def compute_fg(self, dust_model, presets_dust=None, NSIDE_PATCH=4):

        #print("\nYou're computing thermal dust with model {}\n".format(dust_model))

        dustmaps=np.zeros((self.nfreqs, 3, self.npix))
        settings=[dust_model]#, 's1']
        self.preset_fg(dustmodel=dust_model, dict_dust=presets_dust)
        sky = pysm3.Sky(nside = self.nside, preset_strings=settings)

        if dust_model == 'd1' or dust_model == 'd2' or dust_model == 'd3' or dust_model == 'd6':
            sky.components[0].mbb_temperature=20*sky.components[0].mbb_temperature.unit     # fix temp at 20 K across the sky
            #print('Downgrade pixelization of spectral indices at nside = {}'.format(NSIDE_PATCH))

            for spectral_param in [sky.components[0].mbb_index]:#, sky.components[1].pl_index]:
                spectral_param[:] = hp.ud_grade(hp.ud_grade(spectral_param.value, NSIDE_PATCH),
                                    self.nside) * spectral_param.unit

        for j in range(self.nfreqs):
            dustmaps[j]=np.array(sky.get_emission(self.nus[j]*u.GHz)*utils.bandpass_unit_conversion(self.nus[j]*u.GHz, None, u.uK_CMB))

        return dustmaps
    def get_clBB(self, map1, map2):


        leff, cls, _ = self.Namaster.get_spectra(map1, map2=map2,
                                 purify_e=False,
                                 purify_b=True,
                                 w=self.w,
                                 verbose=False,
                                 beam_correction=None,
                                 pixwin_correction=False)

        return leff, cls[:, 2]
    def _get_noise(self):

        np.random.seed(None)

        N = np.zeros(((self.nfreqs, 3, self.npix)))
        depth_i = self.config['depth_i']
        depth_p = self.config['depth_p']

        for ind_nu, nu in enumerate(self.nus):

            sig_i=depth_i[ind_nu]/(np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60)
            N[ind_nu, 0] = np.random.normal(0, sig_i, 12*self.nside**2)

            sig_p=depth_p[ind_nu]/(np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60)
            N[ind_nu, 1] = np.random.normal(0, sig_p, 12*self.nside**2)*np.sqrt(2)
            N[ind_nu, 2] = np.random.normal(0, sig_p, 12*self.nside**2)*np.sqrt(2)

        return N
    def ana_likelihood(self, rv, leff, fakedata, errors, model, prior,mylikelihood=mcmc.LogLikelihood, covariance_model_funct=None):

        ll = mylikelihood(xvals=leff, yvals=fakedata, errors=errors,
                                model = model, flatprior=prior, covariance_model_funct=covariance_model_funct, nbins=len(leff))

        like = np.zeros_like(rv)
        for i in range(len(rv)):
            like[i] = np.exp(ll([rv[i]]))
            #print(rv[i],ll([rv[i]]),like[i])
        maxL = rv[like == np.max(like)]
        cumint = scipy.integrate.cumtrapz(like, x=rv)
        cumint = cumint / np.max(cumint)
        onesigma = np.interp(0.68, cumint, rv[1:])

        return like, cumint, onesigma
    def explore_like(self, leff, mcl_noise, errors, rv, cov=None, verbose=False, sample_variance=True):

        lbinned, b = self.Namaster.get_binning(self.nside)

        def myclth(ell, r):
            clth = _get_Cl_cmb(Alens=self.Alens, r=r)[2, ell.astype(int)-1]
            return clth

        def myBBth(ell, r):
            clBB = cl2dl(ell, _get_Cl_cmb(Alens=self.Alens, r=r)[2, ell.astype(int)-1])
            return clBB

        fakedata = mcl_noise.copy()

        if sample_variance:
            covariance_model_funct = self.Namaster.knox_covariance
        else:
            covariance_model_funct = None

        like, cumint, allrlim = self.ana_likelihood(rv, leff, fakedata,
                                                errors,
                                                myBBth, [[0,1]],
                                               covariance_model_funct=covariance_model_funct)


        return like, cumint, allrlim
    def RUN_MC(self, N, dust_model, dict_dust, nside_fgb):

        instr, comp = self.instrument()
        clBB=np.zeros((N, len(self.Namaster.ell_binned)))
        seed=np.zeros(N)

        cmb_est=np.zeros((N, 2, np.sum(self.pixok)))
        index_est=np.zeros((N, 12*nside_fgb**2))
        cl_fg=np.zeros((N, self.lmax+1))
        for i in range(N):
            print('Iteration {:.0f} over {:.0f}'.format(i+1, N))

            s=np.random.randint(10000000)
            seed[i]=s
            #print('\nSeed of the CMB is {}\n'.format(s))
            mycmb=self.compute_cmb(seed=s)
            mycmb[:, :, ~self.pixok]=0

            maps_for_namaster=np.zeros((2, 3, self.npix))
            myfg=self.compute_fg(dust_model, presets_dust=dict_dust, NSIDE_PATCH=nside_fgb)
            for nb in range(2):

                maps_for_fgb=np.zeros((self.nfreqs, 3, self.npix))
                maps_for_fgb+=myfg.copy()
                maps_for_fgb+=mycmb.copy()
                noise=self._get_noise()
                #print("RMS of noise {:.0f} : {:.6f} muK".format(nb+1, np.std(noise[0, 1, self.pixok])))
                maps_for_fgb+=noise.copy()

                maps_for_fgb[:, :, ~self.pixok]=hp.UNSEEN


                r=fgbuster.separation_recipes.basic_comp_sep(comp,
                                                             instr,
                                                             maps_for_fgb[:, 1:, :], nside=nside_fgb, tol=1e-18, method='TNC')

                cmb_est[i]=r.s[0, :, self.pixok].T.copy()
                index_est[i]=r.x[0].copy()
                maps_for_namaster[nb, 1:]=r.s[0].copy()

            #print('Computing foregrounds residuals in Cl space...\n')

            ell, cl_fg[i] = self.residuals_fg_cl(mycmb[0], np.mean(maps_for_namaster, axis=0))
            #print('First bins of cl_fg -> {}'.format(cl_fg[i, 10:30]))
            #print('Done\n')
            #print("\nComputing cross-spectra...")
            leff, clBB[i] = self.get_clBB(maps_for_namaster[0], maps_for_namaster[1])
            #print('Estimated power spectrum : {}'.format(clBB[i]))
        mydata=np.mean(clBB, axis=0)
        myerr=np.std(clBB, axis=0)
        rv=np.linspace(0, 0.2, 10000)

        print('\nComputing cosmological parameters...\n')

        like, _, allrlim = self.explore_like(leff[:-1], mydata[:-1], myerr[:-1], rv, cov=self.covmap, verbose=False, sample_variance=True)

        lim, szlim = confidence_interval(rv,like)
        CL95=lim[1][1]
        maxL=rv[like == np.max(like)][0]
        sigma=lim[0][1]

        return leff, clBB, like, maxL, sigma, CL95, cmb_est, seed, cl_fg, index_est
