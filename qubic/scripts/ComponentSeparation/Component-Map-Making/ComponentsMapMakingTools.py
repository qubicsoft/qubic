import qubic
from pyoperators import *
from pysimulators import *
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
import component_model as c
import mixing_matrix as mm
import healpy as hp
import numpy as np
import time
from scipy.optimize import minimize





def get_allA(nc, nf, npix, beta, nus, comp):
    # Initialize arrays to store mixing matrix values
    allA = np.zeros((beta.shape[0], nf, nc))
    allA_pix = np.zeros((npix, nf, nc))

    # Loop through each element of beta to calculate mixing matrix
    for i in range(beta.shape[0]):
        allA[i] = get_mixingmatrix(beta[i], nus, comp)

    # Check if beta and npix are equal
    if beta.shape[0] != npix:
        # Upgrade resolution if not equal
        for i in range(nf):
            for j in range(nc):
                allA_pix[:, i, j] = hp.ud_grade(allA[:, i, j], hp.npix2nside(npix))
        # Return upgraded mixing matrix
        return allA_pix
    else:
        # Return original mixing matrix
        return allA
def get_mixing_operator_verying_beta(nc, nside, A):

    npix = 12*nside**2
    
    D = BlockDiagonalOperator([DenseBlockDiagonalOperator(A, broadcast='rightward', shapein=(npix, nc)),
                           DenseBlockDiagonalOperator(A, broadcast='rightward', shapein=(npix, nc)),
                           DenseBlockDiagonalOperator(A, broadcast='rightward', shapein=(npix, nc))], new_axisin=0, new_axisout=2)
    return D
def get_mixingmatrix(beta, nus, comp, active=False):
    A = mm.MixingMatrix(*comp)
    if active:
        i = A.components.index('COLine')
        comp[i] = c.COLine(nu=comp[i].nu, active=True)
        A = mm.MixingMatrix(*comp)
        A_ev = A.evaluator(nus)
        if beta.shape[0] == 0:
            A_ev = A_ev()
        else:
            A_ev = A_ev(beta)
            for ii in range(len(comp)):
                #print('ii : ', ii)
                if ii == i:
                    pass
                else:
                    #print('to zero', ii)
                    A_ev[0, ii] = 0
            #print(i, A, A.shape)    
    
    
    else:
        A_ev = A.evaluator(nus)
        if beta.shape[0] == 0:
            A_ev = A_ev()
        else:
            A_ev = A_ev(beta)
        try:
            
            i = A.components.index('COLine')
            A_ev[0, i] = 0
        except:
            pass
    return A_ev
def get_mixing_operator(beta, nus, comp, nside, active=False):
    
    """
    This function returns a mixing operator based on the input parameters: beta and nus.
    The mixing operator is either a constant operator, or a varying operator depending on the input.
    """

    nc = len(comp)
    if beta.shape[0] != 1 and beta.shape[0] != 2:
        
        nside_fit = hp.npix2nside(beta.shape[0])
    else:
        nside_fit = 0

    # Check if the length of beta is equal to the number of channels minus 1
    if nside_fit == 0: # Constant indice on the sky
        #beta = np.mean(beta)

        # Get the mixing matrix
        A = get_mixingmatrix(beta, nus, comp, active)
        
        # Get the shape of the mixing matrix
        _, nc = A.shape
        
        # Create a ReshapeOperator
        R = ReshapeOperator(((1, 12*nside**2, 3)), ((12*nside**2, 3)))
        
        # Create a DenseOperator with the first row of A
        D = DenseOperator(A[0], broadcast='rightward', shapein=(nc, 12*nside**2, 3), shapeout=(1, 12*nside**2, 3))

    else: # Varying indice on the sky
        
        # Get all A matrices nc, nf, npix, beta, nus, comp
        A = get_allA(nc, 1, 12*nside**2, beta, nus, comp)
        
        # Get the varying mixing operator
        D = get_mixing_operator_verying_beta(nc, nside, A)

    return D
def give_me_intercal(D, d):
    return 1/np.sum(D[:]**2, axis=1) * np.sum(D[:] * d[:], axis=1)
def get_gain_detector(H, mysolution, tod, Nsamples, Ndets, number_FP):
    if number_FP == 2:
        R = ReshapeOperator((2 * Nsamples * Ndets), (2, Ndets, Nsamples))
        r_ = ReshapeOperator((Nsamples * Ndets), (Ndets, Nsamples))
        data_qu = R(tod[:(2 * Nsamples * Ndets)]).copy()
        data150_qu, data220_qu = data_qu[0], data_qu[1]
        
        H150 = H.operands[0].operands[0]
        H220 = H.operands[0].operands[1]
        data150_s_qu = r_(H150(mysolution)).copy()
        data220_s_qu = r_(H220(mysolution)).copy()
            
        I150 = give_me_intercal(data150_s_qu, data150_qu)
        I220 = give_me_intercal(data220_s_qu, data220_qu)
        I = np.array([I150, I220])
    else:
        R = ReshapeOperator((Nsamples * Ndets), (Ndets, Nsamples))
        data_qu = R(tod[:(Nsamples * Ndets)]).copy()
        H150 = R * H.operands[0]
        data_s_qu = H150(mysolution).copy()
        I = give_me_intercal(data_s_qu, data_qu)
    return I
def get_dictionary(nsub, nside, pointing, band):
    dictfilename = 'dicts/pipeline_demo.dict'
    
    # Read dictionary chosen
    d = qubic.qubicdict.qubicDict()
    d.read_from_file(dictfilename)
    d['nf_recon'] = nsub
    d['nf_sub'] = nsub
    d['nside'] = nside
    d['RA_center'] = 100
    d['DEC_center'] = -157
    center = qubic.equ2gal(d['RA_center'], d['DEC_center'])
    d['effective_duration'] = 3
    d['npointings'] = pointing
    d['filter_nu'] = int(band*1e9)
    d['photon_noise'] = True
    d['config'] = 'FI'
    d['MultiBand'] = True
    
    return d, center

def put_zero_outside_patch(obs, seenpix, nside, nus):
    R = ReshapeOperator(obs.shape, (len(nus), 12*nside**2, 3))
    obs_maps = R(obs)
    obs_maps[:, ~seenpix, :] = 0
    return R.T(obs_maps)
def myChi2(beta, betamap, Hi, solution, data, patch_id, acquisition):
    newbeta = betamap.copy()
    if patch_id is not None:
        newbeta[patch_id] = beta.copy()
    else:
        newbeta = beta.copy()

    Hi = acquisition.update_A(Hi, newbeta)

    fakedata = Hi(solution)
    #print('chi2 ', np.sum((fakedata - data)**2))
    return np.sum((fakedata - data)**2)

def normalize_tod(tod, external_nus, npix):

    R = ReshapeOperator(len(external_nus)*npix*3, (len(external_nus), npix, 3))
    R1 = ReshapeOperator((npix, 3), npix*3)
    
    tod_external = R(tod)   # size -> (Nf, Npix, 3)
    tod_normalized = []
    
    for ii, i in enumerate(external_nus):
        tod_external[ii] = tod_external[ii]/tod_external[ii].max()
        
        tod_normalized = np.r_[tod_normalized, R1(tod_external[ii])]

    return tod_normalized

class Spectra:
    
    def __init__(self, lmin, lmax, dl, r=0, Alens=1, icl=2, CMB_CL_FILE=None):
        self.lmin = lmin
        self.lmax = lmax
        self.icl = icl
        self.r = r
        self.dl = dl
        self.CMB_CL_FILE = CMB_CL_FILE
        self.Alens = Alens
        self.ell_theo = np.arange(2, self.lmax, 1)
        self.cl_theo = self._get_Cl_cmb(r=r)[self.icl]
        self.dl_theo = self._cl2dl(self.ell_theo, self.cl_theo)
        self.ell_obs = np.arange(lmin, lmax, dl)
        
        
    def _get_Cl_cmb(self, r):
        power_spectrum = hp.read_cl(self.CMB_CL_FILE%'lensed_scalar')[:, :self.lmax]
        if self.Alens != 1.:
            power_spectrum[2] *= self.Alens
        if r:
            power_spectrum += r * hp.read_cl(self.CMB_CL_FILE%'unlensed_scalar_and_tensor_r1')[:, :self.lmax]
        return power_spectrum
    
    def _cl2dl(self, ell, cl):
        dl=np.zeros(ell.shape[0])
        for i in range(ell.shape[0]):
            dl[i]=(ell[i]*(ell[i]+1)*cl[i])/(2*np.pi)
        return dl
    
    def binning(self, cl):
        
        nbins = len(self.ell_obs)
        cl_binned = np.zeros(nbins)
        for i in range(nbins):
            cl_binned[i] = np.mean(cl[self.ell_obs[i]-int(self.dl/2) : self.ell_obs[i]+int(self.dl/2)])
        return cl_binned
    
    def get_observed_spectra(self, s):
        
        alm = hp.sphtfunc.map2alm(s, lmax=self.lmax)
        cl = hp.sphtfunc.alm2cl(alm)[self.icl, :]

        #cl_binned = self.binning(cl)
        #print(cl_binned)
        dl_binned = self._cl2dl(self.ell_theo, cl)
        
        #print(dl_binned)
        return dl_binned
    
    def chi2(self, r, dobs):
        #print(r)
        cl = self._get_Cl_cmb(r)[self.icl]
        d = self._cl2dl(self.ell_theo, cl)#[np.array(self.ell_obs, dtype=int)]
        #d = self._cl2dl(self.ell_theo, self.cl_theo)[np.array(self.ell_obs, dtype=int)]
        #print(d)
        #print(dobs)
        return np.sum((d - dobs)**2)
        
    def give_rbias(self, cl_obs):
        
        cl_theo_binned = self.dl_theo.copy()#[np.array(self.ell_obs, dtype=int)]
        s = minimize(self.chi2, x0=np.ones(1), args=(cl_obs), method='TNC', tol=1e-10, bounds=[(0, 1)])
        return s.x[0] - self.r
    