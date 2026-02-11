import qubic
from pyoperators import *
from pysimulators import *
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
import component_model as c
import mixing_matrix as mm
import healpy as hp
import numpy as np
import time


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

'''
def get_mixingmatrix(beta, nus, comp):

    A = mm.MixingMatrix(*comp)
    A_ev = A.evaluator(nus)
    if beta.shape[0] == 0:
        A = A_ev()
    else:
        A = A_ev(beta)

    return A
'''

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

    print(A_ev)
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
        data_qu = R(tod[:(2 * Nsamples * Ndets)]).copy()
        data150_qu, data220_qu = data_qu[0], data_qu[1]
        H150 = CompositionOperator(H.operands[0].operands[0].operands[1:])
        H220 = CompositionOperator(H.operands[0].operands[1].operands[1:])
        data150_s_qu = H150(mysolution).copy()
        data220_s_qu = H220(mysolution).copy()
            
        I150 = give_me_intercal(data150_s_qu, data150_qu)
        I220 = give_me_intercal(data220_s_qu, data220_qu)
        I = np.array([I150, I220])
    else:
        R = ReshapeOperator((Nsamples * Ndets), (Ndets, Nsamples))
        data_qu = R(tod[:(Nsamples * Ndets)]).copy()
        H150 = CompositionOperator(H.operands[0].operands[1:])
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
    d['RA_center'] = 0
    d['DEC_center'] = -57
    center = qubic.equ2gal(d['RA_center'], d['DEC_center'])
    d['effective_duration'] = 3
    d['npointings'] = pointing
    d['filter_nu'] = int(band*1e9)
    d['photon_noise'] = False
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

def normalize_tod(tod, ndets, nsamples, number_of_FP):

    tod_qubic = tod[:ndets*nsamples*number_of_FP]
    tod_external = tod[ndets*nsamples*number_of_FP:]

    newtod_qubic = tod_qubic / tod_qubic.max()
    newtod_external = tod_external / tod_external.max()

    return np.r_[newtod_qubic, newtod_external]
