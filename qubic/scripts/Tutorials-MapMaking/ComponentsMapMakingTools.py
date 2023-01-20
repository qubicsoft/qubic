import qubic
from pyoperators import *
from pysimulators import *
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
import component_model as c
import mixing_matrix as mm
import healpy as hp
import numpy as np


def get_allA(nc, nf, npix, beta, nus, comp):
    
    allA = np.zeros((npix, nf, nc))
    for i in range(npix):
        allA[i] = get_mixingmatrix(beta[i], nus, comp)
    return allA

def get_mixing_operator_verying_beta(nc, nside, A):

    npix = 12*nside**2
    if A.shape[0] != npix:
        print(f'Upgrade pixelization of beta from {A.shape[0]} to {12*nside**2}')
        Adown = np.zeros((12*nside**2, A.shape[1], A.shape[2]))
        for i in range(nc):
            Adown[:, 0, i] = hp.ud_grade(A[:, 0, i], nside)
    else:
        Adown=A.copy()
    
    R = ReshapeOperator((npix, 1, 3), (npix, 3))
    D = R * BlockDiagonalOperator([DenseBlockDiagonalOperator(Adown, broadcast='rightward', shapein=(npix, nc)),
                           DenseBlockDiagonalOperator(Adown, broadcast='rightward', shapein=(npix, nc)),
                           DenseBlockDiagonalOperator(Adown, broadcast='rightward', shapein=(npix, nc))], new_axisin=0, new_axisout=2)
    return D

def get_mixingmatrix(beta, nus, comp):

    A = mm.MixingMatrix(*comp)
    A_ev = A.evaluator(nus)
    A = A_ev(beta)

    return A

def get_mixing_operator(beta, nus, comp, nside):
    
    """
    This function returns a mixing operator based on the input parameters: beta and nus.
    The mixing operator is either a constant operator, or a varying operator depending on the input.
    """

    nc = len(comp)
    if beta.shape[0] != 1:
        nside_fit = hp.npix2nside(beta.shape[0])
    else:
        nside_fit = 0

    # Check if the length of beta is equal to the number of channels minus 1
    if nside_fit == 0:
        beta = np.mean(beta)
        # Get the mixing matrix
        A = get_mixingmatrix(beta, nus, comp)
        
        # Get the shape of the mixing matrix
        _, nc = A.shape
        
        # Create a ReshapeOperator
        R = ReshapeOperator(((1, 12*nside**2, 3)), ((12*nside**2, 3)))
        
        # Create a DenseOperator with the first row of A
        D = R * DenseOperator(A[0], broadcast='rightward', shapein=(nc, 12*nside**2, 3), shapeout=(1, 12*nside**2, 3))
    else:
        print(beta.shape[0])
        if beta.shape[0] != 12*nside**2:
            beta = hp.ud_grade(beta, nside_fit)
        # Get all A matrices nc, nf, npix, beta, nus, comp
        A = get_allA(nc, 1, len(beta), beta, nus, comp)
        
        # Get the varying mixing operator
        D = get_mixing_operator_verying_beta(nc, nside, A)

    return D