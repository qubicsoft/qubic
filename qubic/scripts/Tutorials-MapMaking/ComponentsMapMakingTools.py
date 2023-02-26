import qubic
from pyoperators import *
from pysimulators import *
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
import component_model as c
import mixing_matrix as mm
import healpy as hp
import numpy as np


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

def get_mixingmatrix(beta, nus, comp):

    A = mm.MixingMatrix(*comp)
    A_ev = A.evaluator(nus)
    if beta.shape[0] == 0:
        A = A_ev()
    else:
        A = A_ev(beta)

    return A

def get_mixing_operator(beta, nus, comp, nside):
    
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
        A = get_mixingmatrix(beta, nus, comp)
        
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