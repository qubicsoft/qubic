import qubic
from pyoperators import BlockDiagonalOperator, CompositionOperator, DenseBlockDiagonalOperator, ReshapeOperator, DenseOperator
#from pysimulators import *
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
import fgbuster.component_model as c
from fgbuster.component_model import AnalyticComponent, K_RJ2K_CMB
import fgbuster.mixingmatrix as mm
import healpy as hp
import numpy as np
import pysm3
import pysm3.units as u
from pysm3.utils import bandpass_unit_conversion
from .data import DATA_PATH


class COLine(AnalyticComponent):
    """ Cosmic microwave background

    Parameters
    ----------
    units:
        Output units (K_CMB and K_RJ available)
    """

    active = False

    def __init__(self, nu, active, units: str = 'K_CMB'):
        # Prepare the analytic expression
        self.nu = nu
        if active :
            analytic_expr = ('1')
        else:
            analytic_expr = ('0')
        if units == 'K_CMB':
            pass
        elif units == 'K_RJ':
            analytic_expr += ' / ' + K_RJ2K_CMB
        else:
            raise ValueError(f"Unsupported units: {units}")
        
        kwargs = {'active': active}
        super().__init__(analytic_expr, **kwargs)


class Sky:
    
    def __init__(self, sky_config, qubic, nu0=150):
        

        """
        
        This class allow to compute the sky at different frequency according to a given SED with astrophysical foregrounds.

        """
        self.qubic = qubic
        self.sky_config = sky_config
        self.nside = self.qubic.scene.nside
        self.allnus = self.qubic.allnus
        self.nu0 = nu0

        self.is_cmb = False
        self.is_dust = False
        map_ref = []
        self.comp = []
        k = 0
        for i in self.sky_config.keys():
            if i == 'cmb':
                self.is_cmb = True
                self.cmb = self.get_cmb(self.sky_config[i])
                self.i_cmb = k
                self.comp += [c.CMB()]
                map_ref += [self.cmb]
            elif i == 'dust':
                self.is_dust = True
                self.dust = self.get_dust(self.nu0, self.sky_config[i])
                map_ref += [self.dust]
                self.comp += [c.Dust(nu0=self.nu0, temp=20)]
                self.i_dust = k
            k+=1
        self.map_ref = np.array(map_ref)

        self.A = mm.MixingMatrix(*self.comp).evaluator(self.allnus)

    def get_SED(self, beta=None):
        if beta is None:
            sed = self.A()
        else:
            sed = self.A(beta)

        return sed
        
    def scale_component(self, beta=None):
        m_nu = np.zeros((len(self.allnus), 12*self.nside**2, 3))
        sed = self.get_SED(beta)
        
        if self.is_cmb == True and self.is_dust == True:
            sed = np.array([sed[:, 1]]).T

        if self.is_dust:
            for i in range(3):
                m_nu[:, :, i] = sed @ np.array([self.map_ref[self.i_dust, :, i]])

        if self.is_cmb:
            for i in range(len(self.allnus)):
                m_nu[i] += self.map_ref[self.i_cmb]

        return m_nu

    def get_cmb(self, seed):
        mycls = self._give_cl_cmb()
        np.random.seed(seed)
        return hp.synfast(mycls, self.nside, verbose=False, new=True).T

    @staticmethod
    def _give_cl_cmb(r=0, Alens=1.):
        power_spectrum = hp.read_cl(DATA_PATH / 'Cls_Planck2018_lensed_scalar.fits')[:,:4000]
        if Alens != 1.:
            power_spectrum[2] *= Alens
        if r:
            power_spectrum += r * hp.read_cl(DATA_PATH / 'Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits')[:,:4000]
        return power_spectrum

    def get_dust(self, nu0, model):
        sky=pysm3.Sky(nside=self.nside, preset_strings=[model])
        myfg=np.array(sky.get_emission(nu0 * u.GHz, None).T * pysm3.utils.bandpass_unit_conversion(nu0*u.GHz, None, u.uK_CMB))
        
        return myfg


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


def get_mixing_operator_varying_beta(nc, nside, A):

    npix = 12*nside**2
    
    D = BlockDiagonalOperator(
        [
            DenseBlockDiagonalOperator(A, broadcast='rightward', shapein=(npix, nc)),
            DenseBlockDiagonalOperator(A, broadcast='rightward', shapein=(npix, nc)),
            DenseBlockDiagonalOperator(A, broadcast='rightward', shapein=(npix, nc)),
        ],
        new_axisin=0,
        new_axisout=2,
    )
    return D


def get_mixingmatrix(beta, nus, comp, active=False):
    A = mm.MixingMatrix(*comp)
    if active:
        i = A.components.index('COLine')
        comp[i] = COLine(nu=comp[i].nu, active=True)
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
        D = get_mixing_operator_varying_beta(nc, nside, A)

    return D
