#########################################################################################################################
#########################################################################################################################
########                                                                                                         ########
########      This file contain the acquisition models to perform the component map-making for QUBIC. We can     ########
########      choose to simulate one FP of the instrument, two FP or the WideBand instrument.                    ########
########                                                                                                         ########
#########################################################################################################################
#########################################################################################################################

# QUBIC stuff
import qubic

# General stuff
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pysm3
import gc
import os
import sys
path = os.getcwd() + '/data/'
print('path', path)
import time
import warnings
warnings.filterwarnings("ignore")
import pysm3.units as u
from importlib import reload
from pysm3 import utils

from qubic.lib.Instrument.Qinstrument import compute_freq
from frequency_acquisition import compute_fwhm_to_convolve, arcmin2rad, give_cl_cmb, create_array, get_preconditioner, QubicPolyAcquisition, QubicAcquisition
import instrument as instr
# FG-Buster packages
import component_model as c
import mixing_matrix as mm
import pickle
# PyOperators stuff
from pysimulators import *
from pyoperators import *
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

__all__ = ['QubicIntegratedComponentsMapMaking',
           'QubicWideBandComponentsMapMaking',
           'QubicTwoBandsComponentsMapMaking',
           'QubicOtherIntegratedComponentsMapMaking',
           'OtherData']

def polarized_I(m, nside, polarization_fraction=0.01):
    
    polangle = hp.ud_grade(hp.read_map(path+'psimap_dust90_512.fits'), nside)
    depolmap = hp.ud_grade(hp.read_map(path+'gmap_dust90_512.fits'), nside)
    cospolangle = np.cos(2.0 * polangle)
    sinpolangle = np.sin(2.0 * polangle)
    #print(depolmap.shape)
    P_map = polarization_fraction * depolmap * hp.ud_grade(m, nside)
    return P_map * np.array([cospolangle, sinpolangle])
def get_allA(nc, nf, npix, beta, nus, comp, active):
    # Initialize arrays to store mixing matrix values
    allA = np.zeros((beta.shape[0], nf, nc))
    allA_pix = np.zeros((npix, nf, nc))

    # Loop through each element of beta to calculate mixing matrix
    for i in range(beta.shape[0]):
        allA[i] = get_mixingmatrix(beta[i], nus, comp, active)

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

    #D = DenseBlockDiagonalOperator(A, broadcast='leftward', shapein=(3, 12*nside**2, nc))
    D = BlockDiagonalOperator([DenseBlockDiagonalOperator(A, broadcast='rightward', shapein=(12*nside**2, nc)),
                           DenseBlockDiagonalOperator(A, broadcast='rightward', shapein=(12*nside**2, nc)),
                           DenseBlockDiagonalOperator(A, broadcast='rightward', shapein=(12*nside**2, nc))], new_axisin=0, new_axisout=2)
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
def get_mixing_operator(beta, nus, comp, nside, Amm=None, active=False):
    
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
        if Amm is None:
            A = get_mixingmatrix(beta, nus, comp, active)
        else:
            A = np.array([Amm]).copy()
        
        # Get the shape of the mixing matrix
        _, nc = A.shape
        
        # Create a ReshapeOperator
        R = ReshapeOperator(((1, 12*nside**2, 3)), ((12*nside**2, 3)))
        
        # Create a DenseOperator with the first row of A
        D = DenseOperator(A[0], broadcast='rightward', shapein=(nc, 12*nside**2, 3), shapeout=(1, 12*nside**2, 3))

    else: # Varying indice on the sky
        
        # Get all A matrices nc, nf, npix, beta, nus, comp
        A = get_allA(nc, 1, 12*nside**2, beta, nus, comp, active)
        
        # Get the varying mixing operator
        D = get_mixing_operator_verying_beta(nc, nside, A)

    return D

def get_mixing_operator_blind(Amm, nus, comp, nside, active=False):
    
    """
    This function returns a mixing operator based on the input parameters: beta and nus.
    The mixing operator is either a constant operator, or a varying operator depending on the input.
    """

    nc = len(comp)
    
    if len(Amm) == nc:
        constant = True
    else:
        constant = False

    # Check if the length of beta is equal to the number of channels minus 1
    if constant: # Constant indice on the sky
        #beta = np.mean(beta)

        # Get the mixing matrix
        A = np.array([Amm]).copy()
        
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


class QubicIntegratedComponentsMapMaking(QubicPolyAcquisition):

    def __init__(self, d, comp, Nsub):


        self.d = d
        self.sampling = qubic.get_pointing(self.d)
        self.scene = qubic.QubicScene(self.d)
        
        #QubicPolyAcquisition.__init__(self, self.multiinstrument, self.sampling, self.scene, self.d)
        
        self.Nsub = Nsub
        self.d['nf_sub'] = self.Nsub
        self.Ndets = 992
        self.Nsamples = self.sampling.shape[0]
        self.number_FP = 1


        _, allnus, _, _, _, _ = compute_freq(self.d['filter_nu']/1e9, Nfreq=self.Nsub, relative_bandwidth=self.d['filter_relative_bandwidth'])
        
        self.multiinstrument = instr.QubicMultibandInstrument(self.d)
        self.nside = self.scene.nside
        self.allnus = allnus
        self.comp = comp
        self.nc = len(self.comp)
        self.npix = 12*self.nside**2
        self.allnus = np.array([q.filter.nu / 1e9 for q in self.multiinstrument])
        
        self.subacqs = [QubicAcquisition(self.multiinstrument[i], self.sampling, self.scene, self.d) for i in range(len(self.multiinstrument))]

        for a in self[1:]:
            a.comm = self[0].comm
        #self.subacqs = [qubic.QubicAcquisition(self.multiinstrument[i], self.sampling, self.scene, self.d) for i in range(len(self.multiinstrument))]

        
        self.allfwhm = np.zeros(len(self.multiinstrument))
        for i in range(len(self.multiinstrument)):
            self.allfwhm[i] = self.subacqs[i].get_convolution_peak_operator().fwhm

        self.alltarget = compute_fwhm_to_convolve(np.min(self.allfwhm), self.allfwhm)
    def get_monochromatic_acquisition(self, nu):
        
        '''
        
        Return a monochromatic acquisition for a specific nu. nu parameter must be in Hz.
        
        '''

        self.d['filter_nu'] = nu

        sampling = qubic.get_pointing(self.d)
        scene = qubic.QubicScene(self.d)
        instrument = instr.QubicInstrument(self.d)
        fwhm = qubic.QubicAcquisition(instrument, sampling, scene, self.d).get_convolution_peak_operator().fwhm
        H = qubic.QubicAcquisition(instrument, sampling, scene, self.d).get_operator()
        return H, fwhm
    def get_PySM_maps(self, config, r=0, Alens=1):

        '''
        
        Read configuration dictionary which contains every components adn the model. 
        
        Example : d = {'cmb':42, 'dust':'d0', 'synchrotron':'s0'}

        The CMB is randomly generated fram specific seed. Astrophysical foregrounds come from PySM 3.
        
        '''

        allmaps = np.zeros((self.nc, 12*self.nside**2, 3))
        ell=np.arange(2*self.nside-1)
        mycls = give_cl_cmb(r=r, Alens=Alens)

        for k, kconf in enumerate(config.keys()):
            if kconf == 'cmb':

                np.random.seed(config[kconf])
                cmb = hp.synfast(mycls, self.nside, verbose=False, new=True).T

                allmaps[k] = cmb.copy()
            
            elif kconf == 'dust':

                nu0 = self.comp[k].__dict__['_fixed_params']['nu0']
                print(nu0)
                sky=pysm3.Sky(nside=self.nside, preset_strings=[config[kconf]], output_unit="uK_CMB")
                #sky.components[0].mbb_index = hp.ud_grade(sky.components[0].mbb_index, 8)
                sky.components[0].mbb_temperature = 20*sky.components[0].mbb_temperature.unit
                #sky.components[0].mbb_index = hp.ud_grade(np.array(sky.components[0].mbb_index), 8)
                mydust=np.array(sky.get_emission(nu0 * u.GHz, None).T * utils.bandpass_unit_conversion(nu0*u.GHz, None, u.uK_CMB))
                    
                allmaps[k] = mydust.copy()
            elif kconf == 'synchrotron':
                nu0 = self.comp[k].__dict__['_fixed_params']['nu0']
                sky=pysm3.Sky(nside=self.nside, preset_strings=[config[kconf]], output_unit="uK_CMB")
                mysync=np.array(sky.get_emission(nu0 * u.GHz, None).T * utils.bandpass_unit_conversion(nu0*u.GHz, None, u.uK_CMB))
                allmaps[k] = mysync.copy()
            elif kconf == 'coline':
                
                sky = pysm3.Sky(nside=self.nside, preset_strings=['co2'], output_unit="uK_CMB")
                #nu0 = sky.components[0].line_frequency['21'].value
                
                #myco=np.array(sky.get_emission(nu0 * u.GHz, None).T * utils.bandpass_unit_conversion(nu0*u.GHz, None, u.uK_CMB))
                # 10 is for reproduce the PsYM template
                m = np.array(sky.components[0].read_map(CMB_FILE+'CO_line.fits', unit=u.K_CMB)) * 10    
                mP = polarized_I(m, self.nside)
                myco = np.zeros((12*self.nside**2, 3))
                myco[:, 0] = m.copy()
                myco[:, 1:] = mP.T.copy()
                allmaps[k] = myco.copy()
            else:
                raise TypeError('Choose right foreground model (d0, s0, ...)')

        #if len(nus) == 1:
        #    allmaps = allmaps[0].copy()
            
        return allmaps
    def _get_average_instrument_acq(self):
        
        """
        Create and return a QubicAcquisition instance of a monochromatic
            instrument with frequency correspondent to the mean of the
            frequency range.
        """
        
        #if len(self) == 1:
        #    return self[0]
        q0 = self.multiinstrument[0]
        nu_min = self.multiinstrument[0].filter.nu
        nu_max = self.multiinstrument[-1].filter.nu
        nep = q0.detector.nep
        fknee = q0.detector.fknee
        fslope = q0.detector.fslope

        d1 = self.d.copy()
        d1['filter_nu'] = (nu_max + nu_min) / 2.
        d1['filter_relative_bandwidth'] = (nu_max - nu_min) / ((nu_max + nu_min) / 2.)
        d1['detector_nep'] = nep
        d1['detector_fknee'] = fknee
        d1['detector_fslope'] = fslope

        qq = instr.QubicInstrument(d1, FRBW=q0.FRBW)
        qq.detector = q0.detector
        #s_ = self.sampling
        #nsamplings = self.multiinstrument[0].comm.allreduce(len(s_))

        d1['random_pointing'] = True
        d1['sweeping_pointing'] = False
        d1['repeat_pointing'] = False
        d1['RA_center'] = 0.
        d1['DEC_center'] = 0.
        d1['npointings'] = self.d['npointings']
        d1['dtheta'] = 10.
        d1['period'] = self.d['period']

        # s = create_random_pointings([0., 0.], nsamplings, 10., period=s_.period)
        a = qubic.QubicAcquisition(qq, self.sampling, self.scene, d1)
        return a
    def get_noise(self):

        '''
        
        Return noise array according the focal plane you are considering which have shape (Ndets, Nsamples).
        
        '''

        a = self._get_average_instrument_acq()
        return a.get_noise()
    def _get_array_of_operators(self, nu_co=None):

        '''
        
        Compute all the Nsub sub-acquisition in one list. Each sub-acquisition contain the instrument specificities and describe the 
        synthetic beam for each frequencies.
        
        '''

        Operator = []
        for _, i in enumerate(self.subacqs):
            Operator.append(i.get_operator())
        if nu_co is not None:
            Hco, fwhmco = self.get_monochromatic_acquisition(nu_co)
            Operator.append(Hco)
        return Operator
    def get_operator(self, beta, convolution, gain=None, list_fwhm=None, nu_co=None):


        '''
        
        Method that allows to compute the reconstruction operator of QUBIC. 

        Parameter
        ---------

        beta : float of healpix format to describe the astrophysical foregrounds
        convolution : bool which allow to include convolution inside the operator. The convolution process assume 

        
        '''

        list_op = self._get_array_of_operators()
        self.Ndets, self.Nsamples = list_op[0].operands[0].shapein
        if beta.shape[0] <= 2:
            R = ReshapeOperator((1, 12*self.nside**2, 3), (12*self.nside**2, 3))
        else:
            R = ReshapeOperator((12*self.nside**2, 1, 3), (12*self.nside**2, 3))

        if gain is not None:
            G = DiagonalOperator(gain, broadcast='rightward')
        else:
            G = DiagonalOperator(1 + 1e-8 * np.random.randn(self.Ndets), broadcast='rightward')
        
        for inu, nu in enumerate(self.allnus):
            if convolution:
                if list_fwhm is not None:
                    C =  HealpixConvolutionGaussianOperator(fwhm=list_fwhm[inu])
                else:
                    C =  HealpixConvolutionGaussianOperator(fwhm=self.allfwhm[inu])
            else:
                C = IdentityOperator()
            
            A = get_mixing_operator(beta, np.array([nu]), comp=self.comp, nside=self.nside, active=False)
            
            list_op[inu] = list_op[inu] * C * R * A

        Rflat = ReshapeOperator((self.Ndets, self.Nsamples), self.Ndets*self.Nsamples)
        H = Rflat * BlockColumnOperator([G * np.sum(list_op, axis=0)], axisout=0)

        if nu_co is not None:
            Hco, myfwhmco = self.get_monochromatic_acquisition(nu_co)
            target = np.sqrt(myfwhmco**2 - np.min(self.allfwhm)**2)
            if convolution:
                if list_fwhm is not None:
                    C = HealpixConvolutionGaussianOperator(fwhm = target)
                else:
                    C = HealpixConvolutionGaussianOperator(fwhm = myfwhmco)
            else:
                C = IdentityOperator()
            Aco = get_mixing_operator(beta, np.array([nu_co]), comp=self.comp, nside=self.nside, active=True)
            Hco = Rflat * G * Hco * C * R * Aco
            H += Hco

        return H
    def update_A(self, H, newbeta):
        
        '''

        

        '''
        # If CO line
        if len(H.operands) == 2:
            for inu, nu in enumerate(self.allnus):
                newA = get_mixing_operator(newbeta, np.array([nu]), comp=self.comp, nside=self.nside, active=False)
                H.operands[0].operands[2].operands[inu].operands[-1] = newA
        else:
            for inu, nu in enumerate(self.allnus):
                newA = get_mixing_operator(newbeta, np.array([nu]), comp=self.comp, nside=self.nside)
                H.operands[2].operands[inu].operands[-1] = newA

        return H
    def get_coverage(self):
        return self.subacqs[0].get_coverage()
    def get_invntt_operator(self):
        invN = self.subacqs[0].get_invntt_operator()
        R = ReshapeOperator(invN.shapeout, invN.shape[0])
        return R(invN(R.T))




class QubicFullBandComponentsMapMakingParametric(QubicPolyAcquisition):

    def __init__(self, d, comp, Nsub, kind='Two'):

        ### relative_bandwidth=0.6138613861386139

        self.kind = kind
        if self.kind == 'Two': self.number_FP = 2
        elif self.kind == 'Wide': self.number_FP = 1

        
        #self.relative_bandwidth = relative_bandwidth

        if Nsub < 2:
            raise TypeError('You should use Nsub > 1')
        
        self.d = d
        self.Nsub = Nsub
        self.comp = comp
        self.d['nf_sub'] = Nsub
        self.d['nf_recon'] = 1
        
        self.nu_down = 131.25
        self.nu_up = 247.5

        self.nu_average = np.mean(np.array([self.nu_down, self.nu_up]))
        self.d['filter_nu'] = self.nu_average * 1e9
        
        _, allnus150, _, _, _, _ = compute_freq(150, Nfreq=self.Nsub-1, relative_bandwidth=0.25)
        _, allnus220, _, _, _, _ = compute_freq(220, Nfreq=self.Nsub-1, relative_bandwidth=0.25)
        self.allnus = np.array(list(allnus150) + list(allnus220))
        #print(self.nu_average, self.allnus)

        self.multiinstrument = instr.QubicMultibandInstrument(self.d)
        self.sampling = qubic.get_pointing(self.d)
        self.scene = qubic.QubicScene(self.d)

        self.subacqs = [QubicAcquisition(self.multiinstrument[i], self.sampling, self.scene, self.d) for i in range(len(self.multiinstrument))]
        self.subacqs150 = self.subacqs[:int(self.Nsub/2)]
        self.subacqs220 = self.subacqs[int(self.Nsub/2):self.Nsub]

        ### For MPI distribution
        QubicPolyAcquisition.__init__(self, self.multiinstrument, self.sampling, self.scene, self.d)

        self.allfwhm = np.zeros(len(self.multiinstrument))
        for i in range(len(self.multiinstrument)):
            self.allfwhm[i] = self.subacqs[i].get_convolution_peak_operator().fwhm

        self.nside = self.scene.nside
        self.npix = 12*self.nside**2

        self.comp = comp
        invn = self.get_invntt_operator()
        self.Ndets, self.Nsamples = invn.shapein

        self.nc = len(self.comp)

    def get_monochromatic_acquisition(self, nu):
        
        '''
        
        Return a monochromatic acquisition for a specific nu. nu parameter must be in Hz.
        
        '''

        self.d['filter_nu'] = nu

        sampling = qubic.get_pointing(self.d)
        scene = qubic.QubicScene(self.d)
        instrument = instr.QubicInstrument(self.d)
        fwhm = qubic.QubicAcquisition(instrument, sampling, scene, self.d).get_convolution_peak_operator().fwhm
        H = qubic.QubicAcquisition(instrument, sampling, scene, self.d).get_operator()
        return H, fwhm

    def _get_array_operators(self, beta, convolution=False, list_fwhm=None, co=None):


        '''
        
        Compute all the Nsub sub-acquisition in one list. Each sub-acquisition contain the instrument specificities and describe the 
        synthetic beam for each frequencies.

        '''

        self.operator = []
        if beta.shape[0] > 2:
            R = ReshapeOperator((12*self.nside**2, 1, 3), (12*self.nside**2, 3))
        else:
            R = ReshapeOperator((1, 12*self.nside**2, 3), (12*self.nside**2, 3))
        NF = np.ones(len(self.allnus))
        for inu, i in enumerate(self.subacqs):
            if convolution:
                if list_fwhm is not None:
                    C = HealpixConvolutionGaussianOperator(fwhm = list_fwhm[inu])
                else:
                    C = HealpixConvolutionGaussianOperator(fwhm = self.allfwhm[inu])
            else:
                C = IdentityOperator()

            A = get_mixing_operator(beta, np.array([self.allnus[inu]]), comp=self.comp, nside=self.nside, active=False)
            
            P = HomothetyOperator(NF[inu]) * i.get_operator() * C * R * A

            self.operator.append(P)

        if co is not None:

            A = get_mixing_operator(beta, np.array([self.allnus[inu]]), comp=self.comp, nside=self.nside, active=True)
            Hco, fwhm_co = self.get_monochromatic_acquisition(co)
            
            if self.d['comm'] is not None:
                mpidist = self.operator[0].operands[5]
                Hco.operands[-1] = mpidist
            else:
                mpidist = IdentityOperator()
                Hco.operands.append(mpidist)

            if convolution:
                Cco = HealpixConvolutionGaussianOperator(fwhm=np.sqrt(fwhm_co**2 - self.allfwhm[-1]**2))
            else:
                Cco = HealpixConvolutionGaussianOperator(fwhm=0)

            self.operator.append(HomothetyOperator(1 / self.Nsub) * Hco * Cco * R * A)
            


        self.Ndets, self.Nsamples = self.operator[0].shapeout

        return self.operator
    

    def get_operator(self, beta, convolution, list_fwhm=None, co=None):

        operator = self._get_array_operators(beta=beta, convolution=convolution, list_fwhm=list_fwhm, co=co)
        array_operator = np.array(operator)

        if self.kind == 'Two':
            index_down = np.where(self.allnus < self.nu_average)[0]
            index_up = np.where(self.allnus >= self.nu_average)[0]
            h150 = AdditionOperator(list(array_operator[index_down]))
            h220 = AdditionOperator(list(array_operator[np.max(index_down)+1:]))
            
            H = BlockColumnOperator([h150, h220], axisout=0)

            return H
        
        elif self.kind == 'Wide':
            H = AdditionOperator(operator)
            return H
        
        else:
            raise TypeError(f'{self.kind} not exist')


    def get_invntt_operator(self):
        """
        
        Method to compute the inverse noise covariance matrix in time-domain.

        """

        if self.kind == 'Two':

            invn150 = self.subacqs150[0].get_invntt_operator(det_noise=True, photon_noise=True)
            invn220 = self.subacqs220[0].get_invntt_operator(det_noise=True, photon_noise=True)

            return BlockDiagonalOperator([invn150, invn220], axisout=0)
        
        elif self.kind == 'Wide':

            invn150 = self.subacqs150[0].get_invntt_operator(det_noise=True, photon_noise=True)
            invn220 = self.subacqs220[-1].get_invntt_operator(det_noise=False, photon_noise=True)

            return invn150 + invn220


    def update_A(self, op, newbeta):

        if self.kind == 'Two':
            
            k=0
            for ifp in range(self.number_FP):
                for jnu in range(int(self.Nsub)):
                    A = get_mixing_operator(newbeta, np.array([self.allnus[k]]), comp=self.comp, nside=self.nside, active=False)
                    op.operands[ifp].operands[jnu].operands[-1] = A
                    k+=1

        elif self.kind == 'Wide':
            k=0
            for jnu in range(self.Nsub*2):

                A = get_mixing_operator(newbeta, np.array([self.allnus[k]]), comp=self.comp, nside=self.nside, active=False)

                op.operands[jnu].operands[-1] = A
                k+=1
        return op
    

    def get_PySM_maps(self, config, r=0, Alens=1):

        '''
        
        Read configuration dictionary which contains every components adn the model. 
        
        Example : d = {'cmb':42, 'dust':'d0', 'synchrotron':'s0'}

        The CMB is randomly generated fram specific seed. Astrophysical foregrounds come from PySM 3.
        
        '''

        allmaps = np.zeros((self.nc, 12*self.nside**2, 3))
        ell=np.arange(2*self.nside-1)
        mycls = give_cl_cmb(r=r, Alens=Alens)

        for k, kconf in enumerate(config.keys()):
            if kconf == 'cmb':

                np.random.seed(config[kconf])
                cmb = hp.synfast(mycls, self.nside, verbose=False, new=True).T

                allmaps[k] = cmb.copy()
            
            elif kconf == 'dust':

                nu0 = self.comp[k].__dict__['_fixed_params']['nu0']
                sky=pysm3.Sky(nside=self.nside, preset_strings=[config[kconf]], output_unit="uK_CMB")
                #sky.components[0].mbb_index = hp.ud_grade(sky.components[0].mbb_index, 8)
                sky.components[0].mbb_temperature = 20*sky.components[0].mbb_temperature.unit
                #sky.components[0].mbb_index = hp.ud_grade(np.array(sky.components[0].mbb_index), 8)
                mydust=np.array(sky.get_emission(nu0 * u.GHz, None).T * utils.bandpass_unit_conversion(nu0*u.GHz, None, u.uK_CMB))
                    
                allmaps[k] = mydust.copy()
            elif kconf == 'synchrotron':
                nu0 = self.comp[k].__dict__['_fixed_params']['nu0']
                sky=pysm3.Sky(nside=self.nside, preset_strings=[config[kconf]], output_unit="uK_CMB")
                mysync=np.array(sky.get_emission(nu0 * u.GHz, None).T * utils.bandpass_unit_conversion(nu0*u.GHz, None, u.uK_CMB))
                allmaps[k] = mysync.copy()
            elif kconf == 'coline':
                
                #sky = pysm3.Sky(nside=self.nside, preset_strings=['co2'], output_unit="uK_CMB")
                #nu0 = sky.components[0].line_frequency['21'].value
                
                #myco=np.array(sky.get_emission(nu0 * u.GHz, None).T * utils.bandpass_unit_conversion(nu0*u.GHz, None, u.uK_CMB))
                # 10 is for reproduce the PsYM template
                m = hp.ud_grade(hp.read_map(path+'CO_line.fits') * 10, self.nside)
                #print(self.nside)   
                mP = polarized_I(m, self.nside)
                #print(mP.shape)
                myco = np.zeros((12*self.nside**2, 3))
                myco[:, 0] = m.copy()
                myco[:, 1:] = mP.T.copy()
                allmaps[k] = myco.copy()
            else:
                raise TypeError('Choose right foreground model (d0, s0, ...)')

        return allmaps


class QubicFullBandComponentsMapMakingBlind(QubicPolyAcquisition):

    def __init__(self, d, comp, Nsub, kind='Two'):

        ### relative_bandwidth=0.6138613861386139

        self.kind = kind
        if self.kind == 'Two': self.number_FP = 2
        elif self.kind == 'Wide': self.number_FP = 1

        
        #self.relative_bandwidth = relative_bandwidth

        if Nsub < 2:
            raise TypeError('You should use Nsub > 1')
        
        self.d = d
        self.Nsub = Nsub
        self.comp = comp
        self.d['nf_sub'] = Nsub
        self.d['nf_recon'] = 1
        
        self.nu_down = 131.25
        self.nu_up = 247.5

        self.nu_average = np.mean(np.array([self.nu_down, self.nu_up]))
        self.d['filter_nu'] = self.nu_average * 1e9
        
        _, allnus150, _, _, _, _ = compute_freq(150, Nfreq=self.Nsub-1, relative_bandwidth=0.25)
        _, allnus220, _, _, _, _ = compute_freq(220, Nfreq=self.Nsub-1, relative_bandwidth=0.25)
        self.allnus = np.array(list(allnus150) + list(allnus220))
        #print(self.nu_average, self.allnus)

        self.multiinstrument = instr.QubicMultibandInstrument(self.d)
        self.sampling = qubic.get_pointing(self.d)
        self.scene = qubic.QubicScene(self.d)

        self.subacqs = [QubicAcquisition(self.multiinstrument[i], self.sampling, self.scene, self.d) for i in range(len(self.multiinstrument))]
        self.subacqs150 = self.subacqs[:int(self.Nsub/2)]
        self.subacqs220 = self.subacqs[int(self.Nsub/2):self.Nsub]

        ### For MPI distribution
        QubicPolyAcquisition.__init__(self, self.multiinstrument, self.sampling, self.scene, self.d)

        self.allfwhm = np.zeros(len(self.multiinstrument))
        for i in range(len(self.multiinstrument)):
            self.allfwhm[i] = self.subacqs[i].get_convolution_peak_operator().fwhm

        self.nside = self.scene.nside
        self.npix = 12*self.nside**2

        self.comp = comp
        invn = self.get_invntt_operator()
        self.Ndets, self.Nsamples = invn.shapein

        self.nc = len(self.comp)

    def _get_array_operators(self, A, convolution=False, list_fwhm=None):


        '''
        
        Compute all the Nsub sub-acquisition in one list. Each sub-acquisition contain the instrument specificities and describe the 
        synthetic beam for each frequencies.

        '''

        self.operator = []
        if A.shape != (len(self.allnus), len(self.comp)):
            R = ReshapeOperator((12*self.nside**2, 1, 3), (12*self.nside**2, 3))
        else:
            R = ReshapeOperator((1, 12*self.nside**2, 3), (12*self.nside**2, 3))
        NF = np.ones(len(self.allnus))
        for inu, i in enumerate(self.subacqs):
            if convolution:
                if list_fwhm is not None:
                    C = HealpixConvolutionGaussianOperator(fwhm = list_fwhm[inu])
                else:
                    C = HealpixConvolutionGaussianOperator(fwhm = self.allfwhm[inu])
            else:
                C = IdentityOperator()

            D = get_mixing_operator_blind(A[inu], np.array([self.allnus[inu]]), comp=self.comp, nside=self.nside, active=False)
            
            P = HomothetyOperator(NF[inu]) * i.get_operator() * C * R * D

            self.operator.append(P)

        self.Ndets, self.Nsamples = self.operator[0].shapeout

        return self.operator
    

    def get_operator(self, A, convolution, list_fwhm=None):

        operator = self._get_array_operators(A=A, convolution=convolution, list_fwhm=list_fwhm)
        array_operator = np.array(operator)

        if self.kind == 'Two':

            index_down = np.where(self.allnus < self.nu_average)[0]
            index_up = np.where(self.allnus >= self.nu_average)[0]
            h150 = AdditionOperator(list(array_operator[index_down]))
            h220 = AdditionOperator(list(array_operator[index_up]))
            
            H = BlockColumnOperator([h150, h220], axisout=0)

            return H
        
        elif self.kind == 'Wide':
            H = AdditionOperator(operator)
            return H
        
        else:
            raise TypeError(f'{self.kind} not exist')


    def get_invntt_operator(self):
        """
        
        Method to compute the inverse noise covariance matrix in time-domain.

        """

        if self.kind == 'Two':

            invn150 = self.subacqs150[0].get_invntt_operator(det_noise=True, photon_noise=True)
            invn220 = self.subacqs220[0].get_invntt_operator(det_noise=True, photon_noise=True)

            return BlockDiagonalOperator([invn150, invn220], axisout=0)
        
        elif self.kind == 'Wide':

            invn150 = self.subacqs150[0].get_invntt_operator(det_noise=True, photon_noise=True)
            invn220 = self.subacqs220[-1].get_invntt_operator(det_noise=False, photon_noise=True)

            return invn150 + invn220


    def update_A(self, op, newA):

        if self.kind == 'Two':
            
            k=0
            for ifp in range(self.number_FP):
                for jnu in range(int(self.Nsub)):
                    A = get_mixing_operator_blind(newA[k], np.array([self.allnus[k]]), comp=self.comp, nside=self.nside, active=False)
                    op.operands[ifp].operands[jnu].operands[-1] = A
                    k+=1

        elif self.kind == 'Wide':
            k=0
            for jnu in range(self.Nsub*2):

                A = get_mixing_operator_blind(newA[jnu], np.array([self.allnus[k]]), comp=self.comp, nside=self.nside, active=False)

                op.operands[jnu].operands[-1] = A
                k+=1
        return op
    

    def get_PySM_maps(self, config, r=0, Alens=1):

        '''
        
        Read configuration dictionary which contains every components adn the model. 
        
        Example : d = {'cmb':42, 'dust':'d0', 'synchrotron':'s0'}

        The CMB is randomly generated fram specific seed. Astrophysical foregrounds come from PySM 3.
        
        '''

        allmaps = np.zeros((self.nc, 12*self.nside**2, 3))
        ell=np.arange(2*self.nside-1)
        mycls = give_cl_cmb(r=r, Alens=Alens)

        for k, kconf in enumerate(config.keys()):
            if kconf == 'cmb':

                np.random.seed(config[kconf])
                cmb = hp.synfast(mycls, self.nside, verbose=False, new=True).T

                allmaps[k] = cmb.copy()
            
            elif kconf == 'dust':

                nu0 = self.comp[k].__dict__['_fixed_params']['nu0']
                sky=pysm3.Sky(nside=self.nside, preset_strings=[config[kconf]], output_unit="uK_CMB")
                #sky.components[0].mbb_index = hp.ud_grade(sky.components[0].mbb_index, 8)
                sky.components[0].mbb_temperature = 20*sky.components[0].mbb_temperature.unit
                #sky.components[0].mbb_index = hp.ud_grade(np.array(sky.components[0].mbb_index), 8)
                mydust=np.array(sky.get_emission(nu0 * u.GHz, None).T * utils.bandpass_unit_conversion(nu0*u.GHz, None, u.uK_CMB))
                    
                allmaps[k] = mydust.copy()
            elif kconf == 'synchrotron':
                nu0 = self.comp[k].__dict__['_fixed_params']['nu0']
                sky=pysm3.Sky(nside=self.nside, preset_strings=[config[kconf]], output_unit="uK_CMB")
                mysync=np.array(sky.get_emission(nu0 * u.GHz, None).T * utils.bandpass_unit_conversion(nu0*u.GHz, None, u.uK_CMB))
                allmaps[k] = mysync.copy()
            elif kconf == 'coline':
                
                #sky = pysm3.Sky(nside=self.nside, preset_strings=['co2'], output_unit="uK_CMB")
                #nu0 = sky.components[0].line_frequency['21'].value
                
                #myco=np.array(sky.get_emission(nu0 * u.GHz, None).T * utils.bandpass_unit_conversion(nu0*u.GHz, None, u.uK_CMB))
                # 10 is for reproduce the PsYM template
                m = np.array(sky.components[0].read_map(path+'CO_line.fits', unit=u.K_CMB)) * 10    
                mP = polarized_I(m, self.nside)
                myco = np.zeros((12*self.nside**2, 3))
                myco[:, 0] = m.copy()
                myco[:, 1:] = mP.T.copy()
                allmaps[k] = myco.copy()
            else:
                raise TypeError('Choose right foreground model (d0, s0, ...)')

        #if len(nus) == 1:
        #    allmaps = allmaps[0].copy()
            
        return allmaps


class QubicOtherIntegratedComponentsMapMakingParametric:

    def __init__(self, qubic, external_nus, comp, nintegr=1):

        #if nintegr == 1:
        #    raise TypeError('nintegr should be higher than 1')

        self.qubic = qubic
        self.external_nus = external_nus
        self.comp = comp
        self.nside = self.qubic.scene.nside
        self.npix = 12*self.nside**2
        self.nintegr = nintegr
        self.ndets = 992
        self.Nsamples = self.qubic.Nsamples
        self.Nsub = self.qubic.Nsub
        self.allnus = self.qubic.allnus
        self.number_FP = self.qubic.number_FP
        self.length_external_nus = len(self.external_nus) * 12*self.nside**2 * 3

        self.allresolution = self.qubic.allfwhm
        self.qubic_resolution = self.allresolution.copy()


        pkl_file = open(path+'AllDataSet_Components_MapMaking.pkl', 'rb')
        dataset = pickle.load(pkl_file)
        self.dataset = dataset
        self.bw = []
        if self.external_nus is not None:
            for ii, i in enumerate(self.external_nus):
                self.bw.append(self.dataset['bw{}'.format(i)])

        self.fwhm = []
        if self.external_nus is not None:
            for ii, i in enumerate(self.external_nus):
                self.fwhm.append(arcmin2rad(self.dataset['fwhm{}'.format(i)]))
                self.allresolution = np.append(self.allresolution, arcmin2rad(self.dataset['fwhm{}'.format(i)]))

        self.allresolution_external = self.allresolution[-len(self.external_nus):]
        self.alltarget = compute_fwhm_to_convolve(self.allresolution, np.max(self.allresolution))
        self.alltarget_external = self.alltarget[-len(self.external_nus):]

    def get_external_invntt_operator(self):

        allsigma=np.array([])
        for _, nu in enumerate(self.external_nus):
            sigma = hp.ud_grade(self.dataset['noise{}'.format(nu)].T, self.nside).T
            allsigma = np.append(allsigma, sigma.ravel())

        invN = DiagonalOperator(1 / allsigma ** 2, broadcast='leftward', shapein=(len(self.external_nus)*12*self.nside**2*3))

        R = ReshapeOperator(invN.shapeout, invN.shape[0])
        return R(invN(R.T))
    def get_operator(self, beta, convolution, list_fwhm=None, co=None):

        Hqubic = self.qubic.get_operator(beta=beta, convolution=convolution, list_fwhm=list_fwhm, co=co)
        
        if self.qubic.d['comm'] is not None:
            if self.qubic.kind == 'Wide':
                mpidist = Hqubic.operands[0].operands[5]
            elif self.qubic.kind == 'Two':
                mpidist = Hqubic.operands[0].operands[0].operands[5]
        else:
            mpidist = IdentityOperator()
        Rqubic = ReshapeOperator(Hqubic.shapeout, Hqubic.shape[0])

        Operator=[Rqubic * Hqubic]

        #if self.external_nus is not None:
        #    for ii, i in enumerate(self.external_nus):
        #        
        #        # Setting BandWidth
        #        bw = self.dataset['bw{}'.format(i)]
        #        # Generate instanciation for external data
        other = OtherDataParametric(self.external_nus, self.nside, self.comp, nintegr=self.nintegr)
        # Add operator
        Hother = other.get_operator(beta, convolution=convolution, myfwhm=[0]*len(self.external_nus), comm=mpidist, nu_co=co)
        #print(Hother.operands)
                
        Operator.append(Hother)
        return BlockColumnOperator(Operator, axisout=0)
    def get_maps(self):
        return ReshapeOperator(3*len(self.external_nus)*self.npix, (len(self.external_nus), self.npix, 3))
    def reconvolve_to_worst_resolution(self, tod):

        sh = tod.shape[0]
        sh_external = len(self.external_nus)*3*12*self.nside**2
        shape_tod_qubic = sh - sh_external
        tod_qubic = tod[:shape_tod_qubic]
        tod_external = tod[shape_tod_qubic:]
        R = self.get_maps()
        maps_external = R(tod_external)
        for ii, i in enumerate(self.external_nus):
            target = compute_fwhm_to_convolve(0, np.min(self.qubic_resolution))
            #print('target -> ', self.fwhm[ii], np.min(self.qubic_resolution), target)
            C = HealpixConvolutionGaussianOperator(fwhm=target)
            maps_external[ii] = C(maps_external[ii])

        tod_external = R.T(maps_external)

        return np.r_[tod_qubic, tod_external]
    
    def update_A(self, op, newbeta):

        op.operands[0].operands[1] = self.qubic.update_A(op.operands[0].operands[1], newbeta)
        other = OtherDataParametric(self.external_nus, self.nside, self.comp, self.nintegr)
        #print(op.operands[1].operands)
        #stop
        op.operands[1] = other.update_A(op.operands[1], newbeta)
        ### EXTERNAL DATA
        #k=0
        #for inu, nu in enumerate(self.external_nus):
                
        #    A = get_mixing_operator(newbeta, np.array([nu]), comp=self.comp, nside=self.nside)
            
        #    if nu == 217:
        #        #print(op.operands[1].operands[inu].operands[2].__name__)
        #        if op.operands[1].operands[inu].operands[2].__name__ == 'AdditionOperator':
        #            #print('mytest -> ', op.operands[1].operands[inu].operands[2])
        #            op.operands[1].operands[inu].operands[2].operands[0].operands[-1] = A
        #    else:
        #        op.operands[1].operands[inu].operands[-1] = A
                
        #    k+=1

        return op
    

    def get_invntt_operator(self, fact=None, mask=None):

        invNq = self.qubic.get_invntt_operator()
        R = ReshapeOperator(invNq.shapeout, invNq.shape[0])
        invNe = OtherDataParametric(self.external_nus, self.nside, self.comp).get_invntt_operator(fact=fact, mask=mask)

        return BlockDiagonalOperator([R(invNq(R.T)), invNe], axisout=0)
class OtherDataParametric:

    def __init__(self, nus, nside, comp, nintegr=2):
        
        if nintegr == 1:
            raise TypeError('The integration of external data should be greater than 1')
        
        self.nintegr = nintegr
        pkl_file = open(path+'AllDataSet_Components_MapMaking.pkl', 'rb')
        dataset = pickle.load(pkl_file)
        self.dataset = dataset

        self.nus = nus
        self.nside = nside
        self.npix = 12*self.nside**2
        self.bw = []
        for ii, i in enumerate(self.nus):
            self.bw.append(self.dataset['bw{}'.format(i)])
        
        self.fwhm = arcmin2rad(create_array('fwhm', self.nus, self.nside))
        self.comp = comp
        self.nc = len(self.comp)

        if nintegr == 1:
            self.allnus = self.nus
        else:
            self.allnus = []
            for inu, nu in enumerate(self.nus):
                self.allnus += list(np.linspace(nu-self.bw[inu]/2, nu+self.bw[inu]/2, self.nintegr))
            self.allnus = np.array(self.allnus)
        ### Compute all external nus
    def get_invntt_operator(self, fact=None, mask=None):
        # Create an empty array to store the values of sigma
        allsigma = np.array([])

        # Iterate through the frequency values
        for inu, nu in enumerate(self.nus):
            # Determine the scaling factor for the noise
            if fact is None:
                f=1
            else:
                f=fact[inu]

            # Get the noise value for the current frequency and upsample to the desired nside
            sigma = f * hp.ud_grade(self.dataset['noise{}'.format(nu)].T, self.nside).T

            if mask is not None:
                sigma /= np.array([mask, mask, mask]).T

            # Append the noise value to the list of all sigmas
            allsigma = np.append(allsigma, sigma.ravel())

        # Flatten the list of sigmas and create a diagonal operator
        allsigma = allsigma.ravel().copy()
        invN = DiagonalOperator(1 / allsigma ** 2, broadcast='leftward', shapein=(3*len(self.nus)*12*self.nside**2))

        # Create reshape operator and apply it to the diagonal operator
        R = ReshapeOperator(invN.shapeout, invN.shape[0])
        return R(invN(R.T))
    

    def update_A(self, op, newbeta):

        k=0
        for inu, nu in enumerate(self.nus):
            for i in range(self.nintegr):
                #print(inu, nu, i)
                A = get_mixing_operator(newbeta, np.array([self.allnus[k]]), comp=self.comp, nside=self.nside)

                if len(self.nus) == 1:
                    op.operands[2].operands[i].operands[-1] = A
                else:
                    if len(op.operands[inu].operands) == 3:  ### If no MPI distribution
                        #print(op.operands[inu].operands[2].operands[i].operands[-1])
                        op.operands[inu].operands[2].operands[i].operands[-1] = A
                    else:                                    ### If MPI distribution
                        op.operands[inu].operands[3].operands[i].operands[-1] = A
                k+=1
        return op

    def get_operator(self, beta, convolution, myfwhm=None, nu_co=None, comm=None):
        R2tod = ReshapeOperator((12*self.nside**2, 3), (3*12*self.nside**2))
        if beta.shape[0] <= 2:
            R = ReshapeOperator((1, 12*self.nside**2, 3), (12*self.nside**2, 3))
        else:
            R = ReshapeOperator((12*self.nside**2, 1, 3), (12*self.nside**2, 3))
        
        Operator=[]
        
        k=0
        for ii, i in enumerate(self.nus):
            ope_i=[]
            for j in range(self.nintegr):
                
                if convolution:
                    if myfwhm is not None:
                        fwhm = myfwhm[ii]
                    else:
                        fwhm = self.fwhm[ii]
                else:
                    fwhm = 0
                #fwhm = fwhm_max if convolution and fwhm_max is not None else (self.fwhm[ii] if convolution else 0)
                C = HealpixConvolutionGaussianOperator(fwhm=fwhm)
            
                
                D = get_mixing_operator(beta, np.array([self.allnus[k]]), comp=self.comp, nside=self.nside, active=False)
                ope_i += [C * R * D]

                
                k+=1
            
            if i == 217:
                #print('co line')
                if nu_co is not None:
                    Dco = get_mixing_operator(beta, np.array([nu_co]), comp=self.comp, nside=self.nside, active=True)
                    ope_i += [C * R * Dco]

            if comm is not None:
                Operator.append(comm*R2tod(AdditionOperator(ope_i)/self.nintegr))
            else:
                Operator.append(R2tod(AdditionOperator(ope_i)/self.nintegr))

                

                
        return BlockColumnOperator(Operator, axisout=0)
    def get_noise(self, seed=None, fact=None, seenpix=None):
        state = np.random.get_state()
        np.random.seed(seed)
        out = np.zeros((len(self.nus), self.npix, 3))
        R2tod = ReshapeOperator((len(self.nus), 12*self.nside**2, 3), (len(self.nus)*3*12*self.nside**2))
        for inu, nu in enumerate(self.nus):
            if fact is None:
                f=1
            else:
                f=fact[inu]
            sigma = f * hp.ud_grade(self.dataset['noise{}'.format(nu)].T, self.nside).T
            out[inu] = np.random.standard_normal((self.npix,3)) * sigma
        if seenpix is not None:
            out[:, seenpix, :] = 0
        np.random.set_state(state)
        return R2tod(out)


class QubicOtherIntegratedComponentsMapMakingBlind:

    def __init__(self, qubic, external_nus, comp, nintegr=1):

        #if nintegr == 1:
        #    raise TypeError('nintegr should be higher than 1')

        self.qubic = qubic
        self.external_nus = external_nus
        self.comp = comp
        self.nside = self.qubic.scene.nside
        self.npix = 12*self.nside**2
        self.nintegr = nintegr
        self.ndets = 992
        self.Nsamples = self.qubic.Nsamples
        self.Nsub = self.qubic.Nsub
        self.allnus = self.qubic.allnus
        self.number_FP = self.qubic.number_FP
        self.length_external_nus = len(self.external_nus) * 12*self.nside**2 * 3

        self.allresolution = self.qubic.allfwhm
        self.qubic_resolution = self.allresolution.copy()


        pkl_file = open(path+'AllDataSet_Components_MapMaking.pkl', 'rb')
        dataset = pickle.load(pkl_file)
        self.dataset = dataset
        self.bw = []
        if self.external_nus is not None:
            for ii, i in enumerate(self.external_nus):
                self.bw.append(self.dataset['bw{}'.format(i)])

        self.fwhm = []
        if self.external_nus is not None:
            for ii, i in enumerate(self.external_nus):
                self.fwhm.append(arcmin2rad(self.dataset['fwhm{}'.format(i)]))
                self.allresolution = np.append(self.allresolution, arcmin2rad(self.dataset['fwhm{}'.format(i)]))

        self.allresolution_external = self.allresolution[-len(self.external_nus):]
        self.alltarget = compute_fwhm_to_convolve(self.allresolution, np.max(self.allresolution))
        self.alltarget_external = self.alltarget[-len(self.external_nus):]

    def get_external_invntt_operator(self):

        allsigma=np.array([])
        for _, nu in enumerate(self.external_nus):
            sigma = hp.ud_grade(self.dataset['noise{}'.format(nu)].T, self.nside).T
            allsigma = np.append(allsigma, sigma.ravel())

        invN = DiagonalOperator(1 / allsigma ** 2, broadcast='leftward', shapein=(len(self.external_nus)*12*self.nside**2*3))

        R = ReshapeOperator(invN.shapeout, invN.shape[0])
        return R(invN(R.T))
    def get_operator(self, A, convolution, list_fwhm=None):

        Hqubic = self.qubic.get_operator(A=A[:2*self.qubic.Nsub], convolution=convolution, list_fwhm=list_fwhm)
        if self.qubic.d['comm'] is not None:
            if self.qubic.kind == 'Wide':
                mpidist = Hqubic.operands[0].operands[5]
            elif self.qubic.kind == 'Two':
                mpidist = Hqubic.operands[0].operands[0].operands[5]
        else:
            mpidist = IdentityOperator()
        Rqubic = ReshapeOperator(Hqubic.shapeout, Hqubic.shape[0])

        Operator=[Rqubic * Hqubic]

        #if self.external_nus is not None:
        #    for ii, i in enumerate(self.external_nus):
        #        
        #        # Setting BandWidth
        #        bw = self.dataset['bw{}'.format(i)]
        #        # Generate instanciation for external data
        other = OtherDataBlind(self.external_nus, self.nside, self.comp)
        # Add operator
        Hother = other.get_operator(A[2*self.qubic.Nsub:], convolution=convolution, myfwhm=[0], comm=mpidist)
        #print(Hother.operands)
                
        Operator.append(Hother)
        return BlockColumnOperator(Operator, axisout=0)
    def get_maps(self):
        return ReshapeOperator(3*len(self.external_nus)*self.npix, (len(self.external_nus), self.npix, 3))
    def reconvolve_to_worst_resolution(self, tod):

        sh = tod.shape[0]
        sh_external = len(self.external_nus)*3*12*self.nside**2
        shape_tod_qubic = sh - sh_external
        tod_qubic = tod[:shape_tod_qubic]
        tod_external = tod[shape_tod_qubic:]
        R = self.get_maps()
        maps_external = R(tod_external)
        for ii, i in enumerate(self.external_nus):
            target = compute_fwhm_to_convolve(0, np.min(self.qubic_resolution))
            #print('target -> ', self.fwhm[ii], np.min(self.qubic_resolution), target)
            C = HealpixConvolutionGaussianOperator(fwhm=target)
            maps_external[ii] = C(maps_external[ii])

        tod_external = R.T(maps_external)

        return np.r_[tod_qubic, tod_external]
    
    def update_A(self, op, newA):

        op.operands[0].operands[1] = self.qubic.update_A(op.operands[0].operands[1], newA[:2*self.qubic.Nsub])

        ### EXTERNAL DATA
        k=0
        for inu, nu in enumerate(self.external_nus):
                
            A = get_mixing_operator_blind(newA[2*self.qubic.Nsub:][inu], np.array([nu]), comp=self.comp, nside=self.nside)
                
            op.operands[1].operands[inu].operands[-1] = A
                
            k+=1

        return op
    

    def get_invntt_operator(self, fact=None, mask=None):

        invNq = self.qubic.get_invntt_operator()
        R = ReshapeOperator(invNq.shapeout, invNq.shape[0])
        invNe = OtherDataParametric(self.external_nus, self.nside, self.comp).get_invntt_operator(fact=fact, mask=mask)

        return BlockDiagonalOperator([R(invNq(R.T)), invNe], axisout=0)
class OtherDataBlind:

    def __init__(self, nus, nside, comp):

        pkl_file = open(path+'AllDataSet_Components_MapMaking.pkl', 'rb')
        dataset = pickle.load(pkl_file)
        self.dataset = dataset

        self.nus = nus
        self.nside = nside
        self.npix = 12*self.nside**2
        self.bw = []
        for ii, i in enumerate(self.nus):
            self.bw.append(self.dataset['bw{}'.format(i)])
        
        self.fwhm = arcmin2rad(create_array('fwhm', self.nus, self.nside))
        self.comp = comp
        self.nc = len(self.comp)

        self.allnus = self.nus
        ### Compute all external nus
    def get_invntt_operator(self, fact=None, mask=None):
        # Create an empty array to store the values of sigma
        allsigma = np.array([])

        # Iterate through the frequency values
        for inu, nu in enumerate(self.nus):
            # Determine the scaling factor for the noise
            if fact is None:
                f=1
            else:
                f=fact[inu]

            # Get the noise value for the current frequency and upsample to the desired nside
            sigma = f * hp.ud_grade(self.dataset['noise{}'.format(nu)].T, self.nside).T

            if mask is not None:
                sigma /= np.array([mask, mask, mask]).T

            # Append the noise value to the list of all sigmas
            allsigma = np.append(allsigma, sigma.ravel())

        # Flatten the list of sigmas and create a diagonal operator
        allsigma = allsigma.ravel().copy()
        invN = DiagonalOperator(1 / allsigma ** 2, broadcast='leftward', shapein=(3*len(self.nus)*12*self.nside**2))

        # Create reshape operator and apply it to the diagonal operator
        R = ReshapeOperator(invN.shapeout, invN.shape[0])
        return R(invN(R.T))
    def get_operator(self, A, convolution, myfwhm=None, nu_co=None, comm=None):
        R2tod = ReshapeOperator((12*self.nside**2, 3), (3*12*self.nside**2))
        if A.shape[1] == len(self.comp):
            R = ReshapeOperator((1, 12*self.nside**2, 3), (12*self.nside**2, 3))
        else:
            R = ReshapeOperator((12*self.nside**2, 1, 3), (12*self.nside**2, 3))
        
        Operator=[]
        
        k=0
        for ii, i in enumerate(self.nus):
            
            if convolution:
                if myfwhm is not None:
                    fwhm = myfwhm[ii]
                else:
                    fwhm = self.fwhm[ii]
            else:
                fwhm = 0
            #fwhm = fwhm_max if convolution and fwhm_max is not None else (self.fwhm[ii] if convolution else 0)
            C = HealpixConvolutionGaussianOperator(fwhm=fwhm)
            
            D = get_mixing_operator_blind(A[ii], np.array([self.allnus[k]]), comp=self.comp, nside=self.nside, active=False)
            op = C * R * D
            k+=1

            #if i == 217:
            #    if nu_co is not None:
            #        Dco = get_mixing_operator(beta, np.array([nu_co]), comp=self.comp, nside=self.nside, active=True)
            #        op.append(C * R * Dco)
            if comm is not None:
                Operator.append(comm*R2tod(op))
            else:
                Operator.append(R2tod(op))

                
        return BlockColumnOperator(Operator, axisout=0)
    def get_noise(self, seed=None, fact=None):
        state = np.random.get_state()
        np.random.seed(seed)
        out = np.zeros((len(self.nus), self.npix, 3))
        R2tod = ReshapeOperator((len(self.nus), 12*self.nside**2, 3), (len(self.nus)*3*12*self.nside**2))
        for inu, nu in enumerate(self.nus):
            if fact is None:
                f=1
            else:
                f=fact[inu]
            sigma = f * hp.ud_grade(self.dataset['noise{}'.format(nu)].T, self.nside).T
            out[inu] = np.random.standard_normal((self.npix,3)) * sigma
        np.random.set_state(state)
        return R2tod(out)


