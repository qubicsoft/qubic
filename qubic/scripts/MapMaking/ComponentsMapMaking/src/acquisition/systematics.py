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
path_to_data = os.getcwd() + '/data/'

import time
import warnings
warnings.filterwarnings("ignore")
import pysm3.units as u
from pysm3 import utils
from importlib import reload
from qubic.data import PATH

from acquisition.frequency_acquisition import compute_fwhm_to_convolve, arcmin2rad, give_cl_cmb, create_array, get_preconditioner, QubicPolyAcquisition, QubicAcquisition
import acquisition.instrument as instr
# FG-Buster packages
import fgb.component_model as c
import fgb.mixing_matrix as mm
import pickle
# PyOperators stuff
from pysimulators import *
from pyoperators import *
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
import pyoperators 
pyoperators.memory.verbose = False

def polarized_I(m, nside, polarization_fraction=0):
    
    polangle = hp.ud_grade(hp.read_map(path_to_data+'psimap_dust90_512.fits'), nside)
    depolmap = hp.ud_grade(hp.read_map(path_to_data+'gmap_dust90_512.fits'), nside)
    cospolangle = np.cos(2.0 * polangle)
    sinpolangle = np.sin(2.0 * polangle)
    #print(depolmap.shape)
    P_map = polarization_fraction * depolmap * hp.ud_grade(m, nside)
    return P_map * np.array([cospolangle, sinpolangle])
def create_array(name, nus, nside):

    if name == 'noise':
        shape=(2, 12*nside**2, 3)
    else:
        shape=len(nus)
    pkl_file = open(path_to_data+'AllDataSet_Components_MapMaking.pkl', 'rb')
    dataset = pickle.load(pkl_file)

    myarray=np.zeros(shape)

    for ii, i in enumerate(nus):
        myarray[ii] = dataset[name+str(i)]

    return myarray
def get_preconditioner(cov):
    if cov is not None:
        cov_inv = 1 / cov
        cov_inv[np.isinf(cov_inv)] = 0.
        preconditioner = DiagonalOperator(cov_inv, broadcast='rightward')
    else:
        preconditioner = None
    return preconditioner
def arcmin2rad(arcmin):
    return arcmin * 0.000290888
def give_cl_cmb(r=0, Alens=1.):
    power_spectrum = hp.read_cl(path_to_data+'Cls_Planck2018_lensed_scalar.fits')[:,:4000]
    if Alens != 1.:
        power_spectrum[2] *= Alens
    if r:
        power_spectrum += r * hp.read_cl(path_to_data+'Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits')[:,:4000]
    return power_spectrum
def rad2arcmin(rad):
    return rad / 0.000290888
def circular_mask(nside, center, radius):
    lon = center[0]
    lat = center[1]
    vec = hp.ang2vec(lon, lat, lonlat=True)
    disc = hp.query_disc(nside, vec, radius=np.deg2rad(radius))
    m = np.zeros(hp.nside2npix(nside))
    m[disc] = 1
    return np.array(m, dtype=bool)
def compute_fwhm_to_convolve(allres, target):
    s = np.sqrt(target**2 - allres**2)
    #if s == np.nan:
    #    s = 0
    return s
def find_co(comp, nus_edge):
    return np.sum(nus_edge < comp[-1].nu) - 1
def parse_addition_operator(operator):

    if isinstance(operator, AdditionOperator):
        for op in operator.operands:
            parse_addition_operator(op)

    else:
        parse_composition_operator(operator)
    return operator
def parse_composition_operator(operator):
    for i, op in enumerate(operator.operands):
        if isinstance(op, HealpixConvolutionGaussianOperator):
            operator.operands[i] = HealpixConvolutionGaussianOperator(fwhm=10)
def insert_inside_list(operator, element, position):

    list = operator.operands
    list.insert(position, element)
    return CompositionOperator(list)
def delete_inside_list(operator, position):

    list = operator.operands
    list.pop(position)
    return CompositionOperator(list)
def mychi2(beta, obj, Hqubic, data, solution, nsamples):

    H_for_beta = obj.get_operator(beta, convolution=False, H_qubic=Hqubic)
    fakedata = H_for_beta(solution)
    fakedata_norm = obj.normalize(fakedata, nsamples)
    print(beta)
    return np.sum((fakedata_norm - data)**2)
def fit_beta(tod, nsamples, obj, H_qubic, outputs):

    tod_norm = obj.normalize(tod, nsamples)
    r = minimize(mychi2, method='TNC', tol=1e-15, x0=np.array([1.]), args=(obj, H_qubic, tod_norm, outputs, nsamples))

    return r.x
def fill_hwp_position(nsamples, angle):
    ang = np.zeros(nsamples)
    nangle = len(angle)
    x = int(nsamples/nangle)
    print(x)
    
    for ii, i in enumerate(angle):
        ang[x*ii:x*(ii+1)] = i
        
    return ang
def get_allA(nc, nf, npix, beta, nus, comp, active):
    # Initialize arrays to store mixing matrix values
    allA = np.zeros((beta.shape[0], nf, nc))
    allA_pix = np.zeros((npix, nf, nc))

    # Loop through each element of beta to calculate mixing matrix
    for i in range(beta.shape[0]):
        allA[i] = get_mixingmatrix(beta[i], nus, comp, active)

    # Check if beta and npix are equal
    #print(beta.shape[0], npix)
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

class PlanckAcquisition:

    def __init__(self, band, scene):
        if band not in (30, 44, 70, 143, 217, 353):
            raise ValueError("Invalid band '{}'.".format(band))
        self.scene = scene
        self.band = band
        self.nside = self.scene.nside
        
        if band == 30:
            filename = 'Variance_Planck30GHz_Kcmb2_ns256.fits'
            var=np.zeros((12*self.scene.nside**2, 3))
            for i in range(3):
                var[:, i] = hp.ud_grade(hp.fitsfunc.read_map(filename, field=i), self.scene.nside)
            sigma = 1e6 * np.sqrt(var)
        elif band == 44:
            filename = 'Variance_Planck44GHz_Kcmb2_ns256.fits'
            var=np.zeros((12*self.scene.nside**2, 3))
            for i in range(3):
                var[:, i] = hp.ud_grade(hp.fitsfunc.read_map(filename, field=i), self.scene.nside)
            sigma = 1e6 * np.sqrt(var)
        elif band == 70:
            filename = 'Variance_Planck70GHz_Kcmb2_ns256.fits'
            var=np.zeros((12*self.scene.nside**2, 3))
            for i in range(3):
                var[:, i] = hp.ud_grade(hp.fitsfunc.read_map(filename, field=i), self.scene.nside)
            sigma = 1e6 * np.sqrt(var)
        elif band == 143:
            filename = 'Variance_Planck143GHz_Kcmb2_ns256.fits'
            self.var = np.array(FitsArray(PATH + filename))
            sigma = 1e6 * np.sqrt(self.var)
        elif band == 217:
            filename = 'Variance_Planck217GHz_Kcmb2_ns256.fits'
            self.var = np.array(FitsArray(PATH + filename))
            sigma = 1e6 * np.sqrt(self.var)
        else:
            filename = 'Variance_Planck353GHz_Kcmb2_ns256.fits'
            var=np.zeros((12*self.scene.nside**2, 3))
            for i in range(3):
                var[:, i] = hp.ud_grade(hp.fitsfunc.read_map(filename, field=i), self.scene.nside)
            sigma = 1e6 * np.sqrt(var)

        


        if scene.kind == 'I':
            sigma = sigma[:, 0]
        elif scene.kind == 'QU':
            sigma = sigma[:, :2]
        if self.nside != 256:
            sigma = np.array(hp.ud_grade(sigma.T, self.nside, power=2),
                             copy=False).T
        self.sigma = sigma

    
    def get_operator(self, nintegr=1):
        Hp = DiagonalOperator(np.ones((12*self.nside**2, 3)), broadcast='rightward',
                                shapein=self.scene.shape, shapeout=np.ones((12*self.nside**2, 3)).ravel().shape)


        if nintegr == 1 :
            return Hp

    def get_invntt_operator(self, beam_correction=0, mask=None, seenpix=None):
        
        if beam_correction != 0:
            factor = (4*np.pi*(np.rad2deg(beam_correction)/2.35/np.degrees(hp.nside2resol(self.scene.nside)))**2)
            #print(f'corrected by {factor}')
            varnew = hp.smoothing(self.var.T, fwhm=beam_correction/np.sqrt(2)) / factor
            self.sigma = 1e6 * np.sqrt(varnew.T)
        
        if mask is not None:
            for i in range(3):
                self.sigma[:, i] /= mask.copy()
                
        myweight = 1 / (self.sigma ** 2)
        
        return DiagonalOperator(myweight, broadcast='leftward',
                                shapein=myweight.shape)

    def get_noise(self, seed):
        state = np.random.get_state()
        np.random.seed(seed)
        out = np.random.standard_normal(np.ones((12*self.nside**2, 3)).shape) * self.sigma
        np.random.set_state(state)
        return out
    
    def get_map(self, nu_min, nu_max, Nintegr, sky_config, d, fwhm = None):

        print(f'Integration from {nu_min:.2f} to {nu_max:.2f} GHz with {Nintegr} steps')
        obj = QubicIntegrated(d, Nsub=Nintegr, Nrec=Nintegr)
        if Nintegr == 1:
            allnus = np.array([np.mean([nu_min, nu_max])])
        else:
            allnus = np.linspace(nu_min, nu_max, Nintegr)
        m = obj.get_PySM_maps(sky_config, nus=allnus)
    
        if fwhm is None:
            fwhm = [0]*Nintegr
        
        for i in range(Nintegr):
            C = HealpixConvolutionGaussianOperator(fwhm=fwhm)
            m[i] = C(m[i])
    
        return np.mean(m, axis=0)
class QubicFullBandSystematic(QubicPolyAcquisition):

    def __init__(self, d, Nsub, Nrec=1, comp=[], kind='Two', nu_co=None, H=None, effective_duration150=3, effective_duration220=3):
        
        #if Nsub % 2 != 0:
        #    raise TypeError('Nsub should not be odd')

        if Nrec > 1 and len(comp) > 0:
            raise TypeError('For Components Map-Making, there must be Nrec = 1')
        
        self.d = d
        self.comp = comp
        self.Nsub = int(Nsub/2)
        self.kind = kind
        self.Nrec = Nrec
        self.nu_co = nu_co
        self.effective_duration150=effective_duration150
        self.effective_duration220=effective_duration220
        
        if self.kind == 'Two' and self.Nrec == 1 and len(self.comp) == 0:
            raise TypeError('Dual band instrument can not reconstruct one band')

        if self.kind == 'Two': self.number_FP = 2
        elif self.kind == 'Wide': self.number_FP = 1

        
        #self.relative_bandwidth = relative_bandwidth

        if Nsub < 2:
            raise TypeError('You should use Nsub > 1')
        
        self.d['nf_sub'] = self.Nsub
        self.d['nf_recon'] = 1
        
        self.nu_down = 131.25
        self.nu_up = 247.5

        self.nu_average = np.mean(np.array([self.nu_down, self.nu_up]))
        self.d['filter_nu'] = self.nu_average * 1e9
        
        _, allnus150, _, _, _, _ = qubic.compute_freq(150, Nfreq=self.Nsub-1, relative_bandwidth=0.25)
        _, allnus220, _, _, _, _ = qubic.compute_freq(220, Nfreq=self.Nsub-1, relative_bandwidth=0.25)
        self.allnus = np.array(list(allnus150) + list(allnus220))
        #print(self.nu_average, self.allnus)

        self.multiinstrument = instr.QubicMultibandInstrument(self.d)
       #print(self.multiinstrument.subinstruments)
       # stop
        self.sampling = qubic.get_pointing(self.d)
        self.scene = qubic.QubicScene(self.d)

        self.Proj = []
        self.subacqs = []
        
        self.H = []
        self.subacqs = [QubicAcquisition(self.multiinstrument[i], self.sampling, self.scene, self.d) for i in range(len(self.multiinstrument))]
        
        self.allfwhm = np.zeros(len(self.multiinstrument))
        for i in range(len(self.multiinstrument)):
            self.allfwhm[i] = self.subacqs[i].get_convolution_peak_operator().fwhm
        
        if nu_co is not None:
            #dmono = self.d.copy()
            self.d['filter_nu'] = nu_co * 1e9
            sampling = qubic.get_pointing(self.d)
            scene = qubic.QubicScene(self.d)
            instrument_co = instr.QubicInstrument(self.d)
            self.multiinstrument.subinstruments += [instrument_co]
            self.Proj += [QubicAcquisition(self.multiinstrument[-1], sampling, scene, self.d).get_projection_operator()]
            self.subacqs += [QubicAcquisition(self.multiinstrument[-1], sampling, scene, self.d)]
        
        QubicPolyAcquisition.__init__(self, self.multiinstrument, self.sampling, self.scene, self.d)
        
        if H is None:
            self.H = [self.subacqs[i].get_operator() for i in range(len(self.subacqs))]
        else:
            self.H = H
        #print(self.d['nprocs_instrument'])
        #stop
        if self.d['nprocs_instrument'] != 1:
            self.mpidist = self.H[0].operands[-1]

        self.ndets = len(self.subacqs[0].instrument)
        self.nsamples = len(self.sampling)
        
        self.coverage = self.subacqs[0].get_coverage()
        

    def get_hwp_operator(self, angle_hwp):
        """
        Return the rotation matrix for the half-wave plate.

        """
        return Rotation3dOperator('X', -4 * angle_hwp,
                                  degrees=True, shapein=self.Proj[0].shapeout)
    def get_components_operator(self, beta, nu, Amm=None, active=False):
        
        if beta.shape[0] != 0 and beta.shape[0] != 1 and beta.shape[0] != 2:
            r = ReshapeOperator((12*self.scene.nside**2, 1, 3), (12*self.scene.nside**2, 3))
        else:
            r = ReshapeOperator((1, 12*self.scene.nside**2, 3), (12*self.scene.nside**2, 3))
        return  r * get_mixing_operator(beta, nu, self.comp, self.scene.nside, Amm=Amm, active=active)
    def sum_over_band(self, h, gain=None):
        op_sum = []
        f = int(2*self.Nsub / self.Nrec)
        
        
        ### Frequency Map-Making
        if len(self.comp) == 0:
            h = np.array(h)
            for irec in range(self.Nrec):
                imin = irec*f
                imax = (irec+1)*f-1
                op_sum += [h[(self.allnus >= self.allnus[imin]) * (self.allnus <= self.allnus[imax])].sum(axis=0)]
            
            if self.kind == 'wide':
                return BlockRowOperator(op_sum, new_axisin=0)
            else:
                if self.Nrec > 2:
                    return BlockDiagonalOperator([BlockRowOperator(op_sum[:int(self.Nrec/2)], new_axisin=0),
                                                  BlockRowOperator(op_sum[int(self.Nrec/2):int(self.Nrec)], new_axisin=0)], axisout=0)
                else:
                    return ReshapeOperator((2, self.ndets, self.nsamples), (2*self.ndets, self.nsamples)) * \
                           BlockDiagonalOperator([BlockRowOperator(op_sum[:int(self.Nrec/2)], new_axisin=0),
                                                  BlockRowOperator(op_sum[int(self.Nrec/2):int(self.Nrec)], new_axisin=0)], new_axisin=0)

                       
        
        ### Components Map-Making
        else:
            if self.kind == 'wide':
                if gain is None:
                    G = DiagonalOperator(np.ones(self.ndets), broadcast='rightward', shapein=(self.ndets, self.nsamples))
                else:
                    G = DiagonalOperator(gain, broadcast='rightward', shapein=(self.ndets, self.nsamples))
                return G * AdditionOperator(h)
            else:
                if gain is None:
                    G150 = DiagonalOperator(np.ones(self.ndets), broadcast='rightward', shapein=(self.ndets, self.nsamples))
                    G220 = DiagonalOperator(np.ones(self.ndets), broadcast='rightward', shapein=(self.ndets, self.nsamples))
                else:
                    G150 = DiagonalOperator(gain[:, 0], broadcast='rightward', shapein=(self.ndets, self.nsamples))
                    G220 = DiagonalOperator(gain[:, 1], broadcast='rightward', shapein=(self.ndets, self.nsamples))
                return BlockColumnOperator([G150 * AdditionOperator(h[:int(self.Nsub)]), 
                                            G220 * AdditionOperator(h[int(self.Nsub):])], axisout=0)
    def get_operator(self, beta=None, Amm=None, angle_hwp=None, gain=None, fwhm=None):
        
        self.operator = []

        if angle_hwp is None:
            angle_hwp = self.sampling.angle_hwp
        else:
            angle_hwp = fill_hwp_position(self.Proj[0].shapeout[1], angle_hwp)

        #G = DiagonalOperator(g, broadcast='rightward', shapein=(myqubic.Ndets, myqubic.Nsamples))
        for isub in range(self.Nsub*2):

            if beta is None:
                Acomp = IdentityOperator()
            else:
                if Amm is not None:
                    Acomp = self.get_components_operator(beta, np.array([self.allnus[isub]]), Amm=Amm[isub])
                else:
                    Acomp = self.get_components_operator(beta, np.array([self.allnus[isub]]))

            if fwhm is None:
                convolution = IdentityOperator()
            else:
                convolution = HealpixConvolutionGaussianOperator(fwhm=fwhm[isub], lmax=2*self.d['nside'])
            with rule_manager(inplace=True):
                hi = CompositionOperator([
                            self.H[isub], convolution, Acomp])
            
            self.operator.append(hi)

        if self.nu_co is not None:
            
            if beta is None:
                Acomp = IdentityOperator()
            else:
                
                Acomp = self.get_components_operator(beta, np.array([self.nu_co]), active=True)
            distribution = self.subacqs[-1].get_distribution_operator()
            temp = self.subacqs[-1].get_unit_conversion_operator()
            aperture = self.subacqs[-1].get_aperture_integration_operator()
            filter = self.subacqs[-1].get_filter_operator()
            projection = self.Proj[-1]
            #hwp = self.get_hwp_operator(angle_hwp)
            hwp = self.subacqs[-1].get_hwp_operator()
            polarizer = self.subacqs[-1].get_polarizer_operator()
            integ = self.subacqs[-1].get_detector_integration_operator()
            trans = self.multiinstrument[-1].get_transmission_operator()
            trans_atm = self.subacqs[-1].scene.atmosphere.transmission
            response = self.subacqs[-1].get_detector_response_operator()
            if fwhm is None:
                convolution = IdentityOperator()
            else:
                convolution = HealpixConvolutionGaussianOperator(fwhm=fwhm[isub], lmax=2*self.d['nside'])
            with rule_manager(inplace=True):
                hi = CompositionOperator([
                            HomothetyOperator(1 / (2*self.Nsub)), response, trans_atm, trans, integ, polarizer, (hwp * projection),
                            filter, aperture, temp, distribution, convolution, Acomp])
            
            self.operator.append(hi)

            
        H = self.sum_over_band(self.operator, gain=gain)
        
        return H
    def get_invntt_operator(self):
        """
        
        Method to compute the inverse noise covariance matrix in time-domain.

        """
        d150 = self.d.copy()
        d150['filter_nu'] = 150 * 1e9
        d150['effective_duration'] = self.effective_duration150
        ins150 = instr.QubicInstrument(d150)

        d220 = self.d.copy()
        d220['effective_duration'] = self.effective_duration220
        d220['filter_nu'] = 220 * 1e9
        
        ins220 = instr.QubicInstrument(d220)

        subacq150 = QubicAcquisition(ins150, self.sampling, self.scene, d150)
        subacq220 = QubicAcquisition(ins220, self.sampling, self.scene, d220)
        if self.kind == 'two':

            invn150 = subacq150.get_invntt_operator(det_noise=True, photon_noise=True)
            invn220 = subacq220.get_invntt_operator(det_noise=True, photon_noise=True)

            return BlockDiagonalOperator([invn150, invn220], axisout=0)
        
        elif self.kind == 'wide':

            invn150 = subacq150.get_invntt_operator(det_noise=True, photon_noise=True)
            invn220 = subacq220.get_invntt_operator(det_noise=False, photon_noise=True)

            return invn150 + invn220
    def get_PySM_maps(self, config, r=0, Alens=1):

        '''
        
        Read configuration dictionary which contains every components adn the model. 
        
        Example : d = {'cmb':42, 'dust':'d0', 'synchrotron':'s0'}

        The CMB is randomly generated fram specific seed. Astrophysical foregrounds come from PySM 3.
        
        '''

        allmaps = np.zeros((len(config), 12*self.scene.nside**2, 3))
        ell=np.arange(2*self.scene.nside-1)
        mycls = give_cl_cmb(r=r, Alens=Alens)

        for k, kconf in enumerate(config.keys()):
            if kconf == 'cmb':

                np.random.seed(config[kconf])
                cmb = hp.synfast(mycls, self.scene.nside, verbose=False, new=True).T

                allmaps[k] = cmb.copy()
            
            elif kconf == 'dust':

                nu0 = self.comp[k].__dict__['_fixed_params']['nu0']
                sky=pysm3.Sky(nside=self.scene.nside, preset_strings=[config[kconf]], output_unit="uK_CMB")
                #sky.components[0].mbb_index = hp.ud_grade(sky.components[0].mbb_index, 8)
                sky.components[0].mbb_temperature = 20*sky.components[0].mbb_temperature.unit
                #sky.components[0].mbb_index = hp.ud_grade(np.array(sky.components[0].mbb_index), 8)
                mydust=np.array(sky.get_emission(nu0 * u.GHz, None).T * utils.bandpass_unit_conversion(nu0*u.GHz, None, u.uK_CMB))
                    
                allmaps[k] = mydust.copy()
            elif kconf == 'synchrotron':
                nu0 = self.comp[k].__dict__['_fixed_params']['nu0']
                sky=pysm3.Sky(nside=self.scene.nside, preset_strings=[config[kconf]], output_unit="uK_CMB")
                mysync=np.array(sky.get_emission(nu0 * u.GHz, None).T * utils.bandpass_unit_conversion(nu0*u.GHz, None, u.uK_CMB))
                allmaps[k] = mysync.copy()
            elif kconf == 'coline':
                
                #sky = pysm3.Sky(nside=self.nside, preset_strings=['co2'], output_unit="uK_CMB")
                #nu0 = sky.components[0].line_frequency['21'].value
                
                #myco=np.array(sky.get_emission(nu0 * u.GHz, None).T * utils.bandpass_unit_conversion(nu0*u.GHz, None, u.uK_CMB))
                # 10 is for reproduce the PsYM template
                m = hp.ud_grade(hp.read_map(path_to_data+'CO_line.fits') * 10, self.scene.nside)
                #print(self.nside)   
                mP = polarized_I(m, self.scene.nside)
                #print(mP.shape)
                myco = np.zeros((12*self.scene.nside**2, 3))
                myco[:, 0] = m.copy()
                myco[:, 1:] = mP.T.copy()
                allmaps[k] = myco.copy()
            else:
                raise TypeError('Choose right foreground model (d0, s0, ...)')

        #if len(nus) == 1:
        #    allmaps = allmaps[0].copy()
            
        return allmaps

class OtherDataParametric:

    def __init__(self, nus, nside, comp, nintegr=2):
        
        if nintegr == 1:
            raise TypeError('The integration of external data should be greater than 1')
        
        self.nintegr = nintegr
        pkl_file = open(path_to_data+'AllDataSet_Components_MapMaking.pkl', 'rb')
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
    def get_operator(self, beta, convolution, Amm=None, myfwhm=None, nu_co=None, comm=None):
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
                C = HealpixConvolutionGaussianOperator(fwhm=fwhm, lmax=2*self.nside)
            
                if Amm is not None:
                    D = get_mixing_operator(beta, np.array([self.allnus[k]]), Amm=Amm[k], comp=self.comp, nside=self.nside, active=False)
                else:
                    D = get_mixing_operator(beta, np.array([self.allnus[k]]), Amm=None, comp=self.comp, nside=self.nside, active=False)
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

class JointAcquisitionFrequencyMapMaking:

    def __init__(self, d, kind, Nrec, Nsub, H=None):

        self.kind = kind
        self.d = d
        self.Nrec = Nrec
        self.Nsub = Nsub
        #self.qubic = qubic
        self.qubic = QubicFullBandSystematic(self.d, comp=[], Nsub=self.Nsub, Nrec=self.Nrec, kind=self.kind, H=H)
        self.scene = self.qubic.scene
        self.pl143 = PlanckAcquisition(143, self.scene)
        self.pl217 = PlanckAcquisition(217, self.scene)




    def get_operator(self, angle_hwp=None, fwhm=None):
        
        if self.kind == 'QubicIntegrated':   # Classic intrument
            
            # Get QUBIC operator
            H_qubic = self.qubic.get_operator(convolution=convolution, myfwhm=myfwhm)
            R_qubic = ReshapeOperator(H_qubic.operands[0].shapeout, H_qubic.operands[0].shape[0])
            R_planck = ReshapeOperator((12*self.qubic.scene.nside**2, 3), (12*self.qubic.scene.nside**2*3))

            # Create an empty list to hold operators
            full_operator = []

            if self.Nrec == 1:
                Operator = [R_qubic(H_qubic), R_planck]
                return BlockColumnOperator(Operator, axisout=0)
            
            else:
                
                for irec in range(self.Nrec):
                    Operator = [R_qubic(H_qubic.operands[irec])]
                    for jrec in range(self.Nrec):
                        if irec == jrec:
                            Operator += [R_planck]
                        else:
                            Operator += [R_planck*0]
                        
                    full_operator += [BlockColumnOperator(Operator, axisout=0)]
                
                return BlockRowOperator(full_operator, new_axisin=0)

        
        elif self.kind == 'wide':      # WideBand intrument

            # Get QUBIC operator
            H_qubic = self.qubic.get_operator(angle_hwp=angle_hwp, fwhm=fwhm)
            R_qubic = ReshapeOperator(H_qubic.operands[0].shapeout, H_qubic.operands[0].shape[0])
            
            R_planck = ReshapeOperator((12*self.qubic.scene.nside**2, 3), (12*self.qubic.scene.nside**2*3))
            

            if self.Nrec == 1:
                operator = [R_qubic(H_qubic), R_planck, R_planck]
                return BlockColumnOperator(operator, axisout=0)
            
            else:

                full_operator = []
                for irec in range(self.Nrec):
                    operator = [R_qubic(H_qubic.operands[irec])]
                    for jrec in range(self.Nrec):
                        if irec == jrec:
                            operator += [R_planck]
                        else:
                            operator += [R_planck*0]
                    full_operator += [BlockColumnOperator(operator, axisout=0)]
                
                return BlockRowOperator(full_operator, new_axisin=0)
            
        elif self.kind == 'two':

            # Get QUBIC operator
            if self.Nrec == 2:
                H_qubic = self.qubic.get_operator(angle_hwp=angle_hwp, fwhm=fwhm).operands[1]
            else:
                H_qubic = self.qubic.get_operator(angle_hwp=angle_hwp, fwhm=fwhm)
            R_qubic = ReshapeOperator(H_qubic.operands[0].shapeout, H_qubic.operands[0].shape[0])
            R_planck = ReshapeOperator((12*self.qubic.scene.nside**2, 3), (12*self.qubic.scene.nside**2*3))
            opefull = []
            for ifp in range(2):
                ope_per_fp = []
                for irec in range(int(self.Nrec/2)):
                    if self.Nrec > 2:
                        operator = [R_qubic * H_qubic.operands[ifp].operands[irec]]
                    else:
                        operator = [R_qubic * H_qubic.operands[ifp]]
                    for jrec in range(int(self.Nrec/2)):
                        if irec == jrec:
                            operator += [R_planck]
                        else:
                            operator += [R_planck*0]
                    ope_per_fp += [BlockColumnOperator(operator, axisout=0)]
                opefull += [BlockRowOperator(ope_per_fp, new_axisin=0)]
            if self.Nrec == 2:
                h = BlockDiagonalOperator(opefull, new_axisin=0)
                _r = ReshapeOperator((h.shapeout[0], h.shapeout[1]), (h.shapeout[0]*h.shapeout[1]))
                return _r * h
            else:
                return BlockDiagonalOperator(opefull, axisout=0)

        else:
            raise TypeError(f'Instrument type {self.kind} is not recognize')
        

    def get_invntt_operator(self, weight_planck=1, beam_correction=None, seenpix=None, mask=None):
        

        if beam_correction is None :
                beam_correction = [0]*self.Nrec

        if self.kind == 'wide':

            invn_q = self.qubic.get_invntt_operator()
            R = ReshapeOperator(invn_q.shapeout, invn_q.shape[0])
            invn_q = [R(invn_q(R.T))]


            invntt_planck143 = weight_planck*self.pl143.get_invntt_operator(beam_correction=beam_correction[0], mask=mask, seenpix=seenpix)
            invntt_planck217 = weight_planck*self.pl217.get_invntt_operator(beam_correction=beam_correction[0], mask=mask, seenpix=seenpix)
            R_planck = ReshapeOperator(invntt_planck143.shapeout, invntt_planck143.shape[0])
            invN_143 = R_planck(invntt_planck143(R_planck.T))
            invN_217 = R_planck(invntt_planck217(R_planck.T))
            if self.Nrec == 1:
                invNe = [invN_143, invN_217]
            else:
                invNe = [invN_143]*int(self.Nrec/2) + [invN_217]*int(self.Nrec/2)
            invN = invn_q + invNe
            return BlockDiagonalOperator(invN, axisout=0)
        
        elif self.kind == 'two':

            invn_q_150 = self.qubic.get_invntt_operator().operands[0]
            invn_q_220 = self.qubic.get_invntt_operator().operands[1]
            R = ReshapeOperator(invn_q_150.shapeout, invn_q_150.shape[0])


            invntt_planck143 = weight_planck*self.pl143.get_invntt_operator(beam_correction=beam_correction[0], mask=mask, seenpix=seenpix)
            invntt_planck217 = weight_planck*self.pl217.get_invntt_operator(beam_correction=beam_correction[0], mask=mask, seenpix=seenpix)
            R_planck = ReshapeOperator(invntt_planck143.shapeout, invntt_planck143.shape[0])
            invN_143 = R_planck(invntt_planck143(R_planck.T))
            invN_217 = R_planck(invntt_planck217(R_planck.T))
            invN = [R(invn_q_150(R.T))]
            for i in range(int(self.Nrec/2)):
                invN += [R_planck(invntt_planck143(R_planck.T))]#, 
                    #R(invn_q_220(R.T)), R_planck(invntt_planck217(R_planck.T))]
            
            invN += [R(invn_q_220(R.T))]

            for i in range(int(self.Nrec/2)):
                invN += [R_planck(invntt_planck217(R_planck.T))]
            
            return BlockDiagonalOperator(invN, axisout=0)

            
        '''
        elif self.kind == 'QubicIntegrated':
            if beam_correction is None :
                beam_correction = [0]*self.Nrec
            else:
                if type(beam_correction) is not list:
                    raise TypeError('Beam correction should be a list')
                if len(beam_correction) != self.Nrec:
                    raise TypeError('List of beam correction should have Nrec elements')


            invntt_qubic = self.qubic.get_invntt_operator(det_noise, photon_noise)
            R_qubic = ReshapeOperator(invntt_qubic.shapeout, invntt_qubic.shape[0])
            Operator = [R_qubic(invntt_qubic(R_qubic.T))]

            for i in range(self.Nrec):
                invntt_planck = weight_planck*self.planck.get_invntt_operator(beam_correction=beam_correction[i], mask=mask, seenpix=seenpix)
                R_planck = ReshapeOperator(invntt_planck.shapeout, invntt_planck.shape[0])
                Operator.append(R_planck(invntt_planck(R_planck.T)))

            return BlockDiagonalOperator(Operator, axisout=0)
        '''
class JointAcquisitionComponentsMapMaking:

    def __init__(self, d, kind, comp, Nsub, nus_external, nintegr, nu_co=None, H=None, ef150=3, ef220=3):

        self.kind = kind
        self.d = d
        self.Nsub = Nsub
        self.comp = comp
        self.nus_external = nus_external
        self.nintegr = nintegr
        #self.qubic = qubic
        self.qubic = QubicFullBandSystematic(self.d, comp=self.comp, Nsub=self.Nsub, Nrec=1, kind=self.kind, nu_co=nu_co, H=H, effective_duration150=ef150, effective_duration220=ef220)
        self.scene = self.qubic.scene
        self.external = OtherDataParametric(self.nus_external, self.scene.nside, self.comp, self.nintegr)

    def get_operator(self, beta, Amm=None, gain=None, fwhm=None, nu_co=None):
        
        if Amm is not None:
            Aq = Amm[:self.Nsub]
            Ap = Amm[self.Nsub:]
        else:
            Aq = None
            Ap = None
        
        Hq = self.qubic.get_operator(beta=beta, gain=gain, fwhm=fwhm, Amm=Aq)
        
        
        Rq = ReshapeOperator(Hq.shapeout, (Hq.shapeout[0]*Hq.shapeout[1]))
        try:
            mpidist = self.qubic.mpidist
        except:
            mpidist = None

        
        He = self.external.get_operator(beta=beta, convolution=False, comm=mpidist, nu_co=nu_co, Amm=Ap)
        
        return BlockColumnOperator([Rq * Hq, He], axisout=0)
    
    def get_invntt_operator(self, fact=None, mask=None):

        invNq = self.qubic.get_invntt_operator()
        R = ReshapeOperator(invNq.shapeout, invNq.shape[0])
        invNe = self.external.get_invntt_operator(fact=fact, mask=mask)

        return BlockDiagonalOperator([R(invNq(R.T)), invNe], axisout=0)
