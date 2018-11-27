from __future__ import division
import healpy as hp
import numpy as np
import matplotlib.pyplot as mp
import qubic

# These two functions should be rewritten and added to qubic package
def scaling_dust(freq1, freq2, sp_index=1.59): 
    '''
    Calculate scaling factor for dust contamination
    Frequencies are in GHz
    '''
    freq1 = float(freq1)
    freq2 = float(freq2)
    x1 = freq1 / 56.78
    x2 = freq2 / 56.78
    S1 = x1**2. * np.exp(x1) / (np.exp(x1) - 1)**2.
    S2 = x2**2. * np.exp(x2) / (np.exp(x2) - 1)**2.
    vd = 375.06 / 18. * 19.6
    scaling_factor_dust = (np.exp(freq1 / vd) - 1) / \
                          (np.exp(freq2 / vd) - 1) * \
                          (freq2 / freq1)**(sp_index + 1)
    scaling_factor_termo = S1 / S2 * scaling_factor_dust
    return scaling_factor_termo

def cmb_plus_dust(cmb, dust, Nbsubbands, sub_nus, kind='IQU'):
    '''
    Sum up clean CMB map with dust using proper scaling coefficients
    '''
    Nbpixels = cmb.shape[0]
    nstokes =  len(kind) #Number of stokes parameters used in the simu
    x0 = np.zeros((Nbsubbands, Nbpixels, 3))
    #Let's fill the maps:
    for i in range(Nbsubbands):
      for istokes in xrange(nstokes):
        if kind == 'QU': #This condition keeps the order IQU in the healpix map
          x0[i, :, istokes+1] = cmb.T[istokes+1] + dust.T[istokes+1] * scaling_dust(150, sub_nus[i], 1.59)
        else:
          x0[i, :, istokes] = cmb.T[istokes] + dust.T[istokes] * scaling_dust(150, sub_nus[i], 1.59)
    return x0

def create_input_sky(d, skypars):
    Nf = int(d['nf_sub'])
    band = d['filter_nu']/1e9
    filter_relative_bandwidth = d['filter_relative_bandwidth']
    _, _, nus_in, _, _, Nbbands_in = qubic.compute_freq(band, filter_relative_bandwidth, Nf)
    # seed
    if d['seed']:
        np.random.seed(d['seed'])
        # Generate the input CMB map
        sp = qubic.read_spectra(skypars['r'])
        cmb = np.array(hp.synfast(sp, d['nside'], new=True, pixwin=True, verbose=False)).T
        # Generate the dust map
        coef = skypars['dust_coeff']
        ell = np.arange(1, 3*d['nside'])
        fact = (ell * (ell + 1)) / (2 * np.pi)
        spectra_dust = [np.zeros(len(ell)), 
                        coef * (ell / 80.)**(-0.42) / (fact * 0.52), 
                        coef * (ell / 80.)**(-0.42) / fact, 
                        np.zeros(len(ell))]
        dust = np.array(hp.synfast(spectra_dust, d['nside'], new=True, pixwin=True, verbose=False)).T
        
        # Combine CMB and dust. As output we have N 3-component maps of sky.
        x0 = cmb_plus_dust(cmb, dust, Nbbands_in, nus_in, d['kind'])
        return x0
    
    
def create_acquisition_operator_TOD(pointing, d):
    # Polychromatic instrument model
    q = qubic.QubicMultibandInstrument(d)
    # scene
    s = qubic.QubicScene(d)
    # number of sub frequencies to build the TOD
    _, nus_edge_in, _, _, _, _ = qubic.compute_freq(d['filter_nu']/1e9, 
                                                    d['filter_relative_bandwidth'], d['nf_sub']) # Multiband instrument model
    # Multi-band acquisition model for TOD fabrication
    atod = qubic.QubicMultibandAcquisition(q, pointing, s, d, nus_edge_in)
    return atod

def create_acquisition_operator_REC(pointing, d, nf_sub_rec):
    # Polychromatic instrument model
    q = qubic.QubicMultibandInstrument(d)
    # scene
    s = qubic.QubicScene(d)
    # number of sub frequencies for reconstruction
    _, nus_edge, _, _, _, _ = qubic.compute_freq(d['filter_nu']/1e9, 
                                                 d['filter_relative_bandwidth'], nf_sub_rec)
    ### Operators for Maps Reconstruction ################################################
    arec = qubic.QubicMultibandAcquisition(q, pointing, s, d, nus_edge)
    return arec


def create_TOD(d, pointing, x0):
    atod = create_acquisition_operator_TOD(pointing, d)
    TOD, _ = atod.get_observation(x0, noiseless=d['noiseless'])
    return TOD

def reconstruct_maps(TOD, d, pointing, nf_sub_rec, x0=None):
    _, nus_edge, nus, _, _, _ = qubic.compute_freq(d['filter_nu']/1e9, 
                                                   d['filter_relative_bandwidth'], nf_sub_rec)
    arec = create_acquisition_operator_REC(pointing, d, nf_sub_rec)
    maps_recon = arec.tod2map(TOD, tol=d['tol'], maxiter=1500)
    cov = arec.get_coverage()
    if x0 is None:
        return maps_recon, cov, nus, nus_edge
    else:
        _, maps_convolved = arec.get_observation(x0)
        maps_convolved = np.array(maps_convolved)
        return maps_recon, cov, nus, nus_edge, maps_convolved
    
    
    






















