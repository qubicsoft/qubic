import os
import sys
import glob

import healpy as hp
import numpy as np

from joblib import Parallel, delayed
import multiprocessing

import qubic
from pysimulators import FitsArray
from qubic import Xpol
from qubic import apodize_mask


#============ Functions to get maps ===========#
def get_seenmap(files):
    """ 
    Return a list of len the number of pixels, 
    True or False if the pixel is observed or not in files.
    files : list of .fits files

    seenmap : list of len npix
    """
    print('\nGetting Observed pixels map')
    m = FitsArray(files[0])
    npix = np.shape(m)[1]
    seenmap = np.zeros(npix) == 0
    for i in xrange(len(files)):
        sys.stdout.flush()
        sys.stdout.write('\r Reading: '+files[i]+' ({:3.1f} %)'.format(100*(i+1)*1./len(files)))
        m = FitsArray(files[i])
        bla = np.mean(m, axis=(0,2)) != hp.UNSEEN
        seenmap *= bla
    sys.stdout.flush()
    return seenmap

def read_all_maps(files, nsub, seenmap):
    """
    Read all realisations of I,Q,U maps for one number of subbands 
    Only seen pixels are stored in order to save memory.

    nsub : number of subbands
    seenmap : list of booleans for pixels observed or not

    mapsout : array of shape (nreals, nsub, npixok, 3)
        
    """
    print('\nReading all maps')

    nfiles = len(files)
    print(files, nfiles)
    npixok  = np.sum(seenmap)
    mapsout = np.zeros((nfiles, nsub, npixok, 3))
    for ifile in xrange(nfiles):
        print('Doing: ' + files[ifile], 'nsub=' + str(nsub))
        sys.stdout.flush()
        sys.stdout.write('\r Reading: ' + files[ifile] + ' ({:3.1f} %)'.format(100 * (ifile + 1) * 1. / nfiles))
        mm = FitsArray(files[ifile])
        mapsout[ifile, :, :, :] = mm[:, seenmap, :]
    sys.stdout.flush()    
    return mapsout

def get_all_maps(rep, archetype, nsubvals):
    """
    This function conbines get_seenmap and read_all_maps 
    for several numbers of subbands.
    
    nsubvals : list containing the numbers of subbands used

    rep : repository path 
    archetype : list of the .fits files in rep

    allmapsout : list of arrays of len len(nsubvals)
        Each array is a mapsout for one number of subbands
    """
    
    seenmap = True
    for i in xrange(len(archetype)):
        files = glob.glob(rep + '/' + archetype[i])
        seenmap *= get_seenmap(files)
    
    allmapsout = []
    for j in xrange(len(nsubvals)):
        files = glob.glob(rep + '/' + archetype[j])
        mapsout = read_all_maps(files, nsubvals[j], seenmap)
        allmapsout.append(mapsout)

    return allmapsout, seenmap


def maps_from_files(files, silent=False):
    """allmost equivalent to get_all_maps, 
    used in the functions that compute spectra 
    """
    if not silent: print('Reading Files')
    nn = len(files)
    mm = FitsArray(files[0])
    sh = np.shape(mm)
    maps = np.zeros((nn, sh[0], sh[1], sh[2]))
    for i in xrange(nn):
        maps[i,:,:,:] = FitsArray(files[i])
        
    totmap = np.sum(np.sum(np.sum(maps, axis=0), axis=0),axis=1)
    seenmap = totmap > -1e20
    return maps, seenmap


def get_maps_residuals(frec, fconv=None, silent=False):
    mrec, seenmap = maps_from_files(frec)
    if fconv==None:
        if not silent: print('Getting Residuals from average MC')
        resid = np.zeros_like(mrec)
        mean_mrec = np.mean(mrec, axis =0)
        for i in xrange(len(frec)):
            resid[i,:,:,:] = mrec[i,:,:,:]- mean_mrec[:,:,:]
    else:
        if not silent: print('Getting Residuals from convolved input maps')
        mconv, _ = maps_from_files(fconv)
        resid = mrec-mconv
    resid[:,:,~seenmap,:] = 0
    return mrec, resid, seenmap


#============ Functions do statistical tests on maps ===========#
def get_rms_covar(nsubvals, seenmap, allmapsout):
    """Test done by Matthieu Tristram :
Calculate the variance map in each case accounting for the band-band covariance matrix for each pixel from the MC. 
This is pretty noisy so it may be interesting to get the average matrix.
We calculate all the matrices for each pixel and normalize them to average 1 
and then calculate the average matrix over the pixels.

variance_map : array of shape (len(nsubvals), 3, npixok)

allmeanmat : list of arrays (nsub, nsub, 3)
        Mean over pixels of the cov matrices freq-freq
allstdmat : list of arrays (nsub, nsub, 3)
        Std over pixels of the cov matrices freq-freq
"""
    print('\nCalculating variance map with freq-freq cov matrix for each pixel from MC')
    seen =  np.where(seenmap == 1)[0]
    npixok = np.sum(seenmap)
    variance_map = np.zeros((len(nsubvals), 3, npixok)) + hp.UNSEEN
    allmeanmat = []
    allstdmat = []
    
    for isub in xrange(len(nsubvals)):
        print('for nsub = {}'.format(nsubvals[isub]))
        mapsout = allmapsout[isub]
        covmat_freqfreq = np.zeros((nsubvals[isub], nsubvals[isub], len(seen), 3))
        #Loop over pixels
        for p in xrange(len(seen)):
            #Loop over I Q U
            for i in xrange(3):
                mat = np.cov(mapsout[:,:,p,i].T)
                #Normalisation
                if np.size(mat) == 1: 
                    variance_map[isub, i, p] = mat
                else: 
                    variance_map[isub, i, p] = 1. / np.sum(np.linalg.inv(mat))
                covmat_freqfreq[:, :, p, i] = mat / np.mean(mat) ### its normalization is irrelevant for the later average
        #Average and std over pixels
        meanmat = np.zeros((nsubvals[isub], nsubvals[isub], 3))
        stdmat = np.zeros((nsubvals[isub], nsubvals[isub], 3))
        for i in xrange(3):
            meanmat[:, :, i] = np.mean(covmat_freqfreq[:, :, :, i], axis=2)
            stdmat[:, :, i] = np.std(covmat_freqfreq[:, :, :, i], axis=2)

        allmeanmat.append(meanmat)
        allstdmat.append(stdmat)
    return np.sqrt(variance_map), allmeanmat, allstdmat



def get_mean_cov(vals, invcov):
    AtNid = np.sum(np.dot(invcov, vals))
    AtNiA_inv = 1. / np.sum(invcov)
    mean_cov = AtNid * AtNiA_inv
    return mean_cov 
    
 
def get_rms_covarmean(nsubvals, seenmap, allmapsout, allmeanmat):
    """
    RMS map and mean map over the realisations using the pixel
    averaged freq-freq covariance matrix computed with get_rms_covar
    meanmap_cov : array of shape (len(nsubvals), 3, npixok)
    
    rmsmap_cov : array of shape (len(nsubvals), 3, npixok)

    """
    
    print('\n\nCalculating variance map with pixel averaged freq-freq cov matrix from MC')
    npixok = np.sum(seenmap)

    rmsmap_cov = np.zeros((len(nsubvals), 3, npixok)) + hp.UNSEEN
    meanmap_cov = np.zeros((len(nsubvals), 3, npixok)) + hp.UNSEEN

    for isub in xrange(len(nsubvals)):
        print('For nsub = {}'.format(nsubvals[isub]))
        mapsout = allmapsout[isub]
        sh = mapsout.shape
        nreals = sh[0]
        for iqu in xrange(3):
            #cov matrice freq-freq averaged over pixels
            covmat = allmeanmat[isub][:,:,iqu] 
            invcovmat = np.linalg.inv(covmat)
            #Loop over pixels
            for p in xrange(npixok):
                mean_cov = np.zeros(nreals)

                #Loop over realisations
                for real in xrange(nreals):
                    vals = mapsout[real,:,p,iqu]
                    mean_cov[real] = get_mean_cov(vals, invcovmat)
                #Mean and rms over realisations
                meanmap_cov[isub,iqu,p] = np.mean(mean_cov)
                rmsmap_cov[isub,iqu,p] = np.std(mean_cov)
              
    return meanmap_cov, rmsmap_cov

#============ Functions to get auto and cross spectra from maps ===========#
def get_xpol(seenmap, ns, lmin=20, delta_ell=20, apodization_degrees=5.):
    """
    Returns a Xpoll object to get spectra, the bin used and the pixel window function.
    """
    #### Create a mask
    mymask = apodize_mask(seenmap, apodization_degrees)

    #### Create XPol object
    lmax = 2 * ns
    xpol = Xpol(mymask, lmin, lmax, delta_ell)
    ell_binned = xpol.ell_binned
    # Pixel window function
    pw = hp.pixwin(ns)
    pwb = xpol.bin_spectra(pw[:lmax+1])

    return xpol, ell_binned, pwb

def allcross_par(xpol, allmaps, silent=False, verbose=1):
    num_cores = multiprocessing.cpu_count()
    nmaps = len(allmaps)
    nbl = len(xpol.ell_binned)
    autos = np.zeros((nmaps,6,nbl))
    ncross = nmaps*(nmaps-1)/2
    cross = np.zeros((ncross, 6, nbl))
    jcross = 0
    if not silent: 
        print('Computing spectra:')

    #### Auto spectra ran in //
    if not silent: print('  Doing All Autos ({}):'.format(nmaps))
    results_auto = Parallel(n_jobs=num_cores,verbose=verbose)(delayed(xpol.get_spectra)(allmaps[i]) for i in xrange(nmaps))
    for i in xrange(nmaps): autos[i,:,:] = results_auto[i][1]

    #### Cross Spectra ran in // - need to prepare indices in a global variable
    if not silent: print('  Doing All Cross ({}):'.format(ncross))
    global cross_indices 
    cross_indices = np.zeros((2, ncross), dtype=int)
    for i in xrange(nmaps):
        for j in xrange(i+1, nmaps):
            cross_indices[:,jcross] = np.array([i,j])
            jcross += 1
    results_cross = Parallel(n_jobs=num_cores,verbose=verbose)(delayed(xpol.get_spectra)(allmaps[cross_indices[0,i]], allmaps[cross_indices[1,i]]) for i in xrange(ncross))
    for i in xrange(ncross): cross[i,:,:] = results_cross[i][1]

    if not silent: 
        sys.stdout.write(' Done \n')
        sys.stdout.flush()

    #### The error-bars are absolutely incorrect if calculated as the following... 
    # There is an analytical estimate in Xpol paper. See if implemented in the gitlab xpol from Tristram instead of in qubic.xpol...
    m_autos = np.mean(autos, axis = 0)
    s_autos = np.std(autos, axis = 0) / np.sqrt(nmaps)
    m_cross = np.mean(cross, axis = 0)
    s_cross = np.std(cross, axis = 0) / np.sqrt(ncross)
    return m_autos, s_autos, m_cross, s_cross



def get_maps_cl(frec, fconv=None, lmin=20, delta_ell=40, apodization_degrees=5.):
    mrec, resid, seenmap = get_maps_residuals(frec,fconv=fconv)
    sh = np.shape(mrec)
    print(sh, np.shape(resid))
    nbsub = sh[1]
    ns = hp.npix2nside(sh[2])

    from qubic import apodize_mask
    mymask = apodize_mask(seenmap, apodization_degrees)


    #### Create XPol object
    from qubic import Xpol
    lmax = 2*ns
    xpol = Xpol(mymask, lmin, lmax, delta_ell)
    ell_binned = xpol.ell_binned
    nbins = len(ell_binned)
    # Pixel window function
    pw = hp.pixwin(ns)
    pwb = xpol.bin_spectra(pw[:lmax+1])

    #### Calculate all crosses and auto
    m_autos = np.zeros((nbsub, 6, nbins))
    s_autos = np.zeros((nbsub, 6, nbins))
    m_cross = np.zeros((nbsub, 6, nbins))
    s_cross = np.zeros((nbsub, 6, nbins))
    fact = ell_binned * (ell_binned+1) /2. /np.pi
    for isub in xrange(nbsub):
        m_autos[isub, :, :], s_autos[isub, :, :], m_cross[isub, :, :], s_cross[isub, :, :] = allcross_par(xpol, mrec[:,isub,:,:], silent=False, verbose=0)

    return mrec, resid, seenmap, ell_binned, m_autos*fact/pwb**2, s_autos*fact/pwb**2, m_cross*fact/pwb**2, s_cross*fact/pwb**2

#===================== Functions for dust ==================#

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


def dust_spectra(ll, nu):
    fact = (ll * (ll + 1)) / (2 * np.pi)
    coef = 1.39e-2
    spectra_dust = [np.zeros(len(ll)), 
                  coef * (ll / 80.)**(-0.42) / (fact * 0.52), 
                  coef * (ll / 80.)**(-0.42) / fact, 
                  np.zeros(len(ll))]
    sc_dust = scaling_dust(150, nu)
    return fact * sc_dust * spectra_dust



