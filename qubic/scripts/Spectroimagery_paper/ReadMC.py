import os
import sys
import glob

import healpy as hp
import numpy as np

import qubic
from pysimulators import FitsArray

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




