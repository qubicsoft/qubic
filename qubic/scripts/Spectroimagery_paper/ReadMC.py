import sys
import glob

import healpy as hp
import numpy as np
from astropy.io import fits

from joblib import Parallel, delayed
import multiprocessing

from pysimulators import FitsArray
from qubic import Xpol
from qubic import apodize_mask


# =============== Save a simulation ==================
def save_simu_fits(maps_recon, cov, nus, nus_edge, maps_convolved,
                   save_dir, simu_name):
    """
    Save a complete simulation in a .fits file for one number of reconstructed subbands.
    Parameters
    ----------
    maps_recon : array
        Reconstructed maps in each subband.
    cov : array
        Coverage of the sky.
    nus : array
        Central frequencies of each subband reconstructed.
    nus_edge : array
        Edge frequencies of each subband reconstructed.
    maps_convolved : array
        Maps convolved at the resolution of each subband.
    save_dir : str
        Directory where the .fits is saved.
    simu_name : str
        Name of the simulation.

    """

    hdu_primary = fits.PrimaryHDU()
    hdu_recon = fits.ImageHDU(data=maps_recon, name='maps_recon')
    hdu_cov = fits.ImageHDU(data=cov, name='coverage')
    hdu_nus = fits.ImageHDU(data=nus, name='central_freq', )
    hdu_nus_edge = fits.ImageHDU(data=nus_edge, name='edge_freq')
    hdu_convolved = fits.ImageHDU(data=maps_convolved, name='maps_convolved')

    the_file = fits.HDUList([hdu_primary, hdu_recon, hdu_cov, hdu_nus,
                             hdu_nus_edge, hdu_convolved])
    the_file.writeto(save_dir + simu_name, 'warn')


# =============== Read saved maps ==================
def get_seenmap_new(file):
    """
    Returns an array with the pixels seen or not.
    Parameters
    ----------
    file : str
        A fits file saved from a simulation.

    Returns
    -------
    seenmap : array
        Array of booleans of shape #pixels,
        True inside the patch and False outside.
    """
    simu = fits.open(file)
    map = simu['MAPS_RECON'].data
    npix = np.shape(map)[1]
    seenmap = np.full(npix, True, dtype=bool)

    bla = np.mean(map, axis=(0, 2)) != hp.UNSEEN
    seenmap *= bla
    return seenmap


def get_maps(file):
    """
    Returns the full maps of a simulation.
    Parameters
    ----------
    file : str
        A fits file saved from a simulation.

    Returns
    -------
    Reconstructed maps, convolved maps and the difference between both,
    all with a shape (#subbands, #pixels, 3).

    """

    simu = fits.open(file)

    maps_recon = simu['MAPS_RECON'].data
    maps_convo = simu['MAPS_CONVOLVED'].data

    diff = maps_recon - maps_convo

    return maps_recon, maps_convo, diff


def get_patch(file, seenmap):
    """
        Returns the observed patch in the maps to save memory.
        Parameters
        ----------
        file : str
            A fits file saved from a simulation.
        seenmap : array
            Array of booleans of shape #pixels,
            True inside the patch and False outside.

        Returns
        -------
        Reconstructed patches, convolved patches and difference between both,
        all with a shape (#subbands, #pixels_seen, 3).

        """

    maps_recon, maps_convo, diff = get_maps(file)

    maps_recon_cut = maps_recon[:, seenmap, :]
    maps_convo_cut = maps_convo[:, seenmap, :]
    diff_cut = diff[:, seenmap, :]

    return maps_recon_cut, maps_convo_cut, diff_cut


def get_patch_many_files(rep_simu, name):
    """
    Get all the patches you want to analyze from many fits files.
    Parameters
    ----------
    rep_simu : str
        Repository where the fits files are.
    name : str
        Name of the files you are interested in.
    Returns
    -------
    A list with the names of all the files you took.
    Three arrays containing the reconstructed patches, the convolved patches
    and the difference between both.

    """
    all_fits = glob.glob(rep_simu + name)
    nfiles = len(all_fits)
    print('{} files have been found.'.format(nfiles))

    seenmap = get_seenmap_new(all_fits[0])

    all_patch_recon = np.empty((nfiles,), dtype=object)
    all_patch_convo = np.empty((nfiles,), dtype=object)
    all_patch_diff = np.empty((nfiles,), dtype=object)

    for i, fits in enumerate(all_fits):
        patch_recon, patch_convo, patch_diff = get_patch(fits, seenmap)
        all_patch_recon[i] = patch_recon
        all_patch_convo[i] = patch_convo
        all_patch_diff[i] = patch_diff

    return all_fits, all_patch_recon, all_patch_convo, all_patch_diff


def get_maps_many_files(rep_simu, name):
    """
    Get all the maps you want to analyze from many fits files.
    Parameters
    ----------
    rep_simu : str
        Repository where the fits files are.
    name : str
        Name of the files you are interested in.
    Returns
    -------
    A list with the names of all the files you took.
    Three arrays containing the reconstructed maps, the convolved maps
    and the difference between both.

    """
    all_fits = glob.glob(rep_simu + name)
    nfiles = len(all_fits)
    print('{} files have been found.'.format(nfiles))

    all_maps_recon = np.empty((nfiles,), dtype=object)
    all_maps_convo = np.empty((nfiles,), dtype=object)
    all_maps_diff = np.empty((nfiles,), dtype=object)

    for i, fits in enumerate(all_fits):
        map_recon, map_convo, map_diff = get_maps(fits)
        all_maps_recon[i] = map_recon
        all_maps_convo[i] = map_convo
        all_maps_diff[i] = map_diff

    return all_fits, all_maps_recon, all_maps_convo, all_maps_diff


# ============ OLD Functions to get maps ===========#
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
        sys.stdout.write('\r Reading: ' + files[i] + ' ({:3.1f} %)'.format(100 * (i + 1) * 1. / len(files)))
        m = FitsArray(files[i])
        bla = np.mean(m, axis=(0, 2)) != hp.UNSEEN
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
    npixok = np.sum(seenmap)
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
    """
    allmost equivalent to get_all_maps, used in the functions that compute spectra
    """
    if not silent:
        print('Reading Files')
    nn = len(files)
    mm = FitsArray(files[0])
    sh = np.shape(mm)
    maps = np.zeros((nn, sh[0], sh[1], sh[2]))
    for i in xrange(nn):
        maps[i, :, :, :] = FitsArray(files[i])

    totmap = np.sum(np.sum(np.sum(maps, axis=0), axis=0), axis=1)
    seenmap = totmap > -1e20
    return maps, seenmap


def get_maps_residuals(frec, fconv=None, silent=False):
    mrec, seenmap = maps_from_files(frec)
    if fconv is None:
        if not silent:
            print('Getting Residuals from average MC')
        resid = np.zeros_like(mrec)
        mean_mrec = np.mean(mrec, axis=0)
        for i in xrange(len(frec)):
            resid[i, :, :, :] = mrec[i, :, :, :] - mean_mrec[:, :, :]
    else:
        if not silent:
            print('Getting Residuals from convolved input maps')
        mconv, _ = maps_from_files(fconv)
        resid = mrec - mconv
    resid[:, :, ~seenmap, :] = 0
    return mrec, resid, seenmap


# ============ Functions do statistical tests on maps ===========#
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
    seen = np.where(seenmap == 1)[0]
    npixok = np.sum(seenmap)
    variance_map = np.zeros((len(nsubvals), 3, npixok)) + hp.UNSEEN
    allmeanmat = []
    allstdmat = []

    for isub in xrange(len(nsubvals)):
        print('for nsub = {}'.format(nsubvals[isub]))
        mapsout = allmapsout[isub]
        covmat_freqfreq = np.zeros((nsubvals[isub], nsubvals[isub], len(seen), 3))
        # Loop over pixels
        for p in xrange(len(seen)):
            # Loop over I Q U
            for i in xrange(3):
                mat = np.cov(mapsout[:, :, p, i].T)
                # Normalisation
                if np.size(mat) == 1:
                    variance_map[isub, i, p] = mat
                else:
                    variance_map[isub, i, p] = 1. / np.sum(np.linalg.inv(mat))
                covmat_freqfreq[:, :, p, i] = mat / np.mean(
                    mat)  # its normalization is irrelevant for the later average
        # Average and std over pixels
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
            # cov matrice freq-freq averaged over pixels
            covmat = allmeanmat[isub][:, :, iqu]
            invcovmat = np.linalg.inv(covmat)
            # Loop over pixels
            for p in xrange(npixok):
                mean_cov = np.zeros(nreals)

                # Loop over realisations
                for real in xrange(nreals):
                    vals = mapsout[real, :, p, iqu]
                    mean_cov[real] = get_mean_cov(vals, invcovmat)
                # Mean and rms over realisations
                meanmap_cov[isub, iqu, p] = np.mean(mean_cov)
                rmsmap_cov[isub, iqu, p] = np.std(mean_cov)

    return meanmap_cov, rmsmap_cov


# ============ Functions to get auto and cross spectra from maps ===========#
def get_xpol(seenmap, ns, lmin=20, delta_ell=20, apodization_degrees=5.):
    """
    Returns a Xpoll object to get spectra, the bin used and the pixel window function.
    """
    # Create a mask
    mymask = apodize_mask(seenmap, apodization_degrees)

    # Create XPol object
    lmax = 2 * ns
    xpol = Xpol(mymask, lmin, lmax, delta_ell)
    ell_binned = xpol.ell_binned
    # Pixel window function
    pw = hp.pixwin(ns)
    pwb = xpol.bin_spectra(pw[:lmax + 1])

    return xpol, ell_binned, pwb


def allcross_par(xpol, allmaps, silent=False, verbose=1):
    num_cores = multiprocessing.cpu_count()
    nmaps = len(allmaps)
    nbl = len(xpol.ell_binned)
    autos = np.zeros((nmaps, 6, nbl))
    ncross = nmaps * (nmaps - 1) / 2
    cross = np.zeros((ncross, 6, nbl))
    jcross = 0
    if not silent:
        print('Computing spectra:')

    # Auto spectra ran in //
    if not silent:
        print('  Doing All Autos ({}):'.format(nmaps))
    results_auto = Parallel(n_jobs=num_cores, verbose=verbose)(
        delayed(xpol.get_spectra)(allmaps[i]) for i in xrange(nmaps))
    for i in xrange(nmaps):
        autos[i, :, :] = results_auto[i][1]

    # Cross Spectra ran in // - need to prepare indices in a global variable
    if not silent:
        print('  Doing All Cross ({}):'.format(ncross))
    global cross_indices
    cross_indices = np.zeros((2, ncross), dtype=int)
    for i in xrange(nmaps):
        for j in xrange(i + 1, nmaps):
            cross_indices[:, jcross] = np.array([i, j])
            jcross += 1
    results_cross = Parallel(n_jobs=num_cores, verbose=verbose)(
        delayed(xpol.get_spectra)(allmaps[cross_indices[0, i]], allmaps[cross_indices[1, i]]) for i in xrange(ncross))
    for i in xrange(ncross):
        cross[i, :, :] = results_cross[i][1]

    if not silent:
        sys.stdout.write(' Done \n')
        sys.stdout.flush()

    # The error-bars are absolutely incorrect if calculated as the following...
    # There is an analytical estimate in Xpol paper.
    # See if implemented in the gitlab xpol from Tristram instead of in qubic.xpol...
    m_autos = np.mean(autos, axis=0)
    s_autos = np.std(autos, axis=0) / np.sqrt(nmaps)
    m_cross = np.mean(cross, axis=0)
    s_cross = np.std(cross, axis=0) / np.sqrt(ncross)
    return m_autos, s_autos, m_cross, s_cross


def get_maps_cl(frec, fconv=None, lmin=20, delta_ell=40, apodization_degrees=5.):
    mrec, resid, seenmap = get_maps_residuals(frec, fconv=fconv)
    sh = np.shape(mrec)
    print(sh, np.shape(resid))
    nbsub = sh[1]
    ns = hp.npix2nside(sh[2])

    from qubic import apodize_mask
    mymask = apodize_mask(seenmap, apodization_degrees)

    # Create XPol object
    from qubic import Xpol
    lmax = 2 * ns
    xpol = Xpol(mymask, lmin, lmax, delta_ell)
    ell_binned = xpol.ell_binned
    nbins = len(ell_binned)
    # Pixel window function
    pw = hp.pixwin(ns)
    pwb = xpol.bin_spectra(pw[:lmax + 1])

    # Calculate all crosses and auto
    m_autos = np.zeros((nbsub, 6, nbins))
    s_autos = np.zeros((nbsub, 6, nbins))
    m_cross = np.zeros((nbsub, 6, nbins))
    s_cross = np.zeros((nbsub, 6, nbins))
    fact = ell_binned * (ell_binned + 1) / 2. / np.pi
    for isub in xrange(nbsub):
        m_autos[isub, :, :], s_autos[isub, :, :], m_cross[isub, :, :], s_cross[isub, :, :] = \
            allcross_par(xpol, mrec[:, isub, :, :], silent=False, verbose=0)

    return mrec, resid, seenmap, ell_binned, m_autos * fact / pwb ** 2, \
           s_autos * fact / pwb ** 2, m_cross * fact / pwb ** 2, s_cross * fact / pwb ** 2
