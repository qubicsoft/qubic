import sys
import healpy as hp
import numpy as np
from scipy import interpolate

import ReadMC as rmc

from qubic import apodize_mask
from qubic import Xpol


def cov2corr(mat):
    """
    Converts a Covariance Matrix in a Correlation Matrix
    """
    newmat = mat.copy()
    sh = np.shape(mat)
    for i in xrange(sh[0]):
        for j in xrange(sh[1]):
            newmat[i, j] = mat[i, j] / np.sqrt(mat[i, i] * mat[j, j])
    return newmat


def std_profile(many_patch, nbins, nside, center, seenmap):
    """
    Get the std profile of a patch over pixels and realisations
    from the center to the border.

    Parameters
    ----------
    many_patch : array of shape (nreal, nsub, npixok, 3)
        Many realisations of one patch.
    nbins : int
        Number of bins.
    nside : int
        NSIDE in the patch.
    center : array
        Coordinates of the center of the patch in degree (lon, lat)
    seenmap : array
        Array of booleans of shape #pixels,
        True inside the patch and False outside.

    Returns
    -------
    bin_centers : array with angles associated to each bin.
    std_bin : array of shape (nbins, nsub, 3)
        Std value in each bin, for each subband and IQU.
    std_profile : array of shape (npixok, nsub, 3)
        Quadratic interpolation of the std_bin.

    """
    ang = rmc.pix2ang(nside, center, seenmap)

    bin_edges = np.linspace(0, np.max(ang), nbins + 1)
    bin_centers = 0.5 * (bin_edges[0:nbins] + bin_edges[1:])

    # Std in each bin
    nsub = np.shape(many_patch)[1]
    std_bin = np.empty((nbins, nsub, 3))
    for b in xrange(nbins):
        ok = (ang > bin_edges[b]) & (ang < bin_edges[b + 1])
        std_bin[b, :, :] = np.std(many_patch[:, :, ok, :], axis=(0, 2))

    # Interpolation to get a profile
    fit = interpolate.interp1d(bin_centers, std_bin, axis=0, kind='quadratic', fill_value='extrapolate', )
    std_profile = fit(ang)

    return bin_centers, std_bin, std_profile


def covariance_IQU_subbands(allmaps):
    """
    Returns the mean maps, averaged over pixels and realisations and the
    covariance matrices of the maps.

    Parameters
    ----------
    allmaps : list of arrays of shape (nreals, nsub, npix, 3)
        list of maps for each number of subband

    Returns
    -------
    allmean : list of arrays of shape 3*nsub
        mean for I, Q, U for each subband
    allcov : list of arrays of shape (3*nsub, 3*nsub)
        covariance matrices between stokes parameters and sub frequency bands

    """
    allmean, allcov = [], []
    for isub in xrange(len(allmaps)):
        sh = allmaps[isub].shape
        nsub = sh[1]  # Number of subbands

        mean = np.zeros(3 * nsub)
        cov = np.zeros((3 * nsub, 3 * nsub))

        for iqu in xrange(3):
            for band in xrange(nsub):
                i = 3 * band + iqu
                map_i = allmaps[isub][:, band, :, iqu]
                mean[i] = np.mean(map_i)
                for iqu2 in xrange(3):
                    for band2 in xrange(nsub):
                        j = 3 * band2 + iqu2
                        map_j = allmaps[isub][:, band2, :, iqu2]
                        cov[i, j] = np.mean((map_i - np.mean(map_i)) * (map_j - np.mean(map_j)))
        allmean.append(mean)
        allcov.append(cov)

    return allmean, allcov


# ============ Functions do statistical tests on maps ===========#
def get_covcorr1pix(maps, ipix, verbose=False):
    """

    This function return the covariance matrix for one pixel given a list of maps.
    Each of the Nreal maps has the shape (nfrec, npix, 3) (see save_simu_fits() ).

    Parameters
    -------
    maps: array
        Input maps with shape (nrealizations, nfrecons, npix, 3)
    ipix: int
        pixel where the covariance will be computed

    Return
    -------
    cov1pix: np.array
        covariance matrix for a given pixel (ipix) with shape 3*nfrec x 3*nfrec

    corr1pix: np.array
        correlation matrix for a given pixel (ipix) with shape 3*nfrec x 3*nfrec

    """

    if type(ipix) != int:
        raise TypeError('ipix has to be an integer number')

    nfrec = maps[0].shape[0]  # Sub-bands
    nreal = maps.shape[0]  # Sample realizations

    if verbose:
        print('The shape of the input map has to be: (nsample, nfrecons, npix, 3): {}'.format(maps.shape))
        print('Number of reconstructed sub-bands to analyze: {}'.format(nfrec))
        print('Number of realizations: {}'.format(nreal))
        print('Computing covariance matrix in pixel {}'.format(ipix))

    data = np.zeros((nreal, nfrec * 3))

    for j in range(nreal):
        for irec in range(nfrec):
            for istokes in range(3):
                data[j, 3 * irec + istokes] = maps[j, irec, ipix, istokes]

    cov1pix = np.cov(data, rowvar=False)
    corr1pix = np.corrcoef(data, rowvar=False)

    return cov1pix, corr1pix


def get_covcorr_patch(patch):
    """
    This function computes the covariance matrix and the correlation one for a given patch in the sky. 
    It uses get_covcorr1pix() to compute the covariance and correlation matrix for each pixel (ipix) 
    and then computes the average in the choosen pixels (patch).
    
    Asumptions: patch.shape = (nsamples, nrecons, npix_patch, 3) --> to be able to use get_covcorr1pix

    Parameters:
    -----------
    patch: np.array
        Sky patch observed (see get_patch_many_files() from ReadMC module)

    Returns:
    -----------
    cov: np.array
        Covariance matrix for the given patch in the sky. Is the simple mean between the pixels inside the patch.

    corr: np.array
        Correlation matrix for a given patch in the sky. Is the simple mean between the pixels inside the patch. 
    """

    npix = patch.shape[2]

    covpix, corrpix = [], []

    for ipix in xrange(npix):
        mat = get_covcorr1pix(patch, ipix)
        covpix.append(mat[0])
        corrpix.append(mat[1])

    meancov = np.mean(np.asarray(covpix), axis=0)
    meancorr = np.mean(np.asarray(corrpix), axis=0)

    return meancov, meancorr


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
