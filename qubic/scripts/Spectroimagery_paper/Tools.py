import healpy as hp
import numpy as np
from scipy import interpolate


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


def pix2ang(ns, center, seenmap):
    """
    Return the angles between the vector of the central pixel
    and the vector of each pixel seen.
    """
    # central pixel vector
    v0 = hp.ang2vec(center[0], center[1], lonlat=True)
    # indices of the pixels seen
    ip = np.arange(12 * ns ** 2)[seenmap]
    # vectors associated to each pixel seen
    vpix = hp.pix2vec(ns, ip)

    return np.degrees(np.arccos(np.dot(v0, vpix)))


def myprofile(ang, maps, nbins):
    """
    Return the std profile over realisations

    Parameters
    ----------
    ang
    maps : array of shape (nreals, nsub, npixok, 3)
    nbins

    Returns
    -------
    std_bin : array of shape (nbins, nsub, 3)
    allstd_profile : list of len 3*nsub
    """
    sh = maps.shape
    bin_edges = np.linspace(0, np.max(ang), nbins + 1)
    bin_centers = 0.5 * (bin_edges[0:nbins] + bin_edges[1:])

    std_bin = np.zeros((nbins, sh[1], 3))
    allstd_profile = []
    for l in xrange(sh[1]):
        for i in xrange(3):
            for b in xrange(nbins):
                ok = (ang > bin_edges[b]) & (ang < bin_edges[b + 1])
                std_bin[b, l, i] = np.std(maps[:, l, ok, i])
            fit = interpolate.interp1d(bin_centers, std_bin[:, l, i], fill_value='extrapolate')
            std_profile = fit(ang)
            allstd_profile.append(std_profile)
    return bin_centers, std_bin, allstd_profile


def covariance_IQU_subbands(allmaps):
    """
    Returns the mean maps, averaged over pixels and realisations and the covariance matrices of the maps.

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
