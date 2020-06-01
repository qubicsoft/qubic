from __future__ import division, print_function
import healpy as hp
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

import ReadMC as rmc

import qubic

from qubicpack.utilities import Qubic_DataDir


# ============ Functions do statistical tests on maps ===========#
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
    ang : array with angles associated to each pixel
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
    for b in range(nbins):
        ok = (ang > bin_edges[b]) & (ang < bin_edges[b + 1])
        std_bin[b, :, :] = np.std(many_patch[:, :, ok, :], axis=(0, 2))

    # Interpolation to get a profile
    fit = interpolate.interp1d(bin_centers, std_bin, axis=0, kind='linear', fill_value='extrapolate')
    std_profile = fit(ang)

    return bin_centers, ang, std_bin, std_profile


def get_residuals(name, rep_simu, residuals_way, irec):
    """
    Compute residuals in a given way.

    Parameters
    ----------
    name : str
        Simulation file.
    rep_simu : str
        Path containing the simulations.
    residuals_way : str
        Way to compute residuals. 3 keywords : noiseless, conv or mean_recon
    irec : int
        Number of reconstructed subbands.

    Returns
    -------
        residuals : array of shape (#reals, #bands, #pixels, 3)
    """

    # Dictionary saved during the simulation
    d = qubic.qubicdict.qubicDict()
    d.read_from_file(rep_simu + name + '.dict')
    nf_recon = d['nf_recon']
    if irec not in nf_recon:
        raise ValueError('Invalid number of freq. {0} not in {1}'.format(irec, nf_recon))

    _, maps_recon_patch, _, maps_diff_patch = rmc.get_patch_many_files(rep_simu + name,
                                                                       '*nfrecon{}*False*'.format(irec),
                                                                       verbose=False)

    if residuals_way == 'noiseless':
        _, patch_recon_nl, patch_conv_nl, patch_diff_nl = rmc.get_patch_many_files(rep_simu + name,
                                                                                   '*nfrecon{}*True*'.format(irec),
                                                                                   verbose=False)
        residuals = maps_recon_patch - patch_recon_nl

    elif residuals_way == 'conv':
        residuals = maps_diff_patch

    elif residuals_way == 'mean_recon':
        residuals = maps_recon_patch - np.mean(maps_recon_patch, axis=0)

    else:
        raise ValueError('The way to compute residuals is not valid.')

    return residuals


def rms_method(name, residuals_way, zones=1):
    """
    Get the std of the residuals from one simulation. 
    STD are computed over realisations and pixels for I, Q, U separately.

    Parameters
    ----------
    name : str
        Simulation file.
    residuals_way : str
        Way to compute residuals. 3 keywords : noiseless, conv or mean_recon
    zones : int
        Number of zones to divide the patch.

    Returns
    -------
    rms_I, rms_Q, rms_U : dictionarys containing RMS for IQU 
    setpar : a dict with some parameters of the simu.
    """
    # Get the repository where the simulation is
    rep_simu = Qubic_DataDir(datafile=name + '.dict') + '/'
    # print('rep_simu : ', rep_simu)

    # Dictionary saved during the simulation
    d = qubic.qubicdict.qubicDict()
    d.read_from_file(rep_simu + name + '.dict')
    setpar = {'tol': d['tol'], 'nep': d['detector_nep'], 'npoint': d['npointings']}

    nf_recon = d['nf_recon']

    rms_I, rms_Q, rms_U = dict(), dict(), dict()

    for irec in nf_recon:
        residuals = get_residuals(name, rep_simu, residuals_way, irec)
        files, maps_recon_patch, maps_conv_patch, maps_diff_patch = \
            rmc.get_patch_many_files(rep_simu + name, '*nfrecon{}*False*'.format(irec), verbose=False)

        npix_patch = maps_diff_patch.shape[2]
        setpar.update({'pixpatch': npix_patch})
        #         print(setpar)

        nreals = np.shape(residuals)[0]

        # This if is for the number of zones (1 or more)
        if zones == 1:
            rms_i, rms_q, rms_u = np.empty((irec,)), np.empty((irec,)), np.empty((irec,))

            for i in range(irec):
                # STD over pixels and realisations
                rms_i[i] = np.std(residuals[:, i, :, 0])
                rms_q[i] = np.std(residuals[:, i, :, 1])
                rms_u[i] = np.std(residuals[:, i, :, 2])
        else:
            angle = False
            if zones == 2:
                angle = True

            center = qubic.equ2gal(d['RA_center'], d['DEC_center'])
            seenmap = rmc.get_seenmap(files[0])
            nside = d['nside']

            residuals_zones = np.empty((nreals, zones, irec, npix_patch, 3))
            for real in range(nreals):
                pix_zones, residuals_zones[real] = rmc.make_zones(residuals[real], zones, nside, center, seenmap,
                                                                  angle=angle, dtheta=d['dtheta'], verbose=False,
                                                                  doplot=False)

            rms_i, rms_q, rms_u = np.empty((zones, irec,)), np.empty((zones, irec,)), np.empty((zones, irec,))
            for izone in range(zones):
                for i in range(irec):
                    rms_i[izone, i] = np.std(residuals_zones[:, izone, i, :, 0])
                    rms_q[izone, i] = np.std(residuals_zones[:, izone, i, :, 1])
                    rms_u[izone, i] = np.std(residuals_zones[:, izone, i, :, 2])

        rms_I.update({str(irec): rms_i})
        rms_Q.update({str(irec): rms_q})
        rms_U.update({str(irec): rms_u})

    return rms_I, rms_Q, rms_U, setpar


def get_covcorr1pix(maps, ipix, verbose=False, stokesjoint=False):
    """

    This function return the covariance matrix for one pixel given a list of maps.
    Each of the Nreal maps has the shape (nfrec, npix, 3) (see save_simu_fits() ).

    Parameters
    -------
    maps: array
        Input maps with shape (nrealizations, nfrecons, npix, 3)
    ipix: int
        pixel where the covariance will be computed
    verbose : bool
        If True, print information. False by default.
    stokesjoint: bool 
        If True return Stokes parameter together 
        I0,I1,..., Q0,Q1,..., U0, U1, ... . Otherwise will return
        I0,Q0,U0,  I1,Q1,U1, ... 
        Default: False

    Return
    -------
    cov1pix: np.array
        covariance matrix for a given pixel (ipix) with shape 3*nfrec x 3*nfrec

    corr1pix: np.array
        correlation matrix for a given pixel (ipix) with shape 3*nfrec x 3*nfrec

    """

    if type(ipix) != int:
        raise TypeError('ipix has to be an integer number')

    nfrec = maps.shape[1]  # Sub-bands
    nreal = maps.shape[0]  # Sample realizations

    if verbose:
        print('The shape of the input map has to be: (nsample, nfrecons, npix, 3): {}'.format(maps.shape))
        print('Number of reconstructed sub-bands to analyze: {}'.format(nfrec))
        print('Number of realizations: {}'.format(nreal))
        print('Computing covariance matrix in pixel {}'.format(ipix))

    data = np.reshape(maps[:, :, ipix, :], (nreal, nfrec * 3))

    if stokesjoint:
        if nfrec == 1:
            pass
        elif nfrec > 1:
            permutation = []
            for istk in range(3):
                for isub in range(nfrec):
                    permutation.append(3 * isub + istk)
            data = np.reshape(maps[:, :, ipix, :], (nreal, nfrec * 3))
            data = data[:, permutation]

    cov1pix = np.cov(data, rowvar=False)
    corr1pix = np.corrcoef(data, rowvar=False)

    return cov1pix, corr1pix


def get_covcorr_patch(patch, stokesjoint=False, doplot=False):
    """
    This function computes the covariance matrix and the correlation matrix for a given patch in the sky.
    It uses get_covcorr1pix() to compute the covariance and correlation matrix for each pixel (ipix)
    and then computes a histogram for each term (I_0I_0,I_0Q_0,I_0U_0, etc) (patch).

    Asumptions: patch.shape = (nsamples, nrecons, npix_patch, 3) --> to be able to use get_covcorr1pix

    Parameters:
    -----------
    patch: np.array
        Sky patch observed (see get_patch_many_files() from ReadMC module)

    stokesjoint: bool 
        If True return Stokes parameter together 
        I0,I1,..., Q0,Q1,..., U0, U1, ... . Otherwise will return
        I0,Q0,U0,  I1,Q1,U1, ... 
        Default: False

    doplot: If True return a imshow plot of the matrix

    Returns:
    -----------
    covterm: np.array
        Covariance matrix for each pixel in a given patch in the sky.
        Shape = (3xnfreq, 3xnfreq, npix).

    corrterm: np.array
        Correlation matrix for each pixel in a given patch in the sky.
        Shape = (3xnfreq, 3xnfreq, npix)

    plot: Mean over the pixels of the covariance and correlation matrices
    """

    nrecons = patch.shape[1]
    npix = patch.shape[2]
    nstokes = patch.shape[3]
    dim = nrecons * nstokes

    cov = np.zeros((dim, dim, npix))
    corr = np.zeros((dim, dim, npix))

    for ipix in range(npix):
        cov1pix, corr1pix = get_covcorr1pix(patch, ipix, stokesjoint=stokesjoint)
        cov[:, :, ipix] = cov1pix
        corr[:, :, ipix] = corr1pix

    if doplot:
        plt.figure('Mean over pixels')
        plt.subplot(121)
        plt.imshow(np.mean(cov, axis=2), interpolation=None)
        plt.title('Mean cov over pixels')
        plt.colorbar()

        plt.subplot(122)
        plt.imshow(np.mean(corr, axis=2))
        plt.title('Mean corr over pixels')
        plt.colorbar()

    return cov, corr


def plot_hist(mat_npix, bins, title_prefix, ymax=0.5, color='b'):
    """
    Plots the histograms of each element of the matrix.
    Each histogram represents the distribution of a given
    term over the pixels..

    Parameters
    ----------
    mat_npix : array of shape (3 x nfreq, 3 x nfreq, npix)
        Cov or corr matrix for each pixel.
    bins : int
        Numbers of bins for the histogram
    title_prefix : str
        Prefix for the title of the plot.
    ymax : float
        Limit max on the y axis (between 0 and 1)
    color : str
        Color of the histogram

    """
    stokes = ['I', 'Q', 'U']
    dim = np.shape(mat_npix)[0]
    min = np.min(mat_npix)
    max = np.max(mat_npix)

    ttl = title_prefix + ' Hist over pix for each matrix element'
    plt.figure(ttl, figsize=(10, 10))
    for iterm in range(dim):
        for jterm in range(dim):
            idx = dim * iterm + jterm + 1

            mean = np.mean(mat_npix[iterm, jterm, :])
            std = np.std(mat_npix[iterm, jterm, :])

            plt.subplot(dim, dim, idx)
            plt.hist(mat_npix[iterm, jterm, :], color=color, density=True,
                     bins=bins, label='m={0:.2f} \n $\sigma$={1:.2f}'.format(mean, std))
            # no yticks for histogram in middle
            if idx % dim != 1:
                plt.yticks([])
            # no xticks for histogram in middle
            if idx < dim * (dim - 1):
                plt.xticks([])

            # Names
            if iterm == (dim - 1):
                plt.xlabel(stokes[jterm % 3] + '{}'.format(jterm / 3))
            if jterm == 0:
                plt.ylabel(stokes[iterm % 3] + '{}'.format(iterm / 3))

            # same scale for each plot
            plt.xlim((min, max))
            plt.ylim((0., ymax))

            plt.legend(fontsize='xx-small')
            plt.subplots_adjust(hspace=0., wspace=0.)


def get_covcorr_between_pix(maps, verbose=False):
    """
    Compute the pixel covariance matrix and correlation matrix
    minus the identity over many realisations. You will obtain nsub x 3
    matrices of shape (npix x npix).

    Parameters
    ----------
    maps: array
        Input maps with shape (nreal, nsub, npix, 3)
    verbose : bool
        If True, print information. False by default.

    Returns
    -------
    cov_pix : array of shape (nsub, nstokes, npix, npix)
        The covariance matrices for each subband and I, Q, U.
    corr_pix : array of shape (nsub, nstokes, npix, npix)
        The correlation matrices minus the identity (0. on the diagonal).

    """

    nreal, nsub, npix, nstokes = np.shape(maps)

    if verbose:
        print('The shape of the input map has to be: (nreal, nsub, npix, 3)')
        print('Number of reconstructed sub-bands to analyze: {}'.format(nsub))
        print('Number of realizations: {}'.format(nreal))
        print('Number of pixels {}'.format(npix))

    cov_pix = np.empty((nsub, nstokes, npix, npix))
    corr_pix = np.empty((nsub, nstokes, npix, npix))

    for sub in range(nsub):
        for s in range(nstokes):
            cov_pix[sub, s, :, :] = np.cov(maps[:, sub, :, s], rowvar=False)
            corr_pix[sub, s, :, :] = np.corrcoef(maps[:, sub, :, s], rowvar=False) - np.identity(npix)

    return cov_pix, corr_pix


def distance_square(matrix):
    """
    Return a distance associated to a matrix (n*n).
    Sum of the square elements normalized by n**2.

    """
    n = np.shape(matrix)[0]
    d = np.sum(np.square(matrix))
    return d / n ** 2


def distance_sup(matrix):
    """
    Return a distance associated to a matrix (n*n).
    Normalized by n.

    """
    n = np.shape(matrix)[0]
    d = np.max(np.sum(np.square(matrix), axis=1))
    return d / n


def cov2corr(mat):
    """
    Converts a Covariance Matrix in a Correlation Matrix
    """
    newmat = np.empty_like(mat)
    ll, cc = np.shape(mat)
    for i in range(ll):
        for j in range(cc):
            newmat[i, j] = mat[i, j] / np.sqrt(mat[i, i] * mat[j, j])
    return newmat


def covariance_IQU_subbands(allmaps, stokesjoint=False):
    """
    Returns the mean maps, averaged over pixels and realisations and the
    covariance matrices of the maps.

    Parameters
    ----------
    allmaps : list of arrays of shape (nreals, nsub, npix, 3)
        list of maps for each number of subband
    
    stokesjoint: if True return Stokes parameter together 
        I0,I1,..., Q0,Q1,..., U0,U1, ... . Otherwise will return
        I0,Q0,U0,  I1,Q1,U1, ... 
        Default: False

    Returns
    -------
    allmean : list of arrays of shape 3*nsub
        mean for I, Q, U for each subband
    allcov : list of arrays of shape (3*nsub, 3*nsub)
        covariance matrices between stokes parameters and sub frequency bands

    """
    allmean, allcov = [], []
    for isub in range(len(allmaps)):

        sh = allmaps[isub].shape
        nsub = sh[1]  # Number of subbands
        mean = np.zeros(3 * nsub)
        cov = np.zeros((3 * nsub, 3 * nsub))

        for iqu in range(3):
            for band in range(nsub):
                if stokesjoint:
                    i = 3 * iqu + band
                else:
                    i = 3 * band + iqu
                map_i = allmaps[:, band, :, iqu]
                mean[i] = np.mean(map_i)
                for iqu2 in range(3):
                    for band2 in range(nsub):
                        if stokesjoint:
                            j = 3 * iqu2 + band2
                        else:
                            j = 3 * band2 + iqu2
                        map_j = allmaps[:, band2, :, iqu2]
                        cov[i, j] = np.mean((map_i - np.mean(map_i)) * (map_j - np.mean(map_j)))

        allmean.append(mean)
        allcov.append(cov)

    return allmean, allcov


def get_weighted_correlation_average(x, cov):
    """
    Compute a weighted average taking into account the correlations between the variables.
    The mean obtained is the one that has the minimal variance possible.

    Parameters
    ----------
    x : 1D array
        Values you want to average.
    cov : 2D array
        Covariance matrix associated to the values in x.

    Returns
    -------
    The weighted mean and the variance on that mean.

    """
    inv_cov = np.linalg.inv(cov)
    sig2 = 1. / np.sum(inv_cov)
    weighted_mean = sig2 * np.sum(np.dot(inv_cov, x))
    return weighted_mean, sig2


def get_Cp(patch, nfrecon, verbose=True, doplot=True):
    """
    Returns covariance matrices between subbands for each Stokes parameter
    and each pixel.

    Parameters
    ----------
    patch: array of shape (#reals, #bands, #pixels, 3)
    nfrecon: list
        Numbers of reconstructed subbands.
    verbose: Bool
        If True makes a lot of prints.
    doplot: Bool
        If True, plot the mean cov and corr matrices over pixels.

    Returns
    -------
    Cp: array of shape (#bands, #bands, 3, #pixels)
        The covariance matrices.

    """
    irec = np.shape(patch)[1]
    npix_patch = np.shape(patch)[2]
    # if irec == 1:
    #     raise ValueError('If you already have 1 band, you do not need Cp which is computed to average subbands')

    if irec not in nfrecon:
        raise ValueError('Invalid number of freq. {0} not in {1}'.format(irec, nfrecon))

    # Prepare to save
    if verbose:
        print('==== Computing Cp matrix ====')
        print('irec = ', irec)
        print('nfrecon = ', nfrecon)
        print('patch.shape = ', patch.shape)
        print('npix_patch = ', npix_patch)

    # The full one
    covariance, _ = get_covcorr_patch(patch, stokesjoint=True, doplot=doplot)

    if verbose:
        print('covariance.shape =', covariance.shape)

    # Cut the covariance matrix for each Stokes parameter
    Cp = np.empty((irec, irec, 3, npix_patch))
    for istokes in range(3):
        a = istokes * irec
        b = (istokes + 1) * irec
        Cp[:, :, istokes, :] = covariance[a:b, a:b, :]
    if verbose:
        print('Cp.shape = ', Cp.shape)

        # Look at the value in Cp and the determinant
        for ipix in range(10):
            for istokes in range(3):
                det = np.linalg.det(Cp[:, :, istokes, ipix])
                print('det = ', det)

        print('==== Done. Cp matrix computed ====')

    return Cp


def make_weighted_av(patch, Cp, verbose=False):
    """
    Average the maps over subbands using the covariance matrix between subbands.
    Parameters
    ----------
    patch: array of shape (#reals, #bands, #pixels, 3)
    Cp: array of shape (#bands, #bands, 3, #pixels)
        The covariance matrices.
    verbose: bool

    Returns
    -------
    weighted_av: map averaged over bands
    sig2: variances over realisations on the map

    """
    nreals = np.shape(patch)[0]
    npix_patch = np.shape(Cp)[-1]
    if verbose:
        print('Cp.shape = ', np.shape(Cp))
        print('# realizations = ', nreals)
        print('npix_patch = ', npix_patch)

    weighted_av = np.zeros((nreals, npix_patch, 3))
    sig2 = np.zeros((npix_patch, 3))

    nsing = 0
    for ireal in range(nreals):
        for ipix in range(npix_patch):
            for istokes in range(3):
                x = patch[ireal, :, ipix, istokes]
                # Only do it if the matrix is not singular:
                if np.linalg.det(Cp[:, :, istokes, ipix]) != 0.:
                    weighted_av[ireal, ipix, istokes], sig2[ipix, istokes] = \
                        get_weighted_correlation_average(x, Cp[:, :, istokes, ipix])
                else:
                    nsing += 1
    if verbose:
        print('# singular matrices: ', nsing)
        print('Weigthed mean matrix per pixel, shape: ', weighted_av.shape)
        print('Variance in MC simulation, shape: ', sig2.shape)

    return weighted_av, sig2


def average_pix_sig2(sig2, ang, ang_threshold):
    """
    Average the variances over pixels in a given angle.
    """
    sig2mean = np.empty((3,))
    npix = np.shape(ang[ang < ang_threshold])
    print('npix =', npix)
    for istokes in range(3):
        sig2mean[istokes] = np.mean(sig2[:, istokes][ang < ang_threshold])
    return sig2mean


def Cp2Cp_prime(Cp, verbose=True):
    """
    Average the covariance matrices over pixels. A normalization by the 00 element is done
    before averaging.
    Parameters
    ----------
    Cp: array of shape (#bands, #bands, 3, #pixels)
        The covariance matrices for each pixel.
    verbose: bool

    Returns
    -------
    Cp_prime: array of shape (#bands, #bands, 3, #pixels)
        The covariance matrices for each pixel.

    """
    npix_patch = np.shape(Cp)[-1]
    if verbose: print('npix_patch =', npix_patch)

    # Normalize each matrix by the first element
    Np = np.empty_like(Cp)
    for istokes in range(3):
        for ipix in range(npix_patch):
            Np[:, :, istokes, ipix] = Cp[:, :, istokes, ipix] / Cp[0, 0, istokes, ipix]

    # We can now average the matrices over the pixels
    N = np.mean(Np, axis=3)

    if verbose:
        print('N shape:', N.shape)

    # We re-multiply N by the first term
    Cp_prime = np.empty_like(Cp)
    for istokes in range(3):
        for ipix in range(npix_patch):
            Cp_prime[:, :, istokes, ipix] = Cp[0, 0, istokes, ipix] * N[:, :, istokes]

    if verbose:
        print('Cp_prime.shape =', Cp_prime.shape)

    return Cp_prime


def Cp2Cp_prime_viaCorr(Cp, verbose=True):
    """
    Average the covariance matrices over pixels. A normalization is done
    on pixels before the average.
    Parameters
    ----------
    Cp: array of shape (#bands, #bands, 3, #pixels)
        The covariance matrices for each pixel.
    verbose: bool

    Returns
    -------
    Cp_prime: array of shape (#bands, #bands, 3, #pixels)
        The covariance matrices for each pixel.

    """
    nfrec = np.shape(Cp)[0]
    npix_patch = np.shape(Cp)[-1]
    if verbose: print('npix_patch =', npix_patch)

    # Convert cov matrices to correlation matrices
    Np = np.empty_like(Cp)
    for istokes in range(3):
        for ipix in range(npix_patch):
            Np[:, :, istokes, ipix] = cov2corr(Cp[:, :, istokes, ipix])

    # We can now average the correlation matrices over the pixels
    N = np.mean(Np, axis=3)

    if verbose:
        print('N shape:', N.shape)

    # We re-multiply N to get back to covariance matrices
    Cp_prime = np.empty_like(Cp)
    for istokes in range(3):
        for ipix in range(npix_patch):
            for i in range(nfrec):
                for j in range(nfrec):
                    coeff = np.sqrt(Cp[i, i, istokes, ipix] * Cp[j, j, istokes, ipix])
                    Cp_prime[i, j, istokes, ipix] = N[i, j, istokes] * coeff

    if verbose:
        print('Cp_prime.shape =', Cp_prime.shape)

    return Cp_prime


def get_corrections(nf_sub, nf_recon, band=150, relative_bandwidth=0.25):
    """
    The reconstructed subbands have different widths.
    Here, we compute the corrections you can applied to
    the variances and covariances to take this into account.

    Parameters
    ----------
    nf_sub : int
        Number of input subbands
    nf_recon : int
        Number of reconstructed subbands
    band : int
        QUBIC frequency band, in GHz. 150 by default
        Typical values: 150, 220.
    relative_bandwidth : float
        Ratio of the difference between the edges of the
        frequency band over the average frequency of the band
        Typical value: 0.25

    Returns
    ----------
    corrections : list
        Correction coefficients for each subband.
    correction_mat : array of shape (3xnf_recon, 3xnf_recon)
        Matrix containing the corrections.
        It can be multiplied term by term to a covariance matrix.

    """
    nb = nf_sub // nf_recon  # Number of input subbands in each reconstructed subband

    _, nus_edge, nus, deltas, Delta, _ = qubic.compute_freq(band, nf_sub, relative_bandwidth)

    corrections = []
    for isub in range(nf_recon):
        # Compute wide of the sub-band
        sum_delta_i = deltas[isub * nb: isub * nb + nb].sum()
        corrections.append(Delta / sum_delta_i)

    correction_mat = np.empty((3 * nf_recon, 3 * nf_recon))
    for i in range(3 * nf_recon):
        for j in range(3 * nf_recon):
            freq_i = i // nf_recon
            freq_j = j // nf_recon
            sum_delta_i = deltas[freq_i * nb: freq_i * nb + nb].sum()
            sum_delta_j = deltas[freq_j * nb: freq_j * nb + nb].sum()
            correction_mat[i, j] = Delta / np.sqrt(sum_delta_i * sum_delta_j)

    return corrections, correction_mat


# =================== Old functions ================
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

    for isub in range(len(nsubvals)):
        print('for nsub = {}'.format(nsubvals[isub]))
        mapsout = allmapsout[isub]
        covmat_freqfreq = np.zeros((nsubvals[isub], nsubvals[isub], len(seen), 3))
        # Loop over pixels
        for p in range(len(seen)):
            # Loop over I Q U
            for i in range(3):
                mat = np.cov(mapsout[:, :, p, i].T)
                # Normalisation
                if np.size(mat) == 1:
                    variance_map[isub, i, p] = mat
                else:
                    variance_map[isub, i, p] = 1. / np.sum(np.linalg.inv(mat))
                covmat_freqfreq[:, :, p, i] = mat / np.mean(mat)
                # its normalization is irrelevant for the later average
        # Average and std over pixels
        meanmat = np.zeros((nsubvals[isub], nsubvals[isub], 3))
        stdmat = np.zeros((nsubvals[isub], nsubvals[isub], 3))
        for i in range(3):
            meanmat[:, :, i] = np.mean(covmat_freqfreq[:, :, :, i], axis=2)
            stdmat[:, :, i] = np.std(covmat_freqfreq[:, :, :, i], axis=2)

        allmeanmat.append(meanmat)
        allstdmat.append(stdmat)
    return np.sqrt(variance_map), allmeanmat, allstdmat


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

    for isub in range(len(nsubvals)):
        print('For nsub = {}'.format(nsubvals[isub]))
        mapsout = allmapsout[isub]
        sh = mapsout.shape
        nreals = sh[0]
        for iqu in range(3):
            # cov matrice freq-freq averaged over pixels
            covmat = allmeanmat[isub][:, :, iqu]
            invcovmat = np.linalg.inv(covmat)
            # Loop over pixels
            for p in range(npixok):
                mean_cov = np.zeros(nreals)

                # Loop over realisations
                for real in range(nreals):
                    vals = mapsout[real, :, p, iqu]
                    mean_cov[real] = get_mean_cov(vals, invcovmat)
                # Mean and rms over realisations
                meanmap_cov[isub, iqu, p] = np.mean(mean_cov)
                rmsmap_cov[isub, iqu, p] = np.std(mean_cov)

    return meanmap_cov, rmsmap_cov


def get_mean_cov(vals, invcov):
    """
    This function does the same as: get_weighted_correlation_average
    """
    AtNid = np.sum(np.dot(invcov, vals))
    AtNiA_inv = 1. / np.sum(invcov)
    mean_cov = AtNid * AtNiA_inv
    return mean_cov