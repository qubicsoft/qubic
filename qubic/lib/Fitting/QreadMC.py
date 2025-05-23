from __future__ import division, print_function
import glob

import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


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

    if save_dir[-1] != '/':
        save_dir = save_dir + '/'

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
def get_seenmap(file, badval=-1.6375e+30, rtol=1e-05, atol=1e-08):
    """
    Returns an array with the pixels seen or not.
    Parameters
    ----------
    file : str
        A fits file saved from a simulation.
    badval : float
        The value of the pixel considered as UNSEEN
    rtol : float
        Relative tolerance for badval
    atol : float
        Absolute tolerance for badval

    Returns
    -------
    seenmap : array
        Array of booleans of shape #pixels,
        True inside the patch and False outside.
    """
    simu = fits.open(file)
    map = simu['MAPS_RECON'].data
    map_mean = np.mean(map, axis=(0, 2))
    seenmap = hp.pixelfunc.mask_good(map_mean,
                                     badval=badval,
                                     rtol=rtol,
                                     atol=atol)
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


def get_patch_many_files(rep_simu, name, badval=-1.6375e+30, rtol=1e-05, atol=1e-08, verbose=True):
    """
    Get all the patches you want to analyze from many fits files.
    Parameters
    ----------
    rep_simu : str
        Repository where the fits files are.
    name : str
        Name of the files you are interested in.
    badval : float
        The value of the pixel considered as UNSEEN
    rtol : float
        Relative tolerance for badval
    atol : float
        Absolute tolerance for badval
    verbose: bool
    Returns
    -------
    A list with the names of all the files you took.
    Three arrays containing the reconstructed patches, the convolved patches
    and the difference between both.

    """
    all_fits = glob.glob(rep_simu + name)
    nfiles = len(all_fits)
    if verbose:
        print('{} files have been found.'.format(nfiles))

    seenmap = get_seenmap(all_fits[0], badval=badval, rtol=rtol, atol=atol)

    all_patch_recon = []
    all_patch_convo = []
    all_patch_diff = []

    for i, fits in enumerate(all_fits):
        patch_recon, patch_convo, patch_diff = get_patch(fits, seenmap)
        if i == 0:
            right_shape = patch_recon.shape
        else:
            if patch_recon.shape != right_shape:
                raise ValueError('You should take maps with identical shapes.')
        all_patch_recon.append(patch_recon)
        all_patch_convo.append(patch_convo)
        all_patch_diff.append(patch_diff)

    return all_fits, np.asarray(all_patch_recon), \
           np.asarray(all_patch_convo), np.asarray(all_patch_diff)


def get_maps_many_files(rep_simu, name, verbose=True):
    """
    Get all the maps you want to analyze from many fits files.
    Parameters
    ----------
    rep_simu : str
        Repository where the fits files are.
    name : str
        Name of the files you are interested in.
    verbose : bool
    Returns
    -------
    A list with the names of all the files you took.
    Three arrays containing the reconstructed maps, the convolved maps
    and the difference between both.

    """
    all_fits = glob.glob(rep_simu + name)
    nfiles = len(all_fits)
    if verbose:
        print('{} files have been found.'.format(nfiles))

    all_maps_recon = []
    all_maps_convo = []
    all_maps_diff = []

    for i, fits in enumerate(all_fits):
        map_recon, map_convo, map_diff = get_maps(fits)
        if i == 0:
            right_shape = map_recon.shape
        else:
            if map_recon.shape != right_shape:
                raise ValueError('You should take maps with identical shapes.')
        all_maps_recon.append(map_recon)
        all_maps_convo.append(map_convo)
        all_maps_diff.append(map_diff)

    return all_fits, np.asarray(all_maps_recon), np.asarray(all_maps_convo), np.asarray(all_maps_diff)


# ================== Cut a patch in different zones ====================
def pix2ang(ns, center, seenmap=None):
    """
    For each pixel observed, returns the angle between the pixel vector and
    the central pixel vector.
    """
    # central pixel vector
    v0 = hp.ang2vec(center[0], center[1], lonlat=True)
    if seenmap is not None:
        # Only pixel seen (in the patch)
        ip = np.arange(12 * ns ** 2)[seenmap]
    else:
        # all pixels
        ip = np.arange(12 * ns ** 2)

    # vectors associated to each pixel seen
    vpix = hp.pix2vec(ns, ip)

    return np.degrees(np.arccos(np.dot(v0, vpix)))


def make_zones(patch, nzones, nside, center, seenmap, angle=False, dtheta=15., verbose=True, doplot=True):
    """
    Mask a path to get different concentric zones.

    Parameters
    ----------
    patch : array
        Patch you want to cut of shape (#subbands, #pix_seen, 3)
    nzones : int
        Number of zones you want to make.
    nside : int
    center : array
        Coordinates of the center of the patch in degree (lon, lat)
    seenmap : array
        Array of booleans of shape #pixels,
        True inside the patch and False outside.
    angle : bool
        If True, you make only 2 zones, inside and outside dtheta.
    dtheta : float
        Size of the patch were the pointing was done (degree),
        by default it is 15°.
    verbose: bool
    doplot : bool
        If True, makes a plot with the different zones obtained.

    Returns
    -------
    A list with the number of pixels in each zone.
    A list with the patch masked to get each zone.

    """
    npixok = patch.shape[1]

    # Angle associated to each pixel in the patch
    ang = pix2ang(nside, center, seenmap)

    # Angles at the border of each zone
    if not angle:
        angles_zone = np.linspace(0, np.max(ang), nzones + 1)[1:]
    elif angle:
        angles_zone = np.array([dtheta, np.max(ang)])

    angles_zone = np.insert(angles_zone, 0, 0.)

    # Make a list with the masks
    allmask = [np.zeros_like(patch) for _ in range(nzones)]
    for pix in range(npixok):
        for a, angle in enumerate(angles_zone[1:]):
            if ang[pix] <= angle:
                allmask[a][:, pix, :] = 1.
                break
    # for a, angle in enumerate(angles_zone[1:]):
    #    #print(a,angle)
    #    if a == 0:
    #        mask0 = np.where(ang < angle)
    #        maski = mask0[0]
    #    else:
    #        mask0 = np.where(ang < angles_zone[a-1])
    #        if mask0[0].size == 0:
    #            mask0 = np.array([True])
    #        mask1 = np.where(ang > angles_zone[a])
    #        maski = mask0*mask1
    #    allmask[a][:, maski, :] = 1.

    # Apply the masks on the patch
    allmaps_mask = allmask * patch
    # print('allmaps_maks',np.shape(allmaps_mask))
    # Compute the numbers of pixels in each zone
    pix_per_zone = [np.count_nonzero(m[0, :, 0]) for m in allmask]
    if verbose:
        print('Number of pixels in each zones : {}. Angles limit for each zone: {}'.format(pix_per_zone, angles_zone))

    # Plot the patch masked
    if doplot:
        istokes = 0
        plt.figure('Zones')
        for i in range(nzones):
            map = np.zeros((patch.shape[0], 12 * nside ** 2, 3))
            map[:, seenmap, :] = allmaps_mask[i]
            map[:, ~seenmap, :] = hp.UNSEEN
            hp.gnomview(map[0, :, istokes], sub=(1, nzones, i + 1),
                        rot=center, reso=15,
                        title='Zone {}, npix = {}, istokes = {}'.format(i, pix_per_zone[i], istokes))
    return pix_per_zone, allmaps_mask
