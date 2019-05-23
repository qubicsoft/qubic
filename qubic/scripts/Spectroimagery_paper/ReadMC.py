import glob

import healpy as hp
import numpy as np
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
def get_seenmap(file):
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

    seenmap = get_seenmap(all_fits[0])

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

    return all_fits, np.asarray(all_maps_recon), \
           np.asarray(all_maps_convo), np.asarray(all_maps_diff)


def pix2ang(ns, center, seenmap):
    """
    Return the angles between the vector of the central pixel
    and the vector of each pixel seen.
    """
    # central pixel vector
    v0 = hp.ang2vec(center[0], center[1], lonlat=True)
    # seen pixel indices
    ip = np.arange(12 * ns ** 2)[seenmap]
    # vectors associated to each pixel seen
    vpix = hp.pix2vec(ns, ip)

    return np.degrees(np.arccos(np.dot(v0, vpix)))

