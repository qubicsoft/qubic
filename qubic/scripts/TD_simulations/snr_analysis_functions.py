import pickle
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np

def load_maps(file, stokes_parameter=0, plot=False):
    """
    Load input, reconstructed, and residual maps from a pickle file. The raw data dictionary is deleted to avoid memory saturation.

    Parameters
    ----------
    file : str
        Path to the pickle file containing the simulation output.
    stokes_parameter : int, optional
        Index of the Stokes component to extract (0 = I, 1 = Q, 2 = U).
        Default is 0.
    plot : bool, optional
        If True, display coverage, convergence, and simulated maps. Default is False.

    Returns
    -------
    maps_input : ndarray, shape (Nrec, Npix)
        Convolved input maps for the selected Stokes parameter.
    maps_rec : ndarray, shape (Nrec, Npix)
        Reconstructed maps for the selected Stokes parameter.
    maps_res : ndarray, shape (Nrec, Npix)
        Residual maps for the selected Stokes parameter.
    center : tuple
        Sky coordinates (lon, lat) of the observed patch centre.
    seenpix : ndarray of bool, shape (Npix,)
        Boolean mask of the seenpixels in the observed region.
    coverage : ndarray, shape (Npix,)
        Coverage map
    """

    # Load the full data dictionary from disk
    data = pickle.load(open(file, "rb"))

    # Frequency array and number of reconstructed sub-bands
    nus = data["nus"]
    Nrec = nus.shape[0] - 7

    # Maps
    maps_input = data["maps_in_convolved"]      
    maps_rec   = data["maps"][:Nrec]          
    maps_res   = maps_input - maps_rec          

    center  = data["center"]   
    coverage = data["coverage"] 
    seenpix  = data["seenpix"] 

    maps_input[:, ~seenpix, :] = hp.UNSEEN
    maps_rec[:, ~seenpix, :]   = hp.UNSEEN
    maps_res[:, ~seenpix, :]   = hp.UNSEEN

    istk = stokes_parameter


    # Plots
    if plot == True:

        # 1) Coverage map + PCG convergence curve
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        hp.mollview(coverage, title='Coverage', cmap='Spectral_r',
                    fig=fig.number, sub=(1, 2, 1))
        plt.subplot(1, 2, 1).axis('off')

        axes[1].plot(data["convergence"])
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Convergence")
        axes[1].set_title("PCG Convergence")
        axes[1].set_yscale("log")
        axes[1].set_ylim(1e-8, 1e0)
        axes[1].grid()

        plt.subplots_adjust(wspace=0.4)
        plt.show()

        # 2) Gnomonic projections: input / reconstructed / residual per sub-band
        STOKES = ["I", "Q", "U"]
        n = 3
        reso = 15  # arcmin per pixel in the gnomonic projection
        sigma_res = np.std(maps_rec[:, seenpix, istk] - maps_input[:, seenpix, istk], axis=1)

        k = 0
        plt.figure(figsize=(20, 10))
        for inu in range(Nrec):
            hp.gnomview(maps_input[inu, :, istk], rot=center,
                        reso=reso, notext=True,
                        title=f"{nus[inu].round(2)} GHz - {STOKES[istk]} input",
                        cmap="jet",
                        unit=r"$\mu K_{CMB}$",
                        sub=(Nrec, 3, k + 1))

            hp.gnomview(maps_rec[inu, :, istk], rot=center,
                        reso=reso, notext=True,
                        title=f"{nus[inu].round(2)} GHz - {STOKES[istk]} Output",
                        cmap="jet",
                        unit=r"$\mu K_{CMB}$",
                        sub=(Nrec, 3, k + 2))

            hp.gnomview(maps_res[inu, :, istk], rot=center,
                        reso=reso, notext=True,
                        title=f"{nus[inu].round(2)} GHz - {STOKES[istk]} Residual",
                        cmap="jet",
                        min = - n * sigma_res[inu],
                        max = n * sigma_res[inu],
                        unit=r"$\mu K_{CMB}$",
                        sub=(Nrec, 3, k + 3))
            k += 3

    # Free the raw data dictionary from memory
    del data

    return maps_input[:, :, istk], maps_rec[:, :, istk], maps_res[:, :, istk], center, seenpix, coverage
    
    
def maps_stacking(datafiles, stokes_parameter=0):
    """
    Stack multiple simulation and compute cumulative averages.


    Parameters
    ----------
    datafiles : list of str
        Paths to the pickle files produced by the map-making pipeline.
    stokes_parameter : int, optional
        Index of the Stokes component to extract (0 = I, 1 = Q, 2 = U).
        Default is 0.
   
    Returns
    -------
    result : dict
        Dictionary with the following entries:

        Per-realisation arrays (axis 0 = realisation index):
            maps_in             – convolved input maps, shape (N, Nrec, Npix)
            maps_in_nobeam      – input maps without beam convolution, shape (N, Nrec, Npix)
            maps_rec            – reconstructed maps, shape (N, Nrec, Npix)
            maps_res            – residual maps, shape (N, Nrec, Npix) 
            coverage            – coverage maps, shape (N, Npix)
            seenpix             – observed-pixel masks, shape (N, Npix)

        Cumulative-average arrays (axis 0 = number of stacked realisations):
            maps_in_cum         – cumulative mean of convolved input maps
            maps_in_nobeam_cum  – cumulative mean of input maps without beam convolution
            maps_cum            – cumulative mean of reconstructed maps
            maps_res_cum        – cumulative mean of residual maps
            coverage_cum        – summed coverage
            seenpix_cum         – observed-pixel mask obtained form cumulative coverage

      
            nus                 – frequency array
            center              – sky patch center coordinates
    """

    nfiles = len(datafiles)

    # Load the first file to infer array shapes and shared metadata
    with open(datafiles[0], "rb") as f:
        data = pickle.load(f)

    nus  = data["nus"]
    Nrec = nus.shape[0] - 7          # number of reconstruction sub-bands
    center = data["center"]


    # Preallocate output arrays 
    istk = stokes_parameter
    
    maps_input_arr        = np.empty((nfiles, *data["maps_in_convolved"][:, :, istk].shape), dtype=np.float32)
    maps_input_nobeam_arr = np.empty((nfiles, *data["maps_in"][:, :, istk].shape),          dtype=np.float32)
    maps_rec_arr          = np.empty((nfiles, *data["maps"][:Nrec, :, istk].shape),         dtype=np.float32)
    coverage_arr          = np.empty((nfiles, *data["coverage"].shape),                   dtype=np.float32)
    seenpix_arr           = np.empty((nfiles, *data["seenpix"].shape),                    dtype=bool)

    # Store the already-loaded first realisation
    maps_input_arr[0]        = data["maps_in_convolved"][:, :, istk]
    maps_input_nobeam_arr[0] = data["maps_in"][:, :, istk]
    maps_rec_arr[0]          = data["maps"][:Nrec, :, istk]
    coverage_arr[0]          = data["coverage"]
    seenpix_arr[0]           = data["seenpix"]


    # Load the remaining files one by one
    for i, fname in enumerate(datafiles[1:], start=1):
        with open(fname, "rb") as f:
            data = pickle.load(f)
        maps_input_arr[i]        = data["maps_in_convolved"][:, :, istk]
        maps_input_nobeam_arr[i] = data["maps_in"][:, :, istk]
        maps_rec_arr[i]          = data["maps"][:Nrec, :, istk]
        coverage_arr[i]          = data["coverage"]
        seenpix_arr[i]           = data["seenpix"]


    # Cumulative averages. Element k of each array is the mean of the first k+1 realisations.
    divisor = np.arange(1, nfiles + 1)[:, None, None]   	

    maps_input_cum        = np.cumsum(maps_input_arr, axis=0)        / divisor
    maps_input_nobeam_cum = np.cumsum(maps_input_nobeam_arr, axis=0) / divisor
    maps_rec_cum          = np.cumsum(maps_rec_arr, axis=0)          / divisor
    maps_res_cum          = maps_input_cum - maps_rec_cum


    # Cumulative coverage and derived seen-pixel mask.
    coverage_cum = np.cumsum(coverage_arr, axis=0)
    seenpix_cum  = coverage_cum / np.max(coverage_cum, axis=1)[:, None] > 0.1

    # Free the last loaded dictionary to reclaim memory
    del data

    return {
        "maps_in":             maps_input_arr,
        "maps_in_nobeam":      maps_input_nobeam_arr,
        "maps_rec":            maps_rec_arr,
        "maps_res":            maps_input_arr - maps_rec_arr,
        "maps_in_cum":         maps_input_cum,
        "maps_in_nobeam_cum":  maps_input_nobeam_cum,
        "maps_cum":            maps_rec_cum,
        "maps_res_cum":        maps_res_cum,
        "nus":                 nus,
        "coverage":            coverage_arr,
        "coverage_cum":        coverage_cum,
        "center":              center,
        "seenpix":             seenpix_arr,
        "seenpix_cum":         seenpix_cum,
    }
    
 
def snr_from_coverage_bins(maps_input, maps_res, coverage, seenpix, center=[0, 0], nbins=20, plot=False, verbose=False):
    """
    Estimate per-pixel noise and signal-to-noise ratio using coverage-based binning.

    The observed pixels are divided into `nbins` quantile bins according to
    their coverage value.  Within each bin the RMS of the residual map is
    taken as the noise estimate (sigma), so that every pixel in the same
    coverage bin shares the same sigma.  The SNR map is then simply
    maps_input / sigma.

    Parameters
    ----------
    maps_input : ndarray, shape (Npix,)
        Input (convolved) sky map for a single Stokes parameter.
    maps_res : ndarray, shape (Npix,)
        Residual map for the same Stokes parameter.
    coverage : ndarray, shape (Npix,)
        Coverage map.
    seenpix : ndarray of bool, shape (Npix,)
        Boolean mask identifying observed pixels.
    center : list of float, optional
        Sky coordinates [lon, lat] of the patch centre (used only for
        the gnomonic plot). Default is [0, 0].
    nbins : int, optional
        Number of coverage quantile bins. Default is 20.
    plot : bool, optional
        If True, display a gnomonic projection of the SNR map.
    verbose : bool, optional
        If True, compute and print the Pearson correlation of sigma with
        coverage and with 1/sqrt(coverage).

    Returns
    -------
    snr : ndarray, shape (Npix,)
        Signal-to-noise map (maps_input / sigma).
    sigma : ndarray, shape (Npix,)
        Noise estimate per pixel, constant within each coverage bin.
    corr_cov_sigma : float
        Pearson correlation between coverage and sigma (0 if verbose is False).
    corr_inv_sqrt_cov_sigma : float
        Pearson correlation between 1/sqrt(coverage) and sigma
        (0 if verbose is False).
    """

    # Restrict to observed pixels
    hits  = coverage[seenpix]
    noise = maps_res[seenpix]

    # Build quantile-based bin edges over the coverage distribution
    edges = np.quantile(hits, np.linspace(0, 1, nbins + 1))

    hits_centers = []
    sigma_bins   = np.full(nbins, np.nan, dtype=float)

    for i in range(nbins):
        # Select pixels whose coverage falls in the current bin
        sel = (hits >= edges[i]) & (hits < edges[i + 1])

        # Skip bins with too few pixels for a reliable RMS estimate
        if np.sum(sel) < 50:
            continue

        hits_centers.append(np.mean(hits[sel]))
        sigma_bins[i] = np.sqrt(np.mean(noise[sel] ** 2))  # RMS noise in this bin

    hits_centers = np.array(hits_centers)

    # Assign each observed pixel to its coverage bin
    bin_idx = np.digitize(coverage[seenpix], edges) - 1      # map to 0 .. nbins-1
    bin_idx[bin_idx == nbins] = nbins - 1                    

    # Build the full-sky sigma array (NaN for unobserved pixels)
    sigma = np.full_like(coverage, np.nan, dtype=float)
    sigma[seenpix] = sigma_bins[bin_idx]

    # Signal-to-noise ratio map
    snr = maps_input / sigma

 
    # Optional diagnostics: correlation between noise and coverage
    corr_cov_sigma          = 0
    corr_inv_sqrt_cov_sigma = 0

    if verbose:
        if np.any(~np.isfinite(sigma[seenpix])):
            raise ValueError("The noise is NaN for some pixel.")

        corr_cov_sigma = np.corrcoef(coverage[seenpix], sigma[seenpix])[0, 1]
        corr_inv_sqrt_cov_sigma = np.corrcoef(1 / np.sqrt(coverage[seenpix]), sigma[seenpix])[0, 1]

        print("Correlation between coverage and binned noise:", corr_cov_sigma)
        print("Correlation between 1/sqrt(coverage) and binned noise:",
              corr_inv_sqrt_cov_sigma)


    # Optional gnomonic projection of the SNR map
    if plot:
        snr[~seenpix] = hp.UNSEEN
        hp.gnomview(snr, rot=center, reso=15, notext=True, title="SNR", cmap="jet", min=0)

    return snr, sigma, corr_cov_sigma, corr_inv_sqrt_cov_sigma


def snr_from_local_std(maps_input, maps_res, coverage, seenpix, center=[0, 0], radius_deg=1.0, plot=False, verbose=False):
    """
    Estimate per-pixel noise and SNR from a local neighbourhood RMS. The SNR is then maps_input / sigma.

    Parameters
    ----------
    maps_input : ndarray, shape (Npix,)
        Input (convolved) sky map for a single Stokes parameter.
    maps_res : ndarray, shape (Npix,)
        Residual map.
    coverage : ndarray, shape (Npix,)
        Coverage map.
    seenpix : ndarray of bool, shape (Npix,)
        Boolean mask identifying observed pixels.
    center : list of float, optional
        Sky coordinates [lon, lat] of the patch centre (used only for
        the gnomonic plot). Default is [0, 0].
    radius_deg : float, optional
        Angular radius of the local neighbourhood in degrees. Default is 1.0.
    plot : bool, optional
        If True, display a gnomonic projection of the SNR map.
    verbose : bool, optional
        If True, print the Pearson correlations of sigma with coverage
        and with 1/sqrt(coverage).

    Returns
    -------
    snr : ndarray, shape (Npix,)
        Signal-to-noise map (maps_input / sigma).
    sigma : ndarray, shape (Npix,)
        Local RMS noise estimate per pixel.
    corr_cov_sigma : float
        Pearson correlation between coverage and sigma.
    corr_inv_sqrt_cov_sigma : float
        Pearson correlation between 1/sqrt(coverage) and sigma.
    """

    # Convert the neighbourhood radius to radians
    radius_rad = np.radians(radius_deg)

    nside = 128

    sigma = np.full_like(maps_res, np.nan, dtype=float)

    for ipix in range(len(maps_res)):
        # Find all pixels within the disk centred on the current pixel
        neighbors = hp.query_disc(nside, hp.pix2vec(nside, ipix), radius_rad)

        # Restrict to observed pixels only
        neighbors = neighbors[seenpix[neighbors]]
        if neighbors.size == 0:
            continue

        # Local RMS of the residual inside the disk
        sigma[ipix] = np.sqrt(np.mean(maps_res[neighbors] ** 2))

    # Per-pixel signal-to-noise ratio
    snr = maps_input / sigma


    # Optional diagnostics: correlation between noise and coverage
    if np.any(~np.isfinite(sigma[seenpix])):
        raise ValueError("The noise is NaN for some pixel.")

    corr_cov_sigma = np.corrcoef(coverage[seenpix], sigma[seenpix])[0, 1]
    corr_inv_sqrt_cov_sigma = np.corrcoef(
        1 / np.sqrt(coverage[seenpix]), sigma[seenpix]
    )[0, 1]

    if verbose:
        print("Correlation between coverage and binned noise:", corr_cov_sigma)
        print("Correlation between 1/sqrt(coverage) and binned noise:",
              corr_inv_sqrt_cov_sigma)


    # Optional gnomonic projection of the SNR map
    if plot:
        snr[~seenpix] = hp.UNSEEN
        hp.gnomview(snr, rot=center,
                    reso=15, notext=True, title="SNR",
                    cmap="jet", min=0)

    return snr, sigma, corr_cov_sigma, corr_inv_sqrt_cov_sigma
