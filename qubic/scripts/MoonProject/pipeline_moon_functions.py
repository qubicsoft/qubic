import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator, LinearNDInterpolator
import matplotlib.pyplot as plt
from scipy.fft import fftfreq, fft2, ifft2
import sys
import healpy as hp
import time
from scipy.signal import butter, filtfilt, bessel, sosfiltfilt
import pandas as pd
import jax
import jax.numpy as jnp
from fast_histogram import histogram2d

import fitting as fit
import pickle
from datetime import datetime

import iminuit
from iminuit.cost import LeastSquares

#########################

### General imports
from joblib import Parallel, delayed
from multiprocessing import Manager, Lock
from scipy.signal import medfilt

### Astropy configuration
from astropy.visualization import quantity_support
quantity_support()
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_moon

#### QUBIC IMPORT
from qubicpack.qubicfp import qubicfp
import qubic.lib.Calibration.Qfiber as ft

from qubic.lib import Qdictionary
from qubic.lib.Instrument import Qacquisition

import pipeline_moon_plotting as pmp

#########################

from qubic.lib.Qbeams import BeamFitted, BeamGaussian, MultiFreqBeam
from qubic.lib.Calibration.Qcalibration import QubicCalibration
from qubic.lib.Qripples import BeamGaussianRippled, ConvolutionRippledGaussianOperator
from qubic.lib.Qutilities import _compress_mask
from qubic.lib.Instrument.Qinstrument import SyntheticBeam

#########################

conv_reso_fwhm = 2.35482

#########################
# import matplotlib.style as style
# style.use("/Users/huchet/Documents/phd_code/matplotlib_styles/ah_basic_style.mplstyle")
# plt.rc('text', usetex=False)
# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r"\usepackage{bm}")
# plt.style.use('default')

#########################

# def timer(f, *args):
#     starttime = time.time()
#     result = f(*args)
#     result = jax.block_until_ready(result) # ensure the result is done running for timing purposes
#     endtime = time.time()
#     return endtime - starttime

def timer(f, *args):
    """
    wrap a function to monitor its runtime and compile time:
    result = timer(f, inputs)
    """
    # running the function twice
    starttime = time.time()
    result1 = f(*args)
    result1 = jax.block_until_ready(result1) # ensure the result is done running for timing purposes
    midtime = time.time()
    result2 = f(*args)
    result2 = jax.block_until_ready(result2) # ensure the result is done running for timing purposes
    endtime = time.time()
    # deducing runtime and compile time
    runtime1 = midtime - starttime
    runtime2 = endtime - midtime
    compiletime = runtime1 - runtime2
    print(f"runtime: {runtime2} compiletime: {compiletime}")
    # returning the result
    return result2
    
def mean_bin_data_nd(pos, values, bins): # data and bins have shape (ndims, ...)
    # left limit of bins is excluded
    # shape of pos has to be (ndims, npoints), with npoints in the TOD
    # print(np.shape(values))
    ndims = len(pos)
    bin_df = []
    for i in range(ndims):
        # print("in loop")
        # print(np.shape(pos[i]))
        ipos_df = pd.DataFrame(pos[i])
        # print(ipos_df[0])
        # test = pd.cut(ipos_df[0], bins=bins[i])
        bin_df.append(pd.cut(ipos_df[0], bins=bins[i]))
        # print("")
        # print("cut")
        # print(np.shape(test))
    
    values_df = pd.DataFrame(values)
    values_binned = values_df.groupby([bin_df[i] for i in range(ndims)])

    return  np.array(values_binned[0].mean())


def cond(pred, true_fun, false_fun, operand):
  if pred:
    return true_fun(operand)
  else:
    return false_fun(operand)


# def mean_2d_bin_data(pos, values, bins): # pos and bins have shape (2, ...)
#     # left limit of bins is excluded
#     # shape of pos has to be (2, npoints), with npoints in the TOD
#     n_i = len(bins[0]) - 1
#     n_j = len(bins[1]) - 1
#     res = jnp.zeros((n_i, n_j))
#     # res = jnp.arange(n_i*n_j).reshape(n_i, n_j)
#     tot_n_count = 0
#     def n_count_not_zero(operand):
#         values, mask_values_ij, n_count = operand
#         # res.at[i, j].set(jnp.sum(values*mask_values_ij)/n_count)
#         # print("here we are")
#         return jnp.sum(values*mask_values_ij)/n_count
#     def n_count_zero(operand):
#         values, mask_values_ij, n_count = operand
#         return jnp.sum(values*jnp.nan)
#     def loop_j(j, val):
#         res_i, i, mask_values_i, tot_n_count = val
#         # i, mask_values_i, tot_n_count = val
#         mask_values_j = (bins[1][j] < pos[1]) & (pos[1] <= bins[1][j + 1])
#         mask_values_ij = mask_values_i * mask_values_j
#         n_count = jnp.sum(mask_values_ij)
#         operand = values, mask_values_ij, n_count
#         new_res_ij = jax.lax.cond(n_count > 0, 
#                     n_count_not_zero, 
#                     n_count_zero,
#                     operand)
#         # res.at[i, j].set(new_res_ij)
#         # tot_n_count = tot_n_count + n_count
#         tot_n_count = tot_n_count + new_res_ij
#         res_i = res_i.at[j].set(new_res_ij)
#         return res_i, i, mask_values_i, tot_n_count
#     def loop_i(i, val):
#         res, tot_n_count = val
#         res_i = res[i]
#         mask_values_i = (bins[0][i] < pos[0]) & (pos[0] <= bins[0][i + 1])
#         res_i, i, mask_values_i, tot_n_count = jax.lax.fori_loop(0, n_j, loop_j, init_val=(res_i, i, mask_values_i, tot_n_count))
#         res = res.at[i].set(res_i)
#         return res, tot_n_count
#     res, tot_n_count = jax.lax.fori_loop(0, n_i, loop_i, init_val=(res, tot_n_count))
#     return res#, tot_n_count

# def mean_2d_bin_data(pos, values, bins): # pos and bins have shape (2, ...)
#     # left limit of bins is excluded
#     # shape of pos has to be (2, npoints), with npoints in the TOD
#     n_i = len(bins[0]) - 1
#     n_j = len(bins[1]) - 1
#     res = jnp.zeros((n_i, n_j))
#     # res = jnp.arange(n_i*n_j).reshape(n_i, n_j)
#     tot_n_count = 0
#     def n_count_not_zero(operand):
#         values, mask_values_ij, n_count = operand
#         # res.at[i, j].set(jnp.sum(values*mask_values_ij)/n_count)
#         # print("here we are")
#         return jnp.sum(values*mask_values_ij)/n_count
#     def n_count_zero(operand):
#         values, mask_values_ij, n_count = operand
#         return jnp.sum(values*jnp.nan)
#     def loop_j(j, val):
#         res, i, mask_values_i, tot_n_count = val
#         # i, mask_values_i, tot_n_count = val
#         mask_values_j = (bins[1][j] < pos[1]) & (pos[1] <= bins[1][j + 1])
#         mask_values_ij = mask_values_i * mask_values_j
#         n_count = jnp.sum(mask_values_ij)
#         operand = values, mask_values_ij, n_count
#         new_res_ij = jax.lax.cond(n_count > 0, 
#                     n_count_not_zero, 
#                     n_count_zero,
#                     operand)
#         # res.at[i, j].set(new_res_ij)
#         # tot_n_count = tot_n_count + n_count
#         tot_n_count = tot_n_count + new_res_ij
#         res = res.at[i, j].set(new_res_ij)
#         return res, i, mask_values_i, tot_n_count
#     def loop_i(i, val):
#         res, tot_n_count = val
#         mask_values_i = (bins[0][i] < pos[0]) & (pos[0] <= bins[0][i + 1])
#         res, i, mask_values_i, tot_n_count = jax.lax.fori_loop(0, n_j, loop_j, init_val=(res, i, mask_values_i, tot_n_count))
#         # res = res.at[i].set(res.at[i])
#         return res, tot_n_count
#     res, tot_n_count = jax.lax.fori_loop(0, n_i, loop_i, init_val=(res, tot_n_count))
#     return res#, tot_n_count

# def mean_2d_bin_data(pos, values, bins): # pos and bins have shape (2, ...)
#     # left limit of bins is excluded
#     # shape of pos has to be (2, npoints), with npoints in the TOD
#     n_i = len(bins[0]) - 1
#     n_j = len(bins[1]) - 1
#     # res = jnp.zeros((n_i, n_j))
#     # res = jnp.arange(n_i*n_j).reshape(n_i, n_j)
#     tot_n_count = 0
#     def n_count_not_zero(operand):
#         values, mask_values_ij, n_count = operand
#         # res.at[i, j].set(jnp.sum(values*mask_values_ij)/n_count)
#         # print("here we are")
#         return jnp.sum(values*mask_values_ij)/n_count
#     def n_count_zero(operand):
#         values, mask_values_ij, n_count = operand
#         return jnp.sum(values*0)

#     def loop_j(pos, values, bin_j, in_bin_i):
#         # pos (2, npoints)
#         # bin_i (2)
#         # bin_j (2)
#         # values (npoints,)
#         # print(np.shape(bin_j))
#         in_bin_j = (bin_j[0] < pos[1]) & (pos[1] <= bin_j[1])
#         in_bin_ij = in_bin_i * in_bin_j
#         n_count = jnp.sum(in_bin_ij)
#         operand = values, in_bin_ij, n_count
#         new_res_ij = jax.lax.cond(n_count > 0, 
#                     n_count_not_zero, 
#                     n_count_zero,
#                     operand)
#         return new_res_ij
    
#     loopj_vmap = jax.vmap(loop_j, (None, None, 1, None), 0)
    
#     def loop_i(pos, values, bin_i, bin_j):
#         # pos (2, npoints)
#         # bin_i (2)
#         # bin_j (2)
#         # values (npoints,)
#         # print(np.shape(bin_i))
#         in_bin_i = (bin_i[0] < pos[0]) & (pos[0] <= bin_i[1])
#         res_i = loopj_vmap(pos, values, jnp.array([bin_j[:-1], bin_j[1:]]), in_bin_i)
#         return res_i

#     loopi_vmap = jax.vmap(loop_i, (None, None, 1, None), 0)
#     # utiliser vmap
#     # print(np.shape(pos))
#     # print(np.shape(values))
#     # print(np.shape((bins[0][:-1], bins[1][:-1])))
#     # print(np.shape((bins[0][1:], bins[1][1:])))
#     res = loopi_vmap(pos, values, jnp.array([bins[0][:-1], bins[0][1:]]), bins[1])
#     return res

def mean_2d_bin_data(pos, values, bins): # pos and bins have shape (2, ...)
    # left limit of bins is excluded
    # shape of pos has to be (2, npoints), with npoints in the TOD

    # values_bini = jnp.digitize(pos[0], bins[0]) # right edge of each bin excluded, left edges included
    # values_binj = jnp.digitize(pos[1], bins[1])
    # res, _, _ = jnp.histogram2d(pos[0], pos[1], bins=bins, weights=values)
    res = histogram2d(pos[0], pos[1], range=((bins[0][0], bins[0][-1]), (bins[1][0], bins[1][-1])), bins=(len(bins[0]) - 1, len(bins[1]) - 1), weights=values)
    return res
# mean_2d_bin_data_jitted = jax.jit(mean_2d_bin_data)
mean_2d_bin_data_jitted = mean_2d_bin_data
# mean_2d_bin_data_jitted = mean_bin_data_nd

#########################

def healpix_map(azt, elt, tod, flags=None, flaglimit=0, nside=128, countcut=0, unseen_val=hp.UNSEEN):
    if flags is None:
        flags = np.zeros(len(azt))
    
    ok = flags <= flaglimit 
    return healpix_map_(azt[ok], elt[ok], tod[ok], nside=nside, countcut=countcut, unseen_val=unseen_val)


def healpix_map_(azt, elt, tod, nside=128, countcut=0, unseen_val=hp.UNSEEN):
# def healpix_map_(elt, azt, tod, nside=128, countcut=0, unseen_val=hp.UNSEEN):
    ips = hp.ang2pix(nside, azt, elt, lonlat=True)
    mymap = np.zeros(12*nside**2)
    mapcount = np.zeros(12*nside**2)
    for i in range(len(azt)):
        mymap[ips[i]] += tod[i]
        mapcount[ips[i]] += 1
    unseen = mapcount <= countcut
    mymap[unseen] = unseen_val
    mapcount[unseen] = unseen_val
    mymap[~unseen] = mymap[~unseen] / mapcount[~unseen]
    return mymap, mapcount

def sparse_2d_to_fullmap(mymap, ipos_grid, jpos_grid):
    # time_0 = time.time()
    # mymap_2 = mymap**2
    # mymap_OK = (np.isfinite(mymap)) & (mymap_2 > 1e-5*np.max(np.isfinite(mymap_2)))
    mymap_OK = np.isfinite(mymap)
    # mymap_OK = mymap != 0
    # mymap_OK = np.ascontiguousarray(mymap_OK_[:])
    # mymap_OK = jax.block_until_ready(mymap_OK) # ensure the result is done running for timing purposes
    # time_1 = time.time()
    # print("elated time in = {}".format(time_1 - time_0))

    # Ni, Nj = np.shape(ipos_grid)
    # plt.figure()
    # plt.imshow(mymap.reshape(Ni, Nj))
    # plt.show()

    # 2D interpolation
    # interp = LinearNDInterpolator(list(zip(np.ravel(ipos_grid)[mymap_OK], np.ravel(jpos_grid)[mymap_OK])), mymap[mymap_OK], fill_value=0)
    interp = LinearNDInterpolator(list(zip((ipos_grid).reshape(-1)[mymap_OK], (jpos_grid).reshape(-1)[mymap_OK])), mymap[mymap_OK], fill_value=0)
    # time_2 = time.time()
    # print("elated time in = {}".format(time_2 - time_1))
    mymap_interp = interp(ipos_grid, jpos_grid)
    # time_3 = time.time()
    # print("elated time in = {}".format(time_3 - time_2))

    # hp.gnomview result puts a decreasing azt as j coord
    mymap_interp = np.flip(mymap_interp, axis=1)
    # time_4 = time.time()
    # print("elated time in = {}".format(time_4 - time_3))

    # plt.figure()
    # img = plt.imshow(mymap_interp)
    # plt.colorbar(img)
    # plt.show()
    # sys.exit()
    return mymap_interp

# def TOD_to_flat_map(azt, elt, vals, img_azt, img_elt):
def TOD_to_flat_map(ipos, jpos, vals, ipos_img, jpos_img):
    """
    Create a flat (Ni, Nj) map from fake TOD created from a flat map (in order to skip the HEALPix stage).
    It is the invert of img_to_TOD.
    ipos, jpos: arrays (n)
        Positions of TOD n points in i and j directions
    vals: array (n)
        Values of TOD n points
    ipos_img: array (Ni)
        Positions of image Ni points in i direction
    jpos_img: array (Nj)
        Positions of image Nj points in j direction
    """

    coord = [ipos, jpos]
    Ni = len(ipos_img)
    Nj = len(jpos_img)
    img_coord = [ipos_img, jpos_img]
    min_coord = np.min(img_coord)
    max_coord = np.max(img_coord)
    Npix = [Ni, Nj]

    i_range = np.linspace(min_coord, max_coord, Ni)
    j_range = np.linspace(min_coord, max_coord, Nj)
    ipos_grid, jpos_grid = np.meshgrid(i_range, j_range, indexing="ij")

    azel_pixsize = []
    azel_bin = []
    for k in range(len(coord)):
        azel_pixsize.append((min_coord - max_coord)/(Npix[k] - 1)) # There are Npix - 1 pixels between the center of the left-most and the center of the right-most pixels
        azel_bin.append(np.linspace(min_coord - azel_pixsize[k]/2, max_coord + azel_pixsize[k]/2, Npix[k] + 1)) # Edges of pixels
    
    # Le code prend plus de temps qu'avant ??

    # time_0 = time.time()
    # timer(mean_bin_data_nd, coord, vals, azel_bin)
    # # timer(mean_2d_bin_data, coord, vals, azel_bin)
    # timer(mean_2d_bin_data_jitted, coord, vals, azel_bin)
    # sys.exit()
    mymap = mean_bin_data_nd(pos=coord, values=vals, bins=azel_bin)#.reshape(Ni, Nj)
    # mymap = np.ravel(mean_2d_bin_data(pos=coord, values=vals, bins=azel_bin))#.reshape(Ni, Nj)
    # mymap = mean_2d_bin_data_jitted(pos=coord, values=vals, bins=azel_bin).reshape(-1)
    # mymap = jax.block_until_ready(mymap) # ensure the result is done running for timing purposes
    # time_0_0 = time.time()
    # print("elated time = {}".format(time_0_0 - time_0))
    # mymap = mean_2d_bin_data_jitted(pos=coord, values=vals, bins=azel_bin)
    # mymap = mean_2d_bin_data(pos=coord, values=vals, bins=azel_bin)
    # mymap = mymap.reshape(-1)
    # mymap_ = np.ravel(mymap_)
    # mymap = jax.block_until_ready(mymap) # ensure the result is done running for timing purposes
    # time_1 = time.time()
    # print("elated time = {}".format(time_1 - time_0_0))
    # sys.exit()

    # mymap_interp_test = sparse_2d_to_fullmap(mymap_test, ipos_grid, jpos_grid)
    mymap_interp = sparse_2d_to_fullmap(mymap, ipos_grid, jpos_grid)

    # time_2 = time.time()
    # print("elated time = {}".format(time_2 - time_1))

    # print("total time = {}".format(time_2 - time_0))

    # plt.figure()
    # img = plt.imshow(mymap_interp - mymap_interp_test)
    # plt.colorbar(img)
    # plt.show()

    # sys.exit()

    return mymap_interp


#######################

def gaussian(x, mu, reso):
		sig = reso / conv_reso_fwhm
		res = 1 / (np.sqrt(2 * np.pi) * sig) * np.exp(-((x - mu) / sig)**2 / 2)
		return res / np.sum(res) # area under the curve = 1

def gauss2D(Nx, Ny, x0, y0, reso, amp=None, normal=True):
    # don't forget to convert all values (x0, y0, reso) in pixel space
    x, y = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    if len(reso) == 1:
        reso = np.array([reso, reso])
    sig = reso / conv_reso_fwhm
    res = np.exp(-(x - x0)**2/(2*sig[0]**2)) * np.exp(-(y - y0)**2/(2*sig[1]**2))
    if normal:
        return(res/np.sum(res))
    else:
        return amp*res


def get_new_azel(azt, elt, azmoon, elmoon):
    newazt = (azt - azmoon) * np.cos(np.radians(elt))
    # newelt = -(elt - elmoon) # so the Moon is higher than trees in maps (?)
    newelt = (elt - elmoon)
    return newazt, newelt


def make_coadded_maps_TES(tt, tod, azt, elt, scantype, newazt, newelt, nside=256, doplot=True, check_back_forth=False):

    # Inversion in signal
    mytod = -tod.copy()

    # Filter the TOD
    mytod_1 = my_filt(mytod.copy())

    mytod_2 = my_filt_2(mytod.copy())

    if doplot:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(tt, -tod, label="TOD")
        ax.plot(tt, mytod_1, label="filtered TOD")
        ax.plot(tt, mytod_2, label="filtered TOD 2")
        ax.legend()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Flux [ADU]")
        plt.tight_layout()
        plt.savefig("tod_filtering.pdf")
        plt.show()
        
    # Map-making

    # newelt = -newelt
    # Calculate center of maps from pointing w.r.t. Moon
    center=[np.mean(newazt), np.mean(newelt)]

    # To compare the map created with only forth scans with the map created with only back scans
    if check_back_forth:
        mapsb_forth, mapcount_forth = healpix_map(newazt[scantype > 0], newelt[scantype > 0], mytod[scantype > 0], nside=nside)
        mapsb_back, mapcount_back = healpix_map(newazt[scantype < 0], newelt[scantype < 0], mytod[scantype < 0], nside=nside)
        plt.figure()
        hp.gnomview(mapsb_forth, reso=10, sub=(1, 3, 1), min=-5e3, max=1.2e4, 
                title="forth scans", rot=center)
        hp.gnomview(mapsb_back, reso=10, sub=(1, 3, 2), min=-5e3, max=1.2e4, 
                title="back scans", rot=center)
        hp.gnomview(mapsb_forth - mapsb_back, reso=10, sub=(1, 3, 3), min=-5e3, max=1.2e4, 
                title="forth - back scans", rot=center)
        plt.show()
        # return mapsb_forth, mapcount_forth, mapsb_back, mapcount_back
    
    mapsb, mapcount = healpix_map(newazt[scantype != 0], newelt[scantype != 0], mytod_1[scantype != 0], nside=nside)

    mapsb_2, mapcount = healpix_map(newazt[scantype != 0], newelt[scantype != 0], mytod_2[scantype != 0], nside=nside)

    plt.figure()
    # hp.gnomview(testmap, reso=10, sub=(1, 2, 1), min=-5e3, max=1.2e4, 
    #             title="gaussian map", rot=center)
    hp.gnomview(mapsb, reso=10, sub=(1, 2, 2), min=-5e3, max=1.2e4, 
                title="final map", rot=center)
    plt.savefig("figures/mapsb.pdf")
    plt.show()

    plt.figure()
    hp.gnomview(mapsb_2, reso=10, sub=(1, 2, 2), min=-5e3, max=1.2e4, 
                title="final map 2", rot=center)
    plt.savefig("figures/mapsb_2.pdf")
    plt.show()


    plt.figure()
    hp.gnomview(mapsb_2 - mapsb, reso=10, sub=(1, 2, 2), min=-5e3, max=1.2e4, 
                title="mapsb_2 - mapsb", rot=center)
    plt.savefig("figures/mapsb_2-mapsb.pdf")
    plt.show()
    # stop
    return mapsb, mapcount


# https://stackoverflow.com/questions/14695367/most-efficient-way-to-filter-a-long-time-series-python
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, *args, **kwargs):
    sos = butter_bandpass(*args, **kwargs)
    return sosfiltfilt(sos, data) # no phase but filter applied twice (forwards and backwards), because I use filtfilt instead of lfilter

def butter_pseudo_bandpass_filter(data, lowcut, highcut, fs, order=2):
    sos = butter_lowpass(highcut, fs, order=order)
    data_altered = sosfiltfilt(sos, data)
    sos = butter_highpass(lowcut, fs, order=order)
    return sosfiltfilt(sos, data_altered) # no phase but filter applied twice (forwards and backwards), because I use filtfilt instead of lfilter

def butter_lowpass(highcut, fs, order):
    nyq = 0.5*fs
    high = highcut/nyq
    sos = butter(order, high, btype='lowpass', output='sos')
    return sos

def butter_highpass(lowcut, fs, order):
    nyq = 0.5*fs
    low = lowcut/nyq
    sos = butter(order, low, btype='highpass', output='sos')
    return sos

def butter_highpass_filter(data, *args, **kwargs):
    sos = butter_highpass(*args, **kwargs)
    return sosfiltfilt(sos, data) # no phase but filter applied twice (forwards and backwards), because I use filtfilt instead of lfilter

def bessel_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    b,a = bessel(order, [low, high], btype='band')
    return b,a

def bessel_bandpass_filter(data, *args, **kwargs):
    b, a = bessel_bandpass(*args, **kwargs)
    return filtfilt(b, a, data) # no phase but filter applied twice (forwards and backwards), because I use filtfilt instead of lfilter

def bessel_highpass(lowcut, fs, order):
    nyq = 0.5*fs
    low = lowcut/nyq
    sos = bessel(order, low, btype='highpass', output="sos")
    return sos

def bessel_highpass_filter(data, *args, **kwargs):
    sos = bessel_highpass(*args, **kwargs)
    return sosfiltfilt(sos, data) # no phase but filter applied twice (forwards and backwards), because I use filtfilt instead of lfilter


def my_filt_2(mytod): # utiliser cette fonction ?
    # Cuts are expressed in Hz, a back and forth scan takes 107.5 seconds
    fs = 157.36 # Hz # could be computed directly on TOD
    # lowcut = 4/107.5 # 4/107.5, i.e. half a forth (or back) scan
    lowcut = 4/107.5
    # highcut = 2/107.5*100/2 # 2/107.5*100/4, i.e. approx. 4 % of a forth (or back) scan --> passer à 2% parce que 4% est trop proche de la taille de la Lune (2/107.5*100/6 makes the Moon round but it's fine-tuned for it...)
    # print("lowcut = {} Hz, highcut = {} Hz".format(lowcut, highcut))
    # filt_tod = butter_bandpass_filter(mytod, lowcut=lowcut, highcut=highcut, fs=fs, order=1) # Hz
    # filt_tod = butter_pseudo_bandpass_filter(mytod, lowcut=lowcut, highcut=highcut, fs=fs, order=1) # Hz
    # filt_tod = butter_highpass_filter(mytod, lowcut=lowcut, fs=fs, order=1) # Hz
    # filt_tod = bessel_bandpass_filter(mytod, lowcut=lowcut, highcut=highcut, fs=fs, order=1) # Hz
    filt_tod = bessel_highpass_filter(mytod, lowcut=lowcut, fs=fs, order=5) # Hz
    return filt_tod

def my_filt(mytod): # utiliser cette fonction ?
    # Cuts are expressed in Hz, a back and forth scan takes 107.5 seconds
    fs = 157.36 # Hz # could be computed directly on TOD
    lowcut = 4/107.5 # 4/107.5, i.e. half a forth (or back) scan
    # highcut = 2/107.5*100/2 # 2/107.5*100/4, i.e. approx. 4 % of a forth (or back) scan --> passer à 2% parce que 4% est trop proche de la taille de la Lune (2/107.5*100/6 makes the Moon round but it's fine-tuned for it...)
    # filt_tod = butter_bandpass_filter(mytod, lowcut=lowcut, highcut=highcut, fs=fs, order=2) # Hz
    # filt_tod = butter_pseudo_bandpass_filter(mytod, lowcut=lowcut, highcut=highcut, fs=fs, order=2) # Hz
    filt_tod = butter_highpass_filter(mytod, lowcut=lowcut, fs=fs, order=2) # Hz
    # filt_tod = bessel_bandpass_filter(mytod, lowcut=lowcut, highcut=highcut, fs=fs, order=3) # Hz
    return filt_tod
        

class gauss2dfit:
    def __init__(self, xx, yy):
        self.xx = xx
        self.yy = yy
    def __call__(self, x, pars):
        amp, xc, yc, sig = pars
        mygauss = amp * np.exp(-0.5*((self.xx-xc)**2+(self.yy-yc)**2)/sig**2)
        return np.ravel(mygauss)

class filtgauss2dfit:
    def __init__(self, ipos, jpos, scantype, allipos, alljpos, nside): # it seems that I should use ipos = elt and jpos = -azt
        # self.xx = xx
        # self.yy = yy
        self.ipos = ipos
        self.jpos = jpos
        self.scantype = scantype
        # We can cut newazt and newelt to the values of xx and yy?
        self.allipos = allipos
        self.alljpos = alljpos
        self.nside = nside
        # self.amp_azt = np.array([np.min(self.xx), np.max(self.xx)])
        # self.amp_elt = np.array([np.min(self.yy), np.max(self.yy)])
        self.Npix = 1000
        self.allipos_range = np.linspace(np.min(allipos), np.max(allipos), self.Npix)
        self.alljpos_range = np.linspace(np.min(alljpos), np.max(alljpos), self.Npix)
        self.iipos_large, self.jjpos_large = np.meshgrid(self.allipos_range, self.alljpos_range, indexing="ij")
        self.amp_ipos = np.array([np.min(self.iipos_large), np.max(self.iipos_large)])
        self.amp_jpos = np.array([np.min(self.jjpos_large), np.max(self.jjpos_large)])

        self.nIter = 0
    def __call__(self, x, pars):
        amp, xc, yc, sig = pars
        # mygauss = amp * np.exp(-0.5*((self.xx - xc)**2+(self.yy - yc)**2)/sig**2)
        mygauss = amp * np.exp(-0.5*((self.iipos_large - xc)**2+(self.jjpos_large - yc)**2)/sig**2)

        # mygauss = jax.block_until_ready(mygauss)
        # timer(img_to_TOD, mygauss, self.amp_ipos, self.amp_jpos, self.allipos, self.alljpos)
        # sys.exit()
        my_gauss_tod = img_to_TOD(mygauss, self.amp_ipos, self.amp_jpos, self.allipos, self.alljpos)
        # timer(img_to_TOD, mygauss, self.amp_ipos, self.amp_jpos, self.allipos, self.alljpos)
        # timer(img_to_TOD_jitted, mygauss, self.amp_ipos, self.amp_jpos, self.allipos, self.alljpos)
        # sys.exit()
        # my_gauss_tod = np.array(img_to_TOD_jitted(mygauss, self.amp_ipos, self.amp_jpos, self.allipos, self.alljpos))

        # my_gauss_tod = jax.block_until_ready(my_gauss_tod)
        # timer(my_filt, my_gauss_tod)
        # sys.exit()
        my_gauss_tod = my_filt(my_gauss_tod)

        # myfiltgauss_hp, _ = healpix_map(azt=self.alljpos[self.scantype > 0], elt=self.allipos[self.scantype > 0], tod=my_gauss_tod[self.scantype > 0], nside=self.nside)
        # rot = np.array([0, 0, 0])
        # reso = 5
        # xs = 201
        # myfiltgauss = hp.gnomview(myfiltgauss_hp, reso=reso, rot=rot, return_projected_map=True, xsize=xs, no_plot=True).data
        # myfiltgauss[myfiltgauss == hp.UNSEEN] = 0

        # plt.figure()
        # plt.imshow(myfiltgauss)
        # plt.show()
        # sys.exit()

        # Get a flat map from the TOD without passing through Healpy --> it actually takes more time...
        myfiltgauss = TOD_to_flat_map(self.allipos[self.scantype > 0], self.alljpos[self.scantype > 0], my_gauss_tod[self.scantype > 0], self.ipos, self.jpos)

        # plt.figure()
        # plt.imshow(myfiltgauss)
        # plt.show()
        # sys.exit()

        self.nIter += 1

        return np.ravel(myfiltgauss)
    

def get_dict(params):
    """QUBIC dictionary.

    Method to modify the qubic dictionary.

    Parameters
    ----------
    key : str, optional
        Can be "in" or "out".
        It is used to build respectively the instances to generate the TODs or to reconstruct the sky maps,
        by default "in".

    Returns
    -------
    dict_qubic: dict
        Modified QUBIC dictionary.

    """
    ### Arguments used when simulating the maps

    args = {
                "npointings": params["QUBIC"]["npointings"],
                "nf_recon": params["QUBIC"]["nrec"],
                "nf_sub": params["QUBIC"]["nsub_in"],
                "nside": params["SKY"]["nside"],
                "MultiBand": True,
                "period": 1,
                "RA_center": params["SKY"]["RA_center"],
                "DEC_center": params["SKY"]["DEC_center"],
                "filter_nu": 150 * 1e9,
                "noiseless": False,
                "beam_shape": 'gaussian',
                #"comm": comm,
                "dtheta": params["QUBIC"]["dtheta"],
                "nprocs_sampling": 1,
                #"nprocs_instrument": comm.size,
                "photon_noise": True,
                "nhwp_angles": 3,
                #'effective_duration':3,
                "effective_duration150": 3,
                "effective_duration220": 3,
                "filter_relative_bandwidth": 0.25,
                "type_instrument": "two",
                "TemperatureAtmosphere150": None,
                "TemperatureAtmosphere220": None,
                "EmissivityAtmosphere150": None,
                "EmissivityAtmosphere220": None,
                "detector_nep": float(params["QUBIC"]["NOISE"]["detector_nep"]),
                "synthbeam_kmax": params["QUBIC"]["SYNTHBEAM"]["synthbeam_kmax"] #,
                #"synthbeam_fraction": params["QUBIC"]["SYNTHBEAM"]["synthbeam_fraction"],
            }

    dictfilename = "pipeline_demo.dict"
    qubic_dict = Qdictionary.qubicDict()
    qubic_dict.read_from_file(dictfilename)

    for i in args.keys():
        qubic_dict[str(i)] = args[i]
    return qubic_dict


def img_to_TOD(img, amp_azt, amp_elt, newazt, newelt):
    # To create fake TOD quickly from an input image
    Npix = np.shape(img) # vérifier ordre lignes colonnes (lignes = azimuth ou elevation ?)
    azel_arr = []
    amplitude = np.array([amp_azt, amp_elt]) # Here amplitude contains the intervals in azimuth and elevation for the pixels' centers
    for i, coord in enumerate(["az", "el"]):
        azel_arr.append(np.linspace(amplitude[i, 0], amplitude[i, 1], Npix[i]))
    # print("new azt in ({}, {}), new elt in ({}, {})".format(np.min(azel_arr[0]), np.max(azel_arr[0]), np.min(azel_arr[1]), np.max(azel_arr[1])))
    # print(np.min(newazt), np.max(newazt), np.min(newelt), np.max(newelt))
    grid_interp = RegularGridInterpolator( (azel_arr[0], azel_arr[1]), img, method='linear' ) # linear is faster than nearest??
    img_tod = grid_interp((newazt, newelt))
    return img_tod
def img_to_TOD_(img, amp_azt, amp_elt, newazt, newelt):
    # To create fake TOD quickly from an input image
    Npix = jnp.shape(img) # vérifier ordre lignes colonnes (lignes = azimuth ou elevation ?)
    azel_arr = []
    amplitude = jnp.array([amp_azt, amp_elt]) # Here amplitude contains the intervals in azimuth and elevation for the pixels' centers
    for i, coord in enumerate(["az", "el"]):
        azel_arr.append(jnp.linspace(amplitude[i, 0], amplitude[i, 1], Npix[i]))
    # print("new azt in ({}, {}), new elt in ({}, {})".format(jnp.min(azel_arr[0]), jnp.max(azel_arr[0]), jnp.min(azel_arr[1]), jnp.max(azel_arr[1])))
    # print(jnp.min(newazt), jnp.max(newazt), jnp.min(newelt), jnp.max(newelt))
    grid_interp = jax.scipy.interpolate.RegularGridInterpolator( (azel_arr[0], azel_arr[1]), img, method='nearest' ) 
    img_tod = grid_interp((newazt, newelt))
    return img_tod
img_to_TOD_jitted = jax.jit(img_to_TOD_)

def map_to_TOD(hp_map, newazt, newelt):
    # To create fake TOD quickly from an input HEALPix map
    # Use azimuth, elevation as latitude and longitude in degrees
    map_tod = hp.get_interp_val(hp_map, newazt, newelt, lonlat=True)
    return map_tod

def fitgauss_img(mapxy, ipos, jpos, xs, guess=None, doplot=False, distok=3, mytit='', nsig=1, mini=None, maxi=None, ms=10, renorm=False, mynum=33, axs=None, verbose=False, reso=None, pack=None):
    # xx, yy = np.meshgrid(x, y, indexing="xy")
    iipos, jjpos = np.meshgrid(ipos, jpos, indexing="ij")
    iipos = jax.block_until_ready(iipos)
    time_0 = time.time()
    
    ### Displays the image as an array
    mm, ss = ft.meancut(mapxy, 3)
    if mini is None:
        mini = mm-nsig*ss
    if maxi is None:
        maxi = np.max(mapxy)


    # g2d = gauss2dfit(iipos, jjpos)
    scantype, newazt, newelt, nside = pack
    guess = jax.block_until_ready(guess)
    time_1 = time.time()
    print("Time before filtgauss2dfit initialization is: {}".format(time_1 - time_0))

    g2d = filtgauss2dfit(ipos, jpos, scantype, newelt, newazt, nside)


    ### Guess where the maximum is and the other parameters with a matched filter
    if guess is None:
        Nx = len(mapxy)
        Ny = len(mapxy[0])
        lobe_pos = (Nx//2, Ny//2)
        Kx, Ky, K = get_K(Nx, Ny)
        ft_phase = get_ft_phase(lobe_pos, Nx, Ny)
        cos_win = cos_window(Nx, Ny, lx=20, ly=20)
        deltaK = 1
        Kbin = get_Kbin(deltaK, K)
        nKbin = len(Kbin) - 1  # nb of bins
        Kcent = (Kbin[:-1] + Kbin[1:])/2
        size_pix = reso/60 # degree
        # reso_instr = 0.92 # degree
        reso_img = 1.036 # degree # test
        ft_shape = fft2(gauss2D(Nx, Ny, x0=lobe_pos[0], y0=lobe_pos[1], reso=[reso_img/size_pix], normal=True))

        # Test with a filtered Gaussian
        # pars = 1, iipos[Nx//2, Ny//2], jjpos[Ny//2, Ny//2], reso_img/conv_reso_fwhm
        # myfiltgauss_flat = g2d(0, pars)
        # myfiltgauss = myfiltgauss_flat.reshape(Nx, Ny)
        # ft_shape = fft2(myfiltgauss)

        # plt.figure()
        # plt.imshow(myfiltgauss)
        # plt.show()

        filtmapsn = get_filtmapsn(mapxy * cos_win, nKbin, K, Kbin, Kcent, ft_shape, ft_phase)

        # plt.figure()
        # plt.imshow(filtmapsn)
        # plt.show()
        # sys.exit()
        
        maxii = filtmapsn == np.nanmax(filtmapsn)
        max_i = np.mean(iipos[maxii])
        max_j = np.mean(jjpos[maxii])
        guess = np.array([1e4, max_i, max_j, reso_img/conv_reso_fwhm])
        if verbose:
            print(guess)
    else:
        max_i = guess[1]
        max_j = guess[2]
        
    ### Do the fit putting the UNSEEN to a very low weight
    errpix = iipos*0 + ss
    errpix[mapxy==0] *= 1e5


    g2d = jax.block_until_ready(g2d)
    time_2 = time.time()
    print("Time before fit.Data is: {}".format(time_2 - time_1))

    data = fit.Data(np.ravel(iipos), np.ravel(mapxy), np.ravel(errpix), g2d)

    data = jax.block_until_ready(data)
    time_3 = time.time()
    print("Time before data.fit_minuit is: {}".format(time_3 - time_2))

    m, ch2, ndf = data.fit_minuit(guess, limits=[[0, 1e3, 1e8], [1, max_i - distok, max_i + distok], [2, max_j - distok, max_j + distok], [3, 0.6/conv_reso_fwhm, 1.2/conv_reso_fwhm]], renorm=renorm)

    m = jax.block_until_ready(m)
    time_4 = time.time()
    print("Time taken by data.fit_minuit is: {}".format(time_4 - time_3))

    print("{} iterations of filtgauss2dfit needed.".format(g2d.nIter))
    
    ### Image of the fitted Gaussian
    fitted = np.reshape(g2d(ipos, m.values), (xs, xs))

    if doplot:
        origin = "upper" #"lower" swaps the y-axis and the guess doesn't match, default is "upper"
        if axs is None:
            fig, axs = plt.subplots(1, 4, width_ratios=(1, 1, 1, 0.05), figsize=(16, 5))
            axs[1].imshow(fitted, origin=origin, extent=[np.min(ipos), np.max(ipos), np.min(jpos), np.max(jpos)], vmin=mini, vmax=maxi)
            im = axs[2].imshow(mapxy - fitted, origin=origin, extent=[np.min(ipos), np.max(ipos), np.min(jpos), np.max(jpos)], vmin=mini, vmax=maxi)
            axs[0].set_ylabel('Degrees')
            for i in range(3):
                axs[i].set_xlabel('Degrees')
            axs[2].set_title('Residuals')
    
    if doplot:
        axs = pmp.plot_fit_img(mapxy, axs, ipos, jpos, iguess=guess[1], jguess=guess[2], ifit=m.values[1], jfit=m.values[2], vmin=mini, vmax=maxi, ms=ms, origin=origin)
        return m, fitted, axs
    return m, fitted
    
    

def fit_one_tes(mymap, xs, reso, rot=np.array([0., 0., 0.]), doplot=False, verbose=False, guess=None, distok=3, mytit='', return_images=False, ms=10, renorm=False, xycreid_corr=None, axs=None, pack=None):
    a = 0
    a = jax.block_until_ready(a)
    time_0 = time.time()
    ### get the gnomview back into a np.array in order to fit it
    mm = mymap.copy()
    badpix = mm == hp.UNSEEN
    mm[badpix] = 0          ### Set bad pixels to zero before returning the np.array()
    mapxy = hp.gnomview(mm, reso=reso, rot=rot, return_projected_map=True, xsize=xs, no_plot=True).data

    ### np.array coordinates
    # Doesn't work with the fit plot but is ok with final gnomview plot of the Moon map corrected
    # But in order to stack the maps I now have to use (azt, -elt) position fitted here (why??)
    # x = -(np.arange(xs) - (xs - 1)/2)*reso/60
    # y = x.copy()
    # x += rot[0]
    # y -= rot[1]

    # Works on fit plot but then azt and elt are with the wrong sign on the final gnomview plot. Weird!!
    # x = (np.arange(xs) - (xs - 1)/2)*reso/60
    # y = x.copy()
    # x -= rot[0]
    # y += rot[1]

    # Other tests
    x = (np.arange(xs) - (xs - 1)/2)*reso/60
    y = -x.copy()
    x += rot[1]
    y -= rot[0]


    # print(np.min(y), np.max(y))
    # sys.exit()

    if xycreid_corr is not None:
        try:
            guess = np.array([1e4, xycreid_corr[0], xycreid_corr[1], 0.92])
            if verbose:
                print(guess)
        except:
            guess = None
            if verbose:
                print("TES has no position on sky")
                print(guess)
        
    y = jax.block_until_ready(y)
    time_1 = time.time()
    print("Time before fitgauss_img: {}".format(time_1 - time_0))
    if doplot:
        m, fitted, fig_axs = fitgauss_img(mapxy, x, y, xs, guess=guess, doplot=doplot, distok=distok, mytit=mytit, ms=ms, renorm=renorm, axs=axs, verbose=verbose, reso=reso, pack=pack)
        if verbose:
            print(m.values)
    else:
        m, fitted = fitgauss_img(mapxy, x, y, xs, guess=guess, doplot=doplot, distok=distok, mytit=mytit, ms=ms, renorm=renorm, verbose=verbose, reso=reso, pack=pack)
    # try:
    #     m, fitted = fitgauss_img(mapxy, x, y, guess=guess, doplot=doplot, distok=distok, mytit=mytit, ms=ms, renorm=renorm)
    # except:
    #     m = None
    #     fitted = None
    
    m = jax.block_until_ready(m)
    time_2 = time.time()
    print("Total time is: {}".format(time_2 - time_0))

    if return_images:
        return m, mapxy, fitted, [np.min(x), np.max(x), np.min(y), np.max(y)], fig_axs

    return m
    

def get_close(deltax, deltay, tolerance):
    return np.sqrt(deltax**2 + deltay**2) <= tolerance

def assign_TES(x, y, xc, yc, tolerance, doplot=True):
    # xc and yc have been corrected for the shift and rotation
    # We want to check if some TES have the wrong number assigned to them

    # We first get the TES that are correctly numbered
    OK_1 = get_close(x - xc, y - yc, tolerance)

    if doplot:
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect(1)
        ax.plot(x, y, "ro", alpha=0.2)
        ax.plot(xc, yc, "ko", label="Creidhe rotated ({})".format(len(xc)))
        ax.plot(x[OK_1], y[OK_1], "go", label="Well assigned ({})".format(np.sum(OK_1)))
        plt.legend()
        plt.show()
        
    # We then iterate


# From fitting.py of JC
class Data:
    def __init__(self, x, y, cov, model, pnames=None):
        self.x = x
        self.y = y
        self.model = model
        self.cov = cov
        if np.prod(np.shape(x)) == np.prod(np.shape(cov)):
            self.diag = True
            self.errors = cov
        else:
            self.diag = False
            self.errors = 1./np.sqrt(np.diag(cov))
            self.invcov = np.linalg.inv(cov)
        self.fit = None
        self.fitinfo = None
        self.pnames = pnames
        
    def __call__(self):
        return 0

    def plot(self, nn=1000, color=None, mylabel=None, nostat=False):
        p=plt.errorbar(self.x, self.y, yerr=self.errors, fmt='o', color=color, alpha=1)
        if self.fit is not None:
            xx = np.linspace(np.min(self.x), np.max(self.x), nn)
            plt.plot(xx, self.model(xx, self.fit), color=p[0].get_color(), alpha=1, label=mylabel)
        if mylabel is None:
            if nostat == False:
                plt.legend(title="\n".join(self.fit_info))
        else:
            plt.legend()


    
    def fit_minuit(self, guess, fixpars = None, limits=None, scan=None, renorm=False, simplex=False, minimizer=LeastSquares):
        ok = np.isfinite(self.x) & (self.errors != 0)

        ### Prepare Minimizer
        if self.diag == True:
            myminimizer = minimizer(self.x[ok], self.y[ok], self.errors[ok], self.model)
        else:
            print('Non diagoal covariance not yet implemented: using only diagonal')
            myminimizer = minimizer(self.x[ok], self.y[ok], self.errors[ok], self.model)

        ### Instanciate the minuit object
        if simplex == False:
            m = iminuit.Minuit(myminimizer, guess, name=self.pnames)
        else:
            m = iminuit.Minuit(myminimizer, guess, name=self.pnames).simplex()
        
        ### Limits
        if limits is not None:
            mylimits = []
            for k in range(len(guess)):
                mylimits.append((None, None))
            for k in range(len(limits)):
                mylimits[limits[k][0]] = (limits[k][1], limits[k][2])
            m.limits = mylimits

        ### Fixed parameters
        if fixpars is not None:
            for k in range(len(guess)):
                m.fixed["x{}".format(k)]=False
            for k in range(len(fixpars)):
                m.fixed["x{}".format(fixpars[k])]=True

        ### If requested, perform a scan on the parameters
        if scan is not None:
            m.scan(ncall=scan)

        ### Call the minimization
        m.migrad()  

        ### accurately computes uncertainties
        m.hesse()   

        ch2 = m.fval
        ndf = len(self.x[ok]) - m.nfit
        self.fit = m.values

        self.fit_info = [
            f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {ch2:.1f} / {ndf}",
        ]
        for i in range(len(guess)):
            vi = m.values[i]
            ei = m.errors[i]
            self.fit_info.append(f"{m.parameters[i]} = ${vi:.3f} \\pm {ei:.3f}$")

        if renorm:
            m.errors *= 1./np.sqrt(ch2/ndf)

        return m, ch2, ndf
    

def get_K(Nx, Ny):
    '''
    Parameters
    ----------
    hd : map header

    Returns
    -------
    Kx : 2D numpy array (Nx,Ny)
        K values for x dimension for the map.
    Ky : 2D numpy array (Nx,Ny)
        K values for y dimension for the map.
    K : 2D numpy array (Nx,Ny)
        K values for the map.
    '''
    Kx, Ky = np.meshgrid(fftfreq(Nx,d=1/Nx),fftfreq(Nx,d=1/Ny),indexing='ij')
    K=np.sqrt(Kx**2+Ky**2)
    return Kx ,Ky, K

def get_Kbin(deltaK, K):
    Kmax = np.ceil(np.max(K))
    k = np.arange(3+deltaK/2,Kmax+deltaK-1,deltaK)
    Kbin = np.concatenate(([0,1.5],k[:-2],[Kmax]))  # same def as JB bins (the middle of the bins are JB kp) except for the last bin
    return Kbin

def get_ft_phase(lobe_pos, Nx, Ny):  # problème si pas de round ! pourquoi ? parce que image décalée d'un nombre non entier de pixels/modes ?
    '''
    Parameters
    ----------
    lobe_pos : int tuple (2)
        x position and y position of the lobe.
    hd : map header

    Returns
    -------
    ft_phase : double
        corrective ft_phase of beam.
    '''
    px = lobe_pos[0]
    py = lobe_pos[1]
    x, y = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    ft_phase = np.exp(-2*np.pi*1j*x*round(-px)/Nx)*np.exp(-2*np.pi*1j*y*round(-py)/Ny)
    return ft_phase


def get_filtmapsn(mapj, nKbin, K, Kbin, Kcent, ft_beam_map, ft_phase):
    ftmapj = fft2(mapj)
    result = np.zeros((nKbin))
    modu2 = ftmapj*np.conj(ftmapj)
    for i in range(nKbin):
        iKbin = np.logical_and(K>=Kbin[i], K<Kbin[i + 1])
        if len(K[iKbin])>0:
            result[i] = np.abs(np.mean(modu2[iKbin]))
        else:
            print("Kbin [{}, {}] is empty.".format(Kbin[i], Kbin[i + 1]))
    gp = np.interp(K, Kcent, result)   # Pk bins interpolated
    
    ftfilt = np.conj(ft_beam_map)/gp
    normfilt = np.sum(np.abs(ft_beam_map)**2/gp)
    filtmapsn = np.real(ifft2(ftfilt*ftmapj*ft_phase)/np.sqrt(normfilt))  # M convol T / sigma
    return filtmapsn

def cos_window(Nx, Ny, lx=None,ly=None):
    """
    Jean-Baptiste appelle le code avec :
    lx = Nx*0.05/2
    ly = Ny*0.05/2  # donc 2,5% de l'image de chaque côté
    """
    if lx==None:
        lx=Nx*0.05/2
    if ly==None:
        ly=Ny*0.05/2
    result = np.ones((Nx,Ny))
    X,Y = np.meshgrid(np.arange(Nx),np.arange(Ny),indexing='ij')
    whx = X <= lx
    result[whx]=1/2.*(1-np.cos(np.pi/lx*X[whx]))
    whx = X >= Nx-1-lx
    result[whx]=1/2.*(1-np.cos(np.pi/lx*(Nx-1-X[whx])))
    why = Y <= ly
    result[why]=result[why]*1/2.*(1-np.cos(np.pi/ly*Y[why]))  # pour faire les coins aussi
    why = Y >= Ny-1-ly
    result[why]=result[why]*1/2.*(1-np.cos(np.pi/ly*(Ny-1-Y[why])))  # pour faire les coins aussi
    return result


def read_data(datadir, remove_t0=True):
    """
    Reads QUBIC raw data: time and TOD, as well azimuth, elevation and 
    their corresponding time

    Parameters
    ----------
    datadir : string
        Full path of the directory where the raw data is stored
        ex/ '/Volumes/QubicData/Calibration/2022-07-14/
    remove_t0 : bool
        subtracts the time of the first sample to the time vector, by default True.

    Returns
    -------
    tt : time for TOD
    tod : the TODs for all detectors
    thk : time for housekeeping data
    az : azimuth of the mount
    el : elevation of the mount
    """
    
    a = qubicfp()
    a.read_qubicstudio_dataset(datadir)
    tt, alltod = a.tod()
    az = a.azimuth()
    el = a.elevation()
    thk = a.timeaxis(datatype='hk')
    tinit = tt[0]
    if remove_t0:
        ### We remove tt[0]
        tinit = tt[0]
        tt -= tinit
        thk -= tinit
    del(a)
    return tt, alltod, thk, az, el, tinit

def get_azel_moon(ObsSite, tt, tinit, doplot=True):
    MySite = EarthLocation(lat=ObsSite['lat'], lon=ObsSite['lon'], height=ObsSite['height'])
    # utcoffset = ObsSite['UTC_Offset']

    dt0 = datetime.utcfromtimestamp(int((tt + tinit)[0]))
    print(dt0)

    nbtime = 100
    tt_hours_loc = tt/3600
    delta_time = np.linspace(np.min(tt_hours_loc), np.max(tt_hours_loc), nbtime)*u.hour

    alltimes = Time(dt0) + delta_time

    ### Local coordinates
    frame_Site = AltAz(obstime=alltimes, location=MySite)

    ### Source
    moon_Site = get_moon(alltimes)
    moonaltazs_Site = moon_Site.transform_to(frame_Site)  

    myazmoon = moonaltazs_Site.az.value
    myelmoon = moonaltazs_Site.alt.value

    azmoon = np.interp(tt_hours_loc, delta_time/u.hour, myazmoon)
    elmoon = np.interp(tt_hours_loc, delta_time/u.hour, myelmoon)
    if doplot:
        plt.figure()
        plt.plot(myazmoon, myelmoon, 'ro')
        plt.plot(azmoon, elmoon)
        plt.show()
    return azmoon, elmoon


def make_coadded_maps(datadir, ObsSite, allTESNum, start_tt=10000, data=None, speedmin=0.05, 
                      doplot=True, nside=256, az_qubic=0, parallel=False, check_back_forth=False):
    ### First read the data from disk if needed
    if data is None:
        print('Reading data from disk: '+datadir)
        tt, alltod, thk, az, el, tinit = read_data(datadir, remove_t0=False)
        az += az_qubic
        tt_save = np.copy(tt)
        alltod_save = np.copy(alltod)
        thk_save = np.copy(thk)
        az_save = np.copy(az)
        el_save = np.copy(el)
        data = [tt_save, alltod_save, thk_save, az_save, el_save, tinit]
        print("tinit = {}".format(tinit))
    else:
        print('Using data already stored in memory - not read from disk')
        tt_save, alltod_save, thk_save, az_save, el_save, tinit = data
        tt = np.copy(tt_save)
        alltod = np.copy(alltod_save)
        thk = np.copy(thk_save)
        az = np.copy(az_save)
        el = np.copy(el_save)
        print(np.shape(tt))
        print(np.shape(alltod))
        print("tinit = {}".format(tinit))
    
    # Remove the first start_tt points (out of 1998848)
    tinit = tt[start_tt]
    print("tinit = {}".format(tinit))
    alltod = alltod[:, start_tt:]
    tt = tt[start_tt:]
    # I need to put tt[0] to zero, but be careful of real time
    # Also, I would have to adjust the mount time?
    tt -= tinit + 0.21 # delta_t seen in plotting back and forth images
    thk -= tinit
    print(np.shape(tt))
    print(np.shape(alltod))
    print("tinit = {}".format(tinit))

    ### Azimuth and Elevation of the Moon at the same timestamps from the observing site
    azmoon, elmoon = get_azel_moon(ObsSite, tt, tinit, doplot=False)

    
    ### Identify scan types and numbers
    scantype_hk, azt, elt, scantype, vmean = identify_scans(thk, az, el, 
                                                                tt=tt, doplot=False, 
                                                                plotrange=[0, 2000], 
                                                                thr_speedmin=speedmin)

    # New coordinates centered on the Moon
    newazt, newelt = get_new_azel(azt, elt, azmoon, elmoon)

    ### Loop over TES to do the maps
    print('\nLooping coaddition mapmaking over selected TES')
    print('nside = ',nside)
    start_time = time.perf_counter()
    if parallel is False:
        print('Using sequential loop')
        allmaps = np.zeros((len(allTESNum), 12*nside**2))
        for i in range(len(allTESNum)):
            TESNum = allTESNum[i]
            print('TES# {}'.format(TESNum), end=" ")
            tod = alltod[TESNum-1,:]
            
            allmaps[i,:], mapscounts = make_coadded_maps_TES(tt, tod, azt, elt, scantype, newazt, newelt,
                                                             nside=nside, 
                                                             doplot=doplot, check_back_forth=check_back_forth)
            print('OK', flush=True)
    else:
        print('using a parallel loop : no output will be given while processing... be patient...')
        ### Note that this code has been generated using ChatGPT
        def process_TES(i, TESNum, allmaps, alltod, tt, azt, elt, scantype, newazt, newelt, nside, doplot):
            # Create a lock for each process to ensure safe access to shared memory
            lock = Lock()
            
            tod = alltod[TESNum - 1, :]

            map_result, mapscounts = make_coadded_maps_TES(tt, tod, azt, elt, scantype, newazt, newelt,
                                                           nside=nside, doplot=doplot)        
            # Use lock to ensure safe access to shared memory inside the inner function
            with lock:
                # Directly assign the result to the correct index in allmaps
                # allmaps is a list of numpy arrays, so we can use allmaps[i] directly
                allmaps[i] = map_result
        
        def parallel_coadded_maps(allTESNum, alltod, tt, azt, elt, scantype, azmoon, elmoon, nside, doplot):
            # Use Manager to create a shared list that will be modified by parallel processes
            with Manager() as manager:
                # Create a list of NumPy arrays initialized to zeros
                allmaps = manager.list([np.zeros(12 * nside ** 2) for _ in range(len(allTESNum))])
        
                # Run the parallel processing with the correct arguments
                Parallel(n_jobs=-1)(delayed(process_TES)(i, allTESNum[i], allmaps, alltod, tt, azt, elt, scantype, azmoon, elmoon, nside, doplot)
                                    for i in range(len(allTESNum)))
        
                # Convert the manager list back to a NumPy array (this ensures allmaps is a numpy array of arrays)
                allmaps_np = np.array([np.array(allmaps[i]) for i in range(len(allTESNum))])
        
            return allmaps_np

        allmaps = parallel_coadded_maps(allTESNum, alltod, tt, azt, elt, 
                                        scantype, azmoon, elmoon, nside, doplot)
    
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds => average of {(elapsed_time/len(allTESNum)):.4f} per TES")    
        

    # Get central Az and El from pointing
    # newazt, newelt = get_new_azel(azt, elt, azmoon, elmoon)
    center = [np.mean(newazt), np.mean(newelt)]
    return allmaps, data, center, newazt, newelt, scantype


# from QdataHandling
def identify_scans(thk, az, el, tt=None, median_size=101, thr_speedmin=0.1, doplot=False, plotrange=[0,1000]):
    """
    This function identifies and assign numbers the various regions of a back-and-forth scanning using the housepkeeping time, az, el
        - a numbering for each back & forth scan
        - a region to remove at the end of each scan (bad data due to FLL reset, slowingg down of the moiunt, possibly HWP rotation
        - is the scan back or forth ?
    It optionnaly iinterpolate this information to the TOD sampling iif provided.
    Parameters
    ----------
    input
    thk : np.array()
            time samples (seconds) for az and el at the housekeeeping sampling rate
    az : np.array()
            azimuth in degrees at the housekeeping sampling rate
    el : np.array()
            elevation in degrees at the housekeeping sampling rate
    tt : Optional : np.array()
            None buy default, if not None:
            time samples (seconds) at the TOD sampling rate
            Then. the output will also containe az,el and scantype interpolated at TOD sampling rate
    thr_speedmin : Optional : float
            Threshold for angular velocity to be considered as slow
    doplot : [Optional] : Boolean
            If True displays some useeful plot
    output :
    scantype_hk: np.array(int)
            type of scan for each sample at housekeeping sampling rate:
            * 0 = speed to slow - Bad data
            * n = scanning towards positive azimuth
            * -n = scanning towards negative azimuth
            where n is the scan number (starting at 1)
    azt : [optional] np.array()
            azimuth (degrees) at the TOD sampling rate
    elt : [optional] np.array()
            elevation (degrees) at the TOD sampling rate
    scantype : [optiona] np.array()
            same as scantype_hk, but interpolated at TOD sampling rate
    """

    def get_az_vel(time, azimuth, order=2): # get the angular azimuth velocity
        az_vel = np.zeros(len(time))
        az_vel[:order] = (azimuth[1:order + 1] - azimuth[:order])/(time[1:order + 1] - time[:order])
        az_vel[-order:] = (azimuth[-order:] - azimuth[-order - 1:-1])/(time[-order:] - time[-order - 1:-1])
        dt_ = time[2*order:] - time[:-2*order]
        az_vel[order:-order] = (az[2*order:] - az[:-2*order])/dt_
        return az_vel
    medaz_dt_ = get_az_vel(thk, az, order=50) # high order necessary to remove glitches
    medaz_dt = medfilt(medaz_dt_, median_size)
    if doplot:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(thk, medaz_dt_)
        plt.plot(thk, medaz_dt)
        plt.xlim(plotrange[0],plotrange[1])

        plt.subplot(1, 2, 2)
        plt.plot(thk, az)
        plt.xlim(plotrange[0],plotrange[1])
        plt.show()
    ### Identify regions of change
    # Low velocity -> Bad
    c0 = np.abs(medaz_dt) < thr_speedmin
    # Positive velicity => Good
    cpos = (~c0) * (medaz_dt >= 0)
    # Negative velocity => Good
    cneg = (~c0) * (medaz_dt < 0)

    ### Scan identification at HK sampling
    scantype_hk = np.zeros(len(thk), dtype='int')-10
    scantype_hk[c0] = 0
    scantype_hk[cpos] = 1
    scantype_hk[cneg] = -1
    # check that we have them all
    count_them = np.sum(scantype_hk==0) + np.sum(scantype_hk==-1) + np.sum(scantype_hk==1)
    if count_them != len(scantype_hk):
        ValueError('Identify_scans: Bad Scan counting at HK sampling level - Error')

    ### Now give a number to each back and forth scan
    num = 0
    previous = 0
    for i in range(len(scantype_hk)):
        if scantype_hk[i] <= 0:
            previous = 0
        elif previous == 0:
            # we have a change
            num += 1
            previous = 1
        scantype_hk[i] *= num

    dead_time = np.sum(c0) / len(thk)

    if doplot:
        ### Some plotting (a lot), moved to other file not to take too much space here
        pmp.plots_identify_scans(thk, plotrange, az, medaz_dt, c0, cpos, cneg, dead_time, el, scantype_hk)

        

    vmean = 0.5 * (np.abs(np.mean(medaz_dt[cpos])) +  np.abs(np.mean(medaz_dt[cneg])))
    if tt is not None:
        ### We propagate these at TOD sampling rate  (this is an "step interpolation": we do not want intermediatee values")
        scantype = interp1d(thk, scantype_hk, kind='previous', fill_value='extrapolate')(tt)
        scantype = scantype.astype(int)
        count_them = np.sum(scantype==0) + np.sum(scantype<=-1) + np.sum(scantype>=1)
        if count_them != len(scantype):
            ValueError('Bad Scan counting at data sampling level - Error')
        ### Interpolate azimuth and elevation to TOD sampling
        azt = np.interp(tt, thk, az)
        elt = np.interp(tt, thk, el)
        ### Return evereything
        return scantype_hk, azt, elt, scantype, vmean
    else:
        ### Return scantype at HK sampling only
        return scantype_hk
    
### DBSCAN
from sklearn.cluster import DBSCAN
def run_DBSCAN(params, eps=0.5, min_samples=10):
    # clustering = DBSCAN(eps=1.3, min_samples=10).fit(params)
    # clustering = DBSCAN(eps=0.5, min_samples=10).fit(params)
    # clustering = DBSCAN(eps=0.25, min_samples=10).fit(params)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(params)
    labels = clustering.labels_
    return labels


# transformer en fonction pour réutiliser avec les fits sur les valeurs corrigées de Créidhe
def get_DBscan_res(x_fit, y_fit, x_theo, y_theo, x_err, y_err, FWHM, errFWHM, visibly_ok_arr, doplot, eps=0.5, min_samples=10):

    delta_az = x_fit - x_theo
    err_delta_az = x_err
    delta_el = y_fit - y_theo
    err_delta_el = y_err

    params_dbscan = np.array([delta_az, delta_el, err_delta_az, err_delta_el, FWHM, errFWHM]).T
    rng_nan = np.random.default_rng(seed=12345)
    params_dbscan[np.isnan(params_dbscan)] = rng_nan.uniform(low=1, high=2, size=(len(params_dbscan[np.isnan(params_dbscan)]),)) * 1e8

    labels = run_DBSCAN(params_dbscan, eps=eps, min_samples=min_samples)
    DB_ok = labels == 0
    if doplot:
        plt.figure()
        plt.subplot().set_aspect(1)
        plt.plot(delta_az[visibly_ok_arr], delta_el[visibly_ok_arr], 'ko', label='all visibly ok ({})'.format(len(delta_az[visibly_ok_arr])))
        plt.plot(delta_az[DB_ok], delta_el[DB_ok], 'ro', label='DBSCAN selected ({})'.format(len(delta_az[DB_ok])))
        plt.xlabel('$\Delta_{az}^{Moon} - Offset_{Creidhe}$')
        plt.ylabel('$\Delta_{el}^{Moon} - Offset_{Creidhe}$')
        plt.legend()
        plt.show()

    return DB_ok


### Function to rotate a set of points around a given center
def rotate_translate_scale_2d(xin, theta, center, scale):
    rotmat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    return scale * np.dot(rotmat, (xin-center).T).T

def rot_trans_scale_pts(x, pars):
    pts = np.reshape(x, (len(x)//2, 2))
    return np.ravel(rotate_translate_scale_2d(pts, np.radians(pars[0]), np.array([pars[1],pars[2]]), pars[3]))


def get_synthbeam_freqs(idet, nsub=16, nside=512):
    # Not finished but the aim is to fit a synthesized beam on the Moon maps
    # It means it should be fake TOD with synthesized beam

    # I can try to change the pointing direction maybe
    dictfilename = "pipeline_demo.dict"
    qubic_dict = Qdictionary.qubicDict()
    qubic_dict.read_from_file(dictfilename)

    qubic_dict["nside"] = nside
    qubic_dict["nf_sub"] = nsub
    qubic_dict["MultiBand"] = True
    qubic_dict["filter_nu"] = 150 * 1e9
    qubic_dict["synthbeam_kmax"] = 1
    qubic_dict["noiseless"] = True
    qubic_dict["npointings"] = 2

    # print(qubic_dict)

    # idet = 67
    acq = Qacquisition.QubicMultiAcquisitions(qubic_dict, nsub=qubic_dict['nf_sub'], nrec=2)
    synthbeam_list = [acq.subacqs[ifreq].instrument[idet].get_synthbeam(acq.subacqs[0].scene)[0] for ifreq in range(nsub//2)] # only first band sub bands

    return synthbeam_list, qubic_dict


def moon_with_synthbeam(synthbeam_list, moon_spectrum, rot, reso=4, xs=401):
    tot_signal = np.sum([synthbeam_list[i] * moon_spectrum[i] for i in range(len(synthbeam_list))])
    # sb_img = hp.gnomview(np.log10(sb/np.max(sb)), rot=[0, 95], reso=reso, min=-3, xsize=xs, title="Synthesized Beam - log scale", return_projected_map=True, no_plot=True).data
    sb_img = hp.gnomview(tot_signal, reso=reso, rot=rot, min=-5e3, max=1.2e4, return_projected_map=True, xsize=xs, no_plot=True).data
    # fig, (ax, cax) = plt.subplots(1, 2, width_ratios=(1, 0.05))
    # img = ax.imshow(sb_map, vmin=-3, origin="lower")
    # fig.colorbar(img, cax=cax)
    # plt.show()
    # sys.exit()
    return sb_img


def get_synthbeam_fit_tod(newazt, newelt, moon_pos, reso, amp):
    # On donne les coord des tod et on calcule des faux tod à partir du lobe synthétique centré en
    # la position de la Lune
    if np.any(np.isnan(moon_pos)):
        return np.zeros_like(newazt), np.zeros_like(newazt)
    Nx = 10000
    Ny = Nx
    min_azt = np.min(newazt)
    max_azt = np.max(newazt)
    min_elt = np.min(newelt)
    max_elt = np.max(newelt)
    pixsize = np.array([(max_azt - min_azt) / (Nx - 1), (max_elt - min_elt) / (Ny - 1)])
    azt_pix = np.linspace(min_azt, max_azt, Nx)
    elt_pix = np.linspace(min_elt, max_elt, Ny)
    if not ((min_azt <= moon_pos[0] <= max_azt) and (min_elt <= moon_pos[1] <= max_elt)):
        print("moon_pos = ({}, {})".format(moon_pos[0], moon_pos[1]), flush=True)
        print("azimuth in [{}, {}], elevation in [{}, {}]".format(min_azt, max_azt, min_elt, max_elt))
        raise TypeError("Position of the Moon outside of map... Aborting.")
    # Moon position in pixel space
    moon_pos_pix = [np.argmin((azt_pix - moon_pos[0])**2), np.argmin((elt_pix - moon_pos[1])**2)]
    print("moon_pos_pix = {}".format(moon_pos_pix), flush=True)

    # Do it with the synthesized beam instead of a simple gaussian
    # synthbeam = get_synthbeam()
    # print("np.shape(synthbeam)", np.shape(synthbeam))
    # plt.figure()
    # plt.imshow(synthbeam, vmin=-1e3, vmax=1e4)
    # plt.show()
    # sys.exit()

    synthbeam_list, qubic_dict = get_synthbeam_freqs(nsub=16, nside=256) # array of the synthesized beams from the instrument for nsub//2 freqs
    # Fixed values, not to be in the fit loop

    # First thing is to fit an possible rotation
    # Then fit the Moon spectrum

    # The Moon spectrum is the changing parameter
    moon_spectrum = np.zeros(len(synthbeam_list))

    rot=[]
    moon_with_synthbeam(synthbeam_list, moon_spectrum, rot, reso=4, xs=401)

    # Gaussian centred on the position of the Moon in pixel space
    gauss = gauss2D(Nx, Ny, moon_pos_pix[0], moon_pos_pix[1], reso=reso/pixsize, amp=amp, normal=False)
    print("old azt in ({}, {}), old elt in ({}, {})".format(np.min(azt_pix), np.max(azt_pix), np.min(elt_pix), np.max(elt_pix)))
    print(np.min(newazt), np.max(newazt), np.min(newelt), np.max(newelt))
    # Create tod from this with a better version of the computing of the Moon position (much faster)
    # grid_interp = RegularGridInterpolator( (azt_pix, elt_pix), gauss, method='nearest' ) 
    # gauss_tod = grid_interp((newazt, newelt))
    # mapsb, mapcount = healpix_map(newazt[scantype != 0], newelt[scantype != 0], mytod[scantype != 0], nside=nside)
    # gauss_tod = img_to_TOD(gauss, ((min_azt + max_azt)/2, (min_elt + max_elt)/2), pixsize, newazt, newelt)
    gauss_tod = img_to_TOD(gauss, [min_azt, max_azt], [min_elt, max_elt], newazt, newelt)
    # img, center, pixsize, newazt, newelt
    filt_tod = my_filt(gauss_tod)
    return gauss_tod - filt_tod, gauss_tod



def fitsb_img(mapxy, x, y, xs, guess=None, doplot=False, distok=3, mytit='', nsig=1, mini=None, maxi=None, ms=10, renorm=False, mynum=33, axs=None, verbose=False, reso=None):
    xx, yy = np.meshgrid(x, y, indexing="xy")
    
    ### Displays the image as an array
    mm, ss = ft.meancut(mapxy, 3)
    if mini is None:
        mini = mm-nsig*ss
    if maxi is None:
        maxi = np.max(mapxy)

    ### Guess where the maximum is and the other parameters with a matched filter
    if guess is None:
        Nx = len(mapxy)
        Ny = len(mapxy[0])
        lobe_pos = (Nx//2, Ny//2)
        Kx, Ky, K = get_K(Nx, Ny)
        ft_phase = get_ft_phase(lobe_pos, Nx, Ny)
        cos_win = cos_window(Nx, Ny, lx=20, ly=20)
        deltaK = 1
        Kbin = get_Kbin(deltaK, K)
        nKbin = len(Kbin) - 1  # nb of bins
        Kcent = (Kbin[:-1] + Kbin[1:])/2
        size_pix = reso/60 # degree
        reso_instr = 0.92
        # ft_shape = fft2(gauss2D(Nx, Ny, x0=lobe_pos[0], y0=lobe_pos[1], reso=[reso_instr/size_pix], normal=True))
        synthbeam_list, qubic_dict = get_synthbeam_freqs(nsub=16, nside=256) # array of the synthesized beams from the instrument for nsub//2 freqs
        tot_sb = np.sum(synthbeam_list, axis=0)
        sb_img = hp.gnomview(tot_sb, reso=reso, rot=[0, 90], min=-5e2, max=1e6, return_projected_map=True, xsize=xs, no_plot=True).data
        sb_img = np.flip(np.swapaxes(sb_img, 0, 1), 1)
        ft_shape = fft2(sb_img)
        
        filtmapsn = get_filtmapsn(mapxy * cos_win, nKbin, K, Kbin, Kcent, ft_shape, ft_phase)

        # plt.figure()
        # plt.imshow(sb_img, origin="lower")
        # plt.show()
    
        # plt.figure()
        # plt.imshow(mapxy)
        # plt.show()

        # plt.figure()
        # plt.imshow(filtmapsn)
        # plt.show()
        maxii = filtmapsn == np.nanmax(filtmapsn)
        maxx = np.mean(xx[maxii])
        maxy = np.mean(yy[maxii])
        guess = np.array([1e4, maxx, maxy, reso_instr])
        if verbose:
            print(guess)
    else:
        maxx = guess[1]
        maxy = guess[2]
        
    ### Do the fit putting the UNSEEN to a very low weight
    errpix = xx*0+ss
    errpix[mapxy==0] *= 1e5
    g2d = gauss2dfit(xx, yy)
    sb2d = synthbeam2dfit(sb_hp_map)
    data = fit.Data(np.ravel(xx), np.ravel(mapxy), np.ravel(errpix), sb2d)
    m, ch2, ndf = data.fit_minuit(guess, limits=[[0, 1e3, 1e8], [1, maxx - distok, maxx + distok], [2, maxy - distok, maxy + distok], [3, 0.6/conv_reso_fwhm, 1.2/conv_reso_fwhm]], renorm=renorm)

    limits=[[0, -180, 180], [1, -180, 180], [2, -180, 180], [3, -180, 180]]

    ### Image of the fitted Gaussian
    fitted = np.reshape(g2d(x, m.values), (xs, xs))

    if doplot:
        origin = "upper" #"lower" swaps the y-axis and the guess doesn't match 
        if axs is None:
            fig, axs = plt.subplots(1, 4, width_ratios=(1, 1, 1, 0.05), figsize=(16, 5))
            axs[1].imshow(fitted, origin=origin, extent=[np.min(x), np.max(x), np.min(y), np.max(y)], vmin=mini, vmax=maxi)
            im = axs[2].imshow(mapxy - fitted, origin=origin, extent=[np.min(x), np.max(x), np.min(y), np.max(y)], vmin=mini, vmax=maxi)
            axs[0].set_ylabel('Degrees')
            for i in range(3):
                axs[i].set_xlabel('Degrees')
            axs[2].set_title('Residuals')
    
    if doplot:
        axs = pmp.plot_fit_img(mapxy, axs, x, y, xguess=guess[1], yguess=guess[2], xfit=m.values[1], yfit=m.values[2], vmin=mini, vmax=maxi, ms=ms, origin=origin)
        return m, fitted, axs
    return m, fitted
    
    

def fit_one_tes_sb(mymap, xs, reso, rot=np.array([0., 0., 0.]), doplot=False, verbose=False, guess=None, distok=3, mytit='', return_images=False, ms=10, renorm=False, xycreid_corr=None, axs=None):
    ### get the gnomview back into a np.array in order to fit it
    mm = mymap.copy()
    badpix = mm == hp.UNSEEN
    mm[badpix] = 0          ### Set bad pixels to zero before returning the np.array()
    mapxy = hp.gnomview(mm, reso=reso, rot=rot, return_projected_map=True, xsize=xs, no_plot=True).data

    ### np.array coordinates
    # Doesn't work with the fit plot but is ok with final gnomview plot of the Moon map corrected
    # But in order to stack the maps I now have to use (azt, -elt) position fitted here (why??)
    x = -(np.arange(xs) - (xs - 1)/2)*reso/60
    y = x.copy()
    x += rot[0]
    y -= rot[1]

    # Works on fit plot but then azt and elt are with the wrong sign on the final gnomview plot. Weird!!
    # x = (np.arange(xs) - (xs - 1)/2)*reso/60
    # y = x.copy()
    # x -= rot[0]
    # y += rot[1]

    # Other tests
    # x = (np.arange(xs) - (xs - 1)/2)*reso/60
    # y = x.copy()
    # x -= rot[0]
    # y += rot[1]


    # print(np.min(y), np.max(y))
    # sys.exit()

    if xycreid_corr is not None:
        try:
            guess = np.array([1e4, xycreid_corr[0], xycreid_corr[1], 0.92])
            if verbose:
                print(guess)
        except:
            guess = None
            if verbose:
                print("TES has no position on sky")
                print(guess)
        
        
    if doplot:
        m, fitted, fig_axs = fitsb_img(mapxy, x, y, xs, guess=guess, doplot=doplot, distok=distok, mytit=mytit, ms=ms, renorm=renorm, axs=axs, verbose=verbose, reso=reso)
        if verbose:
            print(m.values)
    else:
        m, fitted = fitsb_img(mapxy, x, y, xs, guess=guess, doplot=doplot, distok=distok, mytit=mytit, ms=ms, renorm=renorm, verbose=verbose, reso=reso)
    # try:
    #     m, fitted = fitgauss_img(mapxy, x, y, guess=guess, doplot=doplot, distok=distok, mytit=mytit, ms=ms, renorm=renorm)
    # except:
    #     m = None
    #     fitted = None
    
    if return_images:
        return m, mapxy, fitted, [np.min(x), np.max(x), np.min(y), np.max(y)], fig_axs
    return m
    
class synthbeam2dfit:
    def __init__(self, sb_hp_map, newazt, newelt, scantype, xx, yy):
        self.xx = xx
        self.yy = yy
        self.sb_hp_map = sb_hp_map
        self.newazt = newazt
        self.newelt = newelt
        self.scantype = scantype
        self.nside = np.sqrt(len(self.sb_hp_map)/12)
    def __call__(self, x, pars):
        rot1, rot2, rot3 = pars
        rotator = hp.rotator.Rotator(rot=(rot1, rot2, rot3), eulertype="X", deg=True)
        self.synthbeam_rot = rotator.rotate_map_alms(self.synthbeam) * self.amp_slider.val
        fake_TOD = map_to_TOD(self.synthbeam_rot, self.newazt, self.newelt)
        fake_TOD = my_filt(fake_TOD)
        synthbeam_rot_filt, _ = healpix_map(self.newazt[self.scantype != 0], self.newelt[self.scantype != 0], fake_TOD[self.scantype != 0], nside=self.nside)
        sb_map = hp.gnomview(synthbeam_rot_filt, reso=self.reso, rot=[0, 0, 0], min=-5e2, max=1e6, return_projected_map=True, xsize=self.xs, no_plot=True).data

        return np.ravel(sb_map)