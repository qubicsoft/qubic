import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator, LinearNDInterpolator
import matplotlib.pyplot as plt
from scipy.fft import fftfreq, fft2, ifft2, fft, ifft
import sys
import os
import healpy as hp
import time
from scipy.signal import butter, filtfilt, bessel, sosfiltfilt, find_peaks
import pandas as pd
import jax
import jax.numpy as jnp
from fast_histogram import histogram2d
import glob
from scipy.optimize import curve_fit

import fitting as fit
import pickle
from datetime import datetime

import iminuit
from iminuit.cost import LeastSquares

#########################

### General imports
from joblib import Parallel, delayed
from multiprocessing import Manager, Lock

### Astropy configuration
from astropy.visualization import quantity_support
quantity_support()
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_moon, get_body

#### QUBIC IMPORT
import qubicpack as qp
from qubicpack.qubicfp import qubicfp
import qubicpack.pixel_translation as pt
import qubic.lib.Calibration.Qfiber as ft

from qubic.lib import Qdictionary
from qubic.lib.Instrument import Qacquisition

import pipeline_moon_plotting as pmp

#########################

from qubic.lib.Qdictionary import qubicDict
from qubic.lib.Instrument.Qinstrument import QubicMultibandInstrument
from qubic.lib.Qscene import QubicScene
#########################

conv_reso_fwhm = 2.35482
AU_meters = 149597870700 # m

#########################
# import matplotlib.style as style
# style.use("/Users/huchet/Documents/phd_code/matplotlib_styles/ah_basic_style.mplstyle")
# plt.rc('text', usetex=False)
# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r"\usepackage{bm}")
# plt.style.use('default')

#########################

def centre_coord_at_0(coord):
    return np.mod(coord - 180, 360) - 180

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

# trying to build only patch instead of full map?
def healpix_patch(azt, elt, tod, nside): # , vec_centre, radius=25
    # good_pix = hp.query_disc(nside, vec_centre, np.radians(radius))
    cell_ids = hp.ang2pix(nside, azt, elt, lonlat=True)
    df_pix = pd.DataFrame(tod).groupby(cell_ids)
    patch = np.array(df_pix[0].mean())
    hitcount = np.array((pd.DataFrame(np.ones_like(tod)).groupby(cell_ids))[0].sum()).astype(int)
    print("patch", patch)
    print("hitcount", hitcount)
    cell_ids = np.array((pd.DataFrame(cell_ids).groupby(cell_ids))[0].mean()).astype(int)
    print("cell_ids", cell_ids)
    return patch, hitcount, cell_ids


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

# Function to go from QubicSoft (Sims) indices (0-247) to QubicPack (data) indices (0-255)
### The 8 thermometers are not in QubicSoft

# QP is real data (QubicPack), QS is QubicSoft
def iQS2iQP(indexQS):
    qpnumi, qpasici = qp.pix2tes.pix2tes(indexQS+1)
    return qpnumi+(qpasici-1)*128-1

def iQP2iQS(indexQP):
    QStesnum = qp.pix2tes.tes2pix(indexQP%128+1, indexQP//128+1) # doesn't work since cloned and pip install . qubicpack? 25/06/2026
    # QStesnum = tes2pix(indexQP%128+1, indexQP//128+1)
    return QStesnum-1

def get_ObsSite(name):
    if name == "salta":
        Salta_CNEA = {'lat':-24.731358*u.deg,
                    'lon':-65.409535*u.deg,
                    'height':1152*u.m,
                    'UTC_Offset':-3*u.hour}
        Obs_Site = Salta_CNEA
    else:
        raise ValueError("Site name '{}' is unknown. Only 'salta' is implemented for now.")
    return Obs_Site

def get_vel(time, position, order=2):
    """Function to get the velocity from time and position.
    """
    vel = np.zeros(len(time))
    vel[:order] = (position[1:order + 1] - position[:order])/(time[1:order + 1] - time[:order])
    vel[-order:] = (position[-order:] - position[-order - 1:-1])/(time[-order:] - time[-order - 1:-1])
    dt_ = time[2*order:] - time[:-2*order]
    vel[order:-order] = (position[2*order:] - position[:-2*order])/dt_
    return vel

#######################

def gaussian(x, mu, reso):
		sig = reso / conv_reso_fwhm
		res = 1 / (np.sqrt(2 * np.pi) * sig) * np.exp(-((x - mu) / sig)**2 / 2)
		return res / np.sum(res) # area under the curve = 1

def gauss2D(Nx, Ny, x0, y0, reso, amp=None, normal=True): # reso is the fwhm
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


def spherical2cartesian(rho, theta, phi, coord="spherical", axis="first"): # axis is the axis where the coords for each point will be
    # print("in spherical2cartesian", flush=True)
    if coord == "horizontal": # theta is azimuth and phi is elevation
        # print("coord == 'horizontal'", flush=True)
        theta_ = theta.copy()
        theta = np.pi/2 - np.radians(phi.copy())
        phi = np.radians(theta_)
        # print("go get it", flush=True)
        # theta, phi = np.pi/2 - np.radians(phi), np.radians(theta)
    elif coord != "spherical":
        raise ValueError("Argument coord = {} not understood.".format(coord))
    x = rho * np.sin(theta) * np.cos(phi)
    y = rho * np.sin(theta) * np.sin(phi)
    z = rho * np.cos(theta)
    # print("x, y, z alright", flush=True)
    res = np.array([x, y, z])
    if axis == "first":
        return res
    elif axis == "last":
        return np.moveaxis(res, 0, -1)
    else:
        raise ValueError("Argument axis = {} not understood.".format(axis))

def cartesian2spherical(x, y, z, coord="spherical", axis="first"):
    rho = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/rho)
    phi = np.arctan2(y, x)
    if coord == "horizontal":
        theta_ = theta.copy()
        theta = np.degrees(phi).copy()
        phi = 90 - np.degrees(theta_)
        # theta, phi = np.degrees(phi), 90 - np.degrees(theta)
    elif coord != "spherical":
        raise ValueError("Argument coord = {} not understood.".format(coord))
    res = np.array([rho, theta, phi])
    if axis == "first":
        return res
    elif axis == "last":
        return np.moveaxis(res, 0, -1)
    else:
        raise ValueError("Argument axis = {} not understood.".format(axis))

def get_perp_vect(point_A, point_B, point_C): # get the vector perpendicular to the plane with these three points of shape (3, ...)
    vec_1 = point_A - match_shape(point_C, point_A.shape) # vec_1 = CA
    vec_2 = point_B - match_shape(point_C, point_B.shape) # vec_2 = CB
    vec_2 = match_shape(vec_2, np.shape(vec_1))
    res_nonorm = np.cross(vec_1, vec_2, axis=0)
    return res_nonorm/np.linalg.norm(res_nonorm, axis=0)

def get_plane(point_A, point_B, point_C, offset=0): # plane equation ax + by + cz + d = 0
    perp_vect = get_perp_vect(point_A, point_B, point_C)
    a, b, c = perp_vect
    d = -np.dot(perp_vect, point_A)
    return np.array([a, b, c, d])

def get_rotation_matrix(point_pos, target_pos):
    # rotation necessary to put a point at the position target_pos in cartesian coordinates
    center_pos = np.array([0, 0, 0])
    perp_vect = get_perp_vect(target_pos, point_pos, center_pos)
    print("perp_vect", perp_vect)
    x, y, z = perp_vect[0], perp_vect[1], perp_vect[2]
    if len(np.shape(point_pos)) == 2:
        vec_B = point_pos - center_pos[:, None]
    else:
        vec_B = point_pos - center_pos
    if len(np.shape(target_pos)) == 2:
        vec_A = target_pos - center_pos[:, None]
        angle = np.arccos(np.einsum("ij,ij->j", vec_A, vec_B)/(np.linalg.norm(vec_A, axis=0) * np.linalg.norm(vec_B, axis=0)))
    else:
        vec_A = target_pos - center_pos
        angle = np.arccos(np.dot(vec_A, vec_B)/(np.linalg.norm(vec_A) * np.linalg.norm(vec_B, axis=0))) # compute the angle between vec_A and vec_B (the angle to rotate around perp_vect)
    print(np.shape(angle))
    c = np.cos(angle)
    s = np.sin(angle)

    # print(np.all(np.isclose(np.sum(perp_vect**2, axis=0), 1)))
    rot_matrix = np.array([[x**2 * (1 - c) + c, x*y*(1 - c) - z*s, x*z*(1 - c) + y*s],
                           [x*y*(1 - c) + z*s, y**2*(1 - c) + c, y*z*(1 - c) - x*s],
                           [x*z*(1 - c) - y*s, y*z*(1 - c) + x*s, z**2*(1 - c) + c]])
    return rot_matrix

def get_simple_rotation_matrix(axis, angle):
    one = np.ones_like(angle)
    zero = np.zeros_like(angle)
    if axis == "x":
        R = np.array([[one, zero, zero],
                      [zero, np.cos(angle), -np.sin(angle)],
                      [zero, np.sin(angle), np.cos(angle)]])
    elif axis == "y":
        R = np.array([[np.cos(angle), zero, np.sin(angle)],
                      [zero, one, zero],
                      [-np.sin(angle), zero, np.cos(angle)]])
    elif axis == "z":
        R = np.array([[np.cos(angle), -np.sin(angle), zero],
                      [np.sin(angle), np.cos(angle), zero],
                      [zero, zero, one]])
    else:
        raise ValueError("{} is not a good axis.".format(axis))
    return R


def get_circle_values(A, B, C):
    # https://stackoverflow.com/questions/20314306/find-arc-circle-equation-given-three-points-in-space-3d
    # get circle centre and radius from 3 points in cartesian coordinates
    # note that a is the distance of the side opposite to A (i.e. BC)
    a = np.linalg.norm(C - B, axis=-1) # we use only the last dimension for norm
    b = np.linalg.norm(C - A, axis=-1)
    c = np.linalg.norm(B - A, axis=-1)
    s = (a + b + c) / 2
    R = a*b*c / 4 / np.sqrt(s * (s - a) * (s - b) * (s - c)) # radius
    b1 = a*a * (b*b + c*c - a*a)
    b2 = b*b * (a*a + c*c - b*b)
    b3 = c*c * (a*a + b*b - c*c)
    P = np.einsum("ijk,ik->ij", np.stack((A, B, C), axis=-1), np.stack((b1, b2, b3), axis=-1)) # much faster than np.dot
    # P = np.stack((A, B, C), axis=-1).dot(np.stack((b1, b2, b3), axis=0)) # np.dot is dot product on last axis of first array and second to last of second array
    P /= (b1 + b2 + b3)[:, None] # circle centre (renormalised here because above formula works for normalised positions)
    return P, R


def detect_peaks_TOD(tt, mytod, resolution, doplot=False, freq_sampling=157.36):

    mytod = mytod.copy()
    slope = (mytod[-1] - mytod[0])/(tt[-1] - tt[0])
    mytod -= (slope * tt + (mytod[0] - slope*tt[0]))

    # freq_sampling = 157.36 # Hz
    ### Use gaussian smoothing to remove noise

    Nx = len(mytod)
    gauss_pos = Nx//2
    gauss_kernel = gaussian(tt, np.mean(tt), resolution) # resolution in seconds
    K = fftfreq(len(mytod), d=1/freq_sampling)
    # K = fftfreq(len(mytod), d=1/len(mytod))

    ft_phase = get_ft_phase_1D(gauss_pos, Nx)
    # deltaK = 1
    # Kbin = get_Kbin(deltaK, K)
    Kbin = np.concatenate(([0, 0.1], np.geomspace(1/8, 1, 100), [1.1, np.max(K)]))
    nKbin = len(Kbin) - 1  # nb of bins
    Kcent = (Kbin[:-1] + Kbin[1:])/2
    ft_shape = fft(gauss_kernel)
    filtmapsn = get_filtmapsn_1D(mytod, nKbin, K, Kbin, Kcent, ft_shape, ft_phase)

    if doplot:
        plt.figure()
        plt.plot(tt, mytod)
        # plt.plot(tt, np.roll(gauss_kernel, len(gauss_kernel)//4 + int(15.2*freq_sampling))*5900000 - 120000) # order 0 peak for TES 152
        plt.plot(tt, np.roll(gauss_kernel, len(gauss_kernel)//4 + int(1083.2*freq_sampling))*5900000 - 190000) # order 1 peak for TES 83
        plt.xlim(xlim)
        plt.ylim(pmp.get_ylim(tt, xlim, mytod))
        plt.show()

        plt.figure()
        # plt.scatter(tt, mytod, label="TOD")
        plt.plot(tt, filtmapsn, label="peak detection")
        # plt.xlim(xlim)
        # plt.ylim(pmp.get_ylim(tt, xlim, filtmapsn))
        plt.legend()
        plt.show()
        # zf
    return filtmapsn

def remove_peaks(tt, tod, peaks_detected, interval, mask=None, control_size=None):
    index = np.arange(len(tod))
    if mask is not None:
        index_masked = index[mask]
    else:
        index_masked = index

    # plt.figure()
    # plt.plot(tt, tod)
    for i_peak, peak in enumerate(peaks_detected):
        if peak in index_masked:
            if control_size is None:
                control_size = 3*interval[i_peak]
            low = (index >= peak - (interval[i_peak] + control_size)) & (index <= peak - interval[i_peak])
            high = (index >= peak + interval[i_peak]) & (index <= peak + (interval[i_peak] + control_size))
            slope = (np.median(tod[high]) - np.median(tod[low]))/(np.median(tt[high]) - np.median(tt[low])) # or use Lagrange polynome?
            origin = np.median(tod[low]) - slope*np.median(tt[low])
            tod[peak - interval[i_peak] : peak + interval[i_peak]] = slope*tt[peak - interval[i_peak] : peak + interval[i_peak]] + origin
            # plt.plot(tt, tod)
            # plt.show()
            # aer
    return tod

def add_peaks(tt, nopeak_tod, tod, peaks_detected, interval, mask):
    index = np.arange(len(tod))
    index_masked = index[mask]
    for i_peak, peak in enumerate(peaks_detected):
        if peak in index_masked:
            low = (index >= peak - 3*interval[i_peak]) & (index <= peak - interval[i_peak])
            high = (index >= peak + interval[i_peak]) & (index <= peak + 3*interval[i_peak])
            slope = (np.median(tod[high]) - np.median(tod[low]))/(np.median(tt[high]) - np.median(tt[low]))
            origin = np.median(tod[low])- slope*np.median(tt[low])
            nopeak_tod[peak - interval[i_peak] : peak + interval[i_peak]] = tod[peak - interval[i_peak] : peak + interval[i_peak]] - (slope*tt[peak - interval[i_peak] : peak + interval[i_peak]] + origin)
    return nopeak_tod

# xlim = [9480, 9550]
xlim = [10540, 10640]

def make_coadded_maps_TES(tt, tod, azt, elt, scantype, newazt, newelt, ifile, Tbath=None, T1K=None, TES_number="", nside=256, doplot=True,
                          check_back_forth=False, also_tod=False, det_pos=None, clean_tod=True, manual=False,
                          ObsName=None, new_method_clean=False, theo_sb=None, more="", simu=False):

    # What worked best so far:
    # - filter raw TOD (bandpass, to get rid of large and small scales)
    # - detect peaks from the result
    # - remove the detected peaks from raw TOD and replace them by a linear fit of raw data around peaks
    # - filter the result (highpass, to get rid of large scales)
    # - add the removed peaks again (difference between raw peaks and linear fit of filtered data around peaks)

    ObsDate = ObsName[:10]

    if more != "":
        more = "_" + more

    if theo_sb is not None:
        # we get the theoretical synthbeam
        thetas = theo_sb[0] # shape (n_nus, npeaks)
        phis = theo_sb[1]
        n_nus = len(thetas)
        n_peaks = len(thetas[0])

    # freq_sampling = 157.36 # Hz
    freq_sampling = 1/np.median(tt[1:] - tt[:-1]) # Hz
    # print("freq_sampling", freq_sampling)

    if ObsDate[:4] == "2022":
        # Inversion in signal
        mytod = -tod.copy()
        reso=10
        xsize = 200 # default
    elif ObsDate[:4] == "2026":
        mytod = tod.copy()
        ang_size = 40 # degrees
        xsize = 201
        reso = ang_size*60/xsize # arcmin/pix
        # print(reso)


    if clean_tod:
        min_plot = -5e3
        max_plot = 1.2e4
    else:
        min_plot = np.min(mapsb[mapsb != hp.UNSEEN])
        max_plot = np.max(mapsb[mapsb != hp.UNSEEN])

    if new_method_clean:
        rho = 1
        dist_min = 2 # deg
        tod_pos = spherical2cartesian(rho, newazt, newelt, coord="horizontal", axis="last")
        protected_tod = np.zeros_like(mytod)
        mask_elt = np.ones_like(mytod, dtype=bool) # we don't mask in elevation
        for i_nu in range(n_nus):
            for i_peak in range(n_peaks):
                el_peak = 90 - np.degrees(thetas[i_nu, i_peak])
                az_peak = np.degrees(phis[i_nu, i_peak])
                # azimuth differences won't work at az=+/-180
                # tod_close = np.logical_and(np.abs(az_peak - newazt) < 1, np.abs(el_peak - newelt) < 1) # in a 1 deg^2 square
                peak_pos = spherical2cartesian(rho, az_peak, el_peak, coord="horizontal", axis="last")
                dist_peak = np.abs(np.degrees(dist_angle(tod_pos, peak_pos)))
                # we should add a selection on the peak's position:
                # if it is too close to the border of the map the peak is not counted
                # not needed for simulations
                if not simu:
                    if np.min(dist_peak[scantype == 0]) < dist_min: # needed for real data because of low frequency noise! --> see with Noah how to remove it
                        # print("skipped the peak", i_nu, i_peak)
                        continue
                tod_close = dist_peak < dist_min
                protected_tod[tod_close] = 1
        # Just to avoid having different numbers of start and end peaks
        protected_tod[0] = 0
        protected_tod[-1] = 0
        new_peak_ = np.append(protected_tod[1:] - protected_tod[:-1], 0)
        start_peak = np.argwhere(new_peak_ == 1)[:, 0]
        end_peak = np.argwhere(new_peak_ == -1)[:, 0]
        # print(TES_number, "shape new_peak_", np.shape(new_peak_), flush=True)
        # print(TES_number, "shape start_peak end_peak", np.shape(start_peak), np.shape(end_peak), flush=True)
        pos_peak = ((end_peak + start_peak)/2).astype(int)
        border_margin = 0 # 0.2*freq_sampling # border_margin is xx seconds (approx. xx deg)
        interval_peak = (((end_peak - start_peak) + border_margin*2)/2).astype(int) # we add the border margin on both sides
        # print(TES_number, "shape interval_peak", np.shape(interval_peak), flush=True)
        # print(interval_peak)
        # ar
        # ind_peak = 100
        # tod_no_peak = remove_peaks(tt, mytod.copy(), [pos_peak[ind_peak]], interval=[interval_peak[ind_peak]], mask=mask_elt)ind_peak = 100
        control_size = 1*freq_sampling # one second of data is used to compute the continuum in remove_peaks
        tod_no_peak = remove_peaks(tt, mytod.copy(), pos_peak, interval=interval_peak, mask=mask_elt, control_size=control_size)
        mytod_3 = my_filt(tod_no_peak.copy())
        mytod_4 = add_peaks(tt, mytod_3.copy(), mytod, pos_peak, interval=interval_peak, mask=mask_elt)
        if doplot and True: # plot with raw tod vs filtered tod
            pmp.plot_tod_filtering(ObsName, TES_number, tt, mytod, mytod_4, protected_tod == 1, Tbath=Tbath, T1K=T1K, savefig=True)
        final_tod = mytod_4


    elif clean_tod:
        if ObsDate[:4] == "2022":
            # Filter the TOD
            mytod_1 = my_filt(mytod.copy())

            min_elt, max_elt = 39, 51 # 29.97844299316406 50.08891662597656
            # # mask_elt = (elt >= min_elt) & (elt <= max_elt)
            mask_elt = elt >= min_elt
            # mask_elt = azt >= 335
            tod_ma_filt = my_filt_2(mytod.copy()) # bandpass instead of moving average then highpass
            prominence = (3*np.std(tod_ma_filt[mask_elt]), None) # good filter
            # prominence = None
            data_peaks = tod_ma_filt
            peaks_detected, peaks_properties = find_peaks(data_peaks, height=None, threshold=None, distance=10*freq_sampling, prominence=prominence, width=(1*freq_sampling, 8*freq_sampling), wlen=10*freq_sampling, rel_height=0.5, plateau_size=None)
            # width up to 8 seconds for order 1 peaks
            # might have to go even higher for the peaks aligned with elevation scans
            # distance=None helps with order 1 peaks that are a bit irregular (close to border of map/dead time)
            # kept distance=10s to remove some foregrounds
            widths = peaks_properties["widths"]
        elif ObsDate == "2026-03-11":
            mask_elt = np.ones_like(elt, dtype=bool)
            # Filter the TOD
            test_tod = my_filt_4(mytod.copy()) # only a high pass to detect the glitches
            if manual:
                period_tt = np.mean(tt[1:] - tt[:-1])
                intervals = [[570, 650], [3650, 3750], [3955, 4000]] #s
                peaks_s = [(interval[1] + interval[0])/2 for interval in intervals]
                glitches_detected = [np.argmin(np.abs(peaks_s[i_peak] - tt)) for i_peak in range(len(peaks_s))]
                interval_glitch = np.array([(interval[1] - interval[0])/(2*period_tt) for interval in intervals]).astype(int)
                test_tod_no_glitch = remove_peaks(tt, test_tod.copy(), glitches_detected, interval=interval_glitch, mask=mask_elt)
            else:
                glitches_detected, glitches_properties = find_peaks(test_tod, height=80000, threshold=None, distance=None, width=0, wlen=10*freq_sampling) # totally empirical cut at 80000 ADUs
                interval_glitch = (glitches_properties["widths"]).astype(int) * 10
                test_tod_no_glitch = remove_peaks(tt, test_tod.copy(), glitches_detected, interval=interval_glitch, mask=mask_elt)
                glitches_detected_ = np.isin(np.arange(len(tt)), glitches_detected)
            plt.figure()
            plt.plot(tt, mytod, label="raw TOD")
            plt.plot(tt, test_tod, label="test_tod")
            plt.plot(tt, test_tod_no_glitch, label="TOD no peak")
            if not manual:
                plt.scatter(tt[glitches_detected_ & mask_elt], mytod[glitches_detected_ & mask_elt], c="r", label="peaks_detected", zorder=1000)
            plt.legend()
            plt.show()

            mytod_1 = my_filt(mytod.copy())
            tod_ma_filt = my_filt_3(test_tod_no_glitch.copy()) # bandpass instead of moving average then highpass
            sigma_clipped = tod_ma_filt.copy()
            mean_tod = 0 # approximation, since the data is filtered a lot
            for i_clip in range(2):
                sigma_clipped = sigma_clipped[np.abs(sigma_clipped - mean_tod) < 6*np.std(sigma_clipped)]
            std_prominence = np.std(sigma_clipped)
            prominence = (3*std_prominence, None)
            data_peaks = tod_ma_filt

            # peaks_detected, peaks_properties = find_peaks(data_peaks, height=None, threshold=None, distance=10*freq_sampling, prominence=prominence, width=(1*freq_sampling, 8*freq_sampling), wlen=10*freq_sampling, rel_height=0.5, plateau_size=None)
            peaks_detected, peaks_properties = find_peaks(data_peaks, height=None, threshold=None, distance=10*freq_sampling, prominence=prominence, width=(1*freq_sampling, 10*freq_sampling), wlen=10*freq_sampling, rel_height=0.5, plateau_size=None)
            widths = peaks_properties["widths"]

        else: #if ObsDate == "2026-03-13":
            # Filter the TOD
            mytod_1 = np.zeros_like(mytod)
            data_peaks = np.zeros_like(mytod)
            widths = []
            peaks_detected = []
            for i in range(np.max(ifile) + 1):
                print("i", i)
                mask_elt = None
                mask = ifile == i
                if np.sum(mask) == 0: # happens when we skip a bad file when reading the data
                    print("No data in file {}".format(i))
                    continue
                first_index = np.min(np.argwhere(mask))
                mytod_1[mask] = my_filt(mytod[mask])
                tod_ma_filt = my_filt_2(mytod[mask]) # bandpass instead of moving average then highpass
                prominence = (5*np.std(tod_ma_filt), None) # 3 good filter
                # prominence = None
                data_peaks[mask] = tod_ma_filt
                peaks_detected_, peaks_properties = find_peaks(data_peaks[mask], height=None, threshold=None, distance=10*freq_sampling, prominence=prominence, width=(1*freq_sampling, 8*freq_sampling), wlen=10*freq_sampling, rel_height=0.5, plateau_size=None)
                peaks_detected.append(peaks_detected_ + first_index)
                widths.append(peaks_properties["widths"])
            print("peaks_detected", peaks_detected)
            peaks_detected = np.concatenate(peaks_detected)
            widths = np.concatenate(widths)

        if doplot and True:
            plt.figure()
            plt.plot(tt, data_peaks, label="bandpass filtered TOD")
            plt.scatter(tt[peaks_detected], data_peaks[peaks_detected], c="r", label="peaks_detected", zorder=1000)
            # plt.scatter(tt[peaks_detected], peaks_detected[peaks_detected], c="g", label="peaks_detected")
            # plt.axhline(y=prominence[0], c="k", ls="--", label="prominence cut") # not as easy to plot
            plt.legend()
            plt.tight_layout()
            plt.savefig("figures/peak_detection_TES_{}.pdf".format(TES_number), dpi=300)
            plt.show()

            # plt.figure()
            # plt.plot(tt, mytod, label="TOD")
            # plt.scatter(tt[peaks_detected], mytod[peaks_detected], c="r", label="peaks_detected", zorder=1000)
            # # plt.scatter(tt[peaks_detected], peaks_detected[peaks_detected], c="g", label="peaks_detected")
            # plt.legend()
            # plt.show()

        # interval_peak = int(time_moon*freq_sampling * 1.5) # 1.5 is good
        interval_peak = (widths).astype(int)
        if ObsDate == "2026-03-11":
            tod_no_glitches = remove_peaks(tt, mytod.copy(), glitches_detected, interval=interval_glitch, mask=mask_elt) #this doesn't deal well with flux jumps
            # tod_no_glitches = test_tod_no_glitch.copy() # you can use this for flux jumps, but there is a bit of filtering ringing around Moon
        else:
            tod_no_glitches = mytod
        # tod_no_peak = remove_peaks(tt, mytod.copy(), peaks_detected, interval=interval_peak, mask=mask_elt)
        # print(np.shape(peaks_detected))
        # print(peaks_detected)
        # print(np.shape(interval_peak))
        # print(interval_peak)
        # aetzr
        tod_no_peak = remove_peaks(tt, tod_no_glitches.copy(), peaks_detected, interval=interval_peak, mask=mask_elt)
        mytod_3 = my_filt(tod_no_peak.copy())
        mytod_4 = add_peaks(tt, mytod_3.copy(), mytod, peaks_detected, interval=interval_peak, mask=mask_elt)

        # tod_no_peak = remove_peaks(tt, tod_ma.copy(), peaks_detected, interval=interval_peak, mask=mask_elt)
        # mytod_3 = my_filt(tod_no_peak.copy())
        # mytod_4 = add_peaks(tt, mytod_3.copy(), mytod, peaks_detected, interval=interval_peak, mask=mask_elt)

        if doplot and True: # these plots are peaks, Moon TOD and Moon TOD spectrum
            # Lagrange_poly(x, points_x, points_y)
            # binned_tt, binned_TOD, scan_sorted, bin_id = bin_data_by_scan(tt, mytod, scantype)
            # print(scan_sorted)
            # print(bin_id)

            peaks_detected_ = np.isin(np.arange(len(tt)), peaks_detected)
            if mask_elt is None:
                mask_elt = np.ones_like(peaks_detected_, dtype=bool)

            pmp.plot_tod_filtering(ObsName, TES_number, tt, mytod, mytod_4, peaks_detected_ & mask_elt, Tbath=Tbath, T1K=T1K, savefig=True)

            # fit_lagrange_scans(tt, mytod, scantype)
            # azr

            # tmin, tmax = 4300, 4900
            tmin, tmax = 4000, 4600
            fig, ax = plt.subplots(figsize=(7.5, 6))
            # ax.set_title("raw TOD")
            ax.plot(tt, mytod)
            ax.set_xlim([tmin, tmax])
            ax.set_xlabel("time [s]")
            ax.set_ylabel("amplitude [arbitrary units]")
            ax.set_yticks([])
            plt.tight_layout()
            # plt.savefig("figures/moon_tod.pdf")
            plt.savefig("figures/moon_tod_.png")
            plt.show()

        final_tod = mytod_4 # good filter
        # final_tod = mytod # no filter
        comparison_tod = mytod_1
    else:
        final_tod = mytod
    mask_scan = scantype != 0
    mask_map = mask_scan# * mask_elt

    center = [0, 90] # zenith

    # To compare the map created with only forth scans with the map created with only back scans
    if check_back_forth:
    
        mapsb_forth, mapcount_forth = healpix_map(newazt[scantype > 0], newelt[scantype > 0], final_tod[scantype > 0], nside=nside)
        mapsb_back, mapcount_back = healpix_map(newazt[scantype < 0], newelt[scantype < 0], final_tod[scantype < 0], nside=nside)



        if ObsDate[:4] == "2026":
            mapsb_fb_proj = []
            for mapsb_ in [mapsb_forth, mapsb_back]:
                mapsb_proj = hp.gnomview(mapsb_, reso=reso, min=min_plot, max=max_plot, xsize=xsize,
                        rot=center, return_projected_map=True, no_plot=True)
                X = np.arange(len(mapsb_proj))
                Y = np.arange(len(mapsb_proj[0]))
                XX, YY = np.meshgrid(X, Y) # pixel units, just for the interpolation

                no_UNSEEN_mask = ~mapsb_proj.mask

                print(np.shape(XX))
                print(np.shape(XX[no_UNSEEN_mask]))

                mapsb_interpolator = LinearNDInterpolator(np.moveaxis([XX[no_UNSEEN_mask], YY[no_UNSEEN_mask]], 0, -1), mapsb_proj[no_UNSEEN_mask])
                new_mapsb_proj = mapsb_interpolator(np.moveaxis([XX, YY], 0, -1))
                mapsb_fb_proj.append(new_mapsb_proj)

            if doplot:
                fig, axs = plt.subplots(1, 3, figsize=(18, 6))
                axs[0].set_title("forth")
                axs[0].imshow(mapsb_fb_proj[0], vmin=min_plot, vmax=max_plot)
                axs[1].set_title("back")
                axs[1].imshow(mapsb_fb_proj[1], vmin=min_plot, vmax=max_plot)
                axs[2].set_title("back - forth")
                axs[2].imshow(mapsb_fb_proj[1] - mapsb_fb_proj[0], vmin=min_plot, vmax=max_plot)
                fig.savefig("figures/{}_{}_back-forth_plot.pdf".format(ObsName, TES_number), dpi=150)
                plt.show()

        else:
            if doplot:
                plt.figure()
                hp.gnomview(mapsb_forth, reso=reso, sub=(1, 3, 1), min=min_plot, max=max_plot, xsize=xsize,
                        title="forth scans", rot=center)
                hp.gnomview(mapsb_back, reso=reso, sub=(1, 3, 2), min=min_plot, max=max_plot, xsize=xsize,
                        title="back scans", rot=center)
                hp.gnomview(mapsb_forth - mapsb_back, reso=reso, sub=(1, 3, 3), min=min_plot, max=max_plot, xsize=xsize,
                        title="forth - back scans", rot=center)
                plt.show()

    mapsb, mapcount = healpix_map(newazt[mask_map], newelt[mask_map], final_tod[mask_map], nside=nside)

    if doplot and True:
        
        plt.figure()
        hp.gnomview(mapsb, reso=reso, min=-5e3, max=1.2e4, xsize=xsize,
                    title="{}, TES {}".format(ObsName, TES_number), rot=center)
        if theo_sb is not None: 
            for i in range(n_nus):
                hp.projscatter(thetas[i,:], phis[i,:], c="r", 
                            marker='.')
        plt.tight_layout()
        plt.savefig("figures/map_TES_{}.pdf".format(TES_number), dpi=300)
        plt.show()
        # azr
        
    if also_tod:
        return mapsb, mapcount, final_tod
    
    if ObsDate[:4] == "2026":
        mapsb_proj = hp.gnomview(mapsb, reso=reso, min=min_plot, max=max_plot, xsize=xsize,
                    rot=center, return_projected_map=True, no_plot=True)
        X = np.arange(len(mapsb_proj))
        Y = np.arange(len(mapsb_proj[0]))
        XX, YY = np.meshgrid(X, Y) # pixel units, just for the interpolation

        no_UNSEEN_mask = ~mapsb_proj.mask
        mapsb_interpolator = LinearNDInterpolator(np.moveaxis([XX[no_UNSEEN_mask], YY[no_UNSEEN_mask]], 0, -1), mapsb_proj[no_UNSEEN_mask])
        new_mapsb_proj = mapsb_interpolator(np.moveaxis([XX, YY], 0, -1))
        if doplot:
            fig, ax = plt.subplots(figsize=(5, 5))
            plt.imshow(new_mapsb_proj, vmin=min_plot, vmax=max_plot)
            for minor in [True, False]:
                ax.set_xticks([], minor=minor)
                ax.set_yticks([], minor=minor)
            plt.tight_layout()
            plt.savefig("figures/{}{}_TES{}_v2.pdf".format(ObsDate, more, TES_number))
            plt.show()
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

def butter_lowpass_filter(data, *args, **kwargs):
    sos = butter_lowpass(*args, **kwargs)
    return sosfiltfilt(sos, data) # no phase but filter applied twice (forwards and backwards), because I use filtfilt instead of lfilter

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


def my_filt_4(mytod): # utiliser cette fonction ?
    # Cuts are expressed in Hz, a back and forth scan takes 107.5 seconds
    fs = 157.36 # Hz # could be computed directly on TOD
    # lowcut = 4/107.5 # 4/107.5, i.e. half a forth (or back) scan
    lowcut = 2/107.5
    filt_tod = butter_highpass_filter(mytod, lowcut=lowcut, fs=fs, order=1) # Hz
    return filt_tod

def my_filt_3(mytod): # utiliser cette fonction ?
    # Cuts are expressed in Hz, a back and forth scan takes 107.5 seconds
    fs = 157.36 # Hz # could be computed directly on TOD
    scan_period = 112 # s
    # lowcut = 4/107.5 # 4/107.5, i.e. half a forth (or back) scan
    # lowcut = 16/scan_period
    lowcut = 16/scan_period
    highcut = 2/scan_period*100/2 # 2/107.5*100/4, i.e. approx. 4 % of a forth (or back) scan --> passer à 2% parce que 4% est trop proche de la taille de la Lune (2/107.5*100/6 makes the Moon round but it's fine-tuned for it...)
    filt_tod = butter_bandpass_filter(mytod, lowcut=lowcut, highcut=highcut, fs=fs, order=6) # Hz
    return filt_tod

def my_filt_2(mytod): # utiliser cette fonction ?
    # Cuts are expressed in Hz, a back and forth scan takes 107.5 seconds
    fs = 157.36 # Hz # could be computed directly on TOD
    # lowcut = 4/107.5 # 4/107.5, i.e. half a forth (or back) scan
    lowcut = 8/107.5
    highcut = 2/107.5*100/4 # 2/107.5*100/4, i.e. approx. 4 % of a forth (or back) scan --> passer à 2% parce que 4% est trop proche de la taille de la Lune (2/107.5*100/6 makes the Moon round but it's fine-tuned for it...)
    # print("lowcut = {} Hz, highcut = {} Hz".format(lowcut, highcut))
    filt_tod = butter_bandpass_filter(mytod, lowcut=lowcut, highcut=highcut, fs=fs, order=8) # Hz
    # filt_tod = butter_pseudo_bandpass_filter(mytod, lowcut=lowcut, highcut=highcut, fs=fs, order=1) # Hz
    # filt_tod = butter_highpass_filter(mytod, lowcut=lowcut, fs=fs, order=1) # Hz
    # filt_tod = butter_lowpass_filter(mytod, highcut=highcut, fs=fs, order=1) # Hz
    # filt_tod = bessel_bandpass_filter(mytod, lowcut=lowcut, highcut=highcut, fs=fs, order=1) # Hz
    # filt_tod = bessel_highpass_filter(mytod, lowcut=lowcut, fs=fs, order=5) # Hz
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
    def __init__(self, ii, jj):
        self.ii = ii
        self.jj = jj
    def __call__(self, x, pars):
        amp, ic, jc, sig = pars
        mygauss = amp * np.exp(-0.5*((self.ii-ic)**2+(self.jj-jc)**2)/sig**2)
        return np.ravel(mygauss)
    

class gaussfitsphere:
    def __init__(self, elt, azt, mask=None): # might want to change to azt, elt to avoid mistakes
        self.pix_pos = spherical2cartesian(1, azt, elt, coord="horizontal", axis="last") # here we save the coordinates of each pixel of the map
        # the mask is here to put the masked pixels to zero so they don't influence the fit
        if mask is None:
            self.mask = np.zeros_like(elt)
        else:
            self.mask = mask
    def __call__(self, x, pars):
        amp, ic, jc, sig = pars # here the position is given in pixels and later converted to azel
        if np.isnan(ic) or np.isnan(jc): # for some reason, maybe when we fall outise of the image, ic or jc can be NaNs
            shape_pix_pos = np.shape(self.pix_pos)
            return np.zeros(shape_pix_pos[0]*shape_pix_pos[1])
        centre_pos = self.pix_pos[int(ic), int(jc)]
        dist_deg = np.degrees(dist_angle(self.pix_pos, centre_pos))
        mygauss = amp * np.exp(-0.5*dist_deg**2/sig**2)
        return np.ravel(mygauss * (1 - self.mask))
    
class gaussfitgnomproj:
    def __init__(self, elt, azt, nside, rot, reso, xs, mask=None): # might want to change to azt, elt to avoid mistakes
        self.nside = nside
        self.rot = rot
        self.reso = reso
        self.xs = xs
        if mask is None:
            self.mask = np.zeros_like(elt)
        else:
            self.mask = mask
        # vec_centre = hp.ang2vec(self.rot[0], self.rot[1], lonlat=True)
        # radius = 40 # radius of patch we take into account for fit in degrees
        # self.useful_pix = hp.query_disc(self.nside, vec_centre, np.radians(radius))
        self.useful_pix = np.arange(12*nside**2)
        self.pix_pos_patch = hp.pix2vec(self.nside, self.useful_pix) # here we save the coordinates of each pixel of the patch before proj
        # self.pix_pos_patch = np.swapaxes(self.pix_pos_patch, axis1=0, axis2=-1)
        self.pix_pos_patch = np.moveaxis(self.pix_pos_patch, source=[0, 1], destination=[1, 0])
        self.pix_pos_proj = spherical2cartesian(1, azt, elt, coord="horizontal", axis="last") # here we save the coordinates of each pixel of the map after proj

    def __call__(self, ij, amp, ic, jc, sig): # for curve_fit this time, not minuit
        # amp, ic, jc, sig = pars # here the position is given in pixels and later converted to azel
        if np.isnan(ic) or np.isnan(jc): # for some reason, maybe when we fall outise of the image, ic or jc can be NaNs
            shape_pix_pos = np.shape(self.pix_pos_proj)
            return np.zeros(shape_pix_pos[0]*shape_pix_pos[1])
        ic = (ic - 1)*1e8 # trick to force curve_fit to do bigger steps (otherwise it stays at initial position)
        jc = (jc - 1)*1e8
        if not 0 < ic < self.xs or  not 0 < jc < self.xs: # if we fall outside of map
            shape_pix_pos = np.shape(self.pix_pos_proj)
            return np.zeros(shape_pix_pos[0]*shape_pix_pos[1])
        centre_pos = self.pix_pos_proj[int(ic), int(jc)] # this is the vector associated with the pixel after proj, but it should be perfectly usable with vectors of pixels before proj
        dist_deg = np.abs(np.degrees(dist_angle(self.pix_pos_patch, centre_pos)))
        mygauss = amp * np.exp(-0.5*dist_deg**2/sig**2)
        full_map = np.zeros(12*self.nside**2)
        full_map[self.useful_pix] = mygauss
        proj_map = hp.gnomview(full_map, rot=self.rot, reso=self.reso, xsize=self.xs, return_projected_map=True, no_plot=True).data
        return np.ravel(proj_map * (1 - self.mask))

def fitgauss_img(mapij, ipos, jpos, xs, guess=None, doplot=False, distok=3, mytit='', nsig=1,
                mini=None, maxi=None, ms=10, renorm=False, mynum=33, axs=None, verbose=False, reso=None, pack=None, g2d=None):
    # iipos, jjpos = np.meshgrid(ipos, jpos, indexing="ij")
    iipos = ipos # already 2D
    jjpos = jpos

    # we want to keep the mask for the fit
    mask_badpix = mapij.mask
    mapij = mapij.data
    mapij[mask_badpix] = 0
    
    ### Displays the image as an array
    mm, ss = ft.meancut(mapij[mapij>1e-3], 3)
    if mini is None:
        mini = mm-nsig*ss
    if maxi is None:
        maxi = np.max(mapij)

    # g2d = gauss2dfit(iipos, jjpos) # has to be in the same order as in m from the fit
    # mask_badpix = None
    # g2d = gaussfitsphere(iipos, jjpos, mask=mask_badpix) # elt, azt

    # test_gauss = g2d(None, np.array([1, 90, 0, 1])).reshape((xs, xs))
    # plt.figure()
    # plt.imshow(test_gauss)
    # plt.show()

    ### Guess where the maximum is and the other parameters with a matched filter
    if guess is None:
        Ni = len(mapij)
        Nj = len(mapij[0])
        lobe_pos = (Ni//2, Nj//2)
        _, _, K = get_K(Ni, Nj)
        ft_phase = get_ft_phase(lobe_pos, Ni, Nj)
        border_size_i = Ni*0.1 # Ni = xs
        border_size_j = Nj*0.1 # Nj = xs
        cos_win = cos_window(Ni, Nj, lx=border_size_i, ly=border_size_j)
        deltaK = 1
        Kbin = get_Kbin(deltaK, K)
        nKbin = len(Kbin) - 1  # nb of bins
        Kcent = (Kbin[:-1] + Kbin[1:])/2
        size_pix = reso/60 # degree
        # reso_instr = 0.92 # degree
        reso_img = 1.036 # degree # test
        ft_shape = fft2(gauss2D(Ni, Nj, x0=lobe_pos[0], y0=lobe_pos[1], reso=[reso_img/size_pix], normal=True))

        filtmapsn = get_filtmapsn(mapij * cos_win, nKbin, K, Kbin, Kcent, ft_shape, ft_phase)

        # plt.figure()
        # plt.imshow(filtmapsn * 1e4)
        # plt.show()
        maxii = filtmapsn == np.nanmax(filtmapsn)
        ### in data coords
        # max_i = np.mean(iipos[maxii])
        # max_j = np.mean(jjpos[maxii])
        # guess = np.array([1e4, max_i, max_j, reso_img/conv_reso_fwhm])
        ### in pixel coords (new version, easier to plot). distok is now in pixels
        iipix, jjpix = np.meshgrid(np.arange(Ni), np.arange(Nj), indexing="ij")
        max_i = int(np.mean(iipix[maxii]))
        max_j = int(np.mean(jjpix[maxii]))
        guess = np.array([1e6, max_i, max_j, reso_img/conv_reso_fwhm])
        # guess = np.array([1e6, 95, 77, 0.5]) # fine-tuned for test
        if verbose:
            print("guess: amp = {}, i = {}, j = {}, sig = {}".format(guess[0], guess[1], guess[2], guess[3]))

        # # we bypass Minuit
        # fake_m = empty_class()
        # fake_m.values = guess
        # if doplot:
        #     return fake_m, None, None
        # else:
        #     return fake_m, None
    else:
        max_i = guess[1]
        max_j = guess[2]

    ### Do the fit putting the UNSEEN to a very low weight
    errpix = iipos*0 + ss
    errpix[mapij==0] *= 1e5

    ######### Minuit #########
    # data = fit.Data(np.ravel(iipos), np.ravel(mapij), np.ravel(errpix), g2d)
    # m, ch2, ndf = data.fit_minuit(guess, limits=[[0, 1e3, 1e8], [1, max_i - distok, max_i + distok], [2, max_j - distok, max_j + distok], [3, 0.6/conv_reso_fwhm, 1.5/conv_reso_fwhm]], renorm=renorm)
    # # m: amplitude, elevation (i), azimuth ((-)j), sigma Gaussian fit
    # where_res = np.array([int(m.values[1]), int(m.values[2])])
    # # adjust fit "by ha,d" for testing purposes
    # # delta_fit = np.array([5, 2])
    # # where_res += delta_fit
    # # ifit = m.values[1]
    # # jfit = m.values[2]
    # ijerr = np.array([m.errors[1], m.errors[2]]) * reso/60 # pix to deg
    # # g2d_ = gaussfitsphere(iipos, jjpos, mask=mask_badpix) # elt, azt
    # ### Image of the fitted Gaussian
    # fitted = np.reshape(g2d(ipos, m.values), (xs, xs))

    ######## curve_fit #######
    # ii, jj = np.meshgrid(np.arange(int(max_i - distok), int(max_i + distok) + 1), np.arange(int(max_j - distok), int(max_j + distok) + 1))
    xx = None
    # bounds=[[1e3, max_i - distok, max_j - distok, 0.6/conv_reso_fwhm], [1e8, max_i + distok, max_j + distok, 1.5/conv_reso_fwhm]]
    # popt, pcov = curve_fit(g2d, xx, mapij.ravel(), p0=guess, bounds=bounds)#, sigma=errpix.ravel())
    fact_renorm = 1e8 # trick to force curve_fit to do bigger steps (otherwise it stays at initial position)
    guess[1] = guess[1]/fact_renorm + 1
    guess[2] = guess[2]/fact_renorm + 1
    popt, pcov = curve_fit(g2d, xx, mapij.ravel(), p0=guess)#, sigma=errpix.ravel())
    fitted = np.reshape(g2d(ipos, popt[0], popt[1], popt[2], popt[3]), (xs, xs))
    popt[1] = (popt[1] - 1)*fact_renorm
    popt[2] = (popt[2] - 1)*fact_renorm
    guess[1] = (guess[1] - 1)*fact_renorm
    guess[2] = (guess[2] - 1)*fact_renorm
    where_res = np.array([int(popt[1]), int(popt[2])])
    m = type("Foo", (object,), {})()
    m.values = popt
    # m.errors = pcov
    m.errors = np.diag(pcov) # to be compatible with old code
    ijerr = np.array([m.errors[1], m.errors[2]]) * reso/60 # pix to deg
    
    ifit = where_res[0]
    jfit = where_res[1]
    ires = iipos[where_res[0], where_res[1]]
    jres = jjpos[where_res[0], where_res[1]]
    ijres = np.array([ires, jres])




    if doplot:
        origin = "upper" #"lower" swaps the y-axis and the guess doesn't match, default is "upper", and lower matches the hp.gnomview display orientation
        if axs is None:
            fig, axs = plt.subplots(1, 4, width_ratios=(1, 1, 1, 0.05), figsize=(16, 5))
            # axs[1].imshow(fitted, origin=origin, extent=[np.min(ipos), np.max(ipos), np.min(jpos), np.max(jpos)], vmin=mini, vmax=maxi)
            # im = axs[2].imshow(mapij - fitted, origin=origin, extent=[np.min(ipos), np.max(ipos), np.min(jpos), np.max(jpos)], vmin=mini, vmax=maxi)
            axs[1].imshow(fitted, origin=origin, vmin=mini, vmax=maxi)
            im = axs[2].imshow(mapij - fitted, origin=origin, vmin=mini, vmax=maxi)
            axs[0].set_ylabel('Pixel number [{} arcmin]'.format(reso))
            for i in range(3):
                axs[i].set_xlabel('Pixel number [{} arcmin]'.format(reso))
            axs[2].set_title('Residuals')
        axs = pmp.plot_fit_img(mapij, axs, ipos, jpos, iguess=guess[1], jguess=guess[2], ifit=ifit, jfit=jfit, vmin=mini, vmax=maxi, ms=ms, origin=origin)

        # plt.show()
        ### Look at result after shifting the fit a little (by eye)
        # amp, ii, jj, fwhm = m.values
        # ii += 4
        # jj -= 0
        # fitted_ = np.reshape(g2d(ipos, [amp, ii, jj, fwhm]), (xs, xs))
        # fig, axs = plt.subplots(1, 4, width_ratios=(1, 1, 1, 0.05), figsize=(16, 5))
        # axs[1].imshow(fitted_, origin=origin, vmin=mini, vmax=maxi)
        # im = axs[2].imshow(mapij - fitted_, origin=origin, vmin=mini, vmax=maxi)
        # axs[0].set_ylabel('Elevation [degrees]')
        # for i in range(3):
        #     axs[i].set_xlabel('Azimuth [degrees]')
        # axs[2].set_title('Residuals')
        # axs = pmp.plot_fit_img(mapij, axs, ipos, jpos, iguess=guess[1], jguess=guess[2], ifit=ii, jfit=jj, vmin=mini, vmax=maxi, ms=ms, origin=origin)
        return m, fitted, axs, ijres, ijerr
    return m, fitted, ijres, ijerr
    
    

def fit_one_tes(mymap, xs, reso, rot=np.array([0., 0., 0.]), doplot=False, verbose=False, guess=None, distok=3, mytit='', return_images=False, ms=10, renorm=False, azelguess=None, axs=None, pack=None):
    ### get the gnomview back into a np.array in order to fit it
    mm = mymap.copy()
    mapxy = hp.gnomview(mm, reso=reso, rot=rot, return_projected_map=True, xsize=xs, no_plot=True)#.data

    gnom_proj = hp.projector.GnomonicProj(rot=rot, reso=reso, xsize=xs)
    x, y = gnom_proj.ij2xy()
    azt_proj, elt_proj = gnom_proj.xy2ang(x=x.flatten(), y=y.flatten(), lonlat=True)
    azt_proj = azt_proj.reshape(np.shape(x))
    elt_proj = elt_proj.reshape(np.shape(y))

    i_elt = elt_proj
    j_azt = azt_proj

    nside = int(np.sqrt(len(mm)/12))
    g2d = gaussfitgnomproj(i_elt, j_azt, nside, rot, reso, xs, mask=mapxy.mask)

    if azelguess is not None:
        # try:
            # guess = np.array([1e4, azelguess[0], azelguess[1], 0.92])
            # guess_pos_cart = spherical2cartesian(1, -azelguess[0], azelguess[1], coord="horizontal", axis="last")
            guess_pos_cart = spherical2cartesian(1, azelguess[0], azelguess[1], coord="horizontal", axis="last")
            pix_pos_cart = spherical2cartesian(1, j_azt, i_elt, coord="horizontal", axis="last")
            dist_to_guess = dist_angle(guess_pos_cart, pix_pos_cart)
            argpix = np.argmin(np.abs(dist_to_guess))
            Npix_side = len(dist_to_guess)
            # guess = np.array([1e4, argpix//Npix_side, argpix%Npix_side, 0.92])
            guess = np.array([1e6, argpix//Npix_side, argpix%Npix_side, 0.5])

            # plt.figure()
            # plt.imshow(dist_to_guess)
            # plt.show()
            # print(np.shape(dist_to_guess))
            # print(guess)
            
            if verbose:
                print(guess)
        # except:
        #     guess = None
        #     if verbose:
        #         print("TES has no position on sky")
        #         print(guess)
        
    # # test méli mélo
    # i_elt_ = i_elt.copy()
    # i_elt = j_azt.copy()
    # j_azt = i_elt_.copy()
    if doplot:
        m, fitted, fig_axs, ijres, ijerr = fitgauss_img(mapxy, i_elt, j_azt, xs, guess=guess, doplot=doplot, distok=distok, mytit=mytit, ms=ms, renorm=renorm, axs=axs, verbose=verbose, reso=reso, pack=pack, g2d=g2d)
        if verbose:
            print(m.values)
    else:
        m, fitted, ijres, ijerr = fitgauss_img(mapxy, i_elt, j_azt, xs, guess=guess, doplot=doplot, distok=distok, mytit=mytit, ms=ms, renorm=renorm, verbose=verbose, reso=reso, pack=pack, g2d=g2d)

    if return_images:
        return m, mapxy, fitted, [np.min(i_elt), np.max(i_elt), np.min(j_azt), np.max(j_azt)], fig_axs, ijres, ijerr

    return m, ijres, ijerr
    

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

def get_ft_phase_1D(lobe_pos, Nx):  # problème si pas de round ! pourquoi ? parce que image décalée d'un nombre non entier de pixels/modes ?
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
    px = lobe_pos
    x= np.arange(Nx)
    ft_phase = np.exp(-2*np.pi*1j*x*round(-px)/Nx)
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

def get_filtmapsn_1D(mapj, nKbin, K, Kbin, Kcent, ft_beam_map, ft_phase):
    ftmapj = fft(mapj)
    modu2 = ftmapj*np.conj(ftmapj)
    result = np.zeros((nKbin))
    for i in range(nKbin):
        iKbin = np.logical_and(K>=Kbin[i], K<Kbin[i + 1])
        if len(K[iKbin])>0:
            result[i] = np.abs(np.mean(modu2[iKbin]))
        else:
            print("Kbin [{}, {}] is empty.".format(Kbin[i], Kbin[i + 1]))
    gp = np.interp(K, Kcent, result)   # Pk bins interpolated
    
    ftfilt = np.conj(ft_beam_map)/gp
    normfilt = np.sum(np.abs(ft_beam_map)**2/gp)
    # ftfilt = np.conj(ft_beam_map)/modu2
    # normfilt = np.sum(np.abs(ft_beam_map)**2/modu2)
    filtmapsn = np.real(ifft(ftfilt*ftmapj*ft_phase)/np.sqrt(normfilt))  # M convol T / sigma
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


def read_data(datadir, remove_t0=True, year_data="2022"):
    """
    Reads QUBIC raw data: time and TOD, as well azimuth, elevation and 
    their corresponding time.

    In the qubicpack/data/TD_TEMPERATURE_LABELS.txt you can find all
    the labels.

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
    try: # the nodata method is better but harder to redo everytime I update qubicpack
        tt, alltod = a.tod()
    except:
        print("\nThis file ({}) couldn't be read and will not be taken into account.\n".format(datadir))
        return None
    tt, alltod = a.tod()
    az = a.azimuth()
    el = a.elevation()
    tTbath = a.Tbath
    tT1K = [a.timeaxis('1K stage'), a.get_hk("1K stage")] # it is the label used when plotting temperature plots (time, temperature)
    print("tt[0]", tt[0])
    print("year_data", year_data)
    if year_data == "2022":
        thk = a.timeaxis(datatype='hk') # had to add a fix in qubicpack/tools.assign_pointing_data method
        print("len(thk)", len(thk))
        print("len(az)", len(az))
    elif tt[0] < 1774994400: # before the 01/04/2026 (might not be the best date))
        print("Using extern data for thk.")
        thk = a.timeaxis(datatype='extern', asic=1) # there is no intern hk in 2026-03-13 data
        print("shape thk", np.shape(thk))
        print("shape az", np.shape(az))
        print(az)
        print("shape el", np.shape(el))
        print(el)
    else:
        thk = a.timeaxis(datatype='AZ') # since qubicpack update, that's how thk is read to fit azel data
    tinit = tt[0]
    if remove_t0:
        ### We remove tt[0]
        tinit = tt[0]
        tt -= tinit
        thk -= tinit
    del(a)
    return tt, alltod, thk, az, el, tinit, tTbath, tT1K

def get_azel_moon(ObsSite, tt, tinit, doplot=True):
    MySite = EarthLocation(lat=ObsSite['lat'], lon=ObsSite['lon'], height=ObsSite['height'])
    # utcoffset = ObsSite['UTC_Offset']

    # print("tt in get_azel_moon", np.min(tt), np.max(tt))

    dt0 = datetime.utcfromtimestamp(int((tt + tinit)[0])) # starting time
    print(dt0)

    nbtime = 100
    tt_hours_loc = (tt - tt[0])/3600 # number of hours since starting time
    delta_time = np.linspace(np.min(tt_hours_loc), np.max(tt_hours_loc), nbtime)*u.hour

    alltimes = Time(dt0) + delta_time

    # print("alltimes in get_azel_moon", np.min(alltimes), np.max(alltimes))

    ### Local coordinates
    frame_Site = AltAz(obstime=alltimes, location=MySite)

    ### Source
    moon_Site = get_moon(alltimes)
    moonaltazs_Site = moon_Site.transform_to(frame_Site)  

    myazmoon = moonaltazs_Site.az.value
    myelmoon = moonaltazs_Site.alt.value

    azmoon = np.interp(tt_hours_loc, delta_time/u.hour, myazmoon)
    elmoon = np.interp(tt_hours_loc, delta_time/u.hour, myelmoon)

    # print("azel moon in get_azel_moon", np.min(azmoon), np.max(azmoon), np.min(elmoon), np.max(elmoon))
    if doplot:
        plt.figure()
        plt.plot(myazmoon, myelmoon, 'ro')
        plt.plot(azmoon, elmoon)
        plt.show()
    return azmoon, elmoon


def format_data(az_qubic, ObsSite, speedmin, datadir=None, det_pos=None, tshift=0, year_data="2022", doplot=False, simu=False):
    # wrapper that reads data and correct for tshift before separating scans

    # first read data from observation files
    data_step_one = read_many_files(az_qubic, datadir, year_data)

    # then correct for tshift and separate scans
    data = compute_secondary_data(ObsSite, speedmin, data_step_one, det_pos, tshift, doplot, simu)
    return data


def read_many_files(az_qubic, datadir=None, tt_=None, year_data="2022"):
    # we read data from observation files

    ### We flip the numbering of TOD around the diagonal of the quadrant in order to match simulations and data
    FPidentity = pt.make_id_focalplane()
    quadrant = 3
    QPidx = np.array([FPidentity[fp_idx].QPindex for fp_idx in range(len(FPidentity)) if FPidentity[fp_idx].quadrant == quadrant]).reshape(17, 17)
    if quadrant == 3: # TD quadrant = 3
        QPidx[11:15, 0] = np.array([4, 36, 68, 100]) - 1 # thermometers of quadrant 3
        QPidx[-1, 2:6] = np.array([132, 164, 196, 228]) - 1 # thermometers of quadrant 3
    elif quadrant == 2:
        QPidx[2:6, 0] = np.array([4, 36, 68, 100]) - 1 # thermometers (might not be the right numbers at the right place)
        QPidx[0, 11:15] = np.array([132, 164, 196, 228]) - 1 # thermometers
    QPidx_old = QPidx.flatten()[QPidx.flatten()>=0]
    QPidx = np.flip(np.flip(QPidx, axis=0).T, axis=0)
    QPidx = QPidx.flatten()
    QPidx = QPidx[QPidx>=0]
    sort_idx_old = np.argsort(QPidx_old)
    QPidx = QPidx[sort_idx_old]
    # QPidx = QPidx_old # if want old

    tt_full = []
    alltod_full = []
    Tbath_full = []
    T1K_full = []
    ifile_full = []
    tinit = None
    thk_full = []
    az_full = []
    el_full = []

    for i, diri in enumerate(datadir):
        print("\nreading file", i, diri)
        vars = read_data(diri, remove_t0=False, year_data=year_data)
        if vars is None:
            print("\nSkipping data...")
            continue
        tt, alltod, thk, az, el, tinit_, tTbath_hk, tT1K_hk = vars
        if len(thk) != len(az) or len(thk) != len(el):
            raise ValueError("thk has shape '{}' while az is '{}' and el '{}'".format(len(thk), len(az), len(el)))
        if len(thk) < 20: # in case something went wring in the acquisition, the file might be very small and unusable
            print("Housekeeping data has less than 20 data points, skipped...")
            continue
        az += az_qubic

        if tinit is None: # first good file
            tinit = tinit_
            print("tinit = {}".format(tinit))
        else:
            if tinit_ <= tinit:
                raise ValueError("The initial time {} is smaller than the one from the first file {}. Files might not be sorted well.".format(tinit_, tinit))
        
        Tbath = np.interp(tt + tinit, tTbath_hk[0], tTbath_hk[1])
        T1K = np.interp(tt + tinit, tT1K_hk[0], tT1K_hk[1])

        tt_full.append(tt)
        alltod_full.append(alltod)
        Tbath_full.append(Tbath)
        T1K_full.append(T1K)
        ifile_full.append(np.full_like(tt, i, dtype=int))
        thk_full.append(thk)
        az_full.append(az)
        el_full.append(el)

    tt = np.concatenate(tt_full)
    alltod = np.concatenate(alltod_full, axis=1)
    Tbath = np.concatenate(Tbath_full)
    T1K = np.concatenate(T1K_full)
    ifile = np.concatenate(ifile_full)
    thk = np.concatenate(thk_full)
    az = np.concatenate(az_full)
    el = np.concatenate(el_full)

    # this is the data before correcting for the time shift
    data_step_one = [tt, tinit, alltod, QPidx, Tbath, T1K, ifile, thk, az, el] # the data that could be useful later in the analysis

    return data_step_one


def compute_secondary_data(ObsSite, speedmin, data_step_one, det_pos=None, tshift=0, doplot=False, simu=False):
    # we correct for tshift and then separate data by scan

    # the concatenated data from the files of the observation
    tt, tinit, alltod, QPidx, Tbath, T1K, ifile, thk, az, el = data_step_one
    thk = thk.copy() # we will modify the values in thk otherwise
    tt = tt.copy()

    # need to put tt[0] to zero, but be careful of real time
    # tshift seen in plotting back and forth images or with the Earth magnetic field. In the second case, tshift has shape (len(tt),)
    if np.shape(tshift):
        # tshift_ = np.interp(thk - tinit, tt_, tshift)
        tshift_ = np.interp(thk, tt, tshift) # let's use tt because it's the same as tt_ but with tinit
        # tshift_ = np.interp(tt - tinit, tt_, tshift)
    else:
        tshift_ = tshift
    tt -= tinit #+ tshift_
    thk -= tinit - tshift_ # now correcting the thk instead of tt because we trust more tt

    ### Azimuth and Elevation of the Moon at the same timestamps from the observing site
    azmoon, elmoon = get_azel_moon(ObsSite, tt, tinit, doplot=False)

    ### Identify scan types and numbers
    _, azt, elt, scantype, _ = identify_scans(thk, az, el, 
                                            tt=tt, doplot=False, 
                                            plotrange=[tt[0], tt[0] + 2000], 
                                            thr_speedmin=speedmin)
    # good solution
    if simu:
        boresight_angle = 0 # simus
    else:
        boresight_angle = -3.5 #-3.5 # real data
    print("boresight_angle =", boresight_angle)
    newazt, newelt = get_azel_as_zenith(tt, azt, elt, azmoon, elmoon, boresight_angle=boresight_angle, det_pos=det_pos) # change the coordinates at the map creation level from the real posiiton of the Moon first to be able to fit the angular distance and orientation of the shift of each detector on the sky

    print("shape scantype", np.shape(scantype))
    print("shape newazt", np.shape(newazt))
    data = [tt, tinit, alltod, QPidx, azt, elt, newazt, newelt, scantype, Tbath, T1K, ifile, thk, az, el] # the data that could be useful later in the analysis
    if doplot and det_pos is None:
        fig, axs = plt.subplots(1, 2)
        axs[0].set_title("az")
        axs[0].plot(tt, azt, c="b", label="azt")
        axs[0].scatter(tt, newazt, s=1, c="g", label="newazt")
        axs[0].plot(tt, azmoon, c="r", label="moon")
        axs[1].set_title("el")
        axs[1].plot(tt, elt, c="b", label="elt")
        axs[1].scatter(tt, newelt, s=1, c="g", label="newelt")
        axs[1].plot(tt, elmoon, c="r", label="moon")
        plt.legend()
        plt.show()
    return data


def make_coadded_maps(allTESNum, data=None,
                      doplot=True, nside=256, parallel=False, check_back_forth=False,
                      isok_arr=None, det_pos=None, clean_tod=True, manual=False, ObsName=None,
                      new_method_clean=False, theo_sb=None, more=""):
    
    ObsDate = ObsName[:10]
    
    tt, tinit, alltod, _, azt, elt, newazt_, newelt_, scantype, Tbath, T1K, ifile, _, _, _ = data
    if tinit < 1774994400: # before the 01/04/2026 (might not be the best date)
        # the first data points are the telescope going to start position
        nskip = 10000
        tinit += tt[nskip]
        tt = tt[nskip:]
        alltod = alltod[:, nskip:]
        azt = azt[nskip:]
        elt = elt[nskip:]
        if det_pos is None:
            newazt_ = newazt_[nskip:]
            newelt_ = newelt_[nskip:]
        else:
            newazt_ = newazt_[:, nskip:]
            newelt_ = newelt_[:, nskip:]
        scantype = scantype[nskip:]
        Tbath = Tbath[nskip:]
        T1K = T1K[nskip:]
        ifile = ifile[nskip:]
    ### Loop over TES to do the maps
    print('\nLooping coaddition mapmaking over selected TES')
    print('nside = ',nside)
    start_time = time.perf_counter()
    if parallel is False:
        print('Using sequential loop')
        allmaps = np.zeros((len(allTESNum), 12*nside**2)) + 1e-15 # to fool minuit
        for i in range(len(allTESNum)):
            if isok_arr is not None: # only compute good TES (to make testing phase easier)
                if not isok_arr[i]:
                    continue
            TESNum = allTESNum[i]
            print('TES# {}'.format(TESNum), end=" ")
            iTES = TESNum - 1
            if ObsDate[:4] == "2022":
                tod = alltod[iTES, :]
                # tod = alltod[iTES == QPidx][0] # order of TES not well-implemented before? (use this in 2022 analysis?)
            else:
                tod = alltod[iTES, :]
            if det_pos is not None:
                if len(np.shape(newazt_)) == 1:
                    newazt = newazt_
                    newelt = newelt_
                elif np.shape(newazt_)[0] == 1:
                    newazt = newazt_[0]
                    newelt = newelt_[0]
                    # det_pos_i = det_pos
                else:
                    newazt = newazt_[iTES]
                    newelt = newelt_[iTES]
                    # det_pos_i = det_pos[iTES]
            else:
                newazt = newazt_
                newelt = newelt_
            print("shape pos", np.shape(det_pos))

            allmaps[i,:], mapscounts = make_coadded_maps_TES(tt, tod, azt, elt, scantype, newazt, newelt, ifile,
                                                             Tbath=Tbath, T1K=T1K, TES_number=TESNum, nside=nside,
                                                             doplot=doplot, check_back_forth=check_back_forth,
                                                             det_pos=det_pos, clean_tod=clean_tod, manual=manual,
                                                             ObsName=ObsName, new_method_clean=new_method_clean,
                                                             theo_sb=theo_sb, more=more)
            print('OK', flush=True)
    else:
        print('using a parallel loop : no output will be given while processing... be patient...')
        ### Note that this code has been generated using ChatGPT
        def process_TES(i, TESNum, allmaps, alltod, tt, azt, elt, scantype, newazt, newelt, nside, doplot):
            # Create a lock for each process to ensure safe access to shared memory
            lock = Lock()
            iTES = TESNum - 1
            print(i, flush=True)
            # thermom = np.array([4, 36, 68, 100, 132, 164, 196, 228]) - 1 # thermometers of quadrant 3
            
            if ObsDate[:4] == "2022":
                tod = alltod[iTES, :]
                # tod = alltod[iTES == QPidx][0] # order of TES not well-implemented before? (use this in 2022 analysis?)
            else:
                tod = alltod[iTES, :]
            map_result, mapscounts = make_coadded_maps_TES(tt, tod, azt, elt, scantype, newazt, newelt, ifile,
                                                           TES_number=TESNum, nside=nside, doplot=doplot, det_pos=det_pos,
                                                           clean_tod=clean_tod, manual=manual, ObsName=ObsName, new_method_clean=new_method_clean,
                                                           theo_sb=theo_sb, more=more)        
            # Use lock to ensure safe access to shared memory inside the inner function
            with lock:
                # Directly assign the result to the correct index in allmaps
                # allmaps is a list of numpy arrays, so we can use allmaps[i] directly
                allmaps[i] = map_result
        
        def parallel_coadded_maps(allTESNum, alltod, tt, azt, elt, scantype, newazt, newelt, nside, doplot):

            # Use Manager to create a shared list that will be modified by parallel processes
            with Manager() as manager:
                # Create a list of NumPy arrays initialized to zeros
                allmaps = manager.list([np.zeros(12 * nside ** 2) for _ in range(len(allTESNum))])

                if det_pos is None:
                    # Run the parallel processing with the correct arguments
                    Parallel(n_jobs=-1)(delayed(process_TES)(i, allTESNum[i], allmaps, alltod, tt, azt, elt, scantype, newazt, newelt, nside, doplot)
                                        for i in range(len(allTESNum)))
                else:
                    Parallel(n_jobs=-1)(delayed(process_TES)(i, allTESNum[i], allmaps, alltod, tt, azt, elt, scantype, newazt[i], newelt[i], nside, doplot)
                                        for i in range(len(allTESNum)))
                    # i = 4
                    # process_TES(i, allTESNum[i], allmaps, alltod, tt, azt, elt, scantype, newazt[i], newelt[i], nside, doplot)
        
                # Convert the manager list back to a NumPy array (this ensures allmaps is a numpy array of arrays)
                allmaps_np = np.array([np.array(allmaps[i]) for i in range(len(allTESNum))])
        
            return allmaps_np

        if det_pos is not None:
            newazt = newazt_
            newelt = newelt_
            print(np.shape(newazt), np.shape(newelt))
        else:
            newazt = newazt_
            newelt = newelt_

        scantype[0] = 0 # in order to not have peaks too close to start of scan (there would be no start point)
        allmaps = parallel_coadded_maps(allTESNum, alltod, tt, azt, elt, 
                                        scantype, newazt, newelt, nside, doplot=False)
        print("NSIDE = ", nside)
    
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds => average of {(elapsed_time/len(allTESNum)):.4f} per TES")    
        

    # Get central Az and El from pointing
    if det_pos is not None:
        # center = det_pos
        center = [0, 0]
    else:
        # center = [np.mean(newazt), np.mean(newelt)]
        center = [0, 90] # zenith
    return allmaps, center, newazt, newelt, scantype


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
            None by default, if not None:
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

    # medaz_dt_ = get_vel(thk, az, order=50) # high order necessary to remove glitches
    # medaz_dt = medfilt(medaz_dt_, median_size)
    fs = np.median(thk[1:] - thk[:-1]) # sampling frequency
    order = int(1/fs) # ~1s # interval in points usde to compute slope
    medaz_dt_ = get_vel(thk, az, order=order)
    medaz_dt = medaz_dt_
    
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
    # Low velocity => Bad
    c0 = np.abs(medaz_dt) < thr_speedmin
    # Positive velicity => Good
    cpos = (~c0) * (medaz_dt >= 0)
    # Negative velocity => Good
    cneg = (~c0) * (medaz_dt < 0)

    ### Scan identification at HK sampling
    scantype_hk = np.zeros(len(thk), dtype='int') - 10
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
        count_them = np.sum(scantype == 0) + np.sum(scantype <= -1) + np.sum(scantype >= 1)
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
    not_finite = ~np.isfinite(params_dbscan)
    params_dbscan[not_finite] = np.random.randint(low=1e5, high=1e6, size=len(params_dbscan[not_finite])) # get rid of NaN and inf
    print(np.min(params_dbscan), np.max(params_dbscan))
    rng_nan = np.random.default_rng(seed=12345)
    params_dbscan[np.isnan(params_dbscan)] = rng_nan.uniform(low=1, high=2, size=(len(params_dbscan[np.isnan(params_dbscan)]),)) * 1e8

    labels = run_DBSCAN(params_dbscan, eps=eps, min_samples=min_samples)
    DB_ok = labels == 0
    if doplot:
        plt.figure()
        plt.subplot().set_aspect(1)
        plt.scatter(delta_az[visibly_ok_arr], delta_el[visibly_ok_arr], c='k', s=30, label='all visibly ok ({})'.format(len(delta_az[visibly_ok_arr])))
        plt.scatter(delta_az[DB_ok], delta_el[DB_ok], c='r', s=20, label='DBSCAN selected ({})'.format(len(delta_az[DB_ok])))
        plt.xlabel('$\Delta_{az}^{Moon} - Offset_{Creidhe}$')
        plt.ylabel('$\Delta_{el}^{Moon} - Offset_{Creidhe}$')
        plt.legend()
        plt.show()

    return DB_ok

def get_DBscan_res_cart(x_fit, y_fit, z_fit, x_theo, y_theo, z_theo, visibly_ok_arr, doplot, eps=0.5, min_samples=10):

    delta_x = x_fit - x_theo
    delta_y = y_fit - y_theo
    delta_z = z_fit - z_theo

    params_dbscan = np.array([delta_x, delta_y, delta_z]).T
    rng_nan = np.random.default_rng(seed=12345)

    # we don't want the NaNs to crash our scan
    params_dbscan[np.isnan(params_dbscan)] = rng_nan.uniform(low=1, high=2, size=(len(params_dbscan[np.isnan(params_dbscan)]),)) * 1e8

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(params_dbscan)
    labels = clustering.labels_
    DB_ok = labels == 0
    if doplot:
        plt.figure()
        plt.subplot().set_aspect(1)
        plt.scatter(delta_x[visibly_ok_arr], delta_y[visibly_ok_arr], c='k', s=30, label='all visibly ok ({})'.format(len(delta_x[visibly_ok_arr])))
        plt.scatter(delta_x[DB_ok], delta_y[DB_ok], c='r', s=20, label='DBSCAN selected ({})'.format(len(delta_x[DB_ok])))
        plt.xlabel('$\Delta_{x}^{Moon} - Offset_{Creidhe}$')
        plt.ylabel('$\Delta_{y}^{Moon} - Offset_{Creidhe}$')
        plt.legend()
        plt.tight_layout()
        plt.savefig("DBscan_result.pdf")
        plt.show()

    return DB_ok


### Function to rotate a set of points around a given center
def rotate_translate_scale_2d(xin, theta, center, scale):
    rotmat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    return scale * np.dot(rotmat, (xin-center).T).T

def rot_trans_scale_pts(x, pars):
    pts = np.reshape(x, (len(x)//2, 2))
    return np.ravel(rotate_translate_scale_2d(pts, np.radians(pars[0]), np.array([pars[1],pars[2]]), pars[3]))
                    
def rot_trans_scale2d_pts(x, pars):
    pts = np.reshape(x, (len(x)//2, 2))
    return np.ravel(rotate_translate_scale_2d(pts, np.radians(pars[0]), np.array([pars[1],pars[2]]), np.array([pars[3], pars[4]])))


def rotate_2d_zen(data, theta_rot): # data_in (npoints, 3), theta_rot (3) in degrees
    theta_x, theta_y, theta_z = np.radians(theta_rot)
    rot_mat_x = get_simple_rotation_matrix(axis="x", angle=theta_x)
    rot_mat_y = get_simple_rotation_matrix(axis="y", angle=theta_y)
    rot_mat_z = get_simple_rotation_matrix(axis="z", angle=theta_z)
    rot_mats = [rot_mat_x, rot_mat_y, rot_mat_z]
    for rot_mat in rot_mats:
        data = np.einsum("ij,kj->ki", rot_mat, data) # we do the rotations in the order x then y then z
    return data

def rotate_2d_zen_pts(data_flat, theta_rot):
    pts = np.reshape(data_flat, (len(data_flat)//3, 3))
    return np.ravel(rotate_2d_zen(pts, theta_rot))

def fun_minimise(theta_rot, data, target):
    pts = np.reshape(data, (len(data)//3, 3))
    target_pts = np.reshape(target, (len(data)//3, 3))
    rotated_pts = rotate_2d_zen(pts, theta_rot)
    all_dist = dist_angle(target_pts, rotated_pts)
    return np.sum(all_dist**2)

### From ipynb pipeline_moon_fit_order1_peak.ipynb

# get the spherical geodesic of length ang_dist_deg + 2*delta_angle_deg coursing through the two points A and B, +/- delta_angle_deg
def get_great_circle_traj(point_A, point_B, sphere_centre, sphere_radius, npoints, ang_dist_deg, delta_angle_deg, offset=0):
    # offset is in part of the circumference
    if (not np.isclose(np.linalg.norm(point_A - sphere_centre), sphere_radius)) or (not np.isclose(np.linalg.norm(point_B - sphere_centre), sphere_radius)):
        print(np.linalg.norm(point_A - sphere_centre))
        print(np.linalg.norm(point_B - sphere_centre))
        print(sphere_radius)
        raise ValueError("One of the points is not on the sphere.")
    ang_dist_rad = np.radians(ang_dist_deg) # the length of the line is fixed (ang_dist_deg + 2*delta_angle_deg)
    delta_angle_rad = np.radians(delta_angle_deg)
    plane = get_plane(point_A, point_B, sphere_centre, offset=offset) # compute the equation of the plane with A, B and the sphere centre C
    vec_1, vec_2 = get_vect_plane(plane, point_A, sphere_centre, offset=offset) # create two orthogonal vectors on the plane, vec_1 pointing at A
    theta = np.linspace(0 - delta_angle_rad, ang_dist_rad + delta_angle_rad, npoints).reshape(npoints, 1)
    great_circle = sphere_radius * (np.cos(theta) * vec_1 + np.sin(theta) * vec_2) # get the geodesic on the sphere between the two points, +/- delta_angle_deg
    return great_circle, theta.reshape(npoints)


# get two orthogonal vectors on the plane, one pointing at point_ref_end from point_ref_start
def get_vect_plane(plane, point_ref_end, point_ref_start, offset=0): 
    if not np.isclose(np.dot(plane[:3], point_ref_end) + plane[3], 0):
        raise ValueError("The reference end point is not on the plane.")
    if not np.isclose(np.dot(plane[:3], point_ref_start) + plane[3], 0):
        raise ValueError("The reference start point is not on the plane.")
    perp_vect = plane[:3]/np.sqrt(np.sum(plane[:3]**2))
    vec_1 = (point_ref_end - point_ref_start)/np.linalg.norm(point_ref_end - point_ref_start)
    vec_2 = np.cross(perp_vect, vec_1)/np.linalg.norm(np.cross(vec_1, perp_vect))
    return vec_1, vec_2

# wrapper to get spherical geodesic from points A and B in spherical coordinates instead of cartesian
def get_traj(point_A_sph, point_B_sph, npoints, ang_dist_deg, delta_angle_deg):
    point_A = spherical2cartesian(1, point_A_sph[0], point_A_sph[1])
    point_B = spherical2cartesian(1, point_B_sph[0], point_B_sph[1])
    sphere_centre=np.array([0, 0, 0])
    
    great_circle_traj, angle_array = get_great_circle_traj(point_A, point_B, sphere_centre=sphere_centre, sphere_radius=1, npoints=npoints, ang_dist_deg=ang_dist_deg, delta_angle_deg=delta_angle_deg)
    theta_gc = np.arccos(great_circle_traj[:, 2])
    phi_gc = np.arctan2(great_circle_traj[:, 1], great_circle_traj[:, 0])
    return theta_gc, phi_gc, angle_array

# get the value of a map along a spherical geodesic
def get_values_line(map_, point_A_sph, point_B_sph, npoints, ang_dist_deg, delta_angle_deg):
    theta_gc, phi_gc, angle_array = get_traj(point_A_sph, point_B_sph, npoints, ang_dist_deg, delta_angle_deg)
    return hp.get_interp_val(map_, theta_gc, phi_gc), theta_gc, phi_gc, angle_array

# get the mean value of a map along a spherical geodesic in the width +/- widht_angle_deg
def get_values_large_line(map_, theta_gc, phi_gc, width_angle_deg, nlines):
    theta_gc_ = np.tile(theta_gc, (nlines, 1))
    phi_gc_ = phi_gc + np.radians(np.linspace(-width_angle_deg, width_angle_deg, nlines)).reshape(nlines, 1)
    res = hp.get_interp_val(map_, theta_gc_, phi_gc_)
    return np.mean(res, axis=0), res, theta_gc_, phi_gc_

# get the value of a map on line parallel to a spherical geodesic in the direction theta or phi, supposed to represent the elevation
def get_side_lines(hpmap, theta_gc, phi_gc, side, nlines):
    if side == "phi":
        theta_gc_test_diff = np.tile(theta_gc, (nlines, 1))
        phi_gc_test_diff_up = phi_gc + np.radians(np.linspace(2, 3, nlines)).reshape(nlines, 1)
        phi_gc_test_diff_down = phi_gc - np.radians(np.linspace(2, 3, nlines)).reshape(nlines, 1)
        zi_up = hp.get_interp_val(hpmap, theta_gc_test_diff, phi_gc_test_diff_up)
        zi_down = hp.get_interp_val(hpmap, theta_gc_test_diff, phi_gc_test_diff_down)
        coord_up = [theta_gc_test_diff, phi_gc_test_diff_up]
        coord_down = [theta_gc_test_diff, phi_gc_test_diff_down]
    elif side == "theta":
        phi_gc_test_diff = np.tile(phi_gc, (nlines, 1))
        theta_gc_test_diff_up = theta_gc + np.radians(np.linspace(1.5, 2, nlines)).reshape(nlines, 1)
        theta_gc_test_diff_down = theta_gc - np.radians(np.linspace(1.5, 2, nlines)).reshape(nlines, 1)
        zi_up = hp.get_interp_val(hpmap, theta_gc_test_diff_up, phi_gc_test_diff)
        zi_down = hp.get_interp_val(hpmap, theta_gc_test_diff_down, phi_gc_test_diff)
        coord_up = [theta_gc_test_diff_up, phi_gc_test_diff]
        coord_down = [theta_gc_test_diff_down, phi_gc_test_diff]
    return zi_up, zi_down, coord_up, coord_down


# from calibration_dev All_scans_demodulation_src.ipynb

def match_shape(array1, shape_array2): # in the case that array2 has more or the same number of dims
    shape_array1 = np.shape(array1)
    if len(shape_array2) <= len(shape_array1):
        return array1
    # even with this method, the dimensions have to be in the same order!
    # but the new dimensions don't have to be only at the end
    list_expand_dims = []
    i_dim_1 = 0
    for i_dim_2 in range(len(shape_array2)):
        if i_dim_1 >= len(shape_array1):
            list_expand_dims.append(i_dim_1)
            i_dim_1 += 1
            continue
        if shape_array1[i_dim_1] == shape_array2[i_dim_2]:
            i_dim_1 += 1
        else:
            list_expand_dims.append(i_dim_2)
    # print(list_expand_dims)
    return np.expand_dims(array1, list_expand_dims) # quite dangerous stuff here

def polar2cartesian(r, theta, axis="first"):
    x = r * np.cos(np.radians(theta))
    y = r * np.sin(np.radians(theta))
    res = np.array([x, y])
    if axis == "first":
        return res
    elif axis == "last":
        return np.moveaxis(res, 0, -1)

# get the angle between two points (of shape (..., 3))
def dist_angle(vec_A, vec_B):
    cos_angle = np.einsum("...j,...j->...", vec_A, vec_B)/(np.linalg.norm(vec_A, axis=-1) * np.linalg.norm(vec_B, axis=-1))
    mask = np.logical_and(1<cos_angle, cos_angle<1 + 1e-8)
    cos_angle[mask] = 1 # be careful with that, it just seems that it is sometimes a bit higher than 1 because of numerical approximations
    mask = np.logical_and(-1 - 1e-8<cos_angle, cos_angle<-1)
    cos_angle[mask] = -1 # be careful with that, it just seems that it is sometimes a bit lower than -1 because of numerical approximations
    angle = np.arccos(cos_angle) # compute the angle between CA and CB
    sign_angle = np.sign(np.cross(vec_A, vec_B, axis=-1)[..., 2]) # the sign of the z component should give us the sign of the angle
    sign_angle[sign_angle == 0] = 1 # works only if we don't care about the sign
    return sign_angle * angle # radians

# get vector perp to horizontal great circle from az el position
def get_perp_vect_horiz_great_circle(azimuth, elevation, sphere_centre=np.array([0, 0, 0]), sphere_radius=1):
    perp_direction = spherical2cartesian(sphere_radius, centre_coord_at_0(azimuth + 180), 90 - elevation, coord="horizontal", axis="first")
    # print("shape perp_direction", np.shape(perp_direction))
    perp_vect = perp_direction - match_shape(sphere_centre, perp_direction.shape)
    # print("shape perp_vect", np.shape(perp_vect))
    return perp_vect

def get_azel_as_zenith(tt, azt, elt, azt_source, elt_source, boresight_angle=0, det_pos=None):
    """
    This function computes the coordinates of a point or an array of points with respect to a
    given source, taking the source as the zenith of the new coordinates system.

    Parameters
    ----------
    azt, elt : arrays (N,)
        The original coordinates of the array of N points.
    azt_source, elt_source : arrays (N,) or np.float_
        The original coordinates of the source.
    boresight_angle : float
        The boresight rotation to apply (to be checked) in order to retrieve a vertical beam,
        in degrees. Default is 0.

    Returns
    -------
    new_azt, new_elt :
        The coordinates of the array of points in the new system.

    """
    # az_source and el_source need to be either np.float_ or np.array
    sphere_radius = 1
    sphere_centre = np.array([0, 0, 0])
    if det_pos is not None: # if we have already fitted the detector offset on the sky, we can put the position of the Moon for this particular detector as the zenith
        # the idea here is to compute the value of az and el for the given detector position
        # this position is fitted from maps centered at the line of sight of the telescope as if the source were at zenith
        if len(np.shape(det_pos)) == 1:
            az_zen = np.array([det_pos[0]])
            el_zen = np.array([det_pos[1]])
            ndets = 1
            # print("here we are!")
        else:
            az_zen = det_pos[:, 0]
            el_zen = det_pos[:, 1]
            ndets = len(det_pos)

            thermom = np.array([4, 36, 68, 100, 132, 164, 196, 228]) - 1 # thermometers of quadrant 3
            for idx_thermom in thermom:
                az_zen[idx_thermom] = 0 # we remove the NaN
                el_zen[idx_thermom] = 90 # we remove the NaN
        
        az_zen = np.expand_dims(az_zen, 1)
        el_zen = np.expand_dims(el_zen, 1)
        # final shape in cartesian coord should be (3, ndets, ntimes) otherwise it crashed (probably linked to how arrays are stored in memory)
        # below, the deltas are computed from the distance to zenith (90 - el_zen) and the direction (az_zen - 90)
        delta_az = (90 - el_zen) * np.cos(np.radians(az_zen - 90)) / np.cos(np.radians(np.expand_dims(elt_source, 0)))
        delta_el = (90 - el_zen) * np.sin(np.radians(az_zen - 90))
        # we might want instead to do a two-step rotation:
        # - one to get the Moon at zenith for the telescope
        # - one to get the Moon at zenith for the detector
        # this method might be the best, since the detector position is fitted on Moon at zenith 
        # and it might be more accurate instead of converting to az, el (?)
        # actually probably not, the azimuth conversion has to be done just for the elevation of the source (which is known)
        # and the elevation conversion is absolute
        azt_source = np.expand_dims(azt_source, 0) + delta_az
        elt_source = np.expand_dims(elt_source, 0) + delta_el
        # this should be (ndets, ntimes)?
        del az_zen, el_zen, delta_az, delta_el

    azt_zen = np.zeros_like(azt_source)
    elt_zen = np.zeros_like(azt_source)
    len_batch = int(1e5) # number of pointings treated at the same time
    one_more = int(len(azt)%len_batch > 0)
    n_iter = len(azt)//len_batch + one_more
    for i_iter in range(n_iter):
        print("Computing alpha, beta for the {}-point batch {}/{}".format(len_batch, i_iter + 1, n_iter))
        lower_bound = i_iter*len_batch
        higher_bound = min((i_iter + 1)*len_batch, len(azt))
        azt_i = azt[lower_bound:higher_bound]
        elt_i = elt[lower_bound:higher_bound]
        if det_pos is not None:
            azt_source_i = azt_source[:, lower_bound:higher_bound]
            elt_source_i = elt_source[:, lower_bound:higher_bound]
        else:
            azt_source_i = azt_source[lower_bound:higher_bound]
            elt_source_i = elt_source[lower_bound:higher_bound]
        # we get the vector perpendicular to the horizontal great circle at pointing
        perp_vect_pointing = get_perp_vect_horiz_great_circle(azt_i, elt_i, sphere_radius=sphere_radius, sphere_centre=sphere_centre)
        # print("perp_vect_pointing OK", flush=True)

        # we get vector perpendicular to the great circle going through pointing and calsource
        calsource = spherical2cartesian(sphere_radius, azt_source_i, elt_source_i, coord="horizontal", axis="first")
        pointing = spherical2cartesian(sphere_radius, azt_i, elt_i, coord="horizontal", axis="first")
        del azt_i, elt_i, azt_source_i, elt_source_i
        perp_vec_gc_pointing_calsrc = get_perp_vect(calsource, pointing, sphere_centre)

        # we want the coords to be the last axis
        vec_calsource = np.moveaxis(calsource, 0, -1) # sphere centre is [0, 0, 0]
        vec_pointing = np.moveaxis(pointing, 0, -1)
        # the angle between the pointing and the calsource in degrees
        angle_beta = np.abs(np.degrees(dist_angle(vec_calsource, vec_pointing)))

        # the angle between the horizontal great circle and the great circle with the pointing and the calsource
        angle_alpha = np.degrees(dist_angle(np.moveaxis(perp_vect_pointing, 0, -1), np.moveaxis(perp_vec_gc_pointing_calsrc, 0, -1)))
        del perp_vect_pointing, perp_vec_gc_pointing_calsrc

        angle_alpha[~np.isfinite(angle_alpha)] = 0 # at the pixel pointing at calsource or if problem for a scan
        angle_beta[~np.isfinite(angle_beta)] = 30 # if problem for a scan

        # here we want 3D in order to rotate and get the new azimuth elevation that I can compare with the original ones
        new_pointing = spherical2cartesian(sphere_radius, angle_alpha, 90 - angle_beta, coord="horizontal", axis="first") # beta is 90 - elevation!
        pre_rotation_matrix = get_simple_rotation_matrix("z", np.radians(90 + boresight_angle)) # rotation x --> y # boresight_angle is used here to make the final beam "vertical"

        new_pointing = np.einsum("ij,j...k->i...k", pre_rotation_matrix, new_pointing)
        if det_pos is not None:
            _, azt_zen[:, lower_bound:higher_bound], elt_zen[:, lower_bound:higher_bound] = cartesian2spherical(new_pointing[0], new_pointing[1], new_pointing[2], coord="horizontal", axis="first")
        else:
            _, azt_zen[lower_bound:higher_bound], elt_zen[lower_bound:higher_bound] = cartesian2spherical(new_pointing[0], new_pointing[1], new_pointing[2], coord="horizontal", axis="first")

    return azt_zen, elt_zen


def print_keys(d, keys):
    for key in keys:
        print(' - {:25}: {}'.format(key, d[key]))
    print('\n')

### This function retrieves the peaks information as a function of frequency, for each of the MultiBandInstrument sub-bands
def get_peaks_configuration(d, doplot=False, idet=None, debug=False):
    if d['config'] == 'TD':
        Ndet = 248
    elif d['config'] == 'FI':
        Ndet = 992
    else:
        print('Wrong config in dict')
        return 0

    print_keys(d, ['config', 'instrument_type', 'synthbeam', 'use_synthbeam_file', 'synthbeam_fraction', 'synthbeam_kmax'])
    try:
        q_instrument = QubicMultibandInstrument(d)
        q_scene = QubicScene(d)
        print('done', d["config"], d['instrument_type'], 'nf_sub={}'.format(d['nf_sub']))
    except:
        print('oups ! failed instanciating QubicMultibandInstrument()')
        return 0,0,0,0
    
    n_nus = len(q_instrument)
    nus = np.zeros(n_nus)
    dnus = np.zeros(n_nus)
    print()
    print('The instrument has {} sub-frequencies'.format(n_nus))
    for i in range(n_nus):
        print("detector center", q_instrument.subinstruments[i].detector.center)
        nus[i] = q_instrument.subinstruments[i].d['filter_nu']/1e9
        dnus[i] = q_instrument.subinstruments[i].d['filter_relative_bandwidth']*q_instrument.subinstruments[i].d['filter_nu']/1e9
        print('- {0:}: nu = {1:7.2f} GHz ; bw = {2:7.2f}'.format(i, nus[i], dnus[i]))

        q_instrument.subinstruments[i].detector.center[0] = [0, 0, -0.3] # we take a fake central detector because we only care about theta and phi from zenith
    
    thetas = np.zeros((n_nus, Ndet, (2*d['synthbeam_kmax']+1)**2))
    phis = np.zeros((n_nus, Ndet, (2*d['synthbeam_kmax']+1)**2))
    vals = np.zeros((n_nus, Ndet, (2*d['synthbeam_kmax']+1)**2))
    for i in range(n_nus):
        thetas[i,:,:], phis[i,:,:], vals[i,:,:] = q_instrument.subinstruments[i]._peak_angles(q_scene, 
                                                    q_instrument.subinstruments[i].d['filter_nu'], 
                                                    # q_instrument.subinstruments[i].detector.center, 
                                                    np.full_like(q_instrument.subinstruments[i].detector.center, np.array([0, 0, -0.3])), 
                                                    q_instrument.subinstruments[i].synthbeam, 
                                                    q_instrument.subinstruments[i].horn, 
                                                    q_instrument.subinstruments[i].primary_beam)
    if doplot:
        imed = n_nus // 2
        if idet is None:
            # idet = np.random.randint(Ndet)
            idet = 0
        sb = q_instrument.subinstruments[imed].get_synthbeam(q_scene, idet)
        plt.figure(figsize=(8, 4))
        hp.gnomview(np.log10(sb/np.max(sb)), rot=[0,90], reso=20, min=-5, max=0,
             title='Theory {0:} {1:7.2f} GHz: TES #{2:}'.format(d['config'], q_instrument.subinstruments[imed].d['filter_nu']/1e9, idet), 
             sub=(1,2,1))
        for i in range(n_nus):
            hp.projscatter(thetas[i, idet,:], phis[i, idet,:], c=vals[i, idet,:]/np.max(vals[i, idet,:]), 
                           marker='x', cmap='Reds')

        plt.subplot(1,2,2)
        plt.errorbar(nus, dnus, yerr=0, xerr=dnus/2, fmt='ro')
        for i in range(len(nus)):
            plt.axvline(x=nus[i]-dnus[i]/2, ls=':', color='k', alpha=0.5)
            plt.axvline(x=nus[i]+dnus[i]/2, ls=':', color='k', alpha=0.5)
        plt.xlabel('Frequency [GHz]')
        plt.ylabel('Bandwidth [GHz]')
        plt.title('Theory {0:} {1:7.2f} GHz: TES #{2:}'.format(d['config'], q_instrument.subinstruments[imed].d['filter_nu']/1e9, idet))
        plt.tight_layout()

    return thetas, phis, vals, nus, q_instrument


def update_dict(config, instrument_type, nf_sub, nside, dictfilename='qubic/qubic/dicts/pipeline_demo.dict', debug=True):
    d = qubicDict()
    d.read_from_file(dictfilename)
    d['config'] = config
    
    d['instrument_type'] = instrument_type
    d['nf_sub'] = nf_sub
    d['debug'] = debug
    d['nside'] = nside

    d['beam_shape'] = 'gaussian'  # can be 'gaussian', 'fitted_beam' or 'multi_freq'  
    d['synthbeam'] = None         # we put nothing
    d['use_synthbeam_fits_file'] = False
    d['synthbeam_fraction'] = 1
    d['synthbeam_kmax'] = 1
    return d

# from https://stackoverflow.com/questions/8151300/ignore-case-in-glob-on-linux
def insensitive_glob(pattern): # glob.glob but insensitive to case
    def either(c):
        return f'[{c.lower()}{c.upper()}]' if c.isalpha() else c
    return glob.glob(''.join(map(either, pattern)))

def find_sign_change(x, y):
    sign_ = y[:-1]*y[1:] # the only negative values are when the sign changes
    sign_change = np.zeros_like(y, dtype=bool) # to keep the same number of points as y
    sign_change[:-1] = sign_ <= 0 # the zero case has to be taken into account even if unlikely
    where_change = np.nonzero(sign_change)[0] #np.argwhere(sign_change)
    return where_change

def observation_dirs(ObsDate, rise_or_set, datadir):
    """Function that returns the folders containing the chosen observation.

    It uses astropy to find the folders using the start date and Moon rising
    or setting information. The location of the telescope is La Puna.

    Parameters
    ----------
    ObsDate : str
        The date of the start of the observation (UTC) in the format "yyyy-mm-dd".
    rise_or_set: str
        Wether the Moon is rising ("rise") or setting ("set").
    datadir: str
        The data directory where the dated folders are.
    min_el, max_el: float
        The minimum and maximum Moon elevations to take into account in degrees.
        Default is respectively 15° and 85 degrees.
        
    Returns
    -------
    datafiles: numpy 1D array
        The array containing the paths to the data files of the observation.
    """

    # all datasets:
    # july 2026
    # ["2026-07-23_rise",
    # "2026-07-24_set", "2026-07-24_rise",
    # "2026-07-25_set", "2026-07-25_rise",
    # "2026-07-26_set", "2026-07-26_rise",
    # "2026-07-27_set", "2026-07-27_rise",
    # "2026-07-28_set", "2026-07-28_rise",
    # "2026-07-29_set"]
    # august 2026
    # ["2026-08-25_rise",
    # "2026-08-26_set", "2026-08-26_rise",
    # "2026-08-27_set", "2026-08-27_rise",
    # "2026-08-28_set", "2026-08-28_rise",
    # "2026-08-29_set"]

    LaPuna_QUBIC = {"lat":-24.186583*u.deg,
                "lon":-66.478*u.deg,
                "height":4869*u.m,
                "UTC_Offset":-3*u.hour}
    ObsSite = LaPuna_QUBIC
    
    ObsSite = EarthLocation(lat=ObsSite["lat"], lon=ObsSite["lon"], height=ObsSite["height"])

    # from the beginning of the day and for 48 hours
    tinit = Time(ObsDate + "T00:00:00", format='isot', scale='utc')
    two_days = tinit + np.arange(86400 * 2)*u.second # every second

    ### Moon
    moon_gcrs = get_body('Moon', two_days, ObsSite)
    moon_azel = moon_gcrs.transform_to(AltAz(obstime=two_days, location=ObsSite))

    moon_el = moon_azel.alt.deg
    two_days_unix = two_days.to_value("unix")


    ### we find the elevation extrema in order to separate Moon rising and setting
    el_vel = get_vel(two_days, moon_el, order=2)
    sign_ = el_vel[:-1]*el_vel[1:] # the only negative values are when the sign changes
    sign_change = np.zeros_like(el_vel, dtype=bool) # to keep the same number of points as el_vel
    sign_change[:-1] = sign_ <= 0 # the zero case has to be taken into account even if unlikely
    where_change = np.argwhere(sign_change)

    if el_vel[sign_change][0] == 0: # unlikely but we never know
        sign_check = el_vel[where_change[0] - 1]
    else:
        sign_check = el_vel[sign_change][0]
    if sign_check < 0: # that means the first observation started that day should be rising
        order = np.array(["rise", "set"])
    else:
        order = np.array(["set", "rise"])

    def get_extremum(xleft, yleft, xright, yright): # linear approx
        b = (yright - yleft*xright/xleft)/(1 - xright/xleft)
        a = (yleft - b)/xleft
        return -b/a

    extrem_time_unix = get_extremum(two_days_unix[where_change], el_vel[where_change], two_days_unix[where_change + 1], el_vel[where_change + 1])

    # we get all the files
    alldirs = np.array(insensitive_glob(datadir + '/*/*moon*')) # might not be clever when the number of datasets reaches a great number
    allnames = [dir.split(os.sep)[-1] for dir in alldirs] # the name of the files
    alldates = np.array([name[:19].replace("_", "T").replace(".", ":") for name in allnames]) # the starting time of each observation file
    alldates_unix = Time(alldates, format='isot', scale='utc').to_value("unix") # converted to unix time
    argsort_dates = np.argsort(alldates_unix)
    alldirs = alldirs[argsort_dates] # dirs sorted
    alldates_unix = alldates_unix[argsort_dates] # times sorted (.sort() returns None no idea why)

    # we select the files between the two extrema
    first_extr = int(np.argwhere(order == rise_or_set)) # the extremum index just before the start of the acquisition
    # first_extr = np.nonzero(order == rise_or_set) # the extremum index just before the start of the acquisition
    selection = np.logical_and(extrem_time_unix[first_extr] < alldates_unix, alldates_unix < extrem_time_unix[first_extr + 1])
    selected_files = alldirs[selection]

    print(selected_files)
    return selected_files

def do_aperture_photometry(map_in, rot, reso): # not tested
    """
    The aperture photometry is to be done on maps with the Moon at zenith
    for the detector studied. In this basic version, only the order 0 is considered.
    """
    radius_moon = 1 #deg
    radius_aperture = 2 #deg
    xsize = len(map_in)
    gnom_proj = hp.projector.GnomonicProj(rot=rot, reso=reso, xsize=xsize)

    x, y = gnom_proj.ij2xy()
    azt_proj, elt_proj = gnom_proj.xy2ang(x=x.flatten(), y=y.flatten(), lonlat=True)
    azt_proj = azt_proj.reshape(np.shape(x))
    elt_proj = elt_proj.reshape(np.shape(y))

    area_aperture_photo = np.logical_and(elt_proj > radius_moon, elt_proj < radius_aperture)
    data_aperture = map_in[area_aperture_photo]

    sky_emission = np.median(data_aperture)

    # check if data is normally distributed
    if True:
        plt.figure()
        plt.hist(data_aperture)
        plt.show()

    # we compute the standard error of the median
    sky_emission_error = 1.2533 * np.std(data_aperture) / np.sum(area_aperture_photo)

    return sky_emission, sky_emission_error


def do_aperture_photometry_advanced(map_in, theo_sb, rot, reso, const_el=False): # not tested
    """
    The aperture photometry is to be done on maps with the Moon at zenith
    for the detector studied. For this "advanced" version, we take the sky close to each peak.

    Should we compute the sky emission for each peak and frequency, or just each peak?
    Considering that it varies with elevation, in the case of a non-constant-elevation scan,
    this could have an effect.
    But then the error would be larger given the smaller number of pixels used to estimate it.
    """
    rho = 1
    radius_moon = 1 #deg
    radius_aperture = 2 #deg
    xsize = len(map_in)
    gnom_proj = hp.projector.GnomonicProj(rot=rot, reso=reso, xsize=xsize)

    thetas = theo_sb[0] # shape (n_nus, npeaks)
    phis = theo_sb[1]
    n_nus = len(thetas)
    n_peaks = len(thetas[0])

    x, y = gnom_proj.ij2xy()
    azt_proj, elt_proj = gnom_proj.xy2ang(x=x.flatten(), y=y.flatten(), lonlat=True)
    azt_proj = azt_proj.reshape(np.shape(x))
    elt_proj = elt_proj.reshape(np.shape(y))

    pix_pos = spherical2cartesian(rho, azt_proj, elt_proj, coord="horizontal", axis="last")

    if const_el: # if the scan is done at constant elevation, we don't need to differenciate
                 # between the frequencies
        sky_emission = np.zeros(n_peaks)
    else:
        raise ValueError("Only constant elevation scan implemented so far.")
        sky_emission = np.zeros((n_peaks, n_nus))
    sky_emission_error = np.zeros_like(sky_emission)

    for i_peak in range(n_peaks):
        mask_close = np.ones_like(map_in, dtype=bool) # too close = 0, if too close to one peak = 0
        mask_far = np.zeros_like(map_in, dtype=bool) # too far = 0, if close enough to one peak > 0
        for i_nu in range(n_nus):
            el_peak = 90 - np.degrees(thetas[i_nu, i_peak])
            az_peak = np.degrees(phis[i_nu, i_peak])
            # we compute an angular distance from the cartesian positions
            peak_pos = spherical2cartesian(rho, az_peak, el_peak, coord="horizontal", axis="last")
            dist_peak = np.abs(np.degrees(dist_angle(pix_pos, peak_pos)))
            mask_close = np.logical_and(mask_close, dist_peak > radius_moon)
            mask_far = np.logical_or(mask_far, dist_peak < radius_aperture)
            # area_aperture_photo = np.logical_and(dist_peak > radius_moon, dist_peak < radius_aperture)
            # data_aperture = map_in[area_aperture_photo]
        area_aperture_photo = np.logical_and(mask_close, mask_far)
        data_aperture = map_in[area_aperture_photo]

        # const_el case
        sky_emission[i_peak] = np.median(data_aperture)
        # we compute the standard error of the median as if it were normally distributed
        sky_emission_error[i_peak] = 1.2533 * np.std(data_aperture) / np.sum(area_aperture_photo)

    # sky_emission = np.median(data_aperture)
    # # we compute the standard error of the median as if it were normally distributed
    # sky_emission_error = 1.2533 * np.std(data_aperture) / np.sum(area_aperture_photo)

    return sky_emission, sky_emission_error

def bin_data_by_scan(xdata, ydata, scantype): # scantype is positive, negative or null
    bin_id = np.zeros(len(scantype))
    bin_id_i = -1
    scan_sorted = []
    previous = -100
    current_scan = []
    for i in range(len(scantype)):
        if scantype[i] != previous:
            bin_id_i += 1
            previous = scantype[i]
            current_scan.append(bin_id_i)
            if scantype[i] == 0:
                scan_sorted.append(current_scan)
                current_scan = []
                current_scan.append(bin_id_i)
        bin_id[i] = bin_id_i

    values_df = pd.DataFrame(xdata)
    values_binned = values_df.groupby([bin_id])
    # binned_xdata = np.array(values_binned[0].median())
    binned_xdata = np.array(values_binned[0].mean())

    values_df = pd.DataFrame(ydata)
    values_binned = values_df.groupby([bin_id])
    binned_ydata = np.array(values_binned[0].mean())

    return binned_xdata, binned_ydata, scan_sorted, bin_id


def stack_TOD_scan_by_scan(data_TOD_notshift, doplot=False): # will be copy pasted from ipynb file
    """Function to check for a possible shift between science time and mount
    time by comparing the azel positions of the telescope with the variations
    of the signal measured by a TES strongly responsive to the Earth magnetic
    field.

    TES 65, 127 and 128 seem to be blind TES with squids strongly correlated
    to the Earth magnetic field and are therefore useful to compute tshift.
    Other TES also have squids sensitive to the Earth magnetic field and
    can be used.
    """

    # data_TOD_notshift is the data read without tshift correction
    tt, tinit, alltod, QPidx, azt, elt, _, _, scantype, Tbath, T1K, ifile, thk, az, el = data_TOD_notshift

    # parameters of pmf.butter_lowpass_filter
    highcut = 4/110 # Hz
    fs = 160 # Hz
    order = 1

    # TESNum = 128 # TES 128 is blind and responsive to the Earth magnitic field
    # iTES = TESNum - 1
    # tod = alltod[iTES, :]
    ### We will stack multiple TOD in order to maximise the SNR
    all_ind = np.arange(len(alltod[0]))
    # stacked_tod = np.zeros_like(alltod[0])
    blind_TES_ok = [[65, 127, 128], # just looked by eye, might not be the best list (blind TES that see magnitic field)
                    [1, 1, 1]]      # sign of the signal of the TES
    all_TES_ok = [[32, 33, 34, 61, 62, 64, 65, 66, 67, 93, 94, 95, 96, 98, 99, 100, 125, 127, 128], # these are not blind but they see the Earth magnetic field # "2026-07-24_rise" or "2026-07-25_set"
                [-1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1]] # sign of the signal of the TES
    group_2026_07_23_rise = [[65, 97, 127, 128], # not as good a result as with blind_TES_ok
                            [1, -1, 1, 1]]
    
    # it seems that some TES are more sensitive to temperature change than others --> difference between blind TES and not blind ones?
    group_2026_07_27_set = [[32, 33, 34, 93, 94, 95, 96, 98, 99, 65], # these are not blind but they see the Earth magnetic field # "2026-07-24_rise" or "2026-07-25_set"
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, 1]] # sign of the signal of the TES
    
    plot_scan = 131 #150 2026-07-27_set temperature change
    only_plot = False # if we only want the plot

    chosen_group = all_TES_ok # blind_TES_ok works for most, but not precise enough
    chosen_TES = chosen_group[0]
    sign = chosen_group[1]
    fig, ax = plt.subplots(1, 1, figsize=(15, 7))
    tod_iTES = np.zeros((len(alltod[0]), len(chosen_TES)))
    for i_T, TESNum in enumerate(chosen_TES):
        iTES = TESNum - 1
        # tod_iTES = np.zeros_like(alltod[0])
        for i_scan in range(1, int(np.max(scantype)) + 1): # the time shift we are looking for should be small so we can do this
            if i_scan != plot_scan and only_plot:
                continue
            # we want to include the zero in-between (except for normalisation purposes)
            where_iscan = np.nonzero(np.logical_or(scantype == i_scan, scantype == -i_scan))
            if i_scan < np.max(scantype): # we also get the zero before the next scan
                where_next_iscan = np.nonzero(np.logical_or(scantype == i_scan + 1, scantype == -(i_scan + 1)))
            first_ind = np.min(where_iscan)
            last_ind = max(np.max(where_iscan) + 1, np.min(where_next_iscan))# - 1
            mask_iscan = np.logical_and(all_ind >= first_ind, all_ind < last_ind)
            mask_iscan_nonzero = np.logical_and(mask_iscan, scantype != 0) # in order to be able to normalise with less issues
            smooth_scan = pmf.butter_lowpass_filter(alltod[iTES, mask_iscan], highcut, fs, order)
            tod_iTES[mask_iscan, i_T] = sign[i_T]*(alltod[iTES, mask_iscan] - np.mean(alltod[iTES, mask_iscan_nonzero]))/np.std(alltod[iTES, mask_iscan_nonzero])
            if i_scan == plot_scan:
                ax.plot(tt[mask_iscan], tod_iTES[mask_iscan, i_T], label="{}".format(TESNum))

    mean_tod = np.mean(tod_iTES, axis=1)
    median_tod = np.median(tod_iTES, axis=1)
    # fancier mean with sigma clipping
    masked_tod = sigma_clip(tod_iTES, sigma=2, axis=1)
    mean_tod_sigclip = np.mean(masked_tod, axis=1)

    if doplot:
        # plt.xlim(6800, 6950)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("normalised TOD")
        ax.legend(loc="upper right")
        file_name = "figures/{}_{}_norm_TOD_stack.pdf".format(ObsDate, scanning)
        fig.savefig(file_name, dpi=150)
        print("Saved {}".format(file_name))
        plt.show()

    tod = mean_tod_sigclip
    # tod = median_tod
    tod_comp = alltod[128 - 1, :] # we compare with TES 128

    return mean_tod, mean_tod_sigclip, median_tod, binned_tod



def check_tshift_live(data_TOD_notshift, doplot=False): # will be copy pasted from ipynb file
    """Function to check for a possible shift between science time and mount
    time by comparing the azel positions of the telescope with the variations
    of the signal measured by a TES strongly responsive to the Earth magnetic
    field.

    TES 65, 127 and 128 seem to be blind TES with squids strongly correlated
    to the Earth magnetic field and are therefore useful to compute tshift.
    Other TES also have squids sensitive to the Earth magnetic field and
    can be used.
    """
    # it might be better to treat each data file separately to avoid issues with the gaps in the data
    # tod_peaks = pmf.find_sign_change(tt, deriv_smooth_tod) # the indices of the peaks in the TOD
    # az_peaks = pmf.find_sign_change(tt, az_vel) # the indices of the peaks in azimuth
    # az_peaks = np.array([az_peak for az_peak in az_peaks if az_vel[az_peak] != 0]) # we remove where az_vel == 0
    tod_peaks = []
    az_peaks = []
    ifile_peaks = []
    tt_start_file = []
    for i_f in range(np.max(ifile) + 1):
        if i_f not in ifile:
            continue
        mask_f = ifile == i_f
        arg_f = np.nonzero(mask_f)[0]
        tt_start_file.append(tt[arg_f[0]])
        margin = 30*160 # points excluded because too close from file change
        first_ind_f = arg_f[0] + margin
        last_ind_f = arg_f[-1] - margin + 1
        tt_f = tt[first_ind_f:last_ind_f]
        deriv_smooth_tod_f = deriv_smooth_tod[first_ind_f:last_ind_f]
        az_vel_f = az_vel[first_ind_f:last_ind_f]
        tod_peaks_f = pmf.find_sign_change(tt_f, deriv_smooth_tod_f) # the indices of the peaks in the TOD
        dist_ok = 10*160
        tod_peaks_f += first_ind_f
        tod_peaks_f = np.array([tod_peak for tod_peak in tod_peaks_f if np.all(az_vel[max(0, tod_peak - dist_ok):tod_peak + dist_ok] != 0)]) # we remove where az_vel == 0
        tod_peak = tod_peaks_f[0]
        az_peaks_f = pmf.find_sign_change(tt_f, az_vel_f) # the indices of the peaks in azimuth
        az_peaks_f += first_ind_f
        az_peaks_f = np.array([az_peak for az_peak in az_peaks_f if np.all(az_vel[max(0, az_peak - dist_ok):az_peak + dist_ok] != 0)]) # we remove where az_vel == 0
        tod_peaks.append(tod_peaks_f)
        az_peaks.append(az_peaks_f)
        ifile_peaks.append(np.full_like(tod_peaks_f, i_f))
    tod_peaks = np.concatenate(tod_peaks)
    az_peaks = np.concatenate(az_peaks)
    ifile_peaks = np.concatenate(ifile_peaks)
    tt_start_file = np.array(tt_start_file)

    print(np.shape(tod_peaks))
    print(np.shape(az_peaks))

    # # to check where is the problem if not the same number of peaks between tod and az
    # fig, ax = plt.subplots()
    # ax.scatter(tt[tod_peaks], az_vel[tod_peaks], c="b", s=8, label="az_vel[tod_peaks]")
    # ax.scatter(tt[az_peaks], az_vel[az_peaks], c="r", s=4, label="az_vel[az_peaks]")
    # plt.legend()
    # plt.show()


    tshifts = tt[az_peaks] - tt[tod_peaks] # works only if the peaks are the same
    print("shape tshifts", np.shape(tshifts))

    median_tshift = np.median(tshifts)

    # Moving average
    win_w = 15 # npoints
    window_width = win_w + win_w%2 # ensures an even number
    tshifts_mean = np.pad(tshifts, int(window_width/2) , mode='edge')
    cumsum_vec = np.cumsum(tshifts_mean)
    tshifts_mean = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width

    # let's also see the moving median over nb_med tshifts
    nb_med = 15
    shape_pad_start = lambda i: max(0, nb_med//2 - i)
    shape_pad_end = lambda i: max(0, i - nb_med//2)
    padding_start = lambda x: np.full(x, tshifts[0]) # use it with shape_pad_start result
    padding_end = lambda x: np.full(x, tshifts[-1]) # use it with shape_pad_end result
    last_ind = lambda x: None if x == 0 else -x # use it with shape_pad_start result

    # and now the median/mean? of the shift for each data file
    tt_binned, tshifts_binned, test_f, test_f2 = pmf.bin_data_by_scan(tt[az_peaks], tshifts, ifile_peaks)
    print(test_f)
    print(test_f2)

    # a more clever way to do it is just to create a padded array and THEN do the intermediate array

    intermediate_array = np.array([np.concatenate([padding_start(shape_pad_start(i)), tshifts[shape_pad_end(i):last_ind(shape_pad_start(i))], padding_end(shape_pad_end(i))]) for i in range(nb_med)])
    tshifts_med = np.median(intermediate_array, axis=0)

    tshifts_mean_alltt = np.interp(tt, tt[tod_peaks], tshifts_mean)
    tshifts_med_alltt = np.interp(tt, tt[tod_peaks], tshifts_med)
    tshifts_alltt = np.interp(tt, tt[tod_peaks], tshifts)

    tshifts_binned_alltt = interp1d(tt_start_file, tshifts_binned, kind='previous', fill_value='extrapolate')(tt)

    file_name = "{}_tshifts_med_alltt.npy".format(ObsName)
    np.save(file_name, tshifts_med_alltt)
    print("Saved {}".format(file_name))