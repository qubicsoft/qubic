#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# This code was supposed to simulate the fringes in the focal plane.
# It is not working because when we make TOD, the code doesn't care
# about closed horns.

from __future__ import division, print_function
from matplotlib.pyplot import *
import healpy as hp
from astropy.io import fits
import qubic

basedir = '/home/louisemousset/QUBIC/MyGitQUBIC'
dictfilename = basedir + '/qubic/qubic/scripts/global_source.dict'

d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)

# ========== Sweeping pointing in a fix frame ===========
p = qubic.get_pointing(d)
nptg = len(p.index)  # Pointing number

el_angles = np.unique(p.elevation)  # all different elevation angles
az_angles = np.unique(p.azimuth)  # all different azimuth angles
print('el_angles = ' + str(el_angles))
print('az_angles = ' + str(az_angles))

figure('pointing')
subplot(3, 1, 1)
plot(p.index, p.azimuth, '.')
xlabel('index pointing')
ylabel('azimuth angle')
subplot(3, 1, 2)
plot(p.index, p.elevation, '.')
xlabel('index pointing')
ylabel('elevation angle')
subplot(3, 1, 3)
plot(p.azimuth, p.elevation, '.')
xlabel('azimuth angle')
ylabel('elevation angle')

# ============= Point source ==========
az, el = d['fix_azimuth']['az'], d['fix_azimuth']['el']
print('source_coords = {0},{1}'.format(az, el))

npix = 12 * d['nside'] ** 2  # pixel number
sky = np.zeros(npix)
source_pix = hp.pixelfunc.ang2pix(d['nside'], el, az, lonlat=True)
sky[source_pix] = 1.
arcToRad = np.pi / (180 * 60.)  # conversion factor
sky = hp.sphtfunc.smoothing(sky, fwhm=30 * arcToRad)

hp.gnomview(sky, rot=(az, el))

## Sky x0 with only I

# For a Multiband instrument
x0 = np.zeros((d['nf_sub'], npix, 3))
x0[:, :, 0] = sky


# For a PolyInstrument
# x0 = np.zeros((npix, 3))
# x0[:, 0] = sky


# =========== Get TOD for the FI =========
def get_tod(d, p, x0, closed_horns=None):
    """

    Parameters
    ----------
    d : dictionnary
    p : pointing
    x0 : sky
    closed_horns : array
        index of closed horns

    Returns
    -------
    Returns an instrument with closed horns and TOD
    """
    # Polychromatic instrument model
    q = qubic.QubicMultibandInstrument(d)
    # q = qubic.QubicInstrument(d)
    if closed_horns is not None:
        for i in range(d['nf_sub']):
            for h in closed_horns:
                q[i].horn.open[h] = False

    # Scene
    s = qubic.QubicScene(d)

    # Number of sub frequencies to build the TOD
    _, nus_edge_in, _, _, _, _ = qubic.compute_freq(d['filter_nu'] / 1e9, d['nf_sub'], d['filter_relative_bandwidth'])

    # Multi-band acquisition operator
    a = qubic.QubicMultibandAcquisition(q, p, s, d, nus_edge_in)
    # a = qubic.QubicPolyAcquisition(q, p, s, d)

    tod = a.get_observation(x0, convolution=False, noiseless=True)

    return q, tod


q, tod_fi = get_tod(d, p, x0)
q[0].horn.plot()

# ========= Get indices of TES on the focal plane ============
detarray = fits.open(basedir + '/qubic/qubic/calfiles/CalQubic_DetArray_FI.fits')

tes_index = detarray['index'].data
clf()
imshow(tes_index, interpolation='nearest', origin='lower')

# size of one side of the focal plane
size_fi = tes_index.shape[0]

# Position of the TES in the array
tes_position = detarray['removed'].data
clf()
imshow(tes_position, interpolation='nearest', origin='lower')  # 0 on focal plane, 1 elsewhere

clf()
tes_position_rev = tes_position * (-1) + 1
imshow(tes_position_rev, interpolation='nearest', origin='lower')  # reverse

# Indices of TES on the focal plane, 0 outside
focal_plane = tes_index * tes_position_rev
clf()
imshow(focal_plane, interpolation='nearest', origin='lower')


# ========= Project TOD on the focal plane for each pointing ============
def plot_focalplane(size, nptg, ptg_start, ptg_stop, focal_plane, tod):
    figure('focalplane')
    tod_focal_plane = np.zeros((size, size, nptg))
    for ptg in range(ptg_start, ptg_stop + 1):
        j = 0
        for i in range(1156):
            pos = np.where(focal_plane == i)
            # print(pos[0].size)

            #  For the TES 0
            if pos[0].size == 165:
                tod_focal_plane[0, 17, ptg] = tod[j, ptg]
                j += 1

            # For other TES
            elif pos[0].size == 1:
                # print(j)
                tod_focal_plane[pos[0][0], pos[1][0], ptg] = tod[j, ptg]
                j += 1  # print(pos)

        imshow(tod_focal_plane[:, :, ptg], interpolation='nearest', origin='lower')
        title('ptg' + str(ptg) + ' el=' + str(p.elevation[ptg]) + ' az=' + str(p.azimuth[ptg]))
        pause(0.2)
    colorbar()
    return


plot_focalplane(size_fi, nptg, 30, nptg - 1, focal_plane, tod_fi)

# ========= Same thing with closed horns to see the franges ============
i = 40
j = 100

# i closed, j closed, i and j closed
q_i_close, tod_i_close = get_tod(d, p, x0, closed_horns=[i])
q_i_close[0].horn.plot()

q_j_close, tod_j_close = get_tod(d, p, x0, closed_horns=[j])
q_j_close[0].horn.plot()

q_ij_close, tod_ij_close = get_tod(d, p, x0, closed_horns=[i, j])
q_ij_close[0].horn.plot()

# Only i opened, only j open, only i and j opened
closed_horns = np.delete(np.arange(400), [i], 0)
q_i_open, tod_i_open = get_tod(d, p, x0, closed_horns=closed_horns)
q_i_open[0].horn.plot()
plot_focalplane(size_fi, nptg, 30, 30, focal_plane, np.abs(tod_i_open) ** 2)

closed_horns = np.delete(np.arange(400), [j], 0)
q_j_open, tod_j_open = get_tod(d, p, x0, closed_horns=closed_horns)
q_j_open[0].horn.plot()

closed_horns = np.delete(np.arange(400), [i, j], 0)
q_ij_open, tod_ij_open = get_tod(d, p, x0, closed_horns=closed_horns)
q_ij_open[0].horn.plot()

#  Plot the interference pattern on the focal plane for each pointing
tod_frange = tod_fi + tod_ij_close - tod_i_close - tod_j_close
plot_focalplane(size_fi, nptg, 30, 30, focal_plane, tod_frange)
