from __future__ import division, print_function
import os
import qubic
<<<<<<< HEAD
from qubicpack.utilities import Qubic_DataDir
=======
>>>>>>> 817389f4cc3163541fa042c883a3919ba9169a19
import healpy as hp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys

from qubicpack.pixel_translation import plot_id_focalplane

plot_id_focalplane()


def make_detector_subset_instrument(q, dets_FPindex, multiband=True):
    '''
    This function makes an instrument with just a subset of detectors.
    Be careful it modifies the input instrument !!

    INPUTS
    instrument - qubic.QubicMultibandInstrument or qubic.Instrument
    FPindex - list of int - detector FP index (ex: 594 is the 0 element in q.detector.index)

    '''
    if multiband:
        for i in range(q.nsubbands):
            dets_indices = [int(np.where(q[i].detector.index == FPindex)[0])
                            for FPindex in dets_FPindex]
            print(dets_FPindex, dets_indices, q[i].detector.index[dets_indices])
            q[i].detector = q[i].detector[dets_indices]
    else:
        dets_indices = [int(np.where(q.detector.index == FPindex)[0])
                        for FPindex in dets_FPindex]
        print(dets_FPindex, dets_indices, q.detector.index[dets_indices])
        q.detector = q.detector[dets_indices]

    return


stokes = ['I', 'Q', 'U']

mpl.style.use('classic')
name = 'test_scan_source'
resultDir = '%s' % name
os.makedirs(resultDir, exist_ok=True)

<<<<<<< HEAD
alaImager = True     # if True, the beam will be a simple gaussian
component = 1           # Choose the component number to plot (IQU)
oneComponent = False    # True if you want to study only I component, otherwise False if you study IQU
sel_det = True          # True if you want to use one detector, False if you want to use all detectors in focal plane
dets_FPindex = [594]              # if sel_det == True, choose detector number

# Dictionnary
global_dir = Qubic_DataDir(datafile='instrument.py', datadir=os.environ['QUBIC_DATADIR'])
print('global_dir: ', global_dir)
d = qubic.qubicdict.qubicDict()
d.read_from_file(global_dir + '/dicts/global_source_oneDet.dict')
=======
alaImager = True        # if True, the beam will be a simple gaussian
component = 1           # Choose the component number to plot (IQU)
oneComponent = False    # True if you want to study only I component, otherwise False if you study IQU
sel_det = True          # True if you want to use one detector, False if you want to use all detectors in focal plane
dets_FPindex = [594]    # if sel_det == True, choose detector number

# Dictionnary
d = qubic.qubicdict.qubicDict()
d.read_from_file('global_source_oneDet.dict')
>>>>>>> 817389f4cc3163541fa042c883a3919ba9169a19

# Scene
s = qubic.QubicScene(d)

# Instrument
q = qubic.QubicInstrument(d)
if sel_det:
    make_detector_subset_instrument(q, dets_FPindex, multiband=False)

# Pointing
p = qubic.get_pointing(d)

fix_azimuth = d['fix_azimuth']
print('fix_azimuth', fix_azimuth)

plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(p.time, p.azimuth, 'bo')
plt.ylabel('Azimuth')
plt.subplot(412)
plt.plot(p.time, p.elevation, 'bo')
plt.ylabel('Elevation')
plt.subplot(413)
plt.plot(p.time, p.pitch, 'bo')
plt.ylabel('pitch angle')
plt.subplot(414)
plt.plot(p.time, p.angle_hwp, 'bo')
plt.ylabel('HWP angle')
plt.savefig(resultDir+'/%s_pointing.png'%name, bbox_inches='tight')

# Make a point source
npix = 12 * d['nside'] ** 2
m0 = np.zeros(npix)
x0 = np.zeros((npix, 3))
id = hp.pixelfunc.ang2pix(d['nside'], fix_azimuth['el'], fix_azimuth['az'],
                          lonlat=True)
source = m0 * 0
source[id] = 1
arcToRad = np.pi / (180 * 60.)
source = hp.sphtfunc.smoothing(source, fwhm=30 * arcToRad)
x0[:, component] = source
hp.mollview(x0[:, component])
plt.show()

if p.fix_az:
    center = (fix_azimuth['az'], fix_azimuth['el'])
else:
    center = qubic.equ2gal(d['RA_center'], d['DEC_center'])

# Make TOD
a = qubic.QubicAcquisition(q, p, s, d)
TOD = a.get_observation(x0, noiseless=True, convolution=False)

plt.plot(TOD[0, :])
plt.xlabel('pointing index')
plt.ylabel('TOD')
plt.show()

# Map making
nf_sub_rec = 1
if alaImager:

    d['synthbeam_kmax'] = 0
    if oneComponent:
        d['kind'] = 'I'
    q = qubic.QubicInstrument(d)
    if sel_det:
        make_detector_subset_instrument(q, dets_FPindex, multiband=False)
arec = qubic.QubicAcquisition(q, p, s, d)

maps_recon, nit, error = arec.tod2map(TOD, d, cov=None)
print(maps_recon.shape)
hp.gnomview(maps_recon[:, component], rot=center, reso=15)


# Coverage
cov = arec.get_coverage()
hp.gnomview(cov, rot=center, reso=15)
maxcov = np.max(cov)
unseen = cov < maxcov * 0.1

# Convolved maps
TOD_useless, maps_convolved = arec.get_observation(x0)
maps_convolved = np.array(maps_convolved)

diffmap = maps_convolved - maps_recon
maps_convolved[unseen, :] = hp.UNSEEN
maps_recon[unseen, :] = hp.UNSEEN
diffmap[unseen, :] = hp.UNSEEN

xname = ''
if alaImager == True:
    nf_sub_rec = 1
    xname = 'alaImager'

for istokes in [0, 1, 2]:
    plt.figure(istokes, figsize=(12, 12))
    xr = 0.1 * np.max(maps_recon[:, 0])
    im_in = hp.gnomview(maps_convolved[:, istokes], rot=center, reso=5, sub=(1, 2, 1), min=-xr,
                        max=xr, title='Input ' + stokes[istokes],
                        return_projected_map=True)
    # np.savetxt(resultDir + '/in_%s_%s_%s.dat' % (name, stokes[istokes], xname), im_in)
    im_old = hp.gnomview(maps_recon[:, istokes], rot=center, reso=5, sub=(1, 2, 2), min=-xr,
                         max=xr, title='Output ' + stokes[istokes],
                         return_projected_map=True)
    # np.savetxt(resultDir + '/out_%s_%s_%s.dat' % (name, stokes[istokes], xname), im_old)

    # plt.savefig(resultDir + '/%s_map_%s_%s.png' % (name, stokes[istokes], xname), bbox_inches='tight')

plt.figure(figsize=(15,8))
count = 1

if d['kind'] == 'I':
    xr = 0.01 * np.max(maps_recon[:])
    im_old = hp.gnomview(maps_recon[:], rot=center, reso=5, min=-xr, max=xr, title='Output ', return_projected_map=True,
                         hold=True, xsize=500)
    plt.show()
else:
    for istokes in range(3):
        plt.subplot(1, 3, count)
        xr = 0.009
        im_old = hp.gnomview(maps_recon[:, istokes], xsize=500, rot=center, reso=5, min=-xr, max=xr,
                             title='Output ' + stokes[istokes], return_projected_map=True, hold=True)
        count += 1
    plt.show()

    P = np.sqrt(maps_recon[:, 1] ** 2 + maps_recon[:, 2] ** 2)
    sb = q.get_synthbeam(s)

    plt.figure(figsize=(15, 8))

    plt.subplot(1, 2, 1)
    hp.gnomview(P, xsize=500, rot=center, reso=5, title='Output P', return_projected_map=True, hold=True)
    plt.subplot(1, 2, 2)
    hp.gnomview(sb[0], rot=[0, 90], xsize=500, reso=5, title='Synthetic Beam', return_projected_map=True, hold=True)
    plt.show()
