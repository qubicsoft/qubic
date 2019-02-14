from Calibration import fibtools as ft
from Calibration.plotters import *

import numpy as np
from matplotlib.pyplot import *
from pysimulators import FitsArray
import matplotlib.mlab as mlab
import scipy.ndimage.filters as f
import glob

from qubicpack import qubicpack as qp

#### Directory where the files for various angles are located
# basedir = '/Users/hamilton/Qubic/ExternalSource'
basedir = '/home/louisemousset/QUBIC/Qubic_work/Calibration'

# dir = basedir + '/ScanAz2019-01-30/'
# dir = basedir + '/ScanAz2019-01-31/'
# dir = basedir + '/ScanAz2019-01-31-Long/'

dir = basedir + '/2019-01-31/Scan_az/'

#### Now find the az and el corresponding to each directory and prepare the files to read
subdirs = glob.glob(dir + '*')
as1 = glob.glob(dir + '*/Sums/*asic1*.fits')
as2 = glob.glob(dir + '*/Sums/*asic2*.fits')

els = np.zeros(len(subdirs))
azs = np.zeros(len(subdirs))
for i in xrange(len(subdirs)):
    els[i], azs[i] = [float(s) for s in subdirs[i].split('_') if ft.isfloat(s)]

# a = qp()
# a.read_qubicstudio_dataset(subdirs[0])

#### Sort files with azimuth and elevation in ascending order
order = np.argsort(azs)
az = azs[order]
el = els[order]

as1 = np.array(as1)[order]
as2 = np.array(as2)[order]
subdirs = np.array(subdirs)[order]

#### Analyse one to define the list of pixok
name = 'ExtSrc(Auto)'
fnum = 150
fff = 0.333
dc = 0.33

# Saturation value: 2.235174179076e-08

tt, folded, okfinal, params, err, chi2, ndf = ft.run_asic(fnum, 0, fff, dc, as1[0], 1, name=name, initpars=None,
                                                          lowcut=0.05, highcut=15.,
                                                          rangepars=[[0., 1.], [0., 0.5], [0., 1. / fff], [0., 1000.]],
                                                          stop_each=False, reselect_ok=False,
                                                          okfile='ScanAz2019-01-30_OK_Asic1.fits')

tt, folded, okfinal, params, err, chi2, ndf = ft.run_asic(fnum, 0, fff, dc, as2[0], 2, name=name, initpars=None,
                                                          lowcut=0.05, highcut=15., reselect_ok=False,
                                                          rangepars=[[0., 1.], [0., 0.5], [0., 1. / fff], [0., 1000.]],
                                                          okfile='ScanAz2019-01-30_OK_Asic2.fits')

#### Now loop on asics
amps = np.zeros((256, len(az)))
taus = np.zeros((256, len(az)))
erramps = np.zeros((256, len(az)))
errtaus = np.zeros((256, len(az)))
for i in xrange(len(as1)):
    asic = as1[i]
    tt, folded, okfinal, params, err, chi2, ndf = ft.run_asic(fnum, 0, fff, dc, asic, 1, name, doplot=False,
                                                              initpars=None, lowcut=0.05, highcut=15.,
                                                              rangepars=[[0., 1.], [0., 0.5], [0., 1. / fff],
                                                                         [0., 1000.]],
                                                              okfile='ScanAz2019-01-30_OK_Asic1.fits')
    params[~okfinal, :] = np.nan
    amps[:128, i] = params[:, 3]
    erramps[:128, i] = err[:, 3]
    taus[:128, i] = params[:, 1]
    errtaus[:128, i] = err[:, 1]

    asic = as2[i]
    tt, folded, okfinal, params, err, chi2, ndf = ft.run_asic(fnum, 0, fff, dc, asic, 2, name, doplot=False,
                                                              initpars=None, lowcut=0.05, highcut=15.,
                                                              rangepars=[[0., 1.], [0., 0.5], [0., 1. / fff],
                                                                         [0., 1000.]],
                                                              okfile='ScanAz2019-01-30_OK_Asic2.fits')
    params[~okfinal, :] = np.nan
    amps[128:, i] = params[:, 3]
    erramps[128:, i] = err[:, 3]
    taus[128:, i] = params[:, 1]
    errtaus[128:, i] = err[:, 1]

cutval = 1000

allimg = np.zeros((len(as1), 17, 17)) + np.nan
for i in xrange(len(as1)):
    allimg[i, :, :] = ft.image_asics(all1=amps[:, i])
    bad = allimg[i, :, :] > cutval
    allimg[i, :, :][bad] = np.nan
    clf()
    imshow(allimg[i, :, :], vmin=0, vmax=1000, cmap='viridis')
    colorbar()
    title('$\Delta$az={}'.format(az[i]))
    show()
    savefig('imgscan01022019_az_{}.png'.format(1000 + az[i]))
    # raw_input('Press a key')

thepix = 93
clf()
plot(az, amps[thepix, :])

#### Trying intercalibration from old fiber data
#### (probably worthless as at the time they were superconducting and now they are not)

calibration = FitsArray('/Users/hamilton/CMB/Qubic/Fibres/calibration.fits')
calibration_restrict = FitsArray('/Users/hamilton/CMB/Qubic/Fibres/calibration_restrict.fits')

cutval = 200

allimg = np.zeros((len(as1), 17, 17)) + np.nan
allimg_c = np.zeros((len(as1), 17, 17)) + np.nan
allimg_cr = np.zeros((len(as1), 17, 17)) + np.nan
for i in xrange(len(as1)):
    allimg[i, :, :] = ft.image_asics(all1=amps[:, i])
    bad = allimg[i, :, :] > cutval
    allimg[i, :, :][bad] = np.nan

    allimg_c[i, :, :] = ft.image_asics(all1=amps[:, i] * calibration)
    bad = allimg_c[i, :, :] > cutval
    allimg_c[i, :, :][bad] = np.nan

    allimg_cr[i, :, :] = ft.image_asics(all1=amps[:, i] * calibration_restrict)
    bad = allimg_cr[i, :, :] > cutval
    allimg_cr[i, :, :][bad] = np.nan

    cmap = cm.get_cmap('viridis', 10)

    clf()
    subplot(1, 2, 1)
    imshow(allimg[i, :, :], vmin=0, vmax=75, cmap=cmap)
    colorbar()
    title('$\Delta$az={} - Not calibrated'.format(az[i]))
    subplot(1, 2, 2)
    imshow(allimg_c[i, :, :], vmin=0, vmax=75, cmap=cmap)
    colorbar()
    title('$\Delta$az={} - Calibrated'.format(az[i]))
    # subplot(2,2,3)
    # imshow(allimg_cr[i,:,:],vmin=0,vmax=75, cmap=cmap)
    # colorbar()
    # title('$\Delta$az={} - Calibrated (restr)'.format(az[i]))
    show()
    savefig('imgscan01022019_calib_az_{}.png'.format(1000 + az[i]))
    # raw_input('Press a key')
