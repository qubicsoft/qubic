import numpy as np
from matplotlib.pyplot import *
import glob
import time
import datetime

from Calibration import fibtools as ft
from Calibration.plotters import *
from qubicpack import qubicpack as qp

from pysimulators import FitsArray
import matplotlib.mlab as mlab
import scipy.ndimage.filters as f

#### Calsource Stepping
tf, f = np.loadtxt('freq_modulation.txt').T

#### Directory where the files are located
dir = '/Users/hamilton/Qubic/ExternalSource/2019-02-01_17.24.02__Sweepinfrequency/'
init_time = time.mktime(datetime.datetime(2019, 2, 1, 17, 24).timetuple()) - 3600
# init_time = 1549034752.612113


#### Now find the az and el corresponding to each directory and prepare the files to read
subdirs = glob.glob(dir + '*')
as1 = glob.glob(dir + 'Sums/*asic1*.fits')[0]
as2 = glob.glob(dir + 'Sums/*asic2*.fits')[0]

time_ranges = np.zeros((2, len(tf)))
time_ranges[0, :] = tf - init_time
time_ranges[1, :] = np.roll(time_ranges[0, :], -1)
time_ranges[1, -1] = time_ranges[0, -1] + 60

### Analyse one to define the list of pixok
name = 'ExtSrc(Auto)'
fnum = 150
fff = 0.333
dc = 0.33

#### Saturation value: 2.235174179076e-08

tt, folded, okfinal, params, err, chi2, ndf = ft.run_asic(fnum, 0, fff, dc, as1, 1, name=name,
                                                          timerange=time_ranges[:, 0], initpars=None, lowcut=0.05,
                                                          highcut=15.,
                                                          rangepars=[[0., 1.], [0., 0.5], [0., 1. / fff], [0., 5000.]],
                                                          reselect_ok=False,
                                                          okfile='ScanAz2019-02-01_nuscan_OK_Asic1.fits')

tt, folded, okfinal, params, err, chi2, ndf = ft.run_asic(fnum, 0, fff, dc, as2, 2, name=name,
                                                          timerange=time_ranges[:, 0], initpars=None, lowcut=0.05,
                                                          highcut=15.,
                                                          rangepars=[[0., 1.], [0., 0.5], [0., 1. / fff], [0., 5000.]],
                                                          reselect_ok=False,
                                                          okfile='ScanAz2019-02-01_nuscan_OK_Asic2.fits')

## problems: 155, 142.5, 137.5
##            11     5      3

#### Now loop on asics
amps = np.zeros((256, len(tf)))
taus = np.zeros((256, len(tf)))
erramps = np.zeros((256, len(tf)))
errtaus = np.zeros((256, len(tf)))
for i in xrange(len(tf)):
    asic = as1
    tt, folded, okfinal, params, err, chi2, ndf = ft.run_asic(fnum, 0, fff, dc, asic, 1, name=name, doplot=False,
                                                              timerange=time_ranges[:, i], initpars=None, lowcut=0.05,
                                                              highcut=15.,
                                                              rangepars=[[0., 1.], [0., 0.5], [0., 1. / fff],
                                                                         [0., 5000.]],
                                                              okfile='ScanAz2019-02-01_nuscan_OK_Asic1.fits')
    params[~okfinal, :] = np.nan
    amps[:128, i] = params[:, 3]
    erramps[:128, i] = err[:, 3]
    taus[:128, i] = params[:, 1]
    errtaus[:128, i] = err[:, 1]

    asic = as2
    tt, folded, okfinal, params, err, chi2, ndf = ft.run_asic(fnum, 0, fff, dc, asic, 2, name=name, doplot=False,
                                                              timerange=time_ranges[:, i], initpars=None, lowcut=0.05,
                                                              highcut=15.,
                                                              rangepars=[[0., 1.], [0., 0.5], [0., 1. / fff],
                                                                         [0., 5000.]],
                                                              okfile='ScanAz2019-02-01_nuscan_OK_Asic2.fits')
    params[~okfinal, :] = np.nan
    amps[128:, i] = params[:, 3]
    erramps[128:, i] = err[:, 3]
    taus[128:, i] = params[:, 1]
    errtaus[128:, i] = err[:, 1]

cutval = 200000

allimg = np.zeros((len(tf), 17, 17)) + np.nan
for i in xrange(len(tf)):
    allimg[i, :, :] = ft.image_asics(all1=amps[:, i])
    bad = allimg[i, :, :] > cutval
    allimg[i, :, :][bad] = np.nan
    ok = np.isfinite(allimg[i, :, :])
    mm = np.mean(allimg[i, :, :][ok])
    clf()
    imshow(allimg[i, :, :] / mm, vmin=0, vmax=4, cmap='viridis')
    colorbar()
    title(r'$\nu$={} GHz'.format(f[i]))
    show()
    savefig('imgscan01022019_nu_{}.png'.format(f[i]))
    # raw_input('Press a key')
