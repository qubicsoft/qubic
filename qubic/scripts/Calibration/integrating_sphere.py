from Calibration import fibtools as ft
from Calibration.plotters import *

import numpy as np
from matplotlib.pyplot import *
import glob

from pysimulators import FitsArray

#### Directory where the files for various angles are located
# basedir = '/Users/hamilton/Qubic/ExternalSource/'
basedir = '/home/louisemousset/QUBIC/Qubic_work/Calibration/datas/'


def make_amp(dir, name='Integrating_sphere', fnum=150, fff=0.15, dc=0.33):
    """

    Parameters
    ----------
    dir : str
        Directory for data files
    name : str
        Name of the data set
    fnum : float
        Frequency of the calibration source in GHz.
    fff : float
        Modulation frequency of the calibration source.
    dc : float
        Duty cycle of the modulation.

    Returns
    -------

    """
    as1 = glob.glob(dir + '/Sums/*asic1*.fits')
    as2 = glob.glob(dir + '/Sums/*asic2*.fits')

    # Asic 1
    tt, folded, okfinal, params, err, chi2, ndf = ft.run_asic(fnum, 0, fff, dc, as1[0], 1, name=name, initpars=None,
                                                              lowcut=0.05, highcut=15.,
                                                              rangepars=[[0., 1.], [0., 0.5], [0., 1. / fff],
                                                                         [0., 10000.]], stop_each=False,
                                                              reselect_ok=False,
                                                              okfile='TES_OK_2019-02-15_Sphere_Asic1.fits')
    amps1 = params[:, 3]

    # Asic 2
    tt, folded, okfinal, params, err, chi2, ndf = ft.run_asic(fnum, 0, fff, dc, as2[0], 2, name=name, initpars=None,
                                                              lowcut=0.05, highcut=15., reselect_ok=False,
                                                              rangepars=[[0., 1.], [0., 0.5], [0., 1. / fff],
                                                                         [0., 10000.]],
                                                              okfile='TES_OK_2019-02-15_Sphere_Asic2.fits')
    amps2 = params[:, 3]

    # Put the 2 asics together
    amps = np.append(amps1, amps2)

    return amps


# Ref, data 2019-02-14
dir = basedir + '2019-02-14/2019-02-14_16.53.38__Test_int_sphere150Ghz_150mHz'
amps = make_amp(dir)
# img_ref = ft.image_asics(all1=amps / np.nanmean(amps))

amps_all = [amps]

# Test with different positions
allimg = np.zeros((4, 17, 17))  # contains datas from 2019-02-15
for pos in xrange(1, 5):
    dir = basedir + '2019-02-15/*_pos{}'.format(pos)
    amps = make_amp(dir, fff=0.333)
    amps_all.append(amps)
    img = ft.image_asics(all1=amps_all[pos])
    allimg[pos-1, :, :] = img


# Plot for different positions compared to the ref
figure('Integration Sphere, divided the mean of 2019-02-15 measurements ')
for pos in xrange(4):
    # img = ft.image_asics(all1=amps_all[pos] / np.nanmean(amps_all[pos]))
    subplot(2, 2, pos + 1)
    imshow(allimg[pos] / np.nanmean(allimg, axis=0), cmap='viridis', vmin=0, vmax=3, interpolation='nearest')

    # imshow(img / img_ref, cmap='viridis', vmin=0, vmax=3, interpolation='nearest')
    # imshow(img / np.nanmean(img), cmap='viridis', vmin=0, vmax=3, interpolation='nearest')
    colorbar()
    # if pos == 0:
    #     title('Int Sphere Ref 2019-02-14')
    # else:
    title('Int Sphere 2019-02-15 pos{}'.format(pos+1))

# savefig(basedir + 'int_sphere_pos1-4_divided_ref')

# Try intercalibration
intercal = img / img[16, 0]

allimg = np.array(FitsArray('allimg_scan_az.fits'))
az = np.array(FitsArray('az_scan_az.fits'))

for i in xrange(len(az)):
    clf()
    subplot(1, 2, 1)
    imshow(allimg[i, :, :], cmap='viridis', vmin=0, vmax=1000)
    title(az[i])
    subplot(1, 2, 2)
    imshow(allimg[i, :, :] / intercal, cmap='viridis', vmin=0, vmax=1000)
    title('Intercalibrated')
    show()
    savefig('scan_az_int_sphere_{}.png'.format(1100+az[i]))
    #raw_input('press a key')


pixnums = [2,96,67,58+128]
clf()
for i in xrange(len(pixnums)):
    subplot(2,2,i+1)
    ax = plt.gca();
    pixnum = pixnums[i]
    test = np.zeros(256)+1
    test[pixnum-1]=2
    imshow(ft.image_asics(all1=test),vmin=0,vmax=2)
    title('Pix Num (starting at 1) = {}'.format(pixnum))
    # Minor ticks
    ax.set_xticks(np.arange(-.5, 17, 1), minor=True);
    ax.set_yticks(np.arange(-.5, 17, 1), minor=True);
    ax.grid(which='minor',color='w', linestyle='-', linewidth=0.5)

savefig('pixels.png')
