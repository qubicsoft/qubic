from Calibration import fibtools as ft
from Calibration.plotters import *
import matplotlib.pyplot as plt
from pysimulators import FitsArray
import matplotlib.mlab as mlab
import scipy.ndimage.filters as f
import glob
from qubicpack import qubicpack as qp

#### Directory where the files for various angles are located
dir = '/Users/hamilton/Qubic/ExternalSource/2019-02-14_16.53.38__Test_int_sphere150Ghz_150mHz'

as1 = glob.glob(dir+'/Sums/*asic1*.fits')
as2 = glob.glob(dir+'/Sums/*asic2*.fits')


### Analyse one to define the list of pixok
name = 'Integrating_sphere'
fnum = 150
fff = 0.15
dc = 0.33



tt, folded, okfinal, params, err, chi2, ndf = ft.run_asic(fnum,
        0, fff, dc, as1[0], 1, name=name,
        initpars=None, lowcut=0.05, highcut=15., 
        rangepars=[[0.,1.], [0., 0.5], [0.,1./fff], [0., 10000.]],  
        stop_each=False,
        reselect_ok=False, okfile='ScanAz2019-01-30_OK_Asic1.fits')
amps1 = params[:,3]


tt, folded, okfinal, params, err, chi2, ndf = ft.run_asic(fnum,
        0, fff, 
        dc, as2[0], 2, name=name,
        initpars=None, lowcut=0.05, highcut=15., 
        reselect_ok=False,
        rangepars=[[0.,1.], [0., 0.5], [0.,1./fff], [0., 10000.]],  
        okfile='ScanAz2019-01-30_OK_Asic2.fits')
amps2 = params[:,3]


amps = np.append(amps1, amps2)

img = ft.image_asics(all1=amps)
clf()
imshow(img/img[16,0], cmap='viridis', vmin=0,vmax=2)
colorbar()
title('Intercalibration Int. Sphere')



intercal = img/img[16,0]

allimg = np.array(FitsArray('allimg_scan_az.fits'))
az = np.array(FitsArray('az_scan_az.fits'))

for i in xrange(len(az)):
    clf()
    subplot(1,2,1)
    imshow(allimg[i,:,:], cmap='viridis', vmin=0,vmax=1000)
    title(az[i])
    subplot(1,2,2)
    imshow(allimg[i,:,:]/intercal, cmap='viridis', vmin=0,vmax=1000)
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