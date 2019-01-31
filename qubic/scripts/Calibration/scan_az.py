from Calibration import fibtools as ft
from Calibration.plotters import *
import matplotlib.pyplot as plt
from pysimulators import FitsArray
import matplotlib.mlab as mlab
import scipy.ndimage.filters as f
import glob
from qubicpack import qubicpack as qp

#### Directory where the files for various angles are located
dir = '/Users/hamilton/Qubic/Fibres/ScanAz2019-01-30/'

#### Now find the az and el corresponding to each directory and prepare the files to read
subdirs = glob.glob(dir+'*')
as1 = glob.glob(dir+'*/Sums/*asic1*.fits')
as2 = glob.glob(dir+'*/Sums/*asic2*.fits')

els = np.zeros(len(subdirs))
azs = np.zeros(len(subdirs))
for i in xrange(len(subdirs)):
    els[i], azs[i] = [float(s) for s in subdirs[i].split('_') if ft.isfloat(s)]


#a = qp()
#a.read_qubicstudio_dataset(subdirs[0])



order = np.argsort(azs)
az = azs[order]
el = els[order]
as1 = np.array(as1)[order]
as2 = np.array(as2)[order]
subdirs = np.array(subdirs)[order]

### Analyse one to define the list of pixok
name = 'ExtSrc'
fnum = 150
asic1 = as1[0]
fff = 0.333
dc = 0.33

tt, folded, okfinal, params, err, chi2, ndf = ft.run_asic(fnum, 0, fff, 
        dc, asic1, 1, name,
        initpars=None, lowcut=0.05, highcut=15., 
        okfile='ScanAz2019-01-30_OK_Asic1.fits')

#### Now loop on asic1
amps = np.zeros((128,len(az)))
taus = np.zeros((128,len(az)))
erramps = np.zeros((128,len(az)))
errtaus = np.zeros((128,len(az)))
for i in xrange(len(as1)):
    asic = as1[i]
    tt, folded, okfinal, params, err, chi2, ndf = ft.run_asic(fnum, 0, fff, 
        dc, asic, 1, name,
        initpars=None, lowcut=0.05, highcut=15., okfile='ScanAz2019-01-30_OK_Asic1.fits')
    amps[:,i] = params[:,3]
    erramps[:,i] = err[:,3]
    taus[:,i] = params[:,1]
    errtaus[:,i] = err[:,1]

allimg = np.zeros((len(as1),17,17))
for i in xrange(len(as1)):
    allimg[i,:,:] = ft.image_asics(data1=amps[:,i])
    clf()
    imshow(allimg[i,:,:]/allimg[0,:,:],vmin=0,vmax=5)
    colorbar()
    title(az[i])
    show()
    savefig('imgscan_az_{}.png'.format(az[i]))
    raw_input('Press a key')

thepix = 93
clf()
plot(az, amps[thepix,:])

