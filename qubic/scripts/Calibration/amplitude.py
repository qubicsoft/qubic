import numpy as np
from matplotlib.pyplot import *
from pysimulators import FitsArray
import matplotlib.mlab as mlab
import scipy.ndimage.filters as f

from Calibration import fibtools as ft


from qubicpack import qubicpack as qp

########### Read Maynooth Files
num, power1 = np.loadtxt('TESPowOfile150newCF2_CF.qb.txt',skiprows=1).T
num, power2 = np.loadtxt('TESPowOfile150newCF_CF.qb.txt',skiprows=1).T
#num, power = np.loadtxt('TESPowerOutfile.txt',skiprows=1).T
aa = qp()
pixnums = np.ravel(aa.TES2PIX)
pow_maynooth = np.zeros(256)
for i in xrange(len(pixnums)):
    if pixnums[i] < 992:
        pow_maynooth[i] = 1000*power1[pixnums[i]-1] + 1000*power2[pixnums[i]-1]


img_maynooth = ft.image_asics(pow_maynooth, all1=True)
clf()
imshow(img_maynooth)
colorbar()

allfib = [2,3,4]
allcal = np.zeros(len(allfib))
allerrcal = np.zeros(len(allfib))
allnewok = []
for i in xrange(len(allfib)):
    fib = allfib[i]
    free = 'free13'
    allok = np.array(FitsArray('listok_fib{}_{}.fits'.format(fib,free))).astype(bool)
    allparams = np.array(FitsArray('params_fib{}_{}.fits'.format(fib,free)))
    allerr = np.array(FitsArray('err_fib{}_{}.fits'.format(fib,free)))
    allok = allok * np.isfinite(np.sum(allparams, axis=1)) * np.isfinite(np.sum(allerr, axis=1))

    cal, errcal, newok = ft.calibrate(fib, pow_maynooth, allparams, allerr, allok, cutparam=0.4, cuterr=0.03, bootstrap=10000)
    savefig('Calibration-Fib{}_new250119.png'.format(fib))

    allnewok.append(newok)
    allcal[i] = cal[0]
    allerrcal[i] = errcal[0]
    #raw_input('Press a key to continue...')



### TES Intercalibration with i=2 (best S/N)
i=2
fib = allfib[i]
free = 'free13'
allok = np.array(FitsArray('listok_fib{}_{}.fits'.format(fib,free))).astype(bool)
allparams = np.array(FitsArray('params_fib{}_{}.fits'.format(fib,free)))
allerr = np.array(FitsArray('err_fib{}_{}.fits'.format(fib,free)))
allok = allok * np.isfinite(np.sum(allparams, axis=1)) * np.isfinite(np.sum(allerr, axis=1))


calibration = np.zeros(256) + np.nan
calibration[allok] = pow_maynooth[allok] / allparams[allok,3]

cal, errcal, newok = ft.calibrate(fib, pow_maynooth, allparams, allerr, allok, cutparam=0.4, cuterr=0.03, bootstrap=10000)

calibration_restrict = np.zeros(256) + np.nan
calibration_restrict[newok] = pow_maynooth[newok] / allparams[newok,3]

from pysimulators import FitsArray
FitsArray(calibration).save('calibration.fits')
FitsArray(calibration_restrict).save('calibration_restrict.fits')


