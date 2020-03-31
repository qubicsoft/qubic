import numpy as np
import healpy as hp
import matplotlib.pyplot as mp
import pymaster as nmt
from numpy import pi
from qubic import (plot_spectra, read_spectra)


nside = 512
b = nmt.NmtBin(nside, nlb=20, is_Dell=True)
leff = b.get_effective_ells()
# input spectra
spectra = read_spectra(0.01)

# pixwin corrections
pw = hp.pixwin(nside, pol=True)
pw = [pw[0][:3 * nside], pw[1][:3 * nside]]
pwb2 = b.bin_cell(np.array(pw))
pwb3 = pwb2 / (leff * (leff + 1)) * 2 * pi

# read the files with the reconstructed l*(l+1)*Cl
cl_mean = hp.read_cl('cls_BBEE_512.fits')
cl_std = hp.read_cl('scls_BBEE_512.fits')

fact = leff * (leff + 1) / (2 * np.pi)
kw_Xpol = {'fmt': 'bo', 'markersize': 3}
mp.figure()
mp.title('BB')
plot_spectra(spectra[2], color='g', label='Input Spectra')
mp.errorbar(leff, cl_mean[3] / (pwb3[1] * pwb3[1]), cl_std[3] / (pwb3[1] * pwb3[1]), label='Pure-$B$ Estimator',
            fmt='r.')
mp.legend(loc='upper right', frameon=False)
mp.ylim(-0.0002, 0.0009)
mp.xlim(0, 512)
mp.savefig('./figuras/BB_512_beam.pdf', format="pdf")
# mp.show()
mp.close()
mp.figure()
mp.title('EE')
plot_spectra(spectra[1], color='g', label='Input Spectra')
mp.errorbar(leff, cl_mean[0] / (pwb3[1] * pwb3[1]), cl_std[0] / (pwb3[1] * pwb3[1]), label='Pure-$B$ Estimator',
            fmt='r.')
mp.legend(loc='upper right', frameon=False)
mp.xlim(0, 512)
mp.ylim(0, 30)
mp.savefig('./figuras/EE_512_beam.pdf', format="pdf")
mp.close()
