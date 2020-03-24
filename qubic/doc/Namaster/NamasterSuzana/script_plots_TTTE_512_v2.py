import numpy as np
import healpy as hp
import matplotlib.pyplot as mp
import pymaster as nmt
from numpy import pi
from qubic import (plot_spectra, read_spectra)

nside = 512

# binning scheme
b = nmt.NmtBin(nside, nlb=20, is_Dell=True)
leff = b.get_effective_ells()

# input spectra
spectra = read_spectra(0.01)

# pixwin corrections
pw = hp.pixwin(nside, pol=True)
pw = [pw[0][:3 * nside], pw[1][:3 * nside]]
ell = np.arange(3 * nside)
pwb2 = b.bin_cell(np.array(pw))
pwb3 = pwb2 / (leff * (leff + 1)) * 2 * pi

# read the files with the reconstructed l*(l+1)*Cl
cltt_mean = hp.read_cl('cls_tt_512_beam_v2.fits')
cltt_std = hp.read_cl('scls_tt_512_beam_v2.fits')
clte_mean = hp.read_cl('cls_te_512_beam_v2.fits')
clte_std = hp.read_cl('scls_te_512_beam_v2.fits')

# plots

kw_Xpol = {'fmt': 'bo', 'markersize': 3}
mp.figure()
mp.title('TT')
plot_spectra(spectra[0], color='g', label='Input Spectra', lmax=512)
mp.errorbar(leff, cltt_mean / (pwb3[0] * pwb3[0]), cltt_std / (pwb3[0] * pwb3[0]), fmt='m.',
            label='Reconstructed Cls')
mp.legend(loc='upper right', frameon=False)
mp.ylim(0, 7000)
mp.savefig('./figuras/TT_512_beam.pdf', format="pdf")
# mp.show()
mp.figure()
mp.title('TE')
plot_spectra(spectra[3], color='g', label='Input Spectra', lmax=512)
mp.errorbar(leff, clte_mean[0] / (pwb3[0] * pwb3[1]), clte_std[0] / (pwb3[0] * pwb3[0]), fmt='m.',
            label='Reconstructed Cls')
mp.legend(loc='upper right', frameon=False)
mp.ylim(-150, 150)
mp.savefig('./figuras/TE_512_beam.pdf', format="pdf")
# mp.show()
mp.close()
