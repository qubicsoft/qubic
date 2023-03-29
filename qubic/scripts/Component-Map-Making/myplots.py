import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import sys
import pickle
sys.path.append('/home/regnier/work/regnier/mypackages')
import plotter as p
import os
import qubic
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator


N = 11
nc = 3
nstk = 3
nside = 256
nside_fit = 0
convolution = False
noisy = True
band = 150220

path = os.getcwd()
path_i = f'P217353_CMB_DUST_CO_band{band}_convolution{convolution}_noise{noisy}_nsidefit{nside_fit}/'
fullpath = path + '/' + path_i

analysis = p.Analysis(fullpath, N, nside, nc, nside_fit, thr=0, convolution=convolution, path_to_save=f'PLOTS_'+path_i)

### RMS as function of iteration ###
analysis.plot_rms_maps(i=-1, comp_name=['CMB', 'DUST', 'CO'])

### Reconstructed gain ###
analysis.plot_FP_gain(i=-1, iFP=0, vmin=0.9, vmax=1.1)
analysis.plot_FP_gain(i=-1, iFP=1, vmin=0.9, vmax=1.1)

### Histograms of residuals recontructed gain ###
analysis.plot_hist_residuals_gain(i=-1, iFP=0, c='blue', bins=20, figsize=(6, 6))
analysis.plot_hist_residuals_gain(i=-1, iFP=1, c='red', bins=20, figsize=(6, 6))

center = qubic.equ2gal(10, 247)

analysis.plot_maps(i=-1, center=center, reso=25, istk=0, comp_name=['CMB', 'DUST', 'CO'], figsize=(9, 9), min=-250, max=250, rmin=-30, rmax=30)
analysis.plot_maps(i=-1, center=center, reso=25, istk=1, comp_name=['CMB', 'DUST', 'CO'], figsize=(9, 9), min=-8, max=8, rmin=-1, rmax=1)
analysis.plot_maps(i=-1, center=center, reso=25, istk=2, comp_name=['CMB', 'DUST', 'CO'], figsize=(9, 9), min=-8, max=8, rmin=-1, rmax=1)



### FP
#iFP=1
#plt.figure(figsize=(8, 8))
#analysis.histograms_gain(-1, iFP, bins=20)
#plt.savefig(f'hist_band{band}_convolution{convolution}_noise{noisy}_nsidefit{nside_fit}_{stk[istk]}.png')
#plt.show()
#analysis.plot_FP(-1, iFP, s=70, colorbar='jet')

### Panel of components
#analysis.plot_allcomp(-1, istk=istk, figsize=(12, 8), bar=8, r_bar=8, center=center, reso=25, type='gnomview')

### Make panel
#analysis.make_panel(-1, istk=istk, type='gnomview', center=center, reso=25, figsize=(10, 14), alpha=0.3, truth=1.54)

### GIF
#analysis.make_gif_full_panel(f'P353_Panel_CMB_DUST_band{band}_convolution{convolution}_noise{noisy}_nsidefit{nside_fit}_{stk[istk]}.gif', type='gnomview', istk=istk, 
#            center=center, reso=15, figsize=(5, 7), fps=20, truth=1.54)
#plt.savefig(f'FP_band{band}_convolution{convolution}_noise{noisy}_nsidefit{nside_fit}_{stk[istk]}.png')
#plt.savefig(f'P353_CMB_DUST_band{band}_convolution{convolution}_noise{noisy}_nsidefit{nside_fit}_{stk[istk]}.png')
#plt.close()

'''
for i in range(N):
    print(i)
    pkl_file = open(fullpath+f'Iter{i}_maps_beta_gain_rms_maps.pkl', 'rb')
    dataset = pickle.load(pkl_file)
    pixok = dataset['coverage'] > 0
    maps[i] = dataset['maps']
    beta[i] = dataset['beta']
    gain[i] = dataset['gain']
    maps[i, :, ~pixok, :] = hp.UNSEEN
    #if convolution:
    #    fwhm = np.max(dataset['allfwhm'])
    #else:
    #    fwhm = 0

cmbI_residuals = maps[:, 0, :, 0]-maps[0, 0, :, 0]
cmbI_residuals[:, ~pixok] = hp.UNSEEN
cmbQ_residuals = maps[:, 1, :, 0]-maps[0, 1, :, 0]
cmbQ_residuals[:, ~pixok] = hp.UNSEEN




plt.figure(figsize=(8, 8))
analysis.plot_beta(truth=np.ones(12*nside_fit**2)*1.54)
plt.savefig('beta_d.png')
plt.close()

plt.figure()
analysis.plot_gnomview(-1, istk=1, icomp=0, center=center, reso=20)
plt.savefig('gnom.png')
plt.close()

plt.figure()
analysis.plot_mollview(-1, istk=1, icomp=0)
plt.savefig('moll.png')
plt.close()

#analysis.make_one_map_gif(icomp=0, istk=1, filename='test.gif', fps=10)
#analysis.make_one_map_gif(icomp=0, istk=1, filename='test_gnom.gif', fps=10, center=center, reso=15)



plt.figure()
analysis.make_panel(-1, istk=1, icomp=0, type='gnomview', center=center, reso=25)
plt.savefig('panel.png')
plt.close()


p.panel(maps[0, 0, :, 1], maps[-1, 0, :, 1], beta[:, 0, 0], None, nc, type='gnomview', center=center, reso=20, figsize=(14, 10))
plt.savefig('panel.png')
plt.close()


#p.save_healpix_gif(cmbI_residuals, filename='maps_I.gif', center=center, reso=20, fps=20, min=-10, max=10)
#p.save_healpix_gif(cmbQ_residuals, filename='maps_Q.gif', center=center, reso=20, fps=20, min=-0.1, max=0.1)
#p.save_healpix_gif(beta[:, :, 0], filename='beta.gif', center=center, reso=20, fps=20, min=None, max=None)

plt.figure(figsize=(8, 8))
for i in np.arange(0, 12*nside_fit**2, 1):
    p.convergence_of_X(beta[1:, i, 0], style='-x', label=None, truth=1.54, log=False)
plt.savefig('allbeta.png')
plt.close()

plt.figure(figsize=(8, 8))
p.convergence_of_X(np.std(res_beta, axis=1)[1:, 0], style='-x', label=None, truth=None, log=True)
plt.savefig('rms.png')
plt.close()

#print(beta)

ite = -1
plt.figure()
p.plot_maps(maps[0, 1, :, 0], maps[ite, 1, :, 0], center, type='mollview', reso=35, min=-8, max=8)
plt.savefig('maps_moll.png')
plt.show()

plt.figure()
p.plot_maps(maps[0, 1, :, 0], maps[ite, 1, :, 0], center, type='gnomview', reso=25, min=-8, max=8)
plt.savefig('maps_gnom.png')
plt.show()

plt.figure()
p.plot_maps(beta[0, :, 0], beta[ite, :, 0], center, reso=55, min=1.44, max=1.64)
plt.savefig('beta.png')
plt.show()
'''