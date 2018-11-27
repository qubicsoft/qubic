import os
import sys
import glob

import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import copy

import Tools as tl
import ReadMC as rmc

import qubic
from qubic import gal2equ, equ2gal
from qubic import Xpol
from qubic import apodize_mask

thespec = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
stokes = ['I', 'Q', 'U']

#Coordinates of the zone observed in the sky
center = equ2gal(0., -57.)

################## Get the simulation files ########################
#repository where you find
path = '/home/federico/qubic/qubic/scripts/Spectroimagery_paper'


#Number of subbands used during the simulation
nsubvals = np.array([1, 2])


#Archetypes of the files .fits you want to work on
arch_conv, arch_recon = [], []
for isub in xrange(len(nsubvals)):
	arch_conv.append('bpmaps_nf{}_maps_convolved.fits'.format(nsubvals[isub]))
	arch_recon.append('bpmaps_nf{}_maps_recon.fits'.format(nsubvals[isub]))

print arch_conv, arch_recon
#Get all maps
allmaps_conv, seenmap_conv = rmc.get_all_maps(path, arch_conv, nsubvals)
allmaps_recon, seenmap_recon = rmc.get_all_maps(path, arch_recon, nsubvals)

#Number of pixels and nside
npix = len(seenmap_recon)
ns = int(np.sqrt(npix/12))

# Angle associated to each pixel 
ang = tl.pix2ang(ns, center, seenmap_recon)
print(ang.shape)
plt.plot(np.sort(ang))
plt.show()

# ================== Residuals estimation ===============
#Two ways of obtain residuals:
residuals = []
for j in xrange(len(allmaps_conv)): 
	residuals.append(allmaps_recon[j] - allmaps_conv[j])

residuals = []
for j in xrange(len(allmaps_conv)): 
	residuals.append(allmaps_recon[j] - np.mean(allmaps_recon[j], axis=0))


#Histogram of the residuals
plt.clf()
for i in xrange(3):
	plt.subplot(1, 3, i+1)
	plt.hist(np.ravel(residuals[0][:,0,:,i]), range=[-20,20], bins=100)
	plt.title(stokes[i])


# ================ Look at the maps =================
isub = 0
real = 0
freq = 0

maps_conv = np.zeros((12*ns**2, 3))
maps_conv[seenmap_recon, :] = allmaps_conv[isub][real,freq,:,:]

maps_recon = np.zeros((12*ns**2, 3))
maps_recon[seenmap_conv, :] = allmaps_recon[isub][real,freq,:,:]

maps_residuals = np.zeros((12*ns**2, 3))
maps_residuals[seenmap_conv, :] = residuals[isub][real,freq,:,:]

#hp.mollview(maps_conv[:,1], title='maps_conv')

plt.figure('maps')
for i in xrange(3):
	if i==0:
		min=None
		max=None
	else:
		min=None
		max=None
	hp.gnomview(maps_conv[:,i], rot=center, reso=9, sub=(3,3,i+1), title='conv '+stokes[i], min=min, max=max)
	hp.gnomview(maps_recon[:,i], rot=center, reso=9, sub=(3,3,3+i+1), title='recon '+stokes[i], min=min, max=max)
	hp.gnomview(maps_residuals[:,i], rot=center, reso=9, sub=(3,3,6+i+1), title='residuals '+stokes[i], min=min, max=max)
plt.show()


#================= Noise Evolution as a function of the subband number=======================
#To do that, you need many realisations

allmeanmat = rmc.get_rms_covar(nsubvals, seenmap_recon, allmaps_recon)[1]
rmsmap_cov = rmc.get_rms_covarmean(nsubvals, seenmap_recon, allmaps_recon, allmeanmat)[1]
mean_rms_cov = np.sqrt(np.mean(rmsmap_cov**2, axis=2))


plt.plot(nsubvals, np.sqrt(nsubvals), 'k', label='Optimal $\sqrt{N}$', lw=2)
for i in xrange(3):
    plt.plot(nsubvals, mean_rms_cov[:,i] / mean_rms_cov[0,i] * np.sqrt(nsubvals), label=stokes[i], lw=2, ls='--')
plt.xlabel('Number of sub-frequencies')
plt.ylabel('Relative maps RMS')
plt.legend()


#======================= Apply Xpoll to get spectra ============================
lmin = 20
lmax = 2 * ns
delta_ell = 20

#Xpoll needs a mask
mymask = apodize_mask(seenmap_conv, 5)
xpol = Xpol(mymask, lmin, lmax, delta_ell)
ell_binned = xpol.ell_binned
nbins = len(ell_binned)
print('nbins = {}'.format(nbins))

mcls, mcls_in = [], []
scls, scls_in = [], []

#Input : what we should find
mapsconv = np.zeros((12*ns**2, 3))

#Output, what we find
maps_recon = np.zeros((12*ns**2, 3))


for isub in xrange(len(nsubvals)):
	sh = allmaps_conv[isub].shape
	nreals = sh[0]
	nsub = sh[1]
	cells = np.zeros((6, nbins, nsub, nreals))
	cells_in = np.zeros((6, nbins, nsub, nreals))
	print(cells.shape)
	for real in xrange(nreals):
		for n in xrange(nsub):
			mapsconv[seenmap_conv, :] = allmaps_conv[isub][real,n,:,:]
			maps_recon[seenmap_conv, :] = allmaps_recon[isub][real,n,:,:]
			cells_in[:, :, n , real] = xpol.get_spectra(mapsconv)[1]
			cells[:, :, n, real] = xpol.get_spectra(maps_recon)[1]

	mcls.append(np.mean(cells, axis = 3))
	mcls_in.append(np.mean(cells_in, axis = 3))
	scls.append(np.std(cells, axis = 3))
	scls_in.append(np.std(cells_in, axis = 3))

#Plot the spectra
plt.figure('TT_EE_BB spectra')
for isub in xrange(len(nsubvals)):
	for s in [1,2]:
		plt.subplot(4,2,isub*2+s)
		plt.ylabel(thespec[s]+'_ptg='+str((isub+1)*1000))
		plt.xlabel('l')
		sh = mcls[isub].shape
		nsub = sh[2]
		for k in xrange(nsub):
			p = plt.plot(ell_binned, ell_binned * (ell_binned + 1) * mcls_in[isub][s,:,k], '--')
			plt.errorbar(ell_binned, ell_binned * (ell_binned + 1) * mcls[isub][s,:,k], 
				yerr= ell_binned * (ell_binned + 1) * scls[isub][s,:,k], 
				fmt='o', color=p[0].get_color(),
				label='subband'+str(k+1))
		
		if isub == 0 and s==1: 
			plt.legend(numpoints=1, prop={'size': 7})
plt.show()
	

