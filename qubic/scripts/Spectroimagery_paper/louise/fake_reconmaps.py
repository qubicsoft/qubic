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

#============= Get the simulation files ==================
#Simulation repository
rep_simu = '/home/louisemousset/QUBIC/Qubic_work/SpectroImagerie/SimuJCResults/Duration20'


#Number of subbands used during the simulation
nsubvals = np.array([1,2,3,4])


#Archetypes of the files .fits you want to work on
arch_conv, arch_recon = [], []
for isub in xrange(len(nsubvals)):
	arch_conv.append('mpiQ_Nodes_2_Ptg_40000_Noutmax_6_Tol_1e-4_*_nf{}_maps_convolved.fits'.format(nsubvals[isub]))
	arch_recon.append('mpiQ_Nodes_2_Ptg_40000_Noutmax_6_Tol_1e-4_*_nf{}_maps_recon.fits'.format(nsubvals[isub]))


#Get all maps
allmaps_conv, seenmap_conv = rmc.get_all_maps(rep_simu, arch_conv, nsubvals)
allmaps_recon, seenmap_recon = rmc.get_all_maps(rep_simu, arch_recon, nsubvals)

#Number of pixels and nside
npix = len(seenmap_recon)
ns = int(np.sqrt(npix/12))

# Angle associated to each pixel 
ang = tl.pix2ang(ns, center, seenmap_recon)
print(ang.shape)
plt.plot(np.sort(ang))
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


#===================== Residus estimation ===================

#Two ways
residus = []
for j in xrange(len(allmaps_conv)): 
	residus.append(allmaps_recon[j] - allmaps_conv[j])

residus = []
for j in xrange(len(allmaps_conv)): 
	residus.append(allmaps_recon[j] - np.mean(allmaps_recon[j], axis=0))


#Histogram of the residus
plt.clf()
for i in xrange(3):
	plt.subplot(1, 3, i+1)
	plt.hist(np.ravel(residus[0][:,3,:,i]), range=[-20,20], bins=100)
	plt.title(stokes[i])

#================ Look at the maps =================
isub = 0
real = 0
freq = 0

maps_conv = np.zeros((12*ns**2, 3))
maps_conv[seenmap_recon, :] = allmaps_conv[isub][real,freq,:,:]

maps_recon = np.zeros((12*ns**2, 3))
maps_recon[seenmap_conv, :] = allmaps_recon[isub][real,freq,:,:]

maps_residus = np.zeros((12*ns**2, 3))
maps_residus[seenmap_conv, :] = residus[isub][real,freq,:,:]

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
	hp.gnomview(maps_residus[:,i], rot=center, reso=9, sub=(3,3,6+i+1), title='residus '+stokes[i], min=min, max=max)
plt.show()


#==================== Make fake reconstructed maps ======================

######## Old version : Bruit estime pour une seule sous-bande, considere comme la reference

#Mean and std on each pixel, on moyenne sur les realisations 
pixmean = np.mean(residus[0][:,0,:,:], axis=0)
pixstd = np.std(residus[0][:,0,:,:], axis=0)

#On reconstruit une carte complete
mapsmean = np.zeros((12*ns**2, 3))
mapsstd = np.zeros((12*ns**2, 3))
mapsmean[seenmap_recon, :] = pixmean
mapsstd[seenmap_recon, :] = pixstd
		
hp.mollview(mapsstd[:,1], title='std')
hp.mollview(mapsmean[:,1], title='mean')

plt.figure('Mean and Std for 1 subband for I Q U maps')
for i in xrange(3):
	hp.gnomview(mapsmean[:,i], rot=center, reso=12, sub=(2,3,i+4), title='mean'+stokes[i])
	hp.gnomview(mapsstd[:,i], rot=center, reso=12, sub=(2,3,i+1), title='std'+stokes[i])

#Fake map using the noise distribution calculated above
#Correct by a factor sqrt(N)
fake_mapsrecon = []
for j in xrange((len(nsubvals))):
	print('For nsub = {}'.format(nsubvals[i]))
	sh = allmaps_recon[j].shape
	newnoise = np.zeros(sh)
	for k in xrange(sh[0]):
		for l in xrange(sh[1]):
			noise = np.random.randn(len(pixmean),3) * pixstd * np.sqrt(j+1) + pixmean
			newnoise[k,l,:,:] = noise
	fake_mapsrecon.append(allmaps_conv[j] + newnoise)

#Correct by the real evolution, not exactly sqrt(N)
fake_mapsrecon = []
for isub in xrange((len(nsubvals))):
	print('For nsub = {}'.format(nsubvals[i]))
	sh = allmaps_recon[isub].shape
	correction = mean_rms_cov[isub,:] / mean_rms_cov[0,:] * np.sqrt(isub+1)
	newnoise = np.zeros(sh)
	for k in xrange(sh[0]):
		for l in xrange(sh[1]):
			noise = np.random.randn(sh[2], sh[3]) * pixstd * correction + pixmean
			newnoise[k,l,:,:] = noise
			if (isub,k,l) == (2,0,0): 
				print(correction, np.sqrt(isub+1))
				print(noise.shape, newnoise.shape)
	fake_mapsrecon.append(allmaps_conv[isub] + newnoise)


######## New version : Correlations du bruit entre I Q U et les sous-bandes prises en compte
#On plot le std des residus en fonction de l'angle pour les differentes cartes
#Pour se debarrasser de cette dependence, on fabrique des residus_correct en divisant par std_profile
residus_correct = []
plt.clf()
for isub in xrange(len(nsubvals)):
	sh = residus[isub].shape
	nreals = sh[0]
	nsub = sh[1]
	residu_new = np.zeros(sh)
	bin_centers, std_bin, allstd_profile = tl.myprofile(ang, residus[isub], nbins=15)
	for l in xrange(nsub):
		for i in xrange(3):
			plt.subplot(4, 3, 3*isub+i+1)
			p = plt.plot(bin_centers, std_bin[:,l,i], 'o', label='subband'+str(l))
			plt.plot(ang, allstd_profile[3*l+i], ',', color=p[0].get_color())
			#plot(ang, pixstd[:,i], ',', color=p[0].get_color())
			if isub == 0: plt.title(stokes[i])
			plt.xlabel('angle (deg)')
			if i == 0 and isub == 3: 
				plt.legend()

			#Residus corriges	
			residu_new[:,l,:,i] = residus[isub][:,l,:,i] / allstd_profile[3*l+i]	
			
			if i == 0: print(len(allstd_profile), allstd_profile[3*l+i].shape)
	residus_correct.append(residu_new)
	
#On verifie que les residus_correct ont un std autour de 1 qui n'evolue plus avec l'angle
np.std(residus_correct[3][:,0,:,:], axis=0)

plt.figure()
for isub in xrange(len(nsubvals)):
	sh = residus_correct[isub].shape
	nreals = sh[0]
	nsub = sh[1]
	
	bin_centers, std_bin, allstd_profile = tl.myprofile(ang, residus_correct[isub], nbins=15)
	for l in xrange(nsub):
		for i in xrange(3):
			plt.subplot(4, 3, 3*isub +1 +i)
			p = plt.plot(bin_centers, std_bin[:,l,i], 'o', label='subband'+str(l+1))
			print(np.round(np.mean(std_bin[:, l, i], axis=0),5))
			if isub == 0: plt.title(stokes[i])
			plt.xlabel('angle (deg)')
			if i == 0 and isub == 3: 
				plt.legend()

#Covariance matrices and mean of the residus_correct
allmean, allcov = tl.covariance_IQU_subbands(residus_correct)
allmean, allcov = tl.covariance_IQU_subbands(residus)

#test de la fonction covariance_IQU_subbands
toto = []
for isub in xrange(len(nsubvals)):
	toto.append(np.random.randn(10, isub+1, 100014, 3))
m, c = tl.covariance_IQU_subbands(toto)

#Plot of the correlation or covariance matrices
plt.clf()
for isub in xrange(len(nsubvals)):
	plt.subplot(1, len(nsubvals), isub + 1)
	plt.imshow(tl.cov2corr(allcov[isub]), interpolation='nearest', vmin=-1, vmax=1)
	#plt.imshow(allcov[isub], interpolation='nearest')
	if isub == 0: plt.colorbar()
	plt.title(str(isub))


#Fake map using the noise distribution of the covariance matrix 
#On remultiplie par le std_profile pour avoir la dependance du std en fonction de l'angle

residus_fake = []
for isub in xrange((len(nsubvals))):
	print('For nsub = {}'.format(nsubvals[isub]))
	sh = allmaps_recon[isub].shape
	nreals = sh[0]
	nsub = sh[1]
	npixok = sh[2]

	allstd_profile = tl.myprofile(ang, residus[isub], nbins=15)[2]

	noise4D = np.zeros(sh)
	
	for k in xrange(nreals):
		noise = np.random.multivariate_normal(allmean[isub], allcov[isub], size=npixok) #bruit pour la realisation k
		if k == 0: print(noise.shape)
		for l in xrange(nsub):
			noise4D[k, l, :, :] = noise[:, 3*l:3*l+3]
			for i in xrange(3):
				noise4D[k, l, :, i] *= allstd_profile[3*l+i]
	residus_fake.append(noise4D)

############# TEST #################

#Histogramme des residus_fake
plt.clf()
for i in xrange(3):
	plt.subplot(1, 3, i+1)
	plt.hist(np.ravel(residus_fake[0][:,0,:,i]), range=[-2,2], bins=100)
	plt.title(stokes[i])

#Matrice de cov des residus_fake corriges
residus_correct_fake = []
plt.clf()
for isub in xrange(len(nsubvals)):
	sh = residus_fake[isub].shape
	nreals = sh[0]
	nsub = sh[1]

	bin_centers, std_bin, allstd_profile = tl.myprofile(ang, residus_fake[isub], nbins=15)

	residu_new = np.zeros(sh)
	for l in xrange(nsub):
		for i in xrange(3):
			#Std profile en fonction de l'angle
			plt.subplot(4, 3, 3*isub+i+1)
			p = plt.plot(bin_centers, std_bin[:,l,i], 'o', label='subband'+str(l))
			plt.plot(ang, allstd_profile[3*l+i], ',', color=p[0].get_color())
			#plt.plot(ang, pixstd[:,i], ',', color=p[0].get_color())
			if isub == 0: plt.title(stokes[i])
			plt.xlabel('angle (deg)')
			if i == 0 and isub == 3: 
				plt.legend()

			#Residus corriges	
			residu_new[:,l,:,i] = residus_fake[isub][:,l,:,i] / allstd_profile[3*l+i]
	residus_correct_fake.append(residu_new)

allmean_fake, allcov_fake = tl.covariance_IQU_subbands(residus_correct_fake)

#Comparaison entre les matrices de covariance des residus_correct et des residus_correct_fake
alldiff = []
for isub in xrange(len(nsubvals)):
	alldiff.append(allcov_fake[isub] - allcov[isub])

#Plot of the correlation matrices
plt.clf()
for isub in xrange(len(nsubvals)):
	plt.subplot(1, len(nsubvals), isub + 1)
	#imshow(tl.cov2corr(alldiff[isub]), interpolation='nearest', vmin=-0.1, vmax=0.1)
	plt.imshow(alldiff[isub], interpolation='nearest')
	if isub == 0: plt.colorbar()
	plt.title(str(isub))
########################## FIN DU TEST ################################



#Fausse carte
fake_mapsrecon = []
for isub in xrange(len(nsubvals)): 
	#avec les map conv
	fake_mapsrecon.append(allmaps_conv[isub] + residus_fake[isub])
	#que du bruit
	#fake_mapsrecon.append(residus_fake[isub])
	#avec les map recon
	#fake_mapsrecon.append(np.mean(allmaps_recon[isub], axis=0) + residus_fake[isub])


####### Look at one fake map
mapsfake = np.zeros((12*ns**2, 3))
mapsfake[seenmap_conv, :] = fake_mapsrecon[0][0,0,:,:]

hp.mollview(mapsfake[:,0], title='Fake map recon')

plt.figure('Fake map recon')
plt.clf()
for i in xrange(3):
	hp.gnomview(mapsfake[:,i], rot=center, reso=12, sub=(1,3,i+1), title=stokes[i])



#======================= Apply Xpoll on fake maps ============================
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

#Input, ce qu'on doit trouver
mapsconv = np.zeros((12*ns**2, 3))

#Output, ce qu'on trouve
mapsfake = np.zeros((12*ns**2, 3))


for isub in xrange(len(nsubvals)):
	sh = allmaps_conv[isub].shape
	nreals = sh[0]
	nsub = sh[1]
	cells = np.zeros((6, nbins, nsub, nreals))
	cells_in = np.zeros((6, nbins, nsub, nreals))
	print(cells.shape)
	for real in xrange(nreals):
		for n in xrange(nsub):
			for i in xrange(3):
				mapsconv[seenmap_conv, i] = allmaps_conv[isub][real,n,:,i] * mymask[seenmap_conv]

				mapsfake[seenmap_conv, i] = fake_mapsrecon[isub][real,n,:,i]* mymask[seenmap_recon]


			cells_in[:, :, n , real] = xpol.get_spectra(mapsconv)[1]
			
			cells[:, :, n, real] = xpol.get_spectra(mapsfake)[1]

	mcls.append(np.mean(cells, axis = 3))
	mcls_in.append(np.mean(cells_in, axis = 3))
	scls.append(np.std(cells, axis = 3))
	scls_in.append(np.std(cells_in, axis = 3))