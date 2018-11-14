import os
import sys
import glob


import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


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
#Repository with simulation files
rep_simu = '/Users/mousset/Qubic_work/SpectroImagerie/SimuJCResults/Duration20'

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
clf()
plot(np.sort(ang))


####### Look at one map
mapp = np.zeros((12*ns**2, 3))
mapp[seenmap_conv, :] = allmaps_recon[1][6,1,:,:]

hp.mollview(mapp[:,0], title='mapp')

figure('mapp')
clf()
for i in xrange(3):
	hp.gnomview(mapp[:,i], rot=center, reso=12, sub=(1,3,i+1), title=stokes[i])


################# Noise on the maps recon ###############

allmeanmat = rmc.get_rms_covar(nsubvals, seenmap_recon, allmaps_recon)[1]
rmsmap_cov = rmc.get_rms_covarmean(nsubvals, seenmap_recon, allmaps_recon, allmeanmat)[1]
mean_rms_cov = np.sqrt(np.mean(rmsmap_cov**2, axis=2))


#Evolution du bruit en fonction du nombre de sous bandes
clf()
plot(nsubvals, sqrt(nsubvals), 'k', label='Optimal $\sqrt{N}$', lw=2)
for i in xrange(3):
    plot(nsubvals, mean_rms_cov[:,i] / mean_rms_cov[0,i] * sqrt(nsubvals), label=stokes[i], lw=2, ls='--')
xlabel('Number of sub-frequencies')
ylabel('Relative maps RMS')
legend()



#################### Estimation des residus #############################

residus = []
for isub in xrange(len(nsubvals)): 
	# with the convolved maps
	#residus.append(allmaps_recon[isub] - allmaps_conv[isub])

	#with the mean of the recon maps
	residus.append(allmaps_recon[isub] - np.mean(allmaps_recon[isub], axis=0))


#Histogram of the residus
clf()
for i in xrange(3):
	subplot(1, 3, i+1)
	hist(np.ravel(residus[0][:,0,:,i]), range=[-2,2], bins=200)
	title(stokes[i])


#==================== Make fake reconstructed maps ======================

######## But: recuperer la matrice de covariance des residus et l'utiliser pour generer des fausses cartes
######## Correlations du bruit entre I Q U et les sous-bandes prises en compte

#On plot le std des residus en fonction de l'angle entre le pixel considere et le pixel central
#Pour se debarrasser de cette dependence, on fabrique des residus_correct en divisant par std_profile
residus_correct = []
clf()
for isub in xrange(len(nsubvals)):
	sh = residus[isub].shape
	nreals = sh[0]
	nsub = sh[1]
	residu_new = np.zeros(sh)
	bin_centers, std_bin, allstd_profile = tl.myprofile(ang, residus[isub], nbins=15)
	for l in xrange(nsub):
		for i in xrange(3):
			subplot(4, 3, 3*isub+i+1)
			p = plot(bin_centers, std_bin[:,l,i], 'o', label='subband'+str(l))
			plot(ang, allstd_profile[3*l+i], ',', color=p[0].get_color())
			#plot(ang, pixstd[:,i], ',', color=p[0].get_color())
			if isub == 0: title(stokes[i])
			xlabel('angle (deg)')
			if i == 0 and isub == 3: 
				legend()

			#Residus corriges	
			residu_new[:,l,:,i] = residus[isub][:,l,:,i] / allstd_profile[3*l+i]	
			
			if i == 0: print(len(allstd_profile), allstd_profile[3*l+i].shape)
	residus_correct.append(residu_new)
	
#On verifie que les residus_correct ont un std autour de 1 qui n'evolue plus avec l'angle
np.std(residus_correct[3][:,0,:,:], axis=0)

figure()
for isub in xrange(len(nsubvals)):
	sh = residus_correct[isub].shape
	nreals = sh[0]
	nsub = sh[1]
	
	bin_centers, std_bin, allstd_profile = tl.myprofile(ang, residus_correct[isub], nbins=15)
	for l in xrange(nsub):
		for i in xrange(3):
			subplot(4, 3, 3*isub +1 +i)
			p = plot(bin_centers, std_bin[:,l,i], 'o', label='subband'+str(l+1))
			print(np.round(np.mean(std_bin[:, l, i], axis=0),5))
			if isub == 0: title(stokes[i])
			xlabel('angle (deg)')
			if i == 0 and isub == 3: 
				legend()

#Covariance matrices and mean of the residus_correct
allmean, allcov = tl.statistic_distrib(residus_correct)

#Plot of the correlation or covariance matrices
clf()
for isub in xrange(len(nsubvals)):
	subplot(1, len(nsubvals), isub + 1)
	#imshow(tl.cov2corr(allcov[isub]), interpolation='nearest', vmin=-1, vmax=1)
	imshow(allcov[isub], interpolation='nearest')
	if isub == 0: colorbar()
	title(str(isub))


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
#On verifie les residus_fake qu'on a fabrique

#Histogramme des residus_fake
clf()
for i in xrange(3):
	subplot(1, 3, i+1)
	hist(np.ravel(residus_fake[0][:,0,:,i]), range=[-2,2], bins=100)
	title(stokes[i])

#Matrice de cov des residus_fake corriges
residus_correct_fake = []
clf()
for isub in xrange(len(nsubvals)):
	sh = residus_fake[isub].shape
	nreals = sh[0]
	nsub = sh[1]

	bin_centers, std_bin, allstd_profile = tl.myprofile(ang, residus_fake[isub], nbins=15)

	residu_new = np.zeros(sh)
	for l in xrange(nsub):
		for i in xrange(3):
			#Std profile en fonction de l'angle
			subplot(4, 3, 3*isub+i+1)
			p = plot(bin_centers, std_bin[:,l,i], 'o', label='subband'+str(l))
			plot(ang, allstd_profile[3*l+i], ',', color=p[0].get_color())
			#plot(ang, pixstd[:,i], ',', color=p[0].get_color())
			if isub == 0: title(stokes[i])
			xlabel('angle (deg)')
			if i == 0 and isub == 3: 
				legend()

			#Residus corriges	
			residu_new[:,l,:,i] = residus_fake[isub][:,l,:,i] / allstd_profile[3*l+i]
	residus_correct_fake.append(residu_new)

allmean_fake, allcov_fake = tl.statistic_distrib(residus_correct_fake)

#Comparaison entre les matrices de covariance des residus_correct et des residus_correct_fake
alldiff = []
for isub in xrange(len(nsubvals)):
	alldiff.append(allcov_fake[isub] - allcov[isub])

#Plot of the correlation matrices
clf()
for isub in xrange(len(nsubvals)):
	subplot(1, len(nsubvals), isub + 1)
	#imshow(tl.cov2corr(alldiff[isub]), interpolation='nearest', vmin=-0.1, vmax=0.1)
	imshow(alldiff[isub], interpolation='nearest')
	if isub == 0: colorbar()
	title(str(isub))
########################## FIN DU TEST ################################



#Fake map
fake_mapsrecon = []
for isub in xrange(len(nsubvals)): 
	#avec les map conv
	#fake_mapsrecon.append(allmaps_conv[isub] + residus_fake[isub])
	#que du bruit
	#fake_mapsrecon.append(residus_fake[isub])
	#avec les map recon
	fake_mapsrecon.append(np.mean(allmaps_recon[isub], axis=0) + residus_fake[isub])


####### Look at one fake map
mapsfake = np.zeros((12*ns**2, 3))
mapsfake[seenmap_conv, :] = fake_mapsrecon[0][6,0,:,:]

hp.mollview(mapsfake[:,0], title='Fake map recon')

figure('Fake map recon')
clf()
for i in xrange(3):
	hp.gnomview(mapsfake[:,i], rot=center, reso=12, sub=(1,3,i+1), title=stokes[i])




################## APPLY XPOLL##########################
lmin = 20
lmax = 2 * ns
delta_ell = 20

mymask = apodize_mask(seenmap_conv, 5)
xpol = Xpol(mymask, lmin, lmax, delta_ell)
ell_binned = xpol.ell_binned
nbins = len(ell_binned)
print('nbins = {}'.format(nbins))

mcls, mcls_in = [], []
scls, scls_in = [], []

#Input, ce qu'on doit trouver
#mapsconv = np.zeros((12*ns**2, 3))
maps_recon_mean = np.zeros((12*ns**2, 3))
#noise_in = np.zeros((12*ns**2, 3))

#Output, ce qu'on trouve
#maps_recon = np.zeros((12*ns**2, 3))
mapsfake = np.zeros((12*ns**2, 3))


for isub in xrange(len(nsubvals)):
	sh = allmaps_conv[isub].shape
	nreals = sh[0]
	nsub = sh[1]
	cells = np.zeros((6, nbins, nsub, nreals))
	cells_in = np.zeros((6, nbins, nsub, nreals))
	for real in xrange(nreals):
		for n in xrange(nsub):
			#for i in xrange(3):#boucle a mettre uniquement si on met le masque
				#mapsconv[seenmap_conv, i] = allmaps_conv[isub][real,n,:,i] * mymask[seenmap_conv]
			maps_recon_mean[seenmap_conv, :] = np.mean(allmaps_recon[isub][:,n,:,:], axis=0)
			#noise_in[seenmap_conv, :] = residus[isub][real,n,:,:]
			
			mapsfake[seenmap_conv, :] = fake_mapsrecon[isub][real,n,:,:]
			#maps_recon[seenmap_conv, i] = allmaps_recon[isub][real,n,:,i] * mymask[seenmap_recon]
			
			#cells_in[:, :, n , real] = xpol.get_spectra(mapsconv)[1]
			cells_in[:, :, n , real] = xpol.get_spectra(maps_recon_mean)[1]
			#cells_in[:, :, n , real] = xpol.get_spectra(noise_in)[1]
			
			cells[:, :, n, real] = xpol.get_spectra(mapsfake)[1]
			#cells[:, :, n, real] = xpol.get_spectra(maps_recon)[1]

	mcls.append(np.mean(cells, axis = 3))
	mcls_in.append(np.mean(cells_in, axis = 3))
	scls.append(np.std(cells, axis = 3))
	scls_in.append(np.std(cells_in, axis = 3))

#Tous les spectres
figure('test_xpol')

for isub in xrange(len(nsubvals)):
	for s in xrange(3):
		subplot(4,3,isub*3+s+1)
		ylabel(thespec[s])
		xlabel('l')
		
		for k in arange(isub+1):
			p = plot(ell_binned, ell_binned * (ell_binned + 1) * mcls_in[isub][s,:,k], '--')
			errorbar(ell_binned, ell_binned * (ell_binned + 1) * mcls[isub][s,:,k], 
				yerr= ell_binned * (ell_binned + 1) * scls[isub][s,:,k], 
				fmt='o', color=p[0].get_color(),
				label=str(k))
		if s == 0: legend()
		if isub == 0 and s==1: title('exactement comme ana_paper avec le masque')
	

#Spectres pour chaque nb de sous-bande
isub = 0
figure('test_xpol'+str(isub+1))
for s in xrange(3):
	subplot(3,1,s+1)
	ylabel(thespec[s])
	xlabel('l')
	for i in arange(isub+1):
		p = plot(ell_binned, ell_binned * (ell_binned + 1) * mcls_in[isub][s,:,i],'--')
		errorbar(ell_binned, ell_binned * (ell_binned + 1) * mcls[isub][s,:,i], 
			yerr=ell_binned*(ell_binned+1)*scls[isub][s,:,i],fmt='o', color=p[0].get_color(),
			label=str(i))
	if s == 0: 
		legend()
		title('test_xpol'+str(isub+1))
		

