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
# rep_simu = '/home/louisemousset/QUBIC/Qubic_work/SpectroImagerie/SimuJCResults/noiseless211118'
rep_simu = '/home/louisemousset/QUBIC/Qubic_work/SpectroImagerie/SimuLouise/noiseless_I=0'
# rep_simu = '/home/louisemousset/QUBIC/Qubic_work/SpectroImagerie/SimuLouise/test_ptg'
# rep_simu = '/home/louisemousset/QUBIC/Qubic_work/SpectroImagerie/SimuLouise/fix_hwp'
# rep_simu = '/home/louisemousset/QUBIC/Qubic_work/SpectroImagerie/SimuLouise/vary_tol'


#Number of subbands used during the simulation
nsubvals = np.array([1,2,3,4])
#nsubvals = np.tile(4, 3) 

#Archetypes of the files .fits you want to work on
arch_conv, arch_recon = [], []
for isub in xrange(len(nsubvals)):
	# arch_conv.append('mpiQ_Nodes_2_Ptg_40000_Noutmax_6_Tol_1e-4_*_nf{}_maps_convolved.fits'.format(nsubvals[isub]))
	# arch_recon.append('mpiQ_Nodes_2_Ptg_40000_Noutmax_6_Tol_1e-4_*_nf{}_maps_recon.fits'.format(nsubvals[isub]))
	
	arch_conv.append('Test_*_Ptg_16000_Noiseless_True_QP_False_Guess_True_NoI_False_nf{}_maps_convolved.fits'.format(nsubvals[isub]))
	arch_recon.append('Test_*_Ptg_16000_Noiseless_True_QP_False_Guess_True_NoI_False_nf{}_maps_recon.fits'.format(nsubvals[isub]))

	# arch_conv.append('noiseless_05_nf{}_maps_convolved.fits'.format(nsubvals[isub]))
	# arch_recon.append('noiseless_05_nf{}_maps_recon.fits'.format(nsubvals[isub]))

# for ptg in xrange(6):
# 	arch_conv.append('test_ptg02_nf4_ptg{}_maps_convolved.fits'.format(ptg))
# 	arch_recon.append('test_ptg02_nf4_ptg{}_maps_recon.fits'.format(ptg))

# for real in xrange(50):
# 	arch_conv.append('noiseless_50reals_nf4_real{}_maps_convolved.fits'.format(real))
# 	arch_recon.append('noiseless_50reals_nf4_real{}_maps_recon.fits'.format(real))

# for hwp in xrange(4):
#  	arch_conv.append('fix_hwp0{0}_nf4_maps_convolved.fits'.format(hwp+1))
#  	arch_recon.append('fix_hwp0{0}_nf4_maps_recon.fits'.format(hwp+1))
# arch_conv.append('random_hwp01_nf4_maps_convolved.fits')
# arch_recon.append('random_hwp01_nf4_maps_recon.fits')

# for tol in ['1e-4', '5e-5', '1e-5']:
# 	arch_conv.append('mpiQ_Nodes_1_Ptg_40000_Noutmax_6_Tol_{}_*_nf4_maps_convolved.fits'.format(tol))
# 	arch_recon.append('mpiQ_Nodes_1_Ptg_40000_Noutmax_6_Tol_{}_*_nf4_maps_recon.fits'.format(tol))

# for tol in [0.001, 0.0005, 0.0001, 5e-05, 1e-05]:
	# arch_conv.append('vary_tol_nf4_tol{}_maps_convolved.fits'.format(tol))
	# arch_recon.append('vary_tol_nf4_tol{}_maps_recon.fits'.format(tol))


#Get all maps
allmaps_conv, seenmap_conv = rmc.get_all_maps(rep_simu, arch_conv, nsubvals)
allmaps_recon, seenmap_recon = rmc.get_all_maps(rep_simu, arch_recon, nsubvals)

# #test_ptg02: On met tous les ptg dans un même tableau
# sh = allmaps_conv[0].shape
# conv = np.zeros((6,sh[1],sh[2],sh[3]))
# recon = np.zeros((6,sh[1],sh[2],sh[3]))
# for ptg in xrange(6):
# 	conv[ptg,:,:,:] = allmaps_conv[ptg]
# 	recon[ptg,:,:,:] = allmaps_recon[ptg]
# allmaps_conv = []
# allmaps_recon = []
# allmaps_conv.append(conv)
# allmaps_recon.append(recon)

#noiseless_50reals: On met toutes les realisations dans un même tableau
sh = allmaps_conv[0].shape
conv = np.zeros((50,sh[1],sh[2],sh[3]))
recon = np.zeros((50,sh[1],sh[2],sh[3]))
for real in xrange(50):
	conv[real,:,:,:] = allmaps_conv[real]
	recon[real,:,:,:] = allmaps_recon[real]
allmaps_conv = []
allmaps_recon = []
allmaps_conv.append(conv)
allmaps_recon.append(recon)

#Number of pixels and nside
npix = len(seenmap_recon)
ns = int(np.sqrt(npix/12))

# Angle associated to each pixel 
ang = tl.pix2ang(ns, center, seenmap_recon)
print(ang.shape)
plt.plot(np.sort(ang))
plt.show()

#===================== Residus estimation ===================

#Two ways
residus = []
for j in xrange(len(allmaps_conv)): 
	residus.append(allmaps_recon[j] - allmaps_conv[j])

# residus = []
# for j in xrange(len(allmaps_conv)): 
# 	residus.append(allmaps_recon[j] - np.mean(allmaps_recon[j], axis=0))


#Histogram of the residus
plt.clf()
for i in xrange(3):
	plt.subplot(1, 3, i+1)
	plt.hist(np.ravel(residus[0][:,3,:,i]), range=[-20,20], bins=100)
	plt.title(stokes[i])

#Residus en fonction de l'angle
plt.figure('noiseless_05_residus')
for i in xrange(3):
	plt.subplot(1,3,i+1)
	plt.plot(ang, residus[1][0,0,:,i], '.')
	plt.ylabel('residu '+stokes[i])
	plt.xlabel('angle in deg')

#================ Look at the maps =================
isub = 1
real = 0
freq = 0

maps_conv = np.zeros((12*ns**2, 3))
maps_conv[seenmap_recon, :] = allmaps_conv[isub][real,freq,:,:]

maps_recon = np.zeros((12*ns**2, 3))
maps_recon[seenmap_conv, :] = allmaps_recon[isub][real,freq,:,:]

maps_residus = np.zeros((12*ns**2, 3))
maps_residus[seenmap_conv, :] = residus[isub][real,freq,:,:]

#hp.mollview(maps_conv[:,1], title='maps_conv')

plt.figure('noiseless_05_maps_1subbands_freq1')
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

#Plot of residus
isub=0
sh=residus[isub].shape
plt.figure('guess=True_16000_res4')
for freq in xrange(sh[1]):
	maps_residus = np.zeros((12*ns**2, 3))
	maps_residus[seenmap_conv, :] = residus[isub][0,freq,:,:]
	for i in xrange(3):
		if i==0:
			min=None
			max=None
		else:
			min=None
			max=None
		hp.gnomview(maps_residus[:,i], rot=center, reso=9, sub=(sh[1],3,3*freq+i+1), title=stokes[i]+' subband'+str(freq+1), min=min, max=max)



# #####plots pour fix_hwp
# freq = 0
# plt.figure('fix_hwp_map_reconQU_subband'+str(freq+1))
# for hwp in xrange(4):
# 	maps_recon = np.zeros((12*ns**2, 3))
# 	maps_recon[seenmap_conv, :] = allmaps_recon[hwp][0,freq,:,:]
# 	for i in xrange(2):
# 		hp.gnomview(maps_recon[:,i+1], rot=center, reso=9, sub=(2,4,4*i+hwp+1), 
# 					title=(stokes[i+1]+' ptg='+str((hwp+1)*1000)), min=-25, max=25)

# ######plots for test_ptg

# freq = 1
# plt.figure('test_ptg02_map_reconQU_subband'+str(freq+1))
# for ptg in xrange(6):
# 	maps_recon = np.zeros((12*ns**2, 3))
# 	maps_recon[seenmap_conv, :] = allmaps_recon[0][ptg,freq,:,:]
# 	for i in xrange(2):
# 		hp.gnomview(maps_recon[:,i+1], rot=center, reso=12, sub=(2,6,6*i+ptg+1), 
# 					title=(stokes[i+1]+' ptg='+str(ptg*1000+1000)), min=-8, max=8)
# plt.show()

#=============== Separate a map in different zones ====================
nzones = 4
a4 = np.max(ang)
a1 = a4/4
a2 = 2*a1
a3 = 3*a1

sh = allmaps_recon[0].shape
print(sh)
npix_seen = sh[2]		

#TEST: artificial correlation 
# maps_recon_test = copy.copy(allmaps_recon[0])
# for pix in xrange(npix_seen):
# 	if ang[pix] <= a1:
# 		maps_recon_test[:,:,pix,1] = maps_recon_test[:,:,pix,0] + np.random.rand(1)
# 	elif a1<ang[pix]<=a2:
# 		maps_recon_test[:,:,pix,1] = maps_recon_test[:,:,pix,0] + np.random.rand(1)*100
# 	elif a2<ang[pix]<=a3:
# 		maps_recon_test[:,:,pix,1] = maps_recon_test[:,:,pix,0] + np.random.rand(1)*1000
# 	else:
# 		maps_recon_test[:,:,pix,1] = maps_recon_test[:,:,pix,0] + np.random.rand(1)*10000



#On récupère les pixels qui sont dans chaque zone
#On crèe des masks qui valent 1 dans la zone et 0 ailleurs

n0, n1, n2, n3 = 0, 0, 0, 0
recon0, recon1, recon2, recon3 = [], [], [], []
mask0 = np.zeros(sh)
mask1 = np.zeros(sh)
mask2 = np.zeros(sh)
mask3 = np.zeros(sh)
for pix in xrange(npix_seen):
	if ang[pix] <= a1:
		n0+=1
		mask0[:,:,pix,:] = 1.
		recon0.append(allmaps_recon[0][:,:,pix,:])
		#recon0.append(residus[3][:,:,pix,:])		
		#recon0.append(maps_recon_test[:,:,pix,:])	
	elif a1<ang[pix]<=a2:
		n1+=1
		mask1[:,:,pix,:] = 1.
		recon1.append(allmaps_recon[0][:,:,pix,:])
		#recon1.append(residus[3][:,:,pix,:])
		#recon1.append(maps_recon_test[:,:,pix,:])
	elif a2<ang[pix]<=a3:
		n2+=1
		mask2[:,:,pix,:] = 1.
		recon2.append(allmaps_recon[0][:,:,pix,:])
		#recon2.append(residus[3][:,:,pix,:])
		#recon2.append(maps_recon_test[:,:,pix,:])
	else:
		n3+=1
		mask3[:,:,pix,:] = 1.
		recon3.append(allmaps_recon[0][:,:,pix,:])
		#recon3.append(residus[3][:,:,pix,:])
		#recon3.append(maps_recon_test[:,:,pix,:])

#number of pixels in each zone
all_n = [n0, n1, n2, n3]
print(all_n, np.sum(all_n), sh[2])

#Let's look at the zones
allmask =[mask0, mask1, mask2, mask3] 
allrecon_mask= []
for zone in xrange(nzones):
	recon_mask = allmaps_recon[0] * allmask[zone]
	#recon_mask = residus[3] * allmask[zone]
	allrecon_mask.append(recon_mask)

freq = 3
zone = 3
ptg = 0
maps_recon = np.zeros((12*ns**2, 3))
maps_recon[seenmap_conv, :] = allrecon_mask[zone][ptg,freq,:,:]
hp.gnomview(maps_recon[:,2], rot=center, reso=12, title=('ptg='+str(ptg)))

#New reconstructed maps with only the pixels inside the zone
allrecon = [recon0, recon1, recon2, recon3]
allrecon_new = []
for n in xrange(len(all_n)):
	recon_new = np.zeros([sh[0], sh[1], all_n[n], sh[3]])
	for i in xrange(all_n[n]):
		recon_new[:,:,i,:] = allrecon[n][i]
	allrecon_new.append(recon_new)


# ================ Correlations IQU ====================
############Pour test_ptg02 
# Méthode 2: On fait des matrices de covariance IQU en moyennant sur les pixels d'une zone
r_coeff = ['r_IQ', 'r_IU', 'r_QU']
nptg = 6
nsub = 4
ngroup = 1 #pour avoir des barres d'erreur on fait des sous-zones

r = np.zeros((len(r_coeff), nzones, nptg, nsub))
std = np.zeros((len(r_coeff), nzones, nptg, nsub))
for ptg in xrange(nptg):
	for freq in xrange(nsub):
		for zone in xrange(nzones):
			corr_zone = np.zeros([ngroup,3,3])
			for g in xrange(ngroup):
				sh = allrecon_new[zone].shape
				print(sh)
				cov_zone = np.cov(allrecon_new[zone][ptg,freq,g*sh[2]/ngroup:(g+1)*sh[2]/ngroup,:], rowvar=False)
				corr_zone[g,:,:] = tl.cov2corr(cov_zone)
			r_mean = np.mean(corr_zone, axis=0)
			r_std = np.std(corr_zone, axis=0)
			r[0, zone, ptg, freq] = r_mean[0,1]
			r[1, zone, ptg, freq] = r_mean[0,2]
			r[2, zone, ptg, freq] = r_mean[1,2]
			std[0, zone, ptg, freq] = r_std[0,1]
			std[1, zone, ptg, freq] = r_std[0,2]
			std[2, zone, ptg, freq] = r_std[1,2]
		

plt.figure('test_ptg02_rcoeff')
for coeff in xrange(len(r_coeff)):
	for freq in xrange(nsub):
		plt.subplot(3,4,4*coeff+freq+1)
		for ptg in xrange(1):
			plt.errorbar(np.arange(4), r[coeff,:,ptg,freq], yerr = std[coeff,:,ptg,freq], fmt ='-o', label='ptg='+str(1000*(1+ptg)))
		if coeff == 2: plt.xlabel('zone')
		if freq == 0: plt.ylabel(r_coeff[coeff])
		plt.xlim(-0.5, 3.5)
		if coeff == 0: plt.title('subband'+str(freq+1))
		if coeff == 0 and freq == 0: plt.legend()
	


############ pour noiseless_50reals
#### Methode 1: on fait des matrices IQU de covariance par pixel avec 50 realisations)
sh = allmaps_recon[0].shape
print(sh)
npix_seen = sh[2]
nsub = sh[1]

all_corr =[]
for sub in xrange(nsub):
	corr = np.zeros((npix_seen, 3, 3))
	for pix in xrange(npix_seen):
		#cov = np.cov(maps_recon_test[:,sub,pix,:], rowvar = False) 
		cov = np.cov(allmaps_recon[0][:,sub,pix,:], rowvar = False)
		corr[pix,:,:] = tl.cov2corr(cov)
		if pix==0: print(cov)
	if sub==0: print(corr.shape)
	all_corr.append(corr)

#Matrice de correlation pour 5 pixels
ipix = 0
for pix in [50, 200, 500, 1000, 2000]:
	plt.subplot(1,5,ipix+1)
	plt.imshow(all_corr[0][pix, :,:], interpolation='nearest')
	plt.title(ang[pix])
	if ipix==0: plt.colorbar()
	ipix += 1
plt.show()

#Correlation coeff as a function of the angle
ang = tl.pix2ang(ns, center, seenmap_recon)
plt.figure('noiseless_50reals_rQU')
plt.plot(ang, all_corr[0][:, 0, 2], '.')
plt.ylabel('r_QU')
plt.xlabel('angle between the pixel and the central pixel')
plt.show()

####Methode 2: On applique la methode de decoupage en zone à chaque realisation
nreals = sh[0]
nsub = sh[1]

r = np.zeros((len(r_coeff), nzones, nreals, nsub))

for real in xrange(nreals):
	for freq in xrange(nsub):
		for zone in xrange(nzones):
			cov_zone = np.cov(allrecon_new[zone][real,freq,:,:], rowvar=False)
			corr_zone = tl.cov2corr(cov_zone)
			r[0, zone, real, freq] = corr_zone[0,1]
			r[1, zone, real, freq] = corr_zone[0,2]
			r[2, zone, real, freq] = corr_zone[1,2]
			if ptg == 0 and freq==0: print(corr_zone)

#On moyenne sur les realisations
r_mean = np.mean(r, axis=2)
r_std = np.std(r, axis=2)

plt.figure('noiseless_50reals_rcoeff')
for coeff in xrange(len(r_coeff)):
	for freq in xrange(nsub):
		plt.subplot(3,4,4*coeff+freq+1)
		plt.errorbar(np.arange(4), r_mean[coeff,:,freq], yerr = r_std[coeff,:,freq], fmt ='-o')
		if coeff == 2: plt.xlabel('zone')
		if freq == 0: plt.ylabel(r_coeff[coeff])
		plt.xlim(-0.5, 3.5)
		if coeff == 0: plt.title('subband'+str(freq+1))



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

#Input, ce qu'on doit trouver
mapsconv = np.zeros((12*ns**2, 3))
#noise_in = np.zeros((12*ns**2, 3))

#Output, ce qu'on trouve
#maps_recon_mean = np.zeros((12*ns**2, 3))
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
			#noise_in[seenmap_conv, :] = residus[isub][real,n,:,:]
		
			#maps_recon_mean[seenmap_conv, :] = np.mean(allmaps_recon[isub][:,n,:,:], axis=0)
			maps_recon[seenmap_conv, :] = allmaps_recon[isub][real,n,:,:]
		
			#Let's put I=0 (optional)
			# mapsconv[:,0] *= 0
			# maps_recon_mean[:,0] *= 0
			# maps_recon[:,0] *= 0

			cells_in[:, :, n , real] = xpol.get_spectra(mapsconv)[1]
			#cells_in[:, :, n , real] = xpol.get_spectra(noise_in)[1]

			#cells[:, :, n , real] = xpol.get_spectra(maps_recon_mean)[1]
			cells[:, :, n, real] = xpol.get_spectra(maps_recon)[1]

	mcls.append(np.mean(cells, axis = 3))
	mcls_in.append(np.mean(cells_in, axis = 3))
	scls.append(np.std(cells, axis = 3))
	scls_in.append(np.std(cells_in, axis = 3))

tol = [0.001, 0.0005, 0.0001, 5e-05, 1e-05]

#plot all spectra
plt.figure('guess=True_ptg16000')
for isub in xrange(len(nsubvals)):
	for s in xrange(3):
		plt.subplot(4,3,isub*3+s+1)
		plt.ylabel(thespec[s] + ' tol=' + str(tol[isub]))
		plt.xlabel('l')
		sh = mcls[isub].shape
		nsub = sh[2]
		for k in xrange(nsub):
			p = plt.plot(ell_binned, ell_binned * (ell_binned + 1) * mcls_in[isub][s,:,k], '--')
			plt.errorbar(ell_binned, ell_binned * (ell_binned + 1) * mcls[isub][s,:,k], 
				yerr= ell_binned * (ell_binned + 1) * scls[isub][s,:,k], 
				fmt='o', color=p[0].get_color(),
				label='subband'+str(k+1))
		if s==0 and isub==0:
			#plt.title('from 1 to 4 hwp angles') 
			plt.legend(numpoints=1, prop={'size': 7})
plt.show()
	

#Spectres pour chaque nb de sous-bande
isub = 2
plt.figure('test_xpol'+str(isub+1))
for s in xrange(3):
	plt.subplot(3,1,s+1)
	plt.ylabel(thespec[s])
	plt.xlabel('l')

	sh = mcls[isub].shape
	nsub = sh[2]
	for k in xrange(nsub):
		p = plt.plot(ell_binned, ell_binned * (ell_binned + 1) * mcls_in[isub][s,:,k],'--')
		plt.errorbar(ell_binned, ell_binned * (ell_binned + 1) * mcls[isub][s,:,k], 
			 yerr=ell_binned*(ell_binned+1)*scls[isub][s,:,k], fmt='o', color=p[0].get_color(),
			 label=str(k))
	if s == 0: 
		plt.legend(numpoints=1)
		plt.title('test_xpol'+str(isub+1))

#Spectre par numero de sous-bande
plt.figure('test_ptg02_subband')
nptg = len(mcls)
for band in xrange(4):
	for s in xrange(3):
		plt.subplot(4,3,band*3+s+1)
		plt.ylabel(thespec[s]+'_bande'+str(band+1))
		plt.xlabel('l')
		
		plt.plot(ell_binned, ell_binned * (ell_binned + 1) * mcls_in[ptg][s,:,band], '--')
		for ptg in xrange(nptg):
			plt.plot(ell_binned, ell_binned * (ell_binned + 1) * mcls[ptg][s,:,band], 
				'o',label='ptg='+str(ptg*1000+1000))
		if band==0 and s==0: plt.legend(numpoints=1, prop={'size': 7})
		if band==0 and s==1: plt.title('Spectres dans chaque sous-bande pour 6 pointages differents')
plt.show()
	

