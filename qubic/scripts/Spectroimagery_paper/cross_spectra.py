from __future__ import division, print_function
import healpy as hp
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import itertools as it 


import AnalysisMC as amc
import ReadMC as rmc

import qubic
from qubic import gal2equ, equ2gal
from qubic import Xpol
from qubic import apodize_mask

thespec = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
stokes = ['I', 'Q', 'U']

#Coordinates of the zone observed in the sky
center = equ2gal(0., -57.)

rep_simu = '/home/louisemousset/QUBIC/Qubic_work/SpectroImagerie/SimuLouise/fix_hwp_noise/'

#Number of subbands used during the simulation
nsubvals = np.array([1,2,3,4])

#Archetypes of the files .fits you want to work on
fconv, frecon = [], []
fconv.append(rep_simu + 'fix_hwp_noise_13_nf3_maps_convolved.fits')
frecon.append(rep_simu + 'fix_hwp_noise_13_nf3_maps_recon.fits')
fconv.append(rep_simu + 'fix_hwp_noise_14_nf3_maps_convolved.fits')
frecon.append(rep_simu + 'fix_hwp_noise_14_nf3_maps_recon.fits')

nreal = len(frecon)

# Get all maps
mconv, __ = rmc.maps_from_files(fconv)
mrec, resid, seenmap = rmc.get_maps_residuals(frecon,fconv=fconv)

sh = np.shape(mrec)
print(sh, np.shape(resid))

nsub = sh[1]
ns = hp.npix2nside(sh[2])

# Get Xpol object 
xpol, ell_binned, pwb = rmc.get_xpol(seenmap, ns)

fact = ell_binned * (ell_binned+1) / (2. * np.pi)

nbins = len(ell_binned)


#### Auto spectra
nautos = nreal * nsub
print('  Doing All Autos spectra ({}):'.format(nautos))
autos = np.zeros((nautos, 6, nbins))
autos_conv = np.zeros((nautos, 6, nbins))

j = 0
for real in range(nreal):
    for isub in range(nsub):
        print(j)
        autos_conv[j,:,:] = xpol.get_spectra(mconv[real,isub,:,:])[1] * fact / pwb**2
        autos[j,:,:] = xpol.get_spectra(mrec[real,isub,:,:])[1] * fact  / pwb**2
        j += 1

#plot all auto spectra
plt.figure('Auto_spectra')
for real in range(nreal):
    for s in range(3):
        plt.subplot(nreal,3,3*real+s+1)
        plt.ylabel(thespec[s] + ' real_' + str(real))
        plt.xlabel('l')
        for isub in range(nsub):
            j = nsub * real + isub
            # print(j)
            p = plt.plot(ell_binned, autos_conv[j,s,:], '--')
            plt.plot(ell_binned, autos[j,s,:], 'o-', color=p[0].get_color(), label='subband'+str(isub+1)+'/'+str(nsub))
    
        if s==0 and real==0:
            plt.legend()



#### Cross spectra
ncross = int(sc.special.binom(nreal * nsub, 2))

# Combinations for cross spectra
a = []
for real in range(nreal):
    for isub in range(nsub):
        a.append((real, isub))
comb = list(it.combinations(a,2))
comb = np.reshape(comb, (ncross,4))

cross = np.zeros((ncross, 6, nbins))
print('  Doing All Cross Spectra ({}):'.format(ncross))
for c in range(ncross):
    print(c, comb[c])
    cross[c,:,:] = xpol.get_spectra(mrec[comb[c][0],comb[c][1],:,:], mrec[comb[c][2],comb[c][3],:,:])[1] * fact / pwb**2
             
plt.figure('Cross spectra')
for s in range(3):
    for c in range(ncross):
        if c in [2,7,11]:#[3,10,16,21]:#[1,4]: #[2,7,11]:
            #print c
            plt.subplot(3,3,s+1)  
            p = plt.plot(ell_binned, autos_conv[1,s,:], '--')
            plt.plot(ell_binned, cross[c,s,:], 'o-', label='cross'+str(c)) 
            plt.ylabel(thespec[s] + ' same band, other real')
            plt.xlabel('l')
            if s==0: plt.legend(numpoints=1)  

        
        elif c in [0,1,5,12,13,14]:#[0,1,2,7,8,13,22,23,24,25,26,27]:#[0,5]:#[0,1,5,12,13,14]:
            print(c)
            plt.subplot(3,3,3+s+1)
            p = plt.plot(ell_binned, autos_conv[0,s,:], '--')
            plt.plot(ell_binned, cross[c,s,:], 'o-',label='cross'+str(c)) 
            plt.ylabel(thespec[s]+ ' other band, same real')
            plt.xlabel('l')
            if s==0: plt.legend(numpoints=1)  
        
        else:
            plt.subplot(3,3,6+s+1)
            p = plt.plot(ell_binned, autos_conv[0,s,:], '--')
            plt.plot(ell_binned, cross[c,s,:], 'o-',label='cross'+str(c))
            plt.ylabel(thespec[s]+ ' other band, other real')
            plt.xlabel('l')   
            if s==0: plt.legend(numpoints=1)

#Difference
diff = cross[2,:,:] - cross[3,:,:]
for s in range(3):
    plt.subplot(1,3,s+1)
    plt.plot(ell_binned, cross[2,s,:], 'o-',label='cross2') 
    plt.plot(ell_binned, cross[3,s,:], 'o-',label='cross3') 
    plt.plot(ell_binned, diff[s], 'o-',label='diff')
    if s==0: plt.legend()

# ======================= Apply Xpoll to get spectra ============================

xpol, ell_binned, pwb = amc.get_xpol(seenmap_conv, ns)

nbins = len(ell_binned)
print('nbins = {}'.format(nbins))

mcls, mcls_in = [], []
scls, scls_in = [], []

# Input : what we should find
mapsconv = np.zeros((12 * ns ** 2, 3))

# Output, what we find
maps_recon = np.zeros((12 * ns ** 2, 3))

for isub in range(len(nsubvals)):
    sh = allmaps_conv[isub].shape
    nreals = sh[0]
    nsub = sh[1]
    cells = np.zeros((6, nbins, nsub, nreals))
    cells_in = np.zeros((6, nbins, nsub, nreals))
    print(cells.shape)
    for real in range(nreals):
        for n in range(nsub):
            mapsconv[seenmap_conv, :] = allmaps_conv[isub][real, n, :, :]
            maps_recon[seenmap_conv, :] = allmaps_recon[isub][real, n, :, :]
            cells_in[:, :, n, real] = xpol.get_spectra(mapsconv)[1]
            cells[:, :, n, real] = xpol.get_spectra(maps_recon)[1]

    mcls.append(np.mean(cells, axis=3))
    mcls_in.append(np.mean(cells_in, axis=3))
    scls.append(np.std(cells, axis=3))
    scls_in.append(np.std(cells_in, axis=3))

# Plot the spectra
plt.figure('TT_EE_BB spectra')
for isub in range(len(nsubvals)):
    for s in [1, 2]:
        plt.subplot(4, 2, isub * 2 + s)
        plt.ylabel(thespec[s] + '_ptg=' + str((isub + 1) * 1000))
        plt.xlabel('l')
        sh = mcls[isub].shape
        nsub = sh[2]
        for k in range(nsub):
            p = plt.plot(ell_binned, ell_binned * (ell_binned + 1) * mcls_in[isub][s, :, k], '--')
            plt.errorbar(ell_binned, ell_binned * (ell_binned + 1) * mcls[isub][s, :, k],
                         yerr=ell_binned * (ell_binned + 1) * scls[isub][s, :, k],
                         fmt='o', color=p[0].get_color(),
                         label='subband' + str(k + 1))

        if isub == 0 and s == 1:
            plt.legend(numpoints=1, prop={'size': 7})
plt.show()
