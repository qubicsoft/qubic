#!/bin/env python
from __future__ import division
import sys
import os
import time

import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

import qubic
from pysimulators import FitsArray

import SpectroImLib as si

dictfilename = '/sps/hep/qubic/Users/lmousset/myqubic/qubic/scripts/Spectroimagery_paper/test_spectroim.dict' 
rep_out = '/sps/hep/qubic/Users/lmousset/SpectroImaging/'
name = 'testCC_01'

#Numbers of subbands for spectroimaging
noutmin = 1
noutmax = 1

## Input sky parameters
skypars = {'dust_coeff':1e-2, 'r':0}#1.39e-2

d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)

#Print dictionary and others parameters
#Save a file with al parameters
tem = sys.stdout
sys.stdout = f = open(rep_out + name + '.txt','wt')

print('Simulation General Name: ' + name)
print('Dictionnary File: ' + dictfilename)
for k in d.keys(): 
    print(k, d[k])

print('Minimum Number of Sub Frequencies: {}'.format(noutmin))
print('Maximum Number of Sub Frequencies: {}'.format(noutmax))
print(skypars)

sys.stdout = tem
f.close()

####################################################################################################


##### Sky Creation #####

t0 = time.time()
x0 = si.create_input_sky(d, skypars)
t1 = time.time()
print('********************* Input Sky done in {} seconds'.format(t1-t0))

print(x0.shape)

##### Test : Let's put I to 0 ######
# x0[:,:,0] = 0
# x0[:,:,2] = 0


#### Pointing strategy #####
p = qubic.get_pointing(d)
# p = sam.get_pointing(d)
# I = np.array([0,1,3,5])
# p.angle_hwp = np.random.choice(I*11.25, d['npointings'])
# print(np.unique(p.angle_hwp))

##### TOD making #####
TOD = si.create_TOD(d, p, x0)

#### Mapmaking #####
# for tol in [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]:
#     d['tol'] = tol
#     print(tol)
for nf_sub_rec in np.arange(noutmin, noutmax+1):
    print('-------------------------- Map-Making on {} sub-map(s)'.format(nf_sub_rec))
    maps_recon, cov, nus, nus_edge, maps_convolved = si.reconstruct_maps(TOD, d, p, nf_sub_rec, x0=x0)
    if nf_sub_rec==1:
        maps_recon = np.reshape(maps_recon, np.shape(maps_convolved))
    #Look at the coverage of the sky
    cov = np.sum(cov, axis=0)
    maxcov = np.max(cov)
    unseen = cov < maxcov*0.1
    #diffmap = maps_convolved - maps_recon
    maps_convolved[:,unseen,:] = hp.UNSEEN
    maps_recon[:,unseen,:] = hp.UNSEEN
    #diffmap[:,unseen,:] = hp.UNSEEN
    #therms = np.std(diffmap[:,~unseen,:], axis = 1)
        
    print('************************** Map-Making on {} sub-map(s)Done'.format(nf_sub_rec))

    #FitsArray(nus_edge).save(name + '_nf{0}_ptg{1}'.format(nf_sub_rec, ptg) + '_nus_edges.fits')
    #FitsArray(nus).save(name + '_nf{0}_ptg{1}'.format(nf_sub_rec, ptg)+ '_nus.fits')
    FitsArray(maps_convolved).save(rep_out + name + '_nf{}'.format(nf_sub_rec) + '_maps_convolved.fits')
    FitsArray(maps_recon).save(rep_out + name + '_nf{}'.format(nf_sub_rec) + '_maps_recon.fits')
        
    t1 = time.time()
    print('************************** All Done in {} minutes'.format((t1-t0)/60))

##### SIMU fix_hwp
# p = qubic.get_pointing(d)
# #I = np.array([0,2,4,6])
# pi_fraction = 6
# I = np.arange(pi_fraction/2)
# ptg_start = 500
# for simu in xrange(len(I)):

#     if simu==0:
#         pp = p
#         pp.angle_hwp = I[0]*np.rad2deg(np.pi/pi_fraction )#11.25 # pour les simus avec l'angle par palier
#         # pp.angle_hwp = np.random.choice(I*11.25, ptg_start)

#     else:
#         d['npointings'] = (simu+1)*ptg_start
#         pp = qubic.get_pointing(d)

#         pp.azimuth = np.tile(p.azimuth, simu+1)
#         pp.elevation = np.tile(p.elevation, simu+1)
#         pp.pitch = np.tile(p.pitch, simu+1)
#         pp.time = np.tile(p.time, simu+1)

#         #hwp angle fix
#         pp.angle_hwp = np.concatenate((pnew.angle_hwp, np.tile(np.rad2deg(np.pi/pi_fraction )*I[simu], ptg_start)))
        
#         #hwp angle random
#         # hwp_ang = np.zeros(ptg_start)
#         # for ang in xrange(ptg_start):
#         #     ptg = (simu-1)*ptg_start + ang
#         #     if pnew.angle_hwp[ptg]>11.25:
#         #         hwp_ang[ang] = pnew.angle_hwp[ptg] - 22.5
#         #     else : 
#         #         hwp_ang[ang] = 11.25 * 7
#         # pp.angle_hwp = np.concatenate((pnew.angle_hwp, hwp_ang))

#     name = 'test_0' + str(simu+1)

#     TOD = si.create_TOD(d, pp, x0)

#     for nf_sub_rec in np.arange(noutmin, noutmax+1):
#         print('-------------------------- Map-Making on {} sub-map(s)'.format(nf_sub_rec))
#         maps_recon, cov, nus, nus_edge, maps_convolved = si.reconstruct_maps(TOD, d, pp, nf_sub_rec, x0=x0)
#         if nf_sub_rec==1:
#             maps_recon = np.reshape(maps_recon, np.shape(maps_convolved))
#         #Look at the coverage of the sky
#         cov = np.sum(cov, axis=0)
#         maxcov = np.max(cov)
#         unseen = cov < maxcov*0.1

#         #diffmap = maps_convolved - maps_recon
#         maps_convolved[:,unseen,:] = hp.UNSEEN
#         maps_recon[:,unseen,:] = hp.UNSEEN
#         #diffmap[:,unseen,:] = hp.UNSEEN
#         #therms = np.std(diffmap[:,~unseen,:], axis = 1)
            
#         print('************************** Map-Making on {} sub-map(s)Done'.format(nf_sub_rec))

#         #FitsArray(nus_edge).save(name + '_nf{0}_ptg{1}'.format(nf_sub_rec, ptg) + '_nus_edges.fits')
#         #FitsArray(nus).save(name + '_nf{0}_ptg{1}'.format(nf_sub_rec, ptg)+ '_nus.fits')
#         FitsArray(maps_convolved).save(name + '_nf{}'.format(nf_sub_rec) + '_maps_convolved.fits')
#         FitsArray(maps_recon).save(name + '_nf{}'.format(nf_sub_rec) + '_maps_recon.fits')
            
#         t1 = time.time()
#         print('************************** All Done in {} minutes'.format((t1-t0)/60))

#     if simu != 0: print(pnew)
#     print(pp)
#     print(p)
#     pnew = pp

# plt.subplot(321)
# plt.plot(pp.time, '.')
# plt.xlabel('pointing sample')
# plt.ylabel('Time')
# plt.subplot(322)
# plt.plot(pp.angle_hwp, '.')
# plt.xlabel('pointing sample')
# plt.ylabel('HWP angle')
# plt.subplot(323)
# plt.plot(pp.azimuth, '.')
# plt.xlabel('pointing sample')
# plt.ylabel('azimuth')
# plt.subplot(324)
# plt.plot(pp.elevation, '.')
# plt.xlabel('pointing sample')
# plt.ylabel('elevation')
# plt.subplot(325)
# plt.plot(pp.galactic[:,0], '.')
# plt.xlabel('pointing sample')
# plt.ylabel('galactic0')
# plt.subplot(326)
# plt.plot(pp.equatorial[:,1], '.')
# plt.xlabel('pointing sample')
# plt.ylabel('equatorial1')


# f = open('fix.txt', 'w')
# f.write(str(pp.angle_hwp[0:10])+'\n')
# f.write(str(pp.angle_hwp[1000:1010])+'\n')
# f.write(str(pp.angle_hwp[2000:2020])+'\n')
# f.write(str(pp.angle_hwp[3000:3030])+'\n')
# f.close()

# np.savetxt('hwp_angle.txt', p.angle_hwp[0:10], fmt='%2.2f')
# np.savetxt('azimuth.txt', p.azimuth, fmt='%3.8f')


##### Plusieurs pointings #####
# for ptg in xrange(6):
#     p = qubic.get_pointing(d)
#     d['npointings'] += 1000
#     #d['seed'] += 1
#     print(len(p.pitch))
#     print(p.pitch[500], p.pitch[-1]) 
#     t2 = time.time()
#     print('************************** Pointing done in {} seconds'.format(t2-t1))

#     ##### Test : Let's put I to 0 ######
#     #x0[:,:,0] = 0

#     ##### TOD making #####
#     TOD = si.create_TOD(d, p, x0)

#     ##### Mapmaking #####
#     for nf_sub_rec in np.arange(noutmin, noutmax+1):
#         print('-------------------------- Map-Making on {} sub-map(s)'.format(nf_sub_rec))
#         maps_recon, cov, nus, nus_edge, maps_convolved = si.reconstruct_maps(TOD, d, p, nf_sub_rec, x0=x0)
#         if nf_sub_rec==1:
#             maps_recon = np.reshape(maps_recon, np.shape(maps_convolved))
#         #Look at the coverage of the sky
#         cov = np.sum(cov, axis=0)
#         maxcov = np.max(cov)
#         unseen = cov < maxcov*0.1
#         #diffmap = maps_convolved - maps_recon
#         maps_convolved[:,unseen,:] = hp.UNSEEN
#         maps_recon[:,unseen,:] = hp.UNSEEN
#         #diffmap[:,unseen,:] = hp.UNSEEN
#         #therms = np.std(diffmap[:,~unseen,:], axis = 1)
        
#         print('************************** Map-Making on {} sub-map(s)Done'.format(nf_sub_rec))

#         FitsArray(nus_edge).save(name + '_nf{0}_ptg{1}'.format(nf_sub_rec, ptg) + '_nus_edges.fits')
#         FitsArray(nus).save(name + '_nf{0}_ptg{1}'.format(nf_sub_rec, ptg)+ '_nus.fits')
#         FitsArray(maps_convolved).save(name + '_nf{0}_ptg{1}'.format(nf_sub_rec, ptg) + '_maps_convolved.fits')
#         FitsArray(maps_recon).save(name + '_nf{0}_ptg{1}'.format(nf_sub_rec, ptg) + '_maps_recon.fits')
        
#         t1 = time.time()
#         print('************************** All Done in {} minutes'.format((t1-t0)/60))



# ##### Plusieurs realisations #####
# for real in xrange(50):
#     p = qubic.get_pointing(d)
#     d['seed'] += 1 #pour avoir un pointing par real
#     print(len(p.pitch))
#     print(p.pitch[500], p.pitch[-1]) 
#     t2 = time.time()
#     print('************************** Pointing done in {} seconds'.format(t2-t1))

#     ##### Test : Let's put I to 0 ######
#     #x0[:,:,0] = 0

#     ##### TOD making #####
#     TOD = si.create_TOD(d, p, x0)

#     ##### Mapmaking #####
#     for nf_sub_rec in np.arange(noutmin, noutmax+1):
#         print('-------------------------- Map-Making on {} sub-map(s)'.format(nf_sub_rec))
#         maps_recon, cov, nus, nus_edge, maps_convolved = si.reconstruct_maps(TOD, d, p, nf_sub_rec, x0=x0)
#         if nf_sub_rec==1:
#             maps_recon = np.reshape(maps_recon, np.shape(maps_convolved))
#         #Look at the coverage of the sky
#         cov = np.sum(cov, axis=0)
#         maxcov = np.max(cov)
#         unseen = cov < maxcov*0.1
#         #diffmap = maps_convolved - maps_recon
#         maps_convolved[:,unseen,:] = hp.UNSEEN
#         maps_recon[:,unseen,:] = hp.UNSEEN
#         #diffmap[:,unseen,:] = hp.UNSEEN
#         #therms = np.std(diffmap[:,~unseen,:], axis = 1)
        
#         print('************************** Map-Making on {} sub-map(s)Done'.format(nf_sub_rec))

#         #FitsArray(nus_edge).save(name + '_nf{0}_real{1}'.format(nf_sub_rec, real) + '_nus_edges.fits')
#         #FitsArray(nus).save(name + '_nf{0}_real{1}'.format(nf_sub_rec, real)+ '_nus.fits')
#         FitsArray(maps_convolved).save(name + '_nf{0}_real{1}'.format(nf_sub_rec, real) + '_maps_convolved.fits')
#         FitsArray(maps_recon).save(name + '_nf{0}_real{1}'.format(nf_sub_rec, real) + '_maps_recon.fits')
        
#         t1 = time.time()
#         print('************************** All Done in {} minutes'.format((t1-t0)/60))
