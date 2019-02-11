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

#from mpi4py import MPI
from pyoperators import MPI

#### MPI stuff ####
rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

if rank == 0:
      print('**************************')
      print('Master rank {} is speaking:'.format(rank))
      print('mpi is in')
      print('There are {} ranks'.format(size))
      print('**************************')

print '========================================================== Hello ! I am rank number {}'.format(rank)

#### Simulation name ####
name = 'simu'

#### Dictionary ####
dictfilename = 'test_spectroim.dict' 
d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)

##### Numbers of subbands for spectroimaging ####
noutmin = 1
noutmax = 4

#### Input sky parameters ####
skypars = {'dust_coeff':1.39e-2, 'r':0} #1.39e-2

#Print dictionary and others parameters
#Save a file with al parameters
tem = sys.stdout
sys.stdout = f = open(name + '.txt','wt')

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


##### Sky Creation made only on rank 0 #####
if rank==0:
      t0 = time.time()
      x0 = si.create_input_sky(d, skypars)
      t1 = time.time()
      print('********************* Input Sky - Rank {} - done in {} seconds'.format(rank, t1-t0))
else:
      x0 = None
      t0 = time.time()
#I to 0
x0[:,:,0] = 0.

x0 = MPI.COMM_WORLD.bcast(x0)



##### Pointing in not picklable so cannot be broadcasted => done on all ranks simultaneously
t1 = time.time()
p = qubic.get_pointing(d)

t2 = time.time()
print('************************** Pointing - rank {} - done in {} seconds'.format(rank, t2-t1))


##### TOD making is intrinsically parallelized (use of pyoperators)
print('-------------------------- TOD - rank {} Starting'.format(rank))
TOD = si.create_TOD(d, p, x0)
print('************************** TOD - rank {} Done - elaplsed time is {}'.format(rank,time.time()-t0))

##### Wait for all the TOD to be done (is it necessary ?)
MPI.COMM_WORLD.Barrier()
if rank == 0:
      t1 = time.time()
      print('************************** All TOD OK in {} minutes'.format((t1-t0)/60))


##### Mapmaking #####
for nf_sub_rec in np.arange(noutmin, noutmax+1):
    
      if rank == 0:
            print('-------------------------- Map-Making on {} sub-map(s) - Rank {} Starting'.format(nf_sub_rec,rank))
      maps_recon, cov, nus, nus_edge, maps_convolved = si.reconstruct_maps(TOD, d, p, nf_sub_rec, x0=x0)
      if nf_sub_rec==1: maps_recon = np.reshape(maps_recon, np.shape(maps_convolved))
      #Look at the coverage of the sky
      cov = np.sum(cov, axis=0)
      maxcov = np.max(cov)
      unseen = cov < maxcov*0.1
      maps_convolved[:,unseen,:] = hp.UNSEEN
      maps_recon[:,unseen,:] = hp.UNSEEN
      if rank == 0:
            print('************************** Map-Making on {} sub-map(s) - Rank {} Done'.format(nf_sub_rec,rank))

      MPI.COMM_WORLD.Barrier()

      if rank == 0:
            FitsArray(maps_convolved).save(name + '_nf{}'.format(nf_sub_rec) + '_maps_convolved.fits')
            FitsArray(maps_recon).save(name + '_nf{}'.format(nf_sub_rec) + '_maps_recon.fits')
     
            print('************************** rank {} saved fits files'.format(rank))
            t1 = time.time()
            print('************************** All Done in {} minutes'.format((t1-t0)/60))

      MPI.COMM_WORLD.Barrier()

MPI.Finalize()
