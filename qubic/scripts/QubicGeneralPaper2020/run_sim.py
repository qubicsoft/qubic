#!/usr/bin/python
from pylab import *
import os
import sys
import time
import pickle

# Specific science modules
import healpy as hp
import numpy as np

# Specific qubic modules
from qubicpack.utilities import Qubic_DataDir
from pysimulators import FitsArray
from mpi4py import MPI
# from pyoperators import MPI
import pysm
import qubic
from qubic import SpectroImLib as si
from qubic import QubicSkySim as qss
from qubic import NamasterLib as nam

# Instructions:
# python ./run_sim.py <Dictname> <seed> <generic_string> <npointings> <tol>
# Exemple:
# python ./run_sim.py BmodesNoDustNoSystPaper0_2020.dict 42 CMB-Only-12sub_3000_1e-4 3000 1e-4
#
# NB: the  ouput files are for now by default in /qubic/Sims/EndToEndMaps/ (see line 121)
# On a nother machine it  has to be changed before running...

# MPI stuff
rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

if rank == 0:
    print('')
    print('**************************')
    print('Master rank {} is speaking:'.format(rank))
    print('mpi is in')
    print('There are {} ranks'.format(size))
    print('You are using the sys version:')
    print(sys.version)
    print('**************************')
    print('')


##################### Reading Arguments ################################
if rank == 0:
    print('')
    print('**************************')
    print('Master rank {} is speaking:'.format(rank))
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))
    print('**************************')
    print('')

# Dictionary name
dictname = str(sys.argv[1])

# seed
if str(sys.argv[2])=='None':
    seed = None
    if rank == 0:
        print('Master rank {} is speaking:'.format(rank))
        print('Seed is None',seed)
else:
    seed = int(sys.argv[2])
    if rank == 0:
        print('Master rank {} is speaking:'.format(rank))
        print('Seed is NOT None',seed)

# name of the simulation
namesim = str(sys.argv[3])

# number of pointings
nptg = int(sys.argv[4])

# tol for Map-Making
tol = float(sys.argv[5])

if rank==0:
    print('')
    print('**************************')
    print('Rank {} Speaking:'.format(rank))
    print('Dictionnary File: '+dictname)
    print('Simulation General Name: '+namesim)
    print('Number of pointings: {}'.format(nptg))
    print('Mapmaking Tolerance: {}'.format(tol))
    print('**************************')
    print('')
########################################################################


############# Reading Dictionary #######################################
# Repository for dictionary and input maps
global_dir = Qubic_DataDir(datafile='instrument.py', datadir=os.environ['QUBIC_DATADIR'])
dictfilename = global_dir + '/dicts/'+dictname

# Read dictionary chosen
d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)
d['npointings'] = nptg
d['tol'] = tol
center = qubic.equ2gal(d['RA_center'], d['DEC_center'])
#########################################################################




############### Sky  Simulation on rank 0 and broadcasted ###############
# Make a sky using QubicSkySim with r=0 (default simple request - the camb spectrum is calculated inside)
sky_config = {'cmb': seed}
if rank == 0:
    t0 = time.time()
    print('')
    print('**************************************************************')
    print('Rank {} is creating the input sky'.format(rank))
    Qubic_sky = qss.Qubic_sky(sky_config, d)
    x0 = np.reshape(Qubic_sky.get_simple_sky_map(),(d['nf_sub'],d['nside']**2*12,3))
    print('Input SubFrequencies Map with shape (nf_sub, #pixels, #stokes) : ', np.shape(x0))
    t1 = time.time()
    print('********************* Input Sky - Rank {} - done in {} seconds'.format(rank, t1 - t0))
    print('**************************************************************')
else:
    t0 = time.time()
    x0 = None
    Qubic_sky = None

# Now broadcast it everywhere
x0 = MPI.COMM_WORLD.bcast(x0)
Qubic_sky = MPI.COMM_WORLD.bcast(Qubic_sky)
# print('**************************************************************')
# print('X0 test: rank {} {} {}'.format(rank, x0[0], x0[1]))
# print('**************************************************************')

# The input spectra are
input_cl = Qubic_sky.input_cmb_spectra
# print('**************************************************************')
# print('input_cl test: rank {} {} {}'.format(rank, input_cl[:,0], input_cl[:,1]))
# print('**************************************************************')


############### Pointing cannot be broadcasted as it is not pickleable ###############
p = qubic.get_pointing(d)
print('**************************************************************')
print('=== Pointing DONE! at rank {} ==='.format(rank))
print('Pointing Test rank {}: {}'.format(rank, p.equatorial[0]))
print('**************************************************************')



# ==== TOD making ====
print('-------------------------- TOD - rank {} Starting'.format(rank))
TOD, maps_convolved = si.create_TOD(d, p, x0)
maps_convolved = np.array(maps_convolved)
print('--- Rank {} ----- Noiseless TOD with shape: {} - Done ---------'.format(rank, np.shape(TOD)))
print('--- Rank {} ---- Maps Convolved with shape: {} - Done ---------'.format(rank, np.shape(maps_convolved)))

##### Wait for all the TOD to be done (is it necessary ?)
MPI.COMM_WORLD.Barrier()
if rank == 0:
      t1 = time.time()
      print('************************** All TOD OK in {} minutes'.format((t1-t0)/60))


##### QUBIC Instrument and Scene
q = qubic.QubicMultibandInstrument(d)
s = qubic.QubicScene(d)


# ==== Map Making ====
nf_sub_rec = d['nf_recon']
Nfreq_edges, nus_edge, nus, deltas, Delta, Nbbands = qubic.compute_freq(150, nf_sub_rec)
print('Band center:', nus)
print('Band edges:', nus_edge)
print('Band width:', deltas)

len(nus_edge)
for i in range(len(nus_edge) - 1):
    print('base =', nus_edge[i+1] / nus_edge[i])

if rank == 0:
    print('-------------------------- Map-Making on {} sub-map(s) - Rank {} Starting'.format(nf_sub_rec, rank))
maps_recon, cov, nus, nus_edge, maps_convolved = si.reconstruct_maps(TOD, d, p,
                                                                    nf_sub_rec, x0=x0)
maps_convolved = np.reshape(maps_convolved,(d['nf_recon'], 12*d['nside']**2, 3))
maps_recon = np.reshape(maps_recon,(d['nf_recon'], 12*d['nside']**2, 3))
print(maps_recon.shape)

# Look at the coverage of the sky
coverage = np.sum(cov.copy(), axis=0)
maxcov = np.max(coverage)
unseen = coverage < maxcov * 0.1
print(maps_convolved.shape)
maps_convolved[:, unseen, :] = hp.UNSEEN
maps_recon[:, unseen, :] = hp.UNSEEN

# Wait for everyone to finish
MPI.COMM_WORLD.Barrier()

#### Save maps
if rank == 0:
    rnd_name = qss.random_string(10)
    directory = '/global/homes/h/hamilton/qubic/jc/EndToEndMaps/'
    FitsArray(maps_recon).save(directory+namesim+'_maps_recon_seed_'+str(seed)+'_'+rnd_name+'.fits')
    FitsArray(maps_convolved).save(directory+namesim+'_maps_convolved_seed_'+str(seed)+'_'+rnd_name+'.fits')
    FitsArray(coverage).save(directory+namesim+'_maps_coverage_'+rnd_name+'.fits')

    with open(directory+namesim+'_dictionary_'+rnd_name+'.pickle', 'wb') as handle:
        pickle.dump(d, handle, protocol=2)

    with open(directory+namesim+'_input_cell_'+rnd_name+'.pickle', 'wb') as handle:
        pickle.dump(Qubic_sky.input_cmb_spectra, handle, protocol=2)

        print('************************** rank {} saved fits files'.format(rank))
        print('The recon map is:')
        print(directory+namesim+'_maps_recon_seed_'+str(seed)+'_'+rnd_name+'.fits')
        t1 = time.time()
        print('************************** All Done in {} minutes'.format((t1 - t0) / 60))

print('Finished Job {} for rank {}  !!!'.format(namesim, rank))
#MPI.COMM_WORLD.Barrier()
#MPI.Finalize()











